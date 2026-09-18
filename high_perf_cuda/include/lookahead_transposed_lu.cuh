#pragma once

#include "cuda_utils.cuh"
#include "types.hpp"
#include "panel_kernels.cuh"
#include "fused_panel_kernel.cuh"
#include "transposed_hierarchical_lu.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cmath>

namespace high_perf {

class LookaheadTransposedLU {
private:
    int n_;
    int macro_w_;
    int micro_w_;
    cublasHandle_t cublas_handle_panel_ = nullptr;
    cublasHandle_t cublas_handle_gemm_ = nullptr;
    cudaStream_t stream_panel_ = nullptr;
    cudaStream_t stream_gemm_ = nullptr;
    cudaEvent_t ev_next_panel_ready_ = nullptr;
    cudaEvent_t ev_trsm_stripe_ready_ = nullptr;
    cudaEvent_t ev_gemm2_done_ = nullptr;
    cudaEvent_t ev_trsm_done_ = nullptr;

    // Workspaces
    __half* d_B_ = nullptr;
    __half* d_panel_buf_ = nullptr;
    __half* d_L_inv_micro_ = nullptr;
    __half* d_u12_buf_ = nullptr;

    int* d_log_pivot_rows_ = nullptr;
    int* d_log_pivot_cols_ = nullptr;
    unsigned long long* d_counters_ = nullptr;
    float* d_row_scales_ = nullptr;

    bool coop_supported_ = false;
    int fused_num_blocks_ = 0;
    volatile BlockPivotResult* d_block_results_ = nullptr;

public:
    LookaheadTransposedLU(int n, int macro_w = 1024, int micro_w = 32)
        : n_(n), macro_w_(macro_w), micro_w_(micro_w) {

        // Prioritize the panel factorization stream (-1) over the trailing GEMM stream (0)
        // so that the latency-sensitive critical path (pivot search, swap, TRSM) is scheduled
        // immediately by the GPU hardware work distributor, hiding panel latency behind trailing GEMM.
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_panel_, cudaStreamNonBlocking, -1));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_gemm_, cudaStreamNonBlocking, 0));

        CUBLAS_CHECK(cublasCreate(&cublas_handle_panel_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_panel_, stream_panel_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_panel_, CUBLAS_TENSOR_OP_MATH));

        CUBLAS_CHECK(cublasCreate(&cublas_handle_gemm_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_gemm_, stream_gemm_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_gemm_, CUBLAS_TENSOR_OP_MATH));

        // Use timing-disabled events for zero-overhead intra-GPU stream synchronization
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_next_panel_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_trsm_stripe_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_gemm2_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_trsm_done_, cudaEventDisableTiming));

        size_t matrix_bytes = static_cast<size_t>(n_) * n_ * sizeof(__half);
        CUDA_CHECK(cudaMalloc(&d_B_, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_panel_buf_, static_cast<size_t>(n_) * macro_w_ * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_L_inv_micro_, 128 * 128 * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_u12_buf_, static_cast<size_t>(n_) * 128 * sizeof(__half)));

        CUDA_CHECK(cudaMalloc(&d_log_pivot_rows_, n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_log_pivot_cols_, n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counters_, kCounterSlots * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_row_scales_, n_ * sizeof(float)));

        // Cooperative launch detection
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int coop_support = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, device_id));
        coop_supported_ = (coop_support != 0);

        if (coop_supported_) {
            int max_blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                fused_panel_coop_kernel,
                FUSED_COOP_BLOCK, 0
            );
            int num_sms = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
            fused_num_blocks_ = 160; // Optimal 2 blocks/SM: fastest reduction, low sync overhead, leaves 75% SMs free
            if (fused_num_blocks_ > FUSED_MAX_BLOCKS) fused_num_blocks_ = FUSED_MAX_BLOCKS;
            CUDA_CHECK(cudaMalloc((void**)&d_block_results_, fused_num_blocks_ * sizeof(BlockPivotResult)));
        }

        cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }

    void launch_laswp_disjoint(int k1, int k2, const int* ipiv, int row_start, int row_end, cudaStream_t stream) {
        if (k2 <= k1 || row_end <= row_start) return;
        int num_rows = row_end - row_start;
        int num_pairs = num_rows / 2;
        int threads = 256;
        int blocks = std::min((num_pairs + threads - 1) / threads, 160);
        if (blocks < 1) blocks = 1;
        transposed_detail::batched_laswp_transposed_disjoint_kernel<<<blocks, threads, 0, stream>>>(
            d_B_, n_, k1, k2, ipiv, row_start, row_end
        );
    }

    ~LookaheadTransposedLU() {
        if (cublas_handle_panel_) cublasDestroy(cublas_handle_panel_);
        if (cublas_handle_gemm_) cublasDestroy(cublas_handle_gemm_);
        if (stream_panel_) cudaStreamDestroy(stream_panel_);
        if (stream_gemm_) cudaStreamDestroy(stream_gemm_);
        if (ev_next_panel_ready_) cudaEventDestroy(ev_next_panel_ready_);
        if (ev_trsm_stripe_ready_) cudaEventDestroy(ev_trsm_stripe_ready_);
        if (ev_gemm2_done_) cudaEventDestroy(ev_gemm2_done_);
        if (ev_trsm_done_) cudaEventDestroy(ev_trsm_done_);

        if (d_B_) cudaFree(d_B_);
        if (d_panel_buf_) cudaFree(d_panel_buf_);
        if (d_L_inv_micro_) cudaFree(d_L_inv_micro_);
        if (d_u12_buf_) cudaFree(d_u12_buf_);
        if (d_log_pivot_rows_) cudaFree(d_log_pivot_rows_);
        if (d_log_pivot_cols_) cudaFree(d_log_pivot_cols_);
        if (d_counters_) cudaFree(d_counters_);
        if (d_row_scales_) cudaFree(d_row_scales_);
        if (d_block_results_) cudaFree((void*)d_block_results_);
    }

    // Factorize Macro-Panel helper on stream_panel_
    void factorize_macro_panel(int k_macro, int W_actual, const Options& options) {
        const int m_panel = n_ - k_macro;
        const int wb = micro_w_;
        const __half h_one = __float2half(1.0f);
        const __half h_zero = __float2half(0.0f);
        const __half h_minus_one = __float2half(-1.0f);
        dim3 tblk(transposed_detail::TILE_DIM, transposed_detail::BLOCK_ROWS);

        // Step a: Extract Macro-Panel from B to d_panel_buf
        dim3 sgrid((m_panel + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM,
                   (W_actual + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM);
        transposed_detail::transpose_submatrix_kernel<<<sgrid, tblk, 0, stream_panel_>>>(
            d_B_ + k_macro + static_cast<size_t>(k_macro) * n_, n_,
            d_panel_buf_, m_panel,
            m_panel, W_actual
        );

        // Step b: Factorize Macro-Panel using Micro-Panels inside d_panel_buf
        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            int n_arg = m_panel;
            int k_arg = j_micro;
            int w_arg = wb_actual;
            int method_arg = static_cast<int>(options.method);
            float tau_arg = options.tau;
            int* d_pivot_slice = d_log_pivot_rows_ + k_macro;
            void* args[] = {
                (void*)&d_panel_buf_,
                (void*)&n_arg,
                (void*)&k_arg,
                (void*)&w_arg,
                (void*)&method_arg,
                (void*)&tau_arg,
                (void*)&d_row_scales_,
                (void*)&d_pivot_slice,
                (void*)&d_counters_,
                (void*)&d_block_results_
            };

            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)fused_panel_coop_kernel,
                dim3(fused_num_blocks_), dim3(FUSED_COOP_BLOCK),
                args, 0, stream_panel_
            ));

            const int rest_macro_cols = W_actual - (j_micro + wb_actual);
            if (rest_macro_cols > 0) {
                int lt = 256;
                int lg = (rest_macro_cols + lt - 1) / lt;
                batched_laswp_kernel<<<lg, lt, 0, stream_panel_>>>(
                    d_panel_buf_, m_panel, j_micro, j_micro + wb_actual,
                    d_pivot_slice, j_micro + wb_actual, W_actual
                );

                size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
                trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream_panel_>>>(
                    d_panel_buf_, m_panel, j_micro, wb_actual, d_L_inv_micro_
                );

                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_N,
                    wb_actual, rest_macro_cols, wb_actual,
                    &h_one,
                    d_L_inv_micro_, CUDA_R_16F, wb_actual,
                    d_panel_buf_ + (j_micro + static_cast<size_t>(j_micro + wb_actual) * m_panel), CUDA_R_16F, m_panel,
                    &h_zero,
                    d_u12_buf_, CUDA_R_16F, wb_actual,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));

                dim3 sblk(32, 8);
                dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest_macro_cols + sblk.y - 1) / sblk.y);
                store_block_kernel<<<sgrd, sblk, 0, stream_panel_>>>(
                    d_u12_buf_, wb_actual, d_panel_buf_, m_panel, j_micro, j_micro + wb_actual, wb_actual, rest_macro_cols
                );

                const int m_macro_trail = m_panel - (j_micro + wb_actual);
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_N,
                    m_macro_trail, rest_macro_cols, wb_actual,
                    &h_minus_one,
                    d_panel_buf_ + ((j_micro + wb_actual) + static_cast<size_t>(j_micro) * m_panel), CUDA_R_16F, m_panel,
                    d_u12_buf_, CUDA_R_16F, wb_actual,
                    &h_one,
                    d_panel_buf_ + ((j_micro + wb_actual) + static_cast<size_t>(j_micro + wb_actual) * m_panel), CUDA_R_16F, m_panel,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
        }

        // Convert local panel pivots to global row indices
        int opt_threads = 256;
        int opt_grid = (W_actual + opt_threads - 1) / opt_threads;
        transposed_detail::offset_pivots_kernel<<<opt_grid, opt_threads, 0, stream_panel_>>>(
            d_log_pivot_rows_ + k_macro, W_actual, k_macro
        );

        // Apply row swaps of current macro-panel to previous L blocks (columns 0..k_macro of A = rows 0..k_macro of B)
        if (k_macro > 0) {
            launch_laswp_disjoint(k_macro, k_macro + W_actual, d_log_pivot_rows_, 0, k_macro, stream_panel_);
        }

        // Step c: Transpose factorized Macro-Panel back to B
        dim3 bgrid((W_actual + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM,
                   (m_panel + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM);
        transposed_detail::transpose_submatrix_kernel<<<bgrid, tblk, 0, stream_panel_>>>(
            d_panel_buf_, m_panel,
            d_B_ + k_macro + static_cast<size_t>(k_macro) * n_, n_,
            W_actual, m_panel
        );
    }

    // Trailing TRSM on stream_panel_
    void trailing_trsm(int k_macro, int W_actual) {
        const int macro_end = k_macro + W_actual;
        const int n_trail = n_ - macro_end;
        const int m_panel = n_ - k_macro;
        const int wb = std::max(micro_w_, 64);
        const __half h_one = __float2half(1.0f);
        const __half h_zero = __float2half(0.0f);
        const __half h_minus_one = __float2half(-1.0f);

        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream_panel_>>>(
                d_panel_buf_, m_panel, j_micro, wb_actual, d_L_inv_micro_
            );

            __half* A_j_T = d_B_ + macro_end + static_cast<size_t>(k_macro + j_micro) * n_;
            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_T,
                n_trail, wb_actual, wb_actual,
                &h_one,
                A_j_T, CUDA_R_16F, n_,
                d_L_inv_micro_, CUDA_R_16F, wb_actual,
                &h_zero,
                d_u12_buf_, CUDA_R_16F, n_trail,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));

            dim3 sblk(32, 8);
            dim3 sgrd((n_trail + sblk.x - 1) / sblk.x, (wb_actual + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk, 0, stream_panel_>>>(
                d_u12_buf_, n_trail, d_B_, n_, macro_end, k_macro + j_micro, n_trail, wb_actual
            );

            const int rest_macro_rows = W_actual - (j_micro + wb_actual);
            if (rest_macro_rows > 0) {
                __half* L_sub_T = d_B_ + (k_macro + j_micro) + static_cast<size_t>(k_macro + j_micro + wb_actual) * n_;
                __half* A_rest_T = d_B_ + macro_end + static_cast<size_t>(k_macro + j_micro + wb_actual) * n_;

                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_N,
                    n_trail, rest_macro_rows, wb_actual,
                    &h_minus_one,
                    d_u12_buf_, CUDA_R_16F, n_trail,
                    L_sub_T, CUDA_R_16F, n_,
                    &h_one,
                    A_rest_T, CUDA_R_16F, n_,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
        }
    }

    bool factorize(
        __half* d_matrix,
        int n,
        const Options& options,
        std::vector<int>& out_pivot_rows,
        std::vector<int>& out_pivot_cols,
        Counters& out_counters,
        TimeBreakdown* out_breakdown = nullptr
    ) {
        if (!coop_supported_) return false;

        const int W = macro_w_;
        const __half h_one = __float2half(1.0f);
        const __half h_minus_one = __float2half(-1.0f);

        int threads = 256;
        int grid = (n + threads - 1) / threads;
        CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, kCounterSlots * sizeof(unsigned long long), stream_panel_));
        init_identity_perm_kernel<<<grid, threads, 0, stream_panel_>>>(d_log_pivot_rows_, n);
        init_identity_perm_kernel<<<grid, threads, 0, stream_panel_>>>(d_log_pivot_cols_, n);

        // Precompute row norms for ScPP
        if (options.method == Method::kScPP) {
            compute_row_scales_kernel<<<grid, threads, 0, stream_panel_>>>(d_matrix, n, d_row_scales_);
        }

        // Transpose A to B in stream_panel_
        dim3 tblk(transposed_detail::TILE_DIM, transposed_detail::BLOCK_ROWS);
        dim3 tgrid((n + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM,
                   (n + transposed_detail::TILE_DIM - 1) / transposed_detail::TILE_DIM);
        transposed_detail::transpose_matrix_kernel<<<tgrid, tblk, 0, stream_panel_>>>(d_matrix, d_B_, n, n, n, n);

        int k_macro = 0;
        int macro_end = std::min(k_macro + W, n);
        int W_actual = macro_end - k_macro;

        // Step 0: Initial Macro-Panel 0 factorization
        factorize_macro_panel(k_macro, W_actual, options);

        while (macro_end < n) {
            const int n_trail = n - macro_end;

            // Step d: Coalesced Transposed LASWP on Trailing Matrix
            launch_laswp_disjoint(k_macro, macro_end, d_log_pivot_rows_, macro_end, n, stream_panel_);

            // Step e: Trailing TRSM
            trailing_trsm(k_macro, W_actual);

            // Lookahead Setup: Determine Next Macro-Panel (k+1)
            const int k_next = macro_end;
            const int macro_next_end = std::min(k_next + W, n);
            const int W_next = macro_next_end - k_next;

            // Make sure stream_gemm waits for stream_panel's TRSM
            CUDA_CHECK(cudaEventRecord(ev_trsm_done_, stream_panel_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_gemm_, ev_trsm_done_, 0));

            // GEMM 1 (Next Macro-Panel Update: W_next rows x n_trail cols of B22) on stream_gemm_
            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                W_next, n_trail, W_actual,
                &h_minus_one,
                d_B_ + macro_end + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                d_B_ + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                &h_one,
                d_B_ + macro_end + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
            CUDA_CHECK(cudaEventRecord(ev_next_panel_ready_, stream_gemm_));

            // 3-GEMM Quadrant Split on stream_gemm_:
            // GEMM 2a: rest_rows x W_next (TRSM dependency column stripe)
            const int rest_rows = n_trail - W_next;
            if (rest_rows > 0) {
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                    rest_rows, W_next, W_actual,
                    &h_minus_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                    d_B_ + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                    &h_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
                CUDA_CHECK(cudaEventRecord(ev_trsm_stripe_ready_, stream_gemm_));

                // GEMM 2b: rest_rows x rest_rows (Bulk trailing core)
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                    rest_rows, rest_rows, W_actual,
                    &h_minus_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                    d_B_ + k_macro + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
                    &h_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
            CUDA_CHECK(cudaEventRecord(ev_gemm2_done_, stream_gemm_));

            // CONCURRENT EXECUTION: stream_panel_ waits for GEMM 1, then factorizes Panel k+1!
            CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_next_panel_ready_, 0));
            factorize_macro_panel(k_next, W_next, options);

            // Now stream_panel_ waits for GEMM 2 before touching the rest of trailing matrix
            if (rest_rows > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_gemm2_done_, 0));
            }

            // Advance to next macro-panel
            k_macro = k_next;
            macro_end = macro_next_end;
            W_actual = W_next;
        }

        // Transpose B back to A in stream_panel_
        CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_gemm2_done_, 0));
        transposed_detail::transpose_matrix_kernel<<<tgrid, tblk, 0, stream_panel_>>>(d_B_, d_matrix, n, n, n, n);

        CUDA_CHECK(cudaStreamSynchronize(stream_panel_));
        CUDA_CHECK(cudaStreamSynchronize(stream_gemm_));

        // Copy pivots and counters
        out_pivot_rows.resize(n);
        out_pivot_cols.resize(n);
        CUDA_CHECK(cudaMemcpy(out_pivot_rows.data(), d_log_pivot_rows_, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_pivot_cols.data(), d_log_pivot_cols_, n * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<unsigned long long> h_counters(kCounterSlots);
        CUDA_CHECK(cudaMemcpy(h_counters.data(), d_counters_, kCounterSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        out_counters.dp_lookahead_accepts = h_counters[kSlotDpAccepts];
        out_counters.dp_fallbacks = h_counters[kSlotDpFallbacks];
        out_counters.gp_near_ties = h_counters[kSlotGpNearTies];
        out_counters.gp_second_choices = h_counters[kSlotGpSecondChoices];
        out_counters.scap_current = h_counters[kSlotScapCurrent];
        out_counters.scap_middle = h_counters[kSlotScapMiddle];
        out_counters.scap_last = h_counters[kSlotScapLast];
        out_counters.rp_iterations = h_counters[kSlotRpIterations];

        return true;
    }
};

} // namespace high_perf
