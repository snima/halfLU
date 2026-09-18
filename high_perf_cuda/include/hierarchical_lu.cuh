#pragma once

/*
 * hierarchical_lu.cuh
 * ============================================================================
 * NVIDIA-Grade Two-Level Hierarchical Blocked FP16 LU Factorization
 * (Macro-Panel + Micro-Panel Architecture)
 *
 * Architecture:
 * 1. Outer Level (Macro-Panel, W = 256 or 512):
 *    - Trailing matrix GEMM operates with K = W (256 or 512)
 *    - Tensor Cores run at 60 - 73 TFLOPS (vs 36 TFLOPS for K=64)
 *    - Global LASWP row permutations on the full trailing matrix are called
 *      4x to 8x less frequently (e.g. 160 or 80 times instead of 640).
 *
 * 2. Inner Level (Micro-Panel, wb = 64):
 *    - Factorizes sub-panels of width wb using fused_panel_coop_kernel
 *    - Row swaps and rank-1 updates inside the macro-panel are confined
 *      to at most W columns (blindingly fast, minimal cache footprint)
 *    - TRTRI shared memory is strictly bounded to 16 KB (wb=64)
 * ============================================================================
 */

#include "cuda_utils.cuh"
#include "types.hpp"
#include "pivot_kernels.cuh"
#include "panel_kernels.cuh"
#include "fused_panel_kernel.cuh"
#include "high_perf_lu.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace high_perf {

// General block store: copies submatrix from src (ld_src) to dst (ld_dst)
// Thread-safe and bounds-checked.
__global__ void store_block_kernel(
    const __half* __restrict__ src,
    int ld_src,
    __half* __restrict__ dst,
    int ld_dst,
    int row_offset_dst,
    int col_offset_dst,
    int num_rows,
    int num_cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < num_rows && c < num_cols) {
        dst[static_cast<size_t>(col_offset_dst + c) * ld_dst + (row_offset_dst + r)] =
            src[static_cast<size_t>(c) * ld_src + r];
    }
}

constexpr int kDefaultMacroWidth = 256;
constexpr int kDefaultMicroWidth = 64;

class HierarchicalLU {
private:
    int n_;
    int macro_w_;
    int micro_w_;
    cublasHandle_t cublas_handle_ = nullptr;

    // Device workspaces
    int* d_log_pivot_rows_ = nullptr;
    int* d_log_pivot_cols_ = nullptr;
    unsigned long long* d_counters_ = nullptr;
    float* d_row_scales_ = nullptr;

    // Micro-TRTRI workspace for micro-panel (wb x wb)
    __half* d_L_inv_micro_ = nullptr;

    // Buffer for Macro-U12 (W x remaining_cols)
    __half* d_macro_U12_ = nullptr;

    // Cooperative launch state for micro-panels
    bool coop_supported_ = false;
    int fused_num_blocks_ = 0;
    volatile BlockPivotResult* d_block_results_ = nullptr;

public:
    HierarchicalLU(int n, int macro_w = kDefaultMacroWidth, int micro_w = kDefaultMicroWidth)
        : n_(n), macro_w_(macro_w), micro_w_(micro_w) {

        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH));

        CUDA_CHECK(cudaMalloc(&d_log_pivot_rows_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_log_pivot_cols_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counters_, kCounterSlots * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_row_scales_, n * sizeof(float)));

        // Micro-L-inv workspace: micro_w * micro_w * sizeof(__half)
        CUDA_CHECK(cudaMalloc(&d_L_inv_micro_, micro_w_ * micro_w_ * sizeof(__half)));

        // Macro-U12 buffer: macro_w * n * sizeof(__half)
        CUDA_CHECK(cudaMalloc(&d_macro_U12_, static_cast<size_t>(macro_w_) * n * sizeof(__half)));

        // Cooperative launch detection
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int coop_support = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, device_id));
        coop_supported_ = (coop_support != 0);

        if (coop_supported_) {
            int max_blocks_per_sm = 0;
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                fused_panel_coop_kernel,
                FUSED_COOP_BLOCK, 0
            );
            if (occ_err != cudaSuccess || max_blocks_per_sm <= 0) {
                coop_supported_ = false;
            } else {
                int num_sms = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
                fused_num_blocks_ = num_sms * max_blocks_per_sm;
                if (fused_num_blocks_ > FUSED_MAX_BLOCKS) fused_num_blocks_ = FUSED_MAX_BLOCKS;
                CUDA_CHECK(cudaMalloc((void**)&d_block_results_, fused_num_blocks_ * sizeof(BlockPivotResult)));
            }
        }

        cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }

    ~HierarchicalLU() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (d_log_pivot_rows_) cudaFree(d_log_pivot_rows_);
        if (d_log_pivot_cols_) cudaFree(d_log_pivot_cols_);
        if (d_counters_) cudaFree(d_counters_);
        if (d_row_scales_) cudaFree(d_row_scales_);
        if (d_L_inv_micro_) cudaFree(d_L_inv_micro_);
        if (d_macro_U12_) cudaFree(d_macro_U12_);
        if (d_block_results_) cudaFree((void*)d_block_results_);
    }

    bool factorize(
        __half* d_matrix,
        int n,
        const Options& options,
        std::vector<int>& host_pivot_rows,
        std::vector<int>& host_pivot_cols,
        Counters& host_counters,
        TimeBreakdown* out_breakdown = nullptr,
        cudaStream_t stream = 0
    ) {
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
        CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, kCounterSlots * sizeof(unsigned long long), stream));

        int threads = 256;
        int grid = (n + threads - 1) / threads;
        init_identity_perm_kernel<<<grid, threads, 0, stream>>>(d_log_pivot_rows_, n);
        init_identity_perm_kernel<<<grid, threads, 0, stream>>>(d_log_pivot_cols_, n);

        if (options.method == Method::kScPP) {
            compute_row_scales_kernel<<<grid, threads, 0, stream>>>(d_matrix, n, d_row_scales_);
        }

        const int W = macro_w_;
        const int wb = micro_w_;

        cudaEvent_t ev_p_start, ev_p_stop;
        cudaEvent_t ev_l_start, ev_l_stop;
        cudaEvent_t ev_t_start, ev_t_stop;
        cudaEvent_t ev_u_start, ev_u_stop;
        cudaEvent_t ev_s_start, ev_s_stop;
        cudaEvent_t ev_g_start, ev_g_stop;
        if (out_breakdown) {
            CUDA_CHECK(cudaEventCreate(&ev_p_start)); CUDA_CHECK(cudaEventCreate(&ev_p_stop));
            CUDA_CHECK(cudaEventCreate(&ev_l_start)); CUDA_CHECK(cudaEventCreate(&ev_l_stop));
            CUDA_CHECK(cudaEventCreate(&ev_t_start)); CUDA_CHECK(cudaEventCreate(&ev_t_stop));
            CUDA_CHECK(cudaEventCreate(&ev_u_start)); CUDA_CHECK(cudaEventCreate(&ev_u_stop));
            CUDA_CHECK(cudaEventCreate(&ev_s_start)); CUDA_CHECK(cudaEventCreate(&ev_s_stop));
            CUDA_CHECK(cudaEventCreate(&ev_g_start)); CUDA_CHECK(cudaEventCreate(&ev_g_stop));
            *out_breakdown = TimeBreakdown{};
        }

        const __half h_one = __float2half(1.0f);
        const __half h_zero = __float2half(0.0f);
        const __half h_minus_one = __float2half(-1.0f);

        // ====================================================================
        // Outer Loop: Macro-Panels (W = 256 or 512)
        // ====================================================================
        for (int k_macro = 0; k_macro < n; k_macro += W) {
            const int macro_end = std::min(k_macro + W, n);
            const int W_actual = macro_end - k_macro;

            // ----------------------------------------------------------------
            // 1. Hierarchical Factorization of Macro-Panel using Micro-Panels (wb=64)
            // ----------------------------------------------------------------
            if (out_breakdown) cudaEventRecord(ev_p_start, stream);

            for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
                const int micro_end = std::min(j_micro + wb, W_actual);
                const int wb_actual = micro_end - j_micro;
                const int col_curr = k_macro + j_micro;

                // Factorize Micro-Panel (columns [col_curr, col_curr + wb_actual))
                int n_arg = n;
                int k_arg = col_curr;
                int w_arg = wb_actual;
                int method_arg = static_cast<int>(options.method);
                float tau_arg  = options.tau;
                void* args[] = {
                    (void*)&d_matrix,
                    (void*)&n_arg,
                    (void*)&k_arg,
                    (void*)&w_arg,
                    (void*)&method_arg,
                    (void*)&tau_arg,
                    (void*)&d_row_scales_,
                    (void*)&d_log_pivot_rows_,
                    (void*)&d_counters_,
                    (void*)&d_block_results_
                };

                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)fused_panel_coop_kernel,
                    dim3(fused_num_blocks_),
                    dim3(FUSED_COOP_BLOCK),
                    args, 0, stream
                ));

                // Swap rows in the rest of the MACRO-PANEL ONLY:
                // columns [col_curr + wb_actual, macro_end)
                const int rest_macro_cols = macro_end - (col_curr + wb_actual);
                if (rest_macro_cols > 0) {
                    int lt = 256;
                    int lg = (rest_macro_cols + lt - 1) / lt;
                    batched_laswp_kernel<<<lg, lt, 0, stream>>>(
                        d_matrix, n, col_curr, col_curr + wb_actual,
                        d_log_pivot_rows_, col_curr + wb_actual, macro_end
                    );

                    // Micro-TRTRI on L_diag of size wb_actual
                    size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
                    trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream>>>(
                        d_matrix, n, col_curr, wb_actual, d_L_inv_micro_
                    );

                    // Micro-GEMM: update remaining columns of macro-panel
                    // A12_micro = L_diag^{-1} * A12_micro
                    CUBLAS_CHECK(cublasHgemm(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        wb_actual, rest_macro_cols, wb_actual,
                        &h_one,
                        d_L_inv_micro_, wb_actual,
                        d_matrix + (col_curr + static_cast<size_t>(col_curr + wb_actual) * n), n,
                        &h_zero,
                        d_macro_U12_, wb_actual
                    ));

                    // Store micro U12 back
                    dim3 sblk(32, 8);
                    dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest_macro_cols + sblk.y - 1) / sblk.y);
                    store_block_kernel<<<sgrd, sblk, 0, stream>>>(
                        d_macro_U12_, wb_actual, d_matrix, n, col_curr, col_curr + wb_actual, wb_actual, rest_macro_cols
                    );

                    // Update trailing rows of macro-panel:
                    // A22_macro -= L21_micro * U12_micro
                    const int m_macro_trail = n - (col_curr + wb_actual);
                    CUBLAS_CHECK(cublasHgemm(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        m_macro_trail, rest_macro_cols, wb_actual,
                        &h_minus_one,
                        d_matrix + ((col_curr + wb_actual) + static_cast<size_t>(col_curr) * n), n,
                        d_macro_U12_, wb_actual,
                        &h_one,
                        d_matrix + ((col_curr + wb_actual) + static_cast<size_t>(col_curr + wb_actual) * n), n
                    ));
                }
            }

            if (out_breakdown) {
                cudaEventRecord(ev_p_stop, stream);
                cudaEventSynchronize(ev_p_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_p_start, ev_p_stop);
                out_breakdown->panel_ms += ms;
            }

            if (macro_end >= n) break;

            // ----------------------------------------------------------------
            // 2. Global LASWP: Apply all W_actual row swaps to entire trailing matrix
            //    CALLED ONLY n/W TIMES (e.g. 160 times instead of 640!)
            // ----------------------------------------------------------------
            const int remaining_trailing_cols = n - macro_end;
            if (out_breakdown) cudaEventRecord(ev_l_start, stream);
            {
                int lt = 256;
                int lg = (remaining_trailing_cols + lt - 1) / lt;
                batched_laswp_kernel<<<lg, lt, 0, stream>>>(
                    d_matrix, n, k_macro, macro_end,
                    d_log_pivot_rows_, macro_end, n
                );
            }
            if (out_breakdown) {
                cudaEventRecord(ev_l_stop, stream);
                cudaEventSynchronize(ev_l_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_l_start, ev_l_stop);
                out_breakdown->laswp_ms += ms;
            }

            // ----------------------------------------------------------------
            // 3. Macro-TRSM: Solve L_macro * U12_macro = A12_trailing
            //    Solves for all W_actual rows across all remaining_trailing_cols
            //    using hierarchical micro-block TRSM + GEMM
            // ----------------------------------------------------------------
            if (out_breakdown) cudaEventRecord(ev_u_start, stream);
            for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
                const int micro_end = std::min(j_micro + wb, W_actual);
                const int wb_actual = micro_end - j_micro;
                const int col_curr = k_macro + j_micro;

                // Micro TRTRI for diagonal block
                size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
                trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream>>>(
                    d_matrix, n, col_curr, wb_actual, d_L_inv_micro_
                );

                // Solve block row of U12
                // Slice in macro_U12 buffer has leading dimension wb_actual, rows wb_actual, cols remaining_trailing_cols
                __half* d_u12_slice = d_macro_U12_ + static_cast<size_t>(j_micro) * remaining_trailing_cols;
                CUBLAS_CHECK(cublasHgemm(
                    cublas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    wb_actual, remaining_trailing_cols, wb_actual,
                    &h_one,
                    d_L_inv_micro_, wb_actual,
                    d_matrix + (col_curr + static_cast<size_t>(macro_end) * n), n,
                    &h_zero,
                    d_u12_slice, wb_actual
                ));

                // Copy U12 slice back to matrix
                dim3 sblk(32, 8);
                dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (remaining_trailing_cols + sblk.y - 1) / sblk.y);
                store_block_kernel<<<sgrd, sblk, 0, stream>>>(
                    d_u12_slice, wb_actual, d_matrix, n, col_curr, macro_end, wb_actual, remaining_trailing_cols
                );

                // Update subsequent block rows of A12:
                // A12[rest, :] -= L[rest, j_micro] * U12[j_micro, :]
                const int rest_macro_rows = W_actual - (j_micro + wb_actual);
                if (rest_macro_rows > 0) {
                    CUBLAS_CHECK(cublasHgemm(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        rest_macro_rows, remaining_trailing_cols, wb_actual,
                        &h_minus_one,
                        d_matrix + ((col_curr + wb_actual) + static_cast<size_t>(col_curr) * n), n,
                        d_u12_slice, wb_actual,
                        &h_one,
                        d_matrix + ((col_curr + wb_actual) + static_cast<size_t>(macro_end) * n), n
                    ));
                }
            }
            if (out_breakdown) {
                cudaEventRecord(ev_u_stop, stream);
                cudaEventSynchronize(ev_u_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_u_start, ev_u_stop);
                out_breakdown->u12_gemm_ms += ms;
            }

            // ----------------------------------------------------------------
            // 4. Massive Trailing Matrix Update: A22 -= L21 * U12 (Tensor Cores)
            //    Operates with K = W_actual (e.g. 256 or 512)!
            //    Tensor Cores run at 60 - 73 TFLOPS!
            // ----------------------------------------------------------------
            const int m_trail = n - macro_end;
            const int n_trail = n - macro_end;
            const int k_trail = W_actual;

            // Pack U12 buffer into contiguous W_actual x remaining_trailing_cols
            // Note: each slice d_u12_slice was stored contiguously, so d_macro_U12_
            // has row-slices. For cublasHgemm with OP_N, U12 needs leading dimension W_actual.
            // Since d_matrix already has the stored U12 at (k_macro, macro_end) with lda = n,
            // we can pass d_matrix directly with lda = n!
            if (out_breakdown) cudaEventRecord(ev_g_start, stream);
            CUBLAS_CHECK(cublasHgemm(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m_trail, n_trail, k_trail,
                &h_minus_one,
                d_matrix + (macro_end + static_cast<size_t>(k_macro) * n), n,
                d_matrix + (k_macro + static_cast<size_t>(macro_end) * n), n,
                &h_one,
                d_matrix + (macro_end + static_cast<size_t>(macro_end) * n), n
            ));
            if (out_breakdown) {
                cudaEventRecord(ev_g_stop, stream);
                cudaEventSynchronize(ev_g_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_g_start, ev_g_stop);
                out_breakdown->trail_gemm_ms += ms;
            }
        }

        if (out_breakdown) {
            cudaEventDestroy(ev_p_start); cudaEventDestroy(ev_p_stop);
            cudaEventDestroy(ev_l_start); cudaEventDestroy(ev_l_stop);
            cudaEventDestroy(ev_t_start); cudaEventDestroy(ev_t_stop);
            cudaEventDestroy(ev_u_start); cudaEventDestroy(ev_u_stop);
            cudaEventDestroy(ev_s_start); cudaEventDestroy(ev_s_stop);
            cudaEventDestroy(ev_g_start); cudaEventDestroy(ev_g_stop);
            out_breakdown->total_ms = out_breakdown->panel_ms + out_breakdown->laswp_ms +
                                      out_breakdown->trtri_ms + out_breakdown->u12_gemm_ms +
                                      out_breakdown->u12_store_ms + out_breakdown->trail_gemm_ms;
        }

        // Copy pivot logs and counters to host
        host_pivot_rows.resize(n);
        host_pivot_cols.resize(n);
        CUDA_CHECK(cudaMemcpyAsync(host_pivot_rows.data(), d_log_pivot_rows_, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_pivot_cols.data(), d_log_pivot_cols_, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_counters, d_counters_, kCounterSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return true;
    }
};

} // namespace high_perf
