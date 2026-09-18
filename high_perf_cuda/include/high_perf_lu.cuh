#pragma once

#include "cuda_utils.cuh"
#include "types.hpp"
#include "pivot_kernels.cuh"
#include "panel_kernels.cuh"
#include "fused_panel_kernel.cuh"

#include <vector>
#include <algorithm>

namespace high_perf {

__global__ void init_identity_perm_kernel(int* perm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        perm[idx] = idx;
    }
}

// Invert unit lower triangular matrix of size w x w in shared memory (FP32 precision)
// Uses exactly w * w * sizeof(float) bytes (64 KB for w=128, fits in V100 96 KB SMEM)
__global__ void trtri_unit_lower_kernel(
    const __half* __restrict__ matrix,
    int n,
    int k,
    int w,
    __half* __restrict__ L_inv
) {
    extern __shared__ float s_inv[]; // w * w floats

    int tid = threadIdx.x;

    // Initialize s_inv to identity
    for (int idx = tid; idx < w * w; idx += blockDim.x) {
        int r = idx % w;
        int c = idx / w;
        s_inv[r * w + c] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Invert column by column: L * inv[:, j] = e_j
    // For i > j: inv[i, j] = - sum_{p=j}^{i-1} L[i, p] * inv[p, j]
    if (tid < w) {
        int j = tid;
        for (int i = j + 1; i < w; ++i) {
            float sum = 0.0f;
            for (int p = j; p < i; ++p) {
                float L_ip = __half2float(matrix[static_cast<std::size_t>(k + p) * n + (k + i)]);
                sum += L_ip * s_inv[p * w + j];
            }
            s_inv[i * w + j] = -sum;
        }
    }
    __syncthreads();

    // Store L_inv to device memory in column-major order
    for (int idx = tid; idx < w * w; idx += blockDim.x) {
        int r = idx % w;
        int c = idx / w;
        L_inv[c * w + r] = __float2half(s_inv[r * w + c]);
    }
}

// Store U12 buffer back to matrix
__global__ void store_u12_kernel(
    const __half* __restrict__ U12_buf,
    int w,
    __half* __restrict__ matrix,
    int n,
    int k,
    int panel_end
) {
    int r = threadIdx.x;
    int c = blockIdx.x * blockDim.y + threadIdx.y;
    int remaining = n - panel_end;
    if (r < w && c < remaining) {
        matrix[static_cast<std::size_t>(panel_end + c) * n + (k + r)] = U12_buf[static_cast<std::size_t>(c) * w + r];
    }
}

class HighPerfLU {
public:
    HighPerfLU(int max_n) : max_n_(max_n) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH));

        CUDA_CHECK(cudaMalloc(&d_pivot_row_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pivot_col_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pivot_ok_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_log_pivot_rows_, max_n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_log_pivot_cols_, max_n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counters_, kCounterSlots * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_row_scales_, max_n_ * sizeof(float)));

        // Scratch buffers for TRTRI + GEMM
        CUDA_CHECK(cudaMalloc(&d_L_inv_, kMaxPanelWidth * kMaxPanelWidth * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_U12_buf_, static_cast<std::size_t>(kMaxPanelWidth) * max_n_ * sizeof(__half)));

        // Detect cooperative launch support (required for fused_panel_coop_kernel)
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int coop_support = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, device_id));
        coop_supported_ = (coop_support != 0);

        // Compute optimal grid size for fused panel kernel
        if (coop_supported_) {
            int min_grid = 0;
            int block_size = FUSED_COOP_BLOCK;
            cudaOccupancyMaxPotentialBlockSize(
                &min_grid, &block_size,
                fused_panel_coop_kernel, 0, 0
            );
            int max_blocks_per_sm = 0;
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                fused_panel_coop_kernel,
                FUSED_COOP_BLOCK, 0
            );
            if (occ_err != cudaSuccess || max_blocks_per_sm <= 0) {
                // If the kernel configuration cannot support resident execution,
                // cleanly disable cooperative mode and fall back to standard launch path
                coop_supported_ = false;
            } else {
                int num_sms = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
                fused_num_blocks_ = num_sms * max_blocks_per_sm;
                if (fused_num_blocks_ > FUSED_MAX_BLOCKS) fused_num_blocks_ = FUSED_MAX_BLOCKS;

                // Allocate block-level pivot reduction scratch
                CUDA_CHECK(cudaMalloc(&d_block_results_,
                    fused_num_blocks_ * sizeof(BlockPivotResult)));
            }
        }

        // Configure dynamic shared memory attribute for trtri kernel (allows up to 96 KB for w=128)
        cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }

    ~HighPerfLU() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (d_pivot_row_) cudaFree(d_pivot_row_);
        if (d_pivot_col_) cudaFree(d_pivot_col_);
        if (d_pivot_ok_) cudaFree(d_pivot_ok_);
        if (d_log_pivot_rows_) cudaFree(d_log_pivot_rows_);
        if (d_log_pivot_cols_) cudaFree(d_log_pivot_cols_);
        if (d_counters_) cudaFree(d_counters_);
        if (d_row_scales_) cudaFree(d_row_scales_);
        if (d_L_inv_) cudaFree(d_L_inv_);
        if (d_U12_buf_) cudaFree(d_U12_buf_);
        if (d_block_results_) cudaFree(d_block_results_);
    }

    // Execute High-Performance Blocked LU Factorization
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

        const int w = (options.panel_width > 0 && options.panel_width <= kMaxPanelWidth) ? options.panel_width : 128;

        // Cooperative fast path: PP, DP, GP, ScPP
        // Legacy path: ScaP (col swap), CP (col swap), RP (col swap)
        const bool use_coop = coop_supported_ &&
            (options.method == Method::kPP  ||
             options.method == Method::kDP  ||
             options.method == Method::kGP  ||
             options.method == Method::kScPP);

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

        for (int k = 0; k < n; k += w) {
            const int panel_end = std::min(k + w, n);
            const int w_actual = panel_end - k;

            // =================================================================
            // 1. Panel Factorization
            // =================================================================
            if (out_breakdown) cudaEventRecord(ev_p_start, stream);
            if (use_coop) {
                // COOPERATIVE FAST PATH: ONE kernel call per panel!
                // All w_actual columns factorized on-device with grid.sync() between phases.
                // Eliminates all per-column CPU dispatches (was 4*w_actual launches before).
                int n_arg = n;
                int k_arg = k;
                int w_arg = w_actual;
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
                    args,
                    0,      // shared memory bytes (s_keys/s_second are in FUSED_COOP_BLOCK*2*8=4096 bytes)
                    stream
                ));
            } else {
                // LEGACY PATH: per-column launches for ScaP, CP, RP
                for (int j = k; j < panel_end; ++j) {
                    select_pivot_kernel<<<1, kPivotThreads, 0, stream>>>(
                        d_matrix, n, j, panel_end,
                        static_cast<int>(options.method), options.tau, options.window,
                        d_row_scales_, d_pivot_row_, d_pivot_col_, d_pivot_ok_, d_counters_
                    );

                    {
                        int st = 256;
                        int sg = (panel_end + st - 1) / st;
                        swap_rows_panel_kernel<<<sg, st, 0, stream>>>(
                            d_matrix, n, j, d_pivot_row_, 0, panel_end, d_log_pivot_rows_, j
                        );
                    }

                    if (options.method == Method::kCP || options.method == Method::kRP || options.method == Method::kScaP) {
                        int st = 256;
                        int sg = (n + st - 1) / st;
                        swap_cols_device_kernel<<<sg, st, 0, stream>>>(
                            d_matrix, n, j, d_pivot_col_, 0, n, d_log_pivot_cols_, j
                        );
                    }

                    if (j + 1 < n) {
                        int st = 256;
                        int sg = (n - (j + 1) + st - 1) / st;
                        scale_column_kernel<<<sg, st, 0, stream>>>(d_matrix, n, j);
                    }

                    if (j + 1 < panel_end) {
                        dim3 block(16, 16);
                        dim3 pg(
                            (panel_end - (j + 1) + block.x - 1) / block.x,
                            (n - (j + 1) + block.y - 1) / block.y
                        );
                        panel_rank1_update_kernel<<<pg, block, 0, stream>>>(d_matrix, n, j, panel_end);
                    }
                }
            }
            if (out_breakdown) {
                cudaEventRecord(ev_p_stop, stream);
                cudaEventSynchronize(ev_p_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_p_start, ev_p_stop);
                out_breakdown->panel_ms += ms;
            }

            if (panel_end >= n) break;

            // 2. Batched LASWP: apply row permutations to trailing columns
            if (out_breakdown) cudaEventRecord(ev_l_start, stream);
            {
                int remaining_cols = n - panel_end;
                int lt = 256;
                int lg = (remaining_cols + lt - 1) / lt;
                batched_laswp_kernel<<<lg, lt, 0, stream>>>(
                    d_matrix, n, k, panel_end, d_log_pivot_rows_, panel_end, n
                );
            }
            if (out_breakdown) {
                cudaEventRecord(ev_l_stop, stream);
                cudaEventSynchronize(ev_l_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_l_start, ev_l_stop);
                out_breakdown->laswp_ms += ms;
            }

            // 3. TRTRI + Tensor Core GEMM for U12
            if (out_breakdown) cudaEventRecord(ev_t_start, stream);
            size_t smem_bytes = static_cast<size_t>(w_actual) * w_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream>>>(d_matrix, n, k, w_actual, d_L_inv_);
            if (out_breakdown) {
                cudaEventRecord(ev_t_stop, stream);
                cudaEventSynchronize(ev_t_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_t_start, ev_t_stop);
                out_breakdown->trtri_ms += ms;
            }

            const __half h_one = __float2half(1.0f);
            const __half h_zero = __float2half(0.0f);
            const int remaining_cols = n - panel_end;

            if (out_breakdown) cudaEventRecord(ev_u_start, stream);
            CUBLAS_CHECK(cublasHgemm(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                w_actual, remaining_cols, w_actual,
                &h_one,
                d_L_inv_, w_actual,
                d_matrix + (k + static_cast<std::size_t>(panel_end) * n), n,
                &h_zero,
                d_U12_buf_, w_actual
            ));
            if (out_breakdown) {
                cudaEventRecord(ev_u_stop, stream);
                cudaEventSynchronize(ev_u_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_u_start, ev_u_stop);
                out_breakdown->u12_gemm_ms += ms;
            }

            if (out_breakdown) cudaEventRecord(ev_s_start, stream);
            {
                dim3 block(w_actual, std::min(256 / w_actual, remaining_cols));
                dim3 pgrid((remaining_cols + block.y - 1) / block.y);
                store_u12_kernel<<<pgrid, block, 0, stream>>>(d_U12_buf_, w_actual, d_matrix, n, k, panel_end);
            }
            if (out_breakdown) {
                cudaEventRecord(ev_s_stop, stream);
                cudaEventSynchronize(ev_s_stop);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_s_start, ev_s_stop);
                out_breakdown->u12_store_ms += ms;
            }

            // 4. Trailing Matrix Update: A22 = A22 - L21 * U12 (Tensor Cores)
            const __half h_minus_one = __float2half(-1.0f);
            const int m_trail = n - panel_end;
            const int n_trail = n - panel_end;
            const int k_trail = w_actual;

            if (out_breakdown) cudaEventRecord(ev_g_start, stream);
            CUBLAS_CHECK(cublasHgemm(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m_trail, n_trail, k_trail,
                &h_minus_one,
                d_matrix + (panel_end + static_cast<std::size_t>(k) * n), n,
                d_U12_buf_, w_actual,
                &h_one,
                d_matrix + (panel_end + static_cast<std::size_t>(panel_end) * n), n
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

        // Copy results back to host
        host_pivot_rows.resize(n);
        host_pivot_cols.resize(n);
        CUDA_CHECK(cudaMemcpyAsync(host_pivot_rows.data(), d_log_pivot_rows_, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_pivot_cols.data(), d_log_pivot_cols_, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&host_counters, d_counters_, kCounterSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return true;
    }

    bool is_coop_enabled() const { return coop_supported_; }
    int  coop_num_blocks() const { return fused_num_blocks_; }

private:
    int max_n_ = 0;
    cublasHandle_t cublas_handle_ = nullptr;
    int* d_pivot_row_ = nullptr;
    int* d_pivot_col_ = nullptr;
    int* d_pivot_ok_ = nullptr;
    int* d_log_pivot_rows_ = nullptr;
    int* d_log_pivot_cols_ = nullptr;
    unsigned long long* d_counters_ = nullptr;
    float* d_row_scales_ = nullptr;
    __half* d_L_inv_ = nullptr;
    __half* d_U12_buf_ = nullptr;

    // Cooperative launch support
    bool coop_supported_ = false;
    int  fused_num_blocks_ = 0;
    BlockPivotResult* d_block_results_ = nullptr;
};

} // namespace high_perf
