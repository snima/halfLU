#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "include/types.hpp"
#include "include/pivot_kernels.cuh"
#include "include/panel_kernels.cuh"
#include "include/fused_panel_kernel.cuh"
#include "include/high_perf_lu.cuh"
#include "include/hierarchical_lu.cuh"

using namespace high_perf;

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_matrix_kernel(
    const __half* __restrict__ idata,
    __half* __restrict__ odata,
    int width,
    int height,
    int ldi,
    int ldo
) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[static_cast<size_t>(x) * ldi + (y + j)];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[static_cast<size_t>(x) * ldo + (y + j)] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transpose_submatrix_kernel(
    const __half* __restrict__ idata,
    int ldi,
    __half* __restrict__ odata,
    int ldo,
    int width,  // cols in idata
    int height  // rows in idata
) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[static_cast<size_t>(x) * ldi + (y + j)];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[static_cast<size_t>(x) * ldo + (y + j)] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 100% Coalesced Transposed LASWP Kernel
__global__ void batched_laswp_transposed_kernel(
    __half* __restrict__ B, // n x n transposed matrix, ldb = n
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int row_start,
    int row_end
) {
    using VecType = float4; // 8 halfs
    const int num_elements = row_end - row_start;
    const int num_vec = num_elements / 8;
    const int w = k2 - k1;

    for (int i = 0; i < w; ++i) {
        int r1 = k1 + i;
        int r2 = ipiv[k1 + i];
        if (r1 == r2) continue;

        VecType* col1 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r1) * n + row_start);
        VecType* col2 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r2) * n + row_start);

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vec; idx += gridDim.x * blockDim.x) {
            VecType v1 = col1[idx];
            VecType v2 = col2[idx];
            col1[idx] = v2;
            col2[idx] = v1;
        }

        int rem_start = num_vec * 8;
        for (int idx = rem_start + threadIdx.x; idx < num_elements; idx += blockDim.x) {
            __half* c1 = B + static_cast<size_t>(r1) * n + row_start;
            __half* c2 = B + static_cast<size_t>(r2) * n + row_start;
            __half t = c1[idx];
            c1[idx] = c2[idx];
            c2[idx] = t;
        }
        __syncthreads();
    }
}

int main() {
    int n = 2048;
    int W = 256;
    int wb = 64;

    std::cout << "Testing Complete Transposed Hierarchical LU on V100 for n=" << n 
              << ", W=" << W << ", wb=" << wb << "..." << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Generate diagonally dominant test matrix
    std::vector<__half> h_A(n * n);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) {
            float val = dist(rng);
            if (r == c) val += 10.0f;
            h_A[c * n + r] = __float2half(val);
        }
    }
    std::vector<__half> h_A_orig = h_A;

    __half *d_A, *d_B, *d_panel_buf, *d_L_inv_macro, *d_L_inv_micro, *d_u12_buf;
    size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_panel_buf, static_cast<size_t>(n) * W * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_L_inv_macro, W * W * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_L_inv_micro, wb * wb * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u12_buf, static_cast<size_t>(n) * W * sizeof(__half)));

    int *d_log_pivot_rows;
    CUDA_CHECK(cudaMalloc(&d_log_pivot_rows, n * sizeof(int)));
    int init_threads = 256; int init_grid = (n + init_threads - 1) / init_threads;
    init_identity_perm_kernel<<<init_grid, init_threads>>>(d_log_pivot_rows, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long *d_counters;
    CUDA_CHECK(cudaMalloc(&d_counters, kCounterSlots * sizeof(unsigned long long)));

    // Cooperative setup
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fused_panel_coop_kernel, FUSED_COOP_BLOCK, 0));
    int num_sms = 80;
    int fused_blocks = std::min(num_sms * max_blocks_per_sm, FUSED_MAX_BLOCKS);
    volatile BlockPivotResult* d_block_results;
    CUDA_CHECK(cudaMalloc((void**)&d_block_results, fused_blocks * sizeof(BlockPivotResult)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));

    // 1. Initial transpose: B = A^T
    dim3 tgrid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
    dim3 tblk(TILE_DIM, BLOCK_ROWS);
    transpose_matrix_kernel<<<tgrid, tblk>>>(d_A, d_B, n, n, n, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    const __half h_one = __float2half(1.0f);
    const __half h_zero = __float2half(0.0f);
    const __half h_minus_one = __float2half(-1.0f);

    cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // Loop over Macro-Panels
    for (int k_macro = 0; k_macro < n; k_macro += W) {
        const int macro_end = std::min(k_macro + W, n);
        const int W_actual = macro_end - k_macro;
        const int m_panel = n - k_macro;

        // Step a: Extract Macro-Panel from B to d_panel_buf
        // In B: rows [k_macro, macro_end), cols [k_macro, n) -> W_actual rows, m_panel cols, ldi = n
        // Transpose to d_panel_buf: m_panel rows, W_actual cols, ldo = m_panel
        dim3 sgrid((m_panel + TILE_DIM - 1) / TILE_DIM, (W_actual + TILE_DIM - 1) / TILE_DIM);
        transpose_submatrix_kernel<<<sgrid, tblk>>>(
            d_B + k_macro + static_cast<size_t>(k_macro) * n, n,
            d_panel_buf, m_panel,
            m_panel, W_actual
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step b: Factorize Macro-Panel using Micro-Panels inside d_panel_buf
        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            int n_arg = m_panel;
            int k_arg = j_micro;
            int w_arg = wb_actual;
            int method_arg = 0; // PP
            float tau_arg = 1.01f;
            float* null_scales = nullptr;
            void* args[] = {
                (void*)&d_panel_buf,
                (void*)&n_arg,
                (void*)&k_arg,
                (void*)&w_arg,
                (void*)&method_arg,
                (void*)&tau_arg,
                (void*)&null_scales,
                (void*)&d_log_pivot_rows,
                (void*)&d_counters,
                (void*)&d_block_results
            };

            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)fused_panel_coop_kernel,
                dim3(fused_blocks), dim3(FUSED_COOP_BLOCK),
                args, 0, 0
            ));

            const int rest_macro_cols = W_actual - (j_micro + wb_actual);
            if (rest_macro_cols > 0) {
                int lt = 256;
                int lg = (rest_macro_cols + lt - 1) / lt;
                batched_laswp_kernel<<<lg, lt>>>(
                    d_panel_buf, m_panel, j_micro, j_micro + wb_actual,
                    d_log_pivot_rows, j_micro + wb_actual, W_actual
                );

                size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
                trtri_unit_lower_kernel<<<1, 256, smem_bytes>>>(
                    d_panel_buf, m_panel, j_micro, wb_actual, d_L_inv_micro
                );

                CUBLAS_CHECK(cublasHgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    wb_actual, rest_macro_cols, wb_actual,
                    &h_one,
                    d_L_inv_micro, wb_actual,
                    d_panel_buf + (j_micro + static_cast<size_t>(j_micro + wb_actual) * m_panel), m_panel,
                    &h_zero,
                    d_u12_buf, wb_actual
                ));

                dim3 sblk(32, 8);
                dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest_macro_cols + sblk.y - 1) / sblk.y);
                store_block_kernel<<<sgrd, sblk>>>(
                    d_u12_buf, wb_actual, d_panel_buf, m_panel, j_micro, j_micro + wb_actual, wb_actual, rest_macro_cols
                );

                const int m_macro_trail = m_panel - (j_micro + wb_actual);
                CUBLAS_CHECK(cublasHgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m_macro_trail, rest_macro_cols, wb_actual,
                    &h_minus_one,
                    d_panel_buf + ((j_micro + wb_actual) + static_cast<size_t>(j_micro) * m_panel), m_panel,
                    d_u12_buf, wb_actual,
                    &h_one,
                    d_panel_buf + ((j_micro + wb_actual) + static_cast<size_t>(j_micro + wb_actual) * m_panel), m_panel
                ));
            }
        }

        // Offset local pivots in d_log_pivot_rows[0..W_actual) by k_macro, and store at d_log_pivot_rows[k_macro..macro_end)
        std::vector<int> h_macro_pivots(W_actual);
        CUDA_CHECK(cudaMemcpy(h_macro_pivots.data(), d_log_pivot_rows, W_actual * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < W_actual; ++i) h_macro_pivots[i] += k_macro;
        CUDA_CHECK(cudaMemcpy(d_log_pivot_rows + k_macro, h_macro_pivots.data(), W_actual * sizeof(int), cudaMemcpyHostToDevice));

        // Step c: Transpose factorized Macro-Panel back to B
        dim3 bgrid((W_actual + TILE_DIM - 1) / TILE_DIM, (m_panel + TILE_DIM - 1) / TILE_DIM);
        transpose_submatrix_kernel<<<bgrid, tblk>>>(
            d_panel_buf, m_panel,
            d_B + k_macro + static_cast<size_t>(k_macro) * n, n,
            W_actual, m_panel
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        if (macro_end >= n) break;

        // Step d: Coalesced Transposed LASWP on Trailing Matrix
        if (k_macro > 0) {
            batched_laswp_transposed_kernel<<<80 * 4, 256>>>(
                d_B, n, k_macro, macro_end, d_log_pivot_rows, 0, k_macro
            );
        }
        batched_laswp_transposed_kernel<<<80 * 4, 256>>>(
            d_B, n, k_macro, macro_end, d_log_pivot_rows, macro_end, n
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step e: Macro-TRSM on Trailing Matrix
        size_t l_smem = static_cast<size_t>(W_actual) * W_actual * sizeof(float);
        trtri_unit_lower_kernel<<<1, 256, l_smem>>>(
            d_panel_buf, m_panel, 0, W_actual, d_L_inv_macro
        );

        const int n_trail = n - macro_end;
        CUBLAS_CHECK(cublasHgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            n_trail, W_actual, W_actual,
            &h_one,
            d_B + macro_end + static_cast<size_t>(k_macro) * n, n,
            d_L_inv_macro, W_actual,
            &h_zero,
            d_u12_buf, n_trail
        ));

        dim3 ublk(32, 8);
        dim3 ugrd((n_trail + ublk.x - 1) / ublk.x, (W_actual + ublk.y - 1) / ublk.y);
        store_block_kernel<<<ugrd, ublk>>>(
            d_u12_buf, n_trail, d_B, n, macro_end, k_macro, n_trail, W_actual
        );

        // Step f: Trailing Matrix Update in B: B22 -= U12_T * L21_T
        CUBLAS_CHECK(cublasHgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n_trail, n_trail, W_actual,
            &h_minus_one,
            d_B + macro_end + static_cast<size_t>(k_macro) * n, n,
            d_B + k_macro + static_cast<size_t>(macro_end) * n, n,
            &h_one,
            d_B + macro_end + static_cast<size_t>(macro_end) * n, n
        ));
    }

    // Final Transpose: A = B^T
    transpose_matrix_kernel<<<tgrid, tblk>>>(d_B, d_A, n, n, n, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify Backward Error on host
    std::vector<__half> h_LU(n * n);
    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
    std::vector<int> h_pivots(n);
    CUDA_CHECK(cudaMemcpy(h_pivots.data(), d_log_pivot_rows, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<__half> h_PA = h_A_orig;
    for (int i = 0; i < n; ++i) {
        int r1 = i;
        int r2 = h_pivots[i];
        if (r1 != r2) {
            for (int c = 0; c < n; ++c) {
                std::swap(h_PA[c * n + r1], h_PA[c * n + r2]);
            }
        }
    }

    double norm_diff_sq = 0.0;
    double norm_A_sq = 0.0;
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) {
            double orig = __half2float(h_PA[c * n + r]);
            norm_A_sq += orig * orig;

            double recon = 0.0;
            int lim = std::min(r, c);
            for (int k = 0; k <= lim; ++k) {
                double L_rk = (r == k) ? 1.0 : __half2float(h_LU[k * n + r]);
                double U_kc = __half2float(h_LU[c * n + k]);
                recon += L_rk * U_kc;
            }
            double diff = orig - recon;
            norm_diff_sq += diff * diff;
        }
    }

    
    double max_diff_11 = 0, max_diff_21 = 0, max_diff_12 = 0, max_diff_22 = 0;
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) {
            double orig = __half2float(h_PA[c * n + r]);
            double recon = 0.0;
            int lim = std::min(r, c);
            for (int k = 0; k <= lim; ++k) {
                double L_rk = (r == k) ? 1.0 : __half2float(h_LU[k * n + r]);
                double U_kc = __half2float(h_LU[c * n + k]);
                recon += L_rk * U_kc;
            }
            double diff = fabs(orig - recon);
            
            else if (r >= W && c < W) max_diff_21 = std::max(max_diff_21, diff);
            else if (r < W && c >= W) max_diff_12 = std::max(max_diff_12, diff);
            else max_diff_22 = std::max(max_diff_22, diff);
        }
    }
    
    std::cout << "--- h_PA (first 4x4) ---" << std::endl;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            std::cout << __half2float(h_PA[c * n + r]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--- h_LU (first 4x4) ---" << std::endl;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            std::cout << __half2float(h_LU[c * n + r]) << " ";
        }
        std::cout << std::endl;
    }

    
    int max_r_11 = 0, max_c_11 = 0;
    for (int c = 0; c < W; ++c) {
        for (int r = 0; r < W; ++r) {
            double orig = __half2float(h_PA[c * n + r]);
            double recon = 0.0;
            int lim = std::min(r, c);
            for (int k = 0; k <= lim; ++k) {
                double L_rk = (r == k) ? 1.0 : __half2float(h_LU[k * n + r]);
                double U_kc = __half2float(h_LU[c * n + k]);
                recon += L_rk * U_kc;
            }
            double diff = fabs(orig - recon);
            if (diff > max_diff_11) {
                max_diff_11 = diff;
                max_r_11 = r;
                max_c_11 = c;
            }
        }
    }
    std::cout << "Max diff in (0,0) block at (" << max_r_11 << ", " << max_c_11 << ") = " << max_diff_11 << std::endl;

    std::cout << "Max diff in (1,0) block (L21): " << max_diff_21 << std::endl;
    std::cout << "Max diff in (0,1) block (U12): " << max_diff_12 << std::endl;
    std::cout << "Max diff in (1,1) block (A22): " << max_diff_22 << std::endl;

    double backward_error = sqrt(norm_diff_sq) / sqrt(norm_A_sq);
    std::cout << "Backward Error: " << backward_error << std::endl;
    if (backward_error < 2.5e-02) {
        std::cout << "SUCCESS: Transposed Hierarchical LU is 100% numerically sound and verified!" << std::endl;
    } else {
        std::cout << "WARNING: Backward error: " << backward_error << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_panel_buf); cudaFree(d_L_inv_macro); cudaFree(d_L_inv_micro); cudaFree(d_u12_buf);
    cudaFree(d_log_pivot_rows); cudaFree(d_counters); cudaFree((void*)d_block_results);
    cublasDestroy(handle);
    return 0;
}
