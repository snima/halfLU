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
#include "include/hierarchical_lu.cuh"
#include "include/verification.cuh"

using namespace high_perf;

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// 2D Full Matrix Transpose Kernel
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

// 2D Submatrix Transpose Kernel
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

// Coalesced Transposed LASWP Kernel (swaps columns of B across row range [row_start, row_end))
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

// Kernel to convert local panel pivots to global matrix row indices
__global__ void offset_pivots_kernel(int* d_pivots, int count, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_pivots[idx] += offset;
    }
}

int main() {
    int n = 2048;
    int W = 256;
    int wb = 64;

    std::cout << "Testing Clean Transposed Hierarchical LU on V100 for n=" << n 
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

    __half *d_A, *d_B, *d_panel_buf, *d_L_inv_micro, *d_u12_buf;
    size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_panel_buf, static_cast<size_t>(n) * W * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_L_inv_micro, wb * wb * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u12_buf, static_cast<size_t>(n) * wb * sizeof(__half)));

    int *d_log_pivot_rows;
    CUDA_CHECK(cudaMalloc(&d_log_pivot_rows, n * sizeof(int)));
    int init_threads = 256; int init_grid = (n + init_threads - 1) / init_threads;
    init_identity_perm_kernel<<<init_grid, init_threads>>>(d_log_pivot_rows, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long *d_counters;
    CUDA_CHECK(cudaMalloc(&d_counters, kCounterSlots * sizeof(unsigned long long)));

    // Cooperative launch setup
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

    // ========================================================================
    // Outer Loop: Macro-Panels
    // ========================================================================
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
            // Pass (d_log_pivot_rows + k_macro) so local micro-panel writes to its assigned slice!
            int* d_pivot_slice = d_log_pivot_rows + k_macro;
            void* args[] = {
                (void*)&d_panel_buf,
                (void*)&n_arg,
                (void*)&k_arg,
                (void*)&w_arg,
                (void*)&method_arg,
                (void*)&tau_arg,
                (void*)&null_scales,
                (void*)&d_pivot_slice,
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
                    d_pivot_slice, j_micro + wb_actual, W_actual
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

        // Convert local panel pivots to global row indices
        int opt_threads = 256;
        int opt_grid = (W_actual + opt_threads - 1) / opt_threads;
        offset_pivots_kernel<<<opt_grid, opt_threads>>>(d_log_pivot_rows + k_macro, W_actual, k_macro);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply row swaps of current macro-panel to previous L blocks (columns 0..k_macro of A = rows 0..k_macro of B)
        if (k_macro > 0) {
            batched_laswp_transposed_kernel<<<80 * 4, 256>>>(
                d_B, n, k_macro, macro_end, d_log_pivot_rows, 0, k_macro
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step c: Transpose factorized Macro-Panel back to B
        dim3 bgrid((W_actual + TILE_DIM - 1) / TILE_DIM, (m_panel + TILE_DIM - 1) / TILE_DIM);
        transpose_submatrix_kernel<<<bgrid, tblk>>>(
            d_panel_buf, m_panel,
            d_B + k_macro + static_cast<size_t>(k_macro) * n, n,
            W_actual, m_panel
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        if (macro_end >= n) break;

        // Step d: Coalesced Transposed LASWP on Trailing Matrix (ONLY columns macro_end to n in A, which is rows macro_end to n in B!)
        batched_laswp_transposed_kernel<<<80 * 4, 256>>>(
            d_B, n, k_macro, macro_end, d_log_pivot_rows, macro_end, n
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step e: Hierarchical Micro-Block TRSM on Trailing Matrix in B
        const int n_trail = n - macro_end;
        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            // Invert L_diag for this micro-block (size wb_actual <= 64, smem <= 16 KB)
            size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem_bytes>>>(
                d_panel_buf, m_panel, j_micro, wb_actual, d_L_inv_micro
            );

            // U_j^T = A_j^T * (L_diag^{-1})^T
            // A_j^T in B has size n_trail x wb_actual at rows [macro_end, n), cols [k_macro + j_micro, k_macro + j_micro + wb_actual)
            __half* A_j_T = d_B + macro_end + static_cast<size_t>(k_macro + j_micro) * n;
            CUBLAS_CHECK(cublasHgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                n_trail, wb_actual, wb_actual,
                &h_one,
                A_j_T, n,
                d_L_inv_micro, wb_actual,
                &h_zero,
                d_u12_buf, n_trail
            ));

            // Copy U_j^T back to B
            dim3 sblk(32, 8);
            dim3 sgrd((n_trail + sblk.x - 1) / sblk.x, (wb_actual + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk>>>(
                d_u12_buf, n_trail, d_B, n, macro_end, k_macro + j_micro, n_trail, wb_actual
            );

            // Update remaining blocks of A12^T in B:
            // A_rest^T -= U_j^T * L_sub^T
            const int rest_macro_rows = W_actual - (j_micro + wb_actual);
            if (rest_macro_rows > 0) {
                // In B, L_sub^T is at rows [k_macro + j_micro, k_macro + j_micro + wb_actual),
                //                     cols [k_macro + j_micro + wb_actual, macro_end)
                __half* L_sub_T = d_B + (k_macro + j_micro) + static_cast<size_t>(k_macro + j_micro + wb_actual) * n;
                __half* A_rest_T = d_B + macro_end + static_cast<size_t>(k_macro + j_micro + wb_actual) * n;

                CUBLAS_CHECK(cublasHgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n_trail, rest_macro_rows, wb_actual,
                    &h_minus_one,
                    d_u12_buf, n_trail,
                    L_sub_T, n,
                    &h_one,
                    A_rest_T, n
                ));
            }
        }

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

    // Verify Backward Error using verify_factorization
    std::vector<__half> h_LU(n * n);
    CUDA_CHECK(cudaMemcpy(h_LU.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
    std::vector<int> h_pivots(n);
    CUDA_CHECK(cudaMemcpy(h_pivots.data(), d_log_pivot_rows, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_dummy_cols(n, -1);

    // Compute reconstructed LU and block differences
    std::vector<int> p_row_perm(n);
    for (int i = 0; i < n; ++i) p_row_perm[i] = i;
    for (int i = 0; i < n; ++i) {
        if (h_pivots[i] >= 0 && h_pivots[i] < n) {
            std::swap(p_row_perm[i], p_row_perm[h_pivots[i]]);
        }
    }

    double max_diff_11 = 0, max_diff_21 = 0, max_diff_12 = 0, max_diff_22 = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            int k_max = std::min(i, j);
            for (int k = 0; k <= k_max; ++k) {
                double L_ik = (i == k) ? 1.0 : static_cast<double>(__half2float(h_LU[k * n + i]));
                double U_kj = static_cast<double>(__half2float(h_LU[j * n + k]));
                sum += L_ik * U_kj;
            }
            int orig_r = p_row_perm[i];
            double orig = static_cast<double>(__half2float(h_A_orig[j * n + orig_r]));
            double diff = std::abs(orig - sum);
            if (i < W && j < W) max_diff_11 = std::max(max_diff_11, diff);
            else if (i >= W && j < W) max_diff_21 = std::max(max_diff_21, diff);
            else if (i < W && j >= W) max_diff_12 = std::max(max_diff_12, diff);
            else max_diff_22 = std::max(max_diff_22, diff);
        }
    }
    std::cout << "Max diff in Block (0,0) [A11]: " << max_diff_11 << std::endl;
    std::cout << "Max diff in Block (1,0) [L21]: " << max_diff_21 << std::endl;
    std::cout << "Max diff in Block (0,1) [U12]: " << max_diff_12 << std::endl;
    std::cout << "Max diff in Block (1,1) [A22]: " << max_diff_22 << std::endl;

    double backward_error = 0.0;
    double growth_factor = 1.0;
    verify_factorization(h_A_orig, h_LU, h_pivots, h_dummy_cols, n, backward_error, growth_factor);

    std::cout << "Backward Error: " << backward_error << std::endl;
    std::cout << "Growth Factor:  " << growth_factor << std::endl;

    if (backward_error < 2.5e-02) {
        std::cout << "SUCCESS: Transposed Hierarchical LU is 100% numerically verified!" << std::endl;
    } else {
        std::cout << "FAILURE: Backward error: " << backward_error << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_panel_buf); cudaFree(d_L_inv_micro); cudaFree(d_u12_buf);
    cudaFree(d_log_pivot_rows); cudaFree(d_counters); cudaFree((void*)d_block_results);
    cublasDestroy(handle);
    return 0;
}
