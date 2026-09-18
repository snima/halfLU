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

using namespace high_perf;

int main() {
    int n = 2048;
    int W = 256;
    int wb = 64;

    std::cout << "Debugging Macro-Panel Factorization Difference..." << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

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

    __half *d_A1, *d_panel_buf, *d_L_inv_micro, *d_u12_buf;
    CUDA_CHECK(cudaMalloc(&d_A1, n * n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_panel_buf, n * W * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_L_inv_micro, wb * wb * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u12_buf, n * W * sizeof(__half)));

    int *d_piv1, *d_piv2;
    CUDA_CHECK(cudaMalloc(&d_piv1, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_piv2, n * sizeof(int)));

    unsigned long long *d_cnt1, *d_cnt2;
    CUDA_CHECK(cudaMalloc(&d_cnt1, kCounterSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_cnt2, kCounterSlots * sizeof(unsigned long long)));

    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fused_panel_coop_kernel, FUSED_COOP_BLOCK, 0));
    int num_sms = 80;
    int fused_blocks = std::min(num_sms * max_blocks_per_sm, FUSED_MAX_BLOCKS);
    volatile BlockPivotResult *d_br1, *d_br2;
    CUDA_CHECK(cudaMalloc((void**)&d_br1, fused_blocks * sizeof(BlockPivotResult)));
    CUDA_CHECK(cudaMalloc((void**)&d_br2, fused_blocks * sizeof(BlockPivotResult)));

    CUDA_CHECK(cudaMemcpy(d_A1, h_A.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice));

    // Copy first W columns of h_A into d_panel_buf (leading dimension n)
    CUDA_CHECK(cudaMemcpy(d_panel_buf, h_A.data(), n * W * sizeof(__half), cudaMemcpyHostToDevice));

    // Method 1: HierarchicalLU step 1 (Macro-panel on d_A1)
    const __half h_one = __float2half(1.0f);
    const __half h_zero = __float2half(0.0f);
    const __half h_minus_one = __float2half(-1.0f);

    cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // Run HierarchicalLU micro-panel loop on d_A1
    for (int j_micro = 0; j_micro < W; j_micro += wb) {
        int col_curr = j_micro;
        int wb_actual = wb;
        int n_arg = n;
        int k_arg = col_curr;
        int w_arg = wb_actual;
        int method_arg = 0;
        float tau_arg = 1.01f;
        float* null_scales = nullptr;
        void* args[] = {
            (void*)&d_A1, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
            (void*)&method_arg, (void*)&tau_arg, (void*)&null_scales,
            (void*)&d_piv1, (void*)&d_cnt1, (void*)&d_br1
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)fused_panel_coop_kernel, dim3(fused_blocks), dim3(FUSED_COOP_BLOCK), args, 0, 0));

        int rest = W - (col_curr + wb_actual);
        if (rest > 0) {
            int lt = 256;
            int lg = (rest + lt - 1) / lt;
            batched_laswp_kernel<<<lg, lt>>>(d_A1, n, col_curr, col_curr + wb_actual, d_piv1, col_curr + wb_actual, W);
            size_t smem = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem>>>(d_A1, n, col_curr, wb_actual, d_L_inv_micro);
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, wb_actual, rest, wb_actual, &h_one,
                d_L_inv_micro, wb_actual, d_A1 + (col_curr + static_cast<size_t>(col_curr + wb_actual) * n), n,
                &h_zero, d_u12_buf, wb_actual));
            dim3 sblk(32, 8);
            dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk>>>(d_u12_buf, wb_actual, d_A1, n, col_curr, col_curr + wb_actual, wb_actual, rest);
            int m_trail = n - (col_curr + wb_actual);
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_trail, rest, wb_actual, &h_minus_one,
                d_A1 + ((col_curr + wb_actual) + static_cast<size_t>(col_curr) * n), n,
                d_u12_buf, wb_actual,
                &h_one, d_A1 + ((col_curr + wb_actual) + static_cast<size_t>(col_curr + wb_actual) * n), n));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Method 2: d_panel_buf loop (as in test_transposed_lu_full)
    for (int j_micro = 0; j_micro < W; j_micro += wb) {
        int wb_actual = wb;
        int n_arg = n; // m_panel = n for k_macro = 0
        int k_arg = j_micro;
        int w_arg = wb_actual;
        int method_arg = 0;
        float tau_arg = 1.01f;
        float* null_scales = nullptr;
        void* args[] = {
            (void*)&d_panel_buf, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
            (void*)&method_arg, (void*)&tau_arg, (void*)&null_scales,
            (void*)&d_piv2, (void*)&d_cnt2, (void*)&d_br2
        };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)fused_panel_coop_kernel, dim3(fused_blocks), dim3(FUSED_COOP_BLOCK), args, 0, 0));

        int rest = W - (j_micro + wb_actual);
        if (rest > 0) {
            int lt = 256;
            int lg = (rest + lt - 1) / lt;
            batched_laswp_kernel<<<lg, lt>>>(d_panel_buf, n, j_micro, j_micro + wb_actual, d_piv2, j_micro + wb_actual, W);
            size_t smem = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem>>>(d_panel_buf, n, j_micro, wb_actual, d_L_inv_micro);
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, wb_actual, rest, wb_actual, &h_one,
                d_L_inv_micro, wb_actual, d_panel_buf + (j_micro + static_cast<size_t>(j_micro + wb_actual) * n), n,
                &h_zero, d_u12_buf, wb_actual));
            dim3 sblk(32, 8);
            dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk>>>(d_u12_buf, wb_actual, d_panel_buf, n, j_micro, j_micro + wb_actual, wb_actual, rest);
            int m_trail = n - (j_micro + wb_actual);
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_trail, rest, wb_actual, &h_minus_one,
                d_panel_buf + ((j_micro + wb_actual) + static_cast<size_t>(j_micro) * n), n,
                d_u12_buf, wb_actual,
                &h_one, d_panel_buf + ((j_micro + wb_actual) + static_cast<size_t>(j_micro + wb_actual) * n), n));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare d_A1 and d_panel_buf
    std::vector<__half> h_res1(n * W);
    std::vector<__half> h_res2(n * W);
    CUDA_CHECK(cudaMemcpy(h_res1.data(), d_A1, n * W * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res2.data(), d_panel_buf, n * W * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    int diff_r = -1, diff_c = -1;
    for (int c = 0; c < W; ++c) {
        for (int r = 0; r < n; ++r) {
            float v1 = __half2float(h_res1[c * n + r]);
            float v2 = __half2float(h_res2[c * n + r]);
            float diff = fabsf(v1 - v2);
            if (diff > max_diff) {
                max_diff = diff;
                diff_r = r;
                diff_c = c;
            }
        }
    }
    std::cout << "Macro-panel difference between d_A1 and d_panel_buf: " << max_diff 
              << " at r=" << diff_r << ", c=" << diff_c << std::endl;

    return 0;
}
