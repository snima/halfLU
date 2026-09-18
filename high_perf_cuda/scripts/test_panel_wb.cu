#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include "../include/cuda_utils.cuh"
#include "../include/types.hpp"
#include "../include/panel_kernels.cuh"
#include "../include/fused_panel_kernel.cuh"
#include "../include/transposed_hierarchical_lu.cuh"

using namespace high_perf;

void test_panel_config(int m, int W, int wb, int num_blocks) {
    __half *d_panel;
    CUDA_CHECK(cudaMalloc(&d_panel, static_cast<size_t>(m) * W * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_panel, 0x3c, static_cast<size_t>(m) * W * sizeof(__half)));

    __half *d_L_inv;
    CUDA_CHECK(cudaMalloc(&d_L_inv, 256 * 256 * sizeof(__half)));
    __half *d_u12;
    CUDA_CHECK(cudaMalloc(&d_u12, static_cast<size_t>(m) * 256 * sizeof(__half)));

    int *d_pivots;
    CUDA_CHECK(cudaMalloc(&d_pivots, W * sizeof(int)));
    unsigned long long *d_counters;
    CUDA_CHECK(cudaMalloc(&d_counters, kCounterSlots * sizeof(unsigned long long)));

    BlockPivotResult *d_block_res;
    CUDA_CHECK(cudaMalloc(&d_block_res, num_blocks * sizeof(BlockPivotResult)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    const __half h_one = __float2half(1.0f);
    const __half h_zero = __float2half(0.0f);
    const __half h_minus_one = __float2half(-1.0f);

    cudaEvent_t e1, e2;
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    // Warmup
    for (int j = 0; j < W; j += wb) {
        int wb_act = std::min(wb, W - j);
        int n_arg = m;
        int k_arg = j;
        int w_arg = wb_act;
        int method_arg = 0;
        float tau_arg = 1.0f;
        float *d_row_scales = nullptr;
        void* args[] = {
            (void*)&d_panel, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
            (void*)&method_arg, (void*)&tau_arg, (void*)&d_row_scales,
            (void*)&d_pivots, (void*)&d_counters, (void*)&d_block_res
        };
        cudaLaunchCooperativeKernel((void*)fused_panel_coop_kernel, dim3(num_blocks), dim3(FUSED_COOP_BLOCK), args, 0, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(e1));
    for (int j = 0; j < W; j += wb) {
        int wb_act = std::min(wb, W - j);
        int n_arg = m;
        int k_arg = j;
        int w_arg = wb_act;
        int method_arg = 0; // PP
        float tau_arg = 1.0f;
        float *d_row_scales = nullptr;

        void* args[] = {
            (void*)&d_panel, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
            (void*)&method_arg, (void*)&tau_arg, (void*)&d_row_scales,
            (void*)&d_pivots, (void*)&d_counters, (void*)&d_block_res
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)fused_panel_coop_kernel,
            dim3(num_blocks), dim3(FUSED_COOP_BLOCK),
            args, 0, 0
        ));

        int rest = W - (j + wb_act);
        if (rest > 0) {
            int lt = 256;
            int lg = (rest + lt - 1) / lt;
            batched_laswp_kernel<<<lg, lt, 0, 0>>>(
                d_panel, m, j, j + wb_act, d_pivots, j + wb_act, W
            );

            size_t smem = wb_act * wb_act * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem, 0>>>(
                d_panel, m, j, wb_act, d_L_inv
            );

            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                wb_act, rest, wb_act,
                &h_one,
                d_L_inv, CUDA_R_16F, wb_act,
                d_panel + (j + static_cast<size_t>(j + wb_act) * m), CUDA_R_16F, m,
                &h_zero,
                d_u12, CUDA_R_16F, wb_act,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
            dim3 sblk(32, 8);
            dim3 sgrd((wb_act + sblk.x - 1) / sblk.x, (rest + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk, 0, 0>>>(
                d_u12, wb_act, d_panel, m, j, j + wb_act, wb_act, rest
            );

            int m_trail = m - (j + wb_act);
            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m_trail, rest, wb_act,
                &h_minus_one,
                d_panel + ((j + wb_act) + static_cast<size_t>(j) * m), CUDA_R_16F, m,
                d_u12, CUDA_R_16F, wb_act,
                &h_one,
                d_panel + ((j + wb_act) + static_cast<size_t>(j + wb_act) * m), CUDA_R_16F, m,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
    }
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, e1, e2);

    std::cout << "m = " << m << ", W = " << W << ", wb = " << wb 
              << " -> Total Panel Time = " << total_ms << " ms" << std::endl;

    cudaFree(d_panel);
    cudaFree(d_L_inv);
    cudaFree(d_u12);
    cudaFree(d_pivots);
    cudaFree(d_counters);
    cudaFree(d_block_res);
    cublasDestroy(handle);
}

int main() {
    int m = 40960;
    int W = 1024;
    int num_blocks = 160;

    std::cout << "--- Testing wb = 32, 64, 128 for W = 1024 ---" << std::endl;
    test_panel_config(m, W, 32, num_blocks);
    test_panel_config(m, W, 64, num_blocks);
    test_panel_config(m, W, 128, num_blocks);

    std::cout << "--- Testing W = 512, 1024, 1536, 2048 with optimal wb ---" << std::endl;
    test_panel_config(m, 512, 32, num_blocks);
    test_panel_config(m, 1536, 64, num_blocks);
    test_panel_config(m, 2048, 64, num_blocks);

    return 0;
}
