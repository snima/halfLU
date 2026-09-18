#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include "../include/cuda_utils.cuh"
#include "../include/types.hpp"
#include "../include/panel_kernels.cuh"
#include "../include/fused_panel_kernel.cuh"
#include "../include/transposed_hierarchical_lu.cuh"

using namespace high_perf;

void test_trsm(int n, int W, int wb_trsm) {
    int macro_end = W;
    int n_trail = n - macro_end;
    int m_panel = n;

    __half *d_B, *d_panel_buf, *d_L_inv, *d_u12;
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(n) * n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_panel_buf, static_cast<size_t>(n) * W * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_L_inv, wb_trsm * wb_trsm * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u12, static_cast<size_t>(n) * wb_trsm * sizeof(__half)));

    CUDA_CHECK(cudaMemset(d_B, 0x3c, static_cast<size_t>(n) * n * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_panel_buf, 0x3c, static_cast<size_t>(n) * W * sizeof(__half)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    const __half h_one = __float2half(1.0f);
    const __half h_zero = __float2half(0.0f);
    const __half h_minus_one = __float2half(-1.0f);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int j_micro = 0; j_micro < W; j_micro += wb_trsm) {
        int wb_act = std::min(wb_trsm, W - j_micro);
        size_t smem_bytes = static_cast<size_t>(wb_act) * wb_act * sizeof(float);
        trtri_unit_lower_kernel<<<1, 256, smem_bytes>>>(
            d_panel_buf, m_panel, j_micro, wb_act, d_L_inv
        );
        __half* A_j_T = d_B + macro_end + static_cast<size_t>(j_micro) * n;
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_trail, wb_act, wb_act, &h_one, A_j_T, n, d_L_inv, wb_act, &h_zero, d_u12, n_trail);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 5; ++iter) {
        for (int j_micro = 0; j_micro < W; j_micro += wb_trsm) {
            const int micro_end = std::min(j_micro + wb_trsm, W);
            const int wb_actual = micro_end - j_micro;

            size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem_bytes>>>(
                d_panel_buf, m_panel, j_micro, wb_actual, d_L_inv
            );

            __half* A_j_T = d_B + macro_end + static_cast<size_t>(j_micro) * n;
            CUBLAS_CHECK(cublasHgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                n_trail, wb_actual, wb_actual,
                &h_one,
                A_j_T, n,
                d_L_inv, wb_actual,
                &h_zero,
                d_u12, n_trail
            ));

            dim3 sblk(32, 8);
            dim3 sgrd((n_trail + sblk.x - 1) / sblk.x, (wb_actual + sblk.y - 1) / sblk.y);
            store_block_kernel<<<sgrd, sblk>>>(
                d_u12, n_trail, d_B, n, macro_end, j_micro, n_trail, wb_actual
            );

            const int rest_macro_rows = W - (j_micro + wb_actual);
            if (rest_macro_rows > 0) {
                __half* L_sub_T = d_B + j_micro + static_cast<size_t>(j_micro + wb_actual) * n;
                __half* A_rest_T = d_B + macro_end + static_cast<size_t>(j_micro + wb_actual) * n;

                CUBLAS_CHECK(cublasHgemm(
                    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n_trail, rest_macro_rows, wb_actual,
                    &h_minus_one,
                    d_u12, n_trail,
                    L_sub_T, n,
                    &h_one,
                    A_rest_T, n
                ));
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 5.0f;

    std::cout << "TRSM for n_trail = " << n_trail << ", W = " << W << ", wb_trsm = " << wb_trsm 
              << " -> Time = " << elapsed << " ms" << std::endl;

    cudaFree(d_B);
    cudaFree(d_panel_buf);
    cudaFree(d_L_inv);
    cudaFree(d_u12);
    cublasDestroy(handle);
}

int main() {
    int n = 40960;
    int W = 1024;
    test_trsm(n, W, 32);
    test_trsm(n, W, 64);
    test_trsm(n, W, 128);
    return 0;
}
