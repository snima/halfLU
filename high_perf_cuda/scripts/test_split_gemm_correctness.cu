#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../include/cuda_utils.cuh"

int main() {
    int n = 4096;
    int k_macro = 0;
    int W_actual = 512;
    int macro_end = k_macro + W_actual;
    int n_trail = n - macro_end; // 3584
    int W_next = 512;
    int rest_rows = n_trail - W_next; // 3072

    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    std::vector<__half> h_B_orig(n * n);
    for (int i = 0; i < n * n; ++i) {
        h_B_orig[i] = __float2half(static_cast<float>((i * 13 + 7) % 500) / 250.0f - 1.0f);
    }

    __half *d_B1, *d_B2;
    CUDA_CHECK(cudaMalloc(&d_B1, bytes));
    CUDA_CHECK(cudaMalloc(&d_B2, bytes));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B_orig.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B_orig.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const __half h_one = __float2half(1.0f);
    const __half h_minus_one = __float2half(-1.0f);

    // Method 1: Current standard 2-GEMM split (GEMM 1 + full GEMM 2)
    // GEMM 1: W_next x n_trail
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        W_next, n_trail, W_actual,
        &h_minus_one,
        d_B1 + macro_end + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
        d_B1 + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        &h_one,
        d_B1 + macro_end + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    // GEMM 2: rest_rows x n_trail
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        rest_rows, n_trail, W_actual,
        &h_minus_one,
        d_B1 + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
        d_B1 + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        &h_one,
        d_B1 + (macro_end + W_next) + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Method 2: Advanced 3-GEMM Quadrant split (GEMM 1 + GEMM 2a + GEMM 2b)
    // GEMM 1: W_next x n_trail
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        W_next, n_trail, W_actual,
        &h_minus_one,
        d_B2 + macro_end + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
        d_B2 + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        &h_one,
        d_B2 + macro_end + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    // GEMM 2a (TRSM Dependency Stripe): rest_rows x W_next
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        rest_rows, W_next, W_actual,
        &h_minus_one,
        d_B2 + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
        d_B2 + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        &h_one,
        d_B2 + (macro_end + W_next) + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    // GEMM 2b (Bulk Trailing Core): rest_rows x rest_rows
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        rest_rows, rest_rows, W_actual,
        &h_minus_one,
        d_B2 + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
        d_B2 + k_macro + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
        &h_one,
        d_B2 + (macro_end + W_next) + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_B1(n * n), h_B2(n * n);
    CUDA_CHECK(cudaMemcpy(h_B1.data(), d_B1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B2.data(), d_B2, bytes, cudaMemcpyDeviceToHost));

    int diff_count = 0;
    for (int i = 0; i < n * n; ++i) {
        uint16_t b1 = *reinterpret_cast<uint16_t*>(&h_B1[i]);
        uint16_t b2 = *reinterpret_cast<uint16_t*>(&h_B2[i]);
        if (b1 != b2) {
            diff_count++;
            if (diff_count <= 5) {
                std::cout << "Diff at " << i << ": b1=" << __half2float(h_B1[i]) << ", b2=" << __half2float(h_B2[i]) << std::endl;
            }
        }
    }

    std::cout << "3-GEMM Quadrant Split Test: " << (diff_count == 0 ? "PASSED 100% BIT-EXACT!" : "FAILED") 
              << " (" << diff_count << " mismatches out of " << n*n << ")" << std::endl;

    cudaFree(d_B1);
    cudaFree(d_B2);
    cublasDestroy(handle);
    return diff_count == 0 ? 0 : 1;
}
