#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include "../include/cuda_utils.cuh"

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int M = 30000;
    int N = 30000;
    int K = 1024;

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(__half);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(__half);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

    const __half h_one = __float2half(1.0f);
    const __half h_zero = __float2half(0.0f);
    const float f_one = 1.0f;
    const float f_zero = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &h_one, d_A, M, d_B, K, &h_zero, d_C, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1. cublasHgemm
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &h_one, d_A, M, d_B, K, &h_zero, d_C, M);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_hgemm = 0;
    cudaEventElapsedTime(&ms_hgemm, start, stop);
    ms_hgemm /= 5.0f;

    double flops = 2.0 * M * N * K;
    double tflops_hgemm = (flops / (ms_hgemm * 1e-3)) / 1e12;

    // 2. cublasGemmEx (FP16 in, FP32 compute)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                     &f_one, d_A, CUDA_R_16F, M,
                     d_B, CUDA_R_16F, K,
                     &f_zero, d_C, CUDA_R_16F, M,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_gemmex_fp32 = 0;
    cudaEventElapsedTime(&ms_gemmex_fp32, start, stop);
    ms_gemmex_fp32 /= 5.0f;
    double tflops_gemmex_fp32 = (flops / (ms_gemmex_fp32 * 1e-3)) / 1e12;

    // 3. cublasGemmEx (FP16 in, FP16 compute)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                     &h_one, d_A, CUDA_R_16F, M,
                     d_B, CUDA_R_16F, K,
                     &h_zero, d_C, CUDA_R_16F, M,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_gemmex_fp16 = 0;
    cudaEventElapsedTime(&ms_gemmex_fp16, start, stop);
    ms_gemmex_fp16 /= 5.0f;
    double tflops_gemmex_fp16 = (flops / (ms_gemmex_fp16 * 1e-3)) / 1e12;

    std::cout << "GEMM benchmark M=" << M << ", N=" << N << ", K=" << K << " on Tesla V100:" << std::endl;
    std::cout << "  cublasHgemm:              " << ms_hgemm << " ms (" << tflops_hgemm << " TFLOPS)" << std::endl;
    std::cout << "  cublasGemmEx (FP32 acc):  " << ms_gemmex_fp32 << " ms (" << tflops_gemmex_fp32 << " TFLOPS)" << std::endl;
    std::cout << "  cublasGemmEx (FP16 acc):  " << ms_gemmex_fp16 << " ms (" << tflops_gemmex_fp16 << " TFLOPS)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
