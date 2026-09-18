#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    int M = 30000;
    int N = 30000;
    int max_K = 512;

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)M * max_K * sizeof(__half));
    cudaMalloc(&d_B, (size_t)max_K * N * sizeof(__half));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(__half));
    cudaMemset(d_A, 0x3c, (size_t)M * max_K * sizeof(__half)); // 1.0 in FP16
    cudaMemset(d_B, 0x3c, (size_t)max_K * N * sizeof(__half));
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half));

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "==========================================================" << std::endl;
    std::cout << " Tesla V100 cublasHgemm (Tensor Core) Throughput vs K" << std::endl;
    std::cout << " Matrix dimensions: M = " << M << ", N = " << N << std::endl;
    std::cout << "==========================================================" << std::endl;

    for (int K : {32, 64, 128, 256, 512}) {
        // Warmup
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
        cudaDeviceSynchronize();

        int iters = 3;
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops = (flops / (ms * 1e-3)) / 1e12;

        std::cout << " K = " << std::setw(3) << K 
                  << "  |  Time: " << std::setw(6) << std::fixed << std::setprecision(2) << ms << " ms"
                  << "  |  Throughput: " << std::setw(6) << std::fixed << std::setprecision(2) << tflops << " TFLOPS"
                  << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
