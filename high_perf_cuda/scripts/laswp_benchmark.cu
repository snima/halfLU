#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>

// Baseline batched_laswp_kernel
__global__ void batched_laswp_baseline(
    __half* __restrict__ matrix,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int col_start,
    int col_end
) {
    __shared__ int s_ipiv[1024];
    const int w = k2 - k1;
    for (int idx = threadIdx.x; idx < w; idx += blockDim.x) {
        s_ipiv[idx] = ipiv[k1 + idx];
    }
    __syncthreads();

    const int col = col_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= col_end) return;

    for (int i = 0; i < w; ++i) {
        int r2 = s_ipiv[i];
        int r1 = k1 + i;
        if (r2 != r1) {
            std::size_t idx1 = static_cast<std::size_t>(col) * n + r1;
            std::size_t idx2 = static_cast<std::size_t>(col) * n + r2;
            __half tmp = matrix[idx1];
            matrix[idx1] = matrix[idx2];
            matrix[idx2] = tmp;
        }
    }
}

// Optimized: 4 columns per thread (ILP + spatial reuse)
__global__ void batched_laswp_4col(
    __half* __restrict__ matrix,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int col_start,
    int col_end
) {
    __shared__ int s_ipiv[1024];
    const int w = k2 - k1;
    for (int idx = threadIdx.x; idx < w; idx += blockDim.x) {
        s_ipiv[idx] = ipiv[k1 + idx];
    }
    __syncthreads();

    const int base_col = col_start + (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < w; ++i) {
        int r2 = s_ipiv[i];
        int r1 = k1 + i;
        if (r2 != r1) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int col = base_col + c;
                if (col < col_end) {
                    std::size_t idx1 = static_cast<std::size_t>(col) * n + r1;
                    std::size_t idx2 = static_cast<std::size_t>(col) * n + r2;
                    __half tmp = matrix[idx1];
                    matrix[idx1] = matrix[idx2];
                    matrix[idx2] = tmp;
                }
            }
        }
    }
}

int main() {
    int n = 40960;
    int w = 512;
    int col_start = 512;
    int col_end = n;
    int num_cols = col_end - col_start;

    std::cout << "Benchmarking LASWP on n=" << n << ", w=" << w << ", trailing cols=" << num_cols << std::endl;

    __half* d_matrix;
    cudaMalloc(&d_matrix, static_cast<size_t>(n) * n * sizeof(__half));
    cudaMemset(d_matrix, 0, static_cast<size_t>(n) * n * sizeof(__half));

    std::vector<int> h_ipiv(n);
    std::mt19937 rng(42);
    for (int i = 0; i < n; ++i) {
        std::uniform_int_distribution<int> dist(i, std::min(i + 2000, n - 1));
        h_ipiv[i] = dist(rng);
    }
    int* d_ipiv;
    cudaMalloc(&d_ipiv, n * sizeof(int));
    cudaMemcpy(d_ipiv, h_ipiv.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int lt = 256;
    int lg = (num_cols + lt - 1) / lt;
    batched_laswp_baseline<<<lg, lt>>>(d_matrix, n, 0, w, d_ipiv, col_start, col_end);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_baseline<<<lg, lt>>>(d_matrix, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_baseline = 0;
    cudaEventElapsedTime(&ms_baseline, start, stop);
    std::cout << "Baseline (block=256): " << ms_baseline / 10.0f << " ms" << std::endl;

    lt = 64;
    lg = (num_cols + lt - 1) / lt;
    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_baseline<<<lg, lt>>>(d_matrix, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_b64 = 0;
    cudaEventElapsedTime(&ms_b64, start, stop);
    std::cout << "Baseline (block=64): " << ms_b64 / 10.0f << " ms" << std::endl;

    lt = 32;
    lg = (num_cols + lt - 1) / lt;
    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_baseline<<<lg, lt>>>(d_matrix, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_b32 = 0;
    cudaEventElapsedTime(&ms_b32, start, stop);
    std::cout << "Baseline (block=32): " << ms_b32 / 10.0f << " ms" << std::endl;

    lt = 128;
    lg = (num_cols + 4 * lt - 1) / (4 * lt);
    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_4col<<<lg, lt>>>(d_matrix, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_4col = 0;
    cudaEventElapsedTime(&ms_4col, start, stop);
    std::cout << "Optimized 4 cols/thread: " << ms_4col / 10.0f << " ms" << std::endl;

    cudaFree(d_matrix);
    cudaFree(d_ipiv);
    return 0;
}
