#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// Bank-conflict-free 2D matrix transpose
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

int main() {
    int n = 2048;
    std::cout << "Testing 2D shared-memory transpose kernel on V100 for n=" << n << "..." << std::endl;

    __half *d_in, *d_out;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    std::vector<__half> h_in(n * n);
    for (int i = 0; i < n * n; ++i) {
        h_in[i] = __float2half(static_cast<float>(i % 100));
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimGrid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int r = 0; r < 50; ++r) {
        transpose_matrix_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, n, n, n, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 50.0f;
    double bw = (2.0 * bytes) / (avg_ms * 1e-3) / 1e9;
    std::cout << "Transpose time: " << avg_ms << " ms, Bandwidth: " << bw << " GB/s" << std::endl;

    // Verify correctness
    std::vector<__half> h_out(n * n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int c = 0; c < n && ok; ++c) {
        for (int r = 0; r < n && ok; ++r) {
            float v_in = __half2float(h_in[c * n + r]);
            float v_out = __half2float(h_out[r * n + c]);
            if (fabsf(v_in - v_out) > 1e-3f) {
                std::cout << "Mismatch at (" << r << ", " << c << "): in=" << v_in << ", out=" << v_out << std::endl;
                ok = false;
            }
        }
    }
    if (ok) std::cout << "SUCCESS: Transpose is 100% numerically exact!" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
