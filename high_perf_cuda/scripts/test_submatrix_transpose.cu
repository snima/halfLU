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

// General 2D submatrix transpose
__global__ void transpose_submatrix_kernel(
    const __half* __restrict__ idata,
    int ldi,
    __half* __restrict__ odata,
    int ldo,
    int width,  // columns in idata (rows in odata)
    int height  // rows in idata (columns in odata)
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
    int W = 256;
    int M = 1024;
    std::cout << "Testing transpose_submatrix_kernel for " << W << "x" << M << "..." << std::endl;

    __half *d_src, *d_dst;
    size_t src_bytes = static_cast<size_t>(W) * M * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, src_bytes));

    std::vector<__half> h_src(W * M);
    for (int i = 0; i < W * M; ++i) {
        h_src[i] = __float2half(static_cast<float>(i));
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));

    // Transpose from W x M (width=W, height=M) to M x W
    dim3 grid((W + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    transpose_submatrix_kernel<<<grid, block>>>(d_src, M, d_dst, W, W, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_dst(W * M);
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, src_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int c = 0; c < W && ok; ++c) {
        for (int r = 0; r < M && ok; ++r) {
            float v_in = __half2float(h_src[static_cast<size_t>(c) * M + r]);
            float v_out = __half2float(h_dst[static_cast<size_t>(r) * W + c]);
            if (fabsf(v_in - v_out) > 1e-3f) {
                std::cout << "Mismatch at c=" << c << ", r=" << r << ": in=" << v_in << ", out=" << v_out << std::endl;
                ok = false;
            }
        }
    }
    if (ok) std::cout << "SUCCESS: Submatrix transpose is 100% exact!" << std::endl;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
