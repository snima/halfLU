#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include "../include/cuda_utils.cuh"

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// Current (uncoalesced) transpose
__global__ void transpose_matrix_kernel_old(
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

// 100% Fully Coalesced Transpose
// In column-major: row = index within column (stride 1 in memory).
// threadIdx.x MUST index consecutive rows!
__global__ void transpose_matrix_kernel_coalesced(
    const __half* __restrict__ idata,
    __half* __restrict__ odata,
    int width,
    int height,
    int ldi,
    int ldo
) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    // Reading from idata:
    // row varies with threadIdx.x (consecutive in memory!)
    // col varies with threadIdx.y
    int row_in = blockIdx.y * TILE_DIM + threadIdx.x;
    int col_in = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (row_in < height && (col_in + j) < width) {
            tile[threadIdx.y + j][threadIdx.x] = idata[static_cast<size_t>(col_in + j) * ldi + row_in];
        }
    }
    __syncthreads();

    // Writing to odata (which is transposed: height cols x width rows):
    // In odata: row_out is col_in, col_out is row_in
    // We want consecutive threads (threadIdx.x) to write to consecutive row_out!
    int row_out = blockIdx.x * TILE_DIM + threadIdx.x;
    int col_out = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (row_out < width && (col_out + j) < height) {
            odata[static_cast<size_t>(col_out + j) * ldo + row_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    int n = 40960;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);

    __half *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    transpose_matrix_kernel_old<<<grid, block>>>(d_in, d_out, n, n, n, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Test Old
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        transpose_matrix_kernel_old<<<grid, block>>>(d_in, d_out, n, n, n, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float old_ms = 0;
    cudaEventElapsedTime(&old_ms, start, stop);
    old_ms /= 5.0f;

    // Test Coalesced
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        transpose_matrix_kernel_coalesced<<<grid, block>>>(d_in, d_out, n, n, n, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float coalesced_ms = 0;
    cudaEventElapsedTime(&coalesced_ms, start, stop);
    coalesced_ms /= 5.0f;

    std::cout << "N = " << n << " Matrix Transpose:" << std::endl;
    std::cout << "  Old (uncoalesced):  " << old_ms << " ms" << std::endl;
    std::cout << "  Coalesced:          " << coalesced_ms << " ms" << std::endl;
    std::cout << "  Speedup:            " << (old_ms / coalesced_ms) << "x" << std::endl;

    double bandwidth_gb = (2.0 * bytes) / (coalesced_ms * 1e-3) / 1e9;
    std::cout << "  Coalesced Bandwidth: " << bandwidth_gb << " GB/s" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
