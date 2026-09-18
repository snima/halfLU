#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>

// Current Baseline: Column-major LASWP (non-coalesced row swap)
__global__ void batched_laswp_colmajor(
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

// MAGMA-Style: Transposed LASWP
// In transposed matrix B = A^T (dimension n x n):
// Row r1 of A is Column r1 of B!
// Row r2 of A is Column r2 of B!
// Swapping Row r1 and Row r2 of A means:
// Swapping Column r1 and Column r2 of B over rows [col_start, col_end)!
// Since elements of a column are CONTIGUOUS, threads in a warp access
// contiguous 128-bit memory chunks! 100% Coalesced!
__global__ void laswp_transposed_coalesced(
    __half* __restrict__ B, // transposed matrix, ldb = n
    int n,
    int r1,
    int r2,
    int row_start, // corresponds to col_start in A
    int row_end    // corresponds to col_end in A
) {
    // Vectorized 128-bit (8 halfs) per thread
    using VecType = float4; // 16 bytes = 8 halfs
    const int num_elements = row_end - row_start;
    const int num_vec = num_elements / 8;

    VecType* col1 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r1) * n + row_start);
    VecType* col2 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r2) * n + row_start);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vec; idx += gridDim.x * blockDim.x) {
        VecType v1 = col1[idx];
        VecType v2 = col2[idx];
        col1[idx] = v2;
        col2[idx] = v1;
    }

    // Remainder
    int rem_start = num_vec * 8;
    for (int idx = rem_start + threadIdx.x; idx < num_elements; idx += blockDim.x) {
        __half* c1 = B + static_cast<size_t>(r1) * n + row_start;
        __half* c2 = B + static_cast<size_t>(r2) * n + row_start;
        __half t = c1[idx];
        c1[idx] = c2[idx];
        c2[idx] = t;
    }
}

// Batched Transposed LASWP: applies all w swaps in transposed layout
__global__ void batched_laswp_transposed_coalesced(
    __half* __restrict__ B,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int row_start,
    int row_end
) {
    using VecType = float4;
    const int num_elements = row_end - row_start;
    const int num_vec = num_elements / 8;
    const int w = k2 - k1;

    // Grid processes swaps or vectors
    // Each block processes a slice of vectors for swap i
    for (int i = 0; i < w; ++i) {
        int r1 = k1 + i;
        int r2 = ipiv[k1 + i];
        if (r1 == r2) continue;

        VecType* col1 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r1) * n + row_start);
        VecType* col2 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r2) * n + row_start);

        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vec; idx += gridDim.x * blockDim.x) {
            VecType v1 = col1[idx];
            VecType v2 = col2[idx];
            col1[idx] = v2;
            col2[idx] = v1;
        }
        __syncthreads();
    }
}

int main() {
    int n = 40960;
    int w = 512;
    int col_start = 512;
    int col_end = n;
    int num_cols = col_end - col_start;

    std::cout << "=== LASWP Architectural Comparison on Tesla V100 ===" << std::endl;
    std::cout << "Matrix Order n = " << n << ", Panel Width w = " << w << ", Trailing Columns = " << num_cols << std::endl;

    __half *d_A, *d_B;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMemset(d_A, 0, bytes);
    cudaMemset(d_B, 0, bytes);

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

    // 1. Column-major Baseline
    int lt = 256;
    int lg = (num_cols + lt - 1) / lt;
    batched_laswp_colmajor<<<lg, lt>>>(d_A, n, 0, w, d_ipiv, col_start, col_end);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_colmajor<<<lg, lt>>>(d_A, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_colmajor = 0;
    cudaEventElapsedTime(&ms_colmajor, start, stop);
    float avg_colmajor = ms_colmajor / 10.0f;
    std::cout << "1. Current Column-major LASWP:      " << avg_colmajor << " ms" << std::endl;

    // 2. Transposed Coalesced LASWP
    int t_threads = 256;
    int t_blocks = 80 * 4; // 320 blocks to saturate 80 SMs
    batched_laswp_transposed_coalesced<<<t_blocks, t_threads>>>(d_B, n, 0, w, d_ipiv, col_start, col_end);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int rep = 0; rep < 10; ++rep) {
        batched_laswp_transposed_coalesced<<<t_blocks, t_threads>>>(d_B, n, 0, w, d_ipiv, col_start, col_end);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_trans = 0;
    cudaEventElapsedTime(&ms_trans, start, stop);
    float avg_trans = ms_trans / 10.0f;
    std::cout << "2. MAGMA Transposed Coalesced LASWP: " << avg_trans << " ms" << std::endl;
    std::cout << ">>> SPEEDUP: " << (avg_colmajor / avg_trans) << "x faster!" << std::endl;

    // Bandwidth calculation:
    // w swaps of vectors of length num_cols (2 bytes per element, read + write = 4 bytes per element * 2 vectors = 8 bytes)
    double total_bytes = static_cast<double>(w) * num_cols * 4.0 * 2.0;
    double bw_colmajor = (total_bytes / (avg_colmajor * 1e-3)) / 1e9;
    double bw_trans = (total_bytes / (avg_trans * 1e-3)) / 1e9;
    std::cout << "Effective Bandwidth (Column-major):  " << bw_colmajor << " GB/s" << std::endl;
    std::cout << "Effective Bandwidth (Transposed):    " << bw_trans << " GB/s" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ipiv);
    return 0;
}
