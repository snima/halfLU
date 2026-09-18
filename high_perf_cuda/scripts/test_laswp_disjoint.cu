#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// 100% Race-Free, Fully Coalesced Transposed LASWP Kernel via Disjoint Row Partitioning
// Each thread processes disjoint rows of B (consecutive rows across warp threads).
// No two blocks or warps ever touch the same row of B!
// Zero cross-block synchronization needed, provably zero data races!
__global__ void batched_laswp_transposed_disjoint_kernel(
    __half* __restrict__ B,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int row_start,
    int row_end
) {
    const int num_rows = row_end - row_start;
    const int num_pairs = num_rows / 2; // using __half2
    const int w = k2 - k1;

    // Grid-stride loop over disjoint pairs of rows
    for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x; pair_idx < num_pairs; pair_idx += gridDim.x * blockDim.x) {
        int r_base = row_start + pair_idx * 2;

        for (int i = 0; i < w; ++i) {
            int r1 = k1 + i;
            int r2 = ipiv[k1 + i];
            if (r1 == r2) continue;

            __half2* p1 = reinterpret_cast<__half2*>(B + static_cast<size_t>(r1) * n + r_base);
            __half2* p2 = reinterpret_cast<__half2*>(B + static_cast<size_t>(r2) * n + r_base);

            __half2 v1 = *p1;
            __half2 v2 = *p2;
            *p1 = v2;
            *p2 = v1;
        }
    }

    // Remainder odd row if num_rows is odd
    if ((num_rows & 1) != 0) {
        int odd_row = row_end - 1;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int i = 0; i < w; ++i) {
                int r1 = k1 + i;
                int r2 = ipiv[k1 + i];
                if (r1 == r2) continue;
                __half* p1 = B + static_cast<size_t>(r1) * n + odd_row;
                __half* p2 = B + static_cast<size_t>(r2) * n + odd_row;
                __half t = *p1;
                *p1 = *p2;
                *p2 = t;
            }
        }
    }
}

int main() {
    int n = 40960;
    int w = 1024;
    int row_start = 1024;
    int row_end = 40960;

    std::cout << "Testing batched_laswp_transposed_disjoint_kernel on V100..." << std::endl;

    __half* d_B;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));

    int* d_ipiv;
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    std::vector<int> h_ipiv(n);
    for (int i = 0; i < n; ++i) {
        h_ipiv[i] = std::min(n - 1, i + (i % 13)); // Non-trivial permutation chain
    }
    CUDA_CHECK(cudaMemcpy(d_ipiv, h_ipiv.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int k1 = 0;
    int k2 = w;

    int threads = 256;
    int num_pairs = (row_end - row_start) / 2;
    int blocks = std::min((num_pairs + threads - 1) / threads, 160); // 160 blocks = 2 blocks per SM on V100

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    batched_laswp_transposed_disjoint_kernel<<<blocks, threads>>>(
        d_B, n, k1, k2, d_ipiv, row_start, row_end
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    int iters = 20;
    for (int it = 0; it < iters; ++it) {
        batched_laswp_transposed_disjoint_kernel<<<blocks, threads>>>(
            d_B, n, k1, k2, d_ipiv, row_start, row_end
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    std::cout << "Disjoint LASWP execution time per launch (w=1024, elements=" << (row_end - row_start) << "): "
              << avg_ms << " ms" << std::endl;

    double bytes_transferred = static_cast<double>(w) * (row_end - row_start) * sizeof(__half) * 2.0;
    double gb_s = (bytes_transferred / (avg_ms * 1e-3)) / 1e9;
    std::cout << "Effective Bandwidth with Disjoint Row Partitioning: " << gb_s << " GB/s" << std::endl;

    cudaFree(d_B);
    cudaFree(d_ipiv);
    return 0;
}
