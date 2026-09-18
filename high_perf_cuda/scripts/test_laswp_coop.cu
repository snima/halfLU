#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <numeric>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

__global__ void batched_laswp_transposed_coop_kernel(
    __half* __restrict__ B,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int row_start,
    int row_end
) {
    auto grid = cg::this_grid();
    using VecType = float4; // 8 halfs
    const int num_elements = row_end - row_start;
    const int num_vec = num_elements / 8;
    const int w = k2 - k1;

    for (int i = 0; i < w; ++i) {
        int r1 = k1 + i;
        int r2 = ipiv[k1 + i];
        if (r1 != r2) {
            VecType* col1 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r1) * n + row_start);
            VecType* col2 = reinterpret_cast<VecType*>(B + static_cast<size_t>(r2) * n + row_start);

            for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vec; idx += gridDim.x * blockDim.x) {
                VecType v1 = col1[idx];
                VecType v2 = col2[idx];
                col1[idx] = v2;
                col2[idx] = v1;
            }

            int rem_start = num_vec * 8;
            for (int idx = rem_start + threadIdx.x; idx < num_elements; idx += blockDim.x) {
                __half* c1 = B + static_cast<size_t>(r1) * n + row_start;
                __half* c2 = B + static_cast<size_t>(r2) * n + row_start;
                __half t = c1[idx];
                c1[idx] = c2[idx];
                c2[idx] = t;
            }
        }
        grid.sync(); // Global hardware barrier across all resident blocks
    }
}

int main() {
    int n = 40960;
    int w = 1024;
    int row_start = 1024;
    int row_end = 40960;

    std::cout << "Testing batched_laswp_transposed_coop_kernel on V100..." << std::endl;

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, device_id));
    if (!coop_supported) {
        std::cerr << "Cooperative launch not supported!" << std::endl;
        return 1;
    }

    int block_size = 256;
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        batched_laswp_transposed_coop_kernel,
        block_size, 0
    ));
    int num_sms = 80;
    int grid_size = num_sms * max_blocks_per_sm;
    std::cout << "Resident Grid: " << grid_size << " blocks (" << max_blocks_per_sm << " per SM), threads: " << block_size << std::endl;

    __half* d_B;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));

    int* d_ipiv;
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    std::vector<int> h_ipiv(n);
    for (int i = 0; i < n; ++i) {
        // Interdependent swap chain to stress-test race conditions!
        // Swap i with min(n-1, i + (i % 7))
        h_ipiv[i] = std::min(n - 1, i + (i % 7));
    }
    CUDA_CHECK(cudaMemcpy(d_ipiv, h_ipiv.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int k1 = 0;
    int k2 = w;
    void* args[] = {
        (void*)&d_B, (void*)&n, (void*)&k1, (void*)&k2, (void*)&d_ipiv, (void*)&row_start, (void*)&row_end
    };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)batched_laswp_transposed_coop_kernel,
        dim3(grid_size), dim3(block_size),
        args, 0, 0
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    int iters = 10;
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)batched_laswp_transposed_coop_kernel,
            dim3(grid_size), dim3(block_size),
            args, 0, 0
        ));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    std::cout << "Cooperative LASWP execution time per launch (w=1024, elements=" << (row_end - row_start) << "): "
              << avg_ms << " ms" << std::endl;

    double bytes_transferred = static_cast<double>(w) * (row_end - row_start) * sizeof(__half) * 2.0;
    double gb_s = (bytes_transferred / (avg_ms * 1e-3)) / 1e9;
    std::cout << "Effective Bandwidth with 100% grid.sync() race-free barrier: " << gb_s << " GB/s" << std::endl;

    cudaFree(d_B);
    cudaFree(d_ipiv);
    return 0;
}
