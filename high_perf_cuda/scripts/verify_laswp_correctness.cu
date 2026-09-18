#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

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
    const int num_pairs = num_rows / 2;
    const int w = k2 - k1;

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
    int n = 2048;
    int k1 = 128;
    int k2 = 384;
    int row_start = 512;
    int row_end = 2047; // odd count to test remainder too

    std::vector<float> h_B_cpu(n * n);
    std::vector<__half> h_B_gpu_init(n * n);
    for (int i = 0; i < n * n; ++i) {
        float val = static_cast<float>((i * 17) % 1000);
        h_B_cpu[i] = val;
        h_B_gpu_init[i] = __float2half(val);
    }

    std::vector<int> h_ipiv(n);
    for (int i = 0; i < n; ++i) {
        h_ipiv[i] = i + (rand() % (n - i));
    }

    for (int i = k1; i < k2; ++i) {
        int r1 = i;
        int r2 = h_ipiv[i];
        if (r1 == r2) continue;
        for (int r = row_start; r < row_end; ++r) {
            float t = h_B_cpu[r1 * n + r];
            h_B_cpu[r1 * n + r] = h_B_cpu[r2 * n + r];
            h_B_cpu[r2 * n + r] = t;
        }
    }

    __half* d_B;
    int* d_ipiv;
    cudaMalloc(&d_B, n * n * sizeof(__half));
    cudaMalloc(&d_ipiv, n * sizeof(int));
    cudaMemcpy(d_B, h_B_gpu_init.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ipiv, h_ipiv.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int num_pairs = (row_end - row_start) / 2;
    int blocks = (num_pairs + threads - 1) / threads;

    batched_laswp_transposed_disjoint_kernel<<<blocks, threads>>>(
        d_B, n, k1, k2, d_ipiv, row_start, row_end
    );
    cudaDeviceSynchronize();

    std::vector<__half> h_B_gpu_out(n * n);
    cudaMemcpy(h_B_gpu_out.data(), d_B, n * n * sizeof(__half), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int r1 = 0; r1 < n; ++r1) {
        for (int r = 0; r < n; ++r) {
            float cpu_val = h_B_cpu[r1 * n + r];
            float gpu_val = __half2float(h_B_gpu_out[r1 * n + r]);
            if (cpu_val != gpu_val) {
                errors++;
                if (errors <= 5) {
                    std::cout << "Mismatch at (" << r1 << "," << r << "): CPU=" << cpu_val << " GPU=" << gpu_val << std::endl;
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << "VERIFICATION SUCCESS: 100% EXACT BIT MATCH! Zero errors across " << (n*n) << " elements!" << std::endl;
    } else {
        std::cout << "VERIFICATION FAILED: " << errors << " errors found!" << std::endl;
    }

    cudaFree(d_B);
    cudaFree(d_ipiv);
    return (errors == 0 ? 0 : 1);
}
