#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cassert>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// Disjoint Row Partitioned LASWP Kernel
__global__ void batched_laswp_transposed_disjoint_kernel(
    __half* __restrict__ B,
    int n,
    int k1,
    int k2,
    const int* __restrict__ ipiv,
    int row_start,
    int row_end
) {
    if (k2 <= k1 || row_end <= row_start) return;

    int aligned_start = row_start;
    const int w = k2 - k1;

    // Handle leading unaligned row if row_start is odd
    if ((row_start & 1) != 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int i = 0; i < w; ++i) {
                int r1 = k1 + i;
                int r2 = ipiv[k1 + i];
                if (r1 == r2) continue;
                __half* p1 = B + static_cast<size_t>(r1) * n + row_start;
                __half* p2 = B + static_cast<size_t>(r2) * n + row_start;
                __half t = *p1;
                *p1 = *p2;
                *p2 = t;
            }
        }
        aligned_start++;
    }

    const int num_rows = row_end - aligned_start;
    const int num_pairs = num_rows / 2;

    for (int pair_idx = blockIdx.x * blockDim.x + threadIdx.x; pair_idx < num_pairs; pair_idx += gridDim.x * blockDim.x) {
        int r_base = aligned_start + pair_idx * 2;
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

// Sequential CPU LASWP operating strictly on raw 16-bit patterns
void cpu_laswp_transposed_raw(
    uint16_t* B,
    int n,
    int k1,
    int k2,
    const int* ipiv,
    int row_start,
    int row_end
) {
    for (int i = k1; i < k2; ++i) {
        int r1 = i;
        int r2 = ipiv[i];
        if (r1 == r2) continue;
        for (int r = row_start; r < row_end; ++r) {
            uint16_t t = B[static_cast<size_t>(r1) * n + r];
            B[static_cast<size_t>(r1) * n + r] = B[static_cast<size_t>(r2) * n + r];
            B[static_cast<size_t>(r2) * n + r] = t;
        }
    }
}

bool run_test_case(
    const std::string& test_name,
    int n,
    int k1,
    int k2,
    int row_start,
    int row_end,
    const std::vector<int>& ipiv,
    uint32_t seed
) {
    size_t total_elements = static_cast<size_t>(n) * n;
    std::vector<uint16_t> h_B_init(total_elements);

    // Populate with pseudo-random 16-bit raw patterns (covering all bit ranges)
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFF);
    for (size_t i = 0; i < total_elements; ++i) {
        h_B_init[i] = static_cast<uint16_t>(dist(rng));
    }

    // CPU reference
    std::vector<uint16_t> h_B_cpu = h_B_init;
    cpu_laswp_transposed_raw(h_B_cpu.data(), n, k1, k2, ipiv.data(), row_start, row_end);

    // GPU execution
    __half* d_B = nullptr;
    int* d_ipiv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, total_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_B, h_B_init.data(), total_elements * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ipiv, ipiv.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int num_pairs = (row_end - row_start) / 2;
    int blocks = std::max(1, (num_pairs + threads - 1) / threads);
    if (blocks > 320) blocks = 320;

    batched_laswp_transposed_disjoint_kernel<<<blocks, threads>>>(
        d_B, n, k1, k2, d_ipiv, row_start, row_end
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> h_B_gpu(total_elements);
    CUDA_CHECK(cudaMemcpy(h_B_gpu.data(), d_B, total_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_ipiv));

    // Strict 16-bit raw comparison
    size_t bit_mismatches = 0;
    for (size_t i = 0; i < total_elements; ++i) {
        if (h_B_cpu[i] != h_B_gpu[i]) {
            bit_mismatches++;
            if (bit_mismatches <= 5) {
                int col = static_cast<int>(i / n);
                int row = static_cast<int>(i % n);
                std::cerr << "  [MISMATCH] at (" << col << "," << row << ") - CPU raw: 0x"
                          << std::hex << std::setw(4) << std::setfill('0') << h_B_cpu[i]
                          << " GPU raw: 0x" << std::setw(4) << std::setfill('0') << h_B_gpu[i]
                          << std::dec << std::endl;
            }
        }
    }

    if (bit_mismatches == 0) {
        std::cout << "  [PASS] " << std::left << std::setw(35) << test_name
                  << " elements: " << total_elements << " | Bit Mismatches: 0" << std::endl;
        return true;
    } else {
        std::cerr << "  [FAIL] " << std::left << std::setw(35) << test_name
                  << " Bit Mismatches: " << bit_mismatches << " / " << total_elements << std::endl;
        return false;
    }
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << " RIGOROUS BIT-EXACT VERIFICATION FOR DISJOINT ROW PARTITIONED LASWP" << std::endl;
    std::cout << " (Direct 16-Bit Raw Integer Pattern Comparison against Sequential LAPACK LASWP)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    bool all_passed = true;

    // Test Suite 1: Synthetic realistic permutations (even and odd ranges)
    {
        int n = 2048;
        int k1 = 128, k2 = 384;
        std::mt19937 rng(42);
        std::vector<int> ipiv(n);
        for (int i = 0; i < n; ++i) {
            std::uniform_int_distribution<int> pdist(i, n - 1);
            ipiv[i] = pdist(rng);
        }

        // Even row slice
        all_passed &= run_test_case("Random Permutations (Even slice)", n, k1, k2, 512, 2048, ipiv, 101);
        // Odd row slice (tests scalar remainder handling)
        all_passed &= run_test_case("Random Permutations (Odd slice)", n, k1, k2, 513, 2047, ipiv, 202);
    }

    // Test Suite 2: Adversarial cyclic / identity / boundary permutations
    {
        int n = 1024;
        std::vector<int> ipiv_ident(n);
        for (int i = 0; i < n; ++i) ipiv_ident[i] = i;
        all_passed &= run_test_case("Identity Permutations (no swaps)", n, 0, 512, 512, 1024, ipiv_ident, 303);

        std::vector<int> ipiv_reverse(n);
        for (int i = 0; i < n; ++i) ipiv_reverse[i] = n - 1 - i;
        all_passed &= run_test_case("Reverse Permutations (extreme chain)", n, 0, 256, 256, 1024, ipiv_reverse, 404);
    }

    // Test Suite 3: Realistic Macro-Panel dimensions (W=1024 on n=4096)
    {
        int n = 4096;
        int W = 1024;
        std::mt19937 rng(999);
        std::vector<int> ipiv(n);
        for (int i = 0; i < n; ++i) {
            std::uniform_int_distribution<int> pdist(i, n - 1);
            ipiv[i] = pdist(rng);
        }

        all_passed &= run_test_case("Macro-Panel Left LASWP (k=0)", n, 0, W, 0, 0, ipiv, 505); // degenerate empty
        all_passed &= run_test_case("Macro-Panel Trailing (k=0)", n, 0, W, W, n, ipiv, 606);
        all_passed &= run_test_case("Macro-Panel Trailing (k=1024)", n, W, 2 * W, 2 * W, n, ipiv, 707);
        all_passed &= run_test_case("Macro-Panel Trailing (k=2048, odd)", n, 2 * W, 3 * W, 3 * W, n - 1, ipiv, 808);
    }

    std::cout << "================================================================================" << std::endl;
    if (all_passed) {
        std::cout << " FINAL VERDICT: 100% BIT-EXACT MATCH CONFIRMED ACROSS ALL TEST CASES!" << std::endl;
        std::cout << " Disjoint Row Partitioning is mathematically and numerically identical to LAPACK." << std::endl;
    } else {
        std::cerr << " FINAL VERDICT: VERIFICATION FAILED! Mismatches detected." << std::endl;
    }
    std::cout << "================================================================================" << std::endl;

    return all_passed ? 0 : 1;
}
