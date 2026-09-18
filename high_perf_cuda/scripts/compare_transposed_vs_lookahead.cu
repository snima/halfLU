#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "../include/types.hpp"
#include "../include/verification.cuh"
#include "../include/transposed_hierarchical_lu.cuh"
#include "../include/lookahead_transposed_lu.cuh"

using namespace high_perf;

int main() {
    int n = 2048;
    int W = 512;
    int wb = 32;

    size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
    std::vector<__half> h_orig(n * n);
    for (int i = 0; i < n * n; ++i) {
        float v = static_cast<float>((i * 17 + 5) % 1000) / 500.0f - 1.0f;
        if (std::abs(v) < 0.05f) v = 0.5f;
        h_orig[i] = __float2half(v);
    }

    __half *d_A1, *d_A2;
    CUDA_CHECK(cudaMalloc(&d_A1, bytes));
    CUDA_CHECK(cudaMalloc(&d_A2, bytes));
    CUDA_CHECK(cudaMemcpy(d_A1, h_orig.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A2, h_orig.data(), bytes, cudaMemcpyHostToDevice));

    Options opt;
    opt.n = n;
    opt.macro_width = W;
    opt.micro_width = wb;
    opt.method = Method::kPP;
    opt.tau = 1.01f;

    std::vector<int> p1_rows, p1_cols, p2_rows, p2_cols;
    Counters c1, c2;

    TransposedHierarchicalLU lu_base(n, W, wb);
    lu_base.factorize(d_A1, n, opt, p1_rows, p1_cols, c1);
    CUDA_CHECK(cudaDeviceSynchronize());

    LookaheadTransposedLU lu_look(n, W, wb);
    lu_look.factorize(d_A2, n, opt, p2_rows, p2_cols, c2);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> h_fact1(n * n), h_fact2(n * n);
    CUDA_CHECK(cudaMemcpy(h_fact1.data(), d_A1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fact2.data(), d_A2, bytes, cudaMemcpyDeviceToHost));

    double err1 = 0, growth1 = 0;
    verify_factorization(h_orig, h_fact1, p1_rows, p1_cols, n, err1, growth1);

    double err2 = 0, growth2 = 0;
    verify_factorization(h_orig, h_fact2, p2_rows, p2_cols, n, err2, growth2);

    std::cout << "Base Transposed Error:      " << err1 << std::endl;
    std::cout << "Lookahead Transposed Error: " << err2 << std::endl;

    cudaFree(d_A1);
    cudaFree(d_A2);
    return 0;
}
