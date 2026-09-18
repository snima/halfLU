#pragma once

#include "cuda_utils.cuh"
#include "types.hpp"
#include <random>
#include <vector>
#include <cmath>

namespace high_perf {

inline void generate_matrix_host(
    std::vector<__half>& host_matrix,
    int n,
    MatrixFamily family,
    std::uint64_t seed
) {
    host_matrix.resize(static_cast<std::size_t>(n) * n);
    std::mt19937_64 rng(seed);

    if (family == MatrixFamily::kUniform) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < host_matrix.size(); ++i) {
            host_matrix[i] = __float2half(dist(rng));
        }
    } else if (family == MatrixFamily::kNormal) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < host_matrix.size(); ++i) {
            host_matrix[i] = __float2half(dist(rng));
        }
    } else if (family == MatrixFamily::kWilkinson) {
        for (std::size_t i = 0; i < host_matrix.size(); ++i) {
            host_matrix[i] = __float2half(0.0f);
        }
        for (int i = 0; i < n; ++i) {
            host_matrix[static_cast<std::size_t>(i) * n + i] = __float2half(1.0f); // diagonal = 1
            if (i < n - 1) {
                for (int r = i + 1; r < n; ++r) {
                    host_matrix[static_cast<std::size_t>(i) * n + r] = __float2half(-1.0f); // subdiagonal = -1
                }
            }
            host_matrix[static_cast<std::size_t>(n - 1) * n + i] = __float2half(1.0f); // last column = 1
        }
    } else if (family == MatrixFamily::kGraded) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int c = 0; c < n; ++c) {
            float col_scale = std::pow(10.0f, -4.0f * (static_cast<float>(c) / static_cast<float>(n)));
            for (int r = 0; r < n; ++r) {
                host_matrix[static_cast<std::size_t>(c) * n + r] = __float2half(dist(rng) * col_scale);
            }
        }
    } else { // Fallback / near-tie test families
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < host_matrix.size(); ++i) {
            host_matrix[i] = __float2half(dist(rng));
        }
        // Inject near-tie pairs for dp_equality or gp_second
        if (family == MatrixFamily::kDpEquality || family == MatrixFamily::kGpSecond) {
            for (int k = 0; k < n - 1; k += 4) {
                float v = dist(rng);
                host_matrix[static_cast<std::size_t>(k) * n + k] = __float2half(v);
                host_matrix[static_cast<std::size_t>(k) * n + k + 1] = __float2half(v * 1.002f);
            }
        }
    }
}

} // namespace high_perf
