#pragma once

#include "cuda_utils.cuh"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace high_perf {

inline void verify_factorization(
    const std::vector<__half>& A_orig,
    const std::vector<__half>& A_fact,
    const std::vector<int>& p_rows,
    const std::vector<int>& p_cols,
    int n,
    double& backward_error,
    double& growth_factor
) {
    // 1. Build permutation vectors
    std::vector<int> p_row_perm(n);
    std::vector<int> p_col_perm(n);
    for (int i = 0; i < n; ++i) {
        p_row_perm[i] = i;
        p_col_perm[i] = i;
    }
    for (int i = 0; i < static_cast<int>(p_rows.size()); ++i) {
        if (p_rows[i] >= 0 && p_rows[i] < n) {
            std::swap(p_row_perm[i], p_row_perm[p_rows[i]]);
        }
        if (p_cols[i] >= 0 && p_cols[i] < n) {
            std::swap(p_col_perm[i], p_col_perm[p_cols[i]]);
        }
    }

    // 2. Extract L and U and compute L * U in double precision
    std::vector<double> LU(static_cast<std::size_t>(n) * n, 0.0);
    double max_orig = 0.0;
    double max_fact = 0.0;
    double norm_orig_sq = 0.0;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double v_orig = static_cast<double>(__half2float(A_orig[static_cast<std::size_t>(j) * n + i]));
            double v_fact = static_cast<double>(__half2float(A_fact[static_cast<std::size_t>(j) * n + i]));
            if (std::abs(v_orig) > max_orig) max_orig = std::abs(v_orig);
            if (std::abs(v_fact) > max_fact) max_fact = std::abs(v_fact);
            norm_orig_sq += v_orig * v_orig;
        }
    }

    growth_factor = max_orig > 0.0 ? (max_fact / max_orig) : 1.0;

    // Compute (L * U)_{i, j} = sum_{k=0}^{min(i,j)} L_{i, k} * U_{k, j}
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            int k_max = std::min(i, j);
            for (int k = 0; k <= k_max; ++k) {
                double L_ik = (i == k) ? 1.0 : static_cast<double>(__half2float(A_fact[static_cast<std::size_t>(k) * n + i]));
                double U_kj = static_cast<double>(__half2float(A_fact[static_cast<std::size_t>(j) * n + k]));
                sum += L_ik * U_kj;
            }
            LU[static_cast<std::size_t>(j) * n + i] = sum;
        }
    }

    // 3. Compute norm of P * A_orig * Q^T - LU
    double diff_norm_sq = 0.0;
    for (int j = 0; j < n; ++j) {
        int orig_c = p_col_perm[j];
        for (int i = 0; i < n; ++i) {
            int orig_r = p_row_perm[i];
            double orig_val = static_cast<double>(__half2float(A_orig[static_cast<std::size_t>(orig_c) * n + orig_r]));
            double diff = orig_val - LU[static_cast<std::size_t>(j) * n + i];
            diff_norm_sq += diff * diff;
        }
    }

    double norm_orig = std::sqrt(norm_orig_sq);
    double diff_norm = std::sqrt(diff_norm_sq);
    backward_error = norm_orig > 0.0 ? (diff_norm / norm_orig) : diff_norm;
}

} // namespace high_perf
