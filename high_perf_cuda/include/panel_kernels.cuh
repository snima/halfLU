#pragma once

#include "cuda_utils.cuh"

namespace high_perf {

// Batched LASWP kernel: applies all w row swaps to columns [col_start, col_end)
// Each thread block processes a tile of columns. Coalesced and extremely fast!
__global__ void batched_laswp_kernel(
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

// Panel-only row swap: swaps row r1 and *d_pivot_row ONLY inside panel columns [col_start, col_end)
__global__ void swap_rows_panel_kernel(
    __half* __restrict__ matrix,
    int n,
    int r1,
    const int* __restrict__ d_pivot_row,
    int col_start,
    int col_end,
    int* __restrict__ d_log_pivots,
    int step
) {
    const int r2 = *d_pivot_row;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_log_pivots != nullptr) {
            d_log_pivots[step] = r2;
        }
    }
    if (r1 == r2) return;
    const int col = col_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (col < col_end) {
        const std::size_t idx1 = static_cast<std::size_t>(col) * n + r1;
        const std::size_t idx2 = static_cast<std::size_t>(col) * n + r2;
        __half tmp = matrix[idx1];
        matrix[idx1] = matrix[idx2];
        matrix[idx2] = tmp;
    }
}

// Column swap for CP/RP/ScaP inside panel
__global__ void swap_cols_device_kernel(
    __half* __restrict__ matrix,
    int n,
    int c1,
    const int* __restrict__ d_pivot_col,
    int row_start,
    int row_end,
    int* __restrict__ d_log_pivots,
    int step
) {
    const int c2 = *d_pivot_col;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_log_pivots != nullptr) {
            d_log_pivots[step] = c2;
        }
    }
    if (c1 == c2) return;
    const int row = row_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < row_end) {
        const std::size_t idx1 = static_cast<std::size_t>(c1) * n + row;
        const std::size_t idx2 = static_cast<std::size_t>(c2) * n + row;
        __half tmp = matrix[idx1];
        matrix[idx1] = matrix[idx2];
        matrix[idx2] = tmp;
    }
}

// Scale column elements below diagonal by pivot value
__global__ void scale_column_kernel(
    __half* __restrict__ matrix,
    int n,
    int k
) {
    const int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const __half pivot = matrix[static_cast<std::size_t>(k) * n + k];
        const std::size_t idx = static_cast<std::size_t>(k) * n + row;
        float val = __half2float(matrix[idx]);
        float p_val = __half2float(pivot);
        if (fabsf(p_val) > 0.0f) {
            matrix[idx] = __float2half(val / p_val);
        }
    }
}

// Internal panel rank-1 update for columns inside the current panel
__global__ void panel_rank1_update_kernel(
    __half* __restrict__ matrix,
    int n,
    int k,
    int panel_end
) {
    const int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;

    if (col < panel_end && row < n) {
        const __half mult = matrix[static_cast<std::size_t>(k) * n + row];
        const __half u_val = matrix[static_cast<std::size_t>(col) * n + k];
        const std::size_t entry = static_cast<std::size_t>(col) * n + row;
        matrix[entry] = __hsub(matrix[entry], __hmul(mult, u_val));
    }
}

// High-throughput triangular solve for U12 = L11^{-1} * A12
constexpr int kMaxPanelWidth = 256;
__global__ void trsm_u12_kernel(
    __half* __restrict__ matrix,
    int n,
    int k,
    int panel_end
) {
    extern __shared__ __half s_L_dynamic[];

    const int w = panel_end - k;
    const int tid = threadIdx.x;

    // Cooperatively load L11 into shared memory
    for (int idx = tid; idx < w * w; idx += blockDim.x) {
        int r = idx % w;
        int c = idx / w;
        if (r > c) {
            s_L_dynamic[r * w + c] = matrix[static_cast<std::size_t>(k + c) * n + (k + r)];
        } else if (r == c) {
            s_L_dynamic[r * w + c] = __float2half(1.0f);
        } else {
            s_L_dynamic[r * w + c] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Each thread solves one column of A12
    const int col = panel_end + blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int r = 0; r < w; ++r) {
            float xr = __half2float(matrix[static_cast<std::size_t>(col) * n + (k + r)]);
            for (int p = 0; p < r; ++p) {
                xr -= __half2float(s_L_dynamic[r * w + p]) * __half2float(matrix[static_cast<std::size_t>(col) * n + (k + p)]);
            }
            matrix[static_cast<std::size_t>(col) * n + (k + r)] = __float2half(xr);
        }
    }
}

// Row infinity-norm computation for ScPP
__global__ void compute_row_scales_kernel(
    const __half* __restrict__ matrix,
    int n,
    float* __restrict__ row_scales
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float max_val = 0.0f;
        for (int c = 0; c < n; ++c) {
            float val = fabsf(__half2float(matrix[static_cast<std::size_t>(c) * n + row]));
            if (val > max_val) max_val = val;
        }
        row_scales[row] = max_val;
    }
}

} // namespace high_perf
