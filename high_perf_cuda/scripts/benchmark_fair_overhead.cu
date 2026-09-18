#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <random>

#include "include/cuda_utils.cuh"
#include "include/types.hpp"

namespace cg = cooperative_groups;
using namespace high_perf;

constexpr int FUSED_COOP_BLOCK = 256;
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

struct BlockPivotResult {
    unsigned long long best;
    unsigned long long second;
};

// ============================================================================
// Transposed Helper Kernels
// ============================================================================
__global__ void transpose_matrix_kernel(
    const __half* __restrict__ idata,
    __half* __restrict__ odata,
    int width, int height, int ldi, int ldo
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

__global__ void transpose_submatrix_kernel(
    const __half* __restrict__ idata, int ldi,
    __half* __restrict__ odata, int ldo,
    int width, int height
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

__global__ void batched_laswp_kernel(
    __half* __restrict__ a, int lda,
    int k1, int k2,
    const int* __restrict__ ipiv,
    int col_start, int col_end
) {
    int col = col_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= col_end) return;

    for (int i = k1; i < k2; ++i) {
        int r1 = i;
        int r2 = ipiv[i];
        if (r1 != r2) {
            __half tmp = a[static_cast<size_t>(col) * lda + r1];
            a[static_cast<size_t>(col) * lda + r1] = a[static_cast<size_t>(col) * lda + r2];
            a[static_cast<size_t>(col) * lda + r2] = tmp;
        }
    }
}

__global__ void batched_laswp_transposed_disjoint_kernel(
    __half* __restrict__ B,
    int n, int k1, int k2,
    const int* __restrict__ ipiv,
    int row_start, int row_end
) {
    const int num_rows = row_end - row_start;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = row_start + pair_idx * 2;
    if (r >= row_end) return;

    const int w = k2 - k1;
    bool has_second = (r + 1 < row_end);

    for (int i = 0; i < w; ++i) {
        int r1 = k1 + i;
        int r2 = ipiv[k1 + i];
        if (r1 == r2) continue;

        size_t idx1 = static_cast<size_t>(r1) * n + r;
        size_t idx2 = static_cast<size_t>(r2) * n + r;

        if (has_second) {
            __half2 v1 = *reinterpret_cast<__half2*>(&B[idx1]);
            __half2 v2 = *reinterpret_cast<__half2*>(&B[idx2]);
            *reinterpret_cast<__half2*>(&B[idx1]) = v2;
            *reinterpret_cast<__half2*>(&B[idx2]) = v1;
        } else {
            __half v1 = B[idx1];
            __half v2 = B[idx2];
            B[idx1] = v2;
            B[idx2] = v1;
        }
    }
}

__global__ void trtri_unit_lower_kernel(
    const __half* __restrict__ matrix, int ldl,
    int offset, int w,
    __half* __restrict__ L_inv
) {
    extern __shared__ float s_inv[];
    int tid = threadIdx.x;

    for (int idx = tid; idx < w * w; idx += blockDim.x) {
        int r = idx % w;
        int c = idx / w;
        s_inv[r * w + c] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (tid < w) {
        int j = tid;
        for (int i = j + 1; i < w; ++i) {
            float sum = 0.0f;
            for (int p = j; p < i; ++p) {
                float L_ip = __half2float(matrix[static_cast<size_t>(offset + p) * ldl + (offset + i)]);
                sum += L_ip * s_inv[p * w + j];
            }
            s_inv[i * w + j] = -sum;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < w * w; idx += blockDim.x) {
        int r = idx % w;
        int c = idx / w;
        L_inv[c * w + r] = __float2half(s_inv[r * w + c]);
    }
}

__global__ void store_block_kernel(
    const __half* __restrict__ src, int ld_src,
    __half* __restrict__ dst, int ld_dst,
    int r_offset, int c_offset,
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        dst[static_cast<size_t>(c_offset + c) * ld_dst + (r_offset + r)] = src[static_cast<size_t>(c) * ld_src + r];
    }
}

__global__ void offset_pivots_kernel(int* ipiv, int count, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ipiv[idx] += offset;
    }
}

// Fast vectorized row infinity norms from transposed matrix B
__global__ void compute_row_scales_from_transposed_kernel(
    const __half* __restrict__ B,
    int n,
    float* __restrict__ row_scales
) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const __half* col_ptr = B + static_cast<size_t>(row) * n;
    float local_max = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = fabsf(__half2float(col_ptr[i]));
        if (val > local_max) local_max = val;
    }

    __shared__ float s_max[256];
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (s_max[threadIdx.x + offset] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float res = s_max[0];
        row_scales[row] = (res > 0.0f) ? res : 1.0f;
    }
}

// ============================================================================
// Unified Cooperative Panel Kernel Supporting All 7 Policies
// ============================================================================
__global__ void unified_fused_panel_kernel(
    __half* __restrict__ matrix,
    int n,
    int k,
    int w_actual,
    int method_code,
    float tau,
    float* __restrict__ row_scales,
    int* __restrict__ d_log_pivot_rows,
    int* __restrict__ d_log_pivot_cols,
    unsigned long long* __restrict__ d_counters,
    volatile BlockPivotResult* d_block_results
) {
    cg::grid_group grid = cg::this_grid();
    const int tid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int nblk  = gridDim.x;
    const int gid   = bid * blockDim.x + tid;
    const int gsize = nblk * blockDim.x;

    const int panel_end = k + w_actual;

    __shared__ unsigned long long s_keys[FUSED_COOP_BLOCK];
    __shared__ unsigned long long s_second[FUSED_COOP_BLOCK];
    __shared__ int s_pivot_row;
    __shared__ int s_pivot_col;

    for (int j = k; j < panel_end; ++j) {
        unsigned long long local_best   = 0ull;
        unsigned long long local_second = 0ull;

        if (method_code == 0) {
            // PP (0): Pure 1D column max search
            const size_t off_j = static_cast<size_t>(j) * n;
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[off_j + r]));
                if (bits_finite(bits)) {
                    unsigned long long key = pack_key(bits, r);
                    if (key > local_best) local_best = key;
                }
            }
        } else if (method_code == 6) {
            // ScPP (6): Row-scaled 1D search
            const size_t off_j = static_cast<size_t>(j) * n;
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[off_j + r]));
                if (bits_finite(bits)) {
                    float s = (row_scales) ? fmaxf(row_scales[r], 1e-30f) : 1.0f;
                    float raw_val = fabsf(__half2float(__ushort_as_half(bits))) / s;
                    unsigned short sbits = absolute_bits(__float2half(raw_val));
                    unsigned long long key = pack_key(sbits, r);
                    if (key > local_best) local_best = key;
                }
            }
        } else if (method_code == 1 || method_code == 2) {
            // DP (1) or GP (2): Find top two elements
            const size_t off_j = static_cast<size_t>(j) * n;
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[off_j + r]));
                if (bits_finite(bits)) {
                    unsigned long long key = pack_key(bits, r);
                    if (key > local_best) {
                        local_second = local_best;
                        local_best   = key;
                    } else if (key > local_second) {
                        local_second = key;
                    }
                }
            }
        } else if (method_code == 3) {
            // ScaP (3): 3-column probe within panel
            const int c0 = j;
            const int c1 = j + (panel_end - 1 - j) / 2;
            const int c2 = panel_end - 1;
            const int num_probes = (panel_end - j >= 3) ? 3 : ((panel_end - j == 2) ? 2 : 1);
            const size_t off0 = static_cast<size_t>(c0) * n;
            const size_t off1 = static_cast<size_t>(c1) * n;
            const size_t off2 = static_cast<size_t>(c2) * n;

            for (int r = j + gid; r < n; r += gsize) {
                unsigned short b0 = absolute_bits(__ldg(&matrix[off0 + r]));
                unsigned short max_b = bits_finite(b0) ? b0 : 0;
                int best_p = 0;
                if (num_probes > 1) {
                    unsigned short b1 = absolute_bits(__ldg(&matrix[off1 + r]));
                    if (bits_finite(b1) && b1 > max_b) {
                        max_b = b1;
                        best_p = 1;
                    }
                }
                if (num_probes > 2) {
                    unsigned short b2 = absolute_bits(__ldg(&matrix[off2 + r]));
                    if (bits_finite(b2) && b2 > max_b) {
                        max_b = b2;
                        best_p = 2;
                    }
                }
                if (max_b > 0) {
                    unsigned long long key = (static_cast<unsigned long long>(max_b) << 32) |
                                            ((static_cast<unsigned int>(r) << 8) | static_cast<unsigned int>(best_p));
                    if (key > local_best) local_best = key;
                }
            }
        } else if (method_code == 5) {
            // CP (5): Parallel 2D submatrix search in [j, panel_end) x [j, n)
            const int num_cols = panel_end - j;
            const int num_rows = n - j;
            const int total = num_cols * num_rows;

            for (int idx = gid; idx < total; idx += gsize) {
                int col = j + idx / num_rows;
                int row = j + idx % num_rows;
                unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<size_t>(col) * n + row]));
                if (!bits_finite(bits)) continue;
                unsigned int packed_rc = (static_cast<unsigned int>(row) << 16) | static_cast<unsigned int>(col);
                unsigned long long key = (static_cast<unsigned long long>(bits) << 32) | static_cast<unsigned int>(~packed_rc);
                if (key > local_best) local_best = key;
            }
        } else if (method_code == 4) {
            // RP (4): Initial column max for column j
            const size_t off_j = static_cast<size_t>(j) * n;
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[off_j + r]));
                if (bits_finite(bits)) {
                    unsigned long long key = pack_key(bits, r);
                    if (key > local_best) local_best = key;
                }
            }
        }

        // Tree reduction within block
        s_keys[tid]   = local_best;
        s_second[tid] = local_second;
        __syncthreads();

        for (int offset = FUSED_COOP_BLOCK / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                unsigned long long ob = s_keys[tid + offset];
                unsigned long long os = s_second[tid + offset];
                unsigned long long mb = s_keys[tid];
                unsigned long long ms = s_second[tid];
                if (ob > mb) {
                    ms = (mb > os) ? mb : os;
                    mb = ob;
                } else if (ob > ms) {
                    ms = ob;
                }
                s_keys[tid]   = mb;
                s_second[tid] = ms;
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_block_results[bid].best   = s_keys[0];
            d_block_results[bid].second = s_second[0];
        }
        grid.sync();

        // Phase 1b: Final reduction across blocks (Block 0)
        if (bid == 0) {
            unsigned long long g_best   = 0ull;
            unsigned long long g_second = 0ull;
            for (int b = tid; b < nblk; b += FUSED_COOP_BLOCK) {
                unsigned long long ob = d_block_results[b].best;
                unsigned long long os = d_block_results[b].second;
                if (ob > g_best) {
                    g_second = (g_best > os) ? g_best : os;
                    g_best   = ob;
                } else if (ob > g_second) {
                    g_second = ob;
                }
            }
            s_keys[tid]   = g_best;
            s_second[tid] = g_second;
            __syncthreads();

            for (int offset = FUSED_COOP_BLOCK / 2; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    unsigned long long ob = s_keys[tid + offset];
                    unsigned long long os = s_second[tid + offset];
                    unsigned long long mb = s_keys[tid];
                    unsigned long long ms = s_second[tid];
                    if (ob > mb) {
                        ms = (mb > os) ? mb : os;
                        mb = ob;
                    } else if (ob > ms) {
                        ms = ob;
                    }
                    s_keys[tid]   = mb;
                    s_second[tid] = ms;
                }
                __syncthreads();
            }

            if (tid == 0) {
                unsigned long long best   = s_keys[0];
                unsigned long long second = s_second[0];
                int pivot_r = key_empty(best) ? j : key_index(best);
                int pivot_c = j;

                if (method_code == 1) {
                    // DP
                    float diag_val = fabsf(__half2float(matrix[static_cast<size_t>(j) * n + j]));
                    float max_val  = key_magnitude(best);
                    if (max_val > 0.0f && diag_val >= max_val / tau) {
                        pivot_r = j;
                        atomicAdd(&d_counters[kSlotDpAccepts], 1ull);
                    } else {
                        atomicAdd(&d_counters[kSlotDpFallbacks], 1ull);
                    }
                } else if (method_code == 3) {
                    // ScaP
                    if (!key_empty(best)) {
                        unsigned int packed = static_cast<unsigned int>(best & 0xffffffffull);
                        pivot_r = static_cast<int>(packed >> 8);
                        int probe_id = static_cast<int>(packed & 0xffu);
                        const int c0 = j;
                        const int c1 = j + (panel_end - 1 - j) / 2;
                        const int c2 = panel_end - 1;
                        pivot_c = (probe_id == 0) ? c0 : ((probe_id == 1) ? c1 : c2);

                        if (probe_id == 0) atomicAdd(&d_counters[kSlotScapCurrent], 1ull);
                        else if (probe_id == 1) atomicAdd(&d_counters[kSlotScapMiddle], 1ull);
                        else atomicAdd(&d_counters[kSlotScapLast], 1ull);
                    }
                } else if (method_code == 5) {
                    // CP
                    if (!key_empty(best)) {
                        unsigned int packed = ~static_cast<unsigned int>(best & 0xffffffffull);
                        pivot_r = static_cast<int>(packed >> 16);
                        pivot_c = static_cast<int>(packed & 0xffffu);
                    }
                }

                if (pivot_r >= n || pivot_r < j) pivot_r = j;
                if (pivot_c >= panel_end || pivot_c < j) pivot_c = j;

                s_pivot_row = pivot_r;
                s_pivot_col = pivot_c;
            }
            __syncthreads();

            // Parallel GP Tie-Break using Warp 0 of Block 0 (fully coalesced, zero serial stall)
            if (method_code == 2) {
                unsigned long long best   = s_keys[0];
                unsigned long long second = s_second[0];
                float ratio = separation_ratio_device(best, second);
                if ((n - j) > 1 && ratio < tau && !key_empty(second)) {
                    int r1 = key_index(best);
                    int r2 = key_index(second);
                    int num_pcols = panel_end - j;
                    float v1 = 0.0f;
                    float v2 = 0.0f;
                    if (tid < num_pcols) {
                        int c = j + tid;
                        v1 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r1]));
                        v2 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r2]));
                    }
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        v1 = fmaxf(v1, __shfl_down_sync(0xffffffff, v1, offset));
                        v2 = fmaxf(v2, __shfl_down_sync(0xffffffff, v2, offset));
                    }
                    if (tid == 0) {
                        if (v2 < v1) {
                            s_pivot_row = r2;
                            atomicAdd(&d_counters[kSlotGpSecondChoices], 1ull);
                        }
                        atomicAdd(&d_counters[kSlotGpNearTies], 1ull);
                    }
                }
            }
            __syncthreads();

            if (method_code == 4) {
                // RP: Fast cooperative alternating walk using all 256 threads of Block 0
                int cur_r = s_pivot_row;
                int cur_c = s_pivot_col;
                if (cur_r >= n || cur_r < j) cur_r = j;
                if (cur_c >= panel_end || cur_c < j) cur_c = j;
                int iters = 1;

                for (int it = 0; it < 4; ++it) {
                    // 1. Parallel row search across panel cols [j, panel_end) (<= 32 cols)
                    int num_pcols = panel_end - j;
                    float my_val = -1.0f;
                    int my_c = -1;
                    if (tid < num_pcols) {
                        my_c = j + tid;
                        my_val = fabsf(__half2float(matrix[static_cast<size_t>(my_c) * n + cur_r]));
                    }
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        float other_val = __shfl_down_sync(0xffffffff, my_val, offset);
                        int other_c     = __shfl_down_sync(0xffffffff, my_c, offset);
                        if (other_val > my_val) {
                            my_val = other_val;
                            my_c   = other_c;
                        }
                    }
                    if (tid == 0) {
                        s_pivot_col = (my_c >= j && my_c < panel_end) ? my_c : cur_c;
                    }
                    __syncthreads();

                    if (s_pivot_col == cur_c) break;
                    cur_c = s_pivot_col;
                    iters++;

                    // 2. Parallel column search across rows [j, n) using all 256 threads
                    unsigned long long thread_max = 0ull;
                    for (int r = j + tid; r < n; r += FUSED_COOP_BLOCK) {
                        unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<size_t>(cur_c) * n + r]));
                        if (bits_finite(bits)) {
                            unsigned long long key = pack_key(bits, r);
                            if (key > thread_max) thread_max = key;
                        }
                    }
                    s_keys[tid] = thread_max;
                    __syncthreads();

                    for (int offset = FUSED_COOP_BLOCK / 2; offset > 0; offset >>= 1) {
                        if (tid < offset) {
                            if (s_keys[tid + offset] > s_keys[tid]) s_keys[tid] = s_keys[tid + offset];
                        }
                        __syncthreads();
                    }

                    int max_r = key_empty(s_keys[0]) ? cur_r : key_index(s_keys[0]);
                    if (max_r >= n || max_r < j) max_r = cur_r;
                    if (max_r == cur_r) break;
                    cur_r = max_r;
                    if (tid == 0) s_pivot_row = cur_r;
                    __syncthreads();
                    iters++;
                }

                if (tid == 0) {
                    atomicAdd(&d_counters[kSlotRpIterations], static_cast<unsigned long long>(iters));
                }
            }

            if (tid == 0) {
                if (s_pivot_row >= n || s_pivot_row < j) s_pivot_row = j;
                if (s_pivot_col >= panel_end || s_pivot_col < j) s_pivot_col = j;
                if (d_log_pivot_rows) d_log_pivot_rows[j] = s_pivot_row;
                if (d_log_pivot_cols) d_log_pivot_cols[j] = s_pivot_col;
            }
            __syncthreads();
        }

        // Broadcast decisions
        if (bid == 0 && tid == 0) {
            d_block_results[0].best   = static_cast<unsigned long long>(s_pivot_row);
            d_block_results[0].second = static_cast<unsigned long long>(s_pivot_col);
        }
        grid.sync();

        const int pivot_row = static_cast<int>(d_block_results[0].best);
        const int pivot_col = static_cast<int>(d_block_results[0].second);

        // Phase 1c: Column swap inside panel if pivot_col != j
        if (pivot_col != j) {
            for (int r = gid; r < n; r += gsize) {
                size_t idx1 = static_cast<size_t>(j) * n + r;
                size_t idx2 = static_cast<size_t>(pivot_col) * n + r;
                __half tmp = matrix[idx1];
                matrix[idx1] = matrix[idx2];
                matrix[idx2] = tmp;
            }
        }
        grid.sync();

        // Phase 2: Row swap across panel columns [0, panel_end)
        if (pivot_row != j) {
            for (int col = gid; col < panel_end; col += gsize) {
                size_t idx1 = static_cast<size_t>(col) * n + j;
                size_t idx2 = static_cast<size_t>(col) * n + pivot_row;
                __half tmp = matrix[idx1];
                matrix[idx1] = matrix[idx2];
                matrix[idx2] = tmp;
            }
            if (method_code == 6 && row_scales && tid == 0 && bid == 0) {
                float tmp_scale = row_scales[j];
                row_scales[j] = row_scales[pivot_row];
                row_scales[pivot_row] = tmp_scale;
            }
        }
        grid.sync();

        // Phase 3: Scale column j below diagonal (rows j+1..n-1)
        if (j + 1 < n) {
            const float pivot_val = __half2float(matrix[static_cast<size_t>(j) * n + j]);
            const float inv_pivot  = (fabsf(pivot_val) > 0.0f) ? (1.0f / pivot_val) : 0.0f;
            for (int r = (j + 1) + gid; r < n; r += gsize) {
                size_t idx = static_cast<size_t>(j) * n + r;
                matrix[idx] = __float2half(__half2float(matrix[idx]) * inv_pivot);
            }
        }
        grid.sync();

        // Phase 4: Rank-1 update of remaining panel columns
        if (j + 1 < panel_end) {
            const int num_cols = panel_end - (j + 1);
            const int num_rows = n - (j + 1);
            const int total    = num_cols * num_rows;

            for (int idx = gid; idx < total; idx += gsize) {
                int lc  = idx / num_rows;
                int lr  = idx % num_rows;
                int col = j + 1 + lc;
                int row = j + 1 + lr;

                float mult  = __half2float(matrix[static_cast<size_t>(j) * n + row]);
                float u_val = __half2float(matrix[static_cast<size_t>(col) * n + j]);
                size_t entry = static_cast<size_t>(col) * n + row;
                matrix[entry] = __float2half(__half2float(matrix[entry]) - mult * u_val);
            }
        }
        grid.sync();
    }
}

// ============================================================================
// High Performance Lookahead Transposed Factorizer
// ============================================================================
class LookaheadTransposedFactorizer {
private:
    int n_;
    int macro_w_;
    int micro_w_;
    cublasHandle_t cublas_handle_panel_ = nullptr;
    cublasHandle_t cublas_handle_gemm_ = nullptr;
    cudaStream_t stream_panel_ = nullptr;
    cudaStream_t stream_gemm_ = nullptr;
    cudaEvent_t ev_next_panel_ready_ = nullptr;
    cudaEvent_t ev_trsm_stripe_ready_ = nullptr;
    cudaEvent_t ev_gemm2_done_ = nullptr;
    cudaEvent_t ev_trsm_done_ = nullptr;

    __half* d_B_ = nullptr;
    __half* d_panel_buf_ = nullptr;
    __half* d_L_inv_micro_ = nullptr;
    __half* d_u12_buf_ = nullptr;

    int* d_log_pivot_rows_ = nullptr;
    int* d_log_pivot_cols_ = nullptr;
    unsigned long long* d_counters_ = nullptr;
    float* d_row_scales_ = nullptr;

    int fused_num_blocks_ = 160;
    volatile BlockPivotResult* d_block_results_ = nullptr;

public:
    LookaheadTransposedFactorizer(int n, int macro_w = 512, int micro_w = 32)
        : n_(n), macro_w_(std::min(n, macro_w)), micro_w_(std::min(micro_w, n)) {

        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_panel_, cudaStreamNonBlocking, -1));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_gemm_, cudaStreamNonBlocking, 0));

        CUBLAS_CHECK(cublasCreate(&cublas_handle_panel_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_panel_, stream_panel_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_panel_, CUBLAS_TENSOR_OP_MATH));

        CUBLAS_CHECK(cublasCreate(&cublas_handle_gemm_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_gemm_, stream_gemm_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_gemm_, CUBLAS_TENSOR_OP_MATH));

        CUDA_CHECK(cudaEventCreateWithFlags(&ev_next_panel_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_trsm_stripe_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_gemm2_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_trsm_done_, cudaEventDisableTiming));

        size_t matrix_bytes = static_cast<size_t>(n_) * n_ * sizeof(__half);
        CUDA_CHECK(cudaMalloc(&d_B_, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_panel_buf_, static_cast<size_t>(n_) * macro_w_ * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_L_inv_micro_, 128 * 128 * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_u12_buf_, static_cast<size_t>(n_) * 128 * sizeof(__half)));

        CUDA_CHECK(cudaMalloc(&d_log_pivot_rows_, n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_log_pivot_cols_, n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counters_, kCounterSlots * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_row_scales_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_block_results_, fused_num_blocks_ * sizeof(BlockPivotResult)));

        std::vector<float> h_scales(n_, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_row_scales_, h_scales.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));

        cudaFuncSetAttribute(trtri_unit_lower_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }

    ~LookaheadTransposedFactorizer() {
        if (cublas_handle_panel_) cublasDestroy(cublas_handle_panel_);
        if (cublas_handle_gemm_) cublasDestroy(cublas_handle_gemm_);
        if (stream_panel_) cudaStreamDestroy(stream_panel_);
        if (stream_gemm_) cudaStreamDestroy(stream_gemm_);
        if (ev_next_panel_ready_) cudaEventDestroy(ev_next_panel_ready_);
        if (ev_trsm_stripe_ready_) cudaEventDestroy(ev_trsm_stripe_ready_);
        if (ev_gemm2_done_) cudaEventDestroy(ev_gemm2_done_);
        if (ev_trsm_done_) cudaEventDestroy(ev_trsm_done_);

        if (d_B_) cudaFree(d_B_);
        if (d_panel_buf_) cudaFree(d_panel_buf_);
        if (d_L_inv_micro_) cudaFree(d_L_inv_micro_);
        if (d_u12_buf_) cudaFree(d_u12_buf_);
        if (d_log_pivot_rows_) cudaFree(d_log_pivot_rows_);
        if (d_log_pivot_cols_) cudaFree(d_log_pivot_cols_);
        if (d_counters_) cudaFree(d_counters_);
        if (d_row_scales_) cudaFree(d_row_scales_);
        if (d_block_results_) cudaFree((void*)d_block_results_);
    }

    void launch_laswp_disjoint(int k1, int k2, const int* ipiv, int row_start, int row_end, cudaStream_t stream) {
        if (k2 <= k1 || row_end <= row_start) return;
        int num_rows = row_end - row_start;
        int num_pairs = num_rows / 2;
        int threads = 256;
        int blocks = std::min((num_pairs + threads - 1) / threads, 160);
        if (blocks < 1) blocks = 1;
        batched_laswp_transposed_disjoint_kernel<<<blocks, threads, 0, stream>>>(
            d_B_, n_, k1, k2, ipiv, row_start, row_end
        );
    }

    void factorize_macro_panel(int k_macro, int W_actual, int method_code, float tau) {
        const int m_panel = n_ - k_macro;
        const int wb = micro_w_;
        const __half h_one = __float2half(1.0f);
        const __half h_zero = __float2half(0.0f);
        const __half h_minus_one = __float2half(-1.0f);
        dim3 tblk(TILE_DIM, BLOCK_ROWS);

        // Extract Macro-Panel from B to d_panel_buf
        dim3 sgrid((m_panel + TILE_DIM - 1) / TILE_DIM, (W_actual + TILE_DIM - 1) / TILE_DIM);
        transpose_submatrix_kernel<<<sgrid, tblk, 0, stream_panel_>>>(
            d_B_ + k_macro + static_cast<size_t>(k_macro) * n_, n_,
            d_panel_buf_, m_panel,
            m_panel, W_actual
        );

        // Factorize Macro-Panel using Micro-Panels
        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            int n_arg = m_panel;
            int k_arg = j_micro;
            int w_arg = wb_actual;
            int* d_pivot_slice = d_log_pivot_rows_ + k_macro;
            int* d_pivot_col_slice = d_log_pivot_cols_ + k_macro;
            float* d_row_scales_slice = (d_row_scales_) ? (d_row_scales_ + k_macro) : nullptr;
            void* args[] = {
                (void*)&d_panel_buf_, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
                (void*)&method_code, (void*)&tau, (void*)&d_row_scales_slice,
                (void*)&d_pivot_slice, (void*)&d_pivot_col_slice,
                (void*)&d_counters_, (void*)&d_block_results_
            };

            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)unified_fused_panel_kernel,
                dim3(fused_num_blocks_), dim3(FUSED_COOP_BLOCK),
                args, 0, stream_panel_
            ));

            const int rest_macro_cols = W_actual - (j_micro + wb_actual);
            if (rest_macro_cols > 0) {
                int lt = 256;
                int lg = (rest_macro_cols + lt - 1) / lt;
                batched_laswp_kernel<<<lg, lt, 0, stream_panel_>>>(
                    d_panel_buf_, m_panel, j_micro, j_micro + wb_actual,
                    d_pivot_slice, j_micro + wb_actual, W_actual
                );

                size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
                trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream_panel_>>>(
                    d_panel_buf_, m_panel, j_micro, wb_actual, d_L_inv_micro_
                );

                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_N,
                    wb_actual, rest_macro_cols, wb_actual,
                    &h_one,
                    d_L_inv_micro_, CUDA_R_16F, wb_actual,
                    d_panel_buf_ + (j_micro + static_cast<size_t>(j_micro + wb_actual) * m_panel), CUDA_R_16F, m_panel,
                    &h_zero,
                    d_u12_buf_, CUDA_R_16F, wb_actual,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));

                dim3 sblk(32, 8);
                dim3 sgrd((wb_actual + sblk.x - 1) / sblk.x, (rest_macro_cols + sblk.y - 1) / sblk.y);
                store_block_kernel<<<sgrd, sblk, 0, stream_panel_>>>(
                    d_u12_buf_, wb_actual, d_panel_buf_, m_panel, j_micro, j_micro + wb_actual, wb_actual, rest_macro_cols
                );

                const int m_macro_trail = m_panel - (j_micro + wb_actual);
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_N,
                    m_macro_trail, rest_macro_cols, wb_actual,
                    &h_minus_one,
                    d_panel_buf_ + ((j_micro + wb_actual) + static_cast<size_t>(j_micro) * m_panel), CUDA_R_16F, m_panel,
                    d_u12_buf_, CUDA_R_16F, wb_actual,
                    &h_one,
                    d_panel_buf_ + ((j_micro + wb_actual) + static_cast<size_t>(j_micro + wb_actual) * m_panel), CUDA_R_16F, m_panel,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
        }

        int opt_threads = 256;
        int opt_grid = (W_actual + opt_threads - 1) / opt_threads;
        offset_pivots_kernel<<<opt_grid, opt_threads, 0, stream_panel_>>>(
            d_log_pivot_rows_ + k_macro, W_actual, k_macro
        );

        if (k_macro > 0) {
            launch_laswp_disjoint(k_macro, k_macro + W_actual, d_log_pivot_rows_, 0, k_macro, stream_panel_);
        }

        // Transpose back to B
        dim3 bgrid((W_actual + TILE_DIM - 1) / TILE_DIM, (m_panel + TILE_DIM - 1) / TILE_DIM);
        transpose_submatrix_kernel<<<bgrid, tblk, 0, stream_panel_>>>(
            d_panel_buf_, m_panel,
            d_B_ + k_macro + static_cast<size_t>(k_macro) * n_, n_,
            W_actual, m_panel
        );
    }

    void trailing_trsm(int k_macro, int W_actual) {
        const int macro_end = k_macro + W_actual;
        const int n_trail = n_ - macro_end;
        const int m_panel = n_ - k_macro;
        const int wb = std::max(micro_w_, 64);
        const __half h_one = __float2half(1.0f);
        const __half h_zero = __float2half(0.0f);
        const __half h_minus_one = __float2half(-1.0f);

        for (int j_micro = 0; j_micro < W_actual; j_micro += wb) {
            const int micro_end = std::min(j_micro + wb, W_actual);
            const int wb_actual = micro_end - j_micro;

            size_t smem_bytes = static_cast<size_t>(wb_actual) * wb_actual * sizeof(float);
            trtri_unit_lower_kernel<<<1, 256, smem_bytes, stream_panel_>>>(
                d_panel_buf_, m_panel, j_micro, wb_actual, d_L_inv_micro_
            );

            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_T,
                n_trail, wb_actual, wb_actual,
                &h_one,
                d_B_ + macro_end + static_cast<size_t>(k_macro + j_micro) * n_, CUDA_R_16F, n_,
                d_L_inv_micro_, CUDA_R_16F, wb_actual,
                &h_zero,
                d_u12_buf_, CUDA_R_16F, n_trail,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));

            dim3 ublk(32, 8);
            dim3 ugrd((n_trail + ublk.x - 1) / ublk.x, (wb_actual + ublk.y - 1) / ublk.y);
            store_block_kernel<<<ugrd, ublk, 0, stream_panel_>>>(
                d_u12_buf_, n_trail,
                d_B_, n_,
                macro_end, k_macro + j_micro,
                n_trail, wb_actual
            );

            const int rest_trsm_cols = W_actual - (j_micro + wb_actual);
            if (rest_trsm_cols > 0) {
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_panel_, CUBLAS_OP_N, CUBLAS_OP_T,
                    n_trail, rest_trsm_cols, wb_actual,
                    &h_minus_one,
                    d_u12_buf_, CUDA_R_16F, n_trail,
                    d_panel_buf_ + ((j_micro + wb_actual) + static_cast<size_t>(j_micro) * m_panel), CUDA_R_16F, m_panel,
                    &h_one,
                    d_B_ + macro_end + static_cast<size_t>(k_macro + j_micro + wb_actual) * n_, CUDA_R_16F, n_,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
        }
    }

    void factorize(__half* d_A, int method_code, float tau) {
        const int n = n_;
        const int W = macro_w_;
        const __half h_one = __float2half(1.0f);
        const __half h_minus_one = __float2half(-1.0f);

        CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, kCounterSlots * sizeof(unsigned long long), stream_panel_));

        dim3 tblk(TILE_DIM, BLOCK_ROWS);
        dim3 tgrid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);
        transpose_matrix_kernel<<<tgrid, tblk, 0, stream_panel_>>>(d_A, d_B_, n, n, n, n);

        // Precompute row infinity-norms for ScPP from transposed matrix B
        if (method_code == 6) {
            compute_row_scales_from_transposed_kernel<<<n, 256, 0, stream_panel_>>>(d_B_, n, d_row_scales_);
        }

        int k_macro = 0;
        int macro_end = std::min(W, n);
        int W_actual = macro_end - k_macro;

        factorize_macro_panel(k_macro, W_actual, method_code, tau);

        while (macro_end < n) {
            const int n_trail = n - macro_end;

            launch_laswp_disjoint(k_macro, macro_end, d_log_pivot_rows_, macro_end, n, stream_panel_);
            trailing_trsm(k_macro, W_actual);

            const int k_next = macro_end;
            const int macro_next_end = std::min(k_next + W, n);
            const int W_next = macro_next_end - k_next;

            CUDA_CHECK(cudaEventRecord(ev_trsm_done_, stream_panel_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_gemm_, ev_trsm_done_, 0));

            // GEMM 1: Next panel
            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                W_next, n_trail, W_actual,
                &h_minus_one,
                d_B_ + macro_end + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                d_B_ + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                &h_one,
                d_B_ + macro_end + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
            CUDA_CHECK(cudaEventRecord(ev_next_panel_ready_, stream_gemm_));

            // GEMM 2a & 2b: Trailing quadrant split
            const int rest_rows = n_trail - W_next;
            if (rest_rows > 0) {
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                    rest_rows, W_next, W_actual,
                    &h_minus_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                    d_B_ + k_macro + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                    &h_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(macro_end) * n, CUDA_R_16F, n,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
                CUDA_CHECK(cudaEventRecord(ev_trsm_stripe_ready_, stream_gemm_));

                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle_gemm_, CUBLAS_OP_N, CUBLAS_OP_N,
                    rest_rows, rest_rows, W_actual,
                    &h_minus_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(k_macro) * n, CUDA_R_16F, n,
                    d_B_ + k_macro + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
                    &h_one,
                    d_B_ + (macro_end + W_next) + static_cast<size_t>(macro_end + W_next) * n, CUDA_R_16F, n,
                    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ));
            }
            CUDA_CHECK(cudaEventRecord(ev_gemm2_done_, stream_gemm_));

            // Concurrent Next Panel
            CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_next_panel_ready_, 0));
            factorize_macro_panel(k_next, W_next, method_code, tau);

            if (rest_rows > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_gemm2_done_, 0));
            }

            k_macro = k_next;
            macro_end = macro_next_end;
            W_actual = W_next;
        }

        if (n > W) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_panel_, ev_gemm2_done_, 0));
        }
        transpose_matrix_kernel<<<tgrid, tblk, 0, stream_panel_>>>(d_B_, d_A, n, n, n, n);

        CUDA_CHECK(cudaStreamSynchronize(stream_panel_));
        CUDA_CHECK(cudaStreamSynchronize(stream_gemm_));
    }
};

// ============================================================================
// Benchmark Driver Function
// ============================================================================
struct BenchResult {
    int n;
    int code;
    std::string name;
    float median_ms;
    float sem_ms;
    double tflops;
    float overhead_pct;
};

__device__ float simple_rand(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return -1.0f + 2.0f * (float(seed) / 4294967296.0f);
}

__global__ void init_matrix_kernel(__half* d_A, int n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(n) * n) {
        float val = simple_rand(1337 + n + idx);
        d_A[idx] = __float2half(val);
    }
}

void run_campaign(std::vector<int> orders) {
    if (orders.empty()) {
        orders = {512, 1024, 2048, 10240, 40960, 61440};
    }
    std::vector<std::pair<int, std::string>> methods = {
        {0, "PP"},
        {1, "DP"},
        {2, "GP"},
        {3, "ScaP"},
        {6, "ScPP"},
        {5, "CP"},
        {4, "RP"}
    };

    std::cout << "================================================================================" << std::endl;
    std::cout << "  FULL-SCALE FAIR BENCHMARK FOR ALL 7 FP16 LU PIVOTING POLICIES ON TESLA V100" << std::endl;
    std::cout << "  Every policy fully accelerated using cooperative fused panel & lookahead engine" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::vector<std::vector<BenchResult>> all_results;

    for (int n : orders) {
        std::cout << "\n>>> Benchmarking Order n = " << n << " ... " << std::endl;
        size_t bytes = static_cast<size_t>(n) * n * sizeof(__half);
        
        __half *d_A, *d_A_orig;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_A_orig, bytes));
        
        size_t total_elements = static_cast<size_t>(n) * n;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        // Cap blocks to avoid grid size limit if necessary, but 61440*61440/256 = 14,745,600 which is < 2^31-1.
        init_matrix_kernel<<<blocks, threads>>>(d_A_orig, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        int W = std::min(n, 512);
        int wb = std::min(W, 32);
        LookaheadTransposedFactorizer factorizer(n, W, wb);

        int warmups = (n <= 512) ? 60 : ((n <= 2048) ? 30 : ((n <= 10240) ? 8 : 2));
        int reps    = (n <= 512) ? 250 : ((n <= 2048) ? 50 : ((n <= 10240) ? 20 : 3));

        // GPU Clock Pre-warm to guarantee max boost clock
        for (int w = 0; w < (n <= 512 ? 100 : (n <= 2048 ? 20 : 5)); ++w) {
            CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, bytes, cudaMemcpyDeviceToDevice));
            factorizer.factorize(d_A, 0, 1.01f);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<BenchResult> n_results;
        float pp_time = 0.0f;

        for (const auto& m_pair : methods) {
            int code = m_pair.first;
            std::string name = m_pair.second;
            std::cout << "  [" << name << "] testing... " << std::flush;

            // Warmup
            for (int w = 0; w < warmups; ++w) {
                CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, bytes, cudaMemcpyDeviceToDevice));
                factorizer.factorize(d_A, code, 1.01f);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Timed runs
            std::vector<float> times;
            for (int r = 0; r < reps; ++r) {
                CUDA_CHECK(cudaMemcpy(d_A, d_A_orig, bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaEventRecord(start));
                factorizer.factorize(d_A, code, 1.01f);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                times.push_back(ms);
            }

            float sum_ms = 0.0f;
            for (float t : times) sum_ms += t;
            float clean_ms = sum_ms / times.size();

            float var = 0.0f;
            for (float t : times) var += (t - clean_ms) * (t - clean_ms);
            float std_dev = (times.size() > 1) ? std::sqrt(var / (times.size() - 1)) : 0.0f;
            float sem_ms = std_dev / std::sqrt(static_cast<float>(times.size()));

            double total_flops = (2.0 / 3.0) * std::pow(static_cast<double>(n), 3.0);
            double tflops = (total_flops / (clean_ms * 1e-3)) * 1e-12;

            if (code == 0) {
                pp_time = clean_ms;
            }

            float overhead = (code == 0) ? 0.0f : ((clean_ms - pp_time) / pp_time * 100.0f);

            n_results.push_back({n, code, name, clean_ms, sem_ms, tflops, overhead});
            std::cout << "    [" << std::setw(4) << name << "]  Time = " << std::fixed << std::setprecision(3)
                      << std::setw(8) << clean_ms << " ms ± " << std::setw(5) << sem_ms << " ms  |  "
                      << std::setw(6) << std::setprecision(3) << tflops << " TFLOPS  |  Overhead: "
                      << std::setw(7) << std::showpos << std::setprecision(3) << overhead << "%" << std::noshowpos
                      << std::endl;
        }

        all_results.push_back(n_results);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_A_orig));
    }

    // Print Final Comparative Table matching Paper Table 5 format
    std::cout << "\n\n================================================================================" << std::endl;
    std::cout << "             PAPER TABLE 5 (UPDATED WITH OPTIMIZED VOLTA GPU ENGINE)           " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::left << std::setw(8) << "Size"
              << std::right << std::setw(15) << "PP Time (ms)"
              << std::setw(11) << "DP (%)"
              << std::setw(11) << "GP (%)"
              << std::setw(11) << "ScaP (%)"
              << std::setw(11) << "ScPP (%)"
              << std::setw(11) << "CP (%)"
              << std::setw(11) << "RP (%)"
              << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (const auto& n_res : all_results) {
        int n = n_res[0].n;
        std::string size_str;
        if (n == 512) size_str = "0.5K";
        else size_str = std::to_string(n / 1024) + "K";
        
        float pp_ms = n_res[0].median_ms;
        float pp_sem = n_res[0].sem_ms;
        float dp_ov = n_res[1].overhead_pct;
        float gp_ov = n_res[2].overhead_pct;
        float scap_ov = n_res[3].overhead_pct;
        float scpp_ov = n_res[4].overhead_pct;
        float cp_ov = n_res[5].overhead_pct;
        float rp_ov = n_res[6].overhead_pct;

        std::cout << std::left << std::setw(8) << size_str
                  << std::right << std::setw(9) << std::fixed << std::setprecision(2) << pp_ms
                  << " ±" << std::setw(4) << pp_sem
                  << std::setw(9) << std::showpos << std::setprecision(2) << dp_ov << "%"
                  << std::setw(10) << gp_ov << "%"
                  << std::setw(10) << scap_ov << "%"
                  << std::setw(10) << scpp_ov << "%"
                  << std::setw(10) << cp_ov << "%"
                  << std::setw(10) << rp_ov << "%"
                  << std::noshowpos
                  << std::endl;
    }
    std::cout << "================================================================================" << std::endl;
}

int main(int argc, char** argv) {
    std::vector<int> orders;
    for (int i = 1; i < argc; ++i) {
        orders.push_back(std::stoi(argv[i]));
    }
    run_campaign(orders);
    return 0;
}
