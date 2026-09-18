
#pragma once

/*
 * fused_panel_kernel.cuh  (v2 – Cooperative Groups + Grid-wide sync)
 * ====================================================================
 * Persistent fused panel factorization using CUDA Cooperative Groups.
 *
 * Problem with single-block approach:
 *   Phase 4 (rank-1 update) for j=0, w=128, n=40960 needs to update
 *   ~5.2 million elements. A single block of 256 threads would take
 *   ~20,000 iterations per thread — slower than multiple blocks.
 *
 * Solution: Cooperative Grid Launch
 *   - Launch with a FULL grid (enough blocks to saturate SM occupancy)
 *   - Use cooperative_groups::this_grid().sync() between phases
 *   - Only ONE kernel dispatch per panel (vs 4*w dispatches before)
 *   - All phases (search, swap, scale, rank-1) run at full GPU utilization
 *
 * Requirements:
 *   - SM 7.0+ (V100 supports cooperative groups)
 *   - Launch via cudaLaunchCooperativeKernel() instead of <<<>>>
 *   - Device property cudaDevAttrCooperativeLaunch must be 1
 *
 * Grid sizing:
 *   - Use cudaOccupancyMaxActiveBlocksPerMultiprocessor to determine
 *     maximum blocks, then clamp to what's needed for rank-1 update.
 *   - Each block = 256 threads.
 *
 * Pivot search reduction:
 *   - Block-local reduction → block writes result to shared global array
 *   - Grid sync → block 0 does final reduction across all blocks
 *   - Grid sync → result broadcast
 *
 * For n ≤ 65535 and w ≤ 256 (both trivially satisfied).
 *
 * Methods supported: PP(0), DP(1), GP(2), ScPP(6)
 * NOT supported (use legacy path): ScaP(3), RP(4), CP(5)
 */

#include "cuda_utils.cuh"
#include "types.hpp"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace high_perf {

// Maximum number of blocks we allow for the cooperative launch.
// V100 has 80 SMs; with 256 threads/block, occupancy is typically 8 blocks/SM
// → max 640 blocks. We use up to 1024 to be safe.
constexpr int FUSED_COOP_BLOCK = 256;
constexpr int FUSED_MAX_BLOCKS = 2048;  // safe upper bound; actual is capped by device

// Per-block partial results for two-level pivot reduction
// Block 0 collects from d_block_best[0..num_blocks)
// Allocated once by the host, passed as kernel argument
struct BlockPivotResult {
    unsigned long long best;
    unsigned long long second;
};

// ---------------------------------------------------------------------------
// Cooperative Fused Panel Kernel
// ---------------------------------------------------------------------------
// Called ONCE per panel. The entire factorization of the panel (all w_actual
// columns: pivot search, row swap, column scale, rank-1 update) runs here.
//
// Arguments:
//   matrix          – n×n matrix in column-major FP16
//   n               – matrix order
//   k               – first column of this panel (changes per panel)
//   w_actual        – actual panel width for this step
//   method_code     – 0=PP, 1=DP, 2=GP, 6=ScPP
//   tau             – pivot threshold (for DP/GP)
//   row_scales      – per-row infinity norms (for ScPP only)
//   d_log_pivot_rows– output pivot log [n] (we write entries [k..k+w_actual))
//   d_counters      – atomic diagnostic counters
//   d_block_best    – scratch array [num_blocks] for two-level pivot reduction
// ---------------------------------------------------------------------------
__global__ void fused_panel_coop_kernel(
    __half* __restrict__ matrix,
    int n,
    int k,
    int w_actual,
    int method_code,
    float tau,
    const float* __restrict__ row_scales,
    int* __restrict__ d_log_pivot_rows,
    unsigned long long* __restrict__ d_counters,
    volatile BlockPivotResult* d_block_results
) {
    cg::grid_group grid = cg::this_grid();
    const int tid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int nblk  = gridDim.x;
    const int gid   = bid * blockDim.x + tid;  // global thread index
    const int gsize = nblk * blockDim.x;        // total threads in grid

    const int panel_end = k + w_actual;

    // Shared memory for block-local reduction and pivot row
    __shared__ unsigned long long s_keys[FUSED_COOP_BLOCK];
    __shared__ unsigned long long s_second[FUSED_COOP_BLOCK];
    __shared__ int s_pivot_row;

    // ========================================================================
    // Outer loop: one iteration per column of this panel
    // ========================================================================
    for (int j = k; j < panel_end; ++j) {

        // ====================================================================
        // Phase 1: Block-local pivot search
        // Each block scans a stripe of rows [j, n) for max in column j
        // ====================================================================
        unsigned long long local_best   = 0ull;
        unsigned long long local_second = 0ull;

        if (method_code == 0 || method_code == 3 || method_code == 6) {
            // PP / ScaP / ScPP: simple argmax
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<size_t>(j) * n + r]));
                if (!bits_finite(bits)) continue;
                float raw_val = (method_code == 6 && row_scales) ?
                    fabsf(__half2float(__ushort_as_half(bits))) / fmaxf(row_scales[r], 1e-30f) :
                    __half2float(__ushort_as_half(bits));
                unsigned short sbits = (method_code == 6) ? absolute_bits(__float2half(raw_val)) : bits;
                unsigned long long key = pack_key(sbits, r);
                if (key > local_best) local_best = key;
            }
        } else {
            // DP (1) / GP (2): need top-2 for separation ratio
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<size_t>(j) * n + r]));
                if (!bits_finite(bits)) continue;
                unsigned long long key = pack_key(bits, r);
                if (key > local_best) {
                    local_second = local_best;
                    local_best   = key;
                } else if (key > local_second) {
                    local_second = key;
                }
            }
        }

        // Block-level tree reduction
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

        // Block thread 0 writes to global scratch
        if (tid == 0) {
            d_block_results[bid].best   = s_keys[0];
            d_block_results[bid].second = s_second[0];
        }

        // ====================================================================
        // Grid sync: all blocks have written their partial results
        // ====================================================================
        grid.sync();

        // ====================================================================
        // Phase 1b: Final reduction across all blocks (done by block 0 only)
        //           + pivot decision + write to log
        // ====================================================================
        if (bid == 0) {
            // Final reduction across all blocks
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

            // Tree reduction within block 0
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

            // Thread 0 of block 0 makes the pivot decision
            if (tid == 0) {
                unsigned long long best   = s_keys[0];
                unsigned long long second = s_second[0];
                int pivot = key_empty(best) ? j : key_index(best);

                if (method_code == 1) {
                    // DP: accept diagonal if close enough
                    float diag_val = fabsf(__half2float(matrix[static_cast<size_t>(j) * n + j]));
                    float max_val  = key_magnitude(best);
                    if (max_val > 0.0f && diag_val >= max_val / tau) {
                        pivot = j;
                        atomicAdd(&d_counters[kSlotDpAccepts], 1ull);
                    } else {
                        atomicAdd(&d_counters[kSlotDpFallbacks], 1ull);
                    }
                } else if (method_code == 2) {
                    // GP: near-tie check (scalar, thread 0 only)
                    float ratio = separation_ratio_device(best, second);
                    if ((n - j) > 1 && ratio < tau && !key_empty(second)) {
                        atomicAdd(&d_counters[kSlotGpNearTies], 1ull);
                        int r1 = pivot;
                        int r2 = key_index(second);
                        float norm1 = 0.0f, norm2 = 0.0f;
                        for (int c = j; c < panel_end; ++c) {
                            float v1 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r1]));
                            float v2 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r2]));
                            if (v1 > norm1) norm1 = v1;
                            if (v2 > norm2) norm2 = v2;
                        }
                        if (norm2 < norm1) {
                            pivot = r2;
                            atomicAdd(&d_counters[kSlotGpSecondChoices], 1ull);
                        }
                    }
                } else if (method_code == 3) {
                    atomicAdd(&d_counters[kSlotScapCurrent], 1ull);
                }

                s_pivot_row = pivot;
                if (d_log_pivot_rows) d_log_pivot_rows[j] = pivot;
            }
            __syncthreads();
        }

        // ====================================================================
        // Grid sync: pivot row decided, broadcast s_pivot_row via d_block_results[0]
        // ====================================================================
        if (bid == 0 && tid == 0) {
            // Reuse best field to broadcast pivot_row to all blocks
            d_block_results[0].best = static_cast<unsigned long long>(s_pivot_row);
        }
        grid.sync();

        const int pivot_row = static_cast<int>(d_block_results[0].best);

        // ====================================================================
        // Phase 2: Row swap across panel columns [0, panel_end)
        // All threads in grid participate
        // ====================================================================
        if (pivot_row != j) {
            for (int col = gid; col < panel_end; col += gsize) {
                size_t idx1 = static_cast<size_t>(col) * n + j;
                size_t idx2 = static_cast<size_t>(col) * n + pivot_row;
                __half tmp = matrix[idx1];
                matrix[idx1] = matrix[idx2];
                matrix[idx2] = tmp;
            }
        }
        grid.sync();

        // ====================================================================
        // Phase 3: Scale column j below diagonal (rows j+1..n-1)
        // ====================================================================
        if (j + 1 < n) {
            const float pivot_val = __half2float(matrix[static_cast<size_t>(j) * n + j]);
            const float inv_pivot  = (fabsf(pivot_val) > 0.0f) ? (1.0f / pivot_val) : 0.0f;
            for (int r = (j + 1) + gid; r < n; r += gsize) {
                size_t idx = static_cast<size_t>(j) * n + r;
                matrix[idx] = __float2half(__half2float(matrix[idx]) * inv_pivot);
            }
        }
        grid.sync();

        // ====================================================================
        // Phase 4: Rank-1 update of remaining panel columns
        //   A[col][row] -= A[j][row] * A[col][j]   for col in (j, panel_end), row in (j, n)
        // ====================================================================
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

    } // end column loop
}

} // namespace high_perf
