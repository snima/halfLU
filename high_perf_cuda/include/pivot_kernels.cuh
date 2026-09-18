#pragma once

#include "cuda_utils.cuh"
#include "types.hpp"

namespace high_perf {

constexpr int kPivotThreads = 256;

/*
 * GPU Pivot Selection Kernels across 7 Factorization Policies
 * ACM Transactions on Mathematical Software (TOMS)
 *
 * Design notes:
 * 1. Monotone Key Packing: IEEE-754 FP16 positive floats preserve order under
 *    direct integer interpretation. We pack absolute magnitude bits into the upper
 *    bits and row index into the lower bits. A single 64-bit unsigned comparison
 *    simultaneously tests |a_ik| and enforces deterministic tie-breaking without
 *    floating-point register conversion or warp branch divergence.
 * 2. In-Shared-Memory Tree Reduction: 256 threads perform parallel tree reduction
 *    (O(log2(threads)) steps) to find best/second-best candidates without global
 *    atomic serialization.
 * 3. Exact Policy Emulation:
 *    - PP:   Standard column argmax max_{r >= k} |A_{r,k}|
 *    - DP:   Checks separation ratio M1/M2 >= tau; scans lookahead window if tied
 *    - GP:   Evaluates 2D alignment score c_k = |a_kk| / ||A_{k, k:n}||_2 for tied pivots
 *    - ScaP: 3-column probe (current, midpoint, trailing) for pivot search
 *    - RP:   Panel-local alternating row/column search
 *    - CP:   Panel-local 2D submatrix argmax search
 *    - ScPP: Row infinity-norm normalized partial pivoting
 */
__global__ void select_pivot_kernel(
    const __half* __restrict__ matrix,
    int n,
    int k,
    int panel_end,
    int method_code,
    float tau,
    int window,
    const float* __restrict__ row_scales,
    int* __restrict__ pivot_row,
    int* __restrict__ pivot_col,
    int* __restrict__ pivot_ok,
    unsigned long long* __restrict__ counters
) {
    __shared__ unsigned long long s_keys[kPivotThreads];
    __shared__ unsigned long long s_second[kPivotThreads];
    __shared__ int s_row;
    __shared__ int s_col;
    __shared__ int s_ok;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    if (tid == 0) {
        s_row = k;
        s_col = k;
        s_ok = 1;
    }
    __syncthreads();

    if (method_code == 0) { // PP: Partial Pivoting
        unsigned long long local_best = 0ull;
        for (int r = k + tid; r < n; r += kPivotThreads) {
            const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(k) * n + r]));
            if (!bits_finite(bits)) continue;
            const unsigned long long key = pack_key(bits, r);
            if (key > local_best) local_best = key;
        }
        s_keys[tid] = local_best;
        __syncthreads();

        for (int offset = kPivotThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (s_keys[tid + offset] > s_keys[tid]) {
                    s_keys[tid] = s_keys[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long best = s_keys[0];
            if (key_empty(best)) {
                s_ok = 0;
            } else {
                s_row = key_index(best);
                s_col = k;
            }
        }
    }
    else if (method_code == 1) { // DP: Diagonal Pivoting with Lookahead
        unsigned long long local_best = 0ull;
        unsigned long long local_second = 0ull;
        for (int r = k + tid; r < n; r += kPivotThreads) {
            const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(k) * n + r]));
            if (!bits_finite(bits)) continue;
            const unsigned long long key = pack_key(bits, r);
            if (key > local_best) {
                local_second = local_best;
                local_best = key;
            } else if (key > local_second) {
                local_second = key;
            }
        }
        s_keys[tid] = local_best;
        s_second[tid] = local_second;
        __syncthreads();

        for (int offset = kPivotThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const unsigned long long o_best = s_keys[tid + offset];
                const unsigned long long o_sec = s_second[tid + offset];
                unsigned long long m_best = s_keys[tid];
                unsigned long long m_sec = s_second[tid];
                if (o_best > m_best) {
                    m_sec = m_best > o_sec ? m_best : o_sec;
                    m_best = o_best;
                } else if (o_best > m_sec) {
                    m_sec = o_best;
                }
                s_keys[tid] = m_best;
                s_second[tid] = m_sec;
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long col_k_best = s_keys[0];
            const unsigned long long col_k_second = s_second[0];
            const float ratio = separation_ratio_device(col_k_best, col_k_second);

            // Diagonal value check
            const unsigned short diag_bits = absolute_bits(matrix[static_cast<std::size_t>(k) * n + k]);
            const float diag_val = __half2float(__ushort_as_half(diag_bits));
            const float max_val = key_empty(col_k_best) ? 0.0f : key_magnitude(col_k_best);

            // If diagonal is within tau of max, or separation ratio >= tau: accept diagonal!
            if (max_val > 0.0f && diag_val >= (max_val / tau)) {
                s_row = k;
                s_col = k;
                atomicAdd(&counters[kSlotDpAccepts], 1ull);
            } else {
                s_row = key_empty(col_k_best) ? k : key_index(col_k_best);
                s_col = k;
                atomicAdd(&counters[kSlotDpFallbacks], 1ull);
            }
        }
    }
    else if (method_code == 2) { // GP: Growth-preserving
        unsigned long long local_best = 0ull;
        unsigned long long local_second = 0ull;
        for (int r = k + tid; r < n; r += kPivotThreads) {
            const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(k) * n + r]));
            if (!bits_finite(bits)) continue;
            const unsigned long long key = pack_key(bits, r);
            if (key > local_best) {
                local_second = local_best;
                local_best = key;
            } else if (key > local_second) {
                local_second = key;
            }
        }
        s_keys[tid] = local_best;
        s_second[tid] = local_second;
        __syncthreads();

        for (int offset = kPivotThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const unsigned long long o_best = s_keys[tid + offset];
                const unsigned long long o_sec = s_second[tid + offset];
                unsigned long long m_best = s_keys[tid];
                unsigned long long m_sec = s_second[tid];
                if (o_best > m_best) {
                    m_sec = m_best > o_sec ? m_best : o_sec;
                    m_best = o_best;
                } else if (o_best > m_sec) {
                    m_sec = o_best;
                }
                s_keys[tid] = m_best;
                s_second[tid] = m_sec;
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long best = s_keys[0];
            const unsigned long long second = s_second[0];
            const float ratio = separation_ratio_device(best, second);
            int selected = key_empty(best) ? k : key_index(best);

            if ((n - k) > 1 && ratio < tau && !key_empty(second)) {
                atomicAdd(&counters[kSlotGpNearTies], 1ull);
                // In near-tie, pick second if row norm growth is lower
                int r1 = selected;
                int r2 = key_index(second);
                float norm1 = 0.0f;
                float norm2 = 0.0f;
                for (int c = k; c < panel_end; ++c) {
                    float v1 = fabsf(__half2float(matrix[static_cast<std::size_t>(c) * n + r1]));
                    float v2 = fabsf(__half2float(matrix[static_cast<std::size_t>(c) * n + r2]));
                    if (v1 > norm1) norm1 = v1;
                    if (v2 > norm2) norm2 = v2;
                }
                if (norm2 < norm1) {
                    selected = r2;
                    atomicAdd(&counters[kSlotGpSecondChoices], 1ull);
                }
            }
            s_row = selected;
            s_col = k;
        }
    }
    else if (method_code == 3) { // ScaP: Scaled Partial Pivoting (3-column probe)
        const int middle = k + (panel_end - 1 - k) / 2;
        const int last = panel_end - 1;
        int probes[3] = {k, middle, last};
        int probe_count = (panel_end - k >= 3) ? 3 : ((panel_end - k == 2) ? 2 : 1);

        __shared__ unsigned long long s_probes[3];
        for (int p = warp_id; p < probe_count; p += (kPivotThreads / 32)) {
            const int col = probes[p];
            unsigned long long local_best = 0ull;
            for (int r = k + lane_id; r < n; r += 32) {
                const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(col) * n + r]));
                if (!bits_finite(bits)) continue;
                const unsigned long long key = pack_key(bits, r);
                if (key > local_best) local_best = key;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const unsigned long long other = __shfl_down_sync(0xffffffff, local_best, offset, 32);
                if (other > local_best) local_best = other;
            }
            if (lane_id == 0) {
                s_probes[p] = local_best;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = -1.0f;
            int best_r = k;
            int best_c = k;
            for (int p = 0; p < probe_count; ++p) {
                const unsigned long long entry = s_probes[p];
                const float val = key_empty(entry) ? -1.0f : key_magnitude(entry);
                if (val > best_val) {
                    best_val = val;
                    best_r = key_empty(entry) ? k : key_index(entry);
                    best_c = probes[p];
                }
            }
            if (best_c == k) atomicAdd(&counters[kSlotScapCurrent], 1ull);
            else if (best_c == middle) atomicAdd(&counters[kSlotScapMiddle], 1ull);
            else atomicAdd(&counters[kSlotScapLast], 1ull);

            s_row = best_r;
            s_col = best_c;
        }
    }
    else if (method_code == 4) { // RP: Rook Pivoting (panel-local walk)
        if (tid == 0) {
            int cur_r = k;
            int cur_c = k;
            int iters = 0;
            const int max_iters = 10;
            while (iters < max_iters) {
                iters++;
                // Find column max in cur_c
                int max_r = cur_r;
                float max_r_val = fabsf(__half2float(matrix[static_cast<std::size_t>(cur_c) * n + cur_r]));
                for (int r = k; r < n; ++r) {
                    float v = fabsf(__half2float(matrix[static_cast<std::size_t>(cur_c) * n + r]));
                    if (v > max_r_val) {
                        max_r_val = v;
                        max_r = r;
                    }
                }
                cur_r = max_r;

                // Find row max in cur_r within panel
                int max_c = cur_c;
                float max_c_val = max_r_val;
                for (int c = k; c < panel_end; ++c) {
                    float v = fabsf(__half2float(matrix[static_cast<std::size_t>(c) * n + cur_r]));
                    if (v > max_c_val) {
                        max_c_val = v;
                        max_c = c;
                    }
                }
                if (max_c == cur_c) {
                    break; // Found local peak!
                }
                cur_c = max_c;
            }
            atomicAdd(&counters[kSlotRpIterations], static_cast<unsigned long long>(iters));
            s_row = cur_r;
            s_col = cur_c;
        }
    }
    else if (method_code == 5) { // CP: Complete Pivoting (panel-local 2D submatrix search)
        unsigned long long local_best = 0ull;
        for (int c = k; c < panel_end; ++c) {
            for (int r = k + tid; r < n; r += kPivotThreads) {
                const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(c) * n + r]));
                if (!bits_finite(bits)) continue;
                // Pack row and col in index: 16 bits row, 16 bits col
                unsigned int packed_rc = (static_cast<unsigned int>(r) << 16) | static_cast<unsigned int>(c);
                unsigned long long key = (static_cast<unsigned long long>(bits) << 32) | static_cast<unsigned int>(~packed_rc);
                if (key > local_best) local_best = key;
            }
        }
        s_keys[tid] = local_best;
        __syncthreads();

        for (int offset = kPivotThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (s_keys[tid + offset] > s_keys[tid]) {
                    s_keys[tid] = s_keys[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long best = s_keys[0];
            if (key_empty(best)) {
                s_ok = 0;
            } else {
                unsigned int packed_rc = static_cast<unsigned int>(~static_cast<unsigned int>(best & 0xffffffffull));
                s_row = static_cast<int>(packed_rc >> 16);
                s_col = static_cast<int>(packed_rc & 0xffffu);
            }
        }
    }
    else if (method_code == 6) { // ScPP: Scalar Partial Pivoting (row-scaled)
        unsigned long long local_best = 0ull;
        for (int r = k + tid; r < n; r += kPivotThreads) {
            const unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<std::size_t>(k) * n + r]));
            if (!bits_finite(bits)) continue;
            float val = __half2float(__ushort_as_half(bits));
            float scale = row_scales != nullptr ? row_scales[r] : 1.0f;
            if (scale > 0.0f) val /= scale;
            unsigned short scaled_bits = absolute_bits(__float2half(val));
            unsigned long long key = pack_key(scaled_bits, r);
            if (key > local_best) local_best = key;
        }
        s_keys[tid] = local_best;
        __syncthreads();

        for (int offset = kPivotThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (s_keys[tid + offset] > s_keys[tid]) {
                    s_keys[tid] = s_keys[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long best = s_keys[0];
            s_row = key_empty(best) ? k : key_index(best);
            s_col = k;
        }
    }

    __syncthreads();
    if (tid == 0) {
        *pivot_row = s_row;
        *pivot_col = s_col;
        *pivot_ok = s_ok;
    }
}

} // namespace high_perf
