#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../include/cuda_utils.cuh"
#include "../include/types.hpp"

namespace cg = cooperative_groups;
using namespace high_perf;

constexpr int FUSED_COOP_BLOCK = 256;
constexpr int FUSED_MAX_BLOCKS = 2048;

struct BlockPivotResult {
    unsigned long long best;
    unsigned long long second;
};

// Unified Fused Panel Kernel supporting all 7 policies:
// 0: PP, 1: DP, 2: GP, 3: ScaP, 4: RP, 5: CP, 6: ScPP
__global__ void unified_fused_panel_kernel(
    __half* __restrict__ matrix,
    int n,
    int k,
    int w_actual,
    int method_code,
    float tau,
    const float* __restrict__ row_scales,
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

        if (method_code == 0 || method_code == 6) {
            // PP (0) or ScPP (6)
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
        } else if (method_code == 1 || method_code == 2) {
            // DP (1) or GP (2)
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
        } else if (method_code == 3) {
            // ScaP (3): 3-column probe within panel
            const int c0 = j;
            const int c1 = j + (panel_end - 1 - j) / 2;
            const int c2 = panel_end - 1;
            const int num_probes = (panel_end - j >= 3) ? 3 : ((panel_end - j == 2) ? 2 : 1);

            for (int r = j + gid; r < n; r += gsize) {
                // Probe 0
                unsigned short b0 = absolute_bits(__ldg(&matrix[static_cast<size_t>(c0) * n + r]));
                if (bits_finite(b0)) {
                    unsigned long long k0 = (static_cast<unsigned long long>(b0) << 32) |
                                            ((static_cast<unsigned int>(r) << 8) | 0u);
                    if (k0 > local_best) local_best = k0;
                }
                // Probe 1
                if (num_probes > 1) {
                    unsigned short b1 = absolute_bits(__ldg(&matrix[static_cast<size_t>(c1) * n + r]));
                    if (bits_finite(b1)) {
                        unsigned long long k1 = (static_cast<unsigned long long>(b1) << 32) |
                                                ((static_cast<unsigned int>(r) << 8) | 1u);
                        if (k1 > local_best) local_best = k1;
                    }
                }
                // Probe 2
                if (num_probes > 2) {
                    unsigned short b2 = absolute_bits(__ldg(&matrix[static_cast<size_t>(c2) * n + r]));
                    if (bits_finite(b2)) {
                        unsigned long long k2 = (static_cast<unsigned long long>(b2) << 32) |
                                                ((static_cast<unsigned int>(r) << 8) | 2u);
                        if (k2 > local_best) local_best = k2;
                    }
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
            for (int r = j + gid; r < n; r += gsize) {
                unsigned short bits = absolute_bits(__ldg(&matrix[static_cast<size_t>(j) * n + r]));
                if (!bits_finite(bits)) continue;
                unsigned long long key = pack_key(bits, r);
                if (key > local_best) local_best = key;
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
                } else if (method_code == 2) {
                    // GP
                    float ratio = separation_ratio_device(best, second);
                    if ((n - j) > 1 && ratio < tau && !key_empty(second)) {
                        atomicAdd(&d_counters[kSlotGpNearTies], 1ull);
                        int r1 = pivot_r;
                        int r2 = key_index(second);
                        float norm1 = 0.0f, norm2 = 0.0f;
                        for (int c = j; c < panel_end; ++c) {
                            float v1 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r1]));
                            float v2 = fabsf(__half2float(matrix[static_cast<size_t>(c) * n + r2]));
                            if (v1 > norm1) norm1 = v1;
                            if (v2 > norm2) norm2 = v2;
                        }
                        if (norm2 < norm1) {
                            pivot_r = r2;
                            atomicAdd(&d_counters[kSlotGpSecondChoices], 1ull);
                        }
                    }
                } else if (method_code == 3) {
                    // ScaP
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
                } else if (method_code == 5) {
                    // CP
                    unsigned int packed = ~static_cast<unsigned int>(best & 0xffffffffull);
                    pivot_r = static_cast<int>(packed >> 16);
                    pivot_c = static_cast<int>(packed & 0xffffu);
                }

                s_pivot_row = pivot_r;
                s_pivot_col = pivot_c;
            }
            __syncthreads();

            if (method_code == 4) {
                // RP: Fast cooperative alternating walk using all 256 threads of Block 0
                int cur_r = s_pivot_row;
                int cur_c = s_pivot_col;
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
                        s_pivot_col = my_c;
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

                    int max_r = key_index(s_keys[0]);
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

void benchmark_policy(int method_code, const char* name, int m, int W, int wb) {
    size_t bytes = static_cast<size_t>(m) * W * sizeof(__half);
    __half *d_panel;
    CUDA_CHECK(cudaMalloc(&d_panel, bytes));
    CUDA_CHECK(cudaMemset(d_panel, 0x3c, bytes));

    int *d_piv_rows, *d_piv_cols;
    CUDA_CHECK(cudaMalloc(&d_piv_rows, W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_piv_cols, W * sizeof(int)));

    unsigned long long *d_counters;
    CUDA_CHECK(cudaMalloc(&d_counters, kCounterSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_counters, 0, kCounterSlots * sizeof(unsigned long long)));

    int num_blocks = 160;
    BlockPivotResult *d_block_res;
    CUDA_CHECK(cudaMalloc(&d_block_res, num_blocks * sizeof(BlockPivotResult)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int j = 0; j < W; j += wb) {
        int wb_act = std::min(wb, W - j);
        int n_arg = m;
        int k_arg = j;
        int w_arg = wb_act;
        float tau = 1.01f;
        float *row_scales = nullptr;
        void* args[] = {
            (void*)&d_panel, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
            (void*)&method_code, (void*)&tau, (void*)&row_scales,
            (void*)&d_piv_rows, (void*)&d_piv_cols,
            (void*)&d_counters, (void*)&d_block_res
        };
        cudaLaunchCooperativeKernel((void*)unified_fused_panel_kernel, dim3(num_blocks), dim3(FUSED_COOP_BLOCK), args, 0, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    int reps = 10;
    for (int it = 0; it < reps; ++it) {
        for (int j = 0; j < W; j += wb) {
            int wb_act = std::min(wb, W - j);
            int n_arg = m;
            int k_arg = j;
            int w_arg = wb_act;
            float tau = 1.01f;
            float *row_scales = nullptr;
            void* args[] = {
                (void*)&d_panel, (void*)&n_arg, (void*)&k_arg, (void*)&w_arg,
                (void*)&method_code, (void*)&tau, (void*)&row_scales,
                (void*)&d_piv_rows, (void*)&d_piv_cols,
                (void*)&d_counters, (void*)&d_block_res
            };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)unified_fused_panel_kernel, dim3(num_blocks), dim3(FUSED_COOP_BLOCK), args, 0, 0));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= reps;

    std::vector<unsigned long long> h_counters(kCounterSlots);
    CUDA_CHECK(cudaMemcpy(h_counters.data(), d_counters, kCounterSlots * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::cout << "Policy " << name << " (code " << method_code << "): Total Panel Time = " << ms << " ms";
    if (method_code == 1) std::cout << " | DP accepts: " << h_counters[kSlotDpAccepts];
    if (method_code == 2) std::cout << " | GP near-ties: " << h_counters[kSlotGpNearTies];
    if (method_code == 3) std::cout << " | ScaP probes (cur/mid/last): " << h_counters[kSlotScapCurrent] 
                                    << "/" << h_counters[kSlotScapMiddle] << "/" << h_counters[kSlotScapLast];
    if (method_code == 4) std::cout << " | RP iters: " << h_counters[kSlotRpIterations];
    std::cout << std::endl;

    cudaFree(d_panel);
    cudaFree(d_piv_rows);
    cudaFree(d_piv_cols);
    cudaFree(d_counters);
    cudaFree(d_block_res);
}

int main() {
    int m = 40960;
    int W = 512;
    int wb = 32;

    std::cout << "=== BENCHMARKING UNIFIED COOPERATIVE PANEL FOR ALL 7 POLICIES (m=" << m << ", W=" << W << ", wb=" << wb << ") ===" << std::endl;
    benchmark_policy(0, "PP  ", m, W, wb);
    benchmark_policy(1, "DP  ", m, W, wb);
    benchmark_policy(2, "GP  ", m, W, wb);
    benchmark_policy(3, "ScaP", m, W, wb);
    benchmark_policy(4, "RP  ", m, W, wb);
    benchmark_policy(5, "CP  ", m, W, wb);
    benchmark_policy(6, "ScPP", m, W, wb);

    return 0;
}
