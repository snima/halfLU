// ============================================================================
// TABLE 7 EXTENSION -- Fused device-side pivot search cost validator
// ============================================================================
//
// NEW CODE (post-submission extension). Do not confuse with the frozen
// submission validator.
//
//   Baseline validator:
//     cuda/pivoting_runtime_validation.cu
//   This file:
//     cuda/pivoting_search_cost_validation.cu
//
// PURPOSE
// -------
// The frozen validator performs every pivot search on the host. Each
// elimination step therefore pays 6-8 synchronous host/device round trips, and
// the CP branch additionally copies the whole n x n matrix device->host on
// every step. The published overheads in Table 7 (CP +960.2%, ScaP +25.3%,
// RP +20.3%) are dominated by that harness cost, not by the arithmetic of the
// pivot rules: under the blocked_panel_local schedule CP searches only
// [k, panel_end) x [k, n), i.e. O(w n) comparisons per step, so its own
// complexity predicts a negligible overhead.
//
// This file adds a second, selectable search path that removes both effects
// while leaving the factorization arithmetic bit-for-bit unchanged:
//
//   --search host          Verbatim frozen behaviour. Reproduces Table 7.
//   --search fused_device  (A) The per-column and per-row argmax needed by the
//                              NEXT elimination step is captured as a side
//                              effect of the trailing update that already
//                              writes every one of those elements, using one
//                              64-bit atomicMax on a monotone packed key.
//                          (B) The pivot decision, the swaps, the scaling and
//                              the update all consume device-resident indices,
//                              so the step contains no cudaMemcpy and no
//                              cudaDeviceSynchronize at all.
//
// EQUIVALENCE CONTRACT
// --------------------
// The two paths must select the IDENTICAL pivot sequence and produce a
// bit-identical factorization. The packed key stores the bitwise complement of
// the index so that atomicMax breaks magnitude ties toward the LOWEST index,
// matching the ascending strict-greater-than scan of the host reference in
// maximum_index()/top_two(). verify_equivalence.sh checks this; a run reports
// the outcome in the new CSV columns pivot_log_sha1 and search_mode.
//
// WHAT THIS FILE DOES NOT CHANGE
// ------------------------------
// Dynamic scaling remains disabled, exactly as in the frozen validator, so the
// predictive range certificate is still not attributed to any pivot rule other
// than PP. Matrix generation, the FP16 arithmetic sequence, the reconstruction
// diagnostic and the CSV schema of the parent are preserved.
//
// See README.md and ALGORITHM_CONTRACT_EXT.md in this folder.
// ============================================================================
//
// This executable intentionally does not implement dynamic scaling. The paper's
// primary predictive method is validated by codes/dynamic_scaling_validation.cu
// with fixed partial row pivoting. Keeping the paths separate prevents the
// predictive PP certificate from being attributed to unvalidated pivot rules.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
            cudaGetErrorString(status));
    }
}

std::size_t index_of(int row, int column, int n) {
    return static_cast<std::size_t>(column) * n + row; // column major
}

enum class Method { kPP, kDP, kGP, kScaP, kRP, kCP, kScPP };
enum class Schedule { kUnblockedRank1, kBlockedPanelLocal };

// NEW: which implementation of the pivot search is timed.
//   kHost        = frozen parent behaviour (per-step host round trips).
//   kFusedDevice = argmax fused into the trailing-update epilogue, pivot
//                  decision and swaps consume device-resident indices.
enum class SearchMode { kHost, kFusedDevice };

enum class Family {
    kUniform,
    kNormal,
    kGraded,
    kWilkinson,
    kDpEquality,
    kScaPLast,
    kGpSecond,
};

struct Options {
    Method method = Method::kPP;
    Schedule schedule = Schedule::kUnblockedRank1;
    SearchMode search = SearchMode::kHost; // NEW: default reproduces the parent
    Family family = Family::kUniform;
    int n = 256;
    int panel_width = 0;
    std::uint64_t seed = 20260823ULL;
    float tau = 1.01f;
    int window = 6;
    int warmups = 1;
    int repetitions = 3;
    bool log_pivots = false; // NEW: emit the pivot sequence digest for A/B proof
    // NEW: fused path only. 0 = never poll the non-finite counter inside the
    // elimination loop (fastest; a failing run is classified at the end as
    // "nonfinite_update"). 1 = poll wherever the frozen host path polls, which
    // reproduces its early-abort step and its "nonfinite_factorization" status
    // exactly. Used by verify_equivalence.sh on the overflow families.
    int abort_check_stride = 0;
    std::string output = "pivoting_runs.csv";
    std::string pivot_log_path; // NEW: optional full pivot sequence dump
};

struct Counters {
    std::uint64_t row_swaps = 0;
    std::uint64_t column_swaps = 0;
    std::uint64_t dp_lookahead_accepts = 0;
    std::uint64_t dp_fallbacks = 0;
    std::uint64_t gp_near_ties = 0;
    std::uint64_t gp_second_choices = 0;
    std::uint64_t scap_current = 0;
    std::uint64_t scap_middle = 0;
    std::uint64_t scap_last = 0;
    std::uint64_t rp_iterations = 0;
    std::uint64_t rp_failures = 0;
};

struct PivotChoice {
    int row = -1;
    int column = -1;
    bool success = false;
};

struct RunResult {
    std::string status = "not_run";
    double wall_ms = std::numeric_limits<double>::quiet_NaN();
    float cuda_ms = std::numeric_limits<float>::quiet_NaN();
    std::string reconstruction_kind = "not_run";
    double reconstruction_residual = std::numeric_limits<double>::quiet_NaN();
    double growth_factor = std::numeric_limits<double>::quiet_NaN();
    float max_multiplier = 0.0f;
    unsigned long long nonfinite_count = 0;
    Counters counters;
    // NEW: the realised pivot sequence, so that --search host and
    // --search fused_device can be proven to define the same algorithm.
    std::vector<int> pivot_rows;
    std::vector<int> pivot_columns;
    std::string pivot_digest = "not_run";
};

// NEW: fixed slot layout for the device-side counter mirror. Keeping the
// ordering explicit means the fused path reports exactly the same activation
// frequencies (DP accepts, GP near ties, ScaP current/middle/last, RP
// iterations) that the frozen campaign used as evidence that each rule is
// actually active rather than a disguised alias for PP.
enum CounterSlot {
    kSlotRowSwaps = 0,
    kSlotColumnSwaps,
    kSlotDpAccepts,
    kSlotDpFallbacks,
    kSlotGpNearTies,
    kSlotGpSecondChoices,
    kSlotScapCurrent,
    kSlotScapMiddle,
    kSlotScapLast,
    kSlotRpIterations,
    kSlotRpFailures,
    kCounterSlots,
};

// NEW: device status codes, mirrored from the host status strings so the fused
// path can abort without a synchronising read.
enum DeviceStatus {
    kStatusOk = 0,
    kStatusPivotSearchFailed = 1,
    kStatusNonfinitePivot = 2,
    kStatusSingularPivot = 3,
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count = 0) : count_(count) {
        if (count_ != 0) CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return data_; }
    const T* get() const { return data_; }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void swap_rows_kernel(__half* matrix, int n, int first, int second) {
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < n;
         column += blockDim.x * gridDim.x) {
        const std::size_t a = static_cast<std::size_t>(column) * n + first;
        const std::size_t b = static_cast<std::size_t>(column) * n + second;
        const __half value = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = value;
    }
}

__global__ void swap_columns_kernel(__half* matrix, int n, int first, int second) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n;
         row += blockDim.x * gridDim.x) {
        const std::size_t a = static_cast<std::size_t>(first) * n + row;
        const std::size_t b = static_cast<std::size_t>(second) * n + row;
        const __half value = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = value;
    }
}

__global__ void divide_column_kernel(
    __half* matrix,
    int n,
    int pivot,
    unsigned int* multiplier_max_bits,
    unsigned long long* nonfinite) {
    const int row = pivot + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const __half quotient = __hdiv(
        matrix[static_cast<std::size_t>(pivot) * n + row],
        matrix[static_cast<std::size_t>(pivot) * n + pivot]);
    matrix[static_cast<std::size_t>(pivot) * n + row] = quotient;
    const float magnitude = fabsf(__half2float(quotient));
    if (isfinite(magnitude)) atomicMax(multiplier_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
}

// ---------------------------------------------------------------------------
// NEW: monotone packed search key.
//
// For a non-negative IEEE-754 binary16 value the 16-bit pattern is monotone
// increasing in the value, so an integer comparison of |x| bit patterns is
// exactly a magnitude comparison. The low word stores the BITWISE COMPLEMENT of
// the index, which makes a plain atomicMax break magnitude ties toward the
// LOWEST index -- identical to the ascending strict-greater-than scan used by
// the host reference maximum_index()/top_two().
//
// Ordering the pairs by this single key also makes "top two" order independent,
// which is what allows the device tree reduction to reproduce the host
// sequential scan bit-for-bit.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ unsigned long long pack_key(
    unsigned short magnitude_bits, int index) {
    return (static_cast<unsigned long long>(magnitude_bits) << 32) |
           static_cast<unsigned long long>(~static_cast<unsigned int>(index));
}

__host__ __device__ __forceinline__ unsigned short key_magnitude_bits(unsigned long long key) {
    return static_cast<unsigned short>((key >> 32) & 0xffffull);
}

__host__ __device__ __forceinline__ int key_index(unsigned long long key) {
    return static_cast<int>(~static_cast<unsigned int>(key & 0xffffffffull));
}

// A key of exactly zero can only arise from an untouched slot: any recorded
// element with index i < 2^31 contributes at least ~i > 0 in the low word.
__host__ __device__ __forceinline__ bool key_empty(unsigned long long key) {
    return key == 0ull;
}

__device__ __forceinline__ float key_magnitude(unsigned long long key) {
    return __half2float(__ushort_as_half(key_magnitude_bits(key)));
}

// |x| bit pattern, and a finiteness test that matches the host isfinite() guard.
__device__ __forceinline__ unsigned short absolute_bits(__half value) {
    return static_cast<unsigned short>(__half_as_ushort(value) & 0x7fffu);
}

__device__ __forceinline__ bool bits_finite(unsigned short magnitude_bits) {
    return (magnitude_bits & 0x7c00u) != 0x7c00u;
}

__global__ void rank1_update_kernel(
    __half* matrix,
    int n,
    int pivot,
    int row_start,
    int row_end,
    int column_start,
    int column_end,
    unsigned int* growth_max_bits,
    unsigned long long* nonfinite,
    // NEW: fused search state for the NEXT elimination step. Both may be null,
    // in which case this kernel is byte-for-byte the frozen parent kernel.
    unsigned long long* column_argmax,
    unsigned long long* row_argmax,
    int row_argmax_column_end) {
    const int column = column_start + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_start + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= row_end || column >= column_end) return;
    const std::size_t entry = static_cast<std::size_t>(column) * n + row;
    const __half multiplier = matrix[static_cast<std::size_t>(pivot) * n + row];
    const __half pivot_value = matrix[static_cast<std::size_t>(column) * n + pivot];
    const __half updated = __hsub(matrix[entry], __hmul(multiplier, pivot_value));
    matrix[entry] = updated;
    const float magnitude = fabsf(__half2float(updated));
    if (isfinite(magnitude)) atomicMax(growth_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
    // Fused epilogue: the element is already in a register and already stored,
    // so the argmax costs one integer atomic and zero extra memory traffic.
    const unsigned short bits = absolute_bits(updated);
    if (bits_finite(bits)) {
        if (column_argmax != nullptr) {
            atomicMax(&column_argmax[column], pack_key(bits, row));
        }
        if (row_argmax != nullptr && column < row_argmax_column_end) {
            atomicMax(&row_argmax[row], pack_key(bits, column));
        }
    }
}

__global__ void blocked_update_kernel(
    __half* matrix,
    int n,
    int panel_start,
    int panel_end,
    unsigned int* growth_max_bits,
    unsigned long long* nonfinite,
    unsigned long long* column_argmax,
    unsigned long long* row_argmax,
    int row_argmax_column_end) {
    const int column = panel_end + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = panel_end + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= n) return;
    const std::size_t entry = static_cast<std::size_t>(column) * n + row;
    __half updated = matrix[entry];
    for (int pivot = panel_start; pivot < panel_end; ++pivot) {
        const __half multiplier = matrix[static_cast<std::size_t>(pivot) * n + row];
        const __half pivot_value = matrix[static_cast<std::size_t>(column) * n + pivot];
        updated = __hsub(updated, __hmul(multiplier, pivot_value));
    }
    matrix[entry] = updated;
    const float magnitude = fabsf(__half2float(updated));
    if (isfinite(magnitude)) atomicMax(growth_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
    const unsigned short bits = absolute_bits(updated);
    if (bits_finite(bits)) {
        if (column_argmax != nullptr) {
            atomicMax(&column_argmax[column], pack_key(bits, row));
        }
        if (row_argmax != nullptr && column < row_argmax_column_end) {
            atomicMax(&row_argmax[row], pack_key(bits, column));
        }
    }
}

// ===========================================================================
// NEW: device-resident pivot search (SearchMode::kFusedDevice)
//
// Every kernel below is a no-op once *status leaves kStatusOk, so a failing run
// unwinds without any synchronising read from the host.
// ===========================================================================

__device__ const double kDoubleTiny = 2.2250738585072014e-308; // numeric_limits<double>::min()

// Primes the fused state for the very first elimination step, which has no
// preceding trailing update. O(n^2) once per factorization.
__global__ void prime_argmax_kernel(
    const __half* matrix,
    int n,
    int row_start,
    int column_start,
    int column_end,
    unsigned long long* column_argmax,
    unsigned long long* row_argmax,
    int row_argmax_column_end) {
    const int column = column_start + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_start + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= column_end) return;
    const unsigned short bits = absolute_bits(matrix[static_cast<std::size_t>(column) * n + row]);
    if (bits_finite(bits)) {
        if (column_argmax != nullptr) atomicMax(&column_argmax[column], pack_key(bits, row));
    }
    if (row_argmax != nullptr) {
        unsigned long long r_key = (column < row_argmax_column_end && bits_finite(bits))
            ? pack_key(bits, column)
            : 0ull;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(0xffffffff, r_key, offset, 16);
            if (other > r_key) r_key = other;
        }
        if (threadIdx.x == 0 && r_key != 0ull) {
            atomicMax(&row_argmax[row], r_key);
        }
    }
}

// Computes row infinity-norms directly on GPU for ScPP, eliminating the O(n^2)
// CPU bottleneck at large matrix sizes (e.g. n=61440).
__global__ void compute_row_scales_kernel(const __half* matrix, int n, float* row_scales) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float maximum = 0.0f;
    for (int column = 0; column < n; ++column) {
        const float val = fabsf(__half2float(matrix[static_cast<std::size_t>(column) * n + row]));
        if (val > maximum) maximum = val;
    }
    row_scales[row] = (maximum == 0.0f) ? 1.0f : maximum;
}

// Top two entries of each of `column_count` consecutive columns over rows
// Note: In this optimized fused_device implementation, the top-two reductions
// for DP and GP and the weighted argmax for ScPP are inlined directly into
// select_pivot_device_kernel's preambles, eliminating separate kernel launches.

__device__ __forceinline__ float separation_ratio_device(
    unsigned long long best, unsigned long long second) {
    const float maximum = key_empty(best) ? -1.0f : key_magnitude(best);
    const float runner_up = key_empty(second) ? 0.0f : key_magnitude(second);
    if (runner_up == 0.0f) return maximum > 0.0f ? __int_as_float(0x7f800000) : 0.0f;
    return maximum / runner_up;
}

// One block. Consumes only the small device-side summaries produced above and
// writes the decision, the pivot log entry and the activation counters. No host
// participation, therefore no synchronisation.
__global__ void select_pivot_device_kernel(
    const __half* matrix,
    int n,
    int k,
    int selection_column_end,
    int method_code, // 0 PP, 1 DP, 2 GP, 3 ScaP, 4 RP, 5 CP, 6 ScPP
    float tau,
    int window,
    const unsigned long long* column_argmax,
    const unsigned long long* row_argmax,
    unsigned long long* column_top_two,  // DP still filled by column_top_two_kernel;
                                          // GP is now filled in-kernel below (was
                                          // column_top_two_kernel<<<1,kThreads>>>).
    const float* row_scales,             // NEW: lets ScPP score in-kernel.
    double* scpp_score,                  // now written in-kernel for ScPP (was
    int* scpp_row,                       // scaled_column_argmax_kernel<<<1,kThreads>>>).
    int* choice,
    int* pivot_log,
    unsigned long long* counters,
    int* status) {
    if (*status != kStatusOk) return;

    __shared__ unsigned long long shared_key[kThreads];
    __shared__ int shared_row[kThreads];
    __shared__ int result_row;
    __shared__ int result_column;
    __shared__ int result_ok;
    __shared__ float result_magnitude;

    // ------------------------------------------------------------------------
    // OPTIMISATION V3 (fused_device path only; --search host and its reference
    // select_pivot() on the CPU are untouched byte-for-byte).
    //
    // 1. DP uses Lazy Evaluation: all 256 threads compute column k with full
    //    block parallelism first. If ratio >= tau (>90% of steps), column k
    //    is accepted immediately and columns k+1..k+w are NEVER read from DRAM.
    //    If ratio < tau (near-tie fallback), warps 0..7 compute remaining
    //    columns in parallel via __shfl_down_sync.
    // 2. GP, ScPP, and RP use __ldg read-only caching for matrix and scale reads.
    // ------------------------------------------------------------------------
    if (method_code == 1) { // DP: Lazy top-two evaluation
        __shared__ unsigned long long dp_best[kThreads];
        __shared__ unsigned long long dp_second[kThreads];
        unsigned long long local_best = 0ull;
        unsigned long long local_second = 0ull;
        // Step 1: All 256 threads compute column k in parallel
        for (int row = k + threadIdx.x; row < n; row += kThreads) {
            const unsigned short bits = absolute_bits(matrix[static_cast<std::size_t>(k) * n + row]);
            if (!bits_finite(bits)) continue;
            const unsigned long long key = pack_key(bits, row);
            if (key > local_best) { local_second = local_best; local_best = key; }
            else if (key > local_second) { local_second = key; }
        }
        dp_best[threadIdx.x] = local_best;
        dp_second[threadIdx.x] = local_second;
        __syncthreads();
        for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                const unsigned long long other_best = dp_best[threadIdx.x + offset];
                const unsigned long long other_second = dp_second[threadIdx.x + offset];
                unsigned long long mine_best = dp_best[threadIdx.x];
                unsigned long long mine_second = dp_second[threadIdx.x];
                if (other_best > mine_best) {
                    mine_second = mine_best > other_second ? mine_best : other_second;
                    mine_best = other_best;
                } else if (other_best > mine_second) {
                    mine_second = other_best;
                }
                dp_best[threadIdx.x] = mine_best;
                dp_second[threadIdx.x] = mine_second;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            column_top_two[0] = dp_best[0];
            column_top_two[1] = dp_second[0];
        }
        __syncthreads();

        // Step 2: Check if column k ratio < tau. If near-tie, evaluate remaining columns.
        const float initial_ratio = separation_ratio_device(column_top_two[0], column_top_two[1]);
        if (initial_ratio < tau) {
            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            const int count = min(window + 1, selection_column_end - k);

            for (int slot = 1 + warp_id; slot < count; slot += (kThreads / 32)) {
                const int column = k + slot;
                unsigned long long w_best = 0ull;
                unsigned long long w_second = 0ull;
                for (int row = k + lane_id; row < n; row += 32) {
                    const unsigned short bits = absolute_bits(matrix[static_cast<std::size_t>(column) * n + row]);
                    if (!bits_finite(bits)) continue;
                    const unsigned long long key = pack_key(bits, row);
                    if (key > w_best) { w_second = w_best; w_best = key; }
                    else if (key > w_second) { w_second = key; }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const unsigned long long other_b = __shfl_down_sync(0xffffffff, w_best, offset, 32);
                    const unsigned long long other_s = __shfl_down_sync(0xffffffff, w_second, offset, 32);
                    if (other_b > w_best) {
                        w_second = w_best > other_s ? w_best : other_s;
                        w_best = other_b;
                    } else if (other_b > w_second) {
                        w_second = other_b;
                    }
                }
                if (lane_id == 0) {
                    column_top_two[2 * slot] = w_best;
                    column_top_two[2 * slot + 1] = w_second;
                }
            }
            __syncthreads();
        }
    } else if (method_code == 2) { // GP: transplanted column_top_two_kernel, column == k only.
        __shared__ unsigned long long gp_best[kThreads];
        __shared__ unsigned long long gp_second[kThreads];
        unsigned long long local_best = 0ull;
        unsigned long long local_second = 0ull;
        for (int row = k + threadIdx.x; row < n; row += kThreads) {
            const unsigned short bits = absolute_bits(matrix[static_cast<std::size_t>(k) * n + row]);
            if (!bits_finite(bits)) continue;
            const unsigned long long key = pack_key(bits, row);
            if (key > local_best) { local_second = local_best; local_best = key; }
            else if (key > local_second) { local_second = key; }
        }
        gp_best[threadIdx.x] = local_best;
        gp_second[threadIdx.x] = local_second;
        __syncthreads();
        for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                const unsigned long long other_best = gp_best[threadIdx.x + offset];
                const unsigned long long other_second = gp_second[threadIdx.x + offset];
                unsigned long long mine_best = gp_best[threadIdx.x];
                unsigned long long mine_second = gp_second[threadIdx.x];
                if (other_best > mine_best) {
                    mine_second = mine_best > other_second ? mine_best : other_second;
                    mine_best = other_best;
                } else if (other_best > mine_second) {
                    mine_second = other_best;
                }
                gp_best[threadIdx.x] = mine_best;
                gp_second[threadIdx.x] = mine_second;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            column_top_two[0] = gp_best[0];
            column_top_two[1] = gp_second[0];
        }
        __syncthreads();
    } else if (method_code == 6) { // ScPP: transplanted scaled_column_argmax_kernel.
        __shared__ double scpp_scores_shared[kThreads];
        __shared__ int scpp_rows_shared[kThreads];
        double local_best = -1.0;
        int local_row = k;
        for (int row = k + threadIdx.x; row < n; row += kThreads) {
            const double value = fabs(static_cast<double>(
                __half2float(matrix[static_cast<std::size_t>(k) * n + row])));
            const double scale = fmax(static_cast<double>(row_scales[row]), kDoubleTiny);
            const double score = value / scale;
            if (score > local_best) { local_best = score; local_row = row; }
        }
        scpp_scores_shared[threadIdx.x] = local_best;
        scpp_rows_shared[threadIdx.x] = local_row;
        __syncthreads();
        for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                const double other = scpp_scores_shared[threadIdx.x + offset];
                const int other_row = scpp_rows_shared[threadIdx.x + offset];
                if (other > scpp_scores_shared[threadIdx.x] ||
                    (other == scpp_scores_shared[threadIdx.x] &&
                     other_row < scpp_rows_shared[threadIdx.x])) {
                    scpp_scores_shared[threadIdx.x] = other;
                    scpp_rows_shared[threadIdx.x] = other_row;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            *scpp_score = scpp_scores_shared[0];
            *scpp_row = scpp_rows_shared[0];
        }
        __syncthreads();
    }

    if (method_code == 5) { // CP: reduce the per-column argmax array, not the matrix.
        unsigned long long local_key = 0ull;
        int local_row = k;
        for (int column = k + threadIdx.x; column < selection_column_end;
             column += blockDim.x) {
            const unsigned long long entry = column_argmax[column];
            if (key_empty(entry)) continue;
            // Order by (magnitude, then lowest column); the row already carries
            // the lowest-row tie break from the fused epilogue. This matches the
            // host column-outer/row-inner strict scan.
            const unsigned long long key = pack_key(key_magnitude_bits(entry), column);
            if (key > local_key) { local_key = key; local_row = key_index(entry); }
        }
        shared_key[threadIdx.x] = local_key;
        shared_row[threadIdx.x] = local_row;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                if (shared_key[threadIdx.x + offset] > shared_key[threadIdx.x]) {
                    shared_key[threadIdx.x] = shared_key[threadIdx.x + offset];
                    shared_row[threadIdx.x] = shared_row[threadIdx.x + offset];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (key_empty(shared_key[0])) {
                result_row = k; result_column = k; result_ok = 0; result_magnitude = -1.0f;
            } else {
                result_row = shared_row[0];
                result_column = key_index(shared_key[0]);
                result_ok = 1;
                result_magnitude = __half2float(__ushort_as_half(key_magnitude_bits(shared_key[0])));
            }
        }
        __syncthreads();
    } else if (threadIdx.x == 0) {
        int row = k;
        int column = k;
        int ok = 0;
        float magnitude = -1.0f;

        if (method_code == 0) { // PP
            const unsigned long long entry = column_argmax[k];
            if (key_empty(entry)) { row = k; ok = 0; }
            else { row = key_index(entry); magnitude = key_magnitude(entry); ok = 1; }
        } else if (method_code == 1) { // DP
            const unsigned long long initial_best = column_top_two[0];
            const unsigned long long initial_second = column_top_two[1];
            int selected_row = key_empty(initial_best) ? k : key_index(initial_best);
            float selected_magnitude = key_empty(initial_best) ? -1.0f : key_magnitude(initial_best);
            int selected_column = k;
            bool resolved = false;
            if (separation_ratio_device(initial_best, initial_second) < tau) {
                const int last = min(selection_column_end - 1, k + window);
                for (int candidate = k + 1; candidate <= last; ++candidate) {
                    const int slot = candidate - k;
                    const unsigned long long best = column_top_two[2 * slot];
                    const unsigned long long second = column_top_two[2 * slot + 1];
                    if (separation_ratio_device(best, second) >= tau) {
                        atomicAdd(&counters[kSlotDpAccepts], 1ull);
                        selected_row = key_empty(best) ? k : key_index(best);
                        selected_magnitude = key_empty(best) ? -1.0f : key_magnitude(best);
                        selected_column = candidate;
                        resolved = true;
                        break;
                    }
                }
                if (!resolved) atomicAdd(&counters[kSlotDpFallbacks], 1ull);
            }
            row = selected_row;
            column = selected_column;
            magnitude = selected_magnitude;
            ok = selected_magnitude >= 0.0f ? 1 : 0;
        } else if (method_code == 2) { // GP
            const unsigned long long best = column_top_two[0];
            const unsigned long long second = column_top_two[1];
            const float maximum = key_empty(best) ? -1.0f : key_magnitude(best);
            const float runner_up = key_empty(second) ? 0.0f : key_magnitude(second);
            int first_row = key_empty(best) ? k : key_index(best);
            int second_row = key_empty(second) ? first_row : key_index(second);
            int selected_row = first_row;
            if ((n - k) > 1 && separation_ratio_device(best, second) < tau) {
                atomicAdd(&counters[kSlotGpNearTies], 1ull);
                // Serial ascending FP32 accumulation, deliberately not
                // parallelised: the host reference sums in this exact order and
                // the GP branch decision depends on the rounding. In the blocked
                // panel-local schedule this tail is only (selection_column_end-k)
                // <= w elements long.
                float first_norm_squared = 0.0f;
                float second_norm_squared = 0.0f;
                for (int column_index = k; column_index < selection_column_end; ++column_index) {
                    const float value = __half2float(
                        matrix[static_cast<std::size_t>(column_index) * n + first_row]);
                    first_norm_squared += value * value;
                }
                for (int column_index = k; column_index < selection_column_end; ++column_index) {
                    const float value = __half2float(
                        matrix[static_cast<std::size_t>(column_index) * n + second_row]);
                    second_norm_squared += value * value;
                }
                const float first_norm = sqrtf(first_norm_squared);
                const float second_norm = sqrtf(second_norm_squared);
                const float first_score = first_norm == 0.0f ? 0.0f : maximum / first_norm;
                const float second_score = second_norm == 0.0f ? 0.0f : runner_up / second_norm;
                if (second_score > first_score) {
                    selected_row = second_row;
                    atomicAdd(&counters[kSlotGpSecondChoices], 1ull);
                }
            }
            row = selected_row;
            column = k;
            magnitude = maximum;
            ok = maximum >= 0.0f ? 1 : 0;
        } else if (method_code == 3) { // ScaP
            const int middle = k + (selection_column_end - 1 - k) / 2;
            const int last = selection_column_end - 1;
            int probes[3] = {k, middle, last};
            int probe_count = 1;
            for (int i = 1; i < 3; ++i) {
                if (probes[i] != probes[probe_count - 1]) probes[probe_count++] = probes[i];
            }
            float best = -1.0f;
            int best_row = k;
            int best_column = k;
            for (int i = 0; i < probe_count; ++i) {
                const unsigned long long entry = column_argmax[probes[i]];
                const float candidate = key_empty(entry) ? -1.0f : key_magnitude(entry);
                if (candidate > best) {
                    best = candidate;
                    best_row = key_empty(entry) ? k : key_index(entry);
                    best_column = probes[i];
                }
            }
            if (best_column == k) atomicAdd(&counters[kSlotScapCurrent], 1ull);
            else if (best_column == middle) atomicAdd(&counters[kSlotScapMiddle], 1ull);
            else atomicAdd(&counters[kSlotScapLast], 1ull);
            row = best_row;
            column = best_column;
            magnitude = best;
            ok = best >= 0.0f ? 1 : 0;
        } else if (method_code == 4) { // RP
            // OPTIMISATION: the rook walk is an inherently serial,
            // data-dependent pointer chase (each iteration's address depends
            // on the value just read), so it cannot be parallelised without
            // changing the algorithm -- this loop is left as-is. The one
            // safe, zero-risk fix is here: the original code issued one
            // atomicAdd to a global counter on EVERY iteration, i.e. inside
            // the latency-bound dependency chain itself, serialising an
            // atomic round trip in series with the pointer chase. The
            // counter is diagnostic only (never read by the algorithm), so
            // it is accumulated in a register and flushed with a single
            // atomicAdd after the walk resolves -- identical counter value,
            // identical pivot choice, fewer atomics on the critical path.
            int current_column = k;
            const int cap = 2 * (n - k) + 2;
            int iterations_taken = 0;
            for (int iteration = 0; iteration < cap; ++iteration) {
                ++iterations_taken;
                const unsigned long long column_entry = __ldg(&column_argmax[current_column]);
                const int candidate_row = key_empty(column_entry) ? k : key_index(column_entry);
                const unsigned long long row_entry = __ldg(&row_argmax[candidate_row]);
                const int next_column = key_empty(row_entry) ? k : key_index(row_entry);
                if (next_column == current_column) {
                    row = candidate_row;
                    column = current_column;
                    magnitude = key_empty(column_entry) ? -1.0f : key_magnitude(column_entry);
                    ok = 1;
                    break;
                }
                current_column = next_column;
            }
            atomicAdd(&counters[kSlotRpIterations], static_cast<unsigned long long>(iterations_taken));
            if (ok == 0) atomicAdd(&counters[kSlotRpFailures], 1ull);
        } else { // ScPP
            row = *scpp_row;
            column = k;
            const double score = *scpp_score;
            ok = score >= 0.0 ? 1 : 0;
            magnitude = fabsf(__half2float(matrix[static_cast<std::size_t>(k) * n + row]));
        }

        if (ok == 0) {
            *status = kStatusPivotSearchFailed;
        } else if (!isfinite(magnitude)) {
            // Only reachable through ScPP, which does not filter non-finite
            // candidates; the host reference reports the same status here.
            *status = kStatusNonfinitePivot;
        } else if (magnitude == 0.0f) {
            *status = kStatusSingularPivot;
        }
        result_row = row;
        result_column = column;
        result_ok = ok;
        result_magnitude = magnitude;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (method_code == 5) {
            if (result_ok == 0) *status = kStatusPivotSearchFailed;
            else if (result_magnitude == 0.0f) *status = kStatusSingularPivot;
        }
        choice[0] = result_row;
        choice[1] = result_column;
        pivot_log[2 * k] = result_row;
        pivot_log[2 * k + 1] = result_column;
    }
}

// Swaps and scaling that read their indices from device memory.
__global__ void swap_rows_device_kernel(
    __half* matrix,
    int n,
    int pivot,
    const int* choice,
    float* row_scales,
    unsigned long long* counters,
    const int* status) {
    if (*status != kStatusOk) return;
    const int other = choice[0];
    if (other == pivot) return;
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < n;
         column += blockDim.x * gridDim.x) {
        const std::size_t a = static_cast<std::size_t>(column) * n + pivot;
        const std::size_t b = static_cast<std::size_t>(column) * n + other;
        const __half value = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = value;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float scale = row_scales[pivot];
        row_scales[pivot] = row_scales[other];
        row_scales[other] = scale;
        atomicAdd(&counters[kSlotRowSwaps], 1ull);
    }
}

__global__ void swap_columns_device_kernel(
    __half* matrix,
    int n,
    int pivot,
    const int* choice,
    unsigned long long* counters,
    const int* status) {
    if (*status != kStatusOk) return;
    const int other = choice[1];
    if (other == pivot) return;
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n;
         row += blockDim.x * gridDim.x) {
        const std::size_t a = static_cast<std::size_t>(pivot) * n + row;
        const std::size_t b = static_cast<std::size_t>(other) * n + row;
        const __half value = matrix[a];
        matrix[a] = matrix[b];
        matrix[b] = value;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&counters[kSlotColumnSwaps], 1ull);
    }
}

__global__ void divide_column_device_kernel(
    __half* matrix,
    int n,
    int pivot,
    unsigned int* multiplier_max_bits,
    unsigned long long* nonfinite,
    const int* status) {
    if (*status != kStatusOk) return;
    const int row = pivot + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const __half quotient = __hdiv(
        matrix[static_cast<std::size_t>(pivot) * n + row],
        matrix[static_cast<std::size_t>(pivot) * n + pivot]);
    matrix[static_cast<std::size_t>(pivot) * n + row] = quotient;
    const float magnitude = fabsf(__half2float(quotient));
    if (isfinite(magnitude)) atomicMax(multiplier_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
}

// Guarded variants of the two update kernels: identical arithmetic, but they
// stand down once the run has failed so the host never has to poll.
__global__ void rank1_update_device_kernel(
    __half* matrix, int n, int pivot, int row_start, int row_end,
    int column_start, int column_end, unsigned int* growth_max_bits,
    unsigned long long* nonfinite, unsigned long long* column_argmax,
    unsigned long long* row_argmax, int row_argmax_column_end, const int* status) {
    if (*status != kStatusOk) return;
    const int column = column_start + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_start + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= row_end || column >= column_end) return;
    const std::size_t entry = static_cast<std::size_t>(column) * n + row;
    const __half multiplier = matrix[static_cast<std::size_t>(pivot) * n + row];
    const __half pivot_value = matrix[static_cast<std::size_t>(column) * n + pivot];
    const __half updated = __hsub(matrix[entry], __hmul(multiplier, pivot_value));
    matrix[entry] = updated;
    const float magnitude = fabsf(__half2float(updated));
    if (isfinite(magnitude)) atomicMax(growth_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
    const unsigned short bits = absolute_bits(updated);
    if (bits_finite(bits)) {
        if (column_argmax != nullptr) atomicMax(&column_argmax[column], pack_key(bits, row));
    }
    // RP row_argmax optimisation: reduce across the 16 columns of the same row
    // inside the warp, so only threadIdx.x == 0 fires the global atomicMax.
    // This reduces global atomicMax contention on row_argmax by 16x.
    if (row_argmax != nullptr) {
        unsigned long long r_key = (column < row_argmax_column_end && bits_finite(bits))
            ? pack_key(bits, column)
            : 0ull;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(0xffffffff, r_key, offset, 16);
            if (other > r_key) r_key = other;
        }
        if (threadIdx.x == 0 && r_key != 0ull) {
            atomicMax(&row_argmax[row], r_key);
        }
    }
}

__global__ void blocked_update_device_kernel(
    __half* matrix, int n, int panel_start, int panel_end,
    unsigned int* growth_max_bits, unsigned long long* nonfinite,
    unsigned long long* column_argmax, unsigned long long* row_argmax,
    int row_argmax_column_end, const int* status) {
    if (*status != kStatusOk) return;
    const int column = panel_end + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = panel_end + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= n) return;
    const std::size_t entry = static_cast<std::size_t>(column) * n + row;
    __half updated = matrix[entry];
    for (int pivot = panel_start; pivot < panel_end; ++pivot) {
        const __half multiplier = matrix[static_cast<std::size_t>(pivot) * n + row];
        const __half pivot_value = matrix[static_cast<std::size_t>(column) * n + pivot];
        updated = __hsub(updated, __hmul(multiplier, pivot_value));
    }
    matrix[entry] = updated;
    const float magnitude = fabsf(__half2float(updated));
    if (isfinite(magnitude)) atomicMax(growth_max_bits, __float_as_uint(magnitude));
    else atomicAdd(nonfinite, 1ULL);
    const unsigned short bits = absolute_bits(updated);
    if (bits_finite(bits)) {
        if (column_argmax != nullptr) atomicMax(&column_argmax[column], pack_key(bits, row));
    }
    // RP row_argmax optimisation: in-warp 16-way reduction before global atomicMax
    if (row_argmax != nullptr) {
        unsigned long long r_key = (column < row_argmax_column_end && bits_finite(bits))
            ? pack_key(bits, column)
            : 0ull;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(0xffffffff, r_key, offset, 16);
            if (other > r_key) r_key = other;
        }
        if (threadIdx.x == 0 && r_key != 0ull) {
            atomicMax(&row_argmax[row], r_key);
        }
    }
}

int method_code_of(Method method) {
    switch (method) {
        case Method::kPP: return 0;
        case Method::kDP: return 1;
        case Method::kGP: return 2;
        case Method::kScaP: return 3;
        case Method::kRP: return 4;
        case Method::kCP: return 5;
        case Method::kScPP: return 6;
    }
    return 0;
}

std::string method_name(Method method) {
    switch (method) {
        case Method::kPP: return "PP";
        case Method::kDP: return "DP";
        case Method::kGP: return "GP";
        case Method::kScaP: return "ScaP";
        case Method::kRP: return "RP";
        case Method::kCP: return "CP";
        case Method::kScPP: return "ScPP";
    }
    return "unknown";
}

std::string schedule_name(Schedule schedule) {
    return schedule == Schedule::kBlockedPanelLocal ? "blocked_panel_local" : "unblocked_rank1";
}

// NEW
std::string search_name(SearchMode search) {
    return search == SearchMode::kFusedDevice ? "fused_device" : "host";
}

SearchMode parse_search(const std::string& value) {
    if (value == "host") return SearchMode::kHost;
    if (value == "fused_device") return SearchMode::kFusedDevice;
    throw std::runtime_error("unknown search mode: " + value);
}

std::string family_name(Family family) {
    switch (family) {
        case Family::kUniform: return "uniform";
        case Family::kNormal: return "normal";
        case Family::kGraded: return "graded";
        case Family::kWilkinson: return "wilkinson";
        case Family::kDpEquality: return "dp_equality";
        case Family::kScaPLast: return "scap_last";
        case Family::kGpSecond: return "gp_second";
    }
    return "unknown";
}

Method parse_method(const std::string& value) {
    if (value == "pp") return Method::kPP;
    if (value == "dp") return Method::kDP;
    if (value == "gp") return Method::kGP;
    if (value == "scap") return Method::kScaP;
    if (value == "rp") return Method::kRP;
    if (value == "cp") return Method::kCP;
    if (value == "scpp") return Method::kScPP;
    throw std::runtime_error("unknown method: " + value);
}

Schedule parse_schedule(const std::string& value) {
    if (value == "unblocked_rank1") return Schedule::kUnblockedRank1;
    if (value == "blocked_panel_local") return Schedule::kBlockedPanelLocal;
    throw std::runtime_error("unknown schedule: " + value);
}

int publication_panel_width(int n) {
    if (n <= 4096) return 64;
    if (n <= 10240) return 128;
    if (n <= 20480) return 512;
    return 1024;
}

Family parse_family(const std::string& value) {
    if (value == "uniform") return Family::kUniform;
    if (value == "normal") return Family::kNormal;
    if (value == "graded") return Family::kGraded;
    if (value == "wilkinson") return Family::kWilkinson;
    if (value == "dp_equality") return Family::kDpEquality;
    if (value == "scap_last") return Family::kScaPLast;
    if (value == "gp_second") return Family::kGpSecond;
    throw std::runtime_error("unknown family: " + value);
}

std::string option_value(int argc, char** argv, int* index, const char* option) {
    if (*index + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + option);
    return argv[++*index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--method") options.method = parse_method(option_value(argc, argv, &index, "--method"));
        else if (argument == "--schedule") options.schedule = parse_schedule(option_value(argc, argv, &index, "--schedule"));
        else if (argument == "--family") options.family = parse_family(option_value(argc, argv, &index, "--family"));
        else if (argument == "--n") options.n = std::stoi(option_value(argc, argv, &index, "--n"));
        else if (argument == "--panel-width") options.panel_width = std::stoi(option_value(argc, argv, &index, "--panel-width"));
        else if (argument == "--seed") options.seed = std::stoull(option_value(argc, argv, &index, "--seed"));
        else if (argument == "--tau") options.tau = std::stof(option_value(argc, argv, &index, "--tau"));
        else if (argument == "--window") options.window = std::stoi(option_value(argc, argv, &index, "--window"));
        else if (argument == "--warmups") options.warmups = std::stoi(option_value(argc, argv, &index, "--warmups"));
        else if (argument == "--repetitions") options.repetitions = std::stoi(option_value(argc, argv, &index, "--repetitions"));
        else if (argument == "--output") options.output = option_value(argc, argv, &index, "--output");
        else if (argument == "--search") options.search = parse_search(option_value(argc, argv, &index, "--search"));
        else if (argument == "--log-pivots") options.log_pivots = true;
        else if (argument == "--abort-check-stride") options.abort_check_stride = std::stoi(option_value(argc, argv, &index, "--abort-check-stride"));
        else if (argument == "--pivot-log") options.pivot_log_path = option_value(argc, argv, &index, "--pivot-log");
        else if (argument == "--help") {
            std::printf(
                "Usage: %s --method pp|dp|gp|scap|rp|cp|scpp "
                "--schedule unblocked_rank1|blocked_panel_local "
                "--family uniform|normal|graded|wilkinson|dp_equality|scap_last|gp_second "
                "--n N --seed S "
                "[--search host|fused_device] "
                "[--panel-width 0=automatic] [--tau 1.01] [--window 6] "
                "[--warmups 1] [--repetitions 3] "
                "[--output runs.csv] [--log-pivots] [--pivot-log FILE] "
                "[--abort-check-stride 0]\n",
                argv[0]);
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.n < 2) throw std::runtime_error("--n must be at least two");
    if (options.panel_width < 0) throw std::runtime_error("--panel-width must be non-negative");
    if (!(options.tau > 1.0f)) throw std::runtime_error("--tau must exceed one");
    if (options.window < 0) throw std::runtime_error("--window must be non-negative");
    if (options.warmups < 0 || options.repetitions < 1) throw std::runtime_error("invalid run counts");
    if (options.panel_width == 0) options.panel_width = publication_panel_width(options.n);
    options.panel_width = std::min(options.panel_width, options.n);
    return options;
}

std::vector<__half> generate_matrix(const Options& options) {
    std::vector<__half> result(static_cast<std::size_t>(options.n) * options.n);
    if (options.family == Family::kDpEquality || options.family == Family::kScaPLast ||
        options.family == Family::kGpSecond) {
        if (options.n < 4) throw std::runtime_error("diagnostic families require --n >= 4");
        for (int i = 0; i < options.n; ++i) result[index_of(i, i, options.n)] = __float2half_rn(1.0f);
        if (options.family == Family::kDpEquality) {
            result[index_of(1, 0, options.n)] = __float2half_rn(1.0f);
            result[index_of(0, 1, options.n)] = __float2half_rn(2.0f);
        } else if (options.family == Family::kScaPLast) {
            result[index_of(1, 0, options.n)] = __float2half_rn(0.5f);
            const int last = options.schedule == Schedule::kBlockedPanelLocal
                ? std::min(options.n, options.panel_width) - 1
                : options.n - 1;
            result[index_of(2, last, options.n)] = __float2half_rn(4.0f);
        } else {
            result[index_of(1, 0, options.n)] = __float2half_rn(0.99f);
            result[index_of(0, 1, options.n)] = __float2half_rn(100.0f);
        }
        return result;
    }
    std::mt19937_64 generator(options.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int column = 0; column < options.n; ++column) {
        for (int row = 0; row < options.n; ++row) {
            float value = 0.0f;
            if (options.family == Family::kUniform) value = uniform(generator);
            else if (options.family == Family::kNormal) value = normal(generator);
            else if (options.family == Family::kGraded) {
                const float base = 2.0f * uniform(generator) - 1.0f;
                const int exponent = -10 + (20 * column) / std::max(1, options.n - 1);
                value = std::ldexp(base, exponent);
            } else {
                if (column == options.n - 1 || row == column) value = 1.0f;
                else if (row > column) value = -1.0f;
            }
            result[index_of(row, column, options.n)] = __float2half_rn(value);
        }
    }
    return result;
}

std::vector<__half> copy_column_segment(const __half* matrix, int n, int column, int row_start) {
    std::vector<__half> result(n - row_start);
    CUDA_CHECK(cudaMemcpy(
        result.data(), matrix + index_of(row_start, column, n), result.size() * sizeof(__half),
        cudaMemcpyDeviceToHost));
    return result;
}

std::vector<__half> copy_row_segment(
    const __half* matrix,
    int n,
    int row,
    int column_start,
    int column_end) {
    std::vector<__half> result(column_end - column_start);
    CUDA_CHECK(cudaMemcpy2D(
        result.data(), sizeof(__half), matrix + index_of(row, column_start, n),
        static_cast<std::size_t>(n) * sizeof(__half), sizeof(__half), result.size(),
        cudaMemcpyDeviceToHost));
    return result;
}

std::pair<int, float> maximum_index(const std::vector<__half>& values) {
    int index = 0;
    float maximum = -1.0f;
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        const float value = fabsf(__half2float(values[i]));
        if (std::isfinite(value) && value > maximum) {
            maximum = value;
            index = i;
        }
    }
    return {index, maximum};
}

struct TopTwo { int first = 0; int second = 0; float maximum = 0.0f; float runner_up = 0.0f; };

TopTwo top_two(const std::vector<__half>& values) {
    TopTwo result;
    result.maximum = -1.0f;
    result.runner_up = -1.0f;
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        const float value = fabsf(__half2float(values[i]));
        if (!std::isfinite(value)) continue;
        if (value > result.maximum) {
            result.runner_up = result.maximum;
            result.second = result.first;
            result.maximum = value;
            result.first = i;
        } else if (value > result.runner_up) {
            result.runner_up = value;
            result.second = i;
        }
    }
    if (result.runner_up < 0.0f) {
        result.runner_up = 0.0f;
        result.second = result.first;
    }
    return result;
}

float separation_ratio(const TopTwo& values) {
    if (values.runner_up == 0.0f) return values.maximum > 0.0f
        ? std::numeric_limits<float>::infinity() : 0.0f;
    return values.maximum / values.runner_up;
}

PivotChoice select_pivot(
    Method method,
    const __half* matrix,
    int n,
    int k,
    int selection_column_end,
    float tau,
    int window,
    const std::vector<float>& row_scales,
    Counters* counters) {
    if (method == Method::kPP) {
        const auto values = copy_column_segment(matrix, n, k, k);
        const auto maximum = maximum_index(values);
        return {k + maximum.first, k, maximum.second >= 0.0f};
    }
    if (method == Method::kDP) {
        const auto current = copy_column_segment(matrix, n, k, k);
        const TopTwo initial = top_two(current);
        if (separation_ratio(initial) < tau) {
            for (int column = k + 1;
                 column <= std::min(selection_column_end - 1, k + window);
                 ++column) {
                const auto candidate = copy_column_segment(matrix, n, column, k);
                const TopTwo separated = top_two(candidate);
                if (separation_ratio(separated) >= tau) {
                    ++counters->dp_lookahead_accepts;
                    return {k + separated.first, column, separated.maximum >= 0.0f};
                }
            }
            ++counters->dp_fallbacks;
        }
        return {k + initial.first, k, initial.maximum >= 0.0f};
    }
    if (method == Method::kGP) {
        const auto values = copy_column_segment(matrix, n, k, k);
        const TopTwo candidates = top_two(values);
        int selected = candidates.first;
        if (values.size() > 1 && separation_ratio(candidates) < tau) {
            ++counters->gp_near_ties;
            const int first_row = k + candidates.first;
            const int second_row = k + candidates.second;
            const auto first_tail = copy_row_segment(matrix, n, first_row, k, selection_column_end);
            const auto second_tail = copy_row_segment(matrix, n, second_row, k, selection_column_end);
            float first_norm_sq = 0.0f;
            float second_norm_sq = 0.0f;
            for (const __half value : first_tail) {
                const float x = __half2float(value);
                first_norm_sq += x * x;
            }
            for (const __half value : second_tail) {
                const float x = __half2float(value);
                second_norm_sq += x * x;
            }
            const float first_norm = std::sqrt(first_norm_sq);
            const float second_norm = std::sqrt(second_norm_sq);
            const float first_score = first_norm == 0.0f ? 0.0f : candidates.maximum / first_norm;
            const float second_score = second_norm == 0.0f ? 0.0f : candidates.runner_up / second_norm;
            if (second_score > first_score) {
                selected = candidates.second;
                ++counters->gp_second_choices;
            }
        }
        return {k + selected, k, candidates.maximum >= 0.0f};
    }
    if (method == Method::kScaP) {
        const int middle = k + (selection_column_end - 1 - k) / 2;
        const int last = selection_column_end - 1;
        std::vector<int> columns = {k, middle, last};
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        float best = -1.0f;
        int best_row = k;
        int best_column = k;
        for (const int column : columns) {
            const auto values = copy_column_segment(matrix, n, column, k);
            const auto maximum = maximum_index(values);
            if (maximum.second > best) {
                best = maximum.second;
                best_row = k + maximum.first;
                best_column = column;
            }
        }
        if (best_column == k) ++counters->scap_current;
        else if (best_column == middle) ++counters->scap_middle;
        else ++counters->scap_last;
        return {best_row, best_column, best >= 0.0f};
    }
    if (method == Method::kRP) {
        int column = k;
        const int cap = 2 * (n - k) + 2;
        for (int iteration = 0; iteration < cap; ++iteration) {
            ++counters->rp_iterations;
            const auto column_values = copy_column_segment(matrix, n, column, k);
            const int row = k + maximum_index(column_values).first;
            const auto row_values = copy_row_segment(matrix, n, row, k, selection_column_end);
            const int next_column = k + maximum_index(row_values).first;
            if (next_column == column) return {row, column, true};
            column = next_column;
        }
        ++counters->rp_failures;
        return {-1, -1, false};
    }
    if (method == Method::kCP) {
        std::vector<__half> host(static_cast<std::size_t>(n) * n);
        CUDA_CHECK(cudaMemcpy(host.data(), matrix, host.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        float best = -1.0f;
        int best_row = k;
        int best_column = k;
        for (int column = k; column < selection_column_end; ++column) {
            for (int row = k; row < n; ++row) {
                const float value = fabsf(__half2float(host[index_of(row, column, n)]));
                if (std::isfinite(value) && value > best) {
                    best = value;
                    best_row = row;
                    best_column = column;
                }
            }
        }
        return {best_row, best_column, best >= 0.0f};
    }
    const auto values = copy_column_segment(matrix, n, k, k);
    double best = -1.0;
    int best_row = k;
    for (int offset = 0; offset < static_cast<int>(values.size()); ++offset) {
        const int row = k + offset;
        const double score = fabs(static_cast<double>(__half2float(values[offset]))) /
            std::max(static_cast<double>(row_scales[row]), std::numeric_limits<double>::min());
        if (score > best) {
            best = score;
            best_row = row;
        }
    }
    return {best_row, k, best >= 0.0};
}

bool is_permutation(const std::vector<int>& values) {
    std::vector<int> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < static_cast<int>(sorted.size()); ++i) if (sorted[i] != i) return false;
    return true;
}

double reconstruct(
    const std::vector<__half>& initial,
    const std::vector<__half>& factor,
    const std::vector<int>& rows,
    const std::vector<int>& columns,
    int n,
    std::string* kind) {
    if (!is_permutation(rows) || !is_permutation(columns)) {
        *kind = "invalid_permutation";
        return std::numeric_limits<double>::quiet_NaN();
    }
    for (const __half value : factor) {
        if (!std::isfinite(__half2float(value))) {
            *kind = "nonfinite_factor";
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    const int sample_dimension = n <= 256 ? n : 32;
    *kind = n <= 256 ? "full_fp64" : "sampled_fp64_32x32";
    double maximum_input = 0.0;
    double maximum_residual = 0.0;
    for (int row_index = 0; row_index < sample_dimension; ++row_index) {
        const int row = n <= 256 ? row_index : (row_index * (n - 1)) / (sample_dimension - 1);
        for (int column_index = 0; column_index < sample_dimension; ++column_index) {
            const int column = n <= 256 ? column_index : (column_index * (n - 1)) / (sample_dimension - 1);
            double product = 0.0;
            for (int inner = 0; inner <= std::min(row, column); ++inner) {
                const double lower = row == inner ? 1.0 :
                    static_cast<double>(__half2float(factor[index_of(row, inner, n)]));
                const double upper = static_cast<double>(__half2float(factor[index_of(inner, column, n)]));
                product += lower * upper;
            }
            const double reference = static_cast<double>(
                __half2float(initial[index_of(rows[row], columns[column], n)]));
            if (!std::isfinite(product) || !std::isfinite(reference)) {
                *kind = "nonfinite_reconstruction";
                return std::numeric_limits<double>::quiet_NaN();
            }
            maximum_input = std::max(maximum_input, std::abs(reference));
            maximum_residual = std::max(maximum_residual, std::abs(product - reference));
        }
    }
    return maximum_residual / std::max(maximum_input, std::numeric_limits<double>::min());
}

RunResult run_once(const Options& options, const std::vector<__half>& initial) {
    RunResult result;
    const std::size_t elements = initial.size();
    DeviceBuffer<__half> matrix(elements);
    DeviceBuffer<unsigned int> growth_max_bits(1);
    DeviceBuffer<unsigned int> multiplier_max_bits(1);
    DeviceBuffer<unsigned long long> nonfinite(1);
    CUDA_CHECK(cudaMemcpy(matrix.get(), initial.data(), elements * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(multiplier_max_bits.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(nonfinite.get(), 0, sizeof(unsigned long long)));

    float initial_maximum = 0.0f;
    for (const __half value : initial) {
        const float magnitude = fabsf(__half2float(value));
        if (!std::isfinite(magnitude)) {
            result.status = "nonfinite_input";
            return result;
        }
        initial_maximum = std::max(initial_maximum, magnitude);
    }
    unsigned int initial_bits = 0;
    std::memcpy(&initial_bits, &initial_maximum, sizeof(float));
    CUDA_CHECK(cudaMemcpy(growth_max_bits.get(), &initial_bits, sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<int> rows(options.n);
    std::vector<int> columns(options.n);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(columns.begin(), columns.end(), 0);
    std::vector<float> row_scales(options.n, 1.0f);
    if (options.method == Method::kScPP) {
        for (int row = 0; row < options.n; ++row) {
            float maximum = 0.0f;
            for (int column = 0; column < options.n; ++column) {
                maximum = std::max(
                    maximum, fabsf(__half2float(initial[index_of(row, column, options.n)])));
            }
            row_scales[row] = maximum == 0.0f ? 1.0f : maximum;
        }
    }

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start_event));
    CUDA_CHECK(cudaEventSynchronize(start_event));
    const auto wall_start = std::chrono::steady_clock::now();

    result.status = "completed";
    auto check_nonfinite = [&]() {
        CUDA_CHECK(cudaMemcpy(
            &result.nonfinite_count, nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (result.nonfinite_count != 0) {
            result.status = "nonfinite_factorization";
            return false;
        }
        return true;
    };

    auto pivot_step = [&](int pivot, int selection_column_end, int update_column_end) {
        const PivotChoice choice = select_pivot(
            options.method, matrix.get(), options.n, pivot, selection_column_end,
            options.tau, options.window, row_scales, &result.counters);
        // NEW: record the decision so the host and fused paths can be compared.
        result.pivot_rows.push_back(choice.row);
        result.pivot_columns.push_back(choice.column);
        if (!choice.success) {
            result.status = "pivot_search_failed";
            return false;
        }
        if (choice.row != pivot) {
            swap_rows_kernel<<<(options.n + kThreads - 1) / kThreads, kThreads>>>(
                matrix.get(), options.n, pivot, choice.row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(rows[pivot], rows[choice.row]);
            std::swap(row_scales[pivot], row_scales[choice.row]);
            ++result.counters.row_swaps;
        }
        if (choice.column != pivot) {
            swap_columns_kernel<<<(options.n + kThreads - 1) / kThreads, kThreads>>>(
                matrix.get(), options.n, pivot, choice.column);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(columns[pivot], columns[choice.column]);
            ++result.counters.column_swaps;
        }

        __half pivot_value{};
        CUDA_CHECK(cudaMemcpy(
            &pivot_value, matrix.get() + index_of(pivot, pivot, options.n), sizeof(__half),
            cudaMemcpyDeviceToHost));
        const float pivot_float = __half2float(pivot_value);
        if (!std::isfinite(pivot_float)) {
            result.status = "nonfinite_pivot";
            return false;
        }
        if (pivot_float == 0.0f) {
            result.status = "singular_pivot";
            return false;
        }

        CUDA_CHECK(cudaMemset(multiplier_max_bits.get(), 0, sizeof(unsigned int)));
        const int active = options.n - pivot - 1;
        divide_column_kernel<<<(active + kThreads - 1) / kThreads, kThreads>>>(
            matrix.get(), options.n, pivot, multiplier_max_bits.get(), nonfinite.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned int multiplier_bits = 0;
        CUDA_CHECK(cudaMemcpy(
            &multiplier_bits, multiplier_max_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
        float multiplier_max = 0.0f;
        std::memcpy(&multiplier_max, &multiplier_bits, sizeof(float));
        result.max_multiplier = std::max(result.max_multiplier, multiplier_max);
        if (!check_nonfinite()) return false;

        const int update_columns = update_column_end - pivot - 1;
        if (active > 0 && update_columns > 0) {
            dim3 threads(16, 16);
            dim3 blocks(
                (update_columns + threads.x - 1) / threads.x,
                (active + threads.y - 1) / threads.y);
            rank1_update_kernel<<<blocks, threads>>>(
                matrix.get(), options.n, pivot, pivot + 1, options.n,
                pivot + 1, update_column_end, growth_max_bits.get(), nonfinite.get(),
                nullptr, nullptr, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            if (!check_nonfinite()) return false;
        }
        return true;
    };

    if (options.schedule == Schedule::kUnblockedRank1) {
        for (int pivot = 0; pivot < options.n - 1 && result.status == "completed"; ++pivot) {
            pivot_step(pivot, options.n, options.n);
        }
    } else {
        for (int panel_start = 0;
             panel_start < options.n && result.status == "completed";
             panel_start += options.panel_width) {
            const int panel_end = std::min(options.n, panel_start + options.panel_width);
            for (int pivot = panel_start;
                 pivot < panel_end && pivot < options.n - 1 && result.status == "completed";
                 ++pivot) {
                pivot_step(pivot, panel_end, panel_end);
            }
            if (result.status != "completed" || panel_end >= options.n) continue;

            // Form U12 using the same rankwise update order as the blocked
            // predictive validator, but without scaling guards.
            for (int pivot = panel_start; pivot < panel_end; ++pivot) {
                const int rows_in_panel = panel_end - pivot - 1;
                const int future_columns = options.n - panel_end;
                if (rows_in_panel <= 0 || future_columns <= 0) continue;
                dim3 threads(16, 16);
                dim3 blocks(
                    (future_columns + threads.x - 1) / threads.x,
                    (rows_in_panel + threads.y - 1) / threads.y);
                rank1_update_kernel<<<blocks, threads>>>(
                    matrix.get(), options.n, pivot, pivot + 1, panel_end,
                    panel_end, options.n, growth_max_bits.get(), nonfinite.get(),
                    nullptr, nullptr, 0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                if (!check_nonfinite()) break;
            }
            if (result.status != "completed") continue;

            const int trailing = options.n - panel_end;
            dim3 threads(16, 16);
            dim3 blocks(
                (trailing + threads.x - 1) / threads.x,
                (trailing + threads.y - 1) / threads.y);
            blocked_update_kernel<<<blocks, threads>>>(
                matrix.get(), options.n, panel_start, panel_end,
                growth_max_bits.get(), nonfinite.get(),
                nullptr, nullptr, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            check_nonfinite();
        }
    }

    if (result.status == "completed") {
        __half final_pivot{};
        CUDA_CHECK(cudaMemcpy(
            &final_pivot, matrix.get() + index_of(options.n - 1, options.n - 1, options.n),
            sizeof(__half), cudaMemcpyDeviceToHost));
        const float value = __half2float(final_pivot);
        if (!std::isfinite(value)) result.status = "nonfinite_final_pivot";
        else if (value == 0.0f) result.status = "singular_final_pivot";
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    const auto wall_stop = std::chrono::steady_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    CUDA_CHECK(cudaEventElapsedTime(&result.cuda_ms, start_event, stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    unsigned int growth_bits = 0;
    CUDA_CHECK(cudaMemcpy(&growth_bits, growth_max_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    float growth_maximum = 0.0f;
    std::memcpy(&growth_maximum, &growth_bits, sizeof(float));
    result.growth_factor = initial_maximum == 0.0f
        ? std::numeric_limits<double>::quiet_NaN()
        : static_cast<double>(growth_maximum) / initial_maximum;

    std::vector<__half> factor(elements);
    CUDA_CHECK(cudaMemcpy(factor.data(), matrix.get(), elements * sizeof(__half), cudaMemcpyDeviceToHost));
    result.reconstruction_residual = reconstruct(
        initial, factor, rows, columns, options.n, &result.reconstruction_kind);
    if (result.status == "completed" && !std::isfinite(result.reconstruction_residual)) {
        result.status = "reconstruction_failed";
    }
    return result;
}

// ===========================================================================
// NEW: SearchMode::kFusedDevice driver.
//
// Structurally identical to run_once() above -- same matrix, same panel
// schedule, same FP16 arithmetic, same reconstruction diagnostic -- but the
// elimination loop contains no cudaMemcpy and no cudaDeviceSynchronize. The
// pivot decision is taken by select_pivot_device_kernel from the per-column and
// per-row argmax that the previous trailing update already deposited, and the
// permutations are replayed on the host after the timed region from a single
// copy of the device pivot log.
// ===========================================================================
RunResult run_once_fused(const Options& options, const std::vector<__half>& initial) {
    RunResult result;
    const int n = options.n;
    const std::size_t elements = initial.size();

    DeviceBuffer<__half> matrix(elements);
    DeviceBuffer<unsigned int> growth_max_bits(1);
    DeviceBuffer<unsigned int> multiplier_max_bits(1);
    DeviceBuffer<unsigned long long> nonfinite(1);
    DeviceBuffer<unsigned long long> column_argmax(n);
    DeviceBuffer<unsigned long long> row_argmax(n);
    DeviceBuffer<unsigned long long> column_top_two(2 * (options.window + 2));
    DeviceBuffer<double> scpp_score(1);
    DeviceBuffer<int> scpp_row(1);
    DeviceBuffer<int> choice(2);
    DeviceBuffer<int> pivot_log(2 * n);
    DeviceBuffer<unsigned long long> counters(kCounterSlots);
    DeviceBuffer<int> status(1);
    DeviceBuffer<float> row_scales_device(n);

    CUDA_CHECK(cudaMemcpy(matrix.get(), initial.data(), elements * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(multiplier_max_bits.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(nonfinite.get(), 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(counters.get(), 0, kCounterSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(status.get(), 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(pivot_log.get(), 0, 2 * n * sizeof(int)));

    float initial_maximum = 0.0f;
    for (const __half value : initial) {
        const float magnitude = fabsf(__half2float(value));
        if (!std::isfinite(magnitude)) {
            result.status = "nonfinite_input";
            return result;
        }
        initial_maximum = std::max(initial_maximum, magnitude);
    }
    unsigned int initial_bits = 0;
    std::memcpy(&initial_bits, &initial_maximum, sizeof(float));
    CUDA_CHECK(cudaMemcpy(growth_max_bits.get(), &initial_bits, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // ScPP row scales are computed on GPU, eliminating O(n^2) CPU overhead.
    if (options.method == Method::kScPP) {
        const int blocks = (n + kThreads - 1) / kThreads;
        compute_row_scales_kernel<<<blocks, kThreads>>>(
            matrix.get(), n, row_scales_device.get());
        CUDA_CHECK(cudaGetLastError());
    }

    const int code = method_code_of(options.method);
    const bool needs_row_argmax = options.method == Method::kRP;

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start_event));
    CUDA_CHECK(cudaEventSynchronize(start_event));
    const auto wall_start = std::chrono::steady_clock::now();

    // Resets the fused state, then repopulates it for the search domain that the
    // NEXT elimination step will use.
    auto prime = [&](int row_start, int column_start, int column_end, int row_argmax_column_end) {
        CUDA_CHECK(cudaMemsetAsync(column_argmax.get(), 0, n * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(row_argmax.get(), 0, n * sizeof(unsigned long long)));
        const int columns = column_end - column_start;
        const int rows = n - row_start;
        if (columns <= 0 || rows <= 0) return;
        dim3 threads(16, 16);
        dim3 blocks(
            (columns + threads.x - 1) / threads.x,
            (rows + threads.y - 1) / threads.y);
        prime_argmax_kernel<<<blocks, threads>>>(
            matrix.get(), n, row_start, column_start, column_end,
            column_argmax.get(), needs_row_argmax ? row_argmax.get() : nullptr,
            row_argmax_column_end);
        CUDA_CHECK(cudaGetLastError());
    };

    // Optional early-abort poll, mirroring the frozen host path's check points.
    bool aborted = false;
    auto poll_now = [&]() {
        if (options.abort_check_stride <= 0 || aborted) return true;
        unsigned long long count = 0;
        CUDA_CHECK(cudaMemcpy(&count, nonfinite.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (count != 0) {
            result.nonfinite_count = count;
            aborted = true;
            return false;
        }
        return true;
    };
    auto poll_nonfinite = [&](int pivot) {
        if (options.abort_check_stride <= 0 || aborted) return true;
        if (pivot % options.abort_check_stride != 0) return true;
        unsigned long long count = 0;
        CUDA_CHECK(cudaMemcpy(&count, nonfinite.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (count != 0) {
            result.nonfinite_count = count;
            aborted = true;
            return false;
        }
        return true;
    };

    // Number of elimination steps actually launched, so that the pivot log
    // replayed below has exactly the same length as the host path's, which
    // records one entry per invoked step including the failing one.
    int executed_steps = 0;
    auto pivot_step = [&](int pivot, int selection_column_end, int update_column_end,
                          int next_row_argmax_column_end) {
        if (aborted) return;
        ++executed_steps;
        // OPTIMISATION V2: DP, GP, and ScPP all compute their preambles inside
        // select_pivot_device_kernel directly, so NO auxiliary kernel launches
        // exist for ANY method in this loop.

        select_pivot_device_kernel<<<1, kThreads>>>(
            matrix.get(), n, pivot, selection_column_end, code, options.tau, options.window,
            column_argmax.get(), row_argmax.get(), column_top_two.get(),
            row_scales_device.get(), scpp_score.get(), scpp_row.get(), choice.get(),
            pivot_log.get(), counters.get(), status.get());
        CUDA_CHECK(cudaGetLastError());

        const int swap_blocks = (n + kThreads - 1) / kThreads;
        swap_rows_device_kernel<<<swap_blocks, kThreads>>>(
            matrix.get(), n, pivot, choice.get(), row_scales_device.get(),
            counters.get(), status.get());
        CUDA_CHECK(cudaGetLastError());
        swap_columns_device_kernel<<<swap_blocks, kThreads>>>(
            matrix.get(), n, pivot, choice.get(), counters.get(), status.get());
        CUDA_CHECK(cudaGetLastError());

        const int active = n - pivot - 1;
        if (active > 0) {
            divide_column_device_kernel<<<(active + kThreads - 1) / kThreads, kThreads>>>(
                matrix.get(), n, pivot, multiplier_max_bits.get(), nonfinite.get(), status.get());
            CUDA_CHECK(cudaGetLastError());
        }
        if (!poll_nonfinite(pivot)) return;

        const int update_columns = update_column_end - pivot - 1;
        if (active > 0 && update_columns > 0) {
            CUDA_CHECK(cudaMemsetAsync(column_argmax.get(), 0, n * sizeof(unsigned long long)));
            if (needs_row_argmax) {
                CUDA_CHECK(cudaMemsetAsync(row_argmax.get(), 0, n * sizeof(unsigned long long)));
            }
            dim3 threads(16, 16);
            dim3 blocks(
                (update_columns + threads.x - 1) / threads.x,
                (active + threads.y - 1) / threads.y);
            rank1_update_device_kernel<<<blocks, threads>>>(
                matrix.get(), n, pivot, pivot + 1, n, pivot + 1, update_column_end,
                growth_max_bits.get(), nonfinite.get(), column_argmax.get(),
                needs_row_argmax ? row_argmax.get() : nullptr,
                next_row_argmax_column_end, status.get());
            CUDA_CHECK(cudaGetLastError());
        }
        poll_nonfinite(pivot);
    };

    if (options.schedule == Schedule::kUnblockedRank1) {
        prime(0, 0, n, n);
        for (int pivot = 0; pivot < n - 1 && !aborted; ++pivot) {
            pivot_step(pivot, n, n, n);
        }
    } else {
        for (int panel_start = 0; panel_start < n && !aborted; panel_start += options.panel_width) {
            const int panel_end = std::min(n, panel_start + options.panel_width);
            if (panel_start == 0) prime(0, 0, panel_end, panel_end);
            for (int pivot = panel_start; pivot < panel_end && pivot < n - 1 && !aborted; ++pivot) {
                pivot_step(pivot, panel_end, panel_end, panel_end);
            }
            if (aborted || panel_end >= n) break;

            // U12: same rankwise order as the frozen path. This is not a search
            // domain, so the fused state is not written here.
            for (int pivot = panel_start; pivot < panel_end; ++pivot) {
                const int rows_in_panel = panel_end - pivot - 1;
                const int future_columns = n - panel_end;
                if (rows_in_panel <= 0 || future_columns <= 0) continue;
                dim3 threads(16, 16);
                dim3 blocks(
                    (future_columns + threads.x - 1) / threads.x,
                    (rows_in_panel + threads.y - 1) / threads.y);
                rank1_update_device_kernel<<<blocks, threads>>>(
                    matrix.get(), n, pivot, pivot + 1, panel_end, panel_end, n,
                    growth_max_bits.get(), nonfinite.get(), nullptr, nullptr, 0, status.get());
                CUDA_CHECK(cudaGetLastError());
                if (!poll_now()) break;
            }
            if (aborted) break;

            // The blocked A22 update writes exactly the next panel's first search
            // domain, so it repopulates the fused state for free.
            const int next_panel_end = std::min(n, panel_end + options.panel_width);
            const int trailing = n - panel_end;
            CUDA_CHECK(cudaMemsetAsync(column_argmax.get(), 0, n * sizeof(unsigned long long)));
            if (needs_row_argmax) {
                CUDA_CHECK(cudaMemsetAsync(row_argmax.get(), 0, n * sizeof(unsigned long long)));
            }
            dim3 threads(16, 16);
            dim3 blocks(
                (trailing + threads.x - 1) / threads.x,
                (trailing + threads.y - 1) / threads.y);
            blocked_update_device_kernel<<<blocks, threads>>>(
                matrix.get(), n, panel_start, panel_end, growth_max_bits.get(),
                nonfinite.get(), column_argmax.get(),
                needs_row_argmax ? row_argmax.get() : nullptr,
                next_panel_end, status.get());
            CUDA_CHECK(cudaGetLastError());
            poll_now();
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    const auto wall_stop = std::chrono::steady_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    CUDA_CHECK(cudaEventElapsedTime(&result.cuda_ms, start_event, stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    // --- everything below is outside the timed region ---
    int device_status = kStatusOk;
    CUDA_CHECK(cudaMemcpy(&device_status, status.get(), sizeof(int), cudaMemcpyDeviceToHost));
    switch (device_status) {
        case kStatusPivotSearchFailed: result.status = "pivot_search_failed"; break;
        case kStatusNonfinitePivot: result.status = "nonfinite_pivot"; break;
        case kStatusSingularPivot: result.status = "singular_pivot"; break;
        default: result.status = aborted ? "nonfinite_factorization" : "completed"; break;
    }

    std::vector<unsigned long long> counter_values(kCounterSlots, 0);
    CUDA_CHECK(cudaMemcpy(
        counter_values.data(), counters.get(), kCounterSlots * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    result.counters.row_swaps = counter_values[kSlotRowSwaps];
    result.counters.column_swaps = counter_values[kSlotColumnSwaps];
    result.counters.dp_lookahead_accepts = counter_values[kSlotDpAccepts];
    result.counters.dp_fallbacks = counter_values[kSlotDpFallbacks];
    result.counters.gp_near_ties = counter_values[kSlotGpNearTies];
    result.counters.gp_second_choices = counter_values[kSlotGpSecondChoices];
    result.counters.scap_current = counter_values[kSlotScapCurrent];
    result.counters.scap_middle = counter_values[kSlotScapMiddle];
    result.counters.scap_last = counter_values[kSlotScapLast];
    result.counters.rp_iterations = counter_values[kSlotRpIterations];
    result.counters.rp_failures = counter_values[kSlotRpFailures];

    // Replay the permutations on the host from one copy of the device log.
    std::vector<int> log(2 * n, 0);
    CUDA_CHECK(cudaMemcpy(log.data(), pivot_log.get(), 2 * n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> rows(n);
    std::vector<int> columns(n);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(columns.begin(), columns.end(), 0);
    const int steps = std::min(executed_steps, n - 1);
    for (int pivot = 0; pivot < steps; ++pivot) {
        const int chosen_row = log[2 * pivot];
        const int chosen_column = log[2 * pivot + 1];
        result.pivot_rows.push_back(chosen_row);
        result.pivot_columns.push_back(chosen_column);
        if (chosen_row < 0 || chosen_row >= n || chosen_column < 0 || chosen_column >= n) break;
        if (chosen_row != pivot) std::swap(rows[pivot], rows[chosen_row]);
        if (chosen_column != pivot) std::swap(columns[pivot], columns[chosen_column]);
    }

    CUDA_CHECK(cudaMemcpy(
        &result.nonfinite_count, nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (result.nonfinite_count != 0 && result.status == "completed") {
        // Only reachable with --abort-check-stride 0: the frozen host path polls
        // every step and would have reported nonfinite_factorization earlier.
        result.status = "nonfinite_update";
    }

    unsigned int multiplier_bits = 0;
    CUDA_CHECK(cudaMemcpy(
        &multiplier_bits, multiplier_max_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::memcpy(&result.max_multiplier, &multiplier_bits, sizeof(float));

    if (result.status == "completed") {
        __half final_pivot{};
        CUDA_CHECK(cudaMemcpy(
            &final_pivot, matrix.get() + index_of(n - 1, n - 1, n), sizeof(__half),
            cudaMemcpyDeviceToHost));
        const float value = __half2float(final_pivot);
        if (!std::isfinite(value)) result.status = "nonfinite_final_pivot";
        else if (value == 0.0f) result.status = "singular_final_pivot";
    }

    unsigned int growth_bits = 0;
    CUDA_CHECK(cudaMemcpy(&growth_bits, growth_max_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    float growth_maximum = 0.0f;
    std::memcpy(&growth_maximum, &growth_bits, sizeof(float));
    result.growth_factor = initial_maximum == 0.0f
        ? std::numeric_limits<double>::quiet_NaN()
        : static_cast<double>(growth_maximum) / initial_maximum;

    std::vector<__half> factor(elements);
    CUDA_CHECK(cudaMemcpy(factor.data(), matrix.get(), elements * sizeof(__half), cudaMemcpyDeviceToHost));
    result.reconstruction_residual = reconstruct(
        initial, factor, rows, columns, n, &result.reconstruction_kind);
    if (result.status == "completed" && !std::isfinite(result.reconstruction_residual)) {
        result.status = "reconstruction_failed";
    }
    return result;
}

// FNV-1a digest of the realised pivot sequence. Two runs define the same
// algorithm if and only if these digests agree.
std::string pivot_digest_of(const std::vector<int>& rows, const std::vector<int>& columns) {
    std::uint64_t hash = 1469598103934665603ULL;
    const std::size_t count = std::min(rows.size(), columns.size());
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t values[2] = {
            static_cast<std::uint32_t>(rows[i]), static_cast<std::uint32_t>(columns[i])};
        for (const std::uint32_t value : values) {
            for (int byte = 0; byte < 4; ++byte) {
                hash ^= (value >> (8 * byte)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        }
    }
    char buffer[17] = {};
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
    return buffer;
}

bool file_has_content(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    return input && input.tellg() > 0;
}

std::string input_fingerprint(const std::vector<__half>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const __half value : values) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits & 0xffU;
        hash *= 1099511628211ULL;
        hash ^= bits >> 8U;
        hash *= 1099511628211ULL;
    }
    char buffer[17] = {};
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
    return buffer;
}

void append_result(
    const Options& options,
    const std::string& fingerprint,
    int repetition,
    const RunResult& result) {
    const bool write_header = !file_has_content(options.output);
    std::ofstream output(options.output, std::ios::app);
    if (!output) throw std::runtime_error("cannot open output: " + options.output);
    if (write_header) {
        output << "method,search_mode,abort_check_stride,schedule,panel_width,family,n,seed,input_fingerprint,repetition,tau,window,status,factor_wall_ms,factor_cuda_ms,"
                  "reconstruction_kind,reconstruction_residual,growth_factor,max_multiplier,nonfinite_count,"
                  "row_swaps,column_swaps,dp_lookahead_accepts,dp_fallbacks,gp_near_ties,gp_second_choices,"
                  "scap_current,scap_middle,scap_last,rp_iterations,rp_failures,pivot_digest\n";
    }
    output << method_name(options.method) << ',' << search_name(options.search) << ','
           << options.abort_check_stride << ',' << schedule_name(options.schedule) << ','
           << (options.schedule == Schedule::kBlockedPanelLocal ? options.panel_width : 1) << ','
           << family_name(options.family) << ',' << options.n << ','
           << options.seed << ',' << fingerprint << ',' << repetition << ',' << std::setprecision(9) << options.tau << ','
           << options.window << ',' << result.status << ',' << std::setprecision(12) << result.wall_ms << ','
           << result.cuda_ms << ',' << result.reconstruction_kind << ',' << result.reconstruction_residual << ','
           << result.growth_factor << ',' << result.max_multiplier << ',' << result.nonfinite_count << ','
           << result.counters.row_swaps << ',' << result.counters.column_swaps << ','
           << result.counters.dp_lookahead_accepts << ',' << result.counters.dp_fallbacks << ','
           << result.counters.gp_near_ties << ',' << result.counters.gp_second_choices << ','
           << result.counters.scap_current << ',' << result.counters.scap_middle << ','
           << result.counters.scap_last << ',' << result.counters.rp_iterations << ','
           << result.counters.rp_failures << ',' << result.pivot_digest << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<__half> input = generate_matrix(options);
        const std::string fingerprint = input_fingerprint(input);
        // NEW: dispatch on the search mode. run_once is the frozen host path.
        auto factor = [&](const Options& current) {
            return current.search == SearchMode::kFusedDevice
                ? run_once_fused(current, input)
                : run_once(current, input);
        };
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            const RunResult ignored = factor(options);
            if (ignored.status != "completed") {
                std::fprintf(stderr, "warning: warm-up status=%s\n", ignored.status.c_str());
            }
        }
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            RunResult result = factor(options);
            result.pivot_digest = pivot_digest_of(result.pivot_rows, result.pivot_columns);
            append_result(options, fingerprint, repetition, result);
            if (!options.pivot_log_path.empty() && repetition == 0) {
                std::ofstream log(options.pivot_log_path);
                if (!log) throw std::runtime_error("cannot open pivot log: " + options.pivot_log_path);
                log << "step,row,column\n";
                const std::size_t count =
                    std::min(result.pivot_rows.size(), result.pivot_columns.size());
                for (std::size_t step = 0; step < count; ++step) {
                    log << step << ',' << result.pivot_rows[step] << ','
                        << result.pivot_columns[step] << '\n';
                }
            }
            std::printf(
                "%s search=%s %s %s n=%d b=%d seed=%llu rep=%d status=%s wall_ms=%.3f "
                "cuda_ms=%.3f residual=%.6e mu=%.6g pivots=%s\n",
                method_name(options.method).c_str(), search_name(options.search).c_str(),
                schedule_name(options.schedule).c_str(),
                family_name(options.family).c_str(), options.n,
                options.schedule == Schedule::kBlockedPanelLocal ? options.panel_width : 1,
                static_cast<unsigned long long>(options.seed), repetition, result.status.c_str(),
                result.wall_ms, result.cuda_ms, result.reconstruction_residual,
                result.max_multiplier, result.pivot_digest.c_str());
            if (options.log_pivots) {
                const std::size_t count =
                    std::min(result.pivot_rows.size(), result.pivot_columns.size());
                for (std::size_t step = 0; step < count; ++step) {
                    std::printf("  pivot step=%zu row=%d column=%d\n", step,
                                result.pivot_rows[step], result.pivot_columns[step]);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
