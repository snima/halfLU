#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxScanBlocks = 65535;
constexpr float kHalfMinNormal = 0x1p-14f;
constexpr float kHalfMinSubnormal = 0x1p-24f;
constexpr float kHalfUnitRoundoff = 0x1p-11f;
constexpr float kGradedWideLog2Min = -14.0f;
constexpr float kGradedWideLog2Span = 29.0f;
constexpr double kPi = 3.141592653589793238462643383279502884;

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << file << ':' << line << ": CUDA call " << expression << " failed: "
                << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    void allocate(std::size_t count) {
        release();
        if (count == 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        count_ = count;
    }

    T* get() {
        return data_;
    }

    const T* get() const {
        return data_;
    }

    std::size_t size() const {
        return count_;
    }

private:
    void release() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    template <typename Function>
    float measure(Function&& function) {
        CUDA_CHECK(cudaEventRecord(start_));
        function();
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

enum class Mode {
    kLu,
    kTrigger,
};

// ScalingPolicy selects which signal decides *whether/how much* to scale.
// kProxy and kOracle are the manuscript baseline and the validation upper
// bound (unchanged, byte-for-byte, from the original artifact). kHybrid is
// the new, composable, low-cost candidate family: it starts from the same
// panel-diagonal read as kProxy and can be augmented with a cheap structural
// probe, an integer safety margin, and a bounded/periodic oracle-calibrated
// ratio. With every --probe/--margin-bits/--adaptive-* flag at its default,
// kHybrid is numerically identical to kProxy (checked in NAMING.md / smoke
// test) -- this is intentional so it never silently changes old commands.
enum class ScalingPolicy {
    kProxy,
    kOracle,
    kHybrid,
    // Per-column certified majorant (brainstorm A2 / "Schur-update majorant"):
    // bound[j] is a provable upper bound on column j's max magnitude, updated
    // every panel from data already computed (the panel's own U12 block), and
    // actuated with a PER-COLUMN exponent rather than one shared scalar. By
    // construction (|l_ik| <= 1 under partial pivoting) this has no false
    // negatives, independent of matrix structure -- unlike kProxy/kHybrid,
    // which are samples/witnesses that can miss growth outside what they read.
    kCertified,
    // Range-certified actuation before, rather than after, each vulnerable
    // FP16 update. An initial per-column preflight is followed by guarded
    // panel/TRSM rank-1 steps and a fused exact-bound reanchor after A22.
    kPredictive,
};

// How ScalingPolicy::kCertified's per-column bound c_j is obtained.
// kAccumulate: the original triangle-inequality recursion c_j += sum|u_pj|
// (monotone; never shrinks even when the true entries cancel -- the source
// of the measured over-conservatism/subnormal excess).
// kFused: c_j is *reanchored* every panel to the EXACT trailing value,
// captured as a side effect of the panel's final trailing update -- no
// separate scan, no triangle inequality, no accumulated slack. This is the
// "fuse sensing into the epilogue" pattern: at that point c_j is not a bound
// at all, it is the oracle value, obtained for the memory-traffic price of a
// proxy.
enum class BoundSource {
    kAccumulate,
    kFused,
};

// Computational structure of the factorization itself -- separate from
// ScalingPolicy/Probe, which only affect the *monitoring/scaling decision*.
// kUnblocked (default, unchanged): every single pivot column's trailing
// update touches the FULL remaining matrix (n-pivot-1 columns) -- this is
// what every result before this flag existed used, and --panel-width only
// ever controlled monitoring cadence for it, NOT computation.
// kBlockedFp16: MAGMA/LAPACK-style two levels. The panel (width = panel_width,
// the outer level) first factors only its own n x nb slab, then solves
// U12 = L11^-1 A12 after all panel pivots are final, and finally calls ONE
// rank-nb blocked_trailing_update_kernel for the remaining matrix. The solve
// must not be interleaved with panel pivoting: a row swapped in from below the
// panel would otherwise carry an incompletely updated A12 row.
// kBlockedFp32Accumulate: same two levels, but the rank-nb inner product in
// the trailing update accumulates in fp32 and rounds once, matching what a
// real mixed-precision GEMM does -- a genuinely different rounding path.
enum class Blocking {
    kUnblocked,
    kBlockedFp16,
    kBlockedFp32Accumulate,
};

// Cheap structural augmentation for kHybrid. kDiagonal is a no-op (reduces
// to the manuscript proxy). kLastColumn adds an O(n - panel_end) read of the
// matrix's final column, which is where classical worst-case partial-pivoting
// growth (e.g. the Wilkinson matrix) is structurally concentrated.
enum class Probe {
    kDiagonal,
    kLastColumn,
    // Max over the *whole* U12 block (rows = panel's own rows, cols = every
    // future column) instead of one fixed column. In row-major storage this
    // is a contiguous/coalesced read per row, unlike kLastColumn's strided
    // column read -- a real, measurable GPU-cost difference, not just an
    // algorithmic one. Catches growth in *any* future column, not only the
    // last one, at the same O(panel_width * (n - panel_end)) order.
    kPivotRow,
};

enum class Family {
    kRandom,
    kGraded,
    // A non-diagonally-dominant graded stress family spanning [2^-14, 2^15]
    // across columns. Its largest initial entry remains finite in FP16, while
    // a modest LU growth factor can exercise overflow prevention.
    kGradedWide,
    // A dense, prescribed-condition D0-style construction: before conversion
    // to FP16, its singular values are {2^-e, 1, ..., 1}. It deliberately
    // tests conditioning rather than a named external HHP21 construction.
    kNearSingular,
    kWilkinson,
    kUnderflow,
    // Plain i.i.d. U(-1,1), NO diagonal boost -- unlike kRandom, which is
    // diagonally dominant by construction (diag~1, off-diag~0.01) and was
    // observed empirically in baseline tests to be a materially easier/more
    // benign ensemble than the manuscript's actual `rand`/`randn` test
    // matrices (Table 1). Added specifically to test claims (e.g. "lambda_p
    // ~ 1 under partial pivoting") that assume genuinely competing column
    // entries, which kRandom's diagonal dominance rules out by construction.
    kIidRandom,
};

enum class MatrixNormalization {
    kNone,
    kPowerOfTwoMaxAbs,
};

struct Options {
    Mode mode = Mode::kLu;
    Family family = Family::kRandom;
    int n = 256;
    int panel_width = 8;
    int near_singular_exponent = 10;
    int samples = 1;
    int max_panels = -1;
    std::uint64_t seed = 20260806ULL;
    float threshold = 0.25f;
    float safe_threshold = 32768.0f;  // kPredictive: safe for sequential FP16 update widths <= 1024
    float rank_safe_threshold = 60000.0f;  // kPredictive: safe for one FP16 multiply/subtract
    bool scaling = true;
    ScalingPolicy scaling_policy = ScalingPolicy::kProxy;
    Probe probe = Probe::kDiagonal;
    int margin_bits = 0;             // extra exponent added whenever a trigger fires (kProxy, kHybrid)
    int adaptive_burnin = 0;         // first N panels are oracle-calibrated (kHybrid only)
    int adaptive_resync_every = 0;  // 0 = never; else every K-th panel is also oracle-calibrated
    // If true and probe == kLastColumn, scale ONLY the probed column instead
    // of the whole [panel_end, n) block. Rationale: uniform full-block scaling
    // repeatedly divides every future diagonal too, even though those
    // diagonals are structurally constant (not growing) in matrices like
    // Wilkinson's -- over enough panels this drives a future diagonal to
    // underflow to exact 0 in FP16, which is a *worse* failure (division by
    // zero -> NaN contaminating the whole trailing block) than the overflow
    // it was trying to prevent. Scoping the correction to only the column
    // that was actually flagged avoids manufacturing that new failure mode.
    bool scoped_scaling = false;
    BoundSource bound_source = BoundSource::kAccumulate;  // kCertified only
    bool measure_lambda = false;  // opt-in diagnostic; off by default so it never changes existing timings
    Blocking blocking = Blocking::kUnblocked;
    bool rankwise_u12 = false;  // performance ablation: rankwise GPU U12 schedule without predictive guards
    std::string matrix_market;
    MatrixNormalization matrix_normalization = MatrixNormalization::kNone;
    bool snapshot_original = false;
    bool snapshot_final = false;
    std::vector<int> snapshot_panels;
    std::string output = "results/run";
};

struct ScaleStats {
    unsigned long long subnormal_outputs = 0;
    unsigned long long flushed_to_zero = 0;
    unsigned long long nonfinite_values = 0;
};

struct PanelRecord {
    int sample = 0;
    int panel = 0;
    int panel_start = 0;
    int panel_end = 0;
    float dmax_proxy = 0.0f;
    float trailing_max_oracle = 0.0f;
    float ratio_oracle_over_proxy = 0.0f;
    int proxy_exponent = 0;
    int oracle_exponent = 0;
    int applied_exponent = 0;
    std::string classification;
    int under_scale_bits = 0;
    int over_scale_bits = 0;
    unsigned long long proxy_nonfinite = 0;
    unsigned long long oracle_nonfinite = 0;
    float factorization_ms = 0.0f;
    float proxy_scan_ms = 0.0f;
    float oracle_scan_ms = 0.0f;
    float scale_ms = 0.0f;
    unsigned long long scaled_elements = 0;
    ScaleStats scale_stats;

    // kHybrid bookkeeping. Zero/1.0/0 whenever policy != kHybrid, and also
    // whenever kHybrid runs with default probe/adaptive flags, so old
    // (proxy/oracle) runs and new default-hybrid runs are byte-identical.
    float probe_value = 0.0f;
    float effective_value = 0.0f;
    double ratio_estimate = 1.0;
    int is_calibration_panel = 0;
    int base_exponent = 0;  // exponent applied to the whole future block (excludes any targeted extra)
    float probe_scan_ms = 0.0f;
    // Diagnostic only (kCertified): max/min over this panel's own pivot
    // columns of the largest multiplier magnitude below the pivot. Tests the
    // "lambda_p ~ 1 under partial pivoting" prediction directly, rather than
    // assuming it.
    float lambda_max = 0.0f;
    float lambda_min = 0.0f;
};

struct RunSummary {
    int sample = 0;
    bool completed = false;
    std::string termination = "not_started";
    int panels_observed = 0;
    int proxy_activations = 0;
    int oracle_activations = 0;
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    int true_negative = 0;
    int exponent_mismatch_panels = 0;
    unsigned long long total_under_scale_bits = 0;
    unsigned long long total_over_scale_bits = 0;
    unsigned long long total_scaled_elements = 0;
    ScaleStats scale_stats;
    float factorization_ms = 0.0f;
    float proxy_scan_ms = 0.0f;
    float oracle_scan_ms = 0.0f;
    float scale_ms = 0.0f;
    unsigned long long final_nonfinite = 0;
    int row_swaps = 0;
    bool column_scale_permutation_self_test_passed = false;
    std::string reconstruction_kind = "not_run";
    double reconstruction_relative_residual = std::numeric_limits<double>::quiet_NaN();
};

struct DiagnosticValue {
    float maximum = 0.0f;
    unsigned long long nonfinite = 0;
    float elapsed_ms = 0.0f;
};

struct ReconstructionResult {
    std::string kind;
    double relative_residual = std::numeric_limits<double>::quiet_NaN();
};

__device__ std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ float random_signed(std::uint64_t value) {
    const std::uint64_t mixed = mix64(value);
    const float unit = static_cast<float>(mixed & 0x00ffffffULL) / 16777215.0f;
    return 2.0f * unit - 1.0f;
}

__global__ void convert_to_half_kernel(const float* input, __half* output, unsigned long long count) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long current = index; current < count; current += stride) {
        output[current] = __float2half_rn(input[current]);
    }
}

__global__ void generate_trigger_matrix_kernel(
    __half* matrix,
    int n,
    std::uint64_t seed,
    int family_code,
    int near_singular_exponent) {
    const unsigned long long count = static_cast<unsigned long long>(n) * n;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    for (unsigned long long current = index; current < count; current += stride) {
        const int row = static_cast<int>(current / n);
        const int column = static_cast<int>(current % n);
        float value = 0.0f;
        const float random_value = random_signed(seed + current);

        if (family_code == 0) {
            value = 0.5f * random_value;
            if (row == column) {
                value += 1.0f;
            }
        } else if (family_code == 1) {
            const float position = n > 1 ? static_cast<float>(column) / static_cast<float>(n - 1) : 0.0f;
            const float column_scale = exp2f(-4.0f + 8.0f * position);
            value = 0.05f * random_value * column_scale;
            if (row == column) {
                value += column_scale;
            }
        } else if (family_code == 5) {
            const float position = n > 1 ? static_cast<float>(column) / static_cast<float>(n - 1) : 0.0f;
            const float column_scale = exp2f(kGradedWideLog2Min + kGradedWideLog2Span * position);
            value = 0.25f * random_value * column_scale;
            if (row == column) {
                value += column_scale;
            }
        } else if (family_code == 6) {
            const float normalization = sqrtf(2.0f / static_cast<float>(n + 1));
            const float row_angle = static_cast<float>(kPi * static_cast<double>(row + 1) / static_cast<double>(n + 1));
            const float column_angle = static_cast<float>(kPi * static_cast<double>(column + 1) / static_cast<double>(n + 1));
            const float row_sign = random_signed(seed + static_cast<std::uint64_t>(row)) < 0.0f ? -1.0f : 1.0f;
            const float column_sign = random_signed(seed + static_cast<std::uint64_t>(column)) < 0.0f ? -1.0f : 1.0f;
            const float sigma = ldexpf(1.0f, -near_singular_exponent);
            const float q_row = row_sign * normalization * sinf(row_angle);
            const float q_column = column_sign * normalization * sinf(column_angle);
            value = (row == column ? 1.0f : 0.0f) - (1.0f - sigma) * q_row * q_column;
        } else if (family_code == 2) {
            value = row == column ? 1.0f : kHalfMinNormal;
        } else if (family_code == 4) {
            value = random_value;  // plain i.i.d., no diagonal boost
        } else {
            if (column == n - 1 || row == column) {
                value = 1.0f;
            } else if (row > column) {
                value = -1.0f;
            }
        }
        matrix[current] = __float2half_rn(value);
    }
}

__global__ void choose_partial_pivot_kernel(const __half* matrix, int n, int column, int* pivots) {
    __shared__ float maxima[kThreads];
    __shared__ int indices[kThreads];

    float local_maximum = -1.0f;
    int local_index = column;
    for (int row = column + threadIdx.x; row < n; row += blockDim.x) {
        const float value = fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column]));
        if (isfinite(value) && (value > local_maximum || (value == local_maximum && row < local_index))) {
            local_maximum = value;
            local_index = row;
        }
    }
    maxima[threadIdx.x] = local_maximum;
    indices[threadIdx.x] = local_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const float candidate = maxima[threadIdx.x + offset];
            const int candidate_index = indices[threadIdx.x + offset];
            if (candidate > maxima[threadIdx.x] ||
                (candidate == maxima[threadIdx.x] && candidate_index < indices[threadIdx.x])) {
                maxima[threadIdx.x] = candidate;
                indices[threadIdx.x] = candidate_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        pivots[column] = indices[0];
    }
}

__global__ void swap_rows_kernel(__half* matrix, int n, int row_a, const int* pivot) {
    const int row_b = *pivot;
    if (row_a == row_b) {
        return;
    }
    for (int column = threadIdx.x + blockIdx.x * blockDim.x; column < n; column += blockDim.x * gridDim.x) {
        const unsigned long long first_index = static_cast<unsigned long long>(row_a) * n + column;
        const unsigned long long second_index = static_cast<unsigned long long>(row_b) * n + column;
        const __half temporary = matrix[first_index];
        matrix[first_index] = matrix[second_index];
        matrix[second_index] = temporary;
    }
}

__global__ void divide_pivot_column_kernel(__half* matrix, int n, int pivot_index) {
    const int row = pivot_index + 1 + threadIdx.x + blockIdx.x * blockDim.x;
    if (row < n) {
        const unsigned long long row_index = static_cast<unsigned long long>(row) * n + pivot_index;
        const unsigned long long pivot_diagonal = static_cast<unsigned long long>(pivot_index) * n + pivot_index;
        matrix[row_index] = __hdiv(matrix[row_index], matrix[pivot_diagonal]);
    }
}

// column_running_max_bits, when non-null, is written with an exact per-column
// max as a *side effect* of the update this kernel is already performing --
// zero extra global-memory traffic, since every element it touches here was
// already going to be read and written regardless. In unblocked mode, call
// this on the last pivot of a panel (after resetting the buffer) to capture
// the exact post-panel trailing state, matching a full oracle scan without a
// second pass over the data.
__global__ void update_trailing_matrix_kernel(
    __half* matrix, int n, int pivot_index, int row_start, int row_end, int col_start, int col_end,
    unsigned int* column_running_max_bits) {
    const int column = col_start + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = row_start + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < row_end && column < col_end) {
        const unsigned long long multiplier_index = static_cast<unsigned long long>(row) * n + pivot_index;
        const unsigned long long pivot_row_index = static_cast<unsigned long long>(pivot_index) * n + column;
        const unsigned long long entry_index = static_cast<unsigned long long>(row) * n + column;
        const __half multiplier = matrix[multiplier_index];
        const __half pivot_row_value = matrix[pivot_row_index];
        const __half before = matrix[entry_index];
        const __half updated = __hsub(before, __hmul(multiplier, pivot_row_value));
        matrix[entry_index] = updated;
        if (column_running_max_bits != nullptr) {
            const float magnitude = fabsf(__half2float(updated));
            atomicMax(&column_running_max_bits[column], __float_as_uint(magnitude));
        }
    }
}

// Complete a panel only after its pivot sequence is settled. One thread owns
// one future column and applies forward substitution down the panel rows,
// producing U12 = L11^-1 A12 without any row being skipped before a swap.
__global__ void panel_u12_triangular_solve_kernel(__half* matrix, int n, int panel_start, int panel_end) {
    const int column = panel_end + blockIdx.x * blockDim.x + threadIdx.x;
    if (column < n) {
        for (int pivot = panel_start; pivot < panel_end; ++pivot) {
            const __half pivot_row_value = matrix[static_cast<unsigned long long>(pivot) * n + column];
            for (int row = pivot + 1; row < panel_end; ++row) {
                const unsigned long long multiplier_index = static_cast<unsigned long long>(row) * n + pivot;
                const unsigned long long entry_index = static_cast<unsigned long long>(row) * n + column;
                matrix[entry_index] = __hsub(matrix[entry_index], __hmul(matrix[multiplier_index], pivot_row_value));
            }
        }
    }
}

// MAGMA/LAPACK-style right-looking blocked LU has two levels: an outer PANEL
// (width nb, this file's --panel-width) whose n x nb slab is factored first.
// panel_u12_triangular_solve_kernel then forms U12, and this kernel performs
// the one rank-nb update of A22 rather than nb rank-1 updates. This is the
// realistic GEMM-equivalent fusion point and the natural place to capture an
// exact per-column maximum. fp32_accumulate=false preserves the fp16 update
// sequence; fp32_accumulate=true accumulates the rank-nb inner product in fp32
// and rounds once, matching a mixed-precision GEMM.
__global__ void blocked_trailing_update_kernel(
    __half* matrix,
    int n,
    int panel_start,
    int panel_end,
    bool fp32_accumulate,
    unsigned int* column_running_max_bits) {
    const int column = panel_end + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = panel_end + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n) {
        const unsigned long long entry_index = static_cast<unsigned long long>(row) * n + column;
        float updated_f32 = 0.0f;
        __half updated_f16 = matrix[entry_index];
        if (fp32_accumulate) {
            float accumulator = __half2float(matrix[entry_index]);
            for (int p = panel_start; p < panel_end; ++p) {
                const float multiplier = __half2float(matrix[static_cast<unsigned long long>(row) * n + p]);
                const float pivot_row_value = __half2float(matrix[static_cast<unsigned long long>(p) * n + column]);
                accumulator -= multiplier * pivot_row_value;
            }
            updated_f32 = accumulator;
            updated_f16 = __float2half_rn(accumulator);
        } else {
            for (int p = panel_start; p < panel_end; ++p) {
                const __half multiplier = matrix[static_cast<unsigned long long>(row) * n + p];
                const __half pivot_row_value = matrix[static_cast<unsigned long long>(p) * n + column];
                updated_f16 = __hsub(updated_f16, __hmul(multiplier, pivot_row_value));
            }
            updated_f32 = __half2float(updated_f16);
        }
        matrix[entry_index] = updated_f16;
        if (column_running_max_bits != nullptr) {
            atomicMax(&column_running_max_bits[column], __float_as_uint(fabsf(updated_f32)));
        }
    }
}

__global__ void panel_diagonal_max_kernel(
    const __half* matrix,
    int n,
    int panel_start,
    int panel_end,
    float* maximum,
    unsigned long long* nonfinite) {
    __shared__ float maxima[kThreads];
    __shared__ unsigned long long nonfinite_counts[kThreads];

    float local_maximum = 0.0f;
    unsigned long long local_nonfinite = 0;
    for (int index = panel_start + threadIdx.x; index < panel_end; index += blockDim.x) {
        const float value = fabsf(__half2float(matrix[static_cast<unsigned long long>(index) * n + index]));
        if (isfinite(value)) {
            local_maximum = fmaxf(local_maximum, value);
        } else {
            ++local_nonfinite;
        }
    }
    maxima[threadIdx.x] = local_maximum;
    nonfinite_counts[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
            nonfinite_counts[threadIdx.x] += nonfinite_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        maximum[0] = maxima[0];
        nonfinite[0] = nonfinite_counts[0];
    }
}

__global__ void trailing_max_kernel(
    const __half* matrix,
    int n,
    int start,
    unsigned int* maximum_bits,
    unsigned long long* nonfinite) {
    __shared__ float maxima[kThreads];
    __shared__ unsigned long long nonfinite_counts[kThreads];

    const unsigned long long side = static_cast<unsigned long long>(n - start);
    const unsigned long long count = side * side;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    float local_maximum = 0.0f;
    unsigned long long local_nonfinite = 0;
    for (unsigned long long current = index; current < count; current += stride) {
        const int row = start + static_cast<int>(current / side);
        const int column = start + static_cast<int>(current % side);
        const float value = fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column]));
        if (isfinite(value)) {
            local_maximum = fmaxf(local_maximum, value);
        } else {
            ++local_nonfinite;
        }
    }

    maxima[threadIdx.x] = local_maximum;
    nonfinite_counts[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
            nonfinite_counts[threadIdx.x] += nonfinite_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(maximum_bits, __float_as_uint(maxima[0]));
        if (nonfinite_counts[0] != 0) {
            atomicAdd(nonfinite, nonfinite_counts[0]);
        }
    }
}

// Reduction over a single column, rows [row_start, n). This is the cheap
// O(n - row_start) structural probe used by ScalingPolicy::kHybrid with
// Probe::kLastColumn: it targets exactly the location where classical
// worst-case partial-pivoting growth (the Wilkinson matrix) is concentrated,
// without paying for the full (n - panel_end)^2 trailing-block oracle scan.
__global__ void column_segment_max_kernel(
    const __half* matrix,
    int n,
    int column,
    int row_start,
    unsigned int* maximum_bits,
    unsigned long long* nonfinite) {
    __shared__ float maxima[kThreads];
    __shared__ unsigned long long nonfinite_counts[kThreads];

    float local_maximum = 0.0f;
    unsigned long long local_nonfinite = 0;
    for (int row = row_start + blockIdx.x * blockDim.x + threadIdx.x; row < n; row += blockDim.x * gridDim.x) {
        const float value = fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column]));
        if (isfinite(value)) {
            local_maximum = fmaxf(local_maximum, value);
        } else {
            ++local_nonfinite;
        }
    }

    maxima[threadIdx.x] = local_maximum;
    nonfinite_counts[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
            nonfinite_counts[threadIdx.x] += nonfinite_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(maximum_bits, __float_as_uint(maxima[0]));
        if (nonfinite_counts[0] != 0) {
            atomicAdd(nonfinite, nonfinite_counts[0]);
        }
    }
}

// Reduction over rows [row_start, row_end) x columns [col_start, n) -- the
// U12 block for Probe::kPivotRow. Column-fastest indexing means consecutive
// threads read consecutive columns of the SAME row, which is a coalesced
// access in this row-major layout (matrix[row*n+col]); contrast with
// column_segment_max_kernel above, where consecutive threads read the same
// column at different rows -- a stride-n, uncoalesced access. Same asymptotic
// cost, different real memory-bandwidth cost; both are measured below.
__global__ void row_block_max_kernel(
    const __half* matrix,
    int n,
    int row_start,
    int row_end,
    int col_start,
    unsigned int* maximum_bits,
    unsigned long long* nonfinite) {
    __shared__ float maxima[kThreads];
    __shared__ unsigned long long nonfinite_counts[kThreads];

    const unsigned long long height = static_cast<unsigned long long>(row_end - row_start);
    const unsigned long long width = static_cast<unsigned long long>(n - col_start);
    const unsigned long long count = height * width;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    float local_maximum = 0.0f;
    unsigned long long local_nonfinite = 0;
    for (unsigned long long current = index; current < count; current += stride) {
        const int row = row_start + static_cast<int>(current / width);
        const int column = col_start + static_cast<int>(current % width);
        const float value = fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column]));
        if (isfinite(value)) {
            local_maximum = fmaxf(local_maximum, value);
        } else {
            ++local_nonfinite;
        }
    }

    maxima[threadIdx.x] = local_maximum;
    nonfinite_counts[threadIdx.x] = local_nonfinite;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
            nonfinite_counts[threadIdx.x] += nonfinite_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(maximum_bits, __float_as_uint(maxima[0]));
        if (nonfinite_counts[0] != 0) {
            atomicAdd(nonfinite, nonfinite_counts[0]);
        }
    }
}

// Per-column certified majorant update for ScalingPolicy::kCertified. Under
// partial pivoting |l_ik| <= 1, so for the Schur update
// a_ij <- a_ij - l_ik u_kj, a valid (one-sided, no-false-negative) bound is
//   bound[j] += sum over the panel's own rows p of |u_pj|.
// One thread per future column; height (panel_width) is small so a per-column
// serial loop over it is cheap and needs no reduction/shared memory.
__global__ void column_bound_accumulate_kernel(
    const __half* matrix,
    int n,
    int row_start,
    int row_end,
    int col_start,
    int col_end,
    float* bound) {
    for (int column = col_start + blockIdx.x * blockDim.x + threadIdx.x; column < col_end;
         column += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (int row = row_start; row < row_end; ++row) {
            sum += static_cast<double>(fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column])));
        }
        const float rounded = static_cast<float>(static_cast<double>(bound[column]) + sum);
        bound[column] = nextafterf(rounded, CUDART_INF_F);
    }
}

// One-time O(n^2) initialization of the certified bound: bound[j] = max_i |A_ij|
// over the FULL initial matrix, so the recursion above starts from a true
// upper bound rather than from zero. Paid once per sample, not once per panel.
__global__ void column_bound_init_kernel(const __half* matrix, int n, float* bound) {
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < n; column += blockDim.x * gridDim.x) {
        float maximum = 0.0f;
        for (int row = 0; row < n; ++row) {
            maximum = fmaxf(maximum, fabsf(__half2float(matrix[static_cast<unsigned long long>(row) * n + column])));
        }
        bound[column] = maximum;
    }
}

// Applies a PER-COLUMN power-of-two exponent (exponents_to_apply[j], 0 = skip)
// to columns [col_start, n), all rows -- unlike scale_future_columns_kernel's
// single shared exponent. Also divides the certified bound itself by the same
// factor, so the c_j >= true-max invariant is preserved in the new scale.
__device__ void add_scaling_event_counts(
    unsigned long long local_subnormal,
    unsigned long long local_flushed,
    unsigned long long local_nonfinite,
    unsigned long long* block_subnormal,
    unsigned long long* block_flushed,
    unsigned long long* block_nonfinite,
    unsigned long long* subnormal_outputs,
    unsigned long long* flushed_to_zero,
    unsigned long long* nonfinite_values) {
    const int thread = threadIdx.x;
    block_subnormal[thread] = local_subnormal;
    block_flushed[thread] = local_flushed;
    block_nonfinite[thread] = local_nonfinite;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            block_subnormal[thread] += block_subnormal[thread + offset];
            block_flushed[thread] += block_flushed[thread + offset];
            block_nonfinite[thread] += block_nonfinite[thread + offset];
        }
        __syncthreads();
    }

    if (thread == 0) {
        if (block_subnormal[0] != 0) {
            atomicAdd(subnormal_outputs, block_subnormal[0]);
        }
        if (block_flushed[0] != 0) {
            atomicAdd(flushed_to_zero, block_flushed[0]);
        }
        if (block_nonfinite[0] != 0) {
            atomicAdd(nonfinite_values, block_nonfinite[0]);
        }
    }
}

__global__ void scale_columns_per_column_kernel(
    __half* matrix,
    int n,
    int col_start,
    int col_end,
    const int* exponents_to_apply,
    float* bound,
    unsigned long long* subnormal_outputs,
    unsigned long long* flushed_to_zero,
    unsigned long long* nonfinite_values) {
    const int width = col_end - col_start;
    const unsigned long long count = static_cast<unsigned long long>(n) * width;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned long long local_subnormal = 0;
    unsigned long long local_flushed = 0;
    unsigned long long local_nonfinite = 0;
    __shared__ unsigned long long block_subnormal[kThreads];
    __shared__ unsigned long long block_flushed[kThreads];
    __shared__ unsigned long long block_nonfinite[kThreads];

    for (unsigned long long current = index; current < count; current += stride) {
        const int row = static_cast<int>(current / width);
        const int column = col_start + static_cast<int>(current % width);
        const int exponent = exponents_to_apply[column - col_start];
        if (exponent == 0) {
            continue;
        }
        const unsigned long long matrix_index = static_cast<unsigned long long>(row) * n + column;
        const float input = __half2float(matrix[matrix_index]);
        if (!isfinite(input)) {
            ++local_nonfinite;
            continue;
        }
        const float scaled = ldexpf(input, -exponent);
        const __half stored = __float2half_rn(scaled);
        const float stored_value = __half2float(stored);
        matrix[matrix_index] = stored;
        if (input != 0.0f && stored_value == 0.0f) {
            ++local_flushed;
        } else if (stored_value != 0.0f && fabsf(stored_value) < kHalfMinNormal) {
            ++local_subnormal;
        }
        if (!isfinite(stored_value)) {
            ++local_nonfinite;
        }
        if (row == 0) {
            bound[column] = nextafterf(ldexpf(bound[column], -exponent) + kHalfMinSubnormal, CUDART_INF_F);
        }
    }

    add_scaling_event_counts(
        local_subnormal,
        local_flushed,
        local_nonfinite,
        block_subnormal,
        block_flushed,
        block_nonfinite,
        subnormal_outputs,
        flushed_to_zero,
        nonfinite_values);
}

__device__ int device_exponent_for(float maximum, float threshold) {
    if (!isfinite(maximum) || maximum <= threshold) {
        return 0;
    }
    const int exponent = static_cast<int>(ceilf(log2f(maximum / threshold)));
    return exponent > 0 ? exponent : 0;
}

// One coalesced 32-column tile per block. Lane x owns one column while the y
// dimension partitions rows, so matrix traffic is coalesced across columns.
// Decision, full-column scaling, bound maintenance, exponent bookkeeping, and
// audit counters stay on-device; there is no per-rank host synchronization.
__global__ void predictive_guard_columns_kernel(
    __half* matrix,
    int n,
    int source_row,
    bool add_source_row,
    bool inflate_before_add,
    int col_start,
    int col_end,
    float threshold,
    float* bound,
    int* cumulative_exponents,
    unsigned long long* guard_stats) {
    const int lane = threadIdx.x;
    const int column = col_start + blockIdx.x * blockDim.x + lane;
    __shared__ int block_exponents[32];
    __shared__ unsigned long long block_subnormal[kThreads];
    __shared__ unsigned long long block_flushed[kThreads];
    __shared__ unsigned long long block_nonfinite[kThreads];
    unsigned long long local_subnormal = 0;
    unsigned long long local_flushed = 0;
    unsigned long long local_nonfinite = 0;

    if (threadIdx.y == 0) {
        int exponent = 0;
        if (column < col_end) {
            double next_bound = static_cast<double>(bound[column]);
            if (inflate_before_add) {
                const double inflation =
                    static_cast<double>(1.0f + kHalfUnitRoundoff) * (1.0f + kHalfUnitRoundoff);
                next_bound = next_bound * inflation + kHalfMinSubnormal;
            }
            if (add_source_row) {
                next_bound += fabs(static_cast<double>(
                    __half2float(matrix[static_cast<unsigned long long>(source_row) * n + column])));
            }
            float rounded_bound = nextafterf(static_cast<float>(next_bound), CUDART_INF_F);
            exponent = device_exponent_for(rounded_bound, threshold);
            if (exponent > 0) {
                rounded_bound = nextafterf(
                    ldexpf(rounded_bound, -exponent) + kHalfMinSubnormal, CUDART_INF_F);
                cumulative_exponents[column] += exponent;
                atomicAdd(guard_stats + 3, 1ULL);
                atomicMax(guard_stats + 4, static_cast<unsigned long long>(exponent));
            }
            bound[column] = rounded_bound;
        }
        block_exponents[lane] = exponent;
    }
    __syncthreads();

    if (column < col_end) {
        const int exponent = block_exponents[lane];
        if (exponent > 0) {
            for (int row = threadIdx.y; row < n; row += blockDim.y) {
                const unsigned long long index = static_cast<unsigned long long>(row) * n + column;
                const float input = __half2float(matrix[index]);
                if (!isfinite(input)) {
                    ++local_nonfinite;
                    continue;
                }
                const __half stored = __float2half_rn(ldexpf(input, -exponent));
                const float stored_value = __half2float(stored);
                matrix[index] = stored;
                if (input != 0.0f && stored_value == 0.0f) {
                    ++local_flushed;
                } else if (stored_value != 0.0f && fabsf(stored_value) < kHalfMinNormal) {
                    ++local_subnormal;
                }
                if (!isfinite(stored_value)) {
                    ++local_nonfinite;
                }
            }
        }
    }

    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    block_subnormal[thread] = local_subnormal;
    block_flushed[thread] = local_flushed;
    block_nonfinite[thread] = local_nonfinite;
    __syncthreads();
    for (int offset = blockDim.x * blockDim.y / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            block_subnormal[thread] += block_subnormal[thread + offset];
            block_flushed[thread] += block_flushed[thread + offset];
            block_nonfinite[thread] += block_nonfinite[thread + offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        if (block_subnormal[0] != 0) atomicAdd(guard_stats, block_subnormal[0]);
        if (block_flushed[0] != 0) atomicAdd(guard_stats + 1, block_flushed[0]);
        if (block_nonfinite[0] != 0) atomicAdd(guard_stats + 2, block_nonfinite[0]);
    }
}

// One thread per future column: turn its certified bound into the exponent
// that will actually be applied (kCertified's per-column decision step).
__global__ void decide_column_exponents_kernel(
    const float* bound, int col_start, int col_end, float threshold, int margin_bits, int* exponents_out) {
    for (int column = col_start + blockIdx.x * blockDim.x + threadIdx.x; column < col_end;
         column += blockDim.x * gridDim.x) {
        int exponent = device_exponent_for(bound[column], threshold);
        if (exponent > 0) {
            exponent += margin_bits;
        }
        exponents_out[column - col_start] = exponent;
    }
}

// BoundSource::kFused: overwrite (reanchor) bound[j] from the exact value
// captured as a side effect of the panel's final trailing update -- not an
// accumulation, a replacement.
__global__ void reanchor_column_bound_kernel(
    float* bound, const unsigned int* fused_max_bits, int col_start, int n) {
    for (int column = col_start + blockIdx.x * blockDim.x + threadIdx.x; column < n; column += blockDim.x * gridDim.x) {
        bound[column] = __uint_as_float(fused_max_bits[column]);
    }
}

// Block-reduction max over values[start, n) -- used to summarize bound[] into
// one scalar for logging/comparison against the other policies' dmax/oracle.
__global__ void array_segment_max_kernel(const float* values, int start, int n, unsigned int* maximum_bits) {
    __shared__ float maxima[kThreads];
    float local_maximum = 0.0f;
    for (int index = start + blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        local_maximum = fmaxf(local_maximum, values[index]);
    }
    maxima[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(maximum_bits, __float_as_uint(maxima[0]));
    }
}

__global__ void scale_future_columns_kernel(
    __half* matrix,
    int n,
    int first_column,
    int exponent,
    unsigned long long* subnormal_outputs,
    unsigned long long* flushed_to_zero,
    unsigned long long* nonfinite_values) {
    const int width = n - first_column;
    const unsigned long long count = static_cast<unsigned long long>(n) * width;
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned long long local_subnormal = 0;
    unsigned long long local_flushed = 0;
    unsigned long long local_nonfinite = 0;
    __shared__ unsigned long long block_subnormal[kThreads];
    __shared__ unsigned long long block_flushed[kThreads];
    __shared__ unsigned long long block_nonfinite[kThreads];

    for (unsigned long long current = index; current < count; current += stride) {
        const int row = static_cast<int>(current / width);
        const int column = first_column + static_cast<int>(current % width);
        const unsigned long long matrix_index = static_cast<unsigned long long>(row) * n + column;
        const float input = __half2float(matrix[matrix_index]);

        if (!isfinite(input)) {
            ++local_nonfinite;
            continue;
        }

        const float scaled = ldexpf(input, -exponent);
        const __half stored = __float2half_rn(scaled);
        const float stored_value = __half2float(stored);
        matrix[matrix_index] = stored;

        if (input != 0.0f && stored_value == 0.0f) {
            ++local_flushed;
        } else if (stored_value != 0.0f && fabsf(stored_value) < kHalfMinNormal) {
            ++local_subnormal;
        }
        if (!isfinite(stored_value)) {
            ++local_nonfinite;
        }
    }

    add_scaling_event_counts(
        local_subnormal,
        local_flushed,
        local_nonfinite,
        block_subnormal,
        block_flushed,
        block_nonfinite,
        subnormal_outputs,
        flushed_to_zero,
        nonfinite_values);
}

__global__ void add_column_exponent_kernel(int* exponents, int first_column, int n, int exponent) {
    for (int column = first_column + blockIdx.x * blockDim.x + threadIdx.x; column < n;
         column += blockDim.x * gridDim.x) {
        exponents[column] += exponent;
    }
}

// Per-column version: exponents[col] += exponents_to_apply[col - first_column].
// Used by kCertified, where each future column gets its own exponent instead
// of one shared scalar.
__global__ void add_per_column_exponent_kernel(
    int* exponents, int first_column, int last_column, const int* exponents_to_apply) {
    for (int column = first_column + blockIdx.x * blockDim.x + threadIdx.x; column < last_column;
         column += blockDim.x * gridDim.x) {
        exponents[column] += exponents_to_apply[column - first_column];
    }
}

__global__ void swap_columns_kernel(__half* matrix, int n, int column_a, int column_b) {
    if (column_a == column_b) {
        return;
    }
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += blockDim.x * gridDim.x) {
        const unsigned long long first_index = static_cast<unsigned long long>(row) * n + column_a;
        const unsigned long long second_index = static_cast<unsigned long long>(row) * n + column_b;
        const __half temporary = matrix[first_index];
        matrix[first_index] = matrix[second_index];
        matrix[second_index] = temporary;
    }
}

__global__ void swap_exponents_kernel(int* exponents, int first, int second) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && first != second) {
        const int temporary = exponents[first];
        exponents[first] = exponents[second];
        exponents[second] = temporary;
    }
}

int family_code(Family family) {
    switch (family) {
        case Family::kRandom:
            return 0;
        case Family::kGraded:
            return 1;
        case Family::kGradedWide:
            return 5;
        case Family::kNearSingular:
            return 6;
        case Family::kUnderflow:
            return 2;
        case Family::kWilkinson:
            return 3;
        case Family::kIidRandom:
            return 4;
    }
    return 0;
}

std::string family_name(Family family) {
    switch (family) {
        case Family::kRandom:
            return "random";
        case Family::kGraded:
            return "graded";
        case Family::kGradedWide:
            return "graded_wide";
        case Family::kNearSingular:
            return "near_singular";
        case Family::kWilkinson:
            return "wilkinson";
        case Family::kUnderflow:
            return "underflow";
        case Family::kIidRandom:
            return "iid_random";
    }
    return "unknown";
}

std::string mode_name(Mode mode) {
    return mode == Mode::kLu ? "lu" : "trigger";
}

std::string scaling_policy_name(ScalingPolicy policy) {
    switch (policy) {
        case ScalingPolicy::kProxy:
            return "proxy";
        case ScalingPolicy::kOracle:
            return "oracle";
        case ScalingPolicy::kHybrid:
            return "hybrid";
        case ScalingPolicy::kCertified:
            return "certified";
        case ScalingPolicy::kPredictive:
            return "predictive";
    }
    return "unknown";
}

std::string probe_name(Probe probe) {
    switch (probe) {
        case Probe::kLastColumn:
            return "last_column";
        case Probe::kPivotRow:
            return "pivot_row";
        default:
            return "diagonal";
    }
}

std::string bound_source_name(BoundSource source) {
    return source == BoundSource::kFused ? "fused" : "accumulate";
}

std::string blocking_name(Blocking blocking) {
    switch (blocking) {
        case Blocking::kBlockedFp16:
            return "blocked_fp16";
        case Blocking::kBlockedFp32Accumulate:
            return "blocked_fp32_accumulate";
        default:
            return "unblocked";
    }
}

std::string matrix_normalization_name(MatrixNormalization normalization) {
    return normalization == MatrixNormalization::kPowerOfTwoMaxAbs ? "power2_max_abs" : "none";
}

int blocks_for(unsigned long long count) {
    const unsigned long long blocks = (count + kThreads - 1) / kThreads;
    return static_cast<int>(std::max<unsigned long long>(1, std::min<unsigned long long>(blocks, kMaxScanBlocks)));
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

struct MatrixMarketHeader {
    int rows = 0;
    int columns = 0;
    unsigned long long entries = 0;
    std::string field;
    std::string symmetry;
};

MatrixMarketHeader read_matrix_market_header(std::istream& source, const std::filesystem::path& path) {
    std::string line;
    if (!std::getline(source, line)) {
        throw std::runtime_error("empty Matrix Market file: " + path.string());
    }
    std::istringstream header(line);
    std::string banner;
    std::string object;
    std::string storage;
    MatrixMarketHeader result;
    if (!(header >> banner >> object >> storage >> result.field >> result.symmetry) || banner != "%%MatrixMarket" ||
        lowercase(object) != "matrix" || lowercase(storage) != "coordinate") {
        throw std::runtime_error("expected a Matrix Market coordinate matrix: " + path.string());
    }
    result.field = lowercase(result.field);
    result.symmetry = lowercase(result.symmetry);
    if (result.field != "real" && result.field != "integer" && result.field != "pattern") {
        throw std::runtime_error("Matrix Market field must be real, integer, or pattern: " + path.string());
    }
    if (result.symmetry != "general" && result.symmetry != "symmetric" && result.symmetry != "hermitian" &&
        result.symmetry != "skew-symmetric") {
        throw std::runtime_error("unsupported Matrix Market symmetry: " + path.string());
    }

    while (std::getline(source, line)) {
        if (line.empty() || line.front() == '%') {
            continue;
        }
        std::istringstream dimensions(line);
        long long rows = 0;
        long long columns = 0;
        unsigned long long entries = 0;
        if (!(dimensions >> rows >> columns >> entries) || rows <= 0 || columns <= 0 ||
            rows > std::numeric_limits<int>::max() || columns > std::numeric_limits<int>::max()) {
            throw std::runtime_error("invalid Matrix Market dimensions: " + path.string());
        }
        result.rows = static_cast<int>(rows);
        result.columns = static_cast<int>(columns);
        result.entries = entries;
        return result;
    }
    throw std::runtime_error("missing Matrix Market dimensions: " + path.string());
}

int matrix_market_dimension(const std::filesystem::path& path) {
    std::ifstream source(path);
    if (!source) {
        throw std::runtime_error("cannot open Matrix Market file: " + path.string());
    }
    const MatrixMarketHeader header = read_matrix_market_header(source, path);
    if (header.rows != header.columns) {
        throw std::runtime_error("matrix must be square for this LU harness: " + path.string());
    }
    return header.rows;
}

std::vector<float> load_matrix_market(const std::filesystem::path& path, MatrixNormalization normalization) {
    std::ifstream source(path);
    if (!source) {
        throw std::runtime_error("cannot open Matrix Market file: " + path.string());
    }
    const MatrixMarketHeader header = read_matrix_market_header(source, path);
    if (header.rows != header.columns) {
        throw std::runtime_error("matrix must be square for this LU harness: " + path.string());
    }
    const unsigned long long elements = static_cast<unsigned long long>(header.rows) * header.columns;
    if (elements > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Matrix Market matrix is too large to densify: " + path.string());
    }
    std::vector<float> matrix(static_cast<std::size_t>(elements), 0.0f);
    for (unsigned long long entry = 0; entry < header.entries; ++entry) {
        long long row = 0;
        long long column = 0;
        double value = 1.0;
        if (!(source >> row >> column) || (header.field != "pattern" && !(source >> value))) {
            throw std::runtime_error("truncated Matrix Market entry list: " + path.string());
        }
        if (row < 1 || row > header.rows || column < 1 || column > header.columns || !std::isfinite(value)) {
            throw std::runtime_error("invalid Matrix Market entry: " + path.string());
        }
        const std::size_t index = static_cast<std::size_t>(row - 1) * header.columns + (column - 1);
        matrix[index] += static_cast<float>(value);
        if (header.symmetry != "general" && row != column) {
            const std::size_t transpose_index = static_cast<std::size_t>(column - 1) * header.columns + (row - 1);
            matrix[transpose_index] += static_cast<float>(header.symmetry == "skew-symmetric" ? -value : value);
        }
    }

    float maximum = 0.0f;
    for (float value : matrix) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Matrix Market conversion overflowed FP32: " + path.string());
        }
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum == 0.0f) {
        throw std::runtime_error("Matrix Market matrix is identically zero: " + path.string());
    }
    if (normalization == MatrixNormalization::kPowerOfTwoMaxAbs) {
        const int exponent = static_cast<int>(std::ceil(std::log2(maximum)));
        for (float& value : matrix) {
            value = std::ldexp(value, -exponent);
        }
    }
    return matrix;
}

int exponent_for(float maximum, float threshold) {
    if (!std::isfinite(maximum) || maximum <= threshold) {
        return 0;
    }
    const int exponent = static_cast<int>(std::ceil(std::log2(maximum / threshold)));
    return std::max(0, exponent);
}

std::string classify(bool proxy_active, bool oracle_active) {
    if (proxy_active && oracle_active) {
        return "TP";
    }
    if (proxy_active && !oracle_active) {
        return "FP";
    }
    if (!proxy_active && oracle_active) {
        return "FN";
    }
    return "TN";
}

std::vector<float> make_lu_input(const Options& options, std::uint64_t seed) {
    if (!options.matrix_market.empty()) {
        return load_matrix_market(options.matrix_market, options.matrix_normalization);
    }

    const int n = options.n;
    std::vector<float> matrix(static_cast<std::size_t>(n) * n, 0.0f);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    if (options.family == Family::kNearSingular) {
        const double sigma = std::ldexp(1.0, -options.near_singular_exponent);
        std::vector<double> singular_vector(n);
        double squared_norm = 0.0;
        for (int row = 0; row < n; ++row) {
            singular_vector[row] = static_cast<double>(uniform(generator));
            squared_norm += singular_vector[row] * singular_vector[row];
        }
        if (squared_norm == 0.0) {
            throw std::runtime_error("near_singular generator produced a zero singular vector");
        }
        const double inverse_norm = 1.0 / std::sqrt(squared_norm);
        for (double& value : singular_vector) {
            value *= inverse_norm;
        }
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < n; ++column) {
                const double value = (row == column ? 1.0 : 0.0) -
                    (1.0 - sigma) * singular_vector[row] * singular_vector[column];
                matrix[static_cast<std::size_t>(row) * n + column] = static_cast<float>(value);
            }
        }
        return matrix;
    }

    if (options.family == Family::kWilkinson) {
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < n; ++column) {
                float value = 0.0f;
                if (column == n - 1 || row == column) {
                    value = 1.0f;
                } else if (row > column) {
                    value = -1.0f;
                }
                matrix[static_cast<std::size_t>(row) * n + column] = value;
            }
        }
        return matrix;
    }

    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            float value = 0.0f;
            if (options.family == Family::kRandom) {
                value = 0.01f * uniform(generator);
                if (row == column) {
                    value += 1.0f;
                }
            } else if (options.family == Family::kGraded) {
                const float position = n > 1 ? static_cast<float>(column) / static_cast<float>(n - 1) : 0.0f;
                const float column_scale = std::exp2(-4.0f + 8.0f * position);
                value = 0.0025f * uniform(generator) * column_scale;
                if (row == column) {
                    value += column_scale;
                }
            } else if (options.family == Family::kGradedWide) {
                const float position = n > 1 ? static_cast<float>(column) / static_cast<float>(n - 1) : 0.0f;
                const float column_scale = std::exp2(kGradedWideLog2Min + kGradedWideLog2Span * position);
                value = 0.25f * uniform(generator) * column_scale;
                if (row == column) {
                    value += column_scale;
                }
            } else if (options.family == Family::kIidRandom) {
                value = uniform(generator);  // plain i.i.d., no diagonal boost
            } else {
                value = row == column ? 1.0f : kHalfMinNormal;
            }
            matrix[static_cast<std::size_t>(row) * n + column] = value;
        }
    }
    return matrix;
}

DiagnosticValue read_panel_diagnostic(
    const DeviceBuffer<__half>& matrix,
    int n,
    int panel_start,
    int panel_end,
    DeviceBuffer<float>& d_maximum,
    DeviceBuffer<unsigned long long>& d_nonfinite,
    GpuTimer& timer) {
    DiagnosticValue result;
    result.elapsed_ms = timer.measure([&] {
        panel_diagonal_max_kernel<<<1, kThreads>>>(
            matrix.get(), n, panel_start, panel_end, d_maximum.get(), d_nonfinite.get());
        CUDA_CHECK(cudaGetLastError());
    });
    CUDA_CHECK(cudaMemcpy(&result.maximum, d_maximum.get(), sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.nonfinite, d_nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return result;
}

DiagnosticValue read_trailing_diagnostic(
    const DeviceBuffer<__half>& matrix,
    int n,
    int start,
    DeviceBuffer<unsigned int>& d_maximum_bits,
    DeviceBuffer<unsigned long long>& d_nonfinite,
    GpuTimer& timer) {
    DiagnosticValue result;
    if (start >= n) {
        return result;
    }

    CUDA_CHECK(cudaMemset(d_maximum_bits.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_nonfinite.get(), 0, sizeof(unsigned long long)));
    const unsigned long long side = static_cast<unsigned long long>(n - start);
    result.elapsed_ms = timer.measure([&] {
        trailing_max_kernel<<<blocks_for(side * side), kThreads>>>(
            matrix.get(), n, start, d_maximum_bits.get(), d_nonfinite.get());
        CUDA_CHECK(cudaGetLastError());
    });

    unsigned int maximum_bits = 0;
    CUDA_CHECK(cudaMemcpy(&maximum_bits, d_maximum_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.nonfinite, d_nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::memcpy(&result.maximum, &maximum_bits, sizeof(float));
    return result;
}

// Same reduction pattern as read_trailing_diagnostic, restricted to a single
// column. Used only by ScalingPolicy::kHybrid + Probe::kLastColumn.
DiagnosticValue read_column_segment_diagnostic(
    const DeviceBuffer<__half>& matrix,
    int n,
    int column,
    int row_start,
    DeviceBuffer<unsigned int>& d_maximum_bits,
    DeviceBuffer<unsigned long long>& d_nonfinite,
    GpuTimer& timer) {
    DiagnosticValue result;
    if (row_start >= n) {
        return result;
    }

    CUDA_CHECK(cudaMemset(d_maximum_bits.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_nonfinite.get(), 0, sizeof(unsigned long long)));
    const unsigned long long length = static_cast<unsigned long long>(n - row_start);
    result.elapsed_ms = timer.measure([&] {
        column_segment_max_kernel<<<blocks_for(length), kThreads>>>(
            matrix.get(), n, column, row_start, d_maximum_bits.get(), d_nonfinite.get());
        CUDA_CHECK(cudaGetLastError());
    });

    unsigned int maximum_bits = 0;
    CUDA_CHECK(cudaMemcpy(&maximum_bits, d_maximum_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.nonfinite, d_nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::memcpy(&result.maximum, &maximum_bits, sizeof(float));
    return result;
}

// Host wrapper for Probe::kPivotRow: max over the U12 block (the panel's own
// rows, every future column). Coalesced in this row-major layout -- see
// row_block_max_kernel's comment.
DiagnosticValue read_row_block_diagnostic(
    const DeviceBuffer<__half>& matrix,
    int n,
    int row_start,
    int row_end,
    int col_start,
    DeviceBuffer<unsigned int>& d_maximum_bits,
    DeviceBuffer<unsigned long long>& d_nonfinite,
    GpuTimer& timer) {
    DiagnosticValue result;
    if (col_start >= n || row_end <= row_start) {
        return result;
    }
    CUDA_CHECK(cudaMemset(d_maximum_bits.get(), 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_nonfinite.get(), 0, sizeof(unsigned long long)));
    const unsigned long long count =
        static_cast<unsigned long long>(row_end - row_start) * static_cast<unsigned long long>(n - col_start);
    result.elapsed_ms = timer.measure([&] {
        row_block_max_kernel<<<blocks_for(count), kThreads>>>(
            matrix.get(), n, row_start, row_end, col_start, d_maximum_bits.get(), d_nonfinite.get());
        CUDA_CHECK(cudaGetLastError());
    });
    unsigned int maximum_bits = 0;
    CUDA_CHECK(cudaMemcpy(&maximum_bits, d_maximum_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.nonfinite, d_nonfinite.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::memcpy(&result.maximum, &maximum_bits, sizeof(float));
    return result;
}

// One-time (per sample) initialization of the certified per-column bound.
void initialize_column_bound(const DeviceBuffer<__half>& matrix, int n, DeviceBuffer<float>& bound) {
    column_bound_init_kernel<<<blocks_for(static_cast<unsigned long long>(n)), kThreads>>>(matrix.get(), n, bound.get());
    CUDA_CHECK(cudaGetLastError());
}

// Reads back just the max of bound[col_start, n) as a float, for logging.
float read_bound_max(DeviceBuffer<float>& bound, int col_start, int n, DeviceBuffer<unsigned int>& scratch) {
    if (col_start >= n) {
        return 0.0f;
    }
    CUDA_CHECK(cudaMemset(scratch.get(), 0, sizeof(unsigned int)));
    array_segment_max_kernel<<<blocks_for(static_cast<unsigned long long>(n - col_start)), kThreads>>>(
        bound.get(), col_start, n, scratch.get());
    CUDA_CHECK(cudaGetLastError());
    unsigned int bits = 0;
    CUDA_CHECK(cudaMemcpy(&bits, scratch.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

// The kCertified panel step: accumulate the certified bound from this panel's
// own U12 block, decide a PER-COLUMN exponent from it, and actuate per column
// (so a column that doesn't need correction never gets one, regardless of
// whether risk elsewhere is concentrated in one column or spread over many).
struct CertifiedStepResult {
    float bound_max_before = 0.0f;
    int applied_max_exponent = 0;
    unsigned long long scaled_columns = 0;
    ScaleStats scale_stats;
    float accumulate_ms = 0.0f;
    float decide_ms = 0.0f;
    float scale_ms = 0.0f;
    float lambda_max = 0.0f;
    float lambda_min = 0.0f;
};

CertifiedStepResult run_certified_step(
    const Options& options,
    DeviceBuffer<__half>& d_matrix,
    DeviceBuffer<int>& d_exponents,
    DeviceBuffer<float>& d_bound,
    DeviceBuffer<int>& d_column_exponents,
    DeviceBuffer<unsigned int>& d_scratch_uint,
    DeviceBuffer<unsigned long long>& d_scale_stats,
    DeviceBuffer<unsigned int>& d_fused_max_bits,
    DeviceBuffer<unsigned long long>& d_lambda_nonfinite,
    int panel_start,
    int panel_end,
    GpuTimer& timer) {
    CertifiedStepResult result;
    if (panel_end >= options.n) {
        return result;
    }
    const int width = options.n - panel_end;

    if (options.bound_source == BoundSource::kFused) {
        // The exact value is already sitting in d_fused_max_bits, written by
        // this panel's final trailing update (see the caller's reset/pass
        // logic) -- reanchor, don't accumulate.
        result.accumulate_ms = timer.measure([&] {
            reanchor_column_bound_kernel<<<blocks_for(static_cast<unsigned long long>(width)), kThreads>>>(
                d_bound.get(), d_fused_max_bits.get(), panel_end, options.n);
            CUDA_CHECK(cudaGetLastError());
        });
    } else {
        result.accumulate_ms = timer.measure([&] {
            column_bound_accumulate_kernel<<<blocks_for(static_cast<unsigned long long>(width)), kThreads>>>(
                d_matrix.get(), options.n, panel_start, panel_end, panel_end, options.n, d_bound.get());
            CUDA_CHECK(cudaGetLastError());
        });
    }

    // Diagnostic only: does lambda_p (this panel's own multiplier magnitudes)
    // actually approach 1, as predicted for partial pivoting on typical
    // (non-adversarial) columns? Cheap: panel_width small reductions, each
    // O(n - p), reusing the existing single-column probe kernel. Off by
    // default (--measure-lambda on) so it never perturbs the certified
    // policy's own measured cost.
    if (options.measure_lambda) {
        result.lambda_min = std::numeric_limits<float>::infinity();
        for (int p = panel_start; p < panel_end; ++p) {
            if (p + 1 >= options.n) {
                continue;
            }
            const DiagnosticValue column_max = read_column_segment_diagnostic(
                d_matrix, options.n, p, p + 1, d_scratch_uint, d_lambda_nonfinite, timer);
            result.lambda_max = std::max(result.lambda_max, column_max.maximum);
            result.lambda_min = std::min(result.lambda_min, column_max.maximum);
        }
        if (!std::isfinite(result.lambda_min)) {
            result.lambda_min = 0.0f;
        }
    }

    result.bound_max_before = read_bound_max(d_bound, panel_end, options.n, d_scratch_uint);

    result.decide_ms = timer.measure([&] {
        decide_column_exponents_kernel<<<blocks_for(static_cast<unsigned long long>(width)), kThreads>>>(
            d_bound.get(), panel_end, options.n, options.threshold, options.margin_bits, d_column_exponents.get());
        CUDA_CHECK(cudaGetLastError());
    });

    std::vector<int> host_exponents(width);
    CUDA_CHECK(cudaMemcpy(
        host_exponents.data(), d_column_exponents.get(), width * sizeof(int), cudaMemcpyDeviceToHost));
    int max_exponent = 0;
    unsigned long long scaled_columns = 0;
    for (int exponent : host_exponents) {
        max_exponent = std::max(max_exponent, exponent);
        if (exponent > 0) {
            ++scaled_columns;
        }
    }
    result.applied_max_exponent = max_exponent;
    result.scaled_columns = scaled_columns;

    if (options.scaling && scaled_columns > 0) {
        CUDA_CHECK(cudaMemset(d_scale_stats.get(), 0, 3 * sizeof(unsigned long long)));
        const unsigned long long elements = static_cast<unsigned long long>(options.n) * width;
        result.scale_ms = timer.measure([&] {
            scale_columns_per_column_kernel<<<blocks_for(elements), kThreads>>>(
                d_matrix.get(),
                options.n,
                panel_end,
                options.n,
                d_column_exponents.get(),
                d_bound.get(),
                d_scale_stats.get(),
                d_scale_stats.get() + 1,
                d_scale_stats.get() + 2);
            add_per_column_exponent_kernel<<<blocks_for(static_cast<unsigned long long>(width)), kThreads>>>(
                d_exponents.get(), panel_end, options.n, d_column_exponents.get());
            CUDA_CHECK(cudaGetLastError());
        });
        unsigned long long host_stats[3] = {};
        CUDA_CHECK(cudaMemcpy(host_stats, d_scale_stats.get(), sizeof(host_stats), cudaMemcpyDeviceToHost));
        result.scale_stats.subnormal_outputs = host_stats[0];
        result.scale_stats.flushed_to_zero = host_stats[1];
        result.scale_stats.nonfinite_values = host_stats[2];
    }
    return result;
}

struct PredictiveGuardResult {
    int applied_max_exponent = 0;
    unsigned long long scaled_columns = 0;
    ScaleStats scale_stats;
};

void launch_predictive_guard_device(
    const Options& options,
    DeviceBuffer<__half>& d_matrix,
    DeviceBuffer<int>& d_exponents,
    DeviceBuffer<float>& d_bound,
    DeviceBuffer<unsigned long long>& d_guard_stats,
    int source_row,
    bool add_source_row,
    bool inflate_before_add,
    int col_start,
    int col_end,
    float threshold) {
    if (col_start >= col_end) {
        return;
    }
    const dim3 threads(32, 8);
    const dim3 blocks((col_end - col_start + threads.x - 1) / threads.x);
    predictive_guard_columns_kernel<<<blocks, threads>>>(
        d_matrix.get(),
        options.n,
        source_row,
        add_source_row,
        inflate_before_add,
        col_start,
        col_end,
        threshold,
        d_bound.get(),
        d_exponents.get(),
        d_guard_stats.get());
    CUDA_CHECK(cudaGetLastError());
}

ScaleStats scale_future_columns(
    DeviceBuffer<__half>& matrix,
    DeviceBuffer<int>& exponents,
    int n,
    int first_column,
    int exponent,
    DeviceBuffer<unsigned long long>& d_scale_stats,
    GpuTimer& timer,
    float* elapsed_ms) {
    ScaleStats result;
    *elapsed_ms = 0.0f;
    if (exponent <= 0 || first_column >= n) {
        return result;
    }

    CUDA_CHECK(cudaMemset(d_scale_stats.get(), 0, 3 * sizeof(unsigned long long)));
    const unsigned long long elements = static_cast<unsigned long long>(n) * (n - first_column);
    *elapsed_ms = timer.measure([&] {
        scale_future_columns_kernel<<<blocks_for(elements), kThreads>>>(
            matrix.get(),
            n,
            first_column,
            exponent,
            d_scale_stats.get(),
            d_scale_stats.get() + 1,
            d_scale_stats.get() + 2);
        add_column_exponent_kernel<<<blocks_for(static_cast<unsigned long long>(n - first_column)), kThreads>>>(
            exponents.get(), first_column, n, exponent);
        CUDA_CHECK(cudaGetLastError());
    });
    unsigned long long host_stats[3] = {};
    CUDA_CHECK(cudaMemcpy(host_stats, d_scale_stats.get(), sizeof(host_stats), cudaMemcpyDeviceToHost));
    result.subnormal_outputs = host_stats[0];
    result.flushed_to_zero = host_stats[1];
    result.nonfinite_values = host_stats[2];
    return result;
}

bool run_column_scale_permutation_self_test() {
    constexpr int n = 4;
    std::vector<float> host_float(n * n);
    for (int index = 0; index < n * n; ++index) {
        host_float[index] = static_cast<float>(index + 1);
    }

    DeviceBuffer<float> d_float(host_float.size());
    DeviceBuffer<__half> d_matrix(host_float.size());
    DeviceBuffer<int> d_exponents(n);
    DeviceBuffer<unsigned long long> d_stats(3);
    CUDA_CHECK(cudaMemcpy(d_float.get(), host_float.data(), host_float.size() * sizeof(float), cudaMemcpyHostToDevice));
    convert_to_half_kernel<<<1, kThreads>>>(d_float.get(), d_matrix.get(), host_float.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d_exponents.get(), 0, n * sizeof(int)));

    std::vector<__half> initial(n * n);
    CUDA_CHECK(cudaMemcpy(initial.data(), d_matrix.get(), initial.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    std::vector<int> logical_columns(n);
    std::vector<int> exponents(n, 0);
    std::iota(logical_columns.begin(), logical_columns.end(), 0);

    auto apply_scale = [&](int first_column, int exponent) {
        CUDA_CHECK(cudaMemset(d_stats.get(), 0, 3 * sizeof(unsigned long long)));
        const unsigned long long count = static_cast<unsigned long long>(n) * (n - first_column);
        scale_future_columns_kernel<<<blocks_for(count), kThreads>>>(
            d_matrix.get(), n, first_column, exponent, d_stats.get(), d_stats.get() + 1, d_stats.get() + 2);
        add_column_exponent_kernel<<<1, kThreads>>>(d_exponents.get(), first_column, n, exponent);
        CUDA_CHECK(cudaGetLastError());
        for (int column = first_column; column < n; ++column) {
            exponents[column] += exponent;
        }
    };

    auto apply_swap = [&](int first, int second) {
        swap_columns_kernel<<<1, kThreads>>>(d_matrix.get(), n, first, second);
        swap_exponents_kernel<<<1, 1>>>(d_exponents.get(), first, second);
        CUDA_CHECK(cudaGetLastError());
        std::swap(logical_columns[first], logical_columns[second]);
        std::swap(exponents[first], exponents[second]);
    };

    apply_scale(1, 2);
    apply_swap(1, 2);
    apply_scale(2, 1);
    apply_swap(0, 3);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> observed(n * n);
    std::vector<int> observed_exponents(n);
    CUDA_CHECK(cudaMemcpy(observed.data(), d_matrix.get(), observed.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_exponents.data(), d_exponents.get(), n * sizeof(int), cudaMemcpyDeviceToHost));

    for (int column = 0; column < n; ++column) {
        if (observed_exponents[column] != exponents[column]) {
            return false;
        }
        for (int row = 0; row < n; ++row) {
            const float restored = std::ldexp(__half2float(observed[row * n + column]), exponents[column]);
            const float expected = __half2float(initial[row * n + logical_columns[column]]);
            if (restored != expected) {
                return false;
            }
        }
    }
    return true;
}

ReconstructionResult reconstruct_lu(
    const std::vector<__half>& initial,
    const std::vector<__half>& factorized,
    const std::vector<int>& pivots,
    const std::vector<int>& exponents,
    int n) {
    ReconstructionResult result;
    if (n <= 0) {
        result.kind = "not_run";
        return result;
    }

    const int sample_dimension = n <= 256 ? n : 32;
    result.kind = n <= 256 ? "full_fp64" : "sampled_fp64_32x32";

    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    for (int pivot = 0; pivot < n; ++pivot) {
        if (pivots[pivot] < 0 || pivots[pivot] >= n) {
            result.kind = "invalid_pivots";
            return result;
        }
        std::swap(permutation[pivot], permutation[pivots[pivot]]);
    }

    double max_input = 0.0;
    double max_residual = 0.0;
    for (int row_index = 0; row_index < sample_dimension; ++row_index) {
        const int row = n <= 256 ? row_index : (row_index * (n - 1)) / std::max(1, sample_dimension - 1);
        for (int column_index = 0; column_index < sample_dimension; ++column_index) {
            const int column = n <= 256 ? column_index : (column_index * (n - 1)) / std::max(1, sample_dimension - 1);
            double reconstructed = 0.0;
            const int upper_limit = std::min(row, column);
            for (int inner = 0; inner <= upper_limit; ++inner) {
                const std::size_t row_index_ = static_cast<std::size_t>(row) * n;
                const std::size_t inner_index_ = static_cast<std::size_t>(inner) * n;
                const double lower = row == inner ? 1.0 : static_cast<double>(__half2float(factorized[row_index_ + inner]));
                const double upper = std::ldexp(
                    static_cast<double>(__half2float(factorized[inner_index_ + column])), exponents[column]);
                reconstructed += lower * upper;
            }
            const double permuted_input = static_cast<double>(
                __half2float(initial[static_cast<std::size_t>(permutation[row]) * n + column]));
            if (!std::isfinite(reconstructed) || !std::isfinite(permuted_input)) {
                result.kind = "nonfinite_reconstruction";
                return result;
            }
            max_input = std::max(max_input, std::abs(permuted_input));
            max_residual = std::max(max_residual, std::abs(reconstructed - permuted_input));
        }
    }
    result.relative_residual = max_residual / std::max(max_input, std::numeric_limits<double>::min());
    return result;
}

void accumulate_classification(const PanelRecord& record, RunSummary* summary) {
    ++summary->panels_observed;
    summary->factorization_ms += record.factorization_ms;
    summary->proxy_scan_ms += record.proxy_scan_ms;
    summary->oracle_scan_ms += record.oracle_scan_ms;
    summary->scale_ms += record.scale_ms;

    // The final panel has no future trailing block, so it has no scaling
    // decision and must not bias proxy/oracle classification statistics.
    if (record.classification == "NA_FINAL" || record.classification == "NONFINITE") {
        return;
    }
    if (record.proxy_exponent > 0) {
        ++summary->proxy_activations;
    }
    if (record.oracle_exponent > 0) {
        ++summary->oracle_activations;
    }
    if (record.classification == "TP") {
        ++summary->true_positive;
    } else if (record.classification == "FP") {
        ++summary->false_positive;
    } else if (record.classification == "FN") {
        ++summary->false_negative;
    } else if (record.classification == "TN") {
        ++summary->true_negative;
    }
    if (record.under_scale_bits != 0 || record.over_scale_bits != 0) {
        ++summary->exponent_mismatch_panels;
    }
    summary->total_under_scale_bits += static_cast<unsigned long long>(record.under_scale_bits);
    summary->total_over_scale_bits += static_cast<unsigned long long>(record.over_scale_bits);
    summary->total_scaled_elements += record.scaled_elements;
    summary->scale_stats.subnormal_outputs += record.scale_stats.subnormal_outputs;
    summary->scale_stats.flushed_to_zero += record.scale_stats.flushed_to_zero;
    summary->scale_stats.nonfinite_values += record.scale_stats.nonfinite_values;
}

std::pair<PanelRecord, bool> observe_and_maybe_scale(
    const Options& options,
    int sample,
    int panel,
    int panel_start,
    int panel_end,
    DeviceBuffer<__half>& d_matrix,
    DeviceBuffer<int>& d_exponents,
    DeviceBuffer<float>& d_panel_maximum,
    DeviceBuffer<unsigned int>& d_trailing_maximum_bits,
    DeviceBuffer<unsigned int>& d_probe_maximum_bits,
    DeviceBuffer<unsigned long long>& d_nonfinite,
    DeviceBuffer<unsigned long long>& d_scale_stats,
    GpuTimer& timer,
    double& ratio_estimate,
    float factorization_ms) {
    PanelRecord record;
    record.sample = sample;
    record.panel = panel;
    record.panel_start = panel_start;
    record.panel_end = panel_end;
    record.factorization_ms = factorization_ms;

    CUDA_CHECK(cudaMemset(d_nonfinite.get(), 0, sizeof(unsigned long long)));
    const DiagnosticValue proxy = read_panel_diagnostic(
        d_matrix, options.n, panel_start, panel_end, d_panel_maximum, d_nonfinite, timer);
    const DiagnosticValue oracle = read_trailing_diagnostic(
        d_matrix, options.n, panel_end, d_trailing_maximum_bits, d_nonfinite, timer);

    record.dmax_proxy = proxy.maximum;
    record.trailing_max_oracle = oracle.maximum;
    record.proxy_nonfinite = proxy.nonfinite;
    record.oracle_nonfinite = oracle.nonfinite;
    record.proxy_scan_ms = proxy.elapsed_ms;
    record.oracle_scan_ms = oracle.elapsed_ms;
    record.ratio_oracle_over_proxy = record.dmax_proxy > 0.0f
        ? record.trailing_max_oracle / record.dmax_proxy
        : (record.trailing_max_oracle > 0.0f ? std::numeric_limits<float>::infinity() : 1.0f);

    const bool nonfinite_state = record.proxy_nonfinite != 0 || record.oracle_nonfinite != 0;
    if (nonfinite_state) {
        record.proxy_exponent = -1;
        record.oracle_exponent = -1;
        record.classification = "NONFINITE";
        return {record, false};
    }

    if (panel_end >= options.n) {
        record.classification = "NA_FINAL";
        return {record, true};
    }

    record.proxy_exponent = exponent_for(record.dmax_proxy, options.threshold);
    record.oracle_exponent = exponent_for(record.trailing_max_oracle, options.threshold);
    record.classification = classify(record.proxy_exponent > 0, record.oracle_exponent > 0);
    record.under_scale_bits = std::max(0, record.oracle_exponent - record.proxy_exponent);
    record.over_scale_bits = std::max(0, record.proxy_exponent - record.oracle_exponent);

    // --- ScalingPolicy::kHybrid: cheap structural probe + bounded
    // oracle-calibrated ratio. classification/under_scale_bits/over_scale_bits
    // above are left untouched (still pure panel-diagonal-vs-oracle, matching
    // the manuscript rule) so old interpretation code keeps working; kHybrid
    // only changes what gets *applied* below.
    float raw_effective = record.dmax_proxy;
    if (options.scaling_policy == ScalingPolicy::kHybrid && options.probe == Probe::kLastColumn &&
        panel_end < options.n) {
        const DiagnosticValue probe = read_column_segment_diagnostic(
            d_matrix, options.n, options.n - 1, panel_end, d_probe_maximum_bits, d_nonfinite, timer);
        record.probe_value = probe.maximum;
        record.probe_scan_ms = probe.elapsed_ms;
        raw_effective = std::max(raw_effective, record.probe_value);
    } else if (options.scaling_policy == ScalingPolicy::kHybrid && options.probe == Probe::kPivotRow &&
               panel_end < options.n) {
        const DiagnosticValue probe = read_row_block_diagnostic(
            d_matrix, options.n, panel_start, panel_end, panel_end, d_probe_maximum_bits, d_nonfinite, timer);
        record.probe_value = probe.maximum;
        record.probe_scan_ms = probe.elapsed_ms;
        raw_effective = std::max(raw_effective, record.probe_value);
    }

    const bool is_calibration_panel = options.scaling_policy == ScalingPolicy::kHybrid &&
        (panel < options.adaptive_burnin ||
         (options.adaptive_resync_every > 0 && panel % options.adaptive_resync_every == 0));
    record.is_calibration_panel = is_calibration_panel ? 1 : 0;

    if (is_calibration_panel && raw_effective > 0.0f) {
        // Deliberately double (FP64): the calibration ratio is scalar
        // bookkeeping, not a matrix entry, so keeping it above FP16 (like the
        // FP64 reconstruction check elsewhere in this file) stops the
        // adaptive correction itself from compounding rounding error.
        const double observed_ratio =
            static_cast<double>(record.trailing_max_oracle) / static_cast<double>(raw_effective);
        ratio_estimate = std::max(ratio_estimate, observed_ratio);
    }
    record.ratio_estimate = ratio_estimate;
    record.effective_value = options.scaling_policy == ScalingPolicy::kHybrid
        ? static_cast<float>(static_cast<double>(raw_effective) * ratio_estimate)
        : raw_effective;

    // base_exponent: what a plain panel-diagonal decision alone would apply,
    // to the whole future block, exactly like kProxy/the manuscript rule.
    // targeted_extra_exponent: any ADDITIONAL exponent that the (possibly
    // probe/ratio augmented) decision calls for beyond the plain-diagonal
    // amount. With --scoped-scaling on, this extra is applied ONLY to the
    // probed column, on top of the base -- so a benign matrix (where the
    // diagonal alone already explains the decision) gets exactly the
    // original full-block behavior, and only a matrix where the *probe*
    // reveals something the diagonal could not see (e.g. Wilkinson) gets a
    // surgical correction instead of a blanket one. This is the fix for the
    // regression found empirically: naively scoping the *entire* decision to
    // one column starves every other future column of the correction its
    // own (legitimate) diagonal reading was asking for.
    int base_exponent = 0;
    int targeted_extra_exponent = 0;
    if (options.scaling &&
        (options.scaling_policy == ScalingPolicy::kCertified ||
         options.scaling_policy == ScalingPolicy::kPredictive)) {
        // Per-column policies are actuated by their persistent bound state;
        // this function only supplies the diagnostic fields above.
    } else if (options.scaling) {
        if (options.scaling_policy == ScalingPolicy::kOracle) {
            base_exponent = record.oracle_exponent;
        } else {
            base_exponent = exponent_for(record.dmax_proxy, options.threshold);
            if (base_exponent > 0) {
                base_exponent += options.margin_bits;
            }
            if (options.scaling_policy == ScalingPolicy::kHybrid) {
                int decided_exponent = exponent_for(record.effective_value, options.threshold);
                if (decided_exponent > 0) {
                    decided_exponent += options.margin_bits;
                }
                if (options.scoped_scaling && options.probe == Probe::kLastColumn) {
                    targeted_extra_exponent = std::max(0, decided_exponent - base_exponent);
                } else {
                    base_exponent = decided_exponent;  // unscoped: one exponent, whole future block
                }
            }
        }
        record.applied_exponent = base_exponent + targeted_extra_exponent;
    }
    record.base_exponent = base_exponent;
    if (base_exponent > 0 && panel_end < options.n) {
        record.scaled_elements = static_cast<unsigned long long>(options.n) * (options.n - panel_end);
        record.scale_stats = scale_future_columns(
            d_matrix,
            d_exponents,
            options.n,
            panel_end,
            base_exponent,
            d_scale_stats,
            timer,
            &record.scale_ms);
        if (record.scale_stats.nonfinite_values != 0) {
            return {record, false};
        }
    }
    if (targeted_extra_exponent > 0 && options.n - 1 > panel_end) {
        float extra_ms = 0.0f;
        const ScaleStats extra_stats = scale_future_columns(
            d_matrix,
            d_exponents,
            options.n,
            options.n - 1,
            targeted_extra_exponent,
            d_scale_stats,
            timer,
            &extra_ms);
        record.scaled_elements += static_cast<unsigned long long>(options.n);
        record.scale_ms += extra_ms;
        record.scale_stats.subnormal_outputs += extra_stats.subnormal_outputs;
        record.scale_stats.flushed_to_zero += extra_stats.flushed_to_zero;
        record.scale_stats.nonfinite_values += extra_stats.nonfinite_values;
        if (extra_stats.nonfinite_values != 0) {
            return {record, false};
        }
    }
    return {record, true};
}

bool snapshot_requested(const Options& options) {
    return options.snapshot_original || options.snapshot_final || !options.snapshot_panels.empty();
}

bool snapshot_panel_requested(const Options& options, int panel) {
    return std::find(options.snapshot_panels.begin(), options.snapshot_panels.end(), panel) != options.snapshot_panels.end();
}

void write_snapshot_manifest(const Options& options, const std::filesystem::path& output) {
    if (!snapshot_requested(options)) {
        return;
    }
    const std::filesystem::path directory = output / "snapshots";
    std::filesystem::create_directories(directory);
    std::ofstream file(directory / "snapshot_manifest.json");
    file << "{\n";
    file << "  \"storage\": \"IEEE binary16 little-endian row-major\",\n";
    file << "  \"n\": " << options.n << ",\n";
    file << "  \"panel_capture_point\": \"after policy scaling at the completed panel boundary\",\n";
    file << "  \"original_requested\": " << (options.snapshot_original ? "true" : "false") << ",\n";
    file << "  \"final_requested\": " << (options.snapshot_final ? "true" : "false") << ",\n";
    file << "  \"panel_indices\": [";
    for (std::size_t index = 0; index < options.snapshot_panels.size(); ++index) {
        if (index != 0) {
            file << ", ";
        }
        file << options.snapshot_panels[index];
    }
    file << "]\n";
    file << "}\n";
}

void save_matrix_snapshot(
    const DeviceBuffer<__half>& matrix,
    int n,
    const std::filesystem::path& output,
    int sample,
    const std::string& label) {
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    std::vector<__half> host_matrix(elements);
    CUDA_CHECK(cudaMemcpy(host_matrix.data(), matrix.get(), elements * sizeof(__half), cudaMemcpyDeviceToHost));
    const std::filesystem::path path = output / "snapshots" /
        ("sample_" + std::to_string(sample) + "_" + label + ".fp16");
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot write snapshot: " + path.string());
    }
    file.write(reinterpret_cast<const char*>(host_matrix.data()), static_cast<std::streamsize>(elements * sizeof(__half)));
    if (!file) {
        throw std::runtime_error("failed while writing snapshot: " + path.string());
    }
}

RunSummary run_lu_sample(const Options& options, int sample, std::vector<PanelRecord>* records) {
    const std::vector<float> host_float = make_lu_input(options, options.seed + static_cast<std::uint64_t>(sample));
    const std::size_t elements = host_float.size();

    DeviceBuffer<__half> d_matrix(elements);
    // The FP32 staging buffer is only needed for the initial conversion. Scope it
    // so it is freed before d_initial allocates; otherwise n=65536+ would need
    // d_float + d_matrix + d_initial = 34+ GB on a 32 GB V100.
    {
        DeviceBuffer<float> d_float(elements);
        CUDA_CHECK(cudaMemcpy(d_float.get(), host_float.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
        convert_to_half_kernel<<<blocks_for(elements), kThreads>>>(d_float.get(), d_matrix.get(), elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    DeviceBuffer<__half> d_initial(elements);
    CUDA_CHECK(cudaMemcpy(d_initial.get(), d_matrix.get(), elements * sizeof(__half), cudaMemcpyDeviceToDevice));
    DeviceBuffer<int> d_pivots(options.n);
    DeviceBuffer<int> d_exponents(options.n);
    DeviceBuffer<float> d_panel_maximum(1);
    DeviceBuffer<unsigned int> d_trailing_maximum_bits(1);
    DeviceBuffer<unsigned int> d_probe_maximum_bits(1);
    DeviceBuffer<unsigned long long> d_nonfinite(1);
    DeviceBuffer<unsigned long long> d_scale_stats(5);
    DeviceBuffer<float> d_bound(options.n);
    DeviceBuffer<int> d_column_exponents(options.n);
    DeviceBuffer<unsigned int> d_scratch_uint(1);
    DeviceBuffer<unsigned int> d_fused_max_bits(options.n);
    DeviceBuffer<unsigned long long> d_lambda_nonfinite(1);
    GpuTimer timer;
    GpuTimer guard_timer;

    RunSummary summary;
    summary.sample = sample;
    summary.column_scale_permutation_self_test_passed = run_column_scale_permutation_self_test();
    summary.termination = "completed";

    if (options.snapshot_original) {
        save_matrix_snapshot(d_initial, options.n, options.output, sample, "original");
    }
    CUDA_CHECK(cudaMemset(d_pivots.get(), 0, options.n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_exponents.get(), 0, options.n * sizeof(int)));
    float preflight_ms = 0.0f;
    PredictiveGuardResult preflight;
    if (options.scaling_policy == ScalingPolicy::kCertified ||
        options.scaling_policy == ScalingPolicy::kPredictive) {
        preflight_ms += guard_timer.measure([&] {
            initialize_column_bound(d_matrix, options.n, d_bound);
        });
    }
    if (options.scaling_policy == ScalingPolicy::kPredictive) {
        CUDA_CHECK(cudaMemset(d_scale_stats.get(), 0, 5 * sizeof(unsigned long long)));
        preflight_ms += guard_timer.measure([&] {
            launch_predictive_guard_device(
                options, d_matrix, d_exponents, d_bound, d_scale_stats,
                0, false, false, 0, options.n, options.rank_safe_threshold);
        });
        unsigned long long host_guard_stats[5] = {};
        CUDA_CHECK(cudaMemcpy(
            host_guard_stats, d_scale_stats.get(), sizeof(host_guard_stats), cudaMemcpyDeviceToHost));
        preflight.scale_stats.subnormal_outputs = host_guard_stats[0];
        preflight.scale_stats.flushed_to_zero = host_guard_stats[1];
        preflight.scale_stats.nonfinite_values = host_guard_stats[2];
        preflight.scaled_columns = host_guard_stats[3];
        preflight.applied_max_exponent = static_cast<int>(host_guard_stats[4]);
    }

    if (preflight.scale_stats.nonfinite_values != 0) {
        summary.termination = "nonfinite_during_preflight_scaling";
    }

    double ratio_estimate = 1.0;  // FP64 running calibration state; see kHybrid notes above.
    bool keep_running = preflight.scale_stats.nonfinite_values == 0;
    for (int panel_start = 0, panel = 0; panel_start < options.n && keep_running;
         panel_start += options.panel_width, ++panel) {
        const int panel_end = std::min(options.n, panel_start + options.panel_width);
        float predictive_scale_ms = 0.0f;
        unsigned long long predictive_scaled_elements = 0;
        ScaleStats predictive_scale_stats;
        int predictive_max_exponent = 0;
        bool predictive_failed = false;
        if (options.scaling_policy == ScalingPolicy::kPredictive) {
            CUDA_CHECK(cudaMemset(d_scale_stats.get(), 0, 5 * sizeof(unsigned long long)));
        }

        const float total_factorization_ms = timer.measure([&] {
            for (int pivot = panel_start; pivot < panel_end; ++pivot) {
                choose_partial_pivot_kernel<<<1, kThreads>>>(d_matrix.get(), options.n, pivot, d_pivots.get());
                swap_rows_kernel<<<std::max(1, (options.n + kThreads - 1) / kThreads), kThreads>>>(
                    d_matrix.get(), options.n, pivot, d_pivots.get() + pivot);
                divide_pivot_column_kernel<<<std::max(1, (options.n - pivot - 1 + kThreads - 1) / kThreads), kThreads>>>(
                    d_matrix.get(), options.n, pivot);

                if (options.blocking == Blocking::kUnblocked) {
                    // Unchanged from every prior result in this artifact: each
                    // pivot's trailing update touches the WHOLE remaining matrix.
                    const int active = options.n - pivot - 1;
                    if (active > 0) {
                        const dim3 threads(16, 16);
                        const dim3 blocks((active + threads.x - 1) / threads.x, (active + threads.y - 1) / threads.y);
                        unsigned int* fused_target = nullptr;
                        if (options.scaling_policy == ScalingPolicy::kCertified &&
                            options.bound_source == BoundSource::kFused && pivot == panel_end - 1) {
                            CUDA_CHECK(cudaMemset(d_fused_max_bits.get(), 0, options.n * sizeof(unsigned int)));
                            fused_target = d_fused_max_bits.get();
                        }
                        update_trailing_matrix_kernel<<<blocks, threads>>>(
                            d_matrix.get(), options.n, pivot, pivot + 1, options.n, pivot + 1, options.n, fused_target);
                    }
                } else {
                    // Phase 1: factor only the panel slab across every lower
                    // row. This produces both the remaining L11 entries and
                    // L21 while leaving A12 untouched until pivoting is final.
                    const int height = options.n - pivot - 1;
                    const int width = panel_end - pivot - 1;
                    if (height > 0 && width > 0) {
                        if (options.scaling_policy == ScalingPolicy::kPredictive) {
                            launch_predictive_guard_device(
                                options, d_matrix, d_exponents, d_bound, d_scale_stats,
                                pivot,
                                true,
                                pivot > panel_start,
                                pivot + 1,
                                panel_end,
                                options.rank_safe_threshold);
                        }
                        const dim3 threads(16, 16);
                        const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
                        update_trailing_matrix_kernel<<<blocks, threads>>>(
                            d_matrix.get(), options.n, pivot, pivot + 1, options.n, pivot + 1, panel_end, nullptr);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
            }

            if (options.blocking != Blocking::kUnblocked && panel_end < options.n) {
                // Phase 2: all panel row swaps are now complete, so solve U12
                // from the final L11 and permuted A12 exactly once.
                const dim3 threads(16, 16);
                const int width = options.n - panel_end;
                if (options.scaling_policy == ScalingPolicy::kPredictive) {
                    for (int pivot = panel_start; pivot < panel_end; ++pivot) {
                        launch_predictive_guard_device(
                            options, d_matrix, d_exponents, d_bound, d_scale_stats,
                            pivot,
                            true,
                            pivot > panel_start,
                            panel_end,
                            options.n,
                            options.rank_safe_threshold);
                        const int rows = panel_end - pivot - 1;
                        if (rows > 0) {
                            const dim3 update_threads(16, 16);
                            const dim3 update_blocks(
                                (width + update_threads.x - 1) / update_threads.x,
                                (rows + update_threads.y - 1) / update_threads.y);
                            update_trailing_matrix_kernel<<<update_blocks, update_threads>>>(
                                d_matrix.get(),
                                options.n,
                                pivot,
                                pivot + 1,
                                panel_end,
                                panel_end,
                                options.n,
                                nullptr);
                        }
                    }
                    launch_predictive_guard_device(
                        options,
                        d_matrix,
                        d_exponents,
                        d_bound,
                        d_scale_stats,
                        0,
                        false,
                        false,
                        panel_end,
                        options.n,
                        options.safe_threshold);
                } else if (options.rankwise_u12) {
                    for (int pivot = panel_start; pivot < panel_end; ++pivot) {
                        const int rows = panel_end - pivot - 1;
                        if (rows > 0) {
                            const dim3 update_threads(16, 16);
                            const dim3 update_blocks(
                                (width + update_threads.x - 1) / update_threads.x,
                                (rows + update_threads.y - 1) / update_threads.y);
                            update_trailing_matrix_kernel<<<update_blocks, update_threads>>>(
                                d_matrix.get(),
                                options.n,
                                pivot,
                                pivot + 1,
                                panel_end,
                                panel_end,
                                options.n,
                                nullptr);
                        }
                    }
                } else {
                    panel_u12_triangular_solve_kernel<<<(width + kThreads - 1) / kThreads, kThreads>>>(
                        d_matrix.get(), options.n, panel_start, panel_end);
                    CUDA_CHECK(cudaGetLastError());
                }

                const dim3 blocks((width + threads.x - 1) / threads.x, (width + threads.y - 1) / threads.y);
                unsigned int* fused_target = nullptr;
                if ((options.scaling_policy == ScalingPolicy::kCertified &&
                     options.bound_source == BoundSource::kFused) ||
                    options.scaling_policy == ScalingPolicy::kPredictive) {
                    // Natural fusion point under real blocking: reset right
                    // before the ONE trailing GEMM-equivalent call for this
                    // panel, so the exact per-column max falls out of it for free.
                    CUDA_CHECK(cudaMemset(d_fused_max_bits.get(), 0, options.n * sizeof(unsigned int)));
                    fused_target = d_fused_max_bits.get();
                }
                blocked_trailing_update_kernel<<<blocks, threads>>>(
                    d_matrix.get(), options.n, panel_start, panel_end,
                    options.blocking == Blocking::kBlockedFp32Accumulate, fused_target);
                CUDA_CHECK(cudaGetLastError());
                if (options.scaling_policy == ScalingPolicy::kPredictive) {
                    reanchor_column_bound_kernel<<<blocks_for(static_cast<unsigned long long>(width)), kThreads>>>(
                        d_bound.get(), d_fused_max_bits.get(), panel_end, options.n);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        });
        const float factorization_ms = total_factorization_ms;
        if (options.scaling_policy == ScalingPolicy::kPredictive) {
            unsigned long long host_guard_stats[5] = {};
            CUDA_CHECK(cudaMemcpy(
                host_guard_stats, d_scale_stats.get(), sizeof(host_guard_stats), cudaMemcpyDeviceToHost));
            predictive_scale_stats.subnormal_outputs = host_guard_stats[0];
            predictive_scale_stats.flushed_to_zero = host_guard_stats[1];
            predictive_scale_stats.nonfinite_values = host_guard_stats[2];
            predictive_scaled_elements = host_guard_stats[3] * static_cast<unsigned long long>(options.n);
            predictive_max_exponent = static_cast<int>(host_guard_stats[4]);
            predictive_failed = host_guard_stats[2] != 0;
        }

        auto [record, panel_is_finite] = observe_and_maybe_scale(
            options,
            sample,
            panel,
            panel_start,
            panel_end,
            d_matrix,
            d_exponents,
            d_panel_maximum,
            d_trailing_maximum_bits,
            d_probe_maximum_bits,
            d_nonfinite,
            d_scale_stats,
            timer,
            ratio_estimate,
            factorization_ms);
        if (options.scaling_policy == ScalingPolicy::kPredictive) {
            if (panel == 0) {
                predictive_scale_ms += preflight_ms;
                predictive_scaled_elements += preflight.scaled_columns * static_cast<unsigned long long>(options.n);
                predictive_max_exponent = std::max(predictive_max_exponent, preflight.applied_max_exponent);
                predictive_scale_stats.subnormal_outputs += preflight.scale_stats.subnormal_outputs;
                predictive_scale_stats.flushed_to_zero += preflight.scale_stats.flushed_to_zero;
                predictive_scale_stats.nonfinite_values += preflight.scale_stats.nonfinite_values;
            }
            record.probe_value = options.safe_threshold;
            record.effective_value = options.safe_threshold;
            record.applied_exponent = predictive_max_exponent;
            record.scaled_elements += predictive_scaled_elements;
            record.scale_ms += predictive_scale_ms;
            record.scale_stats.subnormal_outputs += predictive_scale_stats.subnormal_outputs;
            record.scale_stats.flushed_to_zero += predictive_scale_stats.flushed_to_zero;
            record.scale_stats.nonfinite_values += predictive_scale_stats.nonfinite_values;
            if (predictive_failed || predictive_scale_stats.nonfinite_values != 0) {
                panel_is_finite = false;
                summary.termination = "nonfinite_during_predictive_scaling";
            }
        }
        if (options.scaling_policy == ScalingPolicy::kCertified && panel_is_finite &&
            record.classification != "NA_FINAL") {
            const CertifiedStepResult certified = run_certified_step(
                options, d_matrix, d_exponents, d_bound, d_column_exponents, d_scratch_uint, d_scale_stats,
                d_fused_max_bits, d_lambda_nonfinite, panel_start, panel_end, timer);
            record.probe_value = certified.bound_max_before;
            record.effective_value = certified.bound_max_before;
            record.applied_exponent = certified.applied_max_exponent;
            record.scaled_elements = certified.scaled_columns * static_cast<unsigned long long>(options.n);
            record.scale_ms += certified.accumulate_ms + certified.decide_ms + certified.scale_ms;
            record.scale_stats.subnormal_outputs += certified.scale_stats.subnormal_outputs;
            record.scale_stats.flushed_to_zero += certified.scale_stats.flushed_to_zero;
            record.scale_stats.nonfinite_values += certified.scale_stats.nonfinite_values;
            record.lambda_max = certified.lambda_max;
            record.lambda_min = certified.lambda_min;
            if (certified.scale_stats.nonfinite_values != 0) {
                panel_is_finite = false;
                summary.termination = "nonfinite_during_scaling";
            }
        }
        if (record.classification == "NONFINITE") {
            summary.termination = "nonfinite_before_or_during_scaling";
        } else if (!panel_is_finite) {
            summary.termination = "nonfinite_during_scaling";
        }
        accumulate_classification(record, &summary);
        records->push_back(record);
        if (snapshot_panel_requested(options, panel)) {
            save_matrix_snapshot(d_matrix, options.n, options.output, sample, "panel_" + std::to_string(panel));
        }
        keep_running = panel_is_finite;
    }

    const DiagnosticValue final_state = read_trailing_diagnostic(
        d_matrix, options.n, 0, d_trailing_maximum_bits, d_nonfinite, timer);
    summary.final_nonfinite = final_state.nonfinite;
    if (keep_running && summary.final_nonfinite == 0) {
        std::vector<__half> initial(elements);
        std::vector<__half> factorized(elements);
        std::vector<int> pivots(options.n);
        std::vector<int> exponents(options.n);
        CUDA_CHECK(cudaMemcpy(initial.data(), d_initial.get(), elements * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(factorized.data(), d_matrix.get(), elements * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pivots.data(), d_pivots.get(), options.n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(exponents.data(), d_exponents.get(), options.n * sizeof(int), cudaMemcpyDeviceToHost));
        summary.row_swaps = 0;
        for (int pivot = 0; pivot < options.n; ++pivot) {
            if (pivots[pivot] != pivot) {
                ++summary.row_swaps;
            }
        }
        const ReconstructionResult reconstruction = reconstruct_lu(initial, factorized, pivots, exponents, options.n);
        summary.reconstruction_kind = reconstruction.kind;
        summary.reconstruction_relative_residual = reconstruction.relative_residual;
        summary.completed = true;
    } else {
        summary.reconstruction_kind = "skipped_after_nonfinite";
    }
    if (options.snapshot_final) {
        save_matrix_snapshot(d_matrix, options.n, options.output, sample, "final");
    }
    return summary;
}

RunSummary run_trigger_sample(const Options& options, std::vector<PanelRecord>* records) {
    const unsigned long long elements = static_cast<unsigned long long>(options.n) * options.n;
    DeviceBuffer<__half> d_matrix(static_cast<std::size_t>(elements));
    DeviceBuffer<int> d_exponents(options.n);
    DeviceBuffer<float> d_panel_maximum(1);
    DeviceBuffer<unsigned int> d_trailing_maximum_bits(1);
    DeviceBuffer<unsigned int> d_probe_maximum_bits(1);
    DeviceBuffer<unsigned long long> d_nonfinite(1);
    DeviceBuffer<unsigned long long> d_scale_stats(5);
    DeviceBuffer<float> d_bound(options.n);
    DeviceBuffer<int> d_column_exponents(options.n);
    DeviceBuffer<unsigned int> d_scratch_uint(1);
    DeviceBuffer<unsigned int> d_fused_max_bits(options.n);
    DeviceBuffer<unsigned long long> d_lambda_nonfinite(1);
    GpuTimer timer;

    timer.measure([&] {
        generate_trigger_matrix_kernel<<<blocks_for(elements), kThreads>>>(
            d_matrix.get(), options.n, options.seed, family_code(options.family), options.near_singular_exponent);
        CUDA_CHECK(cudaGetLastError());
    });
    CUDA_CHECK(cudaMemset(d_exponents.get(), 0, options.n * sizeof(int)));
    if (options.scaling_policy == ScalingPolicy::kCertified) {
        initialize_column_bound(d_matrix, options.n, d_bound);
    }

    RunSummary summary;
    summary.sample = 0;
    summary.column_scale_permutation_self_test_passed = run_column_scale_permutation_self_test();
    summary.termination = "trigger_only_completed";
    summary.reconstruction_kind = "not_applicable_trigger_only";

    double ratio_estimate = 1.0;
    const int panel_limit = options.max_panels < 0
        ? (options.n + options.panel_width - 1) / options.panel_width
        : options.max_panels;
    bool keep_running = true;
    for (int panel_start = 0, panel = 0; panel_start < options.n && panel < panel_limit && keep_running;
         panel_start += options.panel_width, ++panel) {
        const int panel_end = std::min(options.n, panel_start + options.panel_width);
        auto [record, panel_is_finite] = observe_and_maybe_scale(
            options,
            0,
            panel,
            panel_start,
            panel_end,
            d_matrix,
            d_exponents,
            d_panel_maximum,
            d_trailing_maximum_bits,
            d_probe_maximum_bits,
            d_nonfinite,
            d_scale_stats,
            timer,
            ratio_estimate,
            0.0f);
        if (options.scaling_policy == ScalingPolicy::kCertified && panel_is_finite &&
            record.classification != "NA_FINAL") {
            const CertifiedStepResult certified = run_certified_step(
                options, d_matrix, d_exponents, d_bound, d_column_exponents, d_scratch_uint, d_scale_stats,
                d_fused_max_bits, d_lambda_nonfinite, panel_start, panel_end, timer);
            record.probe_value = certified.bound_max_before;
            record.effective_value = certified.bound_max_before;
            record.applied_exponent = certified.applied_max_exponent;
            record.scaled_elements = certified.scaled_columns * static_cast<unsigned long long>(options.n);
            record.scale_ms += certified.accumulate_ms + certified.decide_ms + certified.scale_ms;
            record.scale_stats.subnormal_outputs += certified.scale_stats.subnormal_outputs;
            record.scale_stats.flushed_to_zero += certified.scale_stats.flushed_to_zero;
            record.scale_stats.nonfinite_values += certified.scale_stats.nonfinite_values;
            record.lambda_max = certified.lambda_max;
            record.lambda_min = certified.lambda_min;
            if (certified.scale_stats.nonfinite_values != 0) {
                panel_is_finite = false;
            }
        }
        if (record.classification == "NONFINITE") {
            summary.termination = "trigger_only_nonfinite";
        }
        accumulate_classification(record, &summary);
        records->push_back(record);
        keep_running = panel_is_finite;
    }
    const DiagnosticValue final_state = read_trailing_diagnostic(
        d_matrix, options.n, 0, d_trailing_maximum_bits, d_nonfinite, timer);
    summary.final_nonfinite = final_state.nonfinite;
    summary.completed = keep_running && summary.final_nonfinite == 0;
    return summary;
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

void write_metadata(const Options& options, const std::filesystem::path& output) {
    cudaDeviceProp properties{};
    int device = 0;
    int driver_version = 0;
    int runtime_version = 0;
    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    std::ofstream file(output / "metadata.json");
    file << std::setprecision(9);
    file << "{\n";
    file << "  \"schema_version\": 1,\n";
    file << "  \"timestamp_utc\": \"" << utc_timestamp() << "\",\n";
    file << "  \"mode\": \"" << mode_name(options.mode) << "\",\n";
    file << "  \"family\": \""
         << (options.matrix_market.empty() ? family_name(options.family) : "external_matrix_market") << "\",\n";
    if (!options.matrix_market.empty()) {
        file << "  \"input_matrix_market\": \"" << options.matrix_market << "\",\n";
        file << "  \"matrix_normalization\": \"" << matrix_normalization_name(options.matrix_normalization) << "\",\n";
    }
    if (options.family == Family::kNearSingular) {
        file << "  \"near_singular_exponent\": " << options.near_singular_exponent << ",\n";
        file << "  \"target_smallest_singular_value\": "
             << std::ldexp(1.0, -options.near_singular_exponent) << ",\n";
    }
    file << "  \"n\": " << options.n << ",\n";
    file << "  \"panel_width\": " << options.panel_width << ",\n";
    file << "  \"samples\": " << options.samples << ",\n";
    file << "  \"max_panels\": " << options.max_panels << ",\n";
    file << "  \"seed\": " << options.seed << ",\n";
    file << "  \"threshold\": " << options.threshold << ",\n";
    file << "  \"safe_threshold\": " << options.safe_threshold << ",\n";
    file << "  \"rank_safe_threshold\": " << options.rank_safe_threshold << ",\n";
    file << "  \"dynamic_scaling\": " << (options.scaling ? "true" : "false") << ",\n";
    file << "  \"scaling_policy\": \"" << scaling_policy_name(options.scaling_policy) << "\",\n";
    file << "  \"probe\": \"" << probe_name(options.probe) << "\",\n";
    file << "  \"margin_bits\": " << options.margin_bits << ",\n";
    file << "  \"adaptive_burnin\": " << options.adaptive_burnin << ",\n";
    file << "  \"adaptive_resync_every\": " << options.adaptive_resync_every << ",\n";
    file << "  \"scoped_scaling\": " << (options.scoped_scaling ? "true" : "false") << ",\n";
    file << "  \"bound_source\": \"" << bound_source_name(options.bound_source) << "\",\n";
    file << "  \"measure_lambda\": " << (options.measure_lambda ? "true" : "false") << ",\n";
    file << "  \"blocking\": \"" << blocking_name(options.blocking) << "\",\n";
    file << "  \"u12_schedule\": \""
         << ((options.rankwise_u12 || options.scaling_policy == ScalingPolicy::kPredictive) ? "rankwise" : "serial")
         << "\",\n";
    if (snapshot_requested(options)) {
        file << "  \"snapshots\": {\"original\": " << (options.snapshot_original ? "true" : "false")
             << ", \"final\": " << (options.snapshot_final ? "true" : "false") << ", \"panel_count\": "
             << options.snapshot_panels.size() << "},\n";
    }
    file << "  \"precision\": {\n";
    file << "    \"matrix_storage\": \"IEEE binary16 (__half)\",\n";
    file << "    \"lu_update\": \"CUDA __half multiply/subtract\",\n";
    file << "    \"pivot_search_and_trigger_reductions\": \"FP32\",\n";
    file << "    \"hybrid_calibration_ratio\": \"host FP64 (kHybrid only)\",\n";
    file << "    \"reconstruction_check\": \"host FP64\",\n";
    file << "    \"fast_math\": false\n";
    file << "  },\n";
    file << "  \"gpu\": {\n";
    file << "    \"name\": \"" << properties.name << "\",\n";
    file << "    \"compute_capability\": \"" << properties.major << '.' << properties.minor << "\",\n";
    file << "    \"global_memory_bytes\": " << properties.totalGlobalMem << ",\n";
    file << "    \"free_memory_bytes_at_start\": " << free_memory << ",\n";
    file << "    \"reported_total_memory_bytes_at_start\": " << total_memory << ",\n";
    file << "    \"cuda_driver_version\": " << driver_version << ",\n";
    file << "    \"cuda_runtime_version\": " << runtime_version << "\n";
    file << "  },\n";
    file << "  \"scope\": \"lu is correctness-first and executes the requested blocking mode; trigger is diagnostic-only and performs no LU\"\n";
    file << "}\n";
}

void write_panel_records(const std::filesystem::path& output, const std::vector<PanelRecord>& records) {
    std::ofstream file(output / "panels.csv");
    file << std::setprecision(9);
    file << "sample,panel,panel_start,panel_end,dmax_proxy,trailing_max_oracle,ratio_oracle_over_proxy,"
            "proxy_exponent,oracle_exponent,classification,under_scale_bits,over_scale_bits,proxy_nonfinite,"
            "oracle_nonfinite,applied_exponent,factorization_ms,proxy_scan_ms,oracle_scan_ms,scale_ms,scaled_elements,"
            "subnormal_outputs,flushed_to_zero,nonfinite_values,probe_value,effective_value,ratio_estimate,"
            "is_calibration_panel,base_exponent,probe_scan_ms,lambda_max,lambda_min\n";
    for (const PanelRecord& record : records) {
        file << record.sample << ',' << record.panel << ',' << record.panel_start << ',' << record.panel_end << ','
             << record.dmax_proxy << ',' << record.trailing_max_oracle << ',' << record.ratio_oracle_over_proxy << ','
             << record.proxy_exponent << ',' << record.oracle_exponent << ',' << record.classification << ','
             << record.under_scale_bits << ',' << record.over_scale_bits << ',' << record.proxy_nonfinite << ','
             << record.oracle_nonfinite << ',' << record.applied_exponent << ',' << record.factorization_ms << ',' << record.proxy_scan_ms << ','
             << record.oracle_scan_ms << ',' << record.scale_ms << ',' << record.scaled_elements << ','
             << record.scale_stats.subnormal_outputs << ',' << record.scale_stats.flushed_to_zero << ','
             << record.scale_stats.nonfinite_values << ',' << record.probe_value << ',' << record.effective_value
             << ',' << record.ratio_estimate << ',' << record.is_calibration_panel << ',' << record.base_exponent
             << ',' << record.probe_scan_ms << ',' << record.lambda_max << ',' << record.lambda_min << '\n';
    }
}

void write_run_summaries(const std::filesystem::path& output, const std::vector<RunSummary>& summaries) {
    std::ofstream file(output / "runs.csv");
    file << std::setprecision(17);
    file << "sample,completed,termination,panels_observed,proxy_activations,oracle_activations,true_positive,"
            "false_positive,false_negative,true_negative,exponent_mismatch_panels,total_under_scale_bits,"
            "total_over_scale_bits,total_scaled_elements,subnormal_outputs,flushed_to_zero,nonfinite_values,"
            "factorization_ms,proxy_scan_ms,oracle_scan_ms,scale_ms,final_nonfinite,row_swaps,"
            "column_scale_permutation_self_test_passed,reconstruction_kind,reconstruction_relative_residual\n";
    for (const RunSummary& summary : summaries) {
        file << summary.sample << ',' << (summary.completed ? 1 : 0) << ',' << summary.termination << ','
             << summary.panels_observed << ',' << summary.proxy_activations << ',' << summary.oracle_activations << ','
             << summary.true_positive << ',' << summary.false_positive << ',' << summary.false_negative << ','
             << summary.true_negative << ',' << summary.exponent_mismatch_panels << ','
             << summary.total_under_scale_bits << ',' << summary.total_over_scale_bits << ','
             << summary.total_scaled_elements << ',' << summary.scale_stats.subnormal_outputs << ','
             << summary.scale_stats.flushed_to_zero << ',' << summary.scale_stats.nonfinite_values << ','
             << summary.factorization_ms << ',' << summary.proxy_scan_ms << ',' << summary.oracle_scan_ms << ','
             << summary.scale_ms << ',' << summary.final_nonfinite << ',' << summary.row_swaps << ','
             << (summary.column_scale_permutation_self_test_passed ? 1 : 0) << ',' << summary.reconstruction_kind << ','
             << summary.reconstruction_relative_residual << '\n';
    }
}

void print_summary(const std::vector<RunSummary>& summaries, const std::filesystem::path& output) {
    std::cout << "Wrote " << summaries.size() << " run summary row(s) to " << output << '\n';
    for (const RunSummary& summary : summaries) {
        std::cout << "sample=" << summary.sample << " completed=" << (summary.completed ? "yes" : "no")
                  << " termination=" << summary.termination << " TP/FP/FN/TN=" << summary.true_positive << '/'
                  << summary.false_positive << '/' << summary.false_negative << '/' << summary.true_negative
                  << " proxy_ms=" << summary.proxy_scan_ms << " oracle_ms=" << summary.oracle_scan_ms
                  << " scale_ms=" << summary.scale_ms << " subnormal=" << summary.scale_stats.subnormal_outputs
                  << " zero=" << summary.scale_stats.flushed_to_zero
                  << " residual=" << summary.reconstruction_relative_residual << '\n';
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n\n"
              << "  --mode lu|trigger          default: lu\n"
              << "  --family random|graded|graded_wide|near_singular|wilkinson|underflow|iid_random  default: random\n"
              << "  --n INTEGER                default: 256\n"
              << "  --panel-width INTEGER      default: 8\n"
              << "  --matrix-market FILE       load a square Matrix Market coordinate matrix (lu mode only)\n"
              << "  --matrix-normalization none|power2_max_abs  default: none (external matrices only)\n"
              << "  --snapshot original|final|all|none  write optional FP16 matrix snapshots (lu mode only)\n"
              << "  --snapshot-panels CSV      completed zero-based panel indices to snapshot after policy scaling\n"
              << "  --near-singular-exponent INTEGER  sigma_min = 2^-INTEGER (near_singular only; default: 10)\n"
              << "  --samples INTEGER          default: 1 (lu mode only)\n"
              << "  --max-panels INTEGER       default: all; useful in trigger mode\n"
              << "  --seed INTEGER             default: 20260806\n"
              << "  --threshold FLOAT          default: 0.25\n"
              << "  --safe-threshold FLOAT     predictive range target; default: 32768\n"
              << "  --rank-safe-threshold FLOAT predictive rank-1 target; default: 60000\n"
              << "  --scaling on|off           default: on\n"
              << "  --scaling-policy proxy|oracle|hybrid|certified|predictive  default: proxy\n"
              << "  --probe diagonal|last_column|pivot_row default: diagonal (kHybrid only)\n"
              << "  --margin-bits INTEGER      default: 0 (extra exponent on trigger; kProxy/kHybrid)\n"
              << "  --adaptive-burnin INTEGER  default: 0 (oracle-calibrated panels at the start; kHybrid only)\n"
              << "  --adaptive-resync-every INTEGER  default: 0 (0=never; else recalibrate every K panels)\n"
              << "  --scoped-scaling on|off    default: off (kHybrid+last_column: scale only the probed column)\n"
              << "  --bound-source accumulate|fused  default: accumulate (kCertified only; see BoundSource)\n"
              << "  --measure-lambda on|off    default: off (diagnostic: log per-panel multiplier max/min)\n"
              << "  --blocking unblocked|blocked_fp16|blocked_fp32  default: unblocked (MAGMA-style panel/trailing-GEMM split)\n"
              << "  --u12-schedule serial|rankwise default: serial (performance ablation)\n"
              << "  --output DIRECTORY         default: results/run\n";
}

int parse_integer(const std::string& value, const char* option) {
    try {
        std::size_t consumed = 0;
        const int result = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
}

std::vector<int> parse_snapshot_panels(const std::string& value) {
    if (value.empty()) {
        throw std::runtime_error("--snapshot-panels must not be empty");
    }
    std::vector<int> panels;
    std::istringstream values(value);
    std::string token;
    while (std::getline(values, token, ',')) {
        const int panel = parse_integer(token, "--snapshot-panels");
        if (panel < 0) {
            throw std::runtime_error("--snapshot-panels indices must be non-negative");
        }
        panels.push_back(panel);
    }
    std::sort(panels.begin(), panels.end());
    panels.erase(std::unique(panels.begin(), panels.end()), panels.end());
    return panels;
}

std::uint64_t parse_unsigned(const std::string& value, const char* option) {
    try {
        std::size_t consumed = 0;
        const std::uint64_t result = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid unsigned integer for ") + option + ": " + value);
    }
}

float parse_float(const std::string& value, const char* option) {
    try {
        std::size_t consumed = 0;
        const float result = std::stof(value, &consumed);
        if (consumed != value.size() || !std::isfinite(result)) {
            throw std::invalid_argument("not finite");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid float for ") + option + ": " + value);
    }
}

std::string option_value(int argc, char** argv, int* index, const char* option) {
    if (*index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value after ") + option);
    }
    ++*index;
    return argv[*index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool n_explicit = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--mode") {
            const std::string value = option_value(argc, argv, &index, "--mode");
            if (value == "lu") {
                options.mode = Mode::kLu;
            } else if (value == "trigger") {
                options.mode = Mode::kTrigger;
            } else {
                throw std::runtime_error("--mode must be lu or trigger");
            }
        } else if (argument == "--family") {
            const std::string value = option_value(argc, argv, &index, "--family");
            if (value == "random") {
                options.family = Family::kRandom;
            } else if (value == "graded") {
                options.family = Family::kGraded;
            } else if (value == "graded_wide") {
                options.family = Family::kGradedWide;
            } else if (value == "near_singular") {
                options.family = Family::kNearSingular;
            } else if (value == "wilkinson") {
                options.family = Family::kWilkinson;
            } else if (value == "underflow") {
                options.family = Family::kUnderflow;
            } else if (value == "iid_random") {
                options.family = Family::kIidRandom;
            } else {
                throw std::runtime_error("unknown --family: " + value);
            }
        } else if (argument == "--n") {
            options.n = parse_integer(option_value(argc, argv, &index, "--n"), "--n");
            n_explicit = true;
        } else if (argument == "--panel-width") {
            options.panel_width = parse_integer(option_value(argc, argv, &index, "--panel-width"), "--panel-width");
        } else if (argument == "--matrix-market") {
            options.matrix_market = option_value(argc, argv, &index, "--matrix-market");
        } else if (argument == "--matrix-normalization") {
            const std::string value = option_value(argc, argv, &index, "--matrix-normalization");
            if (value == "none") {
                options.matrix_normalization = MatrixNormalization::kNone;
            } else if (value == "power2_max_abs") {
                options.matrix_normalization = MatrixNormalization::kPowerOfTwoMaxAbs;
            } else {
                throw std::runtime_error("--matrix-normalization must be none or power2_max_abs");
            }
        } else if (argument == "--snapshot") {
            const std::string value = option_value(argc, argv, &index, "--snapshot");
            if (value == "none") {
                options.snapshot_original = false;
                options.snapshot_final = false;
                options.snapshot_panels.clear();
            } else if (value == "original") {
                options.snapshot_original = true;
            } else if (value == "final") {
                options.snapshot_final = true;
            } else if (value == "all") {
                options.snapshot_original = true;
                options.snapshot_final = true;
            } else {
                throw std::runtime_error("--snapshot must be original, final, all, or none");
            }
        } else if (argument == "--snapshot-panels") {
            options.snapshot_panels = parse_snapshot_panels(option_value(argc, argv, &index, "--snapshot-panels"));
        } else if (argument == "--near-singular-exponent") {
            options.near_singular_exponent =
                parse_integer(option_value(argc, argv, &index, "--near-singular-exponent"), "--near-singular-exponent");
        } else if (argument == "--samples") {
            options.samples = parse_integer(option_value(argc, argv, &index, "--samples"), "--samples");
        } else if (argument == "--max-panels") {
            options.max_panels = parse_integer(option_value(argc, argv, &index, "--max-panels"), "--max-panels");
        } else if (argument == "--seed") {
            options.seed = parse_unsigned(option_value(argc, argv, &index, "--seed"), "--seed");
        } else if (argument == "--threshold") {
            options.threshold = parse_float(option_value(argc, argv, &index, "--threshold"), "--threshold");
        } else if (argument == "--safe-threshold") {
            options.safe_threshold =
                parse_float(option_value(argc, argv, &index, "--safe-threshold"), "--safe-threshold");
        } else if (argument == "--rank-safe-threshold") {
            options.rank_safe_threshold = parse_float(
                option_value(argc, argv, &index, "--rank-safe-threshold"), "--rank-safe-threshold");
        } else if (argument == "--scaling") {
            const std::string value = option_value(argc, argv, &index, "--scaling");
            if (value == "on") {
                options.scaling = true;
            } else if (value == "off") {
                options.scaling = false;
            } else {
                throw std::runtime_error("--scaling must be on or off");
            }
        } else if (argument == "--scaling-policy") {
            const std::string value = option_value(argc, argv, &index, "--scaling-policy");
            if (value == "proxy") {
                options.scaling_policy = ScalingPolicy::kProxy;
            } else if (value == "oracle") {
                options.scaling_policy = ScalingPolicy::kOracle;
            } else if (value == "hybrid") {
                options.scaling_policy = ScalingPolicy::kHybrid;
            } else if (value == "certified") {
                options.scaling_policy = ScalingPolicy::kCertified;
            } else if (value == "predictive") {
                options.scaling_policy = ScalingPolicy::kPredictive;
            } else {
                throw std::runtime_error("--scaling-policy must be proxy, oracle, hybrid, certified, or predictive");
            }
        } else if (argument == "--probe") {
            const std::string value = option_value(argc, argv, &index, "--probe");
            if (value == "diagonal") {
                options.probe = Probe::kDiagonal;
            } else if (value == "last_column") {
                options.probe = Probe::kLastColumn;
            } else if (value == "pivot_row") {
                options.probe = Probe::kPivotRow;
            } else {
                throw std::runtime_error("--probe must be diagonal, last_column, or pivot_row");
            }
        } else if (argument == "--margin-bits") {
            options.margin_bits = parse_integer(option_value(argc, argv, &index, "--margin-bits"), "--margin-bits");
        } else if (argument == "--adaptive-burnin") {
            options.adaptive_burnin =
                parse_integer(option_value(argc, argv, &index, "--adaptive-burnin"), "--adaptive-burnin");
        } else if (argument == "--adaptive-resync-every") {
            options.adaptive_resync_every =
                parse_integer(option_value(argc, argv, &index, "--adaptive-resync-every"), "--adaptive-resync-every");
        } else if (argument == "--scoped-scaling") {
            const std::string value = option_value(argc, argv, &index, "--scoped-scaling");
            if (value == "on") {
                options.scoped_scaling = true;
            } else if (value == "off") {
                options.scoped_scaling = false;
            } else {
                throw std::runtime_error("--scoped-scaling must be on or off");
            }
        } else if (argument == "--bound-source") {
            const std::string value = option_value(argc, argv, &index, "--bound-source");
            if (value == "accumulate") {
                options.bound_source = BoundSource::kAccumulate;
            } else if (value == "fused") {
                options.bound_source = BoundSource::kFused;
            } else {
                throw std::runtime_error("--bound-source must be accumulate or fused");
            }
        } else if (argument == "--measure-lambda") {
            const std::string value = option_value(argc, argv, &index, "--measure-lambda");
            if (value == "on") {
                options.measure_lambda = true;
            } else if (value == "off") {
                options.measure_lambda = false;
            } else {
                throw std::runtime_error("--measure-lambda must be on or off");
            }
        } else if (argument == "--blocking") {
            const std::string value = option_value(argc, argv, &index, "--blocking");
            if (value == "unblocked") {
                options.blocking = Blocking::kUnblocked;
            } else if (value == "blocked_fp16") {
                options.blocking = Blocking::kBlockedFp16;
            } else if (value == "blocked_fp32") {
                options.blocking = Blocking::kBlockedFp32Accumulate;
            } else {
                throw std::runtime_error("--blocking must be unblocked, blocked_fp16, or blocked_fp32");
            }
        } else if (argument == "--u12-schedule") {
            const std::string value = option_value(argc, argv, &index, "--u12-schedule");
            if (value == "serial") {
                options.rankwise_u12 = false;
            } else if (value == "rankwise") {
                options.rankwise_u12 = true;
            } else {
                throw std::runtime_error("--u12-schedule must be serial or rankwise");
            }
        } else if (argument == "--output") {
            options.output = option_value(argc, argv, &index, "--output");
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.rankwise_u12 && options.blocking == Blocking::kUnblocked) {
        throw std::runtime_error("rankwise U12 requires a blocked factorization mode");
    }

    if (options.n <= 1) {
        throw std::runtime_error("--n must be greater than one");
    }
    if (options.panel_width <= 0) {
        throw std::runtime_error("--panel-width must be positive");
    }
    if (options.near_singular_exponent <= 0 || options.near_singular_exponent > 24) {
        throw std::runtime_error("--near-singular-exponent must be in [1, 24]");
    }
    if (options.samples <= 0) {
        throw std::runtime_error("--samples must be positive");
    }
    if (options.max_panels == 0 || options.max_panels < -1) {
        throw std::runtime_error("--max-panels must be -1 or positive");
    }
    if (options.threshold <= 0.0f) {
        throw std::runtime_error("--threshold must be positive");
    }
    if (options.safe_threshold <= 0.0f || options.safe_threshold > 32768.0f) {
        throw std::runtime_error("--safe-threshold must be in (0, 32768]");
    }
    if (options.rank_safe_threshold <= 0.0f || options.rank_safe_threshold > 60000.0f) {
        throw std::runtime_error("--rank-safe-threshold must be in (0, 60000]");
    }
    if (options.margin_bits < 0) {
        throw std::runtime_error("--margin-bits must be non-negative");
    }
    if (options.adaptive_burnin < 0) {
        throw std::runtime_error("--adaptive-burnin must be non-negative");
    }
    if (options.adaptive_resync_every < 0) {
        throw std::runtime_error("--adaptive-resync-every must be non-negative");
    }
    if (options.scaling_policy == ScalingPolicy::kPredictive) {
        if (options.mode != Mode::kLu || options.blocking != Blocking::kBlockedFp16) {
            throw std::runtime_error("predictive scaling requires --mode lu --blocking blocked_fp16");
        }
        if (!options.scaling) {
            throw std::runtime_error("predictive scaling requires --scaling on");
        }
        if (options.panel_width > 1024) {
            throw std::runtime_error("predictive scaling currently certifies --panel-width <= 1024");
        }
    }
    if (!options.matrix_market.empty()) {
        if (options.mode != Mode::kLu) {
            throw std::runtime_error("--matrix-market is supported only in lu mode");
        }
        const int loaded_dimension = matrix_market_dimension(options.matrix_market);
        if (loaded_dimension <= 1) {
            throw std::runtime_error("Matrix Market matrix dimension must be greater than one");
        }
        if (n_explicit && options.n != loaded_dimension) {
            throw std::runtime_error("--n must match the Matrix Market matrix dimension");
        }
        options.n = loaded_dimension;
    }
    if (snapshot_requested(options) && options.mode != Mode::kLu) {
        throw std::runtime_error("snapshots are supported only in lu mode");
    }
    const int panel_count = (options.n + options.panel_width - 1) / options.panel_width;
    for (int panel : options.snapshot_panels) {
        if (panel >= panel_count) {
            throw std::runtime_error("--snapshot-panels contains an index beyond the final panel");
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::filesystem::path output(options.output);
        std::filesystem::create_directories(output);
        write_metadata(options, output);
        write_snapshot_manifest(options, output);

        std::vector<PanelRecord> records;
        std::vector<RunSummary> summaries;
        if (options.mode == Mode::kLu) {
            for (int sample = 0; sample < options.samples; ++sample) {
                summaries.push_back(run_lu_sample(options, sample, &records));
            }
        } else {
            summaries.push_back(run_trigger_sample(options, &records));
        }
        write_panel_records(output, records);
        write_run_summaries(output, summaries);
        print_summary(summaries, output);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "dynamic_scaling_validation: " << exception.what() << '\n';
        return 1;
    }
}
