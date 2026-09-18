// ==============================================================================
// halfLU — Algorithmic Runtime Validator for Pivoting Policies
// ACM Transactions on Mathematical Software (TOMS)
//
// Purpose:
//   Evaluates numerical behavior, candidate-search step counts, fallback rates,
//   and row/column swap patterns across all 7 pivoting policies (PP, DP, GP,
//   ScaP, RP, CP, ScPP) in native FP16 arithmetic without dynamic scaling.
//
// Note on Performance:
//   This validation binary evaluates algorithmic logic directly. For maximum
//   hardware throughput (up to 38.16 TFLOPS on Tesla V100 via Tensor Cores,
//   warp-shuffle argmax, and dual-stream lookahead), see `high_perf_cuda/`.
// ==============================================================================

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
    Family family = Family::kUniform;
    int n = 256;
    int panel_width = 0;
    std::uint64_t seed = 20260823ULL;
    float tau = 1.01f;
    int window = 6;
    int warmups = 1;
    int repetitions = 3;
    std::string output = "pivoting_runs.csv";
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

__global__ void rank1_update_kernel(
    __half* matrix,
    int n,
    int pivot,
    int row_start,
    int row_end,
    int column_start,
    int column_end,
    unsigned int* growth_max_bits,
    unsigned long long* nonfinite) {
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
}

__global__ void blocked_update_kernel(
    __half* matrix,
    int n,
    int panel_start,
    int panel_end,
    unsigned int* growth_max_bits,
    unsigned long long* nonfinite) {
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
        else if (argument == "--help") {
            std::printf(
                "Usage: %s --method pp|dp|gp|scap|rp|cp|scpp "
                "--schedule unblocked_rank1|blocked_panel_local "
                "--family uniform|normal|graded|wilkinson|dp_equality|scap_last|gp_second "
                "--n N --seed S "
                "[--panel-width 0=automatic] [--tau 1.01] [--window 6] "
                "[--warmups 1] [--repetitions 3] "
                "[--output runs.csv]\n",
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
                pivot + 1, update_column_end, growth_max_bits.get(), nonfinite.get());
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
                    panel_end, options.n, growth_max_bits.get(), nonfinite.get());
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
                growth_max_bits.get(), nonfinite.get());
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
        output << "method,schedule,panel_width,family,n,seed,input_fingerprint,repetition,tau,window,status,factor_wall_ms,factor_cuda_ms,"
                  "reconstruction_kind,reconstruction_residual,growth_factor,max_multiplier,nonfinite_count,"
                  "row_swaps,column_swaps,dp_lookahead_accepts,dp_fallbacks,gp_near_ties,gp_second_choices,"
                  "scap_current,scap_middle,scap_last,rp_iterations,rp_failures\n";
    }
    output << method_name(options.method) << ',' << schedule_name(options.schedule) << ','
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
           << result.counters.rp_failures << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<__half> input = generate_matrix(options);
        const std::string fingerprint = input_fingerprint(input);
        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            const RunResult ignored = run_once(options, input);
            if (ignored.status != "completed") {
                std::fprintf(stderr, "warning: warm-up status=%s\n", ignored.status.c_str());
            }
        }
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            const RunResult result = run_once(options, input);
            append_result(options, fingerprint, repetition, result);
            std::printf(
                "%s %s %s n=%d b=%d seed=%llu rep=%d status=%s wall_ms=%.3f residual=%.6e mu=%.6g\n",
                method_name(options.method).c_str(), schedule_name(options.schedule).c_str(),
                family_name(options.family).c_str(), options.n,
                options.schedule == Schedule::kBlockedPanelLocal ? options.panel_width : 1,
                static_cast<unsigned long long>(options.seed), repetition, result.status.c_str(),
                result.wall_ms, result.reconstruction_residual, result.max_multiplier);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
