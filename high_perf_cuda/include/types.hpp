#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <limits>

namespace high_perf {

enum class Method {
    kPP = 0,   // Partial Pivoting
    kDP = 1,   // Diagonal Pivoting with lookahead
    kGP = 2,   // Growth-preserving / 2nd choice
    kScaP = 3, // Scaled Partial Pivoting (3-column probe)
    kRP = 4,   // Rook Pivoting
    kCP = 5,   // Complete Pivoting (panel-local submatrix search)
    kScPP = 6  // Scalar Partial Pivoting (row-infinity weighted)
};

inline const char* method_to_string(Method m) {
    switch (m) {
        case Method::kPP: return "PP";
        case Method::kDP: return "DP";
        case Method::kGP: return "GP";
        case Method::kScaP: return "ScaP";
        case Method::kRP: return "RP";
        case Method::kCP: return "CP";
        case Method::kScPP: return "ScPP";
        default: return "Unknown";
    }
}

inline Method string_to_method(const std::string& str) {
    if (str == "PP") return Method::kPP;
    if (str == "DP") return Method::kDP;
    if (str == "GP") return Method::kGP;
    if (str == "ScaP") return Method::kScaP;
    if (str == "RP") return Method::kRP;
    if (str == "CP") return Method::kCP;
    if (str == "ScPP") return Method::kScPP;
    throw std::runtime_error("Unknown pivoting method: " + str);
}

enum class MatrixFamily {
    kUniform,
    kNormal,
    kGraded,
    kWilkinson,
    kDpEquality,
    kScaPLast,
    kGpSecond
};

inline const char* family_to_string(MatrixFamily f) {
    switch (f) {
        case MatrixFamily::kUniform: return "uniform";
        case MatrixFamily::kNormal: return "normal";
        case MatrixFamily::kGraded: return "graded";
        case MatrixFamily::kWilkinson: return "wilkinson";
        case MatrixFamily::kDpEquality: return "dp_equality";
        case MatrixFamily::kScaPLast: return "scap_last";
        case MatrixFamily::kGpSecond: return "gp_second";
        default: return "unknown";
    }
}

inline MatrixFamily string_to_family(const std::string& str) {
    if (str == "uniform") return MatrixFamily::kUniform;
    if (str == "normal") return MatrixFamily::kNormal;
    if (str == "graded") return MatrixFamily::kGraded;
    if (str == "wilkinson") return MatrixFamily::kWilkinson;
    if (str == "dp_equality") return MatrixFamily::kDpEquality;
    if (str == "scap_last") return MatrixFamily::kScaPLast;
    if (str == "gp_second") return MatrixFamily::kGpSecond;
    throw std::runtime_error("Unknown matrix family: " + str);
}

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
    kCounterSlots
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

struct Options {
    Method method = Method::kPP;
    MatrixFamily family = MatrixFamily::kUniform;
    int n = 2048;
    int panel_width = 64; // Block size for cuBLAS GEMM trailing update
    std::uint64_t seed = 20260823ULL;
    float tau = 1.01f;
    int window = 6;
    int warmups = 1;
    int repetitions = 3;
    bool verify = true;
    bool profile = false;
    bool hierarchical = false;
    bool transposed = false;
    bool lookahead = false;
    int macro_width = 256;
    int micro_width = 64;
    std::string output = "high_perf_results.csv";
};

struct TimeBreakdown {
    float panel_ms = 0.0f;
    float laswp_ms = 0.0f;
    float trtri_ms = 0.0f;
    float u12_gemm_ms = 0.0f;
    float u12_store_ms = 0.0f;
    float trail_gemm_ms = 0.0f;
    float total_ms = 0.0f;
};

struct BenchmarkResult {
    std::string method_name;
    int n = 0;
    int panel_width = 0;
    float gpu_ms = 0.0f;
    double gflops = 0.0;
    double tflops = 0.0;
    double backward_error = std::numeric_limits<double>::quiet_NaN();
    double growth_factor = std::numeric_limits<double>::quiet_NaN();
    Counters counters;
    TimeBreakdown breakdown;
    bool success = false;
};

} // namespace high_perf
