#include "include/cuda_utils.cuh"
#include "include/types.hpp"
#include "include/matrix_generator.cuh"
#include "include/high_perf_lu.cuh"
#include "include/hierarchical_lu.cuh"
#include "include/transposed_hierarchical_lu.cuh"
#include "include/lookahead_transposed_lu.cuh"
#include "include/verification.cuh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
/*
 * High-performance GPU benchmark harness for half-precision (FP16) LU factorization.
 * ACM Transactions on Mathematical Software (TOMS)
 *
 * Benchmarks execution time and pivoting overhead across 7 pivoting strategies:
 * PP, DP, GP, ScaP, RP, CP, and ScPP.
 *
 * Utilizes cuBLAS GemmEx Tensor Core execution (FP16 math with FP32 accumulation)
 * combined with asynchronous lookahead panel factorization.
 */

#include <string>

using namespace high_perf;

void print_banner(const cudaDeviceProp& prop) {
    std::cout << "================================================================================" << std::endl;
    std::cout << " halfLU — High-Performance GPU Benchmark Suite (ACM TOMS)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << " GPU Device:             " << prop.name << std::endl;
    std::cout << " Compute Capability:     " << prop.major << "." << prop.minor << std::endl;
    std::cout << " Multiprocessors (SMs):  " << prop.multiProcessorCount << std::endl;
    std::cout << " Global Memory:          " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " Tensor Cores / MMA:     ENABLED (via cuBLAS GemmEx)" << std::endl;
    std::cout << "================================================================================" << std::endl;
}

Options parse_arguments(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--method" && i + 1 < argc) {
            opt.method = string_to_method(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            opt.n = std::stoi(argv[++i]);
        } else if (arg == "--panel-width" && i + 1 < argc) {
            opt.panel_width = std::stoi(argv[++i]);
        } else if (arg == "--family" && i + 1 < argc) {
            opt.family = string_to_family(argv[++i]);
        } else if (arg == "--tau" && i + 1 < argc) {
            opt.tau = std::stof(argv[++i]);
        } else if (arg == "--window" && i + 1 < argc) {
            opt.window = std::stoi(argv[++i]);
        } else if (arg == "--warmups" && i + 1 < argc) {
            opt.warmups = std::stoi(argv[++i]);
        } else if (arg == "--repetitions" && i + 1 < argc) {
            opt.repetitions = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opt.seed = std::stoull(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opt.output = argv[++i];
        } else if (arg == "--no-verify") {
            opt.verify = false;
        } else if (arg == "--profile") {
            opt.profile = true;
        } else if (arg == "--hierarchical") {
            opt.hierarchical = true;
        } else if (arg == "--transposed") {
            opt.transposed = true;
            opt.hierarchical = true;
        } else if (arg == "--lookahead") {
            opt.lookahead = true;
            opt.transposed = true;
            opt.hierarchical = true;
        } else if (arg == "--macro-width" && i + 1 < argc) {
            opt.macro_width = std::stoi(argv[++i]);
            opt.hierarchical = true;
        } else if (arg == "--micro-width" && i + 1 < argc) {
            opt.micro_width = std::stoi(argv[++i]);
        }
    }
    return opt;
}

int main(int argc, char** argv) {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    Options opt = parse_arguments(argc, argv);
    print_banner(prop);

    std::cout << " Configuration:" << std::endl;
    std::cout << "   Matrix Order n:       " << opt.n << std::endl;
    if (opt.lookahead) {
        std::cout << "   Mode:                 Lookahead Transposed Hierarchical (Dual-Stream Overlapped LU)" << std::endl;
        std::cout << "   Macro-Panel Width W:  " << opt.macro_width << std::endl;
        std::cout << "   Micro-Panel Width wb: " << opt.micro_width << std::endl;
    } else if (opt.transposed) {
        std::cout << "   Mode:                 Transposed Hierarchical (MAGMA-Style Contiguous Layout)" << std::endl;
        std::cout << "   Macro-Panel Width W:  " << opt.macro_width << std::endl;
        std::cout << "   Micro-Panel Width wb: " << opt.micro_width << std::endl;
    } else if (opt.hierarchical) {
        std::cout << "   Mode:                 Hierarchical Two-Level (Macro/Micro)" << std::endl;
        std::cout << "   Macro-Panel Width W:  " << opt.macro_width << std::endl;
        std::cout << "   Micro-Panel Width wb: " << opt.micro_width << std::endl;
    } else {
        std::cout << "   Mode:                 Single-Level Blocked" << std::endl;
        std::cout << "   Panel Width w:        " << opt.panel_width << std::endl;
    }
    std::cout << "   Pivoting Policy:      " << method_to_string(opt.method) << std::endl;
    std::cout << "   Matrix Family:        " << family_to_string(opt.family) << std::endl;
    std::cout << "   Tau:                  " << opt.tau << std::endl;
    std::cout << "   Window Depth:         " << opt.window << std::endl;
    std::cout << "   Repetitions:          " << opt.repetitions << " (Warmups: " << opt.warmups << ")" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    const std::size_t matrix_elements = static_cast<std::size_t>(opt.n) * opt.n;
    const std::size_t matrix_bytes = matrix_elements * sizeof(__half);

    std::vector<__half> h_A_orig(matrix_elements);
    generate_matrix_host(h_A_orig, opt.n, opt.family, opt.seed);

    __half* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));

    std::vector<int> p_rows, p_cols;
    Counters counters;
    TimeBreakdown breakdown{};
    GpuTimer timer;
    std::vector<float> run_times;
    bool success = true;

    if (opt.lookahead) {
        LookaheadTransposedLU lu_engine(opt.n, opt.macro_width, opt.micro_width);
        // Warmup
        for (int w = 0; w < opt.warmups; ++w) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters);
        }
        for (int r = 0; r < opt.repetitions; ++r) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            timer.start();
            success = lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters, opt.profile ? &breakdown : nullptr);
            float elapsed_ms = timer.stop_and_sync();
            if (success) run_times.push_back(elapsed_ms);
        }
    } else if (opt.transposed) {
        TransposedHierarchicalLU lu_engine(opt.n, opt.macro_width, opt.micro_width);
        // Warmup
        for (int w = 0; w < opt.warmups; ++w) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters);
        }
        for (int r = 0; r < opt.repetitions; ++r) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            timer.start();
            success = lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters, opt.profile ? &breakdown : nullptr);
            float elapsed_ms = timer.stop_and_sync();
            if (success) run_times.push_back(elapsed_ms);
        }
    } else if (opt.hierarchical) {
        HierarchicalLU lu_engine(opt.n, opt.macro_width, opt.micro_width);
        // Warmup
        for (int w = 0; w < opt.warmups; ++w) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters);
        }
        for (int r = 0; r < opt.repetitions; ++r) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            timer.start();
            success = lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters, opt.profile ? &breakdown : nullptr);
            float elapsed_ms = timer.stop_and_sync();
            if (success) run_times.push_back(elapsed_ms);
        }
    } else {
        HighPerfLU lu_engine(opt.n);
        // Warmup
        for (int w = 0; w < opt.warmups; ++w) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters);
        }
        for (int r = 0; r < opt.repetitions; ++r) {
            CUDA_CHECK(cudaMemcpy(d_A, h_A_orig.data(), matrix_bytes, cudaMemcpyHostToDevice));
            timer.start();
            success = lu_engine.factorize(d_A, opt.n, opt, p_rows, p_cols, counters, opt.profile ? &breakdown : nullptr);
            float elapsed_ms = timer.stop_and_sync();
            if (success) run_times.push_back(elapsed_ms);
        }
    }

    if (!success || run_times.empty()) {
        std::cerr << "[-] Error: LU factorization failed or breakdown occurred." << std::endl;
        cudaFree(d_A);
        return 1;
    }

    std::sort(run_times.begin(), run_times.end());
    float median_ms = run_times[run_times.size() / 2];

    // Theoretical FLOPs = 2/3 * n^3
    double total_flops = (2.0 / 3.0) * std::pow(static_cast<double>(opt.n), 3.0);
    double gflops = (total_flops / (median_ms * 1e-3)) / 1e9;
    double tflops = gflops / 1000.0;

    // Verification
    double backward_error = 0.0;
    double growth_factor = 1.0;
    if (opt.verify) {
        std::vector<__half> h_A_fact(matrix_elements);
        CUDA_CHECK(cudaMemcpy(h_A_fact.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
        verify_factorization(h_A_orig, h_A_fact, p_rows, p_cols, opt.n, backward_error, growth_factor);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << " Results:" << std::endl;
    std::cout << "   Median GPU Time:      " << median_ms << " ms" << std::endl;
    std::cout << "   Throughput:           " << std::setprecision(2) << gflops << " GFLOPS  (" 
              << std::setprecision(3) << tflops << " TFLOPS)" << std::endl;
    if (opt.verify) {
        std::cout << "   Backward Error:       " << std::scientific << backward_error << std::endl;
        std::cout << "   Growth Factor:        " << std::fixed << std::setprecision(3) << growth_factor << std::endl;
    }
    std::cout << "   Activation Counters:" << std::endl;
    std::cout << "     DP Accepts:         " << counters.dp_lookahead_accepts << std::endl;
    std::cout << "     DP Fallbacks:       " << counters.dp_fallbacks << std::endl;
    std::cout << "     GP Near-Ties:       " << counters.gp_near_ties << std::endl;
    std::cout << "     GP 2nd Choices:     " << counters.gp_second_choices << std::endl;
    std::cout << "     ScaP Probes:        (cur: " << counters.scap_current << ", mid: " 
              << counters.scap_middle << ", last: " << counters.scap_last << ")" << std::endl;
    std::cout << "     RP Iterations:      " << counters.rp_iterations << std::endl;

    if (opt.profile && breakdown.total_ms > 0.0f) {
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << " Profiling Time Breakdown (Empirical Measurement):" << std::endl;
        if (opt.transposed) {
            std::cout << "   Initial/Final Transpose:" << std::setw(8) << breakdown.trtri_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.trtri_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Panel Factorization:    " << std::setw(8) << breakdown.panel_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.panel_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Coalesced Transp LASWP: " << std::setw(8) << breakdown.laswp_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.laswp_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Trailing TRSM (U12_T):  " << std::setw(8) << breakdown.u12_gemm_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.u12_gemm_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Trailing GEMM (B22):    " << std::setw(8) << breakdown.trail_gemm_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.trail_gemm_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
        } else {
            std::cout << "   Panel Factorization:  " << std::setw(8) << breakdown.panel_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.panel_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Batched LASWP:        " << std::setw(8) << breakdown.laswp_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.laswp_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   TRTRI (L11 Inversion):" << std::setw(8) << breakdown.trtri_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.trtri_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   U12 GEMM:             " << std::setw(8) << breakdown.u12_gemm_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.u12_gemm_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   U12 Store:            " << std::setw(8) << breakdown.u12_store_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.u12_store_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
            std::cout << "   Trailing GEMM (A22):  " << std::setw(8) << breakdown.trail_gemm_ms << " ms (" 
                      << std::setw(5) << std::setprecision(1) << (breakdown.trail_gemm_ms / breakdown.total_ms * 100.0) << "%)" << std::endl;
        }
        std::cout << "   Sum of Profiled:        " << std::setw(8) << breakdown.total_ms << " ms" << std::endl;
    }
    std::cout << "================================================================================" << std::endl;

    // Append to CSV
    std::ofstream out(opt.output, std::ios::app);
    if (out.tellp() == 0) {
        out << "n,panel_width,method,family,tau,window,gpu_ms,gflops,tflops,backward_error,growth_factor,"
            << "dp_accepts,dp_fallbacks,gp_near_ties,gp_second,scap_cur,scap_mid,scap_last,rp_iters\n";
    }
    out << opt.n << "," << opt.panel_width << "," << method_to_string(opt.method) << ","
        << family_to_string(opt.family) << "," << opt.tau << "," << opt.window << ","
        << median_ms << "," << gflops << "," << tflops << ","
        << backward_error << "," << growth_factor << ","
        << counters.dp_lookahead_accepts << "," << counters.dp_fallbacks << ","
        << counters.gp_near_ties << "," << counters.gp_second_choices << ","
        << counters.scap_current << "," << counters.scap_middle << "," << counters.scap_last << ","
        << counters.rp_iterations << "\n";

    cudaFree(d_A);
    return 0;
}
