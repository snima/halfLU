#!/usr/bin/env python3
"""
Comprehensive High-Performance Benchmark Suite
Automates sweeps across matrix dimensions and pivoting methods,
logging exact GPU runtime, GFLOPS, TFLOPS, and pivoting overheads.
"""

import subprocess
import os
import sys
import pandas as pd

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
BINARY = os.path.join(BUILD_DIR, "high_perf_lu_benchmark")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "high_perf_campaign.csv")

METHODS = ["PP", "DP", "GP", "ScaP", "RP", "CP", "ScPP"]
ORDERS = [512, 1024, 2048, 4096, 8192]

def run_benchmark(n, method, panel_width=64, reps=3, warmups=1):
    cmd = [
        BINARY,
        "--n", str(n),
        "--method", method,
        "--panel-width", str(panel_width),
        "--repetitions", str(reps),
        "--warmups", str(warmups),
        "--output", OUTPUT_CSV
    ]
    if n > 2048:
        cmd.append("--no-verify")
    print(f"Running: n={n:5d}, method={method:4s} ... ", end="", flush=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"FAILED!\n{res.stderr}")
        return False
    # Parse GFLOPS and ms from stdout
    ms = None
    gflops = None
    for line in res.stdout.splitlines():
        if "Median GPU Time:" in line:
            ms = line.split(":")[-1].strip().split()[0]
        if "Throughput:" in line:
            gflops = line.split(":")[-1].strip().split()[0]
    print(f"Done: {ms} ms ({gflops} GFLOPS)")
    return True

def main():
    if not os.path.isfile(BINARY):
        print(f"Binary not found at {BINARY}. Please build first.")
        sys.exit(1)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print("================================================================================")
    print(" STARTING HIGH-PERFORMANCE CUDA FP16 BENCHMARK CAMPAIGN")
    print("================================================================================")

    for n in ORDERS:
        reps = 3 if n <= 4096 else 2
        for m in METHODS:
            run_benchmark(n, m, reps=reps)

    print("\nBenchmark completed. Summary of results:")
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(df[["n", "method", "gpu_ms", "gflops", "backward_error"]].to_string(index=False))

if __name__ == "__main__":
    main()
