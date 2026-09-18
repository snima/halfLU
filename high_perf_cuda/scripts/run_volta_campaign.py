#!/usr/bin/env python3
"""
Automated Tesla V100 Full-Scale Benchmark Campaign.
Runs all 7 pivoting policies across multiple dimensions on GPU 0.
Logs results to CSV and prints a real-time Markdown comparative table.
"""

import subprocess
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "build", "high_perf_lu_benchmark"))
BINARY = os.environ.get("HALFLU_BENCHMARK_BIN", DEFAULT_BIN)
OUTPUT_CSV = os.environ.get("HALFLU_CAMPAIGN_CSV", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results_volta_campaign.csv")))

METHODS = ["PP", "DP", "GP", "ScaP", "RP", "CP", "ScPP"]
ORDERS = [2048, 4096, 8192, 10240, 16384, 20480, 32768, 40960]

def main():
    if not os.path.isfile(BINARY):
        print(f"Error: Binary not found at {BINARY}", flush=True)
        sys.exit(1)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print("=" * 80, flush=True)
    print(" TESLA V100 COMPREHENSIVE FP16 LU PIVOTING BENCHMARK CAMPAIGN", flush=True)
    print("=" * 80, flush=True)
    print(f"Output CSV: {OUTPUT_CSV}", flush=True)
    print(f"Methods:    {', '.join(METHODS)}", flush=True)
    print(f"Orders:     {ORDERS}", flush=True)
    print("=" * 80, flush=True)

    header = f"{'Order (n)':<10} {'Method':<8} {'Time (ms)':<12} {'GFLOPS':<12} {'TFLOPS':<10} {'Backward Error':<16} {'Overhead vs PP':<16}"
    print(header, flush=True)
    print("-" * 84, flush=True)

    for n in ORDERS:
        reps = 3 if n <= 4096 else (2 if n <= 20480 else 1)
        pp_time = None

        for m in METHODS:
            cmd = [
                BINARY,
                "--n", str(n),
                "--method", m,
                "--panel-width", "64",
                "--repetitions", str(reps),
                "--warmups", "1",
                "--output", OUTPUT_CSV
            ]
            if n > 2048:
                cmd.append("--no-verify")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if res.returncode != 0:
                print(f"{n:<10} {m:<8} FAILED: {res.stderr.strip()}", flush=True)
                continue

            ms = 0.0
            gflops = 0.0
            b_err = "N/A"
            for line in res.stdout.splitlines():
                if "Median GPU Time:" in line:
                    ms = float(line.split(":")[-1].strip().split()[0])
                elif "Throughput:" in line:
                    gflops = float(line.split(":")[-1].strip().split()[0])
                elif "Backward Error:" in line:
                    b_err = line.split(":")[-1].strip()

            tflops = gflops / 1000.0

            if m == "PP":
                pp_time = ms
                overhead = "Baseline"
            else:
                if pp_time and pp_time > 0:
                    pct = ((ms - pp_time) / pp_time) * 100.0
                    overhead = f"{pct:+.1f}%"
                else:
                    overhead = "N/A"

            print(f"{n:<10} {m:<8} {ms:<12.2f} {gflops:<12.1f} {tflops:<10.3f} {b_err:<16} {overhead:<16}", flush=True)

    print("=" * 84, flush=True)
    print(f"Campaign completed successfully. CSV saved at: {OUTPUT_CSV}", flush=True)

if __name__ == "__main__":
    main()
