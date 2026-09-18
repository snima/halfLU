#!/usr/bin/env python3
"""Fail-closed proof that --search host and --search fused_device define the
same algorithm.

This is the gate that licenses using fused_device timings in the paper: if the
two search paths do not select an identical pivot sequence and produce an
identical factorization, then the speed-up is not a speed-up of the same method
and must not be reported as one.

For every (schedule, family, method, n) cell the two paths must agree on
  * status
  * pivot_digest        (FNV-1a over the whole realised (row, column) sequence)
  * reconstruction_residual
  * max_multiplier
  * growth_factor
  * every activation counter (DP accepts/fallbacks, GP ties/second choices,
    ScaP current/middle/last, RP iterations/failures, row/column swaps)

The fused path is run with --abort-check-stride 1 so that the overflow families
(wilkinson) abort at exactly the same elimination step as the host path. The
timing campaign uses stride 0; see README.md.

Usage:
    python3 verify_equivalence.py --binary ./pivoting_search_cost_validation
    python3 verify_equivalence.py --binary ./... --quick
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
import tempfile

METHODS = ["pp", "dp", "gp", "scap", "rp", "cp", "scpp"]
SCHEDULES = ["blocked_panel_local", "unblocked_rank1"]
FAMILIES = [
    "uniform",
    "normal",
    "graded",
    "wilkinson",
    "dp_equality",
    "scap_last",
    "gp_second",
]

# Fields that must be bit-identical between the two search paths.
COMPARED = [
    "status",
    "pivot_digest",
    "reconstruction_residual",
    "growth_factor",
    "max_multiplier",
    "nonfinite_count",
    "row_swaps",
    "column_swaps",
    "dp_lookahead_accepts",
    "dp_fallbacks",
    "gp_near_ties",
    "gp_second_choices",
    "scap_current",
    "scap_middle",
    "scap_last",
    "rp_iterations",
    "rp_failures",
]


def run_one(binary, out_csv, method, schedule, family, n, seed, search, extra):
    command = [
        binary,
        "--method", method,
        "--schedule", schedule,
        "--family", family,
        "--n", str(n),
        "--seed", str(seed),
        "--search", search,
        "--warmups", "0",
        "--repetitions", "1",
        "--output", out_csv,
    ] + extra
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "run failed: {}\n{}".format(" ".join(command), completed.stderr.strip()))
    with open(out_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no row produced by: " + " ".join(command))
    return rows[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./pivoting_search_cost_validation")
    parser.add_argument("--orders", type=int, nargs="+", default=[128, 256, 320, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 20260823])
    parser.add_argument("--quick", action="store_true",
                        help="one seed and two orders, for a fast pre-flight")
    parser.add_argument("--report", default="equivalence_report.csv")
    arguments = parser.parse_args()

    orders = arguments.orders[:2] if arguments.quick else arguments.orders
    seeds = arguments.seeds[:1] if arguments.quick else arguments.seeds

    if not os.path.exists(arguments.binary):
        print("error: binary not found: " + arguments.binary, file=sys.stderr)
        return 2

    cells = list(itertools.product(SCHEDULES, FAMILIES, METHODS, orders, seeds))
    mismatches = []
    report_rows = []

    for index, (schedule, family, method, n, seed) in enumerate(cells, 1):
        with tempfile.TemporaryDirectory() as directory:
            host_csv = os.path.join(directory, "host.csv")
            fused_csv = os.path.join(directory, "fused.csv")
            host = run_one(arguments.binary, host_csv, method, schedule, family, n, seed,
                           "host", [])
            fused = run_one(arguments.binary, fused_csv, method, schedule, family, n, seed,
                            "fused_device", ["--abort-check-stride", "1"])

        differing = [key for key in COMPARED if host.get(key) != fused.get(key)]
        record = {
            "schedule": schedule,
            "family": family,
            "method": method,
            "n": n,
            "seed": seed,
            "status_host": host["status"],
            "status_fused": fused["status"],
            "digest_host": host["pivot_digest"],
            "digest_fused": fused["pivot_digest"],
            "wall_ms_host": host["factor_wall_ms"],
            "wall_ms_fused": fused["factor_wall_ms"],
            "differing_fields": ";".join(differing),
            "equivalent": not differing,
        }
        report_rows.append(record)
        if differing:
            mismatches.append(record)
            print("[{:4d}/{}] MISMATCH {} {} {} n={} seed={} -> {}".format(
                index, len(cells), schedule, family, method, n, seed,
                ", ".join(differing)))
            for key in differing:
                print("           {}: host={!r} fused={!r}".format(
                    key, host.get(key), fused.get(key)))
        else:
            speedup = float(host["factor_wall_ms"]) / max(float(fused["factor_wall_ms"]), 1e-12)
            print("[{:4d}/{}] ok       {} {} {} n={} seed={} digest={} speedup={:.2f}x".format(
                index, len(cells), schedule, family, method, n, seed,
                host["pivot_digest"], speedup))

    with open(arguments.report, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("cells checked : {}".format(len(report_rows)))
    print("mismatches    : {}".format(len(mismatches)))
    print("report        : {}".format(arguments.report))
    if mismatches:
        print("RESULT: NOT EQUIVALENT -- do not publish fused_device timings.")
        return 1
    print("RESULT: EQUIVALENT -- fused_device timings measure the same algorithm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
