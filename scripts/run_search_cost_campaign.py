#!/usr/bin/env python3
"""Campaign runner for the fused device-side pivot search study (Table 7 EXT).

Runs the SAME grid for both search paths so that every reported overhead is an
A/B comparison on matched inputs:

    7 methods x 2 schedules x families x orders x seeds x repetitions x 2 paths

The host path reproduces the frozen Table 7 numbers; the fused path is the new
device-resident implementation. Both write into one CSV distinguished by the
search_mode column, which is what analyze_search_cost.py consumes.

Timing note (inherited from VOLTA_RUNBOOK.md): the publication metric is
factor_wall_ms, not factor_cuda_ms, because the host path's pivot selectors
include CPU work and synchronous transfers. Keeping wall time for both paths is
what makes the comparison honest -- the fused path's advantage is precisely the
removal of that host work.

The host CP branch copies the whole n x n matrix device->host on every
elimination step, so its cost grows as O(n^3) PCIe bytes. --host-cp-max-n caps
the order at which the host CP cell is attempted; beyond it only the fused CP
cell is run and the analysis reports the host cell as not_attempted.

Usage:
    python3 run_search_cost_campaign.py \
        --binary ./pivoting_search_cost_validation \
        --results-root ./campaign_root
"""

import argparse
import collections
import csv
import datetime
import json
import os
import platform
import subprocess
import sys

METHODS = ["pp", "dp", "gp", "scap", "rp", "cp", "scpp"]


def publication_panel_width(n):
    """The frozen 64/128/512/1024 profile from ALGORITHM_CONTRACT.md."""
    if n <= 4096:
        return 64
    if n <= 10240:
        return 128
    if n <= 20480:
        return 512
    return 1024


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as error:  # noqa: BLE001
        return "unavailable: {}".format(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./pivoting_search_cost_validation")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--schedules", nargs="+", default=["blocked_panel_local"])
    parser.add_argument("--families", nargs="+", default=["uniform", "graded"])
    parser.add_argument("--orders", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260823, 20260824, 20260825])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.01)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--searches", nargs="+", default=["host", "fused_device"])
    parser.add_argument("--host-cp-max-n", type=int, default=2048,
                        help="skip the host CP cell above this order")
    parser.add_argument("--abort-check-stride", type=int, default=0,
                        help="fused path only; 0 for timing runs")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", "--force", action="store_true",
                        help="remove existing results before starting to guarantee fresh measurements")
    arguments = parser.parse_args()

    os.makedirs(arguments.results_root, exist_ok=True)
    output_csv = os.path.join(arguments.results_root, "search_cost_runs.csv")
    log_path = os.path.join(arguments.results_root, "campaign_log.txt")
    manifest_path = os.path.join(arguments.results_root, "campaign_manifest.json")

    if arguments.clean:
        for p in (output_csv, log_path, manifest_path):
            if os.path.exists(p):
                os.remove(p)

    cells = []
    for schedule in arguments.schedules:
        for family in arguments.families:
            for n in arguments.orders:
                for seed in arguments.seeds:
                    for method in arguments.methods:
                        for search in arguments.searches:
                            if (method == "cp" and search == "host"
                                    and n > arguments.host_cp_max_n):
                                continue
                            cells.append((schedule, family, n, seed, method, search))

    manifest = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary": os.path.abspath(arguments.binary),
        "binary_sha256": None,
        "host": platform.node(),
        "gpu_before": gpu_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "arguments": vars(arguments),
        "cell_count": len(cells),
        "panel_width_profile": "64/128/512/1024 (publication_panel_width)",
    }
    try:
        digest = subprocess.run(["sha256sum", arguments.binary],
                                capture_output=True, text=True, check=True)
        manifest["binary_sha256"] = digest.stdout.split()[0]
    except Exception:  # noqa: BLE001
        pass

    print("cells to run : {}".format(len(cells)))
    print("output       : {}".format(output_csv))
    if arguments.dry_run:
        for cell in cells:
            print("  " + " ".join(str(part) for part in cell))
        return 0

    completed_counts = collections.defaultdict(int)
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    key = (
                        row.get("schedule", ""),
                        row.get("family", ""),
                        int(row.get("n", 0)),
                        int(row.get("seed", 0)),
                        row.get("method", "").lower(),
                        row.get("search_mode", "").lower(),
                    )
                    completed_counts[key] += 1
                except (ValueError, KeyError):
                    continue

    failures = 0
    with open(log_path, "a") as log:
        for index, (schedule, family, n, seed, method, search) in enumerate(cells, 1):
            key = (schedule, family, n, seed, method.lower(), search.lower())
            if completed_counts.get(key, 0) >= arguments.repetitions:
                header = "[{:5d}/{}] {} {} {} n={} seed={} search={} (already completed, skipping)".format(
                    index, len(cells), method, schedule, family, n, seed, search)
                print(header, flush=True)
                continue
            command = [
                arguments.binary,
                "--method", method,
                "--schedule", schedule,
                "--family", family,
                "--n", str(n),
                "--panel-width", str(publication_panel_width(n)),
                "--seed", str(seed),
                "--search", search,
                "--tau", str(arguments.tau),
                "--window", str(arguments.window),
                "--warmups", str(arguments.warmups),
                "--repetitions", str(arguments.repetitions),
                "--output", output_csv,
            ]
            if search == "fused_device" and arguments.abort_check_stride > 0:
                command += ["--abort-check-stride", str(arguments.abort_check_stride)]
            header = "[{:5d}/{}] {} {} {} n={} seed={} search={}".format(
                index, len(cells), method, schedule, family, n, seed, search)
            print(header, flush=True)
            log.write(header + "\n")
            log.write("  " + " ".join(command) + "\n")
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=arguments.timeout)
                log.write(completed.stdout)
                if completed.returncode != 0:
                    failures += 1
                    log.write("  FAILED rc={}\n{}\n".format(
                        completed.returncode, completed.stderr))
                    print("  FAILED rc={}: {}".format(
                        completed.returncode, completed.stderr.strip()), flush=True)
                else:
                    print("  " + completed.stdout.strip().splitlines()[-1], flush=True)
            except subprocess.TimeoutExpired:
                failures += 1
                log.write("  TIMEOUT after {}s\n".format(arguments.timeout))
                print("  TIMEOUT after {}s".format(arguments.timeout), flush=True)
            log.flush()

    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["gpu_after"] = gpu_snapshot()
    manifest["failed_cells"] = failures
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as handle:
            manifest["csv_rows"] = sum(1 for _ in csv.DictReader(handle))
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print()
    print("failed cells : {}".format(failures))
    print("manifest     : {}".format(manifest_path))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
