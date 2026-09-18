#!/usr/bin/env python3
"""Run a resumable, matched secondary pivoting-runtime campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("pp", "dp", "gp", "scap", "rp", "cp", "scpp")
SCHEDULES = ("unblocked_rank1", "blocked_panel_local")
FAMILIES = ("uniform", "normal")
ORDERS = (128, 256, 512, 1024, 2048)
SEEDS = (20260823, 20260824, 20260825)
REPETITIONS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def panel_width(n: int) -> int:
    if n <= 4096:
        return 64
    if n <= 10240:
        return 128
    if n <= 20480:
        return 512
    return 1024


def complete(path: Path, repetitions: int, schedule: str) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return len(rows) == repetitions and all(row.get("schedule") == schedule for row in rows)


def selected_cases(
    methods: tuple[str, ...],
    schedules: tuple[str, ...],
    families: tuple[str, ...],
    orders: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[tuple[str, str, str, int, int]]:
    cases: list[tuple[str, str, str, int, int]] = []
    for family in families:
        for n in orders:
            for seed in seeds:
                combinations = [(schedule, method) for schedule in schedules for method in methods]
                random.Random(seed ^ n ^ sum(map(ord, family))).shuffle(combinations)
                cases.extend((schedule, method, family, n, seed) for schedule, method in combinations)
    return cases


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def csv_integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in csv_strings(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--schedules", default=",".join(SCHEDULES))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--orders", default=",".join(map(str, ORDERS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--warmups", type=int, default=1)
    arguments = parser.parse_args()

    methods = csv_strings(arguments.methods)
    schedules = csv_strings(arguments.schedules)
    families = csv_strings(arguments.families)
    orders = csv_integers(arguments.orders)
    seeds = csv_integers(arguments.seeds)
    if (not methods or not schedules or not families or not orders or not seeds or
            arguments.repetitions < 1 or arguments.warmups < 0):
        raise SystemExit("campaign selections and repetitions must be non-empty/positive")
    if any(method not in METHODS for method in methods):
        raise SystemExit(f"methods must be selected from {METHODS}")
    if any(schedule not in SCHEDULES for schedule in schedules):
        raise SystemExit(f"schedules must be selected from {SCHEDULES}")
    if any(family not in FAMILIES for family in families):
        raise SystemExit(f"families must be selected from {FAMILIES}")

    binary = arguments.binary.resolve()
    root = arguments.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    records = {record["label"]: record for record in manifest}
    cases = selected_cases(methods, schedules, families, orders, seeds)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": Path(__file__).name,
        "runner_sha256": sha256(Path(__file__)),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source_sha256": sha256(Path(__file__).resolve().parent / "pivoting_runtime_validation.cu"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_before": command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ]),
        "nvcc": command_output(["nvcc", "--version"]),
        "methods": methods,
        "schedules": schedules,
        "families": families,
        "orders": orders,
        "seeds": seeds,
        "warmups_per_configuration": arguments.warmups,
        "repetitions_per_configuration": arguments.repetitions,
        "tau": 1.01,
        "dp_window": 6,
        "panel_widths": {str(n): panel_width(n) for n in orders},
        "configuration_count": len(cases),
        "expected_raw_runs": len(cases) * arguments.repetitions,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for index, (schedule, method, family, n, seed) in enumerate(cases, 1):
        label = f"{family}__n{n}__seed{seed}__{schedule}__{method}"
        output_dir = root / label
        output_dir.mkdir(exist_ok=True)
        runs = output_dir / "runs.csv"
        if not arguments.rerun and complete(runs, arguments.repetitions, schedule):
            print(f"[{index}/{len(cases)}] skip {label}", flush=True)
            continue
        if runs.exists():
            runs.unlink()
        command = [
            str(binary), "--method", method, "--schedule", schedule,
            "--panel-width", str(panel_width(n)), "--family", family, "--n", str(n),
            "--seed", str(seed), "--tau", "1.01", "--window", "6",
            "--warmups", str(arguments.warmups), "--repetitions", str(arguments.repetitions),
            "--output", str(runs),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=arguments.timeout, check=False
        )
        (logs / f"{label}.stdout").write_text(completed.stdout)
        (logs / f"{label}.stderr").write_text(completed.stderr)
        record = {
            "label": label,
            "method": method,
            "schedule": schedule,
            "panel_width": panel_width(n) if schedule == "blocked_panel_local" else 1,
            "family": family,
            "n": n,
            "seed": seed,
            "samples": arguments.repetitions,
            "return_code": completed.returncode,
            "wall_seconds": time.monotonic() - started,
            "command": shlex.join(command),
            "runs_sha256": sha256(runs) if runs.is_file() else None,
        }
        records[label] = record
        manifest_path.write_text(json.dumps(list(records.values()), indent=2) + "\n")
        print(
            f"[{index}/{len(cases)}] rc={completed.returncode} {label} "
            f"({record['wall_seconds']:.1f}s)",
            flush=True,
        )
        if completed.returncode != 0 or not complete(runs, arguments.repetitions, schedule):
            raise SystemExit(f"campaign stopped at {label}")

    provenance["gpu_after"] = command_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ])
    provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
