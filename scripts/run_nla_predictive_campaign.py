#!/usr/bin/env python3
"""Run the final uniformly replicated NLA/TOMS predictive-scaling campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DIMENSIONS = (256, 512, 1024, 2048, 4096, 8192, 10240, 20480, 30720, 40960, 51200, 61440)
CONTROL_FAMILIES = ("iid_random", "graded", "near_singular", "underflow")


def panel_width(n: int) -> int:
    if n <= 4096:
        return 64
    if n <= 10240:
        return 128
    if n <= 20480:
        return 512
    return 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def complete(output: Path, samples: int) -> bool:
    runs = output / "runs.csv"
    if not runs.is_file():
        return False
    with runs.open(newline="", encoding="utf-8") as source:
        return len(list(csv.DictReader(source))) == samples


def selected_cases() -> list[tuple[str, int, str, int]]:
    cases: list[tuple[str, int, str, int]] = []
    for n in DIMENSIONS:
        for policy in ("off_serial", "off_rankwise", "predictive"):
            cases.append(("random", n, policy, 3))
        for policy in ("oracle", "predictive"):
            cases.append(("graded_wide", n, policy, 3))
        for family in CONTROL_FAMILIES:
            cases.append((family, n, "predictive", 3))
    for n in (256, 512, 1024):
        cases.append(("wilkinson", n, "predictive", 1))
    return cases


def policy_arguments(policy: str) -> list[str]:
    if policy == "off_serial":
        return ["--scaling", "off", "--u12-schedule", "serial"]
    if policy == "off_rankwise":
        return ["--scaling", "off", "--u12-schedule", "rankwise"]
    if policy == "oracle":
        return ["--scaling", "on", "--scaling-policy", "oracle", "--u12-schedule", "serial"]
    if policy == "predictive":
        return [
            "--scaling", "on", "--scaling-policy", "predictive",
            "--safe-threshold", "32768", "--rank-safe-threshold", "60000",
        ]
    raise ValueError(policy)


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--rerun", action="store_true")
    arguments = parser.parse_args()

    binary = arguments.binary.resolve()
    root = arguments.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    records = {record["label"]: record for record in manifest}
    cases = selected_cases()

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": Path(__file__).name,
        "runner_sha256": sha256(Path(__file__)),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "dimensions": DIMENSIONS,
        "panel_widths": {str(n): panel_width(n) for n in DIMENSIONS},
        "stochastic_samples_per_cell": 3,
        "wilkinson_samples_per_cell": 1,
        "configuration_count": len(cases),
        "expected_run_count": sum(samples for _, _, _, samples in cases),
        "seed": arguments.seed,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for index, (family, n, policy, samples) in enumerate(cases, 1):
        width = 1 if family == "wilkinson" else panel_width(n)
        label = f"{family}__n{n}__w{width}__{policy}"
        output = root / label
        if not arguments.rerun and complete(output, samples):
            print(f"[{index}/{len(cases)}] skip {label}", flush=True)
            continue
        command = [
            str(binary), "--mode", "lu", "--family", family,
            "--n", str(n), "--panel-width", str(width),
            "--samples", str(samples), "--seed", str(arguments.seed),
            "--threshold", "0.25", "--blocking", "blocked_fp16",
            "--output", str(output), *policy_arguments(policy),
        ]
        started = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True, timeout=7200, check=False)
        (logs / f"{label}.stdout").write_text(completed.stdout)
        (logs / f"{label}.stderr").write_text(completed.stderr)
        record = {
            "label": label, "family": family, "n": n, "panel_width": width,
            "policy": policy, "samples": samples, "return_code": completed.returncode,
            "wall_seconds": time.monotonic() - started, "command": shlex.join(command),
        }
        records[label] = record
        manifest_path.write_text(json.dumps(list(records.values()), indent=2) + "\n")
        print(f"[{index}/{len(cases)}] rc={completed.returncode} {label} ({record['wall_seconds']:.1f}s)", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
