#!/usr/bin/env python3
"""Run the norm-probe, growth, FP32-accumulation, and IR extension campaign."""

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


PRECISION_DIMENSIONS = (512, 2048, 8192, 10240, 20480, 30720, 40960, 61440)
IR_EXTRA_DIMENSIONS = (1024, 4096)
FP32_DIMENSIONS = (10240, 20480, 30720, 61440)


def panel_width(n: int) -> int:
    if n <= 4096:
        return 64
    if n <= 10240:
        return 128
    if n <= 20480:
        return 512
    return 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def is_complete(output: Path, samples: int) -> bool:
    runs = output / "runs.csv"
    if not runs.is_file():
        return False
    with runs.open(newline="", encoding="utf-8") as source:
        return len(list(csv.DictReader(source))) == samples


def cases() -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for family in ("iid_random", "graded_wide"):
        for n in sorted(set(PRECISION_DIMENSIONS + IR_EXTRA_DIMENSIONS)):
            selected.append({"family": family, "n": n, "blocking": "blocked_fp16", "ir": 10 if n <= 30720 else 0})
    for n in PRECISION_DIMENSIONS:
        selected.append({"family": "underflow", "n": n, "blocking": "blocked_fp16", "ir": 0})
    for family in ("random", "near_singular"):
        for n in (1024, 4096, 10240, 20480, 30720):
            selected.append({"family": family, "n": n, "blocking": "blocked_fp16", "ir": 10})
    for family in ("random", "iid_random", "graded_wide"):
        for n in FP32_DIMENSIONS:
            selected.append({"family": family, "n": n, "blocking": "blocked_fp32", "ir": 10 if n <= 30720 else 0})
    return selected


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--rerun", action="store_true")
    arguments = parser.parse_args()
    binary = arguments.binary.resolve()
    root = arguments.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    manifest = {record["label"]: record for record in old_manifest}
    selected = cases()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": Path(__file__).name, "runner_sha256": digest(Path(__file__)),
        "binary": str(binary), "binary_sha256": digest(binary),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "norm_probes": 32, "samples_per_cell": 3,
        "configuration_count": len(selected), "expected_runs": len(selected) * 3,
        "precision_dimensions": PRECISION_DIMENSIONS,
        "fp32_dimensions": FP32_DIMENSIONS,
        "ir_max_order": 30720, "ir_max_iterations": 10,
        "seed": arguments.seed,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for index, spec in enumerate(selected, 1):
        family = str(spec["family"])
        n = int(spec["n"])
        blocking = str(spec["blocking"])
        ir = int(spec["ir"])
        width = panel_width(n)
        label = f"{family}__n{n}__w{width}__{blocking}__ir{ir}"
        output = root / label
        if not arguments.rerun and is_complete(output, 3):
            print(f"[{index}/{len(selected)}] skip {label}", flush=True)
            continue
        command = [
            str(binary), "--mode", "lu", "--family", family, "--n", str(n),
            "--panel-width", str(width), "--samples", "3", "--seed", str(arguments.seed),
            "--blocking", blocking, "--scaling", "on", "--scaling-policy", "predictive",
            "--safe-threshold", "32768", "--rank-safe-threshold", "60000",
            "--norm-probes", "32", "--sampled-reconstruction", "off", "--track-growth", "on",
            "--ir-iterations", str(ir), "--ir-tolerance", "1e-12", "--output", str(output),
        ]
        started = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True, timeout=7200, check=False)
        (logs / f"{label}.stdout").write_text(completed.stdout)
        (logs / f"{label}.stderr").write_text(completed.stderr)
        record = {
            "label": label, **spec, "panel_width": width, "samples": 3,
            "return_code": completed.returncode, "wall_seconds": time.monotonic() - started,
            "command": shlex.join(command),
        }
        manifest[label] = record
        manifest_path.write_text(json.dumps(list(manifest.values()), indent=2) + "\n")
        print(f"[{index}/{len(selected)}] rc={completed.returncode} {label} ({record['wall_seconds']:.1f}s)", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
