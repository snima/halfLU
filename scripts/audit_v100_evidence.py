#!/usr/bin/env python3
"""Fail-closed audit of the combined final V100 pivoting evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_SOURCE = "3b932cc3859e0164c78d4f83290a3c30ed9d48c8b5e628e0d3b545b09dc3a81f"
EXPECTED_BINARY = "ee9d4ae937306ae62052af77b6e6bcd0ea2f80a68afbc2ec72802d72f75fdab9"
EXPECTED_COVERAGE = {
    "PP": 61440, "DP": 61440, "GP": 61440, "ScaP": 40960,
    "RP": 40960, "CP": 2048, "ScPP": 20480,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("combined_raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = arguments.evidence_root.resolve()
    combined = arguments.combined_raw.resolve()

    with combined.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1300:
        raise RuntimeError(f"expected 1300 rows, found {len(rows)}")

    keys: set[tuple[str, ...]] = set()
    fingerprints: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    coverage: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (
            row["evidence_tier"], row["schedule"], row["family"], row["n"],
            row["seed"], row["repetition"], row["method"],
        )
        if key in keys:
            raise RuntimeError(f"duplicate run key: {key}")
        keys.add(key)
        if row["status"] != "completed":
            raise RuntimeError(f"non-completed run: {key}: {row['status']}")
        for field in ("factor_wall_ms", "reconstruction_residual", "growth_factor", "max_multiplier"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"nonfinite {field}: {key}")
        fingerprints[(row["evidence_tier"], row["family"], row["n"] + ":" + row["seed"])].add(
            row["input_fingerprint"]
        )
        if row["schedule"] == "blocked_panel_local":
            coverage[row["method"]] = max(coverage[row["method"]], int(row["n"]))

    inconsistent = {key: values for key, values in fingerprints.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"input fingerprints differ across matched methods: {inconsistent}")
    if dict(coverage) != EXPECTED_COVERAGE:
        raise RuntimeError(f"coverage mismatch: {dict(coverage)}")

    provenance_paths = [
        evidence / "v100_pivoting_replicated_final_v2" / "provenance.json",
        evidence / "v100_large_frontier_pilot" / "provenance.json",
        evidence / "v100_large_frontier_20480" / "provenance.json",
        evidence / "v100_large_frontier_30k40k" / "provenance.json",
        evidence / "v100_large_frontier_50k60k" / "provenance.json",
    ]
    provenance = []
    for path in provenance_paths:
        record = json.loads(path.read_text())
        if record["source_sha256"] != EXPECTED_SOURCE or record["binary_sha256"] != EXPECTED_BINARY:
            raise RuntimeError(f"source/binary mismatch in {path}")
        if "Tesla V100-PCIE-32GB" not in record["gpu_before"] or "0 %" not in record["gpu_before"]:
            raise RuntimeError(f"GPU was not idle before {path}")
        provenance.append({
            "file": str(path.relative_to(evidence)),
            "configurations": record["configuration_count"],
            "raw_runs": record["expected_raw_runs"],
            "created_utc": record["created_utc"],
            "completed_utc": record["completed_utc"],
        })

    result = {
        "status": "passed",
        "combined_raw_sha256": sha256(combined),
        "rows": len(rows),
        "unique_keys": len(keys),
        "completed": len(rows),
        "matched_input_groups": len(fingerprints),
        "coverage": coverage,
        "source_sha256": EXPECTED_SOURCE,
        "binary_sha256": EXPECTED_BINARY,
        "provenance": provenance,
    }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
