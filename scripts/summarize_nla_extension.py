#!/usr/bin/env python3
"""Aggregate NLA-extension raw runs into compact paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def data(rows: list[dict[str, str]], name: str) -> list[float]:
    result = []
    for row in rows:
        try:
            value = float(row.get(name, ""))
        except ValueError:
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def median(rows: list[dict[str, str]], name: str) -> float | None:
    current = data(rows, name)
    return statistics.median(current) if current else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    for runs_path in sorted(arguments.campaign.resolve().glob("*/runs.csv")):
        metadata = json.loads((runs_path.parent / "metadata.json").read_text())
        with runs_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        n = int(metadata["n"])
        residual = median(rows, "frobenius_residual_estimate")
        record = {
            "family": metadata["family"], "n": n, "panel_width": metadata["panel_width"],
            "blocking": metadata["blocking"], "ir_requested": metadata.get("ir_iterations", 0),
            "samples": len(rows), "completed": sum(int(row["completed"]) for row in rows),
            "fro_residual": residual,
            "residual_over_nu": residual / (n * 2**-11) if residual is not None else None,
            "probe_rse": median(rows, "frobenius_residual_rse"),
            "probe_ms": median(rows, "norm_probe_ms"),
            "factorization_ms": median(rows, "factorization_ms"),
            "growth_envelope": median(rows, "certified_growth_envelope"),
            "scaled_elements": median(rows, "total_scaled_elements"),
            "ir_converged": sum(int(row.get("ir_converged") or 0) for row in rows),
            "ir_iterations": median(rows, "ir_iterations_completed"),
            "ir_initial_backward": median(rows, "ir_initial_backward_error"),
            "ir_final_backward": median(rows, "ir_final_backward_error"),
            "ir_final_forward": median(rows, "ir_final_forward_error"),
            "ir_ms": median(rows, "ir_ms"),
        }
        records.append(record)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "nla_extension_configurations.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lookup = {(record["family"], record["n"], record["blocking"]): record for record in records}

    lines = [
        "# NLA Extension Campaign Summary", "",
        f"- Configurations: {len(records)}", f"- Raw runs: {sum(record['samples'] for record in records)}",
        f"- Completed: {sum(record['completed'] for record in records)}/{sum(record['samples'] for record in records)}",
        "- Norm probes per run: 32", "",
        "## Precision-Wall Diagnostics (Sequential FP16)", "",
        "| Family | n | Fro residual | r/(nu) | Growth envelope | Probe RSE | Probe/factor time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("iid_random", "graded_wide", "underflow"):
        family_records = sorted(
            (record for record in records if record["family"] == family and record["blocking"] == "blocked_fp16"),
            key=lambda item: item["n"],
        )
        for record in family_records:
            ratio = record["probe_ms"] / record["factorization_ms"] if record["factorization_ms"] else None
            lines.append(
                f"| {family} | {record['n']:,} | {fmt(record['fro_residual'])} | "
                f"{fmt(record['residual_over_nu'])} | {fmt(record['growth_envelope'])} | "
                f"{fmt(record['probe_rse'])} | {fmt(ratio)} |"
            )

    lines.extend([
        "", "## FP32-Accumulation Sensitivity", "",
        "| Family | n | FP16 residual | FP32 residual | FP32/FP16 | FP16 growth | FP32 growth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for family in ("random", "iid_random", "graded_wide"):
        for n in (10240, 20480, 30720, 61440):
            fp16 = lookup.get((family, n, "blocked_fp16"))
            fp32 = lookup[(family, n, "blocked_fp32_accumulate")]
            if fp16 is None:
                lines.append(f"| {family} | {n:,} | NA | {fmt(fp32['fro_residual'])} | NA | NA | {fmt(fp32['growth_envelope'])} |")
                continue
            ratio = fp32["fro_residual"] / fp16["fro_residual"]
            lines.append(
                f"| {family} | {n:,} | {fmt(fp16['fro_residual'])} | {fmt(fp32['fro_residual'])} | "
                f"{fmt(ratio)} | {fmt(fp16['growth_envelope'])} | {fmt(fp32['growth_envelope'])} |"
            )

    lines.extend([
        "", "## Ledger-Aware Iterative Refinement", "",
        "| Family | n | Accumulation | Converged | Initial backward | Final backward | Final forward | Iterations | IR ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in sorted(
        (record for record in records if record["ir_requested"]),
        key=lambda item: (item["family"], item["n"], item["blocking"]),
    ):
        lines.append(
            f"| {record['family']} | {record['n']:,} | {record['blocking']} | "
            f"{record['ir_converged']}/{record['samples']} | {fmt(record['ir_initial_backward'])} | "
            f"{fmt(record['ir_final_backward'])} | {fmt(record['ir_final_forward'])} | "
            f"{fmt(record['ir_iterations'], 1)} | {fmt(record['ir_ms'])} |"
        )
    markdown = "\n".join(lines) + "\n"
    (output / "NLA_EXTENSION_SUMMARY.md").write_text(markdown)
    print(output / "NLA_EXTENSION_SUMMARY.md")


if __name__ == "__main__":
    main()
