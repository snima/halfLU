#!/usr/bin/env python3
"""Read NLA-extension run CSVs and emit compact validation statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def finite(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row.get(name, ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = []
    for root in arguments.roots:
        for runs_path in sorted(root.resolve().glob("**/runs.csv")):
            metadata_path = runs_path.parent / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            with runs_path.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    estimate = finite(row, "frobenius_residual_estimate")
                    exact = finite(row, "exact_frobenius_residual")
                    records.append({
                        "case": runs_path.parent.name,
                        "family": metadata.get("family", "unknown"),
                        "n": metadata.get("n"),
                        "blocking": metadata.get("blocking", "unknown"),
                        "sample": int(row["sample"]),
                        "completed": int(row["completed"]),
                        "probe_count": int(row.get("norm_probe_count") or 0),
                        "fro_residual": estimate,
                        "probe_rse": finite(row, "frobenius_residual_rse"),
                        "exact_fro_residual": exact,
                        "probe_relative_error": abs(estimate - exact) / exact if estimate is not None and exact not in (None, 0) else None,
                        "residual_over_nu": estimate / (float(metadata.get("n")) * 2**-11) if estimate is not None and metadata.get("n") else None,
                        "growth_envelope": finite(row, "certified_growth_envelope"),
                        "probe_ms": finite(row, "norm_probe_ms"),
                        "factorization_ms": finite(row, "factorization_ms"),
                        "ir_iterations": int(row.get("ir_iterations_completed") or 0),
                        "ir_converged": int(row.get("ir_converged") or 0),
                        "ir_initial_backward": finite(row, "ir_initial_backward_error"),
                        "ir_final_backward": finite(row, "ir_final_backward_error"),
                        "ir_final_forward": finite(row, "ir_final_forward_error"),
                        "ir_ms": finite(row, "ir_ms"),
                    })
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    probe_errors = [record["probe_relative_error"] for record in records if record["probe_relative_error"] is not None]
    lines = [
        "# NLA Extension Validation",
        "",
        f"- Runs: {len(records)}",
        f"- Completed: {sum(record['completed'] for record in records)}/{len(records)}",
    ]
    if probe_errors:
        lines.extend([
            f"- Exact-Frobenius comparison runs: {len(probe_errors)}",
            f"- Probe relative error median: {statistics.median(probe_errors):.4%}",
            f"- Probe relative error maximum: {max(probe_errors):.4%}",
        ])
    lines.extend([
        "",
        "| Case | n | Blocking | Probe residual | RSE | Exact residual | Probe error | r/(nu) | Growth | IR iters | IR back initial/final | IR forward | IR ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for record in records:
        show = lambda value: "NA" if value is None else f"{value:.6g}"
        lines.append(
            f"| {record['case']} | {record['n']} | {record['blocking']} | {show(record['fro_residual'])} | "
            f"{show(record['probe_rse'])} | {show(record['exact_fro_residual'])} | "
            f"{show(record['probe_relative_error'])} | {show(record['residual_over_nu'])} | "
            f"{show(record['growth_envelope'])} | {record['ir_iterations']} | "
            f"{show(record['ir_initial_backward'])}/{show(record['ir_final_backward'])} | "
            f"{show(record['ir_final_forward'])} | {show(record['ir_ms'])} |"
        )
    markdown = output.with_suffix(".md")
    markdown.write_text("\n".join(lines) + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
