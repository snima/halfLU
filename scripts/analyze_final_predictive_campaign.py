#!/usr/bin/env python3
"""Aggregate the final 10k--60k predictive-scaling campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def values(rows: list[dict[str, str]], name: str) -> list[float]:
    result = []
    for row in rows:
        try:
            value = float(row[name])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def quantile(data: list[float], probability: float) -> float | None:
    if not data:
        return None
    ordered = sorted(data)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric(rows: list[dict[str, str]], name: str) -> dict[str, float | int | None]:
    data = values(rows, name)
    return {
        f"{name}_count": len(data),
        f"{name}_min": min(data) if data else None,
        f"{name}_p10": quantile(data, 0.1),
        f"{name}_median": statistics.median(data) if data else None,
        f"{name}_p90": quantile(data, 0.9),
        f"{name}_max": max(data) if data else None,
    }


def variant(metadata: dict[str, object]) -> str:
    if not metadata["dynamic_scaling"]:
        return "off_rankwise" if metadata.get("u12_schedule") == "rankwise" else "off_serial"
    return str(metadata["scaling_policy"])


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if number != 0 and (abs(number) >= 1e5 or abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.campaign_root.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    raw_by_key: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    for runs_path in sorted(root.glob("*/runs.csv")):
        metadata = json.loads((runs_path.parent / "metadata.json").read_text())
        with runs_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        current_variant = variant(metadata)
        key = (str(metadata["family"]), int(metadata["n"]), current_variant)
        raw_by_key[key] = rows
        record: dict[str, object] = {
            "family": metadata["family"], "n": metadata["n"],
            "panel_width": metadata["panel_width"], "variant": current_variant,
            "u12_schedule": metadata.get("u12_schedule", "unknown"),
            "samples": len(rows), "completed_count": sum(int(row["completed"]) for row in rows),
            "accepted_count": sum(
                1 for row in rows
                if (value := values([row], "reconstruction_relative_residual")) and value[0] < 1
            ),
            "nonfinite_run_count": sum(int(row["final_nonfinite"]) != 0 for row in rows),
        }
        for name in (
            "panels_observed", "factorization_ms", "scale_ms", "oracle_scan_ms",
            "reconstruction_relative_residual", "total_scaled_elements",
            "subnormal_outputs", "flushed_to_zero", "nonfinite_values",
        ):
            record.update(metric(rows, name))
        records.append(record)

    columns = list(records[0])
    with (output / "configurations.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    (output / "configurations.json").write_text(json.dumps(records, indent=2) + "\n")

    lines = [
        "# Final Predictive Scaling Campaign",
        "",
        f"- Configurations: {len(records)}",
        f"- Raw runs: {sum(int(record['samples']) for record in records)}",
        "- Every stochastic family/dimension cell: three runs.",
        "- Wilkinson negative controls: one deterministic run each.",
        "",
        "## Performance Ablation",
        "",
        "| n | w | Serial ms | Rankwise ms | Predictive ms | Schedule effect | Guard effect | Total effect | Residual match |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    dimensions = sorted({int(record["n"]) for record in records})
    indexed = {(str(record["family"]), int(record["n"]), str(record["variant"])): record for record in records}
    for n in dimensions:
        serial = indexed[("random", n, "off_serial")]
        rankwise = indexed[("random", n, "off_rankwise")]
        predictive = indexed[("random", n, "predictive")]
        serial_ms = float(serial["factorization_ms_median"])
        rankwise_ms = float(rankwise["factorization_ms_median"])
        predictive_ms = float(predictive["factorization_ms_median"])
        serial_rows = raw_by_key[("random", n, "off_serial")]
        rankwise_rows = raw_by_key[("random", n, "off_rankwise")]
        predictive_rows = raw_by_key[("random", n, "predictive")]
        residual_match = all(
            left["reconstruction_relative_residual"] == middle["reconstruction_relative_residual"] == right["reconstruction_relative_residual"]
            for left, middle, right in zip(serial_rows, rankwise_rows, predictive_rows)
        )
        lines.append(
            f"| {n:,} | {serial['panel_width']} | {serial_ms:.3f} | {rankwise_ms:.3f} | {predictive_ms:.3f} | "
            f"{100*(rankwise_ms/serial_ms-1):+.1f}% | {100*(predictive_ms/rankwise_ms-1):+.1f}% | "
            f"{100*(predictive_ms/serial_ms-1):+.1f}% | {'yes' if residual_match else 'NO'} |"
        )

    lines.extend([
        "", "## Graded-Wide Frontier", "",
        "| n | w | Oracle completed | Predictive completed | Accepted | Residual median [min,max] | Subnormal | Flushed |",
        "|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for n in dimensions:
        oracle = indexed[("graded_wide", n, "oracle")]
        predictive = indexed[("graded_wide", n, "predictive")]
        lines.append(
            f"| {n:,} | {predictive['panel_width']} | {oracle['completed_count']}/{oracle['samples']} | "
            f"{predictive['completed_count']}/{predictive['samples']} | {predictive['accepted_count']}/{predictive['samples']} | "
            f"{fmt(predictive['reconstruction_relative_residual_median'])} "
            f"[{fmt(predictive['reconstruction_relative_residual_min'])},{fmt(predictive['reconstruction_relative_residual_max'])}] | "
            f"{fmt(predictive['subnormal_outputs_median'], 0)} | {fmt(predictive['flushed_to_zero_median'], 0)} |"
        )

    lines.extend([
        "", "## Predictive Family Coverage", "",
        "| Family | n | Completed | Accepted | Residual | Scaled elements | Flushed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in sorted(
        (record for record in records if record["variant"] == "predictive"),
        key=lambda item: (str(item["family"]), int(item["n"])),
    ):
        lines.append(
            f"| {record['family']} | {int(record['n']):,} | {record['completed_count']}/{record['samples']} | "
            f"{record['accepted_count']}/{record['samples']} | {fmt(record['reconstruction_relative_residual_median'])} | "
            f"{fmt(record['total_scaled_elements_median'], 0)} | {fmt(record['flushed_to_zero_median'], 0)} |"
        )

    markdown = "\n".join(lines) + "\n"
    (output / "FINAL_CAMPAIGN_STATISTICS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
