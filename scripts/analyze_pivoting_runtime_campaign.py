#!/usr/bin/env python3
"""Aggregate the matched pivoting-runtime campaign without pooling schedules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ("PP", "DP", "GP", "ScaP", "RP", "CP", "ScPP")
SCHEDULE_ORDER = ("unblocked_rank1", "blocked_panel_local")


def finite(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fmt(value: object, specifier: str) -> str:
    if value is None:
        return "NA"
    return format(float(value), specifier)


def fmt_percent(value: object) -> str:
    return "NA" if value is None else f"{float(value):+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.campaign_root.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/runs.csv")):
        with path.open(newline="", encoding="utf-8") as source:
            current = list(csv.DictReader(source))
        for row in current:
            if row.get("schedule") not in SCHEDULE_ORDER:
                raise RuntimeError(f"unexpected schedule in {path}")
            row["source_file"] = str(path.relative_to(root))
            rows.append(row)
    if not rows:
        raise RuntimeError("no runs.csv files found")

    raw_columns = list(rows[0])
    with (output / "pivoting_runtime_raw.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=raw_columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    matched: dict[tuple[str, str, int, int, int, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        schedule = row["schedule"]
        family = row["family"]
        n = int(row["n"])
        grouped[(schedule, method, family, n)].append(row)
        matched[(schedule, family, n, int(row["seed"]), int(row["repetition"]), method)] = row

    statistics_rows: list[dict[str, object]] = []
    for (schedule, method, family, n), current in sorted(
        grouped.items(),
        key=lambda item: (item[0][2], item[0][3], SCHEDULE_ORDER.index(item[0][0]), METHOD_ORDER.index(item[0][1])),
    ):
        times = [value for row in current if (value := finite(row, "factor_wall_ms")) is not None]
        residuals = [value for row in current if (value := finite(row, "reconstruction_residual")) is not None]
        overheads: list[float] = []
        if method != "PP":
            for row in current:
                pp = matched.get((schedule, family, n, int(row["seed"]), int(row["repetition"]), "PP"))
                value = finite(row, "factor_wall_ms")
                baseline = finite(pp, "factor_wall_ms") if pp else None
                if value is not None and baseline is not None and baseline > 0:
                    overheads.append(100.0 * (value / baseline - 1.0))
        completed = sum(row["status"] == "completed" for row in current)
        schedule_effects: list[float] = []
        if schedule == "blocked_panel_local":
            for row in current:
                baseline = matched.get((
                    "unblocked_rank1", family, n, int(row["seed"]), int(row["repetition"]), method
                ))
                value = finite(row, "factor_wall_ms")
                baseline_value = finite(baseline, "factor_wall_ms") if baseline else None
                if value is not None and baseline_value is not None and baseline_value > 0:
                    schedule_effects.append(100.0 * (value / baseline_value - 1.0))
        record: dict[str, object] = {
            "schedule": schedule,
            "panel_width": int(current[0]["panel_width"]),
            "method": method,
            "family": family,
            "n": n,
            "runs": len(current),
            "completed": completed,
            "wall_ms_median": statistics.median(times) if times else None,
            "wall_ms_p10": quantile(times, 0.1) if times else None,
            "wall_ms_p90": quantile(times, 0.9) if times else None,
            "overhead_vs_pp_median_pct": statistics.median(overheads) if overheads else (0.0 if method == "PP" else None),
            "blocked_vs_unblocked_median_pct": statistics.median(schedule_effects) if schedule_effects else None,
            "residual_median": statistics.median(residuals) if residuals else None,
            "max_multiplier": max((finite(row, "max_multiplier") or 0.0) for row in current),
            "row_swaps_median": statistics.median(int(row["row_swaps"]) for row in current),
            "column_swaps_median": statistics.median(int(row["column_swaps"]) for row in current),
            "dp_fallbacks_median": statistics.median(int(row["dp_fallbacks"]) for row in current),
            "gp_second_choices_median": statistics.median(int(row["gp_second_choices"]) for row in current),
        }
        statistics_rows.append(record)

    columns = list(statistics_rows[0])
    with (output / "pivoting_runtime_statistics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(statistics_rows)
    (output / "pivoting_runtime_statistics.json").write_text(json.dumps(statistics_rows, indent=2) + "\n")

    lines = [
        "# Pivoting Runtime Campaign",
        "",
        f"- Raw runs: {len(rows)}",
        "- Schedules: global unblocked rank-1 and panel-local blocked FP16",
        "- Primary time: synchronized end-to-end factorization wall time",
        "- Scope: validation implementation, not production MAGMA/cuSOLVER performance",
        "",
        "| Family | n | Schedule | b | Method | Completed | Median ms | P10--P90 ms | Overhead vs PP | Blocked vs global | Median reconstruction | Max multiplier |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in statistics_rows:
        median = record["wall_ms_median"]
        p10 = record["wall_ms_p10"]
        p90 = record["wall_ms_p90"]
        overhead = record["overhead_vs_pp_median_pct"]
        residual = record["residual_median"]
        lines.append(
            f"| {record['family']} | {record['n']:,} | {record['schedule']} | {record['panel_width']} | {record['method']} | "
            f"{record['completed']}/{record['runs']} | "
            f"{fmt(median, '.3f')} | {fmt(p10, '.3f')}--{fmt(p90, '.3f')} | "
            f"{fmt_percent(overhead)} | {fmt_percent(record['blocked_vs_unblocked_median_pct'])} | "
            f"{fmt(residual, '.3e')} | {fmt(record['max_multiplier'], '.6g')} |"
        )
    lines.extend([
        "",
        "The timing includes method-required host pivot decisions, host/device transfers, swaps, division, and FP16 updates. It excludes input generation, allocation, reconstruction, solves, and iterative refinement. Panel-local methods are algorithmic variants and are not numerically identical to their global counterparts.",
    ])

    large_blocked = [
        record for record in statistics_rows
        if record["schedule"] == "blocked_panel_local" and int(record["n"]) >= 512
    ]
    lines.extend([
        "",
        "## Large-Order Decision Summary",
        "",
        "This summary takes the median of the family/order cell medians for blocked cells with `n >= 512`.",
        "",
        "| Method | Completed | Median overhead vs blocked PP | Median blocked-vs-global effect | Largest observed multiplier |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_ORDER:
        selected = [record for record in large_blocked if record["method"] == method]
        if not selected:
            continue
        overhead_values = [
            float(record["overhead_vs_pp_median_pct"])
            for record in selected if record["overhead_vs_pp_median_pct"] is not None
        ]
        schedule_values = [
            float(record["blocked_vs_unblocked_median_pct"])
            for record in selected if record["blocked_vs_unblocked_median_pct"] is not None
        ]
        lines.append(
            f"| {method} | {sum(int(record['completed']) for record in selected)}/"
            f"{sum(int(record['runs']) for record in selected)} | "
            f"{fmt_percent(statistics.median(overhead_values) if overhead_values else None)} | "
            f"{fmt_percent(statistics.median(schedule_values) if schedule_values else None)} | "
            f"{max(float(record['max_multiplier']) for record in selected):.6g} |"
        )
    (output / "PIVOTING_RUNTIME_STATISTICS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
