#!/usr/bin/env python3
"""Combine the replicated V100 grid and progressively sampled large frontier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("PP", "DP", "GP", "ScaP", "RP", "CP", "ScPP")
COLORS = {
    "PP": "#1f2937", "DP": "#d97706", "GP": "#0f766e",
    "ScaP": "#7c3aed", "RP": "#2563eb", "CP": "#b91c1c", "ScPP": "#6b7280",
}
EVIDENCE = (
    ("replicated", "v100_pivoting_replicated_final_v2_analysis", 3, 1),
    ("frontier_4k_10k", "v100_large_frontier_pilot_analysis", 1, 1),
    ("frontier_20k", "v100_large_frontier_20480_analysis", 1, 1),
    ("frontier_30k_40k", "v100_large_frontier_30k40k_analysis", 1, 0),
    ("frontier_50k_60k", "v100_large_frontier_50k60k_analysis", 1, 0),
)


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.evidence_root.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    contracts: list[dict[str, object]] = []
    for tier, directory, repetitions, warmups in EVIDENCE:
        source = root / directory / "pivoting_runtime_raw.csv"
        if not source.is_file():
            raise RuntimeError(f"missing evidence: {source}")
        with source.open(newline="", encoding="utf-8") as stream:
            current = list(csv.DictReader(stream))
        for row in current:
            row["evidence_tier"] = tier
            row["contract_repetitions"] = str(repetitions)
            row["contract_warmups"] = str(warmups)
            rows.append(row)
        contracts.append({
            "tier": tier,
            "rows": len(current),
            "repetitions": repetitions,
            "warmups": warmups,
            "orders": sorted({int(row["n"]) for row in current}),
            "methods": sorted({row["method"] for row in current}, key=METHODS.index),
        })

    columns = list(rows[0])
    with (output / "v100_pivoting_combined_raw.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "v100_pivoting_evidence_contracts.json").write_text(json.dumps(contracts, indent=2) + "\n")

    completed = sum(row["status"] == "completed" for row in rows)
    if completed != len(rows):
        raise RuntimeError(f"only {completed}/{len(rows)} runs completed")

    # Matched blocked overheads and order-level medians.
    matched: dict[tuple[str, int, int, int, str], dict[str, str]] = {}
    for row in rows:
        if row["schedule"] == "blocked_panel_local":
            matched[(row["family"], int(row["n"]), int(row["seed"]), int(row["repetition"]), row["method"])] = row

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["schedule"] == "blocked_panel_local":
            grouped[(row["method"], int(row["n"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (method, n), current in sorted(grouped.items(), key=lambda item: (item[0][1], METHODS.index(item[0][0]))):
        times = [float(row["factor_wall_ms"]) for row in current]
        residuals = [float(row["reconstruction_residual"]) for row in current]
        overheads: list[float] = []
        for row in current:
            baseline = matched[(row["family"], n, int(row["seed"]), int(row["repetition"]), "PP")]
            overheads.append(100.0 * (float(row["factor_wall_ms"]) / float(baseline["factor_wall_ms"]) - 1.0))
        summaries.append({
            "method": method,
            "n": n,
            "panel_width": int(current[0]["panel_width"]),
            "runs": len(current),
            "time_ms_median": median(times),
            "overhead_vs_pp_median_pct": median(overheads),
            "residual_median": median(residuals),
            "residual_min": min(residuals),
            "residual_max": max(residuals),
            "max_multiplier": max(float(row["max_multiplier"]) for row in current),
            "evidence_tier": current[0]["evidence_tier"],
        })

    with (output / "v100_pivoting_blocked_summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "v100_pivoting_blocked_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    coverage = {
        method: max((int(row["n"]) for row in rows if row["method"] == method and row["schedule"] == "blocked_panel_local"), default=0)
        for method in METHODS
    }

    lines = [
        "# Final V100 Pivoting Evidence",
        "",
        "- GPU: Tesla V100-PCIE-32GB (GPU 0)",
        "- CUDA: 11.4",
        "- Source SHA-256: `3b932cc3859e0164c78d4f83290a3c30ed9d48c8b5e628e0d3b545b09dc3a81f`",
        "- Binary SHA-256: `ee9d4ae937306ae62052af77b6e6bcd0ea2f80a68afbc2ec72802d72f75fdab9`",
        f"- Completed runs: {completed}/{len(rows)}",
        "- Arithmetic: FP16 storage, separate FP16 multiplication and subtraction",
        "- Scaling: disabled in the pivoting comparison",
        "",
        "## Evidence Contracts",
        "",
        "| Tier | Raw runs | Repetitions | Warm-ups | Orders | Methods |",
        "|---|---:|---:|---:|---|---|",
    ]
    for contract in contracts:
        lines.append(
            f"| {contract['tier']} | {contract['rows']} | {contract['repetitions']} | "
            f"{contract['warmups']} | {', '.join(f'{n:,}' for n in contract['orders'])} | "
            f"{', '.join(contract['methods'])} |"
        )
    lines.extend([
        "",
        "## Blocked Frontier",
        "",
        "| n | b | Method | Runs | Median time (s) | Overhead vs PP | Median reconstruction | Max multiplier | Tier |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for record in summaries:
        lines.append(
            f"| {record['n']:,} | {record['panel_width']} | {record['method']} | {record['runs']} | "
            f"{record['time_ms_median']/1000.0:.3f} | {record['overhead_vs_pp_median_pct']:+.1f}% | "
            f"{record['residual_median']:.3e} | {record['max_multiplier']:.6g} | {record['evidence_tier']} |"
        )
    lines.extend([
        "",
        "## Method Coverage",
        "",
        "| Method | Largest tested blocked order |",
        "|---|---:|",
    ])
    for method in METHODS:
        lines.append(f"| {method} | {coverage[method]:,} |")
    lines.extend([
        "",
        "The replicated grid supports runtime comparisons through n=2,048. The larger cells are progressively sampled frontier tests and must not be presented as equally replicated confidence evidence. Panel-local methods are algorithmic variants of the global pivoting rules.",
    ])
    (output / "V100_PIVOTING_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")

    # Frontier plot. CP is omitted from the time panel so the lower-overhead
    # methods remain readable; its exact costs remain in the table.
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plotted_methods = ("PP", "DP", "GP", "ScaP", "RP", "ScPP")
    for method in plotted_methods:
        selected = [record for record in summaries if record["method"] == method]
        if not selected:
            continue
        axes[0].plot(
            [record["n"] for record in selected],
            [record["time_ms_median"] / 1000.0 for record in selected],
            marker="o", linewidth=1.7, label=method, color=COLORS[method],
        )
        axes[1].plot(
            [record["n"] for record in selected],
            [record["overhead_vs_pp_median_pct"] for record in selected],
            marker="o", linewidth=1.7, label=method, color=COLORS[method],
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel("Matrix order n")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Factorization wall time (s)")
    axes[0].set_title("Panel-local blocked frontier")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Time relative to blocked PP (%)")
    axes[1].set_title("Pivot-policy overhead")
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output / "v100_pivoting_frontier.pdf", bbox_inches="tight")
    figure.savefig(output / "v100_pivoting_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
