#!/usr/bin/env python3
"""Generate compact ACM-TOMS figures from the final blocked V100 summary."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "PP": "#1f2937",
    "DP": "#0f766e",
    "GP": "#d97706",
    "ScaP": "#7c3aed",
    "RP": "#2563eb",
    "CP": "#b91c1c",
    "ScPP": "#64748b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    arguments = parser.parse_args()
    paper_output = arguments.paper_output.resolve()
    analysis_output = arguments.analysis_output.resolve()
    paper_output.mkdir(parents=True, exist_ok=True)
    analysis_output.mkdir(parents=True, exist_ok=True)

    with arguments.summary_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["n"] = int(row["n"])
        row["time"] = float(row["time_ms_median"]) / 1000.0
        row["overhead"] = float(row["overhead_vs_pp_median_pct"])
        row["residual"] = float(row["residual_median"])

    # Main-paper story: replicated cost hierarchy and large numerical frontier.
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.75))
    cost_methods = ("DP", "GP", "ScaP", "RP", "ScPP")
    for method in cost_methods:
        selected = [row for row in rows if row["method"] == method and row["evidence_tier"] == "replicated"]
        axes[0].plot(
            [row["n"] for row in selected], [row["overhead"] for row in selected],
            marker="o", markersize=4.5, linewidth=1.8, label=method, color=COLORS[method],
        )
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([128, 256, 512, 1024, 2048], ["128", "256", "512", "1k", "2k"])
    axes[0].set_ylim(-6, 36)
    axes[0].set_xlabel("Matrix order $n$")
    axes[0].set_ylabel("Time relative to blocked PP (%)")
    axes[0].set_title("Panel-local pivot cost")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    axes[0].text(
        0.02, 0.03, "CP omitted: 291--3018%", transform=axes[0].transAxes,
        fontsize=8, color=COLORS["CP"],
    )

    for method in ("PP", "DP", "GP"):
        selected = [row for row in rows if row["method"] == method]
        axes[1].plot(
            [row["n"] for row in selected], [row["residual"] for row in selected],
            marker="o", markersize=4.2, linewidth=1.8, label=method, color=COLORS[method],
        )
    axes[1].axhline(1.0, color="#b91c1c", linewidth=1.0, linestyle="--", label="Reporting cutoff")
    axes[1].axvspan(2048, 61440, color="#0f766e", alpha=0.06)
    axes[1].axvline(2048, color="#64748b", linewidth=0.8, linestyle=":")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Matrix order $n$")
    axes[1].set_ylabel("Sampled reconstruction residual")
    axes[1].set_title("Completion and numerical frontier")
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].text(0.58, 0.05, "single-seed frontier", transform=axes[1].transAxes, fontsize=8, color="#0f766e")

    figure.tight_layout()
    for directory in (paper_output, analysis_output):
        figure.savefig(directory / "pivoting_panel_local_story.pdf", bbox_inches="tight")
        figure.savefig(directory / "pivoting_panel_local_story.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    # Supplement/response: complete cost hierarchy for n >= 512 replicated cells.
    replicated_large = [row for row in rows if row["evidence_tier"] == "replicated" and row["n"] >= 512]
    values: dict[str, list[float]] = defaultdict(list)
    for row in replicated_large:
        values[row["method"]].append(row["overhead"])
    methods = ("DP", "GP", "ScPP", "RP", "ScaP", "CP")
    medians = [statistics.median(values[method]) for method in methods]
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    bars = axis.barh(methods, medians, color=[COLORS[method] for method in methods], alpha=0.9)
    axis.set_xscale("symlog", linthresh=10)
    axis.axvline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Median time relative to blocked PP (%)")
    axis.set_title(r"Replicated V100 pivot-policy cost hierarchy, $n\geq512$")
    axis.grid(True, axis="x", alpha=0.22)
    for bar, value in zip(bars, medians):
        label_x = value + 1.0 if value >= 0 else 0.5
        axis.text(label_x, bar.get_y() + bar.get_height() / 2,
                  f"{value:+.1f}%", va="center", ha="left", fontsize=8)
    figure.tight_layout()
    figure.savefig(analysis_output / "pivoting_cost_hierarchy.pdf", bbox_inches="tight")
    figure.savefig(analysis_output / "pivoting_cost_hierarchy.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
