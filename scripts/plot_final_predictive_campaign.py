#!/usr/bin/env python3
"""Create publication plots from final campaign configurations.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"off_serial": "#6B7280", "off_rankwise": "#2563A6", "predictive": "#0B7A75"}
FAMILY_COLORS = {
    "random": "#2563A6", "iid_random": "#D97706", "graded": "#7C3AED",
    "graded_wide": "#B91C1C", "near_singular": "#0B7A75", "underflow": "#64748B",
}


def number(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def dimension_labels(dimensions: list[int]) -> list[str]:
    return [f"{n // 1024}k" if n % 1024 == 0 and n >= 1024 else f"{n:,}" for n in dimensions]


def logarithmic_dimension_axis(axis: plt.Axes, dimensions: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(dimensions, dimension_labels(dimensions), rotation=32, ha="right")


def performance(rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in rows if row["family"] == "random" and row["variant"] in COLORS]
    dimensions = sorted({int(row["n"]) for row in selected})
    lookup = {(int(row["n"]), row["variant"]): row for row in selected}
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    labels = {"off_serial": "Off, serial U12", "off_rankwise": "Off, rankwise U12", "predictive": "Predictive"}
    for variant in ("off_serial", "off_rankwise", "predictive"):
        y = [number(lookup[(n, variant)], "factorization_ms_median") / 1000 for n in dimensions]
        low = [number(lookup[(n, variant)], "factorization_ms_p10") / 1000 for n in dimensions]
        high = [number(lookup[(n, variant)], "factorization_ms_p90") / 1000 for n in dimensions]
        axes[0].plot(dimensions, y, marker="o", linewidth=2, color=COLORS[variant], label=labels[variant])
        axes[0].fill_between(dimensions, low, high, color=COLORS[variant], alpha=0.13)
    axes[0].set_yscale("log")
    logarithmic_dimension_axis(axes[0], dimensions)
    axes[0].set_xlabel("Matrix order n")
    axes[0].set_ylabel("Factorization time (s)")
    axes[0].set_title("Matched end-to-end timing")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    schedule, guard, total = [], [], []
    for n in dimensions:
        serial = number(lookup[(n, "off_serial")], "factorization_ms_median")
        rankwise = number(lookup[(n, "off_rankwise")], "factorization_ms_median")
        predictive = number(lookup[(n, "predictive")], "factorization_ms_median")
        schedule.append(100 * (rankwise / serial - 1))
        guard.append(100 * (predictive / rankwise - 1))
        total.append(100 * (predictive / serial - 1))
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].plot(dimensions, schedule, marker="o", color=COLORS["off_rankwise"], label="Schedule effect")
    axes[1].plot(dimensions, guard, marker="s", color="#D97706", label="Guard cost")
    axes[1].plot(dimensions, total, marker="D", color=COLORS["predictive"], linewidth=2, label="Total effect")
    axes[1].set_xlabel("Matrix order n")
    logarithmic_dimension_axis(axes[1], dimensions)
    axes[1].set_ylabel("Time relative to serial off (%)")
    axes[1].set_title("Schedule and guard contributions")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    save(figure, output, "final_performance_ablation")


def numerical(rows: list[dict[str, str]], output: Path) -> None:
    predictive = [row for row in rows if row["variant"] == "predictive"]
    dimensions = sorted({int(row["n"]) for row in predictive})
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    for family in FAMILY_COLORS:
        family_rows = sorted((row for row in predictive if row["family"] == family), key=lambda row: int(row["n"]))
        if not family_rows:
            continue
        x = [int(row["n"]) for row in family_rows]
        y = [number(row, "reconstruction_relative_residual_median") for row in family_rows]
        axes[0].plot(x, y, marker="o", linewidth=1.8, color=FAMILY_COLORS[family], label=family.replace("_", " "))
        if int(family_rows[0]["samples"]) > 1:
            low = [number(row, "reconstruction_relative_residual_min") for row in family_rows]
            high = [number(row, "reconstruction_relative_residual_max") for row in family_rows]
            axes[0].fill_between(x, low, high, color=FAMILY_COLORS[family], alpha=0.12)
    axes[0].axhline(1, color="#111827", linestyle="--", linewidth=1, label="Reporting cutoff")
    axes[0].set_yscale("log")
    logarithmic_dimension_axis(axes[0], dimensions)
    axes[0].set_xlabel("Matrix order n")
    axes[0].set_ylabel("Sampled reconstruction residual")
    axes[0].set_title("Accuracy frontier across families")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    graded = sorted((row for row in predictive if row["family"] == "graded_wide"), key=lambda row: int(row["n"]))
    x = [int(row["n"]) for row in graded]
    scaled = [number(row, "total_scaled_elements_median") for row in graded]
    subnormal_rate = [number(row, "subnormal_outputs_median") / count for row, count in zip(graded, scaled)]
    flushed = [number(row, "flushed_to_zero_median") for row in graded]
    axes[1].plot(x, subnormal_rate, marker="o", color="#B91C1C", linewidth=2, label="Subnormal / scaled")
    axes[1].set_yscale("log")
    logarithmic_dimension_axis(axes[1], dimensions)
    axes[1].set_xlabel("Matrix order n")
    axes[1].set_ylabel("Subnormal event rate", color="#B91C1C")
    axes[1].tick_params(axis="y", labelcolor="#B91C1C")
    axes[1].set_title("Graded-wide representability")
    axes[1].grid(axis="y", alpha=0.25)
    twin = axes[1].twinx()
    twin.plot(x, flushed, marker="s", linestyle="--", color="#374151", label="Flushed count")
    twin.set_ylabel("Flushed-to-zero count", color="#374151")
    twin.tick_params(axis="y", labelcolor="#374151")
    twin.set_ylim(-0.15, max(flushed) + 0.5)
    twin.set_yticks(range(0, int(max(flushed)) + 1))
    handles = axes[1].lines + twin.lines
    axes[1].legend(handles, [line.get_label() for line in handles], frameon=False, loc="upper left")
    save(figure, output, "final_numerical_frontier")


def outcomes(rows: list[dict[str, str]], output: Path) -> None:
    predictive = [row for row in rows if row["variant"] == "predictive"]
    families = ["random", "iid_random", "graded", "graded_wide", "near_singular", "underflow"]
    dimensions = sorted({int(row["n"]) for row in predictive})
    lookup = {(row["family"], int(row["n"])): row for row in predictive}
    matrix = np.zeros((len(families), len(dimensions)))
    annotations: list[list[str]] = []
    for row_index, family in enumerate(families):
        labels = []
        for column, n in enumerate(dimensions):
            record = lookup[(family, n)]
            samples = int(record["samples"])
            accepted = int(record["accepted_count"])
            completed = int(record["completed_count"])
            matrix[row_index, column] = accepted / samples
            labels.append(f"{accepted}/{samples}" if completed == samples else f"C{completed}/{samples}")
        annotations.append(labels)
    figure, axis = plt.subplots(figsize=(14.0, 4.1))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(dimensions)), dimension_labels(dimensions), rotation=30, ha="right")
    axis.set_yticks(range(len(families)), [family.replace("_", " ") for family in families])
    axis.set_xlabel("Matrix order n")
    axis.set_title("Predictive completion with sampled residual below one")
    for row in range(len(families)):
        for column in range(len(dimensions)):
            axis.text(column, row, annotations[row][column], ha="center", va="center", fontsize=8,
                      color="white" if matrix[row, column] < 0.35 else "#111827", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Accepted fraction")
    save(figure, output, "final_predictive_outcomes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configurations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure()
    rows = load(arguments.configurations.resolve())
    performance(rows, output)
    numerical(rows, output)
    outcomes(rows, output)
    (output / "README.md").write_text(
        "# Final Predictive Campaign Figures\n\n"
        "Generated only from `configurations.csv`. Performance ribbons are P10--P90. "
        "Control-family coverage cells with one sample are reachability checks, not uncertainty estimates.\n"
    )
    print(f"Wrote three PDF/PNG figure pairs to {output}")


if __name__ == "__main__":
    main()
