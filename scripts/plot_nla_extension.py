#!/usr/bin/env python3
"""Plot compact NLA-extension statistics from the configuration CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"random": "#2563A6", "iid_random": "#D97706", "graded_wide": "#B91C1C", "near_singular": "#0B7A75", "underflow": "#64748B"}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def value(row: dict[str, str], name: str) -> float | None:
    try:
        parsed = float(row[name])
    except (KeyError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def configure() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def precision_wall(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for family in ("iid_random", "graded_wide", "underflow"):
        current = sorted((row for row in rows if row["family"] == family and row["blocking"] == "blocked_fp16"), key=lambda row: int(row["n"]))
        x = [int(row["n"]) for row in current]
        y = [value(row, "residual_over_nu") for row in current]
        axes[0].plot(x, y, marker="o", linewidth=2, color=COLORS[family], label=family.replace("_", " "))
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Matrix order n")
    axes[0].set_ylabel(r"$r_F/(nu)$")
    axes[0].set_title("Normalized FP16 precision wall")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for family in ("iid_random", "graded_wide"):
        for blocking, style, label_suffix in (("blocked_fp16", "-", "FP16 accumulate"), ("blocked_fp32_accumulate", "--", "FP32 accumulate")):
            current = sorted((row for row in rows if row["family"] == family and row["blocking"] == blocking), key=lambda row: int(row["n"]))
            axes[1].plot([int(row["n"]) for row in current], [value(row, "fro_residual") for row in current],
                         marker="o", linestyle=style, linewidth=2, color=COLORS[family],
                         label=f"{family.replace('_', ' ')}, {label_suffix}")
    axes[1].axhline(1, color="#111827", linestyle=":", label="Residual one")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Matrix order n")
    axes[1].set_ylabel("Estimated relative Frobenius residual")
    axes[1].set_title("Accumulation precision moves the wall")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5)
    save(figure, output, "nla_precision_wall")


def refinement(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for family in ("random", "iid_random", "graded_wide", "near_singular"):
        for blocking, style in (("blocked_fp16", "-"), ("blocked_fp32_accumulate", "--")):
            current = sorted((row for row in rows if row["family"] == family and row["blocking"] == blocking and int(float(row["ir_requested"])) > 0), key=lambda row: int(row["n"]))
            if not current:
                continue
            label = family.replace("_", " ") + (" FP32acc" if blocking == "blocked_fp32_accumulate" else "")
            x = [int(row["n"]) for row in current]
            axes[0].plot(x, [value(row, "ir_final_backward") for row in current], marker="o", linestyle=style, color=COLORS[family], label=label)
            axes[1].plot(x, [value(row, "ir_final_forward") for row in current], marker="o", linestyle=style, color=COLORS[family], label=label)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Matrix order n")
        axis.grid(axis="y", alpha=0.25)
    axes[0].axhline(1e-12, color="#111827", linestyle=":", label="Stopping target")
    axes[0].set_ylabel("Final matrix-only backward error")
    axes[0].set_title("Ledger-aware iterative refinement")
    axes[1].set_ylabel("Final forward error")
    axes[1].set_title("Conditioning remains decisive")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(figure, output, "nla_iterative_refinement")


def growth(rows: list[dict[str, str]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 4.1))
    for family in ("random", "iid_random", "graded_wide", "near_singular", "underflow"):
        current = sorted((row for row in rows if row["family"] == family and row["blocking"] == "blocked_fp16"), key=lambda row: int(row["n"]))
        if current:
            axis.plot([int(row["n"]) for row in current], [value(row, "growth_envelope") for row in current],
                      marker="o", linewidth=1.8, color=COLORS[family], label=family.replace("_", " "))
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Matrix order n")
    axis.set_ylabel("Certified growth envelope")
    axis.set_title("Original-coordinate growth remains representable")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(figure, output, "nla_growth_envelope")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configurations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure()
    rows = load(arguments.configurations.resolve())
    precision_wall(rows, output)
    refinement(rows, output)
    growth(rows, output)
    print(f"Wrote three NLA figure pairs to {output}")


if __name__ == "__main__":
    main()
