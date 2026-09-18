#!/usr/bin/env python3
"""
===============================================================================
ACM TOMS: Clean Schedule-Matched Performance Plotting Script
===============================================================================

This script generates publication-grade figures (PDF and PNG) comparing:
  1. Matched Parallel Baseline (Rankwise Off)
  2. Predictive Scaling (Proposition 3.1)

Both configurations execute on identical 2D parallel CUDA schedules, isolating
the pure overhead of dynamic predictive scaling guards without any scheduling bias.

Outputs:
  - clean_performance_matched.pdf (Vector graphics for LaTeX)
  - clean_performance_matched.png (300 DPI raster for previews)

Usage:
  python3 plot_clean_performance_matched.py [path_to_configurations.csv] [--output OUTPUT_DIR]
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def number(row: dict[str, str], name: str) -> float | None:
    """Safely parse finite float from CSV row."""
    try:
        val = float(row[name])
        return val if math.isfinite(val) else None
    except (KeyError, ValueError):
        return None


def configure_typography() -> None:
    """Set publication-grade font, line-weight, and DPI parameters."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })


def generate_clean_performance_figure(csv_path: Path, output_dir: Path) -> None:
    """Generate the 2-panel clean performance matched figure."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = [r for r in rows if r["family"] == "random" and r["variant"] in ("off_rankwise", "predictive")]
    if not selected:
        raise ValueError(f"No valid rows found for family='random' in {csv_path}")

    dimensions = sorted({int(r["n"]) for r in selected})
    lookup = {(int(r["n"]), r["variant"]): r for r in selected}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"wspace": 0.28, "bottom": 0.18})

    # Exact dimension labels: 1k = 1024
    dim_labels = [f"{n // 1024}k" if n % 1024 == 0 and n >= 1024 else f"{n:,}" for n in dimensions]

    # Nature / ACM Sapphire & Emerald palette
    c_base = "#1D4ED8"     # Deep Cobalt Blue
    c_pred = "#047857"     # Deep Emerald Green
    c_over = "#D97706"     # Warm Ochre / Amber
    c_dark = "#1F2937"

    # -------------------------------------------------------------
    # Panel (a): End-to-End Factorization Timing (Seconds)
    # -------------------------------------------------------------
    y_base = [number(lookup[(n, "off_rankwise")], "factorization_ms_median") / 1000.0 for n in dimensions]
    low_base = [number(lookup[(n, "off_rankwise")], "factorization_ms_p10") / 1000.0 for n in dimensions]
    high_base = [number(lookup[(n, "off_rankwise")], "factorization_ms_p90") / 1000.0 for n in dimensions]

    y_pred = [number(lookup[(n, "predictive")], "factorization_ms_median") / 1000.0 for n in dimensions]
    low_pred = [number(lookup[(n, "predictive")], "factorization_ms_p10") / 1000.0 for n in dimensions]
    high_pred = [number(lookup[(n, "predictive")], "factorization_ms_p90") / 1000.0 for n in dimensions]

    axes[0].plot(dimensions, y_base, marker="o", markersize=5.0, linewidth=1.8, color=c_base,
                 label="Baseline", zorder=3)
    axes[0].fill_between(dimensions, low_base, high_base, color=c_base, alpha=0.15, zorder=2)

    axes[0].plot(dimensions, y_pred, marker="s", markersize=4.8, linewidth=1.8, color=c_pred,
                 label="Predictive Scaling", zorder=3)
    axes[0].fill_between(dimensions, low_pred, high_pred, color=c_pred, alpha=0.15, zorder=2)

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(dimensions)
    axes[0].set_xticklabels(dim_labels, rotation=42, ha="right", rotation_mode="anchor")
    axes[0].tick_params(axis="x", pad=2, length=3.5)
    axes[0].tick_params(axis="y", length=3.5)
    axes[0].set_xlabel("Matrix order $n$", labelpad=4)
    axes[0].set_ylabel("Factorization time (s)")
    axes[0].set_title("(a) Factorization Time", fontweight="bold", pad=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25, color="#9CA3AF")
    axes[0].grid(axis="x", linestyle=":", alpha=0.20, color="#9CA3AF")
    axes[0].legend(frameon=True, framealpha=0.92, edgecolor="#E5E7EB", loc="upper left")

    # -------------------------------------------------------------
    # Panel (b): Predictive Guard Overhead Monotonic Decay
    # -------------------------------------------------------------
    base_ms = [number(lookup[(n, "off_rankwise")], "factorization_ms_median") for n in dimensions]
    pred_ms = [number(lookup[(n, "predictive")], "factorization_ms_median") for n in dimensions]
    overhead = [100.0 * (p / b - 1.0) for p, b in zip(pred_ms, base_ms)]

    axes[1].plot(dimensions, overhead, marker="D", markersize=5.0, linewidth=2.0, color=c_over,
                 label="Overhead (%)", zorder=3)
    axes[1].fill_between(dimensions, overhead, color=c_over, alpha=0.10, zorder=1)

    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(dimensions)
    axes[1].set_xticklabels(dim_labels, rotation=42, ha="right", rotation_mode="anchor")
    axes[1].tick_params(axis="x", pad=2, length=3.5)
    axes[1].tick_params(axis="y", length=3.5)
    axes[1].set_xlabel("Matrix order $n$", labelpad=4)
    axes[1].set_ylabel("Guard overhead (%)")
    axes[1].set_title("(b) Predictive Guard Overhead", fontweight="bold", pad=8)
    axes[1].set_ylim(-2, 68)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(10))
    axes[1].grid(axis="y", linestyle="--", alpha=0.25, color="#9CA3AF")
    axes[1].grid(axis="x", linestyle=":", alpha=0.20, color="#9CA3AF")
    axes[1].legend(frameon=True, framealpha=0.92, edgecolor="#E5E7EB", loc="upper right")

    # Callout Annotations
    bbox_amber = dict(boxstyle="round,pad=0.25", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9, lw=0.6)
    bbox_green = dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#10B981", alpha=0.95, lw=0.6)

    axes[1].annotate("+59.4%\n($n=256$)", xy=(256, 59.4), xytext=(320, 50),
                     arrowprops=dict(arrowstyle="->", color=c_dark, lw=1.1),
                     fontsize=8.2, color=c_dark, ha="left", bbox=bbox_amber)

    axes[1].annotate("+13.7%\n($n=10\\mathrm{k}$)", xy=(10240, 13.7), xytext=(3000, 26),
                     arrowprops=dict(arrowstyle="->", color=c_dark, lw=1.1),
                     fontsize=8.2, color=c_dark, ha="center", bbox=bbox_amber)

    axes[1].annotate("+0.8%\n($n=60\\mathrm{k}$)", xy=(61440, 0.8), xytext=(24000, 12),
                     arrowprops=dict(arrowstyle="->", color=c_dark, lw=1.1),
                     fontsize=8.2, color=c_dark, ha="center", bbox=bbox_green)

    # Save to output
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = output_dir / "clean_performance_matched.pdf"
    png_out = output_dir / "clean_performance_matched.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Successfully wrote:")
    print(f"     PDF: {pdf_out}")
    print(f"     PNG: {png_out}")


def main() -> None:
    default_csv = Path(__file__).resolve().parent / "nla_final_configurations.csv"
    default_out = Path(__file__).resolve().parent / "reproduced_figures"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configurations", type=Path, nargs="?", default=default_csv,
                        help="Path to nla_final_configurations.csv")
    parser.add_argument("--output", type=Path, default=default_out,
                        help="Directory to save generated figures (default: ./reproduced_figures)")
    args = parser.parse_args()

    csv_path = args.configurations.resolve()
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    configure_typography()
    generate_clean_performance_figure(csv_path, args.output.resolve())


if __name__ == "__main__":
    main()
