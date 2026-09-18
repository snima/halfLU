#!/usr/bin/env python3
"""
Dedicated Plot Generation Codebase with Custom Policy Ordering:
  Order: DP -> GP -> ScaP -> PP -> ScPP -> RP -> CP

Generates publication-quality figures matching the Reviewer Response style:
1. Throughput across matrix dimensions (Full range up to 60k)
2. Throughput across matrix dimensions (Truncated up to 10k)
3. Pivoting overhead across matrix dimensions (Full range up to 60k)
4. Pivoting overhead across matrix dimensions (Truncated up to 10k)

Strict Design Mandates:
- Zero parentheses across all titles, axes, legends, labels, and ticks.
- No redundant bottom card or theoretical ceiling lines.
- Explicit key-based data mapping ensuring 100% data fidelity when ordering changes.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import argparse

# Paths
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
default_fig_dir = repo_root / "figures"
default_fig_dir.mkdir(parents=True, exist_ok=True)
paper_fig_dir = default_fig_dir

# -------------------------------------------------------------
# 1. Dimension Definitions & Labels (Strictly Zero Parentheses)
# -------------------------------------------------------------
size_labels_full = ["0.5Ki", "1Ki", "2Ki", "10Ki", "40Ki", "60Ki"]
size_labels_10k = ["0.5Ki", "1Ki", "2Ki", "10Ki"]

# -------------------------------------------------------------
# 2. Strict Custom Method Ordering & Color/Marker Mapping
# -------------------------------------------------------------
# Custom order specified: DP, GP, ScaP, PP, ScPP, RP, CP
ordered_methods = ["DP", "GP", "ScaP", "PP", "ScPP", "RP", "CP"]

colors = {
    "DP":   "#2563EB",  # Vibrant Blue
    "GP":   "#059669",  # Emerald Green
    "ScaP": "#D97706",  # Amber / Warm Orange
    "PP":   "#0F172A",  # Slate Black / Baseline Reference
    "ScPP": "#DB2777",  # Deep Rose / Magenta
    "RP":   "#7C3AED",  # Purple
    "CP":   "#DC2626",  # Crimson Red
}

markers = {
    "DP":   "s",  # Square
    "GP":   "D",  # Diamond
    "ScaP": "^",  # Triangle Up
    "PP":   "o",  # Circle
    "ScPP": "v",  # Triangle Down
    "RP":   "*",  # Star
    "CP":   "p",  # Pentagon
}

# -------------------------------------------------------------
# 3. Data Definitions with Keyed Binding (No Data Swap)
# -------------------------------------------------------------
# Baseline PP Throughput (TFLOPS)
pp_tflops_full = np.array([0.011, 0.041, 0.156, 3.91, 28.17, 38.16])

# Overhead Mean and SEM (%) relative to PP
overhead_mean_full = {
    "DP":   np.array([1.34, 1.30, 1.00, 1.85, 0.93, 0.52]),
    "GP":   np.array([1.07, 1.21, 0.60, 0.47, 0.30, 0.21]),
    "ScaP": np.array([1.91, 1.97, 1.37, 0.87, 0.29, 0.17]),
    "PP":   np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),  # Exact baseline reference
    "ScPP": np.array([2.70, 2.59, 1.99, 0.23, 1.23, 0.97]),
    "RP":   np.array([6.25, 6.05, 4.57, 3.03, 1.33, 0.70]),
    "CP":   np.array([2.04, 2.34, 2.00, 4.08, 6.58, 6.33]),
}

overhead_sem_full = {
    "DP":   np.array([0.18, 0.16, 0.14, 0.15, 0.12, 0.09]),
    "GP":   np.array([0.15, 0.14, 0.10, 0.08, 0.06, 0.04]),
    "ScaP": np.array([0.22, 0.20, 0.16, 0.11, 0.06, 0.03]),
    "PP":   np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "ScPP": np.array([0.25, 0.23, 0.19, 0.05, 0.14, 0.11]),
    "RP":   np.array([0.35, 0.31, 0.28, 0.21, 0.15, 0.10]),
    "CP":   np.array([0.24, 0.26, 0.22, 0.32, 0.41, 0.55]),
}

# Derived Throughput (TFLOPS) = PP_TFLOPS / (1 + overhead / 100)
throughput_full = {}
for m in ordered_methods:
    if m == "PP":
        throughput_full["PP"] = pp_tflops_full.copy()
    else:
        throughput_full[m] = pp_tflops_full / (1.0 + overhead_mean_full[m] / 100.0)


# -------------------------------------------------------------
# 4. Helper Function to Build Unified Legend Handles
# -------------------------------------------------------------
def get_legend_handles(is_overhead=False):
    handles = []
    for m in ordered_methods:
        c = colors[m]
        mk = markers[m]
        ms = 9.0 if mk == "*" else 7.5
        ls = "--" if (is_overhead and m == "PP") else "-"
        lw = 2.4 if m == "PP" else 2.0
        handles.append(
            mlines.Line2D(
                [], [], color=c, marker=mk, markersize=ms, linestyle=ls,
                linewidth=lw, markeredgecolor="white", markeredgewidth=1.1,
                label=m
            )
        )
    return handles


# -------------------------------------------------------------
# 5. Throughput Plot Generator
# -------------------------------------------------------------
def plot_throughput(subset_10k=False):
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")
    fig.subplots_adjust(top=0.81, bottom=0.13, left=0.09, right=0.96)

    # Title
    title_text = "Throughput across matrix dimensions up to 10Ki" if subset_10k else "Throughput across matrix dimensions"
    fig.text(0.525, 0.94, title_text, fontsize=14, fontweight="bold", ha="center", va="top", color="#0F172A")

    # Legend in specified order: DP, GP, ScaP, PP, ScPP, RP, CP
    handles = get_legend_handles(is_overhead=False)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.525, 0.875),
        ncol=7, frameon=False, fontsize=10.5, columnspacing=1.6, handletextpad=0.4
    )

    # Grid
    ax.grid(True, linestyle=":", color="#CBD5E1", linewidth=0.7, alpha=0.75, zorder=1)

    labels = size_labels_10k if subset_10k else size_labels_full
    n_points = len(labels)
    x_pos = np.arange(n_points)

    # Plot heuristics first, then PP
    for m in ["CP", "RP", "ScPP", "ScaP", "GP", "DP"]:
        y_vals = throughput_full[m][:n_points]
        mk = markers[m]
        ms = 9.0 if mk == "*" else 7.0
        ax.plot(
            x_pos, y_vals, color=colors[m], marker=mk, markersize=ms,
            linewidth=2.0, alpha=0.85, markeredgecolor="white", markeredgewidth=1.1, zorder=4
        )

    # PP Baseline curve on top
    y_pp = throughput_full["PP"][:n_points]
    ax.plot(
        x_pos, y_pp, color=colors["PP"], marker=markers["PP"], markersize=8.0,
        linewidth=2.6, markeredgecolor="white", markeredgewidth=1.3, zorder=5
    )

    # Axes setup
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, rotation=25, ha="right")
    ax.set_xlabel("Matrix order n", fontsize=11.5, fontweight="bold", labelpad=8, color="#1E293B")
    ax.set_ylabel("Throughput [TFLOPS]", fontsize=11.5, fontweight="bold", labelpad=8, color="#1E293B")
    ax.set_xlim(-0.3, n_points - 0.7)

    if subset_10k:
        ax.set_ylim(-0.1, 4.5)
    else:
        ax.set_ylim(-1.0, 42.0)

    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)

    # Save filenames
    stem = "peak_performance_up_to_10k" if subset_10k else "peak_performance_reviewer_style"
    png_paper = paper_fig_dir / f"{stem}.png"
    pdf_paper = paper_fig_dir / f"{stem}.pdf"

    fig.savefig(png_paper, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_paper, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated {stem}.png and .pdf successfully in {paper_fig_dir}.")


# -------------------------------------------------------------
# 6. Overhead Plot Generator
# -------------------------------------------------------------
def plot_overhead(subset_10k=False):
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")
    fig.subplots_adjust(top=0.81, bottom=0.13, left=0.09, right=0.96)

    # Title
    title_text = "Pivoting overhead across matrix dimensions up to 10Ki" if subset_10k else "Pivoting overhead across matrix dimensions"
    fig.text(0.525, 0.94, title_text, fontsize=14, fontweight="bold", ha="center", va="top", color="#0F172A")

    # Legend in specified order: DP, GP, ScaP, PP, ScPP, RP, CP
    handles = get_legend_handles(is_overhead=True)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.525, 0.875),
        ncol=7, frameon=False, fontsize=10.5, columnspacing=1.6, handletextpad=0.4
    )

    # Zero Reference line
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)

    # Grid
    ax.grid(True, linestyle=":", color="#CBD5E1", linewidth=0.7, alpha=0.75, zorder=1)

    labels = size_labels_10k if subset_10k else size_labels_full
    n_points = len(labels)
    x_pos = np.arange(n_points)

    # Plot curves with error bands
    plot_draw_order = ["CP", "RP", "ScPP", "ScaP", "DP", "GP"]
    for m in plot_draw_order:
        mean_vals = overhead_mean_full[m][:n_points]
        sem_vals = overhead_sem_full[m][:n_points]
        c = colors[m]
        mk = markers[m]
        ms = 9.0 if mk == "*" else 7.5

        # Shaded SEM band
        ax.fill_between(x_pos, mean_vals - sem_vals, mean_vals + sem_vals, color=c, alpha=0.16, zorder=3)

        # Main curve
        ax.plot(
            x_pos, mean_vals, color=c, marker=mk, markersize=ms,
            linewidth=2.3, markeredgecolor="white", markeredgewidth=1.2, zorder=4
        )

    # Plot PP Baseline points on zero line
    ax.plot(
        x_pos, np.zeros(n_points), color=colors["PP"], marker=markers["PP"], markersize=7.5,
        linestyle="--", linewidth=1.6, markeredgecolor="white", markeredgewidth=1.1, zorder=5
    )

    # Axes setup
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, rotation=25, ha="right")
    ax.set_xlabel("Matrix order n", fontsize=11.5, fontweight="bold", labelpad=8, color="#1E293B")
    ax.set_ylabel("Median overhead vs PP [%]", fontsize=11.5, fontweight="bold", labelpad=8, color="#1E293B")
    ax.set_xlim(-0.3, n_points - 0.7)

    if subset_10k:
        ax.set_ylim(-0.5, 7.5)
    else:
        ax.set_ylim(-0.8, 7.8)

    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)

    # Save filenames
    stem = "pivoting_overhead_up_to_10k" if subset_10k else "pivoting_overhead_reviewer_style"
    png_paper = paper_fig_dir / f"{stem}.png"
    pdf_paper = paper_fig_dir / f"{stem}.pdf"

    fig.savefig(png_paper, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_paper, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated {stem}.png and .pdf successfully in {paper_fig_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Custom Order Pivoting Figures")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save output figures")
    args = parser.parse_args()
    if args.output:
        paper_fig_dir = args.output
        paper_fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate all 4 requested plots
    print("Generating throughput plots...")
    plot_throughput(subset_10k=False)
    plot_throughput(subset_10k=True)

    print("Generating overhead plots...")
    plot_overhead(subset_10k=False)
    plot_overhead(subset_10k=True)
    print("All plots generated successfully.")
