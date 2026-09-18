#!/usr/bin/env python3
"""Build complete unified scale table (n=512 to 61440) on Tesla V100 and generate publication plots."""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS_SMALL = ROOT / "results/search_cost/search_cost_runs.csv"
RUNS_LARGE = ROOT / "results/search_cost_large/search_cost_runs.csv"
ANALYSIS_DIR = ROOT / "analysis/search_cost"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read small runs (n=512..10240) and large runs (n=20480..61440)
method_map = {
    "pp": "PP",
    "cp": "CP",
    "scap": "ScaP",
    "gp": "GP",
    "dp": "DP",
    "scpp": "ScPP",
    "rp": "RP",
}

runs_by_cell = {} # (n, method, search) -> list of wall_ms
for p in (RUNS_SMALL, RUNS_LARGE):
    if not p.is_file():
        continue
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r["schedule"] != "blocked_panel_local" or r["family"] != "uniform":
                continue
            n = int(r["n"])
            m_raw = r["method"].lower()
            m = method_map.get(m_raw, r["method"])
            s = r["search_mode"].lower()
            w = float(r["factor_wall_ms"])
            runs_by_cell.setdefault((n, m, s), []).append(w)

all_orders = sorted({k[0] for k in runs_by_cell.keys()})
methods = ["PP", "CP", "ScaP", "GP", "DP", "ScPP", "RP"]

summary_data = {}
for n in all_orders:
    # PP fused baseline
    pp_fused_vals = runs_by_cell.get((n, "PP", "fused_device"), [])
    pp_fused_med = sorted(pp_fused_vals)[len(pp_fused_vals)//2] if pp_fused_vals else None

    # PP host baseline
    pp_host_vals = runs_by_cell.get((n, "PP", "host"), [])
    pp_host_med = sorted(pp_host_vals)[len(pp_host_vals)//2] if pp_host_vals else None

    for m in methods:
        fused_vals = runs_by_cell.get((n, m, "fused_device"), [])
        fused_med = sorted(fused_vals)[len(fused_vals)//2] if fused_vals else None
        fused_ovh = ((fused_med - pp_fused_med) / pp_fused_med * 100.0) if (fused_med and pp_fused_med) else None

        host_vals = runs_by_cell.get((n, m, "host"), [])
        host_med = sorted(host_vals)[len(host_vals)//2] if host_vals else None
        host_ovh = ((host_med - pp_host_med) / pp_host_med * 100.0) if (host_med and pp_host_med) else None

        speedup = (host_med / fused_med) if (host_med and fused_med) else None

        summary_data[(n, m)] = {
            "n": n,
            "method": m,
            "fused_ms": fused_med,
            "fused_ovh": fused_ovh,
            "host_ms": host_med,
            "host_ovh": host_ovh,
            "speedup": speedup,
        }

# Write Summary CSV
csv_out = ANALYSIS_DIR / "full_scale_search_cost_summary.csv"
with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "method", "fused_wall_ms", "fused_overhead_pct", "host_wall_ms", "host_overhead_pct", "speedup"])
    for n in all_orders:
        for m in methods:
            d = summary_data[(n, m)]
            writer.writerow([
                n, m,
                f"{d['fused_ms']:.3f}" if d["fused_ms"] is not None else "",
                f"{d['fused_ovh']:.2f}" if d["fused_ovh"] is not None else "",
                f"{d['host_ms']:.3f}" if d["host_ms"] is not None else "",
                f"{d['host_ovh']:.2f}" if d["host_ovh"] is not None else "",
                f"{d['speedup']:.3f}" if d["speedup"] is not None else "",
            ])

# Write Markdown Table
md_out = ANALYSIS_DIR / "full_scale_search_cost_table.md"
with open(md_out, "w") as f:
    f.write("# Full-Scale Table 7 EXT: Pivot Search Cost on Tesla V100 (n = 512 to 61440)\n\n")
    f.write("Metric: `factor_wall_ms`. Schedule: `blocked_panel_local`, Uniform Matrix Family.\n")
    f.write("Panel profile: $w=64$ for $n \\le 4096$, $w=128$ for $n \\in [8192, 10240]$, $w=512$ for $n=20480$, $w=1024$ for $n \\ge 30720$.\n\n")
    f.write("| Matrix Size $n$ | Panel Width $w$ | PP Time (Fused) | **CP** | **ScaP** | **GP** | **DP** | **ScPP** | **RP (Rook)** |\n")
    f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    def pw(order):
        if order <= 4096: return 64
        if order <= 10240: return 128
        if order <= 20480: return 512
        return 1024

    for n in all_orders:
        pp_ms = summary_data[(n, "PP")]["fused_ms"]
        time_str = f"{pp_ms:.2f} ms" if pp_ms < 1000 else f"{pp_ms/1000:.2f} s"
        row_str = f"| **{n:,}** | {pw(n)} | {time_str} | "
        for m in ["CP", "ScaP", "GP", "DP", "ScPP", "RP"]:
            ovh = summary_data[(n, m)]["fused_ovh"]
            if ovh is None:
                row_str += "— | "
            elif abs(ovh) < 0.05:
                row_str += "**0.00%** | "
            else:
                sign = "+" if ovh > 0 else ""
                row_str += f"**{sign}{ovh:.2f}%** | " if m in ("CP", "ScaP") else f"{sign}{ovh:.2f}% | "
        f.write(row_str + "\n")

# Generate Publication Plots
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

colors = {
    "CP": "#d62728",    # Red
    "ScaP": "#2ca02c",  # Green
    "RP": "#ff7f0e",    # Orange
    "DP": "#1f77b4",    # Blue
    "GP": "#9467bd",    # Purple
    "ScPP": "#8c564b",  # Brown
}
markers = {
    "CP": "o",
    "ScaP": "s",
    "RP": "^",
    "DP": "v",
    "GP": "D",
    "ScPP": "P",
}

# Plot 1: Full Scale Line Chart (n = 512 to 61440)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
for m in ["CP", "ScaP", "GP", "DP", "ScPP", "RP"]:
    x_vals = [n for n in all_orders if summary_data[(n, m)]["fused_ovh"] is not None]
    y_vals = [summary_data[(n, m)]["fused_ovh"] for n in x_vals]
    ax.plot(x_vals, y_vals, marker=markers[m], color=colors[m], label=m,
            linewidth=2.0, markersize=7, markeredgewidth=1.2, markeredgecolor="white")

ax.axhline(0, color="#555555", linestyle=":", linewidth=1.2, label="PP Baseline (0%)")
ax.set_xscale("log", base=2)
ax.set_xticks(all_orders)
ax.set_xticklabels([f"{n//1000}k" if n >= 10000 else str(n) for n in all_orders])
ax.set_xlabel(r"Matrix Order $n$ (Log Scale, $w \in [64, 1024]$)")
ax.set_ylabel("Overhead Relative to Fused PP Baseline (%)")
ax.set_title("Full-Scale Pivot Search Overhead on Tesla V100 ($n = 512$ to $61,440$)")
ax.set_ylim(-1, 13)
ax.grid(True)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right", ncol=2)

fig.tight_layout()
fig.savefig(ANALYSIS_DIR / "full_scale_pivoting_overhead_512_to_61440.pdf")
fig.savefig(ANALYSIS_DIR / "full_scale_pivoting_overhead_512_to_61440.png", dpi=300)
plt.close(fig)

print("Full-scale summary table and plots successfully generated!")
