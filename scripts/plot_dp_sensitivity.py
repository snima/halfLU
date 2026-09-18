#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_dp_sensitivity.py
======================
Generates publication-quality figures for the sensitivity study of Delayed Pivoting (DP):
- Panel (a): Trigger %, Substitution %, and Fallback % vs threshold tau (w=6)
- Panel (b): Extra columns scanned per step vs threshold tau (w=6)
- Panel (c): Fallback % vs lookahead depth w (tau=1.01)
- Panel (d): Relative Frobenius residual ||PAQ - LU||_F / ||A||_F vs tau (w=6)
"""

import argparse
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def main():
    parser = argparse.ArgumentParser(description="Plot DP Sensitivity Analysis")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Path to dp_sensitivity_full_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save output figures")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.summary_csv and args.summary_csv.exists():
        summary_csv = args.summary_csv
    elif (repo_root / "data" / "dp_sensitivity_full_summary.csv").exists():
        summary_csv = repo_root / "data" / "dp_sensitivity_full_summary.csv"
    elif Path("dp_sensitivity_full_summary.csv").exists():
        summary_csv = Path("dp_sensitivity_full_summary.csv")
    else:
        summary_csv = Path("data/dp_sensitivity_full_summary.csv")

    out_dir = args.output_dir if args.output_dir else (repo_root / "figures" if (repo_root / "figures").exists() else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))

    tau_slice = sorted([r for r in rows if int(r['w']) == 6], key=lambda x: float(x['tau']))
    w_slice = sorted([r for r in rows if abs(float(r['tau']) - 1.01) < 1e-4], key=lambda x: int(x['w']))

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    # Panel A: Trigger %, Subst %, Fallback % vs tau (w=6)
    taus = [float(r['tau']) for r in tau_slice]
    trig = [float(r['trigger_pct']) for r in tau_slice]
    sub = [float(r['sub_pct']) for r in tau_slice]
    fall = [float(r['fallback_pct']) for r in tau_slice]

    ax = axs[0, 0]
    ax.plot(taus, trig, 'o-', color='#1f77b4', lw=2, label=r'Trigger % ($r_k < \tau$)')
    ax.plot(taus, sub, 's--', color='#2ca02c', lw=2, label=r'Substitution % ($r_j \geq \tau$)')
    ax.plot(taus, fall, '^-.', color='#d62728', lw=2, label='Fallback to PP %')
    ax.axvline(1.01, color='gray', linestyle=':', lw=1.5, label=r'Default $\tau = 1.01$')
    ax.set_xscale('log')
    ax.set_xticks(taus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r'Pivot Threshold $\tau$ (log scale)')
    ax.set_ylabel('Percentage of Steps (%)')
    ax.set_title(r'(a) DP Activity vs. Threshold $\tau$ ($w=6$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='center left')

    # Panel B: Average Scanned Columns per Step vs tau (w=6)
    scans = [float(r['avg_scans']) for r in tau_slice]
    ax = axs[0, 1]
    ax.plot(taus, scans, 'o-', color='#9467bd', lw=2)
    ax.axvline(1.01, color='gray', linestyle=':', lw=1.5, label=r'Default $\tau = 1.01$ (0.10 scans/step)')
    ax.set_xscale('log')
    ax.set_xticks(taus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r'Pivot Threshold $\tau$ (log scale)')
    ax.set_ylabel('Extra Columns Scanned per Step')
    ax.set_title(r'(b) Lookahead Overhead vs. $\tau$ ($w=6$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    # Panel C: Fallback % vs Lookahead Depth w (tau=1.01)
    ws = [int(r['w']) for r in w_slice]
    w_fall = [float(r['fallback_pct']) for r in w_slice]

    ax = axs[1, 0]
    ax.plot(ws, w_fall, 'o-', color='#d62728', lw=2, label='Fallback %')
    ax.axvline(6, color='gray', linestyle=':', lw=1.5, label=r'Default $w = 6$ (0.24% fallback)')
    ax.axhline(0.22, color='#ff7f0e', linestyle='--', alpha=0.7, label='Panel-edge truncation floor (0.22%)')
    ax.set_xlabel('Lookahead Window Depth $w$')
    ax.set_ylabel('Fallback Percentage (%)')
    ax.set_title(r'(c) Fallback Rate vs. Lookahead Depth $w$ ($\tau=1.01$)')
    ax.set_xticks(ws)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # Panel D: Relative Factor Residual ||PAQ - LU||_F / ||A||_F vs tau (w=6)
    resids = [float(r['residual_gmean']) for r in tau_slice]
    ax = axs[1, 1]
    ax.plot(taus, resids, 'o-', color='#17becf', lw=2, label=r'Geo-mean residual ($\|PAQ-LU\|_F/\|A\|_F$)')
    ax.set_xscale('log')
    ax.set_xticks(taus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r'Pivot Threshold $\tau$ (log scale)')
    ax.set_ylabel(r'$\|PAQ - LU\|_F / \|A\|_F$')
    ax.set_ylim([1e-3, 5e-3])
    ax.set_title(r'(d) Numerical Invariance Across $\tau$ ($w=6$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')

    out_pdf = out_dir / 'dp_sensitivity_analysis.pdf'
    out_png = out_dir / 'dp_sensitivity_analysis.png'
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Plots successfully saved to {out_pdf} and {out_png}')

if __name__ == '__main__':
    main()
