#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_dp_fp16.py
================
Parameter sensitivity study for Delayed Pivoting (DP) in blocked FP16 LU:
    - pivot-ambiguity threshold  tau
    - lookahead depth            w

DP rule (as in the paper): at elimination step k let M1 >= M2 be the two
largest magnitudes in the active column A^(k)[k:n, k], r_k = M1/M2
(+inf if M2 == 0). If r_k < tau, scan up to w subsequent columns of the
CURRENT PANEL; swap in the first column j with r_j >= tau; if none
qualifies, fall back to standard partial pivoting (PP). Row partial
pivoting is always applied to the selected column, so |l_ik| <= 1.

Arithmetic model:
    storage: IEEE binary16 everywhere.
    --accumulate fp32 (default): updates computed in fp32, rounded to fp16
        on store (tensor-core style GEMM/rank-1).
    --accumulate fp16: every multiply and subtract rounded to fp16
        (NumPy evaluates fp16 ufuncs via fp32 and rounds back per op,
        i.e. correctly-rounded fp16 mul/sub; no FMA).

Outputs:
    <out>.csv                    raw results, one row per run
    <out>_summary.csv            aggregated per (tau, w)
    <out>_tables.md / .tex       compact tables: tau-sweep at w=6,
                                 w-sweep at tau=1.01 (+ full grid in .md)
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64

TAUS_DEFAULT = [1.001, 1.005, 1.01, 1.02, 1.05, 1.10, 1.20, 1.50]
WS_DEFAULT = [1, 2, 4, 6, 8, 12, 16]
FAMILIES = ["uniform", "normal", "graded"]

PRESETS = {
    "quick": dict(ns=[256, 512], panels=[32], reps=3),
    "full":  dict(ns=[256, 512, 1024, 2048], panels=[32, 64], reps=5),
}

FIELDS = ["family", "n", "seed", "panel", "tau", "w", "accumulate",
          "steps", "triggers", "subs", "fallbacks", "scanned",
          "trigger_pct", "sub_pct", "fallback_pct",
          "sub_success_given_trigger_pct", "avg_scanned_per_step",
          "max_mult", "growth", "residual", "finite", "time_s"]


# --------------------------------------------------------------------------
# Test matrices
# --------------------------------------------------------------------------
def matrix_seed(family, n, rep):
    """Stable seed: depends only on (family, n, rep)."""
    return 20260821 + 7919 * FAMILIES.index(family) + 13 * n + rep


def make_matrix(family, n, rep):
    rng = np.random.default_rng(matrix_seed(family, n, rep))
    if family == "uniform":
        A = rng.uniform(0.0, 1.0, (n, n))
    elif family == "normal":
        A = rng.standard_normal((n, n))
    elif family == "graded":
        # column-graded over 3 decades: stresses the column-swap logic
        A = rng.standard_normal((n, n)) * np.logspace(0.0, -3.0, n)[None, :]
    else:
        raise ValueError(f"unknown family {family!r}")
    return A / np.abs(A).max()          # unit max-norm -> FP16-range safe


# --------------------------------------------------------------------------
# Core factorization
# --------------------------------------------------------------------------
def _top2(colabs):
    """(r, argmax) for a 1-D fp32 magnitude array; r = M1/M2, inf if M2==0."""
    m = colabs.shape[0]
    if m == 1:
        return math.inf, 0
    i2, i1 = np.argpartition(colabs, m - 2)[-2:]
    if colabs[i2] > colabs[i1]:
        i1, i2 = i2, i1
    M1, M2 = float(colabs[i1]), float(colabs[i2])
    if M2 == 0.0:
        return math.inf, int(i1)        # covers the all-zero column, too
    return M1 / M2, int(i1)


def dp_lu_fp16(A64, panel, tau, w, accumulate="fp32"):
    """
    Blocked right-looking LU with row partial pivoting + Delayed Pivoting
    (within-panel column lookahead) on FP16-stored data.

    Returns a dict of statistics; factors are validated internally via the
    FP64 residual ||P A16 Q - L U||_F / ||A16||_F.
    """
    n = A64.shape[0]
    A16 = A64.astype(F16)
    W = A16.copy()
    p_idx = np.arange(n)
    q_idx = np.arange(n)
    fp32acc = (accumulate == "fp32")

    steps = triggers = subs = fallbacks = scanned = 0
    max_mult = 0.0
    g0 = float(np.abs(A16.astype(F32)).max())
    gmax = g0

    for p0 in range(0, n, panel):
        p1 = min(p0 + panel, n)

        # ---------------- panel factorization (right-looking) -------------
        for k in range(p0, p1):
            steps += 1
            r_k, imax = _top2(np.abs(W[k:, k].astype(F32)))
            sel = k
            if r_k < tau:                                # ambiguous pivot
                triggers += 1
                found = False
                for j in range(k + 1, min(k + w, p1 - 1) + 1):
                    scanned += 1
                    r_j, imax_j = _top2(np.abs(W[k:, j].astype(F32)))
                    if r_j >= tau:
                        sel, imax, found = j, imax_j, True
                        subs += 1
                        break
                if not found:
                    fallbacks += 1                       # PP fallback
                    # (includes truncated/empty windows at panel end)
            if sel != k:                                 # column swap -> Q
                W[:, [k, sel]] = W[:, [sel, k]]
                q_idx[[k, sel]] = q_idx[[sel, k]]
            pr = k + imax                                # row pivot   -> P
            if pr != k:
                W[[k, pr], :] = W[[pr, k], :]
                p_idx[[k, pr]] = p_idx[[pr, k]]

            piv = W[k, k]
            if k + 1 < n and piv != F16(0):
                if fp32acc:
                    l32 = W[k + 1:, k].astype(F32) / F32(piv)
                    W[k + 1:, k] = l32.astype(F16)       # round on store
                else:
                    W[k + 1:, k] = W[k + 1:, k] / piv    # fp16 division
                mm = float(np.abs(W[k + 1:, k].astype(F32)).max())
                if mm > max_mult:
                    max_mult = mm
                if k + 1 < p1:                           # update panel only
                    if fp32acc:
                        blk = W[k + 1:, k + 1:p1].astype(F32)
                        blk -= np.outer(W[k + 1:, k].astype(F32),
                                        W[k, k + 1:p1].astype(F32))
                        W[k + 1:, k + 1:p1] = blk.astype(F16)
                    else:
                        W[k + 1:, k + 1:p1] = (
                            W[k + 1:, k + 1:p1]
                            - np.outer(W[k + 1:, k], W[k, k + 1:p1]))
                    g = float(np.abs(W[k + 1:, k + 1:p1].astype(F32)).max())
                    if g > gmax:
                        gmax = g

        # ---------------- U12 solve + Schur complement update -------------
        if p1 < n:
            bw = p1 - p0
            if fp32acc:
                U12 = W[p0:p1, p1:].astype(F32)
                Lp = W[p0:p1, p0:p1].astype(F32)
                for t in range(bw - 1):                  # unit-lower solve
                    U12[t + 1:, :] -= np.outer(Lp[t + 1:, t], U12[t, :])
                W[p0:p1, p1:] = U12.astype(F16)          # round on store
                S = W[p1:, p1:].astype(F32)
                S -= W[p1:, p0:p1].astype(F32) @ W[p0:p1, p1:].astype(F32)
                W[p1:, p1:] = S.astype(F16)
            else:
                for t in range(bw - 1):                  # fp16 solve
                    W[p0 + t + 1:p1, p1:] = (
                        W[p0 + t + 1:p1, p1:]
                        - np.outer(W[p0 + t + 1:p1, p0 + t], W[p0 + t, p1:]))
                for t in range(bw):                      # fp16 rank-1 Schur
                    W[p1:, p1:] = (
                        W[p1:, p1:]
                        - np.outer(W[p1:, p0 + t], W[p0 + t, p1:]))
            g = float(np.abs(W[p0:p1, p1:].astype(F32)).max())
            if g > gmax:
                gmax = g
            g = float(np.abs(W[p1:, p1:].astype(F32)).max())
            if g > gmax:
                gmax = g

    # ---------------- FP64 verification ------------------------------------
    L = np.tril(W.astype(F64), -1) + np.eye(n)
    U = np.triu(W.astype(F64))
    Aref = A16.astype(F64)[p_idx][:, q_idx]              # P A Q
    res = np.linalg.norm(Aref - L @ U) / np.linalg.norm(A16.astype(F64))

    return dict(steps=steps, triggers=triggers, subs=subs,
                fallbacks=fallbacks, scanned=scanned, max_mult=max_mult,
                growth=gmax / g0, residual=float(res),
                finite=bool(np.isfinite(gmax)),
                p_idx=p_idx, q_idx=q_idx)


# --------------------------------------------------------------------------
# One experiment
# --------------------------------------------------------------------------
def run_one(cfg):
    family, n, rep, panel, tau, w, accumulate = cfg
    A = make_matrix(family, n, rep)
    t0 = time.perf_counter()
    st = dp_lu_fp16(A, panel, tau, w, accumulate)
    dt = time.perf_counter() - t0
    s, tr = st["steps"], st["triggers"]
    return dict(
        family=family, n=n, seed=rep, panel=panel, tau=tau, w=w,
        accumulate=accumulate, steps=s, triggers=tr, subs=st["subs"],
        fallbacks=st["fallbacks"], scanned=st["scanned"],
        trigger_pct=100.0 * tr / s,
        sub_pct=100.0 * st["subs"] / s,
        fallback_pct=100.0 * st["fallbacks"] / s,
        sub_success_given_trigger_pct=(100.0 * st["subs"] / tr) if tr
        else float("nan"),
        avg_scanned_per_step=st["scanned"] / s,
        max_mult=st["max_mult"], growth=st["growth"],
        residual=st["residual"], finite=st["finite"], time_s=dt)


# --------------------------------------------------------------------------
# Aggregation and tables
# --------------------------------------------------------------------------
def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(round(float(r["tau"]), 4), int(r["w"]))].append(r)
    out = []
    for (tau, w), rs in sorted(groups.items()):
        col = lambda name: np.array([float(x[name]) for x in rs])
        sgt = col("sub_success_given_trigger_pct")
        out.append(dict(
            tau=tau, w=w, runs=len(rs),
            trigger_pct=col("trigger_pct").mean(),
            sub_pct=col("sub_pct").mean(),
            fallback_pct=col("fallback_pct").mean(),
            sub_given_trig=float(np.nanmean(sgt)) if np.isfinite(sgt).any()
            else float("nan"),
            avg_scans=col("avg_scanned_per_step").mean(),
            max_mult=col("max_mult").max(),
            growth_mean=col("growth").mean(),
            growth_max=col("growth").max(),
            residual_gmean=float(10 ** np.log10(col("residual")).mean()),
            residual_max=col("residual").max()))
    return out


def _md_rows(recs):
    hdr = ("| tau | w | Trigger % | Subst. % | Fallback % | Scans/step "
           "| max|l_ik| | Residual (geo-mean) | Growth (mean/max) |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in recs:
        body += (f"| {r['tau']:g} | {r['w']} | {r['trigger_pct']:.2f} "
                 f"| {r['sub_pct']:.2f} | {r['fallback_pct']:.3f} "
                 f"| {r['avg_scans']:.3f} | {r['max_mult']:.4f} "
                 f"| {r['residual_gmean']:.2e} "
                 f"| {r['growth_mean']:.1f} / {r['growth_max']:.1f} |\n")
    return hdr + body


def _tex_sci(x):
    e = int(math.floor(math.log10(x))) if x > 0 else 0
    return f"${x / 10**e:.1f}\\times 10^{{{e}}}$"


def _tex_rows(recs):
    out = ""
    for r in recs:
        out += (f"{r['tau']:g} & {r['w']} & {r['trigger_pct']:.1f} "
                f"& {r['sub_pct']:.1f} & {r['fallback_pct']:.2f} "
                f"& {r['avg_scans']:.2f} & {r['max_mult']:.3f} "
                f"& {_tex_sci(r['residual_gmean'])} \\\\\n")
    return out


def make_tables(summary, default_tau=1.01, default_w=6):
    tau_slice = [r for r in summary if r["w"] == default_w]
    w_slice = sorted((r for r in summary
                      if math.isclose(r["tau"], default_tau, rel_tol=1e-9)),
                     key=lambda r: r["w"])
    md = ("### (a) tau-sweep at w = %d\n\n%s\n"
          "### (b) w-sweep at tau = %g\n\n%s\n"
          "### (c) Full grid\n\n%s"
          % (default_w, _md_rows(tau_slice), default_tau, _md_rows(w_slice),
             _md_rows(summary)))
    tex = (
        "\\begin{table}\\centering\n"
        "\\caption{Sensitivity of delayed pivoting to $\\tau$ and $w$ "
        "(pooled over families, sizes, panels, seeds; residual is the "
        "geometric mean of $\\|PAQ-LU\\|_F/\\|A\\|_F$).}\n"
        "\\begin{tabular}{@{}rrrrrrrl@{}}\\toprule\n"
        "$\\tau$ & $w$ & Trig.\\,\\% & Subst.\\,\\% & Fallb.\\,\\% & "
        "Scans/step & $\\max|\\ell_{ik}|$ & Residual\\\\ \\midrule\n"
        + _tex_rows(tau_slice) + "\\midrule\n" + _tex_rows(w_slice) +
        "\\bottomrule\\end{tabular}\\end{table}\n")
    return md, tex


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def selftest():
    A = make_matrix("normal", 128, 0)
    for tau, w, acc in [(1.01, 6, "fp32"), (1.5, 16, "fp32"),
                        (1.01, 6, "fp16")]:
        st = dp_lu_fp16(A, 32, tau, w, acc)
        assert sorted(st["p_idx"]) == list(range(128))
        assert sorted(st["q_idx"]) == list(range(128))
        assert st["max_mult"] <= 1.0 + 1e-6, st["max_mult"]
        assert st["residual"] < 1e-2, st["residual"]
        assert st["growth"] >= 1.0
    print("self-test passed: valid P/Q, max|l| <= 1, residual sane.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="quick")
    ap.add_argument("--accumulate", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="dp_sensitivity.csv")
    ap.add_argument("--taus", default=None, help="comma list, overrides grid")
    ap.add_argument("--ws", default=None, help="comma list, overrides grid")
    ap.add_argument("--analyze", default=None, metavar="CSV",
                    help="skip sweep; aggregate an existing raw CSV")
    ap.add_argument("--full-table", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.analyze:
        with open(args.analyze) as f:
            rows = list(csv.DictReader(f))
        stem = os.path.splitext(args.analyze)[0]
    else:
        taus = ([float(x) for x in args.taus.split(",")] if args.taus
                else TAUS_DEFAULT)
        ws = ([int(x) for x in args.ws.split(",")] if args.ws
              else WS_DEFAULT)
        p = PRESETS[args.preset]
        cfgs = [(fam, n, rep, panel, tau, w, args.accumulate)
                for n in p["ns"] for panel in p["panels"]
                for fam in FAMILIES for rep in range(p["reps"])
                for tau in taus for w in ws]
        total = len(cfgs)
        print(f"# {total} factorizations "
              f"(preset={args.preset}, accumulate={args.accumulate}). "
              f"Use --preset full for the paper numbers; set "
              f"OMP_NUM_THREADS=1 when --jobs > 1.")
        rows, t0 = [], time.perf_counter()
        stem = os.path.splitext(args.out)[0]
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            step = max(1, total // 100)

            def emit(row, i):
                rows.append(row)
                wr.writerow(row)
                if i % step == 0 or i == total:
                    el = time.perf_counter() - t0
                    print(f"[{i}/{total}] elapsed {el:7.1f}s "
                          f"eta {el / i * (total - i):7.1f}s", flush=True)
                    f.flush()

            if args.jobs > 1:
                with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                    futs = [ex.submit(run_one, c) for c in cfgs]
                    for i, fu in enumerate(as_completed(futs), 1):
                        emit(fu.result(), i)
            else:
                for i, c in enumerate(cfgs, 1):
                    emit(run_one(c), i)

    summary = aggregate(rows)
    with open(stem + "_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)
    md, tex = make_tables(summary)
    open(stem + "_tables.md", "w").write(md)
    open(stem + "_tables.tex", "w").write(tex)

    mm = max(float(r["max_mult"]) for r in rows)
    bad = sum(1 for r in rows if str(r["finite"]).lower() not in
              ("true", "1"))
    print(f"\nglobal max |l_ik| = {mm:.6f} (must be <= 1); "
          f"non-finite runs: {bad}")
    print(md if args.full_table else
          md.split("### (c)")[0])
    print(f"wrote {stem}_summary.csv, {stem}_tables.md, {stem}_tables.tex")


if __name__ == "__main__":
    main()
