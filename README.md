<!-- ============================================================ -->
<!--  halfLU — datasheet-style README                             -->
<!-- ============================================================ -->

# halfLU — Pivoting and Scaling Policies for Half-Precision LU

[![Paper](https://img.shields.io/badge/paper-TOMS%202026-1F4E79?style=flat-square)](paper/main.tex)
[![Supplement](https://img.shields.io/badge/supplement-derivations-1B5E4B?style=flat-square)](paper/supplement.tex)
[![Code license](https://img.shields.io/badge/code-MIT-243B53?style=flat-square)](LICENSE)
[![Text license](https://img.shields.io/badge/paper-CC--BY--4.0-243B53?style=flat-square)](LICENSE)

> Lightweight pivoting and exact power-of-two scaling that keep LU
> factorization inside the narrow FP16 dynamic range — close to partial
> pivoting's cost, approaching complete pivoting's stability on many
> (not all) problems.

---

## At a glance — FP16 working envelope

| Spec | Value |
|---|---|
| Format | IEEE-754 binary16 (FP16) |
| Stored / effective significand | 10 / 11 bits |
| Unit roundoff `u` | `2^-11 ≈ 4.88e-4` |
| Normal range | `≈ 6.1e-5 … 6.55e4` |
| Overflow ceiling | `65504` |
| Scaling trigger `t` | `2^-2` (exact, exponent-only) |
| Default pivot threshold `τ` / window `w` | `1.01` / `6` |
| Reference hardware | NVIDIA V100, `sm_70`, CUDA 11.2 |

---

## What's here

Four policies, all analyzed through one inequality
`L_{k+1} ≤ (1 + μ_k·α_k)·L_k`, where `μ_k` bounds the multiplier and `α_k`
bounds the pivot-row tail. Each rule is just a choice of `(μ_k, α_k)`.

| Rule | `μ_k` | `α_k` | Growth bound `ρ` |
|---|---|---|---|
| `PP` (baseline) | `1` | `1` | `2^{n-1}` |
| `DP` Delayed Pivoting | `1/τ` (on success) | `1` | `(1+1/τ)^s · 2^{n-1-s}` |
| `GP` Geometric Pivoting | `<τ` | `√(c_k^{-2}-1)` | `∏(1+μ_k√(c_k^{-2}-1))` |
| `ScaP` Scattered Pivoting | `1` | `1` | `2^{n-1}` |

Plus **`DS` — Dynamic Scaling**: panel-diagonal magnitude is used as a cheap
`O(w)` proxy for trailing-block overflow risk, triggering an *exact*
power-of-two column rescaling. Full step-by-step derivations are in
[`paper/supplement.tex`](paper/supplement.tex).

---

## Results at a glance (honest)

On the SuiteSparse-derived tests at `τ=1.01`, dynamic scaling raises finite
refined-error coverage from **26/63 to 55/63** (matrix × method pairs). It is
the main robustness lever; conditioning is the main limit.

It is **not** a universal fix:
- `pores_2`, `west1505` — strongly pivot-dependent (RP / CP win).
- `b2_ss` — unsolved by every tested policy, with or without DS.
- On dense families, DS makes most cases usable; HHP21 and the near-singular
  families stay limited by growth / conditioning, not by overflow.

The takeaway the paper argues for: **expose adaptive pivoting and scaling
policies, not a single classical default.**

---

## Reproduce

```bash
git clone https://github.com/<org>/halfLU.git
cd halfLU
python -m pip install -r requirements.txt   # numpy, scipy, pandas, matplotlib

# 1. Dense synthetic sweep (GPU, native FP16)
make GPU=v100                 # builds the CUDA kernels
bash experiments/run_dense.sh # τ-sweep over n ∈ {64,128,256,512,1024,2048}

# 2. SuiteSparse-derived tests (MATLAB FP16 simulation)
matlab -batch "cd experiments; run_sparse"

# 3. Regenerate every figure from results/
python figures/make_figures.py
