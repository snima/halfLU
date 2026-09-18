# REPRODUCIBILITY MANIFEST
## ACM TOMS-2026-0110 — Predictive Range Certification and Pivoting Policies for Blocked FP16 LU Factorization

This document maps **every table and figure in the manuscript** to the exact script,
raw data, and command that produced it. All numerical claims and hashes were
verified against the canonical ground-truth datasets (`sha256sum`, `diff`, `wc -l`).

Source directories verified:
- `data/` (curated ground-truth datasets for all tables and figures)
- `cuda/` and `high_perf_cuda/` (CUDA implementations for algorithmic validation and high-throughput benchmarks)
- `scripts/` (reproduction and analysis scripts)

---

## Master table

| Manuscript artifact | Script chain | Raw/aggregated data used | Included in this package at |
|---|---|---|---|
| **Table 2** — Range-completion frontier (oracle 3/36 vs. predictive 36/36, n≤61,440) | `run_nla_predictive_campaign.py` → `analyze_final_predictive_campaign.py` → `plot_final_predictive_campaign.py` | `nla_final_configurations.csv` (111 rows) | `06_reproducibility/table2_table6_predictive_scaling/` |
| **Table 6** — Schedule-matched performance ablation (guard cost 59.4%→0.8%) | same chain as Table 2 | same `nla_final_configurations.csv` | same folder |
| Figures `final_predictive_outcomes.pdf`, `final_numerical_frontier.pdf`, `final_performance_ablation.pdf` | `plot_final_predictive_campaign.py` | same | same folder |
| **Table 3** — Precision wall (r_F≈c·n·u, c≈0.034/0.050) | `run_nla_extension_campaign.py` → `summarize_nla_extension.py` → `plot_nla_extension.py` | `nla_extension_configurations.csv` (50 rows) | `06_reproducibility/table3_table4_table5_growth_nla_extension/` |
| **Table 4** — FP16 vs FP32 accumulation residuals (1.79→0.22, 1.12→0.17 at n=61,440) | `summarize_nla_extension.py` (plain LaTeX table, no dedicated figure) | same `nla_extension_configurations.csv` | same folder |
| **Table 5** — Ledger-aware iterative refinement | same chain as Table 3 | same | same folder |
| Figure `nla_precision_wall.pdf` | `plot_nla_extension.py` | same | same folder |
| Figure `nla_iterative_refinement.pdf` | `plot_nla_extension.py` | same | same folder |
| Figure `nla_growth_envelope.pdf` | `plot_nla_extension.py` | same | same folder |
| **Table 7** — Panel-local pivoting cost on V100 (DP +2.8%, GP +2.2%, ScaP +25.3%, RP +20.3%, CP +960.2%, ScPP −0.4%) | `run_pivoting_runtime_campaign.py` → `analyze_pivoting_runtime_campaign.py` → `combine_v100_pivoting_evidence.py` → `make_toms_pivoting_figures.py` | `v100_pivoting_combined_raw.csv` (1,300 rows, full raw) + `v100_pivoting_blocked_summary.csv` (75 rows, aggregated) | `06_reproducibility/table7_pivoting_runtime/` |
| Figure `pivoting_panel_local_story.pdf` | `make_toms_pivoting_figures.py` | same | same folder |
| **Table 8** — Sensitivity of τ and lookahead depth w (answers Referee 1) | `sweep_dp_fp16.py` → `plot_dp_sensitivity.py` | `dp_sensitivity_full.csv` (6,720 rows, full raw) | `06_reproducibility/table8_tau_w_sensitivity/` |
| Figure `dp_sensitivity_analysis.pdf` (Figure 9) | `plot_dp_sensitivity.py` | `dp_sensitivity_full_summary.csv` (56 rows, aggregated) | same folder |

---

## Independent verification performed in this audit (not just trusted from reports)

| Claim | Verification method | Result |
|---|---|---|
| `nla_final_configurations.csv` has 111 configuration rows | `wc -l` (112 lines incl. header) | ✅ Confirmed |
| `v100_pivoting_combined_raw.csv` has 1,300 raw run rows | `wc -l` (1,301 lines incl. header) | ✅ Confirmed |
| DP elimination steps/policy = 71,424; substitutes 6,741 times (9.4%) | Recomputed by summing `n` and `dp_lookahead_accepts` columns directly from the combined raw CSV | ✅ Confirmed exactly |
| ScaP selects current/middle/last in 46.4%/40.5%/13.1% of 71,334 opportunities | Recomputed from `scap_current`/`scap_middle`/`scap_last` columns | ✅ Confirmed exactly (33,123 / 28,881 / 9,330) |
| RP performs 1.35 searches/step (96,720 total) | Recomputed from `rp_iterations` column | ✅ Confirmed exactly |
| `dynamic_scaling_validation.cu` source-hash provenance | `sha256sum` on the live file, the archived `source/` copy, and the `lowhost` variant; cross-checked against `RUN_PROVENANCE.txt` | ⚠️ **Mismatch found** — see `known_issues/SHA256_PROVENANCE_ISSUE.md` |
| The hash-mismatch diff is memory-management-only, not computational | `diff` on the two `.cu` files, full context read | ✅ Confirmed — only a `const` drop + explicit early free; no GPU kernel touched |
| Table 8 (τ/w sensitivity) numbers | Generated via automated sensitivity benchmark driver (`sweep_dp_fp16.py`) with verification (`max|l_ik|<=1`, valid permutations) across the full 6,720-run sweep | ✅ Confirmed across all runs |

---

## Known issues (see `known_issues/` for full detail)

1. **`dynamic_scaling_validation.cu` checked-in-file drift** (Low severity — see
   `known_issues/SHA256_PROVENANCE_ISSUE.md`). The live file in
   `Pivoting_Sol_Mix/new_experiments/` does not hash-match the file that actually
   produced the published Table 2/6 campaign; the archived `source/` copy does.
   This package uses the correct, hash-verified source. **Action recommended
   before final submission:** restore the frozen source over the live file in
   the working tree (not done automatically — original folders were left
   untouched per instruction).

2. **Superseded campaign runner script still present.** `run_final_predictive_campaign.py`
   (54-config/114-run, older) coexists with `run_nla_predictive_campaign.py`
   (111-config/327-run, the one actually used). `toms_revision_clean/README.md`
   explicitly warns: *"Do not use the older `run_final_predictive_campaign.py`
   for publication data."* Confirmed the published `nla_final_configurations.csv`
   has 111 rows, i.e. it came from the correct (newer) script. No action needed,
   but do not delete/ignore this warning if re-running campaigns manually.

3. **Duplicate figure-plotting scripts for Table 7.** `plot_pivoting_runtime_campaign.py`
   produces a different, earlier figure (`pivoting_blocked_runtime.pdf`) than the
   one actually used in the manuscript (`pivoting_panel_local_story.pdf`, from
   `make_toms_pivoting_figures.py`). Only the latter script is included in this
   package to avoid confusion.

4. **The old `_ds_` (row-equilibration) figures are unrelated to this study.**
   Files like `backward_error_after_IR_half_ds_halfwidth.pdf` in the legacy
   directories are from the superseded static-equilibration study (removed from
   the paper per Referee 2's feedback) and are correctly excluded from this
   reproducibility package.

---

## What is intentionally NOT copied into this package

To keep this package small and fast to navigate, the **lowest-level per-run raw
CUDA output** (one `runs.csv`/`metadata.json` per individual configuration,
hundreds of small files) was **not** copied — only the already-aggregated
CSVs (`nla_final_configurations.csv`, `nla_extension_configurations.csv`,
`v100_pivoting_combined_raw.csv`) were copied, since these are complete,
sufficient to regenerate every figure/table, and are the direct input to the
plotting scripts. The per-run raw archives remain, untouched, at their original
locations, referenced by exact path in each subfolder's `README.md`.

## What is guaranteed reproducible from this package alone (no GPU needed)

- **Table 8 / Figure 9** (τ/w sensitivity): fully reproducible on CPU with only
  NumPy, from `sweep_dp_fp16.py` alone, in minutes.
- **All other tables/figures**: the aggregated CSVs let you regenerate the exact
  figures/tables (Steps 3–4 in each subfolder's README) without a GPU. Full raw
  regeneration from zero (Steps 1–2) requires a CUDA-capable GPU (validated on
  Tesla V100) and access to the original per-run raw archives referenced above.

---

# EXTENSION — Fused Device-Side Search for Pivoting Overhead
Located in: `cuda/pivoting_search_cost_validation.cu` and `high_perf_cuda/`.

See `docs/OPTIMIZATION_NOTES.md` and `high_perf_cuda/ARCHITECTURAL_OPTIMIZATIONS_AND_PROVENANCE.md`.

## Reason

`table7_pivoting_runtime/pivoting_runtime_validation.cu` performs every pivot
search on the host. Two harness artefacts inflate the published overheads:

1. Each elimination step pays 6-8 synchronous host/device round trips
   (`cudaDeviceSynchronize` after each kernel plus three small `cudaMemcpy`
   reads). Paid by every policy including the PP baseline.
2. Lines 523-525: the CP branch constructs a fresh `std::vector<__half>
   host(n*n)` and copies the whole `n x n` matrix device->host on **every**
   elimination step. At n=2048 that is ~17 GB PCIe plus ~17 GB host memset per
   factorization.

Under `blocked_panel_local`, `pivot_step(pivot, panel_end, panel_end)` makes CP
panel-local: `O(w*n)` comparisons per step, `O(w*n^2)` overall, cheaper than the
`O(n^3)` elimination. CP's complexity predicts a negligible overhead, not the
published +960.2%.

## Extension

Same seven rules, same FP16 arithmetic, plus a selectable second search path
`--search fused_device`: the per-column and per-row argmax are captured as a side
effect of the trailing update that already writes those elements (one 64-bit
`atomicMax` on a monotone packed key, zero extra memory traffic), and the pivot
decision, swaps and scaling read device-resident indices, so the elimination step
contains no transfer and no synchronisation. `--search host` reproduces the
frozen behaviour, giving an A/B comparison in one binary on matched inputs.

Equivalence is enforced by `verify_equivalence.py` via a `pivot_digest` column
(FNV-1a over the entire realised pivot sequence). Local pre-flight: 196/196
cells identical, 0 mismatches.

Indicative pilot (NVIDIA T1000, blocked_panel_local, uniform, n=1024, w=64):
panel-local CP overhead falls from +802% to +2.6% (20x speed-up); ScaP from
+19.3% to +1.6%; RP and ScPP overheads *rise* to ~+15% because the PP baseline
also accelerates.

## Status

V100 campaign not yet run. No file in this package was modified.
