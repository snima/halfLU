# halfLU: Pivoting and Scaling Policies for Half-Precision LU Factorization

[![Status](https://img.shields.io/badge/Status-Under_Review-orange.svg)]()
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![CUDA 11.2+](https://img.shields.io/badge/CUDA-11.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Official open-source repository and reproducibility artifact package for the research paper:

> **"Pivoting and Scaling Policies for Half-Precision LU Factorization"**  
> **Authors:** [Nima Sahraneshinsamani](mailto:sahrans@uji.es)$^1$, [José I. Aliaga](mailto:aliaga@uji.es)$^1$, [Sandra Catalán](mailto:catalans@uji.es)$^1$, [José R. Herrero](mailto:josepr@ac.upc.edu)$^2$  
> $^1$ *Departament d'Enginyeria i Ciència dels Computadors, Universitat Jaume I, Castelló de la Plana, Spain*  
> $^2$ *Departament d'Arquitectura de Computadors, Universitat Politècnica de Catalunya, Barcelona, Spain*  
> **Status:** Preprint / Under review

---

## Table of Contents
1. [Overview & Problem Statement](#overview--problem-statement)
2. [Algorithmic Policies](#algorithmic-policies)
3. [Repository Layout](#repository-layout)
4. [Hardware & Software Requirements](#hardware--software-requirements)
5. [Quickstart: 30-Second Fast-Track CPU Reproduction](#quickstart-30-second-fast-track-cpu-reproduction)
6. [Detailed Reproduction Walkthrough](#detailed-reproduction-walkthrough)
   - [Table 2: Range-Completion Frontier](#table-2-range-completion-frontier)
   - [Table 3: FP16 Precision Wall](#table-3-fp16-precision-wall)
   - [Table 4: FP16 vs. FP32 Accumulation Residuals](#table-4-fp16-vs-fp32-accumulation-residuals)
   - [Table 5: Ledger-Aware Iterative Refinement](#table-5-ledger-aware-iterative-refinement)
   - [Table 7: Pivoting Execution Overheads on GPU (V100 & Fused Search)](#table-7-pivoting-execution-overheads-on-gpu)
   - [Figures 2, 3, 4, and 5 Reproduction](#figures-2-3-4-and-5-reproduction)
7. [Post-Submission Extension: Fused Device Pivot Search](#post-submission-extension-fused-device-pivot-search)
8. [Audits & Known Issues Disclosure](#audits--known-issues-disclosure)
9. [BibTeX Citation](#bibtex-citation)

---

## Overview & Problem Statement

Half-precision floating-point arithmetic (IEEE-754 **FP16**) delivers massive computing throughput on modern tensor accelerator hardware (e.g., NVIDIA Tensor Cores). However, its narrow dynamic range ($\approx [6.10 \times 10^{-5}, 65{,}504]$ with unit roundoff $u = 2^{-11} \approx 4.88 \times 10^{-4}$) poses severe numerical challenges for matrix factorizations:
1. **Dynamic Range Overflow:** Classical partial pivoting (PP) frequently overflows to $\pm\infty$ or $\mathrm{NaN}$ during unscaled factorization.
2. **Precision Wall:** Standard backward error bounds degrade proportionally to $n \cdot u$, rendering traditional error bounds vacuous as matrix dimensions grow toward $n \ge 2{,}048$.
3. **Heavyweight Search Bottlenecks:** Robust alternatives (such as complete pivoting CP or rook pivoting RP) require extensive multi-column or 2D searches ($O(n^3)$ cost), negating the performance benefits of low precision.

This paper establishes that **dynamic scaling and pivot selection should be treated as separate, adaptive solver policies**:
- **Dynamic Scaling (DS)** acts as the primary safeguard against dynamic range failure by keeping intermediate matrix entries bounded within the representable FP16 ceiling via exact power-of-two scaling.
- **Lightweight Pivoting Policies (DP, GP, ScaP)** intervene only when needed to control element growth and avoid suboptimal rank updates without paying the asymptotic overhead of 2D searches.

---

## Algorithmic Policies

### 1. Dynamic Scaling Policies
- **Diagonal Proxy Heuristic (DS):**  
  Exhaustive trailing submatrix scans cost $O(n^2)$ per panel. Instead, our proxy monitors the factored panel diagonal $d_{\max}^{(k)} = \max_{j \in \text{panel}} |a_{jj}^{(k)}|$ at negligible $O(w)$ cost. Whenever $d_{\max}^{(k)} > t = 2^{-2} = 0.25$, trailing columns are rescaled by exact power-of-two factors:
  $$s_i = 2^{\lceil \log_2(d_{\max}^{(k)} / t) \rceil}$$
  Because scaling adjusts only IEEE-754 exponent bits without mantissa rounding, normal floating-point numbers incur zero arithmetic rounding distortion.
- **Predictive Majorant Guard:**  
  Monitors pre-update column certificates $c_j^+ = c_j^{(k)} + |l_{ij}| \cdot \|u_{k, :}\|_1$ before rank-1 or block updates. Rescaling intervenes only when majorant ceilings threaten the overflow ceiling ($T_{\mathrm{rank}} = 60{,}000$, $T_{\mathrm{block}} = 32{,}768$).

### 2. Lightweight Pivoting Policies
All rules are governed by the unified step growth inequality:
$$L_{k+1} \le (1 + \mu_k \alpha_k) L_k$$

| Policy | Acronym | Trigger Mechanism | Multiplier Bound $\mu_k$ | Search Cost |
|---|---|---|---|---|
| **Delayed Pivoting** | DP | Triggers only when candidate ratio $M_1 / M_2 < \tau = 1.01$; scans up to $w_\ell = 6$ columns for clear pivot separation | $1/\tau$ (on success) | Panel-local $O(w_\ell n)$ |
| **Geometric Pivoting** | GP | Triggers on near-ties ($M_1 / M_2 < \tau$); resolves ties by projecting candidate rows onto the active row tail | $< \tau$ | $O(n)$ row inner product |
| **Scattered Pivoting** | ScaP | Selects among current ($k$), middle ($k + \lfloor(n-k)/2\rfloor$), and trailing ($n$) columns to maximize spatial diversity | $\le 1$ | 3 column scans |
| **Partial Pivoting** | PP | Standard row partial pivoting baseline | $1$ | 1 column scan |
| **Scaled Partial Pivoting** | ScPP | Row maximum normalized partial pivoting | $\le 1$ | 1 column + row norms |
| **Rook Pivoting** | RP | Alternating row and column maximum search | $\le 1$ | $1.35$ scans/step avg |
| **Complete Pivoting** | CP | Full 2D active submatrix search | $\le 1$ | $O((n-k)^2)$ per step |

> **Method Ordering Convention:** Throughout all scripts, figures, and tabular comparisons, methods appear in the fixed order:
> $$\mathbf{DP \longrightarrow GP \longrightarrow ScaP \longrightarrow PP \longrightarrow ScPP \longrightarrow RP \longrightarrow CP}$$

---

## Repository Layout

```text
halfLU_official_repo/
├── README.md                      # This comprehensive ACM TOMS evaluation guide
├── index.html                     # Interactive HTML datasheet for GitHub Pages (https://snima.github.io/halfLU)
├── LICENSE                        # Dual Apache 2.0 (Code) & CC-BY-4.0 (Docs/Data/Figures)
├── CITATION.cff                   # Machine-readable citation metadata
├── Makefile                       # Top-level build orchestration (sm_70, sm_80, sm_90)
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment file
│
├── high_perf_cuda/                # 38 TFLOPS Hardware Tensor Core Benchmark Suite (Table 5 Engine)
│   ├── include/                   # High-throughput kernels (cublasGemmEx, TRSM, fused panel, transposed LU)
│   ├── src/                       # main_benchmark.cu, gemm_benchmark.cu
│   ├── scripts/                   # run_volta_campaign.py, run_comprehensive_benchmark.py
│   ├── CMakeLists.txt             # Native CMake build system
│   ├── ARCHITECTURAL_OPTIMIZATIONS_AND_PROVENANCE.md # Deep technical disclosure on 38 TFLOPS engine
│   └── README.md                  # Detailed build and micro-benchmarking manual
│
├── cuda/                          # Native FP16 Standalone CUDA validators
│   ├── dynamic_scaling_validation.cu      # Table 2 & Table 6 predictive scaling validator
│   ├── dynamic_scaling_validation_nla.cu  # Table 3, 4, 5 NLA extension (Frobenius residual & IR TRSV)
│   ├── pivoting_runtime_validation.cu     # Table 7 frozen baseline (7 pivoting policies, host search)
│   └── pivoting_search_cost_validation.cu # Table 7 extension (fused device-side pivot search)
│
├── scripts/                       # Reproduction, runner, and plotting scripts
│   ├── run_all_figures_and_tables.sh      # Master one-click fast CPU reproduction script (< 30s)
│   ├── plot_paper_figures.py              # Figure generation suite
│   ├── plot_final_predictive_campaign.py  # Predictive campaign plotter (Table 2 & 6)
│   ├── plot_clean_performance_matched.py  # Schedule-matched performance plotter
│   ├── plot_nla_extension.py              # Precision wall, IR, growth envelope plotter (Table 3, 4, 5)
│   ├── make_toms_pivoting_figures.py      # Pivoting panel-local runtime story plotter (Table 7)
│   ├── plot_dp_sensitivity.py             # Delayed Pivoting sensitivity plotter (Table 8 / Fig 5)
│   ├── generate_custom_order_plots.py     # Custom policy ordering plotter (DP->GP->ScaP->PP->ScPP->RP->CP)
│   ├── verify_equivalence.py              # Bit-exact FNV-1a digest equivalence proof
│   ├── run_nla_predictive_campaign.py     # GPU campaign runner for Table 2 & 6
│   ├── run_nla_extension_campaign.py      # GPU campaign runner for Table 3, 4, 5
│   ├── run_pivoting_runtime_campaign.py   # GPU campaign runner for Table 7 frozen baseline
│   ├── run_search_cost_campaign.py        # GPU campaign runner for Table 7 fused search
│   └── sweep_dp_fp16.py                   # CPU sweep for Delayed Pivoting sensitivity
│
├── data/                          # Audited ground-truth benchmark datasets
│   ├── benchmark_results_custom_order.csv # Table 5: 38 TFLOPS GPU throughput & overheads
│   ├── execution_time_table.csv           # Table 5: Wall-clock execution times (ms) across orders
│   ├── nla_final_configurations.csv       # Table 2 & 6 (111 configurations, 327 runs through n=61,440)
│   ├── nla_extension_configurations.csv   # Table 3, 4, 5 (50 configurations, 150 runs through n=61,440)
│   ├── v100_pivoting_combined_raw.csv     # Table 7 frozen raw runs (1,300 runs on Tesla V100)
│   ├── v100_pivoting_blocked_summary.csv  # Table 7 aggregated medians (75 rows)
│   ├── full_scale_search_cost_summary.csv # Table 7 EXT fused search medians (n=512 to 61440)
│   ├── equivalence_report_v100.csv        # Bit-level digest verification log (196/196 cells verified)
│   ├── dp_sensitivity_full.csv            # Table 8 parameter sweep (6,720 factorizations)
│   └── dp_sensitivity_full_summary.csv    # Table 8 aggregated summary (56 rows)
│
├── docs/                          # Technical dossiers, contracts, and runbooks
│   ├── index.html                         # GitHub Pages HTML documentation entrypoint
│   ├── supplement.pdf                     # Extended mathematical derivations
│   ├── ARTIFACT_EVALUATION.md             # Dedicated step-by-step ACM reviewer guide
│   ├── HARDWARE_REQUIREMENTS.md           # Hardware specs (V100/Volta/Ampere/Hopper)
│   ├── REPRODUCIBILITY_MANIFEST.md        # Master manifest mapping all paper numbers to code
│   ├── ALGORITHM_CONTRACT.md              # Scope contract for pivoting & predictive scaling studies
│   ├── ALGORITHM_CONTRACT_EXT.md          # Contract for fused device pivot search
│   ├── OPTIMIZATION_NOTES.md              # CUDA kernel micro-optimizations
│   ├── V100_PIVOTING_FINAL_REPORT.md      # Detailed report on Table 7 V100 execution
│   └── SHA256_PROVENANCE_ISSUE.md         # Full audit disclosure on host memory management
│
└── figures/                       # EXACT figures appearing in manuscript & reviewer response
    ├── paper/                             # Manuscript figures (main_pivoting_and_scaling.pdf)
    │   ├── 4_WP_ds_gray_H.png             # Figure 1(a): Without DS
    │   ├── 4_After_ds_H.png               # Figure 1(b): With DS
    │   ├── fig2_panel_proxy_ratios.pdf    # Figure 2: Trailing-to-diagonal ratio R_k across k/n
    │   ├── dense_story_composite_4row.pdf # Figure 3: Dense synthetic errors (4 rows)
    │   ├── heatmaps/                      # Figure 4: SuiteSparse refined error heatmaps
    │   └── dp_sensitivity_analysis.pdf    # Figure 5: Delayed Pivoting sensitivity (tau, w)
    └── response/                          # Reviewer response figures (response_to_reviewers.pdf)
        ├── fig2_panel_proxy_ratios.pdf    # Revised Figure 2 (876 panels)
        ├── pivoting_overhead_reviewer_style.pdf # Response Figure 2: Overhead (DP->GP->ScaP->PP->ScPP->RP->CP)
        └── peak_performance_reviewer_style.pdf # Peak throughput across dimensions
```

---

## Hardware & Software Requirements

| Operating Mode | Required Hardware | Software Dependencies | Expected Runtime |
|---|---|---|---|
| **CPU Fast Track** | Standard x86_64 CPU ($\ge 2$ cores) | Python $\ge 3.8$, NumPy, Pandas, Matplotlib, SciPy | **$< 30$ seconds** |
| **GPU Full Campaign** | NVIDIA Tesla V100-PCIE-32GB (or compatible Volta/Ampere/Hopper GPU) | CUDA Toolkit $\ge 11.2$ (`nvcc`, `cuBLAS`), C++17 | $\approx 2\text{--}4$ hours |

For hardware details, memory capacity requirements, and architectural flags, see [`docs/HARDWARE_REQUIREMENTS.md`](docs/HARDWARE_REQUIREMENTS.md).

---

## Quickstart: 30-Second Fast-Track CPU Reproduction

To reproduce all publication figures and tabular metrics without requiring a GPU or CUDA:

```bash
# 1. Clone the repository
git clone https://github.com/snima/halfLU.git
cd halfLU

# 2. Install minimal Python dependencies
pip install -r requirements.txt

# 3. Execute master one-click reproduction script
bash scripts/run_all_figures_and_tables.sh
```

### What Happens:
1. Verifies the integrity of the 6 audited ground-truth datasets in `data/`.
2. Regenerates all 16 publication PDF vector graphics in `figures/`.
3. Verifies and displays the exact numerical metrics for Table 2, Table 3, Table 4, Table 5, and Table 7 in the terminal.

---

## Detailed Reproduction Walkthrough

### Table 2: Range-Completion Frontier
- **Scientific Claim:** On challenging graded-wide matrices with wide dynamic ranges ($A = DB$), unscaled FP16 LU factorizations suffer catastrophic overflow. A naive post-panel oracle check completes only **3/36** configurations (failing at all $n \ge 512$), whereas predictive scaling completes **36/36** configurations through $n = 61{,}440$.
- **Ground-Truth Data:** `data/nla_final_configurations.csv` (111 configurations, 327 runs)
- **Fast CPU Verification:**
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_csv('data/nla_final_configurations.csv')
  gw = df[df['family'] == 'graded_wide']
  for v in ['oracle', 'predictive']:
      sub = gw[gw['variant'] == v]
      print(f'{v:<12} completed runs: {sub[\"completed_count\"].sum()}/{sub[\"samples\"].sum()}')
  "
  ```
  *Output:* `oracle completed runs: 3/36` | `predictive completed runs: 36/36`
- **Output Figures:** `figures/final_predictive_outcomes.pdf`, `figures/final_numerical_frontier.pdf`
- **Full GPU Campaign (Tesla V100):**
  ```bash
  make kernels CUDA_ARCH=sm_70
  python3 scripts/run_nla_predictive_campaign.py --binary bin/blocked_predictive_validation --results-root ./campaign_predictive
  python3 scripts/analyze_final_predictive_campaign.py ./campaign_predictive --output ./analysis_predictive
  python3 scripts/plot_final_predictive_campaign.py ./analysis_predictive/configurations.csv --output figures/
  ```

---

### Table 3: FP16 Precision Wall
- **Scientific Claim:** Under sequential FP16 accumulation, the normalized relative Frobenius residual grows strictly proportional to $n \cdot u$:
  $$r_F = \frac{\|A - \hat{L}\hat{U}\|_F}{\|A\|_F} \approx c \cdot n \cdot u$$
  where empirical constants cluster tightly at $c \approx 0.033\text{--}0.042$ on i.i.d. random ensembles and $c \approx 0.048\text{--}0.060$ on graded-wide ensembles across all dimensions $512 \le n \le 61{,}440$.
- **Ground-Truth Data:** `data/nla_extension_configurations.csv` (50 configurations, 150 runs)
- **Fast CPU Verification:**
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_csv('data/nla_extension_configurations.csv')
  b16 = df[df['blocking'] == 'blocked_fp16']
  for fam in ['iid_random', 'graded_wide']:
      print(f'Family: {fam}')
      sub = b16[b16['family'] == fam].sort_values('n')
      for _, r in sub.iterrows():
          print(f'  n = {int(r[\"n\"]):<6} | r_F = {r[\"fro_residual\"]:.4e} | r_F/(nu) = {r[\"residual_over_nu\"]:.4f}')
  "
  ```
- **Output Figure:** `figures/nla_precision_wall.pdf`

---

### Table 4: FP16 vs. FP32 Accumulation Residuals
- **Scientific Claim:** Upgrading the internal matrix multiply accumulation from sequential FP16 to FP32 dramatically lowers the residual: at $n = 61{,}440$, the relative Frobenius residual drops from **1.7910 to 0.2185** on graded-wide matrices, and from **1.1187 to 0.1712** on random controls.
- **Ground-Truth Data:** `data/nla_extension_configurations.csv`
- **Fast CPU Verification:**
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_csv('data/nla_extension_configurations.csv')
  for fam in ['graded_wide', 'iid_random']:
      print(f'Family: {fam} at n=61,440:')
      fp16 = df[(df['family'] == fam) & (df['blocking'] == 'blocked_fp16') & (df['n'] == 61440)]['fro_residual'].values[0]
      fp32 = df[(df['family'] == fam) & (df['blocking'] == 'blocked_fp32_accumulate') & (df['n'] == 61440)]['fro_residual'].values[0]
      print(f'  FP16 acc: r_F = {fp16:.4f}  -->  FP32 acc: r_F = {fp32:.4f}')
  "
  ```
  *Output:*  
  `graded_wide at n=61,440: FP16 acc: r_F = 1.7910 --> FP32 acc: r_F = 0.2185`  
  `iid_random  at n=61,440: FP16 acc: r_F = 1.1187 --> FP32 acc: r_F = 0.1712`

---

### Table 5: Ledger-Aware Iterative Refinement
- **Scientific Claim:** Using the scaling ledger to un-scale intermediate triangular solves in iterative refinement (IR) enables convergence to machine precision on random controls through $n = 20{,}480$, and delivers backward error $\approx 1.25 \times 10^{-10}$ on graded-wide matrices.
- **Ground-Truth Data:** `data/nla_extension_configurations.csv`
- **Output Figure:** `figures/nla_iterative_refinement.pdf`

---

### Table 7: Pivoting Execution Overheads on GPU
- **Scientific Claim:** On Tesla V100 with scaling disabled (isolating pivot-search overhead), panel-local heuristics add minimal runtime relative to partial pivoting (PP):
  - In the **published frozen baseline** ($n=2048$), DP adds $+2.98\%$, GP adds $+2.22\%$, and ScaP adds $+25.29\%$. Complete pivoting (CP) shows an inflated $+3017.78\%$ overhead due to host/device memory transfer synchronization.
  - In the **post-submission Fused Device Pivot Search extension (`TABLE7_EXT`)**, device-resident 64-bit `atomicMax` key packing eliminates round-trip transfers entirely: CP overhead drops from $+3042.56\%$ to **$+4.48\%$** (an $80.8\times$ speedup), while ScaP drops to **$+2.49\%$**.
  - At large dimension ($n = 60\text{Ki}$), $O(w n^2)$ search amortizes completely against $O(n^3)$ GEMM updates, dropping overheads below **$0.52\%$** for all proposed rules (DP $+0.52\%$, GP $+0.21\%$, ScaP $+0.17\%$).
- **Ground-Truth Data:** `data/v100_pivoting_blocked_summary.csv`, `data/full_scale_search_cost_summary.csv`
- **Fast CPU Verification:**
  ```bash
  python3 scripts/make_toms_pivoting_figures.py data/v100_pivoting_blocked_summary.csv --paper-output figures/ --analysis-output figures/
  python3 scripts/generate_custom_order_plots.py --output figures/
  ```
- **Output Figures:** `figures/pivoting_panel_local_story.pdf`, `figures/pivoting_overhead_reviewer_style.pdf`, `figures/peak_performance_reviewer_style.pdf`

---

### Figures 2, 3, 4, and 5 Reproduction

1. **Figure 2: Trailing-to-Diagonal Proxy Ratio $\mathcal{R}_k = T_{\max}^{(k)} / d_{\max}^{(k)}$**  
   Evaluates 876 recorded panels across 4 matrix ensembles (Gaussian, uniform, row-graded $10^6$ and $10^{12}$) up to $n = 20{,}480$. Medians stay near unity ($1.10\text{--}1.33$) with $\max \mathcal{R}_k \le 2.492$, guaranteeing substantial empirical headroom below FP16 overflow ($65{,}504$).  
   Vector plot: [`figures/fig2_panel_proxy_ratios.pdf`](figures/fig2_panel_proxy_ratios.pdf).

2. **Figure 3: Dense Synthetic Errors (4 Rows)**  
   Evaluates backward and forward errors across 6 baseline families and 7 pivoting rules before and after refinement (IR) with and without dynamic scaling (DS).  
   Vector plot: [`figures/dense_story_composite_4row.pdf`](figures/dense_story_composite_4row.pdf).

3. **Figure 4: SuiteSparse Heatmaps & Coverage**  
   Demonstrates that DS raises finite refined-error coverage on SuiteSparse matrices from **26/63 to 55/63** matrix-method pairs ($\tau = 1.01$).

4. **Figure 5: Sensitivity Analysis of Delayed Pivoting (DP)**  
   Evaluates near-tie threshold $\tau \in [1.001, 1.50]$ and lookahead depth $w_\ell \le 16$ across 6,720 factorizations. Confirms that default parameters ($\tau = 1.01, w_\ell = 6$) reside on a wide, stable numerical plateau.  
   Re-run on CPU:
   ```bash
   python3 scripts/plot_dp_sensitivity.py --summary-csv data/dp_sensitivity_full_summary.csv --output-dir figures/
   ```
   Vector plot: [`figures/dp_sensitivity_analysis.pdf`](figures/dp_sensitivity_analysis.pdf).

---

## Post-Submission Extension: Fused Device Pivot Search

In response to in-depth analysis of Table 7, a post-submission extension (`TABLE7_EXT`) was developed to eliminate host-device synchronization artefacts during pivot search:
1. **The Issue:** The initial frozen validator (`cuda/pivoting_runtime_validation.cu`) performed pivot scans on the host, paying 6–8 synchronous `cudaDeviceSynchronize` round trips and PCIe memory transfers per elimination step.
2. **The Fused Architecture (`cuda/pivoting_search_cost_validation.cu`):**
   - Captures column/row argmax values as a side effect of the trailing rank update using a single 64-bit `atomicMax` on a monotone packed key `(magnitude, ~index)`.
   - Drives all pivot decisions, row swaps, and scalings from device-resident indices with **zero host synchronizations and zero PCIe memory transfers**.
3. **Bit-Exact Equivalence Verification:**
   Equivalence is formally verified via `pivot_digest` (FNV-1a hash over the entire realized pivot sequence).
   ```bash
   python3 scripts/verify_equivalence.py --binary bin/pivoting_search_cost_validation --quick
   ```
   *Result:* 196/196 test cells identical; zero mismatches. Logged in [`data/equivalence_report_v100.csv`](data/equivalence_report_v100.csv).

---

## Audits & Known Issues Disclosure

In adherence to open-science rigor, full transparency is provided for all audit artifacts:
1. **Master Manifest:** Every single table and figure is mapped to its exact source script, data row, and command in [`docs/REPRODUCIBILITY_MANIFEST.md`](docs/REPRODUCIBILITY_MANIFEST.md).
2. **Provenance Audit:** Full disclosure of a host memory management optimization patch (`const` removal to free a temporary 15 GB buffer early at $n=61{,}440$ without altering GPU computational kernels) is detailed in [`docs/SHA256_PROVENANCE_ISSUE.md`](docs/SHA256_PROVENANCE_ISSUE.md).

---

## BibTeX Citation

```bibtex
@article{sahraneshin2026pivoting,
  title  = {Pivoting and Scaling Policies for Half-Precision {LU} Factorization},
  author = {Sahraneshinsamani, Nima and Aliaga, Jos{\'e} I. and Catal{\'a}n, Sandra and Herrero, Jos{\'e} R.},
  note   = {Preprint / Under review},
  year   = {2026},
  url    = {https://github.com/snima/halfLU}
}
```

---

## License

This project is licensed under a dual open-source model:
* **Source Code & Scripts (CUDA, C++, Python):** Licensed under the [Apache License, Version 2.0](LICENSE).
* **Documentation, Benchmark Datasets & Figures:** Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

