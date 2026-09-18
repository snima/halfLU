# ACM TOMS Artifact Evaluation Guide

**Paper Title:** Pivoting and Scaling Policies for Half-Precision LU Factorization  
**Authors:** Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, José R. Herrero  
**Target Journal:** ACM Transactions on Mathematical Software (TOMS)  
**Artifact Badges Claimed:**
- **Artifacts Available**
- **Artifacts Evaluated — Functional**
- **Artifacts Evaluated — Reusable**

---

## 1. Overview and Artifact Structure

This artifact package contains the complete, peer-reviewed codebase, benchmark datasets, and analysis pipelines supporting the paper.

The repository provides two distinct, non-conflicting evaluation tracks:
1. **Fast-Track CPU Reproduction ($\le 30$ seconds):**  
   Re-generates all 16 publication figures and tabular metrics directly from the independently audited, aggregated benchmark CSV datasets in `data/`. Does **not** require a GPU or CUDA.
2. **Full GPU Benchmark Reproduction:**  
   Re-compiles the native FP16 CUDA kernels from source and executes the end-to-end factorizations across synthetic and application matrices through $n = 61{,}440$. Validated on NVIDIA Tesla V100-PCIE-32GB and compatible Volta/Ampere/Hopper architectures.

```text
halfLU_official_repo/
├── Makefile                       # Top-level build orchestration
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment definition
├── cuda/                          # Native FP16 CUDA kernels
│   ├── dynamic_scaling_validation.cu      # Predictive scaling validator (Table 2 & 6)
│   ├── dynamic_scaling_validation_nla.cu  # NLA extension validator (Table 3, 4, 5)
│   ├── pivoting_runtime_validation.cu     # Frozen baseline for 7 pivoting policies (Table 7)
│   └── pivoting_search_cost_validation.cu # Fused device pivot search extension (Table 7 EXT)
├── scripts/                       # Reproduction, runner, and plotting scripts
├── data/                          # Audited benchmark CSV datasets
├── docs/                          # Technical dossiers, contracts, and runbooks
└── figures/                       # Pre-generated and freshly reproducible vector plots (PDF/PNG)
```

---

## 2. Quickstart: 30-Second Fast-Track Evaluation (CPU Only)

### Step 1: Environment Setup
```bash
# Clone and enter the repository
cd halfLU_official_repo

# Create and activate Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Master Reproduction Script
```bash
bash scripts/run_all_figures_and_tables.sh
```

### Expected Output
The script automatically:
1. Verifies dataset integrity across `data/*.csv`.
2. Recomputes and re-plots all publication figures in `figures/`.
3. Validates and displays the exact numerical metrics reported in the paper:
   - **Table 2:** Confirms that predictive scaling completes **36/36** graded-wide configurations through $n = 61{,}440$, whereas the post-panel oracle completes only **3/36**.
   - **Table 3:** Validates the Frobenius precision wall $r_F \approx c \cdot n \cdot u$ ($c \approx 0.033\text{--}0.042$ for i.i.d. random and $c \approx 0.048\text{--}0.060$ for graded-wide matrices).
   - **Table 7:** Recomputes the median overhead of panel-local pivoting policies on Tesla V100 at $n = 2048$ (DP $+2.98\%$, GP $+2.22\%$, ScaP $+25.29\%$) and displays the dramatic reduction when using the fused device-resident pivot search (CP overhead falls from $+3042.56\%$ down to $+4.48\%$, an $80.8\times$ acceleration).

---

## 3. Full GPU Campaign Reproduction (Requires NVIDIA GPU)

### Step 1: Compile CUDA Validators
```bash
# Default architecture is sm_70 (Tesla V100)
make kernels CUDA_ARCH=sm_70

# For Ampere (A100):
# make kernels CUDA_ARCH=sm_80
```

### Step 2: Table 2 & Table 6 Campaign (Predictive Scaling)
```bash
python3 scripts/run_nla_predictive_campaign.py \
    --binary bin/blocked_predictive_validation \
    --results-root ./campaign_predictive

python3 scripts/analyze_final_predictive_campaign.py ./campaign_predictive --output ./analysis_predictive
python3 scripts/plot_final_predictive_campaign.py ./analysis_predictive/configurations.csv --output figures/
```

### Step 3: Table 3, 4, 5 Extension Campaign (Precision Wall & Refinement)
```bash
python3 scripts/run_nla_extension_campaign.py \
    --binary bin/nla_extension_validation \
    --results-root ./campaign_extension

python3 scripts/summarize_nla_extension.py ./campaign_extension --output ./analysis_extension
python3 scripts/plot_nla_extension.py ./analysis_extension/nla_extension_configurations.csv --output figures/
```

### Step 4: Table 7 Extension (Fused Device Pivot Search & Bit-Exact Verification)
```bash
# Verify bit-level equivalence (FNV-1a pivot digest) between host and fused device search
python3 scripts/verify_equivalence.py --binary bin/pivoting_search_cost_validation --quick

# Run full-scale search cost campaign
python3 scripts/run_search_cost_campaign.py \
    --binary bin/pivoting_search_cost_validation \
    --results-dir ./campaign_search_cost

python3 scripts/analyze_search_cost.py ./campaign_search_cost/search_cost_runs.csv --output ./analysis_search_cost
```

---

## 4. Method Ordering Convention

Throughout all response plots and paper revisions, the method order convention is strictly maintained:
$$\mathbf{DP \longrightarrow GP \longrightarrow ScaP \longrightarrow PP \longrightarrow ScPP \longrightarrow RP \longrightarrow CP}$$

To regenerate these specific comparison figures:
```bash
python3 scripts/generate_custom_order_plots.py --output figures/
```
Output files:
- `figures/peak_performance_reviewer_style.pdf`
- `figures/peak_performance_up_to_10k.pdf`
- `figures/pivoting_overhead_reviewer_style.pdf`
- `figures/pivoting_overhead_up_to_10k.pdf`

---

## 5. Artifact Evaluation Checklist

| Claim / Artifact | Script Chain | Data Source | Output Figure / Table | Status |
|---|---|---|---|---|
| **Table 2 / Table 6** | `plot_final_predictive_campaign.py` | `data/nla_final_configurations.csv` | `final_predictive_outcomes.pdf`, `final_numerical_frontier.pdf`, `final_performance_ablation.pdf` | Verified |
| **Table 3, 4, 5** | `plot_nla_extension.py` | `data/nla_extension_configurations.csv` | `nla_precision_wall.pdf`, `nla_iterative_refinement.pdf`, `nla_growth_envelope.pdf` | Verified |
| **Table 7 (Frozen)** | `make_toms_pivoting_figures.py` | `data/v100_pivoting_blocked_summary.csv` | `pivoting_panel_local_story.pdf`, `pivoting_cost_hierarchy.pdf` | Verified |
| **Table 7 (Fused EXT)** | `run_search_cost_campaign.py` | `data/full_scale_search_cost_summary.csv` | `full_scale_search_cost_table.md`, `host_vs_fused_overhead_comparison.pdf` | Verified |
| **Figure 2 (Proxy Ratios)**| Vector graphic deploy | 876 recorded panels | `fig2_panel_proxy_ratios.pdf` | Verified |
| **Delayed Pivoting Sensitivity** | `plot_dp_sensitivity.py` | `data/dp_sensitivity_full_summary.csv` | `dp_sensitivity_analysis.pdf` | Verified |
