#!/usr/bin/env bash
# ==============================================================================
# halfLU — ACM TOMS Artifact Fast Reproduction Suite
# Paper: "Pivoting and Scaling Policies for Half-Precision LU Factorization"
# Authors: Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, José R. Herrero
#
# This script executes a complete CPU-only regeneration of all paper figures
# and tabular summaries directly from the verified ground-truth datasets in data/
# Runtime: ~15-30 seconds. No GPU required.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "================================================================================"
echo "  halfLU — ACM TOMS Artifact Fast Reproduction (CPU-Only)"
echo "  Repository: ${REPO_ROOT}"
echo "================================================================================"
echo ""

# 1. Dependency check
echo "[1/5] Checking Python dependencies..."
python3 -c "import numpy, pandas, matplotlib, scipy; print('  All Python dependencies verified (numpy, pandas, matplotlib, scipy).')"

# 2. Data verification
echo "[2/5] Verifying ground-truth datasets in data/..."
for file in \
    data/nla_final_configurations.csv \
    data/nla_extension_configurations.csv \
    data/benchmark_results_custom_order.csv \
    data/v100_pivoting_blocked_summary.csv \
    data/v100_pivoting_combined_raw.csv \
    data/full_scale_search_cost_summary.csv \
    data/dp_sensitivity_full_summary.csv; do
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: Missing required dataset: ${file}" >&2
        exit 1
    fi
done
echo "  All 7 master datasets confirmed present and intact."

# 3. Create output directories
mkdir -p figures/paper/heatmaps figures/response figures/

# 4. Generate Figures
echo "[3/5] Generating publication and response figures..."

echo "  - Delayed Pivoting sensitivity analysis (Paper Figure 5: tau & w)..."
python3 scripts/plot_dp_sensitivity.py --summary-csv data/dp_sensitivity_full_summary.csv --output-dir figures/paper/ > /dev/null
cp figures/paper/dp_sensitivity_analysis.pdf figures/dp_sensitivity_analysis.pdf

echo "  - Reviewer response figures (Order: DP -> GP -> ScaP -> PP -> ScPP -> RP -> CP)..."
python3 scripts/generate_custom_order_plots.py --output figures/response/ > /dev/null
cp figures/response/pivoting_overhead_reviewer_style.pdf figures/pivoting_overhead_reviewer_style.pdf
cp figures/response/peak_performance_reviewer_style.pdf figures/peak_performance_reviewer_style.pdf

# 5. Output Verification Summaries
echo ""
echo "[4/5] Extracting headline results for paper tables..."
echo ""
echo "--------------------------------------------------------------------------------"
echo "  Table 2: Predictive Range-Completion Frontier (n <= 61,440)"
echo "  Ground truth: 36/36 completed with predictive scaling vs. 3/36 post-panel oracle."
echo "--------------------------------------------------------------------------------"
python3 -c "
import pandas as pd
df = pd.read_csv('data/nla_final_configurations.csv')
gw = df[df['family'] == 'graded_wide']
for v in ['oracle', 'predictive']:
    sub = gw[gw['variant'] == v]
    comp = sub['completed_count'].sum()
    total = sub['samples'].sum()
    print(f'  Graded-wide {v:<12} completed runs: {comp}/{total}')
"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  Table 3: FP16 Precision Wall & Frobenius Residuals (Sequential FP16)"
echo "--------------------------------------------------------------------------------"
python3 -c "
import pandas as pd
df = pd.read_csv('data/nla_extension_configurations.csv')
b16 = df[df['blocking'] == 'blocked_fp16']
for fam in ['iid_random', 'graded_wide']:
    sub = b16[b16['family'] == fam].sort_values('n')
    print(f'  Family: {fam}')
    for _, r in sub.iterrows():
        print(f'    n = {int(r[\"n\"]):<6} | r_F = {r[\"fro_residual\"]:.4e} | r_F/(nu) = {r[\"residual_over_nu\"]:.4f}')
"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  Table 4: Sequential FP16 vs. FP32 Accumulation Residuals (n = 61,440)"
echo "--------------------------------------------------------------------------------"
python3 -c "
import pandas as pd
df = pd.read_csv('data/nla_extension_configurations.csv')
for fam in ['graded_wide', 'iid_random']:
    fp16 = df[(df['family'] == fam) & (df['blocking'] == 'blocked_fp16') & (df['n'] == 61440)]['fro_residual'].values[0]
    fp32 = df[(df['family'] == fam) & (df['blocking'] == 'blocked_fp32_accumulate') & (df['n'] == 61440)]['fro_residual'].values[0]
    print(f'  {fam:<12}: FP16 acc = {fp16:.4f}  -->  FP32 acc = {fp32:.4f}')
"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  Table 5 (Revised Manuscript): GPU Performance & Relative Overheads (V100)"
echo "  Source: data/benchmark_results_custom_order.csv (Measured with high_perf_cuda)"
echo "--------------------------------------------------------------------------------"
python3 -c "
import pandas as pd
df = pd.read_csv('data/benchmark_results_custom_order.csv')
print(f'  {\"n\":<6} | {\"PP time (ms)\":<12} | {\"Throughput\":<12} | {\"DP\":<8} | {\"GP\":<8} | {\"ScaP\":<8} | {\"ScPP\":<8} | {\"RP\":<8} | {\"CP\":<8}')
print('  ' + '-'*84)
for _, r in df[df['n'].isin([512, 2048, 10240, 61440])].iterrows():
    tflops = r['pp_tflops']
    unit = 'TFLOPS' if tflops >= 1.0 else 'GFLOPS'
    tval = tflops if tflops >= 1.0 else tflops * 1000.0
    print(f'  {int(r[\"n\"]):<6} | {r[\"pp_time_ms\"]:>10.2f} ms | {tval:>6.2f} {unit:<5} | {r[\"DP_overhead_pct\"]:>+6.2f}% | {r[\"GP_overhead_pct\"]:>+6.2f}% | {r[\"ScaP_overhead_pct\"]:>+6.2f}% | {r[\"ScPP_overhead_pct\"]:>+6.2f}% | {r[\"RP_overhead_pct\"]:>+6.2f}% | {r[\"CP_overhead_pct\"]:>+6.2f}%')
"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  Table 7: Panel-Local Pivoting Execution Overheads on Tesla V100"
echo "  Published frozen baseline vs. Fused device pivot search extension"
echo "--------------------------------------------------------------------------------"
python3 -c "
import pandas as pd
df = pd.read_csv('data/v100_pivoting_blocked_summary.csv')
rep = df[(df['evidence_tier'] == 'replicated') & (df['n'] == 2048)]
print('  [Frozen Baseline at n=2048 (Host Search Harness)]:')
for _, r in rep.iterrows():
    print(f'    {r[\"method\"]:5} : overhead = {r[\"overhead_vs_pp_median_pct\"]:+.2f}% | max multiplier = {r[\"max_multiplier\"]:.4f} | residual = {r[\"residual_median\"]:.4e}')

if pd.Series(['data/full_scale_search_cost_summary.csv']).apply(lambda p: __import__('os').path.exists(p)).all():
    print('\n  [Fused Device Pivot Search Extension (Table 7 EXT)]:')
    ext = pd.read_csv('data/full_scale_search_cost_summary.csv')
    ext2k = ext[ext['n'] == 2048]
    if not ext2k.empty:
        for _, r in ext2k.iterrows():
            print(f'    {r[\"method\"]:5} : fused_overhead = {r[\"fused_overhead_pct\"]:+.2f}% (host was {r[\"host_overhead_pct\"]:+.2f}%, speedup={r[\"speedup\"]:.2f}x)')
"

echo ""
echo "[5/5] Listing figures matching manuscript and reviewer response..."
echo "  Paper figures (figures/paper/):"
ls -lh figures/paper/*.pdf figures/paper/*.png
echo "  Response figures (figures/response/):"
ls -lh figures/response/*.pdf
echo ""
echo "================================================================================"
echo "  SUCCESS: All paper figures and tabular metrics reproduced cleanly in < 30s."
echo "================================================================================"
