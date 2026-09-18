import pandas as pd
import numpy as np

def df_to_md(df):
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---:" if i > 0 else ":---" for i in range(len(cols))) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.environ.get("HALFLU_CAMPAIGN_CSV", os.path.join(SCRIPT_DIR, "..", "results_volta_campaign.csv"))
df = pd.read_csv(csv_path)

print("### TABLE 1: Execution Time (ms) on Tesla V100 32GB across Matrix Dimensions")
pivot_ms = df.pivot(index="n", columns="method", values="gpu_ms")[["PP", "DP", "GP", "ScaP", "ScPP", "CP", "RP"]]
formatted_ms = pivot_ms.reset_index()
for col in ["PP", "DP", "GP", "ScaP", "ScPP", "CP", "RP"]:
    formatted_ms[col] = formatted_ms[col].map("{:.2f} ms".format)
print(df_to_md(formatted_ms))

print("\n### TABLE 2: Compute Throughput (TFLOPS) on Tesla V100")
pivot_tflops = df.pivot(index="n", columns="method", values="tflops")[["PP", "DP", "GP", "ScaP", "ScPP", "CP", "RP"]].reset_index()
for col in ["PP", "DP", "GP", "ScaP", "ScPP", "CP", "RP"]:
    pivot_tflops[col] = pivot_tflops[col].map("{:.3f}".format)
print(df_to_md(pivot_tflops))

print("\n### TABLE 3: Relative Overhead vs PP Baseline (%) on Tesla V100")
pp_series = pivot_ms["PP"]
overhead_pct = pivot_ms.apply(lambda col: ((col - pp_series) / pp_series) * 100.0)
overhead_str = overhead_pct.reset_index()
for col in ["PP", "DP", "GP", "ScaP", "ScPP", "CP", "RP"]:
    if col == "PP":
        overhead_str[col] = "0.0% (Base)"
    else:
        overhead_str[col] = overhead_str[col].apply(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
print(df_to_md(overhead_str))

legacy_pp = {
    2048: 54.12,
    4096: 208.43,
    8192: 1179.0,
    10240: 2224.0,
    20480: 18417.0,
    40960: 148309.0
}

print("\n### TABLE 4: Speedup Comparison: New High-Performance CUDA vs Legacy Code on Tesla V100")
speedup_rows = []
for n, old_ms in legacy_pp.items():
    if n in pivot_ms.index:
        new_ms = pivot_ms.loc[n, "PP"]
        old_gflops = (2.0/3.0 * n**3) / (old_ms * 1e-3) / 1e9
        new_gflops = (2.0/3.0 * n**3) / (new_ms * 1e-3) / 1e9
        speedup = old_ms / new_ms
        speedup_rows.append({
            "Matrix Order n": n,
            "Legacy Time": f"{old_ms:.1f} ms" if old_ms < 1000 else f"{old_ms/1000.0:.2f} s",
            "Legacy GFLOPS": f"{old_gflops:.1f}",
            "New Time": f"{new_ms:.2f} ms" if new_ms < 1000 else f"{new_ms/1000.0:.2f} s",
            "New GFLOPS": f"{new_gflops:.1f}",
            "New TFLOPS": f"{new_gflops/1000.0:.3f}",
            "Speedup Factor": f"{speedup:.2f}x"
        })
print(df_to_md(pd.DataFrame(speedup_rows)))

print("\n### TABLE 5: Activation Counters Across Matrix Dimensions")
counters_df = df[df["method"].isin(["DP", "GP", "ScaP", "RP"])][["n", "method", "dp_accepts", "dp_fallbacks", "gp_near_ties", "gp_second", "scap_cur", "scap_mid", "scap_last", "rp_iters"]]
print(df_to_md(counters_df))
