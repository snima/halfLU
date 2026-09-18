# Fused-Device Search: Fairness Optimization Pass (2026-08-24)

## Why this file exists

The published comparison fuses the search for all seven rules, but does not
put equal engineering effort behind each one: PP/ScaP/CP/RP already needed
zero extra kernel launches per elimination step, while DP/GP/ScPP each paid
one extra launch, and RP's rook walk paid one avoidable atomic per iteration
inside its serial dependency chain. Before attributing DP/GP/RP/ScPP's
overhead to "the cost of the rule," each rule needed a best-effort pass so
the comparison is rule-vs-rule, not rule-vs-effort-invested.

## Provenance

| | |
|:--|:--|
| Base file | `pivoting_search_cost_validation.cu`, sha256 `275f2d2d1af68a3729049b06c0ccab3bbdb463ca03fb4fe6e7f82a68f5e20788` |
| Backup of base (unmodified) | `backups/pivoting_search_cost_validation.cu.bak_20260824_203618` (same sha256 as above) |
| File after this pass | `pivoting_search_cost_validation.cu`, sha256 `5e191bb59b9ca014439a6d575ee705532b0ec91e5aa873b444c3fb1956003173` |
| Binary built from it | `pivoting_search_cost_validation_opt` (`nvcc -O3 -arch=sm_75`, local NVIDIA T1000; rebuild with `-arch=sm_70` for V100) |

**The `--search host` path is untouched by every change below.** Only
`SearchMode::kFusedDevice` behaviour changes. `select_pivot()` (the frozen
CPU reference) was not edited.

## Changes made, and why each is safe

| Rule | Change | Mechanism removed | Why it cannot change `pivot_digest` |
|:--|:--|:--|:--|
| RP | `select_pivot_device_kernel`, method_code 4: the rook-walk loop's `atomicAdd(&counters[kSlotRpIterations], 1ull)` moved from once-per-iteration to a local register count flushed with **one** `atomicAdd` after the walk resolves. | One atomic round-trip per rook-walk iteration, serialised inside the pointer-chasing dependency chain. | The counter is diagnostic-only; it is never read by the selection logic. Same total, same pivot, same digest -- fewer atomics on the latency-bound critical path. |
| GP | `select_pivot_device_kernel`, method_code 2: the single-column top-two reduction (previously `column_top_two_kernel<<<1,kThreads>>>`, launched every step) is now a preamble inside this kernel, transplanted verbatim (same loads, same merge tree, same block size). | One kernel launch per elimination step. | The merge is a MAX-with-tie-break under a total order -- associative and order-independent by construction (same argument the original code already relied on). GP's serial FP32 row-norm accumulation (the part whose rounding must match the host) is **untouched**; only the order-independent argmax preamble moved. |
| ScPP | `select_pivot_device_kernel`, method_code 6: the weighted argmax (previously `scaled_column_argmax_kernel<<<1,kThreads>>>`) is now the same preamble, verbatim, same `double` division, same tie-break. | One kernel launch per elimination step. | Same reduction, same operand order, same precision path (`double`, not switched to a cached reciprocal -- that WOULD risk a different last-bit result and was deliberately not done). |
| DP | **Not merged.** Still calls the standalone `column_top_two_kernel<<<count,kThreads>>>`. | -- | `count = min(window+1, panel_end-k)` depends on the runtime `--window` value. GP/ScPP's merge is safe only because their block already used all `kThreads` for one column; a generic in-block partition for DP's variable `count` needs either a hard compile-time cap (a latent correctness trap if `--window` is later raised) or a cooperative-groups grid-wide redesign. Left as documented future work rather than shipped with an unverified size assumption. |

## Correctness verification performed

```
python3 verify_equivalence.py --binary ./pivoting_search_cost_validation_opt \
    --report equivalence_report_opt_t1000.csv
```

**Result (2026-08-24, NVIDIA T1000, CC 7.5, CUDA 12.0): 784/784 cells
equivalent, 0 mismatches** -- the same grid (2 schedules x 7 families x 7
methods x 4 orders x 2 seeds) as the pre-existing gate, run against the
optimized binary. `pivot_digest`, `reconstruction_residual`, `growth_factor`
and all activation counters agree with `--search host` for every cell,
exactly as required before any timing from this binary is trusted.

## Performance verification: honest status, NOT publication-grade yet

Quick A/B samples on the **same local, shared, desktop T1000** (median of 7-15
repetitions) show a directionally plausible improvement for GP (~1-4% lower
overhead) and a smaller, less consistent one for ScPP, consistent with
removing one ~microsecond-scale kernel launch per elimination step. **These
numbers are not reliable enough to report or publish as-is**: rules I did
NOT touch at all (ScaP, CP, whose code paths are byte-identical to the base
file) fluctuated by up to 5-6% between back-to-back runs on this machine,
which is system noise (desktop GPU shared with the display server and other
GUI processes) of the same order as the effect being measured, not signal.

**Required next step before any number from this pass is used:** rerun
`run_search_cost_campaign.py` + `analyze_search_cost.py` with this binary,
on an *idle, dedicated* GPU (V100, per the benchmark protocol's own
idleness check), with the same seeds/orders/repetitions as the published
grid, and report old-binary-vs-new-binary overhead side by side -- exactly
the same rigor already applied to the host-vs-fused comparison.

## What this does and does not license

* It licenses the claim "DP/GP/RP/ScPP received a documented best-effort
  optimization pass before their overhead was attributed to the rule."
* It does **not** license claiming a specific new percentage for GP/ScPP/RP
  until the campaign above is run on an idle GPU.
* It does not change any published Table 7 (EXT) number; those were
  generated by the pre-optimization binary and remain what they are.
