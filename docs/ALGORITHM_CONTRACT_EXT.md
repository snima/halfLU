# Algorithm Contract — Table 7 Extension

This extends, and does not replace,
`ACM_TOMS_FINAL_SUBMISSION_CLEAN_PACKAGE/04_reproducibility_all_tables/table7_pivoting_runtime/ALGORITHM_CONTRACT.md`.
Every prohibition in that file still applies. The additions below are specific
to the fused device-side search path.

## Scope

This study measures **the cost of the pivot search, and nothing else.** It
introduces no new pivot rule, no new numerical property, and no change to the
factorization arithmetic. The two search paths are proven to select an identical
pivot sequence and to produce an identical factorization.

## Supported claims

* Under the `blocked_panel_local` schedule, the per-column and per-row argmax
  required by every one of the seven pivot rules can be obtained as a side effect
  of the trailing update, with one integer atomic per updated element and no
  additional memory traffic.
* The pivot decision, the row and column interchanges and the column scaling can
  all be driven from device-resident indices, so that the elimination step
  contains no host/device transfer and no device synchronisation.
* With that change, the measured overhead of panel-local CP relative to
  schedule-matched PP collapses by roughly two orders of magnitude. The
  previously reported figure was dominated by a per-step full-matrix
  device→host copy in the search branch, not by the arithmetic of the rule.
* The overhead of ScaP relative to PP likewise falls, because ScaP's cost was
  three synchronous `cudaMemcpy` calls per step, i.e. **latency**, not the
  streaming bandwidth of a trailing-column scan.
* Because the fused path also accelerates the PP baseline, the relative overhead
  of DP, GP, RP and ScPP may **increase**. Those increased figures are the more
  honest measurement of the rules' own cost.

## Prohibited claims

* **Do not** claim that CP is cheap in general. What is shown cheap is
  *panel-local* CP under the blocked schedule, whose search is restricted to
  `[k, panel_end) × [k, n)`. Global complete pivoting over the full trailing
  submatrix is a different algorithm; it additionally forces a per-step argmax
  over all remaining columns, which prevents the right-looking blocked
  formulation and therefore forfeits level-3 BLAS. Nothing here measures that.
* **Do not** claim any stability or accuracy improvement. The factorization is
  bit-identical to the frozen path; residuals, growth factors and
  `max|l_ik|` are unchanged by construction, and the manuscript's conclusion
  that reconstruction residuals exceed one at the largest orders regardless of
  pivot rule is untouched.
* **Do not** compare these timings with cuBLAS, cuSOLVER or MAGMA. The trailing
  update is still the naive frozen kernel: no shared-memory tiling, no register
  blocking, no Tensor Cores. All percentages are relative to a memory-bound
  update.
* **Do not** carry the fused overhead percentages over to a Tensor Core
  implementation. The fused atomics are a fixed cost per updated element; against
  an update that is 10–50× faster they would be a proportionally larger share.
  Re-measurement would be required.
* **Do not** replace the published Table 7 with these numbers. They are a
  *different measurement* of the same rules: cost of the rule rather than cost of
  the rule plus harness. If both are reported, label them as such.
* **Do not** attribute the predictive range certificate to any rule other than
  PP. Dynamic scaling is disabled here, exactly as in the parent study, and the
  observed GP and ScPP multipliers exceeding one still preclude it.
* **Do not** publish a fused-path timing whose `pivot_digest` has not been
  matched against the host path for the same cell. `verify_equivalence.py` is a
  precondition, not a formality.

## Required reporting

Any table or figure derived from this folder must state:

1. that both paths were run on matched inputs with the same binary;
2. the equivalence result (cells checked, mismatches);
3. that the metric is `factor_wall_ms`;
4. that the trailing update is unoptimised and not a BLAS comparison;
5. the panel-width profile used.

## Relationship to the manuscript text

Two statements in the current manuscript are inconsistent with the frozen source
and should be revised regardless of whether this extension is published:

* *"CP is an order of magnitude more expensive, as its worst-case complexity
  predicts."* Under the blocked schedule the measured CP is panel-local, with
  `O(w·n²)` total comparisons — cheaper than the `O(n³)` elimination. Its
  complexity predicts a negligible overhead. The measured cost came from the
  per-step `n × n` device→host copy.
* The ScaP bandwidth model `t_scan/t_gemm ≈ R/(βw) ≈ 24%`. The frozen ScaP branch
  issues three discrete `cudaMemcpy` calls per step; the cost is round-trip
  latency, not streaming bandwidth. The numerical agreement is coincidental.

Also: the Background section describes CP as incurring "irregular memory
access". Storage is column-major (`index_of(row, column, n) = column*n + row`),
so a **column** interchange is a contiguous, fully coalesced swap; it is the
**row** interchange, used by every rule including PP, that is strided.
