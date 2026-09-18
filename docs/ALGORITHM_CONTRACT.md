# Algorithm Contract

This workspace contains two deliberately separate validation paths.

## Primary Path

`blocked_predictive_validation` is built from the frozen final source at
`../codes/dynamic_scaling_validation.cu`.

- Pivoting: partial row pivoting only.
- Storage and updates: FP16.
- Blocked update: separate FP16 multiplication and sequential FP16 subtraction.
- Bound state: FP32 storage with an upward-rounded device-FP64 temporary in the
  current implementation.
- Scaling: predictive complete-column powers of two.
- Thresholds: `T_rank=60000`, `T_block=32768`.
- Valid panel widths: at most 1024.
- Column permutations: none in the validated factorization.

The publication panel-width profile is:

| Matrix order | Panel width |
|---:|---:|
| `n <= 4096` | 64 |
| `4096 < n <= 10240` | 128 |
| `10240 < n <= 20480` | 512 |
| `n > 20480` | 1024 |

This is the only path to which the final `3/36` versus `36/36` result applies.

## Secondary Pivoting Path

`pivoting_runtime_validation` compares PP, DP, GP, ScaP, RP, CP, and ScPP with
scaling disabled under two matched schedules:

- `unblocked_rank1`: the global definitions in the reviewed manuscript;
- `blocked_panel_local`: a right-looking blocked LU in which cross-column
  searches and GP row-tail norms are restricted to the remaining panel.

The unblocked schedule is required to preserve the original manuscript definitions of
the cross-column methods. During a conventional blocked PP panel, columns
outside the panel do not receive the panel's internal updates until U12/A22 is
formed. DP, GP, ScaP, RP, and CP inspect those columns. Applying them directly
inside a wide PP panel would therefore select pivots from stale data and define
a different algorithm. The blocked schedule therefore labels every such rule
as panel-local. It uses the same `64/128/512/1024` panel-width profile as the
primary campaign, forms U12 rankwise, and applies one sequential-FP16 blocked
A22 update per panel.

The secondary timings include host pivot decisions, required host/device
transfers, row/column swaps, division, and FP16 trailing updates. They are
implementation timings, not optimized MAGMA or cuSOLVER performance.

## Prohibited Claims And Combinations

- Do not apply the blocked predictive PP certificate to GP or ScPP.
- Do not attribute predictive completion results to any secondary pivot rule.
- Do not call the old `half_ds` code dynamic scaling; it performs static
  two-sided equilibration.
- Do not describe panel-local variants as numerically identical to the global
  pivoting methods.
- Compare blocked predictive timing only with a schedule-matched blocked PP
  control, not with global unblocked pivoting.
- Do not use a panel width above 1024 with the current blocked threshold.
- Do not call a reconstruction residual a solve residual or a stability proof.

## Future Extension

A predictive rank-1 engine for column-pivoted methods would require a bound
weighted by the observed multiplier maximum, synchronized swaps of the matrix,
majorant, exponent ledger, and logical-column permutation, and new experiments.
It would be a new validated method and is outside the frozen final campaign.
