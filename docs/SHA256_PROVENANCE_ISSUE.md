# Known Reproducibility Issue: `dynamic_scaling_validation.cu` checked-in-file drift

**Severity: Low (results unaffected) — but must be fixed before final submission for
exact reproducibility.**

## The problem

The file currently in `cuda/dynamic_scaling_validation.cu`
matches `lowhost_source_sha256` where host-side memory staging was optimized.

Verified directly:

```
$ sha256sum new_experiments/dynamic_scaling_validation.cu
783db5465258b1b4c4cd49cfa651baa04c108e40177ea4d1df26c7b7da044c3d

$ sha256sum new_experiments/volta_nla_final_.../source/dynamic_scaling_validation.cu
1b5bbe518a1aab2d5acb866e8402e8e231867a5ddf18c67893dc167ccf3be597

$ grep sha256 new_experiments/volta_nla_final_.../RUN_PROVENANCE.txt
primary_source_sha256: 1b5bbe518a1aab2d5acb866e8402e8e231867a5ddf18c67893dc167ccf3be597
lowhost_source_sha256: 783db5465258b1b4c4cd49cfa651baa04c108e40177ea4d1df26c7b7da044c3d
```

So the file checked into the live tree matches `lowhost_source_sha256`, **not**
`primary_source_sha256` — i.e. rebuilding `blocked_predictive_validation` from the
live tree today does not reproduce the exact binary that generated the published
numbers.

## What actually changed (verified by direct diff)

```diff
2237c2237
<     const std::vector<float> host_float = make_lu_input(...);
---
>     std::vector<float> host_float = make_lu_input(...);
2250a2251,2254
>     // The source-precision staging matrix is no longer needed after conversion.
>     // Releasing it here avoids retaining O(n^2) host memory throughout the GPU
>     // factorization (about 15 GiB at n=61440).
>     std::vector<float>().swap(host_float);
```

This is a **host-side memory-management change only** (drops a `const` qualifier
so the staging buffer can be freed early, then explicitly frees it). It does not
touch any GPU kernel, arithmetic, pivoting logic, or scaling decision. This matches
`RUN_PROVENANCE.txt`'s own description of the lowhost variant as "host-staging-only,
GPU kernels unchanged."

## Practical impact
- **Numerically:** none expected — no computational code path differs.
- **Formally / for the reproducibility claim:** the checked-in file does not match
  the exact source hash recorded as authoritative in the campaign's own provenance
  record. A strict reviewer (or a future co-author) who tries to verify byte-exact
  provenance will find a mismatch.

## Recommended fix (not yet applied — original folders were left untouched per
instruction)
Either:
1. **Restore the frozen file as the checked-in one**, i.e. copy
   `source/dynamic_scaling_validation.cu` (hash `1b5bbe51...`) back over
   `new_experiments/dynamic_scaling_validation.cu` and `codes/dynamic_scaling_validation.cu`,
   and record in a commit message / CHANGES.md that the lowhost memory optimization
   was reverted for provenance cleanliness; **or**
2. **Explicitly adopt the lowhost variant going forward**, update
   `RUN_PROVENANCE.txt`/`ALGORITHM_CONTRACT.md` to name `783db546...` as the new
   `primary_source_sha256`, and add one sentence to the artifact/supplementary
   material stating that the shipped source includes a memory-management-only
   patch relative to the exact binary used for the reported campaign, with the
   diff above quoted verbatim as evidence that no computational path changed.

Option 1 is simpler and is what we recommend, since it makes the live tree and the
already-published numbers trivially consistent again with zero risk.

## What this reproducibility package does about it
The file copied into
`06_reproducibility/table2_table6_predictive_scaling/dynamic_scaling_validation.cu`
is the **frozen, hash-verified** `source/dynamic_scaling_validation.cu`
(`1b5bbe51...`), i.e. the one that actually produced the published campaign —
**not** the currently-live file in `new_experiments/`. This was a deliberate choice
so that this package is self-consistent and byte-exact with the published numbers,
independent of the drift described above in the original working tree.
