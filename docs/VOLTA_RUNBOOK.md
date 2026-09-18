# Volta Runbook

Use secure SSH key or SSH-agent authentication. Never place a password in a
command and never transfer `../codes/pass.txt`.

## 1. Verify The Machine

```bash
ssh "$VOLTA_HOST" 'pwd'
ssh "$VOLTA_HOST" 'nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu --format=csv'
ssh "$VOLTA_HOST" 'nvcc --version'
ssh "$VOLTA_HOST" 'ls "$HOME"'
```

The publication target is one idle Tesla V100-PCIE-32GB. Select the intended
physical GPU explicitly with `CUDA_VISIBLE_DEVICES` and record the mapping.

## 2. Stage Source Only

Create a fresh timestamped remote root. Transfer this clean directory and the
frozen predictive source, excluding binaries, results, and credentials.

```bash
rsync -av \
  --exclude 'bin/' \
  --exclude 'build/' \
  --exclude 'smoke_*' \
  ./ \
  "$VOLTA_HOST:$REMOTE_ROOT/halfLU/"
```

## 3. Build For Volta

From `toms_revision_clean`, build with:

```bash
make CUDA_ARCH=sm_70
sha256sum blocked_predictive_validation pivoting_runtime_validation
sha256sum ../codes/dynamic_scaling_validation.cu pivoting_runtime_validation.cu
```

## 4. Smoke Checks

```bash
python3 check_smoke.py --binary ./pivoting_runtime_validation

./blocked_predictive_validation \
  --mode lu --family random --n 256 --panel-width 64 --samples 1 \
  --seed 20260823 --blocking blocked_fp16 --scaling on \
  --scaling-policy predictive --safe-threshold 32768 \
  --rank-safe-threshold 60000 --output smoke_predictive
```

## 5. Pivoting Runtime Campaign

Run serially on an idle selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_pivoting_runtime_campaign.py \
  --binary ./pivoting_runtime_validation \
  --results-root "$REMOTE_ROOT/results/pivoting_runtime"

python3 analyze_pivoting_runtime_campaign.py \
  "$REMOTE_ROOT/results/pivoting_runtime" \
  --output "$REMOTE_ROOT/analysis/pivoting_runtime"
```

The publication time is `factor_wall_ms`, not `factor_cuda_ms`, because the
pivot selectors include CPU work and synchronous transfers. The runner measures
both `unblocked_rank1` and `blocked_panel_local` using matched inputs.

## 6. Frozen Predictive Campaign

Do not rerun the expensive final campaign unless its arithmetic changes or its
provenance cannot be verified. If a rerun is necessary, use only:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_blocked_predictive_campaign.py \
  --binary ./blocked_predictive_validation \
  --results-root "$REMOTE_ROOT/results/predictive_final"
```

That runner enforces the publication panel profile `64/128/512/1024` and the
uniform 111-configuration, 327-run grid.

## 7. Retrieve Compact Evidence

Retrieve CSV, JSON, Markdown, logs, source, and hashes. Do not retrieve large
matrix snapshots unless a failed invariant requires them. Record GPU idleness
again after the campaign.
