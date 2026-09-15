#!/usr/bin/env bash
# Finish one setting's Round 1 candidates in parallel, one candidate per GPU.
#
# Only the candidates still missing a metrics.csv for this setting are launched,
# and each is launched at most once, so this neither re-runs nor skips work.
set -euo pipefail

cd "$HOME/niuyiming/PhaseFormer"
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
FROZEN=research_runs/structured_lowrank_round0_v1/frozen_configs.json
OUT=research_runs/structured_lowrank_round1_scratch
LOGDIR="$HOME/niuyiming/structured_lowrank_round1_logs"
SETTING="$1"; shift
GPUS=("$@")
mkdir -p "$LOGDIR"

DATASET="${SETTING%%:*}"
HORIZON="${SETTING##*:}"
DATASET_KEY=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

ALL=(A_period_lowrank A_period_lowrank_r8 B_segment_basis C_level_shape D_recent_sparse
     E_separable matched_A_period_lowrank matched_A_period_lowrank_r8 matched_B_segment_basis
     matched_C_level_shape matched_D_recent_sparse matched_E_separable)

pending=()
for candidate in "${ALL[@]}"; do
  done_for_setting=0
  for metrics in "$OUT/$candidate"/runs/*"${DATASET_KEY}_h${HORIZON}"*/metrics.csv; do
    [ -f "$metrics" ] || continue
    if head -2 "$metrics" | grep -q test_mse; then done_for_setting=1; break; fi
  done
  [ "$done_for_setting" -eq 0 ] && pending+=("$candidate")
done
echo "pending: ${pending[*]:-none}" | tee -a "$LOGDIR/finish_${DATASET_KEY}.log"

index=0
for candidate in "${pending[@]:-}"; do
  [ -n "$candidate" ] || continue
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  index=$((index + 1))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_structured_lowrank_round1.py \
    --round 1 --frozen "$FROZEN" --setting "$SETTING" --output-root "$OUT" \
    --gpus "$gpu" --only "$candidate" \
    > "$LOGDIR/finish_${candidate}.log" 2>&1 &
  echo "launched $candidate on gpu $gpu" | tee -a "$LOGDIR/finish_${DATASET_KEY}.log"
done
wait
echo "finish ${SETTING} done" | tee -a "$LOGDIR/finish_${DATASET_KEY}.log"
