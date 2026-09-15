#!/usr/bin/env bash
# Finalize the remaining Round 1 setting by spreading its candidates over
# several free GPUs instead of one sequential GPU.
#
# Only safe because every unfinished candidate's hyperparameters are fully
# determined by the frozen per-setting config: each job is independent and
# writes to its own output directory.
set -euo pipefail

cd "$HOME/niuyiming/PhaseFormer"
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
FROZEN=research_runs/structured_lowrank_round0_v1/frozen_configs.json
OUT=research_runs/structured_lowrank_round1_scratch
LOGDIR="$HOME/niuyiming/structured_lowrank_round1_logs"
SETTING="${1:-Electricity:336}"
shift || true
GPUS=("$@")
mkdir -p "$LOGDIR"
git log --oneline -1 > "$LOGDIR/remaining_head.txt"

IFS=',' read -r -a CANDIDATES <<< "A_period_lowrank,A_period_lowrank_r8,B_segment_basis,C_level_shape,D_recent_sparse,E_separable,matched_A_period_lowrank,matched_A_period_lowrank_r8,matched_B_segment_basis,matched_C_level_shape,matched_D_recent_sparse,matched_E_separable"

# Only a run for the requested setting counts as done: the scratch tree holds
# every setting, so a plain status.json glob would treat another setting's run as
# completing this candidate.
SETTING_KEY=$(echo "$SETTING" | tr ':' '_' | tr '[:upper:]' '[:lower:]')
pending=()
for candidate in "${CANDIDATES[@]}"; do
  if ! ls "$OUT/$candidate"/runs/*"${SETTING_KEY}"*/status.json >/dev/null 2>&1; then
    pending+=("$candidate")
  fi
done
echo "pending: ${pending[*]:-none}" | tee -a "$LOGDIR/remaining.log"

index=0
for candidate in "${pending[@]}"; do
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  index=$((index + 1))
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_structured_lowrank_round1.py \
    --round 1 --frozen "$FROZEN" --setting "$SETTING" \
    --output-root "$OUT" --gpus "$gpu" --only "$candidate" \
    > "$LOGDIR/remaining_${candidate}.log" 2>&1 &
  echo "launched $candidate on gpu $gpu" | tee -a "$LOGDIR/remaining.log"
done
wait
echo "remaining jobs finished" | tee -a "$LOGDIR/remaining.log"
