#!/usr/bin/env bash
# Run a queue of Round 1 candidates sequentially on one GPU, one at a time.
#
# Splitting the remaining candidates into one queue per GPU keeps exactly one
# training run per GPU: co-locating two runs on the same GPU shares SMs and
# memory bandwidth and slows both down, which is worse than running them
# back-to-back at full speed.
#
#   bash scripts/remote_gpu_group_serial.sh Electricity:336 0 "A_period_lowrank B_segment_basis"
set -euo pipefail

cd "$HOME/niuyiming/PhaseFormer"
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
FROZEN=research_runs/structured_lowrank_round0_v1/frozen_configs.json
OUT=research_runs/structured_lowrank_round1_scratch
LOGDIR="$HOME/niuyiming/structured_lowrank_round1_logs"
SETTING="$1"; GPU="$2"; QUEUE="$3"
mkdir -p "$LOGDIR"

for candidate in $QUEUE; do
  echo "start $candidate on gpu $GPU $(date -u +%H:%M:%S)" >> "$LOGDIR/gpu${GPU}_serial.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_structured_lowrank_round1.py \
    --round 1 --frozen "$FROZEN" --setting "$SETTING" --output-root "$OUT" \
    --gpus "$GPU" --only "$candidate" \
    > "$LOGDIR/gpu${GPU}_${candidate}.log" 2>&1
  echo "done $candidate on gpu $GPU $(date -u +%H:%M:%S)" >> "$LOGDIR/gpu${GPU}_serial.log"
done
