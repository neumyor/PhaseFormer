#!/usr/bin/env bash
# Round 1 driver: one pilot setting per GPU, all candidates sequential per GPU.
#
# Plan reference: section 6 (Round 1 route matrix) and section 13 (budget).
# Runs on the A800 server.  Logs land outside the repository.
set -euo pipefail

cd "$HOME/niuyiming/PhaseFormer"
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
FROZEN=research_runs/structured_lowrank_round0_v1/frozen_configs.json
OUT=research_runs/structured_lowrank_round1_scratch
LOGDIR="$HOME/niuyiming/structured_lowrank_round1_logs"
mkdir -p "$LOGDIR"

git log --oneline -1 > "$LOGDIR/pipeline_head.txt"

run_setting () {
  local gpu="$1" setting="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_structured_lowrank_round1.py \
    --round 1 --frozen "$FROZEN" --setting "$setting" \
    --output-root "$OUT" --gpus "$gpu" \
    > "$LOGDIR/${setting//:/_}.log" 2>&1
  echo "done ${setting} on gpu ${gpu}" >> "$LOGDIR/pipeline.log"
}

run_setting 0 ETTh2:96 &
run_setting 1 ETTh2:720 &
run_setting 2 ETTm2:192 &
run_setting 3 Electricity:336 &

wait
echo "all Round 1 settings finished" >> "$LOGDIR/pipeline.log"
