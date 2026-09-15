#!/usr/bin/env bash
# Run an explicitly enumerated candidate list, one candidate per GPU slot.
#
# Unlike the auto-detecting driver this script never decides what is missing: the
# caller passes the exact candidates, so a candidate can be launched at most once.
# Use it to finish a setting whose remaining work is known.
#
#   bash scripts/remote_explicit_candidates.sh Electricity:336 \
#        "A_period_lowrank_r8:0" "D_recent_sparse:1" ...
set -euo pipefail

cd "$HOME/niuyiming/PhaseFormer"
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
FROZEN=research_runs/structured_lowrank_round0_v1/frozen_configs.json
OUT=research_runs/structured_lowrank_round1_scratch
LOGDIR="$HOME/niuyiming/structured_lowrank_round1_logs"
SETTING="$1"; shift
mkdir -p "$LOGDIR"
git log --oneline -1 > "$LOGDIR/explicit_head.txt"

for pair in "$@"; do
  candidate="${pair%%:*}"
  gpu="${pair##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_structured_lowrank_round1.py \
    --round 1 --frozen "$FROZEN" --setting "$SETTING" \
    --output-root "$OUT" --gpus "$gpu" --only "$candidate" \
    > "$LOGDIR/explicit_${candidate}.log" 2>&1 &
  echo "launched $candidate on gpu $gpu" | tee -a "$LOGDIR/explicit.log"
done
wait
echo "explicit candidates finished" | tee -a "$LOGDIR/explicit.log"
