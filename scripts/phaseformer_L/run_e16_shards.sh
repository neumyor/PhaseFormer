#!/bin/bash
# E16 sharded across the 8 GPUs: one (setting, arm) combo per shard, with all
# three seeds kept together in the same process.
#
# Why shard this way rather than by seed: `cross_seed_rows` groups by
# (setting, arm) and the dissection rows carry `cross_seed_leading4_*_overlap`,
# so splitting seeds across processes would silently drop the very columns that
# section 4.4 displays.  Keeping one combo per shard makes each shard a
# self-contained, numerically identical slice of the full run -- same code, same
# per-cell arithmetic, just a narrower cell list.
#
# Why shard at all: measured 2026-09-20, the single-process launch spent >36 min
# on its FIRST cell and had completed 0 of 63, while seven of the eight A800s sat
# idle.  The cost is dominated by the dense arm, whose rank_dim is the lookback
# (720) at every horizon, and the contraction runs single-threaded in float64.
# Sharding restores the plan's throughput without touching a single line of the
# evaluator, so the numbers cannot move.
#
# Usage (on the server, from the repository root):
#   nohup bash scripts/phaseformer_L/run_e16_shards.sh > ~/niuyiming/logs/e16_shards.log 2>&1 &
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
LOG="$HOME/niuyiming/logs"
STATUS="$LOG/e16_shards.status"
E14_ROOT=research_runs/phaseformer_L_e14_main_v1
SHARD_PREFIX=research_runs/phaseformer_L_e16_shard
SEEDS=2021,2022,2023
SETTINGS="ETTh2:96 ETTh2:720 ETTm2:96 ETTm2:192 Weather:96 Weather:192 Electricity:336"
ARMS="l_main l_q1_4 l_q1_8"

mkdir -p "$LOG"
{
  echo "e16 shards launched $(date -Is)"
  echo "combos: 7 settings x 3 arms = 21 shards, 3 cells each = 63"
} > "$STATUS"

index=0
pids=""
for setting in $SETTINGS; do
  dataset=${setting%%:*}
  horizon=${setting##*:}
  for arm in $ARMS; do
    shard=$(printf "%s_%02d" "$SHARD_PREFIX" "$index")
    log="$LOG/e16_shard_$(printf %02d "$index").log"
    gpu=$(( index % 8 ))
    mkdir -p "$shard"
    "$PY" scripts/phaseformer_L/e16_dissection.py \
      --e14-root "$E14_ROOT" \
      --output-root "$shard" \
      --datasets "$dataset" --horizons "$horizon" --arms "$arm" \
      --seeds "$SEEDS" \
      --gpus "$gpu" --mem-budget-mb 2048 --num-workers 1 \
      > "$log" 2>&1 &
    pid=$!
    pids="$pids $pid"
    echo "shard $index: ${dataset}-${horizon} ${arm} gpu=$gpu pid=$pid root=$shard" >> "$STATUS"
    index=$(( index + 1 ))
  done
done

echo "all $index shards launched at $(date -Is); waiting" >> "$STATUS"
failures=0
for pid in $pids; do
  if ! wait "$pid"; then
    echo "shard pid=$pid FAILED (rc=$?)" >> "$STATUS"
    failures=$(( failures + 1 ))
  fi
done

echo "e16 shards finished at $(date -Is); failures=$failures" >> "$STATUS"
exit "$failures"
