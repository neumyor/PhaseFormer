#!/bin/bash
# E16 sharded across the 8 GPUs, using the BLAS contraction path (--fast-einsum).
#
# This is the companion of run_e16_shards.sh, which uses the historical bare
# einsum.  It exists as a separate file rather than a flag because the original
# driver is still running when this one is prepared, and bash reads a script
# incrementally -- editing the running file could break it mid-loop.
#
# Only launch this after compare_e16_runs.py has shown the two kernels agree at
# the precision section 4.4 displays, on BOTH a low-rank and a dense cell (the
# dense arm's rank_dim is the lookback, so it is the case where BLAS matters and
# the case that must be proven).
#
# Sharding shape, and why it is not negotiable:
#   one (setting, arm) per shard = 21 shards x 3 seeds = 63 cells.
#   cross_seed_rows groups by (setting, arm) and the dissection rows carry
#   cross_seed_leading4_*_overlap, so splitting seeds would silently drop the
#   cross-seed column section 4.4 displays.
#
# Usage (on the server, from the repository root):
#   nohup bash scripts/phaseformer_L/run_e16_shards_fast.sh \
#     > ~/niuyiming/logs/e16_shards_fast.log 2>&1 &
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
LOG="$HOME/niuyiming/logs"
STATUS="$LOG/e16_shards_fast.status"
E14_ROOT=research_runs/phaseformer_L_e14_main_v1
SHARD_PREFIX=research_runs/phaseformer_L_e16_shard_fast
SEEDS=2021,2022,2023
SETTINGS="ETTh2:96 ETTh2:720 ETTm2:96 ETTm2:192 Weather:96 Weather:192 Electricity:336"
ARMS="l_main l_q1_4 l_q1_8"

mkdir -p "$LOG"
{
  echo "e16 fast-kernel shards launched $(date -Is)"
  echo "kernel: --fast-einsum (EINSUM_OPTIMIZE=True; BLAS contraction order)"
  echo "combos: 7 settings x 3 arms = 21 shards, 3 cells each = 63"
} > "$STATUS"

index=0
pids=""
for setting in $SETTINGS; do
  dataset=${setting%%:*}
  horizon=${setting##*:}
  for arm in $ARMS; do
    shard=$(printf "%s_%02d" "$SHARD_PREFIX" "$index")
    log="$LOG/e16_shard_fast_$(printf %02d "$index").log"
    gpu=$(( index % 8 ))
    mkdir -p "$shard"
    "$PY" scripts/phaseformer_L/e16_dissection.py \
      --fast-einsum \
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

echo "all $index fast shards launched at $(date -Is); waiting" >> "$STATUS"
failures=0
for pid in $pids; do
  if ! wait "$pid"; then
    echo "shard pid=$pid FAILED (rc=$?)" >> "$STATUS"
    failures=$(( failures + 1 ))
  fi
done

echo "e16 fast shards finished at $(date -Is); failures=$failures" >> "$STATUS"
exit "$failures"
