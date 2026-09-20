#!/bin/bash
# E16 sharded rerun (v2) with the corrected bias convention and the fixed snapshot.
#
# Why a NEW shard prefix instead of rerunning into run_e16_shards_fast.sh's dirs:
# the earlier run's per-shard outputs were produced under the double-counted bias
# convention (defect D1) and with the stale element-wise snapshot (defect D2).  Both
# are fixed, so those numbers are superseded.  Writing the rerun into the SAME
# directories would be dangerous rather than merely untidy: a shard that FAILED this
# time would leave the previous, differently-conventioned CSV sitting in place, and
# the merge would happily consume it.  A fresh prefix makes "old" and "new"
# impossible to confuse, and keeps the superseded artifacts available as provenance.
#
# Sharding shape is unchanged and still non-negotiable: one (setting, arm) per shard
# so all three seeds stay together, which the cross-seed columns section 4.4 displays
# require.  Kernel: --fast-einsum, proven equivalent to the historical path at the
# displayed precision on both a low-rank and a dense cell.
#
# Usage (on the server, from the repository root):
#   nohup bash scripts/phaseformer_L/run_e16_shards_v2.sh \
#     > ~/niuyiming/logs/e16_shards_v2.log 2>&1 &
set -u

cd "$(dirname "$0")/../.." || exit 1
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
LOG="$HOME/niuyiming/logs"
STATUS="$LOG/e16_shards_v2.status"
E14_ROOT=research_runs/phaseformer_L_e14_main_v1
SHARD_PREFIX=research_runs/phaseformer_L_e16_shard_v2
SEEDS=2021,2022,2023
SETTINGS="ETTh2:96 ETTh2:720 ETTm2:96 ETTm2:192 Weather:96 Weather:192 Electricity:336"
ARMS="l_main l_q1_4 l_q1_8"

mkdir -p "$LOG"
{
  echo "e16 v2 shards launched $(date -Is)"
  echo "code: D1 fixed (no double-counted W_dec b_enc), D2 fixed (statistics re-read after the arms pass)"
  echo "kernel: --fast-einsum; combos: 7 settings x 3 arms = 21 shards, 3 cells each = 63"
} > "$STATUS"

index=0
pids=""
for setting in $SETTINGS; do
  dataset=${setting%%:*}
  horizon=${setting##*:}
  for arm in $ARMS; do
    shard=$(printf "%s_%02d" "$SHARD_PREFIX" "$index")
    log="$LOG/e16_shard_v2_$(printf %02d "$index").log"
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

echo "all $index v2 shards launched at $(date -Is); waiting" >> "$STATUS"
failures=0
for pid in $pids; do
  if ! wait "$pid"; then
    failures=$(( failures + 1 ))
    echo "shard pid=$pid FAILED" >> "$STATUS"
  fi
done

echo "e16 v2 shards finished at $(date -Is); failures=$failures" >> "$STATUS"
exit "$failures"
