#!/usr/bin/env bash
# Phase-2 orchestration for the PhaseFormer-L experiment suite.
#
# Runs the stages that can only start once E14's stage A has finished, in
# dependency order, logging every step's exit code.  Each stage is invoked with
# its own `--verify`/`--dry-run` gate where the tool provides one, and the script
# stops at the first failure so a broken stage cannot silently starve the next.
#
# Usage (on the server, from the repository root):
#
#   bash scripts/phaseformer_L/run_phase2_after_e14.sh                 # all steps
#   bash scripts/phaseformer_L/run_phase2_after_e14.sh --from 3        # start at step 3
#   bash scripts/phaseformer_L/run_phase2_after_e14.sh --only 1        # just step 1
#   bash scripts/phaseformer_L/run_phase2_after_e14.sh --list
#
# Steps (see docs/PhaseFormer_L_execution_schedule.md for the contracts):
#   1  E14 stage B   single test read for the 411 new cells (e14_read_test.py)
#   2  E19 stage 2   §4.7 rho columns (e19_predictive_power.py)
#   3  E14 writeback §4.2 table + claims A-D + audit (e14_writeback.py)
#   4  E16           §4.4 dissection + 10-arm interventions, 63 cells
#   5  E17           §4.5 four-arm table (24 new runs) + assemble
#   6  E18           §4.6 rows 1 and 5 (78 runs) + row 3 over 28 settings
#
# Nothing here reads or writes outside research_runs/ and the two log files.

set -uo pipefail

PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO" || exit 1

LOGDIR=$HOME/niuyiming/logs

E14_ROOT=research_runs/phaseformer_L_e14_main_v1
E16_ROOT=research_runs/phaseformer_L_e16_dissection_v1
E17_ROOT=research_runs/phaseformer_L_e17_conditional_v1
E18_ROOT=research_runs/phaseformer_L_e18_negative_v1
E19_ROOT=research_runs/phaseformer_L_e19_predictive_v1
GPUS=0,1,2,3,4,5,6,7

FROM=1
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --list)
      sed -n 's/^#   \([0-9]\)  \(.*\)/  \1  \2/p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGDIR"

run_step() {
  local n="$1" name="$2"; shift 2
  if [ -n "$ONLY" ] && [ "$ONLY" != "$n" ]; then return 0; fi
  if [ "$n" -lt "$FROM" ]; then return 0; fi
  local log="$LOGDIR/phase2_step${n}.log"
  echo "=== [$(date -Is)] step $n: $name"
  echo "HEAD: $(git log --oneline -1)"
  "$@" >"$log" 2>&1
  local rc=$?
  echo "--- step $n exit=$rc (log: $log)"
  if [ $rc -ne 0 ]; then
    echo "=== step $n FAILED; stopping so the next stage cannot start on bad input"
    tail -25 "$log"
    exit $rc
  fi
  tail -3 "$log"
}

# --- guard: E14 stage A must be finished -----------------------------------
guard_e14_done() {
  if pgrep -f "scripts/.*search_phaseformer.py" >/dev/null 2>&1; then
    echo "E14 (or another search_phaseformer run) is still active; refusing to start phase 2" >&2
    return 1
  fi
  if ! grep -q "E14_MAIN_EXIT=0" "$LOGDIR/e14_main.log" 2>/dev/null; then
    echo "e14_main.log does not record E14_MAIN_EXIT=0; refusing to start phase 2" >&2
    return 1
  fi
  local trained
  trained=$(find "$E14_ROOT/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  echo "E14 stage A finished; run directories present: $trained (expected 411)"
  if [ "$trained" -ne 411 ]; then
    echo "expected 411 new runs, found $trained; refusing to start phase 2" >&2
    return 1
  fi
  return 0
}

if [ -z "$ONLY" ] || [ "$ONLY" -ge 1 ]; then
  if [ "$FROM" -le 1 ]; then
    echo "=== [$(date -Is)] pre-flight: E14 completion guard"
    guard_e14_done || exit 1
  fi
fi

run_step 1 "E14 stage B: single test read" \
  "$PY" scripts/phaseformer_L/e14_read_test.py \
    --manifest "$E14_ROOT/stage_a_manifest.json" --output-root "$E14_ROOT" \
    --gpus "$GPUS" --retries 1 --poll-seconds 15 --num-workers 4

run_step 2 "E19 stage 2: §4.7 rho columns" \
  "$PY" scripts/phaseformer_L/e19_predictive_power.py \
    --stats "$E19_ROOT/level_statistics.csv" --results "$E14_ROOT/results.csv" \
    --output-root "$E19_ROOT"

run_step 3 "E14 writeback: §4.2 table + claims A-D + audit" \
  "$PY" scripts/phaseformer_L/e14_writeback.py \
    --manifest "$E14_ROOT/stage_a_manifest.json" --results "$E14_ROOT/results.csv" \
    --stats "$E19_ROOT/level_statistics.csv" \
    --golden docs/PhaseFormer_gold_standard.md --output-root "$E14_ROOT"

run_step 4 "E16: §4.4 dissection + interventions (63 cells)" \
  "$PY" scripts/phaseformer_L/e16_dissection.py \
    --e14-root "$E14_ROOT" --output-root "$E16_ROOT" \
    --gpus 0 --mem-budget-mb 2048

run_step 5 "E17: §4.5 four-arm training (24 runs) + assemble" \
  bash -c "cd '$REPO' && '$PY' scripts/phaseformer_L/e17_conditional.py --stage a --verify \
      --gpus '$GPUS' --output-root '$E17_ROOT'"

run_step 6 "E18: §4.6 rows 1+5 (78 runs) then row 3 (28 settings)" \
  bash -c "cd '$REPO' && '$PY' scripts/phaseformer_L/e18_negative.py --stage all --verify \
      --gpus '$GPUS' --output-root '$E18_ROOT' \
    && '$PY' scripts/phaseformer_L/e18_svd_truncation.py \
      --e14-root '$E14_ROOT' --output-root '$E18_ROOT' \
      --ranks 10 --seeds 2021 2022 2023 --evaluation-split val"

echo "=== [$(date -Is)] phase 2 complete"
