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
#   3  E14 params    parameter table + gate fallback (e14_params.py)
#      E14 reuse     reuse ambiguity audit (e14_reuse_audit.py)
#      E14 writeback §4.2 table + claims A-D + audit (e14_writeback.py)
#   4  E16           §4.4 dissection + interventions (63 cells) + write-back
#   5  E17           §4.5 training (24 runs) + assemble + single test read
#                   + §4.5 write-back
#   6  E18           §4.6 rows 1+5 (78 runs) + single test read + row 3
#                   (28 settings) + §4.6 write-back
#   7  audit         stage-5 acceptance audit over every experiment's artifacts
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

# E14's test-bearing CSV.  NOTE the asymmetry, which is a real contract and not a
# style choice: e14_read_test.py fills test_mse/test_mae into `results.csv` IN
# PLACE, whereas read_test_generic.py (used for E17/E18) writes a SEPARATE
# `<results>.with_test.csv` sibling.  Declaring the E14 path once keeps steps 2, 3
# and 6 from drifting apart -- they had, and step 6 pointed at a
# `results.with_test.csv` that E14 never produces.
E14_TEST_CSV="$E14_ROOT/results.csv"

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
    # Static, sub-second, and it runs BEFORE any expensive stage: it proves that
    # every column the write-backs demand is one their producers actually write.
    # This is the generalised form of the e16_writeback label defect, which
    # otherwise only surfaces after the dissection run has already finished.
    echo "=== [$(date -Is)] pre-flight: static column contracts"
    "$PY" scripts/phaseformer_L/check_column_contracts.py --strict \
      --output "$LOGDIR/phase2_column_contracts.json" || {
        echo "column contracts are broken; refusing to spend GPU time on stages that cannot write back" >&2
        exit 1
      }
    # Flag existence is not flag arity: `--seeds 2021 2022 2023` is an argparse
    # error because --seeds is a comma-list, and that mistake would otherwise
    # only surface at step 6, after every expensive stage had already run.
    echo "=== [$(date -Is)] pre-flight: pipeline invocation arity"
    "$PY" scripts/phaseformer_L/check_pipeline_invocations.py || {
        echo "pipeline invocations are malformed; refusing to start the chain" >&2
        exit 1
      }
    # Do the consumers actually accept E14's *real* manifest?  A consumer that
    # mis-reads it fails soft: the step exits 0 and writes a table with empty
    # columns.  E18's baseline index did exactly that for all 78 of its rows
    # (reused cells carry no command), and the recorded --output-dir of every new
    # cell was a template value.  This runs the consumers' own loaders against
    # the live artifact, before anything expensive starts.
    echo "=== [$(date -Is)] pre-flight: phase-2 consumer contracts"
    "$PY" scripts/phaseformer_L/check_phase2_consumers.py \
      --e14-root "$E14_ROOT" \
      --json "$LOGDIR/phase2_consumer_contracts.json" || {
        echo "phase-2 consumers cannot read E14's manifest; refusing to start the chain" >&2
        exit 1
      }
  fi
fi

run_step 1 "E14 stage B: single test read" \
  bash -c "cd '$REPO' && \
    '$PY' scripts/phaseformer_L/e14_read_test.py \
      --manifest '$E14_ROOT/stage_a_manifest.json' --output-root '$E14_ROOT' \
      --gpus '$GPUS' --retries 1 --poll-seconds 15 --num-workers 4 \
    && test -s '$E14_TEST_CSV' \
    && awk -F, 'NR==1{for(i=1;i<=NF;i++) if(\$i==\"test_mse\") c=i} NR>1{if(c&&\$c!=\"\")n++;t++} END{printf \"E14 test-bearing rows=%d with_test=%d\n\",t,n; if(c==0||t==0||n!=t){print \"E14 results.csv is missing a populated test_mse column; refusing to continue\"; exit 1}}' '$E14_TEST_CSV'"

run_step 2 "E19 stage 2: §4.7 rho columns" \
  "$PY" scripts/phaseformer_L/e19_predictive_power.py \
    --stats "$E19_ROOT/level_statistics.csv" --results "$E14_TEST_CSV" \
    --output-root "$E19_ROOT"

# The parameter table and the reuse-ambiguity audit must exist BEFORE the
# write-back, because the write-back reports the parameter columns and takes the
# gate fallback from them.
run_step 3 "E14 parameter table + reuse ambiguity audit + §4.2 writeback" \
  bash -c "cd '$REPO' && \
    '$PY' scripts/phaseformer_L/e14_params.py \
      --manifest '$E14_ROOT/stage_a_manifest.json' --output-root '$E14_ROOT' \
    && '$PY' scripts/phaseformer_L/e14_reuse_audit.py \
      --manifest '$E14_ROOT/stage_a_manifest.json' --output-root '$E14_ROOT' \
    && '$PY' scripts/phaseformer_L/e14_writeback.py \
      --manifest '$E14_ROOT/stage_a_manifest.json' --results '$E14_TEST_CSV' \
      --stats '$E19_ROOT/level_statistics.csv' \
      --golden docs/PhaseFormer_gold_standard.md --output-root '$E14_ROOT' \
    && '$PY' scripts/phaseformer_L/check_builder_outputs.py \
      --csv '$E14_ROOT/main_table.csv' --csv '$E14_ROOT/variant_table.csv' \
      --output '$E14_ROOT/empty_column_report.json' || true"

# E16's own gate is its --dry-run: it resolves every cell's checkpoint and
# refuses when one is missing, and `--verify-checkpoint-heads` additionally
# proves each cell's head kind from the checkpoint keys in seconds (this is the
# check that would have caught the per-cell head bug before any forward pass).
run_step 4 "E16: §4.4 dissection + interventions (63 cells)" \
  bash -c "cd '$REPO' && \
    '$PY' scripts/phaseformer_L/e16_dissection.py --dry-run --verify-checkpoint-heads \
      --e14-root '$E14_ROOT' --output-root '$E16_ROOT' \
    && '$PY' scripts/phaseformer_L/e16_dissection.py \
      --e14-root '$E14_ROOT' --output-root '$E16_ROOT' \
      --gpus 0 --mem-budget-mb 2048 \
    && '$PY' scripts/phaseformer_L/e16_writeback.py \
      --intervention '$E16_ROOT/intervention_table.csv' \
      --dissection '$E16_ROOT/dissection_table.csv' \
      --output-root '$E16_ROOT' \
    && '$PY' scripts/phaseformer_L/check_builder_outputs.py \
      --csv '$E16_ROOT/intervention_table_44.csv' \
      --csv '$E16_ROOT/dissection_table_44.csv' \
      --output '$E16_ROOT/empty_column_report.json' || true"

# E17: train, assemble, then read test ONCE for the 24 new cells.  The runner
# refuses --evaluate-test by design and marks the column
# "pending_single_test_read"; read_test_generic.py is that separate stage, and it
# takes the frozen-subspace basis paths from the results CSV's basis_file column
# (the wrapper owns those paths; the run config only records that a projection
# was used).
run_step 5 "E17: §4.5 four-arm training (24 runs), assemble, single test read" \
  bash -c "cd '$REPO' && \
    '$PY' scripts/phaseformer_L/e17_conditional.py --stage a --verify \
      --gpus '$GPUS' --output-root '$E17_ROOT' \
    && '$PY' scripts/phaseformer_L/e17_conditional.py --stage assemble \
      --output-root '$E17_ROOT' \
    && '$PY' scripts/phaseformer_L/read_test_generic.py \
      --results '$E17_ROOT/results.csv' --gpus '$GPUS' \
    && '$PY' scripts/phaseformer_L/e17_writeback.py \
      --results '$E17_ROOT/results.with_test.csv' \
      --projector-audit '$E17_ROOT/projectors/projector_audit.json' \
      --output-root '$E17_ROOT' \
    && '$PY' scripts/phaseformer_L/check_builder_outputs.py \
      --csv '$E17_ROOT/conditional_table.csv' \
      --output '$E17_ROOT/empty_column_report.json' || true"

run_step 6 "E18: §4.6 rows 1+5 (78 runs), completeness audit, then row 3 (28 settings)" \
  bash -c "cd '$REPO' && \
    '$PY' scripts/phaseformer_L/e18_negative.py --stage all \
      --gpus '$GPUS' --output-root '$E18_ROOT' \
    && '$PY' scripts/phaseformer_L/e18_negative.py --stage all --verify --dry-run \
      --output-root '$E18_ROOT' \
    && '$PY' scripts/phaseformer_L/read_test_generic.py \
      --results '$E18_ROOT/results.csv' --gpus '$GPUS' \
    && '$PY' scripts/phaseformer_L/e18_svd_truncation.py --verify \
      --e14-root '$E14_ROOT' --output-root '$E18_ROOT' \
      --ranks 10 --seeds 2021,2022,2023 --evaluation-split val \
    && '$PY' scripts/phaseformer_L/e18_writeback.py \
      --results '$E18_ROOT/results.with_test.csv' \
      --e14-results '$E14_TEST_CSV' \
      --truncation '$E18_ROOT/svd_truncation_table_28.csv' \
      --output-root '$E18_ROOT' \
    && '$PY' scripts/phaseformer_L/check_builder_outputs.py \
      --csv '$E18_ROOT/negative_table.csv' \
      --output '$E18_ROOT/empty_column_report.json' || true"
# NOTE on --verify semantics, which differ per script and must not be guessed:
#   e18_negative.py     --verify = "fail if any planned cell has no matching
#                        completed run", i.e. a stage-5 COMPLETENESS AUDIT.
#                        Passing it on the training invocation makes it refuse
#                        to start (verified: exit 1, "78 of 78 cells have no
#                        matching completed run").  Hence train first, then
#                        audit, as ordered above.
#   e14_main_matrix.py  --verify = pre-run gate (every declared reuse cell must
#   e17_conditional.py            resolve) -> passed on the training invocation.
#   e18_svd_truncation  --verify = resolve every cell's checkpoint before
#                        evaluating -> passed just before its analysis.

# Stage 5 entry point: verify the documented acceptance criteria of every
# experiment from the artifacts that now exist.  It exits non-zero only when a
# present artifact CONTRADICTS a documented criterion, so a failure here means
# "the numbers need a human before write-up", not "the chain broke".
run_step 7 "stage 5: acceptance audit over the phase-2 artifacts" \
  "$PY" scripts/phaseformer_L/audit_phase2_outputs.py \
    --json "$LOGDIR/phase2_acceptance_audit.json"

echo "=== [$(date -Is)] phase 2 complete"
