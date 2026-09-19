#!/usr/bin/env bash
# Wait for E14 stage A to finish cleanly, then run the phase-2 chain unattended.
#
# Why a separate watcher instead of just starting phase 2 later by hand: E14's
# stage A occupies all 8 GPUs for hours, and `run_phase2_after_e14.sh` refuses to
# start until stage A is complete.  Chaining the two removes the idle gap without
# weakening that refusal -- phase 2 still runs its own `guard_e14_done` pre-flight
# and will abort if the completion conditions are not exactly met.
#
# Usage (on the server, from the repository root):
#
#   nohup bash scripts/phaseformer_L/watch_e14_then_phase2.sh \
#     > ~/niuyiming/logs/phase2_watcher.log 2>&1 &
#
# Tunables:
#   POLL_SECONDS    how often to re-check E14           (default 120)
#   MAX_WAIT_HOURS  give up after this long waiting     (default 24)
#
# Status is written to $LOGDIR/phase2_watch.status so a poller only has to read
# one small file to learn where the chain stands.  Nothing here writes inside
# research_runs/ -- it only reads E14's artifacts and delegates the real work.

set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO" || exit 1

LOGDIR=$HOME/niuyiming/logs
POLL_SECONDS=${POLL_SECONDS:-120}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-24}

E14_ROOT=research_runs/phaseformer_L_e14_main_v1
E14_LOG=$LOGDIR/e14_main.log
MARK=$LOGDIR/phase2_watch.status

mkdir -p "$LOGDIR"

# Expected number of new run directories; must match run_phase2_after_e14.sh.
EXPECTED_RUNS=411

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))

echo "watcher started $(date -Is); waiting for E14 stage A (expect $EXPECTED_RUNS runs)" > "$MARK"

while :; do
  flag=no
  if grep -q "E14_MAIN_EXIT=0" "$E14_LOG" 2>/dev/null; then flag=yes; fi
  runs=$(find "$E14_ROOT/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

  if [ "$flag" = yes ] && [ "$runs" = "$EXPECTED_RUNS" ]; then
    echo "E14 complete (flag=$flag runs=$runs) at $(date -Is); starting phase 2" > "$MARK"
    bash scripts/phaseformer_L/run_phase2_after_e14.sh >> "$LOGDIR/phase2.log" 2>&1
    rc=$?
    {
      echo "phase 2 exit=$rc at $(date -Is)"
      if [ "$rc" -eq 0 ]; then
        echo "PHASE2_OK"
      else
        echo "PHASE2_FAILED rc=$rc -- inspect $LOGDIR/phase2_step*.log"
      fi
    } >> "$MARK"
    exit "$rc"
  fi

  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "TIMEOUT waiting for E14 at $(date -Is) (flag=$flag runs=$runs)" > "$MARK"
    exit 3
  fi

  sleep "$POLL_SECONDS"
done
