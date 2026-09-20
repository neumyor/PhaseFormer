#!/bin/bash
# Server-side finisher for the PhaseFormer-L golden search.
#
# Purpose: the round-1 grid takes ~6 h and the round-2 grid ~10 h, so the
# verdict must not depend on a laptop session staying alive.  This watcher
# waits for whichever driver is currently running to exit, then runs the
# remaining stages itself and, if the pre-registered target (>=4/8 settings
# beating Golden on BOTH metrics) is not met, launches round 2.
#
# It is deliberately conservative:
#   * it syncs code exactly once, only after the driver has exited and only
#     behind an explicit process check, so a live job's source files are never
#     replaced (REMOTE_SERVER.md).  The sync is needed because the verdict must
#     be computed by the current driver, not the older server build;
#   * every stage is idempotent (the driver skips cells that already have
#     metrics), so being re-run after an interruption is safe;
#   * it writes one status file that a later session can read to learn what
#     happened without reading the whole log.
#
# Usage:  nohup bash ~/niuyiming/watch_golden_search.sh >> ~/niuyiming/logs/watch_gs.log 2>&1 &

set -u
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
REPO=$HOME/niuyiming/PhaseFormer
ROOT=research_runs/phaseformer_L_golden_search_v1
LOG=$HOME/niuyiming/logs/golden_search_round2.log
STATUS=$HOME/niuyiming/logs/golden_search_status.txt
TARGET=4

cd "$REPO" || exit 1

status() {
    echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"
}

# ---------------------------------------------------------------- wait for r1
status "watcher start; waiting for the running driver to exit"
while pgrep -f "golden_search.py --stage search" >/dev/null 2>&1; do
    sleep 120
done
status "round-1 driver has exited"

DONE=$(grep -c '^\[done' "$ROOT/_logs/stage1.log" 2>/dev/null || echo 0)
FAIL=$(grep -c '^\[FAIL' "$ROOT/_logs/stage1.log" 2>/dev/null || echo 0)
status "round-1 tally: done=$DONE fail=$FAIL"

# A failure means a cell is missing.  Re-run the same stage: it skips every
# cell that already has test metrics and retries only the missing ones.
if [ "$FAIL" -gt 0 ]; then
    status "retrying the failed cells (idempotent re-run)"
    $PY scripts/phaseformer_L/golden_search.py --stage search \
        --gpus 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
    status "retry pass exit=$?"
fi

# ---------------------------------------------------------------- sync
# The verdict must be computed by the CURRENT driver: the older server build
# has neither the dual-metric ranking nor the phase_only anchor, so running
# select before syncing would report round 1 with a weaker rule.  The sync is
# gated on a flag the running driver does not have, and behind an explicit
# process check, so it can never replace source files under a live job
# (REMOTE_SERVER.md).
if ! $PY scripts/phaseformer_L/golden_search.py --help 2>/dev/null | grep -q 'search-round2'; then
    if pgrep -f 'search_phaseformer.py|golden_search.py' >/dev/null 2>&1; then
        status "ABORT: processes still running, refusing to sync"
        exit 1
    fi
    status "driver is stale -> syncing from the uploaded bundle"
    git fetch origin weak_residual_nlinear_bottleneck >> "$LOG" 2>&1
    git reset --hard FETCH_HEAD >> "$LOG" 2>&1
    status "sync exit=$? head=$(git log --oneline -1)"
fi
if ! $PY scripts/phaseformer_L/golden_search.py --help 2>/dev/null | grep -q 'search-round2'; then
    status "ABORT: driver still lacks search-round2 after sync"
    exit 1
fi
status "driver ready: $(git log --oneline -1)"

# ---------------------------------------------------------------- r1 verdict
status "running round-1 select / confirm / final"
$PY scripts/phaseformer_L/golden_search.py --stage select >> "$LOG" 2>&1
status "select exit=$?"
$PY scripts/phaseformer_L/golden_search.py --stage confirm \
    --gpus 0,1,2,3,4,5,6,7 >> "$LOG" 2>&1
status "confirm exit=$?"
$PY scripts/phaseformer_L/golden_search.py --stage final >> "$LOG" 2>&1
status "final exit=$?"

WINS=$($PY - <<'EOF'
import json, pathlib
p = pathlib.Path("research_runs/phaseformer_L_golden_search_v1/final_selection.json")
if not p.is_file():
    print("NA")
else:
    print(json.loads(p.read_text())["verdict"]["achieved"])
EOF
)
status "round-1 verdict: $WINS/8 settings beat Golden on both metrics"

# ---------------------------------------------------------------- round 2
if [ "$WINS" = "NA" ] || [ "$WINS" -lt "$TARGET" ]; then
    status "target not met -> launching round 2 (pre-registered, narrowed grid)"
    bash "$HOME/niuyiming/run_round2.sh" >> "$LOG" 2>&1
    status "round-2 chain exit=$?"

    WINS2=$($PY - <<'EOF'
import json, pathlib
p = pathlib.Path("research_runs/phaseformer_L_golden_search_v1/final_selection.json")
if not p.is_file():
    print("NA")
else:
    v = json.loads(p.read_text())["verdict"]
    print(f'{v["achieved"]}/{v["n_settings"]} (target {v["target_count"]}, met={v["met"]})')
EOF
)
    status "FINAL after round 2: $WINS2"
    status "ALL DONE -- verdict table at $ROOT/final_selection.csv"
else
    status "target met after round 1 -- ALL DONE"
    status "verdict table at $ROOT/final_selection.csv"
fi
