#!/bin/bash
# Round-3 golden search (pre-registered in docs/PhaseFormer_L_golden_search_plan.md 2c).
# The three settings that still miss Golden: ETTh1-192, ETTh1-336, ETTm1-192.
# 132 runs each = 396 total, ~16 GPU-h, ~2 h wall on 8 GPUs.
#
# New axes (neither round 1 nor 2 touched them):
#   huber_delta 0.05/0.1/0.3/3.0  -- the continuous knob between the huber and
#                                    mae endpoints, which rounds 1-2 sampled only
#   max_epochs 60                 -- round 1 measured 68% of cells exhausting the
#                                    30-epoch budget; NOT comparable with E14
#   lr one rung past each edge     -- ETTh1 1e-2 (winners pressed 3e-3),
#                                    ETTm1-192 1e-4 (its winner pressed 3e-4)
set -u
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer || exit 1
git log --oneline -1 > ~/niuyiming/logs/golden_search_round3_HEAD.txt
echo "=== round3 start $(date '+%F %T') head=$(git log --oneline -1)" >> ~/niuyiming/logs/golden_search_round3.log
$PY scripts/phaseformer_L/golden_search.py --stage search-round3 \
    --gpus 0,1,2,3,4,5,6,7 >> ~/niuyiming/logs/golden_search_round3.log 2>&1
echo "=== round3 search exit=$? $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round3.log
$PY scripts/phaseformer_L/golden_search.py --stage select >> ~/niuyiming/logs/golden_search_round3.log 2>&1
echo "=== round3 select exit=$? $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round3.log
$PY scripts/phaseformer_L/golden_search.py --stage confirm --gpus 0,1,2,3,4,5,6,7 >> ~/niuyiming/logs/golden_search_round3.log 2>&1
echo "=== round3 confirm exit=$? $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round3.log
$PY scripts/phaseformer_L/golden_search.py --stage final >> ~/niuyiming/logs/golden_search_round3.log 2>&1
echo "=== round3 ALL DONE $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round3.log
