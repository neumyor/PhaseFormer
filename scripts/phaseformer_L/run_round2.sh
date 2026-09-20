#!/bin/bash
# Round-2 golden search (pre-registered in docs/PhaseFormer_L_golden_search_plan.md §2b).
# Batch 1: mae   x lr {3e-4,1e-3,3e-3} x gate x head = 642 runs
# Batch 2: huber x lr {3e-4,1e-3,3e-3} x gate x head = 642 runs
# (Electricity-96 is narrowed to gate{0.05,0.2} x head{dense,H/4}; see plan 2b
#  revision 1 for why.  Total 1284 runs ~= 81.5 GPU-h ~= 10.2 h on 8 GPUs.)
set -u
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer || exit 1
git log --oneline -1 > ~/niuyiming/logs/golden_search_round2_HEAD.txt
for LOSS in mae huber; do
  echo "=== round2 batch loss=$LOSS start $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round2.log
  $PY scripts/phaseformer_L/golden_search.py --stage search-round2 --losses $LOSS \
      --gpus 0,1,2,3,4,5,6,7 >> ~/niuyiming/logs/golden_search_round2.log 2>&1
  echo "=== round2 batch loss=$LOSS exit=$? $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round2.log
done
$PY scripts/phaseformer_L/golden_search.py --stage select >> ~/niuyiming/logs/golden_search_round2.log 2>&1
$PY scripts/phaseformer_L/golden_search.py --stage confirm --gpus 0,1,2,3,4,5,6,7 >> ~/niuyiming/logs/golden_search_round2.log 2>&1
$PY scripts/phaseformer_L/golden_search.py --stage final >> ~/niuyiming/logs/golden_search_round2.log 2>&1
echo "=== round2 ALL DONE $(date '+%F %T')" >> ~/niuyiming/logs/golden_search_round2.log
