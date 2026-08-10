#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=2
export MPLCONFIGDIR=/tmp/phaseformer-mpl
mkdir -p "$MPLCONFIGDIR"
# Resolve a python interpreter portable across machines: prefer the active
# conda env, then bash, then system python3.
PY="${CONDA_PREFIX:-}/bin/python"
if [ -n "$CONDA_PREFIX" ] && [ -x "$PY" ]; then
    :
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    PY=python
fi
failed=0
echo "[$(date -Is)] START ETTm1 h96 p24"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 96 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h96 p24"
echo "[$(date -Is)] START ETTm1 h96 p48"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 96 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h96 p48"
echo "[$(date -Is)] START ETTm1 h96 p96"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 96 --stage period_screen --mechanism original --period 96 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h96 p96"
echo "[$(date -Is)] START ETTm1 h720 p24"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 720 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h720 p24"
echo "[$(date -Is)] START ETTm1 h720 p48"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 720 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h720 p48"
echo "[$(date -Is)] START ETTm1 h720 p96"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 720 --stage period_screen --mechanism original --period 96 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 h720 p96"
echo "[$(date -Is)] START Exchange h96 p7"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 96 --stage period_screen --mechanism original --period 7 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h96 p7"
echo "[$(date -Is)] START Exchange h96 p14"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 96 --stage period_screen --mechanism original --period 14 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h96 p14"
echo "[$(date -Is)] START Exchange h96 p30"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 96 --stage period_screen --mechanism original --period 30 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h96 p30"
echo "[$(date -Is)] START Exchange h96 p24"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 96 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h96 p24"
echo "[$(date -Is)] START Exchange h720 p7"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 720 --stage period_screen --mechanism original --period 7 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h720 p7"
echo "[$(date -Is)] START Exchange h720 p14"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 720 --stage period_screen --mechanism original --period 14 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h720 p14"
echo "[$(date -Is)] START Exchange h720 p30"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 720 --stage period_screen --mechanism original --period 30 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h720 p30"
echo "[$(date -Is)] START Exchange h720 p24"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 720 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange h720 p24"
echo "period worker failures=$failed"
exit "$failed"
