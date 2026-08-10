#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=3
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
echo "[$(date -Is)] START ETTm2 h96 p24"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 96 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h96 p24"
echo "[$(date -Is)] START ETTm2 h96 p48"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 96 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h96 p48"
echo "[$(date -Is)] START ETTm2 h96 p96"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 96 --stage period_screen --mechanism original --period 96 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h96 p96"
echo "[$(date -Is)] START ETTm2 h720 p24"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 720 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h720 p24"
echo "[$(date -Is)] START ETTm2 h720 p48"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 720 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h720 p48"
echo "[$(date -Is)] START ETTm2 h720 p96"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 720 --stage period_screen --mechanism original --period 96 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 h720 p96"
echo "[$(date -Is)] START Traffic h96 p12"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 96 --stage period_screen --mechanism original --period 12 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h96 p12"
echo "[$(date -Is)] START Traffic h96 p24"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 96 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h96 p24"
echo "[$(date -Is)] START Traffic h96 p48"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 96 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h96 p48"
echo "[$(date -Is)] START Traffic h720 p12"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 720 --stage period_screen --mechanism original --period 12 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h720 p12"
echo "[$(date -Is)] START Traffic h720 p24"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 720 --stage period_screen --mechanism original --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h720 p24"
echo "[$(date -Is)] START Traffic h720 p48"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 720 --stage period_screen --mechanism original --period 48 --percent 30 --max-epochs 8 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic h720 p48"
echo "period worker failures=$failed"
exit "$failed"
