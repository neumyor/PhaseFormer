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
echo "[$(date -Is)] START ETTh1 336"
$PY scripts/search_phaseformer.py --dataset ETTh1 --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTh1 336"
echo "[$(date -Is)] START ETTh2 336"
$PY scripts/search_phaseformer.py --dataset ETTh2 --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTh2 336"
echo "[$(date -Is)] START ETTm1 336"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 336"
echo "[$(date -Is)] START ETTm2 336"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 336"
echo "[$(date -Is)] START Exchange 336"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange 336"
echo "[$(date -Is)] START Weather 336"
$PY scripts/search_phaseformer.py --dataset Weather --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Weather 336"
echo "[$(date -Is)] START Electricity 336"
$PY scripts/search_phaseformer.py --dataset Electricity --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Electricity 336"
echo "[$(date -Is)] START Traffic 336"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 336 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic 336"
echo "baseline worker failures=$failed"
exit "$failed"
