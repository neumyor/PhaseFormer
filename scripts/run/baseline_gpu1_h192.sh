#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=1
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
echo "[$(date -Is)] START ETTh1 192"
$PY scripts/search_phaseformer.py --dataset ETTh1 --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTh1 192"
echo "[$(date -Is)] START ETTh2 192"
$PY scripts/search_phaseformer.py --dataset ETTh2 --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTh2 192"
echo "[$(date -Is)] START ETTm1 192"
$PY scripts/search_phaseformer.py --dataset ETTm1 --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm1 192"
echo "[$(date -Is)] START ETTm2 192"
$PY scripts/search_phaseformer.py --dataset ETTm2 --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END ETTm2 192"
echo "[$(date -Is)] START Exchange 192"
$PY scripts/search_phaseformer.py --dataset Exchange --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Exchange 192"
echo "[$(date -Is)] START Weather 192"
$PY scripts/search_phaseformer.py --dataset Weather --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Weather 192"
echo "[$(date -Is)] START Electricity 192"
$PY scripts/search_phaseformer.py --dataset Electricity --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Electricity 192"
echo "[$(date -Is)] START Traffic 192"
$PY scripts/search_phaseformer.py --dataset Traffic --horizon 192 --stage baseline --mechanism original --period 24 --percent 100 --max-epochs 30 --seed 2021 --loss huber --num-workers 4 --bad-case-limit 8 --resume || failed=$((failed+1))
$PY scripts/summarize_phaseformer_search.py
echo "[$(date -Is)] END Traffic 192"
echo "baseline worker failures=$failed"
exit "$failed"
