#!/bin/bash
# Full-budget confirm for the next-stage paper plan (stages 1-3 + final model).
# Runs sequentially so each driver owns the whole 4-GPU pool; --resume skips
# runs whose metrics.csv already exists.
set -e
PY=/home/niuyiming/.conda/envs/py310/bin/python
ROOT=/home/niuyiming/PhaseFormer
cd "$ROOT"

MODES=phase_velocity,phase_vel_geo,residual_adaptive,next_full,no_residual

echo "=== [1/3] seed 2021, horizons 336 (all 5 datasets) ==="
$PY scripts/run_dyn_phase_full.py \
  --datasets ETTh2,ETTm1,Electricity,Traffic,ETTh1 \
  --horizons 336 --modes "$MODES" \
  --output-dir research_runs/dyn_phase_full --run-prefix dynphase --seed 2021

echo "=== [2/3] seed 2021, horizons 720 (4 datasets, ETTh1 h720 uses seed 2026) ==="
$PY scripts/run_dyn_phase_full.py \
  --datasets ETTh2,ETTm1,Electricity,Traffic \
  --horizons 720 --modes "$MODES" \
  --output-dir research_runs/dyn_phase_full --run-prefix dynphase --seed 2021

echo "=== [3/3] seed 2026, ETTh1 h720 (matches existing baseline seed) ==="
$PY scripts/run_dyn_phase_full.py \
  --datasets ETTh1 --horizons 720 --modes "$MODES" \
  --output-dir research_runs/dyn_phase_full --run-prefix dynphase --seed 2026

echo "ALL NEXT-STAGE FULL-BUDGET BATCHES DONE"
