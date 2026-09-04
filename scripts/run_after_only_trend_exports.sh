#!/usr/bin/env bash
# Wait without sharing a GPU with the only-A trainer, then export both probes.
set -euo pipefail

scratch_root="research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch/runs"
while true; do
  completed=$(find "$scratch_root" -name status.json -print0 | xargs -0 rg -l '"status": "completed"' | wc -l)
  running=$(find "$scratch_root" -name status.json -print0 | xargs -0 rg -l '"status": "running"' | wc -l)
  if [ "$completed" -eq 15 ]; then
    break
  fi
  if [ "$running" -eq 0 ]; then
    echo "only-A training stopped before 15 completions (completed=$completed)" >&2
    exit 1
  fi
  sleep 60
done

export MPLCONFIGDIR="research_runs/asymmetric_prediction_divergence_cases/mpl"
python_bin="/home/wangjing/miniconda3/envs/raft/bin/python"
"$python_bin" scripts/export_asymmetric_prediction_divergence_cases.py \
  --output research_runs/asymmetric_prediction_divergence_cases/X_minus_A \
  --input-mode minus_component --require-cuda
"$python_bin" scripts/export_asymmetric_prediction_divergence_cases.py \
  --output research_runs/asymmetric_prediction_divergence_cases/Only_A \
  --candidate-root research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch \
  --input-mode component_only --require-cuda
