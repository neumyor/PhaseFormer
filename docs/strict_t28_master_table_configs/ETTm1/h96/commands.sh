#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/search_phaseformer.py --dataset ETTm1 --horizon 96 --stage confirm --mechanism pctf_anchor_repair_strict_t28 --period 24 --cycle-period 24 --lookback 720 --percent 100 --max-epochs 50 --seed 2021 --loss mae --lr-multiplier 0.2 --num-workers 0 --bad-case-limit 0 --overrides '{"anchored_pctf_correction_max":0.6,"anchored_pctf_deformation_max":0.24,"anchored_pctf_global_level_max":0.12,"anchored_pctf_shape_aux_weight":0.01,"anchored_pctf_level_aux_weight":0.01,"anchored_pctf_gate_aux_weight":0.01}' --output-dir research_runs/strict_t28_reproduction --require-cuda --evaluate-test --resume
