#!/usr/bin/env python3
"""Report head parameter budgets for the Round 1 candidates (CPU only).

Prints the residual-head parameter count of every pre-registered Round 1
configuration together with the largest ``time_axis_matched_lowrank`` rank whose
head stays inside that budget, so the matched-control table in
``scripts/run_structured_lowrank_round1.py`` can be checked against measurement.

Plan reference: section 3.3 (parameter matching).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.structured_residual_heads import (  # noqa: E402
    TimeAxisMatchedLowRankHead,
    build_structured_residual_head,
)

CANDIDATES = {
    "A_period_lowrank": (
        "structured_period_lowrank",
        {"residual_period_len": 24, "residual_period_rank": 4},
    ),
    "A_period_lowrank_r8": (
        "structured_period_lowrank",
        {"residual_period_len": 24, "residual_period_rank": 8},
    ),
    "B_segment_basis": (
        "structured_segment_basis",
        {
            "residual_period_len": 24,
            "residual_basis_count": 4,
            "residual_basis_lambda_orth": 0.01,
        },
    ),
    "C_level_shape": (
        "structured_level_shape",
        {"residual_period_len": 24, "residual_level_mode": "dense", "residual_shape_rank": 4},
    ),
    "C_level_only": (
        "structured_level_shape",
        {"residual_period_len": 24, "residual_level_mode": "dense", "residual_shape_rank": None},
    ),
    "C_level_lowrank_shape": (
        "structured_level_shape",
        {
            "residual_period_len": 24,
            "residual_level_mode": "lowrank",
            "residual_level_rank": 1,
            "residual_shape_rank": 4,
        },
    ),
    "D_recent_sparse": (
        "structured_recent_period",
        {
            "residual_period_len": 24,
            "residual_recent_taps": 7,
            "residual_recent_weighting": "fixed_exp",
            "residual_recent_decay": 0.8,
            "residual_recent_rank": 2,
        },
    ),
    "E_separable": (
        "structured_separable",
        {"residual_period_len": 24, "residual_separable_components": 1},
    ),
}

SETTINGS = (("ETTh2", 96), ("ETTh2", 720), ("ETTm2", 192), ("Electricity", 336))
LOOKBACK = 720


class _Config:
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def matched_rank(target, horizon):
    """Largest rank whose control head does not exceed ``target`` parameters.

    The control head is ``Linear(720 -> r)`` followed by ``Linear(r -> H)``, so
    its budget is ``r * (720 + H + 1) + H`` including biases.
    """

    unit = LOOKBACK + horizon + 1
    rank = max(1, (target - horizon) // unit)
    while rank > 1 and rank * unit + horizon > target:
        rank -= 1
    return rank, rank * unit + horizon


def main():
    table = {}
    for dataset, horizon in SETTINGS:
        table[f"{dataset}-H{horizon}"] = {}
        for name, (head_type, overrides) in CANDIDATES.items():
            head = build_structured_residual_head(
                head_type, LOOKBACK, horizon, _Config(dict(overrides)), seed=2021, key=name
            )
            budget = head.parameter_count()
            rank, control = matched_rank(budget, horizon)
            control_head = TimeAxisMatchedLowRankHead(LOOKBACK, horizon, rank=rank)
            assert control_head.parameter_count() == control
            table[f"{dataset}-H{horizon}"][name] = {
                "head_type": head_type,
                "head_params": budget,
                "matched_rank": rank,
                "control_params": control,
                "diff_vs_control_pct": round((budget - control) / budget * 100, 3)
                if budget
                else None,
            }
    print(json.dumps(table, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
