#!/usr/bin/env python3
"""Plan-§8.1 contrasts for the H1/H3/H4 input-component ablation.

Per `docs/PhaseFormer_input_component_H1_H3_H4_plan.md` §8.1,

    Delta(M, H, V)       = L(M, H, V) / L(M, H, full) - 1
    Interaction(M, H, V) = Delta(M, H, V) - Delta(M0, H, V)

with ``M0`` the original PhaseFormer.  The Interaction contrast therefore must
**not** contain a ``sham`` term: the sham adjustment answers a different question
("does the effect survive a cycle-scrambled control input?"), and mixing the two
contrasts produced aggregate values that were off by tens of percentage points
(see the D0 stage report §8-1: frozen H1 ``minus_A`` M1−M0 is ≈+2.4 pp from the
long table while the old aggregate column reported +36.1 pp).

These helpers are deliberately dependency-free (numpy + pandas only) so the
definitions can be unit tested without torch or a GPU.
"""

from __future__ import annotations

import pandas as pd

#: Settings are averaged with equal weight per dataset x horizon (plan §8.1).
SETTING_KEYS = ("dataset", "horizon")

#: ``(new column, minuend, subtrahend)`` implementing plan §8.1 exactly.
INTERACTION_SPEC = (
    (
        "interaction_relative_mse_vs_original",
        "relative_delta_mse",
        "original_relative_delta_mse",
    ),
    (
        "interaction_relative_mae_vs_original",
        "relative_delta_mae",
        "original_relative_delta_mae",
    ),
    ("interaction_mse_vs_original", "delta_mse", "original_delta_mse"),
    ("interaction_mae_vs_original", "delta_mae", "original_delta_mae"),
)

#: Legacy sham-adjusted contrasts, renamed instead of silently redefined.  They
#: answer "is the enhanced model's *sham-corrected* dependence larger than the
#: original's" and are **not** plan-§8.1 Interaction values.
SHAM_ADJUSTED_INTERACTION_RENAMES = (
    ("interaction_mse_vs_original", "sham_adjusted_interaction_mse"),
    (
        "interaction_relative_mse_vs_original",
        "sham_adjusted_interaction_relative_mse",
    ),
    ("interaction_mae_vs_original", "sham_adjusted_interaction_mae"),
    (
        "interaction_relative_mae_vs_original",
        "sham_adjusted_interaction_relative_mae",
    ),
)


def add_plan_interaction_columns(frame: pd.DataFrame, *, require: bool = True):
    """Return a copy of ``frame`` with the plan-§8.1 Interaction columns added.

    ``frame`` must already carry, per row, both the model's own deltas
    (``delta_*`` / ``relative_delta_*``) and the original model's deltas merged
    under the ``original_`` prefix.
    """
    out = frame.copy()
    missing = sorted(
        {
            column
            for _, left, right in INTERACTION_SPEC
            for column in (left, right)
            if column not in out.columns
        }
    )
    if missing:
        if require:
            raise KeyError(
                f"missing columns for plan §8.1 interaction: {missing}"
            )
        for name, _, _ in INTERACTION_SPEC:
            out[name] = float("nan")
        return out
    for name, left, right in INTERACTION_SPEC:
        out[name] = out[left] - out[right]
    return out


def rename_sham_adjusted_interactions(frame: pd.DataFrame):
    """Rename legacy sham-adjusted interaction columns to explicit names."""
    renames = {
        old: new
        for old, new in SHAM_ADJUSTED_INTERACTION_RENAMES
        if old in frame.columns and new not in frame.columns
    }
    return frame.rename(columns=renames)


def macro_mean_by_setting(frame: pd.DataFrame, measure: str, *, setting_keys=SETTING_KEYS):
    """Equal-weight mean over settings (never weighted by seeds or windows)."""
    if measure not in frame.columns or frame.empty:
        return float("nan")
    per_setting = frame.groupby(list(setting_keys), dropna=False)[measure].mean().dropna()
    return float(per_setting.mean()) if len(per_setting) else float("nan")
