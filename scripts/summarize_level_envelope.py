#!/usr/bin/env python3
"""Pool the ``verify_level_envelope.py`` shards into per-setting tables.

Envelope constants and shares are averaged over seeds (with the across-seed
standard deviation).  Grouped gains are pooled over seeds as window-weighted
sums, which keeps the per-window gains of small groups (the out-of-envelope
windows) stable where a per-seed ratio would divide by a near-zero total.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def read(pattern_by_dir: dict[str, set[str] | None], stem: str) -> pd.DataFrame:
    frames = []
    for directory, datasets in pattern_by_dir.items():
        for path in sorted(glob.glob(str(REPO_ROOT / directory / f"{stem}_shard*.csv"))):
            frame = pd.read_csv(path)
            if datasets is not None:
                frame = frame[frame.dataset.isin(datasets)]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/level_envelope_v2")
    args = parser.parse_args()
    sources = {args.output_dir: None}
    cells = read(sources, "envelope_cells")
    groups = read(sources, "envelope_groups")
    keys = ["dataset", "horizon", "seed", "split"]
    if cells.duplicated(keys).any():
        raise SystemExit("duplicate cells")

    checks = {
        "cells": int(len(cells)),
        "premise_violations": sorted((set(cells.phase_premise_violations.fillna(""))
                                      | set(cells.fused_premise_violations.fillna(""))) - {""}),
        "max_phase_util": float(cells.phase_util_max.max()),
        "max_fused_phase_util": float(cells.fused_phase_util_max.max()),
        "max_reconstruction_abs": float(max(cells.phase_reconstruction_max_abs.max(),
                                            cells.fused_reconstruction_max_abs.max())),
    }
    test = cells[cells.split == "test"]
    checks["max_rel_diff_phase_test_mse"] = float(
        ((test.phase_mse - test.phase_recorded_test_mse) / test.phase_recorded_test_mse).abs().max())
    checks["max_rel_diff_fused_test_mse"] = float(
        ((test.fused_mse - test.fused_recorded_test_mse) / test.fused_recorded_test_mse).abs().max())

    metrics = ["c0", "C1", "C_phi", "phase_util_max", "frac_outside", "frac_abs_d_gt_Cphi",
               "abs_d_p99", "level_share", "floor_share", "phase_mse", "fused_mse", "gain",
               "level_gain", "shape_gain", "phase_level_tracking", "fused_C1",
               "fused_phase_level_tracking", "fused_level_tracking", "branch_level_energy_share",
               "phase_level_tracking_data", "fused_phase_level_tracking_data",
               "fused_level_tracking_data", "branch_level_tracking_data",
               "branch_level_energy_share_data", "phase_level_spearman",
               "fused_phase_level_spearman",
               "gate_mean", "fused_phase_frac_outside", "phase_abs_level_mean",
               "fused_phase_abs_level_mean", "frac_phase_util_gt_0.9",
               "spearman_absd_vs_gain", "spearman_phase_level_err_vs_gain"]
    by_setting = cells.groupby(["dataset", "horizon", "split"])[metrics]
    setting = by_setting.mean().add_suffix("_mean").join(
        by_setting.std().add_suffix("_sd")).reset_index()

    groups = groups.copy()
    for column in ("gain", "level_gain", "shape_gain"):
        groups[f"{column}_sum"] = groups[f"{column}_per_window"] * groups.windows
    groups["phase_sse"] = groups.phase_mse * groups.windows
    pooled = groups.groupby(["dataset", "horizon", "split", "grouping", "group"]).agg(
        windows=("windows", "sum"), gain_sum=("gain_sum", "sum"),
        level_gain_sum=("level_gain_sum", "sum"), shape_gain_sum=("shape_gain_sum", "sum"),
        phase_sse=("phase_sse", "sum")).reset_index()
    totals = pooled[pooled.grouping == "abs_d_quartile"].groupby(
        ["dataset", "horizon", "split"]).agg(total_windows=("windows", "sum"),
                                             total_gain=("gain_sum", "sum"),
                                             total_level_gain=("level_gain_sum", "sum"))
    pooled = pooled.join(totals, on=["dataset", "horizon", "split"])
    pooled["window_frac"] = pooled.windows / pooled.total_windows
    pooled["gain_per_window"] = pooled.gain_sum / pooled.windows
    pooled["level_gain_per_window"] = pooled.level_gain_sum / pooled.windows
    pooled["shape_gain_per_window"] = pooled.shape_gain_sum / pooled.windows
    pooled["gain_share"] = pooled.gain_sum / pooled.total_gain
    pooled["level_fraction_of_gain"] = pooled.level_gain_sum / pooled.gain_sum
    pooled["relative_gain"] = pooled.gain_sum / pooled.phase_sse

    envelope = pooled[pooled.grouping == "envelope"].pivot_table(
        index=["dataset", "horizon", "split"], columns="group",
        values=["window_frac", "gain_per_window", "level_gain_per_window",
                "shape_gain_per_window", "gain_share", "relative_gain"])
    envelope.columns = [f"{a}_{b}" for a, b in envelope.columns]
    envelope = envelope.reset_index()
    envelope["per_window_lift"] = envelope.gain_per_window_outside / envelope.gain_per_window_inside

    output_dir = REPO_ROOT / args.output_dir
    setting.to_csv(output_dir / "summary_setting.csv", index=False)
    pooled.to_csv(output_dir / "summary_groups_pooled.csv", index=False)
    envelope.to_csv(output_dir / "summary_p5_envelope.csv", index=False)
    (output_dir / "summary_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
