#!/usr/bin/env python3
"""Audit and summarize retrained or frozen H1/H3/H4 result CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_input_component_checkpoint import moving_block_effect_interval
from scripts.run_input_component_ablation import (
    HORIZONS, SEEDS, expected_full_anchors, parse_scope,
)


EXPECTED = {("none", "full")} | {
    (hypothesis, variant)
    for hypothesis in ("h1", "h3", "h4")
    for variant in ("half_A", "minus_A", "sham")
}


def read_inputs(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(path.rglob("frozen_metrics.csv"))
            files.extend(path.rglob("retrained_metrics.csv"))
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError("no result CSV files found")
    frames = []
    for path in files:
        current = pd.read_csv(path).assign(source_file=str(path))
        if "model" not in current and "mechanism" in current:
            current["model"] = current["mechanism"]
        if "input_hypothesis" not in current and "hypothesis" in current:
            current = current.rename(
                columns={"hypothesis": "input_hypothesis", "variant": "input_variant"}
            )
        if "track" not in current:
            current["track"] = (
                "frozen" if path.name == "frozen_metrics.csv" else "retrain"
            )
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "dataset", "horizon", "seed", "model", "track", "input_hypothesis",
        "input_variant", "test_mse", "test_mae",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return frame


def sample_values(row, metric):
    source = Path(row.source_file)
    if row.track == "frozen" or source.name == "frozen_metrics.csv":
        archive = source.parent / "paired_sample_errors.npz"
        key = f"{row.input_hypothesis}_{row.input_variant}__{metric}"
    else:
        archive = source.parent / "sample_errors.npz"
        key = metric
    if not archive.is_file():
        raise FileNotFoundError(f"missing paired sample errors: {archive}")
    with np.load(archive) as values:
        if key not in values:
            raise KeyError(f"missing {key} in {archive}")
        return values[key].astype(np.float64, copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                       help="comma list of horizons to include (default: all)")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                       help="comma list of seeds to include (default: all)")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--expected-settings-per-track", type=int,
                       help="defaults to the number of (dataset,horizon,seed,model) "
                            "settings in the requested --horizons/--seeds scope")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    horizons, seeds = parse_scope(parser, args.horizons, args.seeds)
    if args.expected_settings_per_track is None:
        args.expected_settings_per_track = expected_full_anchors(horizons, seeds)
    frame = read_inputs(args.inputs)
    frame = frame[frame.horizon.isin(horizons) & frame.seed.isin(seeds)]
    if not args.allow_smoke:
        if "evaluation_scope" not in frame or (frame.evaluation_scope != "formal").any():
            raise ValueError("formal summary refuses smoke or unlabelled evaluation rows")
        if "max_samples" in frame and frame.max_samples.fillna(0).astype(int).ne(0).any():
            raise ValueError("formal summary refuses partial --max-samples results")
    keys = ["dataset", "horizon", "seed", "model", "track"]
    duplicates = frame.duplicated(keys + ["input_hypothesis", "input_variant"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate condition rows detected; do not average repeated/test-selected runs silently")

    # The single none/full evaluation belongs to both tracks.  Reuse the
    # frozen row as Track R's baseline instead of reading test twice.
    if (frame.track == "retrain").any():
        retrain_settings = frame[frame.track == "retrain"][
            ["dataset", "horizon", "seed", "model"]
        ].drop_duplicates()
        shared_full = frame[
            (frame.track == "frozen")
            & (frame.input_hypothesis == "none")
            & (frame.input_variant == "full")
        ].merge(
            retrain_settings,
            on=["dataset", "horizon", "seed", "model"],
            how="inner",
        )
        if len(shared_full) != len(retrain_settings):
            raise ValueError("every retrained setting requires its frozen none/full baseline")
        shared_full["track"] = "retrain"
        frame = pd.concat([frame, shared_full], ignore_index=True)

    if not args.allow_incomplete:
        if set(frame.track) != {"frozen", "retrain"}:
            raise ValueError("formal summary requires both frozen and retrain tracks")
        for setting, group in frame.groupby(keys, dropna=False):
            found = set(zip(group.input_hypothesis, group.input_variant))
            if found != EXPECTED:
                raise ValueError(
                    f"incomplete condition matrix for {setting}: missing={sorted(EXPECTED-found)} "
                    f"extra={sorted(found-EXPECTED)}"
                )
            if (
                setting[-1] == "frozen"
                and "checkpoint_sha256" in group
                and group.checkpoint_sha256.nunique() != 1
            ):
                raise ValueError(f"frozen conditions use different checkpoints for {setting}")
        setting_counts = (
            frame[["dataset", "horizon", "seed", "model", "track"]]
            .drop_duplicates()
            .groupby("track")
            .size()
        )
        if (setting_counts != args.expected_settings_per_track).any():
            raise ValueError(
                "formal summary setting counts differ from "
                f"{args.expected_settings_per_track}: {setting_counts.to_dict()}"
            )

    full = (
        frame[(frame.input_hypothesis == "none") & (frame.input_variant == "full")]
        [keys + ["test_mse", "test_mae"]]
        .rename(columns={"test_mse": "full_mse", "test_mae": "full_mae"})
    )
    result = frame.merge(full, on=keys, how="left", validate="many_to_one")
    result["delta_mse"] = result.test_mse - result.full_mse
    result["relative_delta_mse"] = result.test_mse / result.full_mse - 1.0
    result["delta_mae"] = result.test_mae - result.full_mae
    result["relative_delta_mae"] = result.test_mae / result.full_mae - 1.0

    for metric in ("mse", "mae"):
        for effect in ("absolute", "relative"):
            for bound in ("low", "high"):
                column = f"{metric}_{effect}_effect_ci_{bound}"
                if column not in result:
                    result[column] = np.nan
    for setting, group in result.groupby(keys, dropna=False):
        full_row = group[
            (group.input_hypothesis == "none") & (group.input_variant == "full")
        ].iloc[0]
        for index, row in group.iterrows():
            for metric in ("mse", "mae"):
                full_values = sample_values(full_row, metric)
                variant_values = sample_values(row, metric)
                for relative in (False, True):
                    low, high = moving_block_effect_interval(
                        full_values, variant_values,
                        block_length=int(row.horizon),
                        seed=9102,
                        replicates=args.bootstrap_replicates,
                        relative=relative,
                    )
                    effect = "relative" if relative else "absolute"
                    result.loc[index, f"{metric}_{effect}_effect_ci_low"] = low
                    result.loc[index, f"{metric}_{effect}_effect_ci_high"] = high

    sham = (
        result[result.input_variant == "sham"]
        [keys + ["input_hypothesis", "delta_mse", "delta_mae"]]
        .rename(columns={"delta_mse": "sham_delta_mse", "delta_mae": "sham_delta_mae"})
    )
    result = result.merge(
        sham, on=keys + ["input_hypothesis"], how="left", validate="many_to_one"
    )
    result["sham_adjusted_delta_mse"] = result.delta_mse - result.sham_delta_mse
    result["sham_adjusted_delta_mae"] = result.delta_mae - result.sham_delta_mae
    sham_relative = (
        result[result.input_variant == "sham"]
        [keys + ["input_hypothesis", "relative_delta_mse", "relative_delta_mae"]]
        .rename(columns={
            "relative_delta_mse": "sham_relative_delta_mse",
            "relative_delta_mae": "sham_relative_delta_mae",
        })
    )
    result = result.merge(
        sham_relative,
        on=keys + ["input_hypothesis"],
        how="left",
        validate="many_to_one",
    )
    result["sham_adjusted_relative_mse"] = (
        result.relative_delta_mse - result.sham_relative_delta_mse
    )
    result["sham_adjusted_relative_mae"] = (
        result.relative_delta_mae - result.sham_relative_delta_mae
    )

    original = (
        result[result.model == "original"]
        [[
            "dataset", "horizon", "seed", "track", "input_hypothesis", "input_variant",
            "sham_adjusted_delta_mse", "sham_adjusted_relative_mse",
            "sham_adjusted_delta_mae", "sham_adjusted_relative_mae",
        ]]
        .rename(columns={
            "sham_adjusted_delta_mse": "original_adjusted_delta_mse",
            "sham_adjusted_relative_mse": "original_adjusted_relative_mse",
            "sham_adjusted_delta_mae": "original_adjusted_delta_mae",
            "sham_adjusted_relative_mae": "original_adjusted_relative_mae",
        })
    )
    result = result.merge(
        original,
        on=["dataset", "horizon", "seed", "track", "input_hypothesis", "input_variant"],
        how="left",
        validate="many_to_one",
    )
    result["interaction_mse_vs_original"] = (
        result.sham_adjusted_delta_mse - result.original_adjusted_delta_mse
    )
    result["interaction_relative_mse_vs_original"] = (
        result.sham_adjusted_relative_mse - result.original_adjusted_relative_mse
    )
    result["interaction_mae_vs_original"] = (
        result.sham_adjusted_delta_mae - result.original_adjusted_delta_mae
    )
    result["interaction_relative_mae_vs_original"] = (
        result.sham_adjusted_relative_mae - result.original_adjusted_relative_mae
    )
    result["qc_status"] = "ok"
    if "input_endpoint_max_abs" in result:
        result.loc[result.input_endpoint_max_abs > 1e-5, "qc_status"] = "endpoint_failed"
    if "qc_changed_fraction" in result:
        inactive = (
            (result.input_variant != "full")
            & (result.qc_changed_fraction == 0)
        )
        result.loc[inactive, "qc_status"] = "inactive_or_unidentifiable"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(keys + ["input_hypothesis", "input_variant"]).to_csv(
        args.output, index=False
    )
    aggregate_rows = []
    rng = np.random.default_rng(9102)
    measures = [
        "relative_delta_mse",
        "sham_adjusted_relative_mse",
        "interaction_relative_mse_vs_original",
        "relative_delta_mae",
        "sham_adjusted_relative_mae",
        "interaction_relative_mae_vs_original",
    ]
    grouping = ["track", "model", "input_hypothesis", "input_variant"]
    for group_key, group in result.groupby(grouping, dropna=False):
        clusters = list(group.groupby(["dataset", "horizon"], dropna=False))
        row = dict(zip(grouping, group_key))
        row["settings"] = len(clusters)
        row["rows"] = len(group)
        for measure in measures:
            values = group[measure].dropna().to_numpy(dtype=float)
            row[f"mean_{measure}"] = float(values.mean()) if len(values) else np.nan
            estimates = []
            if len(clusters) >= 2 and args.bootstrap_replicates > 0:
                for _ in range(args.bootstrap_replicates):
                    selected = rng.integers(0, len(clusters), size=len(clusters))
                    sampled = pd.concat([clusters[index][1] for index in selected])
                    sampled_measure_values = sampled[measure].dropna().to_numpy(dtype=float)
                    if len(sampled_measure_values):
                        estimates.append(sampled_measure_values.mean())
            if estimates:
                row[f"{measure}_ci_low"], row[f"{measure}_ci_high"] = np.quantile(
                    estimates, [0.025, 0.975]
                )
            else:
                row[f"{measure}_ci_low"] = np.nan
                row[f"{measure}_ci_high"] = np.nan
        aggregate_rows.append(row)
    aggregate_path = args.output.with_name(
        f"{args.output.stem}_aggregate{args.output.suffix}"
    )
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    print(args.output)
    print(aggregate_path)


if __name__ == "__main__":
    main()
