#!/usr/bin/env python3
"""Audit and summarize retrained or frozen H1/H3/H4 result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {("none", "full")} | {
    (hypothesis, variant)
    for hypothesis in ("h1", "h3", "h4")
    for variant in ("half_A", "minus_A", "sham")
}


def read_inputs(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        files.extend(path.rglob("frozen_metrics.csv") if path.is_dir() else [path])
    if not files:
        raise FileNotFoundError("no result CSV files found")
    frames = [pd.read_csv(path).assign(source_file=str(path)) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    if "model" not in frame and "mechanism" in frame:
        frame["model"] = frame["mechanism"]
    if "hypothesis" in frame:
        frame = frame.rename(columns={"hypothesis": "input_hypothesis", "variant": "input_variant"})
    required = {"dataset", "horizon", "seed", "model", "input_hypothesis", "input_variant", "test_mse"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    frame = read_inputs(args.inputs)
    keys = ["dataset", "horizon", "seed", "model"]
    duplicates = frame.duplicated(keys + ["input_hypothesis", "input_variant"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate condition rows detected; do not average repeated/test-selected runs silently")

    if not args.allow_incomplete:
        for setting, group in frame.groupby(keys, dropna=False):
            found = set(zip(group.input_hypothesis, group.input_variant))
            if found != EXPECTED:
                raise ValueError(
                    f"incomplete condition matrix for {setting}: missing={sorted(EXPECTED-found)} "
                    f"extra={sorted(found-EXPECTED)}"
                )
            if "checkpoint_sha256" in group and group.checkpoint_sha256.nunique() != 1:
                raise ValueError(f"frozen conditions use different checkpoints for {setting}")

    full = (
        frame[(frame.input_hypothesis == "none") & (frame.input_variant == "full")]
        [keys + ["test_mse"]]
        .rename(columns={"test_mse": "full_mse"})
    )
    result = frame.merge(full, on=keys, how="left", validate="many_to_one")
    result["delta_mse"] = result.test_mse - result.full_mse
    result["relative_delta_mse"] = result.test_mse / result.full_mse - 1.0

    sham = (
        result[result.input_variant == "sham"]
        [keys + ["input_hypothesis", "delta_mse"]]
        .rename(columns={"delta_mse": "sham_delta_mse"})
    )
    result = result.merge(
        sham, on=keys + ["input_hypothesis"], how="left", validate="many_to_one"
    )
    result["sham_adjusted_delta_mse"] = result.delta_mse - result.sham_delta_mse
    sham_relative = (
        result[result.input_variant == "sham"]
        [keys + ["input_hypothesis", "relative_delta_mse"]]
        .rename(columns={"relative_delta_mse": "sham_relative_delta_mse"})
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

    original = (
        result[result.model == "original"]
        [[
            "dataset", "horizon", "seed", "input_hypothesis", "input_variant",
            "sham_adjusted_delta_mse", "sham_adjusted_relative_mse",
        ]]
        .rename(columns={
            "sham_adjusted_delta_mse": "original_adjusted_delta_mse",
            "sham_adjusted_relative_mse": "original_adjusted_relative_mse",
        })
    )
    result = result.merge(
        original,
        on=["dataset", "horizon", "seed", "input_hypothesis", "input_variant"],
        how="left",
        validate="many_to_one",
    )
    result["interaction_mse_vs_original"] = (
        result.sham_adjusted_delta_mse - result.original_adjusted_delta_mse
    )
    result["interaction_relative_mse_vs_original"] = (
        result.sham_adjusted_relative_mse - result.original_adjusted_relative_mse
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
    ]
    grouping = ["model", "input_hypothesis", "input_variant"]
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
                    sample_values = sampled[measure].dropna().to_numpy(dtype=float)
                    if len(sample_values):
                        estimates.append(sample_values.mean())
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
