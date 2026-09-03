#!/usr/bin/env python3
"""Summarize matched full-input versus remove-trained D1/D2 experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
D1_PERIODS = (
    ("D1-1", 96.0), ("D1-2", 48.0), ("D1-3", 32.0), ("D1-4", 24.0),
    ("D1-5", 677.6470588235294), ("D1-6", 205.71428571428572),
)
D2_LENGTHS = (24, 48, 96, 192)
D3_COMPONENTS = (
    ("D3-global-linear", "global_linear"),
    ("D3-recent-linear", "recent_linear"),
    ("D3-cycle-levels", "cycle_levels"),
    ("D3-phase-drift", "phase_drift"),
    ("D3-cycle-amplitude", "cycle_amplitude"),
)


def read_rows(directory: Path):
    rows = []
    for path in directory.glob("*/metrics.csv"):
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        rows.append(row)
    return rows


def condition(row: dict[str, str]) -> str:
    if row["input_hypothesis"] == "d1":
        period = float(row["input_d1_period"])
        for label, expected in D1_PERIODS:
            if abs(period - expected) < 1e-8:
                return label
        raise ValueError(f"unexpected D1 period {period}")
    if row["input_hypothesis"] == "d2":
        return f"D2-{int(row['input_d2_recent_length'])}"
    if row["input_hypothesis"] == "d3":
        component = row["input_d3_component"]
        for label, expected in D3_COMPONENTS:
            if component == expected:
                return label
        raise ValueError(f"unexpected D3 component {component}")
    raise ValueError(f"unexpected hypothesis {row['input_hypothesis']}")


def pct(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--remove-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    anchors = {}
    for row in read_rows(args.baseline_dir):
        if row["input_hypothesis"] == "none" and row["input_variant"] == "full" and row["percent"] == "100":
            anchors[row["mechanism"]] = row
    if set(anchors) != set(MODELS):
        raise RuntimeError(f"need one full anchor per model, found {sorted(anchors)}")

    removed = {}
    for row in read_rows(args.remove_dir):
        if row["input_variant"] != "remove_full" or row["percent"] != "100":
            continue
        key = condition(row), row["mechanism"]
        if key in removed:
            raise RuntimeError(f"duplicate remove result for {key}")
        removed[key] = row
    active_groups = {row["input_hypothesis"] for row in removed.values()}
    expected = set()
    if "d1" in active_groups:
        expected |= {label for label, _ in D1_PERIODS}
    if "d2" in active_groups:
        expected |= {f"D2-{length}" for length in D2_LENGTHS}
    if "d3" in active_groups:
        expected |= {label for label, _ in D3_COMPONENTS}
    if set(key[0] for key in removed) != expected or {key[1] for key in removed} != set(MODELS):
        raise RuntimeError("remove results do not cover all 10 conditions and three models")

    output = []
    labels = []
    if "d1" in active_groups:
        labels += [name for name, _ in D1_PERIODS]
    if "d2" in active_groups:
        labels += [f"D2-{length}" for length in D2_LENGTHS]
    if "d3" in active_groups:
        labels += [name for name, _ in D3_COMPONENTS]
    for label in labels:
        per_model = {}
        for model in MODELS:
            base, changed = anchors[model], removed[(label, model)]
            per_model[model] = {
                "full_val_mae": float(base["val_mae"]), "remove_trained_val_mae": float(changed["val_mae"]),
                "mae_training_drop_pct": pct(float(changed["val_mae"]), float(base["val_mae"])),
                "full_val_mse": float(base["val_mse"]), "remove_trained_val_mse": float(changed["val_mse"]),
                "mse_training_drop_pct": pct(float(changed["val_mse"]), float(base["val_mse"])),
            }
        for model in MODELS:
            item = {"condition": label, "model": model, **per_model[model]}
            if model != "original":
                item["mae_interaction_vs_original_pp"] = (
                    per_model[model]["mae_training_drop_pct"] - per_model["original"]["mae_training_drop_pct"]
                )
                item["mse_interaction_vs_original_pp"] = (
                    per_model[model]["mse_training_drop_pct"] - per_model["original"]["mse_training_drop_pct"]
                )
            output.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        fieldnames = sorted({field for row in output for field in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(output)
    print(args.output)


if __name__ == "__main__":
    main()
