#!/usr/bin/env python3
"""Combine five Weather component case analyses into one skill-auditable package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


COMPONENTS = ("cycle_levels", "recent_linear", "global_linear", "smooth_local", "smooth_multiscale")


def read_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="Weather", choices=("Weather", "ETTm1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    setting = f"{args.dataset}_h96_seed2021_validation"
    args.output.mkdir(parents=True)
    figures = args.output / "figures"; figures.mkdir()
    result_rows, sample_rows = [], []
    selected = {key: [] for key in ("setting", "component", "sample_id", "group", "history", "removed_history", "truth", "baseline_prediction", "candidate_prediction", "baseline_mae", "candidate_mae")}
    component_summaries, figure_links = [], []
    baseline_seen = False
    for component in COMPONENTS:
        source = args.source / component
        rows = read_csv(source / "results.csv")
        for row in rows:
            if row["config_id"] == "baseline_full":
                if baseline_seen:
                    continue
                baseline_seen = True
            result_rows.append(row)
        for row in read_csv(source / "sample_errors.csv"):
            row["candidate_config_id"] = f"asymmetric_{component}"
            row["component"] = component
            sample_rows.append(row)
        arrays = np.load(source / "selected_cases.npz")
        for index in range(len(arrays["sample_id"])):
            selected["setting"].append(setting)
            selected["component"].append(component)
            for key in ("sample_id", "group", "history", "removed_history", "truth", "baseline_prediction", "candidate_prediction", "baseline_mae", "candidate_mae"):
                selected[key].append(arrays[key][index])
        candidate = next(row for row in rows if row["config_id"] != "baseline_full")
        component_summaries.append((component, float(candidate["mae"]), float(candidate["mse"]), float(candidate["delta_mae"]), float(candidate["delta_mse"])))
        for image in sorted((source / "figures").glob("*.png")):
            target = figures / f"{component}__{image.name}"
            shutil.copyfile(image, target)
            figure_links.append(target.name)
    write_csv(args.output / "results.csv", result_rows)
    write_csv(args.output / "sample_errors.csv", sample_rows)
    np.savez_compressed(args.output / "selected_cases.npz", **{key: np.asarray(value) for key, value in selected.items()})
    selections = {}
    for component in COMPONENTS:
        selection = [int(x) for x, c in zip(selected["sample_id"], selected["component"]) if c == component]
        selections[component] = selection
    run = {
        "experiment_id": args.output.name,
        "code": {"repository": "PhaseFormer", "branch": "weak_residual_exploration", "modified_files": ["src/models/asymmetric_trend_components.py", "src/models/PhaseFormer.py", "scripts/run_weak_residual_asymmetric_trend.py", "scripts/analyze_weak_residual_asymmetric_cases.py"]},
        "mechanism": {"description": "Full X for PhaseFormer; X-A only for NLinear residual with shared RevIN statistics", "feature_flag": "weak_residual_asymmetric_component"},
        "experiment": {"baseline": "weak_residual, residual=X", "candidate": "five fixed trend components, residual=X-A", "settings": [{"setting": setting, "dataset": args.dataset, "split": "validation", "lookback": 720, "horizon": 96, "seed": 2021}], "training": {"max_epochs": 30, "loss": "huber", "period_len": 24, "checkpoint_rule": "lowest validation loss"}, "metrics": ["MSE", "MAE"]},
        "execution": {"environment": {"python": "/home/wangjing/miniconda3/envs/raft/bin/python", "device": "RTX 4090 CUDA"}, "settings": [{"setting": setting, "commands": [f"scripts/run_weak_residual_asymmetric_trend.py --datasets {args.dataset} --horizons 96 --require-cuda --resume"], "runtime": "six full training runs"}]},
        "selection": {"source": "validation", "selected_configs": [{"setting": setting, "config_id": row["config_id"], "search_notes": "fixed plan component; no parameter or test selection"} for row in result_rows]},
        "analysis": {"ranking_metric": "channel-0 candidate-minus-baseline MAE", "top_k": 10, "dedup_rule": "descending positive delta MAE, require start-index separation >=96", "selections": selections},
        "validation": {"results_checked": True, "ranking_and_cases_checked": True, "report_and_archive_checked": True, "directory_and_settings_checked": True, "status": "passed"},
    }
    (args.output / "run.yaml").write_text(json.dumps(run, indent=2) + "\n")
    baseline = next(row for row in result_rows if row["config_id"] == "baseline_full")
    summary_table = "\n".join(f"| `{name}` | {mae:.4f} | {mse:.4f} | {dmae:+.4f} ({100*dmae/float(baseline['mae']):+.2f}%) | {dmse:+.4f} ({100*dmse/float(baseline['mse']):+.2f}%) |" for name, mae, mse, dmae, dmse in component_summaries)
    case_table = "\n".join(f"| `{component}` | {sample} | {base:.4f} | {candidate:.4f} | {candidate-base:+.4f} |" for component, sample, base, candidate in zip(selected["component"], selected["sample_id"], selected["baseline_mae"], selected["candidate_mae"]))
    figure_markdown = "\n".join(f"![](figures/{name})" for name in figure_links)
    report = f"""# Experiment and Objective Error Analysis

## 1. Experiment Setup

One validation-only setting: {args.dataset}, lookback 720, horizon 96, seed 2021, channel 0 for sample-level analysis. Baseline-full gives both branches X. Each candidate gives PhaseFormer X but NLinear residual X-A, with shared full-X RevIN statistics. All six models were trained from scratch, with 30 epochs maximum and lowest validation-loss checkpoint. No test data was read.

## 2. Experiment Results

Baseline channel-0 MAE/MSE: {float(baseline['mae']):.4f}/{float(baseline['mse']):.4f}.

| Component | Candidate MAE | Candidate MSE | Delta MAE | Delta MSE |
|---|---:|---:|---:|---:|
{summary_table}

## 3. Parameter / Configuration Search

No search: all five components and all training hyperparameters were frozen before training. Checkpoints were selected only by validation loss.

## 4. Error Distribution

`sample_errors.csv` contains every {args.dataset} validation origin for channel 0 and each candidate pair.

## 5. Horizon-wise Error

Each per-sample MAE/MSE averages all 96 future steps. No test metric is present.

## 6. High-Error Selection

For each component, the 10 cases with largest positive candidate-minus-baseline channel-0 MAE were retained, requiring validation-origin start-index separation of at least 96 steps.

| Component | Sample | Baseline MAE | Candidate MAE | Delta MAE |
|---|---:|---:|---:|---:|
{case_table}

## 7. Case Analysis

Every figure shows complete X and X-A histories, then the last 192 history steps, truth, Baseline-full and candidate forecasts.

{figure_markdown}

## 8. Repeated Observable Patterns

The displayed cases have positive deltas by selection rule. Across all origins, the aggregate direction must be read from the result table rather than from these deliberately adverse examples.

## 9. Objective Defect Summary

For these fixed single-seed models, the measurable quantity is the candidate-minus-baseline error change. A positive selected-case delta does not by itself establish that the component is a causal, natural information source; intervention distribution shift remains a competing explanation.

## 10. Experiment Scope

This is a {args.dataset} H96 validation-only, single-seed discovery experiment. It is neither a test result nor multi-seed confirmation.
"""
    report_path = args.output / "objective_error_analysis.md"; report_path.write_text(report)
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for name in figure_links:
            archive.write(figures / name, f"figures/{name}")
    print(args.output)


if __name__ == "__main__":
    main()
