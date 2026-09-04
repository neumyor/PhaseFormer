#!/usr/bin/env python3
"""Audit high-error ETTh1 validation cases for one asymmetric residual input."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.asymmetric_trend_components import TREND_COMPONENTS, extract_trend_component
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args


def find_run(root: Path, dataset: str, horizon: int, component: str | None) -> Path:
    matches = []
    for path in root.glob("runs/*"):
        config = path / "config.json"
        if not config.is_file():
            continue
        data = json.loads(config.read_text())
        if data.get("dataset") != dataset or data.get("horizon") != horizon:
            continue
        actual = data["hyperparams"].get("weak_residual_asymmetric_component")
        if actual == component:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {dataset}-H{horizon} run for {component!r}, found {matches}")
    return matches[0]


def load_model(run_dir: Path, device: torch.device):
    config = json.loads((run_dir / "config.json").read_text())
    hp = config["hyperparams"]
    args = make_exp_args(config["dataset"], config["lookback"], config["horizon"], hp)
    model = PhaseFormer(PhaseFormerPresetConfig(args, config["lookback"], config["horizon"], hp))
    result = next(csv.DictReader((run_dir / "metrics.csv").open()))
    checkpoint = ROOT / result["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.to(device).eval()
    return model, args, checkpoint


def select_cases(baseline_mae: np.ndarray, candidate_mae: np.ndarray, count: int = 10, min_separation: int = 96):
    """Find non-overlapping cases where hiding A most increases channel-0 MAE."""
    ranked = np.argsort(candidate_mae - baseline_mae)[::-1]
    selected, seen = [], set()
    for index in ranked:
        index = int(index)
        if index not in seen and all(abs(index - prior) >= min_separation for _, prior in selected):
            selected.append(("candidate_mae_regression", index))
            seen.add(index)
        if len(selected) == count:
            return selected
    raise RuntimeError("not enough distinct validation cases")


def plot_case(path, component, sample_id, history, residual_history, truth, baseline, candidate, baseline_mae, candidate_mae):
    horizon = len(truth)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.7), dpi=150, sharex=False)
    history_x = np.arange(-len(history), 0)
    future_x = np.arange(horizon)
    axes[0].plot(history_x, history, color="#555555", lw=0.85, label="full history X")
    axes[0].plot(history_x, residual_history, color="#E07A2D", lw=0.85, alpha=0.9, label=f"residual history X-{component}")
    axes[0].axvline(0, color="#999999", lw=0.8)
    axes[0].set_title(f"ETTh1 H96 validation sample {sample_id}, channel 0 — {component}")
    axes[0].set_ylabel("scaled value")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)

    recent = min(192, len(history))
    axes[1].plot(np.arange(-recent, 0), history[-recent:], color="#555555", lw=0.9, label="full history X")
    axes[1].plot(np.arange(-recent, 0), residual_history[-recent:], color="#E07A2D", lw=0.9, alpha=0.9, label="X-A1")
    axes[1].plot(future_x, truth, color="#111111", lw=1.6, label="future truth")
    axes[1].plot(future_x, baseline, color="#2878B5", lw=1.2, label="Baseline-full")
    axes[1].plot(future_x, candidate, color="#C43C39", lw=1.2, label=f"Asymmetric-{component}")
    axes[1].axvline(0, color="#999999", lw=0.8)
    axes[1].set_xlabel("forecast step (history is negative)")
    axes[1].set_ylabel("scaled value")
    axes[1].set_title(f"channel-0 MAE: baseline={baseline_mae:.4f}, asymmetric={candidate_mae:.4f}")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_trend_discovery"))
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTm1", "Weather"])
    parser.add_argument("--horizon", type=int, default=96, choices=[96])
    parser.add_argument("--component", choices=sorted(TREND_COMPONENTS), default="cycle_levels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was requested but CUDA is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_run = find_run(args.discovery_root, args.dataset, args.horizon, None)
    component_name = args.component
    candidate_run = find_run(args.discovery_root, args.dataset, args.horizon, component_name)
    baseline_model, exp_args, baseline_checkpoint = load_model(baseline_run, device)
    candidate_model, _, candidate_checkpoint = load_model(candidate_run, device)
    exp_args.dataset_args.num_workers = args.num_workers
    dataset, loader = data_provider(exp_args.dataset_args, "val")
    setting = f"{args.dataset}_h{args.horizon}_seed2021_validation"

    histories = []; removed_histories = []; truths = []; baseline_preds = []; candidate_preds = []
    with torch.inference_mode():
        for batch in loader:
            x, y, xm, ym = [item.to(device) for item in batch]
            decoder = baseline_model._build_decoder_input(y.float())
            base, _, _ = baseline_model(x.float(), xm.float(), decoder, ym.float())
            candidate, _, _ = candidate_model(x.float(), xm.float(), decoder, ym.float())
            component = extract_trend_component(x.float(), component_name, period_len=24)
            histories.append(x[:, :, 0].cpu().numpy())
            removed_histories.append((x - component)[:, :, 0].cpu().numpy())
            truths.append(y[:, -args.horizon:, 0].float().cpu().numpy())
            baseline_preds.append(base[:, -args.horizon:, 0].cpu().numpy())
            candidate_preds.append(candidate[:, -args.horizon:, 0].cpu().numpy())
    history = np.concatenate(histories); removed_history = np.concatenate(removed_histories)
    truth = np.concatenate(truths); baseline = np.concatenate(baseline_preds); candidate = np.concatenate(candidate_preds)
    baseline_mae = np.abs(baseline - truth).mean(axis=1)
    candidate_mae = np.abs(candidate - truth).mean(axis=1)
    baseline_mse = ((baseline - truth) ** 2).mean(axis=1)
    candidate_mse = ((candidate - truth) ** 2).mean(axis=1)
    selected = select_cases(baseline_mae, candidate_mae)

    args.output.mkdir(parents=True)
    figures = args.output / "figures"; figures.mkdir()
    rows = []
    selected_arrays = {key: [] for key in ("sample_id", "group", "history", "removed_history", "truth", "baseline_prediction", "candidate_prediction", "baseline_mae", "candidate_mae")}
    for sample_id in range(len(truth)):
        rows.append({"setting": setting, "baseline_config_id": "baseline_full", "candidate_config_id": f"asymmetric_{component_name}", "sample_id": sample_id, "channel": 0,
                     "baseline_mae": baseline_mae[sample_id], "candidate_mae": candidate_mae[sample_id],
                     "delta_mae": candidate_mae[sample_id] - baseline_mae[sample_id],
                     "baseline_mse": baseline_mse[sample_id], "candidate_mse": candidate_mse[sample_id],
                     "delta_mse": candidate_mse[sample_id] - baseline_mse[sample_id]})
    with (args.output / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for rank, (group, sample_id) in enumerate(selected, start=1):
        plot_case(figures / f"case_{rank:02d}_sample_{sample_id}.png", component_name, sample_id, history[sample_id], removed_history[sample_id], truth[sample_id], baseline[sample_id], candidate[sample_id], baseline_mae[sample_id], candidate_mae[sample_id])
        for key, value in (("sample_id", sample_id), ("group", group), ("history", history[sample_id]), ("removed_history", removed_history[sample_id]), ("truth", truth[sample_id]), ("baseline_prediction", baseline[sample_id]), ("candidate_prediction", candidate[sample_id]), ("baseline_mae", baseline_mae[sample_id]), ("candidate_mae", candidate_mae[sample_id])):
            selected_arrays[key].append(value)
    np.savez_compressed(args.output / "selected_cases.npz", **{key: np.asarray(value) for key, value in selected_arrays.items()})
    result_rows = [
        {"setting": setting, "config_id": "baseline_full", "dataset": args.dataset, "horizon": args.horizon, "seed": 2021, "model": "weak_residual", "key_params": "residual=X", "mse": float(baseline_mse.mean()), "mae": float(baseline_mae.mean()), "delta_mse": 0.0, "delta_mae": 0.0, "selected": True},
        {"setting": setting, "config_id": f"asymmetric_{component_name}", "dataset": args.dataset, "horizon": args.horizon, "seed": 2021, "model": "weak_residual", "key_params": f"residual=X-{component_name}", "mse": float(candidate_mse.mean()), "mae": float(candidate_mae.mean()), "delta_mse": float(candidate_mse.mean()-baseline_mse.mean()), "delta_mae": float(candidate_mae.mean()-baseline_mae.mean()), "selected": True},
    ]
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    run = {"experiment_id": args.output.name, "setting": setting, "component": component_name, "channel": 0,
           "selection": "top 10 candidate-minus-baseline MAE regressions with >=96-start-index separation", "split": "validation only", "test_read": False,
           "baseline_run": str(baseline_run), "candidate_run": str(candidate_run), "baseline_checkpoint": str(baseline_checkpoint), "candidate_checkpoint": str(candidate_checkpoint)}
    (args.output / "run.yaml").write_text(json.dumps(run, indent=2) + "\n")
    table = "\n".join(f"| {rank} | {group} | {sample_id} | {baseline_mae[sample_id]:.4f} | {candidate_mae[sample_id]:.4f} | {candidate_mae[sample_id]-baseline_mae[sample_id]:+.4f} |" for rank, (group, sample_id) in enumerate(selected, 1))
    report = f"""# Experiment and Objective Error Analysis

## 1. Experiment Setup

Setting: `{setting}`; channel 0; validation only; no test data was loaded.

## 2. Experiment Results

The comparison is independently trained Baseline-full Weak Residual versus Asymmetric-{component_name}, where the PhaseFormer path sees X and only the NLinear residual path sees X-A.  The component is extracted deterministically and endpoint anchored before residual-branch RevIN normalization.

Channel-0 aggregate error: baseline MAE {baseline_mae.mean():.4f}, asymmetric MAE {candidate_mae.mean():.4f} ({100*(candidate_mae.mean()/baseline_mae.mean()-1):+.2f}%); baseline MSE {baseline_mse.mean():.4f}, asymmetric MSE {candidate_mse.mean():.4f} ({100*(candidate_mse.mean()/baseline_mse.mean()-1):+.2f}%).

## 3. Parameter / Configuration Search

No parameter search: the plan-fixed baseline and one plan-fixed component condition were trained with seed 2021 and selected by validation checkpoint loss.

## 4. Error Distribution

`sample_errors.csv` contains every validation origin for channel 0.

## 5. Horizon-wise Error

The ranking metric is the MAE averaged over all {args.horizon} future steps.

## 6. High-Error Selection

The ten cases are the largest positive values of `Asymmetric channel-0 MAE - Baseline channel-0 MAE`, retaining origins at least 96 steps apart.  Thus every displayed case is one where denying this component to NLinear is maximally harmful within this validation split; it is not a significance test.

| Rank | Selection source | Validation sample | Baseline MAE | Asymmetric MAE | Delta |
|---:|---|---:|---:|---:|---:|
{table}

## 7. Case Analysis

Each figure has the full 720-step history and the exact component-removed residual-branch history above; below it has the final 192 history steps, future truth, and both predictions.  Corresponding numeric arrays are in `selected_cases.npz`; all channel-0 validation-origin errors are in `sample_errors.csv`.

## 8. Repeated Observable Patterns

The selected examples all have positive channel-0 MAE deltas by construction; the report does not infer a causal mechanism from their shapes.

## 9. Objective Defect Summary

The measurable defect under this intervention is the reported positive MAE difference.  Whether it represents component use rather than intervention-induced distribution shift requires the broader paired evidence.

## 10. Experiment Scope

This is one dataset/horizon/seed validation-only diagnostic, not a test result or a multi-seed confirmation.
"""
    report_path = args.output / "objective_error_analysis.md"; report_path.write_text(report)
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")
    print(args.output)


if __name__ == "__main__":
    main()
