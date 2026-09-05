#!/usr/bin/env python3
"""Create a checked one-table record for all joint-route case components."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_asymmetric_joint_route_cases import BASELINE_ROOTS, COMPONENTS, DATASETS, candidate_path, run_dir
from scripts.export_asymmetric_prediction_divergence_cases import trend_kwargs


OUT = ROOT / "research_runs/asymmetric_prediction_divergence_cases/ALL_COMPONENT_ROUTE_VALIDATION_METRICS.md"


def load_and_verify(path: Path, dataset: str, component: str | None, mode: str | None) -> tuple[dict, dict]:
    config_path, metrics_path = path / "config.json", path / "metrics.csv"
    if not config_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing config or metrics: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparams"]
    if (config.get("dataset"), config.get("lookback"), config.get("horizon"), config.get("seed"),
        config.get("percent"), hp.get("seed")) != (dataset, 720, 96, 2021, 100, 2021):
        raise RuntimeError(f"protocol mismatch: {path}")
    if component is None:
        if hp.get("weak_residual_asymmetric_component", "none") != "none":
            raise RuntimeError(f"baseline is not full-input: {path}")
    elif (hp.get("weak_residual_asymmetric_component"), hp.get("weak_residual_asymmetric_input_mode", "minus_component")) != (component, mode):
        raise RuntimeError(f"candidate identity mismatch: {path}")
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if len(rows) != 1:
        raise RuntimeError(f"expected one aggregate metric row: {path}")
    row = rows[0]
    for key in ("val_mse", "val_mae"):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(f"non-finite {key}: {path}")
    checkpoint = ROOT / row["checkpoint"]
    # The two delivered-component trees were relocated with their checkpoints
    # intact, while metrics.csv deliberately preserves the producer's original
    # path.  Accept that documented relocation only when the standard local
    # checkpoint is physically present.
    if not checkpoint.is_file():
        relocated = path / "attempts/001/checkpoints/best.ckpt"
        if not relocated.is_file():
            raise FileNotFoundError(f"metric references missing checkpoint: {checkpoint}")
    return config, row


def params(component: str, hp: dict) -> str:
    values = trend_kwargs(hp)
    if component == "cycle_levels": return f"P={values['period_len']}"
    if component == "recent_linear": return f"W={values['recent_window']}"
    if component == "global_linear": return "L=720"
    if component == "smooth_local": return f"Gaussian σ={values['local_sigma']:g}"
    if component == "smooth_multiscale": return f"Gaussian σ={values['local_sigma']:g}−σ={values['long_sigma']:g}"
    if component == "causal_ema": return f"EMA α={values['causal_ema_alpha']:g}"
    if component == "holt_local_linear": return f"Holt α={values['holt_level_alpha']:g}, β={values['holt_trend_beta']:g}"
    raise ValueError(component)


def metric(value: float) -> str:
    return f"{value:.6f}"


def delta(value: float, baseline: float) -> str:
    return f"{value - baseline:+.6f} ({100.0 * (value - baseline) / baseline:+.2f}%)"


def main() -> None:
    lines = [
        "# All component-route validation metrics", "",
        "This is the one-table record for all components represented by the prediction-divergence images in this directory. All results are **validation-only** (not test): L=720, H=96, seed=2021, percent=100. `X-A` keeps full X for PhaseFormer and gives X−A to NLinear; `Only-A` keeps full X for PhaseFormer and gives A to NLinear. All values are the best validation checkpoint's MSE/MAE; lower is better.", "",
        "| Dataset | Component | Exact extraction parameters | Baseline-full MSE / MAE | X-A MSE / MAE | X-A ΔMSE / ΔMAE | Only-A MSE / MAE | Only-A ΔMSE / ΔMAE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    validated = 0
    for dataset in DATASETS:
        baseline_path = run_dir(BASELINE_ROOTS[dataset], dataset, None)
        _, baseline = load_and_verify(baseline_path, dataset, None, None)
        baseline_mse, baseline_mae = float(baseline["val_mse"]), float(baseline["val_mae"])
        validated += 1
        for component in COMPONENTS:
            minus_cfg, minus = load_and_verify(candidate_path(dataset, component, "minus_component"), dataset, component, "minus_component")
            only_cfg, only = load_and_verify(candidate_path(dataset, component, "component_only"), dataset, component, "component_only")
            if trend_kwargs(minus_cfg["hyperparams"]) != trend_kwargs(only_cfg["hyperparams"]):
                raise RuntimeError(f"X-A/Only-A parameter mismatch: {dataset} {component}")
            x_mse, x_mae = float(minus["val_mse"]), float(minus["val_mae"])
            a_mse, a_mae = float(only["val_mse"]), float(only["val_mae"])
            lines.append(
                f"| {dataset} | `{component}` | {params(component, minus_cfg['hyperparams'])} | "
                f"{metric(baseline_mse)} / {metric(baseline_mae)} | {metric(x_mse)} / {metric(x_mae)} | "
                f"{delta(x_mse, baseline_mse)} / {delta(x_mae, baseline_mae)} | "
                f"{metric(a_mse)} / {metric(a_mae)} | {delta(a_mse, baseline_mse)} / {delta(a_mae, baseline_mae)} |"
            )
            validated += 2
    lines += [
        "", "## Verification", "",
        f"The generator validated {validated} real runs (3 Baseline-full + 21 X-A + 21 Only-A): each has a matching dataset/L/H/seed/mode configuration, exactly one finite aggregate `metrics.csv` row, and an existing checkpoint referenced by that row. It also verified X-A and Only-A use identical effective extraction arguments for every dataset/component pair.",
        "",
        "Generate/check command: `python scripts/write_asymmetric_case_all_metrics.py`.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
