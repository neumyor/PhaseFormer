#!/usr/bin/env python3
"""Export one joint X-A/Only-A validation case per dataset and component.

Selection uses neither ground truth nor per-model error: it maximizes the
96-step forecast-curve MAD between the X-A and Only-A predictions.  Within a
dataset the selected origins for different components must be at least one
horizon apart, so the final component catalogue does not repeat one sample.
The resulting two-row plot shows X and A, then GT and all three forecasts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_asymmetric_multichannel_cases import DATASETS, run_dir
from scripts.export_asymmetric_prediction_divergence_cases import (
    find_candidate_run,
    load_model,
    trend_kwargs,
)
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component


COMPONENTS = (
    "cycle_levels", "recent_linear", "global_linear", "smooth_local",
    "smooth_multiscale", "causal_ema", "holt_local_linear",
)
DELIVERED = {"causal_ema", "holt_local_linear"}
BASELINE_ROOTS = {
    "ETTh1": ROOT / "research_runs/weak_residual_asymmetric_trend_discovery",
    "Weather": ROOT / "research_runs/weak_residual_asymmetric_weather_h96_scratch",
    "ETTm1": ROOT / "research_runs/weak_residual_asymmetric_ettm1_h96_scratch",
}
X_MINUS_A_ROOTS = BASELINE_ROOTS
ONLY_A_ROOT = ROOT / "research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch"
DELIVERY_ROOT = ROOT / "research_runs/weak_residual_trend_2comp_3ds_experiment_scratch"


def candidate_path(dataset: str, component: str, mode: str) -> Path:
    if component in DELIVERED:
        return find_candidate_run(DELIVERY_ROOT, dataset, component, mode, "delivery")
    root = X_MINUS_A_ROOTS[dataset] if mode == "minus_component" else ONLY_A_ROOT
    return find_candidate_run(root, dataset, component, mode, "runs")


def predict_all(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full validation X, truth, prediction and time marks in loader order."""
    xs, truths, preds, xmarks = [], [], [], []
    with torch.inference_mode():
        for x, y, xm, ym in loader:
            x, y, xm, ym = (v.to(device) for v in (x, y, xm, ym))
            decoder = model._build_decoder_input(y.float())
            pred, _, _ = model(x.float(), xm.float(), decoder, ym.float())
            xs.append(x.cpu().numpy())
            truths.append(y[:, -pred.size(1):].float().cpu().numpy())
            preds.append(pred.float().cpu().numpy())
            xmarks.append(xm.cpu().numpy())
    return tuple(np.concatenate(parts, axis=0) for parts in (xs, truths, preds, xmarks))


def plot_case(path: Path, dataset: str, component: str, origin: int, channel: int,
              history: np.ndarray, extracted: np.ndarray, truth: np.ndarray,
              baseline: np.ndarray, minus_a: np.ndarray, only_a: np.ndarray) -> dict:
    horizon = truth.size
    mae = {
        "baseline": float(np.abs(baseline - truth).mean()),
        "X-A": float(np.abs(minus_a - truth).mean()),
        "Only-A": float(np.abs(only_a - truth).mean()),
    }
    mse = {
        "baseline": float(((baseline - truth) ** 2).mean()),
        "X-A": float(((minus_a - truth) ** 2).mean()),
        "Only-A": float(((only_a - truth) ** 2).mean()),
    }
    best = min(mse, key=mse.get)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), constrained_layout=True)
    hist_t = np.arange(-history.size, 0)
    future_t = np.arange(horizon)
    axes[0].plot(hist_t, history, color="#555555", lw=.9, label="full history X")
    axes[0].plot(hist_t, extracted, color="#9467bd", lw=.9, label=f"extracted A ({component})")
    axes[0].axvline(0, color="#999999", lw=.8)
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"{dataset} | validation origin {origin} | channel {channel} | "
        f"lookback {history.size}, horizon {horizon}"
    )
    axes[0].set_xlabel("time step (forecast begins at 0)")
    axes[1].plot(future_t, truth, color="#1f1f1f", lw=1.35, label="GT")
    axes[1].plot(future_t, baseline, color="#2878b5", lw=1.15, label="Baseline-full")
    axes[1].plot(future_t, minus_a, color="#c43c39", lw=1.15, label="X-A")
    axes[1].plot(future_t, only_a, color="#2a9d62", lw=1.15, label="Only-A")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_xlabel("forecast step")
    axes[1].set_title(
        "MAE / MSE: "
        f"Baseline {mae['baseline']:.4f} / {mse['baseline']:.4f}; "
        f"X-A {mae['X-A']:.4f} / {mse['X-A']:.4f}; "
        f"Only-A {mae['Only-A']:.4f} / {mse['Only-A']:.4f}. "
        f"Best by MSE: {best}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {**{f"{key}_mae": value for key, value in mae.items()},
            **{f"{key}_mse": value for key, value in mse.items()}, "best_by_mse": best}


def main() -> None:
    output = ROOT / "research_runs/asymmetric_prediction_divergence_cases"
    if output.exists():
        existing = [p for p in output.iterdir() if p.name not in {"mpl"}]
        if existing:
            raise FileExistsError(
                f"refuse to mix outputs in a non-empty directory: {output}; {existing}"
            )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = []
    for dataset in DATASETS:
        baseline_path = run_dir(BASELINE_ROOTS[dataset], dataset, None)
        baseline, args, _, _ = load_model(baseline_path)
        baseline.to(device).eval()
        dataset_obj, loader = data_provider(args.dataset_args, "val")
        histories, truths, baseline_preds, _ = predict_all(baseline, loader, device)
        candidates = []
        for component in COMPONENTS:
            minus, _, _, minus_hp = load_model(candidate_path(dataset, component, "minus_component"))
            only, _, _, _ = load_model(candidate_path(dataset, component, "component_only"))
            minus.to(device).eval(); only.to(device).eval()
            _, _, minus_preds, _ = predict_all(minus, loader, device)
            _, _, only_preds, _ = predict_all(only, loader, device)
            # Channel 0 only. GT is deliberately absent from this score.
            gap_minus = np.abs(minus_preds[:, :, 0] - baseline_preds[:, :, 0]).mean(axis=1)
            gap_only = np.abs(only_preds[:, :, 0] - baseline_preds[:, :, 0]).mean(axis=1)
            route_gap = np.abs(minus_preds[:, :, 0] - only_preds[:, :, 0]).mean(axis=1)
            candidates.append({
                "component": component, "minus_hp": minus_hp,
                "minus_preds": minus_preds, "only_preds": only_preds,
                "gap_minus": gap_minus, "gap_only": gap_only, "route_gap": route_gap,
            })
            del minus, only
            if device.type == "cuda":
                torch.cuda.empty_cache()

        selected_origins = []
        for candidate in candidates:
            component = candidate["component"]
            order = np.argsort(candidate["route_gap"])[::-1]
            origin = next(
                (int(item) for item in order
                 if all(abs(int(item) - previous) >= 96 for previous in selected_origins)),
                None,
            )
            if origin is None:
                raise RuntimeError(f"cannot select a separated origin: {dataset} {component}")
            selected_origins.append(origin)
            x = torch.from_numpy(histories[origin:origin + 1]).to(device).float()
            component_a = extract_trend_component(x, component, **trend_kwargs(candidate["minus_hp"]))
            folder = output / dataset / component
            folder.mkdir(parents=True)
            figure = folder / f"origin_{origin}_channel_0.png"
            metrics = plot_case(
                figure, dataset, component, origin, 0,
                histories[origin, :, 0], component_a[0, :, 0].cpu().numpy(),
                truths[origin, :, 0], baseline_preds[origin, :, 0],
                minus_preds[origin, :, 0], only_preds[origin, :, 0],
            )
            row = {
                "dataset": dataset, "component": component, "origin": origin, "channel": 0,
                "x_minus_a_vs_only_a_forecast_curve_mad": float(candidate["route_gap"][origin]),
                "baseline_forecast_curve_mad": 0.0,
                "x_minus_a_vs_baseline_forecast_curve_mad": float(candidate["gap_minus"][origin]),
                "only_a_vs_baseline_forecast_curve_mad": float(candidate["gap_only"][origin]),
                "figure": str(figure.relative_to(output)), **metrics,
            }
            manifest.append(row)
            np.savez_compressed(folder / "selected_case.npz", **{
                "origin": np.asarray(origin), "channel": np.asarray(0),
                "history_x": histories[origin, :, 0], "component_a": component_a[0, :, 0].cpu().numpy(),
                "ground_truth": truths[origin, :, 0], "baseline_prediction": baseline_preds[origin, :, 0],
                "x_minus_a_prediction": minus_preds[origin, :, 0], "only_a_prediction": only_preds[origin, :, 0],
            })
            (folder / "README.md").write_text(
                "Selection: maximum X-A/Only-A forecast-curve MAD on channel 0, with selected origins across "
                "components in the same dataset separated by at least 96 steps. "
                "Ground truth is not used to select the sample; it is shown to interpret the two model errors.\n"
            )
        del candidates
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    (output / "README.md").write_text(json.dumps({
        "selection": "maximum X-A/Only-A forecast-curve MAD; one horizon of separation between component origins; GT excluded",
        "display": "two rows: full X plus extracted A; GT plus Baseline, X-A, Only-A forecasts",
        "datasets": DATASETS, "components": COMPONENTS, "channel": 0, "lookback": 720, "horizon": 96,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
