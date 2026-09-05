#!/usr/bin/env python3
"""Select bidirectional X-A/Only-A role cases for global trend and causal EMA."""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT))

from scripts.export_asymmetric_joint_route_cases import BASELINE_ROOTS, candidate_path, predict_all
from scripts.export_asymmetric_prediction_divergence_cases import load_model, trend_kwargs
from scripts.analyze_asymmetric_multichannel_cases import DATASETS, run_dir
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component


COMPONENTS = ("global_linear", "causal_ema")
TOP_PER_DIRECTION = 5
SEPARATION = 96
OUT = ROOT / "research_runs/global_ema_route_role_cases"


def select(order: np.ndarray, difference: np.ndarray, used: list[int]) -> list[int]:
    chosen: list[int] = []
    for origin in order:
        origin = int(origin)
        if all(abs(origin - prior) >= SEPARATION for prior in used + chosen):
            chosen.append(origin)
        if len(chosen) == TOP_PER_DIRECTION:
            return chosen
    raise RuntimeError("not enough separated origins")


def descriptors(history: np.ndarray, component: np.ndarray, dataset: str) -> dict[str, float]:
    time = np.arange(history.size, dtype=np.float64)
    slope = float(np.polyfit(time, history, 1)[0])
    recent_shift = float(history[-96:].mean() - history[-192:-96].mean())
    spectrum = np.abs(np.fft.rfft(history - history.mean())) ** 2
    period = 24 if dataset != "ETTm1" else 96
    bin_index = max(1, round(history.size / period))
    cycle_share = float(spectrum[bin_index] / spectrum[1:].sum().clip(1e-12))
    return {
        "history_std": float(history.std()), "history_global_slope": slope,
        "recent_level_shift_96": recent_shift, "cycle_energy_share": cycle_share,
        "component_range": float(component.max() - component.min()),
        "component_curvature": float(np.abs(np.diff(component, n=2)).mean()),
    }


def plot(path: Path, dataset: str, component_name: str, direction: str, origin: int,
         history: np.ndarray, extracted: np.ndarray, truth: np.ndarray,
         baseline: np.ndarray, x_minus_a: np.ndarray, only_a: np.ndarray) -> dict[str, float]:
    errors = {
        "baseline": np.abs(baseline - truth).mean(), "x_minus_a": np.abs(x_minus_a - truth).mean(),
        "only_a": np.abs(only_a - truth).mean(),
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), constrained_layout=True)
    hist_t, future_t = np.arange(-history.size, 0), np.arange(truth.size)
    axes[0].plot(hist_t, history, color="#555", lw=.9, label="full history X")
    axes[0].plot(hist_t, extracted, color="#9467bd", lw=1.1, label=f"extracted A ({component_name})")
    axes[0].axvline(0, color="#999", lw=.8); axes[0].legend(loc="upper left")
    axes[0].set_title(f"{dataset} | {component_name} | {direction} | validation origin {origin} | channel 0 | L=720, H=96")
    axes[1].plot(future_t, truth, color="#1f1f1f", lw=1.35, label="GT")
    axes[1].plot(future_t, baseline, color="#2878b5", lw=1.15, label="Baseline-full")
    axes[1].plot(future_t, x_minus_a, color="#c43c39", lw=1.15, label="X-A")
    axes[1].plot(future_t, only_a, color="#2a9d62", lw=1.15, label="Only-A")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_title(f"channel-0 MAE: Baseline={errors['baseline']:.4f}; X-A={errors['x_minus_a']:.4f}; Only-A={errors['only_a']:.4f}")
    axes[1].set_xlabel("forecast step")
    fig.savefig(path, dpi=160); plt.close(fig)
    return {key: float(value) for key, value in errors.items()}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {OUT}")
    figures = OUT / "figures"; figures.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    error_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    selected: dict[str, list] = {key: [] for key in ("setting", "component", "direction", "origin", "history_x", "component_a", "ground_truth", "baseline_prediction", "x_minus_a_prediction", "only_a_prediction")}
    report = ["# Global-linear and causal-EMA route-role cases", "", "Validation-only audit. Selection uses the signed channel-0 MAE difference `Only-A − X-A`: the five largest positive values identify states where X-A is better, and the five most negative values identify states where Only-A is better. GT is used for this *error-role* selection, unlike the prediction-divergence gallery.", "", "All selected origins within one dataset/component are at least 96 steps apart. PhaseFormer receives full X in both candidates; only NLinear's input differs.", ""]

    for dataset in DATASETS:
        baseline, args, _, _ = load_model(run_dir(BASELINE_ROOTS[dataset], dataset, None))
        baseline.to(device).eval()
        _, loader = data_provider(args.dataset_args, "val")
        histories, truths, baseline_preds, _ = predict_all(baseline, loader, device)
        for component in COMPONENTS:
            minus, _, _, minus_hp = load_model(candidate_path(dataset, component, "minus_component"))
            only, _, _, _ = load_model(candidate_path(dataset, component, "component_only"))
            minus.to(device).eval(); only.to(device).eval()
            _, _, minus_preds, _ = predict_all(minus, loader, device)
            _, _, only_preds, _ = predict_all(only, loader, device)
            x_error = np.abs(minus_preds[:, :, 0] - truths[:, :, 0]).mean(axis=1)
            only_error = np.abs(only_preds[:, :, 0] - truths[:, :, 0]).mean(axis=1)
            difference = only_error - x_error
            route_gap = np.abs(only_preds[:, :, 0] - minus_preds[:, :, 0]).mean(axis=1)
            picked_x = select(np.argsort(difference)[::-1], difference, [])
            picked_only = select(np.argsort(difference), difference, picked_x)
            report.extend([f"## {dataset} / `{component}`", ""])
            for direction, origins in (("x_minus_a_better", picked_x), ("only_a_better", picked_only)):
                report.append(f"### {direction}"); report.append("")
                report.append("| rank | origin | X-A MAE | Only-A MAE | Only-A − X-A | curve MAD | figure |")
                report.append("|---:|---:|---:|---:|---:|---:|---|")
                for rank, origin in enumerate(origins, 1):
                    x = torch.from_numpy(histories[origin:origin + 1]).to(device).float()
                    extracted = extract_trend_component(x, component, **trend_kwargs(minus_hp))[0, :, 0].cpu().numpy()
                    features = descriptors(histories[origin, :, 0], extracted, dataset)
                    name = f"{dataset}__{component}__{direction}__{rank:02d}_origin{origin}_channel0.png"
                    metrics = plot(figures / name, dataset, component, direction, origin, histories[origin, :, 0], extracted, truths[origin, :, 0], baseline_preds[origin, :, 0], minus_preds[origin, :, 0], only_preds[origin, :, 0])
                    row = {"setting": f"{dataset}_h96_seed2021", "dataset": dataset, "component": component, "direction": direction, "rank": rank, "origin": origin, "channel": 0, "x_minus_a_mae": metrics["x_minus_a"], "only_a_mae": metrics["only_a"], "only_minus_x_mae": float(difference[origin]), "route_curve_mad": float(route_gap[origin]), **features}
                    error_rows.append(row)
                    report.append(f"| {rank} | {origin} | {metrics['x_minus_a']:.4f} | {metrics['only_a']:.4f} | {difference[origin]:+.4f} | {route_gap[origin]:.4f} | [figure](figures/{name}) |")
                    for key, value in (("setting", row["setting"]), ("component", component), ("direction", direction), ("origin", origin), ("history_x", histories[origin, :, 0]), ("component_a", extracted), ("ground_truth", truths[origin, :, 0]), ("baseline_prediction", baseline_preds[origin, :, 0]), ("x_minus_a_prediction", minus_preds[origin, :, 0]), ("only_a_prediction", only_preds[origin, :, 0])):
                        selected[key].append(value)
                report.append("")
            for direction in ("x_minus_a_better", "only_a_better"):
                rows = [r for r in error_rows if r["dataset"] == dataset and r["component"] == component and r["direction"] == direction]
                result_rows.append({"setting": f"{dataset}_h96_seed2021", "dataset": dataset, "component": component, "direction": direction, "n": len(rows), "mean_only_minus_x_mae": float(np.mean([r["only_minus_x_mae"] for r in rows])), "mean_recent_level_shift_96": float(np.mean([r["recent_level_shift_96"] for r in rows])), "mean_cycle_energy_share": float(np.mean([r["cycle_energy_share"] for r in rows])), "mean_component_range": float(np.mean([r["component_range"] for r in rows]))})
            del minus, only
            if device.type == "cuda": torch.cuda.empty_cache()
        del baseline
        if device.type == "cuda": torch.cuda.empty_cache()
    with (OUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_rows[0].keys()); writer.writeheader(); writer.writerows(error_rows)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys()); writer.writeheader(); writer.writerows(result_rows)
    np.savez_compressed(OUT / "selected_cases.npz", **{key: np.asarray(value) for key, value in selected.items()})
    (OUT / "run.yaml").write_text("experiment_id: global_ema_route_role_cases\nsplit: validation_only\nsettings: [ETTh1_h96_seed2021, Weather_h96_seed2021, ETTm1_h96_seed2021]\ncomponents: [global_linear, causal_ema]\nselection: five greatest X-A advantages and five greatest Only-A advantages by channel-0 MAE, separated by 96 origins\ntest_accessed: false\n", encoding="utf-8")
    report_path = OUT / "objective_error_analysis.md"; report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with zipfile.ZipFile(OUT / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")): archive.write(figure, f"figures/{figure.name}")
    print(OUT)


if __name__ == "__main__":
    main()
