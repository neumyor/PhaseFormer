#!/usr/bin/env python3
"""Audit the two Gaussian-derived branch components on ETTh1 validation.

``smooth_multiscale`` is retained under its implementation name deliberately:
it is G_24(X)-G_72(X), a difference of smoothers (mid-frequency residual),
not a globally smoothed trend.  Both routes leave PhaseFormer on full X and
only change the input of the NLinear weak-residual branch.
"""

from __future__ import annotations

import csv
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

from scripts.analyze_asymmetric_multichannel_cases import run_dir
from scripts.export_asymmetric_joint_route_cases import BASELINE_ROOTS, candidate_path, predict_all
from scripts.export_asymmetric_prediction_divergence_cases import load_model, trend_kwargs
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component


DATASET = "ETTh1"
COMPONENTS = ("smooth_local", "smooth_multiscale")
TOP_PER_DIRECTION = 8
SEPARATION = 96
OUT = ROOT / "research_runs/etth1_smooth_route_role_cases"


def choose(order: np.ndarray, already: list[int]) -> list[int]:
    selected: list[int] = []
    for raw in order:
        origin = int(raw)
        if all(abs(origin - other) >= SEPARATION for other in already + selected):
            selected.append(origin)
        if len(selected) == TOP_PER_DIRECTION:
            return selected
    raise RuntimeError("insufficient 96-step-separated validation origins")


def descriptors(x: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """Measure properties relevant to the two Gaussian routes, not outcomes."""
    spectrum = np.abs(np.fft.rfft(a - a.mean())) ** 2
    # ETTh1 is hourly: bin 30 of a 720-step history is the 24-hour period.
    cycle_share = float(spectrum[30] / spectrum[1:].sum().clip(1e-12))
    interior_curvature = float(np.abs(np.diff(a[:-72], n=2)).mean())
    tail_curvature = float(np.abs(np.diff(a[-72:], n=2)).mean())
    return {
        "history_std": float(x.std()),
        "recent_level_shift_96": float(x[-96:].mean() - x[-192:-96].mean()),
        "history_cycle24_share": float((np.abs(np.fft.rfft(x - x.mean())) ** 2)[30] /
                                     (np.abs(np.fft.rfft(x - x.mean())) ** 2)[1:].sum().clip(1e-12)),
        "component_range": float(a.max() - a.min()),
        "component_cycle24_share": cycle_share,
        "component_interior_curvature": interior_curvature,
        "component_tail_curvature": tail_curvature,
        "tail_to_interior_curvature": tail_curvature / max(interior_curvature, 1e-12),
        "component_last24_change": float(a[-1] - a[-25]),
    }


def plot(path: Path, component: str, direction: str, origin: int, x: np.ndarray,
         a: np.ndarray, y: np.ndarray, baseline: np.ndarray, minus: np.ndarray,
         only: np.ndarray) -> dict[str, float]:
    errors = {
        "baseline": float(np.abs(baseline - y).mean()),
        "x_minus_a": float(np.abs(minus - y).mean()),
        "only_a": float(np.abs(only - y).mean()),
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), constrained_layout=True)
    ht, ft = np.arange(-x.size, 0), np.arange(y.size)
    axes[0].plot(ht, x, color="#555", lw=.9, label="full history X")
    axes[0].plot(ht, a, color="#9467bd", lw=1.1, label=f"extracted A ({component})")
    axes[0].axvline(0, color="#999", lw=.8); axes[0].legend(loc="upper left")
    axes[0].set_title(f"ETTh1 | {component} | {direction} | validation origin {origin} | channel 0 | L=720, H=96")
    axes[1].plot(ft, y, color="#1f1f1f", lw=1.35, label="GT")
    axes[1].plot(ft, baseline, color="#2878b5", lw=1.15, label="Baseline-full")
    axes[1].plot(ft, minus, color="#c43c39", lw=1.15, label="X-A")
    axes[1].plot(ft, only, color="#2a9d62", lw=1.15, label="Only-A")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_title("channel-0 MAE: " + "; ".join(
        f"{name}={value:.4f}" for name, value in errors.items()))
    axes[1].set_xlabel("forecast step")
    fig.savefig(path, dpi=160); plt.close(fig)
    return errors


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUT}")
    figures = OUT / "figures"; figures.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, args, _, _ = load_model(run_dir(BASELINE_ROOTS[DATASET], DATASET, None))
    baseline.to(device).eval()
    _, loader = data_provider(args.dataset_args, "val")
    histories, truths, baseline_preds, _ = predict_all(baseline, loader, device)
    rows: list[dict[str, object]] = []
    npz: dict[str, list] = {key: [] for key in (
        "setting", "component", "direction", "origin", "history_x", "component_a",
        "ground_truth", "baseline_prediction", "x_minus_a_prediction", "only_a_prediction")}
    report = [
        "# ETTh1 Gaussian-derived component route-role audit", "",
        "Validation-only, channel 0, L=720→H=96, seed=2021. Each model has the same PhaseFormer branch receiving complete X. Only NLinear's weak-residual input differs: X-A or Only-A.", "",
        "Selection uses channel-0 `Only-A MAE − X-A MAE` and GT only for this error-role audit. For each component, eight largest positive values are `x_minus_a_better`; eight most negative values are `only_a_better`. Origins are separated by at least 96 steps within a component.", "",
        "`smooth_local` is endpoint-anchored replicate-padded Gaussian smoothing, σ=24. `smooth_multiscale` is endpoint-anchored `G_24(X)−G_72(X)`: it is a difference-of-smoothers / middle-band signal, **not** a global smooth trend.", "",
    ]
    for component in COMPONENTS:
        minus, _, _, hp = load_model(candidate_path(DATASET, component, "minus_component"))
        only, _, _, _ = load_model(candidate_path(DATASET, component, "component_only"))
        minus.to(device).eval(); only.to(device).eval()
        _, _, minus_preds, _ = predict_all(minus, loader, device)
        _, _, only_preds, _ = predict_all(only, loader, device)
        x_mae = np.abs(minus_preds[:, :, 0] - truths[:, :, 0]).mean(axis=1)
        only_mae = np.abs(only_preds[:, :, 0] - truths[:, :, 0]).mean(axis=1)
        difference = only_mae - x_mae
        route_mad = np.abs(only_preds[:, :, 0] - minus_preds[:, :, 0]).mean(axis=1)
        picks = (("x_minus_a_better", choose(np.argsort(difference)[::-1], [])),
                 ("only_a_better", choose(np.argsort(difference), [])))
        report.extend([f"## `{component}`", ""])
        for direction, origins in picks:
            report.extend([f"### {direction}", "", "| rank | origin | Baseline MAE | X-A MAE | Only-A MAE | Only-A−X-A | curve MAD | 24-step A energy | tail/interior curvature | figure |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
            for rank, origin in enumerate(origins, 1):
                x = torch.from_numpy(histories[origin:origin + 1]).to(device).float()
                a = extract_trend_component(x, component, **trend_kwargs(hp))[0, :, 0].cpu().numpy()
                metrics = plot(figures / (name := f"ETTh1__{component}__{direction}__{rank:02d}_origin{origin}_channel0.png"), component, direction, origin, histories[origin, :, 0], a, truths[origin, :, 0], baseline_preds[origin, :, 0], minus_preds[origin, :, 0], only_preds[origin, :, 0])
                row = {"setting": "ETTh1_h96_seed2021_validation", "dataset": DATASET, "component": component, "direction": direction, "rank": rank, "origin": origin, "channel": 0, "only_minus_x_mae": float(difference[origin]), "route_curve_mad": float(route_mad[origin]), **{f"{key}_mae": value for key, value in metrics.items()}, **descriptors(histories[origin, :, 0], a)}
                rows.append(row)
                report.append(f"| {rank} | {origin} | {metrics['baseline']:.4f} | {metrics['x_minus_a']:.4f} | {metrics['only_a']:.4f} | {difference[origin]:+.4f} | {route_mad[origin]:.4f} | {row['component_cycle24_share']:.3f} | {row['tail_to_interior_curvature']:.2f} | [figure](figures/{name}) |")
                for key, value in (("setting", row["setting"]), ("component", component), ("direction", direction), ("origin", origin), ("history_x", histories[origin, :, 0]), ("component_a", a), ("ground_truth", truths[origin, :, 0]), ("baseline_prediction", baseline_preds[origin, :, 0]), ("x_minus_a_prediction", minus_preds[origin, :, 0]), ("only_a_prediction", only_preds[origin, :, 0])):
                    npz[key].append(value)
            report.append("")
        del minus, only
        if device.type == "cuda": torch.cuda.empty_cache()
    with (OUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    # ``results.csv`` is one row per route-role group, derived strictly from selected cases.
    grouped = []
    for component in COMPONENTS:
        for direction in ("x_minus_a_better", "only_a_better"):
            group = [row for row in rows if row["component"] == component and row["direction"] == direction]
            grouped.append({"setting": "ETTh1_h96_seed2021_validation", "dataset": DATASET, "component": component, "direction": direction, "n": len(group), **{f"mean_{key}": float(np.mean([float(row[key]) for row in group])) for key in ("only_minus_x_mae", "route_curve_mad", "component_cycle24_share", "tail_to_interior_curvature", "component_last24_change")}})
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped[0].keys()); writer.writeheader(); writer.writerows(grouped)
    np.savez_compressed(OUT / "selected_cases.npz", **{key: np.asarray(value) for key, value in npz.items()})
    (OUT / "run.yaml").write_text("experiment_id: etth1_smooth_route_role_cases\nsplit: validation_only\nsettings: [ETTh1_h96_seed2021_validation]\ncomponents: [smooth_local, smooth_multiscale]\nselection: eight greatest X-A advantages and eight greatest Only-A advantages by channel-0 MAE; 96-origin separation within component\ntest_accessed: false\n", encoding="utf-8")
    report_path = OUT / "objective_error_analysis.md"; report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with zipfile.ZipFile(OUT / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")): archive.write(figure, f"figures/{figure.name}")
    print(OUT)


if __name__ == "__main__":
    main()
