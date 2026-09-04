#!/usr/bin/env python3
"""Export all-channel cases with maximal Baseline/Asymmetric forecast divergence.

Ranking intentionally uses only the two model forecasts, never ground truth:
mean_t(abs(asymmetric_prediction - baseline_prediction)).  The model's normal
decoder-context input is still passed to forward for API compatibility, but y is
not used for selection or plotting.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_asymmetric_multichannel_cases import COMPONENTS, DATASETS, load_model, run_dir
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component


def select(entries, count, separation):
    """Choose largest divergence cases with distinct forecast origins."""
    result = []
    for divergence, origin, channel in sorted(entries, reverse=True):
        if all(abs(origin - previous_origin) >= separation for _, previous_origin, _ in result):
            result.append((divergence, origin, channel))
        if len(result) == count:
            return result
    raise RuntimeError("candidate heap did not yield enough separated origins")


def plot_case(path, dataset, component, origin, channel, x, a, baseline, asymmetric):
    horizon = len(baseline)
    history_x = np.arange(-len(x), 0)
    future_x = np.arange(horizon)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False, constrained_layout=True)
    axes[0].plot(history_x, x, color="#626262", lw=.9, label="full history X")
    axes[0].axvline(0, color="#999999", lw=.8)
    axes[0].set_title(f"{dataset} H96 validation origin {origin}, channel {channel}")
    axes[0].legend(loc="upper left")
    axes[1].plot(history_x, a, color="#9467bd", lw=.9, label=f"extracted A: {component}")
    axes[1].axhline(0, color="#999999", lw=.7); axes[1].axvline(0, color="#999999", lw=.8)
    axes[1].legend(loc="upper left")
    divergence = float(np.abs(asymmetric - baseline).mean())
    axes[2].plot(future_x, baseline, color="#2878b5", lw=1.25, label="Baseline-full prediction")
    axes[2].plot(future_x, asymmetric, color="#c43c39", lw=1.25, label="Asymmetric X-A prediction")
    axes[2].set_title(f"forecast-curve MAD = {divergence:.4f}; GT not used or displayed")
    axes[2].set_xlabel("forecast step"); axes[2].legend(loc="upper left")
    figure.savefig(path, dpi=150); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research_runs/asymmetric_prediction_divergence_cases"))
    parser.add_argument("--etth1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_trend_discovery"))
    parser.add_argument("--weather-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_weather_h96_scratch"))
    parser.add_argument("--ettm1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_ettm1_h96_scratch"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--origin-separation", type=int, default=96)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.top_k < 1 or args.origin_separation < 1:
        parser.error("--top-k and --origin-separation must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {"ETTh1": args.etth1_root, "Weather": args.weather_root, "ETTm1": args.ettm1_root}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []

    for dataset in DATASETS:
        baseline, experiment_args, _ = load_model(run_dir(roots[dataset], dataset, None))
        baseline.to(device).eval(); dataset_obj, loader = data_provider(experiment_args.dataset_args, "val")
        for component in COMPONENTS:
            candidate, _, _ = load_model(run_dir(roots[dataset], dataset, component))
            candidate.to(device).eval()
            heap = []
            origin_cursor = 0
            with torch.inference_mode():
                for batch in loader:
                    x, y, xm, ym = batch
                    x, y, xm, ym = (value.to(device) for value in (x, y, xm, ym))
                    decoder = baseline._build_decoder_input(y.float())
                    base, _, _ = baseline(x.float(), xm.float(), decoder, ym.float())
                    asymmetric, _, _ = candidate(x.float(), xm.float(), decoder, ym.float())
                    divergence = (asymmetric - base).abs().mean(dim=1).cpu().numpy()
                    for row in range(x.size(0)):
                        origin = origin_cursor + row
                        for channel, value in enumerate(divergence[row]):
                            heapq.heappush(heap, (float(value), int(origin), channel))
                            if len(heap) > 5000:
                                heapq.heappop(heap)
                    origin_cursor += x.size(0)
            selected = select(heap, args.top_k, args.origin_separation)
            folder = args.output / dataset / component; folder.mkdir(parents=True, exist_ok=True)
            rows, arrays = [], defaultdict(list)
            for rank, (divergence, origin, channel) in enumerate(selected, 1):
                x, y, xm, ym = [torch.as_tensor(value).unsqueeze(0).to(device) for value in dataset_obj[origin]]
                with torch.inference_mode():
                    decoder = baseline._build_decoder_input(y.float())
                    base, _, _ = baseline(x.float(), xm.float(), decoder, ym.float())
                    asymmetric, _, _ = candidate(x.float(), xm.float(), decoder, ym.float())
                component_values = extract_trend_component(x.float(), component, period_len=24)
                history = x[0, :, channel].cpu().numpy(); a = component_values[0, :, channel].cpu().numpy()
                base = base[0, :, channel].cpu().numpy(); asymmetric = asymmetric[0, :, channel].cpu().numpy()
                filename = f"case_{rank:02d}_origin_{origin}_channel_{channel}.png"
                plot_case(folder / filename, dataset, component, origin, channel, history, a, base, asymmetric)
                rows.append({"rank": rank, "origin": origin, "channel": channel, "forecast_curve_mad": divergence, "figure": filename})
                for key, value in (("origin", origin), ("channel", channel), ("forecast_curve_mad", divergence), ("history_x", history), ("component_a", a), ("baseline_prediction", base), ("asymmetric_prediction", asymmetric)):
                    arrays[key].append(value)
                manifest.append({"dataset": dataset, "component": component, **rows[-1]})
            with (folder / "selected_cases.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            np.savez_compressed(folder / "selected_cases.npz", **{key: np.asarray(value) for key, value in arrays.items()})
            (folder / "README.md").write_text(
                "# Prediction-divergence cases\n\n"
                "Ranking metric: mean absolute difference between Baseline-full and Asymmetric X-A forecasts over 96 steps. "
                "Ground truth is neither used for ranking nor shown in figures. Each plot shows X, extracted A, and the two forecasts.\n"
            )
            del candidate; torch.cuda.empty_cache()
        del baseline; torch.cuda.empty_cache()
    with (args.output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys()); writer.writeheader(); writer.writerows(manifest)
    (args.output / "README.md").write_text(json.dumps({"selection": "top forecast-curve MAD, no GT", "top_k": args.top_k, "origin_separation": args.origin_separation, "datasets": DATASETS, "components": COMPONENTS}, indent=2) + "\n")


if __name__ == "__main__":
    main()
