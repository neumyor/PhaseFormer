#!/usr/bin/env python3
"""Export channel-0 cases with maximal Baseline/Asymmetric forecast divergence.

Ranking intentionally uses only the two model forecasts, never ground truth:
mean_t(abs(asymmetric_prediction - baseline_prediction)). Ground truth is shown
in figures for interpretation but never participates in ranking.
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

from scripts.analyze_asymmetric_multichannel_cases import DATASETS, run_dir
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args
import scripts.search_phaseformer as search_phaseformer


DEFAULT_COMPONENTS = (
    "cycle_levels", "recent_linear", "global_linear", "smooth_local", "smooth_multiscale",
)


def select(entries, count, separation):
    """Choose largest divergence cases with distinct forecast origins."""
    result = []
    for divergence, origin, channel in sorted(entries, reverse=True):
        if all(abs(origin - previous_origin) >= separation for _, previous_origin, _ in result):
            result.append((divergence, origin, channel))
        if len(result) == count:
            return result
    raise RuntimeError("candidate heap did not yield enough separated origins")


def load_model(path: Path):
    """Load either a normal local run or a delivered checkpoint directory."""
    config = json.loads((path / "config.json").read_text())
    hp = dict(config["hyperparams"])
    args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hp,
        batch_size=config.get("batch_size"),
    )
    args.dataset_args.percent = config.get("percent", 100)
    args.dataset_args.num_workers = 0
    train_set, _ = search_phaseformer.data_provider(args.dataset_args, "train")
    if hasattr(train_set, "data_stamp"):
        hp["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(PhaseFormerPresetConfig(
        args, config["lookback"], config["horizon"], hp
    ))
    checkpoint = path / "attempts/001/checkpoints/best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing best checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model, args, checkpoint, hp


def trend_kwargs(hyperparams: dict) -> dict:
    """Mirror the candidate model's frozen extraction hyperparameters."""
    return {
        "period_len": hyperparams.get("period_len", 24),
        "recent_window": hyperparams.get("weak_residual_trend_recent_window", 96),
        "local_sigma": hyperparams.get("weak_residual_trend_local_sigma", 24.0),
        "long_sigma": hyperparams.get("weak_residual_trend_long_sigma", 72.0),
        "trend_filter_kappa": hyperparams.get("weak_residual_trend_filter_kappa", 100.0),
        "trend_filter_sample_interval_hours": hyperparams.get(
            "weak_residual_trend_filter_sample_interval_hours", 1.0
        ),
        "trend_filter_iterations": hyperparams.get("weak_residual_trend_filter_iterations", 128),
        "causal_ema_alpha": hyperparams.get("weak_residual_causal_ema_alpha", 0.08),
        "causal_local_linear_window": hyperparams.get(
            "weak_residual_causal_local_linear_window", 72
        ),
        "causal_local_linear_sigma": hyperparams.get(
            "weak_residual_causal_local_linear_sigma", 24.0
        ),
        "holt_level_alpha": hyperparams.get("weak_residual_holt_level_alpha", 0.15),
        "holt_trend_beta": hyperparams.get("weak_residual_holt_trend_beta", 0.03),
    }


def find_candidate_run(root, dataset, component, input_mode, layout):
    if layout == "delivery":
        path = root / "checkpoints" / f"{dataset}_h96_seed2021" / f"{component}-{input_mode}"
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"delivered candidate is missing: {path}")
        config = json.loads(config_path.read_text())
        hp = config["hyperparams"]
        if (config["dataset"] != dataset or config["horizon"] != 96
                or hp.get("weak_residual_asymmetric_component") != component
                or hp.get("weak_residual_asymmetric_input_mode") != input_mode):
            raise RuntimeError(f"delivered candidate config mismatch: {path}")
        return path
    matches = []
    for config_path in (root / "runs").glob("*/config.json"):
        config = json.loads(config_path.read_text())
        hyperparams = config["hyperparams"]
        if config["dataset"] != dataset or config["horizon"] != 96:
            continue
        if hyperparams.get("weak_residual_asymmetric_component") != component:
            continue
        mode = hyperparams.get("weak_residual_asymmetric_input_mode", "minus_component")
        if mode == input_mode:
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"need one candidate run: {dataset=} {component=} {input_mode=}; found {matches}")
    return matches[0]


def plot_case(path, dataset, component, input_mode, origin, channel, x, a, truth, baseline, asymmetric):
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
    axes[2].plot(future_x, truth, color="#1f1f1f", lw=1.3, label="ground truth (not used for selection)")
    axes[2].plot(future_x, baseline, color="#2878b5", lw=1.25, label="Baseline-full prediction")
    branch_label = "Asymmetric X-A" if input_mode == "minus_component" else "Only-A"
    axes[2].plot(future_x, asymmetric, color="#c43c39", lw=1.25, label=f"{branch_label} prediction")
    axes[2].set_title(f"forecast-curve MAD = {divergence:.4f}; GT not used for selection")
    axes[2].set_xlabel("forecast step"); axes[2].legend(loc="upper left")
    figure.savefig(path, dpi=150); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research_runs/asymmetric_prediction_divergence_cases"))
    parser.add_argument("--etth1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_trend_discovery"))
    parser.add_argument("--weather-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_weather_h96_scratch"))
    parser.add_argument("--ettm1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_ettm1_h96_scratch"))
    parser.add_argument("--candidate-root", type=Path, default=None,
                        help="Root holding candidates; omit to reuse the per-dataset baseline roots")
    parser.add_argument("--candidate-layout", choices=("runs", "delivery"), default="runs",
                        help="'delivery' reads checkpoints/<setting>/<component>-<route> from an imported bundle")
    parser.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS),
                        help="Components to export; must match the candidate checkpoints")
    parser.add_argument("--input-mode", choices=("minus_component", "component_only"), default="minus_component")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--origin-separation", type=int, default=96)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.top_k < 1 or args.origin_separation < 1:
        parser.error("--top-k and --origin-separation must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {"ETTh1": args.etth1_root, "Weather": args.weather_root, "ETTm1": args.ettm1_root}
    candidate_roots = {dataset: args.candidate_root or roots[dataset] for dataset in DATASETS}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []

    for dataset in DATASETS:
        baseline, experiment_args, _, _ = load_model(run_dir(roots[dataset], dataset, None))
        baseline.to(device).eval(); dataset_obj, loader = data_provider(experiment_args.dataset_args, "val")
        for component in args.components:
            candidate_path = find_candidate_run(
                candidate_roots[dataset], dataset, component, args.input_mode, args.candidate_layout
            )
            candidate, _, _, candidate_hp = load_model(candidate_path)
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
                    if args.channel < 0 or args.channel >= divergence.shape[1]:
                        raise ValueError(f"channel {args.channel} is unavailable for {dataset}")
                    for row in range(x.size(0)):
                        origin = origin_cursor + row
                        heapq.heappush(heap, (float(divergence[row, args.channel]), int(origin), args.channel))
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
                component_values = extract_trend_component(
                    x.float(), component, **trend_kwargs(candidate_hp)
                )
                history = x[0, :, channel].cpu().numpy(); a = component_values[0, :, channel].cpu().numpy()
                truth = y[0, -len(base[0]):, channel].cpu().numpy()
                base = base[0, :, channel].cpu().numpy(); asymmetric = asymmetric[0, :, channel].cpu().numpy()
                filename = f"case_{rank:02d}_origin_{origin}_channel_{channel}.png"
                plot_case(folder / filename, dataset, component, args.input_mode, origin, channel, history, a, truth, base, asymmetric)
                rows.append({"rank": rank, "origin": origin, "channel": channel, "forecast_curve_mad": divergence, "figure": filename})
                for key, value in (("origin", origin), ("channel", channel), ("forecast_curve_mad", divergence), ("history_x", history), ("component_a", a), ("ground_truth", truth), ("baseline_prediction", base), ("asymmetric_prediction", asymmetric)):
                    arrays[key].append(value)
                manifest.append({"dataset": dataset, "component": component, **rows[-1]})
            with (folder / "selected_cases.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            np.savez_compressed(folder / "selected_cases.npz", **{key: np.asarray(value) for key, value in arrays.items()})
            (folder / "README.md").write_text(
                "# Prediction-divergence cases\n\n"
                f"Candidate branch input mode: `{args.input_mode}`. Ranking metric: mean absolute difference between Baseline-full and candidate forecasts over 96 steps. "
                "Ground truth is not used for ranking, but is shown in figures for interpretation. Each plot shows X, extracted A, ground truth, and the two forecasts.\n"
            )
            del candidate; torch.cuda.empty_cache()
        del baseline; torch.cuda.empty_cache()
    with (args.output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys()); writer.writeheader(); writer.writerows(manifest)
    (args.output / "README.md").write_text(json.dumps({"selection": "top forecast-curve MAD; GT excluded from ranking and included in plots", "input_mode": args.input_mode, "channel": args.channel, "top_k": args.top_k, "origin_separation": args.origin_separation, "datasets": DATASETS, "components": args.components, "candidate_layout": args.candidate_layout}, indent=2) + "\n")


if __name__ == "__main__":
    main()
