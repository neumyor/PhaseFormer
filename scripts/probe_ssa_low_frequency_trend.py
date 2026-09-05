#!/usr/bin/env python3
"""Visual validation-only comparison of low-frequency SSA and slow causal EMA."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_trend_filter_smoothing import CHANNEL, HORIZON, LOOKBACK, ORIGIN_FRACTIONS, SPECS, scaled_validation_values
from src.models.asymmetric_trend_components import extract_trend_component


# These are the currently accepted slow EMA parameters, frozen before this
# diagnostic.  SSA uses one dataset-independent setting in sample steps.
SLOW_EMA = {"ETTh1": 0.006, "Weather": 0.024, "ETTm1": 0.006}
SSA = {"window": 144, "rank": 2, "candidate_rank": 12, "min_period": 144}


def _level(component: np.ndarray, endpoint: float) -> np.ndarray:
    """Convert the residual-branch's endpoint-anchored A back to a level curve."""
    return component + endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cases: list[tuple[str, str, int]] = []

    for spec in SPECS:
        validation, count = scaled_validation_values(spec)
        origins = tuple(round(fraction * (count - 1)) for fraction in ORIGIN_FRACTIONS)
        for origin in origins:
            history = validation[origin : origin + LOOKBACK, CHANNEL]
            ground_truth = validation[origin + LOOKBACK : origin + LOOKBACK + HORIZON, CHANNEL]
            x = torch.as_tensor(history, dtype=torch.float32).view(1, LOOKBACK, 1)
            with torch.no_grad():
                ssa_a = extract_trend_component(x, "ssa_low_frequency", **{
                    "ssa_window": SSA["window"], "ssa_rank": SSA["rank"],
                    "ssa_candidate_rank": SSA["candidate_rank"], "ssa_min_period": SSA["min_period"],
                })[0, :, 0].cpu().numpy()
                ema_a = extract_trend_component(
                    x, "causal_ema", causal_ema_alpha=SLOW_EMA[spec.name]
                )[0, :, 0].cpu().numpy()
            time = np.arange(LOOKBACK)
            future_time = np.arange(LOOKBACK, LOOKBACK + HORIZON)
            fig, axis = plt.subplots(figsize=(15, 5.8), constrained_layout=True)
            axis.plot(time, history, color="#9aa0a6", linewidth=1.0, alpha=0.72, label="History X (scaled)")
            axis.plot(time, _level(ssa_a, history[-1]), color="#1f77b4", linewidth=2.1,
                      label="Low-frequency SSA trend level")
            axis.plot(time, _level(ema_a, history[-1]), color="#ff7f0e", linewidth=2.0,
                      label=f"Slow causal EMA level (α={SLOW_EMA[spec.name]:g})")
            axis.plot(future_time, ground_truth, color="#2ca02c", linewidth=2.0, label="GT (future, not used by extractors)")
            axis.axvline(LOOKBACK - 0.5, color="black", linewidth=0.9, alpha=0.55)
            axis.set_title(f"{spec.name} validation | origin={origin} | channel 0 | L=720, H=96")
            axis.set_xlabel("Step (history 0–719; ground truth 720–815)")
            axis.set_ylabel("Training-standardized value")
            axis.legend(loc="upper left", frameon=True)
            name = f"{spec.name}_validation_origin{origin}_channel0.png"
            fig.savefig(figures / name, dpi=180)
            plt.close(fig)
            setting = f"{spec.name}_validation_origin{origin}_channel0"
            cases.append((setting, spec.name, origin))
            for component, values in (("ssa_low_frequency", ssa_a), ("causal_ema_slow", ema_a)):
                rows.append({
                    "setting": setting, "dataset": spec.name, "origin": origin, "channel": CHANNEL,
                    "component": component, "mean_abs_component": float(np.abs(values).mean()),
                    "mean_abs_second_difference": float(np.abs(np.diff(values, n=2)).mean()),
                    "endpoint": float(values[-1]),
                })

    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "sample_errors.csv").write_text("setting\n", encoding="utf-8")
    np.savez(args.output / "selected_cases.npz", setting=np.asarray([x[0] for x in cases]),
             dataset=np.asarray([x[1] for x in cases]), origin=np.asarray([x[2] for x in cases]),
             channel=np.zeros(len(cases), dtype=np.int64))
    (args.output / "run.yaml").write_text(
        "experiment_id: ssa_low_frequency_trend_visual_probe\n"
        "split: validation_only\n"
        "datasets: [ETTh1, Weather, ETTm1]\nlookback: 720\nhorizon: 96\nchannel: 0\n"
        "origins: [fixed_validation_fractions_0.3757, 0.75]\n"
        "ssa: {window: 144, retained_rank: 2, candidate_rank: 12, min_period_steps: 144}\n"
        "slow_causal_ema: {ETTh1: 0.006, Weather: 0.024, ETTm1: 0.006}\n"
        "forecast_model_training: false\ntest_accessed: false\n", encoding="utf-8")
    report = [
        "# Low-frequency SSA trend visual probe", "",
        "Validation-only extraction diagnostic. No forecasting model was trained and no test data was read.", "",
        "SSA embeds each 720-step history into a 144×577 trajectory matrix, computes an SVD, reconstructs the first 12 elementary components by Hankel diagonal averaging, scores each by Fourier energy at periods ≥144 steps, and sums the two highest-scoring components. The plotted SSA and EMA curves are converted from the model-facing endpoint-anchored component A back to levels by adding X[719].", "",
        "The green curve is the subsequent 96-step validation ground truth; it is shown only for contextual comparison and is not used in either extraction method or parameter selection.", "", "## Figures", "",
    ]
    for setting, dataset, origin in cases:
        report.append(f"- `{setting}`: [figure](figures/{dataset}_validation_origin{origin}_channel0.png)")
    report_path = args.output / "objective_error_analysis.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")


if __name__ == "__main__":
    main()
