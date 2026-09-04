#!/usr/bin/env python3
"""Compare causal smooth trends with the frozen GPU trend-filter extractor."""

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

from probe_trend_filter_smoothing import ORIGIN_FRACTIONS, SPECS, scaled_validation_values
from src.models.asymmetric_trend_components import extract_trend_component


LOOKBACK = 720
CHANNEL = 0
COMPONENTS = ("trend_filter", "causal_ema", "holt_local_linear")
COLORS = {"trend_filter": "#1f77b4", "causal_ema": "#ff7f0e", "holt_local_linear": "#9467bd"}
LABELS = {"trend_filter": "Trend filter A6", "causal_ema": "Spectrally constrained causal EMA", "holt_local_linear": "Spectrally constrained Holt"}
# Frozen before plotting.  ETTh1/ETTm1 use their input-spectrum leakage gate;
# Weather has no narrow short-period peak, so uses the conservative hourly value.
SMOOTHING = {
    "ETTh1": {"alpha": 0.024, "beta": 0.006},
    "Weather": {"alpha": 0.024, "beta": 0.006},
    "ETTm1": {"alpha": 0.006, "beta": 0.0015},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected: list[tuple[str, int]] = []

    for spec in SPECS:
        validation, count = scaled_validation_values(spec)
        origins = tuple(round(fraction * (count - 1)) for fraction in ORIGIN_FRACTIONS)
        for origin in origins:
            history = validation[origin:origin + LOOKBACK, CHANNEL]
            input_tensor = torch.as_tensor(history, dtype=torch.float32).view(1, LOOKBACK, 1)
            extracted: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for component in COMPONENTS:
                    params = SMOOTHING[spec.name]
                    kwargs = {
                        "trend_filter_iterations": 256,
                        "trend_filter_sample_interval_hours": spec.sample_interval_hours,
                        "causal_ema_alpha": params["alpha"],
                        "holt_level_alpha": params["alpha"],
                        "holt_trend_beta": params["beta"],
                    }
                    extracted[component] = extract_trend_component(input_tensor, component, **kwargs)[0, :, 0].numpy()
            time = np.arange(LOOKBACK)
            figure, axis = plt.subplots(figsize=(15, 5.6), constrained_layout=True)
            axis.plot(time, history, color="#9aa0a6", linewidth=1.0, alpha=0.70, label="Original input X (scaled)")
            for component in COMPONENTS:
                curve = extracted[component]
                axis.plot(time, curve, color=COLORS[component], linewidth=2.0, label=LABELS[component])
                rows.append({
                    "setting": f"{spec.name}_validation_origin{origin}_channel0",
                    "dataset": spec.name, "origin": origin, "channel": CHANNEL,
                    "component": component, "mean_abs_component": float(np.mean(np.abs(curve))),
                    "mean_abs_second_difference": float(np.mean(np.abs(np.diff(curve, n=2)))),
                    "endpoint": float(curve[-1]),
                })
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
            axis.set_title(f"{spec.name} validation | origin={origin} | channel 0 | L=720")
            axis.set_xlabel("History step (0 = oldest, 719 = latest)")
            axis.set_ylabel("Training-standardized value / endpoint-anchored component")
            axis.legend(loc="upper left", frameon=True)
            name = f"{spec.name}_validation_origin{origin}_channel0.png"
            figure.savefig(figures / name, dpi=180)
            plt.close(figure)
            selected.append((f"{spec.name}_validation_origin{origin}_channel0", origin))

    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "sample_errors.csv").write_text("setting\n", encoding="utf-8")
    np.savez(args.output / "selected_cases.npz", setting=np.asarray([item[0] for item in selected]), origin=np.asarray([item[1] for item in selected]), channel=np.zeros(len(selected), dtype=np.int64))
    (args.output / "run.yaml").write_text(
        "experiment_id: causal_trend_component_spectral_probe\nsplit: validation_only\ndatasets: [ETTh1, Weather, ETTm1]\nlookback: 720\nhorizon: 96\nchannel: 0\ncomponents: [trend_filter, causal_ema, holt_local_linear]\nparameters: {ETTh1: {alpha: 0.024, beta: 0.006}, Weather: {alpha: 0.024, beta: 0.006}, ETTm1: {alpha: 0.006, beta: 0.0015}}\nforecast_model_training: false\ntest_accessed: false\n",
        encoding="utf-8",
    )
    report = ["# Spectrally constrained causal trend-component visual probe", "", "Validation-only extraction comparison; no forecast model was trained and no test data was read.", "", "All curves use the exact extractor implementation and are endpoint anchored. Trend filter uses the frozen 256-step GPU-compatible Chambolle--Pock approximation; causal components use no right-boundary padding. Causal local-linear is excluded because it failed the predeclared periodic-leakage gate on ETTm1.", "", "ETTh1 uses EMA/Holt alpha=0.024 and beta=0.006; ETTm1 uses alpha=0.006 and beta=0.0015. Weather has no narrow short-period peak, so uses the conservative hourly alpha=0.024 and beta=0.006 rather than a prediction-selected value.", "", "## Figures", ""]
    for setting, origin in selected:
        dataset = setting.split("_validation_")[0]
        name = f"{dataset}_validation_origin{origin}_channel0.png"
        report.append(f"- `{setting}`: [figure](figures/{name})")
    report_path = args.output / "objective_error_analysis.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")


if __name__ == "__main__":
    main()
