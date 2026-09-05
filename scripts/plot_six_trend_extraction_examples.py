#!/usr/bin/env python3
"""Plot six continuous trend candidates on one fixed validation sample per dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_asymmetric_joint_route_cases import candidate_path
from scripts.export_asymmetric_prediction_divergence_cases import load_model, trend_kwargs
from scripts.probe_trend_filter_smoothing import LOOKBACK, SPECS, scaled_validation_values
from src.models.asymmetric_trend_components import extract_trend_component


# Fixed, pre-existing validation origins.  They illustrate a strong hourly
# cycle (ETTh1), non-periodic smooth level changes (Weather), and a 96-step
# minute-level cycle (ETTm1); they are not selected using this plot's output.
ORIGINS = {"ETTh1": 1046, "Weather": 2073, "ETTm1": 9073}
COMPONENTS = ("global_linear", "recent_linear", "smooth_local", "smooth_multiscale", "causal_ema", "holt_local_linear")
LABELS = {
    "global_linear": "Global linear", "recent_linear": "Recent linear (last 96)",
    "smooth_local": "Local Gaussian smooth", "smooth_multiscale": "Short − long Gaussian",
    "causal_ema": "Causal EMA", "holt_local_linear": "Holt local linear",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args(); args.figures.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        values, count = scaled_validation_values(spec)
        origin = ORIGINS[spec.name]
        if not 0 <= origin < count:
            raise ValueError(f"fixed origin outside validation range: {spec.name} {origin}")
        history = values[origin:origin + LOOKBACK, 0]
        x = torch.tensor(history, dtype=torch.float32).view(1, LOOKBACK, 1)
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
        for axis, component in zip(axes.ravel(), COMPONENTS):
            _, _, _, hp = load_model(candidate_path(spec.name, component, "minus_component"))
            with torch.no_grad():
                extracted = extract_trend_component(x, component, **trend_kwargs(hp))[0, :, 0].numpy()
            time = np.arange(-LOOKBACK, 0)
            axis.plot(time, history, color="#7a7a7a", lw=.8, alpha=.75, label="full history X")
            axis.plot(time, extracted, color="#9467bd", lw=1.4, label="extracted A")
            axis.axvline(0, color="#999", lw=.7); axis.axhline(0, color="#999", lw=.55)
            axis.set_title(LABELS[component]); axis.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"{spec.name} | validation origin {origin} | channel 0 | L=720 | six continuous trend candidates", fontsize=13)
        fig.savefig(args.figures / f"{spec.name}__six_continuous_trend_components__origin{origin}_channel0.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
