#!/usr/bin/env python3
"""Visual validation-only probe for trend-filter smoothing scales.

This diagnostic does not train or evaluate a forecasting model.  It compares
the endpoint-anchored continuous 72-step linear-spline component with
first-order trend-filter components across a small, fixed set of validation
windows.  The lambda rule is fixed in physical sampling-time units:
``lambda = kappa * sample_std * (one_hour / sample_interval)**2``.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler


LOOKBACK = 720
HORIZON = 96
CHANNEL = 0
WINDOW = 72
KAPPAS = (25.0, 100.0, 400.0)
# Fractions are frozen.  ETTh1's first origin rounds to the user-inspected
# origin 1046 under the repository's validation indexing.
ORIGIN_FRACTIONS = (1046.0 / 2784.0, 0.75)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    sample_interval_hours: float
    kind: str


SPECS = (
    DatasetSpec("ETTh1", Path("resources/all_datasets/ETT/ETTh1.csv"), 1.0, "ett_hour"),
    DatasetSpec("Weather", Path("resources/all_datasets/weather/weather.csv"), 1.0, "custom"),
    DatasetSpec("ETTm1", Path("resources/all_datasets/ETT/ETTm1.csv"), 0.25, "ett_minute"),
)


def second_difference(values: np.ndarray) -> np.ndarray:
    return values[:-2] - 2.0 * values[1:-1] + values[2:]


def second_difference_transpose(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.size + 2, dtype=np.float64)
    result[:-2] += values
    result[1:-1] -= 2.0 * values
    result[2:] += values
    return result


def trend_filter(values: np.ndarray, lam: float, *, iterations: int = 6000) -> np.ndarray:
    """Solve first-order trend filtering by deterministic ADMM."""
    length = values.size
    # Scaling rho with lambda keeps the ADMM shrinkage threshold and dual
    # residual well conditioned for ETTm's 16x physical-time conversion.
    rho = max(10.0, float(lam))
    operator = np.zeros((length - 2, length), dtype=np.float64)
    indices = np.arange(length - 2)
    operator[indices, indices] = 1.0
    operator[indices, indices + 1] = -2.0
    operator[indices, indices + 2] = 1.0
    factor = cho_factor(np.eye(length) + rho * operator.T @ operator, check_finite=False)
    fitted = values.copy()
    sparse_curvature = np.zeros(length - 2, dtype=np.float64)
    dual = np.zeros(length - 2, dtype=np.float64)
    for _ in range(iterations):
        fitted = cho_solve(
            factor, values + rho * second_difference_transpose(sparse_curvature - dual), check_finite=False
        )
        update = second_difference(fitted) + dual
        previous = sparse_curvature
        sparse_curvature = np.sign(update) * np.maximum(np.abs(update) - lam / rho, 0.0)
        dual += second_difference(fitted) - sparse_curvature
        primal = np.max(np.abs(second_difference(fitted) - sparse_curvature))
        dual_residual = np.max(np.abs(rho * second_difference_transpose(sparse_curvature - previous)))
        if primal < 1e-7 and dual_residual < 1e-5:
            break
    return fitted


def continuous_piecewise_linear(values: np.ndarray) -> np.ndarray:
    """Fit continuous 72-step segments using a linear-spline hinge basis."""
    time = np.arange(values.size, dtype=np.float64)
    knots = np.arange(WINDOW, values.size, WINDOW, dtype=np.float64)
    design = np.column_stack([np.ones(values.size), time] + [np.maximum(time - knot, 0.0) for knot in knots])
    return design @ np.linalg.lstsq(design, values, rcond=None)[0]


def scaled_validation_values(spec: DatasetSpec) -> tuple[np.ndarray, int]:
    frame = pd.read_csv(spec.csv_path)
    if spec.kind == "custom":
        # Match Dataset_Custom_Multi: date first, OT moved to the last variable.
        columns = list(frame.columns)
        columns.remove("OT")
        columns.remove("date")
        frame = frame[["date"] + columns + ["OT"]]
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        train_end = int(len(frame) * 0.7)
        val_end = train_end + (len(frame) - train_end - int(len(frame) * 0.2))
        val_start = train_end - LOOKBACK
    elif spec.kind == "ett_hour":
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        val_start = train_end - LOOKBACK
    else:
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        val_start = train_end - LOOKBACK
    scaled = StandardScaler().fit(values[:train_end]).transform(values)
    validation = scaled[val_start:val_end]
    count = validation.shape[0] - LOOKBACK - HORIZON + 1
    return validation, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for spec in SPECS:
        validation, count = scaled_validation_values(spec)
        origins = tuple(round(fraction * (count - 1)) for fraction in ORIGIN_FRACTIONS)
        for origin in origins:
            history = validation[origin : origin + LOOKBACK, CHANNEL]
            scale = float(history.std(ddof=0))
            spline = continuous_piecewise_linear(history)
            component_spline = spline - spline[-1]
            time = np.arange(LOOKBACK)
            fig, axis = plt.subplots(figsize=(15, 5.6), constrained_layout=True)
            axis.plot(time, history, color="#9aa0a6", linewidth=1.0, alpha=0.72, label="Original input X (scaled)")
            axis.plot(time, component_spline, color="#d62728", linewidth=2.2,
                      label="Continuous 72-step piecewise-linear A")
            for kappa, color in zip(KAPPAS, ("#1f77b4", "#2ca02c", "#9467bd")):
                lam = kappa * scale * (1.0 / spec.sample_interval_hours) ** 2
                filtered = trend_filter(history, lam)
                component = filtered - filtered[-1]
                axis.plot(time, component, color=color, linewidth=1.9,
                          label=f"Trend filter A, κ={kappa:g} (λ={lam:.2f})")
                rows.append({
                    "setting": f"{spec.name}_validation_origin{origin}_channel{CHANNEL}",
                    "dataset": spec.name,
                    "origin": origin,
                    "channel": CHANNEL,
                    "sample_interval_hours": spec.sample_interval_hours,
                    "kappa": kappa,
                    "lambda": lam,
                    "input_std": scale,
                    "component_end": float(component[-1]),
                })
            for knot in range(WINDOW, LOOKBACK, WINDOW):
                axis.axvline(knot, color="#d62728", linewidth=0.65, alpha=0.16)
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
            axis.set_title(f"{spec.name} validation | origin={origin} | channel {CHANNEL} | L=720")
            axis.set_xlabel("History step (0 = oldest, 719 = latest)")
            axis.set_ylabel("Training-standardized value / endpoint-anchored component")
            axis.legend(loc="upper left", ncol=1, frameon=True)
            axis.text(0.995, 0.015, "All A curves are f(t) − f(719); red guides mark 72-step knots.",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color="#444")
            figure_path = figures / f"{spec.name}_validation_origin{origin}_channel0.png"
            fig.savefig(figure_path, dpi=180)
            plt.close(fig)

    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = [row["setting"] for row in rows if row["kappa"] == KAPPAS[0]]
    np.savez(
        args.output / "selected_cases.npz",
        setting=np.asarray(settings, dtype=str),
        origin=np.asarray([int(row["origin"]) for row in rows if row["kappa"] == KAPPAS[0]]),
        channel=np.full(len(settings), CHANNEL, dtype=np.int64),
    )
    (args.output / "sample_errors.csv").write_text("setting\n", encoding="utf-8")
    (args.output / "run.yaml").write_text(
        "experiment_id: trend_filter_parameter_probe\n"
        "split: validation_only\n"
        "datasets: [ETTh1, Weather, ETTm1]\n"
        "lookback: 720\n"
        "horizon: 96\n"
        "channel: 0\n"
        "origins: [fixed_validation_fractions_0.3757, 0.75]\n"
        "piecewise_window: 72\n"
        "kappas: [25, 100, 400]\n"
        "lambda_rule: kappa * sample_std * (1 hour / sample_interval)^2\n"
        "forecast_model_training: false\n"
        "test_accessed: false\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Trend-filter parameter visual probe",
        "",
        "This is a validation-only extraction diagnostic; no model was trained and no test data was read.",
        "",
        "Each chart compares the original 720-step channel-0 history, a continuous 72-step linear-spline component, and trend-filter components with fixed κ = 25, 100, and 400.",
        "The rule is `lambda = kappa * sample_std * (1 hour / sample_interval)^2`; ETTm1 therefore uses 16× the hourly λ at equal κ.",
        "",
        "## Visual decision",
        "",
        "Across the six fixed samples, κ=25 still retains conspicuous local cyclic variation on ETTh1/ETTm1, while κ=400 often collapses the component toward a near-global drift. κ=100 retains medium-scale turns without tracking the dominant cycles, so it is the provisional single global κ for a later forecasting ablation. This is an extraction-scale decision only, not evidence of forecast improvement or branch utilization.",
        "",
        "## Figures",
        "",
    ]
    for spec in SPECS:
        validation, count = scaled_validation_values(spec)
        for origin in tuple(round(fraction * (count - 1)) for fraction in ORIGIN_FRACTIONS):
            name = f"{spec.name}_validation_origin{origin}_channel0.png"
            report_lines.append(f"- `{spec.name}`, validation origin {origin}, channel 0: [figure](figures/{name})")
    report_path = args.output / "objective_error_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")


if __name__ == "__main__":
    main()
