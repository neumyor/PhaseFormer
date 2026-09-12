#!/usr/bin/env python3
"""Plot MSE/MAE vs smooth_ratio for the 7-setting causal-EMA smoothing sweep.

Data is the verified summary table in
docs/PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md §6 (table 1),
hardcoded here rather than re-read from the raw CSVs, following the same
pattern as scripts/plot_smooth_ratio_sweep.py (the boxcar-sweep counterpart).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

DATA: dict[str, dict[str, list[float]]] = {
    "ETTh2-96": {
        "mse": [0.2721, 0.2730, 0.2743, 0.2758, 0.2770],
        "mae": [0.3328, 0.3336, 0.3348, 0.3363, 0.3376],
    },
    "ETTh2-720": {
        "mse": [0.3909, 0.3912, 0.3920, 0.3940, 0.3958],
        "mae": [0.4279, 0.4279, 0.4284, 0.4297, 0.4311],
    },
    "ETTm2-96": {
        "mse": [0.1585, 0.1585, 0.1587, 0.1599, 0.1604],
        "mae": [0.2480, 0.2481, 0.2484, 0.2500, 0.2506],
    },
    "ETTm2-192": {
        "mse": [0.2157, 0.2158, 0.2162, 0.2169, 0.2174],
        "mae": [0.2881, 0.2883, 0.2888, 0.2895, 0.2901],
    },
    "Weather-96": {
        "mse": [0.1467, 0.1467, 0.1469, 0.1468, 0.1469],
        "mae": [0.1940, 0.1940, 0.1941, 0.1940, 0.1938],
    },
    "Weather-192": {
        "mse": [0.1918, 0.1917, 0.1916, 0.1921, 0.1918],
        "mae": [0.2363, 0.2363, 0.2361, 0.2366, 0.2366],
    },
    "Electricity-336": {
        "mse": [0.1617, 0.1621, 0.1618, 0.1621, 0.1642],
        "mae": [0.2547, 0.2544, 0.2547, 0.2554, 0.2574],
    },
}

SETTINGS_ORDER = list(DATA.keys())


def main() -> None:
    fig, axes = plt.subplots(7, 2, figsize=(9, 20), sharex=True)

    for row, setting in enumerate(SETTINGS_ORDER):
        for col, metric in enumerate(("mse", "mae")):
            ax = axes[row, col]
            values = DATA[setting][metric]
            ax.plot(RATIOS, values, marker="o", color="C0" if metric == "mse" else "C1")
            ax.axvline(0.0, color="grey", linewidth=0.6, linestyle=":")
            best_idx = min(range(len(values)), key=lambda i: values[i])
            ax.scatter(
                [RATIOS[best_idx]],
                [values[best_idx]],
                color="red",
                zorder=5,
                s=30,
                label="test-best" if row == 0 and col == 0 else None,
            )
            ax.set_title(f"{setting} — {metric.upper()}", fontsize=9)
            ax.grid(True, linewidth=0.3, alpha=0.5)
            if row == 6:
                ax.set_xlabel("smooth_ratio")
            if col == 0:
                ax.set_ylabel(metric.upper())

    fig.suptitle(
        "Weak-period residual causal-EMA smoothing sweep — test MSE/MAE vs smooth_ratio\n"
        "(full-rank head, 7 settings, per-setting frozen gate_init/lr, alpha=0.08, "
        "seed 2021, test-exposed)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = ROOT / "research_runs/causal_ema_smooth_sweep_v1/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "causal_ema_smooth_sweep_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
