#!/usr/bin/env python3
"""Plot MSE/MAE vs smooth_ratio for the 7-setting smoothing sweep.

Data is the verified summary table in
docs/PhaseFormer_residual_smooth_ratio_sweep_experiment.md §6 (table 1),
hardcoded here rather than re-read from the raw CSVs, since most of the
raw per-run CSVs currently live server-side only (see docs/agent-log.md,
2026-09-12 entry, for the SSH tooling limitation that caused this).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

DATA: dict[str, dict[str, list[float]]] = {
    "ETTh2-96": {
        "mse": [0.2723, 0.2727, 0.2734, 0.2740, 0.2753],
        "mae": [0.3348, 0.3353, 0.3361, 0.3370, 0.3388],
    },
    "ETTh2-720": {
        "mse": [0.3852, 0.3846, 0.3858, 0.3866, 0.3967],
        "mae": [0.4246, 0.4246, 0.4253, 0.4261, 0.4343],
    },
    "ETTm2-96": {
        "mse": [0.1598, 0.1603, 0.1607, 0.1612, 0.1616],
        "mae": [0.2494, 0.2499, 0.2503, 0.2508, 0.2512],
    },
    "ETTm2-192": {
        "mse": [0.2135, 0.2135, 0.2137, 0.2140, 0.2142],
        "mae": [0.2878, 0.2879, 0.2880, 0.2883, 0.2886],
    },
    "Weather-96": {
        "mse": [0.1463, 0.1463, 0.1467, 0.1474, 0.1485],
        "mae": [0.1948, 0.1946, 0.1950, 0.1956, 0.1961],
    },
    "Weather-192": {
        "mse": [0.1908, 0.1904, 0.1912, 0.1918, 0.1904],
        "mae": [0.2362, 0.2357, 0.2362, 0.2368, 0.2361],
    },
    "Electricity-336": {
        "mse": [0.1619, 0.1621, 0.1620, 0.1629, 0.1628],
        "mae": [0.2550, 0.2551, 0.2557, 0.2559, 0.2560],
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
        "Weak-period residual smoothing sweep — test MSE/MAE vs smooth_ratio\n"
        "(7 settings, per-setting frozen rank/gate_init/lr, seed 2021, test-exposed)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = ROOT / "research_runs/smooth_ratio_sweep_v1/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smooth_ratio_sweep_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
