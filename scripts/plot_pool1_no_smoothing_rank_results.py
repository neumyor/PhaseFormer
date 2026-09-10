#!/usr/bin/env python3
"""Plot only pool=1, no-smoothing pooled-low-rank results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
INITIAL = ROOT / "research_runs/joint_pooled_lowrank_nlinear_h96_v1/results.csv"
PHASE_A = Path("/tmp/phase_a_exports")
OUTPUT = ROOT / "research_runs/joint_pooled_lowrank_nlinear_h96_v1/figures/pool1_no_smoothing_relative_rank_curves.png"
DATASETS = ["ETTh1", "ETTm1"]
COLORS = {"ETTh1": "#2563eb", "ETTm1": "#dc2626"}


def load_initial() -> list[dict]:
    rows = []
    with INITIAL.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    for dataset in DATASETS:
        setting = f"{dataset}_h96_seed2021"
        baseline = next(
            row
            for row in source
            if row["setting"] == setting and row["model"] == "phase_only"
        )
        for row in source:
            if (
                row["setting"] != setting
                or row["model"] != "pooled_lowrank"
                or int(row["pool_factor"]) != 1
                or float(row["smooth_ratio"]) != 0.0
            ):
                continue
            rank = int(row["rank"])
            rows.append(
                {
                    "dataset": dataset,
                    "q": rank / 96.0,
                    "mse": 100 * (float(row["mse"]) - float(baseline["mse"])) / float(baseline["mse"]),
                    "mae": 100 * (float(row["mae"]) - float(baseline["mae"])) / float(baseline["mae"]),
                }
            )
    return rows


def load_phase_a() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for seed in [2021, 2022, 2023]:
            path = PHASE_A / f"phase_a_{dataset}_h96_s{seed}_validation.csv"
            with path.open(newline="") as handle:
                source = list(csv.DictReader(handle))
            baseline = next(row for row in source if row["config_id"] == "direct_nlinear")
            for row in source:
                if not row["relative_rank"] or int(row["pool_factor"]) != 1:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "q": float(row["relative_rank"]),
                        "mse": 100 * (float(row["val_mse"]) - float(baseline["val_mse"])) / float(baseline["val_mse"]),
                        "mae": 100 * (float(row["val_mae"]) - float(baseline["val_mae"])) / float(baseline["val_mae"]),
                    }
                )
    return rows


def format_axis(axis, title):
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1 / 24, 1 / 12, 1 / 6, 1 / 3, 2 / 3, 1])
    axis.set_xticklabels(["1/24", "1/12", "1/6", "1/3", "2/3", "1"])
    axis.set_xlabel("Relative rank q (pool=1)")
    axis.set_ylabel("Delta vs matched control (%)")
    axis.set_title(title)
    axis.grid(alpha=0.25)


def main() -> None:
    initial = load_initial()
    phase_a = load_phase_a()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for dataset in DATASETS:
        values = sorted(
            [row for row in initial if row["dataset"] == dataset],
            key=lambda row: row["q"],
        )
        axes[0].plot(
            [row["q"] for row in values],
            [row["mse"] for row in values],
            marker="o",
            linewidth=1.8,
            color=COLORS[dataset],
            label=f"{dataset} initial test",
        )

        grouped = defaultdict(list)
        for row in phase_a:
            if row["dataset"] == dataset:
                grouped[row["q"]].append(row)
        points = []
        for q in sorted(grouped):
            metric_values = np.asarray([row["mse"] for row in grouped[q]], dtype=float)
            points.append((q, metric_values.mean(), metric_values.std(ddof=1)))
        axes[0].errorbar(
            [point[0] for point in points],
            [point[1] for point in points],
            yerr=[point[2] for point in points],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            capsize=3,
            color=COLORS[dataset],
            label=f"{dataset} Phase A validation",
        )

        axes[1].plot(
            [row["q"] for row in values],
            [row["mae"] for row in values],
            marker="o",
            linewidth=1.8,
            color=COLORS[dataset],
            label=f"{dataset} initial test",
        )
        points = []
        for q in sorted(grouped):
            metric_values = np.asarray([row["mae"] for row in grouped[q]], dtype=float)
            points.append((q, metric_values.mean(), metric_values.std(ddof=1)))
        axes[1].errorbar(
            [point[0] for point in points],
            [point[1] for point in points],
            yerr=[point[2] for point in points],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            capsize=3,
            color=COLORS[dataset],
            label=f"{dataset} Phase A validation",
        )

    format_axis(axes[0], "Pool=1, no smoothing: MSE")
    format_axis(axes[1], "Pool=1, no smoothing: MAE")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Pooled-low-rank results with pool=1 and smooth_ratio=0 only",
        fontsize=13,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
