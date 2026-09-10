#!/usr/bin/env python3
"""Plot all available no-smoothing pooled-low-rank screen results."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
INITIAL = ROOT / "research_runs/joint_pooled_lowrank_nlinear_h96_v1/results.csv"
PHASE_A = Path("/tmp/phase_a_exports")
OUTPUT = ROOT / "research_runs/joint_pooled_lowrank_nlinear_h96_v1/figures/all_no_smoothing_relative_rank_curves.png"

DATASETS = ["ETTh1", "ETTm1"]
POOLS = [1, 2, 4, 8]
COLORS = {1: "#2563eb", 2: "#dc2626", 4: "#16a34a", 8: "#9333ea"}
MARKERS = {"ETTh1": "o", "ETTm1": "s"}


def relative_rank(pool: int, rank: int, horizon: int = 96) -> float:
    maximum = min(math.ceil(720 / pool), horizon)
    return rank / maximum


def load_initial() -> list[dict]:
    rows = []
    with INITIAL.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    for dataset in DATASETS:
        setting = f"{dataset}_h96_seed2021"
        base = next(
            row
            for row in source
            if row["setting"] == setting and row["model"] == "phase_only"
        )
        for row in source:
            if (
                row["setting"] != setting
                or row["model"] != "pooled_lowrank"
                or float(row["smooth_ratio"]) != 0.0
            ):
                continue
            pool = int(row["pool_factor"])
            rank = int(row["rank"])
            rows.append(
                {
                    "dataset": dataset,
                    "pool": pool,
                    "q": relative_rank(pool, rank),
                    "mse": 100 * (float(row["mse"]) - float(base["mse"])) / float(base["mse"]),
                    "mae": 100 * (float(row["mae"]) - float(base["mae"])) / float(base["mae"]),
                    "protocol": "Initial H96 test screen",
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
            base = next(row for row in source if row["config_id"] == "direct_nlinear")
            for row in source:
                if not row["relative_rank"]:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "pool": int(row["pool_factor"]),
                        "q": float(row["relative_rank"]),
                        "mse": 100 * (float(row["val_mse"]) - float(base["val_mse"])) / float(base["val_mse"]),
                        "mae": 100 * (float(row["val_mae"]) - float(base["val_mae"])) / float(base["val_mae"]),
                        "protocol": "Phase A validation",
                    }
                )
    return rows


def plot_initial(axis, rows, metric):
    for dataset in DATASETS:
        for pool in POOLS:
            values = sorted(
                (row for row in rows if row["dataset"] == dataset and row["pool"] == pool),
                key=lambda row: row["q"],
            )
            if not values:
                continue
            axis.plot(
                [row["q"] for row in values],
                [row[metric] for row in values],
                color=COLORS[pool],
                marker=MARKERS[dataset],
                linewidth=1.5,
                markersize=4.5,
                alpha=0.8,
                label=f"{dataset}, pool={pool}",
            )


def plot_phase_a(axis, rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["pool"], row["q"])].append(row[metric])
    for dataset in DATASETS:
        for pool in [1, 2, 4]:
            points = []
            for q in sorted({key[2] for key in grouped if key[0] == dataset and key[1] == pool}):
                values = np.asarray(grouped[(dataset, pool, q)], dtype=float)
                points.append((q, values.mean(), values.std(ddof=1)))
            if not points:
                continue
            axis.errorbar(
                [point[0] for point in points],
                [point[1] for point in points],
                yerr=[point[2] for point in points],
                color=COLORS[pool],
                marker=MARKERS[dataset],
                linestyle="--",
                linewidth=1.8,
                markersize=6,
                capsize=3,
                label=f"{dataset}, pool={pool}",
            )


def format_axis(axis, title):
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1 / 12, 1 / 3, 1])
    axis.set_xticklabels(["1/12", "1/3", "1"])
    axis.set_xlabel("Relative rank q")
    axis.set_ylabel("Delta vs matched control (%)")
    axis.set_title(title)
    axis.grid(alpha=0.25)


def main() -> None:
    initial = load_initial()
    phase_a = load_phase_a()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    plot_initial(axes[0, 0], initial, "mse")
    plot_initial(axes[0, 1], initial, "mae")
    plot_phase_a(axes[1, 0], phase_a, "mse")
    plot_phase_a(axes[1, 1], phase_a, "mae")
    format_axis(axes[0, 0], "Initial H96 test screen: MSE")
    format_axis(axes[0, 1], "Initial H96 test screen: MAE")
    format_axis(axes[1, 0], "Phase A validation: MSE mean +/- seed SD")
    format_axis(axes[1, 1], "Phase A validation: MAE mean +/- seed SD")
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    axes[0, 1].legend(fontsize=8, ncol=2, frameon=False)
    figure.suptitle(
        "All available no-smoothing pooled-low-rank results\n"
        "Initial screen uses phase-only control; Phase A uses same-seed direct-NLinear control",
        fontsize=13,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
