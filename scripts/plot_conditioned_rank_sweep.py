#!/usr/bin/env python3
"""Compression-tier line charts per conditioned setting, one axis per metric.

For each of the seven Stage 1 settings this draws the error curve across the
compression ladder (direct -> q=1 -> ... -> q=1/32). MSE and MAE are drawn on
separate axes, each with its own Golden horizontal baseline. Outputs a 7x2
grid (rows = settings, columns = metric) plus one two-panel figure per
setting.

Reads the ``phase_a_*_results.csv`` files written by the conditioned Stage 1
run. Those files carry a lossy ``config_id`` for the H336/H720 non-divisible
tiers (``exact_rank`` rounds q*max_rank), so rank tiers are recovered by
sorting the pool rows on their parsed q rather than by trusting the suffix.

Usage:
    python scripts/plot_conditioned_rank_sweep.py \
        --results-dir research_runs/rank_sweep_2_stage1 \
        --output-dir research_runs/rank_sweep_2_stage1/figures
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (dataviz skill light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# (dataset, horizon, golden_mse, golden_mae)
SETTINGS = [
    ("ETTh2", 96, 0.275, 0.338),
    ("ETTh2", 720, 0.402, 0.436),
    ("ETTm2", 96, 0.163, 0.256),
    ("ETTm2", 192, 0.219, 0.293),
    ("Weather", 96, 0.148, 0.195),
    ("Weather", 192, 0.193, 0.237),
    ("Electricity", 336, 0.165, 0.257),
]

TIER_LABELS = ["direct", "q=1", "q=1/4", "q=1/8", "q=1/16", "q=1/32"]

_POOL_Q = re.compile(r"pool1_q([0-9.]+)")


def load(results_dir: Path) -> dict:
    data: dict = {}
    for path in sorted(results_dir.glob("phase_a_*_results.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        key = (rows[0]["dataset"], int(rows[0]["horizon"]))
        entry = {"direct": None, "tiers": []}
        for row in rows:
            cid = row["config_id"]
            record = {
                "test_mse": float(row["test_mse"]),
                "test_mae": float(row["test_mae"]),
            }
            if cid == "direct_nlinear":
                entry["direct"] = record
            elif cid == "phase_only":
                continue
            else:
                match = _POOL_Q.search(cid)
                if match:
                    entry["tiers"].append((float(match.group(1)), record))
        # Descending q => uncompressed (q=1) first, deepest compression last.
        entry["tiers"].sort(key=lambda item: item[0], reverse=True)
        entry["tiers"] = [record for _, record in entry["tiers"]]
        data[key] = entry
    return data


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _ladder(entry, metric: str) -> list:
    """[direct, q=1, q=1/4, ...] of the chosen test metric."""
    return [entry["direct"][metric]] + [t[metric] for t in entry["tiers"]]


def _draw_metric(ax, values, gold, color, marker, label, title=None) -> int:
    """Draw one metric ladder with its Golden baseline on its own axis."""
    style_axes(ax)
    x = np.arange(len(TIER_LABELS))
    lo = min(min(values), gold)
    hi = max(max(values), gold)
    pad = (hi - lo) * 0.20 + 1e-4
    ax.set_ylim(lo - pad, hi + pad * 1.7)
    ax.set_xlim(-0.4, len(TIER_LABELS) - 0.6)

    ax.axhline(gold, color=color, lw=1.3, ls=(0, (5, 3)), zorder=1)
    ax.plot(
        x, values, color=color, lw=2.2, marker=marker, ms=7,
        markerfacecolor=color, markeredgecolor=SURFACE,
        markeredgewidth=1.3, zorder=3,
    )

    best = int(np.argmin(values))
    ax.plot(
        x[best], values[best], "o", ms=14, markerfacecolor="none",
        markeredgecolor=color, markeredgewidth=1.6, zorder=4,
    )
    # Keep the callout inside the axes at both ends of the ladder.
    ha = "center" if 0 < best < len(TIER_LABELS) - 1 else (
        "left" if best == 0 else "right"
    )
    ax.annotate(
        f"best: {TIER_LABELS[best]}", (x[best], values[best]),
        textcoords="offset points", xytext=(0, 13), ha=ha,
        fontsize=7.5, color=color,
    )

    ax.set_xticks(x, TIER_LABELS, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.02, 0.96, f"Golden {label} {gold:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.6, color=MUTED,
    )
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    return best


def _legend_handles():
    return [
        plt.Line2D([], [], color=BLUE, marker="o", ls="-", lw=2, ms=6,
                   label="model test MSE"),
        plt.Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=1.4,
                   label="Golden MSE"),
        plt.Line2D([], [], color=ORANGE, marker="s", ls="-", lw=2, ms=6,
                   label="model test MAE"),
        plt.Line2D([], [], color=ORANGE, ls=(0, (5, 3)), lw=1.4,
                   label="Golden MAE"),
    ]


def fig_single(data: dict, out_dir: Path) -> list:
    """Two panels per setting (MSE | MAE), seven files."""
    written = []
    for ds, h, gold_mse, gold_mae in SETTINGS:
        fig, axes = plt.subplots(
            1, 2, figsize=(11.0, 4.0), dpi=200, facecolor=SURFACE,
        )
        entry = data[(ds, h)]
        _draw_metric(axes[0], _ladder(entry, "test_mse"), gold_mse,
                     BLUE, "o", "MSE", title="MSE")
        _draw_metric(axes[1], _ladder(entry, "test_mae"), gold_mae,
                     ORANGE, "s", "MAE", title="MAE")
        for ax in axes:
            ax.set_xlabel("compression tier (left = uncompressed)", fontsize=8.5)
        axes[0].set_ylabel("test MSE (lower = better)", fontsize=9)
        axes[1].set_ylabel("test MAE (lower = better)", fontsize=9)
        fig.legend(
            handles=_legend_handles(), loc="upper center", ncols=4,
            fontsize=8.5, frameon=False, labelcolor=INK_2,
            bbox_to_anchor=(0.5, 0.945),
        )
        fig.suptitle(
            f"{ds} H{h} — test error across compression tiers",
            fontsize=12, color=INK, y=0.985,
        )
        fig.text(
            0.5, 0.015,
            "single seed 2021 · Stage-0 frozen config · conditional "
            "(post-hoc setting selection)",
            ha="center", va="bottom", fontsize=7.8, color=INK_2,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.90))
        name = out_dir / f"compression_{ds}_h{h}.png"
        fig.savefig(name, facecolor=SURFACE)
        plt.close(fig)
        written.append(name)
    return written


def fig_compression_lines(data: dict, out: Path) -> None:
    """7x2 grid: rows = settings, columns = MSE / MAE."""
    fig, axes = plt.subplots(
        len(SETTINGS), 2, figsize=(11.4, 19.0), dpi=200, facecolor=SURFACE,
    )
    for i, (ds, h, gold_mse, gold_mae) in enumerate(SETTINGS):
        entry = data[(ds, h)]
        _draw_metric(
            axes[i, 0], _ladder(entry, "test_mse"), gold_mse, BLUE, "o", "MSE",
            title=f"{ds} H{h} — MSE",
        )
        _draw_metric(
            axes[i, 1], _ladder(entry, "test_mae"), gold_mae, ORANGE, "s",
            "MAE", title=f"{ds} H{h} — MAE",
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("compression tier (left = uncompressed)", fontsize=8.5)
    for i in range(len(SETTINGS)):
        axes[i, 0].set_ylabel("test MSE", fontsize=8.5)
        axes[i, 1].set_ylabel("test MAE", fontsize=8.5)

    fig.legend(
        handles=_legend_handles(), loc="upper center", ncols=4, fontsize=9,
        frameon=False, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.982),
    )
    fig.suptitle(
        "Conditioned sweep: test MSE (left) and MAE (right) across "
        "compression tiers, Golden as baseline",
        fontsize=13, color=INK, y=0.997,
    )
    fig.text(
        0.5, 0.004,
        "single seed 2021 · Stage-0 frozen configs · 7 settings selected "
        "post-hoc on round-1 test wins (conditional, not blind)",
        ha="center", va="bottom", fontsize=8.4, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.016, 1, 0.965))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="research_runs/rank_sweep_2_stage1",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = (
        Path(args.output_dir) if args.output_dir else results_dir / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(results_dir)
    missing = [f"{ds}-{h}" for ds, h, *_ in SETTINGS if (ds, h) not in data]
    if missing:
        raise SystemExit(f"missing results for {missing}")
    fig_compression_lines(data, out_dir / "compression_lines.png")
    singles = fig_single(data, out_dir)
    print(f"grid written to {out_dir / 'compression_lines.png'}")
    for name in singles:
        print(f"single written to {name}")


if __name__ == "__main__":
    main()
