#!/usr/bin/env python3
"""Render analysis figures for the joint low-rank rank sweep.

Reads the ``phase_a_*_results.csv`` summaries produced by
``scripts/run_joint_pooled_lowrank_phase_a.py`` and draws four figures:

1. rank_response.png       - per-setting test-MSE response to relative rank q
                             (small multiples, delta vs q=1)
2. delta_vs_phase_only.png - diverging heatmaps of delta vs matched phase_only
3. delta_vs_golden.png     - diverging heatmaps of delta vs the Golden table
4. val_vs_test_best.png    - val-best vs test-best rank ratio per setting

Usage:
    python scripts/plot_rank_sweep_results.py \
        --results-dir research_runs/joint_lowrank_rank_sweep_v1 \
        --output-dir research_runs/joint_lowrank_rank_sweep_v1/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Reference palette (dataviz skill light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
DIV_MID = "#f0efec"

QS = [1, 0.25, 0.125, 0.0625, 0.03125]
QLABELS = ["1", "1/4", "1/8", "1/16", "1/32"]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
HORIZONS = [96, 192]
CONFIGS = ["direct_nlinear"] + [f"pool1_q{q:g}" for q in QS]
COLLABELS = ["direct", "q=1", "q=1/4", "q=1/8", "q=1/16", "q=1/32"]

GOLD = {
    ("ETTh1", 96): (0.359, 0.382), ("ETTh1", 192): (0.397, 0.404),
    ("ETTh2", 96): (0.275, 0.338), ("ETTh2", 192): (0.341, 0.376),
    ("ETTm1", 96): (0.293, 0.344), ("ETTm1", 192): (0.323, 0.361),
    ("ETTm2", 96): (0.163, 0.256), ("ETTm2", 192): (0.219, 0.293),
    ("Weather", 96): (0.148, 0.195), ("Weather", 192): (0.193, 0.237),
}

SETTINGS = [(ds, h) for ds in DATASETS for h in HORIZONS]


def load(results_dir: Path) -> dict:
    data: dict = {}
    for path in sorted(results_dir.glob("phase_a_*_results.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], int(row["horizon"]))
                config = row["config_id"]
                # Normalize pool1_q1_r96 -> pool1_q1 so lookups do not depend
                # on the rank suffix.
                if config.startswith("pool1_q"):
                    config = "pool1_q" + config.split("_q")[1].split("_")[0]
                data.setdefault(key, {})[config] = {
                    "test_mse": float(row["test_mse"]),
                    "test_mae": float(row["test_mae"]),
                    "val_mse": float(row["val_mse"]),
                    "val_mae": float(row["val_mae"]),
                }
    return data


def cid(q) -> str:
    return f"pool1_q{q:g}"


def pct(reference: float, value: float) -> float:
    return (reference - value) / reference * 100.0


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def fig_rank_response(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(
        2, 5, figsize=(13.2, 5.2), dpi=200, sharex=True, sharey=True,
        facecolor=SURFACE,
    )
    x = np.arange(len(QS))
    for ax, (ds, h) in zip(axes.flat, SETTINGS):
        style_axes(ax)
        q1 = data[(ds, h)][cid(1)]["test_mse"]
        direct = pct(q1, data[(ds, h)]["direct_nlinear"]["test_mse"])
        y = [pct(q1, data[(ds, h)][cid(q)]["test_mse"]) for q in QS]
        ax.axhspan(-1, 1, color=GRID, alpha=0.55, zorder=0, lw=0)
        ax.axhline(0, color=BASELINE, lw=1, zorder=1)
        ax.axhline(direct, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6.5,
                markerfacecolor=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.set_title(f"{ds}  H{h}", fontsize=9, color=INK_2, pad=4)
        ax.set_xticks(x, QLABELS)
        ax.text(
            0.99, 0.02, f"direct {direct:+.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=MUTED,
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Test MSE vs q=1 (%)", fontsize=8.5)
    for ax in axes[1, :]:
        ax.set_xlabel("relative rank q", fontsize=8.5)
    axes[0, 0].set_ylim(-3.6, 2.1)
    axes[0, 0].text(
        0.02, 0.75, "±1% band", transform=axes[0, 0].transAxes,
        fontsize=7, color=MUTED,
    )
    fig.suptitle(
        "Rank response: test MSE of the pooled low-rank branch vs the "
        "uncompressed factorized map (q=1)",
        fontsize=11, color=INK, x=0.5, y=0.985,
    )
    fig.text(
        0.5, 0.925,
        "solid line = pooled low-rank (pool=1, no smoothing) · dashed = "
        "direct_nlinear reference · shaded band = ±1% (single-seed noise "
        "scale)",
        ha="center", fontsize=8, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _heatmap(ax, matrix, vmax, title, xlabels):
    cmap = LinearSegmentedColormap.from_list(
        "div_br", [BLUE, DIV_MID, RED], N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.set_yticks(range(len(SETTINGS)), [f"{ds} H{h}" for ds, h in SETTINGS],
                  fontsize=8)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White gap between cells.
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SETTINGS)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            dark = abs(value) / vmax > 0.55
            ax.text(
                j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7.2,
                color="#ffffff" if dark else INK,
            )
    ax.set_title(title, fontsize=9.5, color=INK_2, pad=6)


def fig_delta_heatmaps(data: dict, reference: str, out: Path) -> None:
    mse = np.zeros((len(SETTINGS), len(CONFIGS)))
    mae = np.zeros_like(mse)
    for i, (ds, h) in enumerate(SETTINGS):
        rows = data[(ds, h)]
        if reference == "phase_only":
            r_mse = rows["phase_only"]["test_mse"]
            r_mae = rows["phase_only"]["test_mae"]
        else:
            r_mse, r_mae = GOLD[(ds, h)]
        for j, config in enumerate(CONFIGS):
            mse[i, j] = pct(r_mse, rows[config]["test_mse"])
            mae[i, j] = pct(r_mae, rows[config]["test_mae"])
    fig, axes = plt.subplots(
        1, 2, figsize=(9.6, 4.6), dpi=200, facecolor=SURFACE,
    )
    ref_name = "phase_only" if reference == "phase_only" else "Golden"
    vmax = 7.0 if reference == "phase_only" else 7.0
    _heatmap(axes[0], mse, vmax, "MSE", COLLABELS)
    _heatmap(axes[1], mae, vmax, "MAE", COLLABELS)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(
        f"Test delta vs {ref_name} (%) - positive = lower error than "
        f"{ref_name}",
        fontsize=11, color=INK, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=LinearSegmentedColormap.from_list("d", [BLUE, DIV_MID, RED]),
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        ),
        ax=axes, orientation="horizontal", fraction=0.045, pad=0.10,
    )
    cbar.set_label("Δ% (positive = better)", fontsize=8, color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    cbar.outline.set_visible(False)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_val_vs_test(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200, facecolor=SURFACE)
    x = np.arange(len(QS))
    yticks, ylabels = [], []
    for i, (ds, h) in enumerate(SETTINGS):
        rows = data[(ds, h)]
        y = len(SETTINGS) - 1 - i
        yticks.append(y)
        ylabels.append(f"{ds} H{h}")
        val_q = min(QS, key=lambda q: rows[cid(q)]["val_mse"])
        test_q = min(QS, key=lambda q: rows[cid(q)]["test_mse"])
        xv, xt = QS.index(val_q), QS.index(test_q)
        offset = 0.0 if xv != xt else 0.10
        ax.plot(
            [x[xv] - (offset if xv == xt else 0), x[xt] + (offset if xv == xt else 0)],
            [y, y], color=BASELINE, lw=1.5, zorder=1,
        )
        ax.plot(
            x[xv] - offset, y, "o", ms=9, color=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
        )
        ax.plot(
            x[xt] + offset, y, "D", ms=8, color=ORANGE,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
        )
    ax.set_yticks(yticks, ylabels, fontsize=8.5)
    ax.set_xticks(x, QLABELS, fontsize=9)
    ax.set_xlabel("relative rank q (left = uncompressed)", fontsize=9)
    ax.set_xlim(-0.5, len(QS) - 0.5)
    ax.set_ylim(-0.6, len(SETTINGS) - 0.4)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    val_handle = plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=8,
                            label="validation-best q")
    test_handle = plt.Line2D([], [], color=ORANGE, marker="D", ls="", ms=7,
                             label="test-best q (MSE)")
    fig.legend(
        handles=[val_handle, test_handle], loc="lower center", ncols=2,
        fontsize=8.5, frameon=False, labelcolor=INK_2,
        bbox_to_anchor=(0.5, -0.02),
    )
    ax.set_title(
        "Validation-best vs test-best rank ratio (argmax over q)",
        fontsize=11, color=INK, pad=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="research_runs/joint_lowrank_rank_sweep_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(results_dir)
    missing = [s for s in SETTINGS if len(data.get(s, {})) < len(CONFIGS) + 1]
    if missing:
        raise SystemExit(f"missing results for {missing}")
    fig_rank_response(data, out_dir / "rank_response.png")
    fig_delta_heatmaps(data, "phase_only", out_dir / "delta_vs_phase_only.png")
    fig_delta_heatmaps(data, "golden", out_dir / "delta_vs_golden.png")
    fig_val_vs_test(data, out_dir / "val_vs_test_best.png")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
