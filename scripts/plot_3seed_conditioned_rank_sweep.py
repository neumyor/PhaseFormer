#!/usr/bin/env python3
"""3-seed conditioned low-rank compression charts (MSE and MAE separated).

Data sources (numeric authority, see docs/PhaseFormer_rank_sweep_conditioned_experiment.md §7):

  * research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/three_seed_summary.csv
    -> per (dataset, horizon, config) test MSE/MAE mean and sample std over seeds
       {2021, 2022, 2023}.
  * research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/audited_results.csv
    -> 105 audited per-seed cells (7 settings x 3 seeds x 5 configs), used only to
       scatter the individual seed points behind the mean line.

Golden reference values come from docs/PhaseFormer_gold_standard.md (3 decimals).

Charts produced:
  * three_seed_MSE_by_setting.png / three_seed_MAE_by_setting.png
      7 panels (one per setting) for a single metric; per panel: red-free mean line
      over compression tiers, semi-transparent mean +/- sample-sd band, and the
      Golden horizontal dashed baseline.
  * three_seed_<Dataset>_h<H>.png
      per-setting dual panel (MSE left, MAE right).
  * three_seed_figure_data.csv
      the exact numbers that were plotted (audit trail).

Replot:
  python scripts/plot_3seed_conditioned_rank_sweep.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOLD_MSE = "#6b4ec9"
GOLD_MAE = "#1f9e77"

# ------------------------------------------------------------- frozen metadata
# (dataset, horizon, Golden MSE, Golden MAE) -- docs/PhaseFormer_gold_standard.md
SETTINGS = [
    ("ETTh2", 96, 0.275, 0.338),
    ("ETTh2", 720, 0.402, 0.436),
    ("ETTm2", 96, 0.163, 0.256),
    ("ETTm2", 192, 0.219, 0.293),
    ("Weather", 96, 0.148, 0.195),
    ("Weather", 192, 0.193, 0.237),
    ("Electricity", 336, 0.165, 0.257),
]

# tier key -> (display label, csv q value or None for the unfactored direct head)
TIERS = [
    ("direct", "direct", None),
    ("q=1/4", "q=1/4", 0.25),
    ("q=1/8", "q=1/8", 0.125),
    ("q=1/16", "q=1/16", 0.0625),
    ("q=1/32", "q=1/32", 0.03125),
]


def fmt_q(value: float) -> str:
    """0.25 -> 'q=1/4' using the same labels as the report tables."""
    return {
        1.0: "q=1",
        0.25: "q=1/4",
        0.125: "q=1/8",
        0.0625: "q=1/16",
        0.03125: "q=1/32",
    }[value]


def load_summary(path: Path) -> dict:
    """dict[(ds, h)][tier_key] = dict(rank, mse_mean, mse_std, mae_mean, mae_std)."""
    data: dict = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ds = row["dataset"]
            h = int(row["horizon"])
            cfg = row["config"]
            if cfg == "direct":
                key = "direct"
            elif cfg.startswith("q="):
                key = fmt_q(float(cfg[2:]))
            else:
                continue
            data.setdefault((ds, h), {})[key] = {
                "rank": int(row["rank"]) if row["rank"] else None,
                "mse_mean": float(row["test_mse_mean"]),
                "mse_std": float(row["test_mse_std"]),
                "mae_mean": float(row["test_mae_mean"]),
                "mae_std": float(row["test_mae_std"]),
            }
    return data


def load_seed_cells(path: Path) -> dict:
    """dict[(ds, h)][tier_key] = [(seed, mse, mae), ...] from audited per-seed rows."""
    data: dict = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ds = row["dataset"]
            h = int(row["horizon"])
            cfg = row["config"]
            key = "direct" if cfg == "direct" else fmt_q(float(cfg[2:]))
            data.setdefault((ds, h), {}).setdefault(key, []).append(
                (int(row["seed"]), float(row["test_mse"]), float(row["test_mae"]))
            )
    for per_setting in data.values():
        for cells in per_setting.values():
            cells.sort()
    return data


def tier_xticklabels(entry: dict, horizon: int) -> list[str]:
    labels = []
    for key, label, _ in TIERS:
        if key == "direct":
            labels.append(f"direct\n(Linear 720→{horizon})")
        else:
            rank = entry.get(key, {}).get("rank")
            labels.append(f"{label}\nr={rank}")
    return labels


def best_tier(entry: dict, metric: str) -> str:
    key = "mse_mean" if metric == "mse" else "mae_mean"
    best = min(TIERS, key=lambda t: entry[t[0]][key])
    return best[1]


def draw_metric_panel(ax, ds, h, entry, seeds, metric, gold, color, gold_color,
                      show_xticklabels=True, annotate_best=True):
    """One axis: mean line + mean±sd band + Golden baseline for a single metric."""
    x = np.arange(len(TIERS))
    means = np.array([entry[k][f"{metric}_mean"] for k, _, _ in TIERS])
    stds = np.array([entry[k][f"{metric}_std"] for k, _, _ in TIERS])

    metric_idx = 1 if metric == "mse" else 2
    seed_values = [
        cell[metric_idx]
        for key, _, _ in TIERS
        for cell in seeds.get(key, [])
    ]

    # Golden baseline
    ax.axhline(gold, color=gold_color, lw=1.6, ls=(0, (6, 3)), zorder=1)
    ax.annotate(
        f"Golden {gold:.3f}",
        xy=(0.995, gold), xycoords=("axes fraction", "data"),
        xytext=(-3, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=7.6, color=gold_color, zorder=5,
    )

    # semi-transparent ±1 sample-sd band over the 3 seeds
    ax.fill_between(
        x, means - stds, means + stds, color=color, alpha=0.20,
        lw=0, zorder=2, label="mean ± 1 sd (3 seeds)",
    )

    # individual seed points (faint) so the sd is traceable
    jitter = np.array([-0.09, 0.0, 0.09])
    for i, (key, _, _) in enumerate(TIERS):
        cells = sorted(seeds.get(key, []), key=lambda c: c[0])
        vals = [c[metric_idx] for c in cells]
        offs = jitter[: len(vals)] if len(vals) == 3 else np.linspace(-0.09, 0.09, len(vals))
        ax.plot(x[i] + offs, vals, ls="none", marker="o", ms=2.6,
                mfc="none", mec=MUTED, mew=0.8, alpha=0.85, zorder=3)

    # mean line on top
    ax.plot(x, means, color=color, lw=2.2, marker="o", ms=7,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.3,
            zorder=4, label="3-seed mean")

    lo = min(float((means - stds).min()), float(np.min(seed_values)), gold)
    hi = max(float((means + stds).max()), float(np.max(seed_values)), gold)
    pad = (hi - lo) * 0.18 + 1e-5
    ax.set_ylim(lo - pad, hi + pad * 2.0)
    ax.set_xlim(-0.45, len(TIERS) - 0.55)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(tier_xticklabels(entry, h), fontsize=7.8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(f"test {metric.upper()}", fontsize=9.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_2)

    if annotate_best:
        b = best_tier(entry, metric)
        delta = (gold - entry[b][f"{metric}_mean"]) / gold * 100.0
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"lowest mean: {b}  ({sign}{delta:.2f}% vs Golden)",
            xy=(0.02, 0.96), xycoords="axes fraction",
            ha="left", va="top", fontsize=7.4, color=INK_2,
        )


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    summary_dir = base / "research_runs/rank_sweep_2_multiseed_stage1_20260914_summary"
    out_dir = summary_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_dir / "three_seed_summary.csv")
    seeds = load_seed_cells(summary_dir / "audited_results.csv")

    missing = [
        (ds, h, k) for ds, h, _, _ in SETTINGS for k, _, _ in TIERS
        if k not in summary.get((ds, h), {})
    ]
    if missing:
        raise SystemExit(f"incomplete summary grid, missing cells: {missing}")

    written: list[Path] = []
    audit_rows: list[dict] = []

    # =============================================== per-metric grids (7 panels)
    for metric, color, gold_color in (
        ("mse", BLUE, GOLD_MSE),
        ("mae", ORANGE, GOLD_MAE),
    ):
        fig, axes = plt.subplots(4, 2, figsize=(12.6, 16.6), dpi=200, facecolor=SURFACE)
        axes = axes.ravel()
        for i, (ds, h, gold_mse, gold_mae) in enumerate(SETTINGS):
            gold = gold_mse if metric == "mse" else gold_mae
            draw_metric_panel(
                axes[i], ds, h, summary[(ds, h)], seeds.get((ds, h), {}),
                metric, gold, color, gold_color, annotate_best=True,
            )
            axes[i].set_title(
                f"{ds}  H{h}   —   Golden {metric.upper()} = {gold:.3f}",
                fontsize=10.5, color=INK, pad=6,
            )
        axes[len(SETTINGS)].axis("off")

        handles = [
            Line2D([], [], color=color, marker="o", lw=2.2, ms=6.5,
                   label="3-seed mean (sample sd over seeds 2021/2022/2023)"),
            Line2D([], [], color=color, lw=8, alpha=0.20, label="mean ± 1 sd band"),
            Line2D([], [], color=gold_color, ls=(0, (6, 3)), lw=1.6,
                   label=f"Golden {metric.upper()} (PhaseFormer paper)"),
            Line2D([], [], color=MUTED, marker="o", ls="none", mfc="none", ms=4,
                   label="individual seed result"),
        ]
        fig.legend(handles=handles, loc="upper center", ncols=4, fontsize=9.5,
                   frameon=False, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.968))
        fig.suptitle(
            f"Conditioned low-rank rank sweep — test {metric.upper()} vs compression tier "
            f"(3-seed mean ± sd, Golden baseline)",
            fontsize=13.5, color=INK, y=0.993,
        )
        fig.text(
            0.5, 0.006,
            "7 settings are the round-1 test-selected \"nominal both-metric-win\" set → conditional, "
            "test-exposed evidence, not an unbiased generalization estimate.   "
            "x = rank compression q = rank/H (H = horizon); direct = unfactored Linear(720→H).   "
            "Source: three_seed_summary.csv / audited_results.csv (105/105 audited cells).",
            ha="center", va="bottom", fontsize=8.2, color=INK_2,
        )
        fig.tight_layout(rect=(0, 0.028, 1, 0.952))
        path = out_dir / f"three_seed_{metric.upper()}_by_setting.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    # ============================================ per-setting dual-panel charts
    for ds, h, gold_mse, gold_mae in SETTINGS:
        entry = summary[(ds, h)]
        seed_entry = seeds.get((ds, h), {})
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=200, facecolor=SURFACE)
        draw_metric_panel(axes[0], ds, h, entry, seed_entry, "mse",
                          gold_mse, BLUE, GOLD_MSE)
        draw_metric_panel(axes[1], ds, h, entry, seed_entry, "mae",
                          gold_mae, ORANGE, GOLD_MAE)
        axes[0].set_title(f"test MSE   (Golden {gold_mse:.3f})", fontsize=10.5, color=INK)
        axes[1].set_title(f"test MAE   (Golden {gold_mae:.3f})", fontsize=10.5, color=INK)

        handles = [
            Line2D([], [], color=BLUE, marker="o", lw=2.2, ms=6.5, label="MSE mean (3 seeds)"),
            Line2D([], [], color=BLUE, lw=8, alpha=0.20, label="MSE mean ± 1 sd"),
            Line2D([], [], color=GOLD_MSE, ls=(0, (6, 3)), lw=1.6, label="Golden MSE"),
            Line2D([], [], color=ORANGE, marker="s", lw=2.2, ms=6.5, label="MAE mean (3 seeds)"),
            Line2D([], [], color=ORANGE, lw=8, alpha=0.20, label="MAE mean ± 1 sd"),
            Line2D([], [], color=GOLD_MAE, ls=(0, (6, 3)), lw=1.6, label="Golden MAE"),
        ]
        fig.legend(handles=handles, loc="upper center", ncols=6, fontsize=8.6,
                   frameon=False, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.945))
        fig.suptitle(
            f"{ds} H{h} — 3-seed test error across rank compression tiers",
            fontsize=13, color=INK, y=0.985,
        )
        fig.text(
            0.5, 0.012,
            "seeds 2021/2022/2023 · seed-2021 Stage 0 frozen (gate_init, lr) per setting · "
            "conditional post-hoc setting selection",
            ha="center", va="bottom", fontsize=8.2, color=INK_2,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.885))
        path = out_dir / f"three_seed_{ds}_h{h}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    # ================================================== audit trail of plotted data
    for ds, h, gold_mse, gold_mae in SETTINGS:
        entry = summary[(ds, h)]
        for key, label, q in TIERS:
            cell = entry[key]
            audit_rows.append({
                "dataset": ds,
                "horizon": h,
                "tier": label,
                "q": "" if q is None else f"{q:g}",
                "rank": "" if cell["rank"] is None else cell["rank"],
                "test_mse_mean": f"{cell['mse_mean']:.6f}",
                "test_mse_std": f"{cell['mse_std']:.6f}",
                "test_mae_mean": f"{cell['mae_mean']:.6f}",
                "test_mae_std": f"{cell['mae_std']:.6f}",
                "golden_mse": f"{gold_mse:.3f}",
                "golden_mae": f"{gold_mae:.3f}",
                "n_seeds": len(seeds.get((ds, h), {}).get(key, [])),
            })
    audit_path = out_dir / "three_seed_figure_data.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    written.append(audit_path)

    print(f"3-seed charts written to {out_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
