#!/usr/bin/env python3
"""Figures for the low-rank functional-rank analysis (plan section 15).

Reads the CSV/NPZ artifacts of ``analyze_lowrank_functional_rank.py`` and writes
PNG figures into ``<output-dir>/figures``.  Read-only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SETTING_ORDER = [
    "ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192",
]
ORDERING_STYLE = {
    "contribution": ("#c0392b", "-", "predictive contribution"),
    "singular": ("#2980b9", "--", "singular value"),
    "activation_energy": ("#27ae60", "-.", "activation energy"),
    "weight_energy": ("#8e44ad", ":", "weight energy"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def figure_rank_curves(curves: list[dict], cells: list[dict], figures: Path) -> None:
    """Recovery versus retained modes, one panel per setting (q=1/8)."""
    keep = {row["setting"] for row in cells}
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for axis, setting in zip(axes.ravel(), SETTING_ORDER):
        if setting not in keep:
            axis.set_visible(False)
            continue
        subset = [
            row for row in curves
            if row["setting"] == setting and row["cell"] == "q=1/8"
        ]
        by_ordering: dict[str, dict[int, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for row in subset:
            by_ordering[row["ordering"]][int(row["k"])].append(float(row["recovery"]))
        for ordering, series in by_ordering.items():
            color, style, label = ORDERING_STYLE[ordering]
            ks = sorted(series)
            means = [st.mean(series[k]) for k in ks]
            axis.plot(ks, means, style, color=color, label=label, linewidth=1.8)
        axis.axhline(0.95, color="grey", linewidth=0.8, linestyle="--")
        axis.set_title(setting)
        axis.set_xlabel("retained modes (k)")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("recovered improvement")
    axes[1, 0].set_ylabel("recovered improvement")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    figure.suptitle("Functional rank: recovery vs retained modes (q=1/8, 3 seeds)")
    figure.tight_layout()
    figure.savefig(figures / "fig1_functional_rank_curves.png", dpi=140)
    plt.close(figure)


def figure_importance_mismatch(modes: list[dict], figures: Path) -> None:
    """Singular, activation and predictive importance of the same modes."""
    setting, cell, seed = "ETTh2-720", "q=1/4", "2021"
    rows = [
        row for row in modes
        if row["setting"] == setting and row["cell"] == cell and row["seed"] == seed
    ]
    rows.sort(key=lambda row: int(row["mode_index"]))
    if not rows:
        return
    index = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(index - 0.27, [float(r["weight_energy_share"]) for r in rows],
             width=0.27, label="weight energy $s_i^2$")
    axis.bar(index, [float(r["activation_energy_share"]) for r in rows],
             width=0.27, label="activation energy $s_i^2 E[a_i^2]$")
    contributions = [float(r["fused_mse_contribution"]) for r in rows]
    total = sum(contributions)
    axis.bar(index + 0.27, [c / total if total else 0.0 for c in contributions],
             width=0.27, label="predictive contribution $I_i$")
    axis.set_xlabel("mode index")
    axis.set_ylabel("share")
    axis.set_title(f"{setting} {cell} seed {seed}: three notions of importance")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(figures / "fig2_importance_mismatch.png", dpi=140)
    plt.close(figure)


def figure_negative_share(cells: list[dict], figures: Path) -> None:
    """Share of modes whose marginal contribution is negative, vs nominal rank."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for setting in SETTING_ORDER:
        rows = [row for row in cells if row["setting"] == setting]
        if not rows:
            continue
        by_rank: dict[int, list[float]] = collections.defaultdict(list)
        for row in rows:
            rank = int(row["rank"])
            by_rank[rank].append(int(row["n_negative_contribution"]) / rank)
        ranks = sorted(by_rank)
        axis.plot(ranks, [st.mean(by_rank[r]) for r in ranks], "o-", label=setting)
    axis.set_xscale("log")
    axis.set_xlabel("nominal rank")
    axis.set_ylabel("share of modes with $I_i < 0$")
    axis.set_title("Redundancy grows with nominal rank")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "fig3_negative_contribution_share.png", dpi=140)
    plt.close(figure)


def figure_dense_alignment(alignment: list[dict], figures: Path) -> None:
    """Low-rank subspace overlap with the dense head's two orderings."""
    rows = [
        row for row in alignment
        if row["dense_reference"] in ("dense_singular", "dense_functional")
        and row["input_overlap"] not in ("", None)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, metric, title in (
        (axes[0], "input_overlap", "input subspace"),
        (axes[1], "output_overlap", "output subspace"),
    ):
        for reference, color in (("dense_singular", "#2980b9"), ("dense_functional", "#c0392b")):
            by_dim: dict[int, list[float]] = collections.defaultdict(list)
            for row in rows:
                if row["dense_reference"] == reference:
                    by_dim[int(row["subspace_dim"])].append(float(row[metric]))
            dims = sorted(by_dim)
            axis.plot(dims, [st.mean(by_dim[d]) for d in dims], "o-",
                      color=color, label=f"dense {reference.split('_')[1]} order")
        axis.set_xscale("log")
        axis.set_xlabel("subspace dimension")
        axis.set_ylabel("projection overlap")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Low-rank head vs dense head: which dense subspace was kept?")
    figure.tight_layout()
    figure.savefig(figures / "fig4_dense_alignment.png", dpi=140)
    plt.close(figure)


def figure_mode_kernels(modes_dir: Path, figures: Path) -> None:
    """Input and output kernels of the top modes of one rich cell."""
    setting, cell, seed = "ETTh2-720", "q=1/4", 2021
    path = modes_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
    if not path.is_file():
        return
    payload = np.load(path)
    u = payload["u"]
    vt = payload["vt"]
    contribution = payload["contribution"]
    order = np.argsort(-contribution)[:5]
    horizon, lookback = u.shape[0], vt.shape[1]
    figure, axes = plt.subplots(2, 5, figsize=(18, 6))
    for column, index in enumerate(order):
        axes[0, column].plot(np.arange(-lookback + 1, 1), vt[index], linewidth=1.4)
        axes[0, column].set_title(
            f"mode {index}: input $v_i$\n$I_i$={contribution[index] / contribution.sum():.1%} share"
        )
        axes[0, column].set_xlabel("lag")
        axes[1, column].plot(np.arange(1, horizon + 1), u[:, index], linewidth=1.4, color="#c0392b")
        axes[1, column].set_title("output $u_i$")
        axes[1, column].set_xlabel("horizon step")
        for row in (0, 1):
            axes[row, column].grid(alpha=0.25)
    figure.suptitle(f"{setting} {cell} seed {seed}: canonical read-write modes")
    figure.tight_layout()
    figure.savefig(figures / "fig5_canonical_modes.png", dpi=140)
    plt.close(figure)


def figure_sparsity(sparsity: list[dict], figures: Path) -> None:
    """Reconstruction fidelity against the exact fused-MSE cost."""
    figure, axis = plt.subplots(figsize=(9, 5))
    groups: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in sparsity:
        groups[row["variant"]].append(
            (float(row["reconstruction_r2"]), max(float(row["fused_mse_increase"]), 0.0))
        )
    for name in sorted(groups):
        if name == "dense":
            continue
        values = groups[name]
        axis.scatter(
            st.mean(v[0] for v in values), st.mean(v[1] for v in values),
            s=45, label=name,
        )
        axis.annotate(name, (st.mean(v[0] for v in values), st.mean(v[1] for v in values)),
                      fontsize=7, xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel("mode reconstruction $R^2$")
    axis.set_ylabel("fused MSE increase (per mode)")
    axis.set_yscale("log")
    axis.set_title("Cost of sparsifying a single mode's temporal kernel")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "fig6_sparsity_tradeoff.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    base = repo_root / args.output_dir
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cells = read_csv(base / "functional_rank_cells.csv")
    curves = read_csv(base / "functional_rank_curves.csv")
    modes = read_csv(base / "mode_contributions.csv")
    alignment = read_csv(base / "dense_alignment.csv")
    sparsity = read_csv(base / "mode_sparsity.csv")

    figure_rank_curves(curves, cells, figures)
    figure_importance_mismatch(modes, figures)
    figure_negative_share(cells, figures)
    figure_dense_alignment(alignment, figures)
    figure_mode_kernels(base / "modes", figures)
    figure_sparsity(sparsity, figures)
    print(f"wrote {len(list(figures.glob('*.png')))} figures to {figures}")


if __name__ == "__main__":
    main()
