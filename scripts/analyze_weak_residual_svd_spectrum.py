#!/usr/bin/env python3
"""Experiment 1: SVD spectrum of the full-rank weak-period-residual weight.

Pure post-hoc analysis of already-trained checkpoints in
research_runs/rank_sweep_2_stage1 -- no new training, no dataset access. For
each of the 7 settings, load the full-rank ("shared" head_type) checkpoint's
weak_period_residual.linear.weight (pred_len x seq_len) and compute its SVD:
singular-value decay, effective rank at 90/95/99% cumulative energy, and the
participation ratio PR = (sum(s))^2 / sum(s^2). Compares the resulting
"elbow" against the rank grid actually tested in the low-rank sweep
(research_runs/rank_sweep_2_stage1/*_results.csv).

See docs/PhaseFormer_lowrank_mechanism_analysis.md for background,
disclosure, and results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _weak_residual_analysis_common import (
    ROOT,
    SETTINGS,
    full_rank_row,
    load_checkpoint_state_dict,
    low_rank_rows,
    setting_label,
)

OUTPUT_ROOT = ROOT / "research_runs/lowrank_mechanism_analysis_v1"


def effective_rank_at_energy(singular_values: np.ndarray, fraction: float) -> int:
    energy = singular_values**2
    cumulative = np.cumsum(energy) / energy.sum()
    return int(np.searchsorted(cumulative, fraction) + 1)


def participation_ratio(singular_values: np.ndarray) -> float:
    return float(singular_values.sum() ** 2 / (singular_values**2).sum())


def analyze_setting(dataset: str, horizon: int) -> dict:
    row = full_rank_row(dataset, horizon)
    state_dict = load_checkpoint_state_dict(row)
    weight = state_dict["weak_period_residual.linear.weight"].double().numpy()
    singular_values = np.linalg.svd(weight, compute_uv=False)
    tested_ranks = [r["rank"] for r in low_rank_rows(dataset, horizon)]
    return {
        "dataset": dataset,
        "horizon": horizon,
        "weight_shape": weight.shape,
        "singular_values": singular_values,
        "tested_ranks": tested_ranks,
        "rank_90": effective_rank_at_energy(singular_values, 0.90),
        "rank_95": effective_rank_at_energy(singular_values, 0.95),
        "rank_99": effective_rank_at_energy(singular_values, 0.99),
        "participation_ratio": participation_ratio(singular_values),
        "top_singular_value": float(singular_values[0]),
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }


def write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "setting", "dataset", "horizon", "weight_rows", "weight_cols",
        "rank_90", "rank_95", "rank_99", "participation_ratio",
        "top_singular_value", "condition_number", "tested_ranks",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "setting": setting_label(result["dataset"], result["horizon"]),
                    "dataset": result["dataset"],
                    "horizon": result["horizon"],
                    "weight_rows": result["weight_shape"][0],
                    "weight_cols": result["weight_shape"][1],
                    "rank_90": result["rank_90"],
                    "rank_95": result["rank_95"],
                    "rank_99": result["rank_99"],
                    "participation_ratio": round(result["participation_ratio"], 4),
                    "top_singular_value": round(result["top_singular_value"], 6),
                    "condition_number": round(result["condition_number"], 2),
                    "tested_ranks": ";".join(str(r) for r in result["tested_ranks"]),
                }
            )


def plot_spectra(results: list[dict], figures_dir: Path) -> None:
    figure, axes = plt.subplots(len(results), 1, figsize=(7, 3 * len(results)))
    for axis, result in zip(axes, results):
        singular_values = result["singular_values"]
        axis.semilogy(np.arange(1, len(singular_values) + 1), singular_values, color="C0")
        for rank in result["tested_ranks"]:
            axis.axvline(rank, color="grey", linewidth=0.6, linestyle=":")
        axis.axvline(result["rank_95"], color="red", linewidth=1.0, linestyle="--", label="rank@95% energy")
        axis.set_title(f"{setting_label(result['dataset'], result['horizon'])}: singular value spectrum")
        axis.set_xlabel("index")
        axis.set_ylabel("singular value")
        axis.legend(fontsize=8)
        axis.grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    out_path = figures_dir / "svd_spectrum_grid.png"
    figure.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = [analyze_setting(setting["dataset"], setting["horizon"]) for setting in SETTINGS]
    csv_path = OUTPUT_ROOT / "svd_spectrum_summary.csv"
    write_csv(results, csv_path)
    print(f"wrote {csv_path}")
    plot_spectra(results, figures_dir)


if __name__ == "__main__":
    main()
