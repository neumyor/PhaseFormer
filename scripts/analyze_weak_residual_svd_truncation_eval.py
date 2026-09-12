#!/usr/bin/env python3
"""Experiment 2: SVD-truncate the full-rank weight (no retraining) and
evaluate real test-set MSE/MAE, compared against the actually-trained
low-rank models and the full-rank baseline.

For each setting and each rank actually tested in the low-rank sweep,
truncate the full-rank checkpoint's weak_period_residual.linear.weight via
SVD to that rank, inject it into a freshly constructed PhaseFormer, and
evaluate on the real test set (no gradient steps). If this closely matches
the trained low-rank model's own test performance, the learned full-rank
mapping was already close to low-rank -- compression did not have much to
lose.

See docs/PhaseFormer_lowrank_mechanism_analysis.md for background,
disclosure, and results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _weak_residual_analysis_common import (
    ROOT,
    SETTINGS,
    build_model_and_loader,
    evaluate_model,
    full_rank_row,
    load_checkpoint_state_dict,
    low_rank_rows,
    pick_device,
    setting_label,
)

OUTPUT_ROOT = ROOT / "research_runs/lowrank_mechanism_analysis_v1"


def svd_truncate(weight: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    truncated = u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]
    return truncated.to(weight.dtype)


def evaluate_setting(dataset: str, horizon: int, device: torch.device) -> list[dict]:
    full_row = full_rank_row(dataset, horizon)
    full_state_dict = load_checkpoint_state_dict(full_row)
    full_weight = full_state_dict["weak_period_residual.linear.weight"]

    model, loader = build_model_and_loader(full_row)
    model.load_state_dict(full_state_dict, strict=True)
    full_rank_metrics = evaluate_model(model, loader, horizon, device)

    rows = []
    for low_row in low_rank_rows(dataset, horizon):
        rank = low_row["rank"]
        model, loader = build_model_and_loader(full_row)
        model.load_state_dict(full_state_dict, strict=True)
        with torch.no_grad():
            model.weak_period_residual.linear.weight.copy_(svd_truncate(full_weight, rank))
        truncated_metrics = evaluate_model(model, loader, horizon, device)
        trained_low_rank_metrics = {
            "mse": float(low_row["metrics"]["test_mse"]),
            "mae": float(low_row["metrics"]["test_mae"]),
        }
        rows.append(
            {
                "setting": setting_label(dataset, horizon),
                "dataset": dataset,
                "horizon": horizon,
                "rank": rank,
                "svd_truncated_mse": truncated_metrics["mse"],
                "svd_truncated_mae": truncated_metrics["mae"],
                "trained_lowrank_mse": trained_low_rank_metrics["mse"],
                "trained_lowrank_mae": trained_low_rank_metrics["mae"],
                "full_rank_mse": full_rank_metrics["mse"],
                "full_rank_mae": full_rank_metrics["mae"],
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(all_rows: list[dict], figures_dir: Path) -> None:
    settings = sorted({row["setting"] for row in all_rows})
    figure, axes = plt.subplots(len(settings), 2, figsize=(10, 3 * len(settings)))
    for row_index, setting in enumerate(settings):
        rows = sorted((r for r in all_rows if r["setting"] == setting), key=lambda r: r["rank"])
        ranks = [r["rank"] for r in rows]
        for col, metric in enumerate(("mse", "mae")):
            axis = axes[row_index, col]
            axis.plot(ranks, [r[f"svd_truncated_{metric}"] for r in rows], marker="o", label="SVD-truncated (no retrain)")
            axis.plot(ranks, [r[f"trained_lowrank_{metric}"] for r in rows], marker="s", label="trained low-rank")
            axis.axhline(rows[0][f"full_rank_{metric}"], color="grey", linestyle="--", linewidth=0.8, label="full-rank baseline")
            axis.set_title(f"{setting} — {metric.upper()}", fontsize=9)
            axis.set_xlabel("rank")
            axis.grid(True, linewidth=0.3, alpha=0.5)
            if row_index == 0 and col == 0:
                axis.legend(fontsize=7)
    figure.tight_layout()
    out_path = figures_dir / "svd_truncation_vs_trained_lowrank.png"
    figure.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    all_rows: list[dict] = []
    for setting in SETTINGS:
        all_rows.extend(evaluate_setting(setting["dataset"], setting["horizon"], device))
    csv_path = OUTPUT_ROOT / "svd_truncation_eval.csv"
    write_csv(all_rows, csv_path)
    print(f"wrote {csv_path}")
    plot_comparison(all_rows, figures_dir)


if __name__ == "__main__":
    main()
