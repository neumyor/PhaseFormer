#!/usr/bin/env python3
"""Analyze the validation-only pooled low-rank Phase A matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rows(root: Path, datasets: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            path = root / f"phase_a_{dataset}_h96_s{seed}_validation.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(newline="") as handle:
                current = list(csv.DictReader(handle))
            if len(current) != 11:
                raise RuntimeError(f"{path}: expected 11 rows, found {len(current)}")
            for row in current:
                row["dataset"] = dataset
                row["seed"] = seed
                row["val_mse"] = float(row["val_mse"])
                row["val_mae"] = float(row["val_mae"])
                row["pool_factor"] = (
                    int(row["pool_factor"]) if row["pool_factor"] else 0
                )
                row["relative_rank"] = (
                    float(row["relative_rank"]) if row["relative_rank"] else 0.0
                )
                rows.append(row)
    return rows


def paired_deltas(rows: list[dict]) -> list[dict]:
    grouped = {(r["dataset"], r["seed"], r["config_id"]): r for r in rows}
    output = []
    for row in rows:
        if row["config_id"] in {"phase_only", "direct_nlinear"}:
            continue
        control = grouped[(row["dataset"], row["seed"], "direct_nlinear")]
        output.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "config_id": row["config_id"],
                "pool_factor": row["pool_factor"],
                "relative_rank": row["relative_rank"],
                "rank": int(row["rank"]),
                "delta_mse": row["val_mse"] - control["val_mse"],
                "delta_mae": row["val_mae"] - control["val_mae"],
            }
        )
    return output


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(20000, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def summarize(values: list[float], rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_interval(array, rng)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def fixed_effects(rows: list[dict], metric: str) -> dict:
    """Fit metric ~ intercept + dataset + log2(pool) + log2(q) + interaction."""
    design = []
    values = []
    for row in rows:
        design.append(
            [
                1.0,
                1.0 if row["dataset"] == "ETTm1" else 0.0,
                math.log2(row["pool_factor"]),
                math.log2(row["relative_rank"]),
                math.log2(row["pool_factor"]) * math.log2(row["relative_rank"]),
            ]
        )
        values.append(row[metric])
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(values, dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    names = ["intercept", "dataset_ETTm1", "log2_pool", "log2_relative_rank", "interaction"]
    return {name: float(value) for name, value in zip(names, coefficients)}


def render_report(rows: list[dict], output: Path) -> dict:
    rng = np.random.default_rng(20260910)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["config_id"])].append(row)

    summary = {
        "protocol": "Phase A validation-only analysis; no test metrics used",
        "settings": sorted({(r["dataset"], r["seed"]) for r in rows}),
        "paired_control": "direct_nlinear",
        "config_summaries": {},
        "fixed_effects": {
            "delta_mse": fixed_effects(rows, "delta_mse"),
            "delta_mae": fixed_effects(rows, "delta_mae"),
        },
    }
    for key, values in sorted(grouped.items()):
        dataset, config_id = key
        summary["config_summaries"][f"{dataset}:{config_id}"] = {
            "dataset": dataset,
            "config_id": config_id,
            "pool_factor": values[0]["pool_factor"],
            "relative_rank": values[0]["relative_rank"],
            "rank": values[0]["rank"],
            "delta_mse": summarize([v["delta_mse"] for v in values], rng),
            "delta_mae": summarize([v["delta_mae"] for v in values], rng),
        }

    lines = [
        "# Phase A Validation Analysis",
        "",
        "This report uses only Phase A validation metrics. No test loader or test metric was read.",
        "",
        "## Paired response",
        "",
        "Each pooled-low-rank cell is compared with the same-seed `direct_nlinear` control.",
        "Positive delta means the candidate has higher validation error.",
        "",
        "| Dataset | Config | MSE delta mean | MSE std | MSE bootstrap 95% | MAE delta mean | MAE std | MAE bootstrap 95% |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for key, item in sorted(summary["config_summaries"].items()):
        mse = item["delta_mse"]
        mae = item["delta_mae"]
        lines.append(
            f"| {item['dataset']} | {item['config_id']} | "
            f"{mse['mean']:.6f} | {mse['std']:.6f} | "
            f"[{mse['bootstrap_95_low']:.6f}, {mse['bootstrap_95_high']:.6f}] | "
            f"{mae['mean']:.6f} | {mae['std']:.6f} | "
            f"[{mae['bootstrap_95_low']:.6f}, {mae['bootstrap_95_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Fixed effects",
            "",
            "The coefficients come from a descriptive least-squares model with dataset stratum, "
            "`log2(pool)`, `log2(relative rank)`, and their interaction.",
            "",
            "```json",
            json.dumps(summary["fixed_effects"], indent=2),
            "```",
            "",
            "## Decision boundary",
            "",
            "This analysis records the preregistered evidence needed before Phase B. "
            "It does not select a final model and does not authorize test evaluation.",
            "",
        ]
    )
    output.write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="research_runs/joint_pooled_lowrank_phase_a_scratch",
    )
    parser.add_argument(
        "--output",
        default="research_runs/joint_pooled_lowrank_phase_a_scratch/phase_a_validation_analysis.md",
    )
    parser.add_argument("--datasets", default="ETTh1,ETTm1")
    parser.add_argument("--seeds", default="2021,2022,2023")
    args = parser.parse_args()
    datasets = [item for item in args.datasets.split(",") if item]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    rows = load_rows(ROOT / args.input_root, datasets, seeds)
    summary = render_report(paired_deltas(rows), ROOT / args.output)
    json_path = ROOT / args.output
    json_path = json_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "report": str(ROOT / args.output)}))


if __name__ == "__main__":
    main()
