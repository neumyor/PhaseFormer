#!/usr/bin/env python3
"""Audit and summarize the conditioned three-seed low-rank sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = [
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
]
SEEDS = (2021, 2022, 2023)
CONFIGS = ("direct", "q=0.25", "q=0.125", "q=0.0625", "q=0.03125")
FROZEN = {
    ("ETTh2", 96): (0.5, 0.001),
    ("ETTh2", 720): (0.5, 0.001),
    ("ETTm2", 96): (0.5, 0.0003),
    ("ETTm2", 192): (0.2, 0.001),
    ("Weather", 96): (0.2, 0.0003),
    ("Weather", 192): (0.5, 0.001),
    ("Electricity", 336): (0.5, 0.001),
}
RANKS = {
    96: {"q=0.25": 24, "q=0.125": 12, "q=0.0625": 6, "q=0.03125": 3},
    192: {"q=0.25": 48, "q=0.125": 24, "q=0.0625": 12, "q=0.03125": 6},
    336: {"q=0.25": 84, "q=0.125": 42, "q=0.0625": 21, "q=0.03125": 10},
    720: {"q=0.25": 180, "q=0.125": 90, "q=0.0625": 45, "q=0.03125": 22},
}
METRICS = ("val_mse", "val_mae", "test_mse", "test_mae")


def finite_metrics(row: dict) -> dict[str, float]:
    values = {metric: float(row[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite metric")
    return values


def config_key(horizon: int, hp: dict) -> str | None:
    head_type = hp.get("weak_period_residual_head_type", "shared")
    if head_type == "shared":
        return "direct"
    if (
        head_type != "pooled_lowrank"
        or int(hp.get("weak_period_residual_pool_factor", -1)) != 1
        or float(hp.get("weak_period_residual_smooth_ratio", 0.0)) != 0.0
    ):
        return None
    rank = int(hp.get("weak_period_residual_rank", -1))
    for config, expected_rank in RANKS[horizon].items():
        if rank == expected_rank:
            return config
    return None


def load_seed_2021(root: Path) -> dict[tuple, dict]:
    rows = {}
    for dataset, horizon in SETTINGS:
        path = root / f"phase_a_{dataset}_h{horizon}_s2021_results.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                config_id = row["config_id"]
                if config_id == "direct_nlinear":
                    config = "direct"
                elif config_id.startswith("pool1_q"):
                    rank = int(row["rank"])
                    config = next(
                        (
                            candidate
                            for candidate, expected_rank in RANKS[horizon].items()
                            if rank == expected_rank
                        ),
                        "",
                    )
                else:
                    continue
                if config not in CONFIGS:
                    continue
                key = (dataset, horizon, 2021, config)
                if key in rows:
                    raise RuntimeError(f"duplicate seed-2021 cell: {key}")
                rows[key] = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": 2021,
                    "config": config,
                    "rank": "" if config == "direct" else int(row["rank"]),
                    **finite_metrics(row),
                    "run_id": row["run_id"],
                    "run_dir": row["run_dir"],
                    "source_root": str(root),
                }
    return rows


def load_run_roots(roots: list[Path]) -> dict[tuple, list[dict]]:
    rows: dict[tuple, list[dict]] = defaultdict(list)
    valid_settings = set(SETTINGS)
    for root in roots:
        for config_path in sorted((root / "runs").glob("*/config.json")):
            metrics_path = config_path.with_name("metrics.csv")
            if not metrics_path.is_file():
                continue
            try:
                payload = json.loads(config_path.read_text())
                dataset = payload["dataset"]
                horizon = int(payload["horizon"])
                seed = int(payload["seed"])
                hp = payload["hyperparams"]
                metrics = next(csv.DictReader(metrics_path.open(newline="")))
                parsed_metrics = finite_metrics(metrics)
            except (
                OSError,
                KeyError,
                StopIteration,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if (dataset, horizon) not in valid_settings or seed not in (2022, 2023):
                continue
            config = config_key(horizon, hp)
            if config not in CONFIGS:
                continue
            gate, learning_rate = FROZEN[(dataset, horizon)]
            if (
                abs(float(hp["weak_period_residual_gate_init"]) - gate) > 1e-9
                or abs(float(hp["learning_rate"]) - learning_rate) > 1e-9
            ):
                continue
            key = (dataset, horizon, seed, config)
            rows[key].append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "config": config,
                    "rank": "" if config == "direct" else RANKS[horizon][config],
                    **parsed_metrics,
                    "run_id": metrics.get("run_id", config_path.parent.name),
                    "run_dir": str(config_path.parent.relative_to(ROOT)),
                    "source_root": str(root),
                }
            )
    return rows


def audit(
    seed_2021_root: Path,
    run_roots: list[Path],
) -> list[dict]:
    seed_2021 = load_seed_2021(seed_2021_root)
    later = load_run_roots(run_roots)
    output = []
    errors = []
    for dataset, horizon in SETTINGS:
        for seed in SEEDS:
            for config in CONFIGS:
                key = (dataset, horizon, seed, config)
                candidates = [seed_2021[key]] if seed == 2021 and key in seed_2021 else later.get(key, [])
                if len(candidates) != 1:
                    errors.append(f"{key}: expected 1 complete run, found {len(candidates)}")
                    continue
                output.append(candidates[0])
    if errors:
        raise RuntimeError("rank-sweep audit failed:\n" + "\n".join(errors))
    if len(output) != len(SETTINGS) * len(SEEDS) * len(CONFIGS):
        raise RuntimeError(f"expected 105 three-seed rows, found {len(output)}")
    return output


def sample_summary(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    indexed = {
        (row["dataset"], row["horizon"], row["seed"], row["config"]): row
        for row in rows
    }
    paired = []
    for row in rows:
        direct = indexed[
            (row["dataset"], row["horizon"], row["seed"], "direct")
        ]
        paired.append(
            {
                **row,
                "delta_test_mse_pct": (
                    (direct["test_mse"] - row["test_mse"])
                    / direct["test_mse"]
                    * 100.0
                ),
                "delta_test_mae_pct": (
                    (direct["test_mae"] - row["test_mae"])
                    / direct["test_mae"]
                    * 100.0
                ),
            }
        )

    summary = []
    for dataset, horizon in SETTINGS:
        for config in CONFIGS:
            current = [
                row
                for row in paired
                if row["dataset"] == dataset
                and row["horizon"] == horizon
                and row["config"] == config
            ]
            item = {
                "dataset": dataset,
                "horizon": horizon,
                "config": config,
                "rank": current[0]["rank"],
            }
            for metric in ("test_mse", "test_mae", "delta_test_mse_pct", "delta_test_mae_pct"):
                mean, std = sample_summary([row[metric] for row in current])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            item["both_better_seeds"] = sum(
                row["delta_test_mse_pct"] > 0.0
                and row["delta_test_mae_pct"] > 0.0
                for row in current
            )
            summary.append(item)

    diagnostics = []
    paired_index = {
        (row["dataset"], row["horizon"], row["seed"], row["config"]): row
        for row in paired
    }
    summary_index = {
        (row["dataset"], row["horizon"], row["config"]): row for row in summary
    }
    for dataset, horizon in SETTINGS:
        direct_mse = statistics.mean(
            indexed[(dataset, horizon, seed, "direct")]["test_mse"]
            for seed in SEEDS
        )
        direct_mae = statistics.mean(
            indexed[(dataset, horizon, seed, "direct")]["test_mae"]
            for seed in SEEDS
        )
        compressed = [
            paired_index[(dataset, horizon, seed, config)]
            for seed in SEEDS
            for config in CONFIGS[1:]
        ]
        config_means = {
            config: summary_index[(dataset, horizon, config)]
            for config in CONFIGS[1:]
        }
        best_mse = min(
            config_means,
            key=lambda config: config_means[config]["test_mse_mean"],
        )
        best_mae = min(
            config_means,
            key=lambda config: config_means[config]["test_mae_mean"],
        )
        mse_means = [
            config_means[config]["test_mse_mean"] for config in CONFIGS[1:]
        ]
        mae_means = [
            config_means[config]["test_mae_mean"] for config in CONFIGS[1:]
        ]
        diagnostics.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "best_mean_mse_config": best_mse,
                "best_mean_mae_config": best_mae,
                "same_best_config": best_mse == best_mae,
                "compressed_mse_range_pct_of_direct": (
                    (max(mse_means) - min(mse_means)) / direct_mse * 100.0
                ),
                "compressed_mae_range_pct_of_direct": (
                    (max(mae_means) - min(mae_means)) / direct_mae * 100.0
                ),
                "compressed_both_better_cells": sum(
                    row["delta_test_mse_pct"] > 0.0
                    and row["delta_test_mae_pct"] > 0.0
                    for row in compressed
                ),
                "compressed_cells": len(compressed),
            }
        )

    aggregate = []
    for config in CONFIGS[1:]:
        current = [row for row in paired if row["config"] == config]
        setting_rows = [row for row in summary if row["config"] == config]
        aggregate.append(
            {
                "config": config,
                "paired_cells": len(current),
                "both_better_cells": sum(
                    row["delta_test_mse_pct"] > 0.0
                    and row["delta_test_mae_pct"] > 0.0
                    for row in current
                ),
                "both_better_setting_means": sum(
                    row["delta_test_mse_pct_mean"] > 0.0
                    and row["delta_test_mae_pct_mean"] > 0.0
                    for row in setting_rows
                ),
                "macro_delta_test_mse_pct": statistics.mean(
                    row["delta_test_mse_pct"] for row in current
                ),
                "macro_delta_test_mae_pct": statistics.mean(
                    row["delta_test_mae_pct"] for row in current
                ),
            }
        )
    return paired, summary, diagnostics, aggregate


def render_markdown(
    summary: list[dict],
    diagnostics: list[dict],
    aggregate: list[dict],
) -> str:
    indexed = {
        (row["dataset"], row["horizon"], row["config"]): row
        for row in summary
    }
    lines = [
        "# Conditioned Rank Sweep: Three-Seed Summary",
        "",
        "Mean +/- sample standard deviation over seeds 2021, 2022, and 2023.",
        "Positive paired delta means lower error than same-seed direct NLinear.",
        "",
        "| Setting | direct | q=1/4 | q=1/8 | q=1/16 | q=1/32 |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, horizon in SETTINGS:
        cells = []
        for config in CONFIGS:
            row = indexed[(dataset, horizon, config)]
            cells.append(
                f"{row['test_mse_mean']:.6f}+/-{row['test_mse_std']:.6f} / "
                f"{row['test_mae_mean']:.6f}+/-{row['test_mae_std']:.6f}"
            )
        lines.append(f"| {dataset}-{horizon} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| Setting | q=1/4 delta% | q=1/8 delta% | q=1/16 delta% | q=1/32 delta% |",
            "|---|---|---|---|---|",
        ]
    )
    for dataset, horizon in SETTINGS:
        cells = []
        for config in CONFIGS[1:]:
            row = indexed[(dataset, horizon, config)]
            cells.append(
                f"{row['delta_test_mse_pct_mean']:+.3f}+/-{row['delta_test_mse_pct_std']:.3f} / "
                f"{row['delta_test_mae_pct_mean']:+.3f}+/-{row['delta_test_mae_pct_std']:.3f} "
                f"({row['both_better_seeds']}/3)"
            )
        lines.append(f"| {dataset}-{horizon} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Parentheses report seeds where both test metrics beat same-seed direct NLinear.",
            "",
            "| Config | Macro delta MSE% | Macro delta MAE% | Both-better cells | Both-better setting means |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['config']} | {row['macro_delta_test_mse_pct']:+.3f} | "
            f"{row['macro_delta_test_mae_pct']:+.3f} | "
            f"{row['both_better_cells']}/{row['paired_cells']} | "
            f"{row['both_better_setting_means']}/7 |"
        )
    lines.extend(
        [
            "",
            "| Setting | Best mean MSE q | Best mean MAE q | Same q | MSE range% | MAE range% | Both-better cells |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics:
        lines.append(
            f"| {row['dataset']}-{row['horizon']} | "
            f"{row['best_mean_mse_config']} | {row['best_mean_mae_config']} | "
            f"{'yes' if row['same_best_config'] else 'no'} | "
            f"{row['compressed_mse_range_pct_of_direct']:.3f} | "
            f"{row['compressed_mae_range_pct_of_direct']:.3f} | "
            f"{row['compressed_both_better_cells']}/{row['compressed_cells']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-2021-root",
        default="research_runs/rank_sweep_2_stage1",
    )
    parser.add_argument(
        "--run-roots",
        default=(
            "research_runs/rank_sweep_2_multiseed_stage1_20260914_v4,"
            "research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="research_runs/rank_sweep_2_multiseed_stage1_20260914_summary",
    )
    args = parser.parse_args()
    run_roots = [ROOT / item for item in args.run_roots.split(",") if item]
    output_root = ROOT / args.output_root
    rows = audit(ROOT / args.seed_2021_root, run_roots)
    paired, summary, diagnostics, aggregate = summarize(rows)
    write_csv(output_root / "audited_results.csv", paired)
    write_csv(output_root / "three_seed_summary.csv", summary)
    write_csv(output_root / "three_seed_setting_diagnostics.csv", diagnostics)
    write_csv(output_root / "three_seed_aggregate.csv", aggregate)
    (output_root / "three_seed_summary.md").write_text(
        render_markdown(summary, diagnostics, aggregate)
    )
    print(
        json.dumps(
            {
                "audited_rows": len(rows),
                "settings": len(SETTINGS),
                "seeds": list(SEEDS),
                "configs": list(CONFIGS),
                "output_root": str(output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
