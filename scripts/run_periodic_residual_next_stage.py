#!/usr/bin/env python3
"""Run the pre-registered periodic-complementary residual experiment matrix."""

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)
MODES = (
    "original",
    "gold_combo_reliability_s2",
    "rcrf_pe_lff",
    "rcrf_icpt_none",
    "rcrf_icpt_horizon_none",
    "rcrf_phase_error_memory",
    "rcrf_dual_reliability_lff",
    "rcrf_multiperiod",
)
DEFAULT_OUTPUT = "research_runs/periodic_residual_next_stage_v1"
GOLDEN = {
    ("ETTh1", 96): (0.359, 0.382),
    ("ETTh1", 192): (0.397, 0.404),
    ("ETTh2", 96): (0.275, 0.338),
    ("ETTh2", 192): (0.341, 0.376),
    ("ETTm1", 96): (0.293, 0.344),
    ("ETTm1", 192): (0.323, 0.361),
    ("ETTm2", 96): (0.163, 0.256),
    ("ETTm2", 192): (0.219, 0.293),
    ("Weather", 96): (0.148, 0.195),
    ("Weather", 192): (0.193, 0.237),
    ("Electricity", 96): (0.129, 0.221),
    ("Electricity", 192): (0.148, 0.238),
}
INCUMBENT = "rcrf_pe_lff"


def _parse_csv(value, allowed, cast=str):
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported values: {unknown}")
    if not values:
        raise ValueError("selection must not be empty")
    return values


def build_commands(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    modes = _parse_csv(args.modes, MODES)
    commands = []
    for dataset in datasets:
        for horizon in horizons:
            for seed in seeds:
                command = [
                    sys.executable,
                    "scripts/benchmark_phaseformer_suite.py",
                    "--datasets", dataset,
                    "--horizons", str(horizon),
                    "--modes", ",".join(modes),
                    "--lookback", "720",
                    "--seed", str(seed),
                    "--num-workers", str(args.num_workers),
                    "--bad-case-limit", "0",
                    "--bad-case-batches", "0",
                    "--output-dir", args.output_dir,
                    "--run-prefix", (
                        f"periodic_residual_next_{dataset.lower()}_h{horizon}"
                    ),
                ]
                if args.resume:
                    command.append("--resume")
                if args.progress:
                    command.append("--progress")
                commands.append(command)
    return commands, len(datasets) * len(horizons) * len(seeds) * len(modes)


def summarize(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    modes = _parse_csv(args.modes, MODES)
    expected = {
        (dataset, horizon, mode, seed)
        for dataset in datasets
        for horizon in horizons
        for mode in modes
        for seed in seeds
    }
    rows = {}
    for path in sorted((REPO_ROOT / args.output_dir).glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values:
            continue
        row = values[0]
        key = (
            row.get("dataset"),
            int(row.get("horizon", -1)),
            row.get("mode"),
            int(row.get("seed", -1)),
        )
        if key in expected:
            if key in rows:
                raise RuntimeError(f"duplicate formal result for {key}")
            rows[key] = row
    missing = sorted(expected - set(rows))
    if missing:
        print(f"formal matrix incomplete: {len(missing)} missing", file=sys.stderr)
        for key in missing[:20]:
            print(f"  {key}", file=sys.stderr)
        return 2

    summary = []
    keyed_summary = {}
    for dataset in datasets:
        for horizon in horizons:
            gold_mse, gold_mae = GOLDEN[(dataset, horizon)]
            for mode in modes:
                group = [rows[(dataset, horizon, mode, seed)] for seed in seeds]
                mse_values = [float(row["test_mse"]) for row in group]
                mae_values = [float(row["test_mae"]) for row in group]
                mse_mean = statistics.mean(mse_values)
                mae_mean = statistics.mean(mae_values)
                mse_std = statistics.stdev(mse_values) if len(group) > 1 else 0.0
                mae_std = statistics.stdev(mae_values) if len(group) > 1 else 0.0
                item = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "mode": mode,
                    "mse_mean": mse_mean,
                    "mse_std": mse_std,
                    "mae_mean": mae_mean,
                    "mae_std": mae_std,
                    "golden_mse": gold_mse,
                    "golden_mae": gold_mae,
                    "stable_below_golden": (
                        all(value < gold_mse for value in mse_values)
                        and all(value < gold_mae for value in mae_values)
                        and mse_mean + mse_std < gold_mse
                        and mae_mean + mae_std < gold_mae
                    ),
                    "elapsed_sec": sum(float(row["elapsed_sec"]) for row in group),
                }
                summary.append(item)
                keyed_summary[(dataset, horizon, mode)] = item

    for item in summary:
        baseline = keyed_summary[(item["dataset"], item["horizon"], INCUMBENT)]
        item["mse_ratio_vs_a2"] = item["mse_mean"] / baseline["mse_mean"]
        item["mae_ratio_vs_a2"] = item["mae_mean"] / baseline["mae_mean"]

    output_root = REPO_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "formal_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    decisions = {}
    for mode in modes:
        model_rows = [item for item in summary if item["mode"] == mode]
        ratios = [
            value
            for item in model_rows
            for value in (item["mse_ratio_vs_a2"], item["mae_ratio_vs_a2"])
        ]
        both_improve = sum(
            item["mse_ratio_vs_a2"] < 1.0 and item["mae_ratio_vs_a2"] < 1.0
            for item in model_rows
        )
        macro_ratio = statistics.mean(ratios)
        worst_ratio = max(ratios)
        decisions[mode] = {
            "macro_ratio_vs_a2": macro_ratio,
            "both_metric_improve_settings": both_improve,
            "worst_ratio_vs_a2": worst_ratio,
            "stable_below_golden_settings": sum(
                bool(item["stable_below_golden"]) for item in model_rows
            ),
            "eligible_to_replace_a2": (
                mode != INCUMBENT
                and both_improve >= 8
                and macro_ratio < 0.998
                and worst_ratio <= 1.005
            ),
        }
    decision_path = output_root / "decision_summary.json"
    decision_path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {decision_path}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("dry-run", "full", "summarize"), required=True
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.stage == "summarize":
        return summarize(args)

    commands, run_count = build_commands(args)
    print(f"commands={len(commands)} model_runs={run_count}")
    for command in commands:
        print(shlex.join(command))
    if args.stage == "dry-run":
        return 0

    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
