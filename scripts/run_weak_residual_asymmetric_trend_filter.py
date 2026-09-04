#!/usr/bin/env python3
"""Run frozen trend-filter A under X-A and Only-A residual routing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wangjing/miniconda3/envs/raft/bin/python")
DATASETS = ("ETTh1", "Weather", "ETTm1")
INTERVAL_HOURS = {"ETTh1": 1.0, "Weather": 1.0, "ETTm1": 0.25}


def command(dataset: str, input_mode: str, args: argparse.Namespace) -> list[str]:
    overrides = json.dumps({
        "weak_residual_asymmetric_component": "trend_filter",
        "weak_residual_asymmetric_input_mode": input_mode,
        "weak_residual_trend_filter_kappa": 100.0,
        "weak_residual_trend_filter_sample_interval_hours": INTERVAL_HOURS[dataset],
        "weak_residual_trend_filter_iterations": args.iterations,
    }, separators=(",", ":"))
    result = [
        str(PYTHON), "scripts/search_phaseformer.py",
        "--dataset", dataset, "--horizon", "96", "--stage", "mechanism_full8",
        "--mechanism", "weak_residual_asymmetric_trend", "--period", "24",
        "--lookback", "720", "--max-epochs", str(args.max_epochs),
        "--seed", "2021", "--loss", "huber", "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir, "--overrides", overrides,
    ]
    if args.require_cuda:
        result.append("--require-cuda")
    if args.resume:
        result.append("--resume")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/weak_residual_asymmetric_trend_filter_h96_scratch")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--modes", default="minus_component,component_only")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not datasets or set(datasets) - set(DATASETS):
        parser.error("--datasets must be a non-empty subset of " + ",".join(DATASETS))
    if not modes or set(modes) - {"minus_component", "component_only"}:
        parser.error("--modes must contain minus_component and/or component_only")
    schedule = [(dataset, mode) for mode in modes for dataset in datasets]
    print(f"validation-only trend-filter schedule: {len(schedule)} candidate trainings", flush=True)
    for index, (dataset, mode) in enumerate(schedule, 1):
        print(f"[{index}/{len(schedule)}] {dataset} H96 {mode} trend_filter", flush=True)
        if not args.dry_run:
            subprocess.run(command(dataset, mode, args), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
