#!/usr/bin/env python3
"""Launch/resume the validation-only weak-residual trend-component discovery grid."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# `raft` is the available GPU environment on this host (torch + Lightning).
PYTHON = Path("/home/wangjing/miniconda3/envs/raft/bin/python")
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192)
COMPONENTS = (
    "cycle_levels",
    "recent_linear",
    "global_linear",
    "smooth_local",
    "smooth_multiscale",
)


def command(dataset, horizon, component, args):
    mechanism = "weak_residual" if component == "none" else "weak_residual_asymmetric_trend"
    overrides = "{}" if component == "none" else (
        '{"weak_residual_asymmetric_component":"' + component + '"}'
    )
    result = [
        str(PYTHON), "scripts/search_phaseformer.py",
        "--dataset", dataset, "--horizon", str(horizon),
        "--stage", "mechanism_full8", "--mechanism", mechanism,
        "--period", "24", "--lookback", "720", "--max-epochs", str(args.max_epochs),
        "--seed", "2021", "--loss", "huber", "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir, "--overrides", overrides,
    ]
    if args.require_cuda:
        result.append("--require-cuda")
    if args.resume:
        result.append("--resume")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/weak_residual_asymmetric_trend_discovery")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--component", choices=("all",) + COMPONENTS, default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    components = COMPONENTS if args.component == "all" else (args.component,)
    schedule = [(dataset, horizon, "none") for dataset in DATASETS for horizon in HORIZONS]
    schedule += [
        (dataset, horizon, component)
        for component in components
        for dataset in DATASETS
        for horizon in HORIZONS
    ]
    print(f"validation-only discovery schedule: {len(schedule)} runs")
    for index, (dataset, horizon, component) in enumerate(schedule, start=1):
        cmd = command(dataset, horizon, component, args)
        print(f"[{index}/{len(schedule)}] {dataset} H{horizon} {component}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
