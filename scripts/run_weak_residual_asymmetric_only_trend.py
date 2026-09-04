#!/usr/bin/env python3
"""Train the validation-only complementary probe: NLinear sees only A."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wangjing/miniconda3/envs/raft/bin/python")
DATASETS = ("ETTh1", "Weather", "ETTm1")
COMPONENTS = (
    "cycle_levels", "recent_linear", "global_linear", "smooth_local",
    "smooth_multiscale",
)


def command(dataset, component, args):
    overrides = (
        '{"weak_residual_asymmetric_component":"' + component +
        '","weak_residual_asymmetric_input_mode":"component_only"}'
    )
    result = [
        str(PYTHON), "scripts/search_phaseformer.py",
        "--dataset", dataset, "--horizon", "96", "--stage", "mechanism_full8",
        "--mechanism", "weak_residual_asymmetric_trend", "--period", "24",
        "--lookback", "720", "--max-epochs", str(args.max_epochs), "--seed", "2021",
        "--loss", "huber", "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir, "--overrides", overrides,
    ]
    if args.require_cuda:
        result.append("--require-cuda")
    if args.resume:
        result.append("--resume")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--component", choices=("all",) + COMPONENTS, default="all")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if not datasets or set(datasets) - set(DATASETS):
        parser.error("--datasets must be a non-empty subset of " + ",".join(DATASETS))
    components = COMPONENTS if args.component == "all" else (args.component,)
    schedule = [(dataset, component) for component in components for dataset in datasets]
    print(f"validation-only component-only schedule: {len(schedule)} runs", flush=True)
    for index, (dataset, component) in enumerate(schedule, 1):
        print(f"[{index}/{len(schedule)}] {dataset} H96 component_only {component}", flush=True)
        if not args.dry_run:
            subprocess.run(command(dataset, component, args), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
