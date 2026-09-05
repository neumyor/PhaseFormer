#!/usr/bin/env python3
"""Run the frozen trend-filter, causal-EMA and Holt asymmetric comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wangjing/miniconda3/envs/raft/bin/python")
DATASETS = ("ETTh1", "Weather", "ETTm1")
COMPONENTS = ("trend_filter", "causal_ema", "holt_local_linear")
INTERVAL_HOURS = {"ETTh1": 1.0, "Weather": 1.0, "ETTm1": 0.25}
# ETTm1's strong approximately-96-step peak needs this many fixed CP updates
# before A6 meets the frozen <=0.10 periodic-leakage criterion.
TREND_FILTER_ITERATIONS = {"ETTh1": 256, "Weather": 256, "ETTm1": 4096}
CAUSAL_PARAMS = {
    # ETTh1 parameter-debug follow-up: use the slower filters to suppress the
    # dominant 24-step harmonic.
    "ETTh1": {"alpha": 0.006, "beta": 0.0015},
    # Weather has no narrow short-period peak; this is the predeclared
    # conservative hourly setting, not a prediction-selected parameter.
    "Weather": {"alpha": 0.024, "beta": 0.006},
    "ETTm1": {"alpha": 0.006, "beta": 0.0015},
}


def command(dataset: str, component: str, input_mode: str, args: argparse.Namespace) -> list[str]:
    params = CAUSAL_PARAMS[dataset]
    overrides = {
        "weak_residual_asymmetric_component": component,
        "weak_residual_asymmetric_input_mode": input_mode,
        "weak_residual_trend_filter_kappa": 100.0,
        "weak_residual_trend_filter_sample_interval_hours": INTERVAL_HOURS[dataset],
        "weak_residual_trend_filter_iterations": TREND_FILTER_ITERATIONS[dataset],
        "weak_residual_causal_ema_alpha": params["alpha"],
        "weak_residual_holt_level_alpha": params["alpha"],
        "weak_residual_holt_trend_beta": params["beta"],
    }
    result = [
        str(PYTHON), "scripts/search_phaseformer.py",
        "--dataset", dataset, "--horizon", "96", "--stage", "mechanism_full8",
        "--mechanism", "weak_residual_asymmetric_trend", "--period", "24",
        "--lookback", "720", "--max-epochs", str(args.max_epochs),
        "--seed", "2021", "--loss", "huber", "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir, "--overrides", json.dumps(overrides, separators=(",", ":")),
    ]
    if args.require_cuda:
        result.append("--require-cuda")
    if args.resume:
        result.append("--resume")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/weak_residual_trend_comparison_h96_scratch")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--components", default=",".join(COMPONENTS))
    parser.add_argument("--modes", default="minus_component,component_only")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    components = tuple(value.strip() for value in args.components.split(",") if value.strip())
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not datasets or set(datasets) - set(DATASETS):
        parser.error("--datasets must be a non-empty subset of " + ",".join(DATASETS))
    if not components or set(components) - set(COMPONENTS):
        parser.error("--components must be a non-empty subset of " + ",".join(COMPONENTS))
    if not modes or set(modes) - {"minus_component", "component_only"}:
        parser.error("--modes must contain minus_component and/or component_only")
    schedule = [(dataset, component, mode) for component in components for dataset in datasets for mode in modes]
    print(f"validation-only trend comparison schedule: {len(schedule)} candidate trainings", flush=True)
    for index, (dataset, component, mode) in enumerate(schedule, 1):
        cmd = command(dataset, component, mode, args)
        print(f"[{index}/{len(schedule)}] {dataset} H96 {component} {mode}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
