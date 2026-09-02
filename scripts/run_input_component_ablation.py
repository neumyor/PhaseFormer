#!/usr/bin/env python3
"""Plan or execute the preregistered Track-R input-component training matrix."""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic")
MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
CONDITIONS = (("none", "full"),) + tuple(
    (hypothesis, variant)
    for hypothesis in ("h1", "h3", "h4")
    for variant in ("half_A", "minus_A", "sham")
)


def csv_values(value, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="research_runs/input_components_h134_scratch")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Explicitly allow CPU; formal runs require CUDA by default",
    )
    args = parser.parse_args()

    datasets = csv_values(args.datasets)
    models = csv_values(args.models)
    horizons = csv_values(args.horizons, int)
    seeds = csv_values(args.seeds, int)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")
    unknown = set(models) - set(MODELS)
    if unknown:
        parser.error(f"unknown models: {sorted(unknown)}")
    total = len(datasets) * len(models) * len(horizons) * len(seeds) * len(CONDITIONS)
    print(f"Track-R runs: {total}", flush=True)

    for dataset, model, horizon, seed, condition in itertools.product(
        datasets, models, horizons, seeds, CONDITIONS
    ):
        hypothesis, variant = condition
        command = [
            sys.executable,
            "scripts/search_phaseformer.py",
            "--dataset", dataset,
            "--horizon", str(horizon),
            "--stage", "input_components",
            "--mechanism", model,
            "--input-hypothesis", hypothesis,
            "--input-variant", variant,
            "--seed", str(seed),
            "--max-epochs", str(args.max_epochs),
            "--percent", str(args.percent),
            "--num-workers", str(args.num_workers),
            "--output-dir", args.output_dir,
        ]
        if not args.allow_cpu:
            command.append("--require-cuda")
        if args.resume:
            command.append("--resume")
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
