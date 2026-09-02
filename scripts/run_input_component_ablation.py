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
# The preregistered first pass is deliberately narrow: one seed at the
# medium horizon.  Full expansion remains available and is resumable.
PRIORITY_HORIZON = 192
PRIORITY_SEED = 2021
CONDITIONS = (("none", "full"),) + tuple(
    (hypothesis, variant)
    for hypothesis in ("h1", "h3", "h4")
    for variant in ("half_A", "minus_A", "sham")
)


def csv_values(value, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_scope(parser, horizons_csv, seeds_csv):
    """Validate --horizons/--seeds and return sorted integer tuples.

    Shared by the Track R driver and the Track F / retrained-test / summarize
    runners so a scoped run (e.g. D0 = horizon 192 x seed 2021) keeps exactly
    the same horizon/seed vocabulary as the full matrix.
    """
    horizons = tuple(sorted(csv_values(horizons_csv, int)))
    seeds = tuple(sorted(csv_values(seeds_csv, int)))
    unknown = set(horizons) - set(HORIZONS)
    if unknown:
        parser.error(f"unknown horizons: {sorted(unknown)}")
    unknown = set(seeds) - set(SEEDS)
    if unknown:
        parser.error(f"unknown seeds: {sorted(unknown)}")
    return horizons, seeds


def expected_full_anchors(horizons, seeds):
    """Number of unique full (none/full) checkpoints in a scope.

    One full checkpoint per (dataset, model, horizon, seed) setting; the frozen
    Track F matrix has exactly this many anchors, and the summarize setting count
    per track matches it too.
    """
    return len(DATASETS) * len(MODELS) * len(horizons) * len(seeds)


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
        "--priority-first", action="store_true", default=True,
        help=f"run priority horizon={PRIORITY_HORIZON}, seed={PRIORITY_SEED} first (default)",
    )
    parser.add_argument(
        "--no-priority-first", dest="priority_first", action="store_false",
        help="preserve the ordinary dataset/model/horizon/seed ordering",
    )
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

    if args.priority_first:
        # Keep the complete matrix and only change scheduling order, so a
        # resumed run never loses coverage.  This puts the single-seed H192
        # pass ahead of H96/H336/H720 and seeds 2022/2023.
        horizons = tuple(sorted(horizons, key=lambda h: h != PRIORITY_HORIZON))
        seeds = tuple(sorted(seeds, key=lambda s: s != PRIORITY_SEED))
        print(
            f"Priority pass first: horizon={PRIORITY_HORIZON}, seed={PRIORITY_SEED}; "
            "use --horizons 192 --seeds 2021 for the isolated priority pass.",
            flush=True,
        )

    settings = list(itertools.product(datasets, models, horizons, seeds))
    if args.priority_first:
        # Sort settings globally, rather than only sorting the inner horizon /
        # seed loops, so every h192/seed2021 condition is completed before any
        # other seed or horizon begins.
        settings.sort(key=lambda item: (item[2] != PRIORITY_HORIZON or item[3] != PRIORITY_SEED))

    for dataset, model, horizon, seed in settings:
        for hypothesis, variant in CONDITIONS:
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
