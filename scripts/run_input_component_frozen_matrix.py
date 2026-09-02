#!/usr/bin/env python3
"""Discover Track-R full checkpoints and plan/execute the Track-F matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input_component_ablation import (
    DATASETS, HORIZONS, PRIORITY_HORIZON, PRIORITY_SEED, SEEDS,
    expected_full_anchors, parse_dataset_scope, parse_scope,
)


REQUIRED = {
    "dataset", "horizon", "seed", "mechanism", "input_hypothesis",
    "input_variant", "checkpoint", "percent", "max_eval_samples", "test_mse",
}


def discover(root: Path, *, smoke: bool, horizons=None, seeds=None,
            datasets=None):
    rows = []
    for path in root.rglob("metrics.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not REQUIRED.issubset(frame.columns):
            continue
        if not smoke and "stage" in frame:
            frame = frame[frame.stage == "input_components"]
        frame = frame[
            (frame.input_hypothesis == "none") & (frame.input_variant == "full")
        ].copy()
        if len(frame):
            frame["source_file"] = str(path)
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no completed Track-R full metrics under {root}")
    result = pd.concat(rows, ignore_index=True)
    keys = ["dataset", "horizon", "seed", "mechanism"]
    # Duplicate detection stays global: a duplicate anywhere in the source is a
    # data-integrity failure, even if it falls outside the requested scope.
    if result.duplicated(keys, keep=False).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys]
        raise ValueError(f"duplicate full checkpoints detected:\n{duplicates.to_string(index=False)}")
    # Restrict to the requested dataset/horizon/seed scope (D0, D1, the full
    # matrix, or a dataset-restricted variant).  All downstream gates then apply
    # to the scoped rows only, so out-of-scope rows (e.g. a dropped dataset) can
    # never block a scoped read.
    if datasets is not None:
        result = result[result.dataset.isin(datasets)].copy()
    if horizons is not None:
        result = result[result.horizon.isin(horizons)].copy()
    if seeds is not None:
        result = result[result.seed.isin(seeds)].copy()
    if result.test_mse.notna().any():
        raise ValueError("full checkpoints already have test metrics; refusing a duplicate test read")
    if not smoke and ((result.percent != 100).any() or (result.max_eval_samples != 0).any()):
        raise ValueError("formal Track-F requires percent=100 and max_eval_samples=0")
    return result.sort_values(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-r-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", default=",".join(DATASETS),
                       help="comma list of datasets to include (default: all)")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)),
                       help="comma list of horizons to include (default: all)")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                       help="comma list of seeds to include (default: all)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--priority-first", action="store_true", default=True,
        help=f"evaluate horizon={PRIORITY_HORIZON}, seed={PRIORITY_SEED} first (default)",
    )
    parser.add_argument(
        "--no-priority-first", dest="priority_first", action="store_false",
        help="preserve ordinary checkpoint ordering",
    )
    args = parser.parse_args()
    if args.max_samples and not args.smoke:
        parser.error("--max-samples requires --smoke")
    datasets = parse_dataset_scope(parser, args.datasets)
    horizons, seeds = parse_scope(parser, args.horizons, args.seeds)
    frame = discover(args.track_r_dir, smoke=args.smoke,
                     horizons=horizons, seeds=seeds, datasets=datasets)
    if args.expected_count is None:
        expected = None if args.smoke else expected_full_anchors(
            horizons, seeds, datasets=datasets)
    else:
        expected = args.expected_count
    if expected is not None and len(frame) != expected:
        parser.error(f"expected {expected} full checkpoints, found {len(frame)}")
    sort_columns = ["dataset", "horizon", "seed", "mechanism"]
    if args.priority_first:
        frame["_priority"] = (
            (frame.horizon != PRIORITY_HORIZON) | (frame.seed != PRIORITY_SEED)
        )
        sort_columns = ["_priority"] + sort_columns
        frame = frame.sort_values(sort_columns)
        print(
            f"Priority pass first: horizon={PRIORITY_HORIZON}, seed={PRIORITY_SEED}",
            flush=True,
        )
    print(f"Track-F checkpoints: {len(frame)}", flush=True)
    for row in frame.itertuples(index=False):
        destination = (
            args.output_dir
            / str(row.dataset)
            / f"h{int(row.horizon)}"
            / str(row.mechanism)
            / f"s{int(row.seed)}"
        )
        metrics = destination / "frozen_metrics.csv"
        if args.resume and metrics.is_file():
            print(f"RESUME completed: {metrics}", flush=True)
            continue
        checkpoint = Path(str(row.checkpoint))
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint recorded by {row.source_file} is missing: {checkpoint}")
        command = [
            sys.executable,
            "scripts/evaluate_input_component_checkpoint.py",
            "--dataset", str(row.dataset),
            "--horizon", str(int(row.horizon)),
            "--model", str(row.mechanism),
            "--seed", str(int(row.seed)),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(destination),
            "--num-workers", str(args.num_workers),
            "--bootstrap-replicates", str(args.bootstrap_replicates),
        ]
        if args.batch_size:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.max_samples:
            command.extend(["--max-samples", str(args.max_samples)])
        if args.smoke:
            command.append("--smoke")
        if not args.allow_cpu:
            command.append("--require-cuda")
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
