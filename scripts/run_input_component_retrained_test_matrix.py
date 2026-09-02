#!/usr/bin/env python3
"""Evaluate the complete validation-selected Track-R matrix exactly once."""

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
    CONDITIONS, DATASETS, HORIZONS, PRIORITY_HORIZON, PRIORITY_SEED, SEEDS,
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
        if "stage" in frame:
            frame = frame[frame.stage == "input_components"]
        if len(frame):
            frame = frame.copy()
            frame["source_file"] = str(path)
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no completed validation-only Track-R metrics under {root}")
    result = pd.concat(rows, ignore_index=True)
    keys = ["dataset", "horizon", "seed", "mechanism", "input_hypothesis", "input_variant"]
    # Duplicate detection stays global: duplicates anywhere in the source are a
    # data-integrity failure, even outside the requested scope.
    if result.duplicated(keys, keep=False).any():
        raise ValueError("duplicate Track-R checkpoints detected")
    # Restrict to the requested dataset/horizon/seed scope (D0, D1, the full
    # matrix, or a dataset-restricted variant).  Completeness / no-test / percent
    # gates then apply to the scoped rows only, so an unrelated partially-complete
    # out-of-scope setting (a D1 setting, or a dropped dataset) cannot block a
    # scoped read.
    if datasets is not None:
        result = result[result.dataset.isin(datasets)].copy()
    if horizons is not None:
        result = result[result.horizon.isin(horizons)].copy()
    if seeds is not None:
        result = result[result.seed.isin(seeds)].copy()
    if result.test_mse.notna().any():
        raise ValueError("Track-R source already contains test metrics; refusing a second test read")
    if not smoke:
        if (result.percent != 100).any() or (result.max_eval_samples != 0).any():
            raise ValueError("formal Track-R test requires percent=100 and max_eval_samples=0")
        expected_conditions = set(CONDITIONS)
        setting_keys = ["dataset", "horizon", "seed", "mechanism"]
        for setting, group in result.groupby(setting_keys, dropna=False):
            found = set(zip(group.input_hypothesis, group.input_variant))
            if found != expected_conditions:
                raise ValueError(f"incomplete Track-R conditions for {setting}")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
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
    all_frame = discover(args.track_r_dir, smoke=args.smoke,
                         horizons=horizons, seeds=seeds, datasets=datasets)
    # none/full is evaluated once by Track F and reused as Track R's common
    # baseline; only the 9 retrained intervention checkpoints need another read.
    frame = all_frame[
        ~((all_frame.input_hypothesis == "none") & (all_frame.input_variant == "full"))
    ].copy()
    expected = args.expected_count
    if expected is None and not args.smoke:
        expected = expected_full_anchors(horizons, seeds, datasets=datasets) \
            * (len(CONDITIONS) - 1)
    if expected is not None and len(frame) != expected:
        parser.error(f"expected {expected} Track-R checkpoints, found {len(frame)}")
    if args.priority_first:
        frame["_priority"] = (
            (frame.horizon != PRIORITY_HORIZON) | (frame.seed != PRIORITY_SEED)
        )
        frame = frame.sort_values(
            ["_priority", "dataset", "horizon", "seed", "mechanism",
             "input_hypothesis", "input_variant"]
        )
        print(
            f"Priority pass first: horizon={PRIORITY_HORIZON}, seed={PRIORITY_SEED}",
            flush=True,
        )
    print(f"Track-R test checkpoints: {len(frame)}", flush=True)

    for row in frame.itertuples(index=False):
        destination = (
            args.output_dir / str(row.dataset) / f"h{int(row.horizon)}"
            / str(row.mechanism) / f"s{int(row.seed)}"
            / f"{row.input_hypothesis}_{row.input_variant}"
        )
        metrics = destination / "retrained_metrics.csv"
        if args.resume and metrics.is_file():
            print(f"RESUME completed: {metrics}", flush=True)
            continue
        checkpoint = Path(str(row.checkpoint))
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint recorded by {row.source_file}: {checkpoint}")
        command = [
            sys.executable, "scripts/evaluate_input_component_retrained_checkpoint.py",
            "--dataset", str(row.dataset), "--horizon", str(int(row.horizon)),
            "--model", str(row.mechanism), "--seed", str(int(row.seed)),
            "--input-hypothesis", str(row.input_hypothesis),
            "--input-variant", str(row.input_variant),
            "--checkpoint", str(checkpoint), "--output-dir", str(destination),
            "--selection-source", str(row.source_file),
            "--num-workers", str(args.num_workers),
        ]
        if args.batch_size:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.smoke:
            command.append("--smoke")
        if args.max_samples:
            command.extend(["--max-samples", str(args.max_samples)])
        if not args.allow_cpu:
            command.append("--require-cuda")
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
