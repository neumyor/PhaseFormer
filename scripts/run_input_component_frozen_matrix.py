#!/usr/bin/env python3
"""Discover Track-R full checkpoints and plan/execute the Track-F matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED = {
    "dataset", "horizon", "seed", "mechanism", "input_hypothesis",
    "input_variant", "checkpoint",
}


def discover(root: Path):
    rows = []
    for path in root.rglob("metrics.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not REQUIRED.issubset(frame.columns):
            continue
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
    if result.duplicated(keys, keep=False).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys]
        raise ValueError(f"duplicate full checkpoints detected:\n{duplicates.to_string(index=False)}")
    return result.sort_values(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-r-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    frame = discover(args.track_r_dir)
    if args.expected_count and len(frame) != args.expected_count:
        parser.error(f"expected {args.expected_count} full checkpoints, found {len(frame)}")
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
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
