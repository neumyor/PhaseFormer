#!/usr/bin/env python3
"""Final loss-function refinement for the strict-T28 Golden search.

MSE and Smooth-L1 are existing trainer options.  This stage is deliberately
kept topology- and horizon-sharing invariant, and only executes for a dataset
that has not already reached the two-horizon, two-metric test-selection gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from run_strict_t28_golden_hunt import GOLDEN, ROOT
from run_strict_t28_golden_refinement import Candidate, command, read_metrics


FIELDS = (
    "dataset", "horizon", "label", "cycle", "loss", "lr_multiplier",
    "max_epochs", "overrides_json", "mse", "mae", "delta_mse_pct",
    "delta_mae_pct", "passes_half_percent", "run_id",
)


def candidates(dataset: str):
    # Stage 1 showed that low-LR MAE is near the Pareto frontier.  Smooth-L1
    # interpolates its robust gradients and MSE's stronger large-error signal;
    # exact-loss MSE is retained as the opposite endpoint.  Bounds/cycle stay
    # in the observed near-miss region for each dataset.
    if dataset == "ETTh1":
        return (
            Candidate("x_smae015", 24, .95, .50, .25, "smae", .15, 50),
            Candidate("x_smae020", 24, .95, .50, .25, "smae", .20, 50),
            Candidate("x_smae030", 24, .95, .50, .25, "smae", .30, 50),
            Candidate("x_mse010", 24, .95, .50, .25, "mse", .10, 50),
            Candidate("x_mse015", 24, .95, .50, .25, "mse", .15, 50),
            Candidate("x_mse020", 24, .95, .50, .25, "mse", .20, 50),
            Candidate("u_smae015", 24, 1.40, .80, .40, "smae", .15, 50),
        )
    return (
        Candidate("w_smae010", 24, .60, .24, .12, "smae", .10, 50),
        Candidate("w_smae015", 24, .60, .24, .12, "smae", .15, 50),
        Candidate("w_smae020", 24, .60, .24, .12, "smae", .20, 50),
        Candidate("w_mse005", 24, .60, .24, .12, "mse", .05, 50),
        Candidate("w_mse010", 24, .60, .24, .12, "mse", .10, 50),
        Candidate("w_mse015", 24, .60, .24, .12, "mse", .15, 50),
        Candidate("x_smae010", 24, .95, .50, .25, "smae", .10, 50),
    )


def already_reached(output: Path, dataset: str) -> bool:
    # A completed success in either earlier ledger makes extra test selection
    # unnecessary.  Require an explicit passing row for each horizon.
    passed = set()
    for name in (f"{dataset.lower()}_test_selection.csv",
                 f"{dataset.lower()}_refinement_test_selection.csv"):
        path = output / name
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("passes_half_percent") == "True":
                    passed.add(int(row["horizon"]))
    return passed == set(GOLDEN[dataset])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(GOLDEN), required=True)
    parser.add_argument("--output-dir", default="research_runs/strict_t28_golden_hunt_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if already_reached(output, args.dataset):
        print(f"TARGET_ALREADY_REACHED dataset={args.dataset}")
        return
    summary = output / f"{args.dataset.lower()}_loss_refinement_test_selection.csv"
    new_file = not summary.exists()
    recorded = set()
    if not new_file:
        with summary.open(newline="") as f:
            recorded = {(row["dataset"], row["horizon"], row["label"])
                        for row in csv.DictReader(f)}
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for candidate in candidates(args.dataset):
            passed = []
            for horizon, (gold_mse, gold_mae) in GOLDEN[args.dataset].items():
                metrics = read_metrics(output, args.dataset, horizon, candidate)
                if metrics is None:
                    cmd = command(args.dataset, horizon, candidate, output)
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue
                    for _ in range(3):
                        if subprocess.run(cmd, cwd=ROOT).returncode == 0:
                            break
                    else:
                        raise RuntimeError(f"failed after three attempts: {cmd}")
                    metrics = read_metrics(output, args.dataset, horizon, candidate)
                    if metrics is None:
                        raise RuntimeError("successful runner did not produce matching metrics")
                mse, mae, run_id = metrics
                success = mse <= gold_mse * .995 and mae <= gold_mae * .995
                passed.append(success)
                key = (args.dataset, str(horizon), candidate.label)
                if key not in recorded:
                    writer.writerow({
                        "dataset": args.dataset, "horizon": horizon,
                        "label": candidate.label, "cycle": candidate.cycle,
                        "loss": candidate.loss, "lr_multiplier": candidate.lr,
                        "max_epochs": candidate.epochs,
                        "overrides_json": json.dumps(candidate.overrides(), sort_keys=True),
                        "mse": mse, "mae": mae,
                        "delta_mse_pct": (mse - gold_mse) / gold_mse * 100,
                        "delta_mae_pct": (mae - gold_mae) / gold_mae * 100,
                        "passes_half_percent": success, "run_id": run_id,
                    })
                    f.flush()
                    recorded.add(key)
            if not args.dry_run and all(passed):
                print(f"TARGET_REACHED dataset={args.dataset} label={candidate.label}", flush=True)
                return


if __name__ == "__main__":
    main()
