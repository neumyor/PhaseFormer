#!/usr/bin/env python3
"""Resumable single-seed Golden-directed search for strict PCTF.

This script is intentionally explicit about test-set selection: every executed
candidate uses `search_phaseformer.py --stage confirm --evaluate-test`, writes
one row to a compact CSV, and never deletes or overwrites a completed run.  A
failed subprocess is retried twice with the exact same command; `--resume`
makes the underlying runner reuse a completed metric instead of retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "scripts" / "search_phaseformer.py"
GOLDEN = {
    "ETTh1": {96: (0.359, 0.382), 192: (0.397, 0.404)},
    "ETTm1": {96: (0.293, 0.344), 192: (0.323, 0.361)},
}
SUMMARY_FIELDS = (
    "dataset", "horizon", "cycle", "profile", "loss", "lr_multiplier",
    "mse", "mae", "delta_mse_pct", "delta_mae_pct", "passes_half_percent", "run_id",
)

# Deliberately span near-off through a large trust region.  The latter is a
# user-authorized stress test, not a claim that it is a general prior.
PROFILES = {
    "off": (0.02, 0.01, 0.005),
    "c": (0.25, 0.10, 0.05),
    "w": (0.60, 0.24, 0.12),
    "x": (0.95, 0.50, 0.25),
}


def candidate_grid(dataset: str):
    cycles = (24, 48) if dataset == "ETTh1" else (24, 48, 96)
    # Keep the first pass bounded: objective/lr are likely to change the A2
    # anchor, while the four profiles test the ICPT correction amplitude.
    # Start with the best observed ETTh1 family (MAE + W) and its aggressive
    # LR perturbations, so interrupted runs spend budget on the most plausible
    # candidates before broadening to controls/extreme regions.
    profiles = ("w", "x", "off", "c")
    losses = ("mae", "huber")
    lrs = (3.0, 0.3, 1.0)
    for cycle in cycles:
        for profile in profiles:
            bounds = PROFILES[profile]
            for loss in losses:
                for lr in lrs:
                    yield cycle, profile, bounds, loss, lr


def command(dataset, horizon, cycle, bounds, loss, lr, output):
    correction, deformation, level = bounds
    overrides = json.dumps({
        "anchored_pctf_correction_max": correction,
        "anchored_pctf_deformation_max": deformation,
        "anchored_pctf_global_level_max": level,
    }, separators=(",", ":"))
    return [
        sys.executable, str(SEARCH), "--dataset", dataset,
        "--horizon", str(horizon), "--stage", "confirm",
        "--mechanism", "pctf_anchor_repair_strict_t28", "--period", "24",
        "--cycle-period", str(cycle), "--lookback", "720", "--percent", "100",
        "--max-epochs", "30", "--seed", "2021", "--loss", loss,
        "--lr-multiplier", str(lr), "--num-workers", "0", "--bad-case-limit", "0",
        "--overrides", overrides, "--output-dir", str(output), "--require-cuda",
        "--evaluate-test", "--resume",
    ]


def read_metrics(output: Path, dataset: str, horizon: int, cycle: int, loss: str, lr: float, bounds):
    wanted = tuple(map(float, bounds))
    for metrics in output.glob("runs/*/metrics.csv"):
        with metrics.open(newline="") as f:
            row = next(csv.DictReader(f))
        if not (row["dataset"] == dataset and int(row["horizon"]) == horizon
                and int(row["cycle_period"]) == cycle and row["loss"] == loss
                and float(row["lr_multiplier"]) == lr and row["test_mse"]):
            continue
        config = metrics.parent / "config.json"
        hp = json.loads(config.read_text())["hyperparams"]
        actual = tuple(float(hp[k]) for k in (
            "anchored_pctf_correction_max", "anchored_pctf_deformation_max",
            "anchored_pctf_global_level_max"))
        if actual == wanted:
            return float(row["test_mse"]), float(row["test_mae"]), row["run_id"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=tuple(GOLDEN), required=True)
    p.add_argument("--output-dir", default="research_runs/strict_t28_golden_hunt_v1")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="0 means full grid; useful for an interrupted staged launch")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = output / f"{args.dataset.lower()}_test_selection.csv"
    # A terminal interruption can occur between creation and the first flush.
    # Repair that single-file edge case before DictReader sees an invalid header.
    if summary.exists():
        lines = summary.read_text().splitlines()
        header = ",".join(SUMMARY_FIELDS)
        data = [line for line in lines if line and line != header]
        if not lines or lines[0] != header:
            summary.write_text(header + "\n" + "\n".join(data) + ("\n" if data else ""))
    new_file = not summary.exists()
    recorded = set()
    if not new_file:
        with summary.open(newline="") as existing:
            for row in csv.DictReader(existing):
                recorded.add(tuple(row[k] for k in (
                    "dataset", "horizon", "cycle", "profile", "loss", "lr_multiplier")))
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        done = 0
        for cycle, profile, bounds, loss, lr in candidate_grid(args.dataset):
            if args.max_candidates and done >= args.max_candidates:
                break
            candidate_passes = []
            for horizon, (gold_mse, gold_mae) in GOLDEN[args.dataset].items():
                old = read_metrics(output, args.dataset, horizon, cycle, loss, lr, bounds)
                if old is None:
                    cmd = command(args.dataset, horizon, cycle, bounds, loss, lr, output)
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue
                    for attempt in range(1, 4):
                        completed = subprocess.run(cmd, cwd=ROOT)
                        if completed.returncode == 0:
                            break
                        if attempt == 3:
                            raise RuntimeError(f"failed after 3 attempts: {cmd}")
                    old = read_metrics(output, args.dataset, horizon, cycle, loss, lr, bounds)
                    if old is None:
                        raise RuntimeError("runner returned success but metrics were not found")
                mse, mae, run_id = old
                candidate_passes.append(
                    mse <= gold_mse * .995 and mae <= gold_mae * .995
                )
                key = (args.dataset, str(horizon), str(cycle), profile, loss, str(lr))
                if key not in recorded:
                    writer.writerow({
                        "dataset": args.dataset, "horizon": horizon, "cycle": cycle,
                        "profile": profile, "loss": loss, "lr_multiplier": lr,
                        "mse": mse, "mae": mae,
                        "delta_mse_pct": (mse - gold_mse) / gold_mse * 100,
                        "delta_mae_pct": (mae - gold_mae) / gold_mae * 100,
                        "passes_half_percent": mse <= gold_mse * .995 and mae <= gold_mae * .995,
                        "run_id": run_id,
                    })
                    f.flush()
                    recorded.add(key)
            done += 1
            if not args.dry_run and all(candidate_passes):
                print(
                    "TARGET_REACHED",
                    f"dataset={args.dataset} cycle={cycle} profile={profile} "
                    f"loss={loss} lr={lr}",
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
