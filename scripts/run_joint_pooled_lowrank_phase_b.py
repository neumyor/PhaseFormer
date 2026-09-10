#!/usr/bin/env python3
"""Run the preregistered validation-only Phase B smoothing matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE_A_ROOT = "research_runs/joint_pooled_lowrank_phase_a_scratch"
DEFAULT_OUTPUT = "research_runs/joint_pooled_lowrank_phase_b_scratch"


def read_phase_a_rows(args: argparse.Namespace) -> dict[tuple[str, int, str], dict]:
    rows = {}
    root = ROOT / args.phase_a_root
    for dataset in args.datasets:
        for seed in args.seeds:
            path = root / f"phase_a_{dataset}_h96_s{seed}_validation.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(newline="") as handle:
                current = list(csv.DictReader(handle))
            if len(current) != 11:
                raise RuntimeError(f"{path}: expected 11 rows, found {len(current)}")
            for row in current:
                for metric in ("val_mse", "val_mae", "best_val_loss"):
                    if not row.get(metric):
                        raise RuntimeError(f"{path}: missing {metric}")
                    float(row[metric])
                rows[(dataset, seed, row["config_id"])] = row
    return rows


def phase_b_cells(args: argparse.Namespace) -> list[dict]:
    cells = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for pool in args.pool_factors:
                for q in args.relative_ranks:
                    for smooth in args.smooth_ratios:
                        cells.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "pool_factor": pool,
                                "relative_rank": q,
                                "smooth_ratio": smooth,
                            }
                        )
    return cells


def phase_a_config_id(pool: int, q: float) -> str:
    max_rank = min((720 + pool - 1) // pool, 96)
    rank = max(4, int(4 * round((q * max_rank) / 4)))
    rank = min(max_rank, rank)
    return f"pool{pool}_q{q:g}_r{rank}"


def build_command(args: argparse.Namespace, cell: dict) -> list[str]:
    config_id = phase_a_config_id(cell["pool_factor"], cell["relative_rank"])
    phase_a_row = args.phase_a_rows[(cell["dataset"], cell["seed"], config_id)]
    rank = int(phase_a_row["rank"])
    overrides = {
        "weak_period_residual_head_type": "pooled_lowrank",
        "weak_period_residual_pool_factor": cell["pool_factor"],
        "weak_period_residual_rank": rank,
        "weak_period_residual_smooth_ratio": cell["smooth_ratio"],
        "weak_period_residual_smooth_window": 24,
    }
    return [
        sys.executable,
        str(ROOT / "scripts/search_phaseformer.py"),
        "--dataset",
        cell["dataset"],
        "--horizon",
        "96",
        "--stage",
        "confirm",
        "--mechanism",
        "weak_residual",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(cell["seed"]),
        "--loss",
        "huber",
        "--output-dir",
        args.output_root,
        "--num-workers",
        str(args.num_workers),
        "--bad-case-limit",
        "0",
        "--require-cuda",
        "--resume",
        "--overrides",
        json.dumps(overrides, sort_keys=True),
    ]


def dispatch(args: argparse.Namespace, jobs: list[dict]) -> None:
    pending = list(jobs)
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    attempts: dict[str, int] = {}
    while pending or active:
        while pending and len(active) < len(args.gpus):
            gpu = next(gpu for gpu in args.gpus if gpu not in active)
            job = pending.pop(0)
            key = json.dumps(job, sort_keys=True)
            attempts[key] = attempts.get(key, 0) + 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = build_command(args, job)
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "job": job,
                        "gpu": gpu,
                        "attempt": attempts[key],
                    }
                ),
                flush=True,
            )
            active[gpu] = (job, subprocess.Popen(command, cwd=ROOT, env=env))
        finished = []
        for gpu, (job, process) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(gpu)
            if return_code != 0:
                key = json.dumps(job, sort_keys=True)
                if attempts[key] <= args.retries:
                    pending.append(job)
                    print(json.dumps({"event": "retry", "job": job, "gpu": gpu}), flush=True)
                else:
                    raise RuntimeError(f"Phase B job failed: {job}, rc={return_code}")
            else:
                print(json.dumps({"event": "finished", "job": job, "gpu": gpu}), flush=True)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(args.poll_seconds)


def write_summary(args: argparse.Namespace, phase_a_rows: dict, cells: list[dict]) -> Path:
    output = ROOT / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        config_id = phase_a_config_id(cell["pool_factor"], cell["relative_rank"])
        if cell["smooth_ratio"] == 0.0:
            source = phase_a_rows[(cell["dataset"], cell["seed"], config_id)]
            row = dict(source)
            row["source_phase"] = "A_reused"
            row["run_dir"] = source["run_dir"]
        else:
            candidates = []
            for config_path in sorted((output / "runs").glob("*/config.json")):
                config = json.loads(config_path.read_text())
                if (
                    config.get("dataset") != cell["dataset"]
                    or int(config.get("seed", -1)) != cell["seed"]
                ):
                    continue
                hp = config.get("hyperparams", {})
                if (
                    hp.get("weak_period_residual_head_type") != "pooled_lowrank"
                    or int(hp.get("weak_period_residual_pool_factor", -1))
                    != cell["pool_factor"]
                    or float(hp.get("weak_period_residual_smooth_ratio", -1))
                    != cell["smooth_ratio"]
                ):
                    continue
                metrics_path = config_path.with_name("metrics.csv")
                if not metrics_path.is_file():
                    continue
                with metrics_path.open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                if not row.get("val_mse") or not row.get("val_mae"):
                    continue
                candidates.append((config_path, config, row))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one completed Phase B row for {cell}, found {len(candidates)}"
                )
            config_path, config, metrics = candidates[0]
            hp = config["hyperparams"]
            row = {
                "dataset": cell["dataset"],
                "horizon": 96,
                "seed": cell["seed"],
                "config_id": config_id,
                "mechanism": config["mechanism"],
                "pool_factor": cell["pool_factor"],
                "relative_rank": cell["relative_rank"],
                "rank": hp["weak_period_residual_rank"],
                "smooth_ratio": cell["smooth_ratio"],
                "val_mse": metrics["val_mse"],
                "val_mae": metrics["val_mae"],
                "best_val_loss": metrics["best_val_loss"],
                "elapsed_sec": metrics["elapsed_sec"],
                "run_id": metrics["run_id"],
                "run_dir": str(config_path.parent.relative_to(ROOT)),
                "source_phase": "B_trained",
            }
        rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], int(row["seed"]), row["config_id"], float(row["smooth_ratio"])))
    path = output / "phase_b_validation.csv"
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ETTh1,ETTm1")
    parser.add_argument("--seeds", default="2021,2022")
    parser.add_argument("--pool-factors", default="1,2,4")
    parser.add_argument("--relative-ranks", default="0.08333333333333333,0.3333333333333333")
    parser.add_argument("--smooth-ratios", default="0.25,0.50")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-a-root", default=PHASE_A_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.datasets = [item for item in args.datasets.split(",") if item]
    args.seeds = [int(item) for item in args.seeds.split(",") if item]
    args.pool_factors = [int(item) for item in args.pool_factors.split(",") if item]
    args.relative_ranks = [float(item) for item in args.relative_ranks.split(",") if item]
    args.smooth_ratios = [float(item) for item in args.smooth_ratios.split(",") if item]
    args.gpus = [int(item) for item in args.gpus.split(",") if item]
    args.phase_a_rows = read_phase_a_rows(args)
    cells = phase_b_cells(args)
    jobs = [cell for cell in cells if cell["smooth_ratio"] != 0.0]
    if len(jobs) != 48:
        raise RuntimeError(f"frozen Phase B budget expects 48 training jobs, found {len(jobs)}")
    if not args.summarize_only:
        dispatch(args, jobs)
    summary = write_summary(args, args.phase_a_rows, cells)
    print(json.dumps({"trained_jobs": len(jobs), "summary": str(summary)}))


if __name__ == "__main__":
    main()
