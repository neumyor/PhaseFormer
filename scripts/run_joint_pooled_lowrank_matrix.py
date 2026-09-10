#!/usr/bin/env python3
"""Run the jointly trained PhaseFormer plus pooled low-rank NLinear matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_completed(root, dataset, horizon, seed):
    rows = []
    for config_path in sorted((root / "runs").glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if (
            config["dataset"] != dataset
            or int(config["horizon"]) != horizon
            or int(config["seed"]) != seed
        ):
            continue
        metrics_path = config_path.with_name("metrics.csv")
        if not metrics_path.is_file():
            continue
        with metrics_path.open() as handle:
            metric = next(csv.DictReader(handle))
        if not metric.get("test_mse"):
            continue
        rows.append(
            {
                "config": config,
                "metrics": metric,
                "run_dir": config_path.parent,
            }
        )
    return rows


def pooled_rows(rows):
    selected = []
    for row in rows:
        hp = row["config"]["hyperparams"]
        if hp.get("weak_period_residual_head_type") != "pooled_lowrank":
            continue
        selected.append(
            {
                **row,
                "pool_factor": int(hp["weak_period_residual_pool_factor"]),
                "rank": int(hp["weak_period_residual_rank"]),
                "smooth_ratio": float(hp["weak_period_residual_smooth_ratio"]),
            }
        )
    return selected


def validation_score(row):
    metrics = row["metrics"]
    return float(metrics["val_mse"]) + float(metrics["val_mae"])


def select_no_smoothing(rows, count, horizon):
    candidates = []
    for row in pooled_rows(rows):
        if row["smooth_ratio"] != 0.0:
            continue
        pooled_len = math.ceil(720 / row["pool_factor"])
        if row["rank"] >= min(pooled_len, horizon):
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: (validation_score(row), row["rank"]))
    selected = []
    seen = set()
    for row in candidates:
        key = (row["pool_factor"], row["rank"])
        if key not in seen:
            seen.add(key)
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"expected {count} completed no-smoothing low-rank configurations, "
            f"found {len(selected)}"
        )
    return selected


def config_overrides(pool_factor, rank, smooth_ratio):
    return {
        "weak_period_residual_head_type": "pooled_lowrank",
        "weak_period_residual_pool_factor": pool_factor,
        "weak_period_residual_rank": rank,
        "weak_period_residual_smooth_ratio": smooth_ratio,
        "weak_period_residual_smooth_window": 24,
    }


def build_command(args, *, baseline=False, pool_factor=None, rank=None, smooth_ratio=None):
    command = [
        sys.executable,
        str(ROOT / "scripts/search_phaseformer.py"),
        "--dataset",
        args.dataset,
        "--horizon",
        str(args.horizon),
        "--stage",
        "confirm",
        "--mechanism",
        "original" if baseline else "weak_residual",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(args.seed),
        "--loss",
        "huber",
        "--output-dir",
        args.output_root,
        "--num-workers",
        str(args.num_workers),
        "--bad-case-limit",
        "0",
        "--evaluate-test",
        "--require-cuda",
        "--resume",
    ]
    if not baseline:
        command.extend(
            ["--overrides", json.dumps(config_overrides(pool_factor, rank, smooth_ratio))]
        )
    return command


def dispatch(args, jobs):
    pending = list(jobs)
    active = {}
    while pending or active:
        while pending and len(active) < len(args.gpus):
            gpu = next(item for item in args.gpus if item not in active)
            job = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = build_command(args, **job)
            print(f"launch gpu={gpu}: {' '.join(command)}", flush=True)
            active[gpu] = (job, subprocess.Popen(command, cwd=ROOT, env=env))
        finished = []
        for gpu, (job, process) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise RuntimeError(f"job {job} on GPU {gpu} failed ({return_code})")
            print(f"finished gpu={gpu}: {job}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(5)


def rank_jobs(args):
    jobs = []
    for pool_factor in args.pool_factors:
        max_rank = min(math.ceil(720 / pool_factor), args.horizon)
        for rank in args.ranks:
            if rank <= max_rank:
                jobs.append(
                    {
                        "baseline": False,
                        "pool_factor": pool_factor,
                        "rank": rank,
                        "smooth_ratio": 0.0,
                    }
                )
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ETTh1", "ETTm1"])
    parser.add_argument("--stage", required=True, choices=["baseline", "rank", "smooth"])
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        default="research_runs/joint_pooled_lowrank_nlinear_scratch",
    )
    parser.add_argument("--pool-factors", default="1,2,4,8")
    parser.add_argument("--ranks", default="4,8,16,32,64,96")
    parser.add_argument("--smooth-ratios", default="0.10,0.25,0.50,0.75")
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()
    args.pool_factors = [int(item) for item in args.pool_factors.split(",") if item]
    args.ranks = [int(item) for item in args.ranks.split(",") if item]
    args.gpus = [int(item) for item in args.gpus.split(",") if item]
    if not args.gpus:
        parser.error("--gpus must include at least one GPU")

    if args.stage == "baseline":
        jobs = [{"baseline": True}]
    elif args.stage == "rank":
        jobs = rank_jobs(args)
    else:
        root = ROOT / args.output_root
        selected = select_no_smoothing(
            read_completed(root, args.dataset, args.horizon, args.seed),
            args.selected_count,
            args.horizon,
        )
        selection_path = root / (
            f"{args.dataset}_h{args.horizon}_s{args.seed}_smooth_selection.json"
        )
        selection_path.write_text(
            json.dumps(
                {
                    "selection_source": "validation",
                    "selected_no_smoothing_configs": [
                        {
                            "pool_factor": row["pool_factor"],
                            "rank": row["rank"],
                            "validation_score": validation_score(row),
                            "run_dir": str(row["run_dir"].relative_to(ROOT)),
                        }
                        for row in selected
                    ],
                    "smooth_ratios": [
                        float(item) for item in args.smooth_ratios.split(",") if item
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        jobs = [
            {
                "baseline": False,
                "pool_factor": row["pool_factor"],
                "rank": row["rank"],
                "smooth_ratio": float(ratio),
            }
            for row in selected
            for ratio in args.smooth_ratios.split(",")
            if ratio
        ]
    dispatch(args, jobs)


if __name__ == "__main__":
    main()
