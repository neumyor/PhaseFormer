#!/usr/bin/env python3
"""Run the frozen Phase A capacity matrix.

This launcher delegates model construction and evaluation to
``search_phaseformer.py``. It is validation-only by default; passing
``--evaluate-test`` performs exactly one test read per run. Every cell is
trained independently with the PhaseFormer path, residual path, and fusion
gate jointly trainable.
"""

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
DEFAULT_OUTPUT = "research_runs/joint_pooled_lowrank_phase_a_scratch"


def round_to_multiple_of_4(value: float) -> int:
    return max(4, int(4 * round(value / 4)))


def relative_rank(pool_factor: int, q: float, horizon: int) -> int:
    max_rank = min(math.ceil(720 / pool_factor), horizon)
    return min(max_rank, round_to_multiple_of_4(q * max_rank))


def exact_rank(pool_factor: int, q: float, horizon: int) -> int:
    # The rank sweep uses q in {1, 1/4, 1/8, 1/16, 1/32}, whose products with
    # the valid maximum are already integers; the multiple-of-4 rounding and
    # floor in relative_rank would silently distort the q=1/16 and q=1/32 cells.
    max_rank = min(math.ceil(720 / pool_factor), horizon)
    return max(1, min(max_rank, int(round(q * max_rank))))


def build_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = [
        {"config_id": "phase_only", "mechanism": "no_residual", "overrides": {}},
        {
            "config_id": "direct_nlinear",
            "mechanism": "weak_residual",
            "overrides": {
                "weak_period_residual_head_type": "shared",
            },
        },
    ]
    rank_rule = exact_rank if args.exact_rank else relative_rank
    for pool_factor in args.pool_factors:
        for q in args.relative_ranks:
            rank = rank_rule(pool_factor, q, args.horizon)
            jobs.append(
                {
                    "config_id": f"pool{pool_factor}_q{q:g}_r{rank}",
                    "mechanism": "weak_residual",
                    "overrides": {
                        "weak_period_residual_head_type": "pooled_lowrank",
                        "weak_period_residual_pool_factor": pool_factor,
                        "weak_period_residual_rank": rank,
                        "weak_period_residual_smooth_ratio": 0.0,
                        "weak_period_residual_smooth_window": 24,
                    },
                    "pool_factor": pool_factor,
                    "relative_rank": q,
                    "rank": rank,
                }
            )
    return jobs


def command(args: argparse.Namespace, job: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/search_phaseformer.py"),
        "--dataset",
        args.dataset,
        "--horizon",
        str(args.horizon),
        "--stage",
        "confirm",
        "--mechanism",
        job["mechanism"],
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
        "--require-cuda",
        "--resume",
    ]
    if args.evaluate_test:
        cmd.append("--evaluate-test")
    if job["overrides"]:
        cmd.extend(["--overrides", json.dumps(job["overrides"], sort_keys=True)])
    return cmd


def dispatch(args: argparse.Namespace, jobs: list[dict]) -> None:
    pending = list(jobs)
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    attempts: dict[str, int] = {}
    while pending or active:
        while pending and len(active) < len(args.gpus):
            gpu = next(gpu for gpu in args.gpus if gpu not in active)
            job = pending.pop(0)
            key = job["config_id"]
            attempts[key] = attempts.get(key, 0) + 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = command(args, job)
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "dataset": args.dataset,
                        "seed": args.seed,
                        "config_id": key,
                        "gpu": gpu,
                        "attempt": attempts[key],
                        "command": cmd,
                    }
                ),
                flush=True,
            )
            active[gpu] = (job, subprocess.Popen(cmd, cwd=ROOT, env=env))

        finished = []
        for gpu, (job, process) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(gpu)
            if return_code != 0:
                key = job["config_id"]
                if attempts[key] <= args.retries:
                    pending.append(job)
                    print(
                        json.dumps(
                            {
                                "event": "retry",
                                "config_id": key,
                                "gpu": gpu,
                                "return_code": return_code,
                            }
                        ),
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        f"Phase A job {key} failed on GPU {gpu} "
                        f"after {attempts[key]} attempts ({return_code})"
                    )
            else:
                print(
                    json.dumps(
                        {"event": "finished", "config_id": job["config_id"], "gpu": gpu}
                    ),
                    flush=True,
                )
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(args.poll_seconds)


def write_manifest(args: argparse.Namespace, jobs: list[dict]) -> Path:
    root = ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"phase_a_{args.dataset}_h{args.horizon}_s{args.seed}_manifest.json"
    payload = {
        "protocol": (
            "Phase A capacity matrix; one test read per run"
            if args.evaluate_test
            else "Phase A validation-only; no test loader or trainer.test call"
        ),
        "evaluate_test": args.evaluate_test,
        "exact_rank": args.exact_rank,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": 720,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "loss": "huber",
        "pool_factors": args.pool_factors,
        "relative_ranks": args.relative_ranks,
        "jobs": jobs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def summarize(args: argparse.Namespace, jobs: list[dict]) -> Path:
    root = ROOT / args.output_root
    rows = []
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        if (
            row.get("dataset") != args.dataset
            or int(row.get("horizon", -1)) != args.horizon
            or int(row.get("seed", -1)) != args.seed
        ):
            continue
        config = json.loads(path.with_name("config.json").read_text())
        hp = config["hyperparams"]
        if config["mechanism"] == "no_residual":
            config_id = "phase_only"
            q = ""
        elif hp.get("weak_period_residual_head_type") == "shared":
            config_id = "direct_nlinear"
            q = ""
        else:
            pool = int(hp["weak_period_residual_pool_factor"])
            rank = int(hp["weak_period_residual_rank"])
            max_rank = min(math.ceil(720 / pool), args.horizon)
            q = rank / max_rank
            config_id = f"pool{pool}_q{q:g}_r{rank}"
        rows.append(
            {
                "dataset": args.dataset,
                "horizon": args.horizon,
                "seed": args.seed,
                "config_id": config_id,
                "mechanism": config["mechanism"],
                "pool_factor": hp.get("weak_period_residual_pool_factor", ""),
                "relative_rank": q,
                "rank": hp.get("weak_period_residual_rank", ""),
                "smooth_ratio": hp.get("weak_period_residual_smooth_ratio", 0.0),
                "val_mse": row.get("val_mse", ""),
                "val_mae": row.get("val_mae", ""),
                "test_mse": row.get("test_mse", ""),
                "test_mae": row.get("test_mae", ""),
                "best_val_loss": row.get("best_val_loss", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "run_id": row.get("run_id", ""),
                "run_dir": str(path.parent.relative_to(ROOT)),
            }
        )
    rows.sort(key=lambda row: row["config_id"])
    suffix = "results" if args.evaluate_test else "validation"
    out = root / f"phase_a_{args.dataset}_h{args.horizon}_s{args.seed}_{suffix}.csv"
    fields = list(rows[0]) if rows else [
        "dataset", "horizon", "seed", "config_id", "mechanism",
        "pool_factor", "relative_rank", "rank", "smooth_ratio",
        "val_mse", "val_mae", "test_mse", "test_mae", "best_val_loss",
        "elapsed_sec", "run_id", "run_dir",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    expected = {job["config_id"] for job in jobs}
    observed = {row["config_id"] for row in rows}
    if expected != observed:
        raise RuntimeError(
            f"incomplete Phase A summary: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"],
    )
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-factors", default="1,2,4")
    parser.add_argument("--relative-ranks", default="0.08333333333333333,0.3333333333333333,1")
    parser.add_argument("--exact-rank", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.pool_factors = [int(item) for item in args.pool_factors.split(",") if item]
    args.relative_ranks = [float(item) for item in args.relative_ranks.split(",") if item]
    args.gpus = [int(item) for item in args.gpus.split(",") if item]
    if not args.gpus:
        parser.error("--gpus must contain at least one device")
    if args.horizon not in {96, 192, 336, 720}:
        parser.error("--horizon must be one of 96, 192, 336, 720")
    jobs = build_jobs(args)
    manifest = write_manifest(args, jobs)
    if not args.summarize_only:
        dispatch(args, jobs)
    summary = summarize(args, jobs)
    print(json.dumps({"manifest": str(manifest), "summary": str(summary), "runs": len(jobs)}))


if __name__ == "__main__":
    main()
