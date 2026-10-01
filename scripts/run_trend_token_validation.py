#!/usr/bin/env python3
"""Resumable multi-GPU validation matrix for the learned trend-token scheme.

The script creates a deterministic manifest, runs one search_phaseformer job per
GPU at a time, and records every attempt.  It intentionally uses validation-only
selection; ``search_phaseformer.py`` remains the single source of truth for data
loading, checkpoint selection, and metrics.  Re-running with ``--resume`` skips
completed jobs and retries failed/incomplete jobs without changing their config
hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "scripts" / "search_phaseformer.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def job_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_jobs(args):
    arms = [
        ("phase_only", {}),
        ("dense_residual", {}),
        ("trend_token_m1_no_interaction", {}),
        ("trend_token_m1", {}),
    ]
    jobs = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                for arm, overrides in arms:
                    payload = {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "arm": arm,
                        "lookback": args.lookback,
                        "period": args.period,
                        "epochs": args.epochs,
                        "capacity": args.capacity,
                        "batch_size": args.batch_size,
                        "percent": args.percent,
                        "loss": args.loss,
                        "overrides": overrides,
                    }
                    config_hash = job_hash(payload)
                    job_id = f"{dataset.lower()}_h{horizon}_s{seed}_{arm}_{config_hash}"
                    cmd = [
                        sys.executable,
                        str(SEARCH),
                        "--dataset", dataset,
                        "--horizon", str(horizon),
                        "--stage", "mechanism_screen_1",
                        "--mechanism", arm,
                        "--lookback", str(args.lookback),
                        "--period", str(args.period),
                        "--max-epochs", str(args.epochs),
                        "--seed", str(seed),
                        "--loss", args.loss,
                        "--capacity", args.capacity,
                        "--percent", str(args.percent),
                        "--num-workers", str(args.num_workers),
                        "--output-dir", str(Path(args.output_dir) / "search"),
                        "--require-cuda",
                        "--resume",
                    ]
                    if args.batch_size:
                        cmd.extend(["--batch-size", str(args.batch_size)])
                    if args.max_eval_samples:
                        cmd.extend(["--max-eval-samples", str(args.max_eval_samples)])
                    jobs.append({
                        "job_id": job_id,
                        "config_hash": config_hash,
                        "payload": payload,
                        "command": cmd,
                    })
    return jobs


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def run_one(job, args, gpu: int, attempt: int):
    out = Path(args.output_dir)
    status_dir = out / "job_status"
    log_dir = out / "logs"
    status_path = status_dir / f"{job['job_id']}.json"
    log_path = log_dir / f"{job['job_id']}.attempt{attempt}.log"
    status = {
        "job_id": job["job_id"],
        "config_hash": job["config_hash"],
        "gpu": gpu,
        "attempt": attempt,
        "status": "running",
        "started_at": now(),
        "command": job["command"],
        "log": str(log_path),
    }
    write_json(status_path, status)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"# started {status['started_at']} gpu={gpu} attempt={attempt}\n")
        log.write("# " + " ".join(job["command"]) + "\n")
        proc = subprocess.run(
            job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    status.update({
        "returncode": proc.returncode,
        "elapsed_sec": time.monotonic() - started,
        "finished_at": now(),
        "status": "completed" if proc.returncode == 0 else "failed",
    })
    write_json(status_path, status)
    return proc.returncode


def collect_status(jobs, args):
    rows = []
    for job in jobs:
        path = Path(args.output_dir) / "job_status" / f"{job['job_id']}.json"
        item = {"job_id": job["job_id"], "config_hash": job["config_hash"], **job["payload"]}
        if path.exists():
            item.update(json.loads(path.read_text()))
        else:
            item["status"] = "pending"
        rows.append(item)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (out / "run_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/trend_token_validation_scratch")
    parser.add_argument("--datasets", default="ETTh1")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--gpus", required=True, help="comma separated physical GPU ids")
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--capacity", choices=("base", "compact"), default="base")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--loss", choices=("mse", "mae", "smae", "huber", "smape"), default="huber")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    args.datasets = parse_csv(args.datasets)
    args.horizons = parse_csv(args.horizons, int)
    args.seeds = parse_csv(args.seeds, int)
    args.gpus = parse_csv(args.gpus, int)
    if not args.gpus:
        raise SystemExit("at least one GPU is required")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest = {
        "experiment_id": out.name,
        "created_at": now(),
        "repository": str(ROOT),
        "python": sys.executable,
        "platform": platform.platform(),
        "args": vars(args),
        "validation_only": True,
        "selection_rule": "lowest validation loss; no test loader",
        "jobs": jobs,
    }
    write_json(out / "manifest.json", manifest)

    # Round-robin allocation keeps every GPU busy while each GPU remains
    # single-process.  A job is considered complete only after the underlying
    # runner writes its root metrics.csv and status=completed.
    pending = []
    for job in jobs:
        path = out / "job_status" / f"{job['job_id']}.json"
        if args.resume and path.exists():
            state = json.loads(path.read_text())
            if state.get("status") == "completed":
                continue
        pending.append(job)
    if not pending:
        collect_status(jobs, args)
        print("No pending jobs; manifest is already complete.")
        return 0

    # Workers are ordinary child processes.  This makes the supervisor safe to
    # launch with nohup and lets it survive SSH disconnects.
    import multiprocessing as mp

    buckets = {gpu: [] for gpu in args.gpus}
    for index, job in enumerate(pending):
        buckets[args.gpus[index % len(args.gpus)]].append(job)

    def worker(gpu, bucket):
        for job in bucket:
            status_path = out / "job_status" / f"{job['job_id']}.json"
            attempt = 1
            if status_path.exists():
                previous = json.loads(status_path.read_text())
                attempt = int(previous.get("attempt", 0)) + 1
            for _ in range(args.max_retries + 1):
                rc = run_one(job, args, gpu, attempt)
                if rc == 0:
                    break
                attempt += 1

    processes = [mp.Process(target=worker, args=(gpu, bucket), daemon=False) for gpu, bucket in buckets.items() if bucket]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    collect_status(jobs, args)
    failed = [
        json.loads(path.read_text())
        for path in (out / "job_status").glob("*.json")
        if json.loads(path.read_text()).get("status") != "completed"
    ] if (out / "job_status").exists() else []
    print(json.dumps({"jobs": len(jobs), "pending_before_run": len(pending), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

