#!/usr/bin/env python3
"""Run a frozen per-setting causal-EMA smoothing sweep for the full-rank
(non-low-rank) weak-period residual head.

Unlike scripts/run_smooth_ratio_sweep.py (which sweeps a boxcar-smoothed
smooth_ratio inside PooledLowRankWeakPeriodResidualHead, i.e. with a rank
bottleneck), this launcher uses the plain WeakPeriodResidualHead (the
"shared"/direct_nlinear head, no pooling, no rank bottleneck) and swaps the
smoothing operator for a one-sided causal EMA (see
src/models/asymmetric_trend_components.py:_causal_ema), matching the operator
used in the prior X-A/Only-A trend-component research. Only
``weak_period_residual_smooth_ratio`` varies across jobs; the EMA ``alpha``
is fixed (see docs/PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md
for the disclosed rationale — it is an analogy choice, not tuned for this
sweep). Rank, gate_init, and (optionally) learning_rate are frozen per-setting
inputs carried over from the prior rank-sweep round, not derived here.
"""

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
DEFAULT_OUTPUT = "research_runs/causal_ema_smooth_sweep_v1"


def build_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    for smooth_ratio in args.smooth_ratios:
        overrides = {
            "weak_period_residual_smooth_ratio": smooth_ratio,
            "weak_period_residual_causal_ema_alpha": args.causal_ema_alpha,
            "weak_period_residual_gate_init": args.gate_init,
        }
        if args.learning_rate is not None:
            overrides["learning_rate"] = args.learning_rate
        jobs.append(
            {
                "config_id": f"causal_ema_s{smooth_ratio:g}",
                "mechanism": "weak_residual",
                "smooth_ratio": smooth_ratio,
                "overrides": overrides,
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
        "--evaluate-test",
    ]
    job_overrides = dict(job["overrides"])
    if args.overrides:
        job_overrides.update(json.loads(args.overrides))
    cmd.extend(["--overrides", json.dumps(job_overrides, sort_keys=True)])
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
                        f"causal ema smooth sweep job {key} failed on GPU {gpu} "
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
    path = root / f"causal_ema_sweep_{args.dataset}_h{args.horizon}_s{args.seed}_manifest.json"
    payload = {
        "protocol": (
            "causal-EMA smoothing sweep on the full-rank (non-low-rank) weak "
            "residual head; one test read per run (frozen gate_init/lr)"
        ),
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": 720,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "loss": "huber",
        "gate_init": args.gate_init,
        "learning_rate": args.learning_rate,
        "causal_ema_alpha": args.causal_ema_alpha,
        "smooth_ratios": args.smooth_ratios,
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
        if hp.get("weak_period_residual_head_type") is not None:
            continue
        if abs(float(hp.get("weak_period_residual_gate_init", -1)) - args.gate_init) > 1e-9:
            continue
        smooth_ratio = float(hp.get("weak_period_residual_smooth_ratio", 0.0))
        config_id = f"causal_ema_s{smooth_ratio:g}"
        rows.append(
            {
                "dataset": args.dataset,
                "horizon": args.horizon,
                "seed": args.seed,
                "config_id": config_id,
                "mechanism": config["mechanism"],
                "gate_init": hp.get("weak_period_residual_gate_init", ""),
                "learning_rate": hp.get("learning_rate", ""),
                "smooth_ratio": smooth_ratio,
                "causal_ema_alpha": hp.get("weak_period_residual_causal_ema_alpha", ""),
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
    rows.sort(key=lambda row: row["smooth_ratio"])
    out = root / f"causal_ema_sweep_{args.dataset}_h{args.horizon}_s{args.seed}_results.csv"
    fields = list(rows[0]) if rows else [
        "dataset", "horizon", "seed", "config_id", "mechanism",
        "gate_init", "learning_rate", "smooth_ratio", "causal_ema_alpha",
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
            f"incomplete causal ema smooth sweep summary: missing={sorted(expected - observed)}, "
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
    parser.add_argument("--gate-init", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--smooth-ratios", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--causal-ema-alpha", type=float, default=0.08)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overrides",
        default="",
        help="JSON dict merged over every job's overrides, for ad-hoc adjustments",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.smooth_ratios = [float(item) for item in args.smooth_ratios.split(",") if item]
    args.gpus = [int(item) for item in args.gpus.split(",") if item]
    if not args.gpus:
        parser.error("--gpus must contain at least one device")
    if args.horizon not in {96, 192, 336, 720}:
        parser.error("--horizon must be one of 96, 192, 336, 720")
    if not 0.0 <= args.gate_init <= 1.0:
        parser.error("--gate-init must be in [0, 1]")
    if not 0.0 < args.causal_ema_alpha <= 1.0:
        parser.error("--causal-ema-alpha must be in (0, 1]")
    for ratio in args.smooth_ratios:
        if not 0.0 <= ratio <= 1.0:
            parser.error(f"--smooth-ratios values must be in [0, 1], got {ratio}")
    jobs = build_jobs(args)
    manifest = write_manifest(args, jobs)
    if not args.summarize_only:
        dispatch(args, jobs)
    summary = summarize(args, jobs)
    print(json.dumps({"manifest": str(manifest), "summary": str(summary), "runs": len(jobs)}))


if __name__ == "__main__":
    main()
