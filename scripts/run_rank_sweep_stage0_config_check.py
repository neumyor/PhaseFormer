#!/usr/bin/env python3
"""Stage 0 config check for the conditioned low-rank rank sweep.

For each setting in the conditioned sweep, trains the unfactorized
``direct_nlinear`` control under a small validation-only grid
(gate_init x learning-rate) and freezes one configuration per setting by
lowest validation MSE. No test loader is instantiated.

Usage:
    python scripts/run_rank_sweep_stage0_config_check.py \
        --settings ETTh2:96,ETTh2:720,ETTm2:96,ETTm2:192,Weather:96,Weather:192,Electricity:336 \
        --output-root research_runs/rank_sweep_2_stage0
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

GATES = [0.2, 0.5]
LRS = ["default", "0.0003"]


def build_jobs(setting: str) -> list[dict]:
    dataset, horizon = setting.split(":")
    jobs = []
    for gate in GATES:
        for lr in LRS:
            overrides = {
                "weak_period_residual_head_type": "shared",
                "weak_period_residual_gate_init": gate,
            }
            if lr != "default":
                overrides["learning_rate"] = float(lr)
            jobs.append(
                {
                    "config_id": f"direct_g{gate:g}_lr{lr}",
                    "mechanism": "weak_residual",
                    "overrides": overrides,
                }
            )
    return jobs


def command(args: argparse.Namespace, setting: str, job: dict) -> list[str]:
    dataset, horizon = setting.split(":")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/search_phaseformer.py"),
        "--dataset",
        dataset,
        "--horizon",
        horizon,
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
    if job["overrides"]:
        cmd.extend(["--overrides", json.dumps(job["overrides"], sort_keys=True)])
    return cmd


def dispatch(args: argparse.Namespace, setting: str, jobs: list[dict]) -> None:
    gpus = list(args.gpus)
    pending = list(jobs)
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    attempts: dict[str, int] = {}
    while pending or active:
        while pending and len(active) < len(gpus):
            gpu = next(g for g in gpus if g not in active)
            job = pending.pop(0)
            key = job["config_id"]
            attempts[key] = attempts.get(key, 0) + 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = command(args, setting, job)
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "setting": setting,
                        "config_id": key,
                        "gpu": gpu,
                        "attempt": attempts[key],
                    }
                ),
                flush=True,
            )
            active[gpu] = (job, subprocess.Popen(cmd, cwd=ROOT, env=env))
        finished = []
        for gpu, (job, process) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(gpu)
            if code != 0:
                key = job["config_id"]
                if attempts[key] <= args.retries:
                    pending.append(job)
                    print(
                        json.dumps(
                            {"event": "retry", "config_id": key, "return_code": code}
                        ),
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        f"Stage 0 job {key} failed on GPU {gpu} after "
                        f"{attempts[key]} attempts ({code})"
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


def summarize(args: argparse.Namespace, setting: str, jobs: list[dict]) -> dict:
    root = ROOT / args.output_root
    rows = []
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        if (
            row.get("dataset") != setting.split(":")[0]
            or int(row.get("horizon", -1)) != int(setting.split(":")[1])
            or int(row.get("seed", -1)) != args.seed
        ):
            continue
        config = json.loads(path.with_name("config.json").read_text())
        hp = config["hyperparams"]
        # Arm labels: the 3e-4 arm is identifiable by its exact value; every
        # dataset in this sweep has a 1e-3 default, so no collision is possible.
        resolved_lr = float(hp["learning_rate"])
        lr_label = "0.0003" if abs(resolved_lr - 0.0003) < 1e-9 else "default"
        rows.append(
            {
                "config_id": (
                    f"direct_g{float(hp['weak_period_residual_gate_init']):g}"
                    f"_lr{lr_label}"
                ),
                "val_mse": float(row["val_mse"]),
                "val_mae": float(row["val_mae"]),
                "run_dir": str(path.parent.relative_to(ROOT)),
            }
        )
    expected = {job["config_id"] for job in jobs}
    observed = {row["config_id"] for row in rows}
    if expected != observed:
        raise RuntimeError(
            f"incomplete Stage 0 summary for {setting}: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    best = min(rows, key=lambda r: (r["val_mse"], r["val_mae"]))
    return {"rows": rows, "frozen": best["config_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        default="ETTh2:96,ETTh2:720,ETTm2:96,ETTm2:192,Weather:96,Weather:192,Electricity:336",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", default="research_runs/rank_sweep_2_stage0")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(",") if g]
    if not args.gpus:
        parser.error("--gpus must contain at least one device")
    settings = [s for s in args.settings.split(",") if s]
    root = ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for setting in settings:
        jobs = build_jobs(setting)
        manifest = root / f"stage0_{setting.replace(':', '_')}_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "protocol": "Stage 0 validation-only config check; no test read",
                    "setting": setting,
                    "seed": args.seed,
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if not args.summarize_only:
            dispatch(args, setting, jobs)
        result = summarize(args, setting, jobs)
        frozen[setting] = result["frozen"]
        out = root / f"stage0_{setting.replace(':', '_')}_validation.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["config_id", "val_mse", "val_mae", "run_dir"]
            )
            writer.writeheader()
            writer.writerows(result["rows"])
        print(json.dumps({"setting": setting, "frozen": result["frozen"]}), flush=True)
    (root / "frozen_configs.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"frozen_configs": frozen}))


if __name__ == "__main__":
    main()
