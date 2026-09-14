#!/usr/bin/env python3
"""Run the audited 14-setting, five-config multi-seed rank sweep.

The setting table is deliberately explicit.  This avoids positional shell
arguments silently shifting a seed into the gate or learning-rate slot.  The
outer dispatcher is intentionally serial: one inner runner owns all GPUs at a
time, so completion order cannot cause resource-group reuse.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_joint_pooled_lowrank_phase_a.py"
RANKS = "0.25,0.125,0.0625,0.03125"
EXPECTED_CONFIGS = [
    "direct_nlinear",
    "pool1_q0.25",
    "pool1_q0.125",
    "pool1_q0.0625",
    "pool1_q0.03125",
]

SETTING_TABLE = {
    ("ETTh2", 96): {"gate": 0.5, "learning_rate": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "learning_rate": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "learning_rate": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "learning_rate": 0.001},
    ("Weather", 96): {"gate": 0.2, "learning_rate": 0.0003},
    ("Weather", 192): {"gate": 0.5, "learning_rate": 0.001},
    ("Electricity", 336): {"gate": 0.5, "learning_rate": 0.001},
}


def build_jobs(output_root: str, gpu_groups: list[list[int]], seeds: list[int]) -> list[dict]:
    jobs = []
    for seed in seeds:
        for (dataset, horizon), frozen in SETTING_TABLE.items():
            jobs.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "gate_init": frozen["gate"],
                    "learning_rate": frozen["learning_rate"],
                    "expected_configs": EXPECTED_CONFIGS,
                    "gpus": gpu_groups[len(jobs) % len(gpu_groups)],
                    "command": [
                        sys.executable,
                        str(RUNNER),
                        "--dataset",
                        dataset,
                        "--horizon",
                        str(horizon),
                        "--seed",
                        str(seed),
                        "--pool-factors",
                        "1",
                        "--relative-ranks",
                        RANKS,
                        "--exact-rank",
                        "--evaluate-test",
                        "--skip-phase-only",
                        "--max-epochs",
                        "30",
                        "--overrides",
                        json.dumps(
                            {
                                "weak_period_residual_gate_init": frozen["gate"],
                                "learning_rate": frozen["learning_rate"],
                            },
                            sort_keys=True,
                        ),
                        "--gpus",
                        ",".join(str(gpu) for gpu in gpu_groups[len(jobs) % len(gpu_groups)]),
                        "--num-workers",
                        "4",
                        "--retries",
                        "1",
                        "--poll-seconds",
                        "10",
                        "--output-root",
                        output_root,
                    ],
                }
            )
    return jobs


def write_manifest(path: Path, output_root: str, gpu_groups: list[list[int]]) -> list[dict]:
    jobs = build_jobs(output_root, gpu_groups, [2022, 2023])
    if len(jobs) != 14:
        raise RuntimeError(f"expected 14 setting-seed jobs, got {len(jobs)}")
    for job in jobs:
        if len(job["expected_configs"]) != 5:
            raise RuntimeError(f"invalid config matrix for {job}")
    payload = {
        "protocol": "v5 audited multi-seed low-rank sweep",
        "output_root": output_root,
        "lookback": 720,
        "period": 24,
        "loss": "huber",
        "max_epochs": 30,
        "evaluate_test": True,
        "phase_only": "reused Golden; not trained",
        "configs": EXPECTED_CONFIGS,
        "jobs": jobs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return jobs


def dispatch(jobs: list[dict], retries: int, poll_seconds: int) -> None:
    pending = list(jobs)
    active: dict[int, tuple[dict, subprocess.Popen[str], int]] = {}
    attempts: dict[tuple[str, int, int], int] = {}
    while pending or active:
        free_groups = [
            index for index in range(len({tuple(job["gpus"]) for job in jobs}))
            if index not in active
        ]
        while pending and free_groups:
            group_index = free_groups.pop(0)
            job = pending.pop(0)
            key = (job["dataset"], job["horizon"], job["seed"])
            attempts[key] = attempts.get(key, 0) + 1
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "dataset": job["dataset"],
                        "horizon": job["horizon"],
                        "seed": job["seed"],
                        "gate_init": job["gate_init"],
                        "learning_rate": job["learning_rate"],
                        "attempt": attempts[key],
                        "gpus": job["gpus"],
                    }
                ),
                flush=True,
            )
            active[group_index] = (
                job,
                subprocess.Popen(job["command"], cwd=ROOT, text=True),
                attempts[key],
            )
        finished = []
        for group_index, (job, process, attempt) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(group_index)
            key = (job["dataset"], job["horizon"], job["seed"])
            if code != 0 and attempt <= retries:
                pending.append(job)
                print(json.dumps({"event": "retry", "key": key, "code": code}), flush=True)
            elif code != 0:
                raise RuntimeError(f"v4 job failed after retries: {key}, code={code}")
            else:
                print(json.dumps({"event": "finished", "key": key}), flush=True)
        for group_index in finished:
            del active[group_index]
        if active:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="research_runs/rank_sweep_2_multiseed_stage1_20260914_v5",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    gpu_groups = [list(range(8))]
    manifest = ROOT / args.output_root / "v5_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    jobs = write_manifest(manifest, args.output_root, gpu_groups)
    print(json.dumps({"manifest": str(manifest), "jobs": len(jobs)}), flush=True)
    if not args.manifest_only:
        dispatch(jobs, args.retries, args.poll_seconds)


if __name__ == "__main__":
    main()
