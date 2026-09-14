#!/usr/bin/env python3
"""Complete an audited rank-sweep matrix with explicit one-run jobs.

This deliberately launches ``search_phaseformer.py`` directly for each
semantic cell.  It does not call a shared sweep runner, so one cell cannot
silently launch or duplicate another cell.
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
SEARCH = ROOT / "scripts" / "search_phaseformer.py"

FROZEN = {
    ("ETTh2", 96): (0.5, 0.001),
    ("ETTh2", 720): (0.5, 0.001),
    ("ETTm2", 96): (0.5, 0.0003),
    ("ETTm2", 192): (0.2, 0.001),
    ("Weather", 96): (0.2, 0.0003),
    ("Weather", 192): (0.5, 0.001),
    ("Electricity", 336): (0.5, 0.001),
}

RANKS = {
    96: {"q=0.25": 24, "q=0.125": 12, "q=0.0625": 6, "q=0.03125": 3},
    192: {"q=0.25": 48, "q=0.125": 24, "q=0.0625": 12, "q=0.03125": 6},
    336: {"q=0.25": 84, "q=0.125": 42, "q=0.0625": 21, "q=0.03125": 10},
    720: {"q=0.25": 180, "q=0.125": 90, "q=0.0625": 45, "q=0.03125": 22},
}

# Frozen from the v4 audit.  This list is the only source of jobs for the
# repair pass; complete cells are never resubmitted.
MISSING_CELLS = [
    ("Weather", 96, 2022, "q=0.03125"),
    *[
        ("Weather", 192, 2023, config)
        for config in ("q=0.125", "q=0.0625", "q=0.03125")
    ],
    *[
        ("Electricity", 336, seed, config)
        for seed in (2022, 2023)
        for config in ("direct", "q=0.25", "q=0.125", "q=0.0625", "q=0.03125")
    ],
]


def expected_config(dataset: str, horizon: int, seed: int, config: str) -> dict:
    gate, learning_rate = FROZEN[(dataset, horizon)]
    overrides = {
        "weak_period_residual_gate_init": gate,
        "learning_rate": learning_rate,
    }
    if config == "direct":
        overrides["weak_period_residual_head_type"] = "shared"
    else:
        overrides.update(
            {
                "weak_period_residual_head_type": "pooled_lowrank",
                "weak_period_residual_pool_factor": 1,
                "weak_period_residual_rank": RANKS[horizon][config],
                "weak_period_residual_smooth_ratio": 0.0,
                "weak_period_residual_smooth_window": 24,
            }
        )
    return {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "config": config,
        "gate_init": gate,
        "learning_rate": learning_rate,
        "overrides": overrides,
    }


def build_command(job: dict, output_root: str, num_workers: int) -> list[str]:
    return [
        sys.executable,
        str(SEARCH),
        "--dataset",
        job["dataset"],
        "--horizon",
        str(job["horizon"]),
        "--stage",
        "confirm",
        "--mechanism",
        "weak_residual",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        "30",
        "--seed",
        str(job["seed"]),
        "--loss",
        "huber",
        "--output-dir",
        output_root,
        "--num-workers",
        str(num_workers),
        "--bad-case-limit",
        "0",
        "--require-cuda",
        "--evaluate-test",
        "--overrides",
        json.dumps(job["overrides"], sort_keys=True),
    ]


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        return all(
            math.isfinite(float(row[key]))
            for key in ("val_mse", "val_mae", "test_mse", "test_mae")
        )
    except (OSError, StopIteration, KeyError, TypeError, ValueError):
        return False


def matches_job(run_dir: Path, job: dict) -> bool:
    try:
        config = json.loads((run_dir / "config.json").read_text())
        hp = config["hyperparams"]
        if (
            config.get("dataset") != job["dataset"]
            or int(config.get("horizon", -1)) != job["horizon"]
            or int(config.get("seed", -1)) != job["seed"]
            or abs(float(hp["weak_period_residual_gate_init"]) - job["gate_init"]) > 1e-9
            or abs(float(hp["learning_rate"]) - job["learning_rate"]) > 1e-9
        ):
            return False
        if job["config"] == "direct":
            return hp.get("weak_period_residual_head_type") == "shared"
        return (
            hp.get("weak_period_residual_head_type") == "pooled_lowrank"
            and int(hp.get("weak_period_residual_pool_factor", -1)) == 1
            and int(hp.get("weak_period_residual_rank", -1))
            == job["overrides"]["weak_period_residual_rank"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_manifest(
    output_root: str,
    existing_roots: list[str],
    num_workers: int,
) -> list[dict]:
    jobs = [expected_config(*cell) for cell in MISSING_CELLS]
    if len(jobs) != 14:
        raise RuntimeError(f"expected 14 explicit repair jobs, got {len(jobs)}")
    path = ROOT / output_root / "missing_cells_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "protocol": "explicit missing-cell repair; no shared runner",
                "output_root": output_root,
                "existing_roots": existing_roots,
                "num_workers": num_workers,
                "resume": False,
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return jobs


def dispatch(
    jobs: list[dict],
    output_root: str,
    existing_roots: list[str],
    gpus: list[int],
    retries: int,
    num_workers: int,
) -> None:
    pending = list(jobs)
    active: dict[int, tuple[dict, subprocess.Popen[str], int]] = {}
    attempts: dict[tuple, int] = {}
    while pending or active:
        while pending and len(active) < len(gpus):
            gpu = next(gpu for gpu in gpus if gpu not in active)
            job = pending.pop(0)
            key = (job["dataset"], job["horizon"], job["seed"], job["config"])
            existing = []
            for candidate_root in [*existing_roots, output_root]:
                root = ROOT / candidate_root / "runs"
                existing.extend(
                    path / "metrics.csv"
                    for path in root.glob(
                        f"confirm_{job['dataset'].lower()}_h{job['horizon']}_*s{job['seed']}_*/"
                    )
                    if matches_job(path, job)
                )
            if any(is_complete(path) for path in existing):
                print(json.dumps({"event": "skip_complete", "key": key}), flush=True)
                continue
            attempts[key] = attempts.get(key, 0) + 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "key": key,
                        "gpu": gpu,
                        "attempt": attempts[key],
                        "gate_init": job["gate_init"],
                        "learning_rate": job["learning_rate"],
                        "head_type": job["overrides"]["weak_period_residual_head_type"],
                        "rank": job["overrides"].get("weak_period_residual_rank", ""),
                    }
                ),
                flush=True,
            )
            active[gpu] = (
                job,
                subprocess.Popen(
                    build_command(job, output_root, num_workers),
                    cwd=ROOT,
                    env=env,
                ),
                attempts[key],
            )
        finished = []
        for gpu, (job, process, attempt) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(gpu)
            key = (job["dataset"], job["horizon"], job["seed"], job["config"])
            if code != 0 and attempt <= retries:
                pending.append(job)
                print(json.dumps({"event": "retry", "key": key, "code": code}), flush=True)
            elif code != 0:
                raise RuntimeError(f"repair job failed after retries: {key}, code={code}")
            else:
                print(json.dumps({"event": "finished", "key": key}), flush=True)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1",
    )
    parser.add_argument(
        "--existing-roots",
        default="research_runs/rank_sweep_2_multiseed_stage1_20260914_v4",
        help="Comma-separated result roots checked before launching a cell.",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item]
    existing_roots = [
        item for item in args.existing_roots.split(",") if item
    ]
    if not gpus:
        parser.error("--gpus must not be empty")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    jobs = write_manifest(args.output_root, existing_roots, args.num_workers)
    print(json.dumps({"jobs": len(jobs), "manifest": str(ROOT / args.output_root / "missing_cells_manifest.json")}), flush=True)
    if not args.manifest_only:
        dispatch(
            jobs,
            args.output_root,
            existing_roots,
            gpus,
            args.retries,
            args.num_workers,
        )


if __name__ == "__main__":
    main()
