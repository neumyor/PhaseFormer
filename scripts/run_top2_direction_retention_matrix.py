#!/usr/bin/env python3
"""Multi-GPU driver for the top-2 predictive-direction retention experiment.

Schedules every (setting, seed, arm) cell of the plan onto one GPU at a time:

    phase_only            mechanism=no_residual               (control)
    direct_nlinear        weak_residual + head_type=shared     (control)
    keep_direction_1      weak_residual + frozen Q1            (variant V1)
    keep_direction_1_2    weak_residual + frozen Q12           (variant V2)

Exactly one training run per GPU; a GPU takes the next pending cell as soon as
its current cell finishes.  Cells are resumed idempotently, so re-running the
driver after an interruption only launches what is still missing.

Frozen per-setting protocol (gate init / learning rate) is taken from the
existing conditioned rank sweep so the new arms are comparable to the controls:

    ETTh2-96 gate 0.5 lr 1e-3 | ETTh2-720 gate 0.5 lr 1e-3
    ETTm2-96 gate 0.5 lr 3e-4 | ETTm2-192 gate 0.2 lr 1e-3
    Weather-96 gate 0.2 lr 3e-4 | Weather-192 gate 0.5 lr 1e-3

Usage::

    python scripts/run_top2_direction_retention_matrix.py \
        --stage a --gpus 0,1,2,3,4,5 --output-root research_runs/top2_direction_retention_v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_top2_direction_retention.py"

# The six settings that already have RRR direction analysis (plan section 3).
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
)

# Frozen protocol from scripts/run_rank_sweep_multiseed_v4.py SETTING_TABLE.
FROZEN = {
    ("ETTh2", 96): {"gate": 0.5, "lr": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "lr": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "lr": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "lr": 0.001},
    ("Weather", 96): {"gate": 0.2, "lr": 0.0003},
    ("Weather", 192): {"gate": 0.5, "lr": 0.001},
}

ARMS = ("phase_only", "direct_nlinear", "keep_direction_1", "keep_direction_1_2")


def arm_command(arm, dataset, horizon, seed, projector_dir, evaluate_test=False):
    """Build the runner argv (without the interpreter) for one cell."""
    frozen = FROZEN[(dataset, horizon)]
    overrides = {
        "weak_period_residual_gate_init": frozen["gate"],
        "learning_rate": frozen["lr"],
        # Records the arm inside the config so the run id and the auditable
        # config.json say which variant produced the checkpoint.
        "weak_residual_projection_arm": arm,
    }
    argv = [
        str(RUNNER),
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm",
        "--lookback", "720",
        "--period", "24",
        "--max-epochs", "30",
        "--seed", str(seed),
        "--loss", "huber",
        "--learning-rate", str(frozen["lr"]),
        "--require-cuda",
        "--resume",
        "--num-workers", "4",
        "--bad-case-limit", "0",
    ]
    if arm == "phase_only":
        argv += ["--mechanism", "no_residual"]
    else:
        argv += ["--mechanism", "weak_residual"]
        overrides["weak_period_residual_head_type"] = "shared"
        if arm in ("keep_direction_1", "keep_direction_1_2"):
            name = "Q1" if arm == "keep_direction_1" else "Q12"
            argv += [
                "--basis",
                f"{projector_dir}/{dataset}_{horizon}_{name}.npy",
            ]
            overrides["weak_residual_projection"] = "frozen_subspace"
    if evaluate_test:
        argv += ["--evaluate-test"]
    argv += ["--overrides", json.dumps(overrides, sort_keys=True)]
    return argv


def build_cells(stage: str):
    cells = []
    if stage in ("a", "all"):
        for dataset, horizon in SETTINGS:
            for arm in ARMS:
                cells.append(
                    {"dataset": dataset, "horizon": horizon, "seed": 2021, "arm": arm}
                )
    if stage in ("b", "all"):
        for seed in (2022, 2023):
            for dataset, horizon in SETTINGS:
                for arm in ("keep_direction_1", "keep_direction_1_2", "phase_only"):
                    cells.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                            "arm": arm,
                        }
                    )
    return cells


def cell_key(cell):
    return f"{cell['dataset']}-{cell['horizon']}-s{cell['seed']}-{cell['arm']}"


def dispatch(cells, gpus, output_root, projector_dir, retries, poll,
             evaluate_test=False):
    pending = list(cells)
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    attempts: dict[str, int] = {}
    completed, failed = [], []
    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell_key(cell)
            attempts[key] = attempts.get(key, 0) + 1
            argv = arm_command(
                cell["arm"], cell["dataset"], cell["horizon"], cell["seed"],
                projector_dir, args.evaluate_test,
            )
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_dir = Path(output_root) / "_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"{key}.log", "w")
            process = subprocess.Popen(
                [sys.executable, *argv],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(
                json.dumps(
                    {
                        "event": "launch",
                        "cell": key,
                        "gpu": gpu,
                        "attempt": attempts[key],
                    }
                ),
                flush=True,
            )
            active[gpu] = (cell, process, log)
        finished = []
        for gpu, (cell, process, log) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(gpu)
            key = cell_key(cell)
            log.close()
            if code == 0:
                completed.append(key)
                print(json.dumps({"event": "done", "cell": key, "gpu": gpu}), flush=True)
            elif attempts[key] <= retries:
                pending.append(cell)
                print(
                    json.dumps(
                        {"event": "retry", "cell": key, "gpu": gpu, "code": code}
                    ),
                    flush=True,
                )
            else:
                failed.append({"cell": key, "return_code": code})
                print(
                    json.dumps({"event": "failed", "cell": key, "code": code}),
                    flush=True,
                )
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(poll)
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["a", "b", "all"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument(
        "--output-root", default="research_runs/top2_direction_retention_v1"
    )
    parser.add_argument("--projector-dir", default="")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="also read test in the same run; leave off during Stage A/B so test "
        "is read exactly once, after every checkpoint is frozen",
    )
    parser.add_argument(
        "--skip-reused",
        action="store_true",
        help="drop direct_nlinear from the new-training matrix (it is reused "
        "from the audited rank sweep instead)",
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    projector_dir = args.projector_dir or f"{args.output_root}/projectors"

    cells = build_cells(args.stage)
    if args.skip_reused:
        cells = [c for c in cells if c["arm"] != "direct_nlinear"]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    manifest = {
        "protocol": "top2-direction-retention-v1",
        "stage": args.stage,
        "lookback": 720,
        "period": 24,
        "loss": "huber",
        "max_epochs": 30,
        "gpus": gpus,
        "frozen_setting_table": {
            f"{d}-{h}": v for (d, h), v in FROZEN.items()
        },
        "projector_dir": projector_dir,
        "cells": [
            {**cell, "key": cell_key(cell), "command": arm_command(
                cell["arm"], cell["dataset"], cell["horizon"], cell["seed"],
                projector_dir, args.evaluate_test)}
            for cell in cells
        ],
    }
    root = ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / f"stage_{args.stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"event": "planned", "cells": len(cells), "gpus": gpus}))

    if args.dry_run:
        for cell in cells:
            print(json.dumps({"cell": cell_key(cell)}))
        return

    completed, failed = dispatch(
        cells, gpus, args.output_root, projector_dir, args.retries,
        args.poll_seconds, args.evaluate_test,
    )
    summary = {
        "stage": args.stage,
        "cells_planned": len(cells),
        "completed": completed,
        "failed": failed,
    }
    (root / f"stage_{args.stage}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"event": "stage_finished", **summary}, indent=2))
    if failed:
        raise SystemExit(f"stage {args.stage} had failed cells: {failed}")



if __name__ == "__main__":
    main()
