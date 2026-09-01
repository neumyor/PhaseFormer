#!/usr/bin/env python3
"""Strict T28 global Golden validation driver (Stages A-D).

Follows docs/PhaseFormer_strict_t28_global_golden_plan.md.  Every search run is
validation-only, CUDA-required, resumable and uses metrics.csv as the single
comparison source.  Stage D uses --stage confirm --evaluate-test.

Usage:
  python scripts/run_strict_t28_global_golden.py \
      --stage stage_a_traffic --gpus 0,1,2,3 [--dry-run] [--num-workers 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MECHANISM = "pctf_anchor_repair_strict_t28"
LOOKBACK = 720
LOSS = "huber"
SEED_A = 2021
SEEDS_C = (2021, 2022)
SEEDS_D = (2021, 2022, 2023)

# Cycle periods frozen from earlier training-stage screens (all four horizons
# divide the period).  ETTm2=24 frozen by the pilot (0.087% < 0.2% tie-break);
# Traffic determined by Stage A.
FROZEN_CYCLES = {
    "ETTh1": 48,
    "ETTh2": 48,
    "ETTm1": 48,
    "ETTm2": 24,
    "Weather": 24,
    "Electricity": 12,
    "Traffic": None,  # Stage A
}

# Trust-region tiers: correction / deformation / global-level.
TIERS = {
    "C": dict(
        anchored_pctf_correction_max=0.25,
        anchored_pctf_deformation_max=0.10,
        anchored_pctf_global_level_max=0.05,
    ),
    "M": dict(
        anchored_pctf_correction_max=0.40,
        anchored_pctf_deformation_max=0.16,
        anchored_pctf_global_level_max=0.08,
    ),
    "S": dict(
        anchored_pctf_correction_max=0.50,
        anchored_pctf_deformation_max=0.20,
        anchored_pctf_global_level_max=0.10,
    ),
    "W": dict(
        anchored_pctf_correction_max=0.60,
        anchored_pctf_deformation_max=0.24,
        anchored_pctf_global_level_max=0.12,
    ),
}

DATASETS = tuple(FROZEN_CYCLES)
ALL_HORIZONS = (96, 192, 336, 720)
SCREEN_HORIZONS = (96, 336)

DEFAULT_OUTPUT = "research_runs/pctf_strict_t28_global_golden_v1"


def _root(args):
    value = Path(args.output_root)
    return value if value.is_absolute() else REPO_ROOT / value


def _command(args, dataset, horizon, cycle_period, stage, *, tier=None,
             percent=30, epochs=8, seed=SEED_A, evaluate_test=False,
             capacity="base"):
    command = [
        sys.executable, "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", stage,
        "--mechanism", MECHANISM,
        "--period", "24",
        "--lookback", str(LOOKBACK),
        "--cycle-period", str(cycle_period),
        "--percent", str(percent),
        "--max-epochs", str(epochs),
        "--seed", str(seed),
        "--loss", LOSS,
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0" if evaluate_test else "8",
        "--capacity", capacity,
        "--output-dir", str(_root(args)),
        "--require-cuda",
        "--resume",
    ]
    if tier is not None:
        command.extend(("--overrides", json.dumps(TIERS[tier], sort_keys=True)))
    if evaluate_test:
        command.append("--evaluate-test")
    if args.progress:
        command.append("--progress")
    return command


def stage_a_traffic_commands(args):
    """Stage A remainder: Traffic cycle candidates at H96/H336 (6 runs)."""
    return [
        _command(args, "Traffic", horizon, cycle, "period_screen")
        for horizon in SCREEN_HORIZONS
        for cycle in (12, 24, 48)
    ]


def stage_b_commands(args):
    """Stage B: 7 datasets x H96/H336 x 4 tiers = 56 runs."""
    frozen = _load_frozen(args)
    commands = []
    for dataset in DATASETS:
        cycle = frozen["cycle"][dataset]
        for horizon in SCREEN_HORIZONS:
            for tier in TIERS:
                commands.append(
                    _command(args, dataset, horizon, cycle, "mechanism_screen_1",
                             tier=tier)
                )
    return commands


def stage_c_commands(args):
    """Stage C: frozen (cycle+tier) x 4 horizons x seeds 2021/2022.

    Runs the frozen combo plus a C-tier reference at the same full protocol so
    the 16-ratio (4H x 2 seeds x MSE/MAE) regression check vs C is computable.
    Traffic froze at C, so its reference is the frozen run itself.
    """
    frozen = _load_frozen(args)
    commands = []
    for dataset in DATASETS:
        cycle = frozen["cycle"][dataset]
        tier = frozen["tier"][dataset]
        for horizon in ALL_HORIZONS:
            for seed in SEEDS_C:
                commands.append(
                    _command(args, dataset, horizon, cycle, "finalist",
                             tier=tier, percent=100, epochs=30, seed=seed)
                )
                if tier != "C":
                    # C-tier reference at the same protocol for the fallback rule.
                    commands.append(
                        _command(args, dataset, horizon, cycle, "finalist",
                                 tier="C", percent=100, epochs=30, seed=seed)
                    )
    return commands


def stage_d_commands(args):
    """Stage D: 28 settings x 3 seeds, full train + one test = 84 runs."""
    frozen = _load_frozen(args)
    commands = []
    for dataset in DATASETS:
        tier = frozen["tier"][dataset]
        cycle = frozen["cycle"][dataset]
        for horizon in ALL_HORIZONS:
            for seed in SEEDS_D:
                commands.append(
                    _command(args, dataset, horizon, cycle, "confirm",
                             tier=tier, percent=100, epochs=30, seed=seed,
                             evaluate_test=True)
                )
    return commands


def _load_frozen(args):
    path = _root(args) / "frozen_decisions.json"
    if not path.is_file():
        raise SystemExit(f"frozen_decisions.json not found: {path}")
    return json.loads(path.read_text())


def _run(commands, dry_run):
    print(f"commands={len(commands)}", flush=True)
    for command in commands:
        print("  " + " ".join(command), flush=True)
    if dry_run:
        return 0
    for index, command in enumerate(commands, 1):
        print(f"RUN {index}/{len(commands)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=(
        "stage_a_traffic", "stage_b", "stage_c", "stage_d",
    ))
    parser.add_argument("--gpus")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generators = {
        "stage_a_traffic": stage_a_traffic_commands,
        "stage_b": stage_b_commands,
        "stage_c": stage_c_commands,
        "stage_d": stage_d_commands,
    }
    commands = generators[args.stage](args)

    if not args.gpus:
        return _run(commands, args.dry_run)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    per_gpu = {gpu: [] for gpu in gpus}
    for index, command in enumerate(commands):
        per_gpu[gpus[index % len(gpus)]].append(command)
    marker_dir = _root(args) / f"driver_{args.stage}"
    marker_dir.mkdir(parents=True, exist_ok=True)
    processes = {}
    for gpu, slice_commands in per_gpu.items():
        script = marker_dir / f"gpu{gpu}.sh"
        lines = ["#!/usr/bin/env bash", "set -e", f"export CUDA_VISIBLE_DEVICES={gpu}"]
        for command in slice_commands:
            lines.append(" ".join(shlex.quote(part) for part in command))
        lines.append(f"touch {marker_dir / f'gpu{gpu}.done'}")
        script.write_text("\n".join(lines) + "\n")
        proc = subprocess.Popen(["bash", str(script)], cwd=REPO_ROOT)
        processes[gpu] = proc
        print(f"GPU {gpu}: {len(slice_commands)} commands, pid {proc.pid}", flush=True)
    failed = False
    for gpu, proc in processes.items():
        returncode = proc.wait()
        print(f"GPU {gpu}: finished rc={returncode}", flush=True)
        failed = failed or returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
