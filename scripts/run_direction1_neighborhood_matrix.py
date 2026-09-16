#!/usr/bin/env python3
"""Run the seven-setting direction-1 neighborhood test sweep.

This is an explicitly disclosed test-set-selection experiment.

``sweep`` trains six arms for seed 2021 on all seven settings:

    direct_nlinear, rrr_direction_1_2, cone_k1, cone_k2, cone_k4, cone_k8

Every run selects its checkpoint on validation and reads test once.  The
separate selection script then chooses one width per dataset from test metrics.

``confirm`` reads that frozen dataset-width map and trains
direct/RRR-2/Cone-1/selected-cone for seeds 2022 and 2023.  Cone-1 and the
selected cone are deduplicated when the selected width is one.  These extra
seeds measure stability after test-set selection; they are not an unbiased
confirmation.
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

ALL_SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
)
SWEEP_WIDTHS = (1, 2, 4, 8)
SWEEP_SEEDS = (2021,)
CONFIRM_SEEDS = (2022, 2023)

FROZEN = {
    ("ETTh2", 96): {"gate": 0.5, "lr": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "lr": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "lr": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "lr": 0.001},
    ("Weather", 96): {"gate": 0.2, "lr": 0.0003},
    ("Weather", 192): {"gate": 0.5, "lr": 0.001},
    ("Electricity", 336): {"gate": 0.5, "lr": 0.001},
}


def parse_widths(raw: str) -> tuple[int, ...]:
    widths = tuple(
        sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    )
    if not widths or any(width < 1 for width in widths):
        raise ValueError("all widths must be positive")
    return widths


def cone_arm(width: int) -> str:
    return f"direction1_neighborhood_k{width}"


def basis_path(
    projector_dir: str, dataset: str, horizon: int, arm: str, width: int | None
) -> str:
    if arm == "rrr_direction_1_2":
        name = "Qrrr2"
    elif arm.startswith("direction1_neighborhood_k") and width is not None:
        name = f"Qcone{width}"
    else:
        raise ValueError(f"arm {arm!r} does not use a frozen basis")
    return f"{projector_dir}/{dataset}_{horizon}_{name}.npy"


def arm_command(
    cell: dict,
    projector_dir: str,
    output_root: str,
) -> list[str]:
    dataset = cell["dataset"]
    horizon = cell["horizon"]
    seed = cell["seed"]
    arm = cell["arm"]
    width = cell.get("width")
    frozen = FROZEN[(dataset, horizon)]
    overrides = {
        "learning_rate": frozen["lr"],
        "weak_period_residual_gate_init": frozen["gate"],
        "weak_period_residual_head_type": "shared",
        "weak_residual_projection_arm": arm,
    }
    argv = [
        str(RUNNER),
        "--output-dir",
        output_root,
        "--dataset",
        dataset,
        "--horizon",
        str(horizon),
        "--stage",
        "confirm",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        "30",
        "--seed",
        str(seed),
        "--loss",
        "huber",
        "--learning-rate",
        str(frozen["lr"]),
        "--mechanism",
        "weak_residual",
        "--require-cuda",
        "--resume",
        "--num-workers",
        "4",
        "--bad-case-limit",
        "0",
        "--evaluate-test",
    ]
    if arm != "direct_nlinear":
        overrides["weak_residual_projection"] = "frozen_subspace"
        argv += [
            "--basis",
            basis_path(projector_dir, dataset, horizon, arm, width),
        ]
    argv += ["--overrides", json.dumps(overrides, sort_keys=True)]
    return argv


def sweep_cells(widths: tuple[int, ...] = SWEEP_WIDTHS) -> list[dict]:
    cells = []
    for dataset, horizon in ALL_SETTINGS:
        for seed in SWEEP_SEEDS:
            cells.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "arm": "direct_nlinear",
                    "width": None,
                }
            )
            cells.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "arm": "rrr_direction_1_2",
                    "width": 2,
                }
            )
            for width in widths:
                cells.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "arm": cone_arm(width),
                        "width": width,
                    }
                )
    return cells


def load_dataset_widths(selection_file: Path) -> dict[str, int]:
    if not selection_file.is_file():
        raise FileNotFoundError(f"test selection file not found: {selection_file}")
    payload = json.loads(selection_file.read_text())
    if payload.get("status") != "selected":
        raise RuntimeError(f"selection status is {payload.get('status')!r}")
    widths = {
        str(dataset): int(width)
        for dataset, width in payload.get("selected_width_by_dataset", {}).items()
    }
    expected = {dataset for dataset, _ in ALL_SETTINGS}
    if set(widths) != expected:
        raise ValueError(
            f"selection must contain datasets {sorted(expected)}, got {sorted(widths)}"
        )
    if any(width not in SWEEP_WIDTHS for width in widths.values()):
        raise ValueError(f"selection contains a width outside {SWEEP_WIDTHS}")
    return widths


def confirm_cells(selected_widths: dict[str, int]) -> list[dict]:
    cells = []
    for dataset, horizon in ALL_SETTINGS:
        for seed in CONFIRM_SEEDS:
            widths = (1,) if selected_widths[dataset] == 1 else (
                1,
                selected_widths[dataset],
            )
            cells.extend(
                [
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "arm": "direct_nlinear",
                        "width": None,
                    },
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "arm": "rrr_direction_1_2",
                        "width": 2,
                    },
                    *[
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                            "arm": cone_arm(width),
                            "width": width,
                        }
                        for width in widths
                    ],
                ]
            )
    return cells


def cell_key(cell: dict) -> str:
    width = "full" if cell["width"] is None else f"k{cell['width']}"
    return (
        f"{cell['dataset']}-{cell['horizon']}-s{cell['seed']}-"
        f"{cell['arm']}-{width}"
    )


def dispatch(
    cells: list[dict],
    gpus: list[int],
    output_root: str,
    projector_dir: str,
    retries: int,
    poll_seconds: int,
) -> tuple[list[str], list[dict]]:
    pending = list(cells)
    active: dict[int, tuple[dict, subprocess.Popen, object]] = {}
    attempts: dict[str, int] = {}
    completed: list[str] = []
    failed: list[dict] = []
    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell_key(cell)
            attempts[key] = attempts.get(key, 0) + 1
            argv = arm_command(cell, projector_dir, output_root)
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
            active[gpu] = (cell, process, log)
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

        finished = []
        for gpu, (cell, process, log) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(gpu)
            key = cell_key(cell)
            log.close()
            if return_code == 0:
                completed.append(key)
                print(json.dumps({"event": "done", "cell": key, "gpu": gpu}))
            elif attempts[key] <= retries:
                pending.append(cell)
                print(
                    json.dumps(
                        {
                            "event": "retry",
                            "cell": key,
                            "gpu": gpu,
                            "return_code": return_code,
                        }
                    )
                )
            else:
                failed.append({"cell": key, "return_code": return_code})
                print(
                    json.dumps(
                        {
                            "event": "failed",
                            "cell": key,
                            "return_code": return_code,
                        }
                    )
                )
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(poll_seconds)
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "confirm"), required=True)
    parser.add_argument("--widths", default="1,2,4,8")
    parser.add_argument("--selection-file", default="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument(
        "--output-root", default="research_runs/direction1_neighborhood_v1"
    )
    parser.add_argument("--projector-dir", default="")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    widths = parse_widths(args.widths)
    if tuple(widths) != SWEEP_WIDTHS:
        raise ValueError(
            f"pre-registered sweep widths are {SWEEP_WIDTHS}, got {tuple(widths)}"
        )
    root = ROOT / args.output_root
    projector_dir = args.projector_dir or f"{args.output_root}/projectors"
    selection_file = (
        Path(args.selection_file)
        if args.selection_file
        else root / "test_selection.json"
    )
    if not selection_file.is_absolute():
        selection_file = ROOT / selection_file

    if args.stage == "sweep":
        selected_widths = {}
        cells = sweep_cells(widths)
    else:
        selected_widths = load_dataset_widths(selection_file)
        cells = confirm_cells(selected_widths)

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    manifest = {
        "protocol": "direction1-bootstrap-neighborhood-test-selection-v1",
        "stage": args.stage,
        "test_set_selection": True,
        "selection_scope": "one shared width per dataset",
        "lookback": 720,
        "period": 24,
        "loss": "huber",
        "max_epochs": 30,
        "sweep_widths": list(widths),
        "selected_width_by_dataset": selected_widths,
        "selection_file": str(selection_file) if args.stage == "confirm" else "",
        "gpus": gpus,
        "projector_dir": projector_dir,
        "frozen_setting_table": {
            f"{dataset}-{horizon}": values
            for (dataset, horizon), values in FROZEN.items()
        },
        "cells": [
            {
                **cell,
                "key": cell_key(cell),
                "command": arm_command(cell, projector_dir, args.output_root),
            }
            for cell in cells
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"{args.stage}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "planned",
                "stage": args.stage,
                "cells": len(cells),
                "manifest": str(manifest_path),
            }
        )
    )
    if args.dry_run:
        for cell in cells:
            print(json.dumps({"cell": cell_key(cell)}))
        return

    completed, failed = dispatch(
        cells,
        gpus,
        args.output_root,
        projector_dir,
        args.retries,
        args.poll_seconds,
    )
    summary = {
        "stage": args.stage,
        "cells_planned": len(cells),
        "completed": completed,
        "failed": failed,
    }
    (root / f"{args.stage}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"event": "stage_finished", **summary}, indent=2))
    if failed:
        raise SystemExit(f"stage {args.stage} had failed cells: {failed}")


if __name__ == "__main__":
    main()
