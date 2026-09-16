#!/usr/bin/env python3
"""Read test exactly once for every frozen checkpoint of this experiment.

Plan section 7 makes the test read a distinct, final step: all training and all
audits must have passed, and no test result may then change the projector,
training configuration or model structure.  This script is that single step.

For each requested (setting, seed, arm) it rebuilds the model exactly as the
training run did (including reinstalling the same frozen projector for V1/V2),
restores the best-validation checkpoint, and evaluates once on the test split.
It also records two diagnostics the plan asks for:

  * ``nlinear_mse`` / ``nlinear_mae`` — the NLinear branch's own output error,
    i.e. the head read out only through the fixed fusion gate;
  * ``gate_value`` — the learned scalar fusion weight of the NLinear branch.

Reused ``direct_nlinear`` runs keep the test numbers their own run already
recorded; this script only verifies their presence and never re-reads them.

Usage::

    python scripts/read_top2_direction_retention_test.py \
        --root research_runs/top2_direction_retention_v1 --gpus 0,1,2,3,4,5
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
)
SEEDS = (2021, 2022, 2023)
ARMS = ("phase_only", "direct_nlinear", "keep_direction_1", "keep_direction_1_2")

# Recomputing validation from the restored checkpoint cannot be bit-identical:
# the evaluation dataloader differs in worker count from the training run's, so
# float summation order differs.  Anything beyond float-level drift means the
# checkpoint or protocol does not match and the cell must not be reported.
VAL_REPRODUCE_WARN = 1e-4
VAL_REPRODUCE_TOL = 1e-3
FROZEN = {
    ("ETTh2", 96): {"gate": 0.5, "lr": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "lr": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "lr": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "lr": 0.001},
    ("Weather", 96): {"gate": 0.2, "lr": 0.0003},
    ("Weather", 192): {"gate": 0.5, "lr": 0.001},
}


def find_run(root: Path, dataset, horizon, seed, arm):
    """Locate the run directory for one cell by reading each config.json."""
    for config_path in sorted((root / "runs").glob("*/config.json")):
        config = json.loads(config_path.read_text())
        hyper = config.get("hyperparams", {})
        if (
            config.get("dataset") != dataset
            or int(config.get("horizon", -1)) != horizon
            or int(config.get("seed", -1)) != seed
        ):
            continue
        found_arm = hyper.get("weak_residual_projection_arm")
        if found_arm is None:
            found_arm = (
                "phase_only"
                if config.get("mechanism") == "no_residual"
                else "direct_nlinear"
            )
        if found_arm != arm:
            continue
        return config_path.parent, config
    return None, None


def build_model(config, checkpoint_path):
    """Rebuild the exact trained model, then restore its best checkpoint."""
    from src.dataset.data_factory import data_provider
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import (
        PhaseFormerPresetConfig,
        make_exp_args,
    )

    hyper = dict(config["hyperparams"])
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hyper,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = 4
    train_set, _ = data_provider(exp_args.dataset_args, "train")
    if hasattr(train_set, "data_stamp"):
        hyper["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(
        PhaseFormerPresetConfig(
            exp_args, config["lookback"], config["horizon"], hyper
        )
    )
    if hyper.get("weak_residual_projection") == "frozen_subspace":
        basis_path = Path(hyper["_projection_basis"])
        model.install_projection_basis(
            torch.as_tensor(np.load(basis_path), dtype=torch.float32),
            source=str(basis_path),
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompat.unexpected_keys}")
    return model, exp_args


def evaluate_once(model, loader, split, target_var_index, head_gate):
    """MSE/MAE of the fused output and of the NLinear branch alone."""
    model.eval()
    device = next(model.parameters()).device
    fused_sq = fused_abs = branch_sq = branch_abs = 0.0
    count = 0
    branch_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            pred = out[:, -model.pred_len :, :]
            true = batch_y.float()[:, -model.pred_len :, :]
            branch = model.last_residual_forecast
            if target_var_index != -1:
                true = true[:, :, target_var_index : target_var_index + 1]
                if branch is not None:
                    branch = branch[:, :, target_var_index : target_var_index + 1]
            err = pred - true
            fused_sq += float(err.pow(2).sum())
            fused_abs += float(err.abs().sum())
            count += err.numel()
            if branch is not None:
                branch_err = branch - true
                branch_sq += float(branch_err.pow(2).sum())
                branch_abs += float(branch_err.abs().sum())
                branch_count += branch_err.numel()
    result = {
        f"{split}_mse": fused_sq / max(count, 1),
        f"{split}_mae": fused_abs / max(count, 1),
        "nlinear_mse": (branch_sq / branch_count) if branch_count else None,
        "nlinear_mae": (branch_abs / branch_count) if branch_count else None,
        "gate_value": head_gate,
        "eval_count": count,
    }
    return result


def resolve_checkpoint(record, run_dir: Path):
    """Find the best-validation checkpoint of a run.

    ``metrics.csv`` stores a repo-relative path recorded when the run finished.
    The runs are relocatable, so fall back to resolving the same relative path
    against the run's own directory before giving up.
    """
    recorded = record.get("checkpoint") or ""
    if not recorded:
        return None
    raw = Path(recorded)
    candidates = [ROOT / raw, run_dir / raw]
    # Strip a leading run-directory name (e.g. a relocated ``runs/<id>/...``).
    parts = raw.parts
    for index, part in enumerate(parts):
        if part == "runs" and index + 1 < len(parts):
            candidates.append(run_dir / Path(*parts[index + 1 :]))
            break
    candidates.append(run_dir / "attempts" / "001" / "checkpoints" / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_one(cell, root: Path, projector_dir: Path) -> dict:
    from src.dataset.data_factory import data_provider

    dataset, horizon, seed, arm = cell
    run_dir, config = find_run(root, dataset, horizon, seed, arm)
    if run_dir is None:
        return {"cell": f"{dataset}-{horizon}-s{seed}-{arm}", "status": "missing_run"}
    metrics_path = run_dir / "metrics.csv"
    record = next(csv.DictReader(metrics_path.open(newline="")))
    if arm == "direct_nlinear":
        # Reused control: its test numbers were produced by its own run.
        return {
            "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
            "status": "reused" if record.get("test_mse") else "missing_test",
            "test_mse": record.get("test_mse"),
            "test_mae": record.get("test_mae"),
            "run_dir": str(run_dir.relative_to(ROOT)),
        }
    if record.get("test_mse"):
        return {
            "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
            "status": "already_read",
            "test_mse": record.get("test_mse"),
            "test_mae": record.get("test_mae"),
            "run_dir": str(run_dir.relative_to(ROOT)),
        }

    checkpoint = resolve_checkpoint(record, run_dir)
    if checkpoint is None:
        return {
            "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
            "status": "missing_checkpoint",
            "checkpoint": record.get("checkpoint", ""),
            "checked": [
                str(ROOT / record["checkpoint"]) if record.get("checkpoint") else "",
                str(run_dir / record["checkpoint"]) if record.get("checkpoint") else "",
            ],
        }

    hyper = dict(config["hyperparams"])
    if arm in ("keep_direction_1", "keep_direction_1_2"):
        name = "Q1" if arm == "keep_direction_1" else "Q12"
        hyper["_projection_basis"] = str(projector_dir / f"{dataset}_{horizon}_{name}.npy")
    config = {**config, "hyperparams": hyper}

    model, exp_args = build_model(config, checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    head_gate = model.learned_residual_gate()
    # Determinism guard: recompute validation first.  The value must reproduce
    # the training run's own recorded val_mse, otherwise the test number this
    # script produces would not belong to the audited protocol and the cell is
    # reported as a mismatch instead of being silently accepted.
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    val_result = evaluate_once(
        model, val_loader, "val", model.target_var_index, head_gate
    )
    recorded_val_mse = float(record["val_mse"]) if record.get("val_mse") else None
    val_reproduces = None
    if recorded_val_mse:
        val_reproduces = abs(val_result["val_mse"] - recorded_val_mse) / recorded_val_mse
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    result = evaluate_once(
        model, test_loader, "test", model.target_var_index, head_gate
    )
    status = "read"
    if val_reproduces is not None:
        if val_reproduces >= VAL_REPRODUCE_TOL:
            status = "val_mismatch"
        elif val_reproduces >= VAL_REPRODUCE_WARN:
            status = "val_drift"
    payload = {
        "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
        "status": status,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": record["checkpoint"],
        "test_size": len(test_set),
        "recorded_val_mse": recorded_val_mse,
        "recomputed_val_mse": val_result["val_mse"],
        "val_relative_difference": val_reproduces,
        "val_reproduce_warn_threshold": VAL_REPRODUCE_WARN,
        **{k: v for k, v in result.items() if k != "eval_count"},
        "eval_count": result["eval_count"],
    }
    out_path = run_dir / "test_read.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--projector-dir", default="")
    parser.add_argument("--cell", default="", help="dataset:horizon:seed:arm")
    parser.add_argument("--cells-file", default="")
    parser.add_argument(
        "--gpus", default="",
        help="comma-separated GPUs; one cell at a time per GPU via subprocesses",
    )
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    projector_dir = Path(args.projector_dir) if args.projector_dir else root / "projectors"

    if args.cell:
        dataset, horizon, seed, arm = args.cell.split(":")
        cells = [(dataset, int(horizon), int(seed), arm)]
    elif args.cells_file:
        cells = []
        for line in Path(args.cells_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            dataset, horizon, seed, arm = line.split(":")
            cells.append((dataset, int(horizon), int(seed), arm))
    else:
        cells = [
            (dataset, horizon, seed, arm)
            for dataset, horizon in SETTINGS
            for seed in SEEDS
            for arm in ARMS
        ]

    if args.gpus:
        results = parallel_read(cells, args, root, projector_dir)
    else:
        results = []
        for cell in cells:
            payload = read_one(cell, root, projector_dir)
            results.append(payload)
            print("TOPTEST " + json.dumps(payload, sort_keys=True), flush=True)

    summary_path = root / "test_read_summary.json"
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text()).get("cells", [])
    merged = {entry["cell"]: entry for entry in existing}
    for entry in results:
        merged[entry["cell"]] = entry
    summary = {
        "protocol": "top2-direction-retention-test-read-v1",
        "note": (
            "one test read per frozen checkpoint; no configuration, projector "
            "or structure change is permitted after this step.  Reused "
            "direct_nlinear cells report the test numbers their own audited run "
            "already recorded and are never re-read."
        ),
        "cells": [merged[key] for key in sorted(merged)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    bad = [
        r for r in results
        if r["status"] not in ("read", "already_read", "reused", "val_drift")
    ]
    drifted = [r["cell"] for r in results if r["status"] == "val_drift"]
    if drifted:
        print(json.dumps({"val_drift_cells": drifted}, indent=2))
    print(json.dumps({"read": len(results), "problems": bad}, indent=2))
    if bad:
        raise SystemExit(f"test read had problems: {bad}")


def parallel_read(cells, args, root: Path, projector_dir: Path):
    """One subprocess per GPU, each walking a disjoint share of the cells."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    queues: dict[str, list[tuple]] = {gpu: [] for gpu in gpus}
    for index, cell in enumerate(cells):
        queues[gpus[index % len(gpus)]].append(cell)

    log_dir = root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    results = []
    for gpu, queue in queues.items():
        if not queue:
            continue
        cells_file = log_dir / f"test_read_gpu{gpu}.txt"
        cells_file.write_text(
            "\n".join(":".join(str(part) for part in cell) for cell in queue) + "\n"
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log = open(log_dir / f"test_read_gpu{gpu}.log", "w")
        process = subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--root", str(root),
                "--projector-dir", str(projector_dir),
                "--cells-file", str(cells_file),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu, process, log))
        print(json.dumps({"event": "test_read_launch", "gpu": gpu, "cells": len(queue)}))

    for gpu, process, log in processes:
        output, _ = process.communicate()
        log.write(output)
        log.close()
        for line in output.splitlines():
            if line.startswith("TOPTEST "):
                results.append(json.loads(line[len("TOPTEST "):]))
        print(
            json.dumps(
                {
                    "event": "test_read_done",
                    "gpu": gpu,
                    "return_code": process.returncode,
                    "cells_reported": sum(
                        1 for line in output.splitlines() if line.startswith("TOPTEST ")
                    ),
                }
            ),
            flush=True,
        )
    return results


if __name__ == "__main__":
    main()
