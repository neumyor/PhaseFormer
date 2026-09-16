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
import sys
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

    checkpoint = ROOT / record["checkpoint"]
    if not checkpoint.is_file():
        return {
            "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
            "status": "missing_checkpoint",
            "checkpoint": str(checkpoint),
        }

    hyper = dict(config["hyperparams"])
    if arm in ("keep_direction_1", "keep_direction_1_2"):
        name = "Q1" if arm == "keep_direction_1" else "Q12"
        hyper["_projection_basis"] = str(projector_dir / f"{dataset}_{horizon}_{name}.npy")
    config = {**config, "hyperparams": hyper}

    model, exp_args = build_model(config, checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    head_gate = model.learned_residual_gate()
    result = evaluate_once(
        model, test_loader, "test", model.target_var_index, head_gate
    )
    payload = {
        "cell": f"{dataset}-{horizon}-s{seed}-{arm}",
        "status": "read",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": record["checkpoint"],
        "test_size": len(test_set),
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
            "or structure change is permitted after this step"
        ),
        "cells": [merged[key] for key in sorted(merged)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    bad = [r for r in results if r["status"] not in ("read", "already_read", "reused")]
    print(json.dumps({"read": len(results), "problems": bad}, indent=2))
    if bad:
        raise SystemExit(f"test read had problems: {bad}")


if __name__ == "__main__":
    main()
