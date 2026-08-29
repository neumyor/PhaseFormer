#!/usr/bin/env python3
"""Train one validation-only multi-anchor router from OOF shadow forecasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.search_phaseformer import (  # noqa: E402
    PLANNED_BATCH_SIZE,
    epoch_count,
    evaluate,
    repo_relative,
    write_csv,
)
from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.PhaseFormer import PhaseFormer  # noqa: E402
from src.models.multi_anchor import ANCHOR_NAMES, MultiAnchorPhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    PhaseFormerPresetConfig,
    make_exp_args,
)
from src.training.runner import (  # noqa: E402
    build_logger,
    build_trainer,
    restore_best_checkpoint,
)


MECHANISMS = {
    "multi_anchor_global_hard": {
        "router_mode": "global", "output_mode": "hard",
        "mean_regret_weight": 0.0, "cvar_weight": 0.0,
    },
    "multi_anchor_structural_hard": {
        "router_mode": "structural", "output_mode": "hard",
        "mean_regret_weight": 0.0, "cvar_weight": 0.0,
    },
    "multi_anchor_guarded_hard": {
        "router_mode": "structural", "output_mode": "hard",
        "mean_regret_weight": 0.05, "cvar_weight": 0.01,
    },
    "multi_anchor_structural_soft": {
        "router_mode": "structural", "output_mode": "soft",
        "mean_regret_weight": 0.0, "cvar_weight": 0.0,
    },
}
EXPECTED_PRESETS = {
    "A1": "gold_combo_reliability_s2",
    "I0": "rcrf_icpt_none",
    "R0": "triaxis_rolling_features",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_reference(run_dir):
    run_dir = resolve(run_dir)
    spec = json.loads((run_dir / "config.json").read_text())
    with (run_dir / "metrics.csv").open(newline="") as handle:
        metrics = next(csv.DictReader(handle))
    checkpoint = resolve(metrics["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return run_dir, spec, metrics, checkpoint


def verify_references(references, args):
    for bank_name, bank in references.items():
        expected_percent = args.shadow_percent if bank_name == "shadow" else args.full_percent
        for anchor_name, (_, spec, _, _) in bank.items():
            expected = {
                "dataset": args.dataset,
                "horizon": args.horizon,
                "lookback": args.lookback,
                "mechanism": EXPECTED_PRESETS[anchor_name],
                "period": args.period,
                "percent": expected_percent,
                "seed": args.seed,
                "loss": "huber",
            }
            mismatch = {
                key: (spec.get(key), value)
                for key, value in expected.items()
                if spec.get(key) != value
            }
            if mismatch:
                raise ValueError(f"{bank_name}/{anchor_name} mismatch: {mismatch}")
            if int(spec["max_epochs"]) != args.anchor_epochs:
                raise ValueError(
                    f"{bank_name}/{anchor_name} epochs={spec['max_epochs']} "
                    f"!= {args.anchor_epochs}"
                )


def make_model(spec, checkpoint):
    hp = spec["hyperparams"]
    exp_args = make_exp_args(
        spec["dataset"], spec["lookback"], spec["horizon"], hp,
        batch_size=spec["batch_size"],
    )
    model = PhaseFormer(
        PhaseFormerPresetConfig(
            exp_args, spec["lookback"], spec["horizon"], hp
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    return model


def build_spec(args, references):
    mechanism = MECHANISMS[args.mechanism]
    paths = {
        bank: {
            anchor: repo_relative(value[0]) for anchor, value in values.items()
        }
        for bank, values in references.items()
    }
    spec = {
        "protocol_version": "multi-anchor-selector-v1",
        "stage": args.stage,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "period": args.period,
        "mechanism": args.mechanism,
        "mechanism_config": mechanism,
        "shadow_percent": args.shadow_percent,
        "full_percent": args.full_percent,
        "anchor_epochs": args.anchor_epochs,
        "router_epochs": args.max_epochs,
        "seed": args.seed,
        "loss": "huber",
        "batch_size": args.batch_size or PLANNED_BATCH_SIZE[args.dataset],
        "router_hidden": 24,
        "router_temperature": 0.2,
        "oracle_temperature": 0.1,
        "route_weight": 0.1,
        "test_accessed": False,
        "reference_runs": paths,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["config_hash"] = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return spec


def run_id(spec):
    return (
        f"{spec['stage']}_{spec['dataset'].lower()}_h{spec['horizon']}_"
        f"{spec['mechanism']}_p{spec['period']}_pct{spec['shadow_percent']}-"
        f"{spec['full_percent']}_e{spec['router_epochs']}_s{spec['seed']}_"
        f"{spec['config_hash']}"
    )


def calibration_loader(exp_args, args):
    exp_args.dataset_args.percent = args.full_percent
    full_set, _ = data_provider(exp_args.dataset_args, "train")
    exp_args.dataset_args.percent = args.shadow_percent
    shadow_set, _ = data_provider(exp_args.dataset_args, "train")
    # A window is OOF only when its forecast begins at or after the shadow
    # cutoff.  Starting at len(shadow_set) would still overlap by H-1 targets.
    start = len(shadow_set.data_x) - args.lookback
    stop = len(full_set)
    if start < 0 or start >= stop:
        raise RuntimeError(f"empty OOF calibration range: start={start}, stop={stop}")
    subset = Subset(full_set, range(start, stop))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size or PLANNED_BATCH_SIZE[args.dataset],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    audit = {
        "shadow_data_length": len(shadow_set.data_x),
        "full_data_length": len(full_set.data_x),
        "calibration_first_index": start,
        "calibration_last_index": stop - 1,
        "calibration_size": len(subset),
        "first_prediction_start": start + args.lookback,
        "shadow_cutoff": len(shadow_set.data_x),
        "target_overlap_count": max(
            0, len(shadow_set.data_x) - (start + args.lookback)
        ),
    }
    if audit["target_overlap_count"]:
        raise RuntimeError(f"OOF target overlap detected: {audit}")
    return full_set, subset, loader, audit


def route_diagnostics(model, loader, pred_len):
    model.eval()
    device = next(model.parameters()).device
    selected = torch.zeros(3, dtype=torch.long)
    oracle_selected = torch.zeros(3, dtype=torch.long)
    agreements = 0
    total_routes = 0
    regret_sum = 0.0
    regret_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = [value.to(device) for value in batch]
            x, y, x_mark, y_mark = batch
            output, _, _ = model(
                x.float(), x_mark.float(), model._build_decoder_input(y.float()),
                y_mark.float(),
            )
            target = y.float()[:, -pred_len:, :]
            anchors = model.last_anchor_outputs
            anchor_mse = torch.stack(
                [model._cyclewise_mse(value, target) for value in anchors], dim=-1
            )
            oracle = anchor_mse.argmin(dim=-1).cpu()
            chosen = model.router.last_hard_weights.argmax(dim=-1).cpu()
            selected += torch.bincount(chosen.reshape(-1), minlength=3)
            oracle_selected += torch.bincount(oracle.reshape(-1), minlength=3)
            agreements += int((chosen == oracle).sum())
            total_routes += chosen.numel()
            candidate_mse = model._cyclewise_mse(output, target)
            envelope = anchor_mse.min(dim=-1).values
            regret = (candidate_mse - envelope) / envelope.clamp_min(1e-8)
            regret_sum += float(regret.sum())
            regret_count += regret.numel()
    return {
        **{f"select_{name.lower()}_rate": float(selected[i]) / total_routes
           for i, name in enumerate(ANCHOR_NAMES)},
        **{f"oracle_{name.lower()}_rate": float(oracle_selected[i]) / total_routes
           for i, name in enumerate(ANCHOR_NAMES)},
        "route_oracle_agreement": agreements / total_routes,
        "mean_relative_oracle_regret": regret_sum / regret_count,
        "route_count": total_routes,
    }


def execute(args):
    references = {
        "shadow": {
            "A1": read_reference(args.shadow_a1),
            "I0": read_reference(args.shadow_i0),
            "R0": read_reference(args.shadow_r0),
        },
        "full": {
            "A1": read_reference(args.full_a1),
            "I0": read_reference(args.full_i0),
            "R0": read_reference(args.full_r0),
        },
    }
    verify_references(references, args)
    spec = build_spec(args, references)
    rid = run_id(spec)
    run_dir = Path(args.output_dir) / "runs" / rid
    complete = run_dir / "metrics.csv"
    if complete.exists():
        if args.resume:
            print(f"RESUME completed: {rid}")
            return
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "status.json", {"status": "running", "started_at": utc_now()})
    atomic_json(run_dir / "config.json", spec)
    (run_dir / "commands.sh").write_text(
        "#!/bin/sh\n" + shlex.join([sys.executable, *sys.argv]) + "\n"
    )

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    a1_spec = references["full"]["A1"][1]
    hp = dict(a1_spec["hyperparams"])
    hp["train_epochs"] = args.max_epochs
    hp["loss_func"] = "huber"
    exp_args = make_exp_args(
        args.dataset, args.lookback, args.horizon, hp,
        batch_size=spec["batch_size"],
    )
    configured_root = Path(exp_args.dataset_args.root_path)
    fallback = REPO_ROOT / "resources" / "all_datasets" / "ETT-small"
    if not configured_root.exists() and args.dataset.startswith("ETT") and fallback.exists():
        exp_args.dataset_args.root_path = str(fallback)
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers
    exp_args.training_args.loss_func = "huber"
    full_train_set, calibration_set, train_loader, calibration_audit = calibration_loader(
        exp_args, args
    )
    exp_args.dataset_args.percent = args.full_percent
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    if hasattr(full_train_set, "data_stamp"):
        hp["time_mark_dim"] = int(full_train_set.data_stamp.shape[-1])
    config = PhaseFormerPresetConfig(exp_args, args.lookback, args.horizon, hp)
    shadow = {
        name: make_model(references["shadow"][name][1], references["shadow"][name][3])
        for name in ANCHOR_NAMES
    }
    full = {
        name: make_model(references["full"][name][1], references["full"][name][3])
        for name in ANCHOR_NAMES
    }
    mechanism = MECHANISMS[args.mechanism]
    model = MultiAnchorPhaseFormer(
        config,
        shadow,
        full,
        router_mode=mechanism["router_mode"],
        output_mode=mechanism["output_mode"],
        hidden=24,
        temperature=0.2,
        oracle_temperature=0.1,
        route_weight=0.1,
        mean_regret_weight=mechanism["mean_regret_weight"],
        cvar_weight=mechanism["cvar_weight"],
    )
    if any(parameter.requires_grad for bank in (shadow, full) for anchor in bank.values() for parameter in anchor.parameters()):
        raise RuntimeError("anchor freezing audit failed")

    audit_batch = next(iter(train_loader))
    audit_batch = [value[:2] for value in audit_batch]
    model.eval()
    with torch.inference_mode():
        x, y, x_mark, y_mark = audit_batch
        output, _, _ = model(
            x.float(), x_mark.float(), model._build_decoder_input(y.float()), y_mark.float()
        )
        full_identity = float((output - model.last_anchor_outputs[0]).abs().max())
    model.train()
    with torch.no_grad():
        output, _, _ = model(
            x.float(), x_mark.float(), model._build_decoder_input(y.float()), y_mark.float()
        )
        shadow_identity = float((output - model.last_anchor_outputs[0]).abs().max())
    hard = mechanism["output_mode"] == "hard"
    if hard and (full_identity != 0.0 or shadow_identity != 0.0):
        raise RuntimeError(
            f"hard initialization is not exact A1: full={full_identity}, shadow={shadow_identity}"
        )

    parameter_count = sum(value.numel() for value in model.parameters())
    trainable_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    attempts = run_dir / "attempts" / "001"
    attempts.mkdir(parents=True, exist_ok=True)
    logger = build_logger(str(attempts / "lightning"), name="MultiAnchor", version="train")
    trainer, checkpoint = build_trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        patience=args.max_epochs,
        checkpoint_dir=str(attempts / "checkpoints"),
        accelerator="gpu",
        progress=args.progress,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - start
    restore_best_checkpoint(model, checkpoint)
    model.to(trainer.strategy.root_device)
    val_metrics = evaluate(
        model, val_loader, val_set, "val", args.horizon, run_dir, bad_case_limit=0
    )
    diagnostics = route_diagnostics(model, val_loader, args.horizon)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    row = {
        "run_id": rid,
        "protocol_version": spec["protocol_version"],
        "stage": args.stage,
        "dataset": args.dataset,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "mechanism": args.mechanism,
        "period": args.period,
        "loss": "huber",
        "learning_rate": exp_args.training_args.learning_rate,
        "shadow_percent": args.shadow_percent,
        "full_percent": args.full_percent,
        "seed": args.seed,
        "batch_size": spec["batch_size"],
        "epochs_requested": args.max_epochs,
        "epochs_completed": epoch_count(logger.log_dir),
        "best_val_loss": float(checkpoint.best_model_score.cpu()),
        **val_metrics,
        "test_mae": "",
        "test_mse": "",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "elapsed_sec": elapsed,
        "peak_memory_bytes": peak,
        "calibration_size": len(calibration_set),
        "val_size": len(val_set),
        "target_overlap_count": calibration_audit["target_overlap_count"],
        "full_anchor_identity_max_abs": full_identity,
        "shadow_anchor_identity_max_abs": shadow_identity,
        "anchor_frozen": True,
        **diagnostics,
        "checkpoint": repo_relative(checkpoint.best_model_path),
        "config_hash": spec["config_hash"],
        "completed_at": utc_now(),
    }
    write_csv(complete, [row])
    atomic_json(run_dir / "calibration_audit.json", calibration_audit)
    atomic_json(run_dir / "environment.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--short")),
        "platform": platform.platform(),
    })
    atomic_json(run_dir / "status.json", {"status": "completed", "completed_at": utc_now()})
    print(json.dumps(row, indent=2, default=str))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(PLANNED_BATCH_SIZE))
    parser.add_argument("--horizon", required=True, type=int, choices=(96, 192))
    parser.add_argument("--stage", required=True, choices=("smoke", "pilot", "stage_a", "stage_b"))
    parser.add_argument("--mechanism", required=True, choices=tuple(MECHANISMS))
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--shadow-percent", type=int, default=24)
    parser.add_argument("--full-percent", type=int, default=30)
    parser.add_argument("--anchor-epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    for bank in ("shadow", "full"):
        for anchor in ("a1", "i0", "r0"):
            parser.add_argument(f"--{bank}-{anchor}", required=True)
    parser.add_argument("--output-dir", default="research_runs/multi_anchor_selector_v1_scratch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not 0 < args.shadow_percent < args.full_percent <= 100:
        parser.error("require 0 < shadow-percent < full-percent <= 100")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        execute(arguments)
    except Exception as error:
        try:
            failure_root = Path(arguments.output_dir) / "failures"
            failure_root.mkdir(parents=True, exist_ok=True)
            atomic_json(failure_root / f"{arguments.dataset}_h{arguments.horizon}_{arguments.mechanism}.json", {
                "status": "failed", "failed_at": utc_now(), "error": repr(error),
                "traceback": traceback.format_exc(),
            })
        finally:
            raise
