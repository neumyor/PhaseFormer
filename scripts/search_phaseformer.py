#!/usr/bin/env python3
"""Validation-isolated, resumable PhaseFormer experiment runner.

Search stages never instantiate a test loader unless --evaluate-test is explicitly
provided. Every run writes the same auditable artifact schema under a deterministic
experiment ID.
"""

import argparse
import csv
import hashlib
import heapq
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

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.runner import build_logger, build_trainer, restore_best_checkpoint

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)

PLANNED_BATCH_SIZE = {
    "ETTh1": 256,
    "ETTh2": 256,
    "ETTm1": 256,
    "ETTm2": 256,
    "Exchange": 32,
    "Weather": 64,
    "Electricity": 64,
    "Traffic": 8,
}

MECHANISMS = {
    "original": {},
    "fixed_residual_g05": {
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.5,
    },
    "fixed_residual_g09": {
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.9,
    },
    "adaptive_residual_g02": {
        "use_weak_period_residual": True,
        "use_adaptive_weak_period_gate": True,
        "weak_period_residual_gate_init": 0.2,
    },
    "adaptive_residual_g05": {
        "use_weak_period_residual": True,
        "use_adaptive_weak_period_gate": True,
        "weak_period_residual_gate_init": 0.5,
    },
    "phase_uncertainty_f035": {
        "use_phase_uncertainty_shrinkage": True,
        "phase_uncertainty_min": 0.35,
        "phase_uncertainty_trend_gate_init": 0.05,
    },
    "phase_uncertainty_f060": {
        "use_phase_uncertainty_shrinkage": True,
        "phase_uncertainty_min": 0.60,
        "phase_uncertainty_trend_gate_init": 0.05,
    },
    "phase_uncertainty_level": {
        "use_phase_uncertainty_shrinkage": True,
        "phase_uncertainty_min": 0.35,
        "phase_uncertainty_trend_gate_init": 0.05,
        "use_phase_period_level_calibration": True,
        "phase_level_slope_window": 3,
        "phase_level_slope_gate_init": 0.05,
        "phase_level_calib_gate_init": 0.1,
    },
    "phase_uncertainty_level_hifreq": {
        "use_phase_uncertainty_shrinkage": True,
        "phase_uncertainty_min": 0.35,
        "phase_uncertainty_trend_gate_init": 0.05,
        "use_phase_period_level_calibration": True,
        "phase_level_slope_window": 3,
        "phase_level_slope_gate_init": 0.05,
        "phase_level_calib_gate_init": 0.1,
        "use_phase_noise_hifreq_damping": True,
        "phase_noise_hifreq_strength": 0.5,
        "phase_noise_hifreq_threshold": 1.0,
        "phase_noise_hifreq_temperature": 0.2,
        "phase_noise_hifreq_window": 7,
    },
    "phase_uncertainty_level_hifreq_sparse": {
        "use_phase_uncertainty_shrinkage": True,
        "phase_uncertainty_min": 0.35,
        "phase_uncertainty_trend_gate_init": 0.05,
        "use_phase_period_level_calibration": True,
        "phase_level_slope_window": 3,
        "phase_level_slope_gate_init": 0.05,
        "phase_level_calib_gate_init": 0.1,
        "use_phase_noise_hifreq_damping": True,
        "phase_noise_hifreq_strength": 0.5,
        "phase_noise_hifreq_threshold": 1.0,
        "phase_noise_hifreq_temperature": 0.2,
        "phase_noise_hifreq_window": 7,
        "use_phase_sparse_event_calibration": True,
        "phase_sparse_event_window": 3,
        "phase_sparse_event_gate_init": 0.05,
        "phase_sparse_event_max_boost": 1.0,
        "phase_sparse_event_temperature": 0.2,
    },
    "lowpass_residual_w25": {
        "use_weak_period_residual": True,
        "weak_period_residual_head_type": "lowpass",
        "weak_period_residual_smooth_window": 25,
        "weak_period_residual_gate_init": 0.5,
    },
    "phase_local_trend": {
        "use_phase_local_trend": True,
        "phase_local_trend_window": 3,
        "phase_local_trend_gate_init": 0.0,
    },
    "phase_align": {
        "use_phase_align": True,
        "phase_align_hidden": 8,
        "phase_align_position_encoding": False,
    },
    "phase_warp": {
        "use_phase_warp": True,
        "phase_warp_hidden": 8,
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def repo_relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_csv(path, rows, fields=None):
    path = Path(path)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def apply_compact(hp):
    heads = int(hp["phase_attn_heads"])
    half = max(heads, int(hp["latent_dim"]) // 2)
    hp["latent_dim"] = max(heads, (half // heads) * heads)
    hp["phase_encoder_hidden"] = max(1, int(hp["phase_encoder_hidden"]) // 2)
    hp["predictor_hidden"] = max(1, int(hp["predictor_hidden"]) // 2)


def build_spec(args):
    overrides = json.loads(args.overrides)
    if args.mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {args.mechanism}")
    base = build_hyperparams(args.dataset, args.horizon, "original")
    # The experiment seed is explicit; do not inherit legacy per-task seeds.
    base["seed"] = args.seed
    base["period_len"] = args.period
    base.update(MECHANISMS[args.mechanism])
    base["scheme_name"] = args.mechanism
    base["train_epochs"] = args.max_epochs
    base["loss_func"] = args.loss
    base["use_huber_loss"] = args.loss == "huber"
    base_lr = float(build_hyperparams(args.dataset, args.horizon, "original")["learning_rate"])
    base["learning_rate"] = args.learning_rate if args.learning_rate else base_lr * args.lr_multiplier
    if args.capacity == "compact":
        apply_compact(base)
    base.update(overrides)
    spec = {
        "protocol_version": "search-plan-v1",
        "stage": args.stage,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "mechanism": args.mechanism,
        "period": args.period,
        "percent": args.percent,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "loss": args.loss,
        "lr_multiplier": args.lr_multiplier,
        "capacity": args.capacity,
        "batch_size": args.batch_size or PLANNED_BATCH_SIZE[args.dataset],
        "evaluate_test": args.evaluate_test,
        "hyperparams": base,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["config_hash"] = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return spec


def run_id(spec):
    lr = spec["hyperparams"]["learning_rate"]
    return (
        f"{spec['stage']}_{spec['dataset'].lower()}_h{spec['horizon']}_"
        f"{spec['mechanism']}_p{spec['period']}_{spec['capacity']}_"
        f"{spec['loss']}_lr{lr:.6g}_pct{spec['percent']}_e{spec['max_epochs']}_"
        f"s{spec['seed']}_{spec['config_hash']}"
    )




def classify_case(pred, true):
    err = pred - true
    bias = abs(float(err.mean()))
    peak_gap = float(np.max(true) - np.max(pred))
    vol_gap = abs(float(np.std(pred) - np.std(true)))
    late = float(np.mean(np.abs(err[len(err) * 2 // 3 :])))
    early = float(np.mean(np.abs(err[: max(1, len(err) // 3)])))
    scores = {
        "systematic_bias": bias,
        "peak_underfit": max(peak_gap, 0.0),
        "volatility_mismatch": vol_gap,
        "late_horizon_drift": max(late - early, 0.0),
    }
    mode = max(scores, key=scores.get)
    actions = {
        "systematic_bias": "review period-level calibration",
        "peak_underfit": "review sparse-event phase calibration",
        "volatility_mismatch": "review uncertainty/high-frequency damping",
        "late_horizon_drift": "review residual or phase-local trend response",
    }
    return mode, actions[mode]


def variable_names(exp_args):
    path = Path(exp_args.dataset_args.root_path) / exp_args.dataset_args.data_path
    columns = list(pd.read_csv(path, nrows=0).columns)
    columns = [c for c in columns if c != "date"]
    target = exp_args.dataset_args.target
    if target in columns:
        columns = [c for c in columns if c != target] + [target]
    return columns


def evaluate(model, loader, dataset, split, pred_len, run_dir, bad_case_limit=0):
    model.eval()
    device = next(model.parameters()).device
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    heap = []
    serial = 0
    names = variable_names(model.args) if bad_case_limit else []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float())
            pred = out[:, -pred_len:, :]
            true = batch_y.float()[:, -pred_len:, :]
            err = pred - true
            abs_sum += torch.abs(err).sum().item()
            sq_sum += torch.square(err).sum().item()
            count += err.numel()
            if bad_case_limit:
                pair_mse = torch.square(err).mean(dim=1).detach().cpu()
                for b in range(pair_mse.shape[0]):
                    global_sample = batch_idx * loader.batch_size + b
                    vals, variables = torch.topk(pair_mse[b], min(bad_case_limit, pair_mse.shape[1]))
                    for value, variable in zip(vals.tolist(), variables.tolist()):
                        item = (
                            float(value), serial, global_sample, int(variable),
                            pred[b, :, variable].detach().cpu().numpy(),
                            true[b, :, variable].detach().cpu().numpy(),
                        )
                        serial += 1
                        if len(heap) < bad_case_limit:
                            heapq.heappush(heap, item)
                        elif item[0] > heap[0][0]:
                            heapq.heapreplace(heap, item)
    result = {f"{split}_mae": abs_sum / count, f"{split}_mse": sq_sum / count}
    if bad_case_limit:
        bad_dir = Path(run_dir) / "bad_cases"
        pred_dir = Path(run_dir) / "predictions"
        bad_dir.mkdir(exist_ok=True)
        pred_dir.mkdir(exist_ok=True)
        rows = []
        timestamps = getattr(dataset, "timestamps", None)
        for rank, item in enumerate(sorted(heap, reverse=True), 1):
            mse, _, sample, variable, pred, true = item
            artifact = pred_dir / f"val_bad_case_{rank:02d}.npz"
            np.savez_compressed(artifact, prediction=pred, truth=true)
            mode, action = classify_case(pred, true)
            timestamp = "unavailable"
            start = sample + int(model.seq_len)
            if timestamps is not None and start < len(timestamps):
                timestamp = str(timestamps[start])
            rows.append({
                "rank": rank,
                "split": split,
                "sample_index": sample,
                "variable_index": variable,
                "variable": names[variable] if variable < len(names) else str(variable),
                "forecast_start_timestamp": timestamp,
                "mae": float(np.mean(np.abs(pred - true))),
                "mse": mse,
                "prediction_path": repo_relative(artifact),
                "truth_path": repo_relative(artifact),
                "error_mode": mode,
                "next_action": action,
            })
        write_csv(bad_dir / "val_bad_cases.csv", rows)
    return result


def epoch_count(logger_dir):
    metrics = Path(logger_dir) / "metrics.csv"
    if not metrics.exists():
        return 0
    frame = pd.read_csv(metrics)
    return int(frame["epoch"].dropna().max() + 1) if "epoch" in frame and frame["epoch"].notna().any() else 0


def execute(args):
    spec = build_spec(args)
    rid = run_id(spec)
    run_dir = Path(args.output_dir) / "runs" / rid
    complete = run_dir / "metrics.csv"
    if complete.exists():
        if args.resume:
            print(f"RESUME completed: {rid}")
            return
        raise FileExistsError(f"completed experiment already exists: {rid}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "status.json", {"status": "running", "started_at": utc_now()})
    atomic_json(run_dir / "config.json", spec)
    (run_dir / "commands.sh").write_text("#!/bin/sh\n" + shlex.join([sys.executable, *sys.argv]) + "\n")

    hp = spec["hyperparams"]
    pl.seed_everything(spec["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")
    exp_args = make_exp_args(
        spec["dataset"], spec["lookback"], spec["horizon"], hp,
        batch_size=spec["batch_size"],
    )
    exp_args.dataset_args.percent = spec["percent"]
    configured_root = Path(exp_args.dataset_args.root_path)
    ett_fallback = REPO_ROOT / "resources" / "all_datasets" / "ETT-small"
    if not configured_root.exists() and spec["dataset"].startswith("ETT") and ett_fallback.exists():
        exp_args.dataset_args.root_path = str(ett_fallback)
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers
    train_set, train_loader = data_provider(exp_args.dataset_args, "train")
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    if hasattr(train_set, "data_stamp"):
        hp["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(PhaseFormerPresetConfig(exp_args, spec["lookback"], spec["horizon"], hp))
    parameter_count = sum(p.numel() for p in model.parameters())

    attempts_root = run_dir / "attempts"
    attempts_root.mkdir(exist_ok=True)
    attempt_index = max([int(x.name) for x in attempts_root.iterdir() if x.is_dir() and x.name.isdigit()] or [0]) + 1
    attempt_dir = attempts_root / f"{attempt_index:03d}"
    attempt_dir.mkdir()
    logger = build_logger(str(attempt_dir / "lightning"), name="PhaseFormer", version="train")
    trainer, checkpoint = build_trainer(
        max_epochs=spec["max_epochs"],
        logger=logger,
        patience=int(hp.get("patience", 8)),
        checkpoint_dir=str(attempt_dir / "checkpoints"),
        accelerator="gpu",
        progress=args.progress,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - start
    if not checkpoint.best_model_path:
        raise RuntimeError("training completed without a best checkpoint")
    restore_best_checkpoint(model, checkpoint)
    model.to(trainer.strategy.root_device)
    val_metrics = evaluate(
        model, val_loader, val_set, "val", spec["horizon"], run_dir,
        bad_case_limit=min(args.bad_case_limit, 8),
    )
    test_metrics = {"test_mae": "", "test_mse": ""}
    test_size = ""
    if spec["evaluate_test"]:
        test_set, test_loader = data_provider(exp_args.dataset_args, "test")
        test_metrics = evaluate(model, test_loader, test_set, "test", spec["horizon"], run_dir)
        test_size = len(test_set)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    row = {
        "run_id": rid, "protocol_version": spec["protocol_version"], "stage": spec["stage"],
        "dataset": spec["dataset"], "lookback": spec["lookback"], "horizon": spec["horizon"],
        "mechanism": spec["mechanism"], "period": spec["period"], "capacity": spec["capacity"],
        "loss": spec["loss"], "learning_rate": hp["learning_rate"], "lr_multiplier": spec["lr_multiplier"],
        "percent": spec["percent"], "seed": spec["seed"], "batch_size": spec["batch_size"],
        "epochs_requested": spec["max_epochs"], "epochs_completed": epoch_count(logger.log_dir),
        "best_val_loss": float(checkpoint.best_model_score.cpu()), **val_metrics, **test_metrics,
        "parameter_count": parameter_count, "elapsed_sec": elapsed, "peak_memory_bytes": peak,
        "train_size": len(train_set), "val_size": len(val_set), "test_size": test_size,
        "checkpoint": repo_relative(checkpoint.best_model_path),
        "config_hash": spec["config_hash"], "completed_at": utc_now(),
    }
    write_csv(complete, [row])
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "lightning": pl.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--short")),
        "platform": platform.platform(),
    }
    atomic_json(run_dir / "environment.json", environment)
    atomic_json(run_dir / "status.json", {"status": "completed", "completed_at": utc_now()})
    print(json.dumps(row, indent=2, default=str))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(PLANNED_BATCH_SIZE))
    p.add_argument("--horizon", required=True, type=int, choices=[96, 192, 336, 720])
    p.add_argument("--stage", required=True, choices=["smoke", "baseline", "period_screen", "mechanism_screen_1", "mechanism_screen_2", "mechanism_full8", "hp_low", "hp_mid", "finalist", "confirm"])
    p.add_argument("--mechanism", default="original", choices=list(MECHANISMS))
    p.add_argument("--period", type=int, default=24)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--percent", type=int, default=100)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--loss", choices=["huber", "mae"], default="huber")
    p.add_argument("--lr-multiplier", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--capacity", choices=["base", "compact"], default="base")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bad-case-limit", type=int, default=8)
    p.add_argument("--overrides", default="{}", help="JSON object applied last")
    p.add_argument("--evaluate-test", action="store_true", help="Allowed only for frozen confirm runs")
    p.add_argument("--output-dir", default="research_runs/search_v1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()
    if args.evaluate_test and args.stage != "confirm":
        p.error("--evaluate-test is restricted to frozen confirm runs")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        execute(args)
    except Exception as exc:
        # The deterministic run directory may not yet exist if config parsing failed.
        try:
            spec = build_spec(args)
            path = Path(args.output_dir) / "runs" / run_id(spec)
            path.mkdir(parents=True, exist_ok=True)
            atomic_json(path / "status.json", {
                "status": "failed", "failed_at": utc_now(), "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            raise
