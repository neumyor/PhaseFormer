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
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.runner import build_logger, build_trainer, restore_best_checkpoint

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    ABLATION_MODES,
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
    "phase_amp_calib": {
        "use_phase_warp": True,
        "phase_warp_hidden": 8,
        "use_phase_amp_calib": True,
        "phase_amp_calib_hidden": 8,
        "phase_amp_calib_max_scale": 2.0,
    },
    "phase_rape": {
        "use_phase_rape": True,
        "phase_warp_hidden": 8,
        "phase_amp_calib_hidden": 8,
        "phase_amp_calib_max_scale": 2.0,
        "phase_rape_gate_hidden": 8,
    },
    # ---- dynamic-phase mechanisms (weak-residual-phaseformer plan) ----
    # Stage 1: residual-branch contribution. `residual_full` enables both
    # residual heads on the original phase path; `no_residual` disables them.
    # Together the pair isolates the residual-branch contribution (same seed,
    # same phase path, only the two heads differ).
    "residual_full": {
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.5,
        "use_phase_local_trend": True,
        "phase_local_trend_window": 3,
        "phase_local_trend_gate_init": 0.0,
    },
    "no_residual": {"use_residual_head": False},
    # Stage 2: dynamic phase correction (per-slot phase offset on latent tokens).
    "phase_correction": {"use_phase_correction": True},
    # Stage 3: circular (Fourier) phase geometry replacing the learnable pos embed.
    "circular_geometry": {"phase_use_circular_pos": True},
    # Stage 4: phase rotation of latent feature pairs.
    "phase_rotation": {
        "use_phase_rotation": True,
        "phase_rotation_hidden": 8,
    },
    # Stage 5: harmonic feature modulation (gamma*z + beta) before the predictor.
    "harmonic_modulation": {
        "use_harmonic_modulation": True,
        "harmonic_modulation_hidden": 8,
        "harmonic_modulation_max_scale": 2.0,
    },
    # Stage 9 final structure: correction + geometry + rotation + harmonic.
    "dyn_stack": {
        "use_phase_correction": True,
        "phase_use_circular_pos": True,
        "use_phase_rotation": True,
        "phase_rotation_hidden": 8,
        "use_harmonic_modulation": True,
        "harmonic_modulation_hidden": 8,
        "harmonic_modulation_max_scale": 2.0,
    },
    # Cumulative ablation ladder (plan stage 10): A(+Correction),
    # B(+Geometry), C(+Rotation), D(+Harmonic = dyn_stack).
    "dyn_geo": {
        "use_phase_correction": True,
        "phase_use_circular_pos": True,
    },
    "dyn_geo_rot": {
        "use_phase_correction": True,
        "phase_use_circular_pos": True,
        "use_phase_rotation": True,
        "phase_rotation_hidden": 8,
    },
    # Final plan structure: dynamic-phase stack + residual reconstruction heads.
    "dyn_full": {
        "use_phase_correction": True,
        "phase_use_circular_pos": True,
        "use_phase_rotation": True,
        "phase_rotation_hidden": 8,
        "use_harmonic_modulation": True,
        "harmonic_modulation_hidden": 8,
        "harmonic_modulation_max_scale": 2.0,
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.5,
        "use_phase_local_trend": True,
        "phase_local_trend_window": 3,
        "phase_local_trend_gate_init": 0.0,
    },
    # ---- next-stage paper plan (Adaptive Phase-Residual Trajectory Modeling) ----
    # Stage 1: phase velocity trajectory (upgrades offset phi'=phi+delta to
    # phi_t = phi_{t-1} + delta_phi_t via velocity cumsum).
    "phase_velocity": {
        "use_phase_velocity": True,
        "phase_velocity_hidden": 8,
        "phase_velocity_scale": 0.1,
    },
    # Stage 2: velocity + circular attention bias (QK^T - B_circle, geometry at
    # the interaction layer instead of only the position embedding).
    "phase_vel_geo": {
        "use_phase_velocity": True,
        "phase_velocity_hidden": 8,
        "phase_velocity_scale": 0.1,
        "phase_use_circular_attn_bias": True,
        "phase_circular_attn_bias_scale": 1.0,
    },
    # Stage 3: adaptive residual fusion gate (y = (1-alpha) y_p + alpha y_r,
    # alpha from the latent phase feature).
    "residual_adaptive": {
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.5,
        "use_phase_local_trend": True,
        "phase_local_trend_window": 3,
        "phase_local_trend_gate_init": 0.0,
        "use_adaptive_residual_gate": True,
        "adaptive_residual_gate_hidden": 8,
        "adaptive_residual_gate_init": 0.5,
    },
    # Final next-stage model: velocity + circular bias + adaptive residual gate.
    "next_full": {
        "use_phase_velocity": True,
        "phase_velocity_hidden": 8,
        "phase_velocity_scale": 0.1,
        "phase_use_circular_attn_bias": True,
        "phase_circular_attn_bias_scale": 1.0,
        "use_weak_period_residual": True,
        "weak_period_residual_gate_init": 0.5,
        "use_phase_local_trend": True,
        "phase_local_trend_window": 3,
        "phase_local_trend_gate_init": 0.0,
        "use_adaptive_residual_gate": True,
        "adaptive_residual_gate_hidden": 8,
        "adaptive_residual_gate_init": 0.5,
    },
    # ---- residual topology experiment ----
    "residual_output_convex": {
        "use_topology_output_convex_residual": True,
        "topology_output_convex_gate_init": 0.5,
    },
    "residual_output_additive": {
        "use_additive_output_residual": True,
        "additive_output_residual_gate_init": 0.5,
    },
    "residual_latent_long": {
        "use_latent_long_residual": True,
    },
    "residual_latent_layerwise": {
        "use_layerwise_latent_residual": True,
    },
    "residual_hybrid": {
        "use_layerwise_latent_residual": True,
        "use_additive_output_residual": True,
        "additive_output_residual_gate_init": 0.5,
    },
    # Layer-wise output residuals (A1/A2): same fusion form at every routing
    # depth instead of only the final output.
    "residual_output_layerwise_convex": {
        "use_topology_output_convex_residual": True,
        "topology_output_convex_gate_init": 0.5,
        "use_layerwise_output_convex": True,
        "layerwise_output_convex_gate_init": 0.0,
    },
    "residual_output_layerwise_additive": {
        "use_additive_output_residual": True,
        "additive_output_residual_gate_init": 0.5,
        "use_layerwise_output_additive": True,
        "layerwise_output_additive_gate_init": 0.5,
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
    if args.mechanism in MECHANISMS:
        base = build_hyperparams(args.dataset, args.horizon, "original")
        base.update(MECHANISMS[args.mechanism])
    elif args.mechanism in ABLATION_MODES or args.mechanism in ("latest", "best_nonresidual"):
        # Preset modes reachable through build_hyperparams (latest and the
        # gold_combo_* cross-dataset mechanisms).
        base = build_hyperparams(args.dataset, args.horizon, args.mechanism)
    else:
        raise ValueError(f"unknown mechanism: {args.mechanism}")
    # The experiment seed is explicit; do not inherit legacy per-task seeds.
    base["seed"] = args.seed
    base["period_len"] = args.period
    cycle_period = getattr(args, "cycle_period", None)
    require_cuda = bool(getattr(args, "require_cuda", False))
    if cycle_period is not None:
        base["anchored_pctf_cycle_period_len"] = cycle_period
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
        "protocol_version": (
            "input-components-h134-v1"
            if getattr(args, "stage", "") == "input_components"
            else (
                "search-plan-v2"
                if cycle_period is not None or require_cuda
                else "search-plan-v1"
            )
        ),
        "stage": args.stage,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "mechanism": args.mechanism,
        "period": args.period,
        "cycle_period": base.get("anchored_pctf_cycle_period_len", ""),
        "percent": args.percent,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "loss": args.loss,
        "lr_multiplier": args.lr_multiplier,
        "capacity": args.capacity,
        "batch_size": args.batch_size or PLANNED_BATCH_SIZE[args.dataset],
        "evaluate_test": args.evaluate_test,
        "input_hypothesis": getattr(args, "input_hypothesis", "none"),
        "input_variant": getattr(args, "input_variant", "full"),
        "intervention_seed": getattr(args, "intervention_seed", 9102),
        "max_eval_samples": getattr(args, "max_eval_samples", 0),
        "require_cuda": require_cuda,
        "init_checkpoint": repo_relative(getattr(args, "init_checkpoint", ""))
        if getattr(args, "init_checkpoint", "") else "",
        "hyperparams": base,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["config_hash"] = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return spec


def run_id(spec):
    lr = spec["hyperparams"]["learning_rate"]
    cycle = f"_cp{spec['cycle_period']}" if spec["cycle_period"] != "" else ""
    return (
        f"{spec['stage']}_{spec['dataset'].lower()}_h{spec['horizon']}_"
        f"{spec['mechanism']}_p{spec['period']}{cycle}_{spec['capacity']}_"
        f"{spec['input_hypothesis']}-{spec['input_variant']}_"
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
    anchor_abs_sum = 0.0
    anchor_sq_sum = 0.0
    update_sq_sum = 0.0
    attribution_count = 0
    confidence_moments = [0.0] * 5  # x, y, x2, y2, xy
    coefficient_moments = [0.0] * 5
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
            if model.target_var_index != -1:
                true = true[:, :, model.target_var_index : model.target_var_index + 1]
            err = pred - true
            abs_sum += torch.abs(err).sum().item()
            sq_sum += torch.square(err).sum().item()
            count += err.numel()
            if model.use_anchored_phase_cycle_fusion:
                anchor = model.anchored_pctf_anchor_output[:, -pred_len:, :]
                if model.target_var_index != -1:
                    anchor = anchor[
                        :, :, model.target_var_index : model.target_var_index + 1
                    ]
                anchor_err = anchor - true
                update = pred - anchor
                anchor_abs_sum += anchor_err.abs().sum().item()
                anchor_sq_sum += anchor_err.square().sum().item()
                update_sq_sum += update.square().sum().item()
                regret = (
                    anchor_err.square().mean(dim=1)
                    - err.square().mean(dim=1)
                ).reshape(-1)
                composer = model.anchored_phase_cycle_fusion
                confidence = 0.5 * (
                    composer.last_level_confidence
                    + composer.last_shape_confidence
                ).mean(dim=-1)
                coefficient = 0.5 * (
                    composer.last_level_coefficient.abs()
                    + composer.last_shape_coefficient.abs()
                ).mean(dim=-1)
                if model.target_var_index != -1:
                    index = model.target_var_index
                    confidence = confidence[:, index : index + 1]
                    coefficient = coefficient[:, index : index + 1]

                def add_moments(moments, value):
                    moments[0] += value.sum().item()
                    moments[1] += regret.sum().item()
                    moments[2] += value.square().sum().item()
                    moments[3] += regret.square().sum().item()
                    moments[4] += (value * regret).sum().item()

                add_moments(confidence_moments, confidence.reshape(-1))
                add_moments(coefficient_moments, coefficient.reshape(-1))
                attribution_count += regret.numel()
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
    if model.use_anchored_phase_cycle_fusion:
        def correlation(values):
            sx, sy, sx2, sy2, sxy = values
            numerator = attribution_count * sxy - sx * sy
            denominator = (
                (attribution_count * sx2 - sx * sx)
                * (attribution_count * sy2 - sy * sy)
            )
            return numerator / max(denominator, 0.0) ** 0.5 if denominator > 0 else 0.0

        anchor_mse = anchor_sq_sum / count
        anchor_mae = anchor_abs_sum / count
        result.update({
            f"{split}_anchor_mae": anchor_mae,
            f"{split}_anchor_mse": anchor_mse,
            f"{split}_mse_ratio_vs_internal_anchor": (
                result[f"{split}_mse"] / anchor_mse
            ),
            f"{split}_mae_ratio_vs_internal_anchor": (
                result[f"{split}_mae"] / anchor_mae
            ),
            f"{split}_update_rms": (update_sq_sum / count) ** 0.5,
            f"{split}_confidence_regret_corr": correlation(confidence_moments),
            f"{split}_coefficient_regret_corr": correlation(coefficient_moments),
        })
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
    if spec["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError(
            "this experiment requires CUDA; refusing a mixed CPU/GPU matrix"
        )
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
    exp_args.dataset_args.input_hypothesis = spec["input_hypothesis"]
    exp_args.dataset_args.input_variant = spec["input_variant"]
    exp_args.dataset_args.input_period_len = spec["period"]
    exp_args.dataset_args.intervention_seed = spec["intervention_seed"]
    exp_args.training_args.num_workers = args.num_workers
    train_set, train_loader = data_provider(exp_args.dataset_args, "train")
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    if len(train_loader) == 0:
        raise ValueError(
            "training split produced zero batches; increase --percent or reduce "
            f"--batch-size (percent={spec['percent']}, batch_size={spec['batch_size']})"
        )
    if spec["max_eval_samples"]:
        val_set = Subset(val_set, range(min(spec["max_eval_samples"], len(val_set))))
        val_loader = DataLoader(
            val_set,
            batch_size=spec["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
    if hasattr(train_set, "data_stamp"):
        hp["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(PhaseFormerPresetConfig(exp_args, spec["lookback"], spec["horizon"], hp))
    init_missing = []
    init_unexpected = []
    anchor_identity_max_abs = ""
    if model.use_safe_triaxis:
        if not spec["init_checkpoint"]:
            raise ValueError("Safe-Regret candidates require --init-checkpoint")
        init_path = Path(spec["init_checkpoint"])
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        if not init_path.is_file():
            raise FileNotFoundError(f"A1 init checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        incompat = model.load_state_dict(state_dict, strict=False)
        init_missing = list(incompat.missing_keys)
        init_unexpected = list(incompat.unexpected_keys)
        unsafe_missing = [
            key for key in init_missing if not key.startswith("safe_triaxis_")
        ]
        if unsafe_missing or init_unexpected:
            raise RuntimeError(
                "A1 checkpoint is not a strict nested subset: "
                f"unsafe_missing={unsafe_missing}, unexpected={init_unexpected}"
            )
        model.eval()
        audit_batch = next(iter(train_loader))
        audit_x, audit_y, audit_x_mark, audit_y_mark = [
            value[:2] if torch.is_tensor(value) else value
            for value in audit_batch
        ]
        with torch.inference_mode():
            audit_dec = model._build_decoder_input(audit_y.float())
            audit_out, _, _ = model(
                audit_x.float(), audit_x_mark.float(), audit_dec,
                audit_y_mark.float(),
            )
        anchor_identity_max_abs = float(
            (audit_out - model.safe_triaxis_anchor_output).abs().max()
        )
        if anchor_identity_max_abs != 0.0:
            raise RuntimeError(
                "Safe-Regret initialization is not exactly the A1 anchor: "
                f"max_abs={anchor_identity_max_abs}"
            )
        model.freeze_safe_triaxis_anchor()
    elif model.use_anchored_phase_cycle_fusion:
        if spec["init_checkpoint"]:
            init_path = Path(spec["init_checkpoint"])
            if not init_path.is_absolute():
                init_path = REPO_ROOT / init_path
            if not init_path.is_file():
                raise FileNotFoundError(f"A2 init checkpoint not found: {init_path}")
            payload = torch.load(init_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("state_dict", payload)
            incompat = model.load_state_dict(state_dict, strict=False)
            init_missing = list(incompat.missing_keys)
            init_unexpected = list(incompat.unexpected_keys)
            unsafe_missing = [
                key for key in init_missing
                if not key.startswith("anchored_phase_cycle_fusion.")
            ]
            if unsafe_missing or init_unexpected:
                raise RuntimeError(
                    "A2 checkpoint is not a strict nested subset: "
                    f"unsafe_missing={unsafe_missing}, "
                    f"unexpected={init_unexpected}"
                )
        elif model.anchored_pctf_freeze_anchor:
            raise ValueError(
                "frozen anchored PCTF diagnostics require --init-checkpoint"
            )
        model.eval()
        audit_batch = next(iter(train_loader))
        audit_x, audit_y, audit_x_mark, audit_y_mark = [
            value[:2] if torch.is_tensor(value) else value
            for value in audit_batch
        ]
        with torch.inference_mode():
            audit_dec = model._build_decoder_input(audit_y.float())
            audit_out, _, _ = model(
                audit_x.float(), audit_x_mark.float(), audit_dec,
                audit_y_mark.float(),
            )
        anchor_identity_max_abs = float(
            (audit_out - model.anchored_pctf_anchor_output).abs().max()
        )
        if anchor_identity_max_abs != 0.0:
            raise RuntimeError(
                "anchored PCTF initialization is not exactly A2: "
                f"max_abs={anchor_identity_max_abs}"
            )
        if model.anchored_pctf_freeze_anchor:
            model.freeze_anchored_pctf_anchor()
    elif spec["init_checkpoint"]:
        init_path = Path(spec["init_checkpoint"])
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

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
        # Use CUDA when the installed PyTorch build exposes it, otherwise
        # fall back to CPU so the experiment protocol remains runnable in
        # environments with a GPU driver but CPU-only PyTorch.
        accelerator="auto",
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
        if spec["max_eval_samples"]:
            test_set = Subset(
                test_set, range(min(spec["max_eval_samples"], len(test_set)))
            )
            test_loader = DataLoader(
                test_set,
                batch_size=spec["batch_size"],
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
        test_metrics = evaluate(model, test_loader, test_set, "test", spec["horizon"], run_dir)
        test_size = len(test_set)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    row = {
        "run_id": rid, "protocol_version": spec["protocol_version"], "stage": spec["stage"],
        "dataset": spec["dataset"], "lookback": spec["lookback"], "horizon": spec["horizon"],
        "mechanism": spec["mechanism"], "period": spec["period"],
        "input_hypothesis": spec["input_hypothesis"],
        "input_variant": spec["input_variant"],
        "intervention_seed": spec["intervention_seed"],
        "max_eval_samples": spec["max_eval_samples"],
        "cycle_period": spec["cycle_period"], "capacity": spec["capacity"],
        "loss": spec["loss"], "learning_rate": hp["learning_rate"], "lr_multiplier": spec["lr_multiplier"],
        "percent": spec["percent"], "seed": spec["seed"], "batch_size": spec["batch_size"],
        "epochs_requested": spec["max_epochs"], "epochs_completed": epoch_count(logger.log_dir),
        "best_val_loss": float(checkpoint.best_model_score.cpu()), **val_metrics, **test_metrics,
        "parameter_count": parameter_count, "elapsed_sec": elapsed, "peak_memory_bytes": peak,
        "trainable_parameter_count": trainable_parameter_count,
        "init_checkpoint": spec["init_checkpoint"],
        "init_missing_count": len(init_missing),
        "init_unexpected_count": len(init_unexpected),
        "anchor_identity_max_abs": anchor_identity_max_abs,
        "anchor_frozen": bool(
            getattr(model, "safe_triaxis_anchor_frozen", False)
            or (
                model.use_anchored_phase_cycle_fusion
                and model.anchored_pctf_freeze_anchor
            )
        ),
        "final_correction_scale": (
            float(model.anchored_phase_cycle_fusion.correction_scale)
            if model.use_anchored_phase_cycle_fusion else ""
        ),
        "required_cuda": spec["require_cuda"],
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
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
    p.add_argument("--stage", required=True, choices=["smoke", "baseline", "period_screen", "mechanism_screen_1", "mechanism_screen_2", "mechanism_full8", "anchor_attribution", "input_components", "hp_low", "hp_mid", "finalist", "confirm"])
    p.add_argument(
        "--mechanism",
        default="original",
        choices=list(MECHANISMS) + ["latest", "best_nonresidual"] + sorted(ABLATION_MODES),
    )
    p.add_argument("--period", type=int, default=24)
    p.add_argument(
        "--input-hypothesis", choices=["none", "h1", "h3", "h4"], default="none",
        help="History-only component extraction applied after train-fitted scaling",
    )
    p.add_argument(
        "--input-variant", choices=["full", "half_A", "minus_A", "sham"],
        default="full",
    )
    p.add_argument("--intervention-seed", type=int, default=9102)
    p.add_argument(
        "--max-eval-samples", type=int, default=0,
        help="Non-formal smoke limit; zero evaluates the complete split",
    )
    p.add_argument(
        "--cycle-period", type=int,
        help="ICPT cycle period; independent of the PhaseFormer phase period",
    )
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--percent", type=int, default=100)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=2021)
    # All listed losses are already implemented by DefaultModule._get_criterion.
    # Exposing the complete set lets the Golden-directed tuner test the
    # MSE--MAE trade-off without adding a new forecasting mechanism.
    p.add_argument(
        "--loss", choices=["mse", "mae", "smae", "huber", "smape"],
        default="huber",
    )
    p.add_argument("--lr-multiplier", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--capacity", choices=["base", "compact"], default="base")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bad-case-limit", type=int, default=8)
    p.add_argument("--overrides", default="{}", help="JSON object applied last")
    p.add_argument(
        "--init-checkpoint",
        help="Pretrained A1 Lightning checkpoint required by Safe-Regret modes",
    )
    p.add_argument("--evaluate-test", action="store_true", help="Allowed only for frozen confirm runs")
    p.add_argument(
        "--require-cuda", action="store_true",
        help="Fail instead of silently falling back to CPU",
    )
    p.add_argument("--output-dir", default="research_runs/search_v1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()
    if args.evaluate_test and args.stage not in ("confirm", "input_components"):
        p.error("--evaluate-test is restricted to confirm or preregistered input-component runs")
    if args.max_eval_samples and args.stage not in ("smoke", "input_components"):
        p.error("--max-eval-samples is a smoke-only diagnostic and cannot be used here")
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
