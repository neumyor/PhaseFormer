#!/usr/bin/env python3
"""Train one frozen-PhaseFormer pooled low-rank NLinear correction setting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytorch_lightning as pl
import torch

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.frozen_nlinear_correction import FrozenPhasePooledLowRankCorrection
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    make_exp_args,
)
from src.training.runner import build_logger, build_trainer, restore_best_checkpoint


def digest(module):
    hasher = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def load_phaseformer(config_path, checkpoint_path, num_workers):
    spec = json.loads(Path(config_path).read_text())
    hp = dict(spec["hyperparams"])
    exp = make_exp_args(
        spec["dataset"],
        spec["lookback"],
        spec["horizon"],
        hp,
        batch_size=spec["batch_size"],
    )
    exp.dataset_args.num_workers = num_workers
    model = PhaseFormer(
        PhaseFormerPresetConfig(exp, spec["lookback"], spec["horizon"], hp)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return spec, exp, model


def evaluate(model, loader, device, *, capture_arrays=False):
    model.eval()
    totals = {
        "candidate_squared": 0.0,
        "candidate_absolute": 0.0,
        "phase_squared": 0.0,
        "phase_absolute": 0.0,
        "count": 0,
    }
    rows = []
    captured = {
        "history": [],
        "truth": [],
        "phase_prediction": [],
        "candidate_prediction": [],
        "sample_id": [],
    }
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, xm, ym = [value.to(device).float() for value in batch]
            output, phase, _ = model(x, xm, y, ym)
            target = model._target(y)
            candidate_error = output - target
            phase_error = phase - target
            totals["candidate_squared"] += candidate_error.square().sum().item()
            totals["candidate_absolute"] += candidate_error.abs().sum().item()
            totals["phase_squared"] += phase_error.square().sum().item()
            totals["phase_absolute"] += phase_error.abs().sum().item()
            totals["count"] += candidate_error.numel()
            candidate_mse = candidate_error.square().mean(dim=1).cpu().numpy()
            candidate_mae = candidate_error.abs().mean(dim=1).cpu().numpy()
            phase_mse = phase_error.square().mean(dim=1).cpu().numpy()
            phase_mae = phase_error.abs().mean(dim=1).cpu().numpy()
            batch_size = candidate_mse.shape[0]
            for local_index in range(batch_size):
                for channel in range(candidate_mse.shape[1]):
                    rows.append(
                        {
                            "sample_id": offset + local_index,
                            "channel": channel,
                            "phase_mse": float(phase_mse[local_index, channel]),
                            "candidate_mse": float(candidate_mse[local_index, channel]),
                            "phase_mae": float(phase_mae[local_index, channel]),
                            "candidate_mae": float(candidate_mae[local_index, channel]),
                        }
                    )
            if capture_arrays:
                captured["history"].append(x.detach().cpu().numpy())
                captured["truth"].append(target.detach().cpu().numpy())
                captured["phase_prediction"].append(phase.detach().cpu().numpy())
                captured["candidate_prediction"].append(output.detach().cpu().numpy())
                captured["sample_id"].append(
                    np.arange(offset, offset + batch_size, dtype=np.int64)
                )
            offset += batch_size
    count = totals.pop("count")
    metrics = {
        "mse": totals["candidate_squared"] / count,
        "mae": totals["candidate_absolute"] / count,
        "phase_mse": totals["phase_squared"] / count,
        "phase_mae": totals["phase_absolute"] / count,
        "sample_rows": rows,
    }
    if capture_arrays:
        metrics["arrays"] = {
            key: np.concatenate(value, axis=0) for key, value in captured.items()
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-config", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--pool-factor", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--smooth-ratio", type=float, required=True)
    parser.add_argument("--smooth-window", type=int, default=24)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        default="research_runs/pooled_lowrank_nlinear_scratch",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    spec, exp, phaseformer = load_phaseformer(
        args.phase_config, args.phase_checkpoint, args.num_workers
    )
    pooled_len = math.ceil(int(spec["lookback"]) / args.pool_factor)
    max_rank = min(pooled_len, int(spec["horizon"]))
    if args.rank > max_rank:
        parser.error(
            f"--rank {args.rank} exceeds factorized-map limit {max_rank} "
            f"for pool_factor={args.pool_factor}"
        )
    run_name = (
        f"{spec['dataset']}_h{spec['horizon']}_s{args.seed}_"
        f"{args.config_id}"
    )
    run_dir = ROOT / args.output_root / run_name
    if (run_dir / "result.json").exists():
        print(f"completed: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(args.seed, workers=True)
    _, train_loader = data_provider(exp.dataset_args, "train")
    _, val_loader = data_provider(exp.dataset_args, "val")
    test_loader = None
    if args.evaluate_test:
        _, test_loader = data_provider(exp.dataset_args, "test")
    before_hash = digest(phaseformer)
    model = FrozenPhasePooledLowRankCorrection(
        phaseformer,
        learning_rate=exp.training_args.learning_rate,
        loss_name=exp.training_args.loss_func,
        huber_delta=exp.training_args.huber_delta,
        pool_factor=args.pool_factor,
        rank=args.rank,
        smooth_ratio=args.smooth_ratio,
        smooth_window=args.smooth_window,
    )
    logger = build_logger(str(run_dir / "lightning"), name="pooled_lowrank", version="train")
    trainer, checkpoint = build_trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        patience=int(spec["hyperparams"].get("patience", 8)),
        checkpoint_dir=str(run_dir / "checkpoints"),
        accelerator="auto",
        progress=args.progress,
    )
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    restore_best_checkpoint(model, checkpoint)
    after_hash = digest(model.phaseformer)
    if before_hash != after_hash:
        raise RuntimeError("frozen PhaseFormer hash changed during training")
    # Lightning >= 2.5 moves modules back to CPU during teardown.
    model.to(trainer.strategy.root_device)
    val_metrics = evaluate(model, val_loader, trainer.strategy.root_device)
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model, test_loader, trainer.strategy.root_device, capture_arrays=True
        )
        arrays = test_metrics.pop("arrays")
        np.savez_compressed(run_dir / "test_arrays.npz", **arrays)
    for split, metrics in (("val", val_metrics), ("test", test_metrics)):
        if metrics is None:
            continue
        rows = metrics.pop("sample_rows")
        with (run_dir / f"{split}_sample_metrics.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    result = {
        "dataset": spec["dataset"],
        "horizon": spec["horizon"],
        "lookback": spec["lookback"],
        "seed": args.seed,
        "config_id": args.config_id,
        "pool_factor": args.pool_factor,
        "pooled_len": pooled_len,
        "rank": args.rank,
        "rank_ratio": args.rank / max_rank,
        "smooth_ratio": args.smooth_ratio,
        "smooth_window": args.smooth_window,
        "split": "validation_and_test" if args.evaluate_test else "validation",
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "val_phase_mse": val_metrics["phase_mse"],
        "val_phase_mae": val_metrics["phase_mae"],
        "test_mse": test_metrics["mse"] if test_metrics else None,
        "test_mae": test_metrics["mae"] if test_metrics else None,
        "test_phase_mse": test_metrics["phase_mse"] if test_metrics else None,
        "test_phase_mae": test_metrics["phase_mae"] if test_metrics else None,
        "phase_checkpoint": str(Path(args.phase_checkpoint).resolve()),
        "phase_config": str(Path(args.phase_config).resolve()),
        "phase_hash_before": before_hash,
        "phase_hash_after": after_hash,
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "best_val_loss": float(checkpoint.best_model_score.cpu()),
        "elapsed_sec": time.monotonic() - started,
        "params_total": sum(parameter.numel() for parameter in model.parameters()),
        "params_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "device": str(trainer.strategy.root_device),
        "torch": torch.__version__,
        "lightning": pl.__version__,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
