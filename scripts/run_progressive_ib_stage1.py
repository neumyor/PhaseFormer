#!/usr/bin/env python3
"""Run one frozen-PhaseFormer Stage-1 correction setting.

Checkpoints and full prediction arrays are intentionally written below the
ignored ``*_scratch`` root.  The Stage-1 aggregation command converts completed
runs into the six-file audit directory required by the experiment protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.models.frozen_nlinear_correction import FrozenPhaseNLinearCorrection
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args
from src.training.runner import build_logger, build_trainer, restore_best_checkpoint

def digest(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_phaseformer(config_path, checkpoint_path, num_workers):
    spec = json.loads(Path(config_path).read_text())
    hp = dict(spec["hyperparams"])
    exp = make_exp_args(spec["dataset"], spec["lookback"], spec["horizon"], hp,
                        batch_size=spec["batch_size"])
    exp.dataset_args.num_workers = num_workers
    model = PhaseFormer(PhaseFormerPresetConfig(exp, spec["lookback"], spec["horizon"], hp))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return spec, exp, model


def evaluate(model, loader, device):
    model.eval()
    squared = absolute = count = 0.0
    rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            x, y, xm, ym = [value.to(device).float() for value in batch]
            output, phase, correction = model(x, xm, y, ym)
            target = model._target(y)
            err = output - target
            phase_err = target - phase
            dot = (phase_err * correction).sum(dim=1)
            cosine = dot / (phase_err.square().sum(dim=1).sqrt() * correction.square().sum(dim=1).sqrt()).clamp_min(1e-12)
            squared += err.square().sum().item()
            absolute += err.abs().sum().item()
            count += err.numel()
            per_mae = err.abs().mean(dim=1).cpu().numpy()
            per_mse = err.square().mean(dim=1).cpu().numpy()
            for local_index in range(per_mae.shape[0]):
                origin = batch_index * loader.batch_size + local_index
                for channel in range(per_mae.shape[1]):
                    rows.append({
                        "sample_id": origin, "channel": channel,
                        "mse": float(per_mse[local_index, channel]),
                        "mae": float(per_mae[local_index, channel]),
                        "dot": float(dot[local_index, channel].cpu()),
                        "cosine": float(cosine[local_index, channel].cpu()),
                        "correction_l1": float(correction[local_index, :, channel].abs().mean().cpu()),
                        "correction_l2": float(correction[local_index, :, channel].square().mean().sqrt().cpu()),
                    })
    return {"mse": squared / count, "mae": absolute / count, "sample_rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-config", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(FrozenPhaseNLinearCorrection.MODES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="research_runs/progressive_ib_stage1_scratch")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by Stage 1 but unavailable; no CPU fallback is permitted")
    spec, exp, phaseformer = load_phaseformer(args.phase_config, args.phase_checkpoint, args.num_workers)
    run_name = f"{spec['dataset']}_h{spec['horizon']}_s{args.seed}_{args.mode}"
    run_dir = ROOT / args.output_root / run_name
    if (run_dir / "result.json").exists():
        print(f"completed: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(args.seed, workers=True)
    train_set, train_loader = data_provider(exp.dataset_args, "train")
    val_set, val_loader = data_provider(exp.dataset_args, "val")
    before_hash = digest(phaseformer)
    model = FrozenPhaseNLinearCorrection(
        phaseformer, mode=args.mode, learning_rate=exp.training_args.learning_rate,
        loss_name=exp.training_args.loss_func, huber_delta=exp.training_args.huber_delta,
    )
    logger = build_logger(str(run_dir / "lightning"), name="stage1", version="train")
    trainer, checkpoint = build_trainer(
        max_epochs=args.max_epochs, logger=logger, patience=int(spec["hyperparams"].get("patience", 8)),
        checkpoint_dir=str(run_dir / "checkpoints"), accelerator="auto", progress=args.progress,
    )
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    restore_best_checkpoint(model, checkpoint)
    after_hash = digest(model.phaseformer)
    if before_hash != after_hash:
        raise RuntimeError("frozen PhaseFormer hash changed during Stage 1 training")
    metrics = evaluate(model, val_loader, trainer.strategy.root_device)
    rows = metrics.pop("sample_rows")
    with (run_dir / "sample_metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    result = {
        "dataset": spec["dataset"], "horizon": spec["horizon"], "lookback": spec["lookback"],
        "seed": args.seed, "mode": args.mode, "split": "validation", "mse": metrics["mse"], "mae": metrics["mae"],
        "phase_checkpoint": str(Path(args.phase_checkpoint).resolve()), "phase_config": str(Path(args.phase_config).resolve()),
        "phase_hash_before": before_hash, "phase_hash_after": after_hash,
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "elapsed_sec": time.monotonic() - started,
        "params_total": sum(x.numel() for x in model.parameters()),
        "params_trainable": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "device": str(trainer.strategy.root_device),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
