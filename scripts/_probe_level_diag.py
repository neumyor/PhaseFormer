#!/usr/bin/env python3
"""Diagnose the level-probe fit: magnitudes of every quantity in the pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/yyk/yyk03/niuyiming/PhaseFormer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lowrank_checkpoint_model import build_loaders, build_model, load_checkpoint_into  # noqa: E402

spec = importlib.util.spec_from_file_location("probe", REPO / "scripts/probe_level_underutilisation.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

device = torch.device("cuda")
rows = [r for r in __import__("csv").DictReader((REPO / "research_runs/phaseformer_L_e14_main_v1/results.csv").open())
        if r["arm"] == "phase_only" and r["status"] == "read" and r["dataset"] == "ETTh2" and r["seed"] == "2021"]
rows.sort(key=lambda r: int(r["horizon"]))
print("available phase_only cells:", [(r["setting"], r["status"]) for r in rows])

row = rows[0]
horizon = int(row["horizon"])
run_dir = REPO / row["run_dir"]
config = json.loads((run_dir / "config.json").read_text())
hyperparams = dict(config["hyperparams"])
print("scheme:", hyperparams.get("scheme_name"), "| use_residual_head:", hyperparams.get("use_residual_head"))
period = probe.DATASET_PERIOD[row["dataset"]]
cycle_count = probe.LOOKBACK // period

exp_args, handles = build_loaders(row["dataset"], probe.LOOKBACK, horizon, hyperparams,
                                  int(config.get("batch_size") or 256), REPO, splits=("train",))
model = build_model(exp_args, probe.LOOKBACK, horizon, hyperparams)
ckpt = sorted((run_dir / "attempts").glob("*/checkpoints/best.ckpt"))[0]
load_checkpoint_into(model, ckpt)
model.to(device).eval()
print("has residual head:", hasattr(model, "weak_period_residual")
      and model.weak_period_residual is not None)

with torch.inference_mode():
    for batch in handles["train"][1]:
        batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
        x, y, x_mark, y_mark = batch
        dec = model._build_decoder_input(y.float())
        outputs = model(x.float(), x_mark.float(), dec, y_mark.float())
        print("model returns", type(outputs), len(outputs) if isinstance(outputs, tuple) else "-")
        out, _, _ = outputs
        truth = y.float()[:, -out.shape[1]:, :]
        xn = x.double().cpu().numpy()
        mu, sigma = probe.revin_stats(xn)
        residual = (truth.double().cpu().numpy() - out.double().cpu().numpy()) / sigma
        print(f"  out      shape {tuple(out.shape)} mean {out.mean():+.4f} std {out.std():.4f}")
        print(f"  truth    shape {tuple(truth.shape)} mean {truth.mean():+.4f} std {truth.std():.4f}")
        print(f"  sigma    shape {sigma.shape} mean {sigma.mean():.4f} min {sigma.min():.4f}")
        print(f"  residual_norm mean {residual.mean():+.4f} std {residual.std():.4f} "
              f"absmax {np.abs(residual).max():.3f}")
        level = probe.cycle_means(xn, period, cycle_count)
        print(f"  level    shape {level.shape} mean {level.mean():+.4f} std {level.std():.4f} "
              f"absmax {np.abs(level).max():.3f}")
        # baseline MSE in value space vs the recorded metric
        err = (truth - out).double().cpu().numpy()
        print(f"  phase-only MSE on this batch: {np.mean(err ** 2):.6f}")
        print(f"  recorded val MSE (full split): {row['val_mse']}")
        break
