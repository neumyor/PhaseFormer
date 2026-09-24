#!/usr/bin/env python3
"""Is the level probe overfitting, or is something structurally wrong?"""

from __future__ import annotations

import csv
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
rows = [r for r in csv.DictReader((REPO / "research_runs/phaseformer_L_e14_main_v1/results.csv").open())
        if r["arm"] == "phase_only" and r["setting"] == "ETTh2-192" and r["seed"] == "2021"]
row = rows[0]
horizon = int(row["horizon"])
run_dir = REPO / row["run_dir"]
config = json.loads((run_dir / "config.json").read_text())
hyperparams = dict(config["hyperparams"])
period = probe.DATASET_PERIOD[row["dataset"]]
cycle_count = probe.LOOKBACK // period

exp_args, handles = build_loaders(row["dataset"], probe.LOOKBACK, horizon, hyperparams,
                                  int(config.get("batch_size") or 256), REPO,
                                  splits=("train", "val", "test"))
model = build_model(exp_args, probe.LOOKBACK, horizon, hyperparams)
load_checkpoint_into(model, sorted((run_dir / "attempts").glob("*/checkpoints/best.ckpt"))[0])
model.to(device).eval()

splits = {name: probe.forward_split(model, handles[name][1], device, period, cycle_count)
          for name in ("train", "val", "test")}
for name, split in splits.items():
    print(f"{name}: samples={split['samples']} level{np.asarray(split['level']).shape} "
          f"target absmax={np.abs(split['target']).max():.3f}")

train = splits["train"]
print("\n--- level features ---")
level = train["level"]
print(f"  |f| max {np.abs(level).max():.3f}   per-feature std: "
      f"{np.round(level.reshape(-1, level.shape[-1]).std(axis=0)[:6], 4)} ...")
print(f"  target std {train['target'].std():.4f}  mean {train['target'].mean():+.4f}")

# condition of the feature Gram matrix
s_ff, s_yf = train["statistics"]["level"]
eigs = np.linalg.eigvalsh(0.5 * (s_ff + s_ff.T))
print(f"  S_ff eigenvalues: max {eigs.max():.4e} min {eigs.min():.4e} "
      f"condition {eigs.max() / max(eigs.min(), 1e-300):.3e}")

print("\n--- train / val / test MSE before and after, by penalty ---")
for name, split in splits.items():
    before = float(np.mean(split["target"] ** 2))
    print(f"  {name:5s} phase-only MSE = {before:.6f}")
print()
header = f"{'penalty':>10s} {'train_after':>12s} {'val_after':>12s} {'test_after':>12s}"
print(header)
for penalty in (0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e8):
    weight = probe.ridge_fit(s_ff, s_yf, penalty)
    cells = []
    for name in ("train", "val", "test"):
        split = splits[name]
        value = probe.score(probe.with_intercept(split["level"]), split["target"], weight)
        cells.append(value["mse_after"])
    print(f"{penalty:>10.0e} {cells[0]:>12.6f} {cells[1]:>12.6f} {cells[2]:>12.6f}")

print("\n--- same for the PCA control ---")
rng = np.random.default_rng(2021)
principal = probe._principal_axes(train["window_moment"], train["window_sum"],
                                  train["samples"], cycle_count)
proj = {name: probe.forward_split(model, handles[name][1], device, period, cycle_count,
                                  projections={"pca": principal})
        for name in ("train", "val", "test")}
for penalty in (0.1, 1, 10, 100, 1e3, 1e4, 1e6):
    weight = probe.ridge_fit(*probe.moments(
        probe.with_intercept(proj["train"]["projected"]["pca"]),
        proj["train"]["target"])[:2], penalty)
    cells = [probe.score(probe.with_intercept(proj[name]["projected"]["pca"]),
                         proj[name]["target"], weight)["mse_after"]
             for name in ("train", "val", "test")]
    print(f"{penalty:>10.0e} {cells[0]:>12.6f} {cells[1]:>12.6f} {cells[2]:>12.6f}")
