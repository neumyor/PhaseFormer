#!/usr/bin/env python3
"""Validation-only D1 spectral and D2 recent-innovation remove screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input_candidate_discovery_frozen import evaluate, load_model
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import (
    CandidateDataset,
    GaussianNotchBank,
    TailZeroBank,
)


# Fixed before validation evaluation from the aggregated, train-only ETTm1
# periodogram.  The non-integer values are exact DFT periods (N_train / bin).
D1_PERIODS = (
    ("D1-1", 96.0), ("D1-2", 48.0), ("D1-3", 32.0), ("D1-4", 24.0),
    ("D1-5", 677.6470588235294), ("D1-6", 205.71428571428572),
)
D2_LENGTHS = (24, 48, 96, 192)
MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")


def qc(full, changed, limit=256):
    rms, changed_count = [], 0
    for index in range(min(len(full), limit)):
        x = np.asarray(full[index][0], dtype=np.float64)
        z = np.asarray(changed[index][0], dtype=np.float64)
        delta = z - x
        rms.append(float(np.sqrt(np.mean(np.square(delta)))))
        changed_count += int(np.any(delta != 0.0))
    return {"input_change_rms": float(np.mean(rms)), "qc_changed_fraction": changed_count / min(len(full), limit)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--weak-checkpoint", type=Path, required=True)
    parser.add_argument("--rcrf-checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checkpoints = dict(zip(MODELS, (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)))
    if not all(path.is_file() for path in checkpoints.values()):
        parser.error("all three anchor checkpoints must exist")
    pl.seed_everything(2021, workers=True)
    models, exp_args = {}, None
    for name, checkpoint in checkpoints.items():
        model, exp_args = load_model(name, checkpoint, 192, 720, 2021)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        models[name] = model
    exp_args.dataset_args.num_workers = args.num_workers
    full, _ = data_provider(exp_args.dataset_args, "val")
    loader = DataLoader(full, batch_size=exp_args.dataset_args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    rows = []
    full_metrics = {}
    for name, model in models.items():
        metric, _ = evaluate(model, loader, 192)
        full_metrics[name] = metric
        rows.append({"track": "full", "component": "full", "period_or_length": "", "model": name, **metric})

    interventions = []
    for label, period in D1_PERIODS:
        interventions.append(("D1", label, period, GaussianNotchBank(full.seq_len, period, 1.0 / full.seq_len)))
    for length in D2_LENGTHS:
        bank = TailZeroBank(full.seq_len, length)
        # D2 directly zeroes the selected final history observations.
        interventions.append(("D2", f"D2-{length}", length, bank))

    for track, label, value, bank in interventions:
        changed = CandidateDataset(full, bank)
        changed_loader = DataLoader(changed, batch_size=exp_args.dataset_args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        quality = qc(full, changed)
        for name, model in models.items():
            metric, _ = evaluate(model, changed_loader, 192)
            base = full_metrics[name]
            row = {"track": track, "component": label, "period_or_length": value, "model": name, **quality, **metric}
            row["relative_mae_vs_full"] = metric["mae"] / base["mae"] - 1.0
            row["relative_mse_vs_full"] = metric["mse"] / base["mse"] - 1.0
            rows.append(row)
    with (args.output_dir / "d1_d2_remove_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "d1_d2_protocol.json").write_text(json.dumps({
        "split": "validation", "dataset": "ETTm1", "horizon": 192, "seed": 2021,
        "intervention": "remove only; D1 Gaussian frequency notch (sigma=1/720); D2 tail zeros",
        "d1_periods": D1_PERIODS, "d2_lengths": D2_LENGTHS,
    }, indent=2) + "\n")
    print(args.output_dir / "d1_d2_remove_validation.csv")


if __name__ == "__main__":
    main()
