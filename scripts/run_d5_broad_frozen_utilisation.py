#!/usr/bin/env python3
"""Broad, no-retraining frozen utilisation screen for current D1/D2/D3 inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_input_component_checkpoint import moving_block_effect_interval
from scripts.run_d1_d2_remove_screen import D1_PERIODS, D2_LENGTHS
from scripts.run_d4_complementary_frozen_probe import branch_counterfactual
from scripts.run_input_candidate_discovery_frozen import evaluate, load_model
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import (
    CandidateDataset, GaussianNotchBank, TailZeroBank, TrajectoryComponentBank,
)


MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
D3_COMPONENTS = (
    ("D3-global-linear", "global_linear"),
    ("D3-recent-linear", "recent_linear"),
    ("D3-cycle-levels", "cycle_levels"),
    ("D3-phase-drift", "phase_drift"),
    ("D3-cycle-amplitude", "cycle_amplitude"),
)


def interventions(seq_len: int):
    for label, period in D1_PERIODS:
        yield "D1", label, GaussianNotchBank(seq_len, period, 1.0 / seq_len)
    for length in D2_LENGTHS:
        yield "D2", f"D2-tail-zero-{length}", TailZeroBank(seq_len, length)
    for label, component in D3_COMPONENTS:
        yield "D3", label, TrajectoryComponentBank(seq_len, component)


def qc(full, changed, limit=256):
    squared, count, endpoint, changed_count = 0.0, 0, 0.0, 0
    for index in range(min(limit, len(full))):
        x = np.asarray(full[index][0], dtype=np.float64)
        z = np.asarray(changed[index][0], dtype=np.float64)
        delta = z - x
        squared += np.square(delta).sum(); count += delta.size
        endpoint = max(endpoint, float(np.abs(delta[-1]).max()))
        changed_count += int(np.any(delta != 0.0))
    return {"input_change_rms": float(np.sqrt(squared / count)),
            "input_endpoint_max_abs": endpoint,
            "qc_changed_fraction": changed_count / min(limit, len(full))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--weak-checkpoint", type=Path, required=True)
    parser.add_argument("--rcrf-checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples", type=int, default=512,
        help="Uniformly spaced validation origins; 0 means all origins for a confirmed finalist.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    checks = (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)
    if not all(path.is_file() for path in checks):
        parser.error("all three checkpoints must exist")
    args.output_dir.mkdir(parents=True)
    pl.seed_everything(2021, workers=True)
    checkpoints = dict(zip(MODELS, checks))
    models, exp_args = {}, None
    for name, checkpoint in checkpoints.items():
        model, exp_args = load_model(name, checkpoint, 192, 720, 2021)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        models[name] = model
    exp_args.dataset_args.num_workers = args.num_workers
    full, _ = data_provider(exp_args.dataset_args, "val")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    sample_count = len(full) if args.max_samples == 0 else min(args.max_samples, len(full))
    indices = np.linspace(0, len(full) - 1, sample_count, dtype=int)
    evaluated_full = Subset(full, indices.tolist())
    full_loader = DataLoader(evaluated_full, batch_size=exp_args.dataset_args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    rows, samples, full_metrics = [], {}, {}
    for name, model in models.items():
        metric, per_sample = evaluate(model, full_loader, 192)
        full_metrics[name] = (metric, per_sample)
        rows.append({"family": "full", "component": "full", "model": name, "path": "fused", **metric})
        for key in ("mae", "mse"):
            samples[f"{name}__full__{key}"] = per_sample[key]

    for family, component, bank in interventions(full.seq_len):
        changed = CandidateDataset(full, bank)
        changed_loader = DataLoader(Subset(changed, indices.tolist()), batch_size=exp_args.dataset_args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)
        quality = qc(full, changed)
        for name, model in models.items():
            metric, per_sample = evaluate(model, changed_loader, 192)
            baseline, baseline_samples = full_metrics[name]
            row = {"family": family, "component": component, "model": name, "path": "fused", **quality, **metric}
            for key in ("mae", "mse"):
                row[f"delta_{key}_vs_full"] = metric[key] - baseline[key]
                row[f"relative_{key}_vs_full"] = metric[key] / baseline[key] - 1.0
                low, high = moving_block_effect_interval(baseline_samples[key], per_sample[key], 192, 9151,
                                                          args.bootstrap_replicates, relative=True)
                row[f"relative_{key}_ci_low"] = low; row[f"relative_{key}_ci_high"] = high
                samples[f"{name}__{component}__fused_{key}"] = per_sample[key]
            rows.append(row)
            if name != "original":
                counterfactual = branch_counterfactual(model, full_loader, changed_loader, 192)
                cfrow = {"family": family, "component": component, "model": name,
                         "path": "phase_full_nlinear_changed", "mae": counterfactual["mae"],
                         "mse": counterfactual["mse"],
                         "reconstruction_max_abs": counterfactual["reconstruction_max_abs"]}
                for key in ("mae", "mse"):
                    cfrow[f"delta_{key}_vs_full"] = counterfactual[key] - baseline[key]
                    cfrow[f"relative_{key}_vs_full"] = counterfactual[key] / baseline[key] - 1.0
                    low, high = moving_block_effect_interval(
                        baseline_samples[key], counterfactual[f"sample_{key}"], 192, 9151,
                        args.bootstrap_replicates, relative=True,
                    )
                    cfrow[f"relative_{key}_ci_low"] = low; cfrow[f"relative_{key}_ci_high"] = high
                    samples[f"{name}__{component}__cf_{key}"] = counterfactual[f"sample_{key}"]
                rows.append(cfrow)
    with (args.output_dir / "frozen_broad_utilisation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output_dir / "paired_sample_effects.npz", **samples)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "dataset": "ETTm1", "split": "validation", "lookback": 720, "horizon": 192, "seed": 2021,
        "models": MODELS, "D1": "Gaussian notch at fixed train-spectrum periods, sigma=1/720",
        "D2": "zero final standardized input window", "D3": "endpoint-anchored trajectory removal",
        "branch_counterfactual": "(1-alpha_full)*phase_full + alpha_full*nlinear_changed",
        "sample_count": sample_count, "bootstrap_replicates": args.bootstrap_replicates,
    }, indent=2) + "\n")
    print(args.output_dir / "frozen_broad_utilisation_results.csv")


if __name__ == "__main__":
    main()
