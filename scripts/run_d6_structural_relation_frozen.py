#!/usr/bin/env python3
"""D6 frozen screen for order/co-occurrence relations, without retraining."""

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
from scripts.run_d4_complementary_frozen_probe import branch_counterfactual
from scripts.run_input_candidate_discovery_frozen import evaluate, load_model
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import CandidateDataset, StructuralRelationBank

MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
VARIANTS = ("cycle_order_reverse", "phase_desync", "adjacent_pair_swap")


def qc(full, changed, indices):
    endpoint, cycle_mean_error, changed_count = 0.0, 0.0, 0
    for index in indices[: min(128, len(indices))]:
        x = np.asarray(full[int(index)][0], dtype=np.float64)
        z = np.asarray(changed[int(index)][0], dtype=np.float64)
        endpoint = max(endpoint, float(np.abs(x[-1] - z[-1]).max()))
        cycle_mean_error = max(cycle_mean_error, float(np.abs(
            x.reshape(-1, 24, x.shape[1]).mean(axis=1) - z.reshape(-1, 24, z.shape[1]).mean(axis=1)
        ).max()))
        changed_count += int(np.any(x != z))
    return {"input_endpoint_max_abs": endpoint, "cycle_mean_max_abs_delta": cycle_mean_error,
            "qc_changed_fraction": changed_count / min(128, len(indices))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--weak-checkpoint", type=Path, required=True)
    parser.add_argument("--rcrf-checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available(): parser.error("CUDA required")
    if args.max_samples <= 0: parser.error("D6 discovery requires a positive bounded --max-samples")
    if args.output_dir.exists(): parser.error(f"refusing to overwrite {args.output_dir}")
    paths = (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)
    if not all(path.is_file() for path in paths): parser.error("all checkpoints must exist")
    args.output_dir.mkdir(parents=True); pl.seed_everything(2021, workers=True)
    models, exp_args = {}, None
    for name, path in zip(MODELS, paths):
        model, exp_args = load_model(name, path, 192, 720, 2021)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval(); models[name] = model
    full, _ = data_provider(exp_args.dataset_args, "val")
    indices = np.linspace(0, len(full) - 1, min(args.max_samples, len(full)), dtype=int)
    full_loader = DataLoader(Subset(full, indices.tolist()), batch_size=exp_args.dataset_args.batch_size, shuffle=False)
    rows, samples, anchors = [], {}, {}
    for name, model in models.items():
        metric, per = evaluate(model, full_loader, 192); anchors[name] = (metric, per)
        rows.append({"variant": "full", "model": name, "path": "fused", **metric})
        for key in ("mae", "mse"): samples[f"{name}__full__{key}"] = per[key]
    for variant in VARIANTS:
        changed = CandidateDataset(full, StructuralRelationBank(full.seq_len, variant))
        loader = DataLoader(Subset(changed, indices.tolist()), batch_size=exp_args.dataset_args.batch_size, shuffle=False)
        quality = qc(full, changed, indices)
        for name, model in models.items():
            metric, per = evaluate(model, loader, 192); base, base_per = anchors[name]
            row = {"variant": variant, "model": name, "path": "fused", **quality, **metric}
            for key in ("mae", "mse"):
                row[f"relative_{key}_vs_full"] = metric[key] / base[key] - 1
                low, high = moving_block_effect_interval(base_per[key], per[key], 192, 9163, args.bootstrap_replicates, relative=True)
                row[f"relative_{key}_ci_low"] = low; row[f"relative_{key}_ci_high"] = high
                samples[f"{name}__{variant}__fused_{key}"] = per[key]
            rows.append(row)
            if name != "original":
                cf = branch_counterfactual(model, full_loader, loader, 192)
                row = {"variant": variant, "model": name, "path": "phase_full_nlinear_changed",
                       "mae": cf["mae"], "mse": cf["mse"], "reconstruction_max_abs": cf["reconstruction_max_abs"]}
                for key in ("mae", "mse"):
                    row[f"relative_{key}_vs_full"] = cf[key] / base[key] - 1
                    low, high = moving_block_effect_interval(base_per[key], cf[f"sample_{key}"], 192, 9163, args.bootstrap_replicates, relative=True)
                    row[f"relative_{key}_ci_low"] = low; row[f"relative_{key}_ci_high"] = high
                    samples[f"{name}__{variant}__cf_{key}"] = cf[f"sample_{key}"]
                rows.append(row)
    with (args.output_dir / "frozen_structural_relation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output_dir / "paired_sample_effects.npz", **samples)
    (args.output_dir / "protocol.json").write_text(json.dumps({"dataset": "ETTm1", "split": "validation", "lookback":720, "horizon":192, "seed":2021, "sample_count":len(indices), "variants": VARIANTS, "branch_counterfactual":"(1-alpha_full)*phase_full + alpha_full*nlinear_changed"}, indent=2)+"\n")
    print(args.output_dir / "frozen_structural_relation_results.csv")

if __name__ == "__main__": main()
