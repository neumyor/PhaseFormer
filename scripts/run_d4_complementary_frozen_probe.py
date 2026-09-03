#!/usr/bin/env python3
"""Low-cost frozen D4 probe: is a D3 trajectory A sufficient, or is B used?

The experiment never retrains and reads validation only.  For each selected D3
trajectory A it compares the normal history X with (1) ``X-A`` and (2) an
anchor-preserving A-only view, ``last(X)+A``.  For M1/M2 it additionally holds
the full-input phase path and fusion weight fixed while replacing only the
NLinear-style branch by the branch computed from the changed input.
"""

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

from scripts.evaluate_input_component_checkpoint import moving_block_effect_interval
from scripts.run_input_candidate_discovery_frozen import evaluate, load_model
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import CandidateDataset, ComplementaryTrajectoryBank


MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
COMPONENTS = ("recent_linear", "cycle_levels")
VARIANTS = ("remainder", "component_anchor")


def _branch_alpha(model):
    if model.use_rcrf_fusion:
        return model.last_rcrf_alpha[:, None, :]
    return model.last_weak_residual_alpha


def _target(tensor, model):
    if model.target_var_index == -1:
        return tensor
    index = model.target_var_index
    return tensor[:, :, index : index + 1]


def branch_counterfactual(model, full_loader, changed_loader, horizon):
    """Return only-NLinear replacement errors and exact fusion replay QC."""
    total_abs = total_sq = 0.0
    count = 0
    sample_mae, sample_mse = [], []
    reconstruction_max = 0.0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for full_batch, changed_batch in zip(full_loader, changed_loader):
            fx, fy, fxm, fym = [item.to(device) for item in full_batch]
            cx, cy, cxm, cym = [item.to(device) for item in changed_batch]
            if not torch.equal(fy, cy):
                raise RuntimeError("full/changed targets are not aligned")
            fdec = model._build_decoder_input(fy.float())
            fout, _, _ = model(fx.float(), fxm.float(), fdec, fym.float())
            phase_full = _target(model.last_phase_forecast[:, -horizon:, :], model)
            alpha_full = _target(_branch_alpha(model), model)
            cdec = model._build_decoder_input(cy.float())
            cout, _, _ = model(cx.float(), cxm.float(), cdec, cym.float())
            residual_changed = _target(model.last_residual_forecast[:, -horizon:, :], model)
            actual = _target(cout[:, -horizon:, :], model)
            phase_changed = _target(model.last_phase_forecast[:, -horizon:, :], model)
            alpha_changed = _target(_branch_alpha(model), model)
            replay = (1.0 - alpha_changed) * phase_changed + alpha_changed * residual_changed
            reconstruction_max = max(reconstruction_max, float((actual - replay).abs().max().cpu()))
            prediction = (1.0 - alpha_full) * phase_full + alpha_full * residual_changed
            truth = _target(cy.float()[:, -horizon:, :], model)
            error = prediction - truth
            total_abs += error.abs().sum().item()
            total_sq += error.square().sum().item()
            count += error.numel()
            sample_mae.extend(error.abs().mean(dim=(1, 2)).cpu().tolist())
            sample_mse.extend(error.square().mean(dim=(1, 2)).cpu().tolist())
    if reconstruction_max >= 2e-5:
        raise RuntimeError(f"branch fusion replay failed: {reconstruction_max}")
    return {
        "mae": total_abs / count, "mse": total_sq / count,
        "reconstruction_max_abs": reconstruction_max,
        "sample_mae": np.asarray(sample_mae), "sample_mse": np.asarray(sample_mse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--weak-checkpoint", type=Path, required=True)
    parser.add_argument("--rcrf-checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite {args.output_dir}")
    if not all(path.is_file() for path in (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)):
        parser.error("all three checkpoints must exist")
    args.output_dir.mkdir(parents=True)
    pl.seed_everything(2021, workers=True)
    checkpoints = dict(zip(MODELS, (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)))
    models, exp_args = {}, None
    for name, checkpoint in checkpoints.items():
        model, exp_args = load_model(name, checkpoint, 192, 720, 2021)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        models[name] = model
    exp_args.dataset_args.num_workers = args.num_workers
    full, _ = data_provider(exp_args.dataset_args, "val")
    full_loader = DataLoader(full, batch_size=exp_args.dataset_args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    rows, samples = [], {}
    full_metrics = {}
    for name, model in models.items():
        metric, per_sample = evaluate(model, full_loader, 192)
        full_metrics[name] = (metric, per_sample)
        rows.append({"component": "none", "variant": "full", "model": name, "path": "fused", **metric})
        samples[f"{name}__full__mae"] = per_sample["mae"]
        samples[f"{name}__full__mse"] = per_sample["mse"]

    for component in COMPONENTS:
        for variant in VARIANTS:
            changed = CandidateDataset(full, ComplementaryTrajectoryBank(
                full.seq_len, component, variant=variant
            ))
            changed_loader = DataLoader(changed, batch_size=exp_args.dataset_args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)
            for name, model in models.items():
                metric, per_sample = evaluate(model, changed_loader, 192)
                baseline, base_samples = full_metrics[name]
                row = {"component": component, "variant": variant, "model": name, "path": "fused", **metric}
                for key in ("mae", "mse"):
                    row[f"delta_{key}_vs_full"] = metric[key] - baseline[key]
                    row[f"relative_{key}_vs_full"] = metric[key] / baseline[key] - 1.0
                    low, high = moving_block_effect_interval(base_samples[key], per_sample[key], 192, 9137,
                                                              args.bootstrap_replicates, relative=True)
                    row[f"relative_{key}_ci_low"] = low
                    row[f"relative_{key}_ci_high"] = high
                rows.append(row)
                samples[f"{name}__{component}_{variant}__fused_mae"] = per_sample["mae"]
                samples[f"{name}__{component}_{variant}__fused_mse"] = per_sample["mse"]
                if name != "original":
                    cf = branch_counterfactual(model, full_loader, changed_loader, 192)
                    cfrow = {"component": component, "variant": variant, "model": name,
                             "path": "phase_full_nlinear_changed", "mae": cf["mae"], "mse": cf["mse"],
                             "reconstruction_max_abs": cf["reconstruction_max_abs"]}
                    for key in ("mae", "mse"):
                        cfrow[f"delta_{key}_vs_full"] = cf[key] - baseline[key]
                        cfrow[f"relative_{key}_vs_full"] = cf[key] / baseline[key] - 1.0
                        low, high = moving_block_effect_interval(base_samples[key], cf[f"sample_{key}"], 192, 9137,
                                                                  args.bootstrap_replicates, relative=True)
                        cfrow[f"relative_{key}_ci_low"] = low
                        cfrow[f"relative_{key}_ci_high"] = high
                    rows.append(cfrow)
                    samples[f"{name}__{component}_{variant}__cf_mae"] = cf["sample_mae"]
                    samples[f"{name}__{component}_{variant}__cf_mse"] = cf["sample_mse"]
    with (args.output_dir / "frozen_complementary_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output_dir / "paired_sample_effects.npz", **samples)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "dataset": "ETTm1", "split": "validation", "lookback": 720, "horizon": 192, "seed": 2021,
        "components": COMPONENTS, "views": {
            "remainder": "X-A", "component_anchor": "repeat(last(X))+A"
        }, "models": MODELS, "branch_counterfactual": "phase_full + alpha_full * nlinear_changed",
        "checkpoint_paths": {name: str(path) for name, path in checkpoints.items()},
        "bootstrap_replicates": args.bootstrap_replicates,
    }, indent=2) + "\n")
    print(args.output_dir / "frozen_complementary_results.csv")


if __name__ == "__main__":
    main()
