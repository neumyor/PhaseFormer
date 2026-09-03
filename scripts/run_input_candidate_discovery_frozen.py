#!/usr/bin/env python3
"""Frozen validation/test screen for the preregistered ETTm1 C1--C7 bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.evaluate_input_component_checkpoint import (
    evaluate_rcrf_counterfactuals,
    moving_block_effect_interval,
)
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import (
    CANDIDATES,
    CandidateConfig,
    CandidateDataset,
    ContinuousCandidateBank,
)
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")
VARIANTS = ("remove_025", "remove_050", "sham_025", "sham_050")
SEGMENTS = (("h1_24", 0, 24), ("h25_48", 24, 48), ("h49_96", 48, 96), ("h97_192", 96, 192))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_name: str, checkpoint: Path, horizon: int, lookback: int, seed: int):
    hp = build_hyperparams("ETTm1", horizon, model_name)
    hp["seed"] = seed
    hp["period_len"] = 24
    exp_args = make_exp_args("ETTm1", lookback, horizon, hp)
    exp_args.dataset_args.root_path = str(REPO_ROOT / "resources" / "all_datasets" / "ETT")
    model = PhaseFormer(PhaseFormerPresetConfig(exp_args, lookback, horizon, hp))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch for {model_name}: {incompat}")
    return model, exp_args


def evaluate(model, loader, horizon: int):
    device = next(model.parameters()).device
    total_abs = total_sq = 0.0
    count = 0
    sample_mse, sample_mae = [], []
    segments = {name: {"abs": 0.0, "sq": 0.0, "count": 0} for name, _, _ in SEGMENTS}
    with torch.inference_mode():
        for batch in loader:
            x, y, xm, ym = [value.to(device) for value in batch]
            decoder = model._build_decoder_input(y.float())
            output, _, _ = model(x.float(), xm.float(), decoder, ym.float())
            prediction = output[:, -horizon:, :]
            truth = y.float()[:, -horizon:, :]
            if model.target_var_index != -1:
                truth = truth[:, :, model.target_var_index : model.target_var_index + 1]
            error = prediction - truth
            total_abs += error.abs().sum().item()
            total_sq += error.square().sum().item()
            count += error.numel()
            sample_mse.extend(error.square().mean(dim=(1, 2)).cpu().tolist())
            sample_mae.extend(error.abs().mean(dim=(1, 2)).cpu().tolist())
            for name, start, end in SEGMENTS:
                part = error[:, start:end]
                segments[name]["abs"] += part.abs().sum().item()
                segments[name]["sq"] += part.square().sum().item()
                segments[name]["count"] += part.numel()
    result = {"mse": total_sq / count, "mae": total_abs / count}
    for name, values in segments.items():
        result[f"{name}_mse"] = values["sq"] / values["count"]
        result[f"{name}_mae"] = values["abs"] / values["count"]
    return result, {"mse": np.asarray(sample_mse), "mae": np.asarray(sample_mae)}


def qc(full: Dataset, changed: Dataset, indices: np.ndarray):
    rms, diff_rms, changed_count = [], [], 0
    for index in indices:
        x = np.asarray(full[int(index)][0], dtype=np.float64)
        z = np.asarray(changed[int(index)][0], dtype=np.float64)
        delta = z - x
        rms.append(np.sqrt(np.mean(np.square(delta))))
        diff_rms.append(np.sqrt(np.mean(np.square(np.diff(delta, axis=0)))))
        changed_count += int(np.any(delta != 0.0))
    return {
        "input_change_rms": float(np.mean(rms)),
        "input_change_diff_rms": float(np.mean(diff_rms)),
        "qc_changed_fraction": changed_count / len(indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--weak-checkpoint", type=Path, required=True)
    parser.add_argument("--rcrf-checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--candidates", default=",".join(CANDIDATES),
        help="Comma-separated preregistered candidates; S1b may only receive S1a finalists.",
    )
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.horizon != 192 or args.lookback != 720:
        parser.error("the preregistered screen is fixed at ETTm1 lookback720/horizon192")
    candidates = tuple(item.strip() for item in args.candidates.split(",") if item.strip())
    unknown = set(candidates) - set(CANDIDATES)
    if unknown or not candidates:
        parser.error(f"unknown/empty candidates: {sorted(unknown)}")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    for checkpoint in (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"frozen_{args.split}_discovery.csv"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")

    checkpoints = dict(zip(MODELS, (args.original_checkpoint, args.weak_checkpoint, args.rcrf_checkpoint)))
    pl.seed_everything(args.seed, workers=True)
    models, exp_args = {}, None
    for name, checkpoint in checkpoints.items():
        model, exp_args = load_model(name, checkpoint, args.horizon, args.lookback, args.seed)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        models[name] = model
    exp_args.dataset_args.num_workers = args.num_workers
    full_dataset, _ = data_provider(exp_args.dataset_args, args.split)
    sample_count = len(full_dataset) if args.max_samples == 0 else min(args.max_samples, len(full_dataset))
    indices = np.linspace(0, len(full_dataset) - 1, sample_count, dtype=int)
    rows, sample_store = [], {}
    full_loader = DataLoader(Subset(full_dataset, indices.tolist()), batch_size=exp_args.dataset_args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=False)
    full_metrics = {}
    for name, model in models.items():
        metrics, samples = evaluate(model, full_loader, args.horizon)
        full_metrics[name] = (metrics, samples)
        rows.append({"split": args.split, "model": name, "candidate": "none", "variant": "full",
                     "checkpoint_sha256": sha256(checkpoints[name]), "sample_count": len(indices), **metrics})
        sample_store[f"{name}__none_full__mse"] = samples["mse"]
        sample_store[f"{name}__none_full__mae"] = samples["mae"]

    for candidate in candidates:
        for variant in VARIANTS:
            bank = ContinuousCandidateBank(full_dataset, CandidateConfig(candidate, variant))
            changed = CandidateDataset(full_dataset, bank)
            loader = DataLoader(Subset(changed, indices.tolist()), batch_size=exp_args.dataset_args.batch_size,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)
            quality = qc(full_dataset, changed, indices[: min(128, len(indices))])
            for name, model in models.items():
                metrics, samples = evaluate(model, loader, args.horizon)
                full_value, full_samples = full_metrics[name]
                row = {"split": args.split, "model": name, "candidate": candidate, "variant": variant,
                       "checkpoint_sha256": sha256(checkpoints[name]), "sample_count": len(indices), **quality, **metrics}
                for metric in ("mse", "mae"):
                    row[f"delta_{metric}_vs_full"] = metrics[metric] - full_value[metric]
                    row[f"relative_{metric}_vs_full"] = metrics[metric] / full_value[metric] - 1.0
                    low, high = moving_block_effect_interval(full_samples[metric], samples[metric], args.horizon,
                                                              9102, args.bootstrap_replicates, relative=True)
                    row[f"relative_{metric}_ci_low"] = low
                    row[f"relative_{metric}_ci_high"] = high
                if name == "rcrf_nlinear_plain":
                    row.update(evaluate_rcrf_counterfactuals(model, full_loader, loader, args.horizon))
                    row["cf_nlinear_variant_delta_mse_vs_full"] = row["cf_nlinear_variant_mse"] - full_value["mse"]
                    row["cf_nlinear_variant_delta_mae_vs_full"] = row["cf_nlinear_variant_mae"] - full_value["mae"]
                rows.append(row)
                sample_store[f"{name}__{candidate}_{variant}__mse"] = samples["mse"]
                sample_store[f"{name}__{candidate}_{variant}__mae"] = samples["mae"]

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / f"frozen_{args.split}_paired_samples.npz", **sample_store)
    (args.output_dir / f"frozen_{args.split}_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
