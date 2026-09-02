#!/usr/bin/env python3
"""Evaluate one frozen checkpoint on all preregistered H1/H3/H4 inputs."""

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

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, build_hyperparams, make_exp_args


CONDITIONS = [("none", "full")] + [
    (hypothesis, variant)
    for hypothesis in ("h1", "h3", "h4")
    for variant in ("half_A", "minus_A", "sham")
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args, exp_args):
    hp = build_hyperparams(args.dataset, args.horizon, args.model)
    hp.update(json.loads(args.overrides))
    hp["seed"] = args.seed
    hp["period_len"] = args.period
    model = PhaseFormer(PhaseFormerPresetConfig(exp_args, args.lookback, args.horizon, hp))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompat}")
    return model


def evaluate(model, loader, pred_len):
    device = next(model.parameters()).device
    totals = {
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "count": 0,
        "phase_squared_error": 0.0,
        "residual_squared_error": 0.0,
        "alpha_sum": 0.0,
        "reliability_sum": 0.0,
        "gate_count": 0,
    }
    sample_mse = []
    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [item.to(device) for item in batch]
            decoder = model._build_decoder_input(batch_y.float())
            output, _, _ = model(
                batch_x.float(), batch_x_mark.float(), decoder, batch_y_mark.float()
            )
            prediction = output[:, -pred_len:, :]
            truth = batch_y.float()[:, -pred_len:, :]
            if model.target_var_index != -1:
                truth = truth[:, :, model.target_var_index : model.target_var_index + 1]
            error = prediction - truth
            totals["absolute_error"] += error.abs().sum().item()
            totals["squared_error"] += error.square().sum().item()
            totals["count"] += error.numel()
            sample_mse.extend(error.square().mean(dim=(1, 2)).cpu().tolist())
            if model.last_phase_forecast is not None:
                phase = model.last_phase_forecast[:, -pred_len:, :]
                residual = model.last_residual_forecast[:, -pred_len:, :]
                if model.target_var_index != -1:
                    index = model.target_var_index
                    phase = phase[:, :, index : index + 1]
                    residual = residual[:, :, index : index + 1]
                totals["phase_squared_error"] += (phase - truth).square().sum().item()
                totals["residual_squared_error"] += (residual - truth).square().sum().item()
                totals["alpha_sum"] += model.last_rcrf_alpha.sum().item()
                totals["reliability_sum"] += model.last_rcrf_reliability.sum().item()
                totals["gate_count"] += model.last_rcrf_alpha.numel()
    count = totals["count"]
    result = {
        "test_mae": totals["absolute_error"] / count,
        "test_mse": totals["squared_error"] / count,
        "sample_mse_mean": float(np.mean(sample_mse)),
        "sample_mse_median": float(np.median(sample_mse)),
        "sample_count": len(sample_mse),
    }
    if totals["gate_count"]:
        result.update(
            phase_test_mse=totals["phase_squared_error"] / count,
            residual_test_mse=totals["residual_squared_error"] / count,
            rcrf_alpha_mean=totals["alpha_sum"] / totals["gate_count"],
            rcrf_reliability_mean=(
                totals["reliability_sum"] / totals["gate_count"]
            ),
        )
    return result, np.asarray(sample_mse, dtype=np.float64)


def input_qc(full_dataset, transformed_dataset, limit):
    sum_squared = 0.0
    count = 0
    endpoint_max = 0.0
    changed = 0
    checked = min(len(full_dataset), len(transformed_dataset), limit)
    for index in range(checked):
        full_x = np.asarray(full_dataset[index][0], dtype=np.float64)
        changed_x = np.asarray(transformed_dataset[index][0], dtype=np.float64)
        difference = changed_x - full_x
        sum_squared += np.square(difference).sum()
        count += difference.size
        endpoint_max = max(endpoint_max, float(np.abs(difference[-1]).max()))
        changed += int(np.any(difference != 0.0))
    return {
        "input_change_rms": float(np.sqrt(sum_squared / max(count, 1))),
        "input_endpoint_max_abs": endpoint_max,
        "qc_changed_fraction": changed / max(checked, 1),
    }


def moving_block_interval(delta, block_length, seed, replicates):
    """Paired moving-block bootstrap over consecutive forecast origins."""
    delta = np.asarray(delta, dtype=np.float64)
    if len(delta) == 0 or replicates <= 0:
        return np.nan, np.nan
    if len(delta) <= block_length:
        return np.nan, np.nan
    block_length = max(int(block_length), 1)
    block_count = int(np.ceil(len(delta) / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        starts = rng.integers(0, len(delta), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(delta)
        estimates[replicate] = delta[indices.reshape(-1)[: len(delta)]].mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument(
        "--model", required=True,
        choices=["original", "weak_residual", "rcrf_nlinear_plain"],
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--intervention-seed", type=int, default=9102)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--qc-samples", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Smoke-test limit; keep zero for every formal evaluation",
    )
    parser.add_argument("--overrides", default="{}")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    pl.seed_everything(args.seed, workers=True)
    hp = build_hyperparams(args.dataset, args.horizon, args.model)
    hp.update(json.loads(args.overrides))
    exp_args = make_exp_args(
        args.dataset, args.lookback, args.horizon, hp, batch_size=args.batch_size
    )
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers
    configured = Path(exp_args.dataset_args.root_path)
    fallback = REPO_ROOT / "resources" / "all_datasets" / "ETT-small"
    if not configured.exists() and args.dataset.startswith("ETT") and fallback.exists():
        exp_args.dataset_args.root_path = str(fallback)

    exp_args.dataset_args.input_hypothesis = "none"
    exp_args.dataset_args.input_variant = "full"
    full_dataset, _ = data_provider(exp_args.dataset_args, "test")
    model = load_model(args, exp_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    checkpoint_hash = file_sha256(args.checkpoint)
    rows = []
    sample_columns = {}
    for hypothesis, variant in CONDITIONS:
        exp_args.dataset_args.input_hypothesis = hypothesis
        exp_args.dataset_args.input_variant = variant
        exp_args.dataset_args.input_period_len = args.period
        exp_args.dataset_args.intervention_seed = args.intervention_seed
        dataset, loader = data_provider(exp_args.dataset_args, "test")
        if args.max_samples:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
            loader = DataLoader(
                dataset,
                batch_size=exp_args.dataset_args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
        metrics, sample_mse = evaluate(model, loader, args.horizon)
        condition = f"{hypothesis}_{variant}"
        sample_columns[condition] = sample_mse
        row = {
            "dataset": args.dataset,
            "horizon": args.horizon,
            "seed": args.seed,
            "model": args.model,
            "hypothesis": hypothesis,
            "variant": variant,
            "checkpoint_sha256": checkpoint_hash,
            **input_qc(full_dataset, dataset, args.qc_samples),
            **metrics,
        }
        rows.append(row)

    full_mse = rows[0]["test_mse"]
    full_samples = sample_columns["none_full"]
    for row in rows:
        condition = f"{row['hypothesis']}_{row['variant']}"
        paired_delta = sample_columns[condition] - full_samples
        ci_low, ci_high = moving_block_interval(
            paired_delta,
            block_length=args.horizon,
            seed=args.intervention_seed,
            replicates=args.bootstrap_replicates,
        )
        row["delta_mse_vs_full"] = row["test_mse"] - full_mse
        row["relative_mse_vs_full"] = row["test_mse"] / full_mse - 1.0
        row["mean_paired_sample_delta"] = float(np.mean(paired_delta))
        row["paired_delta_ci_low"] = ci_low
        row["paired_delta_ci_high"] = ci_high

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "frozen_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "paired_sample_mse.npz", **sample_columns)
    (args.output_dir / "frozen_config.json").write_text(
        json.dumps(vars(args) | {"checkpoint_sha256": checkpoint_hash}, indent=2, default=str)
        + "\n"
    )
    print(metrics_path)


if __name__ == "__main__":
    main()
