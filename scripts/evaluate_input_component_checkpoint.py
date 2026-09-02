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
    sample_mae = []
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
            sample_mae.extend(error.abs().mean(dim=(1, 2)).cpu().tolist())
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
        "sample_mae_mean": float(np.mean(sample_mae)),
        "sample_mae_median": float(np.median(sample_mae)),
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
    return result, {
        "mse": np.asarray(sample_mse, dtype=np.float64),
        "mae": np.asarray(sample_mae, dtype=np.float64),
    }


def input_qc(full_dataset, transformed_dataset, limit, period_len):
    sum_squared = 0.0
    count = 0
    endpoint_max = 0.0
    changed = 0
    nyquist_energy = 0.0
    total_energy = 0.0
    checked = min(len(full_dataset), len(transformed_dataset), limit)
    for index in range(checked):
        full_x = np.asarray(full_dataset[index][0], dtype=np.float64)
        changed_x = np.asarray(transformed_dataset[index][0], dtype=np.float64)
        difference = changed_x - full_x
        sum_squared += np.square(difference).sum()
        count += difference.size
        endpoint_max = max(endpoint_max, float(np.abs(difference[-1]).max()))
        changed += int(np.any(difference != 0.0))
        cycles = full_x.reshape(-1, period_len, full_x.shape[1])
        nyquist = np.sum(
            cycles * ((-1.0) ** np.arange(period_len))[None, :, None], axis=1
        )
        nyquist_energy += np.square(nyquist).sum() / period_len
        total_energy += np.square(cycles).sum()
    return {
        "input_change_rms": float(np.sqrt(sum_squared / max(count, 1))),
        "input_endpoint_max_abs": endpoint_max,
        "qc_changed_fraction": changed / max(checked, 1),
        "nyquist_energy_fraction": nyquist_energy / max(total_energy, 1e-30),
    }


def moving_block_effect_interval(
    full, variant, block_length, seed, replicates, *, relative=False
):
    """Paired moving-block CI for an absolute or relative metric effect."""
    full = np.asarray(full, dtype=np.float64)
    variant = np.asarray(variant, dtype=np.float64)
    if full.shape != variant.shape:
        raise ValueError(f"paired metric shapes differ: {full.shape} != {variant.shape}")
    if len(full) == 0 or replicates <= 0:
        return np.nan, np.nan
    if len(full) <= block_length:
        return np.nan, np.nan
    block_length = max(int(block_length), 1)
    complete_blocks, remainder = divmod(len(full), block_length)
    draw_count = complete_blocks + int(remainder > 0)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(full), size=(replicates, draw_count))

    def circular_block_sums(values, length):
        extended = np.concatenate([values, values[: length - 1]])
        prefix = np.concatenate([[0.0], np.cumsum(extended, dtype=np.float64)])
        return prefix[length : length + len(values)] - prefix[: len(values)]

    full_sums = circular_block_sums(full, block_length)[starts[:, :complete_blocks]].sum(axis=1)
    variant_sums = circular_block_sums(variant, block_length)[starts[:, :complete_blocks]].sum(axis=1)
    if remainder:
        full_sums += circular_block_sums(full, remainder)[starts[:, -1]]
        variant_sums += circular_block_sums(variant, remainder)[starts[:, -1]]
    full_mean = full_sums / len(full)
    variant_mean = variant_sums / len(full)
    estimates = (
        variant_mean / full_mean - 1.0
        if relative
        else variant_mean - full_mean
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def limited_loader(dataset, loader, max_samples, num_workers):
    if not max_samples:
        return dataset, loader
    dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset, DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def evaluate_rcrf_counterfactuals(model, full_loader, variant_loader, pred_len):
    """Recompose the four preregistered RCRF branch/gate counterfactuals."""
    if len(full_loader) != len(variant_loader):
        raise ValueError("full and intervention loaders have different batch counts")
    labels = (
        "branches_variant_gate_full",
        "gate_variant_branches_full",
        "phase_variant",
        "nlinear_variant",
    )
    totals = {label: {"abs": 0.0, "sq": 0.0} for label in labels}
    count = 0
    reconstruction_max = 0.0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for full_batch, variant_batch in zip(full_loader, variant_loader):
            full_x, full_y, full_x_mark, full_y_mark = [item.to(device) for item in full_batch]
            var_x, var_y, var_x_mark, var_y_mark = [item.to(device) for item in variant_batch]
            if not torch.equal(full_y, var_y):
                raise RuntimeError("full and intervention loaders are not target-aligned")

            full_dec = model._build_decoder_input(full_y.float())
            full_out, _, _ = model(
                full_x.float(), full_x_mark.float(), full_dec, full_y_mark.float()
            )
            phase_full = model.last_phase_forecast[:, -pred_len:, :]
            nlinear_full = model.last_residual_forecast[:, -pred_len:, :]
            alpha_full = model.last_rcrf_alpha[:, None, :]

            var_dec = model._build_decoder_input(var_y.float())
            var_out, _, _ = model(
                var_x.float(), var_x_mark.float(), var_dec, var_y_mark.float()
            )
            phase_var = model.last_phase_forecast[:, -pred_len:, :]
            nlinear_var = model.last_residual_forecast[:, -pred_len:, :]
            alpha_var = model.last_rcrf_alpha[:, None, :]
            truth = var_y.float()[:, -pred_len:, :]
            actual = var_out[:, -pred_len:, :]
            if model.target_var_index != -1:
                index = model.target_var_index
                truth = truth[:, :, index : index + 1]
                actual = actual[:, :, index : index + 1]
                phase_full = phase_full[:, :, index : index + 1]
                nlinear_full = nlinear_full[:, :, index : index + 1]
                alpha_full = alpha_full[:, :, index : index + 1]
                phase_var = phase_var[:, :, index : index + 1]
                nlinear_var = nlinear_var[:, :, index : index + 1]
                alpha_var = alpha_var[:, :, index : index + 1]

            reconstructed = (1.0 - alpha_var) * phase_var + alpha_var * nlinear_var
            reconstruction_max = max(
                reconstruction_max, float((actual - reconstructed).abs().max().cpu())
            )
            predictions = {
                "branches_variant_gate_full": (
                    (1.0 - alpha_full) * phase_var + alpha_full * nlinear_var
                ),
                "gate_variant_branches_full": (
                    (1.0 - alpha_var) * phase_full + alpha_var * nlinear_full
                ),
                "phase_variant": (
                    (1.0 - alpha_full) * phase_var + alpha_full * nlinear_full
                ),
                "nlinear_variant": (
                    (1.0 - alpha_full) * phase_full + alpha_full * nlinear_var
                ),
            }
            count += truth.numel()
            for label, prediction in predictions.items():
                error = prediction - truth
                totals[label]["abs"] += error.abs().sum().item()
                totals[label]["sq"] += error.square().sum().item()
    if reconstruction_max >= 2e-5:
        raise RuntimeError(
            f"RCRF branch reconstruction failed: max_abs={reconstruction_max}"
        )
    result = {"fused_reconstruction_max_abs": reconstruction_max}
    for label in labels:
        result[f"cf_{label}_mse"] = totals[label]["sq"] / count
        result[f"cf_{label}_mae"] = totals[label]["abs"] / count
    return result


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overrides", default="{}")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.max_samples and not args.smoke:
        parser.error("--max-samples requires --smoke; partial data cannot be formal")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    metrics_path = args.output_dir / "frozen_metrics.csv"
    if metrics_path.exists():
        parser.error(f"result already exists; use the matrix runner --resume: {metrics_path}")

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
    full_dataset, full_loader = data_provider(exp_args.dataset_args, "test")
    full_dataset, full_loader = limited_loader(
        full_dataset, full_loader, args.max_samples, args.num_workers
    )
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
        dataset, loader = limited_loader(dataset, loader, args.max_samples, args.num_workers)
        metrics, sample_metrics = evaluate(model, loader, args.horizon)
        condition = f"{hypothesis}_{variant}"
        sample_columns[condition] = sample_metrics
        row = {
            "dataset": args.dataset,
            "horizon": args.horizon,
            "seed": args.seed,
            "model": args.model,
            "track": "frozen",
            "hypothesis": hypothesis,
            "variant": variant,
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_scope": "smoke" if args.smoke else "formal",
            "max_samples": args.max_samples,
            **input_qc(full_dataset, dataset, args.qc_samples, args.period),
            **metrics,
        }
        if model.use_rcrf_fusion:
            row.update(
                evaluate_rcrf_counterfactuals(
                    model, full_loader, loader, args.horizon
                )
            )
        rows.append(row)

    full_mse = rows[0]["test_mse"]
    full_mae = rows[0]["test_mae"]
    full_samples = sample_columns["none_full"]
    for row in rows:
        condition = f"{row['hypothesis']}_{row['variant']}"
        variant_samples = sample_columns[condition]
        row["delta_mse_vs_full"] = row["test_mse"] - full_mse
        row["relative_mse_vs_full"] = row["test_mse"] / full_mse - 1.0
        row["delta_mae_vs_full"] = row["test_mae"] - full_mae
        row["relative_mae_vs_full"] = row["test_mae"] / full_mae - 1.0
        for metric in ("mse", "mae"):
            for relative in (False, True):
                low, high = moving_block_effect_interval(
                    full_samples[metric], variant_samples[metric],
                    block_length=args.horizon,
                    seed=args.intervention_seed,
                    replicates=args.bootstrap_replicates,
                    relative=relative,
                )
                prefix = "relative" if relative else "absolute"
                row[f"{metric}_{prefix}_effect_ci_low"] = low
                row[f"{metric}_{prefix}_effect_ci_high"] = high
        if model.use_rcrf_fusion:
            for label in (
                "branches_variant_gate_full", "gate_variant_branches_full",
                "phase_variant", "nlinear_variant",
            ):
                row[f"cf_{label}_relative_mse_vs_full"] = (
                    row[f"cf_{label}_mse"] / full_mse - 1.0
                )
                row[f"cf_{label}_relative_mae_vs_full"] = (
                    row[f"cf_{label}_mae"] / full_mae - 1.0
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "paired_sample_errors.npz",
        **{
            f"{condition}__{metric}": values[metric]
            for condition, values in sample_columns.items()
            for metric in ("mse", "mae")
        },
    )
    (args.output_dir / "frozen_config.json").write_text(
        json.dumps(vars(args) | {"checkpoint_sha256": checkpoint_hash}, indent=2, default=str)
        + "\n"
    )
    print(metrics_path)


if __name__ == "__main__":
    main()
