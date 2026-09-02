#!/usr/bin/env python3
"""Evaluate one retrained Track-R checkpoint on its matching test input."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_input_component_checkpoint import (
    evaluate,
    file_sha256,
    limited_loader,
    load_model,
)
from src.dataset.data_factory import data_provider
from src.models.phaseformer_presets import build_hyperparams, make_exp_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument(
        "--model", required=True,
        choices=["original", "weak_residual", "rcrf_nlinear_plain"],
    )
    parser.add_argument("--input-hypothesis", required=True, choices=["none", "h1", "h3", "h4"])
    parser.add_argument("--input-variant", required=True, choices=["full", "half_A", "minus_A", "sham"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-source", required=True)
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--intervention-seed", type=int, default=9102)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overrides", default="{}")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if (args.input_hypothesis == "none") != (args.input_variant == "full"):
        parser.error("the shared full condition must be exactly none/full")
    if args.max_samples and not args.smoke:
        parser.error("--max-samples requires --smoke; partial data cannot be formal")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("--require-cuda was set but CUDA is unavailable")
    metrics_path = args.output_dir / "retrained_metrics.csv"
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
    exp_args.dataset_args.input_hypothesis = args.input_hypothesis
    exp_args.dataset_args.input_variant = args.input_variant
    exp_args.dataset_args.input_period_len = args.period
    exp_args.dataset_args.intervention_seed = args.intervention_seed

    dataset, loader = data_provider(exp_args.dataset_args, "test")
    dataset, loader = limited_loader(dataset, loader, args.max_samples, args.num_workers)
    model = load_model(args, exp_args)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    metrics, sample_metrics = evaluate(model, loader, args.horizon)
    row = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "model": args.model,
        "input_hypothesis": args.input_hypothesis,
        "input_variant": args.input_variant,
        "track": "retrain",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "selection_source": args.selection_source,
        "evaluation_scope": "smoke" if args.smoke else "formal",
        "max_samples": args.max_samples,
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    np.savez_compressed(
        args.output_dir / "sample_errors.npz",
        mse=sample_metrics["mse"],
        mae=sample_metrics["mae"],
    )
    (args.output_dir / "retrained_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    print(metrics_path)


if __name__ == "__main__":
    main()
