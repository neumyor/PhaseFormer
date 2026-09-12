"""Shared helpers for the low-rank-compression mechanism analysis scripts.

All four `analyze_weak_residual_*.py` scripts reuse the same run-lookup and
model/eval plumbing as ``scripts/analyze_joint_pooled_lowrank_experiment.py``
(``evaluate_run``), factored out here to avoid duplicating the dataset/model
construction boilerplate across scripts. This module does not change any
production code path; it only reads existing checkpoints under
``research_runs/rank_sweep_2_stage1``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.PhaseFormer import PhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    PhaseFormerPresetConfig,
    make_exp_args,
)

SEED = 2021
RANK_SWEEP_ROOT = ROOT / "research_runs/rank_sweep_2_stage1"
CAUSAL_EMA_SWEEP_ROOT = ROOT / "research_runs/causal_ema_smooth_sweep_v1"

# The 7 test-exposed, conditionally selected settings shared by every round of
# this research thread (rank sweep, boxcar sweep, causal-EMA sweep).
SETTINGS = [
    {"dataset": "ETTh2", "horizon": 96},
    {"dataset": "ETTh2", "horizon": 720},
    {"dataset": "ETTm2", "horizon": 96},
    {"dataset": "ETTm2", "horizon": 192},
    {"dataset": "Weather", "horizon": 96},
    {"dataset": "Weather", "horizon": 192},
    {"dataset": "Electricity", "horizon": 336},
]


def setting_label(dataset: str, horizon: int) -> str:
    return f"{dataset}-{horizon}"


def load_weak_residual_rows(dataset: str, horizon: int, seed: int = SEED) -> list[dict]:
    """Load every weak_residual run (full-rank + low-rank) for one setting."""
    rows = []
    for config_path in sorted((RANK_SWEEP_ROOT / "runs").glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if (
            config["dataset"] != dataset
            or int(config["horizon"]) != horizon
            or int(config["seed"]) != seed
            or config["mechanism"] != "weak_residual"
        ):
            continue
        metrics_path = config_path.with_name("metrics.csv")
        with metrics_path.open(newline="") as handle:
            metrics = next(csv.DictReader(handle))
        hp = config["hyperparams"]
        head_type = hp.get("weak_period_residual_head_type", "shared")
        rank = int(hp["weak_period_residual_rank"]) if head_type == "pooled_lowrank" else None
        rows.append(
            {
                "config": config,
                "metrics": metrics,
                "head_type": head_type,
                "rank": rank,
                "run_dir": config_path.parent,
            }
        )
    return rows


def full_rank_row(dataset: str, horizon: int, seed: int = SEED) -> dict:
    rows = [r for r in load_weak_residual_rows(dataset, horizon, seed) if r["head_type"] == "shared"]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one full-rank (shared) run for {dataset} h{horizon}, found {len(rows)}"
        )
    return rows[0]


def low_rank_rows(dataset: str, horizon: int, seed: int = SEED) -> list[dict]:
    rows = [r for r in load_weak_residual_rows(dataset, horizon, seed) if r["head_type"] == "pooled_lowrank"]
    return sorted(rows, key=lambda r: r["rank"])


def causal_ema_row(dataset: str, horizon: int, smooth_ratio: float, seed: int = SEED) -> dict:
    """Load a single run row from the server-side causal-EMA smoothing sweep.

    Only available where research_runs/causal_ema_smooth_sweep_v1/runs exists
    (the remote server) -- the local checkout only carries the summary CSVs.
    """
    matches = []
    for config_path in sorted((CAUSAL_EMA_SWEEP_ROOT / "runs").glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if (
            config["dataset"] != dataset
            or int(config["horizon"]) != horizon
            or int(config["seed"]) != seed
        ):
            continue
        hp = config["hyperparams"]
        if abs(float(hp.get("weak_period_residual_smooth_ratio", 0.0)) - smooth_ratio) > 1e-9:
            continue
        metrics_path = config_path.with_name("metrics.csv")
        with metrics_path.open(newline="") as handle:
            metrics = next(csv.DictReader(handle))
        matches.append({"config": config, "metrics": metrics, "head_type": "shared", "rank": None})
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one causal-EMA run for {dataset} h{horizon} s={smooth_ratio}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_checkpoint_state_dict(row: dict) -> dict:
    checkpoint = ROOT / row["metrics"]["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["state_dict"]


def build_model_and_loader(row: dict, *, num_workers: int = 4):
    """Reconstruct the PhaseFormer model + test dataloader for one run row.

    Mirrors ``evaluate_run`` in scripts/analyze_joint_pooled_lowrank_experiment.py
    but stops short of loading the checkpoint, so callers can monkeypatch
    weights in between construction and evaluation.
    """
    config = row["config"]
    hp = dict(config["hyperparams"])
    exp = make_exp_args(
        config["dataset"],
        config["lookback"],
        config["horizon"],
        hp,
        batch_size=config["batch_size"],
    )
    exp.dataset_args.num_workers = num_workers
    _, loader = data_provider(exp.dataset_args, "test")
    model = PhaseFormer(PhaseFormerPresetConfig(exp, config["lookback"], config["horizon"], hp))
    return model, loader


def evaluate_model(model, loader, horizon: int, device, *, x_transform=None) -> dict:
    """Real test-set MSE/MAE for an already-constructed, already-weighted model.

    ``x_transform`` (optional): callable applied to the input tensor ``x``
    right before the forward pass (used by the SVD-truncation and
    frequency-band-probe scripts); leaves ``y`` untouched.
    """
    model.to(device).eval()
    total_abs = total_sq = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, xm, ym = [value.to(device).float() for value in batch]
            if x_transform is not None:
                x = x_transform(x)
            dec = model._build_decoder_input(y)
            prediction, _, _ = model(x, xm, dec, ym)
            prediction = prediction[:, -horizon:, :]
            target = y[:, -horizon:, :]
            if model.target_var_index != -1:
                target = target[:, :, model.target_var_index : model.target_var_index + 1]
            error = prediction - target
            total_abs += error.abs().sum().item()
            total_sq += error.square().sum().item()
            count += error.numel()
    return {"mse": total_sq / count, "mae": total_abs / count}


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
