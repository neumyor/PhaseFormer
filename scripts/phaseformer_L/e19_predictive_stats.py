#!/usr/bin/env python3
"""E19 stage 1: train-split cross-cycle level statistics for all 28 settings.

minipaper §4.7 asks for three train-set statistics per setting
(``cycle_level_std``, ``last_cycle_shift``, ``tau_hat``) to be correlated with
the §4.2 gain and the fusion gate ``g``.  §3.4.3 freezes the switch

    s(dataset) = 1[ nu_train(dataset) > nu* ],   nu_train in {cycle_level_std,
                                                          last_cycle_shift, tau_hat}

so the statistics must exist, and ``nu*`` must be frozen, *before* the §4.2
matrix starts.  This script produces only the statistics; the correlation
columns need §4.2 and are E19 stage 2.

**This script never constructs a test loader.**  It reads the train split only.

Definitions
-----------
``cycle_level_std``, ``cycle_amplitude_std`` and ``last_cycle_shift`` are taken
verbatim from the D7 window descriptors already registered in
``docs/PhaseFormer_structural_defect_research_narrative.md`` §A.6 and implemented
in ``scripts/run_d7_internal_path_probe.py::features``:

    cyc                = x.reshape(B, 720 // P, P, C)
    means              = cyc.mean(2)                       # l[k, c] per-cycle level
    cycle_level_std    = mean_c std_k(means[k, c])
    cycle_amplitude_std= mean_c std_k(std_p cyc[k, p, c])
    last_cycle_shift   = mean_c |l[K-1, c] - mean_{k<K-1} l[k, c]|

They are computed on the loader's input tensor, i.e. after the train-fitted
global ``StandardScaler`` and before the model's own RevIN, which is exactly the
convention D7 used.  Nothing is re-scaled here.

``tau_hat`` (level memory length) is NEW -- §4.7 names the quantity but no
earlier script defines it, so its definition is frozen here, before any §4.2
result exists:

    rho_c       = Pearson corr over k of (l[0:K-1, c], l[1:K, c])
    tau_c      = -1 / ln(rho_c)      if 0 < rho_c < 1
                = 0                   if rho_c <= 0      (no level memory)
                = TAU_CYCLE_CAP       if rho_c >= 1      (memory exceeds window)
    tau_c      = min(tau_c, TAU_CYCLE_CAP)   # applied to the RESULT: a rho of
                                             # 0.99999 also exceeds the window
    tau_hat     = P * mean_c tau_c                        # expressed in STEPS

The per-window value is then averaged over train windows.  ``tau_hat`` is
reported in steps so that it is directly comparable with the learned EMA
timescale tau = 6..72 reported in minipaper §4.1/§4.7.  The fraction of channels
saturating the cap is recorded so a degenerate fit cannot pass unnoticed.

Known finite-sample bias (disclosed, not corrected)
---------------------------------------------------
``rho_c`` is a lag-1 autocorrelation estimated from only K = 30 cycle levels, so
it is biased toward zero by roughly ``(1 + 3*rho)/K`` (Kendall/Marriott-Pope for
an AR(1) with an estimated mean).  Consequently ``tau_hat`` **understates** the
population level-memory length; on a synthetic AR(1) with phi = 0.5 it returns
about 1.25 cycles where the population value is 1.44.  This is left uncorrected
on purpose:

* K = 30 is identical for every setting (lookback 720 / period 24), so the bias
  is comparable across settings and does not change the *ranking* of datasets,
  which is all §3.4.3's threshold and §4.7's Spearman rho use it for;
* the bias shrinks in absolute terms as the true memory grows, so it is
  monotone-preserving.  Measured on synthetic AR(1) levels
  (``tests/test_phaseformer_L_e19_stats.py::_ar1_tau``, 200 channels x 40
  windows, cycles):

  | phi  | population tau | measured tau_hat | tau_hat / population |
  |-----:|---:|---:|---:|
  | 0.1  | 0.434 | 0.352 | 0.81 |
  | 0.3  | 0.831 | 0.725 | 0.87 |
  | 0.5  | 1.443 | 1.245 | 0.86 |
  | 0.7  | 2.804 | 2.169 | 0.77 |
  | 0.9  | 9.491 | 4.530 | 0.48 |

  The estimate is a faithful *ordinal* instrument on the 0.1-0.7 range that
  real settings occupy; it compresses strongly-memorable series, so §4.7 must
  not read ``tau_hat`` as an absolute memory length.

Any correction would be an additional modelling choice that must be frozen
before use, and §4.7 only requires the quantity to be *defined* and frozen.  The
bias must be stated in the §4.7 table note.

Usage::

    /home/yyk/yyk03/miniconda3/envs/time/bin/python \
        scripts/phaseformer_L/e19_predictive_stats.py \
        --output-root research_runs/phaseformer_L_e19_predictive_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.phaseformer_presets import make_exp_args  # noqa: E402

ALL_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
ALL_HORIZONS = (96, 192, 336, 720)

LOOKBACK = 720
PERIOD = 24
TAU_CYCLE_CAP = 30.0  # 720 // 24; a longer memory cannot be resolved in-window

# Sign of the LevelFormer branch's benefit over phase_only, as registered in
# minipaper §4.1 (single seed, static gate, test-exposed; motivation only).
# Used ONLY to diagnose candidate nu/threshold combinations during the freeze;
# it is not a test-set label and no test data is read to build it.
PILOT_SIGN = {"ETTh2": +1, "ETTm2": +1, "Weather": +1, "ETTh1": -1, "ETTm1": -1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(ALL_DATASETS))
    parser.add_argument("--horizons", default=",".join(str(h) for h in ALL_HORIZONS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = dataset default")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="cap windows per setting (0 = every train window); smoke only",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def window_descriptors(batch_x: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-window D7 descriptors plus tau_hat, all channel-averaged.

    ``batch_x`` is (B, L, C) with L == LOOKBACK, already loader-normalized.
    Returns 1-D arrays of length B.
    """
    a = batch_x.detach().cpu().numpy()
    n, length, channels = a.shape
    if length != LOOKBACK:
        raise ValueError(f"expected lookback {LOOKBACK}, got {length}")
    cycles, phases = LOOKBACK // PERIOD, PERIOD
    cyc = a.reshape(n, cycles, phases, channels)

    means = cyc.mean(2)               # (B, K, C) per-cycle level l[k, c]
    amp = cyc.std(2)                  # (B, K, C) per-cycle amplitude

    # D7 descriptors, verbatim semantics.
    local_diff = np.abs(np.diff(a[:, -96:], axis=1)).mean((1, 2))
    recent_deviation = np.abs(a[:, -96:] - a[:, -96:].mean(1, keepdims=True)).mean((1, 2))
    cycle_level_std = means.std(1).mean(1)
    cycle_amplitude_std = amp.std(1).mean(1)
    last_cycle_shift = np.abs(means[:, -1] - means[:, :-1].mean(1)).mean(1)
    daily_lag_change = np.abs(a[:, -96:] - a[:, -192:-96]).mean((1, 2))

    # tau_hat: exponential fit to the lag-1 autocorrelation of the level series.
    left, right = means[:, :-1, :], means[:, 1:, :]          # (B, K-1, C)
    left = left - left.mean(1, keepdims=True)
    right = right - right.mean(1, keepdims=True)
    denom = np.sqrt((left ** 2).sum(1) * (right ** 2).sum(1))
    numer = (left * right).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denom > 0, numer / np.maximum(denom, 1e-12), np.nan)

    tau_cycles = np.full_like(rho, np.nan)
    ok = np.isfinite(rho) & (rho > 0.0) & (rho < 1.0)
    tau_cycles[ok] = -1.0 / np.log(rho[ok])
    tau_cycles[np.isfinite(rho) & (rho <= 0.0)] = 0.0
    # A rho in (1 - eps, 1) is numerically indistinguishable from a perfectly
    # persistent level but -1/ln(rho) diverges there, so the cap must be applied
    # to the RESULT, not only to the rho >= 1 branch. Without this the estimator
    # silently reports memory far longer than the 30-cycle window (observed:
    # 2176 steps on ETTh2-720 against a 720-step window).
    tau_cycles[np.isfinite(tau_cycles)] = np.minimum(
        tau_cycles[np.isfinite(tau_cycles)], TAU_CYCLE_CAP
    )
    capped = np.isfinite(tau_cycles) & (tau_cycles >= TAU_CYCLE_CAP)
    valid = np.isfinite(tau_cycles)
    tau_steps = np.where(valid, tau_cycles * PERIOD, np.nan)

    return {
        "local_diff": local_diff,
        "recent_deviation": recent_deviation,
        "cycle_level_std": cycle_level_std,
        "cycle_amplitude_std": cycle_amplitude_std,
        "last_cycle_shift": last_cycle_shift,
        "daily_lag_change": daily_lag_change,
        "tau_hat_steps": np.nanmean(tau_steps, axis=1),
        "tau_hat_cycles": np.nanmean(
            np.where(valid, tau_cycles, np.nan), axis=1
        ),
        "tau_capped_frac": capped.mean(axis=1),
        "rho_mean": np.nanmean(np.where(np.isfinite(rho), rho, np.nan), axis=1),
        "n_channels": np.full(n, channels, dtype=float),
    }


DESCRIPTOR_KEYS = (
    "local_diff",
    "recent_deviation",
    "cycle_level_std",
    "cycle_amplitude_std",
    "last_cycle_shift",
    "daily_lag_change",
    "tau_hat_steps",
    "tau_hat_cycles",
    "tau_capped_frac",
    "rho_mean",
)


def build_train_loader(dataset: str, horizon: int, batch_size: int, num_workers: int):
    hyper = {"learning_rate": 1e-3, "train_epochs": 30}
    exp_args = make_exp_args(
        dataset, LOOKBACK, horizon, hyper, batch_size=batch_size or None
    )
    exp_args.dataset_args.percent = 100
    exp_args.dataset_args.num_workers = num_workers
    train_set, train_loader = data_provider(exp_args.dataset_args, "train")
    return train_set, train_loader


def scan_setting(dataset, horizon, batch_size, num_workers, max_windows):
    train_set, loader = build_train_loader(dataset, horizon, batch_size, num_workers)
    acc = {key: [] for key in DESCRIPTOR_KEYS}
    seen = 0
    for batch in loader:
        batch_x = batch[0]
        if max_windows and seen >= max_windows:
            break
        if max_windows:
            batch_x = batch_x[: max_windows - seen]
        stats = window_descriptors(batch_x.float())
        for key in DESCRIPTOR_KEYS:
            acc[key].append(stats[key])
        seen += int(batch_x.shape[0])
        if max_windows and seen >= max_windows:
            break
    if not seen:
        raise RuntimeError(f"{dataset}-{horizon}: train split produced no windows")
    row = {"dataset": dataset, "horizon": horizon, "n_windows": seen,
           "train_size": len(train_set)}
    for key in DESCRIPTOR_KEYS:
        values = np.concatenate(acc[key]).astype(float)
        row[key] = float(np.nanmean(values))
        row[f"{key}__sd"] = float(np.nanstd(values))
    return row


def parse_list(raw: str, cast=str):
    return [cast(item) for item in str(raw).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    datasets = parse_list(args.datasets)
    horizons = parse_list(args.horizons, int)
    out_root = ROOT / args.output_root

    cells = [(d, h) for d in datasets for h in horizons]
    print(json.dumps({"event": "planned", "cells": len(cells), "settings": [
        f"{d}-{h}" for d, h in cells]}), flush=True)
    if args.dry_run:
        return
    if out_root.exists():
        raise SystemExit(f"refusing to overwrite existing {out_root}")

    (out_root / "figures").mkdir(parents=True)
    rows = []
    for dataset, horizon in cells:
        row = scan_setting(
            dataset, horizon, args.batch_size, args.num_workers, args.max_windows
        )
        rows.append(row)
        print(json.dumps({
            "event": "setting_done",
            "setting": f"{dataset}-{horizon}",
            "n_windows": row["n_windows"],
            "cycle_level_std": round(row["cycle_level_std"], 6),
            "last_cycle_shift": round(row["last_cycle_shift"], 6),
            "tau_hat_steps": round(row["tau_hat_steps"], 3),
        }), flush=True)

    fields = ["dataset", "horizon", "n_windows", "train_size"] + [
        f for key in DESCRIPTOR_KEYS for f in (key, f"{key}__sd")
    ]
    with (out_root / "level_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Dataset-level aggregation: s() is indexed by dataset in §3.4.3.
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    dataset_rows = []
    for dataset, group in by_dataset.items():
        entry = {
            "dataset": dataset,
            "n_horizons": len(group),
            "pilot_sign": PILOT_SIGN.get(dataset, 0),
        }
        for key in ("cycle_level_std", "last_cycle_shift", "tau_hat_steps",
                    "cycle_amplitude_std", "tau_capped_frac", "rho_mean"):
            entry[key] = float(np.mean([g[key] for g in group]))
        dataset_rows.append(entry)
    dataset_rows.sort(key=lambda r: -r["cycle_level_std"])
    with (out_root / "dataset_level_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dataset_rows)

    # Diagnostic for the nu* freeze: how each candidate nu ranks the datasets
    # whose pilot sign is known.  No test data is involved.
    diag = []
    for key in ("cycle_level_std", "last_cycle_shift", "tau_hat_steps"):
        labelled = [r for r in dataset_rows if r["pilot_sign"] != 0]
        pos = sorted(r[key] for r in labelled if r["pilot_sign"] > 0)
        neg = sorted(r[key] for r in labelled if r["pilot_sign"] < 0)
        separated = bool(neg and pos and max(neg) < min(pos))
        diag.append({
            "nu": key,
            "positive_datasets": [r["dataset"] for r in labelled if r["pilot_sign"] > 0],
            "negative_datasets": [r["dataset"] for r in labelled if r["pilot_sign"] < 0],
            "max_negative_value": max(neg) if neg else None,
            "min_positive_value": min(pos) if pos else None,
            "perfectly_separated": separated,
        })
    (out_root / "nu_star_diagnostic.json").write_text(
        json.dumps({"note": "diagnostic only; no test data read", "candidates": diag},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    protocol = {
        "experiment": "E19 stage 1 (minipaper §4.7 train-set statistics)",
        "reads_test": False,
        "split": "train",
        "lookback": LOOKBACK,
        "period": PERIOD,
        "datasets": datasets,
        "horizons": horizons,
        "n_settings": len(rows),
        "max_windows": args.max_windows,
        "tau_cycle_cap": TAU_CYCLE_CAP,
        "inherited_definitions": (
            "cycle_level_std / cycle_amplitude_std / last_cycle_shift from "
            "scripts/run_d7_internal_path_probe.py::features, registered in "
            "docs/PhaseFormer_structural_defect_research_narrative.md A.6"
        ),
        "new_definitions": "tau_hat (see module docstring), frozen before any §4.2 result",
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    (out_root / "run.yaml").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "finished", "rows": len(rows),
                      "output": str(out_root)}), flush=True)


if __name__ == "__main__":
    main()
