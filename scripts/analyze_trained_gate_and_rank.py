#!/usr/bin/env python3
"""Learned fusion gate + trained-map spectrum of the NLinear residual branch.

Reads existing checkpoints only: no training, no dataset, no forward passes.

Two quantities matter for interpreting the rank sweep:

  * ``gate``  = sigmoid(weak_period_residual_gate) per channel.  The fused
    forecast is ``(1-g)*phase + g*residual``, so a branch-error increase of
    delta only reaches the reported metric as roughly ``g^2 * delta`` in MSE.
    The learned gate therefore bounds how much compression damage can hide.

  * the singular spectrum of the *trained* effective map (``linear.weight`` for
    the unfactored direct head, ``decoder.weight @ encoder.weight`` for the
    pooled low-rank head).  This can be compared against the spectrum of the
    analytically optimal map computed by
    ``scripts/analyze_optimal_lowrank_capture.py``.

Usage:

    python scripts/analyze_trained_gate_and_rank.py \
        --runs-root research_runs \
        --output-dir research_runs/lowrank_data_property_v1
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

# Lightning checkpoints unpickle experiment objects from the repository itself,
# so the repo root must be importable regardless of where this file lives.
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

RUN_PATTERN = re.compile(
    r"confirm_(?P<dataset>[a-z0-9]+)_h(?P<horizon>\d+)_"
    r"(?P<mechanism>[a-z_]+)_p(?P<period>\d+)_.*_s(?P<seed>\d+)_"
)


def spectral_stats(vector: np.ndarray) -> tuple[float, float]:
    spectrum = np.abs(np.fft.rfft(vector))
    total = float(spectrum.sum()) + 1e-12
    freqs = np.arange(len(spectrum))
    centroid = float((freqs * spectrum).sum() / total) / max(1, len(spectrum))
    cutoff = len(spectrum) // 2
    return centroid, float(spectrum[cutoff:].sum() / total)


def parse_run(run_dir: str) -> dict:
    name = os.path.basename(run_dir.rstrip("/"))
    match = RUN_PATTERN.search(name)
    if not match:
        return {}
    return {
        "dataset": match.group("dataset"),
        "horizon": int(match.group("horizon")),
        "mechanism": match.group("mechanism"),
        "seed": int(match.group("seed")),
    }


def analyse_checkpoint(path: str) -> dict | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if "weak_period_residual_gate" not in state:
        return None
    gate = torch.sigmoid(state["weak_period_residual_gate"].float()).flatten().numpy()

    if "weak_period_residual.linear.weight" in state:
        head = "direct"
        weight = state["weak_period_residual.linear.weight"].float().numpy()
        rank_cfg = ""
    elif "weak_period_residual.encoder.weight" in state:
        head = "pooled_lowrank"
        enc = state["weak_period_residual.encoder.weight"].float().numpy()
        dec = state["weak_period_residual.decoder.weight"].float().numpy()
        weight = dec @ enc
        rank_cfg = int(enc.shape[0])
    else:
        return None

    svals = np.linalg.svd(weight, compute_uv=True)[1]
    energy = np.cumsum(svals**2) / np.sum(svals**2)
    _, _, vwt = np.linalg.svd(weight, full_matrices=False)
    cents, highs = zip(*[spectral_stats(v) for v in vwt[:5]]) if len(vwt) else ((0.0,), (0.0,))
    # .../runs/<run_id>/attempts/001/checkpoints/best.ckpt -> four dirnames up
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    run = parse_run(run_dir)

    return {
        "run_dir": run_dir,
        "head": head,
        "cfg_rank": rank_cfg,
        "gate_mean": round(float(gate.mean()), 3),
        "gate_min": round(float(gate.min()), 3),
        "gate_max": round(float(gate.max()), 3),
        "gate_std": round(float(gate.std()), 3),
        "gate_sq_mean": round(float((gate**2).mean()), 4),
        "n_channels": len(gate),
        "map_rank90": int(np.searchsorted(energy, 0.90) + 1),
        "map_rank95": int(np.searchsorted(energy, 0.95) + 1),
        "map_rank99": int(np.searchsorted(energy, 0.99) + 1),
        "map_participation_ratio": round(float(svals.sum() ** 2 / np.sum(svals**2)), 2),
        "top5_singular_share": [
            round(float(x), 5) for x in (svals[:5] ** 2 / np.sum(svals**2))
        ],
        "top5_input_pattern_high_freq_frac": [round(float(x), 3) for x in highs],
        "top5_input_pattern_centroid": [round(float(x), 3) for x in cents],
        **run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="research_runs")
    parser.add_argument("--output-dir", default="research_runs/lowrank_data_property_v1")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_root, "**", "best.ckpt"), recursive=True))
    rows = []
    for path in paths:
        try:
            row = analyse_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[skip] {path}: {exc}")
            continue
        if row:
            rows.append(row)
            print(
                f"[ok] {os.path.basename(row['run_dir'])[:60]} head={row['head']} "
                f"gate={row['gate_mean']} rank90={row['map_rank90']}",
                flush=True,
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no matching checkpoints")
        return
    out = out_dir / "trained_gate_and_spectrum.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[written] {out} ({len(rows)} runs from {len(paths)} checkpoints)")


if __name__ == "__main__":
    main()
