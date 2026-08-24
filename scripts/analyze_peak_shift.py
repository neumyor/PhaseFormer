#!/usr/bin/env python3
"""Peak-shift error analysis for the next-stage paper plan (stage 1 metric).

The plan requires MSE/MAE/Peak shift error for the phase-trajectory comparison
(A0 baseline / A1 phase offset / A2 phase velocity). This script recomputes the
test predictions of each mode's best.ckpt and reports, per mode and setting:

  - mse / mae (recomputed, should match the runner's metrics.csv)
  - peak shift error: within each period_len segment of the horizon, the
    circular distance between the argmax position of truth and prediction,
    averaged over (sample, period, channel)
  - peak within +/-3 steps fraction (positional agreement)
  - peak amplitude error: |max(truth_seg) - max(pred_seg)| averaged

Reuses analyze_experiment.py's model-reconstruction + checkpoint-load helpers
so the predictions are computed exactly as in the objective error analysis.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_experiment import (  # noqa: E402
    build_model,
    evaluate_test,
    find_run_dir,
    load_checkpoint,
    parse_setting,
)

PERIOD = 24


def peak_metrics(pred, truth, period=PERIOD):
    """Per-period peak position/amplitude metrics.

    pred/truth: (N, H, C). Segments the horizon into period_len windows,
    computes the argmax position inside each window and the window max value.
    Returns dict with peak shift error, within-k fraction, amplitude error.
    """
    N, H, C = truth.shape
    n_per = H // period
    if n_per == 0:
        n_per = 1
    seg_t = truth[:, : n_per * period, :].reshape(N, n_per, period, C)
    seg_p = pred[:, : n_per * period, :].reshape(N, n_per, period, C)
    pos_t = seg_t.argmax(axis=2)  # (N, n_per, C)
    pos_p = seg_p.argmax(axis=2)
    shift = np.abs(pos_t - pos_p).astype(np.float32)
    shift = np.minimum(shift, period - shift)  # circular distance on the cycle
    peak_shift_err = float(shift.mean())
    within3 = float((shift <= 3).mean())
    amp_t = seg_t.max(axis=2)  # (N, n_per, C)
    amp_p = seg_p.max(axis=2)
    amp_err = float(np.abs(amp_t - amp_p).mean())
    return {
        "peak_shift_err": peak_shift_err,
        "peak_within3": within3,
        "peak_amp_err": amp_err,
        "n_periods": n_per,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--modes", required=True, help="comma-separated mode names")
    p.add_argument("--settings", required=True, help="comma-separated ETTh2_h336_seed2021")
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="research_runs/next_stage_peak_shift.csv")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    rows = []
    for setting in settings:
        ds, horizon, seed = parse_setting(setting)
        for mode in modes:
            run_dir = find_run_dir(args.run_dir, args.run_prefix, mode, ds, horizon, seed)
            import json

            hp = json.loads((run_dir / "config.json").read_text())["hyperparams"]
            model, exp_args = build_model(ds, horizon, args.lookback, hp, args.device)
            load_checkpoint(model, run_dir / "checkpoints" / "best.ckpt", args.device)
            pred, truth, _, _ = evaluate_test(model, exp_args, horizon, args.device)
            err = pred - truth
            mse = float(np.square(err).mean())
            mae = float(np.abs(err).mean())
            pk = peak_metrics(pred, truth)
            rows.append(
                {
                    "setting": setting,
                    "mode": mode,
                    "dataset": ds,
                    "horizon": horizon,
                    "seed": seed,
                    "mse": mse,
                    "mae": mae,
                    "peak_shift_err": pk["peak_shift_err"],
                    "peak_within3": pk["peak_within3"],
                    "peak_amp_err": pk["peak_amp_err"],
                    "n_samples": int(pred.shape[0]),
                    "n_channels": int(pred.shape[2]),
                    "n_periods": pk["n_periods"],
                }
            )
            print(
                f"{setting:24s} {mode:18s} mse={mse:.5f} mae={mae:.5f} "
                f"peak_shift={pk['peak_shift_err']:.3f} within3={pk['peak_within3']:.3f} "
                f"amp_err={pk['peak_amp_err']:.5f}",
                flush=True,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
