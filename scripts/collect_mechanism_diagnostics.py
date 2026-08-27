#!/usr/bin/env python3
"""Collect mechanism diagnostics for the periodic-residual next-stage matrix.

For each D1/D2/D3 formal run (12 settings x seed 2021), load the best.ckpt,
run a bounded number of test batches, and capture the head's last_* diagnostic
attributes. Emits one row per (setting, mode) with distribution stats and a
sample-variation check (does the quantity move across cells/batches rather
than collapsing to a constant).

This is post-hoc explanation of already-frozen formal results; it does not
select or modify any candidate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_experiment import (  # noqa: E402
    build_model,
    find_run_dir,
    load_checkpoint,
    parse_setting,
)

RUN_ROOT = Path("research_runs/periodic_residual_next_stage_v1")
RUN_PREFIX = "periodic_residual_next"
SEED = 2021
MODES = {
    "D1": "rcrf_phase_error_memory",
    "D2": "rcrf_dual_reliability_lff",
    "D3": "rcrf_multiperiod",
}


def _moments(name, arr):
    """Mean/std/min/max of an array, with a collapse check across samples."""
    if arr is None or arr.size == 0:
        return {}
    flat = np.asarray(arr, dtype=np.float64)
    n = flat.shape[0]
    if n <= 1:
        collapse = True
    else:
        per_sample = flat.reshape(n, -1).mean(axis=1)
        collapse = bool(float(per_sample.std()) < 1e-9)
    return {
        f"{name}_mean": float(flat.mean()),
        f"{name}_std": float(flat.std()),
        f"{name}_min": float(flat.min()),
        f"{name}_max": float(flat.max()),
        f"{name}_collapsed": collapse,
    }


def collect(model, max_batches, exp_args, horizon, device):
    import torch

    from scripts.analyze_experiment import data_provider

    _, test_loader = data_provider(exp_args.dataset_args, "test")
    attn_ent = []
    d1_gate = []
    d2_r_periodic = []
    d2_gate_periodic = []
    d2_r_phase = []
    d2_gate_phase = []
    d3_rel = []
    d3_weights = []
    d3_gate = []

    d1 = getattr(model, "weak_period_residual", None)
    d2 = getattr(model, "dual_reliability_fusion", None)
    is_d1 = hasattr(d1, "last_attention_entropy")
    is_d3 = hasattr(d1, "last_period_weights")

    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            if max_batches and i >= max_batches:
                break
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            dec = model._build_decoder_input(batch_y.float())
            model(batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float())
            if is_d1:
                ent = getattr(d1, "last_attention_entropy", None)
                if ent is not None:
                    attn_ent.append(ent)
                gate = getattr(d1, "last_correction_gate", None)
                if gate is not None:
                    d1_gate.append(gate.squeeze().cpu().numpy())
            elif is_d3:
                rel = getattr(d1, "last_period_reliability", None)
                if rel is not None:
                    d3_rel.append(rel.cpu().numpy())
                w = getattr(d1, "last_period_weights", None)
                if w is not None:
                    d3_weights.append(w.cpu().numpy())
                gate = getattr(d1, "last_correction_gate", None)
                if gate is not None:
                    d3_gate.append(gate.squeeze().cpu().numpy())
            if d2 is not None:
                for attr, sink in (
                    ("last_periodic_reliability", d2_r_periodic),
                    ("last_periodic_gate", d2_gate_periodic),
                    ("last_phase_reliability", d2_r_phase),
                    ("last_phase_gate", d2_gate_phase),
                ):
                    val = getattr(d2, attr, None)
                    if val is not None:
                        sink.append(val.cpu().numpy())

    def stack(vals):
        return np.concatenate(vals, axis=0) if vals else None

    return {
        **(_moments("attn_entropy", np.asarray(attn_ent, dtype=np.float64) if attn_ent else None)),
        **(_moments("d1_gate", stack(d1_gate))),
        **(_moments("d2_r_periodic", stack(d2_r_periodic))),
        **(_moments("d2_gate_periodic", stack(d2_gate_periodic))),
        **(_moments("d2_r_phase", stack(d2_r_phase))),
        **(_moments("d2_gate_phase", stack(d2_gate_phase))),
        **(_moments("d3_reliability", stack(d3_rel))),
        **(_moments("d3_weights", stack(d3_weights))),
        **(_moments("d3_gate", stack(d3_gate))),
        "d3_period_argmax_share": _period_share(stack(d3_weights)),
    }


def _period_share(weights):
    if weights is None or weights.size == 0:
        return ""
    counts = np.argmax(weights, axis=-1).reshape(-1)
    q = weights.shape[-1]
    n = counts.size
    return ";".join(f"{float((counts == p).mean()):.3f}" for p in range(q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default=str(RUN_ROOT))
    p.add_argument("--run-prefix", default=RUN_PREFIX)
    p.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity")
    p.add_argument("--horizons", default="96,192")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-batches", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="research_runs/periodic_residual_next_stage_v1/mechanism_diagnostics.csv")
    args = p.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    horizons = [int(h) for h in args.horizons.split(",")]
    run_root = Path(args.run_root)
    rows = []
    for label, mode in MODES.items():
        for dataset in datasets:
            for horizon in horizons:
                setting = f"{dataset}_h{horizon}_seed{args.seed}"
                try:
                    run_dir = find_run_dir(run_root, args.run_prefix, mode, dataset, horizon, args.seed)
                except FileNotFoundError:
                    print(f"  !! missing {setting} {mode}", file=sys.stderr)
                    continue
                hp = json.loads((run_dir / "config.json").read_text())["hyperparams"]
                model, exp_args = build_model(dataset, horizon, 720, hp, args.device)
                load_checkpoint(model, run_dir / "checkpoints" / "best.ckpt", args.device)
                diag = collect(model, args.max_batches, exp_args, horizon, args.device)
                row = {"setting": setting, "mode": label, "preset": mode, **diag}
                rows.append(row)
                print(f"{setting} {label}: " + " ".join(
                    f"{k}={v:.4g}" for k, v in diag.items() if isinstance(v, float) and ("_mean" in k or "_std" in k)
                ), flush=True)
                del model
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
