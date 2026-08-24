#!/usr/bin/env python3
"""Aggregate next-stage full-budget results into the plan's comparison tables.

Reads every metrics.csv under the full-budget run dir, keys by
(dataset, horizon, seed, mode), and prints/CSV-exports the plan's tables:

  阶段1 A0/A1/A2, 阶段2 B1/B2, 阶段3 R0/R1/R2,
  模块消融 Baseline/A/B/C, phase evolution Static/Offset/Velocity,
  residual None/Fixed/Adaptive.

delta% convention: (candidate - baseline) / baseline * 100, negative = better.
"""

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="research_runs/dyn_phase_full")
    p.add_argument("--output", default="research_runs/next_stage_summary.csv")
    p.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,Electricity,Traffic")
    p.add_argument("--horizons", default="336,720")
    args = p.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    horizons = [int(h) for h in args.horizons.split(",") if h]

    # key -> {mode -> (mse, mae, epochs)}
    results = defaultdict(dict)
    for f in glob.glob(f"{args.run_dir}/*/metrics.csv"):
        with open(f) as fh:
            row = list(csv.DictReader(fh))[0]
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]))
        results[key][row["mode"]] = (float(row["test_mse"]), float(row["test_mae"]),
                                     int(row["epochs_completed"]))

    rows = []
    def pct(base, cand):
        return (cand - base) / base * 100.0

    def collect(tag, modes, include_seed=False):
        """Print a table row per (dataset, horizon)."""
        print(f"\n== {tag} ==")
        print(f"{'setting':20s}" + "".join(f"{m:>16s}" for m in modes))
        out_rows = []
        for ds in datasets:
            for h in horizons:
                # pick the seed that has baseline original for this (ds,h)
                seeds = [s for (d, hh, s), v in results.items() if d == ds and hh == h and "original" in v]
                if not seeds:
                    continue
                seed = seeds[0]
                base = results[(ds, h, seed)].get("original")
                line = f"{ds} h{h}:".ljust(20)
                cells = {}
                for m in modes:
                    r = results[(ds, h, seed)].get(m)
                    if r is None:
                        line += f"{'--':>16s}"
                        cells[m] = None
                        continue
                    mse, mae, ep = r
                    d = pct(base[0], mse) if base else 0.0
                    line += f"{mse:8.4f}({d:+5.1f}%)".rjust(16)
                    cells[m] = dict(mode=m, dataset=ds, horizon=h, seed=seed,
                                    mse=mse, mae=mae, delta_mse_pct=d, epochs=ep)
                print(line)
                if base is not None:
                    out_rows.append(cells)
        return out_rows

    # 阶段1: A0/A1/A2
    collect("阶段1 Dynamic Phase Trajectory: A0 original / A1 phase_correction / A2 phase_velocity",
            ["original", "phase_correction", "phase_velocity"])
    # 阶段2: B1 velocity / B2 velocity+geometry
    collect("阶段2 Geometry-aware Interaction: B1 phase_velocity / B2 phase_vel_geo",
            ["phase_velocity", "phase_vel_geo"])
    # 阶段3: R0 no_residual / R1 residual_full / R2 residual_adaptive
    collect("阶段3 Adaptive Residual Fusion: R0 no_residual / R1 residual_full / R2 residual_adaptive",
            ["no_residual", "residual_full", "residual_adaptive"])
    # 模块消融: Baseline/A/B/C
    collect("完整消融 模块: Baseline original / A phase_velocity / B phase_vel_geo / C next_full",
            ["original", "phase_velocity", "phase_vel_geo", "next_full"])
    # phase evolution
    collect("消融 phase evolution: Static original / Offset phase_correction / Velocity phase_velocity",
            ["original", "phase_correction", "phase_velocity"])

    # Also write a flat CSV of all rows.
    flat = []
    for (ds, h, seed), modes in sorted(results.items()):
        if ds not in datasets or h not in horizons:
            continue
        base = modes.get("original")
        for mode, (mse, mae, ep) in sorted(modes.items()):
            if mode not in {"original", "phase_correction", "phase_velocity",
                            "phase_vel_geo", "residual_full", "residual_adaptive",
                            "next_full", "no_residual"}:
                continue
            flat.append(dict(
                dataset=ds, horizon=h, seed=seed, mode=mode,
                mse=f"{mse:.6f}", mae=f"{mae:.6f}",
                delta_mse_pct=f"{pct(base[0], mse):+.2f}" if base else "0.00",
                delta_mae_pct=f"{pct(base[1], mae):+.2f}" if base else "0.00",
                epochs=ep,
            ))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"\nWrote {len(flat)} rows -> {out}")


if __name__ == "__main__":
    main()
