#!/usr/bin/env python3
"""Aggregate pure-phase full-budget results into the plan's comparison tables.

Reads every metrics.csv under the full-budget run dir, keys by
(dataset, horizon, seed, mode), and prints/CSV-exports the plan's tables:

  阶段1 Multi-scale Phase Representation: original vs multiscale_phase
  阶段2 Dynamic Phase Deformation: original / phase_velocity(ref) / phase_deformation
  阶段3 Geometry-aware Interaction: original vs phase_geo / phase_graph
  阶段4 Trajectory Decoder: original / predictor_mlp / trajectory_decoder
  完整消融 (Table): original + 4 modules + final model pure_full

delta% convention: (candidate - baseline) / baseline * 100, negative = better.
"""

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="research_runs/dyn_phase_full")
    p.add_argument("--output", default="research_runs/pure_phase_summary.csv")
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

    def pct(base, cand):
        return (cand - base) / base * 100.0

    table_rows = []

    def collect(tag, modes):
        """Print a table row per (dataset, horizon); return the cells."""
        print(f"\n== {tag} ==")
        print(f"{'setting':20s}" + "".join(f"{m:>18s}" for m in modes))
        out_rows = []
        for ds in datasets:
            for h in horizons:
                seeds = [s for (d, hh, s), v in results.items()
                         if d == ds and hh == h and "original" in v]
                if not seeds:
                    continue
                seed = seeds[0]
                base = results[(ds, h, seed)].get("original")
                line = f"{ds} h{h}:".ljust(20)
                cells = {}
                for m in modes:
                    r = results[(ds, h, seed)].get(m)
                    if r is None:
                        line += f"{'--':>18s}"
                        cells[m] = None
                        continue
                    mse, mae, ep = r
                    d = pct(base[0], mse) if base else 0.0
                    line += f"{mse:8.4f}({d:+5.1f}%)".rjust(18)
                    cells[m] = dict(mode=m, dataset=ds, horizon=h, seed=seed,
                                    mse=mse, mae=mae, delta_mse_pct=d, epochs=ep)
                print(line)
                if base is not None:
                    out_rows.append(cells)
        table_rows.append((tag, modes, out_rows))
        return out_rows

    # 阶段1: Multi-scale phase representation.
    collect("阶段1 Multi-scale: original vs multiscale_phase",
            ["original", "multiscale_phase"])
    # 阶段2: Dynamic phase deformation (velocity as prior-stage reference).
    collect("阶段2 Deformation: original / phase_velocity(ref) / phase_deformation",
            ["original", "phase_velocity", "phase_deformation"])
    # 阶段3: Geometry-aware interaction: circular bias vs explicit graph.
    collect("阶段3 Geometry: original / phase_geo / phase_graph",
            ["original", "phase_geo", "phase_graph"])
    # 阶段4: Trajectory decoder (MLP predictor as capacity-matched reference).
    collect("阶段4 Decoder: original / predictor_mlp / trajectory_decoder",
            ["original", "predictor_mlp", "trajectory_decoder"])
    # 完整消融: original + all 4 modules + final model.
    collect("完整消融: original / 4 modules / pure_full",
            ["original", "multiscale_phase", "phase_deformation", "phase_geo",
             "phase_graph", "predictor_mlp", "trajectory_decoder", "pure_full"])

    # Also write a flat CSV of all pure-phase rows (vs original base).
    flat = []
    for (ds, h, seed), modes in sorted(results.items()):
        if ds not in datasets or h not in horizons:
            continue
        base = modes.get("original")
        for mode, (mse, mae, ep) in sorted(modes.items()):
            if mode not in {"original", "multiscale_phase", "phase_deformation",
                            "phase_geo", "phase_graph", "predictor_mlp",
                            "trajectory_decoder", "pure_full"}:
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
