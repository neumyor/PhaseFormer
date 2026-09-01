#!/usr/bin/env python3
"""Compile the strict-T28 Stage D master table vs the Gold standard.

Reads stage=confirm runs that used --evaluate-test (test_mse/test_mae set),
groups by (dataset, horizon) over the 3 seeds, and reports MSE/MAE
mean +/- sample std vs the Gold standard (docs/PhaseFormer_gold_standard.md),
with absolute and percentage change and a per-setting beat flag.

Usage:
  python scripts/report_strict_t28_master_table.py
"""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs/pctf_strict_t28_global_golden_v1"
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)

# Gold standard MSE/MAE (3 decimals), from docs/PhaseFormer_gold_standard.md.
GOLD = {
    "ETTh1":       {96: (0.359, 0.382), 192: (0.397, 0.404), 336: (0.425, 0.424), 720: (0.431, 0.450)},
    "ETTh2":       {96: (0.275, 0.338), 192: (0.341, 0.376), 336: (0.369, 0.405), 720: (0.402, 0.436)},
    "ETTm1":       {96: (0.293, 0.344), 192: (0.323, 0.361), 336: (0.358, 0.381), 720: (0.412, 0.410)},
    "ETTm2":       {96: (0.163, 0.256), 192: (0.219, 0.293), 336: (0.269, 0.326), 720: (0.351, 0.379)},
    "Weather":     {96: (0.148, 0.195), 192: (0.193, 0.237), 336: (0.242, 0.278), 720: (0.309, 0.332)},
    "Electricity": {96: (0.129, 0.221), 192: (0.148, 0.238), 336: (0.165, 0.257), 720: (0.201, 0.285)},
    "Traffic":     {96: (0.361, 0.238), 192: (0.373, 0.243), 336: (0.385, 0.248), 720: (0.428, 0.270)},
}


def read_row(metrics_path):
    with open(metrics_path) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metrics row: {metrics_path}")
    return rows[0]


def fmt(x, width=8):
    return f"{x:>{width}.3f}"


def main():
    data = {}  # (dataset, horizon) -> list of (mse, mae)
    missing_test = []
    for metrics in sorted((RUNS / "runs").glob("*/metrics.csv")):
        row = read_row(metrics)
        if row["stage"] != "confirm":
            continue
        if not row["test_mse"]:
            missing_test.append(metrics.parent.name)
            continue
        dataset, horizon, seed = row["dataset"], int(row["horizon"]), int(row["seed"])
        if seed not in SEEDS:
            continue
        data.setdefault((dataset, horizon), []).append(
            (float(row["test_mse"]), float(row["test_mae"]))
        )

    if missing_test:
        print(f"WARNING: {len(missing_test)} confirm runs lack test metrics, e.g. "
              f"{missing_test[:5]}", file=sys.stderr)

    datasets = tuple(GOLD)
    header = (f"{'Dataset':<11}{'H':>4} | {'MSE mean±std':>20} {'MAE mean±std':>20} "
              f"| {'Gold MSE':>9} {'Gold MAE':>9} | {'ΔMSE%':>8} {'ΔMAE%':>8}  {'Beat'}")
    print(header)
    print("-" * len(header))

    totals = {"settings": 0, "beat_mse": 0, "beat_mae": 0, "beat_both": 0}
    for dataset in datasets:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            if key not in data:
                print(f"{dataset:<11}{horizon:>4} | MISSING all seeds")
                continue
            rows = data[key]
            if len(rows) != 3:
                print(f"{dataset:<11}{horizon:>4} | {len(rows)}/3 seeds present "
                      f"(runs: {len(rows)})")
                continue
            mse = [r[0] for r in rows]
            mae = [r[1] for r in rows]
            gm, ga = GOLD[dataset][horizon]
            mean_mse, mean_mae = statistics.mean(mse), statistics.mean(mae)
            std_mse = statistics.pstdev(mse)
            std_mae = statistics.pstdev(mae)
            dmse = (mean_mse - gm) / gm * 100
            dmae = (mean_mae - ga) / ga * 100
            beat_mse = mean_mse + std_mse < gm
            beat_mae = mean_mae + std_mae < ga
            beat = beat_mse and beat_mae
            tag = "BOTH" if beat else ("MSE" if beat_mse else ("MAE" if beat_mae else "no"))
            totals["settings"] += 1
            totals["beat_mse"] += beat_mse
            totals["beat_mae"] += beat_mae
            totals["beat_both"] += beat
            print(f"{dataset:<11}{horizon:>4} | {fmt(mean_mse)}+-{std_mse:>6.3f} "
                  f"{fmt(mean_mae)}+-{std_mae:>6.3f} | {fmt(gm)} {fmt(ga):>9} | "
                  f"{dmse:>7.2f}% {dmae:>7.2f}%  {tag}")

    print("-" * len(header))
    print(f"Settings: {totals['settings']} | stable-beat both "
          f"{totals['beat_both']} | beat MSE {totals['beat_mse']} | "
          f"beat MAE {totals['beat_mae']}")


if __name__ == "__main__":
    main()
