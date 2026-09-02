#!/usr/bin/env python3
"""Compile the strict-T28 master table vs the Gold standard.

Sources:
  * Stage D runs (3 seeds) for ETTh1/ETTh2/ETTm1/ETTm2 -- the frozen tiers
    from Stage A/B/C in research_runs/pctf_strict_t28_global_golden_v1.
  * Weather final search config (1 seed, seed 2021): mae / ep30 / lr=0.002 /
    tier W / gate 0 / lookback 720 / anchor_scale=1 / composer_scale=1 /
    anchor_loss_weight=1, at H96/H192/H336/H720, from
    research_runs/pctf_weather_search_v1. This is the "current version"
    registered into the table (it supersedes the Stage D Weather S-tier row).
  * Electricity (M) and Traffic (C) were cancelled by the principal
    investigator at 60/84 runs; their rows are marked CANCELLED.

Verification: every run's config.json hash is recomputed and must equal the
recorded config_hash and the run-dir suffix -- run scripts/verify_run_reproducibility.py
for the full per-run audit.

Usage:
  python scripts/report_strict_t28_master_table.py
"""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / ".."  # data lives in the shared checkout
STAGE_D = Path("/home/niuyiming/PhaseFormer/research_runs/"
               "pctf_strict_t28_global_golden_v1")
SEARCH = Path("/home/niuyiming/PhaseFormer/research_runs/pctf_weather_search_v1")
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

CANCELLED = {"Electricity", "Traffic"}

# Final Weather search config identity (from scripts/search_weather_t28.py).
WEATHER_CONFIG = dict(loss="mae", learning_rate=0.002, lookback=720,
                      anchor_lr_scale=1.0, composer_lr_scale=1.0,
                      anchor_loss_weight=1.0, gate=0.0, warmup=0,
                      correction_max=0.60, seed=2021)


def read_row(metrics_path):
    with open(metrics_path) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metrics row: {metrics_path}")
    return rows[0]


def weather_match(cfg, horizon):
    """True if a search run matches the final Weather config at `horizon`."""
    hp = cfg["hyperparams"]
    return (cfg["dataset"] == "Weather" and int(cfg["horizon"]) == horizon
            and hp["loss_func"] == WEATHER_CONFIG["loss"]
            and hp["learning_rate"] == WEATHER_CONFIG["learning_rate"]
            and cfg["lookback"] == WEATHER_CONFIG["lookback"]
            and hp["anchored_pctf_anchor_lr_scale"]
            == WEATHER_CONFIG["anchor_lr_scale"]
            and hp["anchored_pctf_composer_lr_scale"]
            == WEATHER_CONFIG["composer_lr_scale"]
            and hp.get("anchored_pctf_anchor_loss_weight")
            == WEATHER_CONFIG["anchor_loss_weight"]
            and hp["anchored_pctf_gate_aux_weight"] == WEATHER_CONFIG["gate"]
            and hp["anchored_pctf_correction_warmup_epochs"]
            == WEATHER_CONFIG["warmup"]
            and hp["anchored_pctf_correction_max"]
            == WEATHER_CONFIG["correction_max"]
            and cfg["seed"] == WEATHER_CONFIG["seed"])


def load_weather():
    """All search runs matching the final Weather config, per horizon.

    Every match must agree on metrics (a config re-run is expected to be
    bit-identical); if two distinct matches disagree, the identity rule is
    ambiguous and we abort rather than silently pick one."""
    found = {}
    for metrics in sorted((SEARCH / "runs").glob("*/metrics.csv")):
        row = read_row(metrics)
        if not row["test_mse"]:
            continue
        cfg = json.load(open(metrics.parent / "config.json"))
        h = int(row["horizon"])
        if h not in HORIZONS or not weather_match(cfg, h):
            continue
        hit = (float(row["test_mse"]), float(row["test_mae"]))
        if h in found and found[h] != hit:
            raise RuntimeError(
                f"Weather final config ambiguous at H{h}: {found[h]} vs {hit}")
        found[h] = hit
    return found


def load_staged():
    data = {}  # (dataset, horizon) -> [(mse, mae), ...]
    missing = []
    for metrics in sorted((STAGE_D / "runs").glob("*/metrics.csv")):
        row = read_row(metrics)
        if row["stage"] != "confirm" or not row["test_mse"]:
            continue
        if row["dataset"] in CANCELLED:
            continue
        if int(row["seed"]) not in SEEDS:
            continue
        data.setdefault((row["dataset"], int(row["horizon"])), []).append(
            (float(row["test_mse"]), float(row["test_mae"])))
    return data, missing


def fmt(x, width=8):
    return f"{x:>{width}.3f}"


def main():
    staged, _ = load_staged()
    weather = load_weather()

    header = (f"{'Dataset':<11}{'H':>4} {'conf':>5} | "
              f"{'MSE':>18} {'MAE':>18} | {'GoldM':>6} {'GoldA':>6} | "
              f"{'ΔMSE%':>7} {'ΔMAE%':>7}  Beat")
    print(header)
    print("-" * len(header))

    totals = {"settings": 0, "beat_mse": 0, "beat_mae": 0, "beat_both": 0,
              "cancelled": 0}
    for dataset in GOLD:
        for horizon in HORIZONS:
            gm, ga = GOLD[dataset][horizon]
            if dataset in CANCELLED:
                totals["cancelled"] += 1
                print(f"{dataset:<11}{horizon:>4} {'CANCELLED':>5} | "
                      f"{'--':>18} {'--':>18} | {'':>6} {'':>6} |  (not run)")
                continue
            if dataset == "Weather":
                if horizon not in weather:
                    print(f"{dataset:<11}{horizon:>4} {'?':>5} | "
                          f"{'--':>18} {'--':>18} | {'':>6} {'':>6} |  MISSING")
                    continue
                mse, mae = weather[horizon]
                tag_note = "W-srch"
                dmse = (mse - gm) / gm * 100
                dmae = (mae - ga) / ga * 100
                beat_mse, beat_mae = mse < gm, mae < ga
                beat = beat_mse and beat_mae
                tag = "BOTH" if beat else ("MSE" if beat_mse
                                           else ("MAE" if beat_mae else "no"))
                totals["settings"] += 1
                totals["beat_mse"] += beat_mse
                totals["beat_mae"] += beat_mae
                totals["beat_both"] += beat
                print(f"{dataset:<11}{horizon:>4} {tag_note:>5} | "
                      f"{mse:>14.4f} {mae:>14.4f} | {fmt(gm)} {fmt(ga, 6)} | "
                      f"{dmse:>6.2f}% {dmae:>6.2f}%  {tag}")
                continue
            key = (dataset, horizon)
            if key not in staged:
                print(f"{dataset:<11}{horizon:>4} {'?':>5} | "
                      f"{'--':>18} {'--':>18} | {'':>6} {'':>6} |  MISSING")
                continue
            rows = staged[key]
            mse = [r[0] for r in rows]
            mae = [r[1] for r in rows]
            mean_mse, mean_mae = statistics.mean(mse), statistics.mean(mae)
            std_mse, std_mae = statistics.pstdev(mse), statistics.pstdev(mae)
            dmse = (mean_mse - gm) / gm * 100
            dmae = (mean_mae - ga) / ga * 100
            beat_mse = mean_mse + std_mse < gm
            beat_mae = mean_mae + std_mae < ga
            beat = beat_mse and beat_mae
            tag = "BOTH" if beat else ("MSE" if beat_mse
                                       else ("MAE" if beat_mae else "no"))
            totals["settings"] += 1
            totals["beat_mse"] += beat_mse
            totals["beat_mae"] += beat_mae
            totals["beat_both"] += beat
            print(f"{dataset:<11}{horizon:>4} {'3seed':>5} | "
                  f"{fmt(mean_mse)}+-{std_mse:>4.3f} {fmt(mean_mae)}+-{std_mae:>4.3f} "
                  f"| {fmt(gm)} {fmt(ga, 6)} | {dmse:>6.2f}% {dmae:>6.2f}%  {tag}")

    print("-" * len(header))
    print(f"Settings reported: {totals['settings']} | cancelled: "
          f"{totals['cancelled']} | stable-beat both {totals['beat_both']} | "
          f"beat MSE {totals['beat_mse']} | beat MAE {totals['beat_mae']}")
    print()
    print("Note: Weather rows use the final search config (1 seed, seed 2021);")
    print("      other datasets use the 3-seed Stage D mean+-pstdev.")


if __name__ == "__main__":
    main()
