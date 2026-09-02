#!/usr/bin/env python3
"""Collect config.json + commands.sh for every cell of the strict-T28 master
table into a docs subfolder with layout:

    docs/strict_t28_master_table_configs/<Dataset>/<h<horizon>>/
        config.json
        commands.sh

Only cells whose runs physically exist on this machine are collected:
  * ETTh2 / ETTm2 : Stage D confirm runs (3 seeds). The cell's files are the
                    seed-2021 representative; the other two seeds' runs remain
                    in research_runs/pctf_strict_t28_global_golden_v1 and the
                    manifest (commands differ only in --seed / --config-hash).
  * Weather       : final search config (mae/ep30/lr0.002/tier W/gate0/
                    lookback720, seed 2021) at H96/H192/H336/H720.

Cells without on-disk runs are listed in the README but have no files:
  * ETTh1 / ETTm1 : external search results (u_lr020 / w_aux01), no run data
                    on this machine.
  * Electricity / Traffic : cancelled.

Run from the worktree:  python3 scripts/collect_strict_t28_configs.py
"""

import csv
import json
import shutil
from pathlib import Path

STAGE_D = Path("/home/niuyiming/PhaseFormer/research_runs/"
               "pctf_strict_t28_global_golden_v1")
SEARCH = Path("/home/niuyiming/PhaseFormer/research_runs/pctf_weather_search_v1")
DEST = Path(__file__).resolve().parents[1] / "docs/strict_t28_master_table_configs"
HORIZONS = (96, 192, 336, 720)
SEED_2021 = 2021

STAGE_D_DATASETS = ("ETTh2", "ETTm2")  # 3-seed Stage D rows


def read_metrics(metrics_path):
    return list(csv.DictReader(open(metrics_path)))[0]


def weather_match(cfg):
    """True if a search run is the final Weather config (any horizon)."""
    hp = cfg["hyperparams"]
    return (cfg["dataset"] == "Weather" and int(cfg["horizon"]) in HORIZONS
            and hp["loss_func"] == "mae" and hp["learning_rate"] == 0.002
            and cfg["lookback"] == 720
            and hp["anchored_pctf_anchor_lr_scale"] == 1.0
            and hp["anchored_pctf_composer_lr_scale"] == 1.0
            and hp.get("anchored_pctf_anchor_loss_weight") == 1.0
            and hp["anchored_pctf_gate_aux_weight"] == 0.0
            and hp["anchored_pctf_correction_warmup_epochs"] == 0
            and hp["anchored_pctf_correction_max"] == 0.60
            and cfg["seed"] == SEED_2021)


def collect(dataset, horizon, run_dir):
    dest = DEST / dataset / f"h{horizon}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "commands.sh"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
        else:
            print(f"  WARN {dataset} H{horizon}: missing {fname}")
    return dest, run_dir.name


def main():
    collected = []   # (dataset, horizon, run_id)
    missing = []

    # Stage D ETTh2/ETTm2 seed-2021 confirm runs.
    for metrics in sorted((STAGE_D / "runs").glob("*/metrics.csv")):
        row = read_metrics(metrics)
        if (row["stage"] != "confirm" or not row["test_mse"]
                or row["dataset"] not in STAGE_D_DATASETS
                or int(row["seed"]) != SEED_2021):
            continue
        h = int(row["horizon"])
        collected.append((row["dataset"], h, collect(row["dataset"], h,
                                                     metrics.parent)[1]))

    # Weather final-config runs (one per horizon).
    for metrics in sorted((SEARCH / "runs").glob("*/metrics.csv")):
        row = read_metrics(metrics)
        if not row["test_mse"]:
            continue
        cfg = json.load(open(metrics.parent / "config.json"))
        if weather_match(cfg):
            h = int(row["horizon"])
            collected.append(("Weather", h, collect("Weather", h,
                                                    metrics.parent)[1]))

    # README: layout + which cells have no on-disk run.
    external = {
        "ETTh1": "u_lr020: cycle=24, caps 1.40/0.80/0.40, MAE, "
                 "lr multiplier 0.20, ep50 (no run data on this machine)",
        "ETTm1": "w_aux01: cycle=24, caps 0.60/0.24/0.12, MAE, "
                 "lr multiplier 0.20, shape/level/gate aux=0.01, ep50 "
                 "(no run data on this machine)",
    }
    lines = [
        "# strict-T28 master table run configs\n",
        "",
        "Layout: `<Dataset>/<h<horizon>>/config.json` + `commands.sh` for every "
        "cell of the strict-T28 master table that has an on-disk run.",
        "",
        "Collected:",
    ]
    for dataset, h, rid in sorted(collected):
        lines.append(f"- `{dataset}/h{h}` <- `{rid}`")
    lines += [
        "",
        "Not collected (no run data on this machine / not run):",
        "",
    ]
    for ds, note in external.items():
        lines.append(f"- `{ds}` (registered from external search results): {note}")
    lines.append("- `Electricity` / `Traffic`: CANCELLED (not run).")
    lines += [
        "",
        "Note: ETTh2/ETTm2 cells hold the seed-2021 representative of the "
        "3-seed Stage D runs; seeds 2022/2023 runs remain in "
        "`research_runs/pctf_strict_t28_global_golden_v1/` and the manifest. "
        "The exact per-run invocation is preserved in each `commands.sh`.",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines) + "\n")

    print(f"Collected {len(collected)} cell files into {DEST}")
    for dataset, h, rid in sorted(collected):
        print(f"  {dataset}/h{h}  <-  {rid}")
    print(f"README -> {DEST / 'README.md'}")


if __name__ == "__main__":
    main()
