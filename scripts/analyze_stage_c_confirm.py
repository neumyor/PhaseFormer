#!/usr/bin/env python3
"""Analyze Stage C confirmatory validation.

For each dataset the frozen (cycle, tier) is run at 100% train on 4 horizons x
seeds 2021/2022 together with a C-tier reference at the same protocol.  The
16 ratios (4H x 2 seeds x MSE/MAE) of frozen/C are computed; if any single
ratio > 1.005 (regresses more than 0.5% vs C), the dataset falls back to C.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs/pctf_strict_t28_global_golden_v1"
TIER_KEYS = ("anchored_pctf_correction_max", "anchored_pctf_deformation_max",
             "anchored_pctf_global_level_max")
TIERS = {
    "C": (0.25, 0.10, 0.05),
    "M": (0.40, 0.16, 0.08),
    "S": (0.50, 0.20, 0.10),
    "W": (0.60, 0.24, 0.12),
}
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022)


def read_row(metrics_path):
    with open(metrics_path) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metrics row: {metrics_path}")
    return rows[0]


def tier_from_config(config_path):
    cfg = json.loads(Path(config_path).read_text())
    hp = cfg["hyperparams"]
    signature = tuple(float(hp[k]) for k in TIER_KEYS)
    for tier, values in TIERS.items():
        if signature == values:
            return tier
    return None


def main():
    frozen = json.loads((RUNS / "frozen_decisions.json").read_text())
    data = {}  # (dataset, horizon, seed, tier) -> row
    for metrics in sorted((RUNS / "runs").glob("*/metrics.csv")):
        row = read_row(metrics)
        if row["stage"] != "finalist":
            continue
        cfg = metrics.parent / "config.json"
        tier = tier_from_config(cfg)
        if tier is None:
            print(f"unknown tier: {metrics.parent}", file=sys.stderr)
            continue
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]), tier)
        data[key] = row

    print("=== Stage C confirmatory validation ===")
    final_tiers = {}
    any_missing = False
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather",
                    "Electricity", "Traffic"):
        frozen_tier = frozen["tier"][dataset]
        if frozen_tier not in ("C", "M", "S", "W"):
            print(f"{dataset}: missing frozen tier")
            continue
        need = [(h, s, frozen_tier) for h in HORIZONS for s in SEEDS]
        if frozen_tier != "C":
            need += [(h, s, "C") for h in HORIZONS for s in SEEDS]
        missing = [k for k in need if (dataset, k[0], k[1], k[2]) not in data]
        if missing:
            print(f"{dataset}: MISSING {len(missing)} runs e.g. {missing[:4]}")
            any_missing = True
            continue
        ratios = []
        for horizon in HORIZONS:
            for seed in SEEDS:
                fz = data[(dataset, horizon, seed, frozen_tier)]
                if frozen_tier == "C":
                    # Frozen is C itself: reference is the frozen run.
                    mse_ref = float(fz["val_mse"])
                    mae_ref = float(fz["val_mae"])
                else:
                    c = data[(dataset, horizon, seed, "C")]
                    mse_ref = float(c["val_mse"])
                    mae_ref = float(c["val_mae"])
                mse_fz = float(fz["val_mse"])
                mae_fz = float(fz["val_mae"])
                ratios.append(mse_fz / mse_ref)
                ratios.append(mae_fz / mae_ref)
        worst = max(ratios)
        regress = worst > 1.005
        final = "C" if regress else frozen_tier
        final_tiers[dataset] = final
        print(f"{dataset}: frozen={frozen_tier} -> final={final} "
              f"worst_ratio={worst:.5f} ({'FALLBACK' if regress else 'OK'})")
        print(f"    mean_ratio={statistics.mean(ratios):.5f} "
              f"n_ratios={len(ratios)}")

    if any_missing:
        print("\n** Some datasets are missing runs; not writing decision. **")
        return 1

    frozen["tier"] = final_tiers
    (RUNS / "frozen_decisions.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    print("\nWrote frozen_decisions.json with Stage-C-final tiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
