#!/usr/bin/env python3
"""Analyze Stage B trust-region screen results and freeze per-dataset tier.

Decision rule (from plan): for each dataset, over H96/H336 and MSE/MAE (4
cells), score each tier by the mean ratio relative to tier C.  A tier is
admissible if no single cell regresses more than 0.5% vs C.  The frozen tier is
the admissible tier with the best (lowest) score; if only C is admissible, the
dataset freezes C ("no credible correction-expansion gain").
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
    data = {}  # dataset -> {(horizon, tier): row}
    for metrics in sorted((RUNS / "runs").glob("*/metrics.csv")):
        row = read_row(metrics)
        if row["stage"] != "mechanism_screen_1":
            continue
        cfg = metrics.parent / "config.json"
        tier = tier_from_config(cfg)
        if tier is None:
            print(f"unknown tier: {metrics.parent}", file=sys.stderr)
            continue
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        data.setdefault(dataset, {})[(horizon, tier)] = row

    datasets = sorted(data)
    print("=== Stage B per-dataset trust-region decision ===")
    decisions = {}
    for dataset in datasets:
        rows = data[dataset]
        missing = [(h, t) for h in (96, 336) for t in TIERS if (h, t) not in rows]
        if missing:
            print(f"{dataset}: MISSING {missing}")
            continue
        cells = {}
        for horizon in (96, 336):
            c = rows[(horizon, "C")]
            cells[horizon] = {
                "mse_C": float(c["val_mse"]), "mae_C": float(c["val_mae"]),
            }
            for tier in ("M", "S", "W"):
                r = rows[(horizon, tier)]
                cells[horizon][f"mse_{tier}"] = float(r["val_mse"])
                cells[horizon][f"mae_{tier}"] = float(r["val_mae"])
        # ratios relative to C, per tier
        scores = {}
        for tier in ("M", "S", "W"):
            ratios = []
            for horizon in (96, 336):
                ratios.append(cells[horizon][f"mse_{tier}"] / cells[horizon]["mse_C"])
                ratios.append(cells[horizon][f"mae_{tier}"] / cells[horizon]["mae_C"])
            scores[tier] = {
                "ratios": ratios,
                "mean": statistics.mean(ratios),
                "max": max(ratios),
                # Admissible: no single endpoint metric regresses >0.5% vs C
                # AND the tier is a credible improvement over C on average.
                # Otherwise the dataset is marked "no credible correction-
                # expansion gain" and conservatively freezes C.
                "admissible": max(ratios) <= 1.005 and statistics.mean(ratios) < 1.0,
            }
        admissible = [t for t in ("M", "S", "W") if scores[t]["admissible"]]
        if admissible:
            best = min(admissible, key=lambda t: scores[t]["mean"])
        else:
            best = "C"
        decisions[dataset] = best
        print(f"{dataset}: frozen tier = {best}")
        for tier in ("M", "S", "W"):
            s = scores[tier]
            tag = "OK" if s["admissible"] else "REG>0.5%"
            print(f"    {tier}: mean={s['mean']:.5f} max={s['max']:.5f} [{tag}]")

    frozen["tier"] = decisions
    (RUNS / "frozen_decisions.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    print(f"\nWrote frozen_decisions.json (tier for {len(decisions)} datasets)")

    # Print a compact raw table for the plan doc
    print("\n=== raw val table ===")
    print("dataset,h96_C_mse,h96_C_mae,h336_C_mse,h336_C_mae,h96_M,h96_S,h96_W,h336_M,h336_S,h336_W")
    for dataset in datasets:
        rows = data[dataset]
        c96 = rows[(96, "C")]; c336 = rows[(336, "C")]
        def fmt(tier):
            r96 = rows[(96, tier)]; r336 = rows[(336, tier)]
            return (f"{float(r96['val_mse']):.5f}/{float(r96['val_mae']):.5f}"
                    f"/{float(r336['val_mse']):.5f}/{float(r336['val_mae']):.5f}")
        print(f"{dataset},{float(c96['val_mse']):.5f},{float(c96['val_mae']):.5f},"
              f"{float(c336['val_mse']):.5f},{float(c336['val_mae']):.5f},"
              + ",".join(fmt(t) for t in ("M", "S", "W")))


if __name__ == "__main__":
    main()
