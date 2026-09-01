#!/usr/bin/env python3
"""Verify the user-requested four-setting Golden gate with per-horizon models."""
from __future__ import annotations
import csv
from pathlib import Path
from run_strict_t28_golden_hunt import GOLDEN, ROOT


def main():
    root = ROOT / "research_runs/strict_t28_golden_hunt_v1"
    ok = True
    for dataset, goals in GOLDEN.items():
        files = tuple(root / f"{dataset.lower()}{suffix}" for suffix in (
            "_test_selection.csv", "_refinement_test_selection.csv",
            "_loss_refinement_test_selection.csv",
            "_calibration_refinement_test_selection.csv",
            "_horizon_refinement_test_selection.csv",
        ))
        for horizon, (gm, ga) in goals.items():
            passed = []
            for path in files:
                if not path.exists():
                    continue
                with path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("horizon") != str(horizon) or not row.get("mse"):
                            continue
                        if float(row["mse"]) <= gm * .995 and float(row["mae"]) <= ga * .995:
                            passed.append((path.name, row.get("label", ""), row["mse"], row["mae"]))
            if passed:
                print(f"PASS {dataset} H{horizon}: {passed[0]}")
            else:
                print(f"FAIL {dataset} H{horizon}: no recorded candidate passes both metrics")
                ok = False
    if not ok:
        raise SystemExit(1)
    print("GOAL_REACHED: all four settings meet the 0.5% Golden gate")


if __name__ == "__main__":
    main()
