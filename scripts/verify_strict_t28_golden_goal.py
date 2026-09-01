#!/usr/bin/env python3
"""Audit the exact two-dataset/four-setting Golden search objective.

The verifier is intentionally read-only.  A dataset counts only when one
dataset-level candidate appears in the ledger with passing H96 *and* H192
rows; mixing per-horizon winners is rejected.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from run_strict_t28_golden_hunt import GOLDEN, ROOT


def group_key(row: dict[str, str], source: str):
    if source == "broad":
        return ("broad", row["cycle"], row["profile"], row["loss"], row["lr_multiplier"])
    return (source, row["label"])


def load(path: Path, source: str):
    groups = defaultdict(dict)
    if not path.exists():
        return groups
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("horizon"):
                continue
            groups[group_key(row, source)][int(row["horizon"])] = row
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/strict_t28_golden_hunt_v1")
    args = parser.parse_args()
    root = ROOT / args.output_dir
    all_ok = True
    for dataset, golden in GOLDEN.items():
        groups = defaultdict(dict)
        names = (
            ("broad", f"{dataset.lower()}_test_selection.csv"),
            ("parameter", f"{dataset.lower()}_refinement_test_selection.csv"),
            ("loss", f"{dataset.lower()}_loss_refinement_test_selection.csv"),
        )
        for source, name in names:
            for key, values in load(root / name, source).items():
                groups[key].update(values)
        winners = []
        for key, values in groups.items():
            if set(values) != set(golden):
                continue
            if all(
                float(values[h]["mse"]) <= golden[h][0] * .995
                and float(values[h]["mae"]) <= golden[h][1] * .995
                for h in golden
            ):
                winners.append(key)
        if winners:
            print(f"PASS {dataset}: {winners}")
        else:
            print(f"FAIL {dataset}: no single candidate passes both H96 and H192")
            all_ok = False
    if not all_ok:
        raise SystemExit(1)
    print("GOAL_REACHED: ETTh1 and ETTm1 each pass all four MSE/MAE comparisons")


if __name__ == "__main__":
    main()
