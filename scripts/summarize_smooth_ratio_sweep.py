#!/usr/bin/env python3
"""Compute per-setting smooth_ratio deltas vs s=0 and cross-setting verdicts.

Reads the 7 smooth_sweep_*_results.csv files produced by
run_smooth_ratio_sweep.py and prints a compact text summary (data table,
delta table, verdict table) per the pre-registered rule in
docs/PhaseFormer_residual_smooth_ratio_sweep_experiment.md §5.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = str(ROOT / "research_runs/smooth_ratio_sweep_v1/smooth_sweep_*_results.csv")

SETTINGS_ORDER = [
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
]


def load(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["smooth_ratio"]))
    return rows


def main() -> None:
    files = {}
    for path in glob.glob(PATTERN):
        rows = load(path)
        ds = rows[0]["dataset"]
        h = int(rows[0]["horizon"])
        files[(ds, h)] = rows

    print("=== DATA (MSE/MAE per smooth_ratio) ===")
    for key in SETTINGS_ORDER:
        rows = files.get(key)
        if not rows:
            print(f"{key}: MISSING")
            continue
        ds, h = key
        cells = []
        for r in rows:
            cells.append(f"s={float(r['smooth_ratio']):g}:{float(r['test_mse']):.4f}/{float(r['test_mae']):.4f}")
        print(f"{ds}-{h}: " + " | ".join(cells))

    print()
    print("=== DELTA vs s=0 (positive = better than s=0) ===")
    verdicts = {}
    best_ratio = {}
    for key in SETTINGS_ORDER:
        rows = files.get(key)
        if not rows:
            continue
        ds, h = key
        base = next(r for r in rows if float(r["smooth_ratio"]) == 0.0)
        base_mse = float(base["test_mse"])
        base_mae = float(base["test_mae"])
        deltas = []
        detected = False
        detected_dirs = []
        best = (0.0, base_mse, base_mae)
        for r in rows:
            sr = float(r["smooth_ratio"])
            mse = float(r["test_mse"])
            mae = float(r["test_mae"])
            if mse < best[1] or (mse == best[1] and mae < best[2]):
                best = (sr, mse, mae)
            if sr == 0.0:
                continue
            dmse = (base_mse - mse) / base_mse * 100
            dmae = (base_mae - mae) / base_mae * 100
            deltas.append((sr, dmse, dmae))
            if dmse != 0 and dmae != 0 and (dmse > 0) == (dmae > 0) and abs(dmse) >= 1.0 and abs(dmae) >= 1.0:
                detected = True
                detected_dirs.append((sr, dmse, dmae))
        verdicts[key] = (detected, detected_dirs)
        best_ratio[key] = best[0]
        cells = [f"s={sr:g}:{dmse:+.2f}%/{dmae:+.2f}%" for sr, dmse, dmae in deltas]
        print(f"{ds}-{h}: " + " | ".join(cells))

    print()
    print("=== VERDICT (per setting) ===")
    n_detected = 0
    for key in SETTINGS_ORDER:
        if key not in verdicts:
            continue
        ds, h = key
        detected, dirs = verdicts[key]
        n_detected += int(detected)
        best = best_ratio[key]
        dirstr = "; ".join(f"s={sr:g}({dmse:+.2f}/{dmae:+.2f})" for sr, dmse, dmae in dirs) if dirs else "(none meet threshold)"
        print(f"{ds}-{h}: detected={detected} best_s={best:g} qualifying={dirstr}")

    print()
    print(f"=== CROSS-SETTING COUNT: {n_detected}/7 settings show >=1%/1% same-direction effect ===")
    if n_detected >= 5:
        print("=> consistent effect (>=5/7)")
    elif n_detected >= 3:
        print("=> partial signal (3-4/7)")
    else:
        print("=> no detectable effect (<=2/7)")

    print()
    print("=== OPTIMAL RATIO DISTRIBUTION ===")
    from collections import Counter
    c = Counter(best_ratio.values())
    for sr, cnt in sorted(c.items()):
        print(f"s={sr:g}: {cnt} setting(s)")


if __name__ == "__main__":
    main()
