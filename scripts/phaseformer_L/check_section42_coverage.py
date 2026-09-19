#!/usr/bin/env python3
"""Verify the E14 manifest covers exactly what minipaper section 4.2 requires.

The question this answers is the one the whole task started from: does the plan
actually produce every cell the paper's table demands -- no gap, and no invented
extra?  Read from the REAL manifest, not from my memory of it.

Expected shape (from the code, not from the paper):
  * five main arms (phase_only, l_main, l_q1_4, l_q1_8, l_rcrf) on all 28
    settings = 24 main + the 4-setting Traffic appendix;
  * the optional a1 arm on the 24 main settings only (ARM_DATASET_EXCLUSIONS
    excludes Traffic);
  * three seeds each -> 5*84 + 72 = 492 cells.

Usage: python scripts/phaseformer_L/check_section42_coverage.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

REPO = pathlib.Path.cwd()
DEFAULT_MANIFEST = ("research_runs/phaseformer_L_e14_main_v1/"
                    "stage_a_manifest.json")
MANIFEST = REPO / DEFAULT_MANIFEST
MAIN_ARMS = ("phase_only", "l_main", "l_q1_4", "l_q1_8", "l_rcrf")
OPTIONAL = ("a1",)
SEEDS = (2021, 2022, 2023)
TRAFFIC = "Traffic"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    cells = manifest["cells"]
    per = collections.defaultdict(lambda: {"settings": set(), "seeds": set(),
                                           "reused": 0, "new": 0})
    for cell in cells:
        arm = cell["arm"]
        entry = per[arm]
        entry["settings"].add((cell["dataset"], int(cell["horizon"])))
        entry["seeds"].add(int(cell["seed"]))
        status = cell.get("status", "new")
        entry[status] = entry.get(status, 0) + 1

    all_settings = set()
    for arm in per:
        all_settings |= per[arm]["settings"]
    traffic = {s for s in all_settings if s[0] == TRAFFIC}
    main_settings = all_settings - traffic
    print(f"all settings: {len(all_settings)} = {len(main_settings)} main + {len(traffic)} Traffic")

    failures = 0
    if len(main_settings) != 24 or len(traffic) != 4:
        print("FAIL: expected 24 main + 4 Traffic settings")
        failures += 1

    print(f"\n{'arm':11s} {'settings':>8s} {'seeds':>5s} {'reused':>6s} {'new':>4s} {'cells':>6s}  expected")
    for arm in sorted(per):
        entry = per[arm]
        expected_settings = 24 if arm in OPTIONAL else 28
        got = len(entry["settings"])
        cells_n = got * len(entry["seeds"])
        note = ""
        if got != expected_settings:
            note = f"  <-- FAIL: expected {expected_settings} settings"
            failures += 1
        if set(entry["seeds"]) != set(SEEDS):
            note += f"  <-- FAIL: seeds {sorted(entry['seeds'])}"
            failures += 1
        if arm in OPTIONAL and any(s[0] == TRAFFIC for s in entry["settings"]):
            note += "  <-- FAIL: a1 must not cover Traffic"
            failures += 1
        print(f"{arm:11s} {got:8d} {len(entry['seeds']):5d} {entry['reused']:6d} "
              f"{entry['new']:4d} {cells_n:6d}  {expected_settings} settings{note}")

    expected_total = len(MAIN_ARMS) * 28 * 3 + len(OPTIONAL) * 24 * 3
    print(f"\ntotal cells: {len(cells)} (expected {expected_total})")
    if len(cells) != expected_total:
        print("FAIL: total cell count differs")
        failures += 1
    missing_arms = [a for a in MAIN_ARMS + OPTIONAL if a not in per]
    if missing_arms:
        print(f"FAIL: arms absent from the manifest: {missing_arms}")
        failures += 1

    print()
    print("PASS: the manifest covers exactly section 4.2's schedule"
          if not failures else f"FAIL: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
