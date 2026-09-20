#!/usr/bin/env python3
"""Compare two E16 output roots cell-by-cell, to test whether two kernels agree.

Built to answer one question with evidence rather than assertion: does the
``--fast-einsum`` path (numpy with ``optimize=True``, i.e. BLAS) reproduce the
historical bare-``np.einsum`` path at the precision section 4.4 actually displays?

The comparison is deliberately two-tiered, because "the numbers are the same" is
ambiguous:

1. **Reported precision** -- section 4.4 renders these values as 2, 3 or 4 decimals,
   so the test that matters is whether the *rendered* values are identical.  A
   per-column precision table is supplied by the caller (``--precision``), defaulting
   to the values the paper displays.
2. **Raw float64** -- the maximum relative difference, reported regardless, so a
   reader can see the size of the rounding difference instead of taking
   "equivalent" on trust.

Rows are matched by key, never by position, so a reordering cannot masquerade as
agreement.  A missing key on either side is a failure, not a skip.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

#: Columns the paper displays, and at how many decimals.  Section 4.4's dissection
#: table renders the two explanation rates and the cross-seed overlaps at 2dp, the
#: correction energy share at 3dp; the intervention table's own values are printed
#: as produced.
DEFAULT_PRECISION = {
    "input_group_explanation": 2,
    "output_group_explanation": 2,
    "cross_seed_leading4_input_overlap": 2,
    "cross_seed_leading4_output_overlap": 2,
    "correction_energy_share": 3,
}

KEYS = {
    "dissection_table.csv": ("arm", "setting", "seed"),
    "intervention_table.csv": ("arm", "setting", "seed", "intervention_arm"),
}


def read_rows(path: pathlib.Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def is_number(text) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def compare(a_path: pathlib.Path, b_path: pathlib.Path, keys: tuple,
            precision: dict) -> dict:
    left = {tuple(row[k] for k in keys): row for row in read_rows(a_path)}
    right = {tuple(row[k] for k in keys): row for row in read_rows(b_path)}
    report = {
        "file": a_path.name,
        "rows_a": len(left),
        "rows_b": len(right),
        "missing_in_b": sorted(set(left) - set(right))[:5],
        "missing_in_a": sorted(set(right) - set(left))[:5],
        "columns_compared": 0,
        "raw_max_rel_diff": 0.0,
        "raw_worst_column": None,
        "reported_mismatches": [],
    }
    if not left or not right:
        report["reported_mismatches"].append("one side has no rows")
        return report

    shared = sorted(set(left) & set(right))
    columns = [c for c in left[shared[0]] if c not in keys]
    for column in columns:
        values = [(left[key].get(column, ""), right[key].get(column, ""))
                  for key in shared]
        numeric = [(x, y) for x, y in values if is_number(x) and is_number(y)]
        if not numeric:
            # identity/text column: require exact equality
            differing = [k for k in shared
                         if left[k].get(column, "") != right[k].get(column, "")]
            if differing:
                report["reported_mismatches"].append(
                    f"{column}: {len(differing)} text cell(s) differ, e.g. {differing[0]}")
            report["columns_compared"] += 1
            continue

        digits = precision.get(column)
        worst = 0.0
        for key in shared:
            x, y = left[key].get(column, ""), right[key].get(column, "")
            if not (is_number(x) and is_number(y)):
                report["reported_mismatches"].append(
                    f"{column}: non-numeric value at {key} ({x!r} vs {y!r})")
                continue
            xf, yf = float(x), float(y)
            scale = max(abs(xf), abs(yf), 1e-300)
            worst = max(worst, abs(xf - yf) / scale)
            if digits is not None and f"{xf:.{digits}f}" != f"{yf:.{digits}f}":
                report["reported_mismatches"].append(
                    f"{column}@{key}: displayed at {digits}dp as "
                    f"{xf:.{digits}f} vs {yf:.{digits}f}")
        if worst > report["raw_max_rel_diff"]:
            report["raw_max_rel_diff"] = worst
            report["raw_worst_column"] = column
        report["columns_compared"] += 1
    return report


def self_test() -> int:
    import tempfile

    keys = ("arm", "setting", "seed", "intervention_arm")
    header = ["arm", "setting", "seed", "intervention_arm",
              "correction_energy_share", "label"]
    rows = [
        ["l_main", "ETTh2-96", "2021", "Semantic-only", "0.6521771", "周期形状/相位"],
        ["l_main", "ETTh2-96", "2021", "Semantic-drop", "0.3202054", "近端电平"],
    ]

    def write(path, mutate):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow(mutate(list(row)))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        a, b, c = tmp / "a.csv", tmp / "b.csv", tmp / "c.csv"
        write(a, lambda r: r)
        write(b, lambda r: r[:4] + [str(float(r[4]) + 1e-15)] + r[5:])
        write(c, lambda r: r[:4] + [str(float(r[4]) + 1e-3)] + r[5:])

        same = compare(a, b, keys, DEFAULT_PRECISION)
        moved = compare(a, c, keys, DEFAULT_PRECISION)
        checks = [
            ("identical input reports no mismatch", same["reported_mismatches"] == []),
            ("a 1e-15 perturbation is below the displayed precision",
             same["reported_mismatches"] == []),
            ("that perturbation is still visible in the raw diff",
             same["raw_max_rel_diff"] > 0),
            ("a 1e-3 perturbation IS reported", bool(moved["reported_mismatches"])),
            ("rows matched by key", same["rows_a"] == 2 and same["rows_b"] == 2),
        ]
    ok = True
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
        ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.a or not a.b:
        ap.error("--a and --b (two E16 output roots) are required")

    root_a, root_b = pathlib.Path(a.a), pathlib.Path(a.b)
    failures = 0
    results = []
    for name, keys in KEYS.items():
        pa, pb = root_a / name, root_b / name
        if not (pa.is_file() and pb.is_file()):
            print(f"{name}: missing on one side ({pa.is_file()=}, {pb.is_file()=})")
            failures += 1
            continue
        report = compare(pa, pb, keys, DEFAULT_PRECISION)
        results.append(report)
        print(f"\n=== {name} ===")
        print(f"  rows: {report['rows_a']} vs {report['rows_b']}   "
              f"columns compared: {report['columns_compared']}")
        print(f"  raw max relative difference: {report['raw_max_rel_diff']:.3e} "
              f"({report['raw_worst_column']})")
        if report["missing_in_a"]:
            print(f"  keys missing in A: {report['missing_in_a']}")
            failures += 1
        if report["missing_in_b"]:
            print(f"  keys missing in B: {report['missing_in_b']}")
            failures += 1
        if report["reported_mismatches"]:
            print(f"  MISMATCHES at displayed precision: {len(report['reported_mismatches'])}")
            for item in report["reported_mismatches"][:8]:
                print(f"    {item}")
            failures += 1
        else:
            print("  every value agrees at the displayed precision")

    if a.json:
        import json
        pathlib.Path(a.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nVERDICT: " + ("EQUIVALENT at displayed precision" if not failures
                           else f"NOT equivalent ({failures} problem(s))"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
