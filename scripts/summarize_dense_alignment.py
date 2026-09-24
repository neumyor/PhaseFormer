#!/usr/bin/env python3
"""Summarize the dense-vs-low-rank alignment table."""

from __future__ import annotations

import collections
import csv
import statistics as st
import sys


def main() -> None:
    path = sys.argv[1]
    rows = list(csv.DictReader(open(path)))
    angle_rows = [r for r in rows if r["dense_reference"] != "input_subspace_restriction"]
    retain_rows = [r for r in rows if r["dense_reference"] == "input_subspace_restriction"]

    print("=== subspace alignment: low-rank top-k vs dense reference ===")
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in angle_rows:
        group[(row["dense_reference"], int(row["subspace_dim"]), row["cell"])].append(row)
    print("{:>18s} {:>4s} {:>7s} {:>4s} {:>14s} {:>14s}".format(
        "dense_ref", "dim", "cell", "n", "input_overlap", "output_overlap"))
    for key in sorted(group):
        values = group[key]
        print("{:>18s} {:>4d} {:>7s} {:>4d} {:>14s} {:>14s}".format(
            key[0], key[1], key[2], len(values),
            "{:.3f}".format(st.mean([float(v["input_overlap"]) for v in values])),
            "{:.3f}".format(st.mean([float(v["output_overlap"]) for v in values])),
        ))

    print()
    print("=== dense improvement retained by restricting to the low-rank input subspace ===")
    print("{:>7s} {:>7s} {:>6s} {:>4s} {:>12s}".format("dim", "cell", "setting", "n", "retained"))
    group2: dict[tuple, list] = collections.defaultdict(list)
    for row in retain_rows:
        group2[(int(row["subspace_dim"]), row["cell"], row["setting"])].append(row)
    for key in sorted(group2):
        values = group2[key]
        print("{:>7d} {:>7s} {:>6s} {:>4d} {:>11.1%}".format(
            key[0], key[1], key[2], len(values),
            st.mean([float(v["dense_improvement_retained"]) for v in values]),
        ))

    print()
    print("=== per-setting retained (dim=8), by dense reference ordering ===")
    for cell in ("q=1/4", "q=1/8"):
        for setting in sorted({r["setting"] for r in retain_rows}):
            subset = [r for r in retain_rows
                      if r["setting"] == setting and r["cell"] == cell
                      and int(r["subspace_dim"]) == 8]
            if subset:
                print("  {:12s} {:7s} retained={:.1%}".format(
                    setting, cell,
                    st.mean([float(r["dense_improvement_retained"]) for r in subset])))


if __name__ == "__main__":
    main()
