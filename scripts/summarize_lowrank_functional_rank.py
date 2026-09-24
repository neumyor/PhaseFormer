#!/usr/bin/env python3
"""Compact aggregate view of the functional-rank sweep (read-only)."""

from __future__ import annotations

import collections
import csv
import statistics as st
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "functional_rank_cells.csv"
    rows = [r for r in csv.DictReader(open(path)) if r["dataset"] != "Electricity"]
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in rows:
        group[(row["setting"], int(row["rank"]))].append(row)

    header = (
        "setting", "rank", "n", "r90_c", "r95_c", "r99_c",
        "r95_sing", "r95_act", "neg", "top1", "mse_full",
    )
    print("{:12s} {:>5s} {:>2s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s} {:>7s} {:>7s} {:>11s}".format(*header))
    for key in sorted(group, key=lambda item: (item[0], item[1])):
        values = group[key]
        def mean_sd(field: str) -> str:
            data = [float(v[field]) for v in values]
            return "{:.1f}±{:.1f}".format(st.mean(data), st.pstdev(data))
        print("{:12s} {:5d} {:2d} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s} {:>7s} {:>7s} {:>11.6f}".format(
            key[0], key[1], len(values),
            mean_sd("r90_contribution"), mean_sd("r95_contribution"), mean_sd("r99_contribution"),
            mean_sd("r95_singular"), mean_sd("r95_activation_energy"),
            mean_sd("n_negative_contribution"),
            "{:.0%}".format(st.mean([float(v["contribution_share_of_top1"]) for v in values])),
            st.mean([float(v["mse_full_lowrank"]) for v in values]),
        ))


if __name__ == "__main__":
    main()
