#!/usr/bin/env python3
"""Cross-check the fresh mode attribution and build the per-mode mechanism table.

Two jobs:

1. Verify the recomputed attribution against the earlier checkpoint-information
   run's ``semantic_alignment.csv``.  The exported modes were confirmed to be
   numerically identical to that run's (max |Δs| ~ 1e-15), so agreement here is a
   genuine reproduction check rather than two views of one computation.
2. Emit the table the analysis was missing: for each setting, the leading modes
   by measured predictive contribution, with what each reads, what each writes,
   and how much of the improvement it carries.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

GROUP_TO_LABEL = {
    "recent_level": "recent level",
    "level_change": "level change",
    "local_trend": "local trend",
    "local_curvature": "local curvature",
    "period_level": "periodic level",
    "period_shape": "periodic shape",
    "fast_local_change": "fast local change",
    "overall_displacement": "displacement",
    "slow_tilt": "tilt",
    "curvature": "curvature",
    "periodic": "periodic correction",
    "recent_shape_continuation": "shape continuation",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cross_check(new_rows: list[dict], old_rows: list[dict]) -> None:
    old = {
        (r["setting"], r["seed"], r["cell"], int(r["mode_index"])): r for r in old_rows
    }
    compared = matched_group = 0
    explanation_gaps = []
    for row in new_rows:
        key = (row["setting"], row["seed"], row["cell"], int(row["mode_index"]))
        if key not in old:
            continue
        compared += 1
        reference = old[key]
        if GROUP_TO_LABEL.get(reference["input_best_group"]) == row["input_best_group"]:
            matched_group += 1
        try:
            explanation_gaps.append(
                abs(float(reference["input_group_explanation"]) - float(row["input_group_explanation"]))
            )
        except (TypeError, ValueError):
            pass
    print(f"cross-check vs the earlier run's semantic_alignment.csv")
    print(f"  modes compared            : {compared}")
    print(f"  input best-group agreement: {matched_group}/{compared} "
          f"({matched_group / max(compared, 1):.1%})")
    if explanation_gaps:
        print(f"  max |Δ input group explanation|: {max(explanation_gaps):.2e}")


def mechanism_table(rows: list[dict], cells: list[dict], cell: str, height: int = 5) -> str:
    r95 = {
        (r["setting"], r["seed"], r["cell"]): int(r["r95_contribution"]) for r in cells
    }
    settings = ["ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192"]
    lines = [
        "| Setting | rank position | contribution share | input reads | group expl. | output writes | group expl. | paired mechanism |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for setting in settings:
        subset = [r for r in rows if r["setting"] == setting and r["cell"] == cell]
        if not subset:
            continue
        by_position: dict[int, list[dict]] = collections.defaultdict(list)
        for row in subset:
            by_position[int(row["contribution_rank"])].append(row)
        k95 = st.mean([
            r95[(r["setting"], r["seed"], r["cell"])]
            for r in subset if (r["setting"], r["seed"], r["cell"]) in r95
        ])
        for position in sorted(by_position):
            if position > height:
                break
            values = by_position[position]
            share = st.mean([float(v["share_of_positive"]) for v in values])
            lines.append(
                f"| {setting} | {position} | {share:.1%} | "
                f"{_mode(values, 'input_best_group')} | "
                f"{st.mean([float(v['input_group_explanation']) for v in values]):.2f} | "
                f"{_mode(values, 'output_best_group')} | "
                f"{st.mean([float(v['output_group_explanation']) for v in values]):.2f} | "
                f"{_mode(values, 'paired_mechanism')} |"
            )
        lines.append(
            f"| **{setting}** | *(r95 = {k95:.1f} of {values[0]['rank']})* | | | | | | |"
        )
    return "\n".join(lines)


def _mode(values: list[dict], key: str) -> str:
    counts = collections.Counter(v[key] for v in values)
    return counts.most_common(1)[0][0]


def composition_table(rows: list[dict], cell: str) -> str:
    """Group-by-group explanation of the leading modes (the plan's section 12.2 form)."""
    lines = [
        "| Setting | position | recent level | periodic shape | fast local change | local curvature | other | output displacement | output tilt | output curvature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in ["ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192"]:
        subset = [r for r in rows if r["setting"] == setting and r["cell"] == cell]
        by_position: dict[int, list[dict]] = collections.defaultdict(list)
        for row in subset:
            by_position[int(row["contribution_rank"])].append(row)
        for position in (1, 2, 3):
            values = by_position.get(position)
            if not values:
                continue
            import json as _json

            def average(field: str, label: str) -> float:
                numbers = []
                for v in values:
                    parsed = _json.loads(v[field])
                    if label in parsed:
                        numbers.append(parsed[label])
                return st.mean(numbers) if numbers else float("nan")

            def other(field: str, labels: list[str]) -> float:
                numbers = []
                for v in values:
                    parsed = _json.loads(v[field])
                    total = sum(val for key, val in parsed.items() if key not in labels)
                    numbers.append(total)
                return st.mean(numbers) if numbers else float("nan")

            in_labels = ["recent level", "periodic shape", "fast local change", "local curvature"]
            out_labels = ["displacement", "tilt", "curvature"]
            lines.append(
                f"| {setting} | {position} | "
                f"{average('input_group_explanations', 'recent level'):.2f} | "
                f"{average('input_group_explanations', 'periodic shape'):.2f} | "
                f"{average('input_group_explanations', 'fast local change'):.2f} | "
                f"{average('input_group_explanations', 'local curvature'):.2f} | "
                f"{other('input_group_explanations', in_labels):.2f} | "
                f"{average('output_group_explanations', 'displacement'):.2f} | "
                f"{average('output_group_explanations', 'tilt'):.2f} | "
                f"{average('output_group_explanations', 'curvature'):.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--reference", default="research_runs/lowrank_checkpoint_information_v1/semantic_alignment.csv")
    parser.add_argument("--cell", default="q=1/8")
    args = parser.parse_args()

    base = Path(args.run_dir)
    rows = read_csv(base / "mode_semantics.csv")
    cells = read_csv(base / "functional_rank_cells.csv")
    cross_check(rows, read_csv(Path(args.reference)))

    print()
    print(f"=== leading modes by predictive contribution ({args.cell}, 3 seeds) ===")
    print(mechanism_table(rows, cells, args.cell))
    print()
    print(f"=== group-by-group explanation ({args.cell}) ===")
    print(composition_table(rows, args.cell))


if __name__ == "__main__":
    main()
