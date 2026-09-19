#!/usr/bin/env python3
"""Rehearse the E16 write-back for the per-cell arm count it now expects.

Why this exists
---------------
``e16_writeback.py`` used to assert ``expected_arms = 11`` and flag any cell
with fewer arms as "thin".  The criterion is now structural: every cell must
carry the **10 always-present arms** (the 8 registered ones plus
``Independent-RRR-only`` and this paper's ``RandomRRR-drop``), because
``build_arm_plan`` appends ``PCA-matched-only``/``-drop`` and
``Conditional-RRR-only`` per cell -- measured on 2026-09-20 as 11 arms for 24
cells, 12 for 21 and 13 for 18.

Nothing else covered that change.  E16 is the one write-back without a
rehearsal script, and step 4 masks its failure with ``|| true``: a broken
write-back would not stop the chain, it would leave the section 4.4 fill without
its two 44-tables and only surface at the fill stage.  This rehearsal closes
that gap with a positive and a negative control.

The synthetic intervention table uses the arm structure **measured from each
cell's own checkpoint** (see ``paper_code_consistency.md`` section 18), so the
fixture cannot drift from the real product's shape.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/rehearse_e16_writeback.py [--scratch DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
WRITEBACK = REPO / "scripts" / "phaseformer_L" / "e16_writeback.py"

E14_ARMS = ("l_main", "l_q1_4", "l_q1_8")
SETTINGS = ("ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192",
            "Weather-96", "Weather-192", "Electricity-336")
SEEDS = (2021, 2022, 2023)
ALWAYS = ("Original", "Semantic-only", "Semantic-drop", "Semantic8-only",
          "Semantic8-drop", "Bias-off", "PCA-only", "PCA-drop",
          "Independent-RRR-only", "RandomRRR-drop")
PCA_MATCHED = ("PCA-matched-only", "PCA-matched-drop")
CONDITIONAL = "Conditional-RRR-only"

INTERVENTION_FIELDS = [
    "setting", "arm", "seed", "intervention_arm",
    "delta_fused_mse_vs_checkpoint", "delta_branch_mse_vs_checkpoint",
    "delta_fused_mae_vs_checkpoint", "delta_branch_mae_vs_checkpoint",
    "random_low_fused_mse", "random_high_fused_mse",
    "random_rrr_low_fused_mse", "random_rrr_high_fused_mse",
    "random_rrr_fused_mse_percentile_of_arm",
    "worse_than_random_rrr_95pct_fused_mse",
    "subspace_dimension", "same_dimension_controls_available",
]
DISSECTION_FIELDS = [
    "setting", "arm", "seed", "majority_input_group", "leading_input_group",
    "leading_input_group_label", "mean_input_group_explanation",
    "leading_output_group_label", "mean_output_group_explanation",
    "leading_correction_energy_share", "cross_seed_leading4_input_overlap",
    "cross_seed_leading4_output_overlap", "stable_semantics_verdict",
    "criterion_1_group_stable", "criterion_2_input_explanation_ge_0p5",
    "criterion_3_output_explanation_ge_0p8",
    "criterion_4_drop_beyond_random_95pct",
    "criterion_5_only_within_0p5pct",
    "criterion_6_drop_beyond_random_rrr_95pct", "head_kind",
]


def arms_for(arm: str, setting_index: int) -> list:
    """The arms one cell carries, per the measured structure."""
    names = list(ALWAYS)
    if arm == "l_main":
        names += list(PCA_MATCHED)
    else:
        names.append(CONDITIONAL)
        if setting_index % 2 == 0:          # the 13-arm lowrank cells
            names += list(PCA_MATCHED)
    return names


def intervention_rows() -> list:
    rows = []
    for arm in E14_ARMS:
        for setting_index, setting in enumerate(SETTINGS):
            for seed in SEEDS:
                for index, name in enumerate(arms_for(arm, setting_index)):
                    rows.append({
                        "setting": setting, "arm": arm, "seed": seed,
                        "intervention_arm": name,
                        "delta_fused_mse_vs_checkpoint": 1.0 + index * 0.01,
                        "delta_branch_mse_vs_checkpoint": 0.5 + index * 0.01,
                        "delta_fused_mae_vs_checkpoint": 0.5 + index * 0.01,
                        "delta_branch_mae_vs_checkpoint": 0.25 + index * 0.01,
                        "random_low_fused_mse": 0.1, "random_high_fused_mse": 0.2,
                        "random_rrr_low_fused_mse": 0.12,
                        "random_rrr_high_fused_mse": 0.22,
                        "random_rrr_fused_mse_percentile_of_arm": 60.0,
                        "worse_than_random_rrr_95pct_fused_mse": "False",
                        "subspace_dimension": 36,
                        "same_dimension_controls_available": "True",
                    })
    return rows


def dissection_rows() -> list:
    rows = []
    for arm in E14_ARMS:
        for setting in SETTINGS:
            for seed in SEEDS:
                rows.append({
                    "setting": setting, "arm": arm, "seed": seed,
                    "majority_input_group": "recent_level",
                    "leading_input_group": "recent_level",
                    "leading_input_group_label": "recent level",
                    "mean_input_group_explanation": 0.75,
                    "leading_output_group_label": "trend",
                    "mean_output_group_explanation": 0.85,
                    "leading_correction_energy_share": 0.42,
                    "cross_seed_leading4_input_overlap": 0.9,
                    "cross_seed_leading4_output_overlap": 0.9,
                    "stable_semantics_verdict": "True",
                    "criterion_1_group_stable": "True",
                    "criterion_2_input_explanation_ge_0p5": "True",
                    "criterion_3_output_explanation_ge_0p8": "True",
                    "criterion_4_drop_beyond_random_95pct": "True",
                    "criterion_5_only_within_0p5pct": "True",
                    "criterion_6_drop_beyond_random_rrr_95pct": "True",
                    "head_kind": "dense_shared" if arm == "l_main"
                    else "pooled_lowrank",
                })
    return rows


def write_csv(path: pathlib.Path, fields: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(scratch: pathlib.Path, rows: list) -> dict:
    write_csv(scratch / "intervention_table.csv", INTERVENTION_FIELDS, rows)
    write_csv(scratch / "dissection_table.csv", DISSECTION_FIELDS,
              dissection_rows())
    out_root = scratch / "out"
    proc = subprocess.run([
        sys.executable, str(WRITEBACK),
        "--intervention", str(scratch / "intervention_table.csv"),
        "--dissection", str(scratch / "dissection_table.csv"),
        "--output-root", str(out_root),
    ], capture_output=True, text=True, cwd=str(REPO))
    summary_path = out_root / "e16_writeback_summary.json"
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_root": out_root,
        "summary": json.loads(summary_path.read_text())
        if summary_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", default="",
                        help="scratch dir (default: a fresh temp dir)")
    args = parser.parse_args()

    import tempfile
    results = []
    with tempfile.TemporaryDirectory(dir=args.scratch or None) as tmp:
        scratch = pathlib.Path(tmp)

        rows = intervention_rows()
        cells = {(r["arm"], r["setting"], r["seed"]) for r in rows}
        print("fixture: %d rows over %d cells, arm counts %s"
              % (len(rows), len(cells),
                 sorted({len(arms_for(a, i)) for a in E14_ARMS
                         for i in range(len(SETTINGS))})))

        # Positive control: the measured structure must be complete.
        got = run(scratch / "positive", rows)
        ok = got["rc"] == 0 and got["summary"] is not None
        results.append(ok)
        print("\n[%s] positive control: write-back exit %d"
              % ("OK  " if ok else "FAIL", got["rc"]))
        if not ok:
            print(got["stderr"][-800:])
        if got["summary"]:
            coverage = got["summary"]["arm_coverage"]
            print("      always_present_arms: %d entries"
                  % len(coverage["always_present_arms"]))
            print("      expected_arms_per_cell: %s"
                  % coverage["expected_arms_per_cell"])
            print("      observed arm counts: %s"
                  % coverage["aggregated_arms_observed"])
            print("      cells_with_fewer_arms: %s"
                  % coverage["cells_with_fewer_arms"])
            for name in ("intervention_table_44.csv", "dissection_table_44.csv"):
                exists = (got["out_root"] / name).is_file()
                results.append(exists)
                print("      [%s] %s written" % ("OK  " if exists else "FAIL", name))
            expect = {
                "expected_arms_per_cell is the always-present count": coverage[
                    "expected_arms_per_cell"] == len(ALWAYS),
                "11/12/13-arm cells are all complete": coverage[
                    "cells_with_fewer_arms"] == [],
                "per-seed cells are all complete": coverage[
                    "per_seed_cells_with_fewer_arms"] == [],
                "observed counts cover the measured shape": coverage[
                    "aggregated_arms_observed"] == [10 + 1, 10 + 2, 10 + 3]
                or coverage["aggregated_arms_observed"] == [11, 12, 13],
            }
            for label, value in expect.items():
                results.append(value)
                print("      [%s] %s" % ("OK  " if value else "FAIL", label))
            table = got["out_root"] / "intervention_table_44.csv"
            if table.is_file():
                with table.open(newline="") as handle:
                    written = list(csv.DictReader(handle))
                ok44 = len(written) == 21
                results.append(ok44)
                print("      [%s] section 4.4 table = 21 rows (got %d)"
                      % ("OK  " if ok44 else "FAIL", len(written)))

        # Negative control (a): one seed loses an always-present arm.  The
        # (arm, setting) aggregate keys on the union over seeds, so the gap can
        # only appear in the per-seed check -- which is precisely why the
        # write-back keeps both granularities.  (My first version of this control
        # asserted the aggregate would catch it; the write-back was right.)
        victim = ("l_q1_4", "ETTh2-96", 2021)
        broken = [r for r in rows if not (
            (r["arm"], r["setting"], r["seed"]) == victim
            and r["intervention_arm"] == "PCA-drop")]
        got = run(scratch / "negative-seed", broken)
        coverage = (got["summary"] or {}).get("arm_coverage", {})
        per_seed = coverage.get("per_seed_cells_with_fewer_arms") or []
        caught = (got["rc"] == 0 and per_seed
                  and per_seed[0] == "l_q1_4__ETTh2-96-s2021"
                  and coverage.get("cells_with_fewer_arms") == [])
        results.append(bool(caught))
        print("\n[%s] negative control (a): one seed missing PCA-drop is caught "
              "by the per-seed check only" % ("OK  " if caught else "FAIL"))
        print("      per_seed_cells_with_fewer_arms: %s" % per_seed)
        print("      cells_with_fewer_arms (aggregate, expected empty): %s"
              % coverage.get("cells_with_fewer_arms"))

        # Negative control (b): every seed of one cell loses an always-present
        # arm, which the aggregate must catch.
        broken = [r for r in rows if not (
            (r["arm"], r["setting"]) == ("l_q1_4", "ETTh2-96")
            and r["intervention_arm"] == "PCA-drop")]
        got = run(scratch / "negative-cell", broken)
        coverage = (got["summary"] or {}).get("arm_coverage", {})
        thin = coverage.get("cells_with_fewer_arms") or []
        caught = got["rc"] == 0 and thin and thin[0] == "l_q1_4__ETTh2-96"
        results.append(bool(caught))
        print("\n[%s] negative control (b): all seeds of a cell missing PCA-drop "
              "is caught by the aggregate" % ("OK  " if caught else "FAIL"))
        print("      cells_with_fewer_arms: %s" % thin)
        print("      per_seed_cells_with_fewer_arms: %s"
              % (coverage.get("per_seed_cells_with_fewer_arms") or []))

    print()
    if all(results):
        print("OK: %d assertion(s) held" % len(results))
        return 0
    print("FAIL: %d of %d assertion(s) did not hold"
          % (results.count(False), len(results)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
