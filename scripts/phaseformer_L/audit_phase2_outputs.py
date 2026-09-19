#!/usr/bin/env python3
"""Stage-5 acceptance audit over the phase-2 artifacts, for minipaper section 4.

Why this exists
---------------
Each experiment's formal-run acceptance criteria were written down before the
runs (E16's are tabulated in ``e16_dissection/02_03_static_check_smoke.md``
section 6.2, E18's in ``e18_negative/02_static_check.md``).  Auditing them by hand
after a multi-hour chain is exactly where a criterion gets skipped, so this
script encodes them and reports every one as PASS / FAIL / PENDING.

States
------
* ``PENDING`` -- the artifact does not exist yet (the stage has not run).  This
  is not a failure, so the script is safe to run at any time, including before
  phase 2 finishes.
* ``FAIL``    -- the artifact exists and contradicts a documented criterion.
  Any FAIL makes the script exit non-zero.
* ``INFO``    -- a quantity whose exact expected value is not fixed by the
  documents; reported for the human auditor without judging it.

Scope
-----
It checks *mechanical* criteria only (counts, flags, non-empty columns, row
structure).  It cannot judge whether a number is scientifically right; that
remains the stage-5 human review.  Criteria are transcribed from the documents,
not invented here.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/audit_phase2_outputs.py
    python scripts/phaseformer_L/audit_phase2_outputs.py --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Root the artifact paths are resolved against.  Overridable with ``--root`` so
#: the auditor can itself be pointed at a synthetic tree -- an audit script that
#: can only ever report "fine" proves nothing, and this is what lets the checks
#: be calibrated against deliberately wrong inputs.
ROOT = REPO

E14 = "research_runs/phaseformer_L_e14_main_v1"
E16 = "research_runs/phaseformer_L_e16_dissection_v1"
E17 = "research_runs/phaseformer_L_e17_conditional_v1"
E18 = "research_runs/phaseformer_L_e18_negative_v1"
E19 = "research_runs/phaseformer_L_e19_predictive_v1"


class Report:
    def __init__(self) -> None:
        self.rows: list = []

    def add(self, experiment: str, criterion: str, state: str, detail: str) -> None:
        self.rows.append({"experiment": experiment, "criterion": criterion,
                          "state": state, "detail": detail})

    def fails(self) -> list:
        return [r for r in self.rows if r["state"] == "FAIL"]

    def print(self) -> None:
        current = None
        for row in self.rows:
            if row["experiment"] != current:
                current = row["experiment"]
                print(f"\n=== {current} ===")
            mark = {"PASS": "OK  ", "FAIL": "FAIL", "PENDING": "--  ",
                    "INFO": "info"}[row["state"]]
            print(f"  {mark} {row['criterion']}: {row['detail']}")


def read_csv(path: pathlib.Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def check_exists(report: Report, experiment: str, rel: str, label: str):
    path = ROOT / rel
    if not path.is_file():
        report.add(experiment, label, "PENDING", f"{rel} does not exist yet")
        return None
    return path


# --------------------------------------------------------------------------
# E14 -- section 4.2 main matrix
# --------------------------------------------------------------------------

def audit_e14(report: Report) -> None:
    path = check_exists(report, "E14 (§4.2)", f"{E14}/results.csv", "results.csv present")
    if path:
        rows = read_csv(path)
        blank = [r for r in rows if not str(r.get("test_mse", "")).strip()
                 or not str(r.get("test_mae", "")).strip()]
        report.add("E14 (§4.2)", "results rows = 492",
                   "PASS" if len(rows) == 492 else "FAIL", f"got {len(rows)}")
        report.add("E14 (§4.2)", "every row carries test_mse and test_mae",
                   "PASS" if not blank else "FAIL",
                   f"{len(blank)} blank row(s)" + (f" e.g. {blank[0].get('setting')}" if blank else ""))

    summary = check_exists(report, "E14 (§4.2)",
                           f"{E14}/test_read_summary.json", "single test read summary")
    if summary:
        obj = read_json(summary)
        by_status = obj.get("by_status") or {}
        unresolved = {k: v for k, v in by_status.items()
                      if k.startswith("missing") and v}
        problems = obj.get("problems")
        bad = bool(unresolved) or (isinstance(problems, list) and problems)
        report.add("E14 (§4.2)", "stage B resolved every cell",
                   "FAIL" if bad else "PASS",
                   f"by_status={by_status} problems="
                   f"{len(problems) if isinstance(problems, list) else problems}")

    params = check_exists(report, "E14 (§4.2)", f"{E14}/parameter_table.csv",
                          "parameter_table.csv present")
    if params:
        rows = read_csv(params)
        report.add("E14 (§4.2)", "parameter table non-empty",
                   "PASS" if rows else "FAIL", f"{len(rows)} rows")

    main = check_exists(report, "E14 (§4.2)", f"{E14}/main_table.csv", "main_table.csv present")
    if main:
        rows = read_csv(main)
        appendix = [r for r in rows
                    if str(r.get("is_traffic_appendix")).strip() == "True"]
        core = len(rows) - len(appendix)
        report.add("E14 (§4.2)", "main table = 28 rows, 24 core + 4 appendix",
                   "PASS" if (len(rows), core, len(appendix)) == (28, 24, 4) else "FAIL",
                   f"{len(rows)} = {core} core + {len(appendix)} appendix")
        blank_prov = [r.get("setting") for r in rows
                      if not str(r.get("provenance_note", "")).strip()]
        report.add("E14 (§4.2)", "provenance column filled on every row",
                   "PASS" if not blank_prov else "FAIL",
                   f"{len(blank_prov)} blank" + (f" e.g. {blank_prov[:2]}" if blank_prov else ""))

    variant = check_exists(report, "E14 (§4.2)", f"{E14}/variant_table.csv",
                           "variant_table.csv present")
    if variant:
        rows = read_csv(variant)
        report.add("E14 (§4.2)", "variant table = 6 arms",
                   "PASS" if len(rows) == 6 else "FAIL", f"got {len(rows)}")

    claims = check_exists(report, "E14 (§4.2)", f"{E14}/claims.json", "claims.json present")
    if claims:
        obj = read_json(claims)
        need = ("A", "B", "C", "D", "must_answer_a", "must_answer_b")
        missing = [k for k in need if k not in obj]
        report.add("E14 (§4.2)", "claims + must-answer blocks present",
                   "PASS" if not missing else "FAIL",
                   f"missing {missing}" if missing else "all present")


# --------------------------------------------------------------------------
# E16 -- section 4.4
# --------------------------------------------------------------------------

def audit_e16(report: Report) -> None:
    summary = check_exists(report, "E16 (§4.4)", f"{E16}/e16_summary.json",
                           "e16_summary.json present")
    if summary:
        obj = read_json(summary)
        expected = {
            "cells": 63,
            "algebra_failures": 0,
            "run_metric_failures": 0,
            "run_metric_not_comparable": 0,
            "reference_parity_passed": True,
            "checkpoint_path_mismatches": [],
            "test_split_read": False,
        }
        for key, want in expected.items():
            if key not in obj:
                report.add("E16 (§4.4)", f"summary.{key}", "FAIL", "key absent")
                continue
            got = obj[key]
            report.add("E16 (§4.4)", f"summary.{key}", 
                       "PASS" if got == want else "FAIL", f"got {got!r}, want {want!r}")

    parity = check_exists(report, "E16 (§4.4)", f"{E16}/reference_parity.json",
                          "reference_parity.json present")
    if parity:
        obj = read_json(parity)
        probes = obj.get("probe_cells")
        # Documented as ~72 = 6 settings x 3 seeds x 2 low-rank arms x 2 files.
        report.add("E16 (§4.4)", "reference_parity.probe_cells near 72",
                   "PASS" if isinstance(probes, int) and abs(probes - 72) <= 2 else "FAIL",
                   f"got {probes!r}")

    inter = check_exists(report, "E16 (§4.4)", f"{E16}/intervention_table.csv",
                         "intervention_table.csv present")
    if inter:
        rows = read_csv(inter)
        arms = sorted({str(r.get("intervention_arm")) for r in rows})
        report.add("E16 (§4.4)", "intervention rows = 63 cells x 11 arms",
                   "PASS" if len(rows) == 63 * 11 else "FAIL", f"got {len(rows)}")
        report.add("E16 (§4.4)", "11 distinct intervention arms",
                   "PASS" if len(arms) == 11 else "FAIL",
                   f"got {len(arms)}: {arms}")
        rrr = [a for a in arms if "rrr" in a.lower()]
        report.add("E16 (§4.4)", "random-RRR subspace control present",
                   "PASS" if rrr else "FAIL", f"arms matching rrr: {rrr}")
        band = [c for c in rows[0] if "random_rrr" in c]
        report.add("E16 (§4.4)", "random-RRR band columns present",
                   "PASS" if band else "FAIL", f"{len(band)} column(s)")

    diss = check_exists(report, "E16 (§4.4)", f"{E16}/dissection_table.csv",
                        "dissection_table.csv present")
    if diss:
        rows = read_csv(diss)
        report.add("E16 (§4.4)", "dissection table = 21 rows (3 models x 7 settings)",
                   "PASS" if len(rows) == 21 else "FAIL", f"got {len(rows)}")

    for rel in (f"{E16}/intervention_table_44.csv", f"{E16}/dissection_table_44.csv"):
        check_exists(report, "E16 (§4.4)", rel, f"{pathlib.Path(rel).name} present")


# --------------------------------------------------------------------------
# E17 -- section 4.5
# --------------------------------------------------------------------------

def audit_e17(report: Report) -> None:
    audit = check_exists(report, "E17 (§4.5)", f"{E17}/projectors/projector_audit.json",
                         "frozen projector audit present")
    if audit:
        obj = read_json(audit)
        report.add("E17 (§4.5)", "projector audit has 7 settings",
                   "PASS" if len(obj) == 7 else "FAIL", f"got {len(obj)}")

    results = check_exists(report, "E17 (§4.5)", f"{E17}/results.with_test.csv",
                           "results.with_test.csv present")
    if results:
        rows = read_csv(results)
        new = [r for r in rows if str(r.get("status")) == "new"]
        unread = [r for r in new if not str(r.get("test_mse", "")).strip()]
        report.add("E17 (§4.5)", "24 newly trained cells present",
                   "PASS" if len(new) == 24 else "FAIL", f"got {len(new)}")
        report.add("E17 (§4.5)", "every new cell has a test metric",
                   "PASS" if not unread else "FAIL", f"{len(unread)} unread")

    table = check_exists(report, "E17 (§4.5)", f"{E17}/conditional_table.csv",
                         "conditional_table.csv present")
    if table:
        rows = read_csv(table)
        report.add("E17 (§4.5)", "conditional table = 7 settings",
                   "PASS" if len(rows) == 7 else "FAIL", f"got {len(rows)}")
        blank = [r.get("setting") for r in rows
                 if not str(r.get("conditional_vs_independent_direction_cos", "")).strip()]
        report.add("E17 (§4.5)", "cosine column recovered from the audit",
                   "PASS" if not blank else "FAIL",
                   f"{len(blank)} blank" + (f" e.g. {blank[:2]}" if blank else ""))


# --------------------------------------------------------------------------
# E18 -- section 4.6
# --------------------------------------------------------------------------

def audit_e18(report: Report) -> None:
    results = check_exists(report, "E18 (§4.6)", f"{E18}/results.with_test.csv",
                           "results.with_test.csv present")
    if results:
        rows = read_csv(results)
        report.add("E18 (§4.6)", "results non-empty",
                   "PASS" if rows else "FAIL", f"{len(rows)} rows")
        stages = sorted({str(r.get("stage")) for r in rows})
        report.add("E18 (§4.6)", "both stages present (smooth, rank12)",
                   "PASS" if set(stages) == {"smooth", "rank12"} else "FAIL",
                   f"got {stages}")
        unread = [r for r in rows if not str(r.get("test_mse", "")).strip()]
        report.add("E18 (§4.6)", "every cell has a test metric",
                   "PASS" if not unread else "FAIL", f"{len(unread)} unread")

    for rel, label in ((f"{E18}/e18_negative_verify.json", "completeness audit (rows 1+5)"),
                       (f"{E18}/e18_svd_truncation_summary.json", "row-3 checkpoint resolution")):
        path = check_exists(report, "E18 (§4.6)", rel, label)
        if path:
            obj = read_json(path)
            problems = obj.get("problems")
            if isinstance(problems, list):
                report.add("E18 (§4.6)", f"{label}: problems empty",
                           "PASS" if not problems else "FAIL", f"{len(problems)} problem(s)")
            else:
                report.add("E18 (§4.6)", f"{label}: problems field",
                           "INFO", f"no problems field; keys={sorted(obj)[:6]}")

    table = check_exists(report, "E18 (§4.6)", f"{E18}/svd_truncation_table_28.csv",
                         "svd_truncation_table_28.csv present")
    if table:
        rows = read_csv(table)
        settings = {str(r.get("setting")) for r in rows}
        report.add("E18 (§4.6)", "row 3 covers 28 settings",
                   "PASS" if len(settings) == 28 else "FAIL",
                   f"{len(settings)} distinct setting(s), {len(rows)} rows")

    negative = check_exists(report, "E18 (§4.6)", f"{E18}/negative_table.csv",
                            "negative_table.csv present")
    if negative:
        rows = read_csv(negative)
        ids = [str(r.get("row")) for r in rows]
        report.add("E18 (§4.6)", "negative table rows 1..5",
                   "PASS" if ids == ["1", "2", "3", "4", "5"] else "FAIL",
                   f"got {ids}")


# --------------------------------------------------------------------------
# E19 -- section 4.7
# --------------------------------------------------------------------------

def audit_e19(report: Report) -> None:
    path = check_exists(report, "E19 (§4.7)", f"{E19}/predictive_power.csv",
                        "predictive_power.csv present")
    if path:
        rows = read_csv(path)
        settings = {str(r.get("setting")) for r in rows}
        state = "PASS" if len(settings) == 28 else "FAIL"
        report.add("E19 (§4.7)", "predictive table covers 28 settings", state,
                   f"{len(settings)} distinct setting(s), {len(rows)} rows")
    check_exists(report, "E19 (§4.7)", f"{E19}/predictive_power_summary.json",
                 "predictive_power_summary.json present")
    check_exists(report, "E19 (§4.7)", f"{E19}/level_statistics.csv",
                 "level_statistics.csv present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="", help="write the report as JSON here")
    parser.add_argument("--root", default=str(REPO),
                        help="root holding research_runs/ (for self-tests)")
    args = parser.parse_args()

    global ROOT
    ROOT = pathlib.Path(args.root)

    report = Report()
    for fn in (audit_e14, audit_e16, audit_e17, audit_e18, audit_e19):
        fn(report)

    report.print()

    counts = {}
    for row in report.rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print(f"\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    failed = report.fails()
    if failed:
        print(f"\nFAILING CRITERIA ({len(failed)}):")
        for row in failed:
            print(f"  - {row['experiment']} / {row['criterion']}: {row['detail']}")
        print("\nNot all criteria are met; the write-back must not be treated as final.")
        return 1

    pending = counts.get("PENDING", 0)
    if pending:
        print(f"\n{pending} criterion/criteria still PENDING (their stage has not "
              f"produced its artifacts yet). No contradictions found so far.")
        return 0
    print("\nAll criteria met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
