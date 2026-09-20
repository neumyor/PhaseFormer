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

    def counts(self) -> dict:
        counts: dict = {}
        for row in self.rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        return counts

    def to_dict(self) -> dict:
        """Machine-readable form of the report.

        Step 7 invokes the auditor with ``--json``, so this is the acceptance
        audit's durable artifact; before 2026-09-20 the flag was declared and
        parsed but never used, so the promised file was silently never written.
        """
        return {
            "root": str(ROOT),
            "criteria": list(self.rows),
            "counts": self.counts(),
            "failing": [f"{r['experiment']} / {r['criterion']}" for r in self.fails()],
            "total_criteria": len(self.rows),
        }

    def write_json(self, path) -> None:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2,
                                     ensure_ascii=False) + "\n", encoding="utf-8")

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


#: The only arms whose mechanism owns a ``weak_period_residual_gate`` parameter.
#: ``phase_only`` (no_residual), ``l_rcrf`` (rcrf_nlinear_plain) and ``a1``
#: (gold_combo_reliability_s2) have no gate, so their rows carry no gate value.
GATE_ARMS = ("l_main", "l_q1_4", "l_q1_8")

#: The intervention arms every E16 cell carries.  The *count* is not fixed:
#: ``build_arm_plan`` adds ``PCA-matched-only``/``-drop`` only when the semantic
#: image is narrower than the head's rank, and ``Conditional-RRR-only`` only for
#: cells whose Stage-3 subspace file exists (the dense head has none).  Measured
#: from each cell's own checkpoint on 2026-09-20: 11 arms (24 cells), 12 (21),
#: 13 (18) -- 750 rows over 63 cells and 13 distinct names.  A hardcoded
#: "63 x 11 = 693" would therefore have failed step 7 after the 3-5 h run.
E16_ALWAYS_ARMS = (
    "Original", "Semantic-only", "Semantic-drop", "Semantic8-only",
    "Semantic8-drop", "Bias-off", "PCA-only", "PCA-drop",
    "Independent-RRR-only", "RandomRRR-drop",
)


def is_true(value) -> bool:
    """Read a boolean from a CSV cell the way ``e14_params.py`` writes it.

    CSV has no booleans: the writer emits Python ``True``/``False``, and a
    missing cell must read as False rather than as the non-empty string "None".
    """
    return str(value).strip().lower() in ("true", "1", "yes")


def check_exists(report: Report, experiment: str, rel: str, label: str):
    """Record presence, and return the path (or None when absent).

    Presence is recorded as PASS so the summary counts stay meaningful; only the
    content checks below can turn a present artifact into a FAIL.
    """
    path = ROOT / rel
    if not path.is_file():
        report.add(experiment, label, "PENDING", f"{rel} does not exist yet")
        return None
    report.add(experiment, label, "PASS", "present")
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
                   f"{len(blank)} blank row(s)"
                   + (f" e.g. {blank[0].get('setting') or blank[0].get('arm') or '<no setting column>'}"
                      if blank else ""))

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
        # `e14_params.py` is REPORT-ONLY: it lists cells it could not resolve in
        # ``unresolved`` and exits 0 anyway, so "non-empty" would pass while half
        # the matrix is missing.  One row is written per cell, so the count is the
        # real criterion -- and each row carries a cross-check against the run's own
        # `metrics.csv:parameter_count`, which must not be False.
        report.add("E14 (§4.2)", "parameter table covers all 492 cells",
                   "PASS" if len(rows) == 492 else "FAIL", f"{len(rows)} rows")
        crossed = [r for r in rows
                   if str(r.get("total_matches_metrics", "")).strip().lower() == "false"]
        report.add("E14 (§4.2)", "parameter counts agree with metrics.csv",
                   "PASS" if not crossed else "FAIL",
                   f"{len(crossed)} row(s) mismatched"
                   + (f" e.g. {crossed[0].get('setting')}/{crossed[0].get('arm')}"
                      if crossed else ""))
        # 2026-09-20 FIX (see 05_audit.md section 17): section 4.2's gate fallback
        # used to key its lookup by (arm, horizon), which pooled the dataset away
        # and handed every reused cell the SMALLEST gate of any dataset at that
        # horizon.  The built table carries no sign of that -- it stays complete and
        # every cell holds a plausible number -- so the criterion has to be on the
        # SCHEMA that makes the correct lookup possible: a parameter table without a
        # populated dataset column cannot identify any cell's gate.
        has_column = bool(rows) and "dataset" in rows[0]
        blank_dataset = [r for r in rows if not str(r.get("dataset", "")).strip()]
        report.add("E14 (§4.2)", "parameter table carries a populated dataset column",
                   "PASS" if (has_column and not blank_dataset) else "FAIL",
                   ("column missing from the header" if not has_column
                    else f"{len(blank_dataset)} of {len(rows)} row(s) without a dataset"
                         + (f" e.g. {blank_dataset[0].get('setting')}"
                            if blank_dataset else "")))
        # And the column must actually DISCRIMINATE: one dataset repeated on every
        # row would satisfy the check above while being just as useless.
        distinct = sorted({str(r.get("dataset", "")).strip() for r in rows})
        report.add("E14 (§4.2)", "parameter table spans the 7 datasets",
                   "PASS" if len(distinct) == 7 else "FAIL",
                   f"{len(distinct)} distinct dataset value(s): {distinct}")
        # The gate only exists in the three weak-residual arms; `phase_only`
        # (no_residual), `l_rcrf` (rcrf_nlinear_plain) and `a1`
        # (gold_combo_reliability_s2) have no such parameter at all, so 85 of the
        # 492 rows legitimately carry no gate value.  Requiring one from *every*
        # row would have failed this gate after all 411 runs had finished.
        # The row's own `gate_param_present` column is the discriminator, and the
        # arm/gate equivalence is asserted separately so that a mis-populated
        # column cannot pass by being uniformly False.
        present = [r for r in rows if is_true(r.get("gate_param_present"))]
        no_gate = [r for r in present
                   if not str(r.get("gate_value_from_checkpoint", "")).strip()]
        mismatched = [r for r in rows
                      if (str(r.get("arm", "")) in GATE_ARMS)
                      != is_true(r.get("gate_param_present"))]
        report.add("E14 (§4.2)", "gate value recovered from every gated checkpoint",
                   "PASS" if not no_gate else "FAIL",
                   f"{len(no_gate)} gated row(s) without a gate value "
                   f"(of {len(present)} gated rows; "
                   f"{len(rows) - len(present)} rows have no gate parameter)"
                   + (f" e.g. {no_gate[0].get('setting')}/{no_gate[0].get('arm')}"
                      if no_gate else ""))
        # A mechanism without the parameter cannot produce a value for it, so a
        # non-empty value on such a row means the table is mis-populated -- the
        # arm/presence comparison alone would not notice it.
        stray = [r for r in rows
                 if not is_true(r.get("gate_param_present"))
                 and str(r.get("gate_value_from_checkpoint", "")).strip()]
        inconsistent = mismatched + stray
        report.add("E14 (§4.2)", "gate presence matches the arm table",
                   "PASS" if not inconsistent else "FAIL",
                   f"{len(mismatched)} row(s) whose arm and gate_param_present "
                   f"disagree, {len(stray)} gateless row(s) carrying a gate value"
                   + (f" e.g. {inconsistent[0].get('setting')}/"
                      f"{inconsistent[0].get('arm')}"
                      if inconsistent else ""))

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
        # Same lesson as above, one level up: `total_params_per_horizon` is the
        # number the paper quotes, and it is NOT one number -- the phase trunk
        # scales with the dataset's channel count (H=192: 140191 on the 7-channel
        # datasets, 411913 Electricity, 412454 Traffic).  A bare per-horizon map
        # therefore lets a reader attribute it to any dataset, so the builder now
        # names the dataset it quoted and publishes the full spread beside it.
        arms_with_params = [r for r in rows
                            if str(r.get("total_params_per_horizon", "")).strip()
                            not in ("", "None")]
        attributed = [r for r in arms_with_params
                      if str(r.get("total_params_reference_dataset", "")).strip()
                      not in ("", "None")]
        report.add("E14 (§4.2)", "per-horizon parameter counts name their dataset",
                   "PASS" if len(attributed) == len(arms_with_params) else "FAIL",
                   f"{len(attributed)} of {len(arms_with_params)} parameter-bearing "
                   f"arm row(s) attribute the quoted count")

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
    # Kept outside the block: the intervention-table criteria below cross-check
    # the table against the runner's own declared counts.
    summary_obj: dict = {}
    if summary:
        obj = read_json(summary)
        summary_obj = obj if isinstance(obj, dict) else {}
        # 2026-09-20 FIX: these were read as TOP-LEVEL keys (`algebra_failures`,
        # `reference_parity_passed`, `test_split_read`, ...), but the producer does
        # not write them there -- it nests them (`invariants.*`,
        # `reference_parity.passed`, `reads_test`).  All six would have reported
        # "key absent" -> FAIL against a perfectly good run, and the criterion that
        # actually matters (`reference_parity.passed`) would never have been read.
        # The paths below come from a REAL summary, not from the plan's prose.
        expected = (
            # NOT `cells` -- that top-level key is the list of cell records;
            # the count lives under counts.cells.
            ("counts.cells", ("counts", "cells"), 63),
            ("invariants.algebra_failures", ("invariants", "algebra_failures"), []),
            ("invariants.run_metric_failures", ("invariants", "run_metric_failures"), []),
            ("invariants.run_metric_not_comparable",
             ("invariants", "run_metric_not_comparable"), {}),
            ("reads_test", ("reads_test",), False),
        )
        for label, path, want in expected:
            value = obj
            for step in path:
                value = value.get(step) if isinstance(value, dict) else None
            if value is None:
                report.add("E16 (§4.4)", label, "FAIL",
                           "key absent (looked at summary" +
                           "".join("." + s for s in path) + ")")
                continue
            report.add("E16 (§4.4)", label, "PASS" if value == want else "FAIL",
                       f"got {value!r}, want {want!r}")
        # Reference parity is a criterion of the plan of record (01_plan.md:272 and
        # 02_03_static_check_smoke.md:133: the full run "must give true on all eight
        # fields"), so a False here is a real contradiction to adjudicate rather
        # than a schema detail to paper over.  It is reported separately because it
        # lives under `reference_parity`.
        parity_obj = obj.get("reference_parity")
        if not isinstance(parity_obj, dict):
            report.add("E16 (§4.4)", "summary.reference_parity", "FAIL", "key absent")
        else:
            passed = parity_obj.get("passed")
            report.add("E16 (§4.4)", "reference_parity.passed",
                       "PASS" if passed is True else "FAIL",
                       f"got {passed!r}, want True (plan 01_plan.md:272)")
            details = parity_obj.get("fields") or {}
            worst = max([v.get("max_abs_diff", 0.0) for v in details.values()
                         if isinstance(v, dict)] or [0.0])
            # The path-mismatched cells are DOCUMENTED as expected (the manifest's
            # reuse tie-break and the low-rank inventory pick different
            # candidates), so their count is not a criterion -- but it must be
            # visible, because it is also the number of cells that could not be
            # value-compared at all.
            report.add("E16 (§4.4)", "reference_parity coverage", "INFO",
                       f"max_abs_diff={worst:.3e} over {len(details)} field(s); "
                       f"value-compared={parity_obj.get('cells_value_compared')}, "
                       f"path-mismatched="
                       f"{len(parity_obj.get('checkpoint_path_mismatches') or [])}")

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
        groups: dict = {}
        for row in rows:
            key = (str(row.get("arm")), str(row.get("setting")),
                   str(row.get("seed")))
            groups.setdefault(key, set()).add(str(row.get("intervention_arm")))
        counts = sorted({len(names) for names in groups.values()})
        report.add("E16 (§4.4)", "intervention table covers 63 cells",
                   "PASS" if len(groups) == 63 else "FAIL",
                   f"{len(groups)} distinct (arm, setting, seed) group(s)")
        missing = {f"{k[0]}__{k[1]}-s{k[2]}": sorted(set(E16_ALWAYS_ARMS) - names)
                   for k, names in groups.items()
                   if set(E16_ALWAYS_ARMS) - names}
        report.add("E16 (§4.4)", "every cell carries the always-present arms",
                   "PASS" if not missing else "FAIL",
                   f"{len(missing)} cell(s) missing an arm"
                   + (f" e.g. {sorted(missing)[0]}: {missing[sorted(missing)[0]]}"
                      if missing else ""))
        # Count-agnostic cross-check against the runner's own declaration: the
        # per-cell count varies by design, so the criterion is agreement, not a
        # constant.
        declared_total = (summary_obj.get("counts") or {}).get("intervention_rows")
        if isinstance(declared_total, int):
            report.add("E16 (§4.4)", "intervention rows match the runner's count",
                       "PASS" if len(rows) == declared_total else "FAIL",
                       f"table {len(rows)} vs summary intervention_rows "
                       f"{declared_total}")
        else:
            report.add("E16 (§4.4)", "intervention rows match the runner's count",
                       "INFO", "e16_summary.json carries no intervention_rows")
        declared_counts = (summary_obj.get("counts") or {}).get(
            "intervention_arms_per_cell")
        if isinstance(declared_counts, list) and declared_counts:
            want = sorted({int(value) for value in declared_counts})
            report.add("E16 (§4.4)", "per-cell arm counts match the runner's",
                       "PASS" if counts == want else "FAIL",
                       f"table {counts} vs summary {want}")
        else:
            report.add("E16 (§4.4)", "per-cell arm counts match the runner's",
                       "INFO", "e16_summary.json carries no per-cell arm counts")
        report.add("E16 (§4.4)", "intervention arms per cell",
                   "INFO",
                   f"{len(arms)} distinct arm(s), per-cell counts {counts}, "
                   f"{len(rows)} rows (the count is data-dependent: PCA-matched "
                   f"and Conditional-RRR-only are per-cell)")
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

    # A missing projector is an infrastructure failure, not a benign skip: the
    # frozen bases are what the two frozen arms train against.  Its consequence is
    # also visible downstream (no runs -> no cells), but an explicit check names the
    # actual cause instead of leaving "0 new cells" as the only symptom.
    summary = check_exists(report, "E17 (§4.5)", f"{E17}/e17_summary.json",
                           "e17_summary.json present")
    if summary:
        obj = read_json(summary)
        present = obj.get("projector_audit_present")
        report.add("E17 (§4.5)", "projector audit flag set by the runner",
                   "PASS" if present else "FAIL", f"projector_audit_present={present!r}")
        missing = obj.get("missing_projectors") or []
        report.add("E17 (§4.5)", "no missing frozen projectors",
                   "PASS" if not missing else "FAIL",
                   f"{len(missing)} missing" + (f" e.g. {missing[:2]}" if missing else ""))

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
        # One row per cell, and the plan is 42 smoothing rows (7 settings x 3 seeds
        # x 2 levels) + 36 boundary rows (6 settings x 3 seeds x 2 ranks) = 78 --
        # the same count the pipeline trains.  "non-empty" would pass a half-run.
        report.add("E18 (§4.6)", "results rows = 78",
                   "PASS" if len(rows) == 78 else "FAIL", f"{len(rows)} rows")
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

    counts = report.counts()
    print(f"\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.json:
        report.write_json(args.json)
        print(f"report written: {args.json}")

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
