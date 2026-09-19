#!/usr/bin/env python3
"""Rehearse the E17 write-back against the REAL frozen projector audit.

Why this exists
---------------
`e17_writeback.py` joins two inputs on ``(dataset, horizon)``: the results CSV
(whose ``horizon`` arrives as a *string* from CSV and is converted with
``int()``) and the projector audit JSON (whose ``horizon`` is a JSON number).
If those two key types ever disagree, the join returns nothing for every
setting and the section 4.5 cosine column comes out **empty** -- a silent loss
of exactly the evidence that says which settings can discriminate the two
frozen arms.  An empty column does not raise, so no exception surfaces.

The projector audit is a real, frozen artifact, so this rehearsal uses it
verbatim and synthesises only the results rows -- and takes that synthetic
schema from ``e17_conditional.RESULTS_FIELDS`` so it cannot drift from the
producer.

It asserts the strongest available property: the cosines the write-back reports
must **equal the audit's own values** for every setting.  On this suite that
means reproducing the known ``0.0066`` for Electricity-336 and ``>= 0.9991``
elsewhere; those were established independently when the projectors were fitted.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/rehearse_e17_writeback.py [--scratch DIR]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_AUDIT = ("research_runs/phaseformer_L_e17_conditional_v1/"
                 "projectors/projector_audit.json")


def literal_constants(path: pathlib.Path, names: set) -> dict:
    """Top-level ``NAME = <literal>`` assignments, read without importing."""
    out: dict = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="/tmp/rehearse_e17")
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    args = parser.parse_args()

    scratch = pathlib.Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    audit_path = pathlib.Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = REPO / audit_path
    if not audit_path.is_file():
        print(f"FATAL: projector audit not found: {audit_path}")
        return 2

    constants = literal_constants(
        REPO / "scripts/phaseformer_L/e17_conditional.py", {"RESULTS_FIELDS", "ARMS"})
    fields = constants.get("RESULTS_FIELDS")
    arms = constants.get("ARMS")
    if not fields or not arms:
        print("FATAL: could not read RESULTS_FIELDS/ARMS from e17_conditional.py")
        return 2

    audit = json.loads(audit_path.read_text())
    settings = sorted({(str(e["dataset"]), int(e["horizon"])) for e in audit})
    expected = {
        (str(e["dataset"]), int(e["horizon"])):
            (e.get("revin") or {}).get("abs_cos_conditional_vs_independent")
        for e in audit
    }
    print(f"real audit: {len(audit)} records, {len(settings)} settings")

    fixture = scratch / "results.with_test.csv"
    with fixture.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for dataset, horizon in settings:
            for arm in arms:
                for seed in (2021, 2022, 2023):
                    row = {key: "" for key in fields}
                    row.update({
                        "arm": arm, "dataset": dataset, "horizon": horizon,
                        "seed": seed, "setting": f"{dataset}-{horizon}",
                        "status": "read", "source": "rehearsal-fixture",
                        "test_mse": 0.2 + 0.01 * (seed - 2021),
                        "test_mae": 0.3 + 0.01 * (seed - 2021),
                        "h1_cond_gt_indep_seed_majority": (
                            "True" if arm == "direct" else ""),
                    })
                    writer.writerow(row)
    print(f"fixture rows: {len(settings) * len(arms) * 3}")

    result = subprocess.run(
        [sys.executable, "scripts/phaseformer_L/e17_writeback.py",
         "--results", str(fixture),
         "--projector-audit", str(audit_path),
         "--output-root", str(scratch)],
        cwd=REPO, capture_output=True, text=True)
    print(f"writeback exit: {result.returncode}")
    if result.returncode:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return 1

    table = list(csv.DictReader((scratch / "conditional_table.csv").open()))
    print(f"table rows: {len(table)} (expected {len(settings)})")
    if len(table) != len(settings):
        print("FAIL: row count does not match the audit's settings")
        return 1

    mismatches = 0
    for row in table:
        raw = str(row.get("conditional_vs_independent_direction_cos", "")).strip()
        if not raw:
            print(f"  {row['setting']:22s} EMPTY cosine -- the join did not bind")
            mismatches += 1
            continue
        got = float(raw)
        want = expected.get((row["dataset"], int(row["horizon"])))
        ok = want is not None and abs(got - want) < 1e-9
        mismatches += 0 if ok else 1
        print(f"  {row['setting']:22s} got={got:.6f} audit="
              f"{'None' if want is None else format(want, '.6f')} "
              f"{'OK' if ok else 'MISMATCH'}")

    distinct = sorted({str(r.get("frozen_arms_are_distinct")) for r in table})
    print(f"frozen_arms_are_distinct values: {distinct}")
    print(f"cosine mismatches: {mismatches}")

    if mismatches:
        print("FAIL: the write-back does not reproduce the audit's cosines")
        return 1
    print("PASS: every setting's cosine was recovered from the real audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
