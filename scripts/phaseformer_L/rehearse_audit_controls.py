"""Calibration controls for ``audit_phase2_outputs.py`` (the step-7 acceptance gate).

An audit script that can only ever report "fine" proves nothing, which is why the
auditor takes ``--root``.  This harness builds deliberately *wrong* synthetic
trees, runs the auditor against each one, and asserts the exact verdict of the
criterion under test.

It exists because a criterion written as "every row has a gate value" looked
right and was wrong: the gate parameter only exists in the three weak-residual
arms (``l_main``/``l_q1_4``/``l_q1_8``), so 85 of the 492 real rows legitimately
carry no gate value, and the criterion would have failed the final gate *after*
all 411 runs had finished.  Every criterion whose verdict depends on the shape of
real data belongs in here.

Usage::

    python scripts/phaseformer_L/rehearse_audit_controls.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDITOR = REPO / "scripts" / "phaseformer_L" / "audit_phase2_outputs.py"
E14 = "research_runs/phaseformer_L_e14_main_v1"

GATE_ARMS = ("l_main", "l_q1_4", "l_q1_8")
GATELESS_ARMS = ("phase_only", "l_rcrf", "a1")
PARAM_COLUMNS = ("arm", "dataset", "horizon", "seed", "setting", "status",
                 "total_params", "gate_value_from_checkpoint",
                 "gate_param_present", "total_matches_metrics")

COVER_CRITERION = "parameter table covers all 492 cells"
CROSS_CRITERION = "parameter counts agree with metrics.csv"
GATE_CRITERION = "gate value recovered from every gated checkpoint"
ARM_CRITERION = "gate presence matches the arm table"


def parameter_row(index: int, arm: str | None = None) -> dict:
    """One parameter-table row, gated or not according to its arm."""
    arm = arm or GATE_ARMS[index % len(GATE_ARMS)]
    gated = arm in GATE_ARMS
    return {
        "arm": arm,
        "dataset": "ETTh2",
        "horizon": 96,
        "seed": 2021,
        "setting": f"ETTh2-96-{index}",
        "status": "new",
        "total_params": 10_000 + index,
        "gate_value_from_checkpoint": "0.2" if gated else "",
        "gate_param_present": str(gated),
        "total_matches_metrics": "True",
    }


def full_table() -> list:
    """492 rows: every third row a gated arm, the rest gateless (the real mix)."""
    rows = []
    for index in range(492):
        rows.append(parameter_row(index, GATE_ARMS[index % 3]
                                  if index % 3 else GATELESS_ARMS[index % 3]))
    return rows


def run_auditor(root: Path) -> dict:
    """Run the auditor against ``root`` and return {criterion: (state, detail)}."""
    proc = subprocess.run(
        [sys.executable, str(AUDITOR), "--root", str(root)],
        capture_output=True, text=True, cwd=str(REPO))
    verdicts = {}
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        for mark, state in (("OK", "PASS"), ("FAIL", "FAIL"),
                            ("--", "PENDING"), ("info", "INFO")):
            prefix = f"{mark} "
            if stripped.startswith(prefix):
                # The marker is padded ("OK  " / "--  "), so strip the rest:
                # without this every criterion name keeps leading spaces and no
                # lookup ever matches.
                rest = stripped[len(prefix):].strip()
                name, _, detail = rest.partition(": ")
                verdicts[name] = (state, detail)
                break
    return {"verdicts": verdicts, "rc": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def check(label: str, verdicts: dict, criterion: str, expected: str,
          results: list) -> None:
    got = verdicts.get(criterion, ("<absent>", ""))[0]
    ok = got == expected
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}: {criterion} -> {got} "
          f"(expected {expected})")
    if not ok:
        print(f"         detail: {verdicts.get(criterion, ('', '<absent>'))[1]}")


def write_fixture(root: Path, rows: list) -> None:
    out = root / E14
    out.mkdir(parents=True, exist_ok=True)
    with (out / "parameter_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PARAM_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    results: list = []
    print("auditor control calibration")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. The legitimate shape: 492 rows, gated rows carry a value, gateless
        #    rows carry none (this is the control for the false-FAIL defect).
        write_fixture(root, full_table())
        verdicts = run_auditor(root)["verdicts"]
        check("healthy table", verdicts, COVER_CRITERION, "PASS", results)
        check("healthy table", verdicts, GATE_CRITERION, "PASS", results)
        check("healthy table", verdicts, ARM_CRITERION, "PASS", results)

        # 2. A gated row that lost its value: the criterion must still bite.
        rows = full_table()
        victim = next(r for r in rows if r["arm"] in GATE_ARMS)
        victim["gate_value_from_checkpoint"] = ""
        write_fixture(root, rows)
        verdicts = run_auditor(root)["verdicts"]
        check("gated row without a value", verdicts, GATE_CRITERION, "FAIL", results)

        # 3. A gateless arm carrying a gate value: the two signals disagree.
        rows = full_table()
        victim = next(r for r in rows if r["arm"] in GATELESS_ARMS)
        victim["gate_value_from_checkpoint"] = "0.2"
        write_fixture(root, rows)
        verdicts = run_auditor(root)["verdicts"]
        check("gateless arm with a value", verdicts, ARM_CRITERION, "FAIL", results)

        # 4. A gated arm marked as having no gate parameter: uniform-False must
        #    not be able to pass the criterion by simply having nothing to check.
        rows = full_table()
        victim = next(r for r in rows if r["arm"] in GATE_ARMS)
        victim["gate_param_present"] = "False"
        victim["gate_value_from_checkpoint"] = ""
        write_fixture(root, rows)
        verdicts = run_auditor(root)["verdicts"]
        check("gated arm marked gateless", verdicts, ARM_CRITERION, "FAIL", results)

        # 5. Half a matrix must not pass "covers all 492 cells".
        write_fixture(root, full_table()[:491])
        verdicts = run_auditor(root)["verdicts"]
        check("491 rows", verdicts, COVER_CRITERION, "FAIL", results)

        # 6. One row disagreeing with its own metrics.csv cross-check.
        rows = full_table()
        rows[7]["total_matches_metrics"] = "False"
        write_fixture(root, rows)
        verdicts = run_auditor(root)["verdicts"]
        check("one cross-check False", verdicts, CROSS_CRITERION, "FAIL", results)

        # 7. The real defect, reproduced: a table in which *every* gateless row
        #    has no value (i.e. the real 85-row mix) must not be a failure --
        #    asserted again here with only gateless rows present.
        write_fixture(root, [parameter_row(i, GATELESS_ARMS[i % 3])
                             for i in range(492)])
        verdicts = run_auditor(root)["verdicts"]
        check("all-gateless table", verdicts, GATE_CRITERION, "PASS", results)
        check("all-gateless table", verdicts, ARM_CRITERION, "PASS", results)

    print()
    if all(results):
        print(f"OK: {len(results)} control(s) behaved as expected")
        return 0
    failed = results.count(False)
    print(f"FAIL: {failed} of {len(results)} control(s) behaved unexpectedly")
    return 1


if __name__ == "__main__":
    sys.exit(main())
