#!/usr/bin/env python3
"""Flag output columns that are empty in every row.

The recurring failure mode in this line has not been a crash: it has been a table
column that exists in the header and stays blank in every row, so the minipaper
silently loses a required field.  Four separate instances were found by hand
(the Golden parse dropping 16 of 28 rows, E16's non-existent
``majority_input_group_label``, the empty §4.2 provenance column, and a missing
must-answer block).  This script turns that hunt into one mechanical check.

For each CSV it reports:

* ``always_empty`` -- columns with no non-empty value in any row.  On real output
  this is a defect; on a deliberately partial fixture it is expected, so pass
  ``--expect-empty`` for the columns a fixture cannot fill.
* ``mostly_empty`` -- columns populated in fewer than ``--min-fill`` of the rows,
  which usually means a join key silently failed for most cells.
* ``constant`` -- columns whose non-empty values never vary, which is worth
  eyeballing (a genuine constant like a threshold is fine; an accidental one is
  not).

Usage::

    python scripts/phaseformer_L/check_builder_outputs.py \\
        --csv research_runs/phaseformer_L_e14_main_v1/main_table.csv \\
        --csv research_runs/phaseformer_L_e16_dissection_v1/dissection_table_44.csv \\
        --expect-empty gate_source_for_phase_only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMPTY_TOKENS = ("", "none", "nan", "null", "[]", "{}")


def is_empty(value) -> bool:
    return str(value if value is not None else "").strip().lower() in EMPTY_TOKENS


def check(path: Path, min_fill: float, expect_empty: set) -> dict:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return {"path": str(path), "rows": 0, "error": "no rows"}
    always_empty, mostly_empty, constant = [], [], []
    for field in fields:
        values = [row.get(field) for row in rows]
        filled = [v for v in values if not is_empty(v)]
        if not filled:
            if field not in expect_empty:
                always_empty.append(field)
            continue
        fill_rate = len(filled) / len(rows)
        if fill_rate < min_fill:
            mostly_empty.append({"column": field, "fill_rate": round(fill_rate, 3)})
        if len({str(v).strip() for v in filled}) == 1:
            constant.append({"column": field, "value": str(filled[0])[:60]})
    return {
        "path": str(path),
        "rows": len(rows),
        "columns": len(fields),
        "always_empty": always_empty,
        "mostly_empty": mostly_empty,
        "constant": constant,
        "expected_empty_allowed": sorted(expect_empty),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True,
                        help="repeatable; each path is checked independently")
    parser.add_argument("--min-fill", type=float, default=0.9,
                        help="a column below this fill rate is reported")
    parser.add_argument("--expect-empty", action="append", default=[],
                        help="column name that is legitimately empty in this fixture")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    reports = []
    for raw in args.csv:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            reports.append({"path": str(path), "error": "missing"})
            continue
        reports.append(check(path, args.min_fill, set(args.expect_empty)))

    problems = 0
    for report in reports:
        print(f"=== {report['path']}")
        if "error" in report:
            print(f"    {report['error']}")
            problems += 1
            continue
        print(f"    rows={report['rows']} columns={report['columns']}")
        if report["always_empty"]:
            print(f"    ALWAYS EMPTY: {report['always_empty']}")
            problems += len(report["always_empty"])
        if report["mostly_empty"]:
            print(f"    MOSTLY EMPTY: {report['mostly_empty']}")
            problems += len(report["mostly_empty"])
        if report["constant"]:
            print(f"    constant    : "
                  f"{[c['column'] for c in report['constant']]}")
        if not (report["always_empty"] or report["mostly_empty"]):
            print("    no empty-column problems")

    summary = {"reports": reports, "problems": problems}
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(json.dumps({"event": "finished", "problems": problems}))
    raise SystemExit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
