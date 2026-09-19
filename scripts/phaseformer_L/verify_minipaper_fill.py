#!/usr/bin/env python3
"""Verify that the minipaper's filled section-4 rows match the produced artifacts.

Why this exists
---------------
Every other stage of the six-stage contract is now machine-checked: the artifacts
are produced by tools, and `audit_phase2_outputs.py` checks their internal
acceptance criteria.  The last stage -- putting those numbers INTO the paper -- was
still done by hand, and no tool writes to the minipaper (the two tools that
mention it only reference it in their docstrings).

Hand-transcription across five tables with dozens of numeric cells is exactly
where a paper silently ends up disagreeing with its own artifacts, and that error
survives every automated check because both sides are individually fine.

Criterion
---------
A displayed value must equal the artifact value **rounded to the displayed
precision**.  So ``0.712`` matches ``0.711673`` (3 dp) and ``1.90`` matches
``1.8987`` (2 dp).  Comparing raw floats would flag correct rounding as an error --
the same mistake that produced a false alarm in the E17 rehearsal.

States
------
* ``match``   -- displayed value equals the artifact at that precision;
* ``MISMATCH``-- it does not (this fails the run);
* ``blank``   -- the paper cell is still empty (reported as PENDING, not a failure,
  so this tool is safe to run while experiments are still producing data);
* ``PENDING`` -- the artifact does not exist yet.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/verify_minipaper_fill.py
    python scripts/phaseformer_L/verify_minipaper_fill.py --json report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
MINIPAPER = REPO / "docs/PhaseFormer_L_minipaper.md"
E15 = "research_runs/phaseformer_L_e15_dimension_v1"
E14 = "research_runs/phaseformer_L_e14_main_v1"

#: displayed column -> (artifact column, decimal places); None dp = exact string.
SECTION_4_3_COLUMNS = {
    2: ("lambda1_share_of_achievable", 3),
    3: ("pred_dims_90", 0),
    4: ("PR", 2),
    6: ("a1_vs_const_abs_cos", 3),
    7: ("used_var_share_r1", 3),
}


class Report:
    def __init__(self) -> None:
        self.rows: list = []

    def add(self, table: str, key: str, column: str, state: str, detail: str) -> None:
        self.rows.append({"table": table, "key": key, "column": column,
                          "state": state, "detail": detail})

    def count(self, state: str) -> int:
        return sum(1 for r in self.rows if r["state"] == state)


def section_text(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    stop = text.index(end, begin)
    return text[begin:stop]


def parse_markdown_table(block: str, header_needle: str):
    """Return (header_cells, [row_cells, ...]) for the table containing the needle."""
    lines = [ln.strip() for ln in block.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("|") and header_needle in line:
            header = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            for later in lines[index + 2:]:          # skip the |---| separator
                if not later.startswith("|"):
                    break
                rows.append([c.strip() for c in later.strip("|").split("|")])
            return header, rows
    return None, []


def as_float(raw: str):
    """Parse a paper cell; tolerate ``+0.0481``, ``−1.2`` (U+2212), ``0.950``."""
    cleaned = (raw.replace("−", "-").replace("–", "-").replace("+", "")
               .replace("−", "-").strip())
    if not cleaned or cleaned in {"—", "-", "–"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def compare_number(paper_cell: str, artifact_value, dp: int) -> tuple:
    """(state, detail) for one numeric cell against the artifact rounding."""
    got = as_float(paper_cell)
    if got is None:
        return "blank", f"paper cell {paper_cell!r}"
    if artifact_value in (None, ""):
        return "PENDING", "artifact value empty"
    want = round(float(artifact_value), dp) if dp else float(artifact_value)
    ok = abs(got - want) < 1e-9
    return ("match" if ok else "MISMATCH",
            f"paper={got} artifact={artifact_value} rounded={want} dp={dp}")


def check_4_3(report: Report, root: pathlib.Path, minipaper: pathlib.Path) -> None:
    path = root / E15 / "dimension_table.csv"
    block = section_text(minipaper.read_text(encoding="utf-8"), "### 4.3", "### 4.4")
    header, paper_rows = parse_markdown_table(block, "λ_1/Σλ")
    if header is None:
        report.add("§4.3", "-", "-", "MISMATCH", "could not find the 4.3 table header")
        return
    if not path.is_file():
        report.add("§4.3", "-", "artifact", "PENDING", f"{path.name} does not exist")
        return
    with path.open(newline="") as handle:
        artifact = {(r["dataset"], str(r["horizon"])): r for r in csv.DictReader(handle)}

    if len(paper_rows) != len(artifact):
        report.add("§4.3", "-", "row count", "MISMATCH",
                   f"paper has {len(paper_rows)} rows, artifact has {len(artifact)}")
    for cells in paper_rows:
        if len(cells) < 8:
            report.add("§4.3", "|".join(cells[:2]), "-", "MISMATCH",
                       f"row has {len(cells)} cells, expected 8")
            continue
        key = (cells[0], cells[1])
        source = artifact.get(key)
        if source is None:
            report.add("§4.3", f"{key[0]}-{key[1]}", "-", "MISMATCH",
                       "no artifact row for this dataset/horizon")
            continue
        label = f"{key[0]}-{key[1]}"
        for column, (artifact_column, dp) in SECTION_4_3_COLUMNS.items():
            state, detail = compare_number(cells[column], source.get(artifact_column), dp)
            report.add("§4.3", label, artifact_column, state, detail)
        # column 5 is the composite `b_1` template plus its |cos|
        template = cells[5]
        if not template:
            report.add("§4.3", label, "b1_best_template", "blank", "empty cell")
        else:
            artifact_template = str(source.get("b1_best_template", "")).replace("_", " ")
            artifact_template = artifact_template.replace("tau=", "τ=")
            artifact_cos = source.get("b1_best_template_abs_cos")
            cos_match = re.search(r"\(([-0-9.]+)\)", template)
            problems = []
            if artifact_template and artifact_template not in template:
                problems.append(f"template {artifact_template!r} not in {template!r}")
            if cos_match and artifact_cos not in (None, ""):
                shown = float(cos_match.group(1))
                if abs(shown - round(float(artifact_cos), 3)) >= 1e-9:
                    problems.append(f"|cos| shown {shown} vs artifact {artifact_cos}")
            report.add("§4.3", label, "b1_best_template",
                       "MISMATCH" if problems else "match",
                       "; ".join(problems) if problems else template)


def check_4_2(report: Report, root: pathlib.Path, minipaper: pathlib.Path) -> None:
    """Section 4.2 main table: compare each cell with the builder's own markdown row.

    This one is a plain field comparison rather than a numeric one, and it can be:
    `e14_writeback` emits its rows with exactly the paper's ten columns in exactly
    this order (dataset | horizon | golden_mse/mae | phase_only | PhaseFormer-L |
    Δ | g | s | stable | provenance).  Comparing the produced row text against the
    paper's row text is therefore the strongest available check -- it verifies not
    just the numbers but the column mapping and the provenance note as well.
    """
    md = root / E14 / "main_table.md"
    block = section_text(minipaper.read_text(encoding="utf-8"), "### 4.2", "### 4.3")
    header, paper_rows = parse_markdown_table(block, "Golden MSE/MAE")
    if header is None:
        report.add("§4.2", "-", "-", "MISMATCH", "could not find the 4.2 table header")
        return
    if len(header) != 10:
        report.add("§4.2", "-", "columns", "MISMATCH",
                   f"header has {len(header)} columns, expected 10")
    if not md.is_file():
        report.add("§4.2", "-", "artifact", "PENDING", f"{md.name} does not exist yet")
        return

    artifact: dict = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            artifact[(cells[0], cells[1])] = cells

    if len(paper_rows) != len(artifact):
        report.add("§4.2", "-", "row count", "MISMATCH",
                   f"paper has {len(paper_rows)} rows, artifact has {len(artifact)}")
    for cells in paper_rows:
        if len(cells) < 10:
            report.add("§4.2", "|".join(cells[:2]), "-", "MISMATCH",
                       f"row has {len(cells)} cells, expected 10")
            continue
        key = (cells[0], cells[1])
        want = artifact.get(key)
        if want is None:
            report.add("§4.2", f"{key[0]}-{key[1]}", "-", "MISMATCH",
                       "no artifact row for this dataset/horizon")
            continue
        for index in range(10):
            got_cell, want_cell = cells[index], want[index]
            if not got_cell:
                report.add("§4.2", f"{key[0]}-{key[1]}", header[index], "blank",
                           "empty paper cell")
            elif got_cell != want_cell:
                report.add("§4.2", f"{key[0]}-{key[1]}", header[index], "MISMATCH",
                           f"paper={got_cell!r} artifact={want_cell!r}")
            else:
                report.add("§4.2", f"{key[0]}-{key[1]}", header[index], "match", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO),
                        help="root holding research_runs/")
    parser.add_argument("--json", default="", help="write the report as JSON here")
    parser.add_argument("--minipaper", default=str(MINIPAPER),
                        help="override the paper path (for self-tests)")
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    minipaper = pathlib.Path(args.minipaper)

    report = Report()
    check_4_3(report, root, minipaper)
    check_4_2(report, root, minipaper)

    for state in ("match", "MISMATCH", "blank", "PENDING"):
        n = report.count(state)
        if n:
            print(f"{state}: {n}")
    mismatches = [r for r in report.rows if r["state"] == "MISMATCH"]
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for row in mismatches[:20]:
            print(f"  {row['table']} / {row['key']} / {row['column']}: {row['detail']}")
    blanks = [r for r in report.rows if r["state"] == "blank"]
    if blanks:
        examples = [f"{r['key']}:{r['column']}" for r in blanks[:5]]
        print(f"\nstill blank ({len(blanks)} cell(s)); e.g. {examples}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report.rows, indent=2))
        print(f"\nreport written to {args.json}")
    if mismatches:
        print("\nThe paper disagrees with its artifacts; fix before submission.")
        return 1
    print("\nOK: every filled section-4 cell matches its artifact at the displayed precision"
          + (" (some cells are still blank)" if blanks else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
