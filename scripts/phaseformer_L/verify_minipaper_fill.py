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
E16 = "research_runs/phaseformer_L_e16_dissection_v1"
E18 = "research_runs/phaseformer_L_e18_negative_v1"
E17 = "research_runs/phaseformer_L_e17_conditional_v1"

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


def is_separator(cells) -> bool:
    """A markdown table separator row (|---|---|) contains only dashes, colons, spaces.

    Written as a named helper because the inverted form of this test silently
    skipped EVERY data row of a table whose cells happen to contain no dash --
    which is what the 4.5 calibration's positive control caught.
    """
    return set("".join(cells)) <= set("-: ")


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
        if len(cells) >= 2 and cells[0] not in ("Dataset", "") \
                and not is_separator(cells):
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


def inventory(minipaper: pathlib.Path) -> int:
    """Count rows/columns/empty cells for every table in section 4.

    Turns "is the fill finished?" from an eyeball question into a number: after a
    fill, every table that should be complete must report zero empty cells (the
    tables that carry plan text on purpose are listed separately below).
    """
    text = minipaper.read_text(encoding="utf-8")
    marks = [("4.2", "### 4.2", "### 4.3"), ("4.3", "### 4.3", "### 4.4"),
             ("4.4", "### 4.4", "### 4.5"), ("4.5", "### 4.5", "### 4.6"),
             ("4.6", "### 4.6", "### 4.7"), ("4.7", "### 4.7", "## 5")]
    total_empty = 0
    for name, start, end in marks:
        block = section_text(text, start, end)
        raw = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("|")]
        # A table is "header line, then a |---| separator, then rows".  Without the
        # lookahead an earlier version merged adjacent tables (it only opened one
        # table per section), which is exactly the kind of quiet structural error
        # this mode exists to expose -- the TOTAL stayed right while the per-table
        # breakdown was wrong.
        is_sep = lambda cells: set("".join(cells)) <= set("-: ")
        cells_of = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
        tables = []
        for index, line in enumerate(raw):
            cells = cells_of(line)
            following = cells_of(raw[index + 1]) if index + 1 < len(raw) else None
            if following is not None and is_sep(following):
                tables.append({"cols": len(cells), "rows": []})
                continue
            # Rows belong to the most recently opened table; the header line
            # itself opened it above and is deliberately not counted as a row.
            if not tables or is_sep(cells):
                continue
            tables[-1]["rows"].append(cells)
        print(f"\n=== §{name} ===")
        for index, table in enumerate(tables, start=1):
            rows = table["rows"]
            empty = sum(1 for cells in rows for c in cells if not c)
            total_empty += empty
            print(f"  table {index}: {len(rows)} rows x {table['cols']} cols, "
                  f"empty cells = {empty}")
    print(f"\nsection 4 empty cells in total: {total_empty}")
    print("(§4.2 variant, §4.3 and §4.7's second table carry intentional text; "
          "see minipaper_fill_mapping.md section 4)")
    return total_empty


def aggregate_h1(root: pathlib.Path) -> dict:
    """setting -> (supporting seeds, total seeds) or the string "evidence_missing".

    The paper's H1 column header reads "（seed 数）", so the cell must carry a SEED
    COUNT, not the per-seed verdict.  The write-back emits the per-seed verdict
    (``true``/``false``/``evidence_missing``) in `conditional_table.md`, so the cell
    has to be aggregated from `results.with_test.csv` instead -- grouping the three
    seeds of each setting and counting the ones whose verdict is "true".  That is
    the same aggregation that produced the independently computed H1 summary
    ("ETTh2-96/720, ETTm2-96/192 = 3/3 seeds; Weather-96 = 2/3; Weather-192 = 0/3;
    Electricity-336 = evidence_missing"), which is why this is the口径 to encode.
    """
    path = root / E17 / "results.with_test.csv"
    if not path.is_file():
        return {}
    groups: dict = {}
    for row in csv.DictReader(path.open(newline="")):
        key = (str(row.get("dataset")), str(row.get("horizon")))
        verdict = str(row.get("h1_cond_gt_indep_seed_majority", "")).strip().lower()
        entry = groups.setdefault(key, {"true": 0, "total": 0, "missing": 0})
        entry["total"] += 1
        if verdict == "true":
            entry["true"] += 1
        elif verdict == "evidence_missing" or not verdict:
            entry["missing"] += 1
    out = {}
    for (dataset, horizon), entry in groups.items():
        if entry["total"] and entry["missing"] == entry["total"]:
            out[f"{dataset}-{horizon}"] = "evidence_missing"
        else:
            out[f"{dataset}-{horizon}"] = (entry["true"], entry["total"])
    return out


def check_4_5(report: Report, root: pathlib.Path, minipaper: pathlib.Path) -> None:
    """Section 4.5: six columns copy from `conditional_table.md`, H1 is aggregated."""
    md = root / E17 / "conditional_table.md"
    block = section_text(minipaper.read_text(encoding="utf-8"), "### 4.5", "### 4.6")
    header, paper_rows = parse_markdown_table(block, "冻结独立")
    if header is None:
        report.add("§4.5", "-", "-", "MISMATCH", "could not find the 4.5 table header")
        return
    if len(header) != 7:
        report.add("§4.5", "-", "columns", "MISMATCH",
                   f"header has {len(header)} columns, expected 7")
    if not md.is_file():
        report.add("§4.5", "-", "artifact", "PENDING", f"{md.name} does not exist yet")
        return

    artifact = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] not in ("Dataset", "") \
                and not is_separator(cells):
            artifact[f"{cells[0]}-{cells[1]}"] = cells
    h1 = aggregate_h1(root)

    if len(paper_rows) != len(artifact):
        report.add("§4.5", "-", "row count", "MISMATCH",
                   f"paper has {len(paper_rows)} rows, artifact has {len(artifact)}")
    for cells in paper_rows:
        if len(cells) < 7:
            report.add("§4.5", "|".join(cells[:2]), "-", "MISMATCH",
                       f"row has {len(cells)} cells, expected 7")
            continue
        key = f"{cells[0]}-{cells[1]}"
        want = artifact.get(key)
        if want is None:
            report.add("§4.5", key, "-", "MISMATCH", "no artifact row for this setting")
            continue
        for index in range(6):                      # six direct-copy columns
            got, expect = cells[index], want[index]
            if not got:
                report.add("§4.5", key, header[index], "blank", "empty paper cell")
            elif got != expect:
                report.add("§4.5", key, header[index], "MISMATCH",
                           f"paper={got!r} artifact={expect!r}")
            else:
                report.add("§4.5", key, header[index], "match", "")
        # seventh column: seed count, compared tolerantly on the numbers so the
        # exact suffix ("3/3" vs "3/3 seed") does not raise a false alarm.
        cell = cells[6]
        expected = h1.get(key)
        if not cell:
            report.add("§4.5", key, header[6], "blank", "empty paper cell")
        elif expected is None:
            report.add("§4.5", key, header[6], "PENDING",
                       "no H1 verdicts in results.with_test.csv for this setting")
        elif expected == "evidence_missing":
            ok = "evidence_missing" in cell.lower()
            report.add("§4.5", key, header[6], "match" if ok else "MISMATCH",
                       cell if ok else f"paper={cell!r} expected evidence_missing")
        else:
            match = re.search(r"(\d+)\s*/\s*(\d+)", cell)
            ok = match is not None and (int(match.group(1)), int(match.group(2))) == expected
            report.add("§4.5", key, header[6], "match" if ok else "MISMATCH",
                       cell if ok else f"paper={cell!r} expected {expected[0]}/{expected[1]}")


def check_4_4_intervention(report: Report, root: pathlib.Path,
                           minipaper: pathlib.Path) -> None:
    """Section 4.4's intervention table: the artifact row minus its first field.

    `e16_writeback` emits 11 fields per row -- `model | dataset | H | q/r | ...seven
    values` -- while the paper's table has 10 columns and NO separate model column,
    because the model identity is already inside the `q/r` label
    (`dense（r=96）` / `q=1/4（r=24）` / `q=1/8（r=12）`).  So the mapping is

        artifact_row[1:] == paper_row[0:]

    and the check is a plain field comparison of the ten values.
    """
    md = root / E16 / "intervention_table_44.md"
    block = section_text(minipaper.read_text(encoding="utf-8"), "### 4.4", "### 4.5")
    header, paper_rows = parse_markdown_table(block, "随机 RRR 子空间 drop")
    if header is None:
        report.add("§4.4 干预表", "-", "-", "MISMATCH", "could not find the table header")
        return
    if len(header) != 10:
        report.add("§4.4 干预表", "-", "columns", "MISMATCH",
                   f"header has {len(header)} columns, expected 10")
    if not md.is_file():
        report.add("§4.4 干预表", "-", "artifact", "PENDING", f"{md.name} does not exist yet")
        return

    artifact = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4 and cells[1] not in ("Dataset", "") \
                and not is_separator(cells):
            artifact[(cells[1], cells[2], cells[3])] = cells

    if len(paper_rows) != len(artifact):
        report.add("§4.4 干预表", "-", "row count", "MISMATCH",
                   f"paper has {len(paper_rows)} rows, artifact has {len(artifact)}")
    for cells in paper_rows:
        if len(cells) < 10:
            report.add("§4.4 干预表", "|".join(cells[:2]), "-", "MISMATCH",
                       f"row has {len(cells)} cells, expected 10")
            continue
        key = (cells[0], cells[1], cells[2])
        want = artifact.get(key)
        label = f"{key[0]}-{key[1]}-{key[2]}"
        if want is None:
            report.add("§4.4 干预表", label, "-", "MISMATCH",
                       "no artifact row for this dataset/horizon/q-r")
            continue
        for index in range(10):
            got, expect = cells[index], want[index + 1]
            if not got:
                report.add("§4.4 干预表", label, header[index], "blank", "empty paper cell")
            elif got != expect:
                report.add("§4.4 干预表", label, header[index], "MISMATCH",
                           f"paper={got!r} artifact={expect!r}")
            else:
                report.add("§4.4 干预表", label, header[index], "match", "")


def check_4_6(report: Report, root: pathlib.Path, minipaper: pathlib.Path) -> None:
    """Section 4.6: only the fifth column is filled; columns 1-4 keep the paper's text.

    This is narrower than it first appeared.  `e18_writeback` emits five fields per
    row, and comparing them with the paper shows the two agree **verbatim for rows
    2, 3 and 4** (the write-back's `KEPT_AS_DASH` constants are literally the paper's
    own strings) but differ for rows 1 and 5:

        row 1  操作: paper "输入平滑（boxcar / causal EMA，各 5 档）"
                     artifact "输入平滑（causal EMA 两个强度）"
        row 5  作用对象: paper "支路容量（网格之外）"   artifact "支路容量（低秩网格之外）"
        row 5  口径:     paper "6 setting × 3 seed"     artifact "6 setting × 3 seed × 2 rank"

    The paper's wording describes the PRIOR experiments that columns 3-4 report, so
    replacing the whole row would rewrite accurate prose.  The fill is therefore the
    fifth column (本文补做), which takes the artifact's `addendum`; the scale text it
    displaces belongs in the table note.
    """
    md = root / E18 / "negative_table.md"
    block = section_text(minipaper.read_text(encoding="utf-8"), "### 4.6", "### 4.7")
    header, paper_rows = parse_markdown_table(block, "本文补做")
    if header is None:
        report.add("§4.6", "-", "-", "MISMATCH", "could not find the 4.6 table header")
        return
    if len(header) != 5:
        report.add("§4.6", "-", "columns", "MISMATCH",
                   f"header has {len(header)} columns, expected 5")
    if not md.is_file():
        report.add("§4.6", "-", "artifact", "PENDING", f"{md.name} does not exist yet")
        return

    artifact = []
    for line in md.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and not is_separator(cells) and cells[0] != "操作":
            artifact.append(cells)
    if len(paper_rows) != len(artifact):
        report.add("§4.6", "-", "row count", "MISMATCH",
                   f"paper has {len(paper_rows)} rows, artifact has {len(artifact)}")
        return
    for index, (cells, want) in enumerate(zip(paper_rows, artifact), start=1):
        label = f"row {index}"
        if len(cells) < 5:
            report.add("§4.6", label, "-", "MISMATCH", f"row has {len(cells)} cells")
            continue
        got = cells[4]
        if not got:
            report.add("§4.6", label, header[4], "blank", "empty paper cell")
        elif got != want[4]:
            report.add("§4.6", label, header[4], "MISMATCH",
                       f"paper={got!r} artifact={want[4]!r}")
        else:
            report.add("§4.6", label, header[4], "match", "")
        # cross-check the descriptive columns that ARE expected to agree verbatim
        for column in range(4):
            if cells[column] == want[column]:
                report.add("§4.6", label, header[column], "match", "")
            else:
                report.add("§4.6", label, header[column], "INFO",
                           f"paper={cells[column]!r} artifact={want[column]!r} "
                           f"(allowed to differ; see the docstring)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO),
                        help="root holding research_runs/")
    parser.add_argument("--json", default="", help="write the report as JSON here")
    parser.add_argument("--minipaper", default=str(MINIPAPER),
                        help="override the paper path (for self-tests)")
    parser.add_argument("--inventory", action="store_true",
                        help="just count rows/columns/empty cells per table")
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    minipaper = pathlib.Path(args.minipaper)

    if args.inventory:
        inventory(minipaper)
        return 0

    report = Report()
    check_4_3(report, root, minipaper)
    check_4_2(report, root, minipaper)
    check_4_5(report, root, minipaper)
    check_4_4_intervention(report, root, minipaper)
    check_4_6(report, root, minipaper)

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
