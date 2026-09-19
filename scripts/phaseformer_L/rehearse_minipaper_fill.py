#!/usr/bin/env python3
"""Rehearse the section 4.4 dissection checker of `verify_minipaper_fill.py`.

Why this exists
---------------
That checker read four of its columns under the **raw** `dissection_table.csv`
names (`leading_input_group_label`, `mean_input_group_explanation`,
`mean_output_group_explanation`, `leading_correction_energy_share`) while
`e16_writeback.build_dissection` *renames* them when it aggregates into
`dissection_table_44.csv` (`input_group_label`, `input_group_explanation`,
`output_group_explanation`, `correction_energy_share`).  Every lookup therefore
returned None -- and because a check whose artifact side is empty was skipped,
the cells were reported as **match** without ever being compared.  Measured on
2026-09-20, before the fix: 3 of the 5 composite columns (63 of the 105 cells)
were either silently passing or PENDING.

This rehearsal pins the contract down both ways and, just as usefully, pins the
**fill format** for section 4.4's dissection table: it fills a temporary copy of
the paper from a synthetic 44-table and asserts the checker's verdicts.

The synthetic 44-table is built from the column names the producer actually
writes, read out of `e16_writeback.build_dissection` by AST -- so the fixture
cannot drift from the producer.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/rehearse_minipaper_fill.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
VERIFIER = REPO / "scripts" / "phaseformer_L" / "verify_minipaper_fill.py"
WRITEBACK = REPO / "scripts" / "phaseformer_L" / "e16_writeback.py"
MINIPAPER = REPO / "docs" / "PhaseFormer_L_minipaper.md"

MODELS = ("PhaseFormer-L", "L-q1/4", "L-q1/8")
SETTINGS = (("ETTh2", 96), ("ETTh2", 720), ("ETTm2", 96), ("ETTm2", 192),
            ("Weather", 96), ("Weather", 192), ("Electricity", 336))

#: label / rate / share / in / out, per (model, setting) -- fixed values so the
#: paper cells this rehearsal writes can be compared exactly.
VALUES = {
    "input_group_label": "recent level",
    "input_group_explanation": 0.75,
    "output_group_label": "trend",
    "output_group_explanation": 0.85,
    "correction_energy_share": 0.42,
    "leading4_input_overlap": 0.91,
    "leading4_output_overlap": 0.88,
    "stable_semantics_verdict": True,
}


def writer_columns() -> list:
    """The 44-table's columns, from the producer's own code.

    `build_dissection` writes dict-literal keys, assigns a few through a loop
    variable (`entry[target] = ...`), and computes the renamed ones -- all three
    shapes are collected here so the fixture matches what the writer emits.
    """
    source = WRITEBACK.read_text()
    tree = ast.parse(source)
    keys: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_dissection":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for key in sub.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
                if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
                    target = sub.slice
                    if isinstance(target, ast.Constant) and isinstance(target.value, str):
                        keys.add(target.value)
    loop = re.search(r"for column, target in \((.*?)\):\n", source, re.S).group(1)
    for _column, target in re.findall(r'\("([a-z_0-9]+)",\s*\n?\s*"([a-z_0-9]+)"\)', loop):
        keys.add(target)
    return sorted(keys)


def build_artifact(columns: list, rename: str = "") -> list:
    rows = []
    for model in MODELS:
        for dataset, horizon in SETTINGS:
            row = {name: "" for name in columns}
            row.update({"model": model, "dataset": dataset, "horizon": horizon})
            for name, value in VALUES.items():
                if name in row:
                    row[name] = value
            if rename and rename in row:
                row[rename + "_raw"] = row.pop(rename)   # simulate a rename
            rows.append(row)
    return rows


def fill_paper(text: str, rows: list) -> str:
    """Fill section 4.4's dissection table in the prescribed composite format."""
    lines = text.splitlines()
    # Scope to section 4.4 exactly as the checker does: the phrase 主模式输入组 also
    # occurs in earlier sections, and matching it globally made an earlier version
    # of this rehearsal fill section 4.2's variant table instead.
    start44 = next(i for i, l in enumerate(lines) if l.startswith("### 4.4"))
    end44 = next(i for i, l in enumerate(lines) if l.startswith("### 4.5"))
    header_index = next(
        i for i in range(start44, end44)
        if lines[i].lstrip().startswith("|") and "主模式输入组" in lines[i]
        and lines[i].count("|") == 9)          # 8 columns -> 9 pipes
    # rows follow the header and its separator until the first non-table line
    start = header_index + 2
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    by_key = {(r["model"], r["dataset"], str(r["horizon"])): r for r in rows}
    filled = []
    for row in rows:
        filled.append(
            "| {model} | {dataset} | {horizon} | {ilabel} / {irate:.2f} | "
            "{olabel} / {orate:.2f} | {share:.3f} | {idin:.2f} / {odout:.2f} | {verdict} |"
            .format(model=row["model"], dataset=row["dataset"], horizon=row["horizon"],
                    ilabel=row["input_group_label"], irate=row["input_group_explanation"],
                    olabel=row["output_group_label"], orate=row["output_group_explanation"],
                    share=row["correction_energy_share"],
                    idin=row["leading4_input_overlap"],
                    odout=row["leading4_output_overlap"],
                    verdict="✓" if row["stable_semantics_verdict"] else "✗"))
    assert len(by_key) == len(filled) == 21, (len(by_key), len(filled))
    return "\n".join(lines[:start] + filled + lines[end:]) + "\n"


def run(root: pathlib.Path, minipaper: pathlib.Path) -> dict:
    proc = subprocess.run([sys.executable, str(VERIFIER), "--root", str(root),
                           "--minipaper", str(minipaper)],
                          capture_output=True, text=True, cwd=str(REPO))
    counts = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(match|MISMATCH|blank|PENDING): (\d+)$", line.strip())
        if m:
            counts[m.group(1)] = int(m.group(2))
    return {"rc": proc.returncode, "stdout": proc.stdout, "counts": counts}


def dissection_verdicts(stdout: str) -> dict:
    verdicts = {}
    for line in stdout.splitlines():
        if "主模式输入组" in line or "修正能量份额" in line or "稳定语义判定" in line:
            verdicts.setdefault("rows", 0)
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    columns = writer_columns()
    print("columns extracted from e16_writeback.build_dissection: %d" % len(columns))
    for name in ("input_group_label", "input_group_explanation", "output_group_label",
                 "output_group_explanation", "correction_energy_share",
                 "leading4_input_overlap", "leading4_output_overlap",
                 "stable_semantics_verdict"):
        mark = "OK  " if name in columns else "MISS"
        print(f"  [{mark}] {name}")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = pathlib.Path(tmp)
        out_dir = scratch / "research_runs" / "phaseformer_L_e16_dissection_v1"
        out_dir.mkdir(parents=True)
        artifact = out_dir / "dissection_table_44.csv"
        paper_text = MINIPAPER.read_text(encoding="utf-8")

        def write_artifact(rows: list) -> None:
            fields = sorted({k for row in rows for k in row})
            with artifact.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        # 1. healthy: correct names, matching values -> no MISMATCH anywhere
        rows = build_artifact(columns)
        write_artifact(rows)
        paper = scratch / "paper_ok.md"
        paper.write_text(fill_paper(paper_text, rows), encoding="utf-8")
        got = run(scratch, paper)
        ok = got["counts"].get("MISMATCH", 0) == 0
        results.append(ok)
        print(f"\n[{'OK  ' if ok else 'FAIL'}] filled paper + correctly named artifact: "
              f"counts={got['counts']} rc={got['rc']}")
        matched = got["counts"].get("match", 0)
        ok = matched >= 105
        results.append(ok)
        print(f"[{'OK  ' if ok else 'FAIL'}] at least 105 §4.4 cells compared: "
              f"match={matched}")

        # 2. one rate perturbed in the paper -> MISMATCH
        paper2 = scratch / "paper_bad.md"
        text = fill_paper(paper_text, rows).replace("recent level / 0.75",
                                                    "recent level / 0.11", 1)
        paper2.write_text(text, encoding="utf-8")
        got = run(scratch, paper2)
        ok = got["counts"].get("MISMATCH", 0) >= 1
        results.append(ok)
        print(f"[{'OK  ' if ok else 'FAIL'}] a perturbed rate is reported: "
              f"MISMATCH={got['counts'].get('MISMATCH', 0)}")

        # 3. regression control for the defect: the artifact carries the RAW names
        raw = build_artifact(columns, rename="input_group_explanation")
        write_artifact(raw)
        paper3 = scratch / "paper_raw.md"
        paper3.write_text(fill_paper(paper_text, rows), encoding="utf-8")
        got = run(scratch, paper3)
        text = got["stdout"]
        loud = "lacks" in text and "input_group_explanation" in text
        results.append(loud)
        print(f"[{'OK  ' if loud else 'FAIL'}] an artifact with the raw column name is "
              f"reported as a column MISMATCH, not a silent pass")
        detail = [l for l in text.splitlines() if "lacks" in l]
        if detail:
            print("      " + detail[0].strip()[:150])

    print()
    if all(results):
        print(f"OK: {len(results)} assertion(s) held")
        return 0
    print(f"FAIL: {results.count(False)} of {len(results)} assertion(s) did not hold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
