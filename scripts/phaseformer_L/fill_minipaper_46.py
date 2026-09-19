#!/usr/bin/env python3
"""Fill minipaper section 4.6's fifth column ("本文补做") from negative_table.md.

Why only the fifth column, and why positionally:

* ``e18_writeback`` emits five fields per row, and the first four agree VERBATIM with
  the paper for rows 2-4 but differ for rows 1 and 5 -- the paper's wording describes
  the prior experiments those columns report, so copying the artifact over them would
  replace accurate prose with a different (also accurate, but not the paper's)
  phrasing.  Only the fifth column, the new work, is filled.
* The match is POSITIONAL, not keyed: row 1's first cell differs between the paper and
  the artifact ("输入平滑（boxcar / causal EMA，各 5 档）" vs "输入平滑（causal EMA 两个
  强度）"), so there is no shared key to match on.  The row count is therefore checked
  explicitly, and a mismatch refuses the whole write.

Dry-run by default; ``--write`` applies.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

COLUMN = 4


def cells_of(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(cells: list) -> bool:
    return set("".join(cells)) <= set("-: ")


def locate_table(lines: list, start: str, end: str, needle: str):
    begin = next((i for i, ln in enumerate(lines) if ln.strip().startswith(start)), None)
    if begin is None:
        raise SystemExit(f"section start {start!r} not found")
    stop = len(lines)
    for i in range(begin + 1, len(lines)):
        if lines[i].strip().startswith(end):
            stop = i
            break
    for i in range(begin, stop):
        stripped = lines[i].strip()
        if not stripped.startswith("|") or needle not in stripped:
            continue
        if i + 1 < len(lines) and is_separator(cells_of(lines[i + 1].strip())):
            first = i + 2
            last = first
            while last < len(lines) and lines[last].strip().startswith("|"):
                last += 1
            return i, first, last
    raise SystemExit(f"no table with header needle {needle!r} in {start}..{end}")


def parse_artifact(text: str) -> list:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = cells_of(stripped)
        if len(cells) == 5 and not is_separator(cells) and cells[0] != "操作":
            rows.append(cells)
    return rows


def self_test() -> int:
    paper = [
        "### 4.6 x",
        "",
        "| 操作 | 对照 | 口径 | 既有结果 | 本文补做 |",
        "|---|---|---|---|---|",
        "| 输入平滑（boxcar / causal EMA，各 5 档） | a | b | c |  |",
        "| 同一操作 | a | b | c |  |",
        "",
        "### 4.7 y",
    ]
    artifact = "\n".join([
        "| 操作 | 对照 | 口径 | 既有结果 | 本文补做 |",
        "|---|---|---|---|---|",
        "| 输入平滑（causal EMA 两个强度） | a | b | c | NEW-1 |",
        "| 同一操作 | a | b | c | NEW-2 |",
    ])
    want = parse_artifact(artifact)
    lines = list(paper)
    _, first, last = locate_table(lines, "### 4.6", "### 4.7", "本文补做")
    for offset in range(last - first):
        cells = cells_of(lines[first + offset].strip())
        cells[COLUMN] = want[offset][COLUMN]
        lines[first + offset] = "| " + " | ".join(cells) + " |"

    checks = [
        ("artifact parsed two rows", len(want) == 2),
        ("fifth column filled positionally", "NEW-1" in lines[4] and "NEW-2" in lines[5]),
        ("rows not swapped", lines[4].index("NEW-1") > 0 and "NEW-2" not in lines[4]),
        ("column 1 keeps the PAPER's wording (not the artifact's)",
         lines[4].startswith("| 输入平滑（boxcar / causal EMA，各 5 档）")),
        ("section 4.7 untouched", lines[7] == "### 4.7 y"),
    ]
    ok = True
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
        ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipaper")
    ap.add_argument("--artifact")
    ap.add_argument("--start", default="### 4.6")
    ap.add_argument("--end", default="### 4.7")
    ap.add_argument("--header-needle", default="本文补做")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.minipaper or not a.artifact:
        ap.error("--minipaper and --artifact are required (or use --self-test)")

    artifact_path = pathlib.Path(a.artifact)
    if not artifact_path.is_file():
        print(f"artifact {artifact_path} does not exist yet; nothing to fill", file=sys.stderr)
        return 2
    want = parse_artifact(artifact_path.read_text(encoding="utf-8"))

    paper = pathlib.Path(a.minipaper)
    lines = paper.read_text(encoding="utf-8").splitlines()
    _, first, last = locate_table(lines, a.start, a.end, a.header_needle)

    if len(want) != last - first:
        print(f"artifact has {len(want)} rows but the paper's table has {last - first}; "
              f"refusing a positional fill when the counts differ", file=sys.stderr)
        return 2

    out = list(lines)
    for offset in range(last - first):
        cells = cells_of(out[first + offset].strip())
        if len(cells) != 5:
            print(f"row {offset + 1} has {len(cells)} cells, expected 5", file=sys.stderr)
            return 2
        cells[COLUMN] = want[offset][COLUMN]
        out[first + offset] = "| " + " | ".join(cells) + " |"
        print(f"  row {offset + 1}: {cells[COLUMN]}")

    print(f"artifact rows: {len(want)}   paper rows: {last - first}")
    if not a.write:
        print("dry run; pass --write to apply")
        return 0
    paper.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
