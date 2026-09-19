#!/usr/bin/env python3
"""Fill a minipaper table from a producer's markdown artifact, keyed by cell columns.

Built for section 4.2's main table (28 rows x 10 cols from ``main_table.md``), but
the mechanics are generic: locate the table in the paper whose header contains a
needle, then replace each row's cells with the artifact row carrying the same key
(the first ``--key-columns`` cells).

Why a keyed replacement instead of pasting the artifact block wholesale:

* ``verify_minipaper_fill.check_4_2`` matches rows by ``(dataset, horizon)``, so
  only the cells matter -- row order is cosmetic.  Preserving the PAPER's order
  keeps the diff small and reviewable, which matters because a 192-cell fill is
  exactly where a silent row shift would otherwise hide.
* The replacement is verified independently afterwards by
  ``verify_minipaper_fill.py``, so this tool does not need to be trusted: a row
  shift, a wrong column mapping, or a stale number all surface as MISMATCH.

Dry-run by default; ``--write`` applies the change in place.
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def cells_of(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(cells: list) -> bool:
    return set("".join(cells)) <= set("-: ")


def parse_artifact(text: str, key_columns: int) -> dict:
    """Map key tuple -> the artifact's own cell list (separators/headers excluded)."""
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = cells_of(stripped)
        if is_separator(cells) or len(cells) < key_columns:
            continue
        if cells[0] in ("Dataset", "统计量", ""):
            continue
        out[tuple(cells[:key_columns])] = cells
    return out


def locate_table(lines: list, start: str, end: str, needle: str):
    """Return (header_index, first_row_index, last_row_index) for the matching table."""
    try:
        begin = next(i for i, ln in enumerate(lines) if ln.strip().startswith(start))
    except StopIteration:
        raise SystemExit(f"section start {start!r} not found")
    stop = len(lines)
    for i in range(begin + 1, len(lines)):
        if lines[i].strip().startswith(end):
            stop = i
            break

    header = None
    for i in range(begin, stop):
        stripped = lines[i].strip()
        if not stripped.startswith("|"):
            continue
        cells = cells_of(stripped)
        following = cells_of(lines[i + 1].strip()) if i + 1 < len(lines) else []
        if following and is_separator(following) and needle in stripped:
            header = i
            break
    if header is None:
        raise SystemExit(f"no table with header needle {needle!r} in {start}..{end}")

    first = header + 2  # skip header and separator
    last = first
    while last < len(lines) and lines[last].strip().startswith("|"):
        last += 1
    return header, first, last


def fill(lines: list, artifact: dict, start: str, end: str, needle: str,
         key_columns: int) -> tuple:
    header, first, last = locate_table(lines, start, end, needle)
    out = list(lines)
    filled, missing = 0, []
    for i in range(first, last):
        cells = cells_of(out[i].strip())
        if is_separator(cells):
            continue
        key = tuple(cells[:key_columns])
        want = artifact.get(key)
        if want is None:
            missing.append(key)
            continue
        if out[i].strip().endswith("|"):
            indent = out[i][:len(out[i]) - len(out[i].lstrip())]
            out[i] = f"{indent}| " + " | ".join(want) + " |"
            filled += 1
    return out, filled, missing, (first, last)


def self_test() -> int:
    paper = [
        "### 4.2 x",
        "",
        "| Dataset | H | Golden MSE/MAE | A |",
        "|---|---|---|---|",
        "| ETTh1 | 96 | 0.359/0.382 |  |",
        "| ETTh2 | 96 | 0.275/0.338 |  |",
        "",
        "### 4.3 y",
    ]
    artifact_text = "\n".join([
        "| Dataset | H | Golden MSE/MAE | A |",
        "|---|---|---|---|",
        "| ETTh2 | 96 | 0.275/0.338 | 0.111 |",
        "| ETTh1 | 96 | 0.359/0.382 | 0.222 |",
    ])
    art = parse_artifact(artifact_text, 2)
    out, filled, missing, _ = fill(paper, art, "### 4.2", "### 4.3", "Golden MSE/MAE", 2)
    ok = True
    checks = [
        ("artifact parsed both rows", len(art) == 2),
        ("both rows filled", filled == 2),
        ("nothing missing", missing == []),
        ("PAPER row order preserved (ETTh1 first)", out[4].startswith("| ETTh1")),
        ("ETTh1 got its own value 0.222", "0.222" in out[4]),
        ("ETTh2 kept its own value 0.111", "0.111" in out[5]),
        ("section 4.3 untouched", out[7] == "### 4.3 y"),
    ]
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
        ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipaper")
    ap.add_argument("--artifact")
    ap.add_argument("--start", default="### 4.2")
    ap.add_argument("--end", default="### 4.3")
    ap.add_argument("--header-needle", default="Golden MSE/MAE")
    ap.add_argument("--key-columns", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.minipaper or not a.artifact:
        ap.error("--minipaper and --artifact are required (or use --self-test)")

    paper = pathlib.Path(a.minipaper)
    art_path = pathlib.Path(a.artifact)
    if not art_path.is_file():
        print(f"artifact {art_path} does not exist yet; nothing to fill", file=sys.stderr)
        return 2

    artifact = parse_artifact(art_path.read_text(encoding="utf-8"), a.key_columns)
    lines = paper.read_text(encoding="utf-8").splitlines()
    out, filled, missing, (first, last) = fill(
        lines, artifact, a.start, a.end, a.header_needle, a.key_columns)

    print(f"artifact rows: {len(artifact)}")
    print(f"paper table rows: {last - first} (lines {first + 1}..{last})")
    print(f"filled: {filled}   missing from artifact: {len(missing)}")
    for key in missing:
        print(f"  MISSING {'|'.join(key)}")
    emptied = []
    for i in range(first, last):
        cells = cells_of(out[i].strip())
        if not is_separator(cells) and any(c == "" for c in cells[2:]):
            emptied.append("|".join(cells[:2]))
    print(f"rows still carrying empty data cells afterwards: {len(emptied)}")
    for key in emptied[:10]:
        print(f"  still-empty {key}")

    if not a.write:
        print("dry run; pass --write to apply")
        return 0
    paper.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
