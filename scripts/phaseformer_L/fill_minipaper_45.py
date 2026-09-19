#!/usr/bin/env python3
"""Fill minipaper section 4.5 from conditional_table.md plus an aggregated H1 column.

Six of the seven columns are a direct copy from `conditional_table.md`.  The seventh
is NOT: the paper's header reads "H1: cond distance < indep distance （seed 数）", so
the cell must be a SEED COUNT, while the artifact emits a per-seed verdict
(true/false/evidence_missing).  Pasting the artifact's cell would put `true` under a
header asking for a count.

The aggregation is imported from the verifier (``aggregate_h1``) rather than
re-implemented here.  That is deliberate: the filler and the verifier must agree on
this number, and two copies of the grouping rule would be free to drift.  If the
import fails, this tool refuses to run instead of guessing.

Keyed by ``dataset-horizon``; the paper's row order is preserved.  Dry-run by
default; ``--write`` applies.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

H1_COLUMN = 6


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


def render_h1(value) -> str:
    """(supporting, total) -> "3/3 seed"; the string "evidence_missing" passes through."""
    if value == "evidence_missing":
        return "evidence_missing"
    if isinstance(value, tuple):
        return f"{value[0]}/{value[1]} seed"
    raise SystemExit(f"unexpected H1 aggregate {value!r}")


def import_h1(repo: str):
    sys.path.insert(0, repo)
    try:
        from scripts.phaseformer_L.verify_minipaper_fill import aggregate_h1
    except Exception as exc:                     # pragma: no cover - env dependent
        raise SystemExit(f"could not import the verifier's aggregate_h1: {exc!r}; refusing "
                         "to aggregate with a second, possibly divergent implementation")
    return aggregate_h1


def self_test() -> int:
    # Running this file directly puts its own directory on sys.path, not the repo
    # root, so the verifier import below needs the root added explicitly.
    repo = str(pathlib.Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.phaseformer_L.verify_minipaper_fill import parse_markdown_table

    paper = [
        "### 4.5 x",
        "",
        "| Dataset | H | direct | 冻结独立 | 冻结条件 | joint | H1 |",
        "|---|---:|---|---|---|---|---|",
        "| ETTh2 | 96 | 0.1 | 0.2 | 0.3 | 0.4 |  |",
        "| Weather | 192 | 0.5 | 0.6 | 0.7 | 0.8 |  |",
        "",
        "### 4.6 y",
    ]
    artifact = {
        "ETTh2-96": ["ETTh2", "96", "0.1", "0.2", "0.3", "0.4", "true"],
        "Weather-192": ["Weather", "192", "0.5", "0.6", "0.7", "0.8", "false"],
    }
    h1 = {"ETTh2-96": (3, 3), "Weather-192": "evidence_missing"}

    lines = list(paper)
    _, first, last = locate_table(lines, "### 4.5", "### 4.6", "冻结独立")
    for index in range(first, last):
        cells = cells_of(lines[index].strip())
        want = artifact[f"{cells[0]}-{cells[1]}"]
        cells[:6] = want[:6]
        cells[H1_COLUMN] = render_h1(h1[f"{cells[0]}-{cells[1]}"])
        lines[index] = "| " + " | ".join(cells) + " |"

    # The verifier's own parser must accept the result and see a matching seed count.
    header, rows = parse_markdown_table("\n".join(lines[0:first + 3]), "冻结独立")
    import re
    match = re.search(r"(\d+)\s*/\s*(\d+)", rows[0][H1_COLUMN])
    checks = [
        ("header parsed with 7 columns", header is not None and len(header) == 7),
        ("two rows filled", last - first == 2),
        ("six copied columns took the artifact's text", rows[0][2] == "0.1"),
        ("H1 rendered as a seed count", rows[0][H1_COLUMN] == "3/3 seed"),
        ("the verifier's regex reads that count", match and (int(match.group(1)), int(match.group(2))) == (3, 3)),
        ("evidence_missing passes through", rows[1][H1_COLUMN] == "evidence_missing"),
        ("verifier accepts evidence_missing cell",
         "evidence_missing" in rows[1][H1_COLUMN].lower()),
        ("section 4.6 untouched", lines[7] == "### 4.6 y"),
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
    ap.add_argument("--repo", default="")
    ap.add_argument("--e17-root")
    ap.add_argument("--start", default="### 4.5")
    ap.add_argument("--end", default="### 4.6")
    ap.add_argument("--header-needle", default="冻结独立")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.minipaper or not a.artifact or not a.e17_root:
        ap.error("--minipaper, --artifact and --e17-root are required (or --self-test)")

    artifact_path = pathlib.Path(a.artifact)
    if not artifact_path.is_file():
        print(f"artifact {artifact_path} does not exist yet; nothing to fill", file=sys.stderr)
        return 2
    aggregate_h1 = import_h1(a.repo or str(pathlib.Path(__file__).resolve().parents[2]))
    h1 = aggregate_h1(pathlib.Path(a.e17_root))
    if not h1:
        print("no H1 verdicts in results.with_test.csv yet; refusing to write blanks",
              file=sys.stderr)
        return 2

    artifact = {}
    for line in artifact_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = cells_of(stripped)
        if is_separator(cells) or len(cells) < 7 or cells[0] == "Dataset":
            continue
        artifact[f"{cells[0]}-{cells[1]}"] = cells

    paper = pathlib.Path(a.minipaper)
    lines = paper.read_text(encoding="utf-8").splitlines()
    _, first, last = locate_table(lines, a.start, a.end, a.header_needle)

    out = list(lines)
    filled, problems = 0, []
    for index in range(first, last):
        cells = cells_of(out[index].strip())
        key = f"{cells[0]}-{cells[1]}"
        want = artifact.get(key)
        if want is None:
            problems.append((key, "no artifact row"))
            continue
        if key not in h1:
            problems.append((key, "no H1 verdicts in results.with_test.csv"))
            continue
        cells[:6] = want[:6]
        cells[H1_COLUMN] = render_h1(h1[key])
        out[index] = "| " + " | ".join(cells) + " |"
        filled += 1
        print(f"  {key}: " + " | ".join(cells[2:]))

    print(f"artifact rows: {len(artifact)}")
    print(f"paper table rows: {last - first} (lines {first + 1}..{last})")
    print(f"filled: {filled}   problems: {len(problems)}")
    for key, why in problems:
        print(f"  PROBLEM {key}: {why}")
    if problems:
        print("refusing to write a partial table", file=sys.stderr)
        return 2
    if not a.write:
        print("dry run; pass --write to apply")
        return 0
    paper.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
