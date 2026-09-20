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


def parse_artifact(text: str, key_columns: int, skip: int = 0,
                   strip_paren: bool = False) -> dict:
    """Map key tuple -> the artifact's own cell list (separators/headers excluded).

    ``skip`` drops that many leading artifact cells from the value AND from the key,
    which is how section 4.4's intervention table is mapped: the artifact carries a
    leading ``model`` field the paper does not have, so

        artifact_row[1:] == paper_row[0:]     (see check_4_4_intervention)

    and the key becomes (dataset, H, q/r) on both sides.  Keying by those three
    cells rather than by position is what keeps the fill safe: the `q/r` label is
    what identifies the model, so a row shift would be caught rather than silently
    renaming a model.
    """
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = cells_of(stripped)
        if is_separator(cells) or len(cells) < key_columns + skip:
            continue
        if cells[0] in ("Dataset", "模型", "统计量", "") or "Dataset" in cells[:skip + 1]:
            continue
        out[normalize_key(cells[skip:skip + key_columns], strip_paren)] = cells[skip:]
    return out


def normalize_key(cells: list, strip_paren: bool) -> tuple:
    """Key components, optionally cut at the first bracket.

    Section 4.4's paper table pre-fills the dense rows' q/r cell as the generic
    ``dense（r=H）`` while the artifact writes the concrete ``dense（r=96）``.  Both
    sides therefore have to be keyed on the part before the bracket, or those rows
    never match and the fill leaves them empty without saying so.
    """
    if not strip_paren:
        return tuple(cells)
    out = []
    for cell in cells:
        cut = min([i for i in (cell.find("（"), cell.find("(")) if i >= 0] or [len(cell)])
        out.append(cell[:cut].strip())
    return tuple(out)


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
         key_columns: int, strip_paren: bool = False, marker: str = "") -> tuple:
    header, first, last = locate_table(lines, start, end, needle)
    out = list(lines)
    filled, missing = 0, []
    for i in range(first, last):
        cells = cells_of(out[i].strip())
        if is_separator(cells):
            continue
        key = normalize_key(cells[:key_columns], strip_paren)
        want = artifact.get(key)
        if want is None:
            missing.append(key)
            if marker:
                # An experiment that was never run (or was stopped) must say so IN THE
                # TABLE.  Leaving the cells empty makes them indistinguishable from
                # "forgot to fill", and a marker keeps the registered row visible.
                indent = out[i][:len(out[i]) - len(out[i].lstrip())]
                out[i] = f"{indent}| " + " | ".join(cells[:key_columns]
                                                    + [marker] * (len(cells) - key_columns)) + " |"
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

    # Second case: section 4.4's intervention mapping, where the artifact carries a
    # leading `model` column the paper lacks.  Both rows below have the SAME
    # (dataset, H) but different models, so a tool that ignored the third key cell
    # would put the wrong model's numbers in a row while still looking complete.
    paper44 = [
        "### 4.4 x",
        "",
        "| Dataset | H | q/r | branch | fused |",
        "|---|---|---|---|---|",
        "| ETTh2 | 96 | dense（r=96） |  |  |",
        "| ETTh2 | 96 | q=1/8（r=12） |  |  |",
        "",
        "### 4.5 y",
    ]
    artifact44 = "\n".join([
        "| 模型 | Dataset | H | q/r | branch | fused |",
        "|---|---|---|---|---|---|",
        "| L-q1/8 | ETTh2 | 96 | q=1/8（r=12） | 0.1111 | 0.2222 |",
        "| PhaseFormer-L | ETTh2 | 96 | dense（r=96） | 0.3333 | 0.4444 |",
    ])
    art44 = parse_artifact(artifact44, 3, skip=1)
    out44, filled44, missing44, _ = fill(paper44, art44, "### 4.4", "### 4.5",
                                         "q/r", 3)
    # Full-row replacement is required here: the paper's dense row says `r=H` while
    # the artifact says `r=96`, so a fill that only wrote the empty cells would
    # leave that row disagreeing with its artifact.
    dense_row = [ln for ln in out44 if ln.startswith("| ETTh2 | 96 | dense")][0]
    low_row = [ln for ln in out44 if "q=1/8" in ln][0]

    ok = True
    checks = [
        ("artifact parsed both rows", len(art) == 2),
        ("both rows filled", filled == 2),
        ("nothing missing", missing == []),
        ("PAPER row order preserved (ETTh1 first)", out[4].startswith("| ETTh1")),
        ("ETTh1 got its own value 0.222", "0.222" in out[4]),
        ("ETTh2 kept its own value 0.111", "0.111" in out[5]),
        ("section 4.3 untouched", out[7] == "### 4.3 y"),
        ("skip=1 keyed the 44 table by (dataset, H, q/r)", len(art44) == 2),
        ("44: both rows filled", filled44 == 2 and missing44 == []),
        ("44: dense row got the DENSE model's numbers", "0.3333" in dense_row),
        ("44: low-rank row got the LOW-RANK model's numbers", "0.1111" in low_row),
        ("44: the two same-setting rows were not swapped",
         "0.1111" not in dense_row and "0.3333" not in low_row),
        ("44: the pre-filled q/r cell was rewritten (r=H -> r=96)",
         "r=H" not in dense_row),
        ("44: section 4.5 untouched", out44[7] == "### 4.5 y"),
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
    ap.add_argument("--skip-artifact-columns", type=int, default=0,
                    help="leading artifact columns to drop (section 4.4 intervention: 1)")
    ap.add_argument("--strip-paren-in-key", action="store_true",
                    help="key on the part before the first bracket, so the paper's "
                         "generic `dense（r=H）` matches the artifact's `dense（r=96）`")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write a marker into rows with no artifact row instead of "
                         "refusing; default is to refuse, so a partial fill cannot "
                         "happen silently")
    ap.add_argument("--marker", default="未跑（按指示停止）",
                    help="marker written into every data cell of a row with no "
                         "artifact row (used only with --allow-missing)")
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

    artifact = parse_artifact(art_path.read_text(encoding="utf-8"), a.key_columns,
                              a.skip_artifact_columns, a.strip_paren_in_key)
    lines = paper.read_text(encoding="utf-8").splitlines()
    if not a.allow_missing:
        pre, _f, pre_missing, _r = fill(
            lines, artifact, a.start, a.end, a.header_needle, a.key_columns,
            a.strip_paren_in_key)
        if pre_missing:
            print(f"{len(pre_missing)} paper row(s) have no artifact row "
                  f"(e.g. {'|'.join(pre_missing[0])}); pass --allow-missing with "
                  f"--marker to write them as explicitly not run, or fix the artifact",
                  file=sys.stderr)
            return 2
    out, filled, missing, (first, last) = fill(
        lines, artifact, a.start, a.end, a.header_needle, a.key_columns,
        a.strip_paren_in_key, a.marker)

    print(f"artifact rows: {len(artifact)}")
    print(f"paper table rows: {last - first} (lines {first + 1}..{last})")
    print(f"filled: {filled}   missing from artifact: {len(missing)}")
    for key in missing:
        print(f"  MISSING {'|'.join(key)}")
    emptied = []
    for i in range(first, last):
        cells = cells_of(out[i].strip())
        if not is_separator(cells) and any(c == "" for c in cells[a.key_columns:]):
            emptied.append("|".join(cells[:a.key_columns]))
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
