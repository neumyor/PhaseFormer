#!/usr/bin/env python3
"""Fill minipaper section 4.4's DISSECTION table from dissection_table_44.csv.

This is the one fill whose cells are not copied but *composed*: five of the eight
paper columns are built out of the artifact's columns.  The rendering below is not
invented -- it mirrors ``verify_minipaper_fill.check_4_4_dissection``, which parses
the paper cell the same way it is produced here:

    col 3  main-mode input group / explanation   <- input_group_label  + input_group_explanation
    col 4  main-mode output group / explanation  <- output_group_label + output_group_explanation
    col 5  correction energy share               <- correction_energy_share
    col 6  cross-seed leading4 overlap           <- leading4_input_overlap + leading4_output_overlap
    col 7  stable-semantics verdict              <- stable_semantics_verdict

Renderings: ``{label} / {rate:.2f}``, ``{share:.3f}``, ``{in:.2f} / {out:.2f}``, and
``✓``/``✗``.  Group labels may themselves contain a slash (a real label is
``周期形状/相位``), which is why the verifier checks "the cell starts with the label
and its LAST number equals the rate" rather than splitting on the slash; putting the
rate last is therefore part of the contract.

Fail-closed: if any required value is missing for a row, nothing is written at all.
A half-rendered cell would compare as blank (which the acceptance criteria only
count for prose), so refusing loudly is the safer failure.

Dry-run by default; ``--write`` applies.  Rows are keyed by (model, dataset, horizon)
and the paper's row order is preserved.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

REQUIRED = ("input_group_label", "input_group_explanation",
            "output_group_label", "output_group_explanation",
            "correction_energy_share",
            "leading4_input_overlap", "leading4_output_overlap",
            "stable_semantics_verdict")


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


def render(row: dict) -> list:
    """The five composed cells for one artifact row (indices 3..7 of the paper row)."""
    inp_label = str(row.get("input_group_label") or "").strip()
    out_label = str(row.get("output_group_label") or "").strip()
    inp_rate = float(row["input_group_explanation"])
    out_rate = float(row["output_group_explanation"])
    share = float(row["correction_energy_share"])
    lin = float(row["leading4_input_overlap"])
    lout = float(row["leading4_output_overlap"])
    verdict = "✓" if str(row["stable_semantics_verdict"]).strip().lower() in ("true", "1") else "✗"
    return [
        f"{inp_label} / {inp_rate:.2f}",
        f"{out_label} / {out_rate:.2f}",
        f"{share:.3f}",
        f"{lin:.2f} / {lout:.2f}",
        verdict,
    ]


def self_test() -> int:
    lines = [
        "### 4.4 x",
        "",
        "| 模型 | Dataset | H | 主模式输入组 / 解释率 | 主模式输出组 / 解释率 | 修正能量份额 | 跨 seed leading4 重叠 | 稳定语义判定 |",
        "|---|---|---:|---|---|---:|---|---|",
        "| PhaseFormer-L | ETTh2 | 96 |  |  |  |  |  |",
        "| L-q1/8 | ETTh2 | 96 |  |  |  |  |  |",
        "",
        "### 4.5 y",
    ]
    rows = {
        ("PhaseFormer-L", "ETTh2", "96"): {
            "input_group_label": "周期形状/相位", "input_group_explanation": "0.6649",
            "output_group_label": "近端电平", "output_group_explanation": "0.5",
            "correction_energy_share": "0.6521771",
            "leading4_input_overlap": "0.777", "leading4_output_overlap": "0.666",
            "stable_semantics_verdict": "True",
        },
        ("L-q1/8", "ETTh2", "96"): {
            "input_group_label": "电平", "input_group_explanation": "0.31",
            "output_group_label": "常数", "output_group_explanation": "0.94",
            "correction_energy_share": "0.3202054",
            "leading4_input_overlap": "0.5", "leading4_output_overlap": "0.25",
            "stable_semantics_verdict": "false",
        },
    }
    _, first, last = locate_table(lines, "### 4.4", "### 4.5", "主模式输入组")
    out = list(lines)
    for index in range(first, last):
        cells = cells_of(out[index].strip())
        rendered = render(rows[(cells[0], cells[1], cells[2])])
        out[index] = "| " + " | ".join(cells[:3] + rendered) + " |"

    dense, low = out[4], out[5]
    ok = True
    checks = [
        ("row count found (2 rows)", last - first == 2),
        ("label leads the cell and rate is LAST (rate 0.66 -> 0.66)",
         "周期形状/相位 / 0.66" in dense),
        ("rate rounded to 2dp (0.5 -> 0.50)", "近端电平 / 0.50" in dense),
        ("share rounded to 3dp", "0.652" in dense),
        ("overlap rendered in / out at 2dp", "0.78 / 0.67" in dense),
        ("truthy artifact value -> check mark", "✓" in dense),
        ("falsy artifact value -> cross", "✗" in low),
        ("key columns preserved", dense.startswith("| PhaseFormer-L | ETTh2 | 96 |")),
        ("rows not swapped", "0.320" in low and "0.320" not in dense),
        ("section 4.5 untouched", out[7] == "### 4.5 y"),
    ]
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
        ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipaper")
    ap.add_argument("--artifact")
    ap.add_argument("--start", default="### 4.4")
    ap.add_argument("--end", default="### 4.5")
    ap.add_argument("--header-needle", default="主模式输入组")
    ap.add_argument("--key-columns", type=int, default=3)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write markers into rows with no artifact row instead of "
                         "refusing.  Used when an experiment was stopped, so the table "
                         "states the gap instead of leaving cells that look unfilled")
    ap.add_argument("--marker", default="未跑（按指示停止）",
                    help="marker written into the five composed cells of a row with no "
                         "artifact row (only with --allow-incomplete)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.minipaper or not a.artifact:
        ap.error("--minipaper and --artifact are required (or use --self-test)")

    art_path = pathlib.Path(a.artifact)
    if not art_path.is_file():
        print(f"artifact {art_path} does not exist yet; nothing to fill", file=sys.stderr)
        return 2

    with art_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        absent = [name for name in REQUIRED if name not in columns]
        if absent:
            print(f"{art_path.name} lacks {absent}; refusing to guess", file=sys.stderr)
            return 2
        artifact = {}
        for row in reader:
            key = (str(row.get("model")), str(row.get("dataset")), str(row.get("horizon")))
            artifact[key] = row

    paper = pathlib.Path(a.minipaper)
    lines = paper.read_text(encoding="utf-8").splitlines()
    _, first, last = locate_table(lines, a.start, a.end, a.header_needle)

    out = list(lines)
    filled, incomplete = 0, []
    for index in range(first, last):
        cells = cells_of(out[index].strip())
        if len(cells) < a.key_columns:
            raise SystemExit(f"row {index + 1} has {len(cells)} cells, expected {a.key_columns}+")
        key = tuple(cells[:a.key_columns])
        row = artifact.get(key)
        if row is None:
            incomplete.append(("|".join(key), "no artifact row"))
            if a.allow_incomplete:
                out[index] = ("| " + " | ".join(cells[:a.key_columns]
                                                + [a.marker] * (len(cells) - a.key_columns))
                              + " |")
            continue
        gaps = [name for name in REQUIRED if str(row.get(name) or "").strip() == ""]
        if gaps:
            incomplete.append(("|".join(key), f"artifact gaps {gaps}"))
            continue
        rendered = render(row)
        out[index] = "| " + " | ".join(cells[:a.key_columns] + rendered) + " |"
        filled += 1
        print(f"  {key[0]} | {key[1]}-{key[2]}: " + " | ".join(rendered))

    print(f"artifact rows: {len(artifact)}")
    print(f"paper table rows: {last - first} (lines {first + 1}..{last})")
    print(f"filled: {filled}   incomplete: {len(incomplete)}")
    for key, why in incomplete:
        print(f"  INCOMPLETE {key}: {why}")

    if incomplete and not a.allow_incomplete:
        print("refusing to write a partial dissection table (pass --allow-incomplete "
              "to mark those rows as not run instead)", file=sys.stderr)
        return 2
    if not a.write:
        print("dry run; pass --write to apply")
        return 0
    paper.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
