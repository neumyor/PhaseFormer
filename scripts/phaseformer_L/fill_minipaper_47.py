#!/usr/bin/env python3
"""Fill minipaper section 4.7's two rho columns from predictive_power_summary.json.

Row mapping is BY INDEX, matching ``verify_minipaper_fill.check_4_7``: row i of the
paper's table corresponds to ``STATISTICS[i]``.  That means this tool must never
reorder rows -- it rewrites the two rho cells in place and leaves row order alone.

Precision is 3 decimals because the verifier compares with
``compare_number(cell, value, 3)``.

A non-finite rho (None or NaN) is written literally as ``NaN`` rather than
substituted with a number: the verifier reports such a cell as PENDING, and the
pre-registered handling is to name those cells explicitly instead of inventing a
value (see minipaper_fill_mapping.md section 1.8).

The ``STATISTICS`` tuple is not redeclared blindly: when the verifier module can be
imported, this tool compares its own copy against the verifier's and refuses to run
if they disagree, so the two cannot drift apart silently.

Dry-run by default; ``--write`` applies the change in place.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

STATISTICS = ("cycle_level_std", "last_cycle_shift", "tau_hat_steps")


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


def format_rho(value) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, float) and value != value:
        return "NaN"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if number != number:
        return "NaN"
    return f"{number:.3f}"


def spec_check(repo: str) -> list:
    """Guard against STATISTICS drift: compare with the verifier's own tuple."""
    if not repo:
        return ["(verifier not imported: --repo not given, drift guard skipped)"]
    sys.path.insert(0, repo)
    try:
        from scripts.phaseformer_L.verify_minipaper_fill import STATISTICS as theirs
    except Exception as exc:                       # pragma: no cover - env dependent
        return [f"(could not import the verifier's STATISTICS: {exc!r})"]
    if tuple(theirs) != tuple(STATISTICS):
        raise SystemExit(
            "STATISTICS drift: this tool has "
            f"{STATISTICS} but the verifier has {tuple(theirs)}; refusing to fill")
    return [f"STATISTICS matches the verifier: {tuple(theirs)}"]


def self_test() -> int:
    paper = [
        "### 4.7 x",
        "",
        "| 统计量 | 与 ΔMSE 的 Spearman ρ | 与 `g` 的 ρ | 预期符号 |",
        "|---|---:|---:|---|",
        "| `cycle_level_std` |  |  | + |",
        "| `last_cycle_shift` |  |  | + |",
        "| `tau_hat_steps` |  |  | 与学到的 EMA τ 正相关 |",
        "",
        "## 5. next",
    ]
    summary = {"predictive_power": {"spearman": {
        "cycle_level_std_vs_delta_mse_pct": {"rho": 0.12345},
        "cycle_level_std_vs_gate_value": {"rho": -0.5},
        "last_cycle_shift_vs_delta_mse_pct": {"rho": 0.6789},
        "last_cycle_shift_vs_gate_value": {"rho": None},
        "tau_hat_steps_vs_delta_mse_pct": {"rho": -0.999},
        "tau_hat_steps_vs_gate_value": {"rho": float("nan")},
    }}}
    _, first, last = locate_table(paper, "### 4.7", "## 5", "与 ΔMSE 的 Spearman")
    spearman = summary["predictive_power"]["spearman"]
    rows = list(paper)
    for index in range(first, last):
        stat = STATISTICS[index - first]
        cells = cells_of(rows[index])
        cells[1] = format_rho(spearman.get(f"{stat}_vs_delta_mse_pct", {}).get("rho"))
        cells[2] = format_rho(spearman.get(f"{stat}_vs_gate_value", {}).get("rho"))
        rows[index] = "| " + " | ".join(cells) + " |"
    checks = [
        ("row 1 gets 3-decimal rounding", "0.123" in rows[4]),
        ("row 1 negative rho kept", "-0.500" in rows[4]),
        ("row 2 rounded", "0.679" in rows[4 + 1]),
        ("None becomes NaN", "NaN" in rows[5]),
        ("row 3 negative", "-0.999" in rows[6]),
        ("NaN float becomes NaN", "NaN" in rows[6]),
        ("row order untouched (statistic names intact)",
         "cycle_level_std" in rows[4] and "tau_hat_steps" in rows[6]),
        ("next section untouched", rows[8] == "## 5. next"),
    ]
    ok = True
    for name, good in checks:
        print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
        ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipaper")
    ap.add_argument("--summary")
    ap.add_argument("--repo", default="")
    ap.add_argument("--start", default="### 4.7")
    ap.add_argument("--end", default="## 5")
    ap.add_argument("--header-needle", default="与 ΔMSE 的 Spearman")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.minipaper or not a.summary:
        ap.error("--minipaper and --summary are required (or use --self-test)")

    for note in spec_check(a.repo):
        print(note)

    summary_path = pathlib.Path(a.summary)
    if not summary_path.is_file():
        print(f"summary {summary_path} does not exist yet; nothing to fill", file=sys.stderr)
        return 2
    paper = pathlib.Path(a.minipaper)
    lines = paper.read_text(encoding="utf-8").splitlines()

    raw = summary_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{summary_path} is not strict JSON ({exc}); "
                         f"known boundary: the summary can emit a bare NaN token")

    spearman = (data.get("predictive_power") or {}).get("spearman")
    if not isinstance(spearman, dict):
        raise SystemExit("summary has no predictive_power.spearman block")

    _, first, last = locate_table(lines, a.start, a.end, a.header_needle)
    if last - first != len(STATISTICS):
        raise SystemExit(f"paper table has {last - first} rows, expected {len(STATISTICS)}")

    out = list(lines)
    filled, nan_cells = 0, []
    for offset in range(last - first):
        index = first + offset
        stat = STATISTICS[offset]
        cells = cells_of(out[index].strip())
        if len(cells) < 4:
            raise SystemExit(f"row {offset + 1} has {len(cells)} cells, expected 4")
        for column, suffix in ((1, "_vs_delta_mse_pct"), (2, "_vs_gate_value")):
            entry = spearman.get(f"{stat}{suffix}")
            if not isinstance(entry, dict):
                raise SystemExit(f"summary lacks {stat}{suffix}")
            text = format_rho(entry.get("rho"))
            if text == "NaN":
                nan_cells.append(f"{stat}{suffix}")
            cells[column] = text
            filled += 1
        out[index] = "| " + " | ".join(cells) + " |"
        print(f"  {stat}: delta_mse={cells[1]}  gate={cells[2]}")

    print(f"cells filled: {filled}   non-finite: {len(nan_cells)} {nan_cells}")
    if not a.write:
        print("dry run; pass --write to apply")
        return 0
    paper.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
