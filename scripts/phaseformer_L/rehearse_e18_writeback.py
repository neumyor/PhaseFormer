#!/usr/bin/env python3
"""Rehearse the E18 write-back (minipaper section 4.6) without any training.

Why this exists
---------------
`e18_writeback.py` is the LAST stage of the phase-2 chain.  A crash here loses
the section 4.6 table after every expensive stage has already run, so the cheap
thing to do is prove -- in seconds, with no GPU -- that it survives both good and
degraded inputs.

Sources of truth are the producers' own declarations, so the fixtures cannot
drift from what the real runs will emit:

* E18's ``RESULTS_FIELDS`` and its level names (``causal_ema_mid`` /
  ``causal_ema_max`` for the smoothing rows; the rank rows carry an empty level
  and an integer ``rank``);
* E14's ``RESULTS_FIELDS`` for the ``l_main`` baseline the rows are differenced
  against;
* ``e18_svd_truncation.TABLE_FIELDS`` for the row-3 truncation table.

Three cases are exercised, and the last two matter most because a missing value
is the realistic failure mode (a cell that could not be evaluated, a baseline
that is not in the reuse set):

  A  fully populated          -- expect 5 rows and non-empty verdicts
  B  no E14 baseline at all   -- must not crash, must degrade to empty deltas
  C  E18 rows with blank test metrics -- must not crash

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/rehearse_e18_writeback.py [--scratch DIR]
"""

from __future__ import annotations

import argparse
import ast
import csv
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the contract checker's source-derived extractor instead of re-writing it:
# the merged file the write-back consumes is the runner's own fields PLUS the
# columns read_test_generic stamps on.  Deriving both means this fixture cannot
# drift from the real artifact -- getting it wrong once made the first run fail
# with "dict contains fields not in fieldnames: test_mse, test_mae", because E18
# records `test_mse_recorded` and only the reader supplies `test_mse`.
from scripts.phaseformer_L import check_column_contracts as cc  # noqa: E402

SETTINGS = (("ETTh1", 96), ("ETTh2", 96), ("ETTm2", 192), ("Weather", 96))
SMOOTH_LEVELS = ("causal_ema_mid", "causal_ema_max")


def literal_constants(path: pathlib.Path, names: set) -> dict:
    out: dict = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def write_csv(path: pathlib.Path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            full = {key: "" for key in fields}
            full.update(row)
            writer.writerow(full)


def build_fixtures(scratch: pathlib.Path, baseline: bool, blank_metrics: bool):
    stamped = cc._subscript_store_keys(
        REPO / "scripts/phaseformer_L/read_test_generic.py")
    e18_fields = list(dict.fromkeys(
        literal_constants(REPO / "scripts/phaseformer_L/e18_negative.py",
                          {"RESULTS_FIELDS"})["RESULTS_FIELDS"] + sorted(stamped)))
    e14_fields = literal_constants(
        REPO / "scripts/phaseformer_L/e14_read_test.py", {"RESULTS_FIELDS"})["RESULTS_FIELDS"]
    trunc_fields = literal_constants(
        REPO / "scripts/phaseformer_L/e18_svd_truncation.py", {"TABLE_FIELDS"})["TABLE_FIELDS"]

    def metric(seed, blank):
        if blank:
            return {"test_mse": "", "test_mae": ""}
        return {"test_mse": 0.2 + 0.001 * (seed - 2021), "test_mae": 0.3}

    e18_rows = []
    for dataset, horizon in SETTINGS:
        for level in SMOOTH_LEVELS:
            for seed in (2021, 2022, 2023):
                e18_rows.append({
                    "stage": "smooth", "dataset": dataset, "horizon": horizon,
                    "seed": seed, "level": level, "rank": "",
                    "baseline_status": "reused", **metric(seed, blank_metrics),
                })
        for rank in (1, 2):
            for seed in (2021, 2022, 2023):
                e18_rows.append({
                    "stage": "rank12", "dataset": dataset, "horizon": horizon,
                    "seed": seed, "level": "", "rank": rank,
                    "baseline_status": "reused", **metric(seed, blank_metrics),
                })
    write_csv(scratch / "e18_results.csv", e18_fields, e18_rows)

    e14_rows = []
    if baseline:
        for dataset, horizon in SETTINGS:
            for seed in (2021, 2022, 2023):
                e14_rows.append({
                    "arm": "l_main", "dataset": dataset, "horizon": horizon,
                    "seed": seed, "setting": f"{dataset}-{horizon}",
                    "status": "read", "source": "rehearsal-fixture",
                    "test_mse": 0.19 + 0.001 * (seed - 2021), "test_mae": 0.29,
                })
    write_csv(scratch / "e14_results.csv", e14_fields, e14_rows)

    trunc_rows = []
    for dataset, horizon in SETTINGS:
        trunc_rows.append({
            "setting": f"{dataset}-{horizon}", "dataset": dataset,
            "horizon": horizon, "primary_rank": 10, "seed": 2021, "split": "val",
            "truncated_mse": 0.25, "truncated_mae": 0.35,
            "trained_lowrank_mse": 0.21, "trained_lowrank_mae": 0.31,
            "gap_truncated_vs_trained_mse_pct": 19.0,
            "gap_truncated_vs_trained_mae_pct": 12.9,
            "full_rank_mse": 0.19, "full_rank_mae": 0.29,
            "records_test": "False",
        })
    write_csv(scratch / "svd_truncation.csv", trunc_fields, trunc_rows)

    return len(e18_rows), len(e14_rows), len(trunc_rows)


def run_case(name: str, scratch: pathlib.Path, baseline: bool, blank: bool) -> int:
    scratch.mkdir(parents=True, exist_ok=True)
    n18, n14, ntr = build_fixtures(scratch, baseline, blank)
    print(f"--- {name}: e18 rows={n18} e14 rows={n14} truncation rows={ntr}")
    result = subprocess.run(
        [sys.executable, "scripts/phaseformer_L/e18_writeback.py",
         "--results", str(scratch / "e18_results.csv"),
         "--e14-results", str(scratch / "e14_results.csv"),
         "--truncation", str(scratch / "svd_truncation.csv"),
         "--output-root", str(scratch)],
        cwd=REPO, capture_output=True, text=True)
    print(f"    exit={result.returncode}")
    if result.returncode:
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        return 1

    table = list(csv.DictReader((scratch / "negative_table.csv").open()))
    rows = [r["row"] for r in table]
    print(f"    negative_table rows={rows}")
    if rows != ["1", "2", "3", "4", "5"]:
        print("    FAIL: expected exactly rows 1..5")
        return 1
    for row in table:
        addendum = str(row.get("addendum", ""))
        print(f"      row {row['row']}: scope={row.get('scope')!r} "
              f"addendum={addendum[:60]!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="/tmp/rehearse_e18")
    args = parser.parse_args()
    base = pathlib.Path(args.scratch)

    failures = 0
    failures += run_case("case A: fully populated", base / "A", True, False)
    failures += run_case("case B: no E14 baseline", base / "B", False, False)
    failures += run_case("case C: blank test metrics", base / "C", True, True)

    print()
    if failures:
        print(f"FAIL: {failures} case(s) did not survive")
        return 1
    print("PASS: e18_writeback survives populated and degraded inputs, emitting rows 1..5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
