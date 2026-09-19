#!/usr/bin/env python3
"""Rehearse the E14 write-back -- the minipaper section 4.2 main table.

Why this exists
---------------
This is the paper's centrepiece artifact (24 main settings + the 4-setting
Traffic appendix), and it is produced by the busiest builder in the suite: it
joins the manifest, the test-bearing results, the Golden reference, E19's level
statistics and E14's parameter table, then evaluates claims A-D against frozen
thresholds.

Two properties are worth proving before the real run, cheaply and with no GPU:

* **structure** -- exactly 24 rows in ``main_table.csv`` and 4 in
  ``variant_table.csv`` (the "24+4" the minipaper demands);
* **survival** -- it does not crash when inputs are degraded (no test metrics,
  no parameter table), because a crash here loses the main result.

The fixture uses the REAL manifest, the REAL Golden table and the REAL level
statistics; only the results and parameter rows are synthetic, and their columns
come from the producers' own declarations so they cannot drift.

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/rehearse_e14_writeback.py [--scratch DIR]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

MANIFEST = "research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json"
STATS = "research_runs/phaseformer_L_e19_predictive_v1/level_statistics.csv"
GOLDEN = "docs/PhaseFormer_gold_standard.md"


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


def build_fixtures(scratch: pathlib.Path, golden: dict, manifest: dict,
                   empty_metrics: bool, with_params: bool):
    results_fields = literal_constants(
        REPO / "scripts/phaseformer_L/e14_read_test.py", {"RESULTS_FIELDS"})["RESULTS_FIELDS"]
    params_fields = literal_constants(
        REPO / "scripts/phaseformer_L/e14_params.py", {"PARAMETER_FIELDS"}).get(
            "PARAMETER_FIELDS")
    if not params_fields:
        # The parameter table's columns are built dynamically; take them from the
        # module that writes it via the same extractor the contract checker uses.
        sys.path.insert(0, str(REPO))
        from scripts.phaseformer_L import check_column_contracts as cc  # noqa: E402
        params_fields = sorted(cc._enclosing_function_keys(
            REPO / "scripts/phaseformer_L/e14_params.py", '"parameter_table.csv"'))

    arms = sorted({str(cell["arm"]) for cell in manifest.get("cells", [])})
    seeds = (2021, 2022, 2023)

    rows = []
    for (dataset, horizon), (gmse, gmae) in sorted(golden.items()):
        for arm in arms:
            for seed in seeds:
                # Arm-dependent so the claim logic has something to compare;
                # the absolute numbers are meaningless by construction.
                factor = 1.0 + 0.02 * (hash((arm, seed)) % 5) - 0.01 * arms.index(arm)
                row = {
                    "arm": arm, "dataset": dataset, "horizon": horizon,
                    "seed": seed, "setting": f"{dataset}-{horizon}",
                    "status": "read", "source": "rehearsal-fixture",
                    "gate_value": round(0.2 + 0.001 * arms.index(arm), 6),
                }
                if not empty_metrics:
                    row["test_mse"] = round(gmse * factor, 6)
                    row["test_mae"] = round(gmae * factor, 6)
                rows.append(row)
    write_csv(scratch / "results.csv", results_fields, rows)

    param_rows = []
    if with_params:
        for arm in arms:
            for horizon in sorted({h for (_d, h) in golden}):
                param_rows.append({
                    "arm": arm, "horizon": horizon, "seed": 2021,
                    "total_params": 100000 + 100 * horizon,
                    "residual_params": 1000 + horizon,
                    "residual_share": round((1000 + horizon) / (100000 + 100 * horizon), 6),
                    "gate_value_from_checkpoint": round(0.2 + 0.001 * arms.index(arm), 6),
                })
        write_csv(scratch / "parameter_table.csv", params_fields, param_rows)

    return len(rows), len(param_rows)


def run_case(name: str, scratch: pathlib.Path, golden: dict, manifest: dict,
             empty_metrics: bool, with_params: bool) -> int:
    scratch.mkdir(parents=True, exist_ok=True)
    n_res, n_par = build_fixtures(scratch, golden, manifest, empty_metrics, with_params)
    print(f"--- {name}: results rows={n_res} parameter rows={n_par}")
    result = subprocess.run(
        [sys.executable, "scripts/phaseformer_L/e14_writeback.py",
         "--manifest", str(REPO / MANIFEST),
         "--results", str(scratch / "results.csv"),
         "--stats", str(REPO / STATS),
         "--params", str(scratch / "parameter_table.csv"),
         "--golden", str(REPO / GOLDEN),
         "--output-root", str(scratch)],
        cwd=REPO, capture_output=True, text=True)
    print(f"    exit={result.returncode}")
    if result.returncode:
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        return 1

    main = list(csv.DictReader((scratch / "main_table.csv").open()))
    variant = list(csv.DictReader((scratch / "variant_table.csv").open()))
    claims = json.loads((scratch / "claims.json").read_text())

    # The shipped layout is: main_table.csv = one row per Golden setting (28),
    # of which the Traffic ones carry is_traffic_appendix=True (4) -- that flag,
    # not a separate file, is what separates the 24 main settings from the
    # 4-setting appendix; variant_table.csv is the per-arm summary (6 arms).
    # An earlier version of this rehearsal asserted "main 24 + variant 4" and
    # reported a false failure against a correct builder.
    appendix = [r for r in main if str(r.get("is_traffic_appendix")).strip() == "True"]
    core = len(main) - len(appendix)
    print(f"    main_table rows={len(main)} (core {core} + appendix {len(appendix)}), "
          f"variant_table rows={len(variant)}")

    failures = 0
    if len(main) != 28 or core != 24 or len(appendix) != 4:
        print(f"    FAIL: expected 28 rows = 24 core + 4 appendix, got "
              f"{len(main)} = {core} + {len(appendix)}")
        failures += 1
    else:
        print("    OK: the 24+4 structure holds (via is_traffic_appendix)")

    # Regression check for the provenance column, which was emitted empty once.
    blank_prov = [r.get("setting") for r in main
                  if not str(r.get("provenance_note", "")).strip()]
    print(f"    rows with an empty provenance_note: {len(blank_prov)}"
          f"{' e.g. ' + str(blank_prov[:3]) if blank_prov else ''}")
    if blank_prov:
        failures += 1

    # claims.json carries A-D plus the two must-answer blocks; the flattened
    # verdicts are emitted on stdout as a single {"event": "finished", ...} line.
    for key in ("A", "B", "C", "D", "must_answer_a", "must_answer_b"):
        present = key in claims
        print(f"    claims.json[{key!r}] present: {present}"
              f"{'' if present else '   <-- MISSING'}")
        if not present:
            failures += 1
    verdicts = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"event": "finished"' in line:
            try:
                verdicts = json.loads(line)
            except Exception:
                pass
    for key in ("claim_A_either_metric", "claim_B", "claim_C_stable_beyond_golden",
                "claim_C_double_metric_improvement", "claim_D",
                "must_answer_a_reaches_fits", "must_answer_b_etth1_ettm1_all_s0"):
        present = key in verdicts
        print(f"    stdout.{key}: {verdicts.get(key)!r}"
              f"{'' if present else '   <-- MISSING'}")
        if not present:
            failures += 1
    print(f"    claims.A verdict_either_metric={claims.get('A', {}).get('verdict_either_metric')!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="/tmp/rehearse_e14")
    args = parser.parse_args()
    base = pathlib.Path(args.scratch)

    sys.path.insert(0, str(REPO))
    from scripts.phaseformer_L import e14_writeback as wb  # noqa: E402

    golden = wb.read_golden(REPO / GOLDEN)
    manifest = json.loads((REPO / MANIFEST).read_text())
    print(f"real Golden settings: {len(golden)}  real manifest cells: "
          f"{len(manifest.get('cells', []))}")

    failures = 0
    failures += run_case("case A: fully populated", base / "A", golden, manifest,
                         False, True)
    failures += run_case("case B: no test metrics", base / "B", golden, manifest,
                         True, True)
    failures += run_case("case C: no parameter table", base / "C", golden, manifest,
                         False, False)

    print()
    if failures:
        print(f"FAIL: {failures} assertion(s) did not hold")
        return 1
    print("PASS: the section 4.2 builder holds the 24+4 structure and survives "
          "degraded inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
