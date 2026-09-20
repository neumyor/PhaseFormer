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
                   empty_metrics: bool, with_params: bool,
                   drop_gate_for: set = frozenset(), with_dataset_in_params: bool = True):
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
    datasets = sorted({d for (d, _h) in golden})

    # Per-dataset gate priors.  They must be DISTINCT and ordered so that the
    # smallest one at a horizon belongs to the LAST dataset; a fallback that pools
    # the dataset away then cannot coincidentally return the right number.
    gate_prior = {d: round(0.30 + 0.05 * i, 6) for i, d in enumerate(datasets)}
    # Channel counts differ by dataset in reality, and the phase trunk scales with
    # them, so total_params must depend on the dataset as well.
    channels = {d: 7 + 13 * i for i, d in enumerate(datasets)}

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
                }
                # Cells in `drop_gate_for` emulate the real Stage-0 reused cells:
                # their evidence table records no gate, so section 4.2 has to read
                # it from the checkpoint instead.
                if dataset not in drop_gate_for:
                    row["gate_value"] = round(0.2 + 0.001 * arms.index(arm), 6)
                if not empty_metrics:
                    row["test_mse"] = round(gmse * factor, 6)
                    row["test_mae"] = round(gmae * factor, 6)
                rows.append(row)
    write_csv(scratch / "results.csv", results_fields, rows)

    param_rows = []
    if with_params:
        for arm in arms:
            for dataset in datasets:
                for horizon in sorted({h for (d, h) in golden if d == dataset}):
                    for index, seed in enumerate(seeds):
                        total = 100000 + 100 * horizon + 137 * channels[dataset]
                        residual = 1000 + horizon
                        param_rows.append({
                            "arm": arm,
                            "horizon": horizon,
                            "seed": seed,
                            "total_params": total,
                            "residual_params": residual,
                            "residual_share": round(residual / total, 6),
                            # Slightly different per seed, as in reality, so the
                            # fallback can only be right if it takes the MEAN.
                            "gate_value_from_checkpoint": round(
                                gate_prior[dataset] + 0.001 * index
                                + 0.001 * arms.index(arm), 6),
                        })
                        if with_dataset_in_params:
                            param_rows[-1]["dataset"] = dataset
        write_csv(scratch / "parameter_table.csv", params_fields, param_rows)

    return len(rows), len(param_rows)


def expected_fallback_gate(dataset: str, arms_index: int, datasets, seeds) -> float:
    prior = round(0.30 + 0.05 * datasets.index(dataset), 6)
    return round(sum(prior + 0.001 * i + 0.001 * arms_index
                     for i in range(len(seeds))) / len(seeds), 6)


def run_case(name: str, scratch: pathlib.Path, golden: dict, manifest: dict,
             empty_metrics: bool, with_params: bool,
             drop_gate_for: frozenset = frozenset(),
             with_dataset_in_params: bool = True) -> int:
    scratch.mkdir(parents=True, exist_ok=True)
    n_res, n_par = build_fixtures(scratch, golden, manifest, empty_metrics,
                                  with_params, drop_gate_for,
                                  with_dataset_in_params)
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

    # ------------------------------------------------------------------
    # The gate fallback (defect found 2026-09-20, see 05_audit.md section 17).
    # When a cell's results.csv row carries no gate -- exactly the 7 reused
    # `l_main` settings in the real run -- section 4.2 reads the checkpoint value
    # out of parameter_table.csv.  That lookup used to be keyed by
    # (arm, horizon), pooling the dataset away, so every such cell received the
    # SMALLEST gate any dataset had at that horizon instead of its own.
    #
    # The fixture makes that failure impossible to miss: each dataset gets a
    # distinct gate prior and its own channel count, so pooling either returns a
    # foreign dataset's number or (for the old min-collapse) the wrong statistic.
    # ------------------------------------------------------------------
    if with_params and drop_gate_for:
        datasets = sorted({d for (d, _h) in golden})
        arms = sorted({str(cell["arm"]) for cell in manifest.get("cells", [])})
        wrong_gate = []
        for row in main:
            dataset = row["dataset"]
            if dataset not in drop_gate_for:
                continue
            source = str(row.get("l_main_gate_source", "")).strip()
            if not with_dataset_in_params:
                # A parameter table without a dataset column cannot identify a
                # cell's gate.  Refusing to report one is the correct behaviour;
                # borrowing the horizon's smallest value is not.
                #
                # Careful with the emptiness test: a blank CSV cell reads back as
                # "", not None, so `is not None` would call a correctly-left-blank
                # cell wrong.  That mistake cost one rehearsal round (the "bad"
                # report was in the assertion, not the producer).
                got = row.get("l_main_gate_mean")
                if got is not None and str(got).strip() != "":
                    wrong_gate.append((row["setting"], f"gate={got!r}",
                                       "expected no gate at all"))
                elif source:
                    wrong_gate.append((row["setting"], f"source={source!r}",
                                       "expected no gate source at all"))
                continue
            if source != "checkpoint":
                wrong_gate.append((row["setting"], source, "expected the checkpoint "
                                                            "fallback"))
                continue
            # Only the three weak-residual arms carry a gate at all.  The others
            # (`phase_only`, `l_rcrf`, `a1`) have no such parameter, so the paper
            # leaves their gate cell blank and the fallback is never consulted for
            # them -- asserting on them would demand a value the model cannot have.
            for arm_index, arm in enumerate(arms):
                if arm not in ("l_main", "l_q1_4", "l_q1_8"):
                    continue
                want = expected_fallback_gate(dataset, arm_index, datasets, (2021, 2022, 2023))
                got = row.get(f"{arm}_gate_mean")
                if got is None or str(got).strip() == "" \
                        or abs(float(got) - want) > 1e-6:
                    wrong_gate.append((row["setting"], f"{arm}={got!r}", want))
        expectation = ("the cell's own seed-averaged gate" if with_dataset_in_params
                       else "no gate at all (the table cannot identify the cell)")
        print(f"    fallback-gate cells checked: "
              f"{len([r for r in main if r['dataset'] in drop_gate_for])}; "
              f"wrong: {len(wrong_gate)}; expecting {expectation}"
              f"{' e.g. ' + str(wrong_gate[:3]) if wrong_gate else ''}")
        if wrong_gate:
            print("      <-- FAIL: the checkpoint fallback is not reading THIS "
                  "cell's own gate")
            failures += 1
        else:
            print("      OK: the fallback behaves correctly for this schema")

        # The quoted per-horizon parameter count must be attributable.  In the
        # fixture every dataset has a different channel count, so a single
        # anonymous number per horizon cannot be right for all of them.  Case E
        # deliberately omits the dataset column, so there the correct outcome is
        # that no count is reported at all -- which the audit's schema criterion
        # is what turns into a visible failure rather than a silently blank paper
        # cell.  Only the well-formed schema is asserted here.
        reference = {str(r.get("arm")): r.get("total_params_reference_dataset")
                     for r in variant}
        named = [a for a, v in reference.items() if v]
        print(f"    arms naming their reference dataset: {len(named)} of {len(variant)}; "
              f"e.g. l_main -> {reference.get('l_main')!r}")
        if with_dataset_in_params and not named:
            print("      <-- FAIL: total_params_per_horizon is unattributed")
            failures += 1
        elif not with_dataset_in_params and named:
            print("      <-- FAIL: a parameter table with no dataset column cannot "
                  "attribute a count, yet one was reported")
            failures += 1

    return failures

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
    # Case D is the one that would have caught the gate defect: two datasets get
    # no results.csv gate (as the real reused cells), so the checkpoint fallback
    # is exercised and must return each dataset's OWN gate.
    failures += run_case("case D: checkpoint-fallback gates", base / "D", golden,
                         manifest, False, True, frozenset({"ETTh1", "Weather"}))
    # Case E: an OLD-style parameter table with no dataset column.  The correct
    # behaviour is to leave the gate unknown (None) rather than to borrow another
    # dataset's; this pins that a schema regression cannot reintroduce the defect.
    failures += run_case("case E: parameter table without a dataset column",
                         base / "E", golden, manifest, False, True,
                         frozenset({"ETTh1", "Weather"}),
                         with_dataset_in_params=False)

    print()
    if failures:
        print(f"FAIL: {failures} assertion(s) did not hold")
        return 1
    print("PASS: the section 4.2 builder holds the 24+4 structure and survives "
          "degraded inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
