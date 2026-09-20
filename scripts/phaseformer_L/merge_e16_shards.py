#!/usr/bin/env python3
"""Assemble the canonical E16 artifacts from the 21 (setting, arm) shards.

E16 was sharded because the single-process launch was measured infeasible (see
``docs/PhaseFormer_L_execution_schedule.md`` section 9.1.6).  Each shard is a
self-contained slice: one (setting, arm) with all three seeds, so its cross-seed
columns are computed inside the shard exactly as the unsharded run would.  This
tool therefore only has to CONCATENATE rows and AGGREGATE counters -- it never
recomputes a number.

What it writes into the canonical root:

* ``dissection_table.csv``, ``intervention_table.csv``, ``canonical_modes.csv``,
  ``semantic_alignment.csv``, ``cross_seed_alignment.csv`` -- concatenated, with a
  header equality check across shards (a shard with a shifted column order would
  otherwise merge silently into nonsense);
* ``e16_summary.json`` -- rebuilding:
  - ``cells`` (right now a LIST of cell records, not a count) by concatenation;
  - ``counts`` by recomputation from the merged tables plus the shards' row counts;
  - ``invariants.*`` by concatenation (failure lists) and dict merge (per-cell);
  - ``reads_test`` by OR (any shard that read test is a catastrophe, so OR is the
    safe direction);
  - ``reference_parity.*`` by summing probe/comparison counts, taking the maximum
    per-field absolute difference, ANDing ``passed``, and concatenating the
    path-mismatch list;
  - a ``merge_provenance`` block naming every shard root and the rule used for
    each aggregate, so the assembled artifact explains itself.
* ``reference_parity.json`` -- the same aggregation, in the tool's own format.

Fail-closed: if a shard is missing an expected file, if headers disagree, if the
shard count is not what the driver planned, or if any ``einsum_optimize`` flag
disagrees with the others, nothing is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

CSV_FILES = (
    "dissection_table.csv",
    "intervention_table.csv",
    "canonical_modes.csv",
    "semantic_alignment.csv",
    "cross_seed_alignment.csv",
)
SUMMARY = "e16_summary.json"
PARITY = "reference_parity.json"


def read_csv_rows(path: pathlib.Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def concat_csv(shards: list, name: str) -> tuple:
    """Concatenate one CSV across shards, with a strict header-equality check."""
    header = None
    rows = []
    per_shard = {}
    for shard in shards:
        path = shard / name
        if not path.is_file():
            raise SystemExit(f"{shard.name} lacks {name}; refusing a partial merge")
        fields, shard_rows = read_csv_rows(path)
        if header is None:
            header = fields
        elif fields != header:
            raise SystemExit(
                f"column mismatch in {name}: {shard.name} has {fields} but the "
                f"first shard has {header}; refusing to concatenate"
            )
        per_shard[shard.name] = len(shard_rows)
        rows.extend(shard_rows)
    return header, rows, per_shard


def write_csv(path: pathlib.Path, header: list, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def merge_parity(parities: list) -> dict:
    merged = {
        "reference_root": parities[0].get("reference_root"),
        "reference_available": all(p.get("reference_available") for p in parities),
        "probe_cells": sum(int(p.get("probe_cells") or 0) for p in parities),
        "cells_value_compared": sum(int(p.get("cells_value_compared") or 0) for p in parities),
        "checkpoint_path_mismatches": [],
        "fields": {},
        # AND, not OR: parity is a claim that holds only if it holds everywhere.
        "passed": all(bool(p.get("passed")) for p in parities),
    }
    for p in parities:
        merged["checkpoint_path_mismatches"].extend(p.get("checkpoint_path_mismatches") or [])
        for field, entry in (p.get("fields") or {}).items():
            if not isinstance(entry, dict):
                continue
            slot = merged["fields"].setdefault(
                field, {"max_abs_diff": 0.0, "comparisons": 0, "tolerance_abs": None})
            slot["max_abs_diff"] = max(slot["max_abs_diff"],
                                       float(entry.get("max_abs_diff") or 0.0))
            slot["comparisons"] += int(entry.get("comparisons") or 0)
            if slot["tolerance_abs"] is None:
                slot["tolerance_abs"] = entry.get("tolerance_abs")
    return merged


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        shards = []
        for index, (rows, passed) in enumerate(((2, True), (1, False))):
            shard = root / f"shard_{index}"
            shard.mkdir()
            for name in CSV_FILES:
                # Header written ONCE, then the data rows: repeating the header per
                # row made DictReader count headers as data (the first version of
                # this fixture did exactly that, and the self-test caught it).
                body = "".join("l_main,ETTh2-96\n" for _ in range(rows))
                (shard / name).write_text("arm,setting\n" + body, encoding="utf-8")
            (shard / SUMMARY).write_text(json.dumps({
                "cells": [{"arm": "l_main"}] * rows,
                "counts": {"cells": rows, "intervention_rows": rows * 11},
                "invariants": {"algebra_failures": [], "run_metric_failures": [],
                               "run_metric_not_comparable": {}, "per_cell": {}},
                "reads_test": False,
                "einsum_optimize": False,
                "reference_parity": {"passed": passed, "probe_cells": 6,
                                     "cells_value_compared": rows,
                                     "checkpoint_path_mismatches": [{"cell": "q=1/8"}],
                                     "fields": {"x": {"max_abs_diff": 0.1 * rows,
                                                      "comparisons": 1,
                                                      "tolerance_abs": 1e-6}}},
            }), encoding="utf-8")
            shards.append(shard)

        header, rows, per_shard = concat_csv(shards, "dissection_table.csv")
        ok = True
        checks = [
            ("headers agree", header == ["arm", "setting"]),
            ("rows concatenated (2+1)", len(rows) == 3),
            ("per-shard counts recorded", per_shard == {"shard_0": 2, "shard_1": 1}),
        ]
        # A header mismatch must abort rather than merge silently.
        (shards[1] / "dissection_table.csv").write_text("setting,arm\nETTh2-96,l_main\n",
                                                        encoding="utf-8")
        try:
            concat_csv(shards, "dissection_table.csv")
            checks.append(("header mismatch aborts", False))
        except SystemExit:
            checks.append(("header mismatch aborts", True))

        parity = merge_parity([
            {"passed": True, "probe_cells": 6, "cells_value_compared": 2,
             "checkpoint_path_mismatches": [{"a": 1}],
             "fields": {"x": {"max_abs_diff": 0.2, "comparisons": 1, "tolerance_abs": 1e-6}}},
            {"passed": False, "probe_cells": 6, "cells_value_compared": 1,
             "checkpoint_path_mismatches": [{"a": 2}],
             "fields": {"x": {"max_abs_diff": 0.5, "comparisons": 2, "tolerance_abs": 1e-6}}},
        ])
        checks += [
            ("probe cells summed (12)", parity["probe_cells"] == 12),
            ("value compared summed (3)", parity["cells_value_compared"] == 3),
            ("passed ANDed to False", parity["passed"] is False),
            ("mismatches concatenated (2)", len(parity["checkpoint_path_mismatches"]) == 2),
            ("per-field max taken (0.5)", parity["fields"]["x"]["max_abs_diff"] == 0.5),
            ("per-field comparisons summed (3)", parity["fields"]["x"]["comparisons"] == 3),
        ]
        for name, good in checks:
            print(f"  [{'OK  ' if good else 'FAIL'}] {name}")
            ok = ok and good
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="research_runs/phaseformer_L_e16_shard_*")
    ap.add_argument("--output-root",
                    default="research_runs/phaseformer_L_e16_dissection_v1")
    ap.add_argument("--expect-shards", type=int, default=21)
    ap.add_argument("--expect-cells", type=int, default=63)
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept fewer shards than the registered 21 (setting, arm) "
                         "combos.  Requires an explicit record of what is missing: the "
                         "merge stores the missing combos in merge_provenance and refuses "
                         "to look complete by omission.")
    ap.add_argument("--repo-root", default="",
                    help="root the shard glob and --output-root resolve against "
                         "(default: the repository root).  Exists so the merge can be "
                         "exercised against a fixture instead of the live run tree.")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    repo = (pathlib.Path(a.repo_root) if a.repo_root
            else pathlib.Path(__file__).resolve().parents[2])
    shards = sorted(pathlib.Path(repo).glob(a.shard_glob))
    if not shards:
        print(f"no shard directories matched {a.shard_glob}", file=sys.stderr)
        return 2
    print(f"shards found: {len(shards)} (expected {a.expect_shards})")
    if a.allow_partial:
        # A stopped shard leaves an empty directory (the tool writes its artifacts only
        # at the end), which must be counted as "not run" rather than aborting the
        # merge.  Its combo falls out of the present list and is recorded as missing.
        usable = [shard for shard in shards
                  if (shard / "dissection_table.csv").is_file()]
        skipped = [shard.name for shard in shards if shard not in usable]
        if skipped:
            print(f"  skipping {len(skipped)} shard dir(s) with no artifacts "
                  f"(stopped before writing): {skipped}")
        shards = usable
    for shard in shards:
        print(f"  {shard.name}")

    merged = {}
    per_shard_rows = {}
    for name in CSV_FILES:
        header, rows, counts = concat_csv(shards, name)
        merged[name] = (header, rows)
        per_shard_rows[name] = counts
        print(f"{name}: {len(rows)} row(s) merged")

    summaries = []
    for shard in shards:
        path = shard / SUMMARY
        if not path.is_file():
            print(f"{shard.name} lacks {SUMMARY}; refusing a partial merge", file=sys.stderr)
            return 2
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    parities = []
    for shard in shards:
        path = shard / PARITY
        if not path.is_file():
            print(f"{shard.name} lacks {PARITY}; refusing a partial merge", file=sys.stderr)
            return 2
        parities.append(json.loads(path.read_text(encoding="utf-8")))

    cells = [cell for summary in summaries for cell in (summary.get("cells") or [])]

    # The registered scope is 7 settings x 3 arms.  Derive which combos are present
    # from the shard roots themselves so a partial merge can name exactly what is
    # missing instead of quietly producing a smaller table.
    settings = ["ETTh2:96", "ETTh2:720", "ETTm2:96", "ETTm2:192",
                "Weather:96", "Weather:192", "Electricity:336"]
    arms = ["l_main", "l_q1_4", "l_q1_8"]
    registered = [(s.replace(":", "-"), arm) for s in settings for arm in arms]
    present = []
    for shard in shards:
        with (shard / "dissection_table.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row:
            present.append((row["setting"], row["arm"]))
    missing = [combo for combo in registered if combo not in present]
    extra = [combo for combo in present if combo not in registered]

    if len(cells) != a.expect_cells:
        if not a.allow_partial:
            print(f"shards declare {len(cells)} cells, expected {a.expect_cells} "
                  f"({len(missing)} registered combo(s) missing: {missing}); pass "
                  f"--allow-partial to assemble what exists and record the gap",
                  file=sys.stderr)
            return 2
        print(f"PARTIAL merge: {len(cells)} of {a.expect_cells} cells; "
              f"missing registered combos: {missing}")
    if extra:
        print(f"refusing to merge: shards carry combos outside the registered scope: "
              f"{extra}", file=sys.stderr)
        return 2

    modes = [summary.get("einsum_optimize") for summary in summaries]
    if len(set(modes)) != 1:
        print(f"shards disagree about the einsum kernel: {sorted(set(modes))}; "
              f"refusing to present a mixed-kernel artifact", file=sys.stderr)
        return 2

    dissection_rows = merged["dissection_table.csv"][1]
    intervention_rows = merged["intervention_table.csv"][1]
    arms_per_cell = {}
    for row in intervention_rows:
        key = (row.get("arm"), row.get("setting"), row.get("seed"))
        arms_per_cell.setdefault(key, set()).add(row.get("intervention_arm"))

    base = summaries[0]
    counts = dict(base.get("counts") or {})
    counts.update({
        "cells": len(cells),
        "dissection_rows": len(dissection_rows),
        "intervention_rows": len(intervention_rows),
        "intervention_arms_per_cell": sorted({len(v) for v in arms_per_cell.values()}),
        "cross_seed_rows": len(merged["cross_seed_alignment.csv"][1]),
        "canonical_mode_rows": len(merged["canonical_modes.csv"][1]),
        "semantic_alignment_rows": len(merged["semantic_alignment.csv"][1]),
    })
    verdicts = {}
    for summary in summaries:
        for name, value in ((summary.get("counts") or {}).get("verdicts") or {}).items():
            verdicts[name] = verdicts.get(name, 0) + int(value)
    if verdicts:
        counts["verdicts"] = verdicts

    invariants = {"per_cell": {}}
    for summary in summaries:
        block = summary.get("invariants") or {}
        for key in ("algebra_failures", "run_metric_failures"):
            invariants.setdefault(key, []).extend(block.get(key) or [])
        invariants.setdefault("run_metric_not_comparable", {}).update(
            block.get("run_metric_not_comparable") or {})
        invariants["per_cell"].update(block.get("per_cell") or {})
        for key in ("algebra_certificate", "algebra_tolerance_relative"):
            if key in block and key not in invariants:
                invariants[key] = block[key]

    parity = merge_parity(parities)
    summary_out = dict(base)
    summary_out.update({
        "cells": cells,
        "counts": counts,
        "invariants": invariants,
        # OR is the safe direction: one shard reading test would be catastrophic.
        "reads_test": any(bool(s.get("reads_test")) for s in summaries),
        "reference_parity": parity,
        "einsum_optimize": modes[0],
        "elapsed_seconds": sum(float(s.get("elapsed_seconds") or 0.0) for s in summaries),
        "merge_provenance": {
            "sharded": True,
            "rule": ("E16 was split into one (setting, arm) shard per process so that "
                     "all three seeds of a setting stay together, which the cross-seed "
                     "columns require. Rows are concatenated; counters are summed; "
                     "boolean invariants are ANDed/ORed in the safe direction; no "
                     "number is recomputed here."),
            "shards": [shard.name for shard in shards],
            "cells_per_shard": {shard.name: len(s.get("cells") or [])
                                for shard, s in zip(shards, summaries)},
            "partial": {
                "expected_cells": a.expect_cells,
                "actual_cells": len(cells),
                "missing_combos": missing,
                "note": ("The registered scope is 7 settings x 3 arms.  Any combo listed "
                         "in missing_combos has NO data: its experiments were not run (or "
                         "were stopped), so section 4.4's tables must mark those rows as "
                         "not run rather than leave them looking unfilled."),
            } if missing else None,
            "rows_per_shard": per_shard_rows,
            "bools": {"reference_parity_passed": parity["passed"],
                      "reads_test": summary_out["reads_test"]},
        },
    })

    root = pathlib.Path(repo) / a.output_root
    print(f"\nmerged: {len(cells)} cells, {len(dissection_rows)} dissection rows, "
          f"{len(intervention_rows)} intervention rows")
    print(f"parity: passed={parity['passed']} probe_cells={parity['probe_cells']} "
          f"value_compared={parity['cells_value_compared']} "
          f"path_mismatched={len(parity['checkpoint_path_mismatches'])}")
    print(f"kernel: einsum_optimize={modes[0]}")

    if not a.write:
        print("\ndry run; pass --write to assemble the canonical artifacts")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    for name in CSV_FILES:
        header, rows = merged[name]
        write_csv(root / name, header, rows)
    (root / SUMMARY).write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=False, sort_keys=True,
                   default=str) + "\n", encoding="utf-8")
    (root / PARITY).write_text(
        json.dumps(parity, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\nwrote the canonical artifacts to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
