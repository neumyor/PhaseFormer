#!/usr/bin/env python3
"""E17 write-back: build the minipaper §4.5 four-arm table.

Consumes the test-bearing results CSV that ``read_test_generic.py`` produces
(``results.with_test.csv``) and emits the §4.5 rows plus the claims block.

The §4.5 table compares four arms per test-selected setting:

* ``direct``                        -- the unconstrained jointly trained corrector;
* ``frozen_independent_direction_1`` -- the independent-RRR direction 1 frozen as
  a bottleneck (E8's ``keep_direction_1`` lineage for 6 settings, newly trained
  for Electricity-336);
* ``frozen_conditional_direction_1`` -- the NEW arm, the conditional-RRR
  direction 1 (``D_cond = y - y_phi``) frozen the same way;
* ``joint`` (= PhaseFormer-L)        -- the same configuration as ``direct``.

Two properties are reported rather than asserted away:

1. ``direct`` and ``joint`` are the SAME configuration here, so those two columns
   carry identical numbers by construction -- not two independent experiments;
2. the conditional and independent leading directions coincide on 6 of the 7
   settings (projector audit: |cos| >= 0.9991), so the frozen-conditional versus
   frozen-independent contrast is only a genuine test on Electricity-336, where
   they are nearly orthogonal (|cos| = 0.0066).  The projector audit's cosines are
   carried into the output so the table note can state which settings actually
   discriminate.

Usage::

    python scripts/phaseformer_L/e17_writeback.py \\
        --results research_runs/phaseformer_L_e17_conditional_v1/results.with_test.csv \\
        --projector-audit research_runs/phaseformer_L_e17_conditional_v1/projectors/projector_audit.json \\
        --output-root research_runs/phaseformer_L_e17_conditional_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("direct", "frozen_independent_direction_1",
        "frozen_conditional_direction_1", "joint")
ARM_LABEL = {
    "direct": "direct",
    "frozen_independent_direction_1": "冻结独立-RRR 方向 1",
    "frozen_conditional_direction_1": "冻结条件-RRR 方向 1",
    "joint": "PhaseFormer-L（联合）",
}
# Above this |cos| the two frozen directions are the same object in practice, so
# the arm pair cannot separate "frozen hurts" from "the target definition moved".
REDUNDANT_DIRECTION_COS = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--projector-audit", default="")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_results(path: Path) -> dict:
    index: dict = {}
    missing: list = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (row["dataset"], int(row["horizon"]), row["arm"])
            except (KeyError, TypeError, ValueError):
                continue
            entry = index.setdefault(key, {"mse": [], "mae": [], "seeds": [],
                                           "h1": None, "source": set()})
            mse = str(row.get("test_mse", "")).strip()
            mae = str(row.get("test_mae", "")).strip()
            if not mse or not mae:
                missing.append(f"{row['arm']} {row['dataset']}-{row['horizon']}"
                               f" s{row.get('seed')}")
                continue
            try:
                entry["mse"].append(float(mse))
                entry["mae"].append(float(mae))
            except ValueError:
                continue
            entry["seeds"].append(int(row["seed"]))
            if row.get("source"):
                entry["source"].add(str(row["source"]))
            if row.get("h1_cond_gt_indep_seed_majority"):
                entry["h1"] = str(row["h1_cond_gt_indep_seed_majority"])
    return index, missing


def read_projector_cosines(path: Path) -> dict:
    """(dataset, horizon) -> |cos| between the conditional and independent Q1."""
    if not path or not Path(path).exists():
        return {}
    try:
        audit = json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for entry in audit:
        revin = entry.get("revin") or {}
        cosine = revin.get("abs_cos_conditional_vs_independent")
        if cosine is None:
            continue
        out[(str(entry["dataset"]), int(entry["horizon"]))] = float(cosine)
    return out


def stat(values):
    if not values:
        return None, None, 0
    if len(values) == 1:
        return float(values[0]), 0.0, 1
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    index, missing = read_results(results_path)
    cosines = read_projector_cosines(Path(args.projector_audit) if args.projector_audit
                                     else None)

    settings = sorted({(d, h) for (d, h, _) in index},
                      key=lambda k: (k[0], k[1]))
    rows = []
    for dataset, horizon in settings:
        entry = {"dataset": dataset, "horizon": horizon,
                 "setting": f"{dataset}-{horizon}"}
        for arm in ARMS:
            bucket = index.get((dataset, horizon, arm))
            mean, sd, n = stat(bucket["mse"] if bucket else [])
            mean_mae, sd_mae, _ = stat(bucket["mae"] if bucket else [])
            entry[f"{arm}_mse"] = None if mean is None else round(mean, 6)
            entry[f"{arm}_mae"] = None if mean_mae is None else round(mean_mae, 6)
            entry[f"{arm}_mse_std"] = None if sd is None else round(sd, 6)
            entry[f"{arm}_mae_std"] = None if sd_mae is None else round(sd_mae, 6)
            entry[f"{arm}_seeds"] = n
            entry[f"{arm}_source"] = ("|".join(sorted(bucket["source"]))
                                      if bucket and bucket["source"] else None)
        # Deltas of the frozen arms against direct, on both metrics.
        for arm in ("frozen_independent_direction_1",
                    "frozen_conditional_direction_1", "joint"):
            for metric in ("mse", "mae"):
                base = entry.get(f"direct_{metric}")
                value = entry.get(f"{arm}_{metric}")
                key = f"{arm}_vs_direct_{metric}_pct"
                entry[key] = (None if base in (None, 0) or value is None
                              else round(100.0 * (value - base) / base, 4))
        entry["h1_seed_majority"] = (
            index.get((dataset, horizon, "direct"), {}).get("h1"))
        cosine = cosines.get((dataset, horizon))
        entry["conditional_vs_independent_direction_cos"] = (
            None if cosine is None else round(cosine, 6))
        entry["frozen_arms_are_distinct"] = (
            None if cosine is None else bool(cosine < REDUNDANT_DIRECTION_COS))
        rows.append(entry)

    direct_equals_joint = all(
        r["direct_mse"] == r["joint_mse"] and r["direct_mae"] == r["joint_mae"]
        for r in rows)
    discriminating = [r["setting"] for r in rows
                      if r["frozen_arms_are_distinct"]]
    redundant = [r["setting"] for r in rows
                 if r["frozen_arms_are_distinct"] is False]
    h1_supported = [r["setting"] for r in rows
                    if r["h1_seed_majority"] not in (None, "", "False", "evidence_missing")]
    h1_missing = [r["setting"] for r in rows
                  if r["h1_seed_majority"] == "evidence_missing"]

    lines = []
    for row in rows:
        def fmt(value):
            return "—" if value is None else f"{value:.3f}"
        lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            row["dataset"], row["horizon"],
            fmt(row["direct_mse"]) + "/" + fmt(row["direct_mae"]),
            fmt(row["frozen_independent_direction_1_mse"]) + "/" +
            fmt(row["frozen_independent_direction_1_mae"]),
            fmt(row["frozen_conditional_direction_1_mse"]) + "/" +
            fmt(row["frozen_conditional_direction_1_mae"]),
            fmt(row["joint_mse"]) + "/" + fmt(row["joint_mae"]),
            row["h1_seed_majority"] or "—",
        ))

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["setting"]
    with (out_root / "conditional_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "conditional_table.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    summary = {
        "experiment": "E17 write-back (minipaper section 4.5)",
        "rows": len(rows),
        "cells_without_test_metrics": missing,
        "direct_equals_joint": direct_equals_joint,
        "direct_equals_joint_note": (
            "PhaseFormer-L's corrector IS the jointly trained unconstrained head, "
            "so the direct and joint columns are the same configuration and their "
            "numbers coincide by construction." if direct_equals_joint else
            "columns differ; investigate before writing them up"),
        "discriminating_settings": discriminating,
        "redundant_direction_settings": redundant,
        "direction_cosine_note": (
            "The conditional and independent leading directions are the same "
            "object where the cosine is near 1, so the two frozen arms cannot "
            "there separate 'freezing hurts' from 'the target definition moved'. "
            "The contrast is a genuine test only on the discriminating settings."
        ),
        "h1_supported_settings": h1_supported,
        "h1_evidence_missing_settings": h1_missing,
        "disclosures": [
            "the seven settings are test-set-selection-derived, not a blind sample",
            "reused cells keep their Stage-0 frozen (gate, lr); the new cells use "
            "the same per-setting frozen values so the two frozen arms stay "
            "hyperparameter-matched, while direct/joint come from E14 and mix D-2 "
            "defaults with Stage-0 values",
            "the frozen projectors are computed on the train split only",
            "H1 is quoted from the registered conditional_rrr_alignment.csv; "
            "Electricity-336 has no H1 evidence and is reported as evidence_missing",
            "no test was read during stage A; the numbers here come from the single "
            "read performed by read_test_generic.py",
        ],
    }
    (out_root / "conditional_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished", "rows": len(rows),
        "missing_test_cells": len(missing),
        "direct_equals_joint": direct_equals_joint,
        "discriminating_settings": discriminating,
        "h1_supported": len(h1_supported),
        "h1_evidence_missing": h1_missing,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
