#!/usr/bin/env python3
"""E14 reuse ambiguity audit: every candidate run per reused cell.

The reuse resolver takes the first match while walking a whitelist of roots, so
it is a *choice*, not a lookup.  That became concrete when a broken batch was
silently accepted for ETTh2-96 seed 2023 (its config had the SEED written into
the gate init and a 500x learning rate; see
docs/PhaseFormer_L/e14_main/05_audit.md section 6).  Guarding against invalid
configs fixed that case, but it does not answer the next question: for how many
cells is there more than one *valid* candidate, and do those candidates actually
agree?

This script enumerates, for every declared reuse cell, all runs under the
whitelisted roots that pass the full validation, reports the one the manifest
actually chose, and flags cells where valid candidates disagree on the recorded
gate init, learning rate, or validation MSE.  A cell whose candidates agree is
insensitive to the tie-break; a cell whose candidates disagree is a disclosed
choice that the §4.2 table note must carry.

Reads configs, metrics and the manifest only.  No dataset, no test split.

Usage::

    python scripts/phaseformer_L/e14_reuse_audit.py \\
        --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \\
        --output-root research_runs/phaseformer_L_e14_main_v1
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

from scripts.phaseformer_L.e14_main_matrix import (  # noqa: E402
    REUSE_ROOT_WHITELIST,
    REUSE_SCOPE,
    SEEDS,
    _arm_match,
    _protocol_ok,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--metrics-round", type=int, default=8,
        help="decimals used when deciding whether candidates agree")
    return parser.parse_args()


def first_metrics_row(run_dir: Path):
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), {}) or {}


def _num(value):
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_candidates(arm: str):
    """(dataset, horizon, seed) -> [candidate dict, ...] over whitelisted roots."""
    wanted = {(d, h, s) for (d, h) in REUSE_SCOPE.get(arm, ()) for s in SEEDS}
    if not wanted:
        return {}
    found: dict = {}
    for root_rel in REUSE_ROOT_WHITELIST:
        root = ROOT / root_rel
        if not root.is_dir():
            continue
        for config_path in sorted(root.glob("runs/*/config.json")):
            try:
                config = json.loads(config_path.read_text())
            except json.JSONDecodeError:
                continue
            key = (str(config.get("dataset")), int(config.get("horizon", -1)),
                   int(config.get("seed", -1)))
            if key not in wanted or not _arm_match(config, arm):
                continue
            ok, failures = _protocol_ok(config)
            run_dir = config_path.parent
            record = first_metrics_row(run_dir)
            hyper = config.get("hyperparams", {}) or {}
            found.setdefault(key, []).append({
                "root": root_rel,
                "config_hash": config.get("config_hash"),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "valid": bool(ok),
                "failures": failures,
                "gate_init": hyper.get("weak_period_residual_gate_init"),
                "learning_rate": hyper.get("learning_rate"),
                "has_test_metrics": bool(str(record.get("test_mse", "")).strip()),
                "val_mse": _num(record.get("val_mse")),
                "test_mse": _num(record.get("test_mse")),
            })
    return found


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    chosen = {}
    for cell in manifest.get("cells", []):
        if cell.get("status") != "reused":
            continue
        source = cell.get("source") or {}
        chosen[(str(cell["arm"]), str(cell["dataset"]), int(cell["horizon"]),
                int(cell["seed"]))] = str(source.get("run_dir", ""))

    rows = []
    for arm in sorted(REUSE_SCOPE):
        for key, candidates in sorted(collect_candidates(arm).items(),
                                     key=lambda item: str(item[0])):
            valid = [c for c in candidates if c["valid"]]
            cell_key = (arm, key[0], key[1], key[2])
            picked = chosen.get(cell_key, "")
            pick = next((c for c in candidates if c["run_dir"] == picked), None)

            def distinct(field, pool):
                values = {json.dumps(c[field], sort_keys=True) for c in pool}
                return values if len(values) > 1 else set()

            disagree_gate = distinct("gate_init", valid)
            disagree_lr = distinct("learning_rate", valid)
            vals = [round(c["val_mse"], args.metrics_round)
                    for c in valid if c["val_mse"] is not None]
            disagree_val = len(set(vals)) > 1

            rows.append({
                "arm": arm,
                "dataset": key[0],
                "horizon": key[1],
                "seed": key[2],
                "setting": f"{key[0]}-{key[1]}",
                "n_candidates": len(candidates),
                "n_valid_candidates": len(valid),
                "n_invalid_candidates": len(candidates) - len(valid),
                "chosen_run_dir": picked,
                "chosen_valid": None if pick is None else bool(pick["valid"]),
                "chosen_in_whitelist_scan": pick is not None,
                "chosen_root": None if pick is None else pick["root"],
                "valid_candidates_disagree_on_gate_init": bool(disagree_gate),
                "valid_candidates_disagree_on_learning_rate": bool(disagree_lr),
                "valid_candidates_disagree_on_val_mse": bool(disagree_val),
                "ambiguous_choice": bool(disagree_gate or disagree_lr or disagree_val),
                "distinct_gate_inits": json.dumps(sorted(g for g in
                                                         (c["gate_init"] for c in valid))),
                "distinct_learning_rates": json.dumps(sorted(
                    lr for lr in (c["learning_rate"] for c in valid))),
                "val_mse_spread": (round(max(vals) - min(vals), 8)
                                   if len(vals) > 1 else 0.0),
                "candidate_run_dirs": json.dumps(
                    [c["run_dir"] for c in valid], ensure_ascii=False),
            })

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "arm", "dataset", "horizon", "seed", "setting", "n_candidates"]
    with (out_root / "reuse_ambiguity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    multi = [r for r in rows if r["n_valid_candidates"] > 1]
    ambiguous = [r for r in rows if r["ambiguous_choice"]]
    missing = [r for r in rows if not r["chosen_in_whitelist_scan"]]
    summary = {
        "cells_audited": len(rows),
        "cells_with_multiple_valid_candidates": len(multi),
        "cells_with_disagreeing_candidates": len(ambiguous),
        "cells_whose_chosen_run_is_not_in_the_scan": len(missing),
        "cells_with_invalid_candidates_rejected": sum(
            1 for r in rows if r["n_invalid_candidates"]),
        "worst_val_mse_spread": (round(max(r["val_mse_spread"] for r in rows), 8)
                                 if rows else 0.0),
        "ambiguous_cells": [f"{r['arm']}__{r['setting']}-s{r['seed']}"
                            for r in ambiguous],
        "not_in_scan": [f"{r['arm']}__{r['setting']}-s{r['seed']}" for r in missing],
        "interpretation": (
            "A cell with one valid candidate is a lookup, not a choice. A cell "
            "whose valid candidates agree on gate init, learning rate and val_mse "
            "is insensitive to the tie-break. Only cells with disagreeing valid "
            "candidates are genuine choices, and those must be disclosed in the "
            "4.2 table note."
        ),
    }
    (out_root / "reuse_ambiguity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished",
        "cells_audited": summary["cells_audited"],
        "multiple_valid": summary["cells_with_multiple_valid_candidates"],
        "disagreeing": summary["cells_with_disagreeing_candidates"],
        "not_in_scan": summary["cells_whose_chosen_run_is_not_in_the_scan"],
        "invalid_rejected_cells": summary["cells_with_invalid_candidates_rejected"],
        "worst_val_mse_spread": summary["worst_val_mse_spread"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
