#!/usr/bin/env python3
"""Fifty-configuration H192 validation screen for strict one-stage PCTF.

The reference is the already completed two-stage Full Repair matrix.  This
script deliberately never passes ``--evaluate-test``: every one of the ten
training policies is selected from the same held-out validation split, paired
by dataset and seed with the historical Full Repair run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "research_runs/pctf_anchor_formal_etts_v1"
OUTPUT_ROOT = "research_runs/pctf_single_stage_h192_tuning_v2"
MECHANISM = "pctf_anchor_repair_full"
SETTINGS = (("ETTh2", 192, 48), ("ETTm2", 192, 96))
SEEDS = (2021, 2022)

def _policies():
    """Return exactly fifty pre-registered, interpretable configurations."""
    values = {"T00_strict_base": dict()}

    def add(prefix, rows):
        for suffix, overrides in rows:
            name = f"T{len(values):02d}_{prefix}_{suffix}"
            if name in values:
                raise AssertionError(f"duplicate policy {name}")
            values[name] = overrides

    # One-factor scans: correction optimization, component supervision and
    # trust-region size.  The ranges are centered on the prior T0 default.
    add("composer_lr", [
        ("025", dict(anchored_pctf_composer_lr_scale=0.25)),
        ("040", dict(anchored_pctf_composer_lr_scale=0.40)),
        ("060", dict(anchored_pctf_composer_lr_scale=0.60)),
        ("080", dict(anchored_pctf_composer_lr_scale=0.80)),
        ("120", dict(anchored_pctf_composer_lr_scale=1.20)),
        ("150", dict(anchored_pctf_composer_lr_scale=1.50)),
        ("200", dict(anchored_pctf_composer_lr_scale=2.00)),
        ("250", dict(anchored_pctf_composer_lr_scale=2.50)),
    ])
    add("equal_aux", [
        ("000", dict(anchored_pctf_shape_aux_weight=0.0, anchored_pctf_level_aux_weight=0.0)),
        ("010", dict(anchored_pctf_shape_aux_weight=0.01, anchored_pctf_level_aux_weight=0.01)),
        ("025", dict(anchored_pctf_shape_aux_weight=0.025, anchored_pctf_level_aux_weight=0.025)),
        ("075", dict(anchored_pctf_shape_aux_weight=0.075, anchored_pctf_level_aux_weight=0.075)),
        ("100", dict(anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.10)),
        ("150", dict(anchored_pctf_shape_aux_weight=0.15, anchored_pctf_level_aux_weight=0.15)),
        ("200", dict(anchored_pctf_shape_aux_weight=0.20, anchored_pctf_level_aux_weight=0.20)),
    ])
    add("gate_aux", [
        ("000", dict(anchored_pctf_gate_aux_weight=0.0)),
        ("010", dict(anchored_pctf_gate_aux_weight=0.01)),
        ("025", dict(anchored_pctf_gate_aux_weight=0.025)),
        ("075", dict(anchored_pctf_gate_aux_weight=0.075)),
        ("100", dict(anchored_pctf_gate_aux_weight=0.10)),
        ("150", dict(anchored_pctf_gate_aux_weight=0.15)),
    ])
    add("trust", [
        ("010", dict(anchored_pctf_correction_max=0.10, anchored_pctf_deformation_max=0.04, anchored_pctf_global_level_max=0.02)),
        ("015", dict(anchored_pctf_correction_max=0.15, anchored_pctf_deformation_max=0.06, anchored_pctf_global_level_max=0.03)),
        ("020", dict(anchored_pctf_correction_max=0.20, anchored_pctf_deformation_max=0.08, anchored_pctf_global_level_max=0.04)),
        ("030", dict(anchored_pctf_correction_max=0.30, anchored_pctf_deformation_max=0.12, anchored_pctf_global_level_max=0.06)),
        ("035", dict(anchored_pctf_correction_max=0.35, anchored_pctf_deformation_max=0.14, anchored_pctf_global_level_max=0.07)),
        ("045", dict(anchored_pctf_correction_max=0.45, anchored_pctf_deformation_max=0.18, anchored_pctf_global_level_max=0.09)),
        ("060", dict(anchored_pctf_correction_max=0.60, anchored_pctf_deformation_max=0.24, anchored_pctf_global_level_max=0.12)),
    ])
    # Shape and cycle-level residuals are identifiable subspaces; asymmetry
    # tests whether the longer H192 error is dominated by one of them.
    add("asymmetric_aux", [
        ("shape100", dict(anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.05)),
        ("shape025", dict(anchored_pctf_shape_aux_weight=0.025, anchored_pctf_level_aux_weight=0.05)),
        ("level100", dict(anchored_pctf_shape_aux_weight=0.05, anchored_pctf_level_aux_weight=0.10)),
        ("level025", dict(anchored_pctf_shape_aux_weight=0.05, anchored_pctf_level_aux_weight=0.025)),
        ("shape100_level025", dict(anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.025)),
        ("shape025_level100", dict(anchored_pctf_shape_aux_weight=0.025, anchored_pctf_level_aux_weight=0.10)),
    ])
    # These combinations are fixed before execution, rather than chosen after
    # observing the preceding one-factor rows.
    add("joint", [
        ("lr060_aux100", dict(anchored_pctf_composer_lr_scale=0.60, anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.10)),
        ("lr150_aux100", dict(anchored_pctf_composer_lr_scale=1.50, anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.10)),
        ("lr060_narrow", dict(anchored_pctf_composer_lr_scale=0.60, anchored_pctf_correction_max=0.15, anchored_pctf_deformation_max=0.06, anchored_pctf_global_level_max=0.03)),
        ("lr150_narrow", dict(anchored_pctf_composer_lr_scale=1.50, anchored_pctf_correction_max=0.15, anchored_pctf_deformation_max=0.06, anchored_pctf_global_level_max=0.03)),
        ("lr060_gate100", dict(anchored_pctf_composer_lr_scale=0.60, anchored_pctf_gate_aux_weight=0.10)),
        ("lr150_gate100", dict(anchored_pctf_composer_lr_scale=1.50, anchored_pctf_gate_aux_weight=0.10)),
        ("wide_lr060_aux100", dict(anchored_pctf_composer_lr_scale=0.60, anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.10, anchored_pctf_correction_max=0.35, anchored_pctf_deformation_max=0.14, anchored_pctf_global_level_max=0.07)),
        ("wide_lr150_aux100", dict(anchored_pctf_composer_lr_scale=1.50, anchored_pctf_shape_aux_weight=0.10, anchored_pctf_level_aux_weight=0.10, anchored_pctf_correction_max=0.35, anchored_pctf_deformation_max=0.14, anchored_pctf_global_level_max=0.07)),
        ("narrow_lr060_gate100", dict(anchored_pctf_composer_lr_scale=0.60, anchored_pctf_gate_aux_weight=0.10, anchored_pctf_correction_max=0.15, anchored_pctf_deformation_max=0.06, anchored_pctf_global_level_max=0.03)),
    ])
    add("convergence", [
        ("e36", dict(max_epochs=36)),
        ("e45", dict(max_epochs=45)),
        ("e60", dict(max_epochs=60)),
        ("e45_p12", dict(max_epochs=45, patience=12)),
        ("e60_p16", dict(max_epochs=60, patience=16)),
        ("warm3", dict(anchored_pctf_correction_warmup_epochs=3)),
    ])
    if len(values) != 50:
        raise AssertionError(f"expected 50 policies, got {len(values)}")
    return values


POLICIES = _policies()

STRICT_DEFAULTS = dict(
    anchored_pctf_anchor_lr_scale=1.0,
    anchored_pctf_composer_lr_scale=1.0,
    anchored_pctf_anchor_loss_weight=1.0,
    anchored_pctf_decouple_anchor_gradient=True,
    anchored_pctf_detach_composer_inputs=True,
    anchored_pctf_correction_warmup_epochs=0,
)


def _root(args):
    path = Path(args.output_root)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_one(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row: {path}")
    return rows[0]


def _reference():
    path = REFERENCE_ROOT / "formal_details.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen Full Repair reference: {path}")
    result = {}
    for row in csv.DictReader(path.open()):
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]))
        if key[:2] not in {(dataset, horizon) for dataset, horizon, _ in SETTINGS}:
            continue
        if row["model"] != MECHANISM:
            continue
        if key[2] not in SEEDS:
            continue
        result[key] = row
    expected = {
        (dataset, horizon, seed)
        for dataset, horizon, _ in SETTINGS for seed in SEEDS
    }
    if set(result) != expected:
        raise RuntimeError(f"incomplete frozen reference: missing={expected - set(result)}")
    return result


def _command(args, policy, dataset, horizon, cycle_period, seed, *, smoke=False):
    values = dict(STRICT_DEFAULTS)
    policy_values = dict(POLICIES[policy])
    max_epochs = 1 if smoke else int(policy_values.pop("max_epochs", args.max_epochs))
    values.update(policy_values)
    command = [
        sys.executable, "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "finalist",
        "--mechanism", MECHANISM,
        "--period", "24",
        "--lookback", "720",
        "--cycle-period", str(cycle_period),
        # ETT H192 with batch size 256 has no complete batch at 5% train.
        # Thirty percent remains a cheap smoke while guaranteeing the audit
        # batch and one real optimizer step exist for both ETT frequencies.
        "--percent", "30" if smoke else "100",
        "--max-epochs", str(max_epochs),
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(
            _root(args) / ("smoke" if smoke else "candidates") / policy
        ),
        "--overrides", json.dumps(values, sort_keys=True),
        "--require-cuda", "--resume",
    ]
    if args.progress:
        command.append("--progress")
    return command


def commands(args):
    return [
        _command(args, policy, dataset, horizon, cycle, seed)
        for policy in POLICIES
        for dataset, horizon, cycle in SETTINGS
        for seed in SEEDS
    ]


def smoke_commands(args):
    # T00 validates the default strict graph, while the last policy exercises
    # the one-stage curriculum.  Both period geometries are covered.
    smoke_policies = (next(iter(POLICIES)), tuple(POLICIES)[-1])
    return [
        _command(args, policy, dataset, horizon, cycle, 2021, smoke=True)
        for policy in smoke_policies
        for dataset, horizon, cycle in SETTINGS
    ]


def _collect_policy(args, policy):
    expected = {
        (dataset, horizon, seed)
        for dataset, horizon, _ in SETTINGS for seed in SEEDS
    }
    found, environments = {}, set()
    run_root = _root(args) / "candidates" / policy / "runs"
    for path in run_root.glob("*/metrics.csv"):
        row = _read_one(path)
        # A prior smoke may exist in the same root.  It is evidence for the
        # launch path, never an admissible full-train tuning observation.
        if int(row.get("percent", -1)) != 100:
            continue
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]))
        if key not in expected:
            continue
        if key in found:
            raise RuntimeError(f"duplicate candidate result: {policy} {key}")
        if row.get("test_mse") or row.get("test_mae"):
            raise RuntimeError(f"test leakage in {policy} {key}")
        if row["mechanism"] != MECHANISM or row["device_type"] != "cuda":
            raise RuntimeError(f"inadmissible candidate run: {policy} {key}")
        found[key] = row
        environment = json.loads((path.parent / "environment.json").read_text())
        environments.add((environment["gpu"], environment["torch"], environment["lightning"]))
    if set(found) != expected:
        raise RuntimeError(f"incomplete policy {policy}: missing={expected - set(found)}")
    if len(environments) != 1:
        raise RuntimeError(f"mixed environments in {policy}: {environments}")
    return found, next(iter(environments))


def summarize(args):
    reference = _reference()
    details, summary = [], []
    reference_environment = None
    for policy in POLICIES:
        rows, environment = _collect_policy(args, policy)
        if reference_environment is None:
            reference_environment = environment
        elif environment != reference_environment:
            raise RuntimeError("mixed candidate environments")
        ratios = []
        both_improved = 0
        for dataset, horizon, _ in SETTINGS:
            for seed in SEEDS:
                key = (dataset, horizon, seed)
                candidate, full = rows[key], reference[key]
                mse_ratio = float(candidate["val_mse"]) / float(full["val_mse"])
                mae_ratio = float(candidate["val_mae"]) / float(full["val_mae"])
                both_improved += mse_ratio < 1.0 and mae_ratio < 1.0
                ratios.extend((mse_ratio, mae_ratio))
                details.append({
                    "policy": policy, "dataset": dataset, "horizon": horizon,
                    "seed": seed,
                    "full_repair_val_mse": full["val_mse"],
                    "full_repair_val_mae": full["val_mae"],
                    "candidate_val_mse": candidate["val_mse"],
                    "candidate_val_mae": candidate["val_mae"],
                    "mse_ratio_vs_full_repair": mse_ratio,
                    "mae_ratio_vs_full_repair": mae_ratio,
                    "epochs_completed": candidate["epochs_completed"],
                    "elapsed_sec": candidate["elapsed_sec"],
                    "parameter_count": candidate["parameter_count"],
                })
        summary.append({
            "policy": policy,
            "combined_macro_ratio_vs_full_repair": statistics.mean(ratios),
            "combined_improvement_pct_vs_full_repair": 100 * (1 - statistics.mean(ratios)),
            "mse_macro_ratio_vs_full_repair": statistics.mean(ratios[::2]),
            "mae_macro_ratio_vs_full_repair": statistics.mean(ratios[1::2]),
            "worst_ratio_vs_full_repair": max(ratios),
            "double_improve_runs": both_improved,
            "mean_elapsed_sec": statistics.mean(float(row["elapsed_sec"]) for row in rows.values()),
        })
    summary.sort(key=lambda item: item["combined_macro_ratio_vs_full_repair"])
    winner = summary[0]
    # Passing means at least 0.5% lower combined validation error while no
    # paired H192 metric regresses over 0.5%.  This is deliberately stricter
    # than the user's aggregate target before authorizing any test read.
    eligible = (
        winner["combined_macro_ratio_vs_full_repair"] <= 0.995
        and winner["worst_ratio_vs_full_repair"] <= 1.005
    )
    root = _root(args)
    _write_csv(root / "tuning_details.csv", details)
    _write_csv(root / "tuning_summary.csv", summary)
    (root / "tuning_decision.json").write_text(json.dumps({
        "protocol": "pctf-single-stage-strict-h192-fifty-policy-v2",
        "selection_source": "validation_only",
        "test_metrics_read": False,
        "reference": "two-stage Full Repair, frozen formal_details.csv validation fields",
        "policies": list(POLICIES),
        "settings": ["ETTh2-H192", "ETTm2-H192"],
        "seeds": list(SEEDS),
        "candidate_environment": reference_environment,
        "winner": winner["policy"],
        "winner_eligible_for_independent_test": eligible,
        "gate": {"combined_ratio_max": 0.995, "worst_ratio_max": 1.005},
    }, indent=2) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry", "smoke", "run", "summarize"), required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize":
        return summarize(args)
    planned = smoke_commands(args) if args.stage == "smoke" else commands(args)
    print(f"commands={len(planned)}")
    for command in planned:
        print(shlex.join(command))
    if args.stage in ("smoke", "run"):
        for index, command in enumerate(planned, 1):
            print(f"RUN {index}/{len(planned)}")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
