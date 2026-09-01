#!/usr/bin/env python3
"""Ten-configuration H192 validation screen for strict one-stage PCTF.

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
OUTPUT_ROOT = "research_runs/pctf_single_stage_h192_tuning_v1"
MECHANISM = "pctf_anchor_repair_full"
SETTINGS = (("ETTh2", 192, 48), ("ETTm2", 192, 96))
SEEDS = (2021, 2022)

# The first row is the strict-gradient repair itself.  The next three isolate
# ICPT/composer optimization speed; the following three isolate supervision;
# the last three search the correction trust region and available convergence
# time.  All rows preserve one random-init Trainer.fit and the same A2 anchor.
POLICIES = {
    "T0_strict_base": dict(),
    "T1_composer_lr_half": dict(anchored_pctf_composer_lr_scale=0.5),
    "T2_composer_lr_1p5": dict(anchored_pctf_composer_lr_scale=1.5),
    "T3_composer_lr_2p0": dict(anchored_pctf_composer_lr_scale=2.0),
    "T4_aux_low": dict(
        anchored_pctf_shape_aux_weight=0.025,
        anchored_pctf_level_aux_weight=0.025,
    ),
    "T5_aux_high": dict(
        anchored_pctf_shape_aux_weight=0.10,
        anchored_pctf_level_aux_weight=0.10,
    ),
    "T6_gate_high": dict(anchored_pctf_gate_aux_weight=0.10),
    "T7_narrow_trust_region": dict(
        anchored_pctf_correction_max=0.15,
        anchored_pctf_deformation_max=0.06,
        anchored_pctf_global_level_max=0.03,
    ),
    "T8_wide_trust_region": dict(
        anchored_pctf_correction_max=0.35,
        anchored_pctf_deformation_max=0.14,
        anchored_pctf_global_level_max=0.07,
    ),
    "T9_long_budget": dict(max_epochs=45),
}

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


def _command(args, policy, dataset, horizon, cycle_period, seed):
    values = dict(STRICT_DEFAULTS)
    policy_values = dict(POLICIES[policy])
    max_epochs = int(policy_values.pop("max_epochs", args.max_epochs))
    values.update(policy_values)
    command = [
        sys.executable, "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "single-stage-h192-tune",
        "--mechanism", MECHANISM,
        "--period", "24",
        "--lookback", "720",
        "--cycle-period", str(cycle_period),
        "--percent", "100",
        "--max-epochs", str(max_epochs),
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(_root(args) / "candidates" / policy),
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


def _collect_policy(args, policy):
    expected = {
        (dataset, horizon, seed)
        for dataset, horizon, _ in SETTINGS for seed in SEEDS
    }
    found, environments = {}, set()
    run_root = _root(args) / "candidates" / policy / "runs"
    for path in run_root.glob("*/metrics.csv"):
        row = _read_one(path)
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
        "protocol": "pctf-single-stage-strict-h192-ten-policy-v1",
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
    parser.add_argument("--stage", choices=("dry", "run", "summarize"), required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize":
        return summarize(args)
    planned = commands(args)
    print(f"commands={len(planned)}")
    for command in planned:
        print(shlex.join(command))
    if args.stage == "run":
        for index, command in enumerate(planned, 1):
            print(f"RUN {index}/{len(planned)}")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
