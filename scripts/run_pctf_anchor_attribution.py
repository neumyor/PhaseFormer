#!/usr/bin/env python3
"""Validation-only causal attribution for the repaired A2-anchored PCTF.

This runner deliberately stops before any test-set evaluation.  It first trains
one matched A2 anchor for every setting/seed, then initializes every candidate
from that exact checkpoint.  Frozen-anchor rows identify optimization drift;
the remaining rows test residual supervision, anchor-safe joint optimization,
marginal gate supervision, and the one-cycle level-space repair in sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INCUMBENT = "rcrf_pe_lff"
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 192),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 192),
    ("Electricity", 96),
)
SEEDS = (2021, 2022)
CYCLE_PERIODS = {
    "ETTh2": 48,
    "ETTm2": 96,
    "Weather": 24,
    "Electricity": 12,
}
CURRENT_CONTROL = "pctf_anchor_mlp"
FROZEN_MODES = (
    "pctf_anchor_diag_frozen_absolute",
    "pctf_anchor_diag_frozen_residual",
)
JOINT_MODES = (
    "pctf_anchor_repair_joint_residual",
    "pctf_anchor_repair_joint_marginal",
    "pctf_anchor_repair_full",
)
CANDIDATES = (CURRENT_CONTROL,) + FROZEN_MODES + JOINT_MODES
DEFAULT_OUTPUT_ROOT = "research_runs/pctf_anchor_attribution_v3"


def _root(args):
    value = Path(args.output_root)
    return value if value.is_absolute() else REPO_ROOT / value


def _parse_seeds(value):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _command(args, dataset, horizon, seed, mechanism, output_stage, checkpoint=None):
    command = [
        sys.executable,
        "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "anchor_attribution",
        "--mechanism", mechanism,
        "--period", "24",
        "--lookback", "720",
        "--cycle-period", str(CYCLE_PERIODS[dataset]),
        "--percent", "30",
        "--max-epochs", "12",
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(_root(args) / output_stage),
        "--require-cuda",
        "--resume",
    ]
    if checkpoint is not None:
        command.extend(("--init-checkpoint", str(checkpoint)))
    if args.progress:
        command.append("--progress")
    return command


def anchor_commands(args):
    seeds = _parse_seeds(args.seeds)
    return [
        _command(args, dataset, horizon, seed, INCUMBENT, "anchors")
        for dataset, horizon in SETTINGS
        for seed in seeds
    ]


def _read_single_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one metrics row: {path}")
    return rows[0]


def _environment(path):
    environment_path = path.parent / "environment.json"
    if not environment_path.is_file():
        raise RuntimeError(f"missing environment audit: {environment_path}")
    value = json.loads(environment_path.read_text())
    if not value.get("cuda_available") or not value.get("gpu"):
        raise RuntimeError(f"non-CUDA result is not admissible: {environment_path}")
    return value


def _collect_stage(stage_root, mechanisms, seeds):
    expected = {
        (dataset, horizon, seed, mechanism)
        for dataset, horizon in SETTINGS
        for seed in seeds
        for mechanism in mechanisms
    }
    found = {}
    environments = []
    for path in sorted((stage_root / "runs").glob("*/metrics.csv")):
        row = _read_single_row(path)
        key = (
            row.get("dataset"),
            int(row.get("horizon", -1)),
            int(row.get("seed", -1)),
            row.get("mechanism"),
        )
        if key not in expected:
            continue
        if key in found:
            raise RuntimeError(f"duplicate result: {key}")
        if str(row.get("test_mse", "")).strip() or str(
            row.get("test_mae", "")
        ).strip():
            raise RuntimeError(f"test leakage in validation-only stage: {key}")
        found[key] = row
        environments.append(_environment(path))
    missing = sorted(expected - set(found), key=str)
    if missing:
        raise RuntimeError(f"incomplete matrix: {len(missing)} missing; {missing[:6]}")
    signatures = {
        (
            item.get("gpu"), item.get("torch"), item.get("cuda_runtime"),
            item.get("lightning"), item.get("git_commit"),
        )
        for item in environments
    }
    if len(signatures) != 1:
        raise RuntimeError(f"heterogeneous run environment: {sorted(signatures, key=str)}")
    return found, next(iter(signatures))


def _anchor_checkpoints(args, dry=False):
    seeds = _parse_seeds(args.seeds)
    if dry:
        return {
            (dataset, horizon, seed): (
                f"<A2_CHECKPOINT:{dataset}:H{horizon}:S{seed}>"
            )
            for dataset, horizon in SETTINGS
            for seed in seeds
        }
    rows, _ = _collect_stage(_root(args) / "anchors", (INCUMBENT,), seeds)
    checkpoints = {}
    for dataset, horizon in SETTINGS:
        for seed in seeds:
            row = rows[(dataset, horizon, seed, INCUMBENT)]
            path = Path(row["checkpoint"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file():
                raise RuntimeError(
                    f"A2 checkpoint recorded but missing for {(dataset, horizon, seed)}: {path}"
                )
            checkpoints[(dataset, horizon, seed)] = path
    return checkpoints


def candidate_commands(args, dry=False):
    seeds = _parse_seeds(args.seeds)
    checkpoints = _anchor_checkpoints(args, dry=dry)
    return [
        _command(
            args, dataset, horizon, seed, mechanism, "candidates",
            checkpoint=checkpoints[(dataset, horizon, seed)],
        )
        for dataset, horizon in SETTINGS
        for seed in seeds
        for mechanism in CANDIDATES
    ]


def _float(row, key):
    value = str(row.get(key, "")).strip()
    if not value:
        raise RuntimeError(f"required metric {key!r} is missing")
    return float(value)


def summarize(args):
    seeds = _parse_seeds(args.seeds)
    anchors, anchor_signature = _collect_stage(
        _root(args) / "anchors", (INCUMBENT,), seeds
    )
    candidates, candidate_signature = _collect_stage(
        _root(args) / "candidates", CANDIDATES, seeds
    )
    if anchor_signature != candidate_signature:
        raise RuntimeError("anchor and candidate environments differ")

    details = []
    for dataset, horizon in SETTINGS:
        for seed in seeds:
            anchor = anchors[(dataset, horizon, seed, INCUMBENT)]
            a2_mse = _float(anchor, "val_mse")
            a2_mae = _float(anchor, "val_mae")
            for mechanism in CANDIDATES:
                row = candidates[(dataset, horizon, seed, mechanism)]
                if _float(row, "anchor_identity_max_abs") != 0.0:
                    raise RuntimeError(
                        f"candidate did not start as exact A2: {(dataset, horizon, seed, mechanism)}"
                    )
                frozen = str(row.get("anchor_frozen", "")).lower() == "true"
                if frozen != (mechanism in FROZEN_MODES):
                    raise RuntimeError(f"unexpected freeze state for {mechanism}")
                details.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "cycle_period": CYCLE_PERIODS[dataset],
                    "candidate": mechanism,
                    "anchor_frozen": frozen,
                    "val_mse": _float(row, "val_mse"),
                    "val_mae": _float(row, "val_mae"),
                    "mse_ratio_vs_matched_a2": _float(row, "val_mse") / a2_mse,
                    "mae_ratio_vs_matched_a2": _float(row, "val_mae") / a2_mae,
                    "internal_anchor_mse_ratio_vs_a2": (
                        _float(row, "val_anchor_mse") / a2_mse
                    ),
                    "internal_anchor_mae_ratio_vs_a2": (
                        _float(row, "val_anchor_mae") / a2_mae
                    ),
                    "fused_mse_ratio_vs_internal_anchor": _float(
                        row, "val_mse_ratio_vs_internal_anchor"
                    ),
                    "fused_mae_ratio_vs_internal_anchor": _float(
                        row, "val_mae_ratio_vs_internal_anchor"
                    ),
                    "update_rms": _float(row, "val_update_rms"),
                    "confidence_regret_corr": _float(
                        row, "val_confidence_regret_corr"
                    ),
                    "coefficient_regret_corr": _float(
                        row, "val_coefficient_regret_corr"
                    ),
                })

    aggregates = []
    for mechanism in CANDIDATES:
        rows = [item for item in details if item["candidate"] == mechanism]
        fused_ratios = [
            ratio for item in rows
            for ratio in (
                item["mse_ratio_vs_matched_a2"],
                item["mae_ratio_vs_matched_a2"],
            )
        ]
        internal_ratios = [
            ratio for item in rows
            for ratio in (
                item["internal_anchor_mse_ratio_vs_a2"],
                item["internal_anchor_mae_ratio_vs_a2"],
            )
        ]
        aggregates.append({
            "candidate": mechanism,
            "macro_ratio_vs_matched_a2": statistics.mean(fused_ratios),
            "worst_ratio_vs_matched_a2": max(fused_ratios),
            "both_metric_improve_rows": sum(
                item["mse_ratio_vs_matched_a2"] < 1.0
                and item["mae_ratio_vs_matched_a2"] < 1.0
                for item in rows
            ),
            "internal_anchor_macro_ratio_vs_a2": statistics.mean(internal_ratios),
            "internal_anchor_worst_ratio_vs_a2": max(internal_ratios),
            "macro_fused_mse_ratio_vs_internal_anchor": statistics.mean(
                item["fused_mse_ratio_vs_internal_anchor"] for item in rows
            ),
            "macro_fused_mae_ratio_vs_internal_anchor": statistics.mean(
                item["fused_mae_ratio_vs_internal_anchor"] for item in rows
            ),
            "mean_update_rms": statistics.mean(item["update_rms"] for item in rows),
            "mean_confidence_regret_corr": statistics.mean(
                item["confidence_regret_corr"] for item in rows
            ),
            "mean_coefficient_regret_corr": statistics.mean(
                item["coefficient_regret_corr"] for item in rows
            ),
        })

    keyed = {item["candidate"]: item for item in aggregates}
    frozen_exact = all(
        abs(item["internal_anchor_macro_ratio_vs_a2"] - 1.0) <= 1e-8
        and abs(item["internal_anchor_worst_ratio_vs_a2"] - 1.0) <= 1e-8
        for item in aggregates if item["candidate"] in FROZEN_MODES
    )
    minimum_double_improve = math.ceil(2 * len(SETTINGS) * len(seeds) / 3)
    decision = {
        "protocol": "pctf-anchor-v3-causal-attribution-validation-only",
        "test_metrics_read": False,
        "environment_signature": anchor_signature,
        "settings": [f"{dataset}-H{horizon}" for dataset, horizon in SETTINGS],
        "seeds": list(seeds),
        "hypotheses": {
            "H1_freeze_is_valid_control": frozen_exact,
            "H2_residual_target_beats_absolute_when_frozen": (
                keyed["pctf_anchor_diag_frozen_residual"]["macro_ratio_vs_matched_a2"]
                < keyed["pctf_anchor_diag_frozen_absolute"]["macro_ratio_vs_matched_a2"]
            ),
            "H3_anchor_safe_joint_limits_drift": (
                keyed["pctf_anchor_repair_joint_residual"]
                ["internal_anchor_worst_ratio_vs_a2"] <= 1.01
            ),
            "H4_marginal_gate_improves_joint_residual": (
                keyed["pctf_anchor_repair_joint_marginal"]["macro_ratio_vs_matched_a2"]
                < keyed["pctf_anchor_repair_joint_residual"]["macro_ratio_vs_matched_a2"]
                and keyed["pctf_anchor_repair_joint_marginal"]
                ["mean_coefficient_regret_corr"]
                > keyed["pctf_anchor_repair_joint_residual"]
                ["mean_coefficient_regret_corr"]
            ),
            "H5_full_repair_passes_preformal_gate": (
                keyed["pctf_anchor_repair_full"]["macro_ratio_vs_matched_a2"] <= 0.998
                and keyed["pctf_anchor_repair_full"]["worst_ratio_vs_matched_a2"] <= 1.01
                and keyed["pctf_anchor_repair_full"]["both_metric_improve_rows"]
                >= minimum_double_improve
            ),
        },
        "preformal_gate": {
            "macro_ratio_vs_matched_a2_max": 0.998,
            "worst_ratio_vs_matched_a2_max": 1.01,
            "both_metric_improve_fraction_min": "2/3",
            "both_metric_improve_rows_min": minimum_double_improve,
        },
        "formal_test_authorized": False,
        "note": "This diagnostic cannot select on or report test metrics.",
    }
    root = _root(args)
    _write_csv(root / "attribution_details.csv", details)
    _write_csv(root / "attribution_aggregates.csv", aggregates)
    _write_json(root / "attribution_decision.json", decision)
    return 0


def _run(commands, execute):
    print(f"commands={len(commands)}")
    for command in commands:
        print(shlex.join(command))
    if execute:
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "anchors-dry", "anchors", "candidates-dry", "candidates",
            "summarize",
        ),
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize":
        return summarize(args)
    if args.stage.startswith("anchors"):
        return _run(anchor_commands(args), args.stage == "anchors")
    return _run(
        candidate_commands(args, dry=args.stage == "candidates-dry"),
        args.stage == "candidates",
    )


if __name__ == "__main__":
    raise SystemExit(main())
