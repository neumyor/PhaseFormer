#!/usr/bin/env python3
"""Formal ETTh2/ETTm2 test comparison of A2 and PCTF full repair.

The candidate was frozen by the preceding validation-only attribution study.
For every dataset/horizon/seed, this runner trains a fresh full-data A2 first,
then initializes the single-checkpoint PCTF candidate from that matched A2.
Both models select checkpoints only by validation loss and read test once.
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
INCUMBENT = "rcrf_pe_lff"
CANDIDATE = "pctf_anchor_repair_full"
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 192),
    ("ETTm2", 96),
    ("ETTm2", 192),
)
SEEDS = (2021, 2022, 2023)
CYCLE_PERIODS = {"ETTh2": 48, "ETTm2": 96}
GOLDEN = {
    ("ETTh2", 96): (0.275, 0.338),
    ("ETTh2", 192): (0.341, 0.376),
    ("ETTm2", 96): (0.163, 0.256),
    ("ETTm2", 192): (0.219, 0.293),
}
DEFAULT_OUTPUT_ROOT = "research_runs/pctf_anchor_formal_etts_v1"


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
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _command(args, dataset, horizon, seed, mechanism, stage, checkpoint=None):
    command = [
        sys.executable,
        "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm",
        "--mechanism", mechanism,
        "--period", "24",
        "--lookback", "720",
        "--cycle-period", str(CYCLE_PERIODS[dataset]),
        "--percent", "100",
        "--max-epochs", "30",
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(_root(args) / stage),
        "--require-cuda",
        "--evaluate-test",
        "--resume",
    ]
    if checkpoint is not None:
        command.extend(("--init-checkpoint", str(checkpoint)))
    if args.progress:
        command.append("--progress")
    return command


def anchor_commands(args):
    return [
        _command(args, dataset, horizon, seed, INCUMBENT, "anchors")
        for dataset, horizon in SETTINGS
        for seed in _parse_seeds(args.seeds)
    ]


def _read_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metrics row: {path}")
    return rows[0]


def _environment(path):
    environment_path = path.parent / "environment.json"
    if not environment_path.is_file():
        raise RuntimeError(f"missing environment audit: {environment_path}")
    value = json.loads(environment_path.read_text())
    if not value.get("cuda_available") or not value.get("gpu"):
        raise RuntimeError(f"non-CUDA run is inadmissible: {environment_path}")
    return value


def _collect(stage_root, mechanism, seeds):
    expected = {
        (dataset, horizon, seed, mechanism)
        for dataset, horizon in SETTINGS for seed in seeds
    }
    found = {}
    environments = []
    for path in sorted((stage_root / "runs").glob("*/metrics.csv")):
        row = _read_row(path)
        key = (
            row.get("dataset"), int(row.get("horizon", -1)),
            int(row.get("seed", -1)), row.get("mechanism"),
        )
        if key not in expected:
            continue
        if key in found:
            raise RuntimeError(f"duplicate formal result: {key}")
        if not str(row.get("test_mse", "")).strip() or not str(
            row.get("test_mae", "")
        ).strip():
            raise RuntimeError(f"formal result lacks test metrics: {key}")
        found[key] = row
        environments.append(_environment(path))
    missing = sorted(expected - set(found), key=str)
    if missing:
        raise RuntimeError(f"formal matrix incomplete: {len(missing)} missing; {missing[:6]}")
    signatures = {
        (
            item.get("gpu"), item.get("torch"), item.get("cuda_runtime"),
            item.get("lightning"), item.get("git_commit"),
        )
        for item in environments
    }
    if len(signatures) != 1:
        raise RuntimeError(f"heterogeneous formal environment: {sorted(signatures, key=str)}")
    return found, next(iter(signatures))


def _anchor_checkpoints(args, dry=False):
    seeds = _parse_seeds(args.seeds)
    if dry:
        return {
            (dataset, horizon, seed): f"<A2:{dataset}:H{horizon}:S{seed}>"
            for dataset, horizon in SETTINGS for seed in seeds
        }
    rows, _ = _collect(_root(args) / "anchors", INCUMBENT, seeds)
    checkpoints = {}
    for dataset, horizon in SETTINGS:
        for seed in seeds:
            path = Path(rows[(dataset, horizon, seed, INCUMBENT)]["checkpoint"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file():
                raise RuntimeError(f"matched A2 checkpoint is missing: {path}")
            checkpoints[(dataset, horizon, seed)] = path
    return checkpoints


def candidate_commands(args, dry=False):
    seeds = _parse_seeds(args.seeds)
    checkpoints = _anchor_checkpoints(args, dry=dry)
    return [
        _command(
            args, dataset, horizon, seed, CANDIDATE, "candidates",
            checkpoint=checkpoints[(dataset, horizon, seed)],
        )
        for dataset, horizon in SETTINGS for seed in seeds
    ]


def _float(row, key):
    value = str(row.get(key, "")).strip()
    if not value:
        raise RuntimeError(f"required metric {key!r} is missing")
    return float(value)


def _mean_std(values):
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize(args):
    seeds = _parse_seeds(args.seeds)
    anchors, anchor_environment = _collect(
        _root(args) / "anchors", INCUMBENT, seeds
    )
    candidates, candidate_environment = _collect(
        _root(args) / "candidates", CANDIDATE, seeds
    )
    if anchor_environment != candidate_environment:
        raise RuntimeError("A2 and candidate formal environments differ")

    details = []
    for dataset, horizon in SETTINGS:
        for seed in seeds:
            a2 = anchors[(dataset, horizon, seed, INCUMBENT)]
            candidate = candidates[(dataset, horizon, seed, CANDIDATE)]
            if _float(candidate, "anchor_identity_max_abs") != 0.0:
                raise RuntimeError(
                    f"candidate is not exact A2 at initialization: {(dataset, horizon, seed)}"
                )
            for mechanism, row in ((INCUMBENT, a2), (CANDIDATE, candidate)):
                internal_anchor_mse = (
                    _float(row, "test_mse") if mechanism == INCUMBENT
                    else _float(row, "test_anchor_mse")
                )
                internal_anchor_mae = (
                    _float(row, "test_mae") if mechanism == INCUMBENT
                    else _float(row, "test_anchor_mae")
                )
                details.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "model": mechanism,
                    "test_mse": _float(row, "test_mse"),
                    "test_mae": _float(row, "test_mae"),
                    "internal_anchor_test_mse": internal_anchor_mse,
                    "internal_anchor_test_mae": internal_anchor_mae,
                    "val_mse": _float(row, "val_mse"),
                    "val_mae": _float(row, "val_mae"),
                    "epochs_completed": int(float(row["epochs_completed"])),
                    "elapsed_sec": _float(row, "elapsed_sec"),
                    "peak_memory_bytes": int(float(row["peak_memory_bytes"])),
                    "parameter_count": int(row["parameter_count"]),
                    "trainable_parameter_count": int(row["trainable_parameter_count"]),
                })

    summary = []
    keyed = {}
    for dataset, horizon in SETTINGS:
        golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
        for mechanism in (INCUMBENT, CANDIDATE):
            group = [
                item for item in details
                if item["dataset"] == dataset
                and item["horizon"] == horizon
                and item["model"] == mechanism
            ]
            mses = [item["test_mse"] for item in group]
            maes = [item["test_mae"] for item in group]
            internal_anchor_mses = [
                item["internal_anchor_test_mse"] for item in group
            ]
            internal_anchor_maes = [
                item["internal_anchor_test_mae"] for item in group
            ]
            mse_mean, mse_std = _mean_std(mses)
            mae_mean, mae_std = _mean_std(maes)
            stage_elapsed = statistics.mean(
                group_item["elapsed_sec"] for group_item in group
            )
            upstream_a2_elapsed = 0.0
            if mechanism == CANDIDATE:
                upstream_group = [
                    detail for detail in details
                    if detail["dataset"] == dataset
                    and detail["horizon"] == horizon
                    and detail["model"] == INCUMBENT
                ]
                upstream_a2_elapsed = statistics.mean(
                    group_item["elapsed_sec"] for group_item in upstream_group
                )
            item = {
                "dataset": dataset,
                "horizon": horizon,
                "model": mechanism,
                "golden_mse": golden_mse,
                "golden_mae": golden_mae,
                "test_mse_mean": mse_mean,
                "test_mse_std": mse_std,
                "test_mae_mean": mae_mean,
                "test_mae_std": mae_std,
                "internal_anchor_test_mse_mean": statistics.mean(
                    internal_anchor_mses
                ),
                "internal_anchor_test_mae_mean": statistics.mean(
                    internal_anchor_maes
                ),
                "mse_ratio_vs_internal_anchor": (
                    mse_mean / statistics.mean(internal_anchor_mses)
                ),
                "mae_ratio_vs_internal_anchor": (
                    mae_mean / statistics.mean(internal_anchor_maes)
                ),
                "mse_improvement_vs_golden_pct": 100 * (golden_mse - mse_mean) / golden_mse,
                "mae_improvement_vs_golden_pct": 100 * (golden_mae - mae_mean) / golden_mae,
                "stable_below_golden": (
                    all(value < golden_mse for value in mses)
                    and all(value < golden_mae for value in maes)
                    and mse_mean + mse_std < golden_mse
                    and mae_mean + mae_std < golden_mae
                ),
                "elapsed_sec_mean": stage_elapsed,
                "stage_elapsed_sec_mean": stage_elapsed,
                "upstream_a2_elapsed_sec_mean": upstream_a2_elapsed,
                "total_training_elapsed_sec_mean": (
                    stage_elapsed + upstream_a2_elapsed
                ),
                "peak_memory_bytes_max": max(
                    item["peak_memory_bytes"] for item in group
                ),
                "parameter_count": group[0]["parameter_count"],
                "trainable_parameter_count": group[0]["trainable_parameter_count"],
            }
            summary.append(item)
            keyed[(dataset, horizon, mechanism)] = item

    mse_ratios, mae_ratios = [], []
    both_improve = 0
    for dataset, horizon in SETTINGS:
        a2 = keyed[(dataset, horizon, INCUMBENT)]
        candidate = keyed[(dataset, horizon, CANDIDATE)]
        mse_ratio = candidate["test_mse_mean"] / a2["test_mse_mean"]
        mae_ratio = candidate["test_mae_mean"] / a2["test_mae_mean"]
        candidate["mse_ratio_vs_a2"] = mse_ratio
        candidate["mae_ratio_vs_a2"] = mae_ratio
        a2["mse_ratio_vs_a2"] = 1.0
        a2["mae_ratio_vs_a2"] = 1.0
        mse_ratios.append(mse_ratio)
        mae_ratios.append(mae_ratio)
        both_improve += mse_ratio < 1.0 and mae_ratio < 1.0

    decision = {
        "protocol": "pctf-anchor-full-repair-etts-three-seed-formal-test-v1",
        "candidate_frozen_before_test": True,
        "test_set_selection_after_this_run": True,
        "settings": [f"{dataset}-H{horizon}" for dataset, horizon in SETTINGS],
        "seeds": list(seeds),
        "environment_signature": anchor_environment,
        "candidate_mse_macro_ratio_vs_a2": statistics.mean(mse_ratios),
        "candidate_mae_macro_ratio_vs_a2": statistics.mean(mae_ratios),
        "candidate_combined_macro_ratio_vs_a2": statistics.mean(
            mse_ratios + mae_ratios
        ),
        "candidate_both_metric_improve_settings": int(both_improve),
        "candidate_worst_ratio_vs_a2": max(mse_ratios + mae_ratios),
        "candidate_stable_below_golden_settings": sum(
            keyed[(dataset, horizon, CANDIDATE)]["stable_below_golden"]
            for dataset, horizon in SETTINGS
        ),
        "a2_stable_below_golden_settings": sum(
            keyed[(dataset, horizon, INCUMBENT)]["stable_below_golden"]
            for dataset, horizon in SETTINGS
        ),
    }
    decision["candidate_replaces_a2_on_etts"] = (
        decision["candidate_combined_macro_ratio_vs_a2"] < 0.998
        and decision["candidate_both_metric_improve_settings"] >= 3
        and decision["candidate_worst_ratio_vs_a2"] <= 1.005
    )
    root = _root(args)
    _write_csv(root / "formal_details.csv", details)
    _write_csv(root / "formal_summary.csv", summary)
    _write_json(root / "formal_decision.json", decision)
    return 0


def _run(commands, execute):
    print(f"commands={len(commands)}")
    for command in commands:
        print(shlex.join(command))
    if execute:
        for index, command in enumerate(commands, 1):
            print(f"RUN {index}/{len(commands)}")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
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
