#!/usr/bin/env python3
"""Screen and formally confirm one-stage PCTF training on ETTh2/ETTm2.

Every candidate starts from random initialization and trains the complete
PhaseFormer + LFF-NLinear + ICPT graph in one Trainer.fit call.  Screening is
validation-only; formal test commands are available only after the shared
training policy passes the preregistered validation gate.
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
CYCLE_PERIODS = {"ETTh2": 48, "ETTm2": 96}
GOLDEN = {
    ("ETTh2", 96): (0.275, 0.338),
    ("ETTh2", 192): (0.341, 0.376),
    ("ETTm2", 96): (0.163, 0.256),
    ("ETTm2", 192): (0.219, 0.293),
}
SCREEN_SEEDS = (2021, 2022)
FORMAL_SEEDS = (2021, 2022, 2023)
DEFAULT_OUTPUT_ROOT = "research_runs/pctf_single_stage_training_v1"

# The first four rows isolate anchor LR and protection.  The next two test a
# within-run correction curriculum; ICPT still trains from epoch zero through
# its residual component and marginal-coefficient auxiliary objectives.  The
# final row is an evidence-driven follow-up: the forward graph stays fused, but
# the fused objective cannot perturb A2, which receives its own matched loss.
POLICIES = {
    "legacy_safe": dict(anchor_lr=0.1, anchor_loss=1.0, warmup=0, decouple=False),
    "uniform_unprotected": dict(anchor_lr=1.0, anchor_loss=0.0, warmup=0, decouple=False),
    "uniform_mild": dict(anchor_lr=1.0, anchor_loss=0.25, warmup=0, decouple=False),
    "uniform_protected": dict(anchor_lr=1.0, anchor_loss=1.0, warmup=0, decouple=False),
    "warm5_mild": dict(anchor_lr=1.0, anchor_loss=0.25, warmup=5, decouple=False),
    "warm5_protected": dict(anchor_lr=1.0, anchor_loss=1.0, warmup=5, decouple=False),
    "decoupled_protected": dict(
        anchor_lr=1.0, anchor_loss=1.0, warmup=0, decouple=True
    ),
}


def _root(args):
    value = Path(args.output_root)
    return value if value.is_absolute() else REPO_ROOT / value


def _parse_seeds(value):
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("at least one seed is required")
    return result


def _selected_policies(args):
    value = getattr(args, "policies", "")
    if not value:
        return tuple(POLICIES)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected).difference(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {', '.join(unknown)}")
    if not selected:
        raise ValueError("at least one policy is required")
    return selected


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _read_row(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metrics row: {path}")
    return rows[0]


def _float(row, key):
    value = str(row.get(key, "")).strip()
    if not value:
        raise RuntimeError(f"missing required metric {key!r}")
    return float(value)


def _environment(path):
    value = json.loads((path.parent / "environment.json").read_text())
    if not value.get("cuda_available") or not value.get("gpu"):
        raise RuntimeError(f"non-CUDA run is inadmissible: {path.parent}")
    return (
        value.get("gpu"), value.get("torch"), value.get("cuda_runtime"),
        value.get("lightning"), value.get("git_commit"),
    )


def _command(
    args, dataset, horizon, seed, mechanism, output_dir, *,
    policy=None, evaluate_test=False,
):
    command = [
        sys.executable, "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm" if evaluate_test else "finalist",
        "--mechanism", mechanism,
        "--period", "24",
        "--lookback", "720",
        "--cycle-period", str(CYCLE_PERIODS[dataset]),
        "--percent", "100",
        "--max-epochs", "30",
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0" if evaluate_test else "8",
        "--output-dir", str(output_dir),
        "--require-cuda",
        "--resume",
    ]
    if policy is not None:
        config = POLICIES[policy]
        overrides = {
            "anchored_pctf_anchor_lr_scale": config["anchor_lr"],
            "anchored_pctf_anchor_loss_weight": config["anchor_loss"],
            "anchored_pctf_correction_warmup_epochs": config["warmup"],
            "anchored_pctf_decouple_anchor_gradient": config["decouple"],
        }
        command.extend(("--overrides", json.dumps(overrides, sort_keys=True)))
    if evaluate_test:
        command.append("--evaluate-test")
    if args.progress:
        command.append("--progress")
    return command


def screen_baseline_commands(args):
    root = _root(args) / "screen" / "baselines"
    return [
        _command(args, dataset, horizon, seed, INCUMBENT, root)
        for dataset, horizon in SETTINGS
        for seed in _parse_seeds(args.screen_seeds)
    ]


def screen_candidate_commands(args):
    root = _root(args) / "screen" / "candidates"
    return [
        _command(
            args, dataset, horizon, seed, CANDIDATE, root / policy,
            policy=policy,
        )
        for policy in _selected_policies(args)
        for dataset, horizon in SETTINGS
        for seed in _parse_seeds(args.screen_seeds)
    ]


def _collect(root, mechanism, seeds, *, require_test):
    expected = {
        (dataset, horizon, seed, mechanism)
        for dataset, horizon in SETTINGS for seed in seeds
    }
    found = {}
    environments = set()
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        row = _read_row(path)
        key = (
            row.get("dataset"), int(row.get("horizon", -1)),
            int(row.get("seed", -1)), row.get("mechanism"),
        )
        if key not in expected:
            continue
        if key in found:
            raise RuntimeError(f"duplicate result: {key}")
        has_test = bool(str(row.get("test_mse", "")).strip())
        if require_test != has_test:
            label = "missing test" if require_test else "test leakage"
            raise RuntimeError(f"{label}: {key}")
        found[key] = row
        environments.add(_environment(path))
    missing = sorted(expected - set(found), key=str)
    if missing:
        raise RuntimeError(f"incomplete matrix: {len(missing)} missing; {missing[:4]}")
    if len(environments) != 1:
        raise RuntimeError(f"heterogeneous environments: {sorted(environments, key=str)}")
    return found, next(iter(environments))


def summarize_screen(args):
    seeds = _parse_seeds(args.screen_seeds)
    root = _root(args)
    baselines, environment = _collect(
        root / "screen" / "baselines", INCUMBENT, seeds,
        require_test=False,
    )
    details = []
    aggregates = []
    for policy in _selected_policies(args):
        candidates, candidate_environment = _collect(
            root / "screen" / "candidates" / policy,
            CANDIDATE, seeds, require_test=False,
        )
        if candidate_environment != environment:
            raise RuntimeError(f"environment mismatch for {policy}")
        rows = []
        for dataset, horizon in SETTINGS:
            for seed in seeds:
                key_a = (dataset, horizon, seed, INCUMBENT)
                key_c = (dataset, horizon, seed, CANDIDATE)
                anchor = baselines[key_a]
                candidate = candidates[key_c]
                mse_ratio = _float(candidate, "val_mse") / _float(anchor, "val_mse")
                mae_ratio = _float(candidate, "val_mae") / _float(anchor, "val_mae")
                item = {
                    "policy": policy,
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "a2_val_mse": _float(anchor, "val_mse"),
                    "a2_val_mae": _float(anchor, "val_mae"),
                    "candidate_val_mse": _float(candidate, "val_mse"),
                    "candidate_val_mae": _float(candidate, "val_mae"),
                    "mse_ratio_vs_a2": mse_ratio,
                    "mae_ratio_vs_a2": mae_ratio,
                    "internal_anchor_mse_ratio_vs_a2": (
                        _float(candidate, "val_anchor_mse")
                        / _float(anchor, "val_mse")
                    ),
                    "internal_anchor_mae_ratio_vs_a2": (
                        _float(candidate, "val_anchor_mae")
                        / _float(anchor, "val_mae")
                    ),
                    "fused_mse_ratio_vs_internal_anchor": _float(
                        candidate, "val_mse_ratio_vs_internal_anchor"
                    ),
                    "fused_mae_ratio_vs_internal_anchor": _float(
                        candidate, "val_mae_ratio_vs_internal_anchor"
                    ),
                    "update_rms": _float(candidate, "val_update_rms"),
                    "coefficient_regret_corr": _float(
                        candidate, "val_coefficient_regret_corr"
                    ),
                    "elapsed_sec": _float(candidate, "elapsed_sec"),
                    "epochs_completed": int(float(candidate["epochs_completed"])),
                    "final_correction_scale": _float(
                        candidate, "final_correction_scale"
                    ),
                }
                rows.append(item)
                details.append(item)
        mse_ratios = [row["mse_ratio_vs_a2"] for row in rows]
        mae_ratios = [row["mae_ratio_vs_a2"] for row in rows]
        combined = statistics.mean(mse_ratios + mae_ratios)
        aggregate = {
            "policy": policy,
            "anchor_lr_scale": POLICIES[policy]["anchor_lr"],
            "anchor_loss_weight": POLICIES[policy]["anchor_loss"],
            "correction_warmup_epochs": POLICIES[policy]["warmup"],
            "decouple_anchor_gradient": POLICIES[policy]["decouple"],
            "mse_macro_ratio_vs_a2": statistics.mean(mse_ratios),
            "mae_macro_ratio_vs_a2": statistics.mean(mae_ratios),
            "combined_macro_ratio_vs_a2": combined,
            "worst_ratio_vs_a2": max(mse_ratios + mae_ratios),
            "both_metric_improve_rows": sum(
                row["mse_ratio_vs_a2"] < 1 and row["mae_ratio_vs_a2"] < 1
                for row in rows
            ),
            "internal_anchor_combined_ratio_vs_a2": statistics.mean([
                *[row["internal_anchor_mse_ratio_vs_a2"] for row in rows],
                *[row["internal_anchor_mae_ratio_vs_a2"] for row in rows],
            ]),
            "fused_combined_ratio_vs_internal_anchor": statistics.mean([
                *[row["fused_mse_ratio_vs_internal_anchor"] for row in rows],
                *[row["fused_mae_ratio_vs_internal_anchor"] for row in rows],
            ]),
            "mean_update_rms": statistics.mean(row["update_rms"] for row in rows),
            "mean_coefficient_regret_corr": statistics.mean(
                row["coefficient_regret_corr"] for row in rows
            ),
            "elapsed_sec_mean": statistics.mean(row["elapsed_sec"] for row in rows),
            "min_final_correction_scale": min(
                row["final_correction_scale"] for row in rows
            ),
        }
        aggregate["eligible"] = (
            combined < 0.998
            and aggregate["worst_ratio_vs_a2"] <= 1.01
            and aggregate["both_metric_improve_rows"] >= 6
            and aggregate["min_final_correction_scale"] == 1.0
        )
        aggregates.append(aggregate)
    eligible = [row for row in aggregates if row["eligible"]]
    winner = min(
        eligible,
        key=lambda row: (
            row["combined_macro_ratio_vs_a2"], row["worst_ratio_vs_a2"]
        ),
    ) if eligible else None
    decision = {
        "protocol": "pctf-single-stage-training-screen-v1",
        "test_metrics_read": False,
        "settings": [f"{dataset}-H{horizon}" for dataset, horizon in SETTINGS],
        "seeds": list(seeds),
        "environment_signature": environment,
        "gate": {
            "combined_macro_ratio_lt": 0.998,
            "worst_ratio_lte": 1.01,
            "both_metric_improve_rows_gte": 6,
            "requires_full_correction_scale": True,
        },
        "winner": winner["policy"] if winner else None,
        "winner_eligible": winner is not None,
    }
    _write_csv(root / "screen_details.csv", details)
    _write_csv(root / "screen_aggregates.csv", aggregates)
    _write_json(root / "screen_decision.json", decision)
    return 0


def _winner(args):
    path = _root(args) / "screen_decision.json"
    if not path.is_file():
        raise RuntimeError("screen must be summarized before formal commands")
    decision = json.loads(path.read_text())
    if not decision.get("winner_eligible") or decision.get("winner") not in POLICIES:
        raise RuntimeError("no single-stage policy passed the validation gate")
    return decision["winner"]


def formal_baseline_commands(args):
    root = _root(args) / "formal" / "baselines"
    return [
        _command(
            args, dataset, horizon, seed, INCUMBENT, root,
            evaluate_test=True,
        )
        for dataset, horizon in SETTINGS
        for seed in _parse_seeds(args.formal_seeds)
    ]


def formal_candidate_commands(args):
    policy = _winner(args)
    root = _root(args) / "formal" / "candidates" / policy
    return [
        _command(
            args, dataset, horizon, seed, CANDIDATE, root,
            policy=policy, evaluate_test=True,
        )
        for dataset, horizon in SETTINGS
        for seed in _parse_seeds(args.formal_seeds)
    ]


def _mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def summarize_formal(args):
    seeds = _parse_seeds(args.formal_seeds)
    root = _root(args)
    policy = _winner(args)
    baselines, environment = _collect(
        root / "formal" / "baselines", INCUMBENT, seeds, require_test=True
    )
    candidates, candidate_environment = _collect(
        root / "formal" / "candidates" / policy,
        CANDIDATE, seeds, require_test=True,
    )
    if candidate_environment != environment:
        raise RuntimeError("formal environments differ")
    summary = []
    mse_ratios, mae_ratios = [], []
    both = 0
    worst = 0.0
    for dataset, horizon in SETTINGS:
        golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
        keyed = {}
        for label, mechanism, source in (
            ("A2", INCUMBENT, baselines),
            (f"one_stage:{policy}", CANDIDATE, candidates),
        ):
            group = [
                source[(dataset, horizon, seed, mechanism)] for seed in seeds
            ]
            mses = [_float(row, "test_mse") for row in group]
            maes = [_float(row, "test_mae") for row in group]
            mse_mean, mse_std = _mean_std(mses)
            mae_mean, mae_std = _mean_std(maes)
            item = {
                "dataset": dataset,
                "horizon": horizon,
                "model": label,
                "golden_mse": golden_mse,
                "golden_mae": golden_mae,
                "test_mse_mean": mse_mean,
                "test_mse_std": mse_std,
                "test_mae_mean": mae_mean,
                "test_mae_std": mae_std,
                "mse_improvement_vs_golden_pct": (
                    100 * (golden_mse - mse_mean) / golden_mse
                ),
                "mae_improvement_vs_golden_pct": (
                    100 * (golden_mae - mae_mean) / golden_mae
                ),
                "elapsed_sec_mean": statistics.mean(
                    _float(row, "elapsed_sec") for row in group
                ),
                "peak_memory_bytes_max": max(
                    int(float(row["peak_memory_bytes"])) for row in group
                ),
                "parameter_count": int(group[0]["parameter_count"]),
            }
            summary.append(item)
            keyed[label] = item
        a2, candidate = keyed["A2"], keyed[f"one_stage:{policy}"]
        mse_ratio = candidate["test_mse_mean"] / a2["test_mse_mean"]
        mae_ratio = candidate["test_mae_mean"] / a2["test_mae_mean"]
        candidate["mse_ratio_vs_a2"] = mse_ratio
        candidate["mae_ratio_vs_a2"] = mae_ratio
        a2["mse_ratio_vs_a2"] = a2["mae_ratio_vs_a2"] = 1.0
        mse_ratios.append(mse_ratio)
        mae_ratios.append(mae_ratio)
        both += mse_ratio < 1 and mae_ratio < 1
        worst = max(worst, mse_ratio, mae_ratio)
    decision = {
        "protocol": "pctf-single-stage-training-formal-v1",
        "policy_frozen_on_validation": policy,
        "test_set_selection_after_this_run": True,
        "environment_signature": environment,
        "candidate_mse_macro_ratio_vs_a2": statistics.mean(mse_ratios),
        "candidate_mae_macro_ratio_vs_a2": statistics.mean(mae_ratios),
        "candidate_combined_macro_ratio_vs_a2": statistics.mean(
            mse_ratios + mae_ratios
        ),
        "candidate_both_metric_improve_settings": int(both),
        "candidate_worst_ratio_vs_a2": worst,
        "candidate_replaces_a2_on_etts": (
            statistics.mean(mse_ratios + mae_ratios) < 0.998
            and both >= 3 and worst <= 1.005
        ),
    }
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
    parser.add_argument("--stage", required=True, choices=(
        "screen-baselines-dry", "screen-baselines",
        "screen-candidates-dry", "screen-candidates", "screen-summarize",
        "formal-baselines-dry", "formal-baselines",
        "formal-candidates-dry", "formal-candidates", "formal-summarize",
    ))
    parser.add_argument("--screen-seeds", default=",".join(map(str, SCREEN_SEEDS)))
    parser.add_argument("--formal-seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--policies", default="",
        help="optional comma-separated subset for an evidence-driven follow-up",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    stages = {
        "screen-baselines-dry": (screen_baseline_commands, False),
        "screen-baselines": (screen_baseline_commands, True),
        "screen-candidates-dry": (screen_candidate_commands, False),
        "screen-candidates": (screen_candidate_commands, True),
        "formal-baselines-dry": (formal_baseline_commands, False),
        "formal-baselines": (formal_baseline_commands, True),
        "formal-candidates-dry": (formal_candidate_commands, False),
        "formal-candidates": (formal_candidate_commands, True),
    }
    if args.stage == "screen-summarize":
        return summarize_screen(args)
    if args.stage == "formal-summarize":
        return summarize_formal(args)
    function, execute = stages[args.stage]
    return _run(function(args), execute)


if __name__ == "__main__":
    raise SystemExit(main())
