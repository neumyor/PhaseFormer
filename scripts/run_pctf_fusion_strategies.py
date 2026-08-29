#!/usr/bin/env python3
"""Validation screen and frozen test confirmation for PCTF fusion strategies."""

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
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)
REFERENCES = (
    "gold_combo_reliability_s2",
    "rcrf_pe_lff",
    "rcrf_icpt_none",
)
INCUMBENT = "rcrf_pe_lff"
PAPER_CANDIDATES = (
    "pctf_dual_fixed",                 # F0
    "pctf_fusion_component_scalar",    # F1a
    "pctf_fusion_component_cycle",     # F1b
    "pctf_fusion_monotonic",           # F2a
    "pctf_fusion_mlp",                 # F2b
    "pctf_fusion_phase_modulation",    # F3
)
NEGATIVE_CONTROLS = (
    "pctf_fusion_uniform_control",      # C0
    "pctf_fusion_softmax_control",      # C1
)
FUSION_MODES = PAPER_CANDIDATES + NEGATIVE_CONTROLS
SCREEN_MODES = REFERENCES + FUSION_MODES
DEFAULT_OUTPUT_ROOT = "research_runs/pctf_fusion_v1"
GOLDEN = {
    ("ETTh1", 96): (0.359, 0.382),
    ("ETTh1", 192): (0.397, 0.404),
    ("ETTh2", 96): (0.275, 0.338),
    ("ETTh2", 192): (0.341, 0.376),
    ("ETTm1", 96): (0.293, 0.344),
    ("ETTm1", 192): (0.323, 0.361),
    ("ETTm2", 96): (0.163, 0.256),
    ("ETTm2", 192): (0.219, 0.293),
    ("Weather", 96): (0.148, 0.195),
    ("Weather", 192): (0.193, 0.237),
    ("Electricity", 96): (0.129, 0.221),
    ("Electricity", 192): (0.148, 0.238),
}


def _parse_csv(value, allowed, cast=str):
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported values: {unknown}")
    if not values:
        raise ValueError("selection must not be empty")
    return values


def _root(args):
    path = Path(args.output_root)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _command(args, dataset, horizon, seed, mode, *, formal):
    command = [
        sys.executable,
        "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm" if formal else "mechanism_screen_2",
        "--mechanism", mode,
        "--period", "24",
        "--lookback", "720",
        "--percent", "100" if formal else "30",
        "--max-epochs", "30" if formal else "8",
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(_root(args) / ("formal" if formal else "screen")),
        "--resume",
    ]
    if formal:
        command.append("--evaluate-test")
    if args.progress:
        command.append("--progress")
    return command


def screen_commands(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    return [
        _command(args, dataset, 96, 2021, mode, formal=False)
        for dataset in datasets
        for mode in SCREEN_MODES
    ]


def _frozen_champion(args, *, dry_override=False):
    if dry_override and args.champion:
        if args.champion not in PAPER_CANDIDATES:
            raise ValueError("negative controls cannot be frozen champions")
        return args.champion
    path = _root(args) / "screen_decision.json"
    if not path.is_file():
        raise RuntimeError("screen_decision.json missing; screen must be summarized")
    decision = json.loads(path.read_text())
    if not decision.get("passed"):
        raise RuntimeError("no paper candidate passed; formal test is blocked")
    champion = decision.get("champion")
    if champion not in PAPER_CANDIDATES:
        raise RuntimeError(f"invalid frozen champion: {champion}")
    return champion


def formal_commands(args, *, dry_override=False):
    champion = _frozen_champion(args, dry_override=dry_override)
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    modes = REFERENCES + (champion,)
    commands = [
        _command(args, dataset, horizon, seed, mode, formal=True)
        for dataset in datasets
        for horizon in horizons
        for seed in seeds
        for mode in modes
    ]
    return commands, champion


def _collect(root, expected):
    rows = {}
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values:
            continue
        row = values[0]
        key = (
            row.get("dataset"), int(row.get("horizon", -1)),
            row.get("mechanism"), int(row.get("seed", -1)),
        )
        if key not in expected:
            continue
        if key in rows:
            raise RuntimeError(f"duplicate result for {key}")
        rows[key] = row
    missing = sorted(expected - set(rows))
    if missing:
        preview = "\n".join(f"  {key}" for key in missing[:20])
        raise RuntimeError(f"matrix incomplete: {len(missing)} missing\n{preview}")
    return rows


def _macro_pair_ratio(rows, datasets, left, right, metric_prefix="val"):
    ratios = []
    double_improve = 0
    for dataset in datasets:
        left_row = rows[(dataset, 96, left, 2021)]
        right_row = rows[(dataset, 96, right, 2021)]
        mse = float(left_row[f"{metric_prefix}_mse"]) / float(
            right_row[f"{metric_prefix}_mse"]
        )
        mae = float(left_row[f"{metric_prefix}_mae"]) / float(
            right_row[f"{metric_prefix}_mae"]
        )
        ratios.extend((mse, mae))
        double_improve += mse < 1.0 and mae < 1.0
    return statistics.mean(ratios), int(double_improve), max(ratios)


def summarize_screen(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    expected = {
        (dataset, 96, mode, 2021)
        for dataset in datasets for mode in SCREEN_MODES
    }
    rows = _collect(_root(args) / "screen", expected)
    leaked = [
        key for key, row in rows.items()
        if row.get("test_mse", "").strip() or row.get("test_mae", "").strip()
    ]
    if leaked:
        raise RuntimeError(f"validation screen contains test metrics: {leaked[:5]}")

    details, aggregates = [], []
    for candidate in FUSION_MODES:
        ratios_a2, ratios_reference, ratios_control = [], [], []
        both_improve = 0
        for dataset in datasets:
            item = rows[(dataset, 96, candidate, 2021)]
            mse, mae = float(item["val_mse"]), float(item["val_mae"])
            a2 = rows[(dataset, 96, INCUMBENT, 2021)]
            a2_mse, a2_mae = float(a2["val_mse"]), float(a2["val_mae"])
            reference_mse = min(
                float(rows[(dataset, 96, mode, 2021)]["val_mse"])
                for mode in REFERENCES
            )
            reference_mae = min(
                float(rows[(dataset, 96, mode, 2021)]["val_mae"])
                for mode in REFERENCES
            )
            control_mse = min(
                float(rows[(dataset, 96, mode, 2021)]["val_mse"])
                for mode in NEGATIVE_CONTROLS
            )
            control_mae = min(
                float(rows[(dataset, 96, mode, 2021)]["val_mae"])
                for mode in NEGATIVE_CONTROLS
            )
            mse_a2, mae_a2 = mse / a2_mse, mae / a2_mae
            mse_ref, mae_ref = mse / reference_mse, mae / reference_mae
            mse_control, mae_control = mse / control_mse, mae / control_mae
            ratios_a2.extend((mse_a2, mae_a2))
            ratios_reference.extend((mse_ref, mae_ref))
            ratios_control.extend((mse_control, mae_control))
            both_improve += mse_a2 < 1.0 and mae_a2 < 1.0
            details.append({
                "dataset": dataset,
                "horizon": 96,
                "candidate": candidate,
                "paper_candidate": candidate in PAPER_CANDIDATES,
                "val_mse": mse,
                "val_mae": mae,
                "mse_ratio_vs_a2": mse_a2,
                "mae_ratio_vs_a2": mae_a2,
                "mse_ratio_vs_reference_envelope": mse_ref,
                "mae_ratio_vs_reference_envelope": mae_ref,
                "mse_ratio_vs_control_envelope": mse_control,
                "mae_ratio_vs_control_envelope": mae_control,
            })
        aggregate = {
            "candidate": candidate,
            "paper_candidate": candidate in PAPER_CANDIDATES,
            "macro_ratio_vs_a2": statistics.mean(ratios_a2),
            "both_metric_improve_datasets_vs_a2": int(both_improve),
            "worst_ratio_vs_a2": max(ratios_a2),
            "macro_ratio_vs_reference_envelope": statistics.mean(
                ratios_reference
            ),
            "macro_ratio_vs_control_envelope": statistics.mean(ratios_control),
        }
        aggregate["passed"] = (
            aggregate["paper_candidate"]
            and aggregate["macro_ratio_vs_a2"] <= 0.998
            and aggregate["both_metric_improve_datasets_vs_a2"] >= 4
            and aggregate["worst_ratio_vs_a2"] <= 1.01
            and aggregate["macro_ratio_vs_reference_envelope"] <= 1.005
            and aggregate["macro_ratio_vs_control_envelope"] <= 1.0
        )
        aggregates.append(aggregate)

    comparisons = []
    for child, parent, question in (
        ("pctf_fusion_component_cycle", "pctf_fusion_component_scalar",
         "Do cycle-specific gates beat scalar gates?"),
        ("pctf_fusion_monotonic", "pctf_fusion_component_cycle",
         "Does monotonic history evidence help?"),
        ("pctf_fusion_mlp", "pctf_fusion_component_cycle",
         "Does unconstrained evidence help?"),
        ("pctf_fusion_phase_modulation", "pctf_fusion_component_cycle",
         "Does phase-template modulation beat component interpolation?"),
    ):
        macro, double, worst = _macro_pair_ratio(rows, datasets, child, parent)
        comparisons.append({
            "child": child, "parent": parent, "question": question,
            "macro_ratio": macro, "double_improve_datasets": double,
            "worst_ratio": worst,
        })

    eligible = [item for item in aggregates if item["passed"]]
    ranked = sorted(
        [item for item in aggregates if item["paper_candidate"]],
        key=lambda item: (
            item["macro_ratio_vs_reference_envelope"],
            item["macro_ratio_vs_a2"],
        ),
    )
    champion = None
    if eligible:
        champion = min(
            eligible,
            key=lambda item: (
                item["macro_ratio_vs_reference_envelope"],
                item["macro_ratio_vs_a2"],
            ),
        )["candidate"]
    decision = {
        "protocol": "pctf-fusion-v1-validation-only",
        "passed": bool(eligible),
        "champion": champion,
        "best_observed_paper_candidate": ranked[0]["candidate"],
        "negative_controls_eligible": False,
        "gate": {
            "macro_ratio_vs_a2_max": 0.998,
            "both_metric_improve_datasets_vs_a2_min": 4,
            "worst_ratio_vs_a2_max": 1.01,
            "macro_ratio_vs_reference_envelope_max": 1.005,
            "macro_ratio_vs_control_envelope_max": 1.0,
        },
        "aggregates": aggregates,
        "design_comparisons": comparisons,
        "test_metrics_read": False,
    }
    root = _root(args)
    _write_csv(root / "screen_summary.csv", details)
    _write_json(root / "screen_decision.json", decision)
    print(f"wrote {root / 'screen_summary.csv'}")
    print(f"wrote {root / 'screen_decision.json'}")
    return 0


def summarize_formal(args):
    champion = _frozen_champion(args)
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    modes = REFERENCES + (champion,)
    expected = {
        (dataset, horizon, mode, seed)
        for dataset in datasets for horizon in horizons
        for mode in modes for seed in seeds
    }
    rows = _collect(_root(args) / "formal", expected)
    summary, keyed = [], {}
    for dataset in datasets:
        for horizon in horizons:
            gold_mse, gold_mae = GOLDEN[(dataset, horizon)]
            for mode in modes:
                group = [rows[(dataset, horizon, mode, seed)] for seed in seeds]
                mses = [float(row["test_mse"]) for row in group]
                maes = [float(row["test_mae"]) for row in group]
                mse_mean, mae_mean = statistics.mean(mses), statistics.mean(maes)
                mse_std = statistics.stdev(mses) if len(mses) > 1 else 0.0
                mae_std = statistics.stdev(maes) if len(maes) > 1 else 0.0
                item = {
                    "dataset": dataset, "horizon": horizon, "mode": mode,
                    "mse_mean": mse_mean, "mse_std": mse_std,
                    "mae_mean": mae_mean, "mae_std": mae_std,
                    "golden_mse": gold_mse, "golden_mae": gold_mae,
                    "stable_below_golden": (
                        all(value < gold_mse for value in mses)
                        and all(value < gold_mae for value in maes)
                        and mse_mean + mse_std < gold_mse
                        and mae_mean + mae_std < gold_mae
                    ),
                }
                summary.append(item)
                keyed[(dataset, horizon, mode)] = item

    for item in summary:
        a2 = keyed[(item["dataset"], item["horizon"], INCUMBENT)]
        item["mse_ratio_vs_a2"] = item["mse_mean"] / a2["mse_mean"]
        item["mae_ratio_vs_a2"] = item["mae_mean"] / a2["mae_mean"]
        item["mse_ratio_vs_reference_envelope"] = item["mse_mean"] / min(
            keyed[(item["dataset"], item["horizon"], mode)]["mse_mean"]
            for mode in REFERENCES
        )
        item["mae_ratio_vs_reference_envelope"] = item["mae_mean"] / min(
            keyed[(item["dataset"], item["horizon"], mode)]["mae_mean"]
            for mode in REFERENCES
        )

    candidate_rows = [item for item in summary if item["mode"] == champion]
    ratios = [
        value for item in candidate_rows
        for value in (item["mse_ratio_vs_a2"], item["mae_ratio_vs_a2"])
    ]
    decision = {
        "protocol": "pctf-fusion-v1-frozen-three-seed-test",
        "champion": champion,
        "macro_ratio_vs_a2": statistics.mean(ratios),
        "both_metric_improve_settings_vs_a2": sum(
            item["mse_ratio_vs_a2"] < 1.0 and item["mae_ratio_vs_a2"] < 1.0
            for item in candidate_rows
        ),
        "worst_ratio_vs_a2": max(ratios),
        "stable_below_golden_settings": sum(
            item["stable_below_golden"] for item in candidate_rows
        ),
    }
    decision["eligible_to_replace_a2"] = (
        decision["macro_ratio_vs_a2"] < 0.998
        and decision["both_metric_improve_settings_vs_a2"] >= 8
        and decision["worst_ratio_vs_a2"] <= 1.005
    )
    root = _root(args)
    _write_csv(root / "formal_summary.csv", summary)
    _write_json(root / "formal_decision.json", decision)
    print(f"wrote {root / 'formal_summary.csv'}")
    print(f"wrote {root / 'formal_decision.json'}")
    return 0


def _run(commands, execute):
    print(f"commands={len(commands)}")
    for command in commands:
        print(shlex.join(command))
    if not execute:
        return 0
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "screen-dry", "screen", "screen-summarize",
            "confirm-dry", "confirm", "confirm-summarize",
        ),
        required=True,
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--champion", choices=PAPER_CANDIDATES,
        help="dry-run preview only; formal runs require the frozen decision",
    )
    args = parser.parse_args()

    if args.stage == "screen-summarize":
        return summarize_screen(args)
    if args.stage == "confirm-summarize":
        return summarize_formal(args)
    if args.stage.startswith("screen"):
        return _run(screen_commands(args), args.stage == "screen")
    commands, _ = formal_commands(
        args, dry_override=args.stage == "confirm-dry"
    )
    return _run(commands, args.stage == "confirm")


if __name__ == "__main__":
    raise SystemExit(main())
