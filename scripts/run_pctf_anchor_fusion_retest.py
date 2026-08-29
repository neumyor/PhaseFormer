#!/usr/bin/env python3
"""Retest A2-anchored PCTF fusion without validation/test leakage.

The protocol has three frozen stages:

1. select one ICPT cycle period per dataset on validation only;
2. compare repaired fusion strategies at the selected periods on validation;
3. run H96/H192, three-seed test confirmation only for the frozen champion.

Every generated training command requires CUDA.  Summarization also rejects a
mixed device/software matrix and rejects any test metric in the two selection
stages.
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
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)
PHASE_PERIOD = 24
PERIOD_CANDIDATES = {
    "ETTh1": (12, 24, 48),
    "ETTh2": (12, 24, 48),
    "ETTm1": (24, 48, 96),
    "ETTm2": (24, 48, 96),
    "Weather": (12, 24, 48),
    "Electricity": (12, 24, 48),
}
REFERENCES = (
    "gold_combo_reliability_s2",  # A1
    "rcrf_pe_lff",                # A2 incumbent and exact anchor
    "rcrf_icpt_none",             # I0 cycle expert reference
)
INCUMBENT = "rcrf_pe_lff"
DIAGNOSTICS = ("pctf_dual_fixed",)  # best legacy PCTF, not eligible
ABLATIONS = (
    "pctf_anchor_shape_only",
    "pctf_anchor_level_only",
)
PAPER_CANDIDATES = (
    "pctf_anchor_component_scalar",
    "pctf_anchor_component_cycle",
    "pctf_anchor_monotonic",
    "pctf_anchor_mlp",
    "pctf_anchor_phase_modulation",
)
ANCHORED_MODES = ABLATIONS + PAPER_CANDIDATES
SCREEN_MODES = REFERENCES + DIAGNOSTICS + ANCHORED_MODES
PERIOD_PROBE = "pctf_anchor_component_cycle"
DEFAULT_OUTPUT_ROOT = "research_runs/pctf_anchor_fusion_v2"
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
    value = Path(args.output_root)
    return value if value.is_absolute() else REPO_ROOT / value


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


def _command(
    args, dataset, horizon, seed, mode, *, output_stage,
    cycle_period=None, formal=False,
):
    command = [
        sys.executable,
        "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm" if formal else "mechanism_screen_2",
        "--mechanism", mode,
        "--period", str(PHASE_PERIOD),
        "--lookback", "720",
        "--percent", "100" if formal else "30",
        "--max-epochs", "30" if formal else "12",
        "--seed", str(seed),
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(_root(args) / output_stage),
        "--require-cuda",
        "--resume",
    ]
    if cycle_period is not None:
        command.extend(("--cycle-period", str(cycle_period)))
    if formal:
        command.append("--evaluate-test")
    if args.progress:
        command.append("--progress")
    return command


def period_commands(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    commands = []
    for dataset in datasets:
        for horizon in horizons:
            commands.append(_command(
                args, dataset, horizon, 2021, INCUMBENT,
                output_stage="period",
            ))
            for period in PERIOD_CANDIDATES[dataset]:
                commands.append(_command(
                    args, dataset, horizon, 2021, PERIOD_PROBE,
                    output_stage="period", cycle_period=period,
                ))
    return commands


def _period_map(args, *, dry_override=False):
    if getattr(args, "period_map", ""):
        values = json.loads(args.period_map)
        result = {dataset: int(values[dataset]) for dataset in DATASETS if dataset in values}
    else:
        path = _root(args) / "period_decision.json"
        if path.is_file():
            decision = json.loads(path.read_text())
            result = {
                dataset: int(period)
                for dataset, period in decision["selected_periods"].items()
            }
        elif dry_override:
            result = {dataset: 24 for dataset in DATASETS}
        else:
            raise RuntimeError(
                "period_decision.json missing; period selection must be frozen first"
            )
    datasets = _parse_csv(args.datasets, DATASETS)
    missing = sorted(set(datasets) - set(result))
    if missing:
        raise ValueError(f"selected period missing for datasets: {missing}")
    for dataset in datasets:
        if result[dataset] not in PERIOD_CANDIDATES[dataset]:
            raise ValueError(
                f"invalid frozen period for {dataset}: {result[dataset]}"
            )
    return result


def screen_commands(args, *, dry_override=False):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    periods = _period_map(args, dry_override=dry_override)
    return [
        _command(
            args, dataset, horizon, 2021, mode,
            output_stage="screen",
            cycle_period=periods[dataset] if mode in ANCHORED_MODES else None,
        )
        for dataset in datasets
        for horizon in horizons
        for mode in SCREEN_MODES
    ]


def _frozen_champion(args, *, dry_override=False):
    if dry_override and args.champion:
        if args.champion not in PAPER_CANDIDATES:
            raise ValueError("only paper candidates can be frozen")
        return args.champion
    path = _root(args) / "screen_decision.json"
    if not path.is_file():
        raise RuntimeError("screen_decision.json missing; screen must be summarized")
    decision = json.loads(path.read_text())
    if not decision.get("passed"):
        raise RuntimeError("no repaired candidate passed; formal test is blocked")
    champion = decision.get("champion")
    if champion not in PAPER_CANDIDATES:
        raise RuntimeError(f"invalid frozen champion: {champion}")
    return champion


def formal_commands(args, *, dry_override=False):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    periods = _period_map(args, dry_override=dry_override)
    champion = _frozen_champion(args, dry_override=dry_override)
    modes = REFERENCES + (champion,)
    commands = [
        _command(
            args, dataset, horizon, seed, mode,
            output_stage="formal",
            cycle_period=periods[dataset] if mode == champion else None,
            formal=True,
        )
        for dataset in datasets
        for horizon in horizons
        for seed in seeds
        for mode in modes
    ]
    return commands, champion


def _cycle_period(row):
    value = str(row.get("cycle_period", "")).strip()
    return int(value) if value else None


def _key(row):
    return (
        row.get("dataset"), int(row.get("horizon", -1)),
        row.get("mechanism"), int(row.get("seed", -1)),
        _cycle_period(row),
    )


def _collect(root, expected):
    rows = {}
    environments = []
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values:
            continue
        row = values[0]
        key = _key(row)
        if key not in expected:
            continue
        if key in rows:
            raise RuntimeError(f"duplicate result for {key}")
        environment_path = path.parent / "environment.json"
        if not environment_path.is_file():
            raise RuntimeError(f"environment audit missing for {key}")
        environment = json.loads(environment_path.read_text())
        if not environment.get("cuda_available") or not environment.get("gpu"):
            raise RuntimeError(f"non-CUDA run in paired matrix: {key}")
        rows[key] = row
        environments.append(environment)
    missing = sorted(expected - set(rows), key=str)
    if missing:
        preview = "\n".join(f"  {item}" for item in missing[:20])
        raise RuntimeError(f"matrix incomplete: {len(missing)} missing\n{preview}")
    signatures = {
        (
            item.get("gpu"), item.get("torch"), item.get("cuda_runtime"),
            item.get("lightning"),
        )
        for item in environments
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "heterogeneous device/software matrix: "
            f"{sorted(signatures, key=str)}"
        )
    return rows, next(iter(signatures))


def _assert_validation_only(rows):
    leaked = [
        key for key, row in rows.items()
        if str(row.get("test_mse", "")).strip()
        or str(row.get("test_mae", "")).strip()
    ]
    if leaked:
        raise RuntimeError(f"selection stage contains test metrics: {leaked[:5]}")


def _assert_anchor_identity(rows, modes):
    failed = []
    for key, row in rows.items():
        if row.get("mechanism") not in modes:
            continue
        value = str(row.get("anchor_identity_max_abs", "")).strip()
        if not value or float(value) != 0.0:
            failed.append((key, value))
    if failed:
        raise RuntimeError(f"candidate is not exact A2 at initialization: {failed[:5]}")


def summarize_period(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    expected = set()
    for dataset in datasets:
        for horizon in horizons:
            expected.add((dataset, horizon, INCUMBENT, 2021, None))
            expected.update(
                (dataset, horizon, PERIOD_PROBE, 2021, period)
                for period in PERIOD_CANDIDATES[dataset]
            )
    rows, signature = _collect(_root(args) / "period", expected)
    _assert_validation_only(rows)
    _assert_anchor_identity(rows, {PERIOD_PROBE})

    details, selected = [], {}
    for dataset in datasets:
        candidates = []
        for period in PERIOD_CANDIDATES[dataset]:
            ratios = []
            for horizon in horizons:
                probe = rows[(dataset, horizon, PERIOD_PROBE, 2021, period)]
                anchor = rows[(dataset, horizon, INCUMBENT, 2021, None)]
                mse_ratio = float(probe["val_mse"]) / float(anchor["val_mse"])
                mae_ratio = float(probe["val_mae"]) / float(anchor["val_mae"])
                ratios.extend((mse_ratio, mae_ratio))
                details.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "cycle_period": period,
                    "val_mse": float(probe["val_mse"]),
                    "val_mae": float(probe["val_mae"]),
                    "mse_ratio_vs_a2": mse_ratio,
                    "mae_ratio_vs_a2": mae_ratio,
                })
            candidates.append((statistics.mean(ratios), max(ratios), period))
        selected[dataset] = min(candidates)[2]

    decision = {
        "protocol": "pctf-anchor-v2-period-validation-only",
        "selected_periods": selected,
        "phase_period_fixed": PHASE_PERIOD,
        "selection_key": "lowest macro ratio vs A2, then lowest worst ratio",
        "environment_signature": signature,
        "test_metrics_read": False,
    }
    root = _root(args)
    _write_csv(root / "period_summary.csv", details)
    _write_json(root / "period_decision.json", decision)
    return 0


def _macro_pair(rows, datasets, horizons, periods, left, right):
    ratios, double = [], 0
    for dataset in datasets:
        for horizon in horizons:
            left_period = periods[dataset] if left in ANCHORED_MODES else None
            right_period = periods[dataset] if right in ANCHORED_MODES else None
            left_row = rows[(dataset, horizon, left, 2021, left_period)]
            right_row = rows[(dataset, horizon, right, 2021, right_period)]
            mse = float(left_row["val_mse"]) / float(right_row["val_mse"])
            mae = float(left_row["val_mae"]) / float(right_row["val_mae"])
            ratios.extend((mse, mae))
            double += mse < 1.0 and mae < 1.0
    return statistics.mean(ratios), int(double), max(ratios)


def summarize_screen(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    periods = _period_map(args)
    expected = {
        (
            dataset, horizon, mode, 2021,
            periods[dataset] if mode in ANCHORED_MODES else None,
        )
        for dataset in datasets for horizon in horizons for mode in SCREEN_MODES
    }
    rows, signature = _collect(_root(args) / "screen", expected)
    _assert_validation_only(rows)
    _assert_anchor_identity(rows, set(ANCHORED_MODES))

    details, aggregates = [], []
    benchmark_modes = REFERENCES + DIAGNOSTICS
    for candidate in ANCHORED_MODES:
        ratios_a2, ratios_envelope = [], []
        double = 0
        for dataset in datasets:
            for horizon in horizons:
                period = periods[dataset]
                item = rows[(dataset, horizon, candidate, 2021, period)]
                a2 = rows[(dataset, horizon, INCUMBENT, 2021, None)]
                mse, mae = float(item["val_mse"]), float(item["val_mae"])
                a2_mse, a2_mae = float(a2["val_mse"]), float(a2["val_mae"])
                envelope_mse = min(
                    float(rows[(dataset, horizon, mode, 2021, None)]["val_mse"])
                    for mode in benchmark_modes
                )
                envelope_mae = min(
                    float(rows[(dataset, horizon, mode, 2021, None)]["val_mae"])
                    for mode in benchmark_modes
                )
                mse_ratio, mae_ratio = mse / a2_mse, mae / a2_mae
                mse_envelope, mae_envelope = mse / envelope_mse, mae / envelope_mae
                ratios_a2.extend((mse_ratio, mae_ratio))
                ratios_envelope.extend((mse_envelope, mae_envelope))
                double += mse_ratio < 1.0 and mae_ratio < 1.0
                details.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "cycle_period": period,
                    "candidate": candidate,
                    "paper_candidate": candidate in PAPER_CANDIDATES,
                    "val_mse": mse,
                    "val_mae": mae,
                    "mse_ratio_vs_a2": mse_ratio,
                    "mae_ratio_vs_a2": mae_ratio,
                    "mse_ratio_vs_reference_envelope": mse_envelope,
                    "mae_ratio_vs_reference_envelope": mae_envelope,
                })
        aggregate = {
            "candidate": candidate,
            "paper_candidate": candidate in PAPER_CANDIDATES,
            "macro_ratio_vs_a2": statistics.mean(ratios_a2),
            "both_metric_improve_settings_vs_a2": int(double),
            "worst_ratio_vs_a2": max(ratios_a2),
            "macro_ratio_vs_reference_envelope": statistics.mean(ratios_envelope),
        }
        aggregate["passed"] = (
            aggregate["paper_candidate"]
            and aggregate["macro_ratio_vs_a2"] <= 0.998
            and aggregate["both_metric_improve_settings_vs_a2"] >= 8
            and aggregate["worst_ratio_vs_a2"] <= 1.01
            and aggregate["macro_ratio_vs_reference_envelope"] <= 1.005
        )
        aggregates.append(aggregate)

    comparisons = []
    for child, parent, question in (
        ("pctf_anchor_component_cycle", "pctf_anchor_component_scalar",
         "Do lead-specific coefficients beat shared coefficients?"),
        ("pctf_anchor_monotonic", "pctf_anchor_component_cycle",
         "Does matched causal evidence improve the anchored correction?"),
        ("pctf_anchor_mlp", "pctf_anchor_component_cycle",
         "Does an unconstrained evidence MLP improve the anchored correction?"),
        ("pctf_anchor_phase_modulation", "pctf_anchor_component_cycle",
         "Does constrained phase modulation beat free component correction?"),
        ("pctf_anchor_component_cycle", "pctf_anchor_shape_only",
         "Does relative cycle-level correction add value?"),
        ("pctf_anchor_component_cycle", "pctf_anchor_level_only",
         "Does within-cycle shape correction add value?"),
    ):
        macro, count, worst = _macro_pair(
            rows, datasets, horizons, periods, child, parent
        )
        comparisons.append({
            "child": child,
            "parent": parent,
            "question": question,
            "macro_ratio": macro,
            "double_improve_settings": count,
            "worst_ratio": worst,
        })

    eligible = [item for item in aggregates if item["passed"]]
    paper = [item for item in aggregates if item["paper_candidate"]]
    ranking = sorted(
        paper,
        key=lambda item: (
            item["macro_ratio_vs_reference_envelope"],
            item["macro_ratio_vs_a2"],
            item["worst_ratio_vs_a2"],
        ),
    )
    champion = None
    if eligible:
        champion = min(
            eligible,
            key=lambda item: (
                item["macro_ratio_vs_reference_envelope"],
                item["macro_ratio_vs_a2"],
                item["worst_ratio_vs_a2"],
            ),
        )["candidate"]
    decision = {
        "protocol": "pctf-anchor-v2-strategy-validation-only",
        "passed": bool(eligible),
        "champion": champion,
        "best_observed_paper_candidate": ranking[0]["candidate"],
        "selected_periods": periods,
        "ablations_eligible": False,
        "gate": {
            "macro_ratio_vs_a2_max": 0.998,
            "both_metric_improve_settings_vs_a2_min": 8,
            "worst_ratio_vs_a2_max": 1.01,
            "macro_ratio_vs_reference_envelope_max": 1.005,
        },
        "aggregates": aggregates,
        "design_comparisons": comparisons,
        "environment_signature": signature,
        "test_metrics_read": False,
    }
    root = _root(args)
    _write_csv(root / "screen_summary.csv", details)
    _write_json(root / "screen_decision.json", decision)
    return 0


def summarize_formal(args):
    datasets = _parse_csv(args.datasets, DATASETS)
    horizons = _parse_csv(args.horizons, HORIZONS, int)
    seeds = _parse_csv(args.seeds, SEEDS, int)
    periods = _period_map(args)
    champion = _frozen_champion(args)
    modes = REFERENCES + (champion,)
    expected = {
        (
            dataset, horizon, mode, seed,
            periods[dataset] if mode == champion else None,
        )
        for dataset in datasets for horizon in horizons
        for seed in seeds for mode in modes
    }
    rows, signature = _collect(_root(args) / "formal", expected)
    _assert_anchor_identity(rows, {champion})
    empty_test = [
        key for key, row in rows.items()
        if not str(row.get("test_mse", "")).strip()
        or not str(row.get("test_mae", "")).strip()
    ]
    if empty_test:
        raise RuntimeError(f"formal matrix lacks test metrics: {empty_test[:5]}")

    summary, keyed = [], {}
    for dataset in datasets:
        for horizon in horizons:
            golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
            for mode in modes:
                period = periods[dataset] if mode == champion else None
                group = [
                    rows[(dataset, horizon, mode, seed, period)]
                    for seed in seeds
                ]
                mses = [float(row["test_mse"]) for row in group]
                maes = [float(row["test_mae"]) for row in group]
                item = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "cycle_period": period if period is not None else "",
                    "mode": mode,
                    "mse_mean": statistics.mean(mses),
                    "mse_std": statistics.stdev(mses) if len(mses) > 1 else 0.0,
                    "mae_mean": statistics.mean(maes),
                    "mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0,
                    "golden_mse": golden_mse,
                    "golden_mae": golden_mae,
                }
                item["stable_below_golden"] = (
                    all(value < golden_mse for value in mses)
                    and all(value < golden_mae for value in maes)
                    and item["mse_mean"] + item["mse_std"] < golden_mse
                    and item["mae_mean"] + item["mae_std"] < golden_mae
                )
                summary.append(item)
                keyed[(dataset, horizon, mode)] = item

    for item in summary:
        a2 = keyed[(item["dataset"], item["horizon"], INCUMBENT)]
        item["mse_ratio_vs_a2"] = item["mse_mean"] / a2["mse_mean"]
        item["mae_ratio_vs_a2"] = item["mae_mean"] / a2["mae_mean"]

    candidates = [item for item in summary if item["mode"] == champion]
    ratios = [
        ratio for item in candidates
        for ratio in (item["mse_ratio_vs_a2"], item["mae_ratio_vs_a2"])
    ]
    decision = {
        "protocol": "pctf-anchor-v2-frozen-three-seed-test",
        "champion": champion,
        "selected_periods": periods,
        "macro_ratio_vs_a2": statistics.mean(ratios),
        "both_metric_improve_settings_vs_a2": sum(
            item["mse_ratio_vs_a2"] < 1.0 and item["mae_ratio_vs_a2"] < 1.0
            for item in candidates
        ),
        "worst_ratio_vs_a2": max(ratios),
        "stable_below_golden_settings": sum(
            item["stable_below_golden"] for item in candidates
        ),
        "environment_signature": signature,
    }
    decision["eligible_to_replace_a2"] = (
        decision["macro_ratio_vs_a2"] < 0.998
        and decision["both_metric_improve_settings_vs_a2"] >= 8
        and decision["worst_ratio_vs_a2"] <= 1.005
    )
    root = _root(args)
    _write_csv(root / "formal_summary.csv", summary)
    _write_json(root / "formal_decision.json", decision)
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
            "period-dry", "period", "period-summarize",
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
        "--period-map",
        help="JSON dataset-to-period map for dry-run preview only",
    )
    parser.add_argument(
        "--champion", choices=PAPER_CANDIDATES,
        help="dry-run preview only; actual confirm requires frozen decision",
    )
    args = parser.parse_args()

    if args.stage == "period-summarize":
        return summarize_period(args)
    if args.stage == "screen-summarize":
        return summarize_screen(args)
    if args.stage == "confirm-summarize":
        return summarize_formal(args)
    if args.stage.startswith("period"):
        return _run(period_commands(args), args.stage == "period")
    if args.stage.startswith("screen"):
        return _run(
            screen_commands(args, dry_override=args.stage == "screen-dry"),
            args.stage == "screen",
        )
    commands, _ = formal_commands(
        args, dry_override=args.stage == "confirm-dry"
    )
    return _run(commands, args.stage == "confirm")


if __name__ == "__main__":
    raise SystemExit(main())
