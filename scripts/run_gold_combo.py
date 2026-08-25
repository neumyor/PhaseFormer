#!/usr/bin/env python3
"""Resumable driver for the gold_combo_stability_v1 experiment.

Stage A (``screen``) is validation-only: 3 settings x 6 modes on 30% data, max 8
epochs, seed 2021.  Stage A never creates a test loader (search_phaseformer.py
restricts --evaluate-test to frozen confirm runs).

Stage B (``full``) is reserved for frozen candidates: original / latest / the
frozen gold_combo mode x 3 settings x 3 seeds, full data, base-preset
epochs/patience, best-validation checkpoint, test evaluation via the benchmark
runner.  ``freeze`` reads the Stage A results, computes the pre-registered
6-ratio score among the four gold_combo_* candidates and writes the freeze
record before any test is read.

The script intentionally runs jobs sequentially on the CUDA device selected by
the caller.  Use ``--dry-run`` to audit the complete command matrix without
training.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GOLDEN = {
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Electricity", 336): (0.165, 0.257),
}

SETTINGS = [("ETTh2", 720), ("ETTm2", 96), ("Electricity", 336)]
SCREEN_MODES = [
    "original",
    "latest",
    "gold_combo_fixed",
    "gold_combo_adaptive",
    "gold_combo_reliability_s0",
    "gold_combo_reliability_s2",
]
CANDIDATE_MODES = [
    "gold_combo_fixed",
    "gold_combo_adaptive",
    "gold_combo_reliability_s0",
    "gold_combo_reliability_s2",
]
FULL_SEEDS = [2021, 2022, 2023]

# Per-setting training config frozen in the plan section 3 (applies to every
# mode run on that setting, Stage A and Stage B).
SETTING_TRAIN = {
    ("ETTh2", 720): {"loss": "huber", "lr": 0.001},
    ("ETTm2", 96): {"loss": "mae", "lr": 0.0003},
    ("Electricity", 336): {"loss": "mae", "lr": 0.0003},
}

SCREEN_OUTPUT = "research_runs/gold_combo_screen_runs"
FULL_OUTPUT = "research_runs/gold_combo_full_runs"
FREEZE_RECORD = Path(REPO_ROOT) / "research_runs/gold_combo_screen_runs/freeze_record.json"


def parse_settings(value):
    settings = []
    for item in value.split(","):
        dataset, horizon = item.strip().split(":", 1)
        settings.append((dataset, int(horizon)))
    return settings


def screen_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        train = SETTING_TRAIN[(dataset, horizon)]
        for mode in args.modes:
            commands.append(
                [
                    sys.executable,
                    "scripts/search_phaseformer.py",
                    "--dataset", dataset,
                    "--horizon", str(horizon),
                    "--stage", "mechanism_screen_1",
                    "--mechanism", mode,
                    "--period", "24",
                    "--percent", "30",
                    "--max-epochs", "8",
                    "--seed", "2021",
                    "--loss", train["loss"],
                    "--learning-rate", str(train["lr"]),
                    "--num-workers", str(args.num_workers),
                    "--output-dir", args.output_dir,
                    "--resume",
                ]
            )
    return commands


def full_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        train = SETTING_TRAIN[(dataset, horizon)]
        for seed in args.seeds:
            commands.append(
                [
                    sys.executable,
                    "scripts/benchmark_phaseformer_suite.py",
                    "--datasets", dataset,
                    "--horizons", str(horizon),
                    "--modes", ",".join(args.modes),
                    "--lookback", "720",
                    "--seed", str(seed),
                    "--loss", train["loss"],
                    "--learning-rate", str(train["lr"]),
                    "--num-workers", str(args.num_workers),
                    "--bad-case-limit", "8",
                    "--output-dir", args.output_dir,
                    "--run-prefix", f"gold_combo_full_{dataset}_{horizon}",
                    "--resume",
                ]
            )
    return commands


def read_metric_rows(output_dir, stage_filter=None):
    rows = []
    seen = set()
    # search_phaseformer.py writes under output_dir/runs/<rid>/metrics.csv while
    # the benchmark runner writes directly under output_dir/<run_id>/metrics.csv.
    for pattern in ["runs/*/metrics.csv", "*/metrics.csv"]:
        for path in sorted((REPO_ROOT / output_dir).glob(pattern)):
            if str(path) in seen:
                continue
            seen.add(str(path))
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if stage_filter is None or row.get("stage") == stage_filter:
                        rows.append(row)
    return rows


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _screen_cell_rows(output_dir):
    rows = read_metric_rows(output_dir, stage_filter="mechanism_screen_1")
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mechanism"]): row
        for row in rows
    }
    return keyed


def summarize_screen(output_dir):
    keyed = _screen_cell_rows(output_dir)
    summary = []
    for dataset, horizon in SETTINGS:
        base = keyed.get((dataset, horizon, "original"))
        if base is None:
            continue
        base_mse = float(base["val_mse"])
        base_mae = float(base["val_mae"])
        for mode in SCREEN_MODES:
            row = keyed.get((dataset, horizon, mode))
            if row is None:
                continue
            val_mse = float(row["val_mse"])
            val_mae = float(row["val_mae"])
            summary.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "mode": mode,
                    "val_mse": f"{val_mse:.8f}",
                    "val_mae": f"{val_mae:.8f}",
                    "mse_ratio": f"{val_mse / base_mse:.6f}",
                    "mae_ratio": f"{val_mae / base_mae:.6f}",
                    "epochs_completed": row.get("epochs_completed", ""),
                    "parameter_count": row.get("parameter_count", ""),
                    "test_mse": row.get("test_mse", ""),
                    "test_mae": row.get("test_mae", ""),
                    "config_hash": row.get("config_hash", ""),
                    "run_id": row.get("run_id", ""),
                }
            )
    write_rows(REPO_ROOT / output_dir / "screen_summary.csv", summary)
    return summary


def compute_scores(summary):
    """Pre-registered Stage A score: mean over settings x metrics of the
    candidate_val/original_val ratio.  Lower is better."""
    ratios = {mode: [] for mode in CANDIDATE_MODES}
    worst = {mode: [] for mode in CANDIDATE_MODES}
    params = {}
    for row in summary:
        if row["mode"] not in CANDIDATE_MODES:
            continue
        ratios[row["mode"]].append(float(row["mse_ratio"]))
        ratios[row["mode"]].append(float(row["mae_ratio"]))
        worst[row["mode"]].append(max(float(row["mse_ratio"]), float(row["mae_ratio"])))
        if row["parameter_count"]:
            params[row["mode"]] = max(
                params.get(row["mode"], 0), int(row["parameter_count"])
            )
    scores = {}
    for mode in CANDIDATE_MODES:
        if ratios[mode]:
            scores[mode] = {
                "score": sum(ratios[mode]) / len(ratios[mode]),
                "worst_ratio": max(worst[mode]),
                "parameter_count": params.get(mode, None),
            }
    return scores


def freeze(output_dir):
    summary = summarize_screen(output_dir)
    scores = compute_scores(summary)
    complete = {m: s for m, s in scores.items() if len([r for r in summary if r["mode"] == m]) == len(SETTINGS)}
    if len(complete) != len(CANDIDATE_MODES):
        missing = [m for m in CANDIDATE_MODES if m not in complete]
        print(f"FREEZE BLOCKED: incomplete candidates {missing}", flush=True)
        return 1
    ranked = sorted(
        complete.items(),
        key=lambda kv: (
            kv[1]["score"],
            kv[1]["parameter_count"] if kv[1]["parameter_count"] is not None else math.inf,
        ),
    )
    winner, winner_info = ranked[0]
    sensitivities = {
        "gold_combo_fixed": 0,
        "gold_combo_adaptive": 0,
        "gold_combo_reliability_s0": 0,
        "gold_combo_reliability_s2": 2,
    }
    record = {
        "frozen_candidate": winner,
        "selection_source": "validation_only",
        "test_read_before_freeze": False,
        "scores": {m: s["score"] for m, s in complete.items()},
        "worst_ratios": {m: s["worst_ratio"] for m, s in complete.items()},
        "parameter_counts": {m: s["parameter_count"] for m, s in complete.items()},
        "ranking": [(m, round(complete[m]["score"], 6)) for m, _ in ranked],
        "tiebreak_sensitivity": sensitivities,
        "note": "Only the four gold_combo_* candidates are eligible. Tiebreak: fewer parameters, then lower initial sensitivity.",
    }
    FREEZE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def summarize_full(output_dir):
    rows = read_metric_rows(output_dir)
    keyed = {}
    for row in rows:
        keyed[(row["dataset"], int(row["horizon"]), int(row["seed"]), row["mode"])] = row
    summary = []
    for dataset, horizon in SETTINGS:
        golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
        for mode in ["original", "latest", "gold_combo_fixed", "gold_combo_adaptive",
                     "gold_combo_reliability_s0", "gold_combo_reliability_s2"]:
            for seed in FULL_SEEDS:
                row = keyed.get((dataset, horizon, seed, mode))
                if row is None:
                    continue
                mse = float(row["test_mse"])
                mae = float(row["test_mae"])
                summary.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "mode": mode,
                        "scheme": row.get("scheme", mode),
                        "test_mse": f"{mse:.8f}",
                        "test_mae": f"{mae:.8f}",
                        "delta_mse_pct_vs_golden": f"{(golden_mse - mse) / golden_mse * 100.0:.4f}",
                        "delta_mae_pct_vs_golden": f"{(golden_mae - mae) / golden_mae * 100.0:.4f}",
                        "epochs_completed": row.get("epochs_completed", ""),
                        "run_id": row.get("run_id", ""),
                    }
                )
    write_rows(REPO_ROOT / output_dir / "full_summary.csv", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "freeze", "full"], required=True)
    parser.add_argument(
        "--settings",
        default=",".join(f"{d}:{h}" for d, h in SETTINGS),
    )
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.modes is None:
        args.modes = SCREEN_MODES if args.stage == "screen" else None
    else:
        args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.seeds is None:
        args.seeds = FULL_SEEDS
    else:
        args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.stage == "screen":
        unknown = sorted(set(args.modes) - set(SCREEN_MODES))
        if unknown:
            parser.error(f"unsupported screen modes: {unknown}")
        if args.output_dir is None:
            args.output_dir = SCREEN_OUTPUT
    elif args.stage == "full":
        if args.modes is None:
            parser.error("full stage requires --modes (original,latest,frozen)")
        unknown = sorted(set(args.modes) - set(SCREEN_MODES))
        if unknown:
            parser.error(f"unsupported full modes: {unknown}")
        if "original" not in args.modes:
            parser.error("full stage requires original for a matched comparison")
        if args.output_dir is None:
            args.output_dir = FULL_OUTPUT
    else:
        if args.output_dir is None:
            args.output_dir = SCREEN_OUTPUT
    return args


def main():
    args = parse_args()
    if args.stage == "screen":
        commands = screen_commands(args)
    elif args.stage == "full":
        commands = full_commands(args)
    else:
        commands = []
    for index, command in enumerate(commands, 1):
        printable = " ".join(command)
        print(f"[{index}/{len(commands)}] {printable}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.dry_run:
        if args.stage == "screen":
            summarize_screen(args.output_dir)
        elif args.stage == "full":
            summarize_full(args.output_dir)
        elif args.stage == "freeze":
            return freeze(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
