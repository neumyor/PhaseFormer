#!/usr/bin/env python3
"""Run the pre-registered ICPT full-horizon position-encoding experiment.

The screen is validation-only. It compares the matched RCRF-NLinear baseline,
an anchor control, and every full-horizon PE candidate on four settings. A
candidate is frozen before any formal test command can be generated.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SETTINGS = (
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("Electricity", 336),
    ("Weather", 336),
)
SEEDS = (2021, 2022, 2023)
TRAIN = {
    ("ETTh2", 720): {"loss": "huber", "lr": 0.001, "batch": 256},
    ("ETTm2", 96): {"loss": "mae", "lr": 0.0003, "batch": 256},
    ("Electricity", 336): {"loss": "mae", "lr": 0.0003, "batch": 16},
    ("Weather", 336): {"loss": "mae", "lr": 0.0003, "batch": 256},
}
GOLDEN = {
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Electricity", 336): (0.165, 0.257),
    ("Weather", 336): (0.242, 0.278),
}

A2 = "gold_combo_reliability_s2"
C0 = "rcrf_icpt_horizon_cycle_anchor"
INDEX_MODES = (
    "rcrf_icpt_horizon_none",
    "rcrf_icpt_horizon_sincos",
    "rcrf_icpt_horizon_learned_abs",
    "rcrf_icpt_horizon_time2vec",
    "rcrf_icpt_horizon_rope",
    "rcrf_icpt_horizon_relative",
    "rcrf_icpt_horizon_alibi",
    "rcrf_icpt_horizon_lff",
    "rcrf_icpt_horizon_sincos_relative",
)
CALENDAR = "rcrf_icpt_horizon_calendar"
SCREEN_MODES = (A2, C0) + INDEX_MODES + (CALENDAR,)

SCREEN_OUTPUT = "research_runs/phaseformer_icpt_horizon_pe_screen"
FULL_OUTPUT = "research_runs/phaseformer_icpt_horizon_pe_full"


def parse_settings(value):
    selected = []
    for item in value.split(","):
        dataset, horizon = item.strip().split(":", 1)
        setting = (dataset, int(horizon))
        if setting not in SETTINGS:
            raise ValueError(f"unsupported setting: {setting}")
        selected.append(setting)
    return selected


def read_rows(output_dir):
    rows, seen = [], set()
    root = REPO_ROOT / output_dir
    for pattern in ("runs/*/metrics.csv", "*/metrics.csv"):
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def screen_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        train = TRAIN[(dataset, horizon)]
        for mode in SCREEN_MODES:
            commands.append([
                sys.executable,
                "scripts/search_phaseformer.py",
                "--dataset", dataset,
                "--horizon", str(horizon),
                "--stage", "mechanism_screen_2",
                "--mechanism", mode,
                "--period", "24",
                "--lookback", "720",
                "--percent", "30",
                "--max-epochs", "8",
                "--seed", "2021",
                "--loss", train["loss"],
                "--learning-rate", str(train["lr"]),
                "--batch-size", str(train["batch"]),
                "--num-workers", str(args.num_workers),
                "--output-dir", args.output_dir,
                "--resume",
            ])
    return commands


def summarize_screen(output_dir):
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mechanism"]): row
        for row in read_rows(output_dir)
        if row.get("stage") == "mechanism_screen_2"
    }
    summary = []
    for dataset, horizon in SETTINGS:
        baseline = keyed.get((dataset, horizon, A2))
        p0 = keyed.get((dataset, horizon, INDEX_MODES[0]))
        if baseline is None:
            continue
        base_mse = float(baseline["val_mse"])
        base_mae = float(baseline["val_mae"])
        p0_mse = float(p0["val_mse"]) if p0 else None
        p0_mae = float(p0["val_mae"]) if p0 else None
        for mode in SCREEN_MODES:
            row = keyed.get((dataset, horizon, mode))
            if row is None:
                continue
            mse = float(row["val_mse"])
            mae = float(row["val_mae"])
            item = {
                "dataset": dataset,
                "horizon": horizon,
                "mode": mode,
                "val_mse": f"{mse:.8f}",
                "val_mae": f"{mae:.8f}",
                "mse_ratio_vs_a2": f"{mse / base_mse:.8f}",
                "mae_ratio_vs_a2": f"{mae / base_mae:.8f}",
                "mse_ratio_vs_p0": "" if p0_mse is None else f"{mse / p0_mse:.8f}",
                "mae_ratio_vs_p0": "" if p0_mae is None else f"{mae / p0_mae:.8f}",
                "parameter_count": row.get("parameter_count", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "config_hash": row.get("config_hash", ""),
                "run_id": row.get("run_id", ""),
            }
            summary.append(item)
    write_rows(REPO_ROOT / output_dir / "screen_summary.csv", summary)
    return summary


def candidate_scores(summary, modes):
    scores = {}
    for mode in modes:
        rows = [row for row in summary if row["mode"] == mode]
        if len(rows) != len(SETTINGS):
            continue
        pairs = [
            (float(row["mse_ratio_vs_a2"]), float(row["mae_ratio_vs_a2"]))
            for row in rows
        ]
        ratios = [value for pair in pairs for value in pair]
        mean_ratio = sum(ratios) / len(ratios)
        worst_ratio = max(ratios)
        both_improve = sum(mse < 1.0 and mae < 1.0 for mse, mae in pairs)
        worst_regression = max(max(pair) - 1.0 for pair in pairs)
        eligible = (
            (mean_ratio < 1.0 and worst_ratio <= 1.01)
            or (both_improve >= 3 and worst_regression <= 0.005)
        )
        scores[mode] = {
            "mean_ratio_vs_a2": mean_ratio,
            "worst_ratio_vs_a2": worst_ratio,
            "both_metric_improve_settings": both_improve,
            "worst_regression_vs_a2": worst_regression,
            "parameter_count": max(int(row["parameter_count"]) for row in rows),
            "elapsed_sec": sum(float(row["elapsed_sec"]) for row in rows),
            "eligible": eligible,
        }
    return scores


def freeze(output_dir):
    summary = summarize_screen(output_dir)
    index_scores = candidate_scores(summary, INDEX_MODES)
    calendar_scores = candidate_scores(summary, (CALENDAR,))
    eligible = [(mode, info) for mode, info in index_scores.items() if info["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["mean_ratio_vs_a2"],
            item[1]["worst_ratio_vs_a2"],
            item[1]["parameter_count"],
            item[1]["elapsed_sec"],
        ),
    )
    frozen = ranked[0][0] if ranked else None
    calendar = calendar_scores.get(CALENDAR)
    calendar_eligible = bool(calendar and calendar["eligible"])
    record = {
        "frozen_index_candidate": frozen,
        "calendar_eligible": calendar_eligible,
        "selection_source": "validation_only",
        "test_read_before_freeze": False,
        "eligibility_rule": (
            "relative to matched RCRF-NLinear A2: mean of 8 ratios <1 and "
            "worst <=1.01, or >=3/4 settings improve both metrics with all "
            "remaining regressions <=0.5%"
        ),
        "index_scores": index_scores,
        "index_ranking": [mode for mode, _ in ranked],
        "calendar_scores": calendar_scores,
        "screen_passed": bool(frozen or calendar_eligible),
    }
    path = REPO_ROOT / output_dir / "freeze_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if record["screen_passed"] else 2


def read_freeze(output_dir):
    path = REPO_ROOT / output_dir / "freeze_record.json"
    if not path.exists():
        raise RuntimeError("freeze record missing; run --stage freeze first")
    record = json.loads(path.read_text())
    if not record.get("screen_passed"):
        raise RuntimeError("screen failed; formal test is not allowed")
    return record


def full_commands(args):
    record = read_freeze(args.screen_output_dir)
    modes = [A2, INDEX_MODES[0]]
    frozen = record.get("frozen_index_candidate")
    if frozen and frozen not in modes:
        modes.append(frozen)
    if record.get("calendar_eligible"):
        modes.append(CALENDAR)
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        train = TRAIN[(dataset, horizon)]
        for seed in args.seeds:
            commands.append([
                sys.executable,
                "scripts/benchmark_phaseformer_suite.py",
                "--datasets", dataset,
                "--horizons", str(horizon),
                "--modes", ",".join(modes),
                "--lookback", "720",
                "--seed", str(seed),
                "--loss", train["loss"],
                "--learning-rate", str(train["lr"]),
                "--batch-size", str(train["batch"]),
                "--num-workers", str(args.num_workers),
                "--bad-case-limit", "8",
                "--output-dir", args.output_dir,
                "--run-prefix", f"icpt_horizon_{dataset}_{horizon}",
                "--resume",
            ])
    return commands


def summarize_full(output_dir, screen_output_dir):
    record = read_freeze(screen_output_dir)
    modes = {A2, INDEX_MODES[0], record.get("frozen_index_candidate")}
    if record.get("calendar_eligible"):
        modes.add(CALENDAR)
    modes.discard(None)
    summary = []
    for row in read_rows(output_dir):
        if row.get("mode") not in modes or not row.get("test_mse"):
            continue
        setting = (row["dataset"], int(row["horizon"]))
        if setting not in SETTINGS:
            continue
        mse = float(row["test_mse"])
        mae = float(row["test_mae"])
        golden_mse, golden_mae = GOLDEN[setting]
        summary.append({
            "dataset": setting[0],
            "horizon": setting[1],
            "seed": int(row["seed"]),
            "mode": row["mode"],
            "test_mse": f"{mse:.8f}",
            "test_mae": f"{mae:.8f}",
            "golden_mse": golden_mse,
            "golden_mae": golden_mae,
            "delta_mse_pct_vs_golden": f"{(golden_mse - mse) / golden_mse * 100:.4f}",
            "delta_mae_pct_vs_golden": f"{(golden_mae - mae) / golden_mae * 100:.4f}",
            "parameter_count": row.get("parameter_count", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "run_id": row.get("run_id", ""),
        })
    summary.sort(key=lambda row: (row["dataset"], row["horizon"], row["seed"], row["mode"]))
    write_rows(REPO_ROOT / output_dir / "full_summary.csv", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "freeze", "full"), required=True)
    parser.add_argument("--settings", default=",".join(f"{d}:{h}" for d, h in SETTINGS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir")
    parser.add_argument("--screen-output-dir", default=SCREEN_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed)
    if args.output_dir is None:
        args.output_dir = SCREEN_OUTPUT if args.stage in ("screen", "freeze") else FULL_OUTPUT
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
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if args.dry_run:
        return 0
    if args.stage == "screen":
        summarize_screen(args.output_dir)
        return 0
    if args.stage == "freeze":
        return freeze(args.output_dir)
    summarize_full(args.output_dir, args.screen_output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
