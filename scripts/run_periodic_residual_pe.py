#!/usr/bin/env python3
"""Run the validation-screen and frozen confirm for periodic residual PE.

Stage A is validation-only (30% data, 8 epochs, seed 2021).  It compares the
current RCRF with seven position encodings and freezes one candidate by the
pre-registered cross-setting ratio rule.  Stage B trains only current RCRF and
the frozen candidate on full data for seeds 2021/2022/2023, then evaluates the
best-validation checkpoints on test.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (("ETTh2", 720), ("ETTm2", 96), ("Electricity", 336))
SEEDS = (2021, 2022, 2023)
BASELINE = "gold_combo_reliability_s2"
CANDIDATES = (
    "rcrf_pe_st",
    "rcrf_pe_cycle",
    "rcrf_pe_harmonic",
    "rcrf_pe_traffic",
    "rcrf_pe_time2vec",
    "rcrf_pe_lff",
    "rcrf_pe_calendar",
)
SCREEN_MODES = (BASELINE,) + CANDIDATES
SETTING_TRAIN = {
    ("ETTh2", 720): {"loss": "huber", "lr": 0.001},
    ("ETTm2", 96): {"loss": "mae", "lr": 0.0003},
    ("Electricity", 336): {"loss": "mae", "lr": 0.0003},
}
GOLDEN = {
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Electricity", 336): (0.165, 0.257),
}
SCREEN_OUTPUT = "research_runs/periodic_residual_pe_screen"
FULL_OUTPUT = "research_runs/periodic_residual_pe_full"
FREEZE_RECORD = REPO_ROOT / SCREEN_OUTPUT / "freeze_record.json"


def parse_settings(value):
    result = []
    for item in value.split(","):
        dataset, horizon = item.strip().split(":", 1)
        setting = (dataset, int(horizon))
        if setting not in SETTINGS:
            raise ValueError(f"unsupported setting: {setting}")
        result.append(setting)
    return result


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
                    "--stage", "mechanism_screen_2",
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
    if not FREEZE_RECORD.exists():
        raise RuntimeError("freeze record is missing; run --stage freeze first")
    freeze = json.loads(FREEZE_RECORD.read_text())
    frozen = freeze.get("frozen_candidate")
    if not frozen:
        raise RuntimeError("Stage A found no eligible PE candidate")
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
                    "--modes", f"{BASELINE},{frozen}",
                    "--lookback", "720",
                    "--seed", str(seed),
                    "--loss", train["loss"],
                    "--learning-rate", str(train["lr"]),
                    "--num-workers", str(args.num_workers),
                    "--bad-case-limit", "8",
                    "--output-dir", args.output_dir,
                    "--run-prefix", f"periodic_residual_pe_full_{dataset}_{horizon}",
                    "--resume",
                ]
            )
    return commands


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


def summarize_screen(output_dir):
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mechanism"]): row
        for row in read_rows(output_dir)
        if row.get("stage") == "mechanism_screen_2"
    }
    summary = []
    for dataset, horizon in SETTINGS:
        base = keyed.get((dataset, horizon, BASELINE))
        if base is None:
            continue
        base_mse, base_mae = float(base["val_mse"]), float(base["val_mae"])
        for mode in SCREEN_MODES:
            row = keyed.get((dataset, horizon, mode))
            if row is None:
                continue
            mse, mae = float(row["val_mse"]), float(row["val_mae"])
            summary.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "mode": mode,
                    "val_mse": f"{mse:.8f}",
                    "val_mae": f"{mae:.8f}",
                    "mse_ratio_vs_rcrf": f"{mse / base_mse:.8f}",
                    "mae_ratio_vs_rcrf": f"{mae / base_mae:.8f}",
                    "parameter_count": row.get("parameter_count", ""),
                    "epochs_completed": row.get("epochs_completed", ""),
                    "test_mse": row.get("test_mse", ""),
                    "test_mae": row.get("test_mae", ""),
                    "config_hash": row.get("config_hash", ""),
                    "run_id": row.get("run_id", ""),
                }
            )
    write_rows(REPO_ROOT / output_dir / "screen_summary.csv", summary)
    return summary


def freeze(output_dir):
    summary = summarize_screen(output_dir)
    scores = {}
    for mode in CANDIDATES:
        rows = [row for row in summary if row["mode"] == mode]
        if len(rows) != len(SETTINGS):
            continue
        ratios = [
            value
            for row in rows
            for value in (
                float(row["mse_ratio_vs_rcrf"]),
                float(row["mae_ratio_vs_rcrf"]),
            )
        ]
        scores[mode] = {
            "mean_ratio": sum(ratios) / len(ratios),
            "worst_ratio": max(ratios),
            "parameter_count": max(int(row["parameter_count"]) for row in rows),
            "eligible": sum(ratios) / len(ratios) < 1.0 and max(ratios) <= 1.01,
        }
    if len(scores) != len(CANDIDATES):
        missing = sorted(set(CANDIDATES) - set(scores))
        raise RuntimeError(f"incomplete Stage A candidates: {missing}")
    eligible = [(mode, info) for mode, info in scores.items() if info["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["mean_ratio"],
            item[1]["worst_ratio"],
            item[1]["parameter_count"],
        ),
    )
    frozen = ranked[0][0] if ranked else None
    record = {
        "frozen_candidate": frozen,
        "selection_source": "validation_only",
        "test_read_before_freeze": False,
        "eligibility_rule": "mean_ratio<1 and worst_ratio<=1.01 versus current RCRF",
        "scores": scores,
        "ranking": [mode for mode, _ in ranked],
        "note": "No post-screen hyperparameter changes are allowed before Stage B.",
    }
    FREEZE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if frozen else 2


def summarize_full(output_dir):
    rows = read_rows(output_dir)
    freeze = json.loads(FREEZE_RECORD.read_text())
    frozen = freeze["frozen_candidate"]
    summary = []
    for row in rows:
        mode = row.get("mode")
        if mode not in (BASELINE, frozen):
            continue
        dataset, horizon = row["dataset"], int(row["horizon"])
        if (dataset, horizon) not in SETTINGS:
            continue
        mse, mae = float(row["test_mse"]), float(row["test_mae"])
        golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
        summary.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "seed": int(row["seed"]),
                "mode": mode,
                "test_mse": f"{mse:.8f}",
                "test_mae": f"{mae:.8f}",
                "golden_mse": golden_mse,
                "golden_mae": golden_mae,
                "delta_mse_pct_vs_golden": f"{(golden_mse - mse) / golden_mse * 100:.4f}",
                "delta_mae_pct_vs_golden": f"{(golden_mae - mae) / golden_mae * 100:.4f}",
                "epochs_completed": row.get("epochs_completed", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "run_id": row.get("run_id", ""),
            }
        )
    summary.sort(key=lambda row: (row["dataset"], row["horizon"], row["seed"], row["mode"]))
    write_rows(REPO_ROOT / output_dir / "full_summary.csv", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "freeze", "full"), required=True)
    parser.add_argument(
        "--settings", default=",".join(f"{d}:{h}" for d, h in SETTINGS)
    )
    parser.add_argument("--modes", default=",".join(SCREEN_MODES))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.modes = tuple(item for item in args.modes.split(",") if item)
    args.seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    unknown = sorted(set(args.modes) - set(SCREEN_MODES))
    if unknown:
        parser.error(f"unsupported modes: {unknown}")
    if args.stage == "screen":
        args.output_dir = args.output_dir or SCREEN_OUTPUT
    elif args.stage == "full":
        args.output_dir = args.output_dir or FULL_OUTPUT
    else:
        args.output_dir = args.output_dir or SCREEN_OUTPUT
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
    summarize_full(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
