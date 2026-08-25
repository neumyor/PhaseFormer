#!/usr/bin/env python3
"""Resumable driver for the residual-topology experiment plan.

The script intentionally runs jobs sequentially on the CUDA device selected by
the caller.  ``screen`` is validation-only; ``full`` is reserved for frozen
candidates and uses the benchmark runner's best-checkpoint test protocol.
Use ``--dry-run`` to audit the complete command matrix without training.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = "ETTh1:336,ETTh2:720,ETTm1:720,Electricity:336"
ALL_MODES = [
    "original",
    "residual_output_convex",
    "residual_output_additive",
    "residual_latent_long",
    "residual_latent_layerwise",
    "residual_hybrid",
    "residual_output_layerwise_convex",
    "residual_output_layerwise_additive",
]


def parse_settings(value):
    settings = []
    for item in value.split(","):
        dataset, horizon = item.strip().split(":", 1)
        settings.append((dataset, int(horizon)))
    return settings


def screen_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        for mode in args.modes:
            commands.append(
                [
                    sys.executable,
                    "scripts/search_phaseformer.py",
                    "--dataset",
                    dataset,
                    "--horizon",
                    str(horizon),
                    "--stage",
                    "mechanism_screen_1",
                    "--mechanism",
                    mode,
                    "--period",
                    "24",
                    "--percent",
                    "30",
                    "--max-epochs",
                    "8",
                    "--seed",
                    "2021",
                    "--loss",
                    "huber",
                    "--num-workers",
                    str(args.num_workers),
                    "--output-dir",
                    args.output_dir,
                    "--resume",
                ]
            )
    return commands


def full_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings):
        commands.append(
            [
                sys.executable,
                "scripts/benchmark_phaseformer_suite.py",
                "--datasets",
                dataset,
                "--horizons",
                str(horizon),
                "--modes",
                ",".join(args.modes),
                "--lookback",
                "720",
                "--seed",
                "2021",
                "--num-workers",
                str(args.num_workers),
                "--bad-case-limit",
                "8",
                "--output-dir",
                args.output_dir,
                "--run-prefix",
                f"residual_topology_full_{dataset}_{horizon}",
                "--resume",
            ]
        )
    return commands


def read_metric_rows(output_dir):
    rows = []
    for path in sorted((REPO_ROOT / output_dir).glob("**/metrics.csv")):
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
    rows = read_metric_rows(output_dir)
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mechanism"]): row
        for row in rows
        if row.get("stage") == "mechanism_screen_1"
    }
    summary = []
    for (dataset, horizon, mechanism), row in sorted(keyed.items()):
        baseline = keyed.get((dataset, horizon, "original"))
        if baseline is None:
            continue
        val_mae = float(row["val_mae"])
        val_mse = float(row["val_mse"])
        base_mae = float(baseline["val_mae"])
        base_mse = float(baseline["val_mse"])
        delta_mae = (base_mae - val_mae) / base_mae * 100.0
        delta_mse = (base_mse - val_mse) / base_mse * 100.0
        summary.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "mechanism": mechanism,
                "val_mae": f"{val_mae:.8f}",
                "val_mse": f"{val_mse:.8f}",
                "delta_mae_pct": f"{delta_mae:.4f}",
                "delta_mse_pct": f"{delta_mse:.4f}",
                "score": f"{0.5 * (delta_mae + delta_mse):.4f}",
                "parameter_count": row["parameter_count"],
                "elapsed_sec": row["elapsed_sec"],
                "run_id": row["run_id"],
            }
        )
    write_rows(REPO_ROOT / output_dir / "screen_summary.csv", summary)


def summarize_full(output_dir):
    rows = read_metric_rows(output_dir)
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mode"]): row
        for row in rows
        if row.get("mode") in ALL_MODES
    }
    summary = []
    for (dataset, horizon, mode), row in sorted(keyed.items()):
        baseline = keyed.get((dataset, horizon, "original"))
        if baseline is None:
            continue
        mae = float(row["test_mae"])
        mse = float(row["test_mse"])
        base_mae = float(baseline["test_mae"])
        base_mse = float(baseline["test_mse"])
        summary.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "mode": mode,
                "test_mae": f"{mae:.8f}",
                "test_mse": f"{mse:.8f}",
                "delta_mae_pct": f"{(base_mae - mae) / base_mae * 100.0:.4f}",
                "delta_mse_pct": f"{(base_mse - mse) / base_mse * 100.0:.4f}",
                "elapsed_sec": row["elapsed_sec"],
                "run_id": row["run_id"],
            }
        )
    write_rows(REPO_ROOT / output_dir / "full_summary.csv", summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "full"], required=True)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--modes",
        default=",".join(ALL_MODES),
        help="comma-separated; full stage should contain original and frozen candidates",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = sorted(set(args.modes) - set(ALL_MODES))
    if unknown:
        parser.error(f"unsupported residual topology modes: {unknown}")
    if args.stage == "full" and "original" not in args.modes:
        parser.error("full stage requires original for a matched comparison")
    if args.output_dir is None:
        args.output_dir = (
            "research_runs/residual_topology_screen_runs"
            if args.stage == "screen"
            else "research_runs/residual_topology_full_runs"
        )
    return args


def main():
    args = parse_args()
    commands = screen_commands(args) if args.stage == "screen" else full_commands(args)
    for index, command in enumerate(commands, 1):
        printable = " ".join(command)
        print(f"[{index}/{len(commands)}] {printable}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.dry_run:
        if args.stage == "screen":
            summarize_screen(args.output_dir)
        else:
            summarize_full(args.output_dir)


if __name__ == "__main__":
    main()
