#!/usr/bin/env python3
"""Aggregate Round 0/1 structured low-rank results into the plan's tables.

Reads every finished run under the Round 1 scratch root plus the reused controls
recorded by ``scripts/audit_structured_lowrank_reuse.py``, and writes:

- ``results.csv``: one row per candidate x setting with test MSE/MAE, deltas
  against the reused direct control, deltas against the reused generic
  ``pooled_lowrank(q=1/8)`` control, head parameter counts and the matched
  control comparison;
- ``results.md``: the plan's Round 1 route matrix and leaderboards.

This script computes no metric from predictions; it only joins recorded metrics,
so it can be re-run at any time while training continues.

Plan reference: sections 6 (route matrix), 11 (selection record) and 14
(metric definitions).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_ORDER = (
    "A_period_lowrank",
    "B_segment_basis",
    "C_level_shape",
    "D_recent_sparse",
    "E_separable",
    "A_period_lowrank_r8",
    "matched_time_axis_lowrank",
)

MATCHED_FOR = {
    "A_period_lowrank": "matched_time_axis_lowrank",
}


def read_csv_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def load_metrics(scratch_root):
    """candidate -> setting -> record for every finished run."""

    table = {}
    for config_path in sorted(Path(scratch_root).glob("*/runs/*/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        metrics = read_csv_rows(metrics_path)[0]
        if not metrics.get("test_mse"):
            continue
        hyper = config["hyperparams"]
        head = hyper.get("weak_period_residual_head_type", "shared")
        label = classify(head, hyper)
        setting = f"{config['dataset']}_h{config['horizon']}_seed{config['seed']}"
        record = {
            "setting": setting,
            "dataset": config["dataset"],
            "horizon": int(config["horizon"]),
            "seed": int(config["seed"]),
            "candidate": label,
            "head_type": head,
            "val_mse": float(metrics["val_mse"]),
            "val_mae": float(metrics["val_mae"]),
            "test_mse": float(metrics["test_mse"]),
            "test_mae": float(metrics["test_mae"]),
            "run_dir": str(run_dir),
            "gate_init": hyper.get("weak_period_residual_gate_init"),
            "params": int(metrics.get("parameter_count") or 0),
            "elapsed_sec": float(metrics.get("elapsed_sec") or 0.0),
            "peak_memory_bytes": int(float(metrics.get("peak_memory_bytes") or 0)),
        }
        table.setdefault(label, {})[setting] = record
    return table


def classify(head, hyper):
    if head == "time_axis_matched_lowrank":
        return "matched_time_axis_lowrank"
    named = {
        ("structured_period_lowrank", "residual_period_rank", 4): "A_period_lowrank",
        ("structured_segment_basis", "residual_basis_count", 4): "B_segment_basis",
        ("structured_level_shape", "residual_level_mode", "dense"): "C_level_shape",
        ("structured_recent_period", "residual_recent_taps", 7): "D_recent_sparse",
        ("structured_separable", "residual_separable_components", 1): "E_separable",
    }
    for (head_type, key, value), label in named.items():
        if head == head_type and hyper.get(key) == value:
            return label
    if head == "structured_period_lowrank":
        return f"A_period_lowrank_r{hyper.get('residual_period_rank')}"
    return head


def delta(control, value):
    if not control:
        return float("nan")
    return (control - value) / control * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--round0-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-params", help="JSON from report_structured_lowrank_params.py")
    args = parser.parse_args()

    scratch = Path(args.scratch_root)
    if not scratch.is_absolute():
        scratch = ROOT / scratch
    round0 = Path(args.round0_dir)
    if not round0.is_absolute():
        round0 = ROOT / round0
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    controls = {}
    for row in read_csv_rows(round0 / "round0_controls.csv"):
        controls.setdefault(row["setting"], {})[row["candidate"]] = {
            "test_mse": float(row["test_mse"]),
            "test_mae": float(row["test_mae"]),
            "val_mse": float(row["val_mse"]),
            "val_mae": float(row["val_mae"]),
            "source": row["source"],
            "source_run": row["source_run"],
        }

    table = load_metrics(scratch)
    settings = sorted({setting for per in table.values() for setting in per})

    rows = []
    for setting in settings:
        direct = controls.get(setting, {}).get("direct_nlinear")
        generic = controls.get(setting, {}).get("pooled_lowrank_q1/8")
        phase = controls.get(setting, {}).get("phase_only")
        for label in CANDIDATE_ORDER:
            record = table.get(label, {}).get(setting)
            if record is None:
                continue
            row = {
                **record,
                "delta_mse_vs_direct": delta(direct["test_mse"], record["test_mse"])
                if direct
                else float("nan"),
                "delta_mae_vs_direct": delta(direct["test_mae"], record["test_mae"])
                if direct
                else float("nan"),
                "delta_mse_vs_generic": delta(generic["test_mse"], record["test_mse"])
                if generic
                else float("nan"),
                "delta_mae_vs_generic": delta(generic["test_mae"], record["test_mae"])
                if generic
                else float("nan"),
                "direct_test_mse": direct["test_mse"] if direct else float("nan"),
                "direct_test_mae": direct["test_mae"] if direct else float("nan"),
                "generic_test_mse": generic["test_mse"] if generic else float("nan"),
                "generic_test_mae": generic["test_mae"] if generic else float("nan"),
                "phase_only_test_mse": phase["test_mse"] if phase else float("nan"),
                "phase_only_test_mae": phase["test_mae"] if phase else float("nan"),
                "control_source": direct["source"] if direct else "",
            }
            rows.append(row)

    fields = [
        "setting",
        "dataset",
        "horizon",
        "seed",
        "candidate",
        "head_type",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "delta_mse_vs_direct",
        "delta_mae_vs_direct",
        "delta_mse_vs_generic",
        "delta_mae_vs_generic",
        "direct_test_mse",
        "direct_test_mae",
        "generic_test_mse",
        "generic_test_mae",
        "phase_only_test_mse",
        "phase_only_test_mae",
        "params",
        "elapsed_sec",
        "peak_memory_bytes",
        "gate_init",
        "control_source",
        "run_dir",
    ]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Round 1 — structured low-rank route matrix",
        "",
        "> **test-set selection.** Every number below comes from a run whose",
        "> validation-selected checkpoint was evaluated on the test split exactly",
        "> once. The reused controls inherit an existing test-exposed lineage. These",
        "> results are exploratory evidence, not a blind or unbiased estimate and not",
        "> a benchmark claim.",
        "",
        "Positive deltas mean the candidate has the lower error (better).",
        "",
        "## Route matrix",
        "",
        "| setting | candidate | test MSE | test MAE | dMSE% vs direct | dMAE% vs direct | "
        "dMSE% vs generic | dMAE% vs generic | params |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {row['candidate']} | {row['test_mse']:.6f} "
            f"| {row['test_mae']:.6f} | {row['delta_mse_vs_direct']:+.3f} "
            f"| {row['delta_mae_vs_direct']:+.3f} | {row['delta_mse_vs_generic']:+.3f} "
            f"| {row['delta_mae_vs_generic']:+.3f} | {row['params']} |"
        )

    lines += ["", "## MSE leaderboard per setting", "", "| setting | rank | candidate | test MSE |", "|---|---:|---|---:|"]
    for setting in settings:
        group = [row for row in rows if row["setting"] == setting]
        for rank, row in enumerate(sorted(group, key=lambda item: item["test_mse"]), 1):
            lines.append(f"| {setting} | {rank} | {row['candidate']} | {row['test_mse']:.6f} |")

    lines += ["", "## MAE leaderboard per setting", "", "| setting | rank | candidate | test MAE |", "|---|---:|---|---:|"]
    for setting in settings:
        group = [row for row in rows if row["setting"] == setting]
        for rank, row in enumerate(sorted(group, key=lambda item: item["test_mae"]), 1):
            lines.append(f"| {setting} | {rank} | {row['candidate']} | {row['test_mae']:.6f} |")

    lines += [
        "",
        "## Signal counts per route",
        "",
        "| route | settings with dMSE>0 | settings with dMAE>0 | settings with both>0 |",
        "|---|---:|---:|---:|",
    ]
    for label in CANDIDATE_ORDER:
        group = [row for row in rows if row["candidate"] == label]
        if not group:
            continue
        mse_positive = sum(1 for row in group if row["delta_mse_vs_direct"] > 0)
        mae_positive = sum(1 for row in group if row["delta_mae_vs_direct"] > 0)
        both = sum(
            1
            for row in group
            if row["delta_mse_vs_direct"] > 0 and row["delta_mae_vs_direct"] > 0
        )
        lines.append(
            f"| {label} | {mse_positive}/{len(group)} | {mae_positive}/{len(group)} "
            f"| {both}/{len(group)} |"
        )

    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {output/'results.csv'} ({len(rows)} rows) and {output/'results.md'}")
    for label in CANDIDATE_ORDER:
        group = [row for row in rows if row["candidate"] == label]
        if not group:
            continue
        print(
            f"  {label:28s} n={len(group):2d} "
            f"dMSE mean {sum(r['delta_mse_vs_direct'] for r in group)/len(group):+7.3f}  "
            f"dMAE mean {sum(r['delta_mae_vs_direct'] for r in group)/len(group):+7.3f}"
        )


if __name__ == "__main__":
    main()
