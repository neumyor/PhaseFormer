#!/usr/bin/env python3
"""Summarize one-mechanism trend-token validation matrices.

The input is the ``search`` directory produced by
``run_trend_token_validation.py``.  The summarizer checks that every candidate
differs from B2 in exactly one declared hyperparameter before computing paired
validation deltas.  It never opens a test loader or selects a configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = (
    "phase_only",
    "trend_token_direct",
    "trend_token_dynamic_gate",
    "trend_token_smooth",
    "trend_token_multi_token",
    "trend_token_selective_attention",
    "trend_token_factorized",
    "trend_token_multiscale",
    "trend_token_robust",
)

HYPOTHESIS_FIELDS = {
    "trend_token_dynamic_gate": "trend_gate_mode",
    "trend_token_smooth": "trend_direct_smooth_window",
    "trend_token_multi_token": "trend_token_count",
    "trend_token_selective_attention": "trend_phase_attention_mode",
    "trend_token_factorized": "trend_readout_type",
    "trend_token_multiscale": "trend_history_multiscale",
    "trend_token_robust": "trend_history_robust",
}

B2_DEFAULTS = {
    "trend_token_dim": 32,
    "trend_encoder_kernel": 5,
    "trend_gate_init": 0.2,
    "trend_readout_type": "direct",
    "trend_use_phase_attention": True,
    "trend_gate_mode": "static",
    "trend_direct_smooth_window": 0,
    "trend_token_count": 1,
    "trend_phase_attention_mode": "dense",
    "trend_history_multiscale": False,
    "trend_history_robust": False,
}


def read_matrix(search_dir: Path):
    rows = []
    for metrics_path in sorted(search_dir.glob("*/metrics.csv")):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        if not metrics:
            continue
        row = metrics[-1]
        if row.get("mechanism") not in ARMS:
            continue
        config = json.loads((metrics_path.parent / "config.json").read_text())
        row["setting"] = f"{row['dataset']}_h{row['horizon']}_seed{row['seed']}"
        row["config"] = config
        rows.append(row)
    return rows


def check_controls(rows):
    by_setting = defaultdict(dict)
    for row in rows:
        by_setting[row["setting"]][row["mechanism"]] = row
    control_rows = []
    for setting, arms in sorted(by_setting.items()):
        b2 = arms.get("trend_token_direct")
        if b2 is None:
            continue
        base_hp = b2["config"].get("hyperparams", {})
        for arm, field in HYPOTHESIS_FIELDS.items():
            candidate = arms.get(arm)
            if candidate is None:
                continue
            hp = candidate["config"].get("hyperparams", {})
            keys = set(B2_DEFAULTS) | set(base_hp) | set(hp)
            changed = {
                key for key in keys
                if hp.get(key, B2_DEFAULTS.get(key))
                != base_hp.get(key, B2_DEFAULTS.get(key))
            }
            control_rows.append({
                "setting": setting,
                "arm": arm,
                "declared_field": field,
                "changed_fields": ",".join(sorted(changed)),
                "control_pass": changed == {field},
            })
    return control_rows


def write_summary(rows, controls, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    control_ok = all(item["control_pass"] for item in controls) if controls else False
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["setting"]][row["mechanism"]] = row
    summary = []
    for arm in ARMS:
        values = [arms[arm] for arms in grouped.values() if arm in arms]
        if not values:
            continue
        mae = np.asarray([float(x["val_mae"]) for x in values])
        mse = np.asarray([float(x["val_mse"]) for x in values])
        b2_mae = np.asarray([
            float(arms["trend_token_direct"]["val_mae"])
            for arms in grouped.values()
            if arm in arms and "trend_token_direct" in arms
        ])
        b2_mse = np.asarray([
            float(arms["trend_token_direct"]["val_mse"])
            for arms in grouped.values()
            if arm in arms and "trend_token_direct" in arms
        ])
        if arm == "trend_token_direct":
            delta_mae = delta_mse = 0.0
            wins_mae = wins_mse = len(values)
            paired_n = len(values)
        elif arm == "phase_only":
            delta_mae = float(mae.mean() - b2_mae.mean()) if len(b2_mae) else float("nan")
            delta_mse = float(mse.mean() - b2_mse.mean()) if len(b2_mse) else float("nan")
            wins_mae = wins_mse = 0
            paired_n = min(len(values), len(b2_mae))
        else:
            paired = [
                (float(arms[arm]["val_mae"]), float(arms["trend_token_direct"]["val_mae"]),
                 float(arms[arm]["val_mse"]), float(arms["trend_token_direct"]["val_mse"]))
                for arms in grouped.values()
                if arm in arms and "trend_token_direct" in arms
            ]
            candidate_mae = np.asarray([x[0] for x in paired])
            reference_mae = np.asarray([x[1] for x in paired])
            candidate_mse = np.asarray([x[2] for x in paired])
            reference_mse = np.asarray([x[3] for x in paired])
            delta_mae = float((candidate_mae - reference_mae).mean()) if len(paired) else float("nan")
            delta_mse = float((candidate_mse - reference_mse).mean()) if len(paired) else float("nan")
            wins_mae = int((candidate_mae < reference_mae).sum())
            wins_mse = int((candidate_mse < reference_mse).sum())
            paired_n = len(paired)
        control_for_arm = [x for x in controls if x["arm"] == arm]
        summary.append({
            "arm": arm,
            "settings": len(values),
            "paired_settings_vs_b2": paired_n,
            "mean_val_mae": float(mae.mean()),
            "mean_val_mse": float(mse.mean()),
            "delta_mae_vs_b2": delta_mae,
            "delta_mse_vs_b2": delta_mse,
            "mae_wins_vs_b2": wins_mae,
            "mse_wins_vs_b2": wins_mse,
            "control_pass": all(x["control_pass"] for x in control_for_arm) if control_for_arm else arm in {"phase_only", "trend_token_direct"},
        })
    fields = list(summary[0]) if summary else []
    with (output_dir / "hypothesis_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    report = [
        "# B2 One-Mechanism Hypothesis Validation",
        "",
        "Validation-only paired matrix. Every candidate is compared with B2 on the same dataset, horizon, and seed.",
        "No test split was loaded and no configuration was selected from test results.",
        "",
        f"- Settings observed: {len(grouped)}.",
        f"- Control-field checks: {'passed' if control_ok else 'incomplete'} ({len(controls)} candidate-setting checks).",
        "",
        "| arm | settings | mean MAE | mean MSE | delta MAE vs B2 | delta MSE vs B2 | MAE wins | MSE wins | control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        report.append(
            f"| {item['arm']} | {item['settings']} | {item['mean_val_mae']:.6f} | "
            f"{item['mean_val_mse']:.6f} | {item['delta_mae_vs_b2']:+.6f} | "
            f"{item['delta_mse_vs_b2']:+.6f} | {item['mae_wins_vs_b2']} | "
            f"{item['mse_wins_vs_b2']} | {item['control_pass']} |"
        )
    report += [
        "",
        "## Control Audit",
        "",
        "Each candidate must differ from B2 in exactly its declared field; the full per-setting audit is in `hypothesis_controls.csv`.",
    ]
    (output_dir / "hypothesis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (output_dir / "hypothesis_controls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(controls[0]) if controls else ["setting"])
        writer.writeheader()
        writer.writerows(controls)
    return summary, control_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_matrix(Path(args.search_dir))
    controls = check_controls(rows)
    summary, control_ok = write_summary(rows, controls, Path(args.output_dir))
    print(json.dumps({"rows": len(rows), "summary": summary, "control_pass": control_ok}, indent=2))
    if not control_ok:
        raise SystemExit("one-mechanism control audit failed or is incomplete")


if __name__ == "__main__":
    main()
