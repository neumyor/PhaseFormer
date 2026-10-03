#!/usr/bin/env python3
"""Aggregate trend-token validation jobs and produce an auditable report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    search_root = root / "search" / "runs"
    rows = []
    diagnostics = []
    cases = []
    for run_dir in sorted(search_root.glob("*")):
        metrics = read_csv(run_dir / "metrics.csv")
        if not metrics:
            continue
        row = dict(metrics[-1])
        config = {}
        if (run_dir / "config.json").exists():
            config = json.loads((run_dir / "config.json").read_text())
        row["config_id"] = config.get("config_hash", run_dir.name)
        row["arm"] = config.get("mechanism", row.get("mechanism", ""))
        row["setting"] = f"{row.get('dataset','')}_h{row.get('horizon','')}_s{row.get('seed','')}"
        rows.append(row)
        for split in ("val", "test"):
            diagnostics.extend(read_csv(run_dir / f"{split}_trend_diagnostics.csv"))
        for case in read_csv(run_dir / "bad_cases" / "val_bad_cases.csv"):
            case = dict(case)
            case["arm"] = row["arm"]
            case["setting"] = row["setting"]
            case["run_dir"] = str(run_dir)
            cases.append(case)

    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        baseline = {}
        for row in rows:
            key = (row.get("dataset"), row.get("horizon"), row.get("seed"))
            if row.get("arm") in {"phase_only", "original"}:
                baseline[key] = row
        fields = sorted({key for row in rows for key in row} | {"delta_mse", "delta_mae"})
        for row in rows:
            base = baseline.get((row.get("dataset"), row.get("horizon"), row.get("seed")))
            row["delta_mse"] = f(f(row.get("val_mse")) - f(base.get("val_mse"))) if base else ""
            row["delta_mae"] = f(f(row.get("val_mae")) - f(base.get("val_mae"))) if base else ""
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    if diagnostics:
        fields = sorted({key for row in diagnostics for key in row})
        with (out / "trend_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostics)

    # The search runner stores only its programmatically selected high-error
    # cases.  Preserve those references and add a compact sample-level table.
    if cases:
        fields = sorted({key for row in cases for key in row})
        with (out / "sample_errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cases)

    report_lines = [
        "# Trend-token validation report",
        "",
        "## Experiment setup",
        "",
        f"- Experiment directory: `{root}`",
        "- Selection: validation loss only; no test loader was requested.",
        "- Arms: phase-only, dense residual, profile trend-token controls, and direct-horizon trend-token controls with and without phase interaction.",
        "- Profile arms are constrained to the fixed profile bank; direct-horizon arms are an explicit higher-output-space control.",
        "",
        "## Results",
        "",
    ]
    if rows:
        report_lines.append("| setting | arm | val MAE | val MSE | delta MAE vs phase-only | params | device |")
        report_lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda x: (x.get("setting", ""), x.get("arm", ""))):
            report_lines.append(
                f"| {row.get('setting','')} | {row.get('arm','')} | {row.get('val_mae','')} | {row.get('val_mse','')} | {row.get('delta_mae','')} | {row.get('parameter_count','')} | {row.get('device_type','')} |"
            )
    else:
        report_lines.append("No completed jobs were found yet.")

    if diagnostics:
        report_lines.extend(["", "## Theory diagnostics", ""])
        report_lines.append("| arm | n | mean gate | mean pooling recency | corr(lambda, level demand) | phase MAE | trend MAE | fused MAE |")
        report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        arms = sorted({row.get("arm", "") for row in rows})
        for arm in arms:
            # Diagnostics do not repeat arm in the legacy CSV, so match by the
            # run's setting through the same search run files below.
            arm_rows = []
            for run_dir in sorted(search_root.glob("*")):
                config_path = run_dir / "config.json"
                if not config_path.exists() or json.loads(config_path.read_text()).get("mechanism") != arm:
                    continue
                arm_rows.extend(read_csv(run_dir / "val_trend_diagnostics.csv"))
            if not arm_rows:
                continue
            x = np.asarray([
                f(r.get("lambda_0"))
                if r.get("lambda_0", "") not in ("", None)
                else f(r.get("direct_delta_mean"))
                for r in arm_rows
            ])
            y = np.asarray([f(r.get("level_demand")) for r in arm_rows])
            mask = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() > 1 else float("nan")
            report_lines.append(
                f"| {arm} | {len(arm_rows)} | {mean([f(r.get('gate')) for r in arm_rows]):.4f} | {mean([f(r.get('pool_recency')) for r in arm_rows]):.4f} | {corr:.4f} | {mean([f(r.get('phase_mae')) for r in arm_rows]):.6f} | {mean([f(r.get('trend_mae')) for r in arm_rows]):.6f} | {mean([f(r.get('fused_mae')) for r in arm_rows]):.6f} |"
            )
        report_lines.extend([
            "",
                "The state correlation uses profile lambda for profile arms and direct-delta mean for direct arms. These diagnostics support the proposed complement interpretation only when they repeat across seeds and settings; they are not causal proof by themselves.",
        ])

    report_lines.extend(["", "## High-error cases", ""])
    if cases:
        for case in sorted(cases, key=lambda x: f(x.get("mae")), reverse=True)[: args.top_k]:
            report_lines.append(
                f"- `{case.get('setting')}` {case.get('arm')}: sample {case.get('sample_index')}, channel {case.get('variable')}, MAE {case.get('mae')}, mode `{case.get('error_mode')}`."
            )
    else:
        report_lines.append("No selected cases are available yet.")

    report_lines.extend([
        "",
        "## Scope and restart",
        "",
        "This is a validation-only screen. A completed job is recognized by its underlying `metrics.csv` and `status=completed`; rerun the same command with `--resume` to skip those jobs and retry incomplete ones. Formal test-set confirmation and the canonical six-file error-analysis package require a completed screen.",
    ])
    report = "\n".join(report_lines) + "\n"
    (out / "objective_error_analysis.md").write_text(report, encoding="utf-8")
    with zipfile.ZipFile(out / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out / "objective_error_analysis.md", "objective_error_analysis.md")

    # A compact machine-readable manifest for downstream audits.
    (out / "analysis_summary.json").write_text(
        json.dumps({"runs": len(rows), "diagnostic_rows": len(diagnostics), "cases": len(cases)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(rows), "diagnostic_rows": len(diagnostics), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
