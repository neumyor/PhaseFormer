#!/usr/bin/env python3
"""Write the aggregate validation comparison used by the joint case figures."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_asymmetric_joint_route_cases import (
    BASELINE_ROOTS,
    COMPONENTS,
    DATASETS,
    candidate_path,
    run_dir,
)


OUT = ROOT / "research_runs/asymmetric_prediction_divergence_cases/XA_OnlyA_Baseline_validation_comparison.md"


def load_config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text())


def load_metrics(run: Path) -> tuple[float, float]:
    with (run / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one aggregate metrics row: {run}")
    row = rows[0]
    return float(row["val_mse"]), float(row["val_mae"])


def check_run(run: Path, dataset: str) -> None:
    cfg = load_config(run)
    if (cfg["dataset"], cfg["lookback"], cfg["horizon"], cfg["percent"]) != (dataset, 720, 96, 100):
        raise ValueError(f"Protocol mismatch: {run}")
    if cfg["hyperparams"].get("seed") != 2021:
        raise ValueError(f"Seed mismatch: {run}")


def metric_cell(mse: float, mae: float) -> str:
    return f"{mse:.6f} / {mae:.6f}"


def delta_cell(value: float, baseline: float) -> str:
    percent = 100.0 * value / baseline
    return f"{value:+.6f} ({percent:+.2f}%)"


def main() -> None:
    lines = [
        "# X-A、Only-A 与 Baseline-full：Validation 聚合结果",
        "",
        "范围：ETTh1、Weather、ETTm1，L=720→H=96，seed=2021，validation-only。",
        "每个数值为最佳 validation checkpoint 的 `MSE / MAE`（均越低越好）。",
        "`Δ` 定义为候选减去 Baseline-full：负值表示候选更好，正值表示候选更差。",
        "该表是聚合 validation 指标；不等同于 `manifest.csv` 中为放大 X-A/Only-A 预测分歧而筛选的局部样本误差。",
        "",
    ]
    provenance = []
    for dataset in DATASETS:
        baseline_run = run_dir(BASELINE_ROOTS[dataset], dataset, None)
        check_run(baseline_run, dataset)
        baseline_mse, baseline_mae = load_metrics(baseline_run)
        lines.extend([
            f"## {dataset}",
            "",
            "| 成分 | Baseline-full MSE / MAE | X-A MSE / MAE | X-A ΔMSE / ΔMAE | Only-A MSE / MAE | Only-A ΔMSE / ΔMAE |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for component in COMPONENTS:
            minus_run = candidate_path(dataset, component, "minus_component")
            only_run = candidate_path(dataset, component, "component_only")
            check_run(minus_run, dataset)
            check_run(only_run, dataset)
            minus_mse, minus_mae = load_metrics(minus_run)
            only_mse, only_mae = load_metrics(only_run)
            lines.append(
                f"| `{component}` | {metric_cell(baseline_mse, baseline_mae)} | "
                f"{metric_cell(minus_mse, minus_mae)} | "
                f"{delta_cell(minus_mse - baseline_mse, baseline_mse)} / "
                f"{delta_cell(minus_mae - baseline_mae, baseline_mae)} | "
                f"{metric_cell(only_mse, only_mae)} | "
                f"{delta_cell(only_mse - baseline_mse, baseline_mse)} / "
                f"{delta_cell(only_mae - baseline_mae, baseline_mae)} |"
            )
            provenance.append((dataset, component, baseline_run, minus_run, only_run))
        lines.append("")
    lines.extend([
        "## 指标来源与可比性",
        "",
        "- `Baseline-full`：PhaseFormer 与弱残差分支均接收完整 X。",
        "- `X-A`：PhaseFormer 接收完整 X，弱残差分支接收 X-A。",
        "- `Only-A`：PhaseFormer 接收完整 X，弱残差分支仅接收 A。",
        "- 三种设置均复用完整 X 的分支归一化统计；该差异表只描述这批单 seed validation checkpoint，不能作为多 seed 或 test 泛化结论。",
        "",
        "| Dataset | 成分 | Baseline-full checkpoint 目录 | X-A checkpoint 目录 | Only-A checkpoint 目录 |",
        "|---|---|---|---|---|",
    ])
    for dataset, component, baseline, minus, only in provenance:
        lines.append(
            f"| {dataset} | `{component}` | `{baseline.relative_to(ROOT)}` | "
            f"`{minus.relative_to(ROOT)}` | `{only.relative_to(ROOT)}` |"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
