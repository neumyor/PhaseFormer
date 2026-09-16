#!/usr/bin/env python3
"""Aggregate the top-2 direction-retention experiment into the plan's tables.

Reads every run's ``metrics.csv`` / ``config.json`` under the experiment root,
groups by (setting, seed, arm), and emits:

    projectors_table.md      plan table 1 (Stage 0 audit; copied from Stage 0)
    stage_a_validation.md    plan table 2 (seed 2021 validation only)
    final_test_table.md      plan table 3 (3-seed mean +- sample std)
    direction2_table.md      plan table 4 (direction-2 incremental value)
    decision_table.md        plan table 5 (pre-registered verdict)
    results.csv              machine-readable long table for all arms/seeds

Metric definitions follow the plan verbatim:

    delta_direct        = (metric_variant / metric_direct      - 1) * 100
    retention(V)        = (metric_phase_only - metric_V)
                          / (metric_phase_only - metric_direct)      * 100
    direction2_recovery = (metric_V1 - metric_V2)
                          / (metric_V1 - metric_direct)              * 100

``retention`` is reported as ``N/A`` whenever ``direct_nlinear`` is not better
than ``phase_only`` on that setting, per plan section 8.2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("phase_only", "direct_nlinear", "keep_direction_1", "keep_direction_1_2")
ARM_LABEL = {
    "phase_only": "phase-only",
    "direct_nlinear": "direct",
    "keep_direction_1": "V1",
    "keep_direction_1_2": "V2",
}
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
)
SEEDS = (2021, 2022, 2023)

# Plan section 9 thresholds.
MACRO_TOLERANCE_PCT = 0.5
RETENTION_MEDIAN_MIN_PCT = 90.0
WIN_COUNT_MIN = 4
WORST_CELL_TOLERANCE_PCT = 2.0
PARTIAL_RETENTION_MEDIAN_PCT = 80.0
NOT_SUPPORTED_WIN_COUNT_MAX = 3
NOT_SUPPORTED_WORST_CELLS = 2


def setting_key(dataset, horizon):
    return f"{dataset}_{horizon}"


def metric_pct(candidate, reference):
    if reference in (None, 0) or candidate is None:
        return None
    return (candidate / reference - 1.0) * 100.0


def retention(candidate, phase_only, direct):
    if candidate is None or phase_only is None or direct is None:
        return None
    denominator = phase_only - direct
    if denominator <= 0:
        return None  # plan 8.2: N/A when direct does not beat phase_only
    return (phase_only - candidate) / denominator * 100.0


def fmt(value, digits=4, scale=None):
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    if scale is not None:
        value = value * scale
    return f"{value:.{digits}f}"


def mean_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], None
    return statistics.fmean(clean), statistics.stdev(clean)


def load_runs(root: Path):
    """Collect one row per (setting, seed, arm) from every metrics.csv."""
    rows = {}
    for path in sorted(root.glob("runs/*/metrics.csv")):
        with path.open(newline="") as handle:
            record = next(csv.DictReader(handle), None)
        if record is None:
            continue
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        hyper = config.get("hyperparams", {})
        arm = hyper.get("weak_residual_projection_arm")
        if arm is None:
            # Legacy runs (the reused controls) predate the arm field.
            if config.get("mechanism") == "no_residual":
                arm = "phase_only"
            elif hyper.get("weak_period_residual_head_type") == "shared":
                arm = "direct_nlinear"
            else:
                continue
        key = (
            record["dataset"],
            int(record["horizon"]),
            int(record["seed"]),
            arm,
        )
        rows[key] = {
            "dataset": record["dataset"],
            "horizon": int(record["horizon"]),
            "seed": int(record["seed"]),
            "arm": arm,
            "val_mse": float(record["val_mse"]) if record.get("val_mse") else None,
            "val_mae": float(record["val_mae"]) if record.get("val_mae") else None,
            "test_mse": float(record["test_mse"]) if record.get("test_mse") else None,
            "test_mae": float(record.get("test_mae")) if record.get("test_mae") else None,
            "parameter_count": record.get("parameter_count", ""),
            "trainable_parameter_count": record.get("trainable_parameter_count", ""),
            "epochs_completed": record.get("epochs_completed", ""),
            "elapsed_sec": record.get("elapsed_sec", ""),
            "peak_memory_bytes": record.get("peak_memory_bytes", ""),
            "run_id": record.get("run_id", ""),
            "run_dir": str(path.parent.relative_to(ROOT)),
            "reused": bool(hyper.get("weak_residual_projection_arm") is None),
            "gate_init": hyper.get("weak_period_residual_gate_init", ""),
            "learning_rate": hyper.get("learning_rate", ""),
        }
    return rows


def setting_rows(rows, dataset, horizon, split):
    out = {}
    for seed in SEEDS:
        for arm in ARMS:
            record = rows.get((dataset, horizon, seed, arm))
            if record is None:
                continue
            out.setdefault(arm, {})[seed] = record.get(f"{split}_mse"), record.get(
                f"{split}_mae"
            )
    return out


def table_stage_a(rows):
    lines = [
        "### 表 2：Stage A validation（seed 2021）",
        "",
        "| Setting | direct MSE/MAE | phase-only MSE/MAE | V1 MSE/MAE | V2 MSE/MAE | "
        "V1 retention | V2 retention | V2−V1 | QC |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for dataset, horizon in SETTINGS:
        cells = {}
        for arm in ARMS:
            record = rows.get((dataset, horizon, 2021, arm))
            cells[arm] = record
        def pair(arm):
            record = cells[arm]
            if record is None:
                return "N/A"
            return f"{fmt(record['val_mse'])} / {fmt(record['val_mae'])}"
        direct, phase_only = cells["direct_nlinear"], cells["phase_only"]
        v1, v2 = cells["keep_direction_1"], cells["keep_direction_1_2"]
        r1 = retention(
            v1["val_mse"] if v1 else None,
            phase_only["val_mse"] if phase_only else None,
            direct["val_mse"] if direct else None,
        )
        r2 = retention(
            v2["val_mse"] if v2 else None,
            phase_only["val_mse"] if phase_only else None,
            direct["val_mse"] if direct else None,
        )
        delta = (
            (v2["val_mse"] - v1["val_mse"]) if (v1 and v2) else None
        )
        qc = []
        if v1 is None or v2 is None:
            qc.append("missing arm")
        if direct is None or phase_only is None:
            qc.append("missing control")
        if direct and phase_only and direct["val_mse"] <= phase_only["val_mse"]:
            qc.append("direct does not beat phase-only -> retention N/A")
        lines.append(
            f"| {dataset}-{horizon} | {pair('direct_nlinear')} | {pair('phase_only')} | "
            f"{pair('keep_direction_1')} | {pair('keep_direction_1_2')} | "
            f"{fmt(r1, 1)}% | {fmt(r2, 1)}% | {fmt(delta, 6)} | "
            f"{'; '.join(qc) if qc else 'OK'} |"
        )
    lines.append("")
    return "\n".join(lines)


def aggregate_test(rows):
    """Per setting: 3-seed mean/std test MSE/MAE for each arm."""
    aggregated = {}
    for dataset, horizon in SETTINGS:
        aggregated[(dataset, horizon)] = {}
        for arm in ARMS:
            mses, maes, gates, params, epochs, elapsed, peak = [], [], [], [], [], [], []
            for seed in SEEDS:
                record = rows.get((dataset, horizon, seed, arm))
                if record is None:
                    continue
                if record["test_mse"] is not None:
                    mses.append(record["test_mse"])
                    maes.append(record["test_mae"])
                epochs.append(record["epochs_completed"])
                if record["elapsed_sec"] not in ("", None):
                    elapsed.append(float(record["elapsed_sec"]))
                if record["peak_memory_bytes"] not in ("", None):
                    peak.append(float(record["peak_memory_bytes"]))
                if record["parameter_count"] not in ("", None):
                    params.append(int(record["parameter_count"]))
            aggregated[(dataset, horizon)][arm] = {
                "mse_mean": mean_std(mses)[0],
                "mse_std": mean_std(mses)[1],
                "mae_mean": mean_std(maes)[0],
                "mae_std": mean_std(maes)[1],
                "n_seeds": len(mses),
                "epochs": epochs,
                "elapsed": elapsed,
                "peak": peak,
                "params": params,
            }
    return aggregated


def median(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.median(clean)


def table_final_test(aggregated):
    lines = [
        "### 表 3：三 seed 正式 test 结果",
        "",
        "| Setting | Model | MSE mean±std | MAE mean±std | ΔMSE vs direct | ΔMAE vs direct | "
        "MSE retention | MAE retention |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset, horizon in SETTINGS:
        per_arm = aggregated[(dataset, horizon)]
        direct = per_arm.get("direct_nlinear", {})
        phase_only = per_arm.get("phase_only", {})
        for arm in ARMS:
            stats = per_arm.get(arm, {})
            if stats.get("n_seeds", 0) == 0:
                lines.append(
                    f"| {dataset}-{horizon} | {ARM_LABEL[arm]} | TBD | TBD | TBD | TBD | TBD | TBD |"
                )
                continue
            mse = (
                f"{fmt(stats['mse_mean'])} ± {fmt(stats['mse_std']) if stats['mse_std'] is not None else '—'}"
            )
            mae = (
                f"{fmt(stats['mae_mean'])} ± {fmt(stats['mae_std']) if stats['mae_std'] is not None else '—'}"
            )
            dmse = metric_pct(stats["mse_mean"], direct.get("mse_mean"))
            dmae = metric_pct(stats["mae_mean"], direct.get("mae_mean"))
            rmse = retention(
                stats["mse_mean"], phase_only.get("mse_mean"), direct.get("mse_mean")
            )
            rmae = retention(
                stats["mae_mean"], phase_only.get("mae_mean"), direct.get("mae_mean")
            )
            lines.append(
                f"| {dataset}-{horizon} | {ARM_LABEL[arm]} | {mse} | {mae} | "
                f"{fmt(dmse, 2)}% | {fmt(dmae, 2)}% | {fmt(rmse, 1)}% | {fmt(rmae, 1)}% |"
            )
    lines.append("")
    return "\n".join(lines)


def table_direction2(rows, aggregated, projector_index):
    lines = [
        "### 表 4：方向 2 增量",
        "",
        "| Setting | λ2 share | V1→V2 ΔMSE | V1→V2 ΔMAE | MSE gap recovery | MAE gap recovery | "
        "三 seed 方向一致 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, horizon in SETTINGS:
        per_arm = aggregated[(dataset, horizon)]
        v1 = per_arm.get("keep_direction_1", {})
        v2 = per_arm.get("keep_direction_1_2", {})
        direct = per_arm.get("direct_nlinear", {})
        lambda2 = projector_index.get(setting_key(dataset, horizon), {}).get(
            "lambda2_share"
        )
        dmse = None
        dmae = None
        if v1.get("mse_mean") is not None and v2.get("mse_mean") is not None:
            dmse = v1["mse_mean"] - v2["mse_mean"]
            dmae = v1["mae_mean"] - v2["mae_mean"]
        rec_mse = None
        rec_mae = None
        if (
            v1.get("mse_mean") is not None
            and v2.get("mse_mean") is not None
            and direct.get("mse_mean") is not None
            and v1["mse_mean"] > direct["mse_mean"]
        ):
            rec_mse = (v1["mse_mean"] - v2["mse_mean"]) / (
                v1["mse_mean"] - direct["mse_mean"]
            ) * 100.0
        if (
            v1.get("mae_mean") is not None
            and v2.get("mae_mean") is not None
            and direct.get("mae_mean") is not None
            and v1["mae_mean"] > direct["mae_mean"]
        ):
            rec_mae = (v1["mae_mean"] - v2["mae_mean"]) / (
                v1["mae_mean"] - direct["mae_mean"]
            ) * 100.0
        agree = direction_agreement(rows, dataset, horizon)
        lines.append(
            f"| {dataset}-{horizon} | {fmt(lambda2, 4)} | {fmt(dmse, 6)} | "
            f"{fmt(dmae, 6)} | {fmt(rec_mse, 1)}% | {fmt(rec_mae, 1)}% | {agree} |"
        )
    lines.append("")
    return "\n".join(lines)


def direction_agreement(rows, dataset, horizon):
    """How many of 3 seeds have V2 better than V1 on BOTH metrics."""
    wins = 0
    for seed in SEEDS:
        v1 = rows.get((dataset, horizon, seed, "keep_direction_1"))
        v2 = rows.get((dataset, horizon, seed, "keep_direction_1_2"))
        if not v1 or not v2:
            continue
        if v1["test_mse"] is None or v2["test_mse"] is None:
            continue
        if v2["test_mse"] < v1["test_mse"] and v2["test_mae"] < v1["test_mae"]:
            wins += 1
    return f"{wins}/3"


def table_decision(rows, aggregated):
    macro_dmse, macro_dmae = [], []
    win_count = 0
    ret_mses, ret_maes = [], []
    worst_mse, worst_mae = 0.0, 0.0
    worst_mse_cells, worst_mae_cells = [], []
    for dataset, horizon in SETTINGS:
        per_arm = aggregated[(dataset, horizon)]
        v2 = per_arm.get("keep_direction_1_2", {})
        v1 = per_arm.get("keep_direction_1", {})
        direct = per_arm.get("direct_nlinear", {})
        phase_only = per_arm.get("phase_only", {})
        if v2.get("mse_mean") is None:
            continue
        dmse = metric_pct(v2["mse_mean"], direct.get("mse_mean"))
        dmae = metric_pct(v2["mae_mean"], direct.get("mae_mean"))
        macro_dmse.append(dmse)
        macro_dmae.append(dmae)
        if dmse is not None and dmse > worst_mse:
            worst_mse, worst_mse_cells = dmse, [f"{dataset}-{horizon}"]
        elif dmse == worst_mse and dmse is not None:
            worst_mse_cells.append(f"{dataset}-{horizon}")
        if dmae is not None and dmae > worst_mae:
            worst_mae, worst_mae_cells = dmae, [f"{dataset}-{horizon}"]
        elif dmae == worst_mae and dmae is not None:
            worst_mae_cells.append(f"{dataset}-{horizon}")
        rmse = retention(
            v2["mse_mean"], phase_only.get("mse_mean"), direct.get("mse_mean")
        )
        rmae = retention(
            v2["mae_mean"], phase_only.get("mae_mean"), direct.get("mae_mean")
        )
        if rmse is not None:
            ret_mses.append(rmse)
        if rmae is not None:
            ret_maes.append(rmae)
        if (
            v1.get("mse_mean") is not None
            and v2["mse_mean"] < v1["mse_mean"]
            and v2["mae_mean"] < v1["mae_mean"]
        ):
            win_count += 1

    macro_mse = statistics.fmean(macro_dmse) if macro_dmse else None
    macro_mae = statistics.fmean(macro_dmae) if macro_dmae else None
    med_mse = median(ret_mses)
    med_mae = median(ret_maes)

    checks = [
        (
            "V2 宏平均 MSE vs direct",
            f"≤ +{MACRO_TOLERANCE_PCT}%",
            f"{fmt(macro_mse, 2)}%",
            macro_mse is not None and macro_mse <= MACRO_TOLERANCE_PCT,
        ),
        (
            "V2 宏平均 MAE vs direct",
            f"≤ +{MACRO_TOLERANCE_PCT}%",
            f"{fmt(macro_mae, 2)}%",
            macro_mae is not None and macro_mae <= MACRO_TOLERANCE_PCT,
        ),
        (
            "V2 MSE retention 中位数",
            f"≥ {RETENTION_MEDIAN_MIN_PCT:.0f}%",
            f"{fmt(med_mse, 1)}%",
            med_mse is not None and med_mse >= RETENTION_MEDIAN_MIN_PCT,
        ),
        (
            "V2 MAE retention 中位数",
            f"≥ {RETENTION_MEDIAN_MIN_PCT:.0f}%",
            f"{fmt(med_mae, 1)}%",
            med_mae is not None and med_mae >= RETENTION_MEDIAN_MIN_PCT,
        ),
        (
            "V2 双指标优于 V1",
            f"≥ {WIN_COUNT_MIN}/6 settings",
            f"{win_count}/6",
            win_count >= WIN_COUNT_MIN,
        ),
        (
            "V2 最坏单格退化",
            f"≤ {WORST_CELL_TOLERANCE_PCT}%",
            f"MSE {fmt(worst_mse, 2)}% ({','.join(worst_mse_cells)}), "
            f"MAE {fmt(worst_mae, 2)}% ({','.join(worst_mae_cells)})",
            worst_mse <= WORST_CELL_TOLERANCE_PCT
            and worst_mae <= WORST_CELL_TOLERANCE_PCT,
        ),
    ]

    strong = all(check[3] for check in checks)
    # Plan section 9: "not supported" triggers.
    badly_degraded = 0
    for dataset, horizon in SETTINGS:
        per_arm = aggregated[(dataset, horizon)]
        v2 = per_arm.get("keep_direction_1_2", {})
        direct = per_arm.get("direct_nlinear", {})
        if v2.get("mse_mean") is None:
            continue
        dmse = metric_pct(v2["mse_mean"], direct.get("mse_mean"))
        dmae = metric_pct(v2["mae_mean"], direct.get("mae_mean"))
        if (dmse is not None and dmse > WORST_CELL_TOLERANCE_PCT) or (
            dmae is not None and dmae > WORST_CELL_TOLERANCE_PCT
        ):
            badly_degraded += 1
    not_supported = (
        win_count <= NOT_SUPPORTED_WIN_COUNT_MAX
        or (med_mse is not None and med_mse < PARTIAL_RETENTION_MEDIAN_PCT)
        or (med_mae is not None and med_mae < PARTIAL_RETENTION_MEDIAN_PCT)
        or badly_degraded >= NOT_SUPPORTED_WORST_CELLS
    )
    verdict = (
        "强支持" if strong else ("不支持" if not_supported else "部分支持（方向 2 有增量价值，"
        "但前两个方向不足以完整替代原输入）")
    )

    lines = [
        "### 表 5：最终决策（预注册门槛）",
        "",
        "| 判定项 | 预注册门槛 | 实测 | 通过 |",
        "|---|---|---|---|",
    ]
    for name, threshold, measured, passed in checks:
        lines.append(f"| {name} | {threshold} | {measured} | {'PASS' if passed else 'FAIL'} |")
    lines.append(f"| 最终结论 | 强支持 / 部分支持 / 不支持 | {verdict} | — |")
    lines.append("")
    details = {
        "macro_delta_mse_pct": macro_mse,
        "macro_delta_mae_pct": macro_mae,
        "median_mse_retention_pct": med_mse,
        "median_mae_retention_pct": med_mae,
        "v2_beats_v1_settings": win_count,
        "worst_cell_delta_mse_pct": worst_mse,
        "worst_cell_delta_mae_pct": worst_mae,
        "cells_degraded_over_2pct": badly_degraded,
        "verdict": verdict,
        "strong_support": strong,
        "not_supported": not_supported,
    }
    return "\n".join(lines), details


def write_results_csv(rows, aggregated, path: Path):
    fields = [
        "dataset", "horizon", "setting", "seed", "arm", "reused", "gate_init",
        "learning_rate", "val_mse", "val_mae", "test_mse", "test_mae",
        "parameter_count", "trainable_parameter_count", "epochs_completed",
        "elapsed_sec", "peak_memory_bytes", "run_id", "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows):
            record = dict(rows[key])
            record["setting"] = setting_key(record["dataset"], record["horizon"])
            writer.writerow({k: record.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--projector-dir", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    projector_dir = Path(args.projector_dir) if args.projector_dir else root / "projectors"
    output = Path(args.output) if args.output else root
    output.mkdir(parents=True, exist_ok=True)

    rows = load_runs(root)
    projector_index = {}
    index_path = projector_dir / "projectors.json"
    if index_path.exists():
        projector_index = json.loads(index_path.read_text()).get("projectors", {})

    aggregated = aggregate_test(rows)
    write_results_csv(rows, aggregated, output / "results.csv")

    decision_md, decision = table_decision(rows, aggregated)
    sections = [
        "# PhaseFormer 前两预测方向数据保留实验结果",
        "",
        f"运行根目录：`{root.relative_to(ROOT)}`",
        "",
        f"覆盖的 (setting, seed, arm) 格子：{len(rows)}",
        "",
        table_stage_a(rows),
        table_final_test(aggregated),
        table_direction2(rows, aggregated, projector_index),
        decision_md,
    ]
    (output / "report_tables.md").write_text("\n".join(sections))
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"cells": len(rows), **decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
