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


def infer_arm(record, hyper):
    """Map a run onto one of the four arms, including pre-arm-field controls."""
    arm = hyper.get("weak_residual_projection_arm")
    if arm:
        return arm
    if record.get("mechanism") == "no_residual":
        return "phase_only"
    if hyper.get("weak_period_residual_head_type") == "shared":
        return "direct_nlinear"
    return None


def row_from_metrics(path: Path):
    """Read one run's metrics.csv + config.json into a flat record."""
    with path.open(newline="") as handle:
        record = next(csv.DictReader(handle), None)
    if record is None:
        return None
    config_path = path.with_name("config.json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    hyper = config.get("hyperparams", {})
    arm = infer_arm(record, hyper)
    if arm is None:
        return None
    try:
        run_dir = str(path.parent.relative_to(ROOT))
    except ValueError:
        run_dir = str(path.parent)
    # The test numbers of freshly trained cells live in ``test_read.json``:
    # test is read exactly once, after every checkpoint is frozen, by a
    # separate step that never rewrites metrics.csv.  Reused controls keep the
    # test numbers their own audited run recorded.
    gate_value = None
    nlinear_mse = None
    nlinear_mae = None
    val_relative_difference = None
    test_read_status = ""
    test_read_path = path.with_name("test_read.json")
    if test_read_path.exists():
        payload = json.loads(test_read_path.read_text())
        test_read_status = payload.get("status", "")
        gate_value = payload.get("gate_value")
        nlinear_mse = payload.get("nlinear_mse")
        nlinear_mae = payload.get("nlinear_mae")
        val_relative_difference = payload.get("val_relative_difference")
        if payload.get("test_mse") is not None:
            record = {**record,
                      "test_mse": payload["test_mse"],
                      "test_mae": payload["test_mae"]}
    return {
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
        "run_dir": run_dir,
        "reused": not bool(hyper.get("weak_residual_projection_arm")),
        "gate_init": hyper.get("weak_period_residual_gate_init", ""),
        "learning_rate": hyper.get("learning_rate", ""),
        "gate_value": gate_value,
        "nlinear_mse": nlinear_mse,
        "nlinear_mae": nlinear_mae,
        "test_read_status": test_read_status or ("reused" if not hyper.get(
            "weak_residual_projection_arm") else ""),
        "val_relative_difference": val_relative_difference,
    }


def load_runs(root: Path, reuse_audit: Path | None = None):
    """Collect one row per (setting, seed, arm).

    New runs come from ``root/runs``.  The reused ``direct_nlinear`` control is
    read from the run directories the reuse audit accepted, in place -- the
    artifacts are never copied into this experiment's own directory, and the
    ``reused`` flag keeps them distinguishable from freshly trained cells.
    """
    rows = {}
    paths: list[Path] = list(root.glob("runs/*/metrics.csv"))
    if reuse_audit is not None and reuse_audit.exists():
        payload = json.loads(reuse_audit.read_text())
        for entry in payload.get("runs", []):
            metrics = ROOT / entry["run_dir"] / "metrics.csv"
            if metrics.exists():
                paths.append(metrics)
    for path in sorted(set(paths)):
        row = row_from_metrics(path)
        if row is None:
            continue
        key = (row["dataset"], row["horizon"], row["seed"], row["arm"])
        existing = rows.get(key)
        if existing is not None and existing["reused"] and row["reused"]:
            # Duplicate reused candidate; keep the first audited one.
            continue
        rows[key] = row
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
        "|---|---|---|---|---|---:|---:|---:|---|",
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
        if direct and phase_only and direct["val_mse"] < phase_only["val_mse"]:
            qc.append("validation: retention defined")
        if direct and phase_only and direct["val_mse"] >= phase_only["val_mse"]:
            qc.append(
                "validation: direct does not beat phase-only -> retention N/A"
            )
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
        "|---|---|---:|---:|---:|---:|---:|---:|",
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


def table_diagnostics(rows, aggregated, stage0, projector_index):
    """Plan section 8.4: gate, NLinear-branch, projection and cost diagnostics."""
    lines = [
        "### 表 6：诊断指标（计划 §8.4）",
        "",
        "| Setting | Arm | learned gate mean±std | NLinear 支路 test MSE | "
        "NLinear 支路 test MAE | 可见中心化输入方差占比 | b1^Tz std | b2^Tz std | "
        "最佳 epoch | 训练时间 (s) | 参数量 | 峰值显存 (MB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for dataset, horizon in SETTINGS:
        entry = projector_index.get(f"{dataset}-{horizon}", {})
        audit = stage0.get(f"{dataset}-{horizon}", {})
        per_arm = aggregated[(dataset, horizon)]
        for arm in ARMS:
            stats = per_arm.get(arm, {})
            if stats.get("n_seeds", 0) == 0:
                continue
            gates, nlin_mse, nlin_mae = [], [], []
            for seed in SEEDS:
                record = rows.get((dataset, horizon, seed, arm))
                if not record:
                    continue
                if record.get("gate_value") is not None:
                    gates.append(record["gate_value"])
                if record.get("nlinear_mse") is not None:
                    nlin_mse.append(record["nlinear_mse"])
                    nlin_mae.append(record["nlinear_mae"])
            gate_mean, gate_std = mean_std(gates)
            if arm == "phase_only":
                gate_text = "N/A（无 NLinear 门控）"
            elif not gates:
                gate_text = "N/A（复用 run 未记录）"
            else:
                gate_text = (
                    f"{fmt(gate_mean, 4)} ± "
                    f"{fmt(gate_std, 4) if gate_std is not None else '—'}"
                )
            nlin_text_mse = (
                fmt(mean_std(nlin_mse)[0], 6) if nlin_mse
                else ("N/A（无 NLinear 支路）" if arm == "phase_only"
                      else "N/A（复用 run 未记录）")
            )
            nlin_text_mae = (
                fmt(mean_std(nlin_mae)[0], 6) if nlin_mae
                else ("N/A（无 NLinear 支路）" if arm == "phase_only"
                      else "N/A（复用 run 未记录）")
            )
            if arm == "keep_direction_1":
                var_share = entry.get("used_var_share_v1")
            elif arm == "keep_direction_1_2":
                var_share = entry.get("used_var_share_v12")
            elif arm == "direct_nlinear":
                var_share = 1.0
            else:
                var_share = 0.0
            b1_std = audit.get("feature_b1_z_std")
            b2_std = audit.get("feature_b2_z_std")
            b2_std_text = fmt(b2_std, 4)
            if arm == "keep_direction_1":
                b2_std_text = "N/A（V1 不可见）"
            elif arm == "phase_only":
                b2_std_text = "N/A"
            params = stats.get("params") or []
            peaks = stats.get("peak") or []
            elapsed = stats.get("elapsed") or []
            epoch_mean, epoch_std = mean_std(
                [float(e) for e in (stats.get("epochs") or []) if e not in ("", None)]
            )
            epoch_text = (
                f"{fmt(epoch_mean, 1)} ± {fmt(epoch_std, 1)}"
                if epoch_std is not None else fmt(epoch_mean, 1)
            )
            lines.append(
                f"| {dataset}-{horizon} | {ARM_LABEL[arm]} | {gate_text} | "
                f"{nlin_text_mse} | {nlin_text_mae} | "
                f"{fmt(var_share, 4) if var_share is not None else 'N/A'} | "
                f"{fmt(b1_std, 4) if b1_std is not None else 'N/A'} | "
                f"{b2_std_text} | "
                f"{epoch_text} | {fmt(mean_std(elapsed)[0], 0)} | "
                f"{params[0] if params else ''} | "
                f"{fmt(mean_std(peaks)[0] / 1e6, 1) if peaks else 'N/A'} |"
            )
    lines.append("")
    lines.append(
        "`可见中心化输入方差占比` 取自 Stage 0 的解析值；`b1^Tz / b2^Tz std` 取自 "
        "Stage 0 训练 split 上的样本标准差。三 seed 的 gate 值在此处为逐 seed 均值。"
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
        lambda2 = projector_index.get(f"{dataset}-{horizon}", {}).get(
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
        # Plan 8.3 defines gap recovery only when V1 is worse than direct; when
        # V1 already matches or beats direct there is no gap to recover, and a
        # signed percentage of a near-zero denominator would be meaningless.
        rec_mse_text = (
            f"{fmt(rec_mse, 1)}%" if rec_mse is not None else
            ("N/A (V1 不差于 direct)" if v1 and v1.get("mse_mean") is not None
             else "N/A")
        )
        rec_mae_text = (
            f"{fmt(rec_mae, 1)}%" if rec_mae is not None else
            ("N/A (V1 不差于 direct)" if v1 and v1.get("mae_mean") is not None
             else "N/A")
        )
        lines.append(
            f"| {dataset}-{horizon} | {fmt(lambda2, 4)} | {fmt(dmse, 6)} | "
            f"{fmt(dmae, 6)} | {rec_mse_text} | {rec_mae_text} | {agree} |"
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
        "nlinear_mse", "nlinear_mae", "gate_value", "test_read_status",
        "val_relative_difference", "parameter_count",
        "trainable_parameter_count", "epochs_completed", "elapsed_sec",
        "peak_memory_bytes", "run_id", "run_dir",
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
    parser.add_argument(
        "--reuse-audit",
        default="",
        help="audit JSON listing the reused direct_nlinear control runs",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    projector_dir = Path(args.projector_dir) if args.projector_dir else root / "projectors"
    output = Path(args.output) if args.output else root
    output.mkdir(parents=True, exist_ok=True)

    reuse_audit = Path(args.reuse_audit) if args.reuse_audit else root / "reuse_audit.json"
    if not reuse_audit.is_absolute():
        reuse_audit = ROOT / reuse_audit
    rows = load_runs(root, reuse_audit)
    projector_index = {}
    index_path = projector_dir / "projectors.json"
    if index_path.exists():
        projector_index = json.loads(index_path.read_text()).get("projectors", {})
    # Stage 0 audit, keyed "Dataset-Horizon" for the diagnostics table.
    stage0 = {}
    stage0_path = projector_dir / "stage0_audit.json"
    if stage0_path.exists():
        for entry in json.loads(stage0_path.read_text()):
            stage0[entry["setting"]] = entry
    stage0_md = ""
    stage0_md_path = projector_dir / "stage0_audit.md"
    if stage0_md_path.exists():
        stage0_md = stage0_md_path.read_text()

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
        table_diagnostics(rows, aggregated, stage0, projector_index),
    ]
    if stage0_md:
        sections += [
            "## 表 1：投影器审计（Stage 0）",
            "",
            "由 `scripts/compute_top2_direction_projectors.py` 生成，只读取训练 split；",
            "完整逐项检查与方向 2 稳定性标注见 `projectors/stage0_audit.md`。",
            "",
            stage0_md,
        ]
    (output / "report_tables.md").write_text("\n".join(sections))
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"cells": len(rows), **decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
