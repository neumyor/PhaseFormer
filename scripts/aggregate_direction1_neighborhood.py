#!/usr/bin/env python3
"""Aggregate the direction-1 neighborhood experiment into the plan's tables.

Reads the Stage 0 projector audit plus every run's ``metrics.csv`` /
``config.json`` under the experiment root and renders the six tables of
``docs/PhaseFormer_direction1_neighborhood_experiment_plan.md``:

    table1_stage0_geometry.md     bootstrap geometry per setting
    table2_stage0_properties.md   neighborhood analytic properties per (setting, k)
    table3_sweep_test.md          Stage T seed-2021 test sweep, all six arms
    table4_width_selection.md     dataset-level width selection record
    table5_multiseed.md           Stage S three-seed stability
    table6_verdict.md             plan section 7 interpretation

Plus ``results.csv`` (one row per setting/seed/arm) and ``aggregate.json``
(the machine-readable payload the tables are rendered from).

This is an explicitly disclosed test-set-selection experiment.  Every table
keeps all sampled widths; nothing is filtered down to the selected width.

Usage::

    python scripts/aggregate_direction1_neighborhood.py \
        --root research_runs/direction1_neighborhood_v1
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

from scripts.aggregate_top2_direction_retention import row_from_metrics  # noqa: E402
from scripts.run_direction1_neighborhood_matrix import (  # noqa: E402
    ALL_SETTINGS,
    CONFIRM_SEEDS,
    SWEEP_SEEDS,
    SWEEP_WIDTHS,
    cone_arm,
)

DIRECT = "direct_nlinear"
RRR2 = "rrr_direction_1_2"
ARM_ORDER = (DIRECT, RRR2, *(cone_arm(width) for width in SWEEP_WIDTHS))
ARM_LABEL = {
    DIRECT: "direct",
    RRR2: "RRR-2",
    **{cone_arm(width): f"Cone-{width}" for width in SWEEP_WIDTHS},
}
DATASET_ORDER = ["ETTh2", "ETTm2", "Weather", "Electricity"]

# Plan section 7 thresholds.
MIN_DATASETS_WITH_WIDER_K = 3
MIN_SETTINGS_BEATING_CONE1 = 4
MIN_SETTINGS_VS_RRR2 = 4
MIN_SEEDS_PER_SETTING = 2


def setting_key(dataset, horizon):
    return f"{dataset}-{horizon}"


def metric_pct(candidate, reference):
    if candidate is None or reference in (None, 0):
        return None
    return (candidate / reference - 1.0) * 100.0


def mean_std(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], None
    return statistics.fmean(clean), statistics.stdev(clean)


def fmt(value, digits=4, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def signed(value, digits=3, suffix="%"):
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return f"{value:+.{digits}f}{suffix}"


def load_runs(root: Path) -> dict:
    rows = {}
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        row = row_from_metrics(path)
        if row is None:
            continue
        key = (row["dataset"], row["horizon"], row["seed"], row["arm"])
        rows[key] = row
    return rows


def get(rows, dataset, horizon, seed, arm):
    return rows.get((dataset, horizon, seed, arm), {})


def test_pair(rows, dataset, horizon, seed, arm):
    row = get(rows, dataset, horizon, seed, arm)
    return row.get("test_mse"), row.get("test_mae")


# --------------------------------------------------------------------------
# Table 1 / 2: Stage 0
# --------------------------------------------------------------------------


def load_stage0(root: Path) -> list[dict]:
    path = root / "projectors" / "stage0_audit.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def legacy_cone1_diagnostic(root: Path, legacy_dir: Path) -> list[dict]:
    """Principal-angle drift between the previous plan's Q1 and this round's Qcone1.

    The plan states this comparison is a diagnostic only and never a blocking
    condition, so it is recorded and never enforced.
    """
    import numpy as np

    records = []
    for dataset, horizon in ALL_SETTINGS:
        legacy = legacy_dir / f"{dataset}_{horizon}_Q1.npy"
        current = root / "projectors" / f"{dataset}_{horizon}_Qcone1.npy"
        if not legacy.is_file() or not current.is_file():
            records.append(
                {
                    "setting": setting_key(dataset, horizon),
                    "available": False,
                    "reason": "missing legacy Q1" if not legacy.is_file()
                    else "missing Qcone1",
                }
            )
            continue
        left = np.load(legacy)
        right = np.load(current)
        singular = np.linalg.svd(left.T @ right, compute_uv=False)
        angle = float(np.degrees(np.arccos(np.clip(singular[0], -1.0, 1.0))))
        records.append(
            {
                "setting": setting_key(dataset, horizon),
                "available": True,
                "principal_angle_deg": angle,
                "legacy_file": legacy.name,
                "current_file": current.name,
            }
        )
    return records


def render_legacy_diagnostic(records: list[dict]) -> str:
    lines = [
        "### 诊断：上一轮 V1 的 Q1 与本轮 Cone-1 的 q1 差异（非阻断项）",
        "",
        "| Setting | principal angle |",
        "|---|---:|",
    ]
    for record in records:
        if not record.get("available"):
            lines.append(
                f"| {record['setting']} | N/A（{record.get('reason', 'unavailable')}） |"
            )
            continue
        lines.append(
            f"| {record['setting']} | {record['principal_angle_deg']:.2f}° |"
        )
    return "\n".join(lines) + "\n"


def render_table1(stage0: list[dict]) -> str:
    by_setting = {}
    for row in stage0:
        by_setting.setdefault(row["setting"], {})[row["width"]] = row
    lines = [
        "### 表 1：Stage 0 bootstrap 几何",
        "",
        "| Setting | blocks | replicates | angle median | angle p90 | "
        "tangent effective rank | QC |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, horizon in ALL_SETTINGS:
        setting = setting_key(dataset, horizon)
        row = by_setting.get(setting, {}).get(1)
        if row is None:
            lines.append(f"| {setting} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {setting} | {row['bootstrap_blocks']} | "
            f"{row['bootstrap_replicates']} | "
            f"{row['angle_median_deg']:.2f}° | {row['angle_p90_deg']:.2f}° | "
            f"{row['bootstrap_tangent_effective_rank']} | "
            f"{'PASS' if row['checks']['all_passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def render_table2(stage0: list[dict]) -> str:
    by_setting = {}
    for row in stage0:
        by_setting.setdefault(row["setting"], {})[row["width"]] = row
    lines = [
        "### 表 2：Stage 0 邻域解析性质",
        "",
        "| Setting | k | tangent variation explained | visible variance | "
        "independent predictive capture | overlap with direction 2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, horizon in ALL_SETTINGS:
        setting = setting_key(dataset, horizon)
        for width in SWEEP_WIDTHS:
            row = by_setting.get(setting, {}).get(width)
            if row is None:
                lines.append(f"| {setting} | {width} | N/A | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {setting} | {width} | "
                f"{fmt(100 * row['tangent_deviation_explained'], 1, '%')} | "
                f"{fmt(100 * row['visible_variance_share'], 2, '%')} | "
                f"{fmt(100 * row['independent_predictive_capture'], 1, '%')} | "
                f"{fmt(100 * row['direction2_overlap'], 2, '%')} |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Table 3: Stage T sweep
# --------------------------------------------------------------------------


def render_table3(rows: dict) -> str:
    lines = [
        "### 表 3：Stage T 全量 test sweep（seed 2021）",
        "",
        "| Setting | Arm | test MSE | test MAE | ΔMSE vs direct | ΔMAE vs direct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, horizon in ALL_SETTINGS:
        setting = setting_key(dataset, horizon)
        direct_mse, direct_mae = test_pair(rows, dataset, horizon, 2021, DIRECT)
        for arm in ARM_ORDER:
            mse, mae = test_pair(rows, dataset, horizon, 2021, arm)
            d_mse = None if arm == DIRECT else metric_pct(mse, direct_mse)
            d_mae = None if arm == DIRECT else metric_pct(mae, direct_mae)
            lines.append(
                f"| {setting} | {ARM_LABEL[arm]} | {fmt(mse)} | {fmt(mae)} | "
                f"{signed(d_mse)} | {signed(d_mae)} |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Table 4: Stage T width selection (copied from test_selection.json)
# --------------------------------------------------------------------------


def render_table4(selection: dict) -> str:
    lines = [
        "### 表 4：按数据集宽度选择",
        "",
        "| Dataset | k=1 ΔMSE/ΔMAE | k=2 ΔMSE/ΔMAE | k=4 ΔMSE/ΔMAE | "
        "k=8 ΔMSE/ΔMAE | selected k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    evaluations = {
        evaluation["dataset"]: evaluation
        for evaluation in selection.get("evaluations", [])
    }
    for dataset in DATASET_ORDER:
        evaluation = evaluations.get(dataset)
        if evaluation is None:
            lines.append(f"| {dataset} | N/A | N/A | N/A | N/A | N/A |")
            continue
        candidates = {item["width"]: item for item in evaluation["candidates"]}
        cells = []
        for width in SWEEP_WIDTHS:
            item = candidates.get(width, {})
            cells.append(
                f"{signed(item.get('macro_delta_mse_vs_direct_pct'), 3)} / "
                f"{signed(item.get('macro_delta_mae_vs_direct_pct'), 3)}"
            )
        lines.append(
            f"| {dataset} | " + " | ".join(cells) + " | "
            f"{evaluation.get('selected_width')} |"
        )
    lines.append("")
    lines.append(
        "选择规则：先取数据集宏平均 test ΔMSE 最优，再把与其差距不超过 "
        f"{selection.get('mse_tie_tolerance_percentage_points')} 个百分点的宽度纳入候选，"
        "在候选中取宏平均 test ΔMAE 最小者，仍相同则取更小的 k。"
    )
    lines.append("")
    lines.append(
        "该选择直接读取 seed 2021 的 test 指标，属于 test-set selection。"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Table 5: Stage S three-seed stability
# --------------------------------------------------------------------------


def multiseed_arms(selected_width: int | None) -> tuple[str, ...]:
    if selected_width is None:
        return (DIRECT, RRR2, cone_arm(1))
    arms = [DIRECT, RRR2, cone_arm(1)]
    selected_arm = cone_arm(selected_width)
    if selected_arm not in arms:
        arms.append(selected_arm)
    return tuple(arms)


def build_multiseed(rows: dict, selection: dict) -> dict:
    selected = selection.get("selected_width_by_dataset", {})
    report = {}
    for dataset, horizon in ALL_SETTINGS:
        setting = setting_key(dataset, horizon)
        width = selected.get(dataset)
        arms = multiseed_arms(width)
        direct_by_seed = {
            seed: test_pair(rows, dataset, horizon, seed, DIRECT)[0]
            for seed in CONFIRM_SEEDS
        }
        direct_by_seed[SWEEP_SEEDS[0]] = test_pair(
            rows, dataset, horizon, SWEEP_SEEDS[0], DIRECT
        )[0]
        rrr2_by_seed = {
            seed: test_pair(rows, dataset, horizon, seed, RRR2)[0]
            for seed in CONFIRM_SEEDS
        }
        rrr2_by_seed[SWEEP_SEEDS[0]] = test_pair(
            rows, dataset, horizon, SWEEP_SEEDS[0], RRR2
        )[0]
        entry = {
            "setting": setting,
            "dataset": dataset,
            "horizon": horizon,
            "selected_width": width,
            "arms": {},
        }
        for arm in arms:
            mse_values, mae_values, seeds_present = [], [], []
            per_seed = {}
            for seed in (*SWEEP_SEEDS, *CONFIRM_SEEDS):
                mse, mae = test_pair(rows, dataset, horizon, seed, arm)
                if mse is None:
                    continue
                seeds_present.append(seed)
                mse_values.append(mse)
                mae_values.append(mae)
                per_seed[seed] = {
                    "test_mse": mse,
                    "test_mae": mae,
                    "delta_mse_vs_direct_pct": metric_pct(
                        mse, direct_by_seed.get(seed)
                    ),
                    "delta_mae_vs_direct_pct": None,
                    "delta_mse_vs_rrr2_pct": metric_pct(
                        mse, rrr2_by_seed.get(seed)
                    ),
                }
                _, direct_mae = test_pair(rows, dataset, horizon, seed, DIRECT)
                per_seed[seed]["delta_mae_vs_direct_pct"] = metric_pct(
                    mae, direct_mae
                )
                _, rrr2_mae = test_pair(rows, dataset, horizon, seed, RRR2)
                per_seed[seed]["delta_mae_vs_rrr2_pct"] = metric_pct(
                    mae, rrr2_mae
                )
            mse_mean, mse_std = mean_std(mse_values)
            mae_mean, mae_std = mean_std(mae_values)
            direct_mse_mean, _ = mean_std(list(direct_by_seed.values()))
            direct_mae_values = [
                test_pair(rows, dataset, horizon, seed, DIRECT)[1]
                for seed in seeds_present
            ]
            direct_mae_mean, _ = mean_std(direct_mae_values)
            entry["arms"][arm] = {
                "label": ARM_LABEL[arm],
                "seeds": seeds_present,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "delta_mse_vs_direct_pct": (
                    None
                    if arm == DIRECT
                    else metric_pct(mse_mean, direct_mse_mean)
                ),
                "delta_mae_vs_direct_pct": (
                    None
                    if arm == DIRECT
                    else metric_pct(mae_mean, direct_mae_mean)
                ),
                "mse_wins_vs_direct": (
                    None
                    if arm == DIRECT
                    else sum(
                        1
                        for seed in seeds_present
                        if per_seed[seed]["test_mse"] is not None
                        and direct_by_seed.get(seed) is not None
                        and per_seed[seed]["test_mse"] < direct_by_seed[seed]
                    )
                ),
                "mse_wins_vs_rrr2": (
                    None
                    if arm == RRR2
                    else sum(
                        1
                        for seed in seeds_present
                        if rrr2_by_seed.get(seed) is not None
                        and per_seed[seed]["test_mse"] < rrr2_by_seed[seed]
                    )
                ),
                "seed_count": len(seeds_present),
                "per_seed": per_seed,
            }
        report[setting] = entry
    return report


def render_table5(multiseed: dict) -> str:
    lines = [
        "### 表 5：三 seed 稳定性结果",
        "",
        "| Setting | Arm | MSE mean±std | MAE mean±std | ΔMSE vs direct | "
        "ΔMAE vs direct | MSE wins vs direct | MSE wins vs RRR-2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, horizon in ALL_SETTINGS:
        entry = multiseed.get(setting_key(dataset, horizon))
        if entry is None:
            continue
        for arm, payload in entry["arms"].items():
            wins_direct = payload["mse_wins_vs_direct"]
            wins_rrr2 = payload["mse_wins_vs_rrr2"]
            n = payload["seed_count"]
            cells = [
                f"{payload['label']}",
                f"{fmt(payload['mse_mean'])} ± {fmt(payload['mse_std'])}",
                f"{fmt(payload['mae_mean'])} ± {fmt(payload['mae_std'])}",
                signed(payload["delta_mse_vs_direct_pct"]),
                signed(payload["delta_mae_vs_direct_pct"]),
                "—" if wins_direct is None else f"{wins_direct}/{n}",
                "—" if wins_rrr2 is None else f"{wins_rrr2}/{n}",
            ]
            lines.append(f"| {setting_key(dataset, horizon)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Table 6: verdict
# --------------------------------------------------------------------------


def build_verdict(multiseed: dict, selection: dict) -> dict:
    selected = selection.get("selected_width_by_dataset", {})
    datasets_wider = sorted(
        dataset for dataset, width in selected.items() if width and width > 1
    )
    per_setting = {}
    mse_better, mae_better, vs_rrr2_better_or_equal, stable = [], [], [], []
    vs_rrr2_mae_better_or_equal = []
    for dataset, horizon in ALL_SETTINGS:
        setting = setting_key(dataset, horizon)
        entry = multiseed.get(setting)
        if entry is None or entry["selected_width"] is None:
            continue
        cone1 = entry["arms"].get(cone_arm(1))
        selected_arm = cone_arm(entry["selected_width"])
        cone_k = entry["arms"].get(selected_arm)
        rrr2 = entry["arms"].get(RRR2)
        if cone1 is None or cone_k is None:
            continue
        record = {
            "setting": setting,
            "selected_width": entry["selected_width"],
            "selected_arm": ARM_LABEL[selected_arm],
            "cone1_mse_mean": cone1["mse_mean"],
            "cone_k_mse_mean": cone_k["mse_mean"],
            "cone1_mae_mean": cone1["mae_mean"],
            "cone_k_mae_mean": cone_k["mae_mean"],
            "delta_mse_vs_cone1_pct": metric_pct(
                cone_k["mse_mean"], cone1["mse_mean"]
            ),
            "delta_mae_vs_cone1_pct": metric_pct(
                cone_k["mae_mean"], cone1["mae_mean"]
            ),
            "delta_mse_vs_rrr2_pct": metric_pct(
                cone_k["mse_mean"], (rrr2 or {}).get("mse_mean")
            ),
            "delta_mae_vs_rrr2_pct": metric_pct(
                cone_k["mae_mean"], (rrr2 or {}).get("mae_mean")
            ),
        }
        shared_seeds = sorted(
            set(cone1["seeds"]) & set(cone_k["seeds"])
        )
        per_seed_wins = sum(
            1
            for seed in shared_seeds
            if cone_k["per_seed"][seed]["test_mse"]
            < cone1["per_seed"][seed]["test_mse"]
        )
        record["seeds_compared"] = len(shared_seeds)
        record["cone_k_mse_wins_vs_cone1_per_seed"] = per_seed_wins
        record["cross_seed_stable"] = per_seed_wins >= min(
            MIN_SEEDS_PER_SETTING, len(shared_seeds)
        )
        if record["delta_mse_vs_cone1_pct"] is not None and (
            record["delta_mse_vs_cone1_pct"] < 0
        ):
            mse_better.append(setting)
        if record["delta_mae_vs_cone1_pct"] is not None and (
            record["delta_mae_vs_cone1_pct"] < 0
        ):
            mae_better.append(setting)
        if record["delta_mse_vs_rrr2_pct"] is not None and (
            record["delta_mse_vs_rrr2_pct"] <= 0
        ):
            vs_rrr2_better_or_equal.append(setting)
        if record["delta_mae_vs_rrr2_pct"] is not None and (
            record["delta_mae_vs_rrr2_pct"] <= 0
        ):
            vs_rrr2_mae_better_or_equal.append(setting)
        if record["cross_seed_stable"] and (
            record["delta_mse_vs_cone1_pct"] is not None
            and record["delta_mse_vs_cone1_pct"] < 0
        ):
            stable.append(setting)
        per_setting[setting] = record

    dataset_macro = {}
    for dataset in DATASET_ORDER:
        widths = [
            record for record in per_setting.values()
            if record["setting"].startswith(f"{dataset}-")
        ]
        if not widths:
            continue
        mse_values = [
            record["delta_mse_vs_cone1_pct"]
            for record in widths
            if record["delta_mse_vs_cone1_pct"] is not None
        ]
        mae_values = [
            record["delta_mae_vs_cone1_pct"]
            for record in widths
            if record["delta_mae_vs_cone1_pct"] is not None
        ]
        if not mse_values or not mae_values:
            continue
        dataset_macro[dataset] = {
            "selected_width": selected.get(dataset),
            "macro_delta_mse_vs_cone1_pct": statistics.fmean(mse_values),
            "macro_delta_mae_vs_cone1_pct": statistics.fmean(mae_values),
        }
    macro_better = [
        dataset
        for dataset, payload in dataset_macro.items()
        if payload["macro_delta_mse_vs_cone1_pct"] < 0
        and payload["macro_delta_mae_vs_cone1_pct"] < 0
    ]

    criteria = {
        "datasets_with_wider_k": len(datasets_wider),
        "datasets_selecting_wider_k": datasets_wider,
        "criterion_1_met": len(datasets_wider) >= MIN_DATASETS_WITH_WIDER_K,
        "datasets_macro_better_than_cone1": macro_better,
        "criterion_2_met": len(macro_better) == len(dataset_macro)
        and bool(dataset_macro),
        "settings_cone_k_mse_better_than_cone1": mse_better,
        "criterion_3_met": len(mse_better) >= MIN_SETTINGS_BEATING_CONE1,
        "settings_cone_k_mae_better_than_cone1": mae_better,
        "settings_cone_k_better_or_equal_rrr2": vs_rrr2_better_or_equal,
        "settings_cone_k_better_or_equal_rrr2_mae": vs_rrr2_mae_better_or_equal,
        "criterion_4_met": len(vs_rrr2_better_or_equal) >= MIN_SETTINGS_VS_RRR2,
        "settings_with_cross_seed_stable_gain": stable,
        "settings_with_mean_gain_not_from_a_single_seed": [
            setting for setting in mse_better if setting in stable
        ],
        "gain_is_single_seed_artifact": [
            setting for setting in mse_better if setting not in stable
        ],
        "criterion_5_met": bool(mse_better)
        and all(setting in stable for setting in mse_better),
    }
    met = sum(
        1
        for key in ("criterion_1_met", "criterion_2_met", "criterion_3_met",
                    "criterion_4_met", "criterion_5_met")
        if criteria[key]
    )
    if met == 5:
        verdict = "支持"
    elif met >= 3:
        verdict = "部分支持"
    else:
        verdict = "不支持"
    criteria["criteria_met"] = met
    criteria["verdict"] = verdict
    return {
        "per_setting": per_setting,
        "dataset_macro": dataset_macro,
        "criteria": criteria,
    }


def render_table6(verdict: dict) -> str:
    criteria = verdict["criteria"]
    lines = [
        "### 表 6：最终解释",
        "",
        "| 判定项 | 实测 |",
        "|---|---|",
        f"| 选择 k>1 的数据集数 | "
        f"{criteria['datasets_with_wider_k']}/4"
        f"（{', '.join(criteria['datasets_selecting_wider_k']) or '无'}） |",
        f"| 选择后 Cone-k 数据集宏平均 MSE 与 MAE 均优于 Cone-1 的数据集 | "
        f"{len(criteria['datasets_macro_better_than_cone1'])}/4"
        f"（{', '.join(criteria['datasets_macro_better_than_cone1']) or '无'}） |",
        f"| Cone-k 三 seed MSE 优于 Cone-1 的 setting | "
        f"{len(criteria['settings_cone_k_mse_better_than_cone1'])}/7"
        f"（{', '.join(criteria['settings_cone_k_mse_better_than_cone1']) or '无'}） |",
        f"| Cone-k 三 seed MAE 优于 Cone-1 的 setting | "
        f"{len(criteria['settings_cone_k_mae_better_than_cone1'])}/7"
        f"（{', '.join(criteria['settings_cone_k_mae_better_than_cone1']) or '无'}） |",
        f"| Cone-k 优于或持平 RRR-2 的 setting（MSE） | "
        f"{len(criteria['settings_cone_k_better_or_equal_rrr2'])}/7"
        f"（{', '.join(criteria['settings_cone_k_better_or_equal_rrr2']) or '无'}） |",
        f"| Cone-k 优于或持平 RRR-2 的 setting（MAE，附加） | "
        f"{len(criteria['settings_cone_k_better_or_equal_rrr2_mae'])}/7"
        f"（{', '.join(criteria['settings_cone_k_better_or_equal_rrr2_mae']) or '无'}） |",
        f"| 宽度收益是否跨 seed 稳定 | 三 seed 均值口径下 Cone-k 优于 Cone-1 的 "
        f"{len(criteria['settings_cone_k_mse_better_than_cone1'])} 个 setting 中，"
        f"{len(criteria['settings_with_mean_gain_not_from_a_single_seed'])} 个在至少 "
        f"{MIN_SEEDS_PER_SETTING} 个 seed 上同向成立"
        f"（{', '.join(criteria['settings_with_mean_gain_not_from_a_single_seed']) or '无'}）；"
        f"单个 seed 异常驱动的 setting："
        f"{', '.join(criteria['gain_is_single_seed_artifact']) or '无'} |",
        f"| 最终结论 | {criteria['verdict']}（{criteria['criteria_met']}/5 判据成立） |",
    ]
    return "\n".join(lines) + "\n"


def write_csv(rows: dict, path: Path) -> None:
    fieldnames = [
        "dataset",
        "horizon",
        "setting",
        "seed",
        "arm",
        "test_mse",
        "test_mae",
        "val_mse",
        "val_mae",
        "delta_mse_vs_direct_pct",
        "delta_mae_vs_direct_pct",
        "gate_value",
        "epochs_completed",
        "elapsed_sec",
        "run_id",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            dataset, horizon, seed, arm = key
            row = rows[key]
            direct_mse, direct_mae = test_pair(rows, dataset, horizon, seed, DIRECT)
            writer.writerow(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "setting": setting_key(dataset, horizon),
                    "seed": seed,
                    "arm": arm,
                    "test_mse": row.get("test_mse"),
                    "test_mae": row.get("test_mae"),
                    "val_mse": row.get("val_mse"),
                    "val_mae": row.get("val_mae"),
                    "delta_mse_vs_direct_pct": (
                        None if arm == DIRECT else metric_pct(
                            row.get("test_mse"), direct_mse
                        )
                    ),
                    "delta_mae_vs_direct_pct": (
                        None if arm == DIRECT else metric_pct(
                            row.get("test_mae"), direct_mae
                        )
                    ),
                    "gate_value": row.get("gate_value"),
                    "epochs_completed": row.get("epochs_completed"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "run_id": row.get("run_id"),
                    "run_dir": row.get("run_dir"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="research_runs/direction1_neighborhood_v1"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--legacy-v1-dir",
        default="research_runs/top2_direction_retention_v1/projectors",
        help="Previous plan's projector directory, used only for the Q1 drift diagnostic",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    output_dir = Path(args.output_dir) if args.output_dir else root / "aggregation"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = load_stage0(root)
    rows = load_runs(root)
    selection_path = root / "test_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    multiseed = build_multiseed(rows, selection)
    verdict = build_verdict(multiseed, selection)
    legacy_dir = Path(args.legacy_v1_dir)
    if not legacy_dir.is_absolute():
        legacy_dir = ROOT / legacy_dir
    legacy = legacy_cone1_diagnostic(root, legacy_dir) if legacy_dir.is_dir() else []

    payload = {
        "root": str(root),
        "test_set_selection": True,
        "runs_collected": len(rows),
        "selection_status": selection.get("status", "missing"),
        "selected_width_by_dataset": selection.get("selected_width_by_dataset", {}),
        "legacy_v1_diagnostic": legacy,
        "verdict": verdict,
        "multiseed": multiseed,
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    write_csv(rows, output_dir / "results.csv")
    (output_dir / "table1_stage0_geometry.md").write_text(render_table1(stage0))
    (output_dir / "table2_stage0_properties.md").write_text(render_table2(stage0))
    (output_dir / "table3_sweep_test.md").write_text(render_table3(rows))
    (output_dir / "table4_width_selection.md").write_text(render_table4(selection))
    (output_dir / "table5_multiseed.md").write_text(render_table5(multiseed))
    (output_dir / "table6_verdict.md").write_text(render_table6(verdict))
    if legacy:
        (output_dir / "table7_legacy_v1_diagnostic.md").write_text(
            render_legacy_diagnostic(legacy)
        )

    print(
        json.dumps(
            {
                "runs_collected": len(rows),
                "selection_status": selection.get("status", "missing"),
                "selected_width_by_dataset": selection.get(
                    "selected_width_by_dataset", {}
                ),
                "verdict": verdict["criteria"]["verdict"],
                "criteria_met": verdict["criteria"]["criteria_met"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
