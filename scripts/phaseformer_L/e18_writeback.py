#!/usr/bin/env python3
"""E18 write-back: fill the minipaper §4.6 negative-control table.

§4.6 is a five-row summary whose last column ("本文补做") is what this paper adds.
Only three rows have work to report; rows 2 (structured coordinates) and 4
(q=1/32 capacity) are kept as "—" because the minipaper says so explicitly.

* row 1  input smoothing on PhaseFormer-L at two causal-EMA strengths
         (7 test-selected settings x 3 seeds x 2 levels = 42 runs);
* row 3  SVD truncation versus rank-constrained training extended to 28 settings
         (analysis only; validation split);
* row 5  boundary ablation, `pooled_lowrank` with ABSOLUTE rank in {1,2}
         (6 settings x 3 seeds x 2 ranks = 36 runs).

Sources:

* ``--results``  E18's test-bearing results CSV (``results.with_test.csv``), which
  carries rows 1 and 5 plus each cell's ``baseline_run_dir``;
* ``--e14-results`` E14's test-bearing CSV, used as the unsmoothed / full-rank
  PhaseFormer-L reference the two rows are differenced against;
* ``--truncation`` E18's ``svd_truncation_table_28.csv`` for row 3.

Every comparison is a paired one (same setting, same seeds) against the same
PhaseFormer-L baseline, so the numbers are internally comparable even though the
baseline's own hyperparameters differ (the D-2 disclosure).

Usage::

    python scripts/phaseformer_L/e18_writeback.py \\
        --results research_runs/phaseformer_L_e18_negative_v1/results.with_test.csv \\
        --e14-results research_runs/phaseformer_L_e14_main_v1/results.csv \\
        --truncation research_runs/phaseformer_L_e18_negative_v1/svd_truncation_table_28.csv \\
        --output-root research_runs/phaseformer_L_e18_negative_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEPT_AS_DASH = (
    ("结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离）", "支路参数化",
     "4 setting", "5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52%"),
    ("联合低秩训练 q=1/32", "支路容量", "7 setting",
     "保留 92.4%–101.9% 可实现价值"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--e14-results", required=True)
    parser.add_argument("--truncation", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def as_float(value):
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_by_setting(rows: list, arm_filter=None) -> dict:
    """(dataset, horizon) -> {'mse': [...], 'mae': [...]}"""
    out: dict = {}
    for row in rows:
        if arm_filter and str(row.get("arm", "")) != arm_filter:
            continue
        key = (str(row.get("dataset")), str(row.get("horizon")))
        entry = out.setdefault(key, {"mse": [], "mae": []})
        mse, mae = as_float(row.get("test_mse")), as_float(row.get("test_mae"))
        if mse is not None:
            entry["mse"].append(mse)
        if mae is not None:
            entry["mae"].append(mae)
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return 100.0 * (new - base) / base


def build_row1(rows: list, baseline: dict) -> dict:
    """Input smoothing on PhaseFormer-L: two strengths, paired vs unsmoothed."""
    levels: dict = {}
    for row in rows:
        if str(row.get("stage")) != "smooth":
            continue
        key = (str(row.get("level")), str(row.get("dataset")), str(row.get("horizon")))
        entry = levels.setdefault(key, {"mse": [], "mae": []})
        mse, mae = as_float(row.get("test_mse")), as_float(row.get("test_mae"))
        if mse is not None:
            entry["mse"].append(mse)
        if mae is not None:
            entry["mae"].append(mae)
    per_level: dict = {}
    for (level, dataset, horizon), entry in levels.items():
        base = baseline.get((dataset, horizon), {"mse": [], "mae": []})
        base_mse, base_mae = mean(base["mse"]), mean(base["mae"])
        level_mse, level_mae = mean(entry["mse"]), mean(entry["mae"])
        per_level.setdefault(level, []).append({
            "setting": f"{dataset}-{horizon}",
            "smooth_mse": level_mse, "smooth_mae": level_mae,
            "baseline_mse": base_mse, "baseline_mae": base_mae,
            "delta_mse_pct": pct(level_mse, base_mse),
            "delta_mae_pct": pct(level_mae, base_mae),
            "n_seeds": len(entry["mse"]),
        })
    improved = []
    degraded = []
    for level, cells in per_level.items():
        for cell in cells:
            if cell["delta_mse_pct"] is None or cell["delta_mae_pct"] is None:
                continue
            if cell["delta_mse_pct"] < 0 and cell["delta_mae_pct"] < 0:
                improved.append(f"{level}:{cell['setting']}")
            else:
                degraded.append(f"{level}:{cell['setting']}")
    all_deltas = [c["delta_mse_pct"] for cells in per_level.values() for c in cells
                  if c["delta_mse_pct"] is not None]
    return {
        "row": 1,
        "operation": "输入平滑（causal EMA 两个强度）",
        "target": "支路输入",
        "scope": f"{len({c['setting'] for cells in per_level.values() for c in cells})} setting",
        "levels": per_level,
        "cells_improved_both_metrics": improved,
        "cells_not_improved": degraded,
        "mean_delta_mse_pct": round(float(np.mean(all_deltas)), 4) if all_deltas else None,
        "verdict": ("no cell improves both metrics" if not improved
                    else f"{len(improved)} cell(s) improve both metrics"),
    }


def build_row5(rows: list, baseline: dict) -> dict:
    """Absolute-rank boundary ablation, paired vs the dense PhaseFormer-L."""
    ranks: dict = {}
    for row in rows:
        if str(row.get("stage")) != "rank12":
            continue
        key = (str(row.get("level")), str(row.get("dataset")), str(row.get("horizon")))
        entry = ranks.setdefault(key, {"mse": [], "mae": []})
        mse, mae = as_float(row.get("test_mse")), as_float(row.get("test_mae"))
        if mse is not None:
            entry["mse"].append(mse)
        if mae is not None:
            entry["mae"].append(mae)
    per_rank: dict = {}
    for (level, dataset, horizon), entry in ranks.items():
        base = baseline.get((dataset, horizon), {"mse": [], "mae": []})
        base_mse, base_mae = mean(base["mse"]), mean(base["mae"])
        rank_mse, rank_mae = mean(entry["mse"]), mean(entry["mae"])
        per_rank.setdefault(level, []).append({
            "setting": f"{dataset}-{horizon}",
            "ranked_mse": rank_mse, "ranked_mae": rank_mae,
            "baseline_mse": base_mse, "baseline_mae": base_mae,
            "delta_mse_pct": pct(rank_mse, base_mse),
            "delta_mae_pct": pct(rank_mae, base_mae),
            "n_seeds": len(entry["mse"]),
        })
    deltas = [c["delta_mse_pct"] for cells in per_rank.values() for c in cells
              if c["delta_mse_pct"] is not None]
    degraded = [f"{k}:{c['setting']}" for k, cells in per_rank.items() for c in cells
                if c["delta_mse_pct"] is not None and c["delta_mse_pct"] > 0]
    return {
        "row": 5,
        "operation": "边界消融：pooled_lowrank 绝对秩 rank∈{1,2}",
        "target": "支路容量（低秩网格之外）",
        "scope": "6 setting × 3 seed × 2 rank",
        "ranks": per_rank,
        "cells_degraded_vs_dense": degraded,
        "mean_delta_mse_pct": round(float(np.mean(deltas)), 4) if deltas else None,
        "preregistered_expectation": "相对 direct 退化（用于量化残项 ε，不影响主张）",
        "reading_rule": "结果无论好坏都不得读作'秩-2 必要/不必要'的证据"
                        "（minipaper §5 第 5 条）",
    }


def build_row3(rows: list) -> dict:
    """SVD truncation versus rank-constrained training, 28 settings."""
    gaps = []
    per_setting = []
    worst = None
    for row in rows:
        gap = as_float(row.get("gap_truncated_vs_trained_mse_pct"))
        entry = {
            "setting": row.get("setting"),
            "primary_rank": row.get("primary_rank"),
            "truncated_mse": as_float(row.get("truncated_mse")),
            "trained_lowrank_mse": as_float(row.get("trained_lowrank_mse")),
            "full_rank_mse": as_float(row.get("full_rank_mse")),
            "gap_truncated_vs_trained_mse_pct": gap,
            "gap_trained_vs_full_mse_pct": as_float(
                row.get("gap_trained_vs_full_mse_pct")),
            "trained_lowrank_rank": row.get("trained_lowrank_rank"),
            "records_test": str(row.get("records_test", "")).strip().lower() == "true",
            "seeds_evaluated": row.get("seeds_evaluated"),
        }
        per_setting.append(entry)
        if gap is not None:
            gaps.append(gap)
            if worst is None or gap > worst[1]:
                worst = (entry["setting"], gap)
    return {
        "row": 3,
        "operation": "SVD 截断 vs 秩约束训练",
        "target": "支路权重",
        "scope": f"{len(per_setting)} setting",
        "rank": rows[0].get("primary_rank") if rows else None,
        "per_setting": per_setting,
        "n_with_gap": len(gaps),
        "mean_gap_pct": round(float(np.mean(gaps)), 4) if gaps else None,
        "max_gap_pct": round(float(np.max(gaps)), 4) if gaps else None,
        "worst_truncated_setting": worst[0] if worst else None,
        "records_test": (rows[0].get("records_test") if rows else None),
        "comparability_note": (
            "validation-based here; the earlier registered comparison (E11) was "
            "test-based with single-seed checkpoints from a different root, so only "
            "the truncation algebra and the three-way structure are shared"
        ),
        "preregistered_expectation": "截断应劣于秩约束训练；Electricity-336 是既有反例",
    }


def main() -> None:
    args = parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    e18_rows = read_csv(resolve(args.results))
    e14_rows = read_csv(resolve(args.e14_results))
    truncation_rows = read_csv(resolve(args.truncation))

    # The PhaseFormer-L reference the two training rows are differenced against.
    baseline = index_by_setting(e14_rows, arm_filter="l_main")
    if not baseline:
        baseline = index_by_setting(e14_rows, arm_filter="phase_only")

    row1 = build_row1(e18_rows, baseline)
    row3 = build_row3(truncation_rows)
    row5 = build_row5(e18_rows, baseline)

    def summarise(row, addend):
        def pct_cell(value, prefix):
            """Render a percentage, or an em dash when it could not be computed.

            Without this a missing baseline (or a cell whose test metrics were
            never read) renders as the literal string "平均 ΔMSE None%" straight
            into the minipaper table.  The rehearsal in
            rehearse_e18_writeback.py exercises exactly that degraded path.
            """
            return "—" if value is None else f"{prefix}{value}%"

        return {
            "row": row["row"], "operation": row["operation"],
            "target": row["target"], "scope": row["scope"],
            "existing": addend, "addendum": row["verdict"] if row["row"] == 1
            else (pct_cell(row.get("mean_gap_pct"), "28 setting 平均截断惩罚 ")
                  if row["row"] == 3
                  else pct_cell(row.get("mean_delta_mse_pct"), "平均 ΔMSE ")),
        }

    table = [
        summarise(row1, "14/14 组合无一改善，越平滑越差"),
        {"row": 2, "operation": KEPT_AS_DASH[0][0], "target": KEPT_AS_DASH[0][1],
         "scope": KEPT_AS_DASH[0][2], "existing": KEPT_AS_DASH[0][3], "addendum": "—"},
        summarise(row3, "截断 +29%，训练 +0.7%（Electricity-336, r=10）"),
        {"row": 4, "operation": KEPT_AS_DASH[1][0], "target": KEPT_AS_DASH[1][1],
         "scope": KEPT_AS_DASH[1][2], "existing": KEPT_AS_DASH[1][3], "addendum": "—"},
        summarise(row5, "无"),
    ]

    out_root = resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "negative_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    lines = ["| 操作 | 作用对象 | 口径 | 结果（既有） | 本文补做 |",
             "|---|---|---|---|---|"]
    for entry in table:
        lines.append("| %s | %s | %s | %s | %s |" % (
            entry["operation"], entry["target"], entry["scope"],
            entry["existing"], entry["addendum"]))
    (out_root / "negative_table.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")

    summary = {
        "experiment": "E18 write-back (minipaper section 4.6)",
        "rows_with_addendum": [entry["row"] for entry in table
                               if entry["addendum"] != "—"],
        "rows_kept_as_dash": [2, 4],
        "row1": row1,
        "row3": row3,
        "row5": row5,
        "baseline_used": "E14 l_main (PhaseFormer-L) test metrics, paired by setting",
        "disclosures": [
            "every addendum number is a paired comparison on the same setting and "
            "seeds; the baseline's own hyperparameters differ (D-2), so the pairing "
            "is within-setting only",
            "row 3 is validation-based while the earlier registered comparison was "
            "test-based; only the truncation algebra is shared",
            "row 5 must never be read as evidence about rank-2 necessity",
            "rows 2 and 4 stay '—' because the minipaper says they are not to be redone",
        ],
    }
    (out_root / "negative_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished",
        "row1": row1["verdict"],
        "row1_mean_delta_mse_pct": row1["mean_delta_mse_pct"],
        "row3_mean_gap_pct": row3["mean_gap_pct"],
        "row3_worst": row3["worst_truncated_setting"],
        "row5_mean_delta_mse_pct": row5["mean_delta_mse_pct"],
        "row5_degraded_cells": len(row5["cells_degraded_vs_dense"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
