#!/usr/bin/env python3
"""Summarize sample-aligned low-rank interaction activation audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODES = [
    "phaseformer_l_i0_late_fusion",
    "phaseformer_l_i1_phase_to_trend",
    "phaseformer_l_i2_trend_to_phase",
    "phaseformer_l_i3_parallel_bidir",
    "phaseformer_l_i6_alternating",
    "phaseformer_l_i7_joint",
]
ACTIVATION_COLUMNS = [
    "trend0_norm", "trend1_norm", "trend_delta_norm",
    "trend_message_norm", "trend_message_cosine",
    "phase0_norm", "phase1_norm", "phase_delta_norm", "phase_cosine",
    "phase_message_norm", "phase_message_cosine",
    "pt_entropy", "pt_max", "tp_entropy", "tp_max",
    "joint_trend_to_phase_mass", "joint_phase_to_trend_mass",
    "joint_phase_to_phase_mass", "joint_trend_query_entropy",
]


def correlation(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 3:
        return float("nan")
    xv = x.to_numpy()[mask]
    yv = y.to_numpy()[mask]
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    cases = []
    correlations = []
    for path in sorted(args.input_root.glob("*/activation_rows.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        setting = f"{frame.dataset.iloc[0]}_H{int(frame.horizon.iloc[0])}_S{int(frame.seed.iloc[0])}"
        key = ["sample_index", "channel"]
        base = frame[frame["mode"] == MODES[0]].set_index(key)
        if base.empty:
            continue
        for mode in MODES:
            cur = frame[frame["mode"] == mode].set_index(key)
            if cur.empty:
                continue
            common = base.join(cur, lsuffix="_i0", rsuffix="_cur", how="inner")
            if mode == MODES[0]:
                delta_mae = pd.Series(0.0, index=common.index)
                delta_mse = pd.Series(0.0, index=common.index)
            else:
                delta_mae = common["fused_mae_cur"] - common["fused_mae_i0"]
                delta_mse = common["fused_mse_cur"] - common["fused_mse_i0"]
            summary = {
                "setting": setting,
                "dataset": str(frame.dataset.iloc[0]),
                "horizon": int(frame.horizon.iloc[0]),
                "seed": int(frame.seed.iloc[0]),
                "mode": mode,
                "rows": int(len(common)),
                "fused_mae": float(common["fused_mae_cur"].mean()),
                "fused_mse": float(common["fused_mse_cur"].mean()),
                "delta_mae_vs_i0": float(delta_mae.mean()),
                "delta_mse_vs_i0": float(delta_mse.mean()),
            }
            for col in ACTIVATION_COLUMNS:
                summary[col] = float(common[f"{col}_cur"].mean())
            summaries.append(summary)

            if mode != MODES[0]:
                for metric_name, delta in [("mae", delta_mae), ("mse", delta_mse)]:
                    for col in ACTIVATION_COLUMNS:
                        correlations.append({
                            "setting": setting, "mode": mode, "metric": metric_name,
                            "activation": col, "correlation": correlation(delta, common[f"{col}_cur"]),
                            "rows": int(len(common)),
                        })
                ranked = pd.DataFrame({"delta_mae": delta_mae, "delta_mse": delta_mse}, index=common.index)
                for case_type, order_col, ascending in [
                    ("improvement_mse", "delta_mse", True),
                    ("regression_mse", "delta_mse", False),
                    ("improvement_mae", "delta_mae", True),
                    ("regression_mae", "delta_mae", False),
                ]:
                    selected = ranked.sort_values(order_col, ascending=ascending).head(args.top_k)
                    for (sample_index, channel), row in selected.iterrows():
                        values = common.loc[(sample_index, channel)]
                        item = {
                            "setting": setting, "dataset": str(frame.dataset.iloc[0]),
                            "horizon": int(frame.horizon.iloc[0]), "seed": int(frame.seed.iloc[0]),
                            "mode": mode, "case_type": case_type,
                            "sample_index": int(sample_index), "channel": int(channel),
                            "delta_mae_vs_i0": float(row.delta_mae),
                            "delta_mse_vs_i0": float(row.delta_mse),
                            "i0_mae": float(values.fused_mae_i0),
                            "candidate_mae": float(values.fused_mae_cur),
                        }
                        for col in ACTIVATION_COLUMNS:
                            item[col] = float(values[f"{col}_cur"])
                        cases.append(item)

    pd.DataFrame(summaries).to_csv(args.output_dir / "activation_setting_summary.csv", index=False)
    pd.DataFrame(correlations).to_csv(args.output_dir / "activation_error_correlations.csv", index=False)
    pd.DataFrame(cases).to_csv(args.output_dir / "activation_selected_cases.csv", index=False)
    print({"settings": len({x["setting"] for x in summaries}), "summary_rows": len(summaries),
           "correlation_rows": len(correlations), "case_rows": len(cases)})


if __name__ == "__main__":
    main()
