#!/usr/bin/env python3
"""Figures for the top-2 predictive-direction retention plan (plan section 11).

Panel 1  MSE/MAE of the four arms across the six settings.
Panel 2  NLinear contribution retention of V1 and V2.
Panel 3  architecture-free diagnostic: Stage-0 lambda shares against the
         measured V1/V2 retention, to show whether the retention tracks the
         spectrum the projectors were built from.
Panel 4  energy/variance share the two variants can still see, versus retention.

Every number is read from the experiment's own ``results.csv`` and the Stage-0
``projectors.json``; nothing is typed in by hand.

Usage::

    python scripts/plot_top2_direction_retention.py \
        --root research_runs/top2_direction_retention_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _use_cjk_font():
    """Prefer an installed CJK font so Chinese axis labels render.

    Matplotlib's default DejaVu Sans has no CJK glyphs and would silently emit
    empty boxes.  Falls back to the default font when none is installed, in
    which case the figure text stays ASCII-safe.
    """
    candidates = [
        "Hiragino Sans GB", "STHeiti", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK_OK = _use_cjk_font()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("phase_only", "direct_nlinear", "keep_direction_1", "keep_direction_1_2")
LABEL = {
    "phase_only": "phase-only",
    "direct_nlinear": "direct",
    "keep_direction_1": "V1 (dir 1)",
    "keep_direction_1_2": "V2 (dir 1+2)",
}
COLOR = {
    "phase_only": "#9e9e9e",
    "direct_nlinear": "#1f77b4",
    "keep_direction_1": "#ff7f0e",
    "keep_direction_1_2": "#2ca02c",
}
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
)


def load_results(root: Path):
    rows = list(csv.DictReader((root / "results.csv").open(newline="")))
    table = {}
    for row in rows:
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]), row["arm"])
        table[key] = {
            "mse": float(row["test_mse"]) if row["test_mse"] else None,
            "mae": float(row["test_mae"]) if row["test_mae"] else None,
        }
    return table


def mean(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def arm_metric(table, dataset, horizon, arm, metric):
    return mean(
        [
            table.get((dataset, horizon, seed, arm), {}).get(metric)
            for seed in (2021, 2022, 2023)
        ]
    )


def retention(table, dataset, horizon, arm, metric):
    direct = arm_metric(table, dataset, horizon, "direct_nlinear", metric)
    phase_only = arm_metric(table, dataset, horizon, "phase_only", metric)
    candidate = arm_metric(table, dataset, horizon, arm, metric)
    if None in (direct, phase_only, candidate) or phase_only - direct <= 0:
        return None
    return (phase_only - candidate) / (phase_only - direct) * 100.0


def figure_arms(table, out_dir: Path):
    labels = [f"{d}-{h}" for d, h in SETTINGS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, metric, name in zip(axes, ("mse", "mae"), ("MSE", "MAE")):
        width = 0.2
        for index, arm in enumerate(ARMS):
            values = [
                arm_metric(table, d, h, arm, metric) or float("nan")
                for d, h in SETTINGS
            ]
            positions = [i + (index - 1.5) * width for i in range(len(labels))]
            axis.bar(positions, values, width, label=LABEL[arm], color=COLOR[arm])
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(f"test {name}")
        axis.set_title(f"test {name}: 三 seed 均值")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("前两预测方向数据保留：四个实验臂的三 seed 平均 test 指标")
    fig.tight_layout()
    path = out_dir / "arms_test_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_retention(table, out_dir: Path):
    labels = [f"{d}-{h}" for d, h in SETTINGS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, metric, name in zip(axes, ("mse", "mae"), ("MSE", "MAE")):
        width = 0.35
        for index, arm in enumerate(("keep_direction_1", "keep_direction_1_2")):
            values = [
                retention(table, d, h, arm, metric) or float("nan")
                for d, h in SETTINGS
            ]
            positions = [i + (index - 0.5) * width for i in range(len(labels))]
            axis.bar(positions, values, width, label=LABEL[arm], color=COLOR[arm])
        axis.axhline(100, color="black", linestyle="--", linewidth=1)
        axis.axhline(80, color="red", linestyle=":", linewidth=1)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(f"{name} 贡献保留率 (%)")
        axis.set_title(f"{name}: NLinear 贡献保留率（100% = 完整 direct）")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("V1/V2 保留了完整 NLinear 多少有效贡献")
    fig.tight_layout()
    path = out_dir / "retention.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_spectrum_vs_retention(table, projectors, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    lambda2 = []
    used_v1, used_v12 = [], []
    ret_v1, ret_v2 = [], []
    labels = []
    for dataset, horizon in SETTINGS:
        entry = projectors.get(f"{dataset}-{horizon}", {})
        labels.append(f"{dataset}-{horizon}")
        lambda2.append(entry.get("lambda2_share", float("nan")))
        used_v1.append(entry.get("used_var_share_v1", float("nan")) * 100.0)
        used_v12.append(entry.get("used_var_share_v12", float("nan")) * 100.0)
        ret_v1.append(retention(table, dataset, horizon, "keep_direction_1", "mse"))
        ret_v2.append(retention(table, dataset, horizon, "keep_direction_1_2", "mse"))

    axis = axes[0]
    axis.plot(labels, lambda2, "o-", label="λ2 share（训练集 RRR 谱）")
    axis.plot(labels, [r if r is not None else float("nan") for r in ret_v2],
              "s--", label="V2 MSE 保留率 / 100")
    axis.set_ylabel("share")
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_title("方向 2 的理论权重 vs 实测增量")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1]
    axis.plot(labels, used_v1, "o-", label="V1 可见中心化输入方差占比 (%)")
    axis.plot(labels, used_v12, "s-", label="V2 可见中心化输入方差占比 (%)")
    axis.plot(labels, [r if r is not None else float("nan") for r in ret_v1],
              "^--", label="V1 MSE 保留率 (%)")
    axis.set_ylabel("percent")
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_title("可见输入方差 vs 保留率")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    fig.suptitle("投影器谱结构与端到端保留率的关系")
    fig.tight_layout()
    path = out_dir / "spectrum_vs_retention.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_results(root)
    projectors = {}
    index_path = root / "projectors" / "projectors.json"
    if index_path.exists():
        projectors = json.loads(index_path.read_text()).get("projectors", {})

    written = [figure_arms(table, out_dir), figure_retention(table, out_dir)]
    if projectors:
        written.append(figure_spectrum_vs_retention(table, projectors, out_dir))
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
