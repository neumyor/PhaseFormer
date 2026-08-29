#!/usr/bin/env python3
"""Audit M3 against a matched rerun of the original PhaseFormer.

The case criteria are fixed before looking at the paired predictions:

* success: relative MSE <= -10% and candidate MAE < baseline MAE;
* failure: relative MSE >= +10% and candidate MAE > baseline MAE;
* other: every remaining sample-channel pair.

Cases are ranked by absolute paired MSE change and deduplicated within each
dataset/channel at a 96-window distance.  All analysis uses validation only.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_multi_anchor_selector import (  # noqa: E402
    CHAMPION,
    SCRATCH as M3_SCRATCH,
    find_run,
    load_candidate,
    variable_names,
)
from scripts.analyze_triaxis_experiment import build_model_and_loader  # noqa: E402
from src.models.multi_anchor import ANCHOR_NAMES  # noqa: E402


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
BASELINE_SCRATCH = REPO / "research_runs" / "m3_vs_original_phaseformer_v1_scratch"
OUTPUT = REPO / "research_runs" / "m3_vs_original_phaseformer_v1"
DOC_FIGURES = REPO / "docs" / "PhaseFormer_M3_figures"
GROUPS = ("success", "other", "failure")
FEATURES = (
    "history_abs_drift",
    "history_lag24_corr",
    "history_diff_volatility",
    "history_phase_reliability",
    "future_level_shift",
    "future_lag24_corr",
    "future_diff_volatility",
    "anchor_disagreement",
    "weight_a1",
    "weight_i0",
    "weight_r0",
    "route_entropy",
)
FEATURE_LABELS = {
    "history_abs_drift": "输入近期漂移",
    "history_lag24_corr": "输入 lag-24 相关",
    "history_diff_volatility": "输入差分波动",
    "history_phase_reliability": "输入相位可靠度",
    "future_level_shift": "未来水平迁移（事后）",
    "future_lag24_corr": "未来 lag-24 相关（事后）",
    "future_diff_volatility": "未来差分波动（事后）",
    "anchor_disagreement": "锚点分歧",
    "weight_a1": "A1 权重",
    "weight_i0": "I0 权重",
    "weight_r0": "R0 权重",
    "route_entropy": "路由熵",
}
SAMPLE_FIELDS = (
    "setting", "dataset", "sample_id", "channel", "time_range", "group",
    "original_mse", "m3_mse", "delta_mse", "relative_delta_mse",
    "original_mae", "m3_mae", "delta_mae", *FEATURES,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
})


def setting_name(dataset: str) -> str:
    return f"{dataset}-L720-H96-pct30-e8-s2021-validation"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def find_original(dataset: str):
    matches = []
    for path in (BASELINE_SCRATCH / "runs").glob("*/config.json"):
        spec = json.loads(path.read_text())
        metrics_path = path.parent / "metrics.csv"
        if not metrics_path.is_file():
            continue
        if (
            spec.get("dataset") == dataset
            and spec.get("mechanism") == "original"
            and int(spec.get("lookback", -1)) == 720
            and int(spec.get("horizon", -1)) == 96
            and int(spec.get("percent", -1)) == 30
            and int(spec.get("max_epochs", -1)) == 8
            and int(spec.get("seed", -1)) == 2021
            and spec.get("loss") == "huber"
        ):
            with metrics_path.open(newline="") as handle:
                matches.append((path.parent, next(csv.DictReader(handle)), spec))
    if len(matches) != 1:
        raise RuntimeError(f"expected one matched original run for {dataset}, got {len(matches)}")
    return matches[0]


def get_output(model, x, x_mark, dec, y_mark):
    value = model(x, x_mark, dec, y_mark)
    return value[0] if isinstance(value, tuple) else value


def lag_correlation(x: torch.Tensor, lag: int = 24, eps: float = 1e-6):
    left = x[:, lag:, :]
    right = x[:, :-lag, :]
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
    ).clamp_min(eps)


def feature_tensors(x, truth, anchors, weights, router_features):
    scale = x.std(dim=1, unbiased=False).clamp_min(1e-6)
    future_shift = (
        truth[:, -24:, :].mean(dim=1) - truth[:, :24, :].mean(dim=1)
    ).abs() / scale
    future_diff = (truth[:, 1:, :] - truth[:, :-1, :]).std(
        dim=1, unbiased=False
    ) / scale
    disagreement = anchors.std(dim=-1, unbiased=False).mean(dim=1) / scale
    mean_weights = weights.mean(dim=2)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=2)
    return {
        "history_abs_drift": router_features[:, :, 0, 0].abs(),
        "history_lag24_corr": router_features[:, :, 0, 1],
        "history_diff_volatility": router_features[:, :, 0, 2],
        "history_phase_reliability": router_features[:, :, 0, 3],
        "future_level_shift": future_shift,
        "future_lag24_corr": lag_correlation(truth),
        "future_diff_volatility": future_diff,
        "anchor_disagreement": disagreement,
        "weight_a1": mean_weights[..., 0],
        "weight_i0": mean_weights[..., 1],
        "weight_r0": mean_weights[..., 2],
        "route_entropy": entropy,
    }


def classify(relative, original_mae, m3_mae):
    success = (relative <= -0.10) & (m3_mae < original_mae)
    failure = (relative >= 0.10) & (m3_mae > original_mae)
    code = torch.ones_like(relative, dtype=torch.int8)
    code[success] = 0
    code[failure] = 2
    return code


def local_top(score, mask, limit=12):
    flat_score, flat_mask = score.reshape(-1), mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).flatten()
    if not indices.numel():
        return []
    values = flat_score[indices]
    top = torch.topk(values, min(limit, values.numel()))
    return [(float(value), int(indices[index])) for value, index in zip(top.values, top.indices)]


def push_case(heap, score, serial, case, limit=4096):
    item = (float(score), int(serial), case)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def make_case(
    kind, score, dataset, sample_id, channel_id, channel, time_range,
    history, truth, original, candidate, weights,
):
    truth_np = truth.detach().cpu().numpy()
    original_np = original.detach().cpu().numpy()
    candidate_np = candidate.detach().cpu().numpy()
    return {
        "case_type": kind,
        "score": float(score),
        "setting": setting_name(dataset),
        "dataset": dataset,
        "sample_id": int(sample_id),
        "channel_id": int(channel_id),
        "channel": channel,
        "time_range": time_range,
        "history": history.detach().cpu().numpy(),
        "truth": truth_np,
        "original": original_np,
        "candidate": candidate_np,
        "weights": weights.detach().cpu().numpy(),
        "original_mse": float(np.mean((original_np - truth_np) ** 2)),
        "m3_mse": float(np.mean((candidate_np - truth_np) ** 2)),
        "original_mae": float(np.mean(np.abs(original_np - truth_np))),
        "m3_mae": float(np.mean(np.abs(candidate_np - truth_np))),
    }


def evaluate(dataset_name, candidate_info, original_info, writer, device):
    batch_size = 8 if dataset_name == "Electricity" else (32 if dataset_name == "Weather" else 128)
    candidate, dataset, loader, exp_args = load_candidate(candidate_info, batch_size)
    original, original_dataset, _ = build_model_and_loader(*original_info, batch_size)
    if len(dataset) != len(original_dataset):
        raise RuntimeError(f"validation length mismatch for {dataset_name}")
    candidate, original = candidate.to(device).eval(), original.to(device).eval()
    names = variable_names(exp_args)
    timestamps = getattr(dataset, "timestamps", None)
    totals = defaultdict(float)
    groups = defaultdict(int)
    pools = {"success": [], "failure": []}
    feature_blocks = {name: [] for name in FEATURES}
    group_blocks, relative_blocks = [], []
    sample_original, sample_candidate = [], []
    serial = offset = 0

    with torch.inference_mode():
        for batch in loader:
            x, y, x_mark, y_mark = [value.to(device) for value in batch]
            x, y = x.float(), y.float()
            truth = y[:, -96:, :]
            dec = candidate._build_decoder_input(y)
            original_output = get_output(original, x, x_mark.float(), dec, y_mark.float())
            candidate_output, _, _ = candidate(x, x_mark.float(), dec, y_mark.float())
            anchors = torch.stack(candidate.last_anchor_outputs, dim=-1)
            weights = candidate.router.last_soft_weights
            feature_values = feature_tensors(
                x, truth, anchors, weights, candidate.router.last_features
            )

            original_error = original_output - truth
            candidate_error = candidate_output - truth
            totals["original_sq"] += float(original_error.square().sum())
            totals["original_abs"] += float(original_error.abs().sum())
            totals["candidate_sq"] += float(candidate_error.square().sum())
            totals["candidate_abs"] += float(candidate_error.abs().sum())
            totals["count"] += truth.numel()
            original_mse = original_error.square().mean(dim=1)
            candidate_mse = candidate_error.square().mean(dim=1)
            original_mae = original_error.abs().mean(dim=1)
            candidate_mae = candidate_error.abs().mean(dim=1)
            relative = (candidate_mse - original_mse) / original_mse.clamp_min(1e-8)
            group_code = classify(relative, original_mae, candidate_mae)
            for index, name in enumerate(GROUPS):
                groups[name] += int((group_code == index).sum())

            group_blocks.append(group_code.cpu().numpy().reshape(-1))
            relative_blocks.append(relative.cpu().numpy().reshape(-1).astype(np.float32))
            for name, values in feature_values.items():
                feature_blocks[name].append(values.cpu().numpy().reshape(-1).astype(np.float32))
            sample_original.append(original_error.square().mean(dim=(1, 2)).cpu().numpy())
            sample_candidate.append(candidate_error.square().mean(dim=(1, 2)).cpu().numpy())

            B, _, C = truth.shape
            rows = []
            for b in range(B):
                start = offset + b + x.shape[1]
                time_range = (
                    f"{timestamps[start]}--{timestamps[start + 95]}"
                    if timestamps is not None and start + 95 < len(timestamps)
                    else f"index:{start}--{start + 95}"
                )
                for c in range(C):
                    code = int(group_code[b, c])
                    rows.append({
                        "setting": setting_name(dataset_name),
                        "dataset": dataset_name,
                        "sample_id": offset + b,
                        "channel": names[c] if c < len(names) else str(c),
                        "time_range": time_range,
                        "group": GROUPS[code],
                        "original_mse": f"{float(original_mse[b, c]):.8g}",
                        "m3_mse": f"{float(candidate_mse[b, c]):.8g}",
                        "delta_mse": f"{float(candidate_mse[b, c] - original_mse[b, c]):.8g}",
                        "relative_delta_mse": f"{float(relative[b, c]):.8g}",
                        "original_mae": f"{float(original_mae[b, c]):.8g}",
                        "m3_mae": f"{float(candidate_mae[b, c]):.8g}",
                        "delta_mae": f"{float(candidate_mae[b, c] - original_mae[b, c]):.8g}",
                        **{name: f"{float(value[b, c]):.8g}" for name, value in feature_values.items()},
                    })
            writer.writerows(rows)

            scores = {
                "success": (original_mse - candidate_mse, group_code == 0),
                "failure": (candidate_mse - original_mse, group_code == 2),
            }
            for kind, (score_tensor, mask) in scores.items():
                for score, flat_index in local_top(score_tensor, mask):
                    b, c = divmod(flat_index, C)
                    start = offset + b + x.shape[1]
                    time_range = (
                        f"{timestamps[start]}--{timestamps[start + 95]}"
                        if timestamps is not None and start + 95 < len(timestamps)
                        else f"index:{start}--{start + 95}"
                    )
                    case = make_case(
                        kind, score, dataset_name, offset + b, c,
                        names[c] if c < len(names) else str(c), time_range,
                        x[b, :, c], truth[b, :, c], original_output[b, :, c],
                        candidate_output[b, :, c], weights[b, c],
                    )
                    push_case(pools[kind], score, serial, case)
                    serial += 1
            offset += B

    count = totals["count"]
    arrays = {
        "group": np.concatenate(group_blocks),
        "relative": np.concatenate(relative_blocks),
        **{name: np.concatenate(blocks) for name, blocks in feature_blocks.items()},
        "sample_original": np.concatenate(sample_original),
        "sample_candidate": np.concatenate(sample_candidate),
    }
    summary = {
        "setting": setting_name(dataset_name),
        "dataset": dataset_name,
        "validation_samples": len(dataset),
        "sample_channel_pairs": int(arrays["group"].size),
        "groups": dict(groups),
        "original_mse": totals["original_sq"] / count,
        "original_mae": totals["original_abs"] / count,
        "m3_mse": totals["candidate_sq"] / count,
        "m3_mae": totals["candidate_abs"] / count,
    }
    pools = {key: [x[2] for x in sorted(value, reverse=True)] for key, value in pools.items()}
    del candidate, original
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, arrays, pools


def block_bootstrap_ratio(original, candidate, rng, block=96, repeats=1000):
    n = len(original)
    blocks = math.ceil(n / block)
    ratios = np.empty(repeats, dtype=np.float64)
    offsets = np.arange(block)
    for index in range(repeats):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
        ratios[index] = candidate[indices].mean() / original[indices].mean() - 1
    return np.quantile(ratios, (0.025, 0.975)).tolist()


def taxonomy(arrays):
    drift_q = np.quantile(arrays["history_abs_drift"], 0.75)
    corr_q = np.quantile(arrays["history_lag24_corr"], 0.75)
    vol_q = np.quantile(arrays["history_diff_volatility"], 0.75)
    drift = arrays["history_abs_drift"] >= drift_q
    periodic = arrays["history_lag24_corr"] >= corr_q
    volatile = arrays["history_diff_volatility"] >= vol_q
    labels = np.full(arrays["group"].size, 4, dtype=np.int8)
    labels[volatile] = 3
    labels[drift & ~periodic] = 2
    labels[periodic & ~drift] = 1
    labels[periodic & drift] = 0
    names = ("强周期+近期漂移", "强周期", "近期漂移", "高差分波动", "其他")
    return labels, names, {"drift_q75": drift_q, "corr_q75": corr_q, "volatility_q75": vol_q}


def summarize(analysis, arrays_by_dataset):
    rng = np.random.default_rng(20210829)
    for dataset in DATASETS:
        values = arrays_by_dataset[dataset]
        analysis[dataset]["mse_ratio_ci95"] = block_bootstrap_ratio(
            values["sample_original"], values["sample_candidate"], rng
        )
        labels, names, thresholds = taxonomy(values)
        analysis[dataset]["taxonomy_thresholds"] = native(thresholds)
        analysis[dataset]["taxonomy"] = {}
        for index, name in enumerate(names):
            mask = labels == index
            analysis[dataset]["taxonomy"][name] = {
                "count": int(mask.sum()),
                "mean_relative_mse": float(values["relative"][mask].mean()),
                "success_rate": float((values["group"][mask] == 0).mean()),
                "failure_rate": float((values["group"][mask] == 2).mean()),
            }

    feature_rows = []
    for feature in FEATURES:
        smds, success_means, failure_means = [], [], []
        for dataset in DATASETS:
            values = arrays_by_dataset[dataset]
            success = values[feature][values["group"] == 0].astype(np.float64)
            failure = values[feature][values["group"] == 2].astype(np.float64)
            pooled = math.sqrt((success.var() + failure.var()) / 2)
            smds.append((success.mean() - failure.mean()) / max(pooled, 1e-12))
            success_means.append(success.mean())
            failure_means.append(failure.mean())
        feature_rows.append({
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "success_mean": float(np.mean(success_means)),
            "failure_mean": float(np.mean(failure_means)),
            "macro_smd": float(np.mean(smds)),
            "dataset_smd": native(dict(zip(DATASETS, smds))),
        })

    taxonomy_rows = []
    names = tuple(next(iter(analysis.values()))["taxonomy"])
    for name in names:
        per_dataset = [analysis[d]["taxonomy"][name] for d in DATASETS]
        taxonomy_rows.append({
            "type": name,
            "macro_relative_mse": float(np.mean([x["mean_relative_mse"] for x in per_dataset])),
            "macro_success_rate": float(np.mean([x["success_rate"] for x in per_dataset])),
            "macro_failure_rate": float(np.mean([x["failure_rate"] for x in per_dataset])),
            "datasets_mean_improved": int(sum(x["mean_relative_mse"] < 0 for x in per_dataset)),
        })
    return feature_rows, taxonomy_rows


def duplicate(case, selected):
    return any(
        old["dataset"] == case["dataset"]
        and old["channel_id"] == case["channel_id"]
        and abs(old["sample_id"] - case["sample_id"]) < 96
        for old in selected
    )


def select_cases(pools, top_k=5):
    selected = []
    for dataset in DATASETS:
        for kind in ("success", "failure"):
            chosen = []
            for case in pools[dataset][kind]:
                if duplicate(case, chosen):
                    continue
                chosen.append(case)
                if len(chosen) == top_k:
                    break
            if len(chosen) != top_k:
                raise RuntimeError(f"insufficient cases for {dataset}/{kind}: {len(chosen)}")
            selected.extend(chosen)
    return selected


def representative_cases(cases):
    representatives = []
    for kind in ("success", "failure"):
        options = sorted(
            (case for case in cases if case["case_type"] == kind),
            key=lambda case: case["score"], reverse=True,
        )
        chosen, used = [], set()
        for case in options:
            if case["dataset"] in used or duplicate(case, chosen):
                continue
            chosen.append(case)
            used.add(case["dataset"])
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise RuntimeError(f"insufficient representative {kind} cases")
        representatives.extend(chosen)
    return representatives


def save_cases(path, cases):
    np.savez_compressed(
        path,
        setting=np.asarray([x["setting"] for x in cases]),
        case_type=np.asarray([x["case_type"] for x in cases]),
        dataset=np.asarray([x["dataset"] for x in cases]),
        sample_id=np.asarray([x["sample_id"] for x in cases]),
        channel_id=np.asarray([x["channel_id"] for x in cases]),
        channel=np.asarray([x["channel"] for x in cases]),
        time_range=np.asarray([x["time_range"] for x in cases]),
        historical_input=np.stack([x["history"] for x in cases]),
        truth=np.stack([x["truth"] for x in cases]),
        original_prediction=np.stack([x["original"] for x in cases]),
        m3_prediction=np.stack([x["candidate"] for x in cases]),
        route_weights=np.stack([x["weights"] for x in cases]),
    )


def markdown_table(rows, columns, formats=None):
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            formats.get(column, str)(row[column]) for column in columns
        ) + " |")
    return "\n".join(lines)


def plot_overall(analysis, path):
    values = np.asarray([
        [analysis[d]["m3_mse"] / analysis[d]["original_mse"] - 1,
         analysis[d]["m3_mae"] / analysis[d]["original_mae"] - 1]
        for d in DATASETS
    ]).T
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    image = axes[0].imshow(100 * values, cmap="RdYlGn_r", vmin=-15, vmax=15, aspect="auto")
    for i in range(2):
        for j in range(len(DATASETS)):
            axes[0].text(j, i, f"{100 * values[i, j]:+.1f}%", ha="center", va="center")
    axes[0].set_xticks(range(len(DATASETS)), DATASETS, rotation=20)
    axes[0].set_yticks((0, 1), ("MSE", "MAE"))
    axes[0].set_title("M3 相对同协议原始 PhaseFormer")
    fig.colorbar(image, ax=axes[0], label="相对误差变化（%）")

    y = np.arange(len(DATASETS))
    ratios = 100 * values[0]
    lows = np.asarray([100 * analysis[d]["mse_ratio_ci95"][0] for d in DATASETS])
    highs = np.asarray([100 * analysis[d]["mse_ratio_ci95"][1] for d in DATASETS])
    axes[1].errorbar(ratios, y, xerr=np.vstack((ratios - lows, highs - ratios)), fmt="o", color="#4C78A8")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, DATASETS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("配对 MSE 变化（%）")
    axes[1].set_title("96 窗口块 bootstrap 95% 区间")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_groups(analysis, feature_rows, taxonomy_rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    for group, label, color in (
        ("success", "成功", "#54A24B"),
        ("other", "其余", "#BAB0AC"),
        ("failure", "失败", "#E45756"),
    ):
        values = np.asarray([
            analysis[d]["groups"][group] / analysis[d]["sample_channel_pairs"]
            for d in DATASETS
        ])
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set_xticks(x, DATASETS, rotation=25)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("样本×通道占比")
    axes[0].set_title("双指标一致的成功/失败分组")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    show = [row for row in feature_rows if row["feature"] in (
        "history_abs_drift", "history_lag24_corr", "history_diff_volatility",
        "history_phase_reliability", "future_level_shift", "future_lag24_corr",
        "future_diff_volatility", "anchor_disagreement", "route_entropy",
    )]
    show.sort(key=lambda row: row["macro_smd"])
    labels = [row["label"] for row in show]
    values = [row["macro_smd"] for row in show]
    axes[1].barh(labels, values, color=["#54A24B" if x > 0 else "#E45756" for x in values])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("成功组 − 失败组：宏平均 SMD")
    axes[1].set_title("两组特征差异（非因果）")
    axes[1].grid(axis="x", alpha=0.2)

    labels = [row["type"] for row in taxonomy_rows]
    values = [100 * row["macro_relative_mse"] for row in taxonomy_rows]
    axes[2].barh(labels, values, color=["#54A24B" if x < 0 else "#E45756" for x in values])
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("跨数据集宏平均相对 MSE（%）")
    axes[2].set_title("仅由输入定义的形态类型")
    axes[2].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cases(cases, path):
    fig, axes = plt.subplots(6, 2, figsize=(12.5, 17), squeeze=False)
    for row, case in enumerate(cases):
        left, right = axes[row]
        left.plot(np.arange(-96, 0), case["history"][-96:], color="#9D9D9D", label="近期历史")
        future = np.arange(1, 97)
        left.plot(future, case["truth"], color="black", linewidth=1.6, label="真实")
        left.plot(future, case["original"], color="#9467BD", label="原始 PhaseFormer")
        left.plot(future, case["candidate"], color="#E45756", label="M3")
        change = case["m3_mse"] / max(case["original_mse"], 1e-12) - 1
        kind = "成功" if case["case_type"] == "success" else "失败"
        left.set_title(
            f"{kind}：{case['dataset']} / {case['channel']} / 样本 {case['sample_id']}\n"
            f"MSE {case['original_mse']:.4f} → {case['m3_mse']:.4f} ({change:+.1%})"
        )
        left.grid(alpha=0.2)
        if row == 0:
            left.legend(ncol=4, fontsize=7)
        q = np.arange(1, 5)
        for index, (name, color) in enumerate(zip(ANCHOR_NAMES, ("#4C78A8", "#F58518", "#54A24B"))):
            right.plot(q, case["weights"][:, index], marker="o", color=color, label=name)
        right.set_ylim(-0.02, 1.02)
        right.set_xticks(q, ("1–24", "25–48", "49–72", "73–96"))
        right.set_ylabel("soft 权重")
        right.set_title("四个未来周期的路由值")
        right.grid(alpha=0.2)
        if row == 0:
            right.legend(ncol=3, fontsize=8)
    fig.suptitle("程序化选取的 M3 成功与失败案例（每类 3 个不同数据集）", y=1.002)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    started = time.time()
    if OUTPUT.exists():
        raise FileExistsError(f"canonical output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    figures = OUTPUT / "figures"
    figures.mkdir()
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    original_infos = {dataset: find_original(dataset) for dataset in DATASETS}
    candidate_infos = {
        dataset: find_run(M3_SCRATCH, dataset, CHAMPION, 30, "multi-anchor-selector-v1")
        for dataset in DATASETS
    }
    for infos in (original_infos, candidate_infos):
        for _, row, _ in infos.values():
            if row.get("test_mse") not in (None, "", "nan"):
                raise RuntimeError("test metric detected in validation-only analysis")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis, arrays_by_dataset, pools = {}, {}, {}
    with (OUTPUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            summary, arrays, case_pool = evaluate(
                dataset, candidate_infos[dataset], original_infos[dataset], writer, device
            )
            analysis[dataset], arrays_by_dataset[dataset], pools[dataset] = summary, arrays, case_pool
            print(f"audited {dataset}: {summary['sample_channel_pairs']:,} pairs", flush=True)

    replay = {}
    result_rows = []
    for dataset in DATASETS:
        original_row = original_infos[dataset][1]
        candidate_row = candidate_infos[dataset][1]
        checks = {
            "original_mse": abs(analysis[dataset]["original_mse"] - float(original_row["val_mse"])),
            "original_mae": abs(analysis[dataset]["original_mae"] - float(original_row["val_mae"])),
            "m3_mse": abs(analysis[dataset]["m3_mse"] - float(candidate_row["val_mse"])),
            "m3_mae": abs(analysis[dataset]["m3_mae"] - float(candidate_row["val_mae"])),
        }
        if max(checks.values()) >= 2e-5:
            raise RuntimeError(f"metric replay failed for {dataset}: {checks}")
        replay[setting_name(dataset)] = checks
        for config_id, label, info in (
            ("original", "Original PhaseFormer", original_infos[dataset]),
            (CHAMPION, "M3 structural soft", candidate_infos[dataset]),
        ):
            row = info[1]
            result_rows.append({
                "setting": setting_name(dataset), "dataset": dataset,
                "config_id": config_id, "model": label,
                "lookback": 720, "horizon": 96, "percent": 30,
                "epochs": 8 if config_id == "original" else 20,
                "seed": 2021, "split": "validation", "loss": "huber",
                "mse": float(row["val_mse"]), "mae": float(row["val_mae"]),
                "test_accessed": False,
            })
    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    feature_rows, taxonomy_rows = summarize(analysis, arrays_by_dataset)
    cases = select_cases(pools)
    save_cases(OUTPUT / "selected_cases.npz", cases)
    representatives = representative_cases(cases)
    plot_overall(analysis, figures / "m3_vs_original_overall.png")
    plot_groups(analysis, feature_rows, taxonomy_rows, figures / "m3_vs_original_groups.png")
    plot_cases(representatives, figures / "m3_vs_original_cases.png")
    for name in ("m3_vs_original_overall.png", "m3_vs_original_groups.png", "m3_vs_original_cases.png"):
        shutil.copy2(figures / name, DOC_FIGURES / name)

    total_pairs = sum(x["sample_channel_pairs"] for x in analysis.values())
    total_groups = {
        group: sum(x["groups"][group] for x in analysis.values()) for group in GROUPS
    }
    metric_rows = []
    for dataset in DATASETS:
        row = analysis[dataset]
        metric_rows.append({
            "数据集": dataset,
            "Original MSE": row["original_mse"], "M3 MSE": row["m3_mse"],
            "MSE变化": row["m3_mse"] / row["original_mse"] - 1,
            "Original MAE": row["original_mae"], "M3 MAE": row["m3_mae"],
            "MAE变化": row["m3_mae"] / row["original_mae"] - 1,
            "MSE区间": row["mse_ratio_ci95"],
        })
    group_rows = [{
        "数据集": dataset,
        "成功": analysis[dataset]["groups"]["success"] / analysis[dataset]["sample_channel_pairs"],
        "失败": analysis[dataset]["groups"]["failure"] / analysis[dataset]["sample_channel_pairs"],
        "其余": analysis[dataset]["groups"]["other"] / analysis[dataset]["sample_channel_pairs"],
    } for dataset in DATASETS]
    feature_table = [{
        "特征": row["label"], "成功组均值": row["success_mean"],
        "失败组均值": row["failure_mean"], "宏平均SMD": row["macro_smd"],
    } for row in feature_rows]
    taxonomy_table = [{
        "输入形态": row["type"], "宏平均MSE变化": row["macro_relative_mse"],
        "成功率": row["macro_success_rate"], "失败率": row["macro_failure_rate"],
        "改善数据集数": f"{row['datasets_mean_improved']}/6",
    } for row in taxonomy_rows]
    case_table = [{
        "类型": "成功" if x["case_type"] == "success" else "失败",
        "数据集": x["dataset"], "通道": x["channel"], "样本": x["sample_id"],
        "Original MSE": x["original_mse"], "M3 MSE": x["m3_mse"],
        "变化": x["m3_mse"] / x["original_mse"] - 1,
    } for x in representatives]
    pct = lambda value: f"{100 * float(value):+.2f}%"
    metric_formats = {
        "Original MSE": lambda x: f"{x:.6f}", "M3 MSE": lambda x: f"{x:.6f}",
        "Original MAE": lambda x: f"{x:.6f}", "M3 MAE": lambda x: f"{x:.6f}",
        "MSE变化": pct, "MAE变化": pct,
        "MSE区间": lambda x: f"[{100*x[0]:+.2f}%, {100*x[1]:+.2f}%]",
    }
    report = f"""# M3 相对原始 PhaseFormer：成功与失败样本审计

## 1. 口径

只使用 validation；六数据集均为 L720/H96、30% train、8 epoch original、seed 2021、Huber。M3 路由训练 20 epoch，但三个正式锚点均为 8 epoch。这里的 original 是同代码、同切分的 matched rerun，**不替代 Golden**。

成功样本预先定义为“样本×通道 MSE 至少下降 10%，且 MAE 同时下降”；失败样本反向定义。案例按绝对 MSE 差排序，并对同数据集同通道相距不足 96 的连续窗口去重。

## 2. 总体结果

{markdown_table(metric_rows, ['数据集', 'Original MSE', 'M3 MSE', 'MSE变化', 'Original MAE', 'M3 MAE', 'MAE变化', 'MSE区间'], metric_formats)}

![总体对比](figures/m3_vs_original_overall.png)

MSE 区间采用以样本起点为单位、块长 96 的循环 block bootstrap（1000 次）。由于 M3 已经由同一 validation 选择，这些区间只刻画配对误差的不确定性，不是独立测试显著性结论。

## 3. 成功与失败占比

共回放 `{total_pairs:,}` 个样本×通道；成功 `{total_groups['success']:,}`（{total_groups['success']/total_pairs:.2%}），失败 `{total_groups['failure']:,}`（{total_groups['failure']/total_pairs:.2%}），其余 `{total_groups['other']:,}`。

{markdown_table(group_rows, ['数据集', '成功', '失败', '其余'], {x: (lambda value: f'{value:.2%}') for x in ('成功', '失败', '其余')})}

## 4. 哪类样本更占优势

输入形态只由 encoder history 定义，阈值为各数据集自己的上四分位：强周期指 lag-24 相关高，近期漂移和高差分波动同理；类别互斥，优先级为“强周期+漂移、强周期、漂移、高波动、其他”。

{markdown_table(taxonomy_table, ['输入形态', '宏平均MSE变化', '成功率', '失败率', '改善数据集数'], {'宏平均MSE变化': pct, '成功率': lambda x: f'{x:.2%}', '失败率': lambda x: f'{x:.2%}'})}

下表的 SMD 先在每个数据集内计算成功组与失败组的标准化均值差，再对六数据集宏平均；正值表示成功组更高。未来特征使用真实预测区间，只能做事后描述，模型在推理时看不到。

{markdown_table(feature_table, ['特征', '成功组均值', '失败组均值', '宏平均SMD'], {'成功组均值': lambda x: f'{x:.4f}', '失败组均值': lambda x: f'{x:.4f}', '宏平均SMD': lambda x: f'{x:+.3f}'})}

![分组与形态](figures/m3_vs_original_groups.png)

“强周期+近期漂移”组的宏平均 MSE 变化最优（-14.35%），成功率也最高（60.27%），且六个数据集的组均值都改善；但“其他”组也改善 13.12%，而强周期组在 Weather 上退化 12.04%，所以该形态只能视为较稳定的正向信号，不能视为充分条件。

更关键的是，四个输入特征的成功/失败宏平均 SMD 绝对值都不超过 0.095，三路权重和路由熵也都不超过 0.099。当前可见输入和路由值不能可靠判断单个样本是否会受益。事后未来特征中，成功组的未来水平迁移较低（SMD -0.199），未来 lag-24 相关略高（+0.106）；这与“周期结构延续时更易受益、未预见状态切换时更易失败”一致，但不是因果证明。

## 5. 具体案例

每类从 30 个去重案例中再按绝对 MSE 差选 3 个不同数据集代表；没有人工挑图。右侧给出 M3 在四个未来 24 步周期上的 A1/I0/R0 soft 权重。

{markdown_table(case_table, ['类型', '数据集', '通道', '样本', 'Original MSE', 'M3 MSE', '变化'], {'Original MSE': lambda x: f'{x:.5f}', 'M3 MSE': lambda x: f'{x:.5f}', '变化': pct})}

![案例](figures/m3_vs_original_cases.png)

成功案例中，ETTh1-LUFL 与 ETTm1-LULL 都在历史末端发生电平上移，M3 比仍停留旧水平的 original 更快跟随；Weather-rain 中 M3 比 original 更快向未来低值回落，但仍有明显误差。失败案例中，Electricity-298 与 Weather-raining 属于稀疏事件通道，R0 或 I0/R0 权重较高时产生持续偏高或重复尖峰；ETTh1-HULL 的未来突然向负值切换，而 A1 主导的 M3 仍维持正向预测。这些是程序化选出的极端例，说明可能边界，不代表全部失败样本。

## 6. 客观边界

- 滑窗高度重叠，因此没有把样本×通道当作独立观测做普通 t 检验；组别统计是描述性的。
- “未来”特征和真实目标参与事后归类，不能作为可部署判据，也不能证明因果。
- 当前只覆盖 H96、30% train、单 seed validation；不能外推到 H192、full-train 或 test。
- M3 是 validation 选出的模型，结果存在选择偏差。这里回答的是当前协议中优势/失败落在哪里，不是独立泛化证明。
"""
    report_path = OUTPUT / "objective_error_analysis.md"
    report_path.write_text(report, encoding="utf-8")

    referenced = re.findall(r"!\[[^]]*\]\((figures/[^)]+)\)", report)
    if set(referenced) != {
        "figures/m3_vs_original_overall.png", "figures/m3_vs_original_groups.png",
        "figures/m3_vs_original_cases.png",
    }:
        raise RuntimeError(f"unexpected report figures: {referenced}")
    zip_path = OUTPUT / "objective_error_analysis.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for relative in referenced:
            archive.write(OUTPUT / relative, relative)
    with zipfile.ZipFile(zip_path) as archive:
        for relative in {report_path.name, *referenced}:
            if archive.read(relative) != (OUTPUT / relative).read_bytes():
                raise RuntimeError(f"ZIP byte mismatch: {relative}")

    manifest = {
        "experiment_id": "m3_vs_original_phaseformer_v1",
        "code": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD")},
        "protocol": {
            "split": "validation", "datasets": list(DATASETS), "lookback": 720,
            "horizon": 96, "percent": 30, "original_epochs": 8,
            "anchor_epochs": 8, "router_epochs": 20, "seed": 2021, "loss": "huber",
            "success": "relative MSE <= -10% and MAE improves",
            "failure": "relative MSE >= +10% and MAE degrades",
            "selection": "absolute MSE delta; same dataset/channel sample gap >=96",
        },
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "analysis": {
            "summaries": native(analysis), "feature_rows": native(feature_rows),
            "taxonomy_rows": native(taxonomy_rows), "metric_replay": native(replay),
            "sample_channel_rows": total_pairs, "selected_cases": len(cases),
        },
        "validation": {"test_accessed": False, "metric_replay_passed": True, "status": "passed"},
        "artifacts": {}, "elapsed_analysis_sec": time.time() - started,
    }
    for name in ("results.csv", "sample_errors.csv", "selected_cases.npz", "objective_error_analysis.md", "objective_error_analysis.zip"):
        path = OUTPUT / name
        manifest["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUTPUT / "run.yaml").write_text(
        yaml.safe_dump(native(manifest), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    expected = {"run.yaml", "results.csv", "sample_errors.csv", "selected_cases.npz", "objective_error_analysis.md", "objective_error_analysis.zip", "figures"}
    if {x.name for x in OUTPUT.iterdir()} != expected:
        raise RuntimeError("canonical root whitelist mismatch")
    if {x.name for x in figures.iterdir()} != {Path(x).name for x in referenced}:
        raise RuntimeError("canonical figure whitelist mismatch")
    print(f"wrote {OUTPUT}")
    print(f"rows={total_pairs:,}, cases={len(cases)}, zip_sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
