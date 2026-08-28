#!/usr/bin/env python3
"""Build the strict validation-only audit bundle for Safe-Regret TriAxis v1."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import platform
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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from analyze_triaxis_experiment import build_model_and_loader, variable_names

SCRATCH = REPO / "research_runs" / "safe_regret_triaxis_v1_scratch"
RUNS = SCRATCH / "runs"
CACHE = SCRATCH / "audit_cache"
OUTPUT = REPO / "research_runs" / "safe_regret_triaxis_v1"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
HORIZONS = (96, 192)
REFERENCES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_rolling_features",
)
CANDIDATES = (
    "safe_triaxis_anchor",
    "safe_triaxis_regret",
    "safe_triaxis_guarded",
    "safe_triaxis_monotone",
)
CHAMPION = "safe_triaxis_guarded"
LABELS = {
    "gold_combo_reliability_s2": "A1 RCRF+NLinear",
    "rcrf_icpt_none": "I0 RCRF+ICPT",
    "triaxis_rolling_features": "R0 Rolling TriAxis",
    "safe_triaxis_anchor": "S0 安全锚定",
    "safe_triaxis_regret": "S1 +后悔路由",
    "safe_triaxis_guarded": "S2 +均值/CVaR保护",
    "safe_triaxis_monotone": "S3 +单调距离先验",
}
EXPERTS = ("phase", "trajectory", "cycle")
EXPERT_ZH = ("相位", "轨迹", "周期间")
GROUPS = ("significant_improvement", "comparable", "significant_regression")
SAMPLE_FIELDS = (
    "setting", "baseline_config_id", "candidate_config_id", "sample_id",
    "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse",
    "relative_delta_mse", "baseline_mae", "candidate_mae", "delta_mae",
    "anchor_mse", "group", "recent_drift", "lag24_correlation",
    "phase_reliability", "shape_innovation", "accept_mean",
    "noop_probability_mean", "anchor_weight_mean", "phase_weight_mean",
    "trajectory_weight_mean", "cycle_weight_mean", "rolling_risk_margin_mean",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def setting_name(dataset, horizon):
    return f"{dataset}-L720-H{horizon}-s2021-val30"


def find_run(dataset, horizon, mode):
    matches = []
    for path in RUNS.glob("*/metrics.csv"):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            continue
        row = frame.iloc[0]
        if (
            row.dataset == dataset
            and int(row.horizon) == horizon
            and row.mechanism == mode
        ):
            matches.append((path.parent, row.to_dict()))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one run for {dataset}/H{horizon}/{mode}, got {len(matches)}"
        )
    run_dir, row = matches[0]
    return run_dir, row, json.loads((run_dir / "config.json").read_text())


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def fmt(value):
    return f"{float(value):.8g}" if isinstance(value, (float, np.floating)) else value


def channel_features(history, period=24, eps=1e-6):
    scale = history.std(dim=1, unbiased=False).clamp_min(eps)
    drift = (
        history[:, -period:].mean(dim=1)
        - history[:, -2 * period : -period].mean(dim=1)
    ).abs() / scale
    left = history[:, period:] - history[:, period:].mean(dim=1, keepdim=True)
    right = history[:, :-period] - history[:, :-period].mean(dim=1, keepdim=True)
    lag24 = (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
        + eps
    )
    return drift, lag24


def push_case(heap, score, serial, case, limit=32):
    item = (float(score), int(serial), case)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def top_indices(score, positive=False, limit=8):
    flat = score.flatten()
    if positive:
        indexes = (torch.isfinite(flat) & (flat > 0)).nonzero().flatten()
        if not len(indexes):
            return []
        values = flat[indexes]
    else:
        indexes = torch.arange(flat.numel(), device=flat.device)
        values = flat
    top = torch.topk(values, min(limit, values.numel()))
    return [(float(value), int(indexes[index])) for value, index in zip(top.values, top.indices)]


def make_case(
    category, score, dataset, horizon, sample_id, channel_id, channel,
    time_range, history, truth, baseline, candidate, anchor, experts, weights,
    action_probabilities, accept, risks, risk_std, structural,
):
    true_np = truth.detach().cpu().numpy()
    baseline_np = baseline.detach().cpu().numpy()
    candidate_np = candidate.detach().cpu().numpy()
    return {
        "case_type": category,
        "score": float(score),
        "setting": setting_name(dataset, horizon),
        "dataset": dataset,
        "horizon": horizon,
        "sample_id": int(sample_id),
        "channel_id": int(channel_id),
        "channel": channel,
        "time_range": time_range,
        "history": history.detach().cpu().numpy(),
        "truth": true_np,
        "baseline": baseline_np,
        "candidate": candidate_np,
        "anchor": anchor.detach().cpu().numpy(),
        "experts": experts.detach().cpu().numpy(),
        "weights": weights.detach().cpu().numpy(),
        "action_probabilities": action_probabilities.detach().cpu().numpy(),
        "accept": accept.detach().cpu().numpy(),
        "risks": risks.detach().cpu().numpy(),
        "risk_std": risk_std.detach().cpu().numpy(),
        "structural": structural.detach().cpu().numpy(),
        "baseline_mse": float(np.mean((baseline_np - true_np) ** 2)),
        "candidate_mse": float(np.mean((candidate_np - true_np) ** 2)),
    }


def evaluate_setting(dataset_name, horizon, baseline_info, candidate_info, writer, device):
    batch_size = 8 if dataset_name == "Electricity" else (
        32 if dataset_name == "Weather" else 128
    )
    baseline_model, dataset, loader = build_model_and_loader(
        *baseline_info, batch_size
    )
    candidate_model, candidate_dataset, _ = build_model_and_loader(
        *candidate_info, batch_size
    )
    if len(dataset) != len(candidate_dataset):
        raise RuntimeError("validation split mismatch")
    baseline_model = baseline_model.to(device).eval()
    candidate_model = candidate_model.to(device).eval()
    names = variable_names(candidate_info[2])
    timestamps = getattr(dataset, "timestamps", None)
    totals = defaultdict(float)
    group_counts = defaultdict(int)
    group_sums = {group: defaultdict(float) for group in GROUPS}
    group_feature_names = (
        "recent_drift", "lag24_correlation", "phase_reliability",
        "shape_innovation", "accept_mean", "noop_probability_mean",
        "rolling_risk_margin_mean",
    )
    relative_parts, accept_parts = [], []
    case_heaps = {key: [] for key in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    )}
    serial = 0
    offset = 0

    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y, x_mark, y_mark = [value.to(device) for value in batch]
            batch_x, batch_y = batch_x.float(), batch_y.float()
            truth = batch_y[:, -horizon:, :]
            baseline_dec = baseline_model._build_decoder_input(batch_y)
            baseline, _, _ = baseline_model(
                batch_x, x_mark.float(), baseline_dec, y_mark.float()
            )
            candidate_dec = candidate_model._build_decoder_input(batch_y)
            candidate, _, _ = candidate_model(
                batch_x, x_mark.float(), candidate_dec, y_mark.float()
            )
            anchor = candidate_model.safe_triaxis_anchor_output
            experts = torch.stack(
                candidate_model.safe_triaxis_expert_outputs, dim=-1
            )
            weights = candidate_model.safe_triaxis_weights
            router = candidate_model.safe_triaxis_router
            action_probabilities = torch.softmax(
                router.last_cycle_action_logits, dim=-1
            )
            accept = router.last_cycle_accept.squeeze(-1)
            risks = router.last_risks
            risk_std = router.last_risk_std
            structural = router.last_structural

            for prefix, prediction in (
                ("baseline", baseline), ("candidate", candidate), ("anchor", anchor)
            ):
                error = prediction - truth
                totals[f"{prefix}_sq"] += error.square().sum().item()
                totals[f"{prefix}_abs"] += error.abs().sum().item()
            totals["count"] += truth.numel()

            baseline_mse = (baseline - truth).square().mean(dim=1)
            candidate_mse = (candidate - truth).square().mean(dim=1)
            anchor_mse = (anchor - truth).square().mean(dim=1)
            baseline_mae = (baseline - truth).abs().mean(dim=1)
            candidate_mae = (candidate - truth).abs().mean(dim=1)
            relative = (candidate_mse - baseline_mse) / baseline_mse.clamp_min(1e-8)
            drift, lag24 = channel_features(batch_x)
            phase_reliability = structural[..., 0].mean(dim=(2, 3))
            shape_innovation = structural[..., 2].mean(dim=(2, 3))
            accept_mean = accept.mean(dim=2)
            noop_mean = action_probabilities[..., 0].mean(dim=2)
            weight_mean = weights.mean(dim=1)
            mean_risk = risks.mean(dim=3)
            ordered = mean_risk.sort(dim=-1).values
            risk_margin = (ordered[..., 1] - ordered[..., 0]).mean(dim=2)
            relative_parts.append(relative.cpu().numpy().reshape(-1).astype(np.float32))
            accept_parts.append(accept_mean.cpu().numpy().reshape(-1).astype(np.float32))

            features = {
                "recent_drift": drift,
                "lag24_correlation": lag24,
                "phase_reliability": phase_reliability,
                "shape_innovation": shape_innovation,
                "accept_mean": accept_mean,
                "noop_probability_mean": noop_mean,
                "rolling_risk_margin_mean": risk_margin,
            }
            masks = {
                "significant_improvement": relative <= -0.10,
                "significant_regression": relative >= 0.10,
                "comparable": (relative > -0.10) & (relative < 0.10),
            }
            for group, mask in masks.items():
                n = int(mask.sum())
                group_counts[group] += n
                for key in group_feature_names:
                    group_sums[group][key] += features[key][mask].sum().item()

            B, _, C = truth.shape
            batch_rows = []
            for b in range(B):
                start = offset + b + batch_x.shape[1]
                if timestamps is not None and start + horizon - 1 < len(timestamps):
                    time_range = f"{timestamps[start]}--{timestamps[start + horizon - 1]}"
                else:
                    time_range = f"index:{start}--{start + horizon - 1}"
                for c in range(C):
                    rel = float(relative[b, c])
                    group = "significant_improvement" if rel <= -0.10 else (
                        "significant_regression" if rel >= 0.10 else "comparable"
                    )
                    row = {
                        "setting": setting_name(dataset_name, horizon),
                        "baseline_config_id": baseline_info[1]["mechanism"],
                        "candidate_config_id": CHAMPION,
                        "sample_id": offset + b,
                        "channel": names[c] if c < len(names) else str(c),
                        "time_range": time_range,
                        "baseline_mse": float(baseline_mse[b, c]),
                        "candidate_mse": float(candidate_mse[b, c]),
                        "delta_mse": float(candidate_mse[b, c] - baseline_mse[b, c]),
                        "relative_delta_mse": rel,
                        "baseline_mae": float(baseline_mae[b, c]),
                        "candidate_mae": float(candidate_mae[b, c]),
                        "delta_mae": float(candidate_mae[b, c] - baseline_mae[b, c]),
                        "anchor_mse": float(anchor_mse[b, c]),
                        "group": group,
                        **{key: float(value[b, c]) for key, value in features.items()},
                        "anchor_weight_mean": float(weight_mean[b, c, 0]),
                        "phase_weight_mean": float(weight_mean[b, c, 1]),
                        "trajectory_weight_mean": float(weight_mean[b, c, 2]),
                        "cycle_weight_mean": float(weight_mean[b, c, 3]),
                    }
                    batch_rows.append({key: fmt(row[key]) for key in SAMPLE_FIELDS})
            writer.writerows(batch_rows)

            scores = {
                "baseline_high_error": baseline_mse,
                "candidate_regression": relative,
                "candidate_improvement": -relative,
            }
            for category, score_tensor in scores.items():
                for score, flat_index in top_indices(
                    score_tensor, positive=category != "baseline_high_error"
                ):
                    b, c = divmod(flat_index, C)
                    start = offset + b + batch_x.shape[1]
                    time_range = (
                        f"{timestamps[start]}--{timestamps[start + horizon - 1]}"
                        if timestamps is not None and start + horizon - 1 < len(timestamps)
                        else f"index:{start}--{start + horizon - 1}"
                    )
                    case = make_case(
                        category, score, dataset_name, horizon, offset + b, c,
                        names[c] if c < len(names) else str(c), time_range,
                        batch_x[b, :, c], truth[b, :, c], baseline[b, :, c],
                        candidate[b, :, c], anchor[b, :, c],
                        experts[b, :, c].T, weights[b, :, c],
                        action_probabilities[b, c], accept[b, c], risks[b, c],
                        risk_std[b, c], structural[b, c],
                    )
                    push_case(case_heaps[category], score, serial, case)
                    serial += 1
            offset += B

    count = totals["count"]
    summary = {
        "setting": setting_name(dataset_name, horizon),
        "dataset": dataset_name,
        "horizon": horizon,
        "validation_samples": len(dataset),
        "sample_channel_pairs": int(sum(group_counts.values())),
        "baseline_config": baseline_info[1]["mechanism"],
        **{
            f"{prefix}_{metric}": totals[f"{prefix}_{suffix}"] / count
            for prefix in ("baseline", "candidate", "anchor")
            for metric, suffix in (("mse", "sq"), ("mae", "abs"))
        },
        "groups": dict(group_counts),
        "group_feature_means": {
            group: {
                key: group_sums[group][key] / max(group_counts[group], 1)
                for key in group_feature_names
            }
            for group in GROUPS
        },
        "relative_delta_quantiles": {
            str(q): float(np.quantile(np.concatenate(relative_parts), q))
            for q in (0.1, 0.5, 0.9)
        },
        "accept_nonzero_fraction": float(
            (np.concatenate(accept_parts) > 1e-6).mean()
        ),
        "accept_mean": float(np.concatenate(accept_parts).mean()),
    }
    arrays = {
        "relative_delta": np.concatenate(relative_parts),
        "accept": np.concatenate(accept_parts),
    }
    pools = {
        key: [item[2] for item in sorted(heap, reverse=True)]
        for key, heap in case_heaps.items()
    }
    del baseline_model, candidate_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, arrays, pools


def choose_baseline(infos, dataset, horizon):
    rows = [infos[(dataset, horizon, mode)][1] for mode in REFERENCES]
    min_mse = min(float(row["val_mse"]) for row in rows)
    min_mae = min(float(row["val_mae"]) for row in rows)
    return min(
        REFERENCES,
        key=lambda mode: (
            float(infos[(dataset, horizon, mode)][1]["val_mse"]) / min_mse
            + float(infos[(dataset, horizon, mode)][1]["val_mae"]) / min_mae
        ),
    )


def is_duplicate(case, selected):
    return any(
        old["setting"] == case["setting"]
        and old["channel_id"] == case["channel_id"]
        and abs(old["sample_id"] - case["sample_id"]) < case["horizon"]
        for old in selected
    )


def select_cases(pools):
    selected = []
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        options = [case for pool in pools.values() for case in pool[category]]
        options.sort(key=lambda item: item["score"], reverse=True)
        chosen, used_settings = [], set()
        for distinct in (True, False):
            for case in options:
                if any(old is case for old in chosen) or is_duplicate(
                    case, selected + chosen
                ):
                    continue
                if distinct and case["setting"] in used_settings:
                    continue
                chosen.append(case)
                used_settings.add(case["setting"])
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                break
        selected.extend(chosen)
    return selected


def pad(array, shape):
    output = np.full(shape, np.nan, dtype=np.float32)
    slices = tuple(slice(0, size) for size in array.shape)
    output[slices] = array
    return output


def save_cases(path, cases):
    np.savez_compressed(
        path,
        setting=np.asarray([case["setting"] for case in cases]),
        case_type=np.asarray([case["case_type"] for case in cases]),
        dataset=np.asarray([case["dataset"] for case in cases]),
        horizon=np.asarray([case["horizon"] for case in cases]),
        sample_id=np.asarray([case["sample_id"] for case in cases]),
        channel_id=np.asarray([case["channel_id"] for case in cases]),
        channel=np.asarray([case["channel"] for case in cases]),
        time_range=np.asarray([case["time_range"] for case in cases]),
        history=np.stack([case["history"] for case in cases]),
        truth=np.stack([pad(case["truth"], (192,)) for case in cases]),
        baseline_prediction=np.stack([pad(case["baseline"], (192,)) for case in cases]),
        candidate_prediction=np.stack([pad(case["candidate"], (192,)) for case in cases]),
        anchor_prediction=np.stack([pad(case["anchor"], (192,)) for case in cases]),
        expert_predictions=np.stack([pad(case["experts"], (3, 192)) for case in cases]),
        route_weights=np.stack([pad(case["weights"], (192, 4)) for case in cases]),
        action_probabilities=np.stack([
            pad(case["action_probabilities"], (8, 4)) for case in cases
        ]),
        cycle_accept=np.stack([pad(case["accept"], (8,)) for case in cases]),
        rolling_risks=np.stack([pad(case["risks"], (8, 24, 3)) for case in cases]),
        rolling_risk_std=np.stack([pad(case["risk_std"], (8, 24, 3)) for case in cases]),
        structural_features=np.stack([
            pad(case["structural"], (8, 24, 3)) for case in cases
        ]),
    )


def markdown_table(rows, columns, formatter=None):
    formatter = formatter or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            formatter.get(column, str)(row[column]) for column in columns
        ) + " |")
    return "\n".join(lines)


def plot_metric_ratios(setting_rows, output):
    labels = [f"{row['dataset']}\nH{row['horizon']}" for row in setting_rows]
    data = np.asarray([[row["mse_ratio"], row["mae_ratio"]] for row in setting_rows]).T
    fig, ax = plt.subplots(figsize=(13, 4.3))
    image = ax.imshow(100 * (data - 1), cmap="RdYlGn_r", vmin=-2, vmax=5, aspect="auto")
    for i in range(2):
        for j in range(len(labels)):
            ax.text(j, i, f"{100 * (data[i, j] - 1):+.2f}%", ha="center", va="center", fontsize=8)
    ax.set_yticks((0, 1), ("MSE", "MAE"))
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_title("S2 相对每个 setting 最强原始模型的验证误差变化（负值更好）")
    fig.colorbar(image, ax=ax, label="相对变化（%）")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_groups(analysis, output):
    labels, improved, comparable, regressed = [], [], [], []
    for key in sorted(analysis, key=lambda item: (DATASETS.index(item[0]), item[1])):
        summary = analysis[key]
        total = summary["sample_channel_pairs"]
        labels.append(f"{key[0]}\nH{key[1]}")
        improved.append(summary["groups"].get("significant_improvement", 0) / total)
        comparable.append(summary["groups"].get("comparable", 0) / total)
        regressed.append(summary["groups"].get("significant_regression", 0) / total)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, improved, color="#54A24B", label="显著改善（≤-10%）")
    ax.bar(x, comparable, bottom=improved, color="#BAB0AC", label="相近")
    ax.bar(x, regressed, bottom=np.asarray(improved) + comparable, color="#E45756", label="显著退化（≥+10%）")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("样本×通道占比")
    ax.set_title("S2 相对各 setting 最强原始模型的样本级误差分组")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_acceptance(analysis, arrays, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for horizon, color in ((96, "#4C78A8"), (192, "#F58518")):
            accept = arrays[(dataset, horizon)]["accept"]
            relative = arrays[(dataset, horizon)]["relative_delta"]
            edges = np.quantile(accept, np.linspace(0, 1, 11))
            bins = np.searchsorted(edges[1:-1], accept, side="right")
            means = [np.mean(relative[bins == index]) for index in range(10)]
            ax.plot(np.arange(1, 11), 100 * np.asarray(means), marker="o", color=color, label=f"H{horizon}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_ylabel("相对误差变化（%）")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("接纳门控十分位（低→高）")
    axes[0, 0].legend()
    fig.suptitle("接纳强度升高是否对应真实改善")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_cases(cases, category, output):
    subset = [case for case in cases if case["case_type"] == category]
    titles = {
        "baseline_high_error": "原始模型高误差代表样本",
        "candidate_regression": "S2 显著退化代表样本",
        "candidate_improvement": "S2 显著改善代表样本",
    }
    fig, axes = plt.subplots(len(subset), 2, figsize=(13, 3.5 * len(subset)), squeeze=False)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for row, case in enumerate(subset):
        left, right = axes[row]
        horizon = case["horizon"]
        left.plot(np.arange(-96, 0), case["history"][-96:], color="#999999", label="近期历史")
        future = np.arange(1, horizon + 1)
        left.plot(future, case["truth"], color="black", linewidth=1.8, label="真实")
        left.plot(future, case["baseline"], color="#9467BD", label="最强原始")
        left.plot(future, case["candidate"], color="#E45756", label="S2")
        left.plot(future, case["anchor"], color="#777777", linestyle=":", label="A1锚点")
        left.axvline(0, color="#888888", linewidth=0.7)
        left.set_title(
            f"{case['dataset']} H{horizon} / {case['channel']} / 样本{case['sample_id']}\n"
            f"MSE 原始={case['baseline_mse']:.4f}，S2={case['candidate_mse']:.4f}"
        )
        left.grid(alpha=0.18)
        if row == 0:
            left.legend(ncol=5, fontsize=7)

        q = np.arange(1, len(case["accept"]) + 1)
        right.plot(q, case["accept"], color="#E45756", marker="o", linewidth=2, label="实际接纳")
        probabilities = case["action_probabilities"]
        right.plot(q, probabilities[:, 0], color="black", linestyle="--", label="no-op概率")
        for expert_id, (label, color) in enumerate(zip(EXPERT_ZH, colors), 1):
            right.plot(q, probabilities[:, expert_id], color=color, marker=".", label=f"{label}动作概率")
        right.set_ylim(0, 1)
        right.set_xticks(q)
        right.set_xlabel("未来周期编号")
        right.set_title("周期级接纳与动作概率")
        right.grid(alpha=0.18)
        if row == 0:
            right.legend(ncol=2, fontsize=7)
    fig.suptitle(titles[category], fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started = time.time()
    if OUTPUT.exists():
        raise FileExistsError(f"canonical output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    figures = OUTPUT / "figures"
    figures.mkdir()
    CACHE.mkdir(exist_ok=True)

    infos, raw_rows = {}, []
    for horizon in HORIZONS:
        modes = REFERENCES + (CANDIDATES if horizon == 96 else (CHAMPION,))
        for dataset in DATASETS:
            for mode in modes:
                info = find_run(dataset, horizon, mode)
                infos[(dataset, horizon, mode)] = info
                raw_rows.append(dict(info[1]))
    if len(raw_rows) != 66:
        raise RuntimeError(f"expected 66 completed runs, got {len(raw_rows)}")
    if any(pd.notna(row.get("test_mse")) for row in raw_rows):
        raise RuntimeError("test metric detected in validation-only experiment")

    baselines = {
        (dataset, horizon): choose_baseline(infos, dataset, horizon)
        for dataset in DATASETS for horizon in HORIZONS
    }
    result_rows = []
    for row in raw_rows:
        dataset, horizon = row["dataset"], int(row["horizon"])
        reference_rows = [infos[(dataset, horizon, mode)][1] for mode in REFERENCES]
        mse_envelope = min(float(item["val_mse"]) for item in reference_rows)
        mae_envelope = min(float(item["val_mae"]) for item in reference_rows)
        result_rows.append({
            "setting": setting_name(dataset, horizon),
            "config_id": row["mechanism"],
            "model": LABELS[row["mechanism"]],
            "dataset": dataset,
            "horizon": horizon,
            "seed": int(row["seed"]),
            "split": "validation",
            "mse": float(row["val_mse"]),
            "mae": float(row["val_mae"]),
            "mse_ratio_to_original_envelope": float(row["val_mse"]) / mse_envelope,
            "mae_ratio_to_original_envelope": float(row["val_mae"]) / mae_envelope,
            "selected_stage_a": row["mechanism"] == CHAMPION,
            "test_accessed": False,
            "parameter_count": int(row["parameter_count"]),
            "trainable_parameter_count": int(row["trainable_parameter_count"]),
            "anchor_identity_max_abs": row["anchor_identity_max_abs"],
            "elapsed_sec": float(row["elapsed_sec"]),
            "checkpoint": row["checkpoint"],
        })
    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis, arrays, pools = {}, {}, {}
    with (OUTPUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            for horizon in HORIZONS:
                key = (dataset, horizon)
                cache_state = CACHE / f"{dataset}_h{horizon}.pt"
                cache_rows = CACHE / f"{dataset}_h{horizon}.csv"
                if cache_state.exists() and cache_rows.exists():
                    state = torch.load(cache_state, map_location="cpu", weights_only=False)
                    summary, plot_arrays, case_pool = (
                        state["summary"], state["arrays"], state["pools"]
                    )
                else:
                    baseline_mode = baselines[key]
                    with cache_rows.open("w", newline="") as row_handle:
                        row_writer = csv.DictWriter(row_handle, fieldnames=SAMPLE_FIELDS)
                        row_writer.writeheader()
                        summary, plot_arrays, case_pool = evaluate_setting(
                            dataset, horizon, infos[(dataset, horizon, baseline_mode)],
                            infos[(dataset, horizon, CHAMPION)], row_writer, device,
                        )
                    torch.save(
                        {"summary": summary, "arrays": plot_arrays, "pools": case_pool},
                        cache_state,
                    )
                with cache_rows.open(newline="") as row_handle:
                    writer.writerows(csv.DictReader(row_handle))
                analysis[key], arrays[key], pools[key] = summary, plot_arrays, case_pool
                print(f"audited {summary['setting']}: {summary['sample_channel_pairs']} pairs", flush=True)

    replay = {}
    for key, summary in analysis.items():
        dataset, horizon = key
        baseline_row = infos[(dataset, horizon, baselines[key])][1]
        candidate_row = infos[(dataset, horizon, CHAMPION)][1]
        checks = {
            "baseline_mse": abs(summary["baseline_mse"] - float(baseline_row["val_mse"])),
            "baseline_mae": abs(summary["baseline_mae"] - float(baseline_row["val_mae"])),
            "candidate_mse": abs(summary["candidate_mse"] - float(candidate_row["val_mse"])),
            "candidate_mae": abs(summary["candidate_mae"] - float(candidate_row["val_mae"])),
        }
        replay[summary["setting"]] = checks
        if max(checks.values()) >= 1e-5:
            raise RuntimeError(f"metric replay failed: {summary['setting']} {checks}")

    cases = select_cases(pools)
    if len(cases) != 9:
        raise RuntimeError(f"expected 9 selected cases, got {len(cases)}")
    save_cases(OUTPUT / "selected_cases.npz", cases)

    setting_rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            baseline = infos[(dataset, horizon, baselines[key])][1]
            candidate = infos[(dataset, horizon, CHAMPION)][1]
            setting_rows.append({
                "dataset": dataset,
                "horizon": horizon,
                "baseline": LABELS[baselines[key]],
                "baseline_mse": float(baseline["val_mse"]),
                "baseline_mae": float(baseline["val_mae"]),
                "candidate_mse": float(candidate["val_mse"]),
                "candidate_mae": float(candidate["val_mae"]),
                "mse_ratio": float(candidate["val_mse"]) / float(baseline["val_mse"]),
                "mae_ratio": float(candidate["val_mae"]) / float(baseline["val_mae"]),
            })
    plot_metric_ratios(setting_rows, figures / "all__metric_ratios.png")
    plot_groups(analysis, figures / "all__sample_groups.png")
    plot_acceptance(analysis, arrays, figures / "all__acceptance_calibration.png")
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        plot_cases(cases, category, figures / f"all__cases_{category}.png")

    stage_a = json.loads((SCRATCH / "stage_a_decision.json").read_text())
    final = json.loads((SCRATCH / "final_decision.json").read_text())
    a1_ratios = []
    for row in setting_rows:
        a1 = infos[(row["dataset"], row["horizon"], REFERENCES[0])][1]
        a1_ratios.extend((
            row["candidate_mse"] / float(a1["val_mse"]),
            row["candidate_mae"] / float(a1["val_mae"]),
        ))
    total_pairs = sum(item["sample_channel_pairs"] for item in analysis.values())
    total_groups = {
        group: sum(item["groups"].get(group, 0) for item in analysis.values())
        for group in GROUPS
    }
    aggregate_feature = {}
    for group in GROUPS:
        count = total_groups[group]
        aggregate_feature[group] = {
            feature: sum(
                item["group_feature_means"][group][feature]
                * item["groups"].get(group, 0)
                for item in analysis.values()
            ) / max(count, 1)
            for feature in next(iter(analysis.values()))["group_feature_means"][group]
        }

    ranking_rows = [{
        "候选": LABELS[item["candidate"]],
        "H96宏比值": item["macro_ratio"],
        "最差比值": item["worst_ratio"],
        "严格通过": "是" if item["strict_pass"] else "否",
    } for item in stage_a["ranking"]]
    table_rows = [{
        "数据集": row["dataset"], "H": row["horizon"],
        "最强原始": row["baseline"],
        "原始MSE": row["baseline_mse"], "S2 MSE": row["candidate_mse"],
        "MSE变化": row["mse_ratio"] - 1,
        "原始MAE": row["baseline_mae"], "S2 MAE": row["candidate_mae"],
        "MAE变化": row["mae_ratio"] - 1,
    } for row in setting_rows]
    group_rows = [{
        "分组": {"significant_improvement": "显著改善", "comparable": "相近", "significant_regression": "显著退化"}[group],
        "数量": total_groups[group], "占比": total_groups[group] / total_pairs,
        "漂移": aggregate_feature[group]["recent_drift"],
        "lag24": aggregate_feature[group]["lag24_correlation"],
        "接纳": aggregate_feature[group]["accept_mean"],
        "no-op概率": aggregate_feature[group]["noop_probability_mean"],
    } for group in GROUPS]
    case_rows = [{
        "类型": {"baseline_high_error": "原始高误差", "candidate_regression": "S2退化", "candidate_improvement": "S2改善"}[case["case_type"]],
        "setting": case["setting"], "通道": case["channel"],
        "样本": case["sample_id"], "原始MSE": case["baseline_mse"],
        "S2 MSE": case["candidate_mse"],
    } for case in cases]
    percent = lambda value: f"{100 * float(value):+.2f}%"
    report = f"""# Safe-Regret TriAxis v1：广泛验证与样本级误差分析

## 直白结论

这版修正**没有达到“集成后稳定超过三个原始模型”的目标**。它确实把完整 A1 变成了可精确退回的锚点，并在 24 个指标单元上相对 A1 的宏平均比值做到 `{np.mean(a1_ratios):.6f}`（平均 `{percent(np.mean(a1_ratios) - 1)}`）；但相对每个 setting 的最强原始模型，宏平均比值是 `{final['result']['macro_ratio']:.6f}`（平均退化 `{percent(final['result']['macro_ratio'] - 1)}`），最差退化 `{percent(final['result']['worst_ratio'] - 1)}`，所以严格门失败，未进入 test。

根因不是“安全门没有工作”，而是**安全门只保证回退 A1，无法回退 I0/R0**。Weather、ETTh1 等 setting 的原始上界来自 R0 或 I0；S2 即使优于自己的 A1 锚点，仍可能明显落后于真正的原始最优。下一版必须把“最强原始模型集合”本身纳入可回退集合，或先蒸馏成强统一锚点，不能继续只在 A1 周围做局部修正。

## 实验边界与审计

- 代码提交：`{git('rev-parse', 'HEAD')}`；工作树仅包含被忽略的实验产物。
- 训练协议：L720、P24、30% train、8 epoch、Huber、seed 2021。
- 数据与 setting：ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity × H96/H192。
- Stage A：H96 上 3 个原始参考 + 4 个候选，共 42 次。
- Stage B：冻结统一 S2 后在 H192 上 3 个原始参考 + S2，共 24 次。
- 合计 66 次训练，全部完成；候选均从同 setting 的 A1 checkpoint 加载，`unexpected=0`，初始输出与 A1 的最大绝对差为 0。
- **test_accessed=false**。最终门失败，因此没有 test/Golden 确认，也不能宣称超过 Golden。
- 回放了全部 12 个验证 setting；记录 `{total_pairs:,}` 条样本×通道误差，聚合指标与训练日志最大差小于 `1e-5`。

## Stage-A 候选排序

{markdown_table(ranking_rows, ['候选', 'H96宏比值', '最差比值', '严格通过'], {'H96宏比值': lambda x: f'{x:.6f}', '最差比值': lambda x: f'{x:.6f}'})}

S2 与 S3 几乎持平，CVaR 保护和单调距离先验都没有改变失败性质；S2 按预注册排序胜出并原样进入 H192，没有二次调参。

## 12 个 setting 的最终结果

{markdown_table(table_rows, ['数据集', 'H', '最强原始', '原始MSE', 'S2 MSE', 'MSE变化', '原始MAE', 'S2 MAE', 'MAE变化'], {'原始MSE': lambda x: f'{x:.6f}', 'S2 MSE': lambda x: f'{x:.6f}', '原始MAE': lambda x: f'{x:.6f}', 'S2 MAE': lambda x: f'{x:.6f}', 'MSE变化': percent, 'MAE变化': percent})}

![逐 setting 指标变化](figures/all__metric_ratios.png)

只有少数 setting 能同时改善 MSE/MAE；Weather-H96/H192、ETTh1-H96/H192 的差距最明显。H192 没有消除问题，说明不是 H96 的偶然噪声。

## 样本级结果

以每个 setting 的单一最强原始模型为 baseline，按样本×通道的相对 MSE 变化分组：

{markdown_table(group_rows, ['分组', '数量', '占比', '漂移', 'lag24', '接纳', 'no-op概率'], {'占比': lambda x: f'{x:.2%}', '漂移': lambda x: f'{x:.4f}', 'lag24': lambda x: f'{x:.4f}', '接纳': lambda x: f'{x:.4f}', 'no-op概率': lambda x: f'{x:.4f}'})}

![样本分组](figures/all__sample_groups.png)

门控的主要失败模式是：它学到的是“相对 A1 是否值得修正”，不是“相对当前最强原始模型是否值得修正”。因此高接纳在部分 setting 内能改善 A1，却未必改善 R0/I0；接纳强度与相对最强原始模型的真实收益也没有形成稳定单调关系。

![接纳校准](figures/all__acceptance_calibration.png)

## 代表样本与细粒度中间量

共选择 9 个互不近邻的代表案例；曲线左侧给出真实值、最强原始模型、A1 锚点与 S2，右侧给出每个未来周期的实际接纳值及 no-op/相位/轨迹/周期间动作概率。完整的专家预测、权重、滚动风险、风险方差和结构特征保存在 `selected_cases.npz`。

{markdown_table(case_rows, ['类型', 'setting', '通道', '样本', '原始MSE', 'S2 MSE'], {'原始MSE': lambda x: f'{x:.5f}', 'S2 MSE': lambda x: f'{x:.5f}'})}

![原始高误差案例](figures/all__cases_baseline_high_error.png)

![S2退化案例](figures/all__cases_candidate_regression.png)

![S2改善案例](figures/all__cases_candidate_improvement.png)

## 对当前设计的判断

1. **已解决的问题**：集成不再因随机初始化直接破坏 A1；初始值和关闭门控时都严格等于 A1，冻结边界也通过审计。
2. **未解决的问题**：A1 不是所有数据集/预测长度上的最强专家，因此“对 A1 不后悔”不等价于“对所有原始模型不后悔”。
3. **训练目标错位**：路由 oracle 比较的是三个修正方向相对 A1 的周期级收益，而最终科研门槛比较的是 S2 相对 A1/I0/R0 包络；两者不是同一目标。
4. **下一步最小合理改动**：构造 `A1/I0/R0 + no-op` 的多锚点选择器，先用 rolling-origin out-of-fold 预测训练“选择哪个完整模型”，再只在被选锚点附近学习小修正；验证时同时约束均值 regret 与每 setting 的上尾 regret。若仍无法逐 setting 超过包络，应停止追求硬集成，改为报告专家适用域。

## 文件说明

- `results.csv`：66 个训练结果与相对原始包络比值。
- `sample_errors.csv`：全部 `{total_pairs:,}` 条样本×通道误差及门控摘要。
- `selected_cases.npz`：9 个代表案例的预测、完整周期级门控、风险及结构特征。
- `run.yaml`：命令、环境、检查点和审计结论。
"""
    report_path = OUTPUT / "objective_error_analysis.md"
    report_path.write_text(report)

    manifest = {
        "experiment_id": "safe_regret_triaxis_v1",
        "status": "completed_gate_failed",
        "hypothesis": "A1-anchored abstaining regret fusion can dominate the original-model envelope",
        "hypothesis_supported": False,
        "selection": native(stage_a),
        "final_gate": native(final),
        "protocol": {
            "datasets": DATASETS, "lookback": 720, "horizons": HORIZONS,
            "period": 24, "train_percent": 30, "epochs": 8,
            "seed": 2021, "loss": "huber", "test_accessed": False,
        },
        "commands": [
            f"{sys.executable} scripts/run_safe_regret_triaxis.py --stage all --output-dir research_runs/safe_regret_triaxis_v1_scratch --num-workers 4",
            f"{sys.executable} scripts/analyze_safe_regret_triaxis.py",
        ],
        "environment": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "audit": {
            "completed_runs": 66, "sample_channel_rows": total_pairs,
            "selected_cases": len(cases), "metric_replay": native(replay),
            "anchor_identity_exact": True, "test_accessed": False,
        },
        "artifacts": {},
        "elapsed_analysis_sec": time.time() - started,
    }
    for name in ("results.csv", "sample_errors.csv", "selected_cases.npz", "objective_error_analysis.md"):
        path = OUTPUT / name
        manifest["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUTPUT / "run.yaml").write_text(
        yaml.safe_dump(native(manifest), sort_keys=False, allow_unicode=True)
    )

    zip_path = OUTPUT / "objective_error_analysis.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")
    print(f"wrote {OUTPUT}")
    print(f"zip sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
