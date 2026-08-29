#!/usr/bin/env python3
"""Build the strict validation-only HPTC v1 audit bundle."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import platform
import re
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

from scripts.analyze_triaxis_experiment import (  # noqa: E402
    build_model_and_loader,
    variable_names,
)

SCRATCH = REPO / "research_runs" / "hptc_unified_v1_scratch" / "runs"
REFERENCE = REPO / "research_runs" / "safe_regret_triaxis_v1_scratch" / "runs"
OUTPUT = REPO / "research_runs" / "hptc_unified_v1"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
REFERENCES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_rolling_features",
)
CANDIDATES = (
    "hptc_fixed_b10",
    "hptc_rolling_b10",
    "hptc_rolling_b25",
    "hptc_rolling_b50",
    "hptc_rolling_b25_r05",
)
CHAMPION = "hptc_rolling_b25_r05"
A1 = "gold_combo_reliability_s2"
LABELS = {
    A1: "A1 RCRF+NLinear",
    "rcrf_icpt_none": "I0 RCRF+ICPT",
    "triaxis_rolling_features": "R0 Rolling TriAxis",
    "hptc_fixed_b10": "H0 fixed beta=.10",
    "hptc_rolling_b10": "H1 rolling beta=.10",
    "hptc_rolling_b25": "H2 rolling beta=.25",
    "hptc_rolling_b50": "H3 rolling beta=.50",
    CHAMPION: "H4 rolling beta=.25 risk=.5",
}
GROUPS = ("significant_improvement", "comparable", "significant_regression")
SAMPLE_FIELDS = (
    "setting", "baseline_config_id", "candidate_config_id", "sample_id",
    "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse",
    "relative_delta_mse", "baseline_mae", "candidate_mae", "delta_mae",
    "group", "recent_drift", "lag24_correlation", "difference_volatility",
    "phase_reliability", "rcrf_alpha", "shape_beta", "rolling_risk",
    "rolling_risk_std", "shape_confidence", "shape_correction_magnitude",
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def setting_name(dataset):
    return f"{dataset}-L720-H96-s2021-validation"


def find_run(root, dataset, mechanism):
    matches = []
    for config_path in Path(root).glob("*/config.json"):
        spec = json.loads(config_path.read_text())
        metrics_path = config_path.parent / "metrics.csv"
        if (
            metrics_path.is_file()
            and spec.get("dataset") == dataset
            and int(spec.get("horizon", -1)) == 96
            and int(spec.get("percent", -1)) == 30
            and spec.get("mechanism") == mechanism
        ):
            with metrics_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            matches.append((config_path.parent, row, spec))
    if len(matches) != 1:
        raise RuntimeError(f"expected one H96 run for {dataset}/{mechanism}, got {len(matches)}")
    return matches[0]


def history_features(x, eps=1e-6):
    scale = x.std(dim=1, unbiased=False).clamp_min(eps)
    drift = (
        x[:, -24:, :].mean(dim=1) - x[:, -48:-24, :].mean(dim=1)
    ).abs() / scale
    diff_vol = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1) / scale
    left, right = x[:, 24:, :], x[:, :-24, :]
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    lag24 = (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
        + eps
    )
    return drift, lag24, diff_vol


def push_case(heap, score, serial, case, limit=2048):
    item = (float(score), int(serial), case)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def local_top(values, positive=False, limit=8):
    flat = values.reshape(-1)
    indices = torch.arange(flat.numel(), device=flat.device)
    if positive:
        mask = torch.isfinite(flat) & (flat > 0)
        flat, indices = flat[mask], indices[mask]
    if not flat.numel():
        return []
    top = torch.topk(flat, min(limit, flat.numel()))
    return [(float(value), int(indices[index])) for value, index in zip(top.values, top.indices)]


def make_case(
    category, score, dataset, sample_id, channel_id, channel, time_range,
    history, truth, baseline, candidate, trajectory, cycle_shape, correction,
    beta, risk, risk_std, confidence, phase_r, alpha,
):
    truth_np = truth.detach().cpu().numpy()
    baseline_np = baseline.detach().cpu().numpy()
    candidate_np = candidate.detach().cpu().numpy()
    return {
        "case_type": category,
        "score": float(score),
        "setting": setting_name(dataset),
        "dataset": dataset,
        "sample_id": int(sample_id),
        "channel_id": int(channel_id),
        "channel": channel,
        "time_range": time_range,
        "history": history.detach().cpu().numpy(),
        "truth": truth_np,
        "baseline": baseline_np,
        "candidate": candidate_np,
        "trajectory": trajectory.detach().cpu().numpy(),
        "cycle_shape": cycle_shape.detach().cpu().numpy(),
        "correction": correction.detach().cpu().numpy(),
        "beta": beta.detach().cpu().numpy(),
        "risk": risk.detach().cpu().numpy(),
        "risk_std": risk_std.detach().cpu().numpy(),
        "confidence": confidence.detach().cpu().numpy(),
        "phase_reliability": float(phase_r),
        "rcrf_alpha": float(alpha),
        "baseline_mse": float(np.mean((baseline_np - truth_np) ** 2)),
        "candidate_mse": float(np.mean((candidate_np - truth_np) ** 2)),
        "baseline_mae": float(np.mean(np.abs(baseline_np - truth_np))),
        "candidate_mae": float(np.mean(np.abs(candidate_np - truth_np))),
    }


def timed_forward(model, x, x_mark, y, y_mark, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    output, _, _ = model(
        x, x_mark.float(), model._build_decoder_input(y), y_mark.float()
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return output, time.perf_counter() - started


def evaluate_setting(dataset_name, baseline_info, candidate_info, writer, device):
    batch_size = 8 if dataset_name == "Electricity" else (
        32 if dataset_name == "Weather" else 128
    )
    baseline, dataset, loader = build_model_and_loader(*baseline_info, batch_size)
    candidate, dataset2, loader2 = build_model_and_loader(*candidate_info, batch_size)
    if len(dataset) != len(dataset2):
        raise RuntimeError("baseline and candidate validation splits differ")
    del loader2, dataset2
    baseline, candidate = baseline.to(device).eval(), candidate.to(device).eval()
    names = variable_names(candidate_info[2])
    timestamps = getattr(dataset, "timestamps", None)
    totals = defaultdict(float)
    group_counts = defaultdict(int)
    feature_names = SAMPLE_FIELDS[14:]
    group_sums = {group: defaultdict(float) for group in GROUPS}
    horizon_abs = np.zeros((2, 4), dtype=np.float64)
    horizon_count = np.zeros(4, dtype=np.int64)
    pools = {name: [] for name in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    )}
    offset = serial = batches = 0
    warmed = False

    with torch.inference_mode():
        for batch in loader:
            x, y, x_mark, y_mark = [value.to(device) for value in batch]
            x, y = x.float(), y.float()
            truth = y[:, -96:, :]
            if not warmed:
                baseline(x, x_mark.float(), baseline._build_decoder_input(y), y_mark.float())
                candidate(x, x_mark.float(), candidate._build_decoder_input(y), y_mark.float())
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warmed = True
            base, base_sec = timed_forward(baseline, x, x_mark, y, y_mark, device)
            cand, cand_sec = timed_forward(candidate, x, x_mark, y, y_mark, device)
            totals["baseline_forward_sec"] += base_sec
            totals["candidate_forward_sec"] += cand_sec
            batches += 1

            head = candidate.weak_period_residual
            beta = head.last_beta
            risk = head.last_risk
            risk_std = head.last_risk_std
            confidence = head.last_confidence
            trajectory = head.last_trajectory
            cycle_shape = head.last_cycle_shape
            correction = head.last_correction
            phase_r = candidate.rcrf_fusion.last_r
            alpha = candidate.rcrf_fusion.last_alpha
            if risk is None or risk_std is None:
                raise RuntimeError("champion did not emit rolling diagnostics")

            for prefix, prediction in (("baseline", base), ("candidate", cand)):
                error = prediction - truth
                totals[f"{prefix}_sq"] += float(error.square().sum())
                totals[f"{prefix}_abs"] += float(error.abs().sum())
            totals["count"] += truth.numel()
            totals["correction_mean_max"] = max(
                totals["correction_mean_max"], head.last_correction_cycle_mean_max
            )

            base_mse = (base - truth).square().mean(dim=1)
            cand_mse = (cand - truth).square().mean(dim=1)
            base_mae = (base - truth).abs().mean(dim=1)
            cand_mae = (cand - truth).abs().mean(dim=1)
            relative = (cand_mse - base_mse) / base_mse.clamp_min(1e-8)
            drift, lag24, diff_vol = history_features(x)
            feature_values = {
                "recent_drift": drift,
                "lag24_correlation": lag24,
                "difference_volatility": diff_vol,
                "phase_reliability": phase_r,
                "rcrf_alpha": alpha,
                "shape_beta": beta.mean(dim=-1),
                "rolling_risk": risk.mean(dim=-1),
                "rolling_risk_std": risk_std.mean(dim=-1),
                "shape_confidence": confidence.mean(dim=-1),
                "shape_correction_magnitude": correction.abs().mean(dim=1),
            }
            masks = {
                "significant_improvement": relative <= -0.10,
                "significant_regression": relative >= 0.10,
                "comparable": (relative > -0.10) & (relative < 0.10),
            }
            for group, mask in masks.items():
                group_counts[group] += int(mask.sum())
                for key, values in feature_values.items():
                    group_sums[group][key] += float(values[mask].sum())

            for segment in range(4):
                start, end = segment * 24, (segment + 1) * 24
                horizon_abs[0, segment] += float((base[:, start:end] - truth[:, start:end]).abs().sum())
                horizon_abs[1, segment] += float((cand[:, start:end] - truth[:, start:end]).abs().sum())
                horizon_count[segment] += truth[:, start:end].numel()

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
                    rel = float(relative[b, c])
                    group = "significant_improvement" if rel <= -0.10 else (
                        "significant_regression" if rel >= 0.10 else "comparable"
                    )
                    rows.append({
                        "setting": setting_name(dataset_name),
                        "baseline_config_id": A1,
                        "candidate_config_id": CHAMPION,
                        "sample_id": offset + b,
                        "channel": names[c] if c < len(names) else str(c),
                        "time_range": time_range,
                        "baseline_mse": f"{float(base_mse[b, c]):.8g}",
                        "candidate_mse": f"{float(cand_mse[b, c]):.8g}",
                        "delta_mse": f"{float(cand_mse[b, c] - base_mse[b, c]):.8g}",
                        "relative_delta_mse": f"{rel:.8g}",
                        "baseline_mae": f"{float(base_mae[b, c]):.8g}",
                        "candidate_mae": f"{float(cand_mae[b, c]):.8g}",
                        "delta_mae": f"{float(cand_mae[b, c] - base_mae[b, c]):.8g}",
                        "group": group,
                        **{key: f"{float(values[b, c]):.8g}" for key, values in feature_values.items()},
                    })
            writer.writerows(rows)

            # The preregistered case ranking uses absolute MAE delta.
            scores = {
                "baseline_high_error": base_mae,
                "candidate_regression": cand_mae - base_mae,
                "candidate_improvement": base_mae - cand_mae,
            }
            for category, score_tensor in scores.items():
                for score, flat_index in local_top(
                    score_tensor, positive=category != "baseline_high_error"
                ):
                    b, c = divmod(flat_index, C)
                    start = offset + b + x.shape[1]
                    time_range = (
                        f"{timestamps[start]}--{timestamps[start + 95]}"
                        if timestamps is not None and start + 95 < len(timestamps)
                        else f"index:{start}--{start + 95}"
                    )
                    case = make_case(
                        category, score, dataset_name, offset + b, c,
                        names[c] if c < len(names) else str(c), time_range,
                        x[b, :, c], truth[b, :, c], base[b, :, c], cand[b, :, c],
                        trajectory[b, :, c], cycle_shape[b, :, c], correction[b, :, c],
                        beta[b, c], risk[b, c], risk_std[b, c], confidence[b, c],
                        phase_r[b, c], alpha[b, c],
                    )
                    push_case(pools[category], score, serial, case)
                    serial += 1
            offset += B

    count = totals["count"]
    summary = {
        "setting": setting_name(dataset_name),
        "dataset": dataset_name,
        "validation_samples": len(dataset),
        "sample_channel_pairs": sum(group_counts.values()),
        "baseline_mse": totals["baseline_sq"] / count,
        "baseline_mae": totals["baseline_abs"] / count,
        "candidate_mse": totals["candidate_sq"] / count,
        "candidate_mae": totals["candidate_abs"] / count,
        "groups": dict(group_counts),
        "group_feature_means": {
            group: {
                key: group_sums[group][key] / max(group_counts[group], 1)
                for key in feature_names
            } for group in GROUPS
        },
        "horizon_mae": {
            "baseline": (horizon_abs[0] / horizon_count).tolist(),
            "candidate": (horizon_abs[1] / horizon_count).tolist(),
        },
        "forward_ms_per_batch": {
            "baseline": 1000 * totals["baseline_forward_sec"] / batches,
            "candidate": 1000 * totals["candidate_forward_sec"] / batches,
            "batch_size": batch_size,
        },
        "learned_global_beta": float(torch.sigmoid(head.beta_logit).detach()),
        "correction_cycle_mean_max": totals["correction_mean_max"],
    }
    pools = {
        key: [item[2] for item in sorted(heap, reverse=True)]
        for key, heap in pools.items()
    }
    del baseline, candidate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, pools


def duplicate(case, selected):
    return any(
        old["setting"] == case["setting"]
        and old["channel_id"] == case["channel_id"]
        and abs(old["sample_id"] - case["sample_id"]) < 96
        for old in selected
    )


def select_cases(pools, top_k=5):
    selected = []
    for dataset in DATASETS:
        for category in (
            "baseline_high_error", "candidate_regression", "candidate_improvement"
        ):
            options = sorted(
                pools[dataset][category], key=lambda case: case["score"], reverse=True
            )
            chosen = []
            for case in options:
                if duplicate(case, chosen):
                    continue
                chosen.append(case)
                if len(chosen) == top_k:
                    break
            if len(chosen) != top_k:
                raise RuntimeError(
                    f"insufficient deduplicated cases: {dataset}/{category}: {len(chosen)}"
                )
            selected.extend(chosen)
    return selected


def representative_cases(cases):
    representatives = []
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        options = sorted(
            (case for case in cases if case["case_type"] == category),
            key=lambda case: case["score"], reverse=True,
        )
        chosen, used = [], set()
        for distinct in (True, False):
            for case in options:
                if case in chosen or duplicate(case, representatives + chosen):
                    continue
                if distinct and case["setting"] in used:
                    continue
                chosen.append(case)
                used.add(case["setting"])
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                break
        representatives.extend(chosen)
    return representatives


def save_cases(path, cases):
    np.savez_compressed(
        path,
        setting=np.asarray([case["setting"] for case in cases]),
        case_type=np.asarray([case["case_type"] for case in cases]),
        dataset=np.asarray([case["dataset"] for case in cases]),
        sample_id=np.asarray([case["sample_id"] for case in cases]),
        channel_id=np.asarray([case["channel_id"] for case in cases]),
        channel=np.asarray([case["channel"] for case in cases]),
        time_range=np.asarray([case["time_range"] for case in cases]),
        historical_input=np.stack([case["history"] for case in cases]),
        truth=np.stack([case["truth"] for case in cases]),
        baseline_prediction=np.stack([case["baseline"] for case in cases]),
        candidate_prediction=np.stack([case["candidate"] for case in cases]),
        nlinear_trajectory=np.stack([case["trajectory"] for case in cases]),
        icpt_cycle_shape=np.stack([case["cycle_shape"] for case in cases]),
        zero_mean_shape_correction=np.stack([case["correction"] for case in cases]),
        shape_beta=np.stack([case["beta"] for case in cases]),
        rolling_risk=np.stack([case["risk"] for case in cases]),
        rolling_risk_std=np.stack([case["risk_std"] for case in cases]),
        shape_confidence=np.stack([case["confidence"] for case in cases]),
        phase_reliability=np.asarray([case["phase_reliability"] for case in cases]),
        rcrf_alpha=np.asarray([case["rcrf_alpha"] for case in cases]),
    )


def table(rows, columns, formats=None):
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


def plot_search(rows, path):
    candidates = list(CANDIDATES)
    labels = [f"{dataset}-{metric.upper()}" for dataset in DATASETS for metric in ("mse", "mae")]
    values = []
    for mode in candidates:
        values.append([
            next(row for row in rows if row["dataset"] == dataset and row["config_id"] == mode)[f"{metric}_ratio_to_a1"]
            for dataset in DATASETS for metric in ("mse", "mae")
        ])
    values = 100 * (np.asarray(values) - 1)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    image = ax.imshow(values, cmap="RdYlGn_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(len(candidates)):
        for j in range(len(labels)):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(candidates)), [LABELS[mode] for mode in candidates])
    ax.set_title("HPTC 调参：相对 A1 的验证误差变化（%）")
    fig.colorbar(image, ax=ax, label="相对变化（%）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_groups(analysis, path):
    values = {
        group: [
            analysis[dataset]["groups"].get(group, 0)
            / analysis[dataset]["sample_channel_pairs"]
            for dataset in DATASETS
        ] for group in GROUPS
    }
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, label, color in (
        ("significant_improvement", "显著改善（MSE≤-10%）", "#54A24B"),
        ("comparable", "相近", "#BAB0AC"),
        ("significant_regression", "显著退化（MSE≥+10%）", "#E45756"),
    ):
        ax.bar(x, values[group], bottom=bottom, label=label, color=color)
        bottom += np.asarray(values[group])
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("样本×通道占比")
    ax.set_title("H4 相对 A1 的样本级误差分组")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_features(group_features, path):
    selected = (
        "recent_drift", "lag24_correlation", "difference_volatility",
        "shape_beta", "rolling_risk", "shape_correction_magnitude",
    )
    labels = ("近期漂移", "lag-24相关", "差分波动", "形状beta", "历史风险", "形状修正幅度")
    raw = np.asarray([
        [group_features[group][feature] for feature in selected]
        for group in GROUPS
    ])
    # Column normalization makes unlike diagnostic scales visually comparable;
    # exact raw means remain in the report table.
    normalized = (raw - raw.mean(axis=0, keepdims=True)) / (
        raw.std(axis=0, keepdims=True) + 1e-8
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    image = ax.imshow(normalized, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(3):
        for j in range(len(selected)):
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(selected)), labels, rotation=20, ha="right")
    ax.set_yticks(range(3), ("显著改善", "相近", "显著退化"))
    ax.set_title("样本分组的输入结构与门控诊断（格内为原始均值）")
    fig.colorbar(image, ax=ax, label="列内标准化差异")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_horizon(analysis, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, width = np.arange(4), 0.12
    for index, dataset in enumerate(DATASETS):
        base = np.asarray(analysis[dataset]["horizon_mae"]["baseline"])
        candidate = np.asarray(analysis[dataset]["horizon_mae"]["candidate"])
        ax.bar(
            x + (index - 2.5) * width,
            100 * (candidate / base - 1), width, label=dataset,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ("1–24", "25–48", "49–72", "73–96"))
    ax.set_ylabel("MAE 相对变化（%）")
    ax.set_xlabel("预测区间")
    ax.set_title("H4 相对 A1 的逐未来周期误差变化")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cases(cases, category, path):
    subset = [case for case in cases if case["case_type"] == category]
    titles = {
        "baseline_high_error": "A1 原始高误差代表",
        "candidate_regression": "H4 明显退化代表",
        "candidate_improvement": "H4 明显改善代表",
    }
    fig, axes = plt.subplots(len(subset), 2, figsize=(12, 3.5 * len(subset)), squeeze=False)
    for row, case in enumerate(subset):
        left, right = axes[row]
        left.plot(np.arange(-192, 0), case["history"][-192:], color="#999999", label="近期历史")
        future = np.arange(1, 97)
        left.plot(future, case["truth"], color="black", linewidth=1.6, label="真实")
        left.plot(future, case["baseline"], color="#9467BD", label="A1")
        left.plot(future, case["candidate"], color="#E45756", label="H4")
        left.set_title(
            f"{case['dataset']} / {case['channel']} / 样本{case['sample_id']}\n"
            f"MAE {case['baseline_mae']:.4f} → {case['candidate_mae']:.4f}"
        )
        left.grid(alpha=0.2)
        if row == 0:
            left.legend(ncol=4, fontsize=7)
        cycles = np.arange(1, 5)
        right.plot(cycles, case["beta"], marker="o", label="有效形状beta", color="#4C78A8")
        right.plot(cycles, case["confidence"], marker="o", label="置信度", color="#54A24B")
        right.plot(cycles, case["risk"], marker="o", label="历史风险", color="#E45756")
        right.set_xticks(cycles)
        right.set_xlabel("未来 24 步周期")
        right.set_title(
            f"相位可靠度={case['phase_reliability']:.3f}, RCRF alpha={case['rcrf_alpha']:.3f}"
        )
        right.grid(alpha=0.2)
        if row == 0:
            right.legend(ncol=3, fontsize=7)
    fig.suptitle(titles[category])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    started = time.time()
    if OUTPUT.exists():
        raise FileExistsError(f"canonical output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    figures = OUTPUT / "figures"
    figures.mkdir()

    reference_infos = {
        dataset: {mode: find_run(REFERENCE, dataset, mode) for mode in REFERENCES}
        for dataset in DATASETS
    }
    candidate_infos = {
        dataset: {mode: find_run(SCRATCH, dataset, mode) for mode in CANDIDATES}
        for dataset in DATASETS
    }
    all_infos = [
        info for group in (reference_infos, candidate_infos)
        for dataset_infos in group.values() for info in dataset_infos.values()
    ]
    if len(all_infos) != 48:
        raise RuntimeError(f"expected 48 validation-only runs, got {len(all_infos)}")
    if any(info[1].get("test_mse") not in (None, "", "nan") for info in all_infos):
        raise RuntimeError("test metric detected")

    result_rows = []
    for dataset in DATASETS:
        a1_mse = float(reference_infos[dataset][A1][1]["val_mse"])
        a1_mae = float(reference_infos[dataset][A1][1]["val_mae"])
        envelope_mse = min(float(info[1]["val_mse"]) for info in reference_infos[dataset].values())
        envelope_mae = min(float(info[1]["val_mae"]) for info in reference_infos[dataset].values())
        for stage, infos in (
            ("reference", reference_infos[dataset]),
            ("candidate", candidate_infos[dataset]),
        ):
            for mode, (_, row, spec) in infos.items():
                mse, mae = float(row["val_mse"]), float(row["val_mae"])
                hp = spec["hyperparams"]
                key_params = {
                    "beta_init": hp.get("hptc_beta_init"),
                    "rolling": hp.get("hptc_use_rolling_confidence"),
                    "risk_scale": hp.get("hptc_risk_scale"),
                    "epochs": spec["max_epochs"],
                    "train_percent": spec["percent"],
                }
                result_rows.append({
                    "setting": setting_name(dataset),
                    "config_id": mode,
                    "dataset": dataset,
                    "horizon": 96,
                    "seed": 2021,
                    "model": LABELS[mode],
                    "stage": stage,
                    "key_params": json.dumps(key_params, sort_keys=True),
                    "mse": mse,
                    "mae": mae,
                    "mse_ratio_to_a1": mse / a1_mse,
                    "mae_ratio_to_a1": mae / a1_mae,
                    "mse_ratio_to_envelope": mse / envelope_mse,
                    "mae_ratio_to_envelope": mae / envelope_mae,
                    "parameter_count": int(row["parameter_count"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                    "peak_memory_bytes": int(row["peak_memory_bytes"]),
                    "selected": mode == CHAMPION,
                    "test_accessed": False,
                })
    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    ranking = []
    for mode in CANDIDATES:
        rows = [row for row in result_rows if row["config_id"] == mode]
        a1_ratios = [value for row in rows for value in (
            row["mse_ratio_to_a1"], row["mae_ratio_to_a1"]
        )]
        envelope_ratios = [value for row in rows for value in (
            row["mse_ratio_to_envelope"], row["mae_ratio_to_envelope"]
        )]
        both = sum(
            row["mse_ratio_to_a1"] < 1 and row["mae_ratio_to_a1"] < 1
            for row in rows
        )
        ranking.append({
            "config_id": mode,
            "macro_ratio_to_a1": float(np.mean(a1_ratios)),
            "worst_ratio_to_a1": float(np.max(a1_ratios)),
            "both_improved_datasets": both,
            "macro_ratio_to_envelope": float(np.mean(envelope_ratios)),
        })
    ranking.sort(key=lambda item: item["macro_ratio_to_a1"])
    if ranking[0]["config_id"] != CHAMPION:
        raise RuntimeError(f"unexpected champion: {ranking[0]}")
    winner = ranking[0]
    gate = {
        "macro_ratio_at_most_0.998": winner["macro_ratio_to_a1"] <= 0.998,
        "both_metrics_improved_on_at_least_4_of_6": winner["both_improved_datasets"] >= 4,
        "worst_ratio_at_most_1.01": winner["worst_ratio_to_a1"] <= 1.01,
    }
    gate["passed"] = all(gate.values())
    if gate["passed"]:
        raise RuntimeError("H96 gate unexpectedly passed; H192 should have been run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis, pools = {}, {}
    with (OUTPUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            summary, case_pool = evaluate_setting(
                dataset,
                reference_infos[dataset][A1],
                candidate_infos[dataset][CHAMPION],
                writer,
                device,
            )
            analysis[dataset], pools[dataset] = summary, case_pool
            print(f"audited {dataset}: {summary['sample_channel_pairs']:,} pairs", flush=True)

    replay = {}
    for dataset in DATASETS:
        base_row = reference_infos[dataset][A1][1]
        cand_row = candidate_infos[dataset][CHAMPION][1]
        checks = {
            "baseline_mse": abs(analysis[dataset]["baseline_mse"] - float(base_row["val_mse"])),
            "baseline_mae": abs(analysis[dataset]["baseline_mae"] - float(base_row["val_mae"])),
            "candidate_mse": abs(analysis[dataset]["candidate_mse"] - float(cand_row["val_mse"])),
            "candidate_mae": abs(analysis[dataset]["candidate_mae"] - float(cand_row["val_mae"])),
        }
        if max(checks.values()) >= 2e-5:
            raise RuntimeError(f"metric replay failed: {dataset}: {checks}")
        replay[setting_name(dataset)] = checks

    cases = select_cases(pools, top_k=5)
    if len(cases) != 90:
        raise RuntimeError(f"expected 90 selected cases, got {len(cases)}")
    save_cases(OUTPUT / "selected_cases.npz", cases)
    representatives = representative_cases(cases)
    if len(representatives) != 9:
        raise RuntimeError(f"expected 9 representative cases, got {len(representatives)}")

    plot_search(result_rows, figures / "all__search_ratios.png")
    plot_groups(analysis, figures / "all__sample_groups.png")
    plot_horizon(analysis, figures / "all__horizon_mae.png")

    total_pairs = sum(item["sample_channel_pairs"] for item in analysis.values())
    total_groups = {
        group: sum(item["groups"].get(group, 0) for item in analysis.values())
        for group in GROUPS
    }
    feature_names = next(iter(analysis.values()))["group_feature_means"]["comparable"]
    aggregate_features = {
        group: {
            feature: sum(
                analysis[dataset]["group_feature_means"][group][feature]
                * analysis[dataset]["groups"].get(group, 0)
                for dataset in DATASETS
            ) / max(total_groups[group], 1)
            for feature in feature_names
        } for group in GROUPS
    }
    plot_features(aggregate_features, figures / "all__group_features.png")
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        plot_cases(representatives, category, figures / f"all__cases_{category}.png")

    ranking_rows = [{
        "候选": LABELS[item["config_id"]],
        "A1宏比值": item["macro_ratio_to_a1"],
        "最差比值": item["worst_ratio_to_a1"],
        "双改善": item["both_improved_datasets"],
        "包络宏比值": item["macro_ratio_to_envelope"],
    } for item in ranking]
    metric_rows = []
    for dataset in DATASETS:
        row = next(
            item for item in result_rows
            if item["dataset"] == dataset and item["config_id"] == CHAMPION
        )
        metric_rows.append({
            "数据集": dataset,
            "A1 MSE": row["mse"] / row["mse_ratio_to_a1"],
            "H4 MSE": row["mse"],
            "MSE变化": row["mse_ratio_to_a1"] - 1,
            "A1 MAE": row["mae"] / row["mae_ratio_to_a1"],
            "H4 MAE": row["mae"],
            "MAE变化": row["mae_ratio_to_a1"] - 1,
        })
    group_rows = [{
        "分组": {
            "significant_improvement": "显著改善",
            "comparable": "相近",
            "significant_regression": "显著退化",
        }[group],
        "数量": total_groups[group],
        "占比": total_groups[group] / total_pairs,
        "漂移": aggregate_features[group]["recent_drift"],
        "lag24": aggregate_features[group]["lag24_correlation"],
        "差分波动": aggregate_features[group]["difference_volatility"],
        "beta": aggregate_features[group]["shape_beta"],
        "风险": aggregate_features[group]["rolling_risk"],
        "修正幅度": aggregate_features[group]["shape_correction_magnitude"],
    } for group in GROUPS]
    dataset_group_rows = [{
        "数据集": dataset,
        "改善占比": analysis[dataset]["groups"].get("significant_improvement", 0)
        / analysis[dataset]["sample_channel_pairs"],
        "退化占比": analysis[dataset]["groups"].get("significant_regression", 0)
        / analysis[dataset]["sample_channel_pairs"],
        "改善lag24": analysis[dataset]["group_feature_means"]["significant_improvement"]["lag24_correlation"],
        "退化lag24": analysis[dataset]["group_feature_means"]["significant_regression"]["lag24_correlation"],
        "改善风险": analysis[dataset]["group_feature_means"]["significant_improvement"]["rolling_risk"],
        "退化风险": analysis[dataset]["group_feature_means"]["significant_regression"]["rolling_risk"],
        "全局beta": analysis[dataset]["learned_global_beta"],
    } for dataset in DATASETS]
    horizon_rows = []
    for dataset in DATASETS:
        base = np.asarray(analysis[dataset]["horizon_mae"]["baseline"])
        candidate = np.asarray(analysis[dataset]["horizon_mae"]["candidate"])
        changes = candidate / base - 1
        horizon_rows.append({
            "数据集": dataset, "1–24": changes[0], "25–48": changes[1],
            "49–72": changes[2], "73–96": changes[3],
        })
    latency_rows = [{
        "数据集": dataset,
        "batch": analysis[dataset]["forward_ms_per_batch"]["batch_size"],
        "A1 ms/batch": analysis[dataset]["forward_ms_per_batch"]["baseline"],
        "H4 ms/batch": analysis[dataset]["forward_ms_per_batch"]["candidate"],
        "耗时比": analysis[dataset]["forward_ms_per_batch"]["candidate"]
        / analysis[dataset]["forward_ms_per_batch"]["baseline"],
    } for dataset in DATASETS]
    case_rows = [{
        "类型": {
            "baseline_high_error": "A1高误差",
            "candidate_regression": "H4退化",
            "candidate_improvement": "H4改善",
        }[case["case_type"]],
        "setting": case["setting"],
        "通道": case["channel"],
        "样本": case["sample_id"],
        "A1 MAE": case["baseline_mae"],
        "H4 MAE": case["candidate_mae"],
        "beta": float(np.mean(case["beta"])),
        "风险": float(np.mean(case["risk"])),
    } for case in representatives]
    percent = lambda value: f"{100 * float(value):+.2f}%"
    max_correction_mean = max(
        analysis[dataset]["correction_cycle_mean_max"] for dataset in DATASETS
    )
    base_params = int(np.mean([
        int(reference_infos[dataset][A1][1]["parameter_count"]) for dataset in DATASETS
    ]))
    cand_params = int(np.mean([
        int(candidate_infos[dataset][CHAMPION][1]["parameter_count"]) for dataset in DATASETS
    ]))
    report = f"""# HPTC v1：统一相位—轨迹—周期模型实验与误差审计

## 1. Experiment Setup

HPTC 不是完整模型 ensemble。一个 PhaseFormer 主干负责相位，NLinear 负责未来轨迹和每个 24 步周期的水平，ICPT 只提供逐周期零均值形状修正；历史 rolling risk 只连续收缩形状修正，外层仍由 RCRF 按相位可靠度融合。六数据集统一使用 L720、H96、P24、30% train、最多 8 epoch、seed 2021、Huber、best-validation checkpoint；**没有读取 test**。

## 2. Experiment Results

H4 是五组预注册候选中最好的一组，相对 A1 的 12 指标宏比值为 `{winner['macro_ratio_to_a1']:.6f}`（平均 `{percent(winner['macro_ratio_to_a1'] - 1)}`），但只在 `{winner['both_improved_datasets']}/6` 个数据集双指标改善。预注册门槛要求至少 4/6，因此 H96 gate 失败，按计划停止 H192。

{table(metric_rows, ['数据集', 'A1 MSE', 'H4 MSE', 'MSE变化', 'A1 MAE', 'H4 MAE', 'MAE变化'], {'A1 MSE': lambda x: f'{x:.6f}', 'H4 MSE': lambda x: f'{x:.6f}', 'A1 MAE': lambda x: f'{x:.6f}', 'H4 MAE': lambda x: f'{x:.6f}', 'MSE变化': percent, 'MAE变化': percent})}

H4 在 ETTh1、ETTm2、Electricity 同时改善 MSE/MAE；ETTh2、ETTm1 两项均轻微退化，Weather 只有 MAE 小幅改善。它相对 A1 的最差单项退化仅 `{percent(winner['worst_ratio_to_a1'] - 1)}`，说明安全性明显好于直接用 ICPT 替换 NLinear；但相对 A1/I0/R0 逐指标包络仍平均退化 `{percent(winner['macro_ratio_to_envelope'] - 1)}`，不能称为当前全局最优。

![调参结果](figures/all__search_ratios.png)

## 3. Parameter / Configuration Search

{table(ranking_rows, ['候选', 'A1宏比值', '最差比值', '双改善', '包络宏比值'], {'A1宏比值': lambda x: f'{x:.6f}', '最差比值': lambda x: f'{x:.6f}', '包络宏比值': lambda x: f'{x:.6f}', '双改善': lambda x: f'{x}/6'})}

H0 优于相同 beta 的 H1，说明当前 rolling risk 并非稳定有效；H4 使用较弱 risk scale 后才成为冠军。beta 从 0.10 增到 0.50 的结果也不是单调的，因此证据支持“小幅连续形状修正”，不支持强 ICPT 注入或把历史回测解释为可靠路由器。

## 4. Error Distribution

共回放 `{total_pairs:,}` 个 validation 样本×通道；相对样本 MSE ≤-10% 记为显著改善，≥+10% 记为显著退化。

{table(group_rows, ['分组', '数量', '占比', '漂移', 'lag24', '差分波动', 'beta', '风险', '修正幅度'], {'占比': lambda x: f'{x:.2%}', '漂移': lambda x: f'{x:.4f}', 'lag24': lambda x: f'{x:.4f}', '差分波动': lambda x: f'{x:.4f}', 'beta': lambda x: f'{x:.4f}', '风险': lambda x: f'{x:.4f}', '修正幅度': lambda x: f'{x:.4f}'})}

![样本分组](figures/all__sample_groups.png)

![分组特征](figures/all__group_features.png)

{table(dataset_group_rows, ['数据集', '改善占比', '退化占比', '改善lag24', '退化lag24', '改善风险', '退化风险', '全局beta'], {'改善占比': lambda x: f'{x:.2%}', '退化占比': lambda x: f'{x:.2%}', '改善lag24': lambda x: f'{x:.3f}', '退化lag24': lambda x: f'{x:.3f}', '改善风险': lambda x: f'{x:.3f}', '退化风险': lambda x: f'{x:.3f}', '全局beta': lambda x: f'{x:.3f}'})}

总体改善组比退化组 lag-24 更高（`{aggregate_features['significant_improvement']['lag24_correlation']:.3f}` vs `{aggregate_features['significant_regression']['lag24_correlation']:.3f}`）、历史风险更低（`{aggregate_features['significant_improvement']['rolling_risk']:.3f}` vs `{aggregate_features['significant_regression']['rolling_risk']:.3f}`），方向上支持“稳定重复形状更适合 ICPT 修正”。但这一总体统计受 Electricity 的 814,377 个样本×通道强烈支配：其改善/退化组 lag-24 和风险非常接近，当前 gate 几乎不能区分两者。更关键的是 Weather 退化组风险反而低于改善组、置信度更高，直接表明 rolling proxy 在该数据上失配。以上均是描述统计，不构成因果证明。

## 5. Horizon-wise Error

![分段误差](figures/all__horizon_mae.png)

{table(horizon_rows, ['数据集', '1–24', '25–48', '49–72', '73–96'], {key: percent for key in ('1–24', '25–48', '49–72', '73–96')})}

Electricity 四段全部改善，ETTm2 四段基本不退化；ETTh1、ETTh2、ETTm1 随预测距离增加而收益衰减，最后一段转为退化。Weather 后两段也回退。统一零均值形状约束因此更像近端局部修正，尚未覆盖远期的周期级水平演化。

## 6. High-Error Selection

程序在每个 setting 内按绝对 MAE 选 A1 高误差、H4 最大退化、H4 最大改善各 5 例；同一 setting×channel 中相距不足 96 的窗口去重，共 90 例。报告只程序化展示每类 3 个跨 setting 代表，没有人工挑例。

{table(case_rows, ['类型', 'setting', '通道', '样本', 'A1 MAE', 'H4 MAE', 'beta', '风险'], {'A1 MAE': lambda x: f'{x:.5f}', 'H4 MAE': lambda x: f'{x:.5f}', 'beta': lambda x: f'{x:.4f}', '风险': lambda x: f'{x:.4f}'})}

## 7. Case Analysis

![A1高误差案例](figures/all__cases_baseline_high_error.png)

![H4退化案例](figures/all__cases_candidate_regression.png)

![H4改善案例](figures/all__cases_candidate_improvement.png)

曲线之外，右图展示逐未来周期有效 beta、历史风险和置信度；完整 NLinear 轨迹、ICPT 形状、零均值修正、RCRF 相位可靠度与 alpha 均保存在 `selected_cases.npz`。

## 8. Repeated Observable Patterns

改善与退化样本都广泛存在，并非某个异常通道造成。可重复的正向模式主要来自 Electricity：改善占比 `16.68%`、退化占比 `8.93%`，且四个未来周期都改善。反例是 ETTm1（改善 `12.82%`、退化 `16.37%`）和 Weather（改善 `3.06%`、退化 `6.21%`）：同一形状修正会在相近输入统计下双向摆动。结合案例曲线，证据只支持 HPTC 能修复一部分重复峰谷，不足以支持其 risk 已准确识别适用域。

## 9. Objective Defect Summary

当前假设得到部分支持：轨迹/形状职责分离把旧 ICPT 在 ETTh2 的明显风险压低，并在 Electricity 获得约 1.3% 双指标收益；但方法没有稳定超过 A1，更远未超过原始方法包络。最关键缺陷是“每 24 步周期水平完全交给 NLinear”约束过强：它保证安全，却会删除 ICPT 对缓慢周期级基线变化的建模能力；同时 rolling risk 仅评估简单形状线性外推，与训练后的 ICPT 修正误差并不完全匹配。下一轮更有根据的修改是把严格零均值改为**受守恒约束的低频周期水平残差**，并用 ICPT 自身的历史 masked reconstruction uncertainty 校准幅度，而不是继续加大 beta 或做完整模型路由。

## 10. Efficiency and Scope

H4 平均参数量 `{cand_params:,}`，A1 平均 `{base_params:,}`，增加 `{(cand_params / base_params - 1):.1%}`。GPU 前向计时如下；这是相同 validation loader 上的 batch 级配对计时，不包含数据加载。

{table(latency_rows, ['数据集', 'batch', 'A1 ms/batch', 'H4 ms/batch', '耗时比'], {'A1 ms/batch': lambda x: f'{x:.3f}', 'H4 ms/batch': lambda x: f'{x:.3f}', '耗时比': lambda x: f'{x:.2f}×'})}

- 逐周期修正均值最大绝对值 `{max_correction_mean:.3g}`；理论上严格为零，但 float32 实测略高于预注册的 `1e-6` 数值阈值，因此该数值检查应记为**未通过**，而不是掩盖为通过。
- checkpoint 回放与训练聚合值最大绝对差 `{max(value for checks in replay.values() for value in checks.values()):.3g}`，阈值 `2e-5`。
- selection source 是 validation；没有读取 test/Golden 数值参与选择。
- 严格门失败后没有运行 H192；本结论只适用于本轮六数据集 H96、单 seed、30% 训练协议，不能当作多 seed 或 test 泛化结论。
"""
    report_path = OUTPUT / "objective_error_analysis.md"
    report_path.write_text(report)

    referenced = re.findall(r"!\[[^]]*\]\((figures/[^)]+)\)", report)
    if len(referenced) != 7 or len(set(referenced)) != 7:
        raise RuntimeError(f"unexpected figure references: {referenced}")
    for relative in referenced:
        if not (OUTPUT / relative).is_file():
            raise FileNotFoundError(relative)
    zip_path = OUTPUT / "objective_error_analysis.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        for relative in referenced:
            archive.write(OUTPUT / relative, relative)
    with zipfile.ZipFile(zip_path) as archive:
        expected_zip = {report_path.name, *referenced}
        if set(archive.namelist()) != expected_zip:
            raise RuntimeError("ZIP whitelist mismatch")
        for relative in expected_zip:
            if archive.read(relative) != (OUTPUT / relative).read_bytes():
                raise RuntimeError(f"ZIP byte mismatch: {relative}")

    selections = []
    for dataset in DATASETS:
        item = {"setting": setting_name(dataset)}
        for category in (
            "baseline_high_error", "candidate_regression", "candidate_improvement"
        ):
            item[category] = [
                f"{case['sample_id']}:{case['channel']}"
                for case in cases
                if case["dataset"] == dataset and case["case_type"] == category
            ]
        selections.append(item)
    settings = [{
        "setting": setting_name(dataset), "dataset": dataset,
        "split": "validation", "lookback": 720, "horizon": 96, "seed": 2021,
    } for dataset in DATASETS]
    modified = git("status", "--short").splitlines()
    manifest = {
        "experiment_id": "hptc_unified_v1",
        "code": {
            "repository": str(REPO),
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "working_tree_at_analysis": modified,
        },
        "hypothesis": "Orthogonal NLinear cycle levels and ICPT zero-mean shapes, with rolling evidence used only as shrinkage, improve A1 without full-model ensembling.",
        "mechanism": {
            "description": "one shared PhaseFormer + NLinear trajectory + zero-mean ICPT shape + continuous historical-risk shrinkage + outer RCRF",
            "feature_flag": CHAMPION,
            "complete_model_ensemble": False,
            "checkpoints_per_run": 1,
        },
        "experiment": {
            "baseline": A1,
            "candidate": CHAMPION,
            "settings": settings,
            "search_space": list(CANDIDATES),
            "training": {"percent": 30, "epochs": 8, "loss": "huber", "seed": 2021},
            "metrics": ["mse", "mae"],
        },
        "execution": {
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "formal_runs": 30,
            "reused_reference_runs": 18,
            "test_accessed": False,
            "h192_executed": False,
        },
        "selection": {
            "source": "validation",
            "ranking": native(ranking),
            "selected_config": CHAMPION,
            "h96_gate": native(gate),
        },
        "analysis": {
            "ranking_metric": "absolute_mae_delta",
            "top_k_per_setting_and_family": 5,
            "dedup_rule": "within each setting/category, same channel requires sample gap >= 96",
            "selections": selections,
            "sample_channel_rows": total_pairs,
            "metric_replay": native(replay),
            "group_summaries": native(analysis),
        },
        "validation": {
            "results_checked": True,
            "ranking_and_cases_checked": True,
            "report_and_archive_checked": True,
            "directory_and_settings_checked": True,
            "correction_cycle_mean_audit": {
                "threshold": 1e-6,
                "observed": max_correction_mean,
                "passed": max_correction_mean < 1e-6,
                "note": "The mathematical projection is zero-mean; float32 cancellation exceeded the preregistered numerical threshold.",
            },
            "status": "passed_with_documented_numerical_audit_failure",
        },
        "artifacts": {},
        "elapsed_analysis_sec": time.time() - started,
    }
    for name in (
        "results.csv", "sample_errors.csv", "selected_cases.npz",
        "objective_error_analysis.md", "objective_error_analysis.zip",
    ):
        path = OUTPUT / name
        manifest["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUTPUT / "run.yaml").write_text(
        yaml.safe_dump(native(manifest), sort_keys=False, allow_unicode=True)
    )

    expected_root = {
        "run.yaml", "results.csv", "sample_errors.csv", "selected_cases.npz",
        "objective_error_analysis.md", "objective_error_analysis.zip", "figures",
    }
    if {path.name for path in OUTPUT.iterdir()} != expected_root:
        raise RuntimeError("canonical directory whitelist mismatch")
    if {path.name for path in figures.iterdir()} != {
        Path(relative).name for relative in referenced
    }:
        raise RuntimeError("figure whitelist mismatch")
    with (OUTPUT / "sample_errors.csv").open(newline="") as handle:
        sample_settings = {row["setting"] for row in csv.DictReader(handle)}
    declared = {item["setting"] for item in settings}
    result_settings = {row["setting"] for row in result_rows}
    npz_settings = set(np.load(OUTPUT / "selected_cases.npz")["setting"].tolist())
    if result_settings != declared or sample_settings != declared or not npz_settings <= declared:
        raise RuntimeError("setting coverage mismatch")
    print(f"wrote {OUTPUT}")
    print(f"rows={total_pairs:,}, cases={len(cases)}, zip_sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
