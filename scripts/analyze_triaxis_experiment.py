#!/usr/bin/env python3
"""Build the canonical validation-only audit for TriAxis experiment v1.

The script consumes the 20 preregistered Stage-A runs, reloads only validation
data, computes expert/oracle/router diagnostics, selects non-overlapping cases,
and emits the strict experiment-and-error-analysis artifact bundle.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import re
import subprocess
import sys
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
SCRATCH = REPO / "research_runs" / "triaxis_self_validating_v1_scratch" / "runs"
OUTPUT = REPO / "research_runs" / "triaxis_self_validating_v1"
DATASETS = ("ETTh2", "ETTm2", "Weather", "Electricity")
MODES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_uniform",
    "triaxis_structural",
    "triaxis_self_validating",
)
LABELS = {
    "gold_combo_reliability_s2": "A1 RCRF+NLinear",
    "rcrf_icpt_none": "I0 RCRF+ICPT",
    "triaxis_uniform": "T0 均匀三轴",
    "triaxis_structural": "T1 结构路由",
    "triaxis_self_validating": "T2 历史自验证",
}
EXPERT_NAMES = ("phase", "trajectory", "cycle")
SETTING = {d: f"{d}-L720-H96-s2021-val30" for d in DATASETS}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # The installed TTC reports the JP family name but contains the full
        # CJK glyph set; using the exact registered name avoids tofu squares.
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def find_run(dataset, mode):
    matches = []
    for path in SCRATCH.glob("*/metrics.csv"):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            continue
        row = frame.iloc[0]
        if row.dataset == dataset and row.mechanism == mode:
            matches.append((path.parent, row.to_dict()))
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {dataset}/{mode}, got {len(matches)}")
    run_dir, row = matches[0]
    spec = json.loads((run_dir / "config.json").read_text())
    return run_dir, row, spec


def build_model_and_loader(run_dir, row, spec, batch_size):
    from src.dataset.data_factory import data_provider
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

    hp = spec["hyperparams"]
    args = make_exp_args(
        spec["dataset"], spec["lookback"], spec["horizon"], hp,
        batch_size=batch_size,
    )
    args.dataset_args.percent = spec["percent"]
    args.dataset_args.num_workers = 0
    args.training_args.num_workers = 0
    dataset, loader = data_provider(args.dataset_args, "val")
    model = PhaseFormer(
        PhaseFormerPresetConfig(args, spec["lookback"], spec["horizon"], hp)
    )
    checkpoint = REPO / row["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    return model, dataset, loader


def variable_names(spec):
    path = REPO / spec["hyperparams"].get("root_path", "")
    if not path.exists():
        path = REPO / spec.get("root_path", "")
    # Paths are authoritative in the experiment spec's dataset arguments.
    from src.models.phaseformer_presets import make_exp_args

    args = make_exp_args(
        spec["dataset"], spec["lookback"], spec["horizon"],
        spec["hyperparams"],
    )
    csv_path = REPO / args.dataset_args.root_path / args.dataset_args.data_path
    columns = [x for x in pd.read_csv(csv_path, nrows=0).columns if x != "date"]
    target = args.dataset_args.target
    return [x for x in columns if x != target] + ([target] if target in columns else [])


def history_features(x, period=24, eps=1e-6):
    scale = x.std(dim=1, unbiased=False).clamp_min(eps)
    drift = (
        (x[:, -period:, :].mean(dim=1) - x[:, -2 * period : -period, :].mean(dim=1)).abs()
        / scale
    ).mean(dim=1)
    highfreq = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1).div(scale).mean(dim=1)
    left = x[:, period:, :]
    right = x[:, :-period, :]
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    corr = (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt() * right.square().mean(dim=1).sqrt() + eps
    )
    return drift, highfreq, corr.mean(dim=1)


def push_case(heap, limit, score, serial, case):
    item = (float(score), int(serial), case)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def local_top(score, limit=8, positive=False):
    flat = score.flatten()
    if positive:
        valid = flat > 0
        if not valid.any():
            return []
        indices = valid.nonzero(as_tuple=False).flatten()
        values = flat[indices]
        k = min(limit, values.numel())
        top = torch.topk(values, k)
        return [(float(v), int(indices[i])) for v, i in zip(top.values, top.indices)]
    k = min(limit, flat.numel())
    top = torch.topk(flat, k)
    return [(float(v), int(i)) for v, i in zip(top.values, top.indices)]


def make_case(
    category, score, setting, dataset_name, sample_index, variable_index,
    names, timestamps, x, truth, baseline, candidate, experts, weights,
    risks, structural,
):
    start = sample_index + x.shape[0]
    timestamp = str(timestamps[start]) if timestamps is not None and start < len(timestamps) else "unavailable"
    true = truth.detach().cpu().numpy()
    base = baseline.detach().cpu().numpy()
    cand = candidate.detach().cpu().numpy()
    expert = experts.detach().cpu().numpy()
    return {
        "case_type": category,
        "score": float(score),
        "setting": setting,
        "dataset": dataset_name,
        "sample_index": int(sample_index),
        "variable_index": int(variable_index),
        "variable": names[variable_index] if variable_index < len(names) else str(variable_index),
        "forecast_start_timestamp": timestamp,
        "history": x.detach().cpu().numpy(),
        "truth": true,
        "baseline": base,
        "candidate": cand,
        "experts": expert,
        "weights": weights.detach().cpu().numpy(),
        "risks": risks.detach().cpu().numpy(),
        "structural": structural.detach().cpu().numpy(),
        "baseline_mse": float(np.mean((base - true) ** 2)),
        "candidate_mse": float(np.mean((cand - true) ** 2)),
        "baseline_mae": float(np.mean(np.abs(base - true))),
        "candidate_mae": float(np.mean(np.abs(cand - true))),
    }


def evaluate_setting(dataset_name, baseline_info, candidate_info, device):
    base_dir, base_row, base_spec = baseline_info
    cand_dir, cand_row, cand_spec = candidate_info
    eval_batch = 16 if dataset_name == "Electricity" else (64 if dataset_name == "Weather" else 256)
    baseline, dataset, loader = build_model_and_loader(
        base_dir, base_row, base_spec, eval_batch
    )
    candidate, dataset2, loader2 = build_model_and_loader(
        cand_dir, cand_row, cand_spec, eval_batch
    )
    if len(dataset) != len(dataset2):
        raise RuntimeError("baseline and candidate validation sets differ")
    # Use one loader; both configs share the exact split and data transform.
    del loader2
    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    names = variable_names(cand_spec)
    timestamps = getattr(dataset, "timestamps", None)

    totals = defaultdict(float)
    sample_rows = []
    pools = {x: [] for x in ("baseline_high_error", "candidate_degraded", "candidate_improved")}
    serial = 0
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [x.to(device) for x in batch]
            batch_x = batch_x.float()
            truth = batch_y.float()[:, -96:, :]
            dec = baseline._build_decoder_input(batch_y.float())
            base_pred, _, _ = baseline(
                batch_x, batch_x_mark.float(), dec, batch_y_mark.float()
            )
            cand_pred, _, _ = candidate(
                batch_x, batch_x_mark.float(), dec, batch_y_mark.float()
            )
            experts = torch.stack(candidate.triaxis_expert_outputs, dim=-1)
            weights = candidate.triaxis_weights
            risks = candidate.triaxis_router.last_risks
            structural = candidate.triaxis_router.last_structural

            base_abs = (base_pred - truth).abs()
            cand_abs = (cand_pred - truth).abs()
            expert_abs = (experts - truth.unsqueeze(-1)).abs()
            count = truth.numel()
            totals["count"] += count
            totals["base_abs"] += base_abs.sum().item()
            totals["base_sq"] += base_abs.square().sum().item()
            totals["cand_abs"] += cand_abs.sum().item()
            totals["cand_sq"] += cand_abs.square().sum().item()
            for i, name in enumerate(EXPERT_NAMES):
                totals[f"{name}_abs"] += expert_abs[..., i].sum().item()
                totals[f"{name}_sq"] += expert_abs[..., i].square().sum().item()
                totals[f"weight_{name}"] += weights[..., i].sum().item()
                totals[f"winner_{name}"] += (expert_abs.argmin(dim=-1) == i).sum().item()
            oracle_abs = expert_abs.min(dim=-1).values
            lower = experts.min(dim=-1).values
            upper = experts.max(dim=-1).values
            convex_abs = torch.where(
                (truth >= lower) & (truth <= upper),
                torch.zeros_like(truth), oracle_abs,
            )
            totals["oracle_abs"] += oracle_abs.sum().item()
            totals["oracle_sq"] += oracle_abs.square().sum().item()
            totals["convex_abs"] += convex_abs.sum().item()
            totals["convex_sq"] += convex_abs.square().sum().item()
            totals["route_agree"] += (
                weights.argmax(dim=-1) == expert_abs.argmin(dim=-1)
            ).sum().item()
            totals["entropy"] += (
                -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).sum().item()
            )

            sample_base_mse = base_abs.square().mean(dim=(1, 2))
            sample_cand_mse = cand_abs.square().mean(dim=(1, 2))
            sample_base_mae = base_abs.mean(dim=(1, 2))
            sample_cand_mae = cand_abs.mean(dim=(1, 2))
            sample_expert_mse = expert_abs.square().mean(dim=(1, 2))
            sample_weights = weights.mean(dim=(1, 2))
            sample_risks = risks.mean(dim=(1, 2))
            sample_structural = structural.mean(dim=(1, 2))
            drift, highfreq, lag24 = history_features(batch_x)
            rel = (sample_cand_mse - sample_base_mse) / sample_base_mse.clamp_min(1e-8)
            for b in range(batch_x.shape[0]):
                relative = float(rel[b])
                group = "candidate_improved" if relative <= -0.10 else (
                    "candidate_degraded" if relative >= 0.10 else "comparable"
                )
                sample_rows.append(
                    {
                        "setting": SETTING[dataset_name],
                        "split": "validation",
                        "sample_index": offset + b,
                        "baseline_mse": float(sample_base_mse[b]),
                        "candidate_mse": float(sample_cand_mse[b]),
                        "baseline_mae": float(sample_base_mae[b]),
                        "candidate_mae": float(sample_cand_mae[b]),
                        "relative_mse_change": relative,
                        "group": group,
                        "recent_drift": float(drift[b]),
                        "high_frequency_ratio": float(highfreq[b]),
                        "lag24_correlation": float(lag24[b]),
                        "phase_reliability": float(sample_structural[b, 0]),
                        "level_drift_evidence": float(sample_structural[b, 1]),
                        "shape_innovation": float(sample_structural[b, 2]),
                        **{f"expert_{name}_mse": float(sample_expert_mse[b, i]) for i, name in enumerate(EXPERT_NAMES)},
                        **{f"weight_{name}": float(sample_weights[b, i]) for i, name in enumerate(EXPERT_NAMES)},
                        **{f"pseudo_risk_{name}": float(sample_risks[b, i]) for i, name in enumerate(EXPERT_NAMES)},
                    }
                )

            pair_base = base_abs.square().mean(dim=1)
            pair_cand = cand_abs.square().mean(dim=1)
            pair_relative = (pair_cand - pair_base) / pair_base.clamp_min(1e-8)
            category_scores = {
                "baseline_high_error": pair_base,
                "candidate_degraded": pair_relative,
                "candidate_improved": -pair_relative,
            }
            B, _, C = truth.shape
            for category, score_tensor in category_scores.items():
                for score, flat_index in local_top(
                    score_tensor, 8, positive=category != "baseline_high_error"
                ):
                    b = flat_index // C
                    c = flat_index % C
                    case = make_case(
                        category, score, SETTING[dataset_name], dataset_name,
                        offset + b, c, names, timestamps,
                        batch_x[b, :, c], truth[b, :, c], base_pred[b, :, c],
                        cand_pred[b, :, c], experts[b, :, c, :].T,
                        weights[b, :, c, :], risks[b, c, :, :],
                        structural[b, c, :, :],
                    )
                    push_case(pools[category], 8, score, serial, case)
                    serial += 1
            offset += B

    denominator = totals["count"]
    summary = {
        "setting": SETTING[dataset_name],
        "dataset": dataset_name,
        "validation_samples": len(dataset),
        "baseline_mse": totals["base_sq"] / denominator,
        "baseline_mae": totals["base_abs"] / denominator,
        "candidate_mse": totals["cand_sq"] / denominator,
        "candidate_mae": totals["cand_abs"] / denominator,
        "oracle_top1_mse": totals["oracle_sq"] / denominator,
        "oracle_top1_mae": totals["oracle_abs"] / denominator,
        "oracle_convex_mse": totals["convex_sq"] / denominator,
        "oracle_convex_mae": totals["convex_abs"] / denominator,
        "route_agreement": totals["route_agree"] / denominator,
        "route_entropy": totals["entropy"] / denominator,
    }
    for name in EXPERT_NAMES:
        summary[f"expert_{name}_mse"] = totals[f"{name}_sq"] / denominator
        summary[f"expert_{name}_mae"] = totals[f"{name}_abs"] / denominator
        summary[f"weight_{name}"] = totals[f"weight_{name}"] / denominator
        summary[f"winner_{name}"] = totals[f"winner_{name}"] / denominator
    candidates = {
        category: [x[2] for x in sorted(heap, reverse=True)]
        for category, heap in pools.items()
    }
    del baseline, candidate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, sample_rows, candidates


def gate_decision(run_rows):
    by = {(x["dataset"], x["mechanism"]): x for x in run_rows}
    decisions = {}
    for candidate in MODES[2:]:
        ratios = []
        both = 0
        for dataset in DATASETS:
            base_mse = min(
                float(by[(dataset, MODES[0])]["val_mse"]),
                float(by[(dataset, MODES[1])]["val_mse"]),
            )
            base_mae = min(
                float(by[(dataset, MODES[0])]["val_mae"]),
                float(by[(dataset, MODES[1])]["val_mae"]),
            )
            rmse = float(by[(dataset, candidate)]["val_mse"]) / base_mse
            rmae = float(by[(dataset, candidate)]["val_mae"]) / base_mae
            ratios.extend((rmse, rmae))
            both += int(rmse < 1.0 and rmae < 1.0)
        macro = float(np.mean(ratios))
        worst = float(np.max(ratios))
        passed = (macro < 0.995 and worst <= 1.005) or (both >= 3 and worst <= 1.005)
        decisions[candidate] = {
            "macro_ratio": macro, "worst_ratio": worst,
            "both_improved_settings": both, "passed": passed,
        }
    return decisions


def select_cases(case_candidates, analysis):
    selected = []
    used = set()
    for category in ("baseline_high_error", "candidate_degraded", "candidate_improved"):
        options = []
        for dataset in DATASETS:
            for case in case_candidates[dataset][category]:
                if category == "baseline_high_error":
                    case["normalized_score"] = case["score"] / analysis[dataset]["baseline_mse"]
                else:
                    case["normalized_score"] = case["score"]
                options.append(case)
        options.sort(key=lambda x: x["normalized_score"], reverse=True)
        chosen_datasets = set()
        for distinct_only in (True, False):
            for case in options:
                key = (case["setting"], case["sample_index"], case["variable_index"])
                if key in used or (distinct_only and case["dataset"] in chosen_datasets):
                    continue
                selected.append(case)
                used.add(key)
                chosen_datasets.add(case["dataset"])
                if sum(x["case_type"] == category for x in selected) == 3:
                    break
            if sum(x["case_type"] == category for x in selected) == 3:
                break
    return selected


def save_npz(path, cases):
    np.savez_compressed(
        path,
        setting=np.asarray([x["setting"] for x in cases]),
        case_type=np.asarray([x["case_type"] for x in cases]),
        dataset=np.asarray([x["dataset"] for x in cases]),
        sample_index=np.asarray([x["sample_index"] for x in cases]),
        variable_index=np.asarray([x["variable_index"] for x in cases]),
        variable=np.asarray([x["variable"] for x in cases]),
        forecast_start_timestamp=np.asarray([x["forecast_start_timestamp"] for x in cases]),
        history=np.stack([x["history"] for x in cases]),
        truth=np.stack([x["truth"] for x in cases]),
        baseline=np.stack([x["baseline"] for x in cases]),
        candidate=np.stack([x["candidate"] for x in cases]),
        experts=np.stack([x["experts"] for x in cases]),
        weights=np.stack([x["weights"] for x in cases]),
        risks=np.stack([x["risks"] for x in cases]),
        structural=np.stack([x["structural"] for x in cases]),
    )


def plot_stage_a(run_rows, output):
    by = {(x["dataset"], x["mechanism"]): x for x in run_rows}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(DATASETS))
    width = 0.24
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for i, mode in enumerate(MODES[2:]):
        ratios = []
        for dataset in DATASETS:
            mse0 = min(float(by[(dataset, MODES[0])]["val_mse"]), float(by[(dataset, MODES[1])]["val_mse"]))
            mae0 = min(float(by[(dataset, MODES[0])]["val_mae"]), float(by[(dataset, MODES[1])]["val_mae"]))
            ratios.append(100 * (0.5 * (float(by[(dataset, mode)]["val_mse"]) / mse0 + float(by[(dataset, mode)]["val_mae"]) / mae0) - 1))
        ax.bar(x + (i - 1) * width, ratios, width, label=LABELS[mode], color=colors[i])
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("相对 A1/I0 较优指标的平均变化（%）\n负值更好")
    ax.set_title("Stage A：三轴候选未形成跨数据集稳定提升")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_oracle(analysis, output):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(DATASETS))
    mse_gain, mae_gain = [], []
    for dataset in DATASETS:
        row = analysis[dataset]
        best_mse = min(row[f"expert_{n}_mse"] for n in EXPERT_NAMES)
        best_mae = min(row[f"expert_{n}_mae"] for n in EXPERT_NAMES)
        mse_gain.append(100 * (1 - row["oracle_top1_mse"] / best_mse))
        mae_gain.append(100 * (1 - row["oracle_top1_mae"] / best_mae))
    ax.bar(x - 0.18, mse_gain, 0.36, label="MSE oracle headroom", color="#4C78A8")
    ax.bar(x + 0.18, mae_gain, 0.36, label="MAE oracle headroom", color="#E45756")
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("相对最佳单专家改善（%）")
    ax.set_title("专家互补性存在，但当前路由没有兑现 oracle 上界")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_routing(analysis, output):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(DATASETS))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    labels = ("相位", "轨迹", "周期间")
    for name, label, color in zip(EXPERT_NAMES, labels, colors):
        values = np.asarray([analysis[d][f"weight_{name}"] for d in DATASETS])
        ax.bar(DATASETS, values, bottom=bottom, label=label, color=color)
        bottom += values
    for i, dataset in enumerate(DATASETS):
        ax.text(i, 1.02, f"H={analysis[dataset]['route_entropy']:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.11)
    ax.set_ylabel("平均路由权重")
    ax.set_title("T2 路由分配（H 为熵；均匀三专家熵为 1.099）")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_cases(cases, category, output):
    subset = [x for x in cases if x["case_type"] == category]
    fig, axes = plt.subplots(len(subset), 2, figsize=(12, 3.3 * len(subset)), squeeze=False)
    titles = {
        "baseline_high_error": "A1 高误差样本",
        "candidate_degraded": "T2 显著退化样本",
        "candidate_improved": "T2 显著改善样本",
    }
    for row, case in enumerate(subset):
        ax, gate_ax = axes[row]
        hist = case["history"][-96:]
        ax.plot(np.arange(-len(hist), 0), hist, color="#777777", linewidth=1.0, label="历史")
        future = np.arange(len(case["truth"]))
        ax.plot(future, case["truth"], color="black", linewidth=1.8, label="真实")
        ax.plot(future, case["baseline"], color="#4C78A8", linewidth=1.4, label="A1")
        ax.plot(future, case["candidate"], color="#E45756", linewidth=1.4, label="T2")
        for expert, label, color in zip(case["experts"], ("相位", "轨迹", "周期间"), ("#72B7B2", "#F2CF5B", "#54A24B")):
            ax.plot(future, expert, color=color, linewidth=0.9, alpha=0.72, linestyle="--", label=label)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(
            f"{case['dataset']} / {case['variable']} / 样本 {case['sample_index']}\n"
            f"MSE: A1 {case['baseline_mse']:.4f}, T2 {case['candidate_mse']:.4f}"
        )
        ax.grid(alpha=0.18)
        if row == 0:
            ax.legend(ncol=6, fontsize=7, loc="upper center")
        for i, (label, color) in enumerate(zip(("相位权重", "轨迹权重", "周期间权重"), ("#4C78A8", "#F58518", "#54A24B"))):
            gate_ax.plot(future, case["weights"][:, i], label=label, color=color)
        risk_text = ", ".join(
            f"{n}={v:.2f}" for n, v in zip(("相位风险", "轨迹风险", "周期风险"), case["risks"].mean(axis=0))
        )
        gate_ax.set_title(f"路由权重；历史伪风险均值：{risk_text}")
        gate_ax.set_ylim(0, 1)
        gate_ax.grid(alpha=0.18)
        if row == 0:
            gate_ax.legend(ncol=3, fontsize=8)
    fig.suptitle(titles[category], fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            values.append(formats.get(col, lambda x: str(x))(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_decimal(value, digits=4):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"canonical output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    figures = OUTPUT / "figures"
    figures.mkdir()

    run_info = {}
    run_rows = []
    for dataset in DATASETS:
        for mode in MODES:
            info = find_run(dataset, mode)
            run_info[(dataset, mode)] = info
            row = dict(info[1])
            row["setting"] = SETTING[dataset]
            row["split"] = "validation"
            run_rows.append(row)
    decisions = gate_decision(run_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis = {}
    all_sample_rows = []
    case_candidates = {}
    for dataset in DATASETS:
        summary, sample_rows, pools = evaluate_setting(
            dataset,
            run_info[(dataset, "gold_combo_reliability_s2")],
            run_info[(dataset, "triaxis_self_validating")],
            device,
        )
        analysis[dataset] = summary
        all_sample_rows.extend(sample_rows)
        case_candidates[dataset] = pools

    # Stage-0 oracle gate: 8 metric cells against the best atomic expert.
    oracle_ratios = []
    for dataset in DATASETS:
        row = analysis[dataset]
        oracle_ratios.extend(
            (
                row["oracle_top1_mse"] / min(row[f"expert_{n}_mse"] for n in EXPERT_NAMES),
                row["oracle_top1_mae"] / min(row[f"expert_{n}_mae"] for n in EXPERT_NAMES),
            )
        )
    oracle_macro_improvement = 1.0 - float(np.mean(oracle_ratios))

    # Enrich only T2 rows with diagnostics while keeping all 20 runs in one table.
    for row in run_rows:
        if row["mechanism"] == "triaxis_self_validating":
            row.update(analysis[row["dataset"]])
    fields = sorted({key for row in run_rows for key in row})
    with (OUTPUT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(run_rows)
    pd.DataFrame(all_sample_rows).to_csv(OUTPUT / "sample_errors.csv", index=False)

    cases = select_cases(case_candidates, analysis)
    if len(cases) != 9:
        raise RuntimeError(f"expected 9 selected cases, got {len(cases)}")
    save_npz(OUTPUT / "selected_cases.npz", cases)

    plot_stage_a(run_rows, figures / "stage_a_metric_change.png")
    plot_oracle(analysis, figures / "oracle_headroom.png")
    plot_routing(analysis, figures / "routing_weights.png")
    for category in ("baseline_high_error", "candidate_degraded", "candidate_improved"):
        plot_cases(cases, category, figures / f"cases_{category}.png")

    sample_frame = pd.DataFrame(all_sample_rows)
    proportion_rows = []
    feature_rows = []
    for dataset in DATASETS:
        part = sample_frame[sample_frame.setting == SETTING[dataset]]
        counts = part.group.value_counts(normalize=True)
        proportion_rows.append(
            {
                "dataset": dataset,
                "improved": counts.get("candidate_improved", 0.0),
                "degraded": counts.get("candidate_degraded", 0.0),
                "comparable": counts.get("comparable", 0.0),
            }
        )
        for group in ("candidate_improved", "candidate_degraded"):
            selected = part[part.group == group]
            feature_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "n": len(selected),
                    "drift": selected.recent_drift.mean(),
                    "lag24": selected.lag24_correlation.mean(),
                    "phase_rel": selected.phase_reliability.mean(),
                    "innovation": selected.shape_innovation.mean(),
                }
            )

    result_table = []
    by = {(x["dataset"], x["mechanism"]): x for x in run_rows}
    for dataset in DATASETS:
        for mode in MODES:
            row = by[(dataset, mode)]
            result_table.append(
                {
                    "数据集": dataset,
                    "模型": LABELS[mode],
                    "MSE": float(row["val_mse"]),
                    "MAE": float(row["val_mae"]),
                }
            )
    oracle_table = []
    for dataset in DATASETS:
        row = analysis[dataset]
        oracle_table.append(
            {
                "数据集": dataset,
                "相位MSE": row["expert_phase_mse"],
                "轨迹MSE": row["expert_trajectory_mse"],
                "周期MSE": row["expert_cycle_mse"],
                "Oracle MSE": row["oracle_top1_mse"],
                "路由命中": row["route_agreement"],
            }
        )
    case_table = []
    for case in cases:
        case_table.append(
            {
                "类型": case["case_type"], "setting": case["setting"],
                "样本": case["sample_index"], "变量": case["variable"],
                "A1 MSE": case["baseline_mse"], "T2 MSE": case["candidate_mse"],
            }
        )

    report = f"""# TriAxis-Former 客观实验与错误分析

## 直白结论

三类专家确实互补：逐点知道真实答案时，oracle 相对最佳单专家的 8 个验证指标宏平均可改善
**{oracle_macro_improvement * 100:.2f}%**。但当前 T2 路由没有稳定学会“什么时候信谁”：它只在
Weather、Electricity 胜出，在 ETTh2、ETTm2 回退，未通过预注册冻结门槛，所以实验在 Stage A
停止，**没有读取新 test，也不更新当前 A1/RCRF+NLinear incumbent**。

## 1. 协议和可比性

- split：validation-only；L=720、H=96、P=24、seed=2021、30% train、最多 8 epoch。
- 所有模型共享数据切分、RevIN、Huber loss、学习率和最低 validation loss checkpoint。
- A1/I0/T0/T1/T2 共 20 次训练均完成；本报告的样本分析比较配对重跑的 A1 与 T2。
- 新机制来自已暴露的历史 test 结果，因此先预注册再筛选；Stage A 失败后按规则不进入 test。
- TriAxis 改变了训练目标（专家辅助 0.2、T2 路由 KL 0.1）及 phase 高频抑制的归属，不能把
  本轮 validation 数值与旧 test 表直接混用；A1/I0 在同一轮重跑用于公平配对。

## 2. Stage A 全部结果

{markdown_table(result_table, ['数据集', '模型', 'MSE', 'MAE'], {'MSE': lambda x: f'{x:.6f}', 'MAE': lambda x: f'{x:.6f}'})}

![Stage A 指标变化](figures/stage_a_metric_change.png)

冻结判定：T2 的 8 指标宏平均比值为 **{decisions['triaxis_self_validating']['macro_ratio']:.4f}**，
最差比值 **{decisions['triaxis_self_validating']['worst_ratio']:.4f}**，仅
**{decisions['triaxis_self_validating']['both_improved_settings']}/4** 个 setting 双指标改善；T0、T1
的宏平均/最差比值分别为 {decisions['triaxis_uniform']['macro_ratio']:.4f}/
{decisions['triaxis_uniform']['worst_ratio']:.4f} 和
{decisions['triaxis_structural']['macro_ratio']:.4f}/
{decisions['triaxis_structural']['worst_ratio']:.4f}。三者均失败。

## 3. 专家互补与路由诊断

{markdown_table(oracle_table, ['数据集', '相位MSE', '轨迹MSE', '周期MSE', 'Oracle MSE', '路由命中'], {k: (lambda x: f'{x:.6f}') for k in ['相位MSE', '轨迹MSE', '周期MSE', 'Oracle MSE']} | {'路由命中': lambda x: f'{x:.2%}'})}

![Oracle 上界](figures/oracle_headroom.png)

![T2 路由权重](figures/routing_weights.png)

oracle 门槛通过，说明继续研究“可校准的专家选择”有价值；但它是使用真实标签的不可部署上界，
不是模型成绩。T2 的可部署路由只读取历史，平均熵接近均匀路由熵 1.099，且实际最佳专家命中率
有限：主要失败不是专家完全同质，而是历史伪风险与未来真实相对误差的映射尚不够可靠。

## 4. 样本统计

以每个 validation 窗口的平均 MSE 为单位，T2 相对 A1 改善/退化至少 10% 才进入对应组：

{markdown_table(proportion_rows, ['dataset', 'improved', 'degraded', 'comparable'], {k: (lambda x: f'{x:.1%}') for k in ['improved', 'degraded', 'comparable']})}

改善组和退化组的历史特征均值如下。`drift` 是最近两周期水平变化，`lag24` 是 24 步相关，
`phase_rel` 是同相位信号占比，`innovation` 是最近周期形状变化，均由输入窗口计算：

{markdown_table(feature_rows, ['dataset', 'group', 'n', 'drift', 'lag24', 'phase_rel', 'innovation'], {k: format_decimal for k in ['drift', 'lag24', 'phase_rel', 'innovation']})}

这些统计不支持一个跨数据集统一的简单规则：Weather/Electricity 的专家混合更常有效，而 ETTm2
仍明显偏好轨迹专家。当前三个标量和三个伪风险不足以稳定区分这两类情况。

## 5. 程序化案例

从所有 sample×channel 对中分别按 A1 误差、T2 相对退化、T2 相对改善排序；每类取 3 个，
要求 setting 内三类不重叠，并优先覆盖不同数据集。原始数组、三专家预测、逐步权重和伪风险均在
`selected_cases.npz`。

{markdown_table(case_table, ['类型', 'setting', '样本', '变量', 'A1 MSE', 'T2 MSE'], {'A1 MSE': lambda x: f'{x:.5f}', 'T2 MSE': lambda x: f'{x:.5f}'})}

![A1 高误差案例](figures/cases_baseline_high_error.png)

![T2 退化案例](figures/cases_candidate_degraded.png)

![T2 改善案例](figures/cases_candidate_improved.png)

## 6. 归因与下一步

1. **保留三轴分解，淘汰当前冻结候选。** Weather/Electricity 与 oracle 结果证明 phase、trajectory、
   cycle 三种归纳偏置有互补性；T2 跨数据集稳定性不足，不能替换 A1。
2. **主要问题是代理任务错配。** “最后一周期伪预测”只验证一步周期迁移，却被外推到未来四个周期；
   它不能充分估计长 horizon 上 NLinear 趋势误差和 ICPT 累积误差。
3. **路由容量不是首要瓶颈。** 仅 247 个参数的路由接近均匀，盲目加深 MLP 会削弱科研逻辑；更有
   价值的是多截点 rolling-origin 自验证、按风险校准而非直接分类，以及对路由 regret 的显式约束。
4. **停止边界。** 按预注册协议不运行 Stage B/C，不读取 test，不报告相对 Golden 的新提升。当前
   Golden/RCRF 结论保持不变。
"""
    report_path = OUTPUT / "objective_error_analysis.md"
    report_path.write_text(report)

    run_yaml = {
        "experiment_id": "triaxis_self_validating_v1",
        "status": "stopped_after_stage_a_gate_failure",
        "implementation_commit": git("rev-parse", "HEAD"),
        "execution_note": "Stage-A runs used the exact source diff committed as implementation_commit; checkpoint environment metadata records the pre-commit parent and dirty flag.",
        "test_split_accessed": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": list(SETTING.values()),
        "models": list(MODES),
        "protocol": {
            "lookback": 720, "horizon": 96, "period": 24, "seed": 2021,
            "train_percent": 30, "max_epochs": 8, "loss": "huber",
            "checkpoint_selection": "lowest validation loss",
            "candidate_selection": "validation only",
        },
        "stage0": {
            "oracle_macro_improvement": oracle_macro_improvement,
            "threshold": 0.01,
            "passed": oracle_macro_improvement >= 0.01,
        },
        "stage_a_decisions": decisions,
        "frozen_candidate": None,
        "stop_reason": "T0, T1, and T2 all failed both preregistered Stage-A gates.",
        "reproduce_training_template": "/home/wangjing/miniconda3/envs/raft/bin/python scripts/search_phaseformer.py --dataset <DATASET> --horizon 96 --stage mechanism_screen_2 --mechanism <MODE> --lookback 720 --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --batch-size <BATCH> --num-workers 2 --output-dir research_runs/triaxis_self_validating_v1_scratch",
        "reproduce_audit": "/home/wangjing/miniconda3/envs/raft/bin/python scripts/analyze_triaxis_experiment.py",
        "selected_case_policy": "validation sample-channel pairs; top 3 per category globally, distinct settings preferred, categories non-overlapping; 9 total",
    }
    (OUTPUT / "run.yaml").write_text(yaml.safe_dump(run_yaml, allow_unicode=True, sort_keys=False))

    referenced = sorted(set(re.findall(r"\(figures/([^)]+\.png)\)", report)))
    existing = sorted(x.name for x in figures.glob("*.png"))
    if referenced != existing:
        raise RuntimeError(f"figure reference mismatch: referenced={referenced}, existing={existing}")
    zip_path = OUTPUT / "objective_error_analysis.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, "objective_error_analysis.md")
        for name in referenced:
            archive.write(figures / name, f"figures/{name}")
    with zipfile.ZipFile(zip_path) as archive:
        expected = ["objective_error_analysis.md"] + [f"figures/{x}" for x in referenced]
        if sorted(archive.namelist()) != sorted(expected):
            raise RuntimeError("ZIP member whitelist validation failed")
        if archive.read("objective_error_analysis.md") != report_path.read_bytes():
            raise RuntimeError("ZIP report differs from canonical report")
        for name in referenced:
            if archive.read(f"figures/{name}") != (figures / name).read_bytes():
                raise RuntimeError(f"ZIP figure differs: {name}")
    roots = sorted(x.name for x in OUTPUT.iterdir())
    expected_roots = sorted(
        ["run.yaml", "results.csv", "sample_errors.csv", "selected_cases.npz",
         "objective_error_analysis.md", "objective_error_analysis.zip", "figures"]
    )
    if roots != expected_roots:
        raise RuntimeError(f"canonical root whitelist failed: {roots}")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(REPO)),
        "stage0_oracle_improvement": oracle_macro_improvement,
        "stage_a": decisions,
        "samples": len(all_sample_rows), "selected_cases": len(cases),
        "zip_members": len(referenced) + 1,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
