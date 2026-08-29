#!/usr/bin/env python3
"""Create the strict validation-only audit bundle for Multi-Anchor Selector v1."""

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

from scripts.search_multi_anchor import (  # noqa: E402
    ANCHOR_NAMES,
    MECHANISMS,
    make_model,
    read_reference,
)
from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.multi_anchor import MultiAnchorPhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    PhaseFormerPresetConfig,
    make_exp_args,
)


SCRATCH = REPO / "research_runs" / "multi_anchor_selector_v1_scratch"
REFERENCE_SCRATCH = REPO / "research_runs" / "safe_regret_triaxis_v1_scratch"
CACHE = SCRATCH / "audit_cache"
OUTPUT = REPO / "research_runs" / "multi_anchor_selector_v1"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
REFERENCES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_rolling_features",
)
CANDIDATES = tuple(MECHANISMS)
CHAMPION = "multi_anchor_structural_soft"
LABELS = {
    "gold_combo_reliability_s2": "A1 RCRF+NLinear",
    "rcrf_icpt_none": "I0 RCRF+ICPT",
    "triaxis_rolling_features": "R0 Rolling TriAxis",
    "multi_anchor_global_hard": "M0 global hard",
    "multi_anchor_structural_hard": "M1 structural hard",
    "multi_anchor_guarded_hard": "M2 guarded hard",
    "multi_anchor_structural_soft": "M3 structural soft",
}
GROUPS = ("significant_improvement", "comparable", "significant_regression")
SAMPLE_FIELDS = (
    "setting", "baseline_config_id", "candidate_config_id", "sample_id",
    "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse",
    "relative_delta_mse", "baseline_mae", "candidate_mae", "delta_mae",
    "group", "recent_drift", "lag24_correlation", "difference_volatility",
    "phase_reliability", "weight_a1", "weight_i0", "weight_r0",
    "route_entropy", "hard_oracle_agreement", "oracle_regret",
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


def setting_name(dataset):
    return f"{dataset}-L720-H96-s2021-validation"


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def find_run(root, dataset, mechanism, percent=None, protocol=None):
    matches = []
    for config_path in (Path(root) / "runs").glob("*/config.json"):
        spec = json.loads(config_path.read_text())
        metrics_path = config_path.parent / "metrics.csv"
        if not metrics_path.is_file():
            continue
        actual_percent = spec.get("percent", spec.get("full_percent"))
        if (
            spec.get("dataset") == dataset
            and int(spec.get("horizon", -1)) == 96
            and spec.get("mechanism") == mechanism
            and (percent is None or int(actual_percent) == percent)
            and (protocol is None or spec.get("protocol_version") == protocol)
        ):
            with metrics_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            matches.append((config_path.parent, row, spec))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one run for {dataset}/{mechanism}/pct{percent}, got {len(matches)}"
        )
    return matches[0]


def select_single_baseline(reference_infos):
    min_mse = min(float(info[1]["val_mse"]) for info in reference_infos.values())
    min_mae = min(float(info[1]["val_mae"]) for info in reference_infos.values())
    return min(
        REFERENCES,
        key=lambda mode: (
            float(reference_infos[mode][1]["val_mse"]) / min_mse
            + float(reference_infos[mode][1]["val_mae"]) / min_mae
        ),
    )


def variable_names(exp_args):
    path = Path(exp_args.dataset_args.root_path) / exp_args.dataset_args.data_path
    columns = [name for name in pd.read_csv(path, nrows=0).columns if name != "date"]
    target = exp_args.dataset_args.target
    return [name for name in columns if name != target] + (
        [target] if target in columns else []
    )


def load_candidate(candidate_info, batch_size):
    run_dir, row, spec = candidate_info
    references = {
        bank: {
            name: read_reference(run_path)
            for name, run_path in spec["reference_runs"][bank].items()
        }
        for bank in ("shadow", "full")
    }
    a1_spec = references["full"]["A1"][1]
    hp = dict(a1_spec["hyperparams"])
    exp_args = make_exp_args(
        spec["dataset"], spec["lookback"], spec["horizon"], hp,
        batch_size=batch_size,
    )
    exp_args.dataset_args.percent = spec["full_percent"]
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    dataset, loader = data_provider(exp_args.dataset_args, "val")
    if hasattr(dataset, "data_stamp"):
        hp["time_mark_dim"] = int(dataset.data_stamp.shape[-1])
    config = PhaseFormerPresetConfig(
        exp_args, spec["lookback"], spec["horizon"], hp
    )
    mechanism = spec["mechanism_config"]
    model = MultiAnchorPhaseFormer(
        config,
        {
            name: make_model(references["shadow"][name][1], references["shadow"][name][3])
            for name in ANCHOR_NAMES
        },
        {
            name: make_model(references["full"][name][1], references["full"][name][3])
            for name in ANCHOR_NAMES
        },
        router_mode=mechanism["router_mode"],
        output_mode=mechanism["output_mode"],
        hidden=24,
        temperature=0.2,
        oracle_temperature=0.1,
        route_weight=0.1,
        mean_regret_weight=mechanism["mean_regret_weight"],
        cvar_weight=mechanism["cvar_weight"],
    )
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = REPO / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    return model, dataset, loader, exp_args


def push_case(heap, score, serial, case, limit=32):
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
    history, truth, baseline, candidate, anchors, weights, hard, oracle, features,
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
        "anchors": anchors.detach().cpu().numpy(),
        "weights": weights.detach().cpu().numpy(),
        "hard_choice": hard.detach().cpu().numpy(),
        "oracle_choice": oracle.detach().cpu().numpy(),
        "features": features.detach().cpu().numpy(),
        "baseline_mse": float(np.mean((baseline_np - truth_np) ** 2)),
        "candidate_mse": float(np.mean((candidate_np - truth_np) ** 2)),
        "baseline_mae": float(np.mean(np.abs(baseline_np - truth_np))),
        "candidate_mae": float(np.mean(np.abs(candidate_np - truth_np))),
    }


def evaluate_setting(dataset_name, candidate_info, baseline_mode, writer, device):
    batch_size = 8 if dataset_name == "Electricity" else (
        32 if dataset_name == "Weather" else 128
    )
    model, dataset, loader, exp_args = load_candidate(candidate_info, batch_size)
    model = model.to(device).eval()
    baseline_index = REFERENCES.index(baseline_mode)
    names = variable_names(exp_args)
    timestamps = getattr(dataset, "timestamps", None)
    totals = defaultdict(float)
    group_counts = defaultdict(int)
    group_sums = {group: defaultdict(float) for group in GROUPS}
    route_weight_sum = torch.zeros(3, dtype=torch.float64)
    oracle_sum = torch.zeros(3, dtype=torch.float64)
    horizon_abs = np.zeros((2, 4), dtype=np.float64)
    horizon_count = np.zeros(4, dtype=np.int64)
    pools = {name: [] for name in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    )}
    serial = 0
    offset = 0

    with torch.inference_mode():
        for batch in loader:
            x, y, x_mark, y_mark = [value.to(device) for value in batch]
            x, y = x.float(), y.float()
            truth = y[:, -96:, :]
            output, _, _ = model(
                x, x_mark.float(), model._build_decoder_input(y), y_mark.float()
            )
            anchors = torch.stack(model.last_anchor_outputs, dim=-1)
            baseline = anchors[..., baseline_index]
            weights = model.router.last_soft_weights
            hard = model.router.last_hard_weights.argmax(dim=-1)
            features = model.router.last_features
            anchor_mse_cycles = torch.stack(
                [model._cyclewise_mse(value, truth) for value in model.last_anchor_outputs],
                dim=-1,
            )
            oracle = anchor_mse_cycles.argmin(dim=-1)
            envelope = anchor_mse_cycles.min(dim=-1).values
            candidate_cycles = model._cyclewise_mse(output, truth)
            regret = (candidate_cycles - envelope) / envelope.clamp_min(1e-8)

            for prefix, prediction in (("baseline", baseline), ("candidate", output)):
                error = prediction - truth
                totals[f"{prefix}_sq"] += float(error.square().sum())
                totals[f"{prefix}_abs"] += float(error.abs().sum())
            totals["count"] += truth.numel()
            for anchor_index, name in enumerate(ANCHOR_NAMES):
                error = anchors[..., anchor_index] - truth
                totals[f"{name}_sq"] += float(error.square().sum())
                totals[f"{name}_abs"] += float(error.abs().sum())

            baseline_mse = (baseline - truth).square().mean(dim=1)
            candidate_mse = (output - truth).square().mean(dim=1)
            baseline_mae = (baseline - truth).abs().mean(dim=1)
            candidate_mae = (output - truth).abs().mean(dim=1)
            relative = (candidate_mse - baseline_mse) / baseline_mse.clamp_min(1e-8)
            history_features = features[:, :, :, :4].mean(dim=2)
            mean_weights = weights.mean(dim=2)
            entropy = -(
                weights * weights.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(dim=2)
            agreement = (hard == oracle).float().mean(dim=2)
            mean_regret = regret.mean(dim=2)
            feature_values = {
                "recent_drift": history_features[..., 0],
                "lag24_correlation": history_features[..., 1],
                "difference_volatility": history_features[..., 2],
                "phase_reliability": history_features[..., 3],
                "weight_a1": mean_weights[..., 0],
                "weight_i0": mean_weights[..., 1],
                "weight_r0": mean_weights[..., 2],
                "route_entropy": entropy,
                "hard_oracle_agreement": agreement,
                "oracle_regret": mean_regret,
            }
            masks = {
                "significant_improvement": relative <= -0.10,
                "significant_regression": relative >= 0.10,
                "comparable": (relative > -0.10) & (relative < 0.10),
            }
            for group, mask in masks.items():
                count = int(mask.sum())
                group_counts[group] += count
                for key, values in feature_values.items():
                    group_sums[group][key] += float(values[mask].sum())

            route_weight_sum += weights.double().sum(dim=(0, 1, 2)).cpu()
            oracle_sum += torch.bincount(oracle.reshape(-1).cpu(), minlength=3).double()
            for segment in range(4):
                start, end = segment * 24, (segment + 1) * 24
                horizon_abs[0, segment] += float((baseline[:, start:end] - truth[:, start:end]).abs().sum())
                horizon_abs[1, segment] += float((output[:, start:end] - truth[:, start:end]).abs().sum())
                horizon_count[segment] += truth[:, start:end].numel()

            B, _, C = truth.shape
            rows = []
            for b in range(B):
                start = offset + b + x.shape[1]
                if timestamps is not None and start + 95 < len(timestamps):
                    time_range = f"{timestamps[start]}--{timestamps[start + 95]}"
                else:
                    time_range = f"index:{start}--{start + 95}"
                for c in range(C):
                    rel = float(relative[b, c])
                    group = "significant_improvement" if rel <= -0.10 else (
                        "significant_regression" if rel >= 0.10 else "comparable"
                    )
                    rows.append({
                        "setting": setting_name(dataset_name),
                        "baseline_config_id": baseline_mode,
                        "candidate_config_id": CHAMPION,
                        "sample_id": offset + b,
                        "channel": names[c] if c < len(names) else str(c),
                        "time_range": time_range,
                        "baseline_mse": f"{float(baseline_mse[b, c]):.8g}",
                        "candidate_mse": f"{float(candidate_mse[b, c]):.8g}",
                        "delta_mse": f"{float(candidate_mse[b, c] - baseline_mse[b, c]):.8g}",
                        "relative_delta_mse": f"{rel:.8g}",
                        "baseline_mae": f"{float(baseline_mae[b, c]):.8g}",
                        "candidate_mae": f"{float(candidate_mae[b, c]):.8g}",
                        "delta_mae": f"{float(candidate_mae[b, c] - baseline_mae[b, c]):.8g}",
                        "group": group,
                        **{key: f"{float(value[b, c]):.8g}" for key, value in feature_values.items()},
                    })
            writer.writerows(rows)

            delta_mse = candidate_mse - baseline_mse
            scores = {
                "baseline_high_error": baseline_mse,
                "candidate_regression": delta_mse,
                "candidate_improvement": -delta_mse,
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
                        x[b, :, c], truth[b, :, c], baseline[b, :, c], output[b, :, c],
                        anchors[b, :, c].T, weights[b, c], hard[b, c], oracle[b, c],
                        features[b, c],
                    )
                    push_case(pools[category], score, serial, case)
                    serial += 1
            offset += B

    count = totals["count"]
    route_total = float(route_weight_sum.sum())
    oracle_total = float(oracle_sum.sum())
    summary = {
        "setting": setting_name(dataset_name),
        "dataset": dataset_name,
        "validation_samples": len(dataset),
        "sample_channel_pairs": sum(group_counts.values()),
        "baseline_config": baseline_mode,
        "baseline_mse": totals["baseline_sq"] / count,
        "baseline_mae": totals["baseline_abs"] / count,
        "candidate_mse": totals["candidate_sq"] / count,
        "candidate_mae": totals["candidate_abs"] / count,
        "anchor_metrics": {
            name: {
                "mse": totals[f"{name}_sq"] / count,
                "mae": totals[f"{name}_abs"] / count,
            } for name in ANCHOR_NAMES
        },
        "groups": dict(group_counts),
        "group_feature_means": {
            group: {
                key: group_sums[group][key] / max(group_counts[group], 1)
                for key in feature_values
            } for group in GROUPS
        },
        "mean_route_weights": {
            name: float(route_weight_sum[index]) / route_total
            for index, name in enumerate(ANCHOR_NAMES)
        },
        "oracle_shares": {
            name: float(oracle_sum[index]) / oracle_total
            for index, name in enumerate(ANCHOR_NAMES)
        },
        "horizon_mae": {
            "baseline": (horizon_abs[0] / horizon_count).tolist(),
            "candidate": (horizon_abs[1] / horizon_count).tolist(),
        },
    }
    pools = {
        key: [item[2] for item in sorted(heap, reverse=True)]
        for key, heap in pools.items()
    }
    del model
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
    """Select each case family independently within every setting."""
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
                if duplicate(case, selected + chosen):
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
    """Choose three cross-setting examples per family for readable figures."""
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
        anchor_predictions=np.stack([case["anchors"] for case in cases]),
        route_weights=np.stack([case["weights"] for case in cases]),
        hard_choice=np.stack([case["hard_choice"] for case in cases]),
        oracle_choice=np.stack([case["oracle_choice"] for case in cases]),
        route_features=np.stack([case["features"] for case in cases]),
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


def plot_ratios(rows, path):
    values = np.asarray([[row["mse_ratio"], row["mae_ratio"]] for row in rows]).T
    fig, ax = plt.subplots(figsize=(9, 3.6))
    image = ax.imshow(100 * (values - 1), cmap="RdYlGn_r", vmin=-2.5, vmax=1.0, aspect="auto")
    for i in range(2):
        for j in range(len(rows)):
            ax.text(j, i, f"{100 * (values[i, j] - 1):+.2f}%", ha="center", va="center")
    ax.set_xticks(range(len(rows)), [row["dataset"] for row in rows])
    ax.set_yticks((0, 1), ("MSE", "MAE"))
    ax.set_title("M3 相对 A1/I0/R0 逐指标包络的验证误差变化")
    fig.colorbar(image, ax=ax, label="相对变化（%）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_groups(analysis, path):
    labels, parts = list(DATASETS), {group: [] for group in GROUPS}
    for dataset in DATASETS:
        summary = analysis[dataset]
        total = summary["sample_channel_pairs"]
        for group in GROUPS:
            parts[group].append(summary["groups"].get(group, 0) / total)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, label, color in (
        ("significant_improvement", "显著改善（≤-10%）", "#54A24B"),
        ("comparable", "相近", "#BAB0AC"),
        ("significant_regression", "显著退化（≥+10%）", "#E45756"),
    ):
        ax.bar(x, parts[group], bottom=bottom, label=label, color=color)
        bottom += np.asarray(parts[group])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("样本×通道占比")
    ax.set_title("M3 相对单一最强完整锚点的样本级误差分组")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_routes(analysis, path):
    labels = list(DATASETS)
    weights = np.asarray([
        [analysis[d]["mean_route_weights"][name] for name in ANCHOR_NAMES]
        for d in labels
    ])
    oracle = np.asarray([
        [analysis[d]["oracle_shares"][name] for name in ANCHOR_NAMES]
        for d in labels
    ])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x, width = np.arange(len(labels)), 0.25
    for index, (name, color) in enumerate(zip(ANCHOR_NAMES, ("#4C78A8", "#F58518", "#54A24B"))):
        axes[0].bar(x + (index - 1) * width, weights[:, index], width, label=name, color=color)
        axes[1].bar(x + (index - 1) * width, oracle[:, index], width, label=name, color=color)
    axes[0].set_title("soft 路由平均权重")
    axes[1].set_title("周期级真实 oracle 占比")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("占比")
    axes[0].legend(ncol=3)
    fig.suptitle("路由权重与真实最优锚点的分布并不等价")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_horizon(analysis, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, width = np.arange(4), 0.12
    for index, dataset in enumerate(DATASETS):
        base = np.asarray(analysis[dataset]["horizon_mae"]["baseline"])
        candidate = np.asarray(analysis[dataset]["horizon_mae"]["candidate"])
        change = 100 * (candidate / base - 1)
        ax.bar(x + (index - 2.5) * width, change, width, label=dataset)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ("1–24", "25–48", "49–72", "73–96"))
    ax.set_ylabel("MAE 相对变化（%）")
    ax.set_xlabel("预测区间")
    ax.set_title("未来区间误差变化")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cases(cases, category, path):
    subset = [case for case in cases if case["case_type"] == category]
    fig, axes = plt.subplots(len(subset), 2, figsize=(12, 3.4 * len(subset)), squeeze=False)
    titles = {
        "baseline_high_error": "原始高误差样本",
        "candidate_regression": "M3 显著退化样本",
        "candidate_improvement": "M3 显著改善样本",
    }
    for row, case in enumerate(subset):
        left, right = axes[row]
        left.plot(np.arange(-96, 0), case["history"][-96:], color="#999999", label="近期历史")
        future = np.arange(1, 97)
        left.plot(future, case["truth"], color="black", linewidth=1.6, label="真实")
        left.plot(future, case["baseline"], color="#9467BD", label="单一最强锚点")
        left.plot(future, case["candidate"], color="#E45756", label="M3")
        left.set_title(
            f"{case['dataset']} / {case['channel']} / 样本{case['sample_id']}\n"
            f"MSE {case['baseline_mse']:.4f} → {case['candidate_mse']:.4f}"
        )
        left.grid(alpha=0.2)
        if row == 0:
            left.legend(ncol=4, fontsize=7)
        q = np.arange(1, 5)
        for index, (name, color) in enumerate(zip(ANCHOR_NAMES, ("#4C78A8", "#F58518", "#54A24B"))):
            right.plot(q, case["weights"][:, index], marker="o", color=color, label=f"{name}权重")
        right.scatter(q, np.full(4, 1.04), c=case["oracle_choice"], cmap="viridis", vmin=0, vmax=2, marker="s", label="oracle(颜色)")
        right.set_ylim(-0.03, 1.1)
        right.set_xticks(q)
        right.set_xlabel("未来 24 步周期")
        right.set_title("soft 权重与周期 oracle")
        right.grid(alpha=0.2)
        if row == 0:
            right.legend(ncol=2, fontsize=7)
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
    CACHE.mkdir(exist_ok=True)

    reference_infos = {
        dataset: {
            mode: find_run(REFERENCE_SCRATCH, dataset, mode, 30)
            for mode in REFERENCES
        } for dataset in DATASETS
    }
    candidate_infos = {
        dataset: {
            mode: find_run(
                SCRATCH, dataset, mode, 30, "multi-anchor-selector-v1"
            ) for mode in CANDIDATES
        } for dataset in DATASETS
    }
    shadow_infos = {
        dataset: {
            mode: find_run(SCRATCH, dataset, mode, 24)
            for mode in REFERENCES
        } for dataset in DATASETS
    }
    all_rows = [
        info[1]
        for group in (reference_infos, candidate_infos, shadow_infos)
        for dataset_infos in group.values()
        for info in dataset_infos.values()
    ]
    if len(all_rows) != 60:
        raise RuntimeError(f"expected 60 formal training runs, got {len(all_rows)}")
    if any(row.get("test_mse") not in (None, "", "nan") for row in all_rows):
        raise RuntimeError("test metric detected")

    baselines = {
        dataset: select_single_baseline(reference_infos[dataset])
        for dataset in DATASETS
    }
    result_rows = []
    for dataset in DATASETS:
        envelope_mse = min(float(info[1]["val_mse"]) for info in reference_infos[dataset].values())
        envelope_mae = min(float(info[1]["val_mae"]) for info in reference_infos[dataset].values())
        for stage, infos in (
            ("full_reference", reference_infos[dataset]),
            ("shadow_anchor", shadow_infos[dataset]),
            ("router_candidate", candidate_infos[dataset]),
        ):
            for mode, (_, row, spec) in infos.items():
                mse, mae = float(row["val_mse"]), float(row["val_mae"])
                key_params = {
                    "train_percent": int(spec.get("percent", spec.get("full_percent"))),
                    "epochs": int(spec.get("max_epochs", spec.get("router_epochs"))),
                    "output_mode": spec.get("mechanism_config", {}).get("output_mode"),
                    "router_mode": spec.get("mechanism_config", {}).get("router_mode"),
                }
                result_rows.append({
                    "setting": setting_name(dataset),
                    "config_id": mode if stage != "shadow_anchor" else f"shadow_{mode}",
                    "dataset": dataset,
                    "horizon": 96,
                    "seed": 2021,
                    "model": LABELS[mode],
                    "stage": stage,
                    "key_params": json.dumps(key_params, sort_keys=True),
                    "mse": mse,
                    "mae": mae,
                    "delta_mse": mse - envelope_mse,
                    "delta_mae": mae - envelope_mae,
                    "mse_ratio_to_envelope": mse / envelope_mse,
                    "mae_ratio_to_envelope": mae / envelope_mae,
                    "selected": mode == CHAMPION and stage == "router_candidate",
                    "test_accessed": False,
                })
    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis, pools = {}, {}
    with (OUTPUT / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            cache_rows = CACHE / f"{dataset}.csv"
            cache_state = CACHE / f"{dataset}.pt"
            if cache_rows.exists() and cache_state.exists():
                state = torch.load(cache_state, map_location="cpu", weights_only=False)
                summary, case_pool = state["summary"], state["pools"]
            else:
                with cache_rows.open("w", newline="") as cache_handle:
                    cache_writer = csv.DictWriter(cache_handle, fieldnames=SAMPLE_FIELDS)
                    cache_writer.writeheader()
                    summary, case_pool = evaluate_setting(
                        dataset,
                        candidate_infos[dataset][CHAMPION],
                        baselines[dataset],
                        cache_writer,
                        device,
                    )
                torch.save({"summary": summary, "pools": case_pool}, cache_state)
            with cache_rows.open(newline="") as cache_handle:
                writer.writerows(csv.DictReader(cache_handle))
            analysis[dataset], pools[dataset] = summary, case_pool
            print(f"audited {dataset}: {summary['sample_channel_pairs']:,} pairs", flush=True)

    replay = {}
    for dataset in DATASETS:
        candidate_row = candidate_infos[dataset][CHAMPION][1]
        baseline_row = reference_infos[dataset][baselines[dataset]][1]
        checks = {
            "candidate_mse": abs(analysis[dataset]["candidate_mse"] - float(candidate_row["val_mse"])),
            "candidate_mae": abs(analysis[dataset]["candidate_mae"] - float(candidate_row["val_mae"])),
            "baseline_mse": abs(analysis[dataset]["baseline_mse"] - float(baseline_row["val_mse"])),
            "baseline_mae": abs(analysis[dataset]["baseline_mae"] - float(baseline_row["val_mae"])),
        }
        if max(checks.values()) >= 2e-5:
            raise RuntimeError(f"metric replay failed: {dataset}: {checks}")
        replay[setting_name(dataset)] = checks

    cases = select_cases(pools, top_k=5)
    if len(cases) != 90:
        raise RuntimeError(f"expected 90 cases, got {len(cases)}")
    save_cases(OUTPUT / "selected_cases.npz", cases)
    representatives = representative_cases(cases)
    if len(representatives) != 9:
        raise RuntimeError(f"expected 9 representative cases, got {len(representatives)}")

    decision = json.loads((SCRATCH / "stage_a_decision.json").read_text())
    winner_rows = []
    for dataset in DATASETS:
        candidate = candidate_infos[dataset][CHAMPION][1]
        envelope_mse = min(float(info[1]["val_mse"]) for info in reference_infos[dataset].values())
        envelope_mae = min(float(info[1]["val_mae"]) for info in reference_infos[dataset].values())
        winner_rows.append({
            "dataset": dataset,
            "mse": float(candidate["val_mse"]),
            "mae": float(candidate["val_mae"]),
            "envelope_mse": envelope_mse,
            "envelope_mae": envelope_mae,
            "mse_ratio": float(candidate["val_mse"]) / envelope_mse,
            "mae_ratio": float(candidate["val_mae"]) / envelope_mae,
        })
    plot_ratios(winner_rows, figures / "all__metric_ratios.png")
    plot_groups(analysis, figures / "all__sample_groups.png")
    plot_routes(analysis, figures / "all__route_weights.png")
    plot_horizon(analysis, figures / "all__horizon_mae.png")
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        plot_cases(representatives, category, figures / f"all__cases_{category}.png")

    total_pairs = sum(summary["sample_channel_pairs"] for summary in analysis.values())
    total_groups = {
        group: sum(summary["groups"].get(group, 0) for summary in analysis.values())
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
    ranking_rows = [{
        "候选": LABELS[item["candidate"]],
        "宏比值": item["macro_ratio"],
        "最差比值": item["worst_ratio"],
        "全部改善": "是" if item["all_below_one"] else "否",
    } for item in decision["ranking"]]
    metric_rows = [{
        "数据集": row["dataset"],
        "包络MSE": row["envelope_mse"], "M3 MSE": row["mse"],
        "MSE变化": row["mse_ratio"] - 1,
        "包络MAE": row["envelope_mae"], "M3 MAE": row["mae"],
        "MAE变化": row["mae_ratio"] - 1,
    } for row in winner_rows]
    group_rows = [{
        "分组": {"significant_improvement": "显著改善", "comparable": "相近", "significant_regression": "显著退化"}[group],
        "数量": total_groups[group], "占比": total_groups[group] / total_pairs,
        "漂移": aggregate_features[group]["recent_drift"],
        "lag24": aggregate_features[group]["lag24_correlation"],
        "差分波动": aggregate_features[group]["difference_volatility"],
        "路由熵": aggregate_features[group]["route_entropy"],
    } for group in GROUPS]
    route_rows = [{
        "数据集": dataset,
        "A1权重": analysis[dataset]["mean_route_weights"]["A1"],
        "I0权重": analysis[dataset]["mean_route_weights"]["I0"],
        "R0权重": analysis[dataset]["mean_route_weights"]["R0"],
        "A1 oracle": analysis[dataset]["oracle_shares"]["A1"],
        "I0 oracle": analysis[dataset]["oracle_shares"]["I0"],
        "R0 oracle": analysis[dataset]["oracle_shares"]["R0"],
    } for dataset in DATASETS]
    case_rows = [{
        "类型": {"baseline_high_error": "原始高误差", "candidate_regression": "M3退化", "candidate_improvement": "M3改善"}[case["case_type"]],
        "setting": case["setting"], "通道": case["channel"], "样本": case["sample_id"],
        "原始MSE": case["baseline_mse"], "M3 MSE": case["candidate_mse"],
    } for case in representatives]
    percent = lambda value: f"{100 * float(value):+.2f}%"
    report = f"""# Multi-Anchor Selector v1：实验与客观误差分析

## 1. Experiment Setup

冻结 A1（RCRF+NLinear）、I0（RCRF+ICPT）和 R0（Rolling TriAxis）三个完整模型。影子锚点只训练前 24%，路由仅在其未见过目标的 24%–30% 时间段校准；正式 validation 时换回 30% 锚点。L720、H96、P24、seed 2021、Huber；**没有读取 test**。

## 2. Experiment Results

最佳候选 M3（结构 soft 路由）相对逐指标原始包络的 12 指标宏比值为 `{decision['ranking'][0]['macro_ratio']:.6f}`，即平均改善 `{percent(decision['ranking'][0]['macro_ratio'] - 1)}`。但 ETTh1-MAE 退化 `{percent(winner_rows[0]['mae_ratio'] - 1)}`、ETTm1-MAE 退化 `{percent(winner_rows[2]['mae_ratio'] - 1)}`，没有做到全部指标严格改善，Stage-A gate 失败，因此按预注册停止 H192/test。

{table(metric_rows, ['数据集', '包络MSE', 'M3 MSE', 'MSE变化', '包络MAE', 'M3 MAE', 'MAE变化'], {'包络MSE': lambda x: f'{x:.6f}', 'M3 MSE': lambda x: f'{x:.6f}', '包络MAE': lambda x: f'{x:.6f}', 'M3 MAE': lambda x: f'{x:.6f}', 'MSE变化': percent, 'MAE变化': percent})}

![指标变化](figures/all__metric_ratios.png)

## 3. Parameter / Configuration Search

{table(ranking_rows, ['候选', '宏比值', '最差比值', '全部改善'], {'宏比值': lambda x: f'{x:.6f}', '最差比值': lambda x: f'{x:.6f}'})}

M0 能学回每个 setting 的全局强锚点，但不能超过包络；两个硬结构路由均退化。只有 M3 的连续凸组合出现稳定平均收益，说明当前证据支持“预测互补可被 soft 插值利用”，不支持“周期级 argmax 已被可靠识别”。所有 60 个正式训练 run（18 个影子、18 个正式锚点、24 个路由候选）均保存在 `results.csv`。

## 4. Error Distribution

样本分析以每个 setting 中 MSE/MAE 联合最强的**单一完整锚点**为 baseline；共 `{total_pairs:,}` 个 validation 样本×通道。相对 MSE ≤-10% 为显著改善，≥+10% 为显著退化。

{table(group_rows, ['分组', '数量', '占比', '漂移', 'lag24', '差分波动', '路由熵'], {'占比': lambda x: f'{x:.2%}', '漂移': lambda x: f'{x:.4f}', 'lag24': lambda x: f'{x:.4f}', '差分波动': lambda x: f'{x:.4f}', '路由熵': lambda x: f'{x:.4f}'})}

![样本分组](figures/all__sample_groups.png)

这些是描述统计，不足以证明漂移、周期性或波动“导致”改善；可验证的事实仅是各组特征均值不同。

## 5. Horizon-wise Error

![分段误差](figures/all__horizon_mae.png)

图中逐数据集比较四个 24 步未来周期的 MAE 变化。M3 并非只改善某一个固定远期区间，因此收益不能简单归结为“只修复长跨度漂移”。

## 6. High-Error Selection

程序在每个 setting 内按绝对 MSE 排名，分别选择原始高误差、M3 最大退化、M3 最大改善各 5 例，并对同 setting×channel 的相距不足 96 的连续窗口去重，共 90 例；表格和图片再按同一分数程序化展示每组 3 个跨 setting 代表。没有人工挑例。

{table(case_rows, ['类型', 'setting', '通道', '样本', '原始MSE', 'M3 MSE'], {'原始MSE': lambda x: f'{x:.5f}', 'M3 MSE': lambda x: f'{x:.5f}'})}

## 7. Case Analysis

![原始高误差案例](figures/all__cases_baseline_high_error.png)

![M3退化案例](figures/all__cases_candidate_regression.png)

![M3改善案例](figures/all__cases_candidate_improvement.png)

每个右图给出四个未来周期的 A1/I0/R0 soft 权重，顶部色块表示用真实目标事后计算的周期 oracle。曲线、三个锚点预测、16 维路由特征和 oracle 均保存在 `selected_cases.npz`。

## 8. Repeated Observable Patterns

{table(route_rows, ['数据集', 'A1权重', 'I0权重', 'R0权重', 'A1 oracle', 'I0 oracle', 'R0 oracle'], {key: (lambda x: f'{x:.3f}') for key in ('A1权重', 'I0权重', 'R0权重', 'A1 oracle', 'I0 oracle', 'R0 oracle')})}

![权重与oracle](figures/all__route_weights.png)

可重复观察是：ETT 上平均权重往往接近 A1，而 Weather/Electricity 会明显调用 I0/R0；但平均权重与样本周期 oracle 的占比仍有较大差距。硬路由因此容易把一个不确定偏好变成错误的 one-hot 决策，soft 路由则保留了锚点间的误差抵消。

## 9. Objective Defect Summary

当前多锚点方向比单 A1 锚定更接近目标：它把六数据集 H96 的宏平均从相对包络退化变成平均改善，并在 10/12 个指标上超过包络。但仍有三个客观缺陷：一是 OOF 影子模型与 30% 正式锚点存在分布/能力偏移；二是只用一个 6% 连续校准段，结构特征覆盖有限；三是 soft 获益主要来自凸插值而非准确专家识别，因此还不是可解释的“适用域路由”。下一轮若继续，应先做多折 rolling-origin OOF 与权重校准，而不是增加专家或继续强化 hard argmax。

## 10. Experiment Scope

- 本轮只到六数据集 H96；Stage-A 严格门失败后没有运行 H192。
- selection source 是 validation；没有 test/Golden 数值参与参数选择。
- 训练聚合值已从 checkpoint 回放，最大绝对差 `{max(value for checks in replay.values() for value in checks.values()):.3g}`，阈值 `2e-5`。
- `sample_errors.csv` 足以按样本×通道重排；`selected_cases.npz` 保存 90 个逐 setting 入选案例，其中报告绘制 9 个跨 setting 代表。
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
    settings = [
        {"setting": setting_name(dataset), "dataset": dataset, "split": "validation", "lookback": 720, "horizon": 96, "seed": 2021}
        for dataset in DATASETS
    ]
    manifest = {
        "experiment_id": "multi_anchor_selector_v1",
        "code": {
            "repository": str(REPO), "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "modified_files": [],
        },
        "mechanism": {
            "description": "OOF-calibrated sample-channel-cycle soft router over frozen A1/I0/R0 complete forecasts",
            "feature_flag": CHAMPION,
        },
        "experiment": {
            "baseline": "per-setting single complete anchor minimizing normalized MSE+MAE sum for sample analysis; per-metric A1/I0/R0 envelope for aggregate gate",
            "candidate": CHAMPION,
            "settings": settings,
            "training": {"shadow_percent": 24, "full_percent": 30, "anchor_epochs": 8, "router_epochs": 20, "loss": "huber", "seed": 2021},
            "metrics": ["mse", "mae"],
        },
        "execution": {
            "environment": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "settings": [
                {"setting": item["setting"], "commands": ["scripts/run_multi_anchor_experiment.py --stage pilot", "scripts/run_multi_anchor_experiment.py --stage a"], "runtime": "recorded per run in scratch metrics.csv"}
                for item in settings
            ],
            "formal_runs": 60, "smoke_runs": 7, "test_accessed": False,
        },
        "selection": {
            "source": "validation",
            "selected_configs": [
                {"setting": item["setting"], "config_id": CHAMPION, "search_notes": "M3 selected by six-dataset H96 macro ratio; strict all-cell gate failed"}
                for item in settings
            ],
            "stage_a_decision": native(decision),
        },
        "analysis": {
            "ranking_metric": "absolute_mse_delta",
            "top_k": 5,
            "dedup_rule": "within each setting/category, same channel requires sample gap >= 96",
            "selections": selections,
            "sample_channel_rows": total_pairs,
            "metric_replay": native(replay),
        },
        "validation": {
            "results_checked": True,
            "ranking_and_cases_checked": True,
            "report_and_archive_checked": True,
            "directory_and_settings_checked": True,
            "status": "passed",
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
    result_settings = {row["setting"] for row in result_rows}
    with (OUTPUT / "sample_errors.csv").open(newline="") as handle:
        sample_settings = {row["setting"] for row in csv.DictReader(handle)}
    npz_settings = set(np.load(OUTPUT / "selected_cases.npz")["setting"].tolist())
    declared = {item["setting"] for item in settings}
    if result_settings != declared or sample_settings != declared or not npz_settings <= declared:
        raise RuntimeError("setting coverage mismatch")
    print(f"wrote {OUTPUT}")
    print(f"rows={total_pairs:,}, cases={len(cases)}, zip_sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
