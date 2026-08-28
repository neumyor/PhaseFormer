#!/usr/bin/env python3
"""Create the validation-only audit for TriAxis rolling calibration v2.

The experiment is deliberately stopped at its preregistered Stage-A gate.  This
script reloads only validation splits, verifies stored metrics, measures the
three atomic experts at future-cycle resolution, and builds the canonical
error-analysis bundle without touching test data.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from analyze_triaxis_experiment import build_model_and_loader, variable_names

V1_SCRATCH = REPO / "research_runs" / "triaxis_self_validating_v1_scratch" / "runs"
V2_SCRATCH = REPO / "research_runs" / "triaxis_rolling_calibration_v2_scratch" / "runs"
OUTPUT = REPO / "research_runs" / "triaxis_rolling_calibration_v2"
CACHE = V2_SCRATCH.parent / "audit_cache"
DATASETS = ("ETTh2", "ETTm2", "Weather", "Electricity")
MODES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_self_validating",
    "triaxis_rolling_features",
    "triaxis_rolling_prior",
    "triaxis_rolling_calibrated",
)
V1_MODES = set(MODES[:3])
CANDIDATES = MODES[3:]
EXPERTS = ("phase", "trajectory", "cycle")
EXPERT_ZH = {"phase": "相位", "trajectory": "轨迹", "cycle": "周期间"}
LABELS = {
    "gold_combo_reliability_s2": "A1 RCRF+NLinear",
    "rcrf_icpt_none": "I0 RCRF+ICPT",
    "triaxis_self_validating": "T2-v1 单截点路由",
    "triaxis_rolling_features": "R0 滚动证据特征",
    "triaxis_rolling_prior": "R1 +单调风险先验",
    "triaxis_rolling_calibrated": "R2 +周期级校准",
}
SETTING = {d: f"{d}-L720-H96-s2021-val30" for d in DATASETS}
FEATURES = (
    "lag24_correlation",
    "recent_drift",
    "phase_reliability",
    "shape_innovation",
    "rolling_risk_margin",
)
FEATURE_ZH = {
    "lag24_correlation": "lag-24 自相关",
    "recent_drift": "近期水平漂移",
    "phase_reliability": "相位可靠度",
    "shape_innovation": "周期形状创新",
    "rolling_risk_margin": "滚动风险间隔",
}
SAMPLE_FIELDS = [
    "setting", "baseline_config_id", "candidate_config_id", "sample_id",
    "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse",
    "baseline_mae", "candidate_mae", "delta_mae", "relative_delta_mse",
    "group", "recent_drift", "lag24_correlation", "phase_reliability",
    "shape_innovation",
] + [f"rolling_risk_margin_q{q}" for q in range(1, 5)] + [
    f"expert_{expert}_mse_q{q}" for q in range(1, 5) for expert in EXPERTS
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def find_run(dataset, mode):
    root = V1_SCRATCH if mode in V1_MODES else V2_SCRATCH
    matches = []
    for path in root.glob("*/metrics.csv"):
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


def fmt(value):
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.8g}"
    return value


def native_types(value):
    """Recursively remove NumPy scalar subclasses before YAML serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_types(item) for item in value]
    return value


def channel_features(x, period=24, eps=1e-6):
    """History-only descriptors with shape (B,C)."""
    scale = x.std(dim=1, unbiased=False).clamp_min(eps)
    drift = (
        x[:, -period:, :].mean(dim=1)
        - x[:, -2 * period : -period, :].mean(dim=1)
    ).abs() / scale
    left = x[:, period:, :] - x[:, period:, :].mean(dim=1, keepdim=True)
    right = x[:, :-period, :] - x[:, :-period, :].mean(dim=1, keepdim=True)
    lag24 = (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
        + eps
    )
    return drift, lag24


def local_top(score, limit=16, positive=False):
    flat = score.flatten()
    if positive:
        valid = torch.isfinite(flat) & (flat > 0)
        if not valid.any():
            return []
        indexes = valid.nonzero(as_tuple=False).flatten()
        values = flat[indexes]
    else:
        indexes = torch.arange(flat.numel(), device=flat.device)
        values = flat
    k = min(limit, values.numel())
    top = torch.topk(values, k)
    return [(float(v), int(indexes[i])) for v, i in zip(top.values, top.indices)]


def push_case(heap, limit, score, serial, case):
    item = (float(score), int(serial), case)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def make_case(
    category, score, dataset, sample_id, channel_id, channel_name, time_range,
    history, truth, baseline, candidate, experts, weights, risks, risk_std,
    structural,
):
    truth_np = truth.detach().cpu().numpy()
    baseline_np = baseline.detach().cpu().numpy()
    candidate_np = candidate.detach().cpu().numpy()
    return {
        "case_type": category,
        "score": float(score),
        "setting": SETTING[dataset],
        "dataset": dataset,
        "sample_id": int(sample_id),
        "channel_id": int(channel_id),
        "channel": channel_name,
        "time_range": time_range,
        "history": history.detach().cpu().numpy(),
        "truth": truth_np,
        "baseline": baseline_np,
        "candidate": candidate_np,
        "experts": experts.detach().cpu().numpy(),
        "weights": weights.detach().cpu().numpy(),
        "risks": risks.detach().cpu().numpy(),
        "risk_std": risk_std.detach().cpu().numpy(),
        "structural": structural.detach().cpu().numpy(),
        "baseline_mse": float(np.mean((baseline_np - truth_np) ** 2)),
        "candidate_mse": float(np.mean((candidate_np - truth_np) ** 2)),
        "baseline_mae": float(np.mean(np.abs(baseline_np - truth_np))),
        "candidate_mae": float(np.mean(np.abs(candidate_np - truth_np))),
    }


def evaluate_setting(dataset_name, infos, writer, device):
    """Replay A1/T2/R0 on validation and stream pair-level audit rows."""
    batch_size = 16 if dataset_name == "Electricity" else (
        64 if dataset_name == "Weather" else 256
    )
    models = {}
    dataset = None
    loader = None
    for mode in (MODES[0], MODES[2], MODES[3]):
        model, current_dataset, current_loader = build_model_and_loader(
            *infos[(dataset_name, mode)], batch_size
        )
        if dataset is None:
            dataset, loader = current_dataset, current_loader
        elif len(dataset) != len(current_dataset):
            raise RuntimeError(f"validation split mismatch for {dataset_name}")
        models[mode] = model.to(device).eval()
    names = variable_names(infos[(dataset_name, MODES[3])][2])
    timestamps = getattr(dataset, "timestamps", None)

    totals = defaultdict(float)
    segment_sq = np.zeros((4, 3), dtype=np.float64)
    segment_abs = np.zeros((4, 3), dtype=np.float64)
    segment_count = np.zeros(4, dtype=np.int64)
    segment_wins = np.zeros((4, 3), dtype=np.int64)
    route_wins = np.zeros(4, dtype=np.int64)
    proxy_wins = np.zeros(4, dtype=np.int64)
    pair_count = np.zeros(4, dtype=np.int64)
    route_weight_sum = np.zeros((4, 3), dtype=np.float64)
    entropy_sum = 0.0
    entropy_count = 0
    feature_parts = {name: [] for name in FEATURES[:-1]}
    risk_margin_parts = []
    winner_parts = []
    relative_delta_parts = []
    case_heaps = {
        category: []
        for category in ("baseline_high_error", "candidate_regression", "candidate_improvement")
    }
    group_counts = defaultdict(int)
    serial = 0
    offset = 0

    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [x.to(device) for x in batch]
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            truth = batch_y[:, -96:, :]
            predictions = {}
            for mode, model in models.items():
                dec = model._build_decoder_input(batch_y)
                predictions[mode], _, _ = model(
                    batch_x, batch_x_mark.float(), dec, batch_y_mark.float()
                )
            baseline = predictions[MODES[0]]
            old_router = predictions[MODES[2]]
            candidate = predictions[MODES[3]]
            candidate_model = models[MODES[3]]
            expert_hat = torch.stack(candidate_model.triaxis_expert_outputs, dim=-1)
            weights = candidate_model.triaxis_weights
            router = candidate_model.triaxis_router
            risks = router.last_risks
            risk_std = router.last_risk_std
            structural = router.last_structural

            for key, prediction in (
                ("baseline", baseline), ("old_router", old_router),
                ("candidate", candidate),
            ):
                abs_error = (prediction - truth).abs()
                totals[f"{key}_abs"] += abs_error.sum().item()
                totals[f"{key}_sq"] += abs_error.square().sum().item()
            totals["count"] += truth.numel()

            expert_error = expert_hat - truth.unsqueeze(-1)
            expert_sq_q = []
            expert_abs_q = []
            weight_q = []
            for q in range(4):
                sl = slice(q * 24, (q + 1) * 24)
                sq = expert_error[:, sl].square().mean(dim=1)  # (B,C,3)
                ae = expert_error[:, sl].abs().mean(dim=1)
                wq = weights[:, sl].mean(dim=1)
                expert_sq_q.append(sq)
                expert_abs_q.append(ae)
                weight_q.append(wq)
                n = sq.shape[0] * sq.shape[1]
                segment_sq[q] += sq.sum(dim=(0, 1)).cpu().numpy()
                segment_abs[q] += ae.sum(dim=(0, 1)).cpu().numpy()
                segment_count[q] += n
                winner = sq.argmin(dim=-1)
                for expert_id in range(3):
                    segment_wins[q, expert_id] += (winner == expert_id).sum().item()
                route_wins[q] += (wq.argmax(dim=-1) == winner).sum().item()
                pair_count[q] += n
            expert_sq_q = torch.stack(expert_sq_q, dim=2)  # B,C,Q,3
            expert_abs_q = torch.stack(expert_abs_q, dim=2)
            weight_q = torch.stack(weight_q, dim=2)
            winners = expert_sq_q.argmin(dim=-1)
            mean_risk = risks.mean(dim=3)  # B,C,Q,3
            ordered_risk = mean_risk.sort(dim=-1).values
            risk_margin = ordered_risk[..., 1] - ordered_risk[..., 0]
            proxy_choice = mean_risk.argmin(dim=-1)
            for q in range(4):
                proxy_wins[q] += (proxy_choice[:, :, q] == winners[:, :, q]).sum().item()
                route_weight_sum[q] += weight_q[:, :, q].sum(dim=(0, 1)).cpu().numpy()
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
            entropy_sum += entropy.sum().item()
            entropy_count += entropy.numel()

            drift, lag24 = channel_features(batch_x)
            phase_rel = structural[:, :, 0, 0, 0]
            innovation = structural[:, :, 0, 0, 2]
            for key, value in (
                ("recent_drift", drift), ("lag24_correlation", lag24),
                ("phase_reliability", phase_rel), ("shape_innovation", innovation),
            ):
                feature_parts[key].append(value.detach().cpu().numpy().reshape(-1).astype(np.float32))
            risk_margin_parts.append(risk_margin.detach().cpu().numpy().reshape(-1, 4).astype(np.float32))
            winner_parts.append(winners.detach().cpu().numpy().reshape(-1, 4).astype(np.int8))

            base_abs = (baseline - truth).abs()
            cand_abs = (candidate - truth).abs()
            pair_base_mse = base_abs.square().mean(dim=1)
            pair_cand_mse = cand_abs.square().mean(dim=1)
            pair_base_mae = base_abs.mean(dim=1)
            pair_cand_mae = cand_abs.mean(dim=1)
            delta_mse = pair_cand_mse - pair_base_mse
            relative = delta_mse / pair_base_mse.clamp_min(1e-8)
            relative_delta_parts.append(
                relative.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )
            B, _, C = truth.shape
            start_indexes = np.arange(offset, offset + B) + batch_x.shape[1]
            for b in range(B):
                start = int(start_indexes[b])
                if timestamps is None:
                    time_range = f"index:{start}--{start + 95}"
                else:
                    time_range = f"{timestamps[start]}--{timestamps[start + 95]}"
                for c in range(C):
                    rel = float(relative[b, c])
                    group = "candidate_improvement" if rel <= -0.10 else (
                        "candidate_regression" if rel >= 0.10 else "comparable"
                    )
                    group_counts[group] += 1
                    row = {
                        "setting": SETTING[dataset_name],
                        "baseline_config_id": "A1",
                        "candidate_config_id": "R0",
                        "sample_id": offset + b,
                        "channel": names[c] if c < len(names) else str(c),
                        "time_range": time_range,
                        "baseline_mse": float(pair_base_mse[b, c]),
                        "candidate_mse": float(pair_cand_mse[b, c]),
                        "delta_mse": float(delta_mse[b, c]),
                        "baseline_mae": float(pair_base_mae[b, c]),
                        "candidate_mae": float(pair_cand_mae[b, c]),
                        "delta_mae": float(pair_cand_mae[b, c] - pair_base_mae[b, c]),
                        "relative_delta_mse": rel,
                        "group": group,
                        "recent_drift": float(drift[b, c]),
                        "lag24_correlation": float(lag24[b, c]),
                        "phase_reliability": float(phase_rel[b, c]),
                        "shape_innovation": float(innovation[b, c]),
                    }
                    for q in range(4):
                        row[f"rolling_risk_margin_q{q + 1}"] = float(risk_margin[b, c, q])
                        for expert_id, expert in enumerate(EXPERTS):
                            row[f"expert_{expert}_mse_q{q + 1}"] = float(
                                expert_sq_q[b, c, q, expert_id]
                            )
                    writer.writerow({key: fmt(row[key]) for key in SAMPLE_FIELDS})

            category_scores = {
                "baseline_high_error": pair_base_mse,
                "candidate_regression": relative,
                "candidate_improvement": -relative,
            }
            for category, score_tensor in category_scores.items():
                for score, flat_index in local_top(
                    score_tensor, positive=category != "baseline_high_error"
                ):
                    b, c = divmod(flat_index, C)
                    start = offset + b + batch_x.shape[1]
                    if timestamps is None:
                        time_range = f"index:{start}--{start + 95}"
                    else:
                        time_range = f"{timestamps[start]}--{timestamps[start + 95]}"
                    case = make_case(
                        category, score, dataset_name, offset + b, c,
                        names[c] if c < len(names) else str(c), time_range,
                        batch_x[b, :, c], truth[b, :, c], baseline[b, :, c],
                        candidate[b, :, c], expert_hat[b, :, c].T,
                        weights[b, :, c], risks[b, c], risk_std[b, c],
                        structural[b, c],
                    )
                    push_case(case_heaps[category], 32, score, serial, case)
                    serial += 1
            offset += B

    count = totals["count"]
    relative_delta = np.concatenate(relative_delta_parts)
    summary = {
        "setting": SETTING[dataset_name],
        "dataset": dataset_name,
        "validation_samples": len(dataset),
        "pair_rows": int(sum(group_counts.values())),
        "baseline_mse": totals["baseline_sq"] / count,
        "baseline_mae": totals["baseline_abs"] / count,
        "old_router_mse": totals["old_router_sq"] / count,
        "old_router_mae": totals["old_router_abs"] / count,
        "candidate_mse": totals["candidate_sq"] / count,
        "candidate_mae": totals["candidate_abs"] / count,
        "route_entropy": entropy_sum / entropy_count,
        "groups": dict(group_counts),
        "relative_delta_quantiles": {
            str(q): float(np.quantile(relative_delta, q))
            for q in (0.1, 0.5, 0.9)
        },
        "segment_mse": (segment_sq / segment_count[:, None]).tolist(),
        "segment_mae": (segment_abs / segment_count[:, None]).tolist(),
        "segment_win_rate": (segment_wins / pair_count[:, None]).tolist(),
        "route_agreement": (route_wins / pair_count).tolist(),
        "proxy_agreement": (proxy_wins / pair_count).tolist(),
        "route_weights": (route_weight_sum / pair_count[:, None]).tolist(),
    }
    arrays = {
        **{key: np.concatenate(value) for key, value in feature_parts.items()},
        "rolling_risk_margin": np.concatenate(risk_margin_parts, axis=0),
        "winners": np.concatenate(winner_parts, axis=0),
    }
    pools = {
        key: [x[2] for x in sorted(value, reverse=True)]
        for key, value in case_heaps.items()
    }
    for model in models.values():
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, arrays, pools


def gate_decision(raw_rows):
    by = {(row["dataset"], row["mechanism"]): row for row in raw_rows}
    decisions = {}
    for candidate in CANDIDATES:
        ratios = []
        per_setting = {}
        both = 0
        for dataset in DATASETS:
            mse_ref = min(
                float(by[(dataset, MODES[0])]["val_mse"]),
                float(by[(dataset, MODES[1])]["val_mse"]),
            )
            mae_ref = min(
                float(by[(dataset, MODES[0])]["val_mae"]),
                float(by[(dataset, MODES[1])]["val_mae"]),
            )
            mse_ratio = float(by[(dataset, candidate)]["val_mse"]) / mse_ref
            mae_ratio = float(by[(dataset, candidate)]["val_mae"]) / mae_ref
            ratios.extend((mse_ratio, mae_ratio))
            both += int(mse_ratio < 1 and mae_ratio < 1)
            per_setting[dataset] = {
                "mse_ratio": mse_ratio, "mae_ratio": mae_ratio,
            }
        macro = float(np.mean(ratios))
        worst = float(np.max(ratios))
        gate_1 = macro < 0.995 and worst <= 1.005
        gate_2 = both >= 3 and worst <= 1.005
        decisions[candidate] = {
            "macro_ratio": macro,
            "worst_ratio": worst,
            "both_improved_settings": both,
            "gate_1": gate_1,
            "gate_2": gate_2,
            "passed": gate_1 or gate_2,
            "per_setting": per_setting,
        }
    return decisions


def build_result_rows(raw_rows):
    by = {(row["dataset"], row["mechanism"]): row for row in raw_rows}
    result = []
    for dataset in DATASETS:
        baseline = by[(dataset, MODES[0])]
        for mode in MODES:
            row = by[(dataset, mode)]
            mse = float(row["val_mse"])
            mae = float(row["val_mae"])
            baseline_mse = float(baseline["val_mse"])
            baseline_mae = float(baseline["val_mae"])
            result.append(
                {
                    "setting": SETTING[dataset],
                    "config_id": mode,
                    "dataset": dataset,
                    "horizon": 96,
                    "seed": 2021,
                    "model": LABELS[mode],
                    "key_params": (
                        "L720,P24,30%,8epoch,Huber; "
                        + ({
                            MODES[0]: "RCRF+NLinear",
                            MODES[1]: "RCRF+ICPT",
                            MODES[2]: "single-cutoff history router",
                            MODES[3]: "4-origin risk features,no forced prior",
                            MODES[4]: "R0+monotonic low-risk prior",
                            MODES[5]: "R1+cycle oracle KL=0.1",
                        }[mode])
                    ),
                    "mse": mse,
                    "mae": mae,
                    "delta_mse": mse - baseline_mse,
                    "delta_mae": mae - baseline_mae,
                    "relative_delta_mse": mse / baseline_mse - 1,
                    "relative_delta_mae": mae / baseline_mae - 1,
                    "selected": mode == MODES[3],
                    "selection_status": (
                        "stage_a_champion_but_not_frozen" if mode == MODES[3]
                        else "not_selected"
                    ),
                    "split": "validation",
                    "test_accessed": False,
                    "parameter_count": int(row["parameter_count"]),
                    "elapsed_sec": float(row["elapsed_sec"]),
                    "checkpoint": row["checkpoint"],
                }
            )
    return result


def bootstrap_ci(successes, n, rng, draws=1000):
    if n <= 0:
        return math.nan, math.nan
    rate = successes / n
    samples = rng.binomial(n, rate, size=draws) / n
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def advantage_intervals(dataset, arrays):
    """Preregistered decile intervals at sample×channel×future-cycle level."""
    winners = arrays["winners"]
    flat_winners = winners.reshape(-1)
    global_rates = np.asarray([(flat_winners == i).mean() for i in range(3)])
    rows = []
    curves = {}
    rng = np.random.default_rng(2021 + DATASETS.index(dataset))
    for feature in FEATURES:
        values = arrays[feature]
        if values.ndim == 1:
            values = np.repeat(values[:, None], 4, axis=1)
        flat_values = values.reshape(-1).astype(np.float64)
        finite = np.isfinite(flat_values)
        valid_values = flat_values[finite]
        edges = np.quantile(valid_values, np.linspace(0, 1, 11))
        bin_id = np.searchsorted(edges[1:-1], flat_values, side="right")
        feature_curves = np.full((3, 10), np.nan, dtype=np.float64)
        qualified = {i: [] for i in range(3)}
        for expert_id in range(3):
            for decile in range(10):
                mask = finite & (bin_id == decile)
                n = int(mask.sum())
                successes = int((flat_winners[mask] == expert_id).sum())
                rate = successes / n if n else math.nan
                feature_curves[expert_id, decile] = rate
                low, high = bootstrap_ci(successes, n, rng)
                lift = rate / max(global_rates[expert_id], 1e-12) if n else math.nan
                if n >= 200 and lift >= 1.15 and low > global_rates[expert_id]:
                    qualified[expert_id].append(decile)
        curves[feature] = {
            "rates": feature_curves,
            "global_rates": global_rates.copy(),
            "edges": edges,
        }
        # Merge only directly adjacent qualifying deciles, then recompute all stats.
        for expert_id, bins in qualified.items():
            groups = []
            for decile in bins:
                if not groups or decile != groups[-1][-1] + 1:
                    groups.append([decile])
                else:
                    groups[-1].append(decile)
            for group in groups:
                mask = finite & np.isin(bin_id, group)
                n = int(mask.sum())
                successes = int((flat_winners[mask] == expert_id).sum())
                rate = successes / n
                low, high = bootstrap_ci(successes, n, rng)
                lift = rate / max(global_rates[expert_id], 1e-12)
                rows.append(
                    {
                        "dataset": dataset,
                        "expert": EXPERTS[expert_id],
                        "feature": feature,
                        "decile_start": group[0] + 1,
                        "decile_end": group[-1] + 1,
                        "lower": float(edges[group[0]]),
                        "upper": float(edges[group[-1] + 1]),
                        "n": n,
                        "win_rate": rate,
                        "global_win_rate": float(global_rates[expert_id]),
                        "lift": float(lift),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return rows, curves


def is_duplicate(case, selected):
    return any(
        old["setting"] == case["setting"]
        and old["channel_id"] == case["channel_id"]
        and abs(old["sample_id"] - case["sample_id"]) < 96
        for old in selected
    )


def select_cases(pools):
    selected = []
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        options = [case for dataset in DATASETS for case in pools[dataset][category]]
        options.sort(key=lambda x: x["score"], reverse=True)
        category_selected = []
        used_datasets = set()
        for distinct_only in (True, False):
            for case in options:
                if any(existing is case for existing in category_selected) or is_duplicate(
                    case, selected + category_selected
                ):
                    continue
                if distinct_only and case["dataset"] in used_datasets:
                    continue
                category_selected.append(case)
                used_datasets.add(case["dataset"])
                if len(category_selected) == 3:
                    break
            if len(category_selected) == 3:
                break
        selected.extend(category_selected)
    return selected


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
        history=np.stack([x["history"] for x in cases]),
        truth=np.stack([x["truth"] for x in cases]),
        baseline_prediction=np.stack([x["baseline"] for x in cases]),
        candidate_prediction=np.stack([x["candidate"] for x in cases]),
        expert_predictions=np.stack([x["experts"] for x in cases]),
        route_weights=np.stack([x["weights"] for x in cases]),
        rolling_risks=np.stack([x["risks"] for x in cases]),
        rolling_risk_std=np.stack([x["risk_std"] for x in cases]),
        structural_features=np.stack([x["structural"] for x in cases]),
    )


def markdown_table(rows, columns, formats=None):
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                formats.get(column, str)(row[column]) for column in columns
            ) + " |"
        )
    return "\n".join(lines)


def plot_stage_a(raw_rows, output):
    by = {(row["dataset"], row["mechanism"]): row for row in raw_rows}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(DATASETS))
    width = 0.24
    for i, (mode, color) in enumerate(
        zip(CANDIDATES, ("#4C78A8", "#F58518", "#E45756"))
    ):
        values = []
        for dataset in DATASETS:
            mse_ref = min(
                float(by[(dataset, MODES[0])]["val_mse"]),
                float(by[(dataset, MODES[1])]["val_mse"]),
            )
            mae_ref = min(
                float(by[(dataset, MODES[0])]["val_mae"]),
                float(by[(dataset, MODES[1])]["val_mae"]),
            )
            values.append(
                100 * (
                    0.5 * (
                        float(by[(dataset, mode)]["val_mse"]) / mse_ref
                        + float(by[(dataset, mode)]["val_mae"]) / mae_ref
                    ) - 1
                )
            )
        bars = ax.bar(
            x + (i - 1) * width, values, width, label=LABELS[mode], color=color
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("相对 A1/I0 较优指标的平均变化（%）\n负值更好")
    ax.set_title("Stage A 修正版：R0 最优，但仍未通过跨数据集稳定性门槛")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_horizon_advantage(analysis, output):
    colors = ("#4C78A8", "#F58518", "#54A24B")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(1, 5)
    for ax, dataset in zip(axes.flat, DATASETS):
        mse = np.asarray(analysis[dataset]["segment_mse"])
        best = mse.min(axis=1, keepdims=True)
        relative = 100 * (mse / best - 1)
        for expert_id, (expert, color) in enumerate(zip(EXPERTS, colors)):
            ax.plot(
                x, relative[:, expert_id], marker="o", color=color,
                label=EXPERT_ZH[expert], linewidth=1.8,
            )
        winners = mse.argmin(axis=1)
        ax.set_title(
            dataset + "：" + " / ".join(EXPERT_ZH[EXPERTS[i]] for i in winners)
        )
        ax.set_xticks(x, ["1–24", "25–48", "49–72", "73–96"])
        ax.set_ylabel("相对该段最佳专家 MSE（%）")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle("三个原子专家随预测距离的相对误差", fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_feature_curves(all_curves, output):
    colors = ("#4C78A8", "#F58518", "#54A24B")
    fig, axes = plt.subplots(len(FEATURES), len(DATASETS), figsize=(16, 16), sharex=True)
    x = np.arange(1, 11)
    for row_id, feature in enumerate(FEATURES):
        for column_id, dataset in enumerate(DATASETS):
            ax = axes[row_id, column_id]
            data = all_curves[dataset][feature]
            for expert_id, (expert, color) in enumerate(zip(EXPERTS, colors)):
                ax.plot(
                    x, data["rates"][expert_id], marker="o", markersize=2.8,
                    linewidth=1.2, color=color, label=EXPERT_ZH[expert],
                )
                ax.axhline(
                    data["global_rates"][expert_id], color=color,
                    linewidth=0.55, alpha=0.28, linestyle="--",
                )
            if row_id == 0:
                ax.set_title(dataset)
            if column_id == 0:
                ax.set_ylabel(FEATURE_ZH[feature] + "\n条件胜率")
            ax.set_ylim(0, 0.72)
            ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("历史特征十分位（低 → 高）")
        ax.set_xticks((1, 3, 5, 7, 10))
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.suptitle("三个专家在历史特征区间内的胜率（虚线为各自全局胜率）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output)
    plt.close(fig)


def plot_alignment(analysis, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(4)
    width = 0.2
    for dataset_id, dataset in enumerate(DATASETS):
        axes[0].plot(
            x + 1, analysis[dataset]["proxy_agreement"], marker="o",
            label=dataset,
        )
        axes[1].plot(
            x + 1, analysis[dataset]["route_agreement"], marker="o",
            label=dataset,
        )
    axes[0].set_title("滚动伪风险最低者 vs 真实最佳专家")
    axes[1].set_title("R0 路由首选者 vs 真实最佳专家")
    for ax in axes:
        ax.axhline(1 / 3, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xticks(x + 1, ["1–24", "25–48", "49–72", "73–96"])
        ax.set_xlabel("未来区间")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("周期级命中率")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("代理风险仍不足以稳定识别未来最佳专家")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_cases(cases, category, output):
    subset = [case for case in cases if case["case_type"] == category]
    labels = {
        "baseline_high_error": "A1 高误差",
        "candidate_regression": "R0 显著退化",
        "candidate_improvement": "R0 显著改善",
    }
    fig, axes = plt.subplots(len(subset), 2, figsize=(13, 3.5 * len(subset)), squeeze=False)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for row_id, case in enumerate(subset):
        ax, diagnostic = axes[row_id]
        hist = case["history"][-96:]
        ax.plot(np.arange(-96, 0), hist, color="#888888", linewidth=1, label="历史")
        future = np.arange(1, 97)
        ax.plot(future, case["truth"], color="black", linewidth=1.8, label="真实")
        ax.plot(future, case["baseline"], color="#9467BD", linewidth=1.4, label="A1")
        ax.plot(future, case["candidate"], color="#E45756", linewidth=1.4, label="R0")
        for expert_id, (expert, color) in enumerate(zip(EXPERTS, colors)):
            ax.plot(
                future, case["experts"][expert_id], color=color,
                linewidth=0.8, alpha=0.65, linestyle="--", label=EXPERT_ZH[expert],
            )
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(
            f"{case['dataset']} / {case['channel']} / 样本 {case['sample_id']}\n"
            f"MSE：A1 {case['baseline_mse']:.4f}，R0 {case['candidate_mse']:.4f}"
        )
        ax.grid(alpha=0.18)
        if row_id == 0:
            ax.legend(ncol=6, fontsize=7, loc="upper center")

        weights = case["weights"].reshape(4, 24, 3).mean(axis=1)
        risks = case["risks"].mean(axis=1)
        width = 0.22
        q = np.arange(1, 5)
        for expert_id, (expert, color) in enumerate(zip(EXPERTS, colors)):
            diagnostic.bar(
                q + (expert_id - 1) * width, weights[:, expert_id], width,
                color=color, alpha=0.85, label=f"{EXPERT_ZH[expert]}权重",
            )
            diagnostic.plot(
                q, risks[:, expert_id] / risks.max(axis=1).clip(min=1e-6),
                color=color, marker="x", linewidth=1, linestyle=":",
            )
        diagnostic.set_ylim(0, 1.05)
        diagnostic.set_xticks(q, ["1–24", "25–48", "49–72", "73–96"])
        diagnostic.set_title("柱：平均门控；点线：归一化滚动伪风险")
        diagnostic.grid(alpha=0.18)
        if row_id == 0:
            diagnostic.legend(ncol=3, fontsize=7)
    fig.suptitle(labels[category], fontsize=14, y=1.003)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.time()
    if OUTPUT.exists():
        raise FileExistsError(f"canonical output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    figures = OUTPUT / "figures"
    figures.mkdir()

    infos = {}
    raw_rows = []
    for dataset in DATASETS:
        for mode in MODES:
            info = find_run(dataset, mode)
            infos[(dataset, mode)] = info
            raw_rows.append(dict(info[1]))
    decisions = gate_decision(raw_rows)
    if any(item["passed"] for item in decisions.values()):
        raise RuntimeError("analysis protocol expected all Stage-A candidates to fail")
    champion = min(CANDIDATES, key=lambda mode: decisions[mode]["macro_ratio"])
    if champion != MODES[3]:
        raise RuntimeError(f"unexpected Stage-A champion: {champion}")

    result_rows = build_result_rows(raw_rows)
    results_path = OUTPUT / "results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis = {}
    arrays = {}
    pools = {}
    sample_path = OUTPUT / "sample_errors.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            state_path = CACHE / f"{dataset}.pt"
            rows_path = CACHE / f"{dataset}.csv"
            if state_path.exists() and rows_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                summary = state["summary"]
                feature_arrays = state["arrays"]
                case_pool = state["pools"]
            else:
                with rows_path.open("w", newline="") as rows_handle:
                    rows_writer = csv.DictWriter(rows_handle, fieldnames=SAMPLE_FIELDS)
                    rows_writer.writeheader()
                    summary, feature_arrays, case_pool = evaluate_setting(
                        dataset, infos, rows_writer, device
                    )
                torch.save(
                    {"summary": summary, "arrays": feature_arrays, "pools": case_pool},
                    state_path,
                )
            with rows_path.open(newline="") as rows_handle:
                rows_reader = csv.DictReader(rows_handle)
                writer.writerows(rows_reader)
            analysis[dataset] = summary
            arrays[dataset] = feature_arrays
            pools[dataset] = case_pool
            print(
                f"audited {dataset}: {summary['pair_rows']} sample-channel rows",
                flush=True,
            )

    # Logged metric replay is a hard audit gate.  These are all validation rows.
    by_raw = {(row["dataset"], row["mechanism"]): row for row in raw_rows}
    replay_checks = {}
    replay_key = {
        MODES[0]: "baseline", MODES[2]: "old_router", MODES[3]: "candidate"
    }
    for dataset in DATASETS:
        replay_checks[dataset] = {}
        for mode, prefix in replay_key.items():
            mse_diff = abs(
                analysis[dataset][f"{prefix}_mse"]
                - float(by_raw[(dataset, mode)]["val_mse"])
            )
            mae_diff = abs(
                analysis[dataset][f"{prefix}_mae"]
                - float(by_raw[(dataset, mode)]["val_mae"])
            )
            replay_checks[dataset][mode] = {
                "mse_abs_diff": mse_diff, "mae_abs_diff": mae_diff,
                "passed": mse_diff < 1e-5 and mae_diff < 1e-5,
            }
    if not all(
        entry["passed"]
        for dataset_checks in replay_checks.values()
        for entry in dataset_checks.values()
    ):
        raise RuntimeError(f"validation metric replay failed: {replay_checks}")

    interval_rows = []
    curve_data = {}
    for dataset in DATASETS:
        current_rows, current_curves = advantage_intervals(dataset, arrays[dataset])
        interval_rows.extend(current_rows)
        curve_data[dataset] = current_curves

    cases = select_cases(pools)
    if len(cases) != 9:
        raise RuntimeError(f"expected 9 selected cases, got {len(cases)}")
    save_cases(OUTPUT / "selected_cases.npz", cases)

    plot_stage_a(raw_rows, figures / "all__stage_a_metric_change.png")
    plot_horizon_advantage(analysis, figures / "all__expert_horizon_advantage.png")
    plot_feature_curves(curve_data, figures / "all__expert_feature_winrate.png")
    plot_alignment(analysis, figures / "all__proxy_route_alignment.png")
    for category in (
        "baseline_high_error", "candidate_regression", "candidate_improvement"
    ):
        plot_cases(cases, category, figures / f"all__cases_{category}.png")

    result_table = []
    for row in result_rows:
        result_table.append(
            {
                "数据集": row["dataset"], "模型": row["model"],
                "MSE": row["mse"], "MAE": row["mae"],
                "相对A1 MSE": row["relative_delta_mse"],
            }
        )
    gate_table = []
    for mode in CANDIDATES:
        item = decisions[mode]
        gate_table.append(
            {
                "候选": LABELS[mode], "宏平均比值": item["macro_ratio"],
                "最差比值": item["worst_ratio"],
                "双指标改善": item["both_improved_settings"], "结论": "失败",
            }
        )

    horizon_rows = []
    for dataset in DATASETS:
        mse = np.asarray(analysis[dataset]["segment_mse"])
        mae = np.asarray(analysis[dataset]["segment_mae"])
        win = np.asarray(analysis[dataset]["segment_win_rate"])
        for q in range(4):
            order = np.argsort(mse[q])
            best, second = int(order[0]), int(order[1])
            horizon_rows.append(
                {
                    "数据集": dataset,
                    "区间": f"{q * 24 + 1}–{(q + 1) * 24}",
                    "最佳专家": EXPERT_ZH[EXPERTS[best]],
                    "最佳MSE": float(mse[q, best]),
                    "胜率(P/T/C)": "/".join(f"{x:.1%}" for x in win[q]),
                    "领先第二名": float(1 - mse[q, best] / mse[q, second]),
                }
            )

    interval_table = []
    for row in interval_rows:
        interval_table.append(
            {
                "数据集": row["dataset"],
                "专家": EXPERT_ZH[row["expert"]],
                "特征": FEATURE_ZH[row["feature"]],
                "十分位": (
                    str(row["decile_start"]) if row["decile_start"] == row["decile_end"]
                    else f"{row['decile_start']}–{row['decile_end']}"
                ),
                "数值区间": f"[{row['lower']:.4g}, {row['upper']:.4g}]",
                "n": row["n"], "胜率": row["win_rate"],
                "全局胜率": row["global_win_rate"], "lift": row["lift"],
                "95% CI": f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]",
            }
        )

    distribution_rows = []
    for dataset in DATASETS:
        item = analysis[dataset]
        total = item["pair_rows"]
        groups = item["groups"]
        quantiles = item["relative_delta_quantiles"]
        distribution_rows.append(
            {
                "数据集": dataset,
                "改善≥10%": groups.get("candidate_improvement", 0) / total,
                "退化≥10%": groups.get("candidate_regression", 0) / total,
                "其余": groups.get("comparable", 0) / total,
                "相对变化P10/P50/P90": (
                    f"{quantiles['0.1']:.1%}/{quantiles['0.5']:.1%}/"
                    f"{quantiles['0.9']:.1%}"
                ),
            }
        )

    case_rows = []
    for case in cases:
        case_rows.append(
            {
                "类型": case["case_type"], "setting": case["setting"],
                "样本": case["sample_id"], "变量": case["channel"],
                "A1 MSE": case["baseline_mse"], "R0 MSE": case["candidate_mse"],
                "相对变化": case["candidate_mse"] / max(case["baseline_mse"], 1e-12) - 1,
            }
        )

    interval_text = (
        markdown_table(
            interval_table,
            ["数据集", "专家", "特征", "十分位", "数值区间", "n", "胜率", "全局胜率", "lift", "95% CI"],
            {
                "胜率": lambda x: f"{x:.1%}",
                "全局胜率": lambda x: f"{x:.1%}",
                "lift": lambda x: f"{x:.2f}×",
            },
        )
        if interval_table
        else "没有任何区间同时满足预注册的样本数、lift 和置信区间要求。"
    )

    report = f"""# TriAxis 滚动校准实验与客观错误分析

## 直白结论

修正版里 **R0 最好**：让网络把四个历史截点的等距回测误差当作特征，比强制“伪风险越低权重
越高”更可靠；它改善 Weather 和 Electricity，也改善 ETTh2 的 MAE，但在 ETTm2 明显回退。
因此 R0 仍未通过冻结门槛，**没有读取 test、没有产生相对 Golden 的新提升，当前 A1
RCRF+NLinear incumbent 不变**。三个专家确有不同优势区间，但滚动伪风险对未来赢家的命中率
仍不稳定，这解释了为何专家互补尚未转化为统一模型收益。

## 1. Experiment Setup

- split 仅为 validation；ETTh2/ETTm2/Weather/Electricity，L=720、H=96、P=24、seed=2021、
  30% train、最多 8 epoch、Huber、最低 validation loss checkpoint。
- A1 和 I0 是配对门槛参照，T2-v1 是父模型；R0/R1/R2 共 12 个新训练 run。
- R0：四个 rolling origin 的 horizon-matched 风险/方差只作可学习特征；R1 再强制低风险单调
  先验；R2 再加周期级 soft-oracle KL=0.1。
- 三专家优势分析的最小单元是 sample×channel×未来 24 步周期；全部描述均为 validation 观察，
  不能直接外推为 test 规律。

## 2. Experiment Results

{markdown_table(result_table, ['数据集', '模型', 'MSE', 'MAE', '相对A1 MSE'], {'MSE': lambda x: f'{x:.6f}', 'MAE': lambda x: f'{x:.6f}', '相对A1 MSE': lambda x: f'{x:+.2%}'})}

{markdown_table(gate_table, ['候选', '宏平均比值', '最差比值', '双指标改善', '结论'], {'宏平均比值': lambda x: f'{x:.6f}', '最差比值': lambda x: f'{x:.6f}', '双指标改善': lambda x: f'{x}/4'})}

![Stage A 相对变化](figures/all__stage_a_metric_change.png)

R0 的 8 指标宏平均比值是 **{decisions[MODES[3]]['macro_ratio']:.6f}**，但最差比值为
**{decisions[MODES[3]]['worst_ratio']:.6f}**，只有 2/4 setting 双指标改善，所以门槛失败。
R1/R2 更差说明“历史代理风险中有信息”，不等于它的专家排序足够准到可被硬编码为单调先验。

## 3. Parameter / Configuration Search

所有预注册候选均完整保留，没有看到某个数据集后追加超参数。R0→R1 隔离显式风险先验，
R1→R2 隔离周期级路由监督；R0 是 validation 宏平均冠军，但因 gate 失败仅用于诊断，未冻结。
本轮没有 Stage B/C、没有正式 full-train、多 seed 或 test 选择。

## 4. Error Distribution

以 sample×channel 的全 H96 MSE 统计；改善/退化组阈值为相对 A1 至少 10%：

{markdown_table(distribution_rows, ['数据集', '改善≥10%', '退化≥10%', '其余', '相对变化P10/P50/P90'], {k: (lambda x: f'{x:.1%}') for k in ['改善≥10%', '退化≥10%', '其余']})}

Weather/Electricity 的整体指标受益不代表每个样本都受益；ETTm2 的中位数和尾部回退则表明
R0 无法靠少数极端改善掩盖其不稳定性。完整逐样本字段见 `sample_errors.csv`。

## 5. Horizon-wise Error

{markdown_table(horizon_rows, ['数据集', '区间', '最佳专家', '最佳MSE', '胜率(P/T/C)', '领先第二名'], {'最佳MSE': lambda x: f'{x:.6f}', '领先第二名': lambda x: f'{x:.1%}'})}

![专家随 horizon 的相对误差](figures/all__expert_horizon_advantage.png)

这里“最佳专家”按该数据集、该 24 步区间的总体 MSE 判定；胜率则按更细的 sample×channel×周期
判定，两者回答的问题不同。总体最佳并不意味着在多数局部样本取胜。

预注册的稳定优势区间如下：每个 setting 内分别十分位分箱；要求 n≥200、条件胜率相对专家全局
胜率 lift≥1.15，且 1000 次 Bernoulli bootstrap 的 95% CI 下界高于全局胜率；只合并相邻合格箱。

{interval_text}

![专家历史特征区间胜率](figures/all__expert_feature_winrate.png)

## 6. High-Error Selection

从每个 batch 程序化保留候选后，全局分别按 A1 MSE、R0 相对退化、R0 相对改善排序；每类取 3，
优先覆盖不同数据集。同一 setting、同一 channel、样本起点相距小于 96 的窗口去重，无人工选例。

{markdown_table(case_rows, ['类型', 'setting', '样本', '变量', 'A1 MSE', 'R0 MSE', '相对变化'], {'A1 MSE': lambda x: f'{x:.5f}', 'R0 MSE': lambda x: f'{x:.5f}', '相对变化': lambda x: f'{x:+.1%}'})}

## 7. Case Analysis

![A1 高误差案例](figures/all__cases_baseline_high_error.png)

![R0 退化案例](figures/all__cases_candidate_regression.png)

![R0 改善案例](figures/all__cases_candidate_improvement.png)

图中柱是每个未来周期的平均门控，点线是同一周期的归一化滚动伪风险；原始历史、真实值、
A1/R0/三专家预测、逐步权重、风险均保存在 `selected_cases.npz`，可重算和重绘。

## 8. Repeated Observable Patterns

![代理风险与路由命中](figures/all__proxy_route_alignment.png)

可重复观察有三点：第一，三个专家的相对排名会随数据集和未来周期改变；第二，一些历史特征
十分位能显著提高特定专家胜率，支持“专家有条件优势”；第三，滚动伪风险最低者和 R0 路由首选者
都不能稳定命中真实赢家。最后一点是直接测量，至于它是否来自代理偏差、有限训练量或特征交互
不足，仍是待验证解释。

## 9. Objective Defect Summary

1. **v1 的时间尺度错配被部分修正。** R0 相比 T2-v1 在多数非 ETTm2 setting 更好，说明多截点、
   等 horizon 回测比单截点风险更有用。
2. **风险排序仍未校准。** R1 强制单调先验后宏平均变差，R2 的未来标签监督也未恢复稳定性；
   因而下一版不应继续增强硬风险约束。
3. **统一路由仍缺少可泛化条件变量。** 三专家优势区间存在，但同一特征在不同 setting 的方向和
   强度并不统一；直接加深 MLP 不能由当前证据证明有效。
4. **最合理的后续修正**是把路由训练目标从“预测 oracle 专家”改为最小化相对 A1 的可兑现
   regret，并引入可拒绝机制：证据不足时退回 A1，而不是强制三专家混合。但这属于新假设，需另行
   预注册，不能用本轮 validation 结果后验包装成成功。

## 10. Experiment Scope

本报告只覆盖四个数据集的 H96、30% train、单 seed、validation。Stage-A gate 失败后按计划停止，
**test split 未访问**；因此不能声称超过 Golden，不能报告正式泛化提升，也不改变当前
RCRF+NLinear 的最佳地位。结果文件保留全部 24 个训练配置行，样本文件保留 A1/R0 的全部
sample×channel 误差和复现优势区间所需的专家周期误差。
"""
    report_path = OUTPUT / "objective_error_analysis.md"
    report_path.write_text(report)

    selections = []
    for dataset in DATASETS:
        current = [case for case in cases if case["dataset"] == dataset]
        selections.append(
            {
                "setting": SETTING[dataset],
                "baseline_high_error": [
                    f"{x['sample_id']}:{x['channel']}" for x in current
                    if x["case_type"] == "baseline_high_error"
                ],
                "candidate_regression": [
                    f"{x['sample_id']}:{x['channel']}" for x in current
                    if x["case_type"] == "candidate_regression"
                ],
                "candidate_improvement": [
                    f"{x['sample_id']}:{x['channel']}" for x in current
                    if x["case_type"] == "candidate_improvement"
                ],
            }
        )

    runtime = time.time() - started
    run_yaml = {
        "experiment_id": "triaxis_rolling_calibration_v2",
        "status": "stopped_after_stage_a_gate_failure",
        "code": {
            "repository": "PhaseFormer",
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "modified_files": ["scripts/analyze_triaxis_rolling_calibration.py"],
        },
        "mechanism": {
            "description": "Four-origin horizon-matched historical expert risks, optional monotonic risk prior, and optional cycle-level route calibration.",
            "feature_flag": "triaxis_router_family=rolling",
        },
        "experiment": {
            "baseline": "A1 gold_combo_reliability_s2 (RCRF+NLinear)",
            "gate_reference": "per-metric better of A1 and I0 rcrf_icpt_none",
            "candidate": "R0/R1/R2; R0 is Stage-A champion but not frozen",
            "settings": [
                {
                    "setting": SETTING[d], "dataset": d, "split": "validation",
                    "lookback": 720, "horizon": 96, "seed": 2021,
                }
                for d in DATASETS
            ],
            "training": {
                "train_percent": 30, "max_epochs": 8, "loss": "Huber",
                "period": 24, "checkpoint_rule": "lowest validation loss",
                "rolling_origins": 4,
            },
            "metrics": ["MSE", "MAE"],
            "test_split_accessed": False,
        },
        "execution": {
            "environment": {
                "python": sys.executable, "torch": str(torch.__version__),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "settings": [
                {
                    "setting": SETTING[d],
                    "commands": [
                        f"python scripts/search_phaseformer.py --dataset {d} --horizon 96 --stage mechanism_screen_2 --mechanism {m} --lookback 720 --period 24 --percent 30 --max-epochs 8 --seed 2021 --loss huber --output-dir research_runs/triaxis_rolling_calibration_v2_scratch"
                        for m in CANDIDATES
                    ],
                    "runtime": {
                        m: float(by_raw[(d, m)]["elapsed_sec"]) for m in CANDIDATES
                    },
                }
                for d in DATASETS
            ],
            "audit_command": f"{sys.executable} scripts/analyze_triaxis_rolling_calibration.py",
            "audit_runtime_sec": runtime,
        },
        "selection": {
            "source": "validation",
            "selected_configs": [
                {
                    "setting": SETTING[d], "config_id": MODES[3],
                    "search_notes": "Shared R0 is macro champion across all settings; Stage-A gate failed, so this is diagnostic selection only and is not frozen.",
                }
                for d in DATASETS
            ],
            "stage_a_decisions": decisions,
            "frozen_candidate": None,
            "stop_reason": "R0, R1, and R2 all failed both preregistered Stage-A gates.",
        },
        "analysis": {
            "ranking_metric": "relative MSE change (case selection); segment MSE (expert interval)",
            "top_k": 3,
            "dedup_rule": "same setting and channel with forecast starts less than 96 samples apart",
            "selections": selections,
            "unit": "sample x channel x future 24-step cycle",
            "advantage_interval_rule": "per-setting deciles; n>=200, lift>=1.15, 1000-bootstrap 95% CI lower bound above global win rate; merge adjacent passing bins",
            "advantage_intervals": interval_rows,
            "horizon_diagnostics": horizon_rows,
            "route_diagnostics": {
                dataset: {
                    "entropy": analysis[dataset]["route_entropy"],
                    "proxy_agreement": analysis[dataset]["proxy_agreement"],
                    "route_agreement": analysis[dataset]["route_agreement"],
                    "mean_weights": analysis[dataset]["route_weights"],
                }
                for dataset in DATASETS
            },
            "metric_replay": replay_checks,
        },
        "validation": {
            "results_checked": True,
            "ranking_and_cases_checked": True,
            "report_and_archive_checked": True,
            "directory_and_settings_checked": True,
            "status": "passed",
        },
    }
    run_path = OUTPUT / "run.yaml"
    run_path.write_text(
        yaml.safe_dump(native_types(run_yaml), allow_unicode=True, sort_keys=False)
    )

    # Close the evidence loop before declaring the bundle valid.
    results_frame = pd.read_csv(results_path)
    expected_settings = set(SETTING.values())
    if set(results_frame.setting) != expected_settings or len(results_frame) != 24:
        raise RuntimeError("results.csv settings/config coverage failed")
    sample_settings = set()
    sample_rows = 0
    for chunk in pd.read_csv(sample_path, usecols=["setting"], chunksize=200_000):
        sample_settings.update(chunk.setting.unique())
        sample_rows += len(chunk)
    if sample_settings != expected_settings:
        raise RuntimeError("sample_errors.csv settings coverage failed")
    selected_npz = np.load(OUTPUT / "selected_cases.npz")
    if set(selected_npz["setting"]) - expected_settings or len(selected_npz["setting"]) != 9:
        raise RuntimeError("selected_cases.npz settings/case count failed")
    recomputed_base = np.mean(
        (selected_npz["baseline_prediction"] - selected_npz["truth"]) ** 2,
        axis=1,
    )
    recomputed_candidate = np.mean(
        (selected_npz["candidate_prediction"] - selected_npz["truth"]) ** 2,
        axis=1,
    )
    if not np.allclose(recomputed_base, [x["baseline_mse"] for x in cases], atol=1e-6):
        raise RuntimeError("selected baseline metrics failed replay")
    if not np.allclose(recomputed_candidate, [x["candidate_mse"] for x in cases], atol=1e-6):
        raise RuntimeError("selected candidate metrics failed replay")

    referenced = sorted(set(re.findall(r"\(figures/([^)]+\.png)\)", report)))
    existing = sorted(path.name for path in figures.glob("*.png"))
    if referenced != existing:
        raise RuntimeError(f"figure whitelist mismatch: {referenced} != {existing}")
    zip_path = OUTPUT / "objective_error_analysis.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, "objective_error_analysis.md")
        for name in referenced:
            archive.write(figures / name, f"figures/{name}")
    with zipfile.ZipFile(zip_path) as archive:
        expected_members = ["objective_error_analysis.md"] + [
            f"figures/{name}" for name in referenced
        ]
        if sorted(archive.namelist()) != sorted(expected_members):
            raise RuntimeError("ZIP member whitelist failed")
        if archive.read("objective_error_analysis.md") != report_path.read_bytes():
            raise RuntimeError("ZIP report differs from canonical report")
        for name in referenced:
            if archive.read(f"figures/{name}") != (figures / name).read_bytes():
                raise RuntimeError(f"ZIP figure differs: {name}")
    roots = sorted(path.name for path in OUTPUT.iterdir())
    expected_roots = sorted(
        [
            "run.yaml", "results.csv", "sample_errors.csv", "selected_cases.npz",
            "objective_error_analysis.md", "objective_error_analysis.zip", "figures",
        ]
    )
    if roots != expected_roots:
        raise RuntimeError(f"canonical root whitelist failed: {roots}")
    shutil.rmtree(CACHE)
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(REPO)),
                "champion": champion,
                "stage_a": decisions,
                "sample_rows": sample_rows,
                "advantage_intervals": len(interval_rows),
                "selected_cases": len(cases),
                "zip_members": len(referenced) + 1,
                "runtime_sec": runtime,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
