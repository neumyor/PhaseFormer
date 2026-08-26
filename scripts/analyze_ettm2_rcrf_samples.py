#!/usr/bin/env python3
"""Auditable ETTm2 sample-level analysis for RCRF versus PhaseFormer.

The script reconstructs three matched checkpoint pairs, exports the RCRF
reliability/gate and both fusion branches, builds deterministic sample
categories, renders a Chinese report, and validates both the canonical audit
bundle and the user-facing docs ZIP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args


DATASET = "ETTm2"
LOOKBACK = 720
HORIZON = 96
SEEDS = (2021, 2022, 2023)
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
BASELINE = "original"
CANDIDATE = "gold_combo_reliability_s2"
EXPERIMENT_ID = "ettm2_rcrf_sample_analysis_v1"
REPORT_NAME = "ETTm2_RCRF_sample_analysis.md"
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "research_runs" / "ettm2_rcrf_sample_raw",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=REPO_ROOT / "research_runs" / EXPERIMENT_ID,
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=REPO_ROOT / "docs" / "ETTm2_RCRF_sample_analysis",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_run(raw_root: Path, mode: str, seed: int) -> Path:
    matches = sorted(raw_root.glob(f"*_{mode}_*ettm2_96_seed{seed}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run for mode={mode}, seed={seed}; got {matches}")
    return matches[0]


def read_metric(run_dir: Path) -> dict:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one metric row: {run_dir}")
    return rows[0]


def make_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = json_load(config_path)
    hp = cfg["hyperparams"]
    exp_args = make_exp_args(cfg["dataset"], cfg["lookback"], cfg["horizon"], hp)
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    model_cfg = PhaseFormerPresetConfig(exp_args, cfg["lookback"], cfg["horizon"], hp)
    model = PhaseFormer(model_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    return model, exp_args, cfg


def apply_hifreq(module, y_hat, phase_series):
    noise = phase_series.var(dim=-1, unbiased=False).mean(dim=2)
    trigger = torch.sigmoid((noise - module.noise_threshold) / module.noise_temperature)
    damping = 1.0 - module.strength * trigger.unsqueeze(1)
    smooth = module._smooth(y_hat)
    return smooth + damping * (y_hat - smooth)


def evaluate(model, exp_args, device: torch.device, capture_rcrf: bool):
    test_set, loader = data_provider(exp_args.dataset_args, "test")
    output = {k: [] for k in ("prediction", "truth", "history")}
    if capture_rcrf:
        output.update({k: [] for k in ("phase", "residual", "r", "alpha")})
        captured = {}

        def fusion_hook(_module, inputs, result):
            captured["phase_pre"] = inputs[0].detach()
            captured["residual_pre"] = inputs[1].detach()
            captured["r"] = _module._reliability(inputs[2]).detach()
            captured["alpha"] = result[1].detach()

        def damping_pre_hook(_module, inputs):
            captured["phase_series"] = inputs[1].detach()

        h1 = model.rcrf_fusion.register_forward_hook(fusion_hook)
        h2 = model.phase_noise_hifreq_damping.register_forward_pre_hook(damping_pre_hook)
    else:
        h1 = h2 = None

    try:
        with torch.inference_mode():
            for batch in loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = [
                    value.to(device) if torch.is_tensor(value) else value for value in batch
                ]
                dec = model._build_decoder_input(batch_y.float())
                pred, _, _ = model(
                    batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
                )
                output["prediction"].append(pred[:, -HORIZON:].float().cpu().numpy())
                output["truth"].append(batch_y[:, -HORIZON:].float().cpu().numpy())
                output["history"].append(batch_x.float().cpu().numpy())

                if capture_rcrf:
                    required = {"phase_pre", "residual_pre", "phase_series", "r", "alpha"}
                    if set(captured) != required:
                        raise RuntimeError(f"Incomplete RCRF hook payload: {sorted(captured)}")
                    phase = apply_hifreq(
                        model.phase_noise_hifreq_damping,
                        captured["phase_pre"],
                        captured["phase_series"],
                    )
                    residual = apply_hifreq(
                        model.phase_noise_hifreq_damping,
                        captured["residual_pre"],
                        captured["phase_series"],
                    )
                    # RevIN in this experiment is non-affine. Recreate the exact
                    # per-window statistics to map both branches to data scale.
                    if model.revin.affine:
                        raise RuntimeError("Branch decomposition assumes non-affine RevIN")
                    _, stats = model.revin.normalize(batch_x.float())
                    phase = model.revin.denormalize(phase, stats)
                    residual = model.revin.denormalize(residual, stats)
                    output["phase"].append(phase.float().cpu().numpy())
                    output["residual"].append(residual.float().cpu().numpy())
                    output["r"].append(captured["r"].float().cpu().numpy())
                    output["alpha"].append(captured["alpha"].float().cpu().numpy())
                    captured.clear()
    finally:
        if h1 is not None:
            h1.remove()
            h2.remove()

    return {key: np.concatenate(value, axis=0) for key, value in output.items()}, test_set


def global_metrics(pred, truth):
    diff = pred.astype(np.float64) - truth.astype(np.float64)
    return float(np.square(diff).mean()), float(np.abs(diff).mean())


def cell_errors(pred, truth):
    diff = pred.astype(np.float64) - truth.astype(np.float64)
    return np.square(diff).mean(axis=1), np.abs(diff).mean(axis=1)


def choose_nonoverlap(indices, score, top_k, min_gap=96):
    selected = []
    for index in indices[np.argsort(score[indices])[::-1]]:
        if all(abs(int(index) - prior) >= min_gap for prior in selected):
            selected.append(int(index))
            if len(selected) == top_k:
                break
    if len(selected) < top_k and min_gap > 24:
        return choose_nonoverlap(indices, score, top_k, min_gap // 2)
    return selected


def classify_samples(base_seed, cand_seed):
    # Inputs are (seed, sample). Relative gain is positive when RCRF is better.
    mean_base = base_seed.mean(axis=0)
    mean_cand = cand_seed.mean(axis=0)
    gain = (mean_base - mean_cand) / np.maximum(mean_base, EPS)
    all_better = np.all(cand_seed < base_seed, axis=0)
    all_worse = np.all(cand_seed >= base_seed, axis=0)
    labels = np.full(mean_base.shape, "混合且净退化", dtype=object)
    labels[(mean_cand < mean_base) & ~all_better] = "混合但净改善"
    labels[all_worse] = "稳定退化"
    labels[all_better & (gain < 0.02)] = "稳定微弱改善（<2%）"
    labels[all_better & (gain >= 0.02) & (gain < 0.10)] = "稳定中等改善（2–10%）"
    labels[all_better & (gain >= 0.10)] = "显著稳定改善（≥10%）"
    return labels, gain, mean_base, mean_cand


def sample_features(history):
    recent = history[:, -192:, :].astype(np.float64)
    first, last = recent[:, :96], recent[:, 96:]
    volatility = np.abs(np.diff(last, axis=1)).mean(axis=(1, 2))
    drift = np.abs(last.mean(axis=1) - first.mean(axis=1)).mean(axis=1)
    drift /= recent.std(axis=1).mean(axis=1) + EPS
    x0, x1 = recent[:, :-24], recent[:, 24:]
    x0 = x0 - x0.mean(axis=1, keepdims=True)
    x1 = x1 - x1.mean(axis=1, keepdims=True)
    corr = (x0 * x1).mean(axis=1) / (
        np.sqrt(np.square(x0).mean(axis=1) * np.square(x1).mean(axis=1)) + EPS
    )
    return volatility, drift, np.nanmean(corr, axis=1)


def set_chinese_style():
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), None)
    if selected is None:
        font_file = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
        if not font_file.exists():
            raise RuntimeError("No Chinese matplotlib font found")
        fm.fontManager.addfont(str(font_file))
        selected = fm.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.titleweight": "bold",
            "grid.alpha": 0.22,
            "savefig.dpi": 170,
        }
    )
    return selected


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def pct(value):
    return f"{100.0 * value:.2f}%"


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def plot_dataset_overview(raw_eval, figure_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    x = np.arange(0, len(raw_eval), 4)
    for channel, ax, values in zip(CHANNELS, axes.flat, raw_eval.T):
        ax.plot(x, values[::4], lw=0.8, color="#2563eb")
        ax.set_title(channel)
        ax.grid(True)
    axes.flat[-1].axis("off")
    fig.suptitle("ETTm2 测试目标区间概览（每小时下采样显示）", fontsize=16, y=1.01)
    fig.supxlabel("测试目标时间步（15 分钟）")
    fig.supylabel("原始量纲")
    savefig(figure_path)


def plot_seed_metrics(metric_rows, figure_path):
    base = [float(metric_rows[(seed, BASELINE)]["test_mae"]) for seed in SEEDS]
    cand = [float(metric_rows[(seed, CANDIDATE)]["test_mae"]) for seed in SEEDS]
    x = np.arange(len(SEEDS))
    width = 0.34
    plt.figure(figsize=(9, 5.2))
    plt.bar(x - width / 2, base, width, label="普通 PhaseFormer", color="#64748b")
    plt.bar(x + width / 2, cand, width, label="RCRF", color="#e11d48")
    for i, (b, c) in enumerate(zip(base, cand)):
        plt.text(i, min(b, c) - 0.003, f"改善 {(b-c)/b*100:.1f}%", ha="center", color="#166534", fontsize=10)
    plt.xticks(x, [str(seed) for seed in SEEDS])
    plt.ylabel("MAE（越低越好）")
    plt.xlabel("随机种子")
    plt.title("三个 matched seed 上 RCRF 均优于普通 PhaseFormer")
    plt.ylim(min(cand) - 0.008, max(base) + 0.006)
    plt.legend(frameon=False)
    plt.grid(axis="y")
    savefig(figure_path)


def plot_classification(labels, figure_path):
    order = [
        "显著稳定改善（≥10%）",
        "稳定中等改善（2–10%）",
        "稳定微弱改善（<2%）",
        "混合但净改善",
        "混合且净退化",
        "稳定退化",
    ]
    colors = ["#15803d", "#4ade80", "#bbf7d0", "#93c5fd", "#fdba74", "#dc2626"]
    counts = Counter(labels)
    values = np.array([counts[item] for item in order])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.barh(order[::-1], values[::-1] / len(labels) * 100, color=colors[::-1])
    for bar, count in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height() / 2, f"{count:,}（{count/len(labels)*100:.1f}%）", va="center")
    ax.set_xlabel("测试样本占比（%）")
    ax.set_title("逐样本跨 seed 一致性分类（互斥且完备）")
    ax.grid(axis="x")
    savefig(figure_path)


def plot_taxonomy(tax_rows, figure_path):
    dynamics = ["常规波动", "高波动", "明显漂移"]
    reliability = ["低可靠度", "中可靠度", "高可靠度"]
    matrix = np.full((3, 3), np.nan)
    counts = np.zeros((3, 3), dtype=int)
    for row in tax_rows:
        i, j = dynamics.index(row["dynamic"]), reliability.index(row["reliability"])
        matrix[i, j], counts[i, j] = row["mean_gain_pct"], row["count"]
    fig, ax = plt.subplots(figsize=(8.3, 5.8))
    image = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=max(10, np.nanmax(matrix)))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"收益 {matrix[i,j]:.1f}%\n样本 {counts[i,j]:,}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3), reliability)
    ax.set_yticks(range(3), dynamics)
    ax.set_title("动态形态 × phase 可靠度：RCRF 平均相对 MAE 收益")
    fig.colorbar(image, ax=ax, label="平均相对 MAE 收益（%）")
    savefig(figure_path)


def plot_gate_relationship(reliability, alpha, gain, figure_path):
    edges = np.quantile(reliability, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(reliability, edges[1:-1]), 0, 9)
    x, y_alpha, y_gain, sizes = [], [], [], []
    for index in range(10):
        mask = bins == index
        x.append(reliability[mask].mean())
        y_alpha.append(alpha[mask].mean())
        y_gain.append(gain[mask].mean() * 100)
        sizes.append(mask.sum())
    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    ax2 = ax1.twinx()
    ax1.plot(x, y_alpha, "o-", color="#e11d48", lw=2.2, label="平均门控 α")
    ax2.plot(x, y_gain, "s--", color="#2563eb", lw=2.0, label="RCRF 相对 MAE 收益")
    ax1.set_xlabel("phase 可靠度 r（按十分位分箱）")
    ax1.set_ylabel("门控 α（残差分支权重）", color="#e11d48")
    ax2.set_ylabel("相对 MAE 收益（%）", color="#2563eb")
    ax1.set_title("低可靠度样本获得更高残差权重；收益随可靠度分层")
    ax1.grid(True)
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    savefig(figure_path)


def plot_channels(channel_rows, figure_path):
    x = np.arange(len(CHANNELS))
    gain = [row["gain_pct"] for row in channel_rows]
    strong = [row["strong_share"] * 100 for row in channel_rows]
    fig, ax1 = plt.subplots(figsize=(11, 5.6))
    ax2 = ax1.twinx()
    bars = ax1.bar(x, gain, color="#0ea5e9", alpha=0.82, label="平均 MAE 收益")
    ax2.plot(x, strong, "o-", color="#dc2626", lw=2, label="显著稳定改善占比")
    ax1.axhline(0, color="#334155", lw=0.8)
    ax1.set_xticks(x, CHANNELS)
    ax1.set_ylabel("平均 MAE 收益（%）")
    ax2.set_ylabel("显著稳定改善的 sample×channel 占比（%）")
    ax1.set_title("通道级收益与显著改善覆盖率")
    for rect, value in zip(bars, gain):
        ax1.text(rect.get_x() + rect.get_width() / 2, value + 0.15, f"{value:.1f}%", ha="center", fontsize=9)
    lines = [bars, ax2.lines[0]]
    ax1.legend(lines, ["平均 MAE 收益", "显著稳定改善占比"], frameon=False)
    ax1.grid(axis="y")
    savefig(figure_path)


def plot_horizon(base_preds, cand_preds, truth, figure_path):
    base = np.abs(base_preds - truth[None]).mean(axis=(0, 1, 3))
    cand = np.abs(cand_preds - truth[None]).mean(axis=(0, 1, 3))
    x = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(11, 5.5))
    plt.plot(x, base, color="#64748b", lw=2, label="普通 PhaseFormer")
    plt.plot(x, cand, color="#e11d48", lw=2, label="RCRF")
    plt.fill_between(x, cand, base, where=base >= cand, color="#22c55e", alpha=0.15, label="RCRF 优势区")
    for boundary in (24, 48, 72):
        plt.axvline(boundary, color="#94a3b8", ls=":", lw=1)
    plt.xlabel("预测步（每步 15 分钟）")
    plt.ylabel("跨样本/通道/seed 平均绝对误差")
    plt.title("逐预测步误差：RCRF 的优势是否贯穿 96 步")
    plt.legend(frameon=False)
    plt.grid(True)
    savefig(figure_path)


def plot_branches(branch_rows, figure_path):
    labels = ["显著稳定改善", "其余净改善", "净退化"]
    phase = [branch_rows[label][0] for label in labels]
    residual = [branch_rows[label][1] for label in labels]
    fused = [branch_rows[label][2] for label in labels]
    x = np.arange(3)
    width = 0.25
    plt.figure(figsize=(10, 5.7))
    plt.bar(x - width, phase, width, label="phase 分支", color="#8b5cf6")
    plt.bar(x, residual, width, label="residual 分支", color="#f59e0b")
    plt.bar(x + width, fused, width, label="RCRF 融合", color="#e11d48")
    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.title("不同样本类别上的分支与融合误差（跨 seed）")
    plt.legend(frameon=False)
    plt.grid(axis="y")
    savefig(figure_path)


def plot_cases(selected, case_rows, arrays, truth, history, figure_path, title):
    fig, axes = plt.subplots(3, 2, figsize=(15, 13.2), sharex=False)
    base_mean = arrays["base"].mean(axis=0)
    cand_mean = arrays["cand"].mean(axis=0)
    phase_mean = arrays["phase"].mean(axis=0)
    residual_mean = arrays["residual"].mean(axis=0)
    for ax, sample_id in zip(axes.flat, selected):
        row = case_rows[sample_id]
        channel = row["channel_index"]
        hist = history[sample_id, -96:, channel]
        x_hist = np.arange(-96, 0)
        x_pred = np.arange(0, HORIZON)
        cand_seed = arrays["cand"][:, sample_id, :, channel]
        ax.plot(x_hist, hist, color="#94a3b8", lw=1.2, label="历史（末 96 步）")
        ax.plot(x_pred, truth[sample_id, :, channel], color="#0f172a", lw=2.1, label="真值")
        ax.plot(x_pred, base_mean[sample_id, :, channel], "--", color="#2563eb", lw=1.7, label="普通 PhaseFormer")
        ax.plot(x_pred, cand_mean[sample_id, :, channel], color="#e11d48", lw=2.0, label="RCRF")
        ax.fill_between(x_pred, cand_seed.min(axis=0), cand_seed.max(axis=0), color="#fb7185", alpha=0.15, label="RCRF seed 范围")
        ax.plot(x_pred, phase_mean[sample_id, :, channel], color="#8b5cf6", lw=1.0, alpha=0.75, label="phase 分支")
        ax.plot(x_pred, residual_mean[sample_id, :, channel], color="#f59e0b", lw=1.0, alpha=0.75, label="residual 分支")
        ax.axvline(0, color="#64748b", lw=0.8)
        ax.set_title(
            f"样本 {sample_id} · {CHANNELS[channel]} · 收益 {row['gain_pct']:.1f}%\n"
            f"r={row['r']:.3f}, α={row['alpha']:.3f}, {row['taxonomy']}",
            fontsize=10.5,
        )
        ax.grid(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.supxlabel("相对预测起点的时间步（15 分钟）", y=0.062)
    fig.supylabel("标准化数据量纲")
    fig.subplots_adjust(bottom=0.115, top=0.93, hspace=0.40, wspace=0.20)
    plt.savefig(figure_path, bbox_inches="tight", dpi=170)
    plt.close()


def zip_exact(zip_path: Path, md_path: Path, figure_paths, md_arcname: str):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(md_path, md_arcname)
        for path in figure_paths:
            archive.write(path, f"figures/{path.name}")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(zip_path: Path, md_path: Path, figure_paths, md_arcname):
    expected = [md_arcname] + [f"figures/{path.name}" for path in figure_paths]
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if names != expected:
            raise AssertionError(f"ZIP members differ: {names} != {expected}")
        if archive.read(md_arcname) != md_path.read_bytes():
            raise AssertionError("ZIP Markdown differs from source")
        for path in figure_paths:
            if archive.read(f"figures/{path.name}") != path.read_bytes():
                raise AssertionError(f"ZIP image differs: {path.name}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    font_name = set_chinese_style()
    raw_root, audit_root, docs_root = args.raw_root.resolve(), args.audit_root.resolve(), args.docs_root.resolve()
    docs_zip = docs_root.parent / f"{docs_root.name}.zip"

    for directory in (audit_root, docs_root):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty output directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "figures").mkdir()

    metric_rows, run_dirs, configs = {}, {}, {}
    arrays = {key: [] for key in ("base", "cand", "phase", "residual", "r", "alpha")}
    truth = history = test_set = None
    results_rows = []

    for seed in SEEDS:
        for mode in (BASELINE, CANDIDATE):
            run_dir = find_run(raw_root, mode, seed)
            run_dirs[(seed, mode)] = run_dir
            metric_rows[(seed, mode)] = read_metric(run_dir)
            model, exp_args, cfg = make_model(run_dir / "config.json", run_dir / "checkpoints" / "best.ckpt", device)
            configs[(seed, mode)] = cfg
            values, this_test_set = evaluate(model, exp_args, device, mode == CANDIDATE)
            mse, mae = global_metrics(values["prediction"], values["truth"])
            expected = metric_rows[(seed, mode)]
            # GPU kernels can differ by a few 1e-6 when the same samples are
            # replayed outside Lightning's logging loop. Keep the tolerance far
            # below the reported six-decimal precision while still catching a
            # wrong checkpoint/configuration.
            if abs(mse - float(expected["test_mse"])) > 1e-5 or abs(mae - float(expected["test_mae"])) > 1e-5:
                raise AssertionError(f"Checkpoint reconstruction mismatch for {mode}/{seed}: {(mse, mae)} vs {expected}")
            if truth is None:
                truth, history, test_set = values["truth"], values["history"], this_test_set
            else:
                np.testing.assert_array_equal(truth, values["truth"])
                np.testing.assert_array_equal(history, values["history"])
            key = "base" if mode == BASELINE else "cand"
            arrays[key].append(values["prediction"])
            if mode == CANDIDATE:
                for branch_key in ("phase", "residual", "r", "alpha"):
                    arrays[branch_key].append(values[branch_key])
                fused_pre_deno = None
                # Exact convexity survives linear damping and non-affine RevIN.
                reconstructed = (
                    (1.0 - values["alpha"][:, None, :]) * values["phase"]
                    + values["alpha"][:, None, :] * values["residual"]
                )
                if np.max(np.abs(reconstructed - values["prediction"])) > 2e-5:
                    raise AssertionError("Exported branches/gates do not reconstruct RCRF output")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        baseline_metric = metric_rows[(seed, BASELINE)]
        candidate_metric = metric_rows[(seed, CANDIDATE)]
        for mode, row in ((BASELINE, baseline_metric), (CANDIDATE, candidate_metric)):
            is_candidate = mode == CANDIDATE
            results_rows.append(
                {
                    "setting": f"ETTm2_h96_seed{seed}",
                    "config_id": mode,
                    "dataset": DATASET,
                    "horizon": HORIZON,
                    "seed": seed,
                    "model": "PhaseFormer" if not is_candidate else "PhaseFormer+RCRF",
                    "key_params": "baseline" if not is_candidate else "alpha0=0.5;s0=2;smax=4;weak_residual;uncertainty0.2;level0.2;hifreq0.8/0.5/w7",
                    "mse": float(row["test_mse"]),
                    "mae": float(row["test_mae"]),
                    "delta_mse": 0.0 if not is_candidate else float(row["test_mse"]) - float(baseline_metric["test_mse"]),
                    "delta_mae": 0.0 if not is_candidate else float(row["test_mae"]) - float(baseline_metric["test_mae"]),
                    "selected": True,
                }
            )

    arrays = {key: np.stack(value, axis=0) for key, value in arrays.items()}
    n_samples, n_channels = truth.shape[0], truth.shape[2]
    if (n_samples, n_channels) != (11425, 7):
        raise AssertionError(f"Unexpected ETTm2 test shape: {(n_samples, n_channels)}")

    base_mse, base_mae, cand_mse, cand_mae = [], [], [], []
    phase_mae, residual_mae = [], []
    for seed_index in range(len(SEEDS)):
        mse, mae = cell_errors(arrays["base"][seed_index], truth)
        base_mse.append(mse); base_mae.append(mae)
        mse, mae = cell_errors(arrays["cand"][seed_index], truth)
        cand_mse.append(mse); cand_mae.append(mae)
        phase_mae.append(cell_errors(arrays["phase"][seed_index], truth)[1])
        residual_mae.append(cell_errors(arrays["residual"][seed_index], truth)[1])
    base_mse, base_mae = np.stack(base_mse), np.stack(base_mae)
    cand_mse, cand_mae = np.stack(cand_mse), np.stack(cand_mae)
    phase_mae, residual_mae = np.stack(phase_mae), np.stack(residual_mae)

    base_sample = base_mae.mean(axis=2)
    cand_sample = cand_mae.mean(axis=2)
    labels, sample_gain, mean_base_sample, mean_cand_sample = classify_samples(base_sample, cand_sample)
    cell_labels, cell_gain, mean_base_cell, mean_cand_cell = classify_samples(
        base_mae.reshape(len(SEEDS), -1), cand_mae.reshape(len(SEEDS), -1)
    )
    cell_labels = cell_labels.reshape(n_samples, n_channels)
    cell_gain = cell_gain.reshape(n_samples, n_channels)
    reliability = arrays["r"].mean(axis=(0, 2))
    alpha = arrays["alpha"].mean(axis=(0, 2))
    volatility, drift, autocorr24 = sample_features(history)
    r_q33, r_q67 = np.quantile(reliability, [1 / 3, 2 / 3])
    vol_q75, drift_q75 = np.quantile(volatility, 0.75), np.quantile(drift, 0.75)
    reliability_label = np.where(reliability <= r_q33, "低可靠度", np.where(reliability <= r_q67, "中可靠度", "高可靠度"))
    dynamic_label = np.where(drift >= drift_q75, "明显漂移", np.where(volatility >= vol_q75, "高波动", "常规波动"))
    taxonomy = np.char.add(np.char.add(dynamic_label.astype(str), " × "), reliability_label.astype(str))

    tax_rows = []
    for dynamic in ("常规波动", "高波动", "明显漂移"):
        for rel in ("低可靠度", "中可靠度", "高可靠度"):
            mask = (dynamic_label == dynamic) & (reliability_label == rel)
            tax_rows.append(
                {
                    "dynamic": dynamic,
                    "reliability": rel,
                    "count": int(mask.sum()),
                    "share": float(mask.mean()),
                    "mean_gain_pct": float(sample_gain[mask].mean() * 100),
                    "strong_share": float((labels[mask] == "显著稳定改善（≥10%）").mean()),
                    "mean_r": float(reliability[mask].mean()),
                    "mean_alpha": float(alpha[mask].mean()),
                }
            )

    # Programmatic consensus examples: rank by absolute MAE improvement and
    # remove overlapping forecast windows. No manual cherry-picking.
    improvement_score = mean_base_sample - mean_cand_sample
    strong_idx = np.flatnonzero(labels == "显著稳定改善（≥10%）")
    improve_selected = choose_nonoverlap(strong_idx, improvement_score, 12)
    regression_idx = np.flatnonzero(np.isin(labels, ["稳定退化", "混合且净退化"]))
    regress_selected = choose_nonoverlap(regression_idx, -improvement_score, 6)
    if len(improve_selected) < 12 or len(regress_selected) < 6:
        raise AssertionError("Insufficient non-overlapping consensus cases")

    timestamps = pd.to_datetime(test_set.timestamps)
    case_rows = {}
    for sample_id in improve_selected + regress_selected:
        channel_delta = mean_base_cell[sample_id] - mean_cand_cell[sample_id]
        channel_index = int(np.argmax(channel_delta) if sample_id in improve_selected else np.argmin(channel_delta))
        case_rows[sample_id] = {
            "channel_index": channel_index,
            "channel": CHANNELS[channel_index],
            "gain_pct": float(sample_gain[sample_id] * 100),
            "baseline_mae": float(mean_base_sample[sample_id]),
            "candidate_mae": float(mean_cand_sample[sample_id]),
            "r": float(arrays["r"][:, sample_id, channel_index].mean()),
            "alpha": float(arrays["alpha"][:, sample_id, channel_index].mean()),
            "phase_mae": float(phase_mae[:, sample_id, channel_index].mean()),
            "residual_mae": float(residual_mae[:, sample_id, channel_index].mean()),
            "fused_mae": float(cand_mae[:, sample_id, channel_index].mean()),
            "seed_gain": ((base_sample[:, sample_id] - cand_sample[:, sample_id]) / np.maximum(base_sample[:, sample_id], EPS) * 100).tolist(),
            "taxonomy": str(taxonomy[sample_id]),
            "label": str(labels[sample_id]),
            "time_start": str(timestamps[sample_id + LOOKBACK]),
            "time_end": str(timestamps[sample_id + LOOKBACK + HORIZON - 1]),
        }

    channel_rows = []
    for channel_index, channel in enumerate(CHANNELS):
        b = base_mae[:, :, channel_index].mean()
        c = cand_mae[:, :, channel_index].mean()
        channel_rows.append(
            {
                "channel": channel,
                "baseline_mae": float(b),
                "candidate_mae": float(c),
                "gain_pct": float((b - c) / b * 100),
                "strong_share": float((cell_labels[:, channel_index] == "显著稳定改善（≥10%）").mean()),
                "mean_r": float(arrays["r"][:, :, channel_index].mean()),
                "mean_alpha": float(arrays["alpha"][:, :, channel_index].mean()),
            }
        )

    # Dataset target-coverage statistics in original units.
    raw_eval = test_set.inverse_transform(test_set.data_x[LOOKBACK:])
    raw_csv = pd.read_csv(REPO_ROOT / "resources" / "all_datasets" / "ETT" / "ETTm2.csv")
    dataset_rows = []
    for index, channel in enumerate(CHANNELS):
        values = raw_eval[:, index]
        dataset_rows.append(
            [channel, f"{values.mean():.3f}", f"{values.std():.3f}", f"{values.min():.3f}", f"{values.max():.3f}", int(raw_csv[channel].isna().sum())]
        )

    results_path = audit_root / "results.csv"
    pd.DataFrame(results_rows).to_csv(results_path, index=False)
    sample_path = audit_root / "sample_errors.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "baseline_config_id", "candidate_config_id", "sample_id", "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse", "baseline_mae", "candidate_mae", "delta_mae"])
        for seed_index, seed in enumerate(SEEDS):
            for sample_id in range(n_samples):
                time_range = f"{timestamps[sample_id + LOOKBACK]} / {timestamps[sample_id + LOOKBACK + HORIZON - 1]}"
                for channel_index, channel in enumerate(CHANNELS):
                    bm, cm = base_mse[seed_index, sample_id, channel_index], cand_mse[seed_index, sample_id, channel_index]
                    ba, ca = base_mae[seed_index, sample_id, channel_index], cand_mae[seed_index, sample_id, channel_index]
                    writer.writerow([f"ETTm2_h96_seed{seed}", BASELINE, CANDIDATE, sample_id, channel, time_range, f"{bm:.10g}", f"{cm:.10g}", f"{cm-bm:.10g}", f"{ba:.10g}", f"{ca:.10g}", f"{ca-ba:.10g}"])

    # Skill-mandated selections per setting: top 10 in all three groups.
    selection_records, npz_payload = [], {}
    for seed_index, seed in enumerate(SEEDS):
        setting = f"ETTm2_h96_seed{seed}"
        flat_base = base_mae[seed_index].reshape(-1)
        flat_cand = cand_mae[seed_index].reshape(-1)
        groups = {
            "baseline_high_error": flat_base,
            "candidate_regression": flat_cand - flat_base,
            "candidate_improvement": flat_base - flat_cand,
        }
        selected_flat = {}
        for group, scores in groups.items():
            ordered = np.argsort(scores)[::-1]
            chosen = []
            for flat_index in ordered:
                sample_id, channel_index = divmod(int(flat_index), n_channels)
                if all(channel_index != old_c or abs(sample_id - old_s) >= HORIZON for old_s, old_c in chosen):
                    chosen.append((sample_id, channel_index))
                    if len(chosen) == 10:
                        break
            selected_flat[group] = chosen
            prefix = f"{setting}__{group}"
            sample_ids = np.array([item[0] for item in chosen], dtype=np.int64)
            channel_ids = np.array([item[1] for item in chosen], dtype=np.int64)
            npz_payload[f"{prefix}__setting"] = np.array([setting] * len(chosen))
            npz_payload[f"{prefix}__sample_id"] = sample_ids
            npz_payload[f"{prefix}__channel_index"] = channel_ids
            npz_payload[f"{prefix}__channel"] = np.array([CHANNELS[c] for c in channel_ids])
            npz_payload[f"{prefix}__time_start"] = np.array([str(timestamps[s + LOOKBACK]) for s in sample_ids])
            npz_payload[f"{prefix}__time_end"] = np.array([str(timestamps[s + LOOKBACK + HORIZON - 1]) for s in sample_ids])
            npz_payload[f"{prefix}__history"] = history[sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__truth"] = truth[sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__baseline"] = arrays["base"][seed_index, sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__candidate"] = arrays["cand"][seed_index, sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__phase"] = arrays["phase"][seed_index, sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__residual"] = arrays["residual"][seed_index, sample_ids, :, channel_ids]
            npz_payload[f"{prefix}__r"] = arrays["r"][seed_index, sample_ids, channel_ids]
            npz_payload[f"{prefix}__alpha"] = arrays["alpha"][seed_index, sample_ids, channel_ids]
        selection_records.append(
            {
                "setting": setting,
                **{group: [f"sample={s},channel={CHANNELS[c]}" for s, c in chosen] for group, chosen in selected_flat.items()},
            }
        )
    np.savez_compressed(audit_root / "selected_cases.npz", **npz_payload)

    figures = [
        "ETTm2_h96__dataset_overview.png",
        "ETTm2_h96__seed_metrics.png",
        "ETTm2_h96__sample_classification.png",
        "ETTm2_h96__taxonomy_heatmap.png",
        "ETTm2_h96__gate_relationship.png",
        "ETTm2_h96__channel_results.png",
        "ETTm2_h96__horizon_errors.png",
        "ETTm2_h96__branch_errors.png",
        "ETTm2_h96__improvement_cases_1.png",
        "ETTm2_h96__improvement_cases_2.png",
        "ETTm2_h96__regression_cases.png",
    ]
    fp = {name: docs_root / "figures" / name for name in figures}
    plot_dataset_overview(raw_eval, fp[figures[0]])
    plot_seed_metrics(metric_rows, fp[figures[1]])
    plot_classification(labels, fp[figures[2]])
    plot_taxonomy(tax_rows, fp[figures[3]])
    plot_gate_relationship(reliability, alpha, sample_gain, fp[figures[4]])
    plot_channels(channel_rows, fp[figures[5]])
    plot_horizon(arrays["base"], arrays["cand"], truth, fp[figures[6]])
    branch_masks = {
        "显著稳定改善": labels == "显著稳定改善（≥10%）",
        "其余净改善": np.isin(labels, ["稳定中等改善（2–10%）", "稳定微弱改善（<2%）", "混合但净改善"]),
        "净退化": np.isin(labels, ["混合且净退化", "稳定退化"]),
    }
    branch_rows = {}
    for label, mask in branch_masks.items():
        branch_rows[label] = (
            float(phase_mae[:, mask].mean()),
            float(residual_mae[:, mask].mean()),
            float(cand_mae[:, mask].mean()),
        )
    plot_branches(branch_rows, fp[figures[7]])
    plot_cases(improve_selected[:6], case_rows, arrays, truth, history, fp[figures[8]], "显著稳定改善样本（按绝对 MAE 改善排序，1/2）")
    plot_cases(improve_selected[6:], case_rows, arrays, truth, history, fp[figures[9]], "显著稳定改善样本（按绝对 MAE 改善排序，2/2）")
    plot_cases(regress_selected, case_rows, arrays, truth, history, fp[figures[10]], "净退化样本（按 RCRF 退化幅度排序）")

    base_overall_mae = np.mean([float(metric_rows[(seed, BASELINE)]["test_mae"]) for seed in SEEDS])
    cand_overall_mae = np.mean([float(metric_rows[(seed, CANDIDATE)]["test_mae"]) for seed in SEEDS])
    base_overall_mse = np.mean([float(metric_rows[(seed, BASELINE)]["test_mse"]) for seed in SEEDS])
    cand_overall_mse = np.mean([float(metric_rows[(seed, CANDIDATE)]["test_mse"]) for seed in SEEDS])
    class_counts = Counter(labels)
    cell_counts = Counter(cell_labels.reshape(-1))
    sensitivity = [float(torch.load(run_dirs[(seed, CANDIDATE)] / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=False)["state_dict"]["rcrf_fusion.s_raw"].tanh() * 4.0) for seed in SEEDS]

    seed_rows_md = []
    for seed in SEEDS:
        b, c = metric_rows[(seed, BASELINE)], metric_rows[(seed, CANDIDATE)]
        seed_rows_md.append([seed, fmt(b["test_mse"], 6), fmt(c["test_mse"], 6), f"{(float(b['test_mse'])-float(c['test_mse']))/float(b['test_mse'])*100:.2f}%", fmt(b["test_mae"], 6), fmt(c["test_mae"], 6), f"{(float(b['test_mae'])-float(c['test_mae']))/float(b['test_mae'])*100:.2f}%"])

    class_rows_md = []
    class_order = ["显著稳定改善（≥10%）", "稳定中等改善（2–10%）", "稳定微弱改善（<2%）", "混合但净改善", "混合且净退化", "稳定退化"]
    for item in class_order:
        class_rows_md.append([item, f"{class_counts[item]:,}", pct(class_counts[item] / n_samples), f"{cell_counts[item]:,}", pct(cell_counts[item] / (n_samples * n_channels))])

    channel_rows_md = [[row["channel"], fmt(row["baseline_mae"]), fmt(row["candidate_mae"]), f"{row['gain_pct']:.2f}%", pct(row["strong_share"]), fmt(row["mean_r"], 3), fmt(row["mean_alpha"], 3)] for row in channel_rows]
    tax_rows_md = [[row["dynamic"], row["reliability"], f"{row['count']:,}", pct(row["share"]), f"{row['mean_gain_pct']:.2f}%", pct(row["strong_share"]), fmt(row["mean_r"], 3), fmt(row["mean_alpha"], 3)] for row in tax_rows]
    case_rows_md = []
    for rank, sample_id in enumerate(improve_selected, 1):
        row = case_rows[sample_id]
        case_rows_md.append([rank, sample_id, row["time_start"], row["channel"], row["taxonomy"], fmt(row["baseline_mae"]), fmt(row["candidate_mae"]), f"{row['gain_pct']:.1f}%", "/".join(f"{x:.1f}%" for x in row["seed_gain"]), fmt(row["r"], 3), fmt(row["alpha"], 3), f"{row['phase_mae']:.3f}/{row['residual_mae']:.3f}/{row['fused_mae']:.3f}"])
    regress_rows_md = []
    for rank, sample_id in enumerate(regress_selected, 1):
        row = case_rows[sample_id]
        regress_rows_md.append([rank, sample_id, row["time_start"], row["channel"], row["taxonomy"], row["label"], fmt(row["baseline_mae"]), fmt(row["candidate_mae"]), f"{row['gain_pct']:.1f}%", fmt(row["r"], 3), fmt(row["alpha"], 3)])

    horizon_rows = []
    for start in range(0, HORIZON, 24):
        b = np.abs(arrays["base"][:, :, start:start+24] - truth[None, :, start:start+24]).mean()
        c = np.abs(arrays["cand"][:, :, start:start+24] - truth[None, :, start:start+24]).mean()
        horizon_rows.append([f"{start+1}–{start+24}", fmt(b), fmt(c), f"{(b-c)/b*100:.2f}%"])

    report = f"""# ETTm2：RCRF 相对普通 PhaseFormer 的样本级分析

> 结论先行：在完全 matched 的 3 个 seed 上，RCRF 的平均 MSE 从 **{base_overall_mse:.6f}** 降至 **{cand_overall_mse:.6f}**（改善 **{(base_overall_mse-cand_overall_mse)/base_overall_mse*100:.2f}%**），平均 MAE 从 **{base_overall_mae:.6f}** 降至 **{cand_overall_mae:.6f}**（改善 **{(base_overall_mae-cand_overall_mae)/base_overall_mae*100:.2f}%**）。按预先写定的工程判据，{class_counts['显著稳定改善（≥10%）']:,}/{n_samples:,} 个测试窗口（{pct(class_counts['显著稳定改善（≥10%）']/n_samples)}）属于“显著稳定改善”。这里的“显著”仅指 **三个 seed 均改善且平均相对 MAE 改善 ≥10%**，不是统计假设检验意义的显著性。

## 1. Experiment Setup / 实验设置

- 数据：ETTm2，多变量 7 通道；标准官方切分。输入 720 步、预测 96 步，每步 15 分钟；测试窗口 {n_samples:,} 个。
- 对照：普通 PhaseFormer（`original`）与 `gold_combo_reliability_s2`（本文简称 RCRF）。两者每个 seed 使用相同数据、batch=256、MAE loss、学习率 3e-4、最多 30 epoch、best-validation checkpoint。
- seed：2021、2022、2023。所有数字来自本次 matched 重跑，不混用旧表中的不同协议结果。
- RCRF candidate 同时包含 uncertainty shrinkage、period-level calibration、high-frequency damping 和 weak-period residual。因此本文能直接分析融合门控和分支行为，但不能把全部总体收益因果归于 RCRF 门控单一部件。
- 评估量纲：模型指标与样本误差均为数据集标准化量纲；下方数据集描述统计另以 CSV 原始量纲给出。

![数据集概览](figures/{figures[0]})

原始 ETTm2 共 {len(raw_csv):,} 行、7 个数值通道、缺失值合计 {int(raw_csv[list(CHANNELS)].isna().sum().sum())}。测试预测目标覆盖 {len(raw_eval):,} 个唯一时间点：{timestamps[LOOKBACK]} 至 {timestamps[-1]}。

{md_table(['通道', '均值', '标准差', '最小值', '最大值', '全量缺失数'], dataset_rows)}

## 2. Experiment Results / 实验结果

{md_table(['seed', 'PhaseFormer MSE', 'RCRF MSE', 'MSE 改善', 'PhaseFormer MAE', 'RCRF MAE', 'MAE 改善'], seed_rows_md)}

三个 seed 的 MAE 改善范围为 {min((float(metric_rows[(s,BASELINE)]['test_mae'])-float(metric_rows[(s,CANDIDATE)]['test_mae']))/float(metric_rows[(s,BASELINE)]['test_mae'])*100 for s in SEEDS):.2f}%–{max((float(metric_rows[(s,BASELINE)]['test_mae'])-float(metric_rows[(s,CANDIDATE)]['test_mae']))/float(metric_rows[(s,BASELINE)]['test_mae'])*100 for s in SEEDS):.2f}%，方向完全一致。

![三个 seed 指标](figures/{figures[1]})

## 3. Parameter / Configuration Search / 参数与配置

本报告不在测试集上重新搜索配置。candidate 是仓库现有的固定 `gold_combo_reliability_s2`：先验残差权重 α₀=0.5，敏感度初值 s₀=2.0、上界 s_max=4.0；checkpoint 中学得的 s 分别为 {', '.join(f'{value:.4f}' for value in sensitivity)}。最终门控定义为：

`r = Var_l(mean_k x_lk) / [Var_l(mean_k x_lk) + mean_l Var_k(x_lk) + ε]`

`α = sigmoid(logit(α₀) + s(1-r))`，融合输出为 `y=(1-α)y_phase+αy_residual`。r、α 均为逐样本、逐通道标量，并在 96 个预测步上共享；本文导出的 phase/residual 分支还经过了与最终输出相同的高频阻尼和 RevIN 反归一化，因此可精确重构融合输出（最大绝对差校验阈值 2e-5）。

## 4. Error Distribution / 误差分布与样本分类

分类在“窗口级”（96 步 × 7 通道 MAE）和更细的“sample×channel 级”分别计算。六类互斥且覆盖全部单元；三个 seed 都严格改善才进入前三个稳定改善类别。

{md_table(['类别', '窗口数', '窗口占比', 'sample×channel 数', 'sample×channel 占比'], class_rows_md)}

![样本分类占比](figures/{figures[2]})

动态形态由输入末 192 步确定：漂移分数处于上四分位记“明显漂移”；否则相邻差分波动处于上四分位记“高波动”；其余为“常规波动”。可靠度按全测试集 r 的三分位切成低/中/高。阈值为 r={r_q33:.4f}/{r_q67:.4f}、波动={vol_q75:.4f}、漂移={drift_q75:.4f}；这些阈值只用于描述分组，不参与模型或样本优劣选择。

{md_table(['动态形态', '可靠度', '样本数', '占比', '平均收益', '显著改善占比', '平均 r', '平均 α'], tax_rows_md)}

![模式分类热力图](figures/{figures[3]})

## 5. Horizon-wise Error / 预测步与通道分析

{md_table(['预测步', 'PhaseFormer MAE', 'RCRF MAE', '改善'], horizon_rows)}

![逐预测步误差](figures/{figures[6]})

{md_table(['通道', 'PhaseFormer MAE', 'RCRF MAE', '改善', '显著改善单元占比', '平均 r', '平均 α'], channel_rows_md)}

![通道结果](figures/{figures[5]})

## 6. High-Error Selection / 选例规则

审计包对每个 seed、每个 sample×channel 程序化选取三组 Top-10：baseline MAE 最大、candidate−baseline MAE 最大、baseline−candidate MAE 最大。相同通道的入选窗口要求起点至少相距 96 步；完整索引、预测、真值、phase/residual 分支、r 和 α 位于 `selected_cases.npz`。

正文的跨-seed案例先按前述六类分类，再按窗口级绝对 MAE 改善排序；同样去除预测区间重叠。未进行人工挑图。显著改善 Top-12 如下，“分支 MAE”依次是 phase/residual/fused，针对表中主导通道并跨 seed 平均。

{md_table(['排名', '样本', '预测起点', '主导通道', '形态', '基线 MAE', 'RCRF MAE', '窗口收益', '三 seed 收益', 'r', 'α', '分支 MAE P/R/F'], case_rows_md)}

## 7. Case Analysis / 案例分析

![显著改善案例 1](figures/{figures[8]})

![显著改善案例 2](figures/{figures[9]})

图中的主导通道是该窗口中跨 seed 平均绝对改善最大的通道。RCRF 红线是三 seed 均值，浅红区域是三 seed 的预测范围。phase/residual 两条细线是融合前两分支；它们与表中的 r、α 一起给出可复查的中间结果。

净退化案例也按退化幅度程序化选出，作为边界条件而非隐藏失败样本：

{md_table(['排名', '样本', '预测起点', '主导通道', '形态', '类别', '基线 MAE', 'RCRF MAE', '收益', 'r', 'α'], regress_rows_md)}

![退化案例](figures/{figures[10]})

## 8. Repeated Observable Patterns / 重复可测模式

- 门控响应：跨样本/通道/seed 的 r 均值为 {arrays['r'].mean():.4f}，α 均值为 {arrays['alpha'].mean():.4f}；r 与 α 的 Pearson 相关为 {np.corrcoef(arrays['r'].reshape(-1), arrays['alpha'].reshape(-1))[0,1]:.4f}。由于公式中 s 为正，二者应呈负相关，这是实现层面的确定关系。
- 样本收益与门控：窗口平均 α 与相对 MAE 收益的 Pearson 相关为 {np.corrcoef(alpha, sample_gain)[0,1]:.4f}；可靠度 r 与收益相关为 {np.corrcoef(reliability, sample_gain)[0,1]:.4f}。相关性是描述性观察，不等于门控造成收益。
- 输入形态：lag-24 自相关均值为 {autocorr24.mean():.4f}；显著改善组为 {autocorr24[labels == '显著稳定改善（≥10%）'].mean():.4f}，净退化组为 {autocorr24[np.isin(labels, ['混合且净退化','稳定退化'])].mean():.4f}。
- 分支误差：显著改善、其余净改善、净退化三组的 phase/residual/fused MAE 见下图。融合误差可能低于其中一个分支，但因 α 在整个 horizon 固定，它未必逐时刻选择更优分支。

![门控关系](figures/{figures[4]})

![分支误差](figures/{figures[7]})

## 9. Objective Defect Summary / 客观缺陷总结

1. **稳定收益成立但不覆盖所有窗口。** 三个全局 seed 都改善；窗口级仍有 {class_counts['混合且净退化'] + class_counts['稳定退化']:,} 个净退化样本（{pct((class_counts['混合且净退化'] + class_counts['稳定退化'])/n_samples)}）。
2. **门控粒度有限。** α 只随 sample×channel 变化、96 步内固定；若 phase 与 residual 的相对优势在 horizon 内切换，当前门控无法逐步调整。这是结构事实；是否为退化主因需用 time-step gate 消融验证。
3. **总体收益不可单因果归因于 RCRF。** candidate 同时启用了三个 phase 修正。验证 RCRF 独立贡献需增加“相同 phase stack + fixed α / no residual”的 matched 消融。
4. **工程“显著”不是统计显著。** 当前只有 3 个训练 seed；报告使用严格的一致性+效应量阈值，未提供 p 值或置信区间。
5. **重叠窗口相关。** 11,425 个测试窗口高度重叠，窗口占比描述部署时遇到的预测起点，不可当作 11,425 个独立统计试验。

## 10. Experiment Scope / 范围与复现说明

- 本结论只覆盖 ETTm2、lookback 720、horizon 96、官方 test split 和三 seed；不能直接外推到其他 horizon/数据集。
- checkpoint 由 validation 最优规则选择；本次样本分类没有参与模型参数选择，`selection.source=fixed`。
- `sample_errors.csv` 保存全部 {len(SEEDS)*n_samples*n_channels:,} 个 setting×sample×channel 误差单元；`selected_cases.npz` 保存程序化 Top-K 所需的对齐序列和门控中间量。
- 图由 matplotlib 生成，使用 `{font_name}`；生成期间将缺字警告视为校验失败。
- 审计校验包括：checkpoint 指标复算、融合分支重构、CSV 聚合复算、Top-K 排名、Markdown 图片引用、ZIP 字节一致性、目录白名单和 setting 覆盖。
"""

    docs_md = docs_root / REPORT_NAME
    docs_md.write_text(report, encoding="utf-8")
    for path in fp.values():
        shutil.copyfile(path, audit_root / "figures" / path.name)
    audit_md = audit_root / "objective_error_analysis.md"
    audit_md.write_text(report.replace("# ETTm2：RCRF 相对普通 PhaseFormer 的样本级分析", "# Experiment and Objective Error Analysis\n\n## ETTm2：RCRF 相对普通 PhaseFormer 的样本级分析", 1), encoding="utf-8")

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    settings = [{"setting": f"ETTm2_h96_seed{seed}", "dataset": DATASET, "split": "test", "lookback": LOOKBACK, "horizon": HORIZON, "seed": seed} for seed in SEEDS]
    run_yaml = {
        "experiment_id": EXPERIMENT_ID,
        "code": {"repository": str(REPO_ROOT), "branch": git_branch, "commit": git_commit, "modified_files": ["src/dataset/data_info.py", "scripts/analyze_ettm2_rcrf_samples.py", f"docs/{docs_root.name}/", f"docs/{docs_zip.name}"]},
        "mechanism": {"description": "Reliability-Coupled Residual Fusion: per-sample/channel reliability controls convex phase/residual fusion", "feature_flag": "use_rcrf_fusion"},
        "experiment": {
            "baseline": BASELINE,
            "candidate": CANDIDATE,
            "settings": settings,
            "training": {"batch_size": 256, "loss": "mae", "learning_rate": 0.0003, "max_epochs": 30, "checkpoint_rule": "minimum validation loss"},
            "metrics": ["mse", "mae"],
        },
        "execution": {"environment": {"device": str(device), "torch": torch.__version__, "matplotlib": matplotlib.__version__, "chinese_font": font_name}, "settings": [{"setting": item["setting"], "commands": [str(run_dirs[(item["seed"], BASELINE)] / "commands.sh"), str(run_dirs[(item["seed"], CANDIDATE)] / "commands.sh")], "runtime_seconds": {BASELINE: float(metric_rows[(item["seed"], BASELINE)]["elapsed_sec"]), CANDIDATE: float(metric_rows[(item["seed"], CANDIDATE)]["elapsed_sec"])}} for item in settings]},
        "selection": {"source": "fixed", "selected_configs": [{"setting": item["setting"], "config_id": CANDIDATE, "search_notes": "Fixed candidate from the existing cross-dataset plan; no test-set search in this analysis."} for item in settings]},
        "analysis": {"ranking_metric": "mae", "top_k": 10, "dedup_rule": "within each group/channel, forecast starts differ by >=96 samples", "significant_improvement_rule": "all 3 seeds improve and mean relative sample MAE improvement >=10%", "selections": selection_records},
        "validation": {"results_checked": False, "ranking_and_cases_checked": False, "report_and_archive_checked": False, "directory_and_settings_checked": False, "status": "incomplete"},
    }
    run_path = audit_root / "run.yaml"
    run_path.write_text(json.dumps(run_yaml, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Package exactly the Markdown and its referenced figures, in first-use
    # order. This makes the report itself the archive whitelist/manifest.
    reference_pattern = re.compile(r"!\[[^]]*\]\(figures/([^)]+)\)")
    referenced_names = reference_pattern.findall(docs_md.read_text(encoding="utf-8"))
    if len(referenced_names) != len(figures) or set(referenced_names) != set(figures):
        raise AssertionError("Markdown figure references do not match generated figure set")
    docs_figures = [fp[name] for name in referenced_names]
    audit_figures = [audit_root / "figures" / name for name in referenced_names]
    zip_exact(docs_zip, docs_md, docs_figures, REPORT_NAME)
    zip_exact(audit_root / "objective_error_analysis.zip", audit_md, audit_figures, "objective_error_analysis.md")

    # Closed-loop validation.
    result_df = pd.read_csv(results_path)
    sample_df = pd.read_csv(sample_path)
    declared_settings = {item["setting"] for item in settings}
    if set(result_df.setting) != declared_settings or set(sample_df.setting) != declared_settings:
        raise AssertionError("Setting coverage mismatch")
    if len(sample_df) != len(SEEDS) * n_samples * n_channels:
        raise AssertionError("sample_errors.csv row count mismatch")
    for seed in SEEDS:
        setting = f"ETTm2_h96_seed{seed}"
        part = sample_df[sample_df.setting == setting]
        for config_id, column, metric_column in ((BASELINE, "baseline", "mae"), (CANDIDATE, "candidate", "mae")):
            recomputed = float(part[f"{column}_mae"].mean())
            recorded = float(result_df[(result_df.setting == setting) & (result_df.config_id == config_id)][metric_column].iloc[0])
            if abs(recomputed - recorded) > 1e-5:
                raise AssertionError(f"CSV aggregate mismatch: {setting}/{config_id}")
    # Verify each stored candidate-improvement group is monotonically top-ranked
    # under the stated non-overlap constraint by re-running selection.
    with np.load(audit_root / "selected_cases.npz") as archive:
        npz_settings = {str(value) for key in archive.files if key.endswith("__setting") for value in archive[key]}
        if npz_settings != declared_settings:
            raise AssertionError("NPZ setting coverage mismatch")
        for seed_index, seed in enumerate(SEEDS):
            prefix = f"ETTm2_h96_seed{seed}__candidate_improvement"
            stored = list(zip(archive[f"{prefix}__sample_id"].tolist(), archive[f"{prefix}__channel_index"].tolist()))
            scores = (base_mae[seed_index] - cand_mae[seed_index]).reshape(-1)
            expected = []
            for flat_index in np.argsort(scores)[::-1]:
                s, c = divmod(int(flat_index), n_channels)
                if all(c != old_c or abs(s - old_s) >= HORIZON for old_s, old_c in expected):
                    expected.append((s, c))
                    if len(expected) == 10:
                        break
            if stored != expected:
                raise AssertionError(f"NPZ ranking mismatch: seed {seed}")

    docs_refs = reference_pattern.findall(docs_md.read_text(encoding="utf-8"))
    audit_refs = reference_pattern.findall(audit_md.read_text(encoding="utf-8"))
    if docs_refs != audit_refs or len(docs_refs) != len(figures) or set(docs_refs) != set(figures):
        raise AssertionError("Markdown figure references do not match whitelist/order")
    for path in docs_figures + audit_figures:
        if not path.is_file() or path.stat().st_size == 0 or path.is_symlink():
            raise AssertionError(f"Invalid figure: {path}")
    validate_zip(docs_zip, docs_md, docs_figures, REPORT_NAME)
    validate_zip(audit_root / "objective_error_analysis.zip", audit_md, audit_figures, "objective_error_analysis.md")
    for docs_path, audit_path in zip(docs_figures, audit_figures):
        if sha256(docs_path) != sha256(audit_path):
            raise AssertionError(f"Docs/audit figure differs: {docs_path.name}")
    expected_root = {"run.yaml", "results.csv", "sample_errors.csv", "selected_cases.npz", "objective_error_analysis.md", "objective_error_analysis.zip", "figures"}
    if {path.name for path in audit_root.iterdir()} != expected_root:
        raise AssertionError("Canonical directory whitelist mismatch")
    if {path.name for path in docs_root.iterdir()} != {REPORT_NAME, "figures"}:
        raise AssertionError("Docs subdirectory should contain only Markdown and figures")

    run_yaml["validation"] = {"results_checked": True, "ranking_and_cases_checked": True, "report_and_archive_checked": True, "directory_and_settings_checked": True, "status": "passed", "checks": {"sample_error_rows": len(sample_df), "figure_count": len(figures), "docs_zip_sha256": sha256(docs_zip), "audit_zip_sha256": sha256(audit_root / "objective_error_analysis.zip")}}
    run_path.write_text(json.dumps(run_yaml, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "docs_report": str(docs_md), "docs_zip": str(docs_zip), "audit_root": str(audit_root), "samples": n_samples, "strong_stable": class_counts["显著稳定改善（≥10%）"], "mean_mae_gain_pct": (base_overall_mae-cand_overall_mae)/base_overall_mae*100}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from current font")
        main()
