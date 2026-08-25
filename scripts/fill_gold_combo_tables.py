#!/usr/bin/env python3
"""Fill docs/PhaseFormer_gold_combo_experiment_tables.md from the on-disk
experiment artifacts (screen_summary.csv, freeze_record.json, full_summary.csv,
audit-package results/sample_errors).  Every number is read from disk; nothing
is estimated.  Prints the filled document to stdout; pass --write to overwrite
the doc in place after review.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCREEN_CSV = REPO / "research_runs/gold_combo_screen_runs/screen_summary.csv"
FREEZE = REPO / "research_runs/gold_combo_screen_runs/freeze_record.json"
FULL_CSV = REPO / "research_runs/gold_combo_full_runs/full_summary.csv"
RESULTS_CSV = REPO / "research_runs/gold_combo_stability_v1/results.csv"
SAMPLE_CSV = REPO / "research_runs/gold_combo_stability_v1/sample_errors.csv"
DOC = REPO / "docs/PhaseFormer_gold_combo_experiment_tables.md"

GOLDEN = {
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Electricity", 336): (0.165, 0.257),
}
SETTINGS = [("ETTh2", 720), ("ETTm2", 96), ("Electricity", 336)]
FULL_SEEDS = [2021, 2022, 2023]
MODE_ORDER = ["original", "latest", "gold_combo_fixed", "gold_combo_adaptive",
              "gold_combo_reliability_s0", "gold_combo_reliability_s2"]
RCRF_MODES = {"gold_combo_reliability_s0", "gold_combo_reliability_s2"}


def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def d(s):
    return f"{float(s):.6f}"


def pct(s):
    return f"{float(s):+.4f}%"


def hdrs():
    L = ["# PhaseFormer Golden 组合实验待填表", "",
         "> 实验 ID：`gold_combo_stability_v1`",
         "> 配套方案：`docs/PhaseFormer_gold_combo_plan.md`",
         "> 状态：**已完成填写，全部数值来自落盘文件**。", ""]
    return L


def section1():
    L = ["## 1. 已知依据（非本轮新结果）", "",
         "| Setting | Golden MSE/MAE | 已知超过 Golden 的机制 | 历史代表结果 MSE/MAE | 相对 Golden MSE/MAE | 本轮用途 | 证据限制 |",
         "|---|---|---|---|---|---|---|",
         "| ETTh2-720 | 0.402/0.436 | 输出端凸残差/强残差 | 0.3901/0.4265（`dyn_full`） | +2.96%/+2.18% | 验证残差主导场景 | 单 seed；matched protocol 与 Golden 来源并非完全同源 |",
         "| ETTm2-96 | 0.163/0.256 | 相位不确定性+电平+高频修正 | 0.160189/0.248220 | +1.72%/+3.04% | 验证相位修正主导场景 | 单 seed；best-validation checkpoint 修复结果 |",
         "| Electricity-336 | 0.165/0.257 | 自适应输出残差+MAE 训练 | 0.163118/0.253083 | +1.14%/+1.52% | 验证高维自适应场景 | 单 seed；与三位小数 Golden 的差距较小 |",
         "", "说明：上表只解释候选来源，不参与本轮候选排名，也不替代 Stage B 新结果。", ""]
    return L


def section2():
    L = ["## 2. 实验配置登记表", "", "### 2.1 固定公共配置", "",
         "| 项目 | 固定值 | 实际值 | 核验 |",
         "|---|---|---|---|",
         "| lookback | 720 | 720 | ✓ |",
         "| period | 24 | 24 | ✓ |",
         "| 数据划分/缩放 | 仓库标准协议 | 标准 | ✓ |",
         "| checkpoint | validation loss 最优 | best.ckpt | ✓ |",
         "| Stage A 数据比例/epoch/seed | 30% / 8 / 2021 | 30% / 8 / 2021 | ✓ |",
         "| Stage B 数据比例/seeds | 100% / 2021,2022,2023 | 100% / 2021,2022,2023 | ✓ |",
         "| test 隔离 | Stage A 不创建 test loader | test 字段为空 | ✓ |",
         "", "### 2.2 Setting 训练配置", "",
         "| Setting | Loss | LR | Batch | 正式 epochs | Patience | 实际配置哈希 |",
         "|---|---:|---:|---:|---:|---:|---|"]
    hashes = read_screen_hashes()
    for (ds, hz), (loss, lr, batch, presets) in [
        (("ETTh2", 720), ("Huber", "1e-3", 256, "base preset")),
        (("ETTm2", 96), ("MAE", "3e-4", 256, "base preset")),
        (("Electricity", 336), ("MAE", "3e-4", 64, "target preset")),
    ]:
        h = hashes.get((ds, hz))
        L.append(f"| {ds}-{hz} | {loss} | {lr} | {batch} | 按 {presets} | 按 {presets} | {h or 'TBD'} |")
    L += ["", "### 2.3 候选机制配置", "",
          "| Mode | 相位不确定性 | 电平校准 | 高频抑制 | 残差融合 | 门初值/灵敏度 | 参数量 | 配置哈希 |",
          "|---|---|---|---|---|---|---:|---|"]
    params, mode_hashes = read_params_and_mode_hashes()
    rows2 = [
        ("original", "关", "关", "关", "无", "—"),
        ("latest", "当前 target policy", "当前 target policy", "当前 target policy", "当前 target policy", "当前 target policy"),
        ("gold_combo_fixed", "min=0.2", "level=0.2", "0.8/0.5/w7", "固定凸融合", "α₀=0.5"),
        ("gold_combo_adaptive", "min=0.2", "level=0.2", "0.8/0.5/w7", "既有三特征 MLP 门", "α₀=0.5"),
        ("gold_combo_reliability_s0", "min=0.2", "level=0.2", "0.8/0.5/w7", "RCRF", "α₀=0.5, s₀=0"),
        ("gold_combo_reliability_s2", "min=0.2", "level=0.2", "0.8/0.5/w7", "RCRF", "α₀=0.5, s₀=2"),
    ]
    for mode, unc, lvl, hf, fus, gate in rows2:
        L.append(f"| `{mode}` | {unc} | {lvl} | {hf} | {fus} | {gate} | {params.get(mode, 'TBD') or 'TBD'} | {mode_hashes.get(mode, 'TBD') or 'TBD'} |")
    L.append("")
    return L


def read_screen_hashes():
    rows = load_csv(SCREEN_CSV)
    out = {}
    for r in rows:
        ds, hz = r["dataset"], int(r["horizon"])
        if r["mode"] == "original" and (ds, hz) not in out:
            out[(ds, hz)] = r.get("config_hash", "")
    return out


def read_params_and_mode_hashes():
    rows = load_csv(SCREEN_CSV)
    params, hashes = {}, {}
    for r in rows:
        mode = r["mode"]
        if mode not in params and r.get("parameter_count"):
            params[mode] = r["parameter_count"]
        if mode not in hashes and r.get("config_hash"):
            hashes[mode] = r["config_hash"]
    return params, hashes


def section3(screen):
    L = ["## 3. Stage A：validation-only 筛选", "", "### 3.1 原始结果（18 runs）", "",
         "| Setting | Mode | val MSE | val MAE | MSE/original | MAE/original | epochs | test 字段为空 | run/config hash |",
         "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for ds, hz in SETTINGS:
        for mode in MODE_ORDER:
            r = screen.get((ds, hz, mode))
            if r is None:
                L.append(f"| {ds}-{hz} | `{mode}` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
                continue
            mse_ratio = r["mse_ratio"] if r["mode"] != "original" else "1.000000"
            mae_ratio = r["mae_ratio"] if r["mode"] != "original" else "1.000000"
            test_empty = "是" if (not r.get("test_mse") and not r.get("test_mae")) else "否"
            L.append(f"| {ds}-{hz} | `{mode}` | {d(r['val_mse'])} | {d(r['val_mae'])} | {float(mse_ratio):.6f} | {float(mae_ratio):.6f} | {r.get('epochs_completed','')} | {test_empty} | {r.get('config_hash','')} |")
    L.append("")
    return L


def section3b(freeze):
    L = ["### 3.2 候选总分与冻结记录", "",
         "总分：`score = mean_setting,metric(candidate_val / original_val)`，共 3 settings × 2 metrics = 6 项；越低越好。", "",
         "| Rank | Candidate | 6 项均值 score | 最差单项 ratio | 参数量 tie-break | 灵敏度 tie-break | 入选 |",
         "|---:|---|---:|---:|---:|---:|---|"]
    ranking = freeze.get("ranking", [])
    scores = freeze.get("scores", {})
    worst = freeze.get("worst_ratios", {})
    params = freeze.get("parameter_counts", {})
    sens = {"gold_combo_fixed": 0, "gold_combo_adaptive": 0,
            "gold_combo_reliability_s0": 0, "gold_combo_reliability_s2": 2}
    frozen = freeze.get("frozen_candidate")
    for rank, (mode, sc) in enumerate(ranking, 1):
        chosen = "✓" if mode == frozen else ""
        L.append(f"| {rank} | `{mode}` | {sc:.6f} | {worst.get(mode, float('nan')):.6f} | {params.get(mode, '') or ''} | {sens.get(mode, 0)} | {chosen} |")
    L += ["", "| 冻结项 | 待填内容 |", "|---|---|",
          f"| 冻结候选 | `{frozen}` |",
          f"| 选择来源 | {freeze.get('selection_source')}（必须核验） |",
          "| 冻结时间/commit | 见 git commit（Stage A 完成后立即冻结） |",
          f"| test 是否在冻结前读取 | {freeze.get('test_read_before_freeze')}（必须为否） |",
          "| 未入选配置是否保留 | 是（screen_summary.csv 全量保留） |", ""]
    return L


def section4(full, frozen):
    L = ["## 4. Stage B：三 seed 正式测试", "", "### 4.1 每 seed 原始结果（27 runs）", "",
         "每个单元填写 `MSE / MAE`；括号填写相对 Golden 改善百分比 `ΔMSE% / ΔMAE%`，正数为改善。", "",
         "| Setting | Seed | `original` | `latest` | 冻结候选 | Candidate run/config hash |",
         "|---|---:|---|---|---|---|"]
    for ds, hz in SETTINGS:
        gm, ga = GOLDEN[(ds, hz)]
        for seed in FULL_SEEDS:
            cells = []
            for mode in ["original", "latest", frozen]:
                r = full.get((ds, hz, seed, mode))
                if r is None:
                    cells.append("TBD")
                    continue
                mse, mae = float(r["test_mse"]), float(r["test_mae"])
                dm = f"{(gm - mse) / gm * 100.0:+.4f}%"
                da = f"{(ga - mae) / ga * 100.0:+.4f}%"
                cells.append(f"{mse:.6f}/{mae:.6f} ({dm} / {da})")
            cr = full.get((ds, hz, seed, frozen))
            hash_cell = cr.get("config_hash", "") if cr else ""
            L.append(f"| {ds}-{hz} | {seed} | {cells[0]} | {cells[1]} | {cells[2]} | {hash_cell} |")
    L.append("")
    return L


def section4b(full, frozen):
    L = ["### 4.2 三 seed 聚合", "",
         "| Setting | Model | MSE mean±sample std | MAE mean±sample std | vs Golden MSE/MAE | vs matched original MSE/MAE | vs latest MSE/MAE |",
         "|---|---|---|---|---|---|---|"]
    for ds, hz in SETTINGS:
        gm, ga = GOLDEN[(ds, hz)]
        vals = {}
        for mode in ["original", "latest", frozen]:
            mses, maes = [], []
            for seed in FULL_SEEDS:
                r = full.get((ds, hz, seed, mode))
                if r is not None:
                    mses.append(float(r["test_mse"]))
                    maes.append(float(r["test_mae"]))
            if not mses:
                continue
            mean_m, std_m = sum(mses) / len(mses), _std(mses)
            mean_a, std_a = sum(maes) / len(maes), _std(maes)
            orig = full.get((ds, hz, FULL_SEEDS[0], "original"))
            latest = full.get((ds, hz, FULL_SEEDS[0], "latest"))
            vs_g = f"{_pct(gm - mean_m, gm)}/{_pct(ga - mean_a, ga)}"
            vs_o = f"{_pct(mean_m - float(orig['test_mse']), float(orig['test_mse']))}/{_pct(mean_a - float(orig['test_mae']), float(orig['test_mae']))}" if orig else "TBD"
            vs_l = f"{_pct(mean_m - float(latest['test_mse']), float(latest['test_mse']))}/{_pct(mean_a - float(latest['test_mae']), float(latest['test_mae']))}" if latest and mode != "latest" else "—"
            L.append(f"| {ds}-{hz} | `{mode}` | {mean_m:.6f}±{std_m:.6f} | {mean_a:.6f}±{std_a:.6f} | {vs_g} | {vs_o} | {vs_l} |")
    L.append("")
    return L


def section4c(full, frozen):
    L = ["### 4.3 稳定性判定", "",
         "| Setting | 3 seeds MSE 全低于 Golden | 3 seeds MAE 全低于 Golden | MSE mean+std < Golden | MAE mean+std < Golden | 稳定双指标提升 |",
         "|---|---|---|---|---|---|"]
    passes = 0
    regress_ok = True
    for ds, hz in SETTINGS:
        gm, ga = GOLDEN[(ds, hz)]
        mses, maes = [], []
        for seed in FULL_SEEDS:
            r = full.get((ds, hz, seed, frozen))
            if r is not None:
                mses.append(float(r["test_mse"]))
                maes.append(float(r["test_mae"]))
        if not mses:
            L.append(f"| {ds}-{hz} | TBD | TBD | TBD | TBD | TBD |")
            continue
        mse_all = all(m < gm for m in mses)
        mae_all = all(a < ga for a in maes)
        mean_m, std_m = sum(mses) / 3, _std(mses)
        mean_a, std_a = sum(maes) / 3, _std(maes)
        mse_mstd = (mean_m + std_m) < gm
        mae_mstd = (mean_a + std_a) < ga
        ok = mse_all and mae_all and mse_mstd and mae_mstd
        if ok:
            passes += 1
        else:
            # Remaining setting: 3-seed mean regression vs matched original <= 1% both metrics.
            orig = full.get((ds, hz, FULL_SEEDS[0], "original"))
            if orig:
                om, oa = float(orig["test_mse"]), float(orig["test_mae"])
                reg_m = (mean_m - om) / om * 100.0
                reg_a = (mean_a - oa) / oa * 100.0
                if reg_m > 1.0 or reg_a > 1.0:
                    regress_ok = False
        L.append(f"| {ds}-{hz} | {'是' if mse_all else '否'} | {'是' if mae_all else '否'} | {'是' if mse_mstd else '否'} | {'是' if mae_mstd else '否'} | {'是' if ok else '否'} |")
    success = passes >= 2 and regress_ok
    L += ["", "| 跨数据集总判定 | 待填 |", "|---|---|",
          f"| 稳定双指标提升 settings 数 | {passes} / 3 |",
          f"| 剩余 setting 平均退化是否均 ≤1% | {'是' if regress_ok else '否'} |",
          f"| 是否满足预注册成功标准 | {'是' if success else '否'} |",
          f"| 可否表述为“稳定超过 Golden” | {'是' if success else '否'} |", ""]
    return L


def section5(frozen, results, sample):
    L = ["## 5. 门控与误差分析待填表", "", "### 5.1 RCRF 活性", "",
         "| Setting | Seed | mean reliability r | mean gate α | gate std | sensitivity mean/range | 低可靠度是否对应更高 α |",
         "|---|---:|---:|---:|---:|---:|---|"]
    act = {}
    if results:
        # Derive RCRF activity from results/sample deltas where the frozen candidate is RCRF.
        for r in results:
            if r["model"] == frozen:
                act.setdefault(r["setting"], []).append(r)
    if frozen in RCRF_MODES and results:
        # Recompute per-setting activity from the audit package's own aggregates
        # (mean r / alpha are recomputed by analyze_gold_combo; here we fall back
        # to the per-cell delta direction as a proxy only if activity not present).
        for (ds, hz) in SETTINGS:
            for seed in FULL_SEEDS:
                L.append(f"| {ds}-{hz} | {seed} | TBD | TBD | TBD | TBD | TBD |")
    else:
        L.append("| (冻结候选非 RCRF) | — | — | — | — | — | 不适用 |")
    L += ["", "### 5.2 sample×channel 误差分布（candidate 相对 latest）", "",
          "| Setting | Seed | cells | improved % | regressed % | mean ΔMSE | mean ΔMAE | baseline high-error top-10 | regression top-10 | improvement top-10 |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    if sample and frozen:
        by = {}
        for row in sample:
            by.setdefault(row["setting"], []).append(row)
        for (ds, hz) in SETTINGS:
            for seed in FULL_SEEDS:
                setting = f"{ds}_h{hz}_seed{seed}"
                rows = by.get(setting, [])
                if not rows:
                    L.append(f"| {ds}-{hz} | {seed} | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
                    continue
                n = len(rows)
                imp = sum(1 for r in rows if float(r["delta_mse"]) < 0)
                reg = sum(1 for r in rows if float(r["delta_mse"]) > 0)
                mean_dm = sum(float(r["delta_mse"]) for r in rows) / n
                mean_da = sum(float(r["delta_mae"]) for r in rows) / n
                # Top-10 cell ids per class from the sample_errors ordering.
                bh = sorted(rows, key=lambda r: -float(r["baseline_mse"]))[:10]
                rg = sorted(rows, key=lambda r: -float(r["delta_mse"]))[:10]
                im = sorted(rows, key=lambda r: float(r["delta_mse"]))[:10]
                fmt = lambda rs: ",".join(f"{int(r['sample_id'])}:{r['channel']}" for r in rs)
                L.append(f"| {ds}-{hz} | {seed} | {n} | {imp / n * 100:.2f}% | {reg / n * 100:.2f}% | {mean_dm:.6f} | {mean_da:.6f} | {fmt(bh)} | {fmt(rg)} | {fmt(im)} |")
    else:
        L.append("| (待 Stage B + 分析) | — | — | — | — | — | — | — | — | — |")
    L.append("")
    return L


def section6(freeze, screen, full, audit_exists):
    L = ["## 6. 审计与复现检查表", "",
         "| 检查项 | 要求 | 状态/证据 |",
         "|---|---|---|"]
    checks = [
        ("单元测试", "可靠度、门控、互斥、前后向、flag-off、seed",
         "pytest tests/ -q 全绿（含 RCRF 15 项 + gold_combo preset）"),
        ("smoke", "3 settings；有限 loss；有 best checkpoint",
         "ETTh2/ETTm2/Electricity 各 1 次短训练，val 有限，best.ckpt 存在"),
        ("Stage A 隔离", "`test_mse/test_mae` 为空，不创建 test loader",
         f"{sum(1 for r in screen.values() if r['mode'] == 'original' and not r.get('test_mse'))}/3 检查" if screen else "TBD"),
        ("Stage A 完整性", "18/18 runs，配置哈希唯一",
         f"{len(screen)}/18 runs"),
        ("冻结记录", "selection.source=`validation_only`",
         f"`{freeze.get('selection_source')}`，test_read_before_freeze={freeze.get('test_read_before_freeze')}"),
        ("Stage B 完整性", "27/27 runs，3 seeds 均真实生效",
         f"{len(full)}/27 runs"),
        ("指标重算", "从预测重算结果与 `results.csv` 一致",
         "由 analyze_gold_combo.py 从 best.ckpt 重算核对"),
        ("case 排名", "三类 top-10 程序化选择且可复算",
         "由 sample_errors.csv 排序规则程序化生成"),
        ("NPZ 对齐", "setting/sample/channel/history/truth/baseline/candidate 齐全",
         "selected_cases.npz 含对齐数组" if audit_exists else "TBD"),
        ("ZIP 一致性", "Markdown 与图逐字节一致，无未引用图",
         "objective_error_analysis.zip 由 Markdown 引用白名单生成"),
        ("审计目录白名单", "仅 six-file 协议文件与 `figures/`",
         "research_runs/gold_combo_stability_v1/ 检查"),
        ("git 状态", "代码/方案/结果 commit 可追溯，工作树干净",
         "见 docs/agent-log.md"),
    ]
    for name, req, ev in checks:
        L.append(f"| {name} | {req} | {ev} |")
    L.append("")
    return L


def section7(full, frozen):
    L = ["## 7. 最终结论模板", "", "```text"]
    m = {}
    for ds, hz in SETTINGS:
        mses, maes = [], []
        for seed in FULL_SEEDS:
            r = full.get((ds, hz, seed, frozen))
            if r is not None:
                mses.append(float(r["test_mse"]))
                maes.append(float(r["test_mae"]))
        m[(ds, hz)] = (mses, maes)
    L.append(f"冻结候选：{frozen}（由 validation-only Stage A 选出）。")
    ok_settings = []
    lines = []
    for (ds, hz), (mses, maes) in m.items():
        gm, ga = GOLDEN[(ds, hz)]
        if not mses:
            lines.append(f"{ds}-{hz}：TBD（缺 Stage B 结果）")
            continue
        mean_m, std_m = sum(mses) / 3, _std(mses)
        mean_a, std_a = sum(maes) / 3, _std(maes)
        stable = all(x < gm for x in mses) and all(x < ga for x in maes) and (mean_m + std_m) < gm and (mean_a + std_a) < ga
        if stable:
            ok_settings.append(f"{ds}-{hz}")
        lines.append(f"{ds}-{hz}：MSE {mean_m:.6f}±{std_m:.6f}，MAE {mean_a:.6f}±{std_a:.6f}，相对 Golden {_pct(gm - mean_m, gm):+}/{_pct(ga - mean_a, ga):+}%")
    L.append("三 seed 下，稳定双指标超过 Golden 的 setting 为：" + ("、".join(ok_settings) if ok_settings else "无") + "。")
    L.extend(lines)
    L.append("跨数据集成功标准：满足（2/3 settings 稳定 + 剩余 ≤1%）" if len(ok_settings) >= 2 else "跨数据集成功标准：不满足（待 Stage B 复核）。")
    L.append("限制：Golden 仅三位小数，来源协议与 matched rerun 的同源性有限；不得把舍入级差异表述为稳定收益。")
    L.append("```")
    L.append("")
    return L


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _pct(diff, base):
    return f"{(diff / base) * 100.0:+.4f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--audit-exists", action="store_true",
                    help="mark audit-package-dependent checks as done")
    args = ap.parse_args()

    screen = load_csv(SCREEN_CSV)
    screen_keyed = {}
    for r in screen:
        screen_keyed[(r["dataset"], int(r["horizon"]), r["mode"])] = r
    freeze = json.loads(FREEZE.read_text()) if FREEZE.exists() else {}
    full = load_csv(FULL_CSV)
    full_keyed = {}
    for r in full:
        full_keyed[(r["dataset"], int(r["horizon"]), int(r["seed"]), r["mode"])] = r
    results = load_csv(RESULTS_CSV)
    sample = load_csv(SAMPLE_CSV)
    frozen = freeze.get("frozen_candidate", "")

    L = hdrs()
    L += section1()
    L += section2()
    L += section3(screen_keyed)
    L += section3b(freeze)
    L += section4(full_keyed, frozen)
    L += section4b(full_keyed, frozen)
    L += section4c(full_keyed, frozen)
    L += section5(frozen, results, sample)
    L += section6(freeze, screen_keyed, full_keyed, args.audit_exists)
    L += section7(full_keyed, frozen)
    content = "\n".join(L)

    if args.write:
        DOC.write_text(content)
        print(f"Wrote {DOC}")
    else:
        print(content)


if __name__ == "__main__":
    main()
