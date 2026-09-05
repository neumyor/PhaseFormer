#!/usr/bin/env python3
"""Append a Chinese interpretation to the ETTh1 Gaussian-route audit."""

from __future__ import annotations

import csv
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_runs/etth1_smooth_route_role_cases"
REPORT = OUT / "objective_error_analysis.md"


def average(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def figure(component: str, direction: str, rank: int, origin: int) -> str:
    return f"figures/ETTh1__{component}__{direction}__{rank:02d}_origin{origin}_channel0.png"


def main() -> None:
    raw = REPORT.read_text(encoding="utf-8")
    rows = list(csv.DictReader((OUT / "sample_errors.csv").open(encoding="utf-8")))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["component"], row["direction"]].append(row)

    lines = [
        "# ETTh1：Local smooth 与 Smooth-multiscale 的 NLinear 路由审计", "",
        "## 结论先行", "",
        "1. 两种 A 都不是‘PhaseFormer 没有用的成分’的直接证据。两条路由的 PhaseFormer 始终得到完整 X；被改变的只有 NLinear 弱残差分支。因此，本审计描述的是 NLinear 在何种状态下偏好 X-A 或 Only-A。", "",
        "2. `smooth_local` 是可用但不干净的局部慢水平候选。它几乎滤掉 ETTh1 的24步振荡（以下选例中 A 的24步能量占比均值低于0.001），但 replicate padding 加上末点锚定使最后72步曲率相对内部放大：两组极端样本的尾部/内部曲率比分别为1.31和1.50。紫线末端被牵向最后观测点的弯折确实可见，不能当作可靠的末端真实趋势。", "",
        "3. `smooth_multiscale` 不应称为 global smooth。其当前实现是 `G_24(X)-G_72(X)`，即 two-scale difference / 中频宽波包，而非全局平滑趋势。它同样几乎不包含精确24步频率，却仍包含百步量级的起伏，并有同样的右端边界风险。", "",
        "4. Only-A 较优通常发生在未来不再遵循历史的快速周期、只需稳定 level/bias 校正时；X-A 较优则发生在预测仍需要局部相位、振幅或尖锐转折时。这是条件性分支信息分工，不是全体样本的平均效果断言。", "",
        "## 1. 审计设置", "",
        "ETTh1 validation、channel 0、L=720→H=96、seed=2021。Baseline-full、X-A、Only-A 均使用已有完整训练 checkpoint；本轮只推理和审阅图片。每种候选取八个 `Only-A MAE−X-A MAE` 最大值（X-A较优）和八个最小值（Only-A较优），同一候选内起点至少相隔96。GT 仅用于误差角色选样，不参与训练或成分提取。", "",
        "所有成分末点锚定为 `A_t=f_t-f_719`。local smooth 使用 `f=G_24(X)`（双侧 replicate-padded Gaussian）；smooth-multiscale 使用 `f=G_24(X)-G_72(X)`。", "",
        "## 2. 全体验证指标（不能用极端样本替代总体）", "",
        "| A | Baseline-full MSE / MAE | X-A MSE / MAE | Only-A MSE / MAE | Only-A 相对 X-A |", "|---|---:|---:|---:|---:|",
        "| local smooth | 0.686489 / 0.557656 | 0.712586 / 0.571168 | 0.671222 / 0.556665 | −5.80% MSE / −2.54% MAE |",
        "| smooth-multiscale | 0.686489 / 0.557656 | 0.689857 / 0.559082 | 0.679850 / 0.560061 | −1.45% MSE / +0.18% MAE |", "",
        "multiscale 的 Only-A MSE 较低但 MAE 略高，说明总体方向本身并不稳定。下表和样图只回答何时两种路由会明显分歧。", "",
        "## 3. 32个代表样本的量化概览", "",
        "正的 `Only-A−X-A` 表示 X-A 更好。每格是8个极端样本的均值，不能解释成总体均值。", "",
        "| A | 更好路由 | Only-A−X-A MAE | 两路预测MAD | A的24步能量占比 | 尾部/内部曲率 |", "|---|---|---:|---:|---:|---:|",
    ]
    for component, label in (("smooth_local", "local smooth"), ("smooth_multiscale", "smooth-multiscale")):
        for direction, display in (("x_minus_a_better", "X-A"), ("only_a_better", "Only-A")):
            group = groups[component, direction]
            lines.append(f"| {label} | {display} | {average(group, 'only_minus_x_mae'):+.4f} | {average(group, 'route_curve_mad'):.4f} | {average(group, 'component_cycle24_share'):.6f} | {average(group, 'tail_to_interior_curvature'):.2f} |")
    lines += [
        "", "## 4. 图片审阅得到的模式", "",
        "### Local smooth", "",
        "- **X-A 更好时，未来仍需要精确日周期形状。** [origin 678](" + figure("smooth_local", "x_minus_a_better", 1, 678) + ") 的 GT 保持清晰、幅度变化的日周期；Only-A 压低了峰谷（MAE 0.5333），而 X-A 为0.2576。[origin 1353](" + figure("smooth_local", "x_minus_a_better", 2, 1353) + ") 也显示仅靠慢水平无法恢复后续峰值高度。", "",
        "- **Only-A 更好时，未来与历史快速振荡脱钩。** [origin 2290](" + figure("smooth_local", "only_a_better", 1, 2290) + ") 与 [origin 857](" + figure("smooth_local", "only_a_better", 2, 857) + ") 的未来没有遵循从历史学到的高幅规则日周期；Only-A 将预测压回较平缓的包络，MAE 从0.9503/0.9663降至0.6764/0.7115。", "",
        "- **边界伪影可见。** 在 [origin 2290](" + figure("smooth_local", "only_a_better", 1, 2290) + ")、[origin 857](" + figure("smooth_local", "only_a_better", 2, 857) + ") 和 [origin 1215](" + figure("smooth_local", "only_a_better", 8, 1215) + ")，紫线在最右端额外弯向锚点；这个形状由 padding 与锚定共同导致，不能被解释成预测性的末端趋势。", "",
        "### Smooth-multiscale", "",
        "- **X-A 更好时，Only-A 丢了未来峰谷的定位。** [origin 1361](" + figure("smooth_multiscale", "x_minus_a_better", 1, 1361) + ")、[origin 496](" + figure("smooth_multiscale", "x_minus_a_better", 2, 496) + ") 与 [origin 615](" + figure("smooth_multiscale", "x_minus_a_better", 6, 615) + ") 的 GT 都仍有日周期；Only-A 持续低估峰值，MAE 比 X-A 高约0.10–0.16。", "",
        "- **Only-A 更好时，历史周期振幅变得误导。** [origin 857](" + figure("smooth_multiscale", "only_a_better", 1, 857) + ") 中 full/X-A 路由延续了过高的周期振幅，而 Only-A 更接近实际较低的后续水平（0.7806 vs 1.1031）。[origin 2535](" + figure("smooth_multiscale", "only_a_better", 3, 2535) + ") 与 [origin 2047](" + figure("smooth_multiscale", "only_a_better", 7, 2047) + ") 也更少过冲。", "",
        "- **不是纯趋势。** 紫线围绕零上下摆动、又强制在最后归零，视觉上是宽波包偏移而非缓慢单调/连续 level。因此它适合诊断‘移除哪些中尺度信息会改变 NLinear’，但不适合作为论文中纯趋势 A 的主证据。", "",
        "## 5. 边界明确的结论", "",
        "证据支持：NLinear 的信息偏好依赖状态——当快速周期仍可预测，X-A 更有价值；当周期相位/振幅失配时，Only-A 的低频/宽尺度输入有时能更稳定地校正预测高度。", "",
        "证据不支持：它不能说明 PhaseFormer 不使用这些成分，也不能证明 local smooth 或 multiscale 是无伪影的最终趋势机制。若需要主线趋势候选，当前更干净的是没有双侧 padding 的 slow causal EMA；global linear 是保守的刚性对照。", "",
        "## 6. 完整可审计选例表", "", raw,
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with zipfile.ZipFile(OUT / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(REPORT, REPORT.name)
        for path in sorted((OUT / "figures").glob("*.png")):
            archive.write(path, f"figures/{path.name}")
    print(REPORT)


if __name__ == "__main__":
    main()
