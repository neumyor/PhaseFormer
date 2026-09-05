#!/usr/bin/env python3
"""Render the narrative report for the completed global/EMA route-role audit."""

from __future__ import annotations

import csv
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_runs/global_ema_route_role_cases"
REPORT = OUT / "objective_error_analysis.md"


def figure(row: dict[str, str]) -> str:
    return (f"{row['dataset']}__{row['component']}__{row['direction']}__"
            f"{int(row['rank']):02d}_origin{row['origin']}_channel0.png")


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def main() -> None:
    rows = list(csv.DictReader((OUT / "sample_errors.csv").open(encoding="utf-8")))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["component"], row["direction"]].append(row)
    for group in groups.values(): group.sort(key=lambda row: int(row["rank"]))

    lines = [
        "# 从趋势候选到 NLinear 分支角色：Global-linear 与 Causal-EMA 双向样本审计", "",
        "## 1. 问题与实验边界", "",
        "目标不是问 PhaseFormer 是否完全不使用趋势。两种候选中 PhaseFormer **始终接收完整 X**；我们只改变 NLinear 弱残差分支的输入：`X-A` 或 `Only-A`。因此，此实验识别的是 **趋势 A 对 NLinear 校正分支的条件性作用**。", "",
        "全部结果为 validation-only、channel 0、L=720→H=96、seed=2021。对于每个数据集与成分，按 `Only-A MAE − X-A MAE` 选取五个最大正值（X-A 更好）和五个最小负值（Only-A 更好），样本间至少相隔一个 horizon。这里 GT 只用于选择“哪条路由误差更低”；它没有用于训练或趋势提取。", "",
        "## 2. 已比较的趋势候选，以及为何聚焦两个成分", "",
        "当前 prediction-divergence gallery 覆盖七种候选：", "",
        "| 成分 | 视觉/定义审阅结论 | 是否进入深入角色分析 |", "|---|---|---|",
        "| `cycle_levels` | 每周期均值的阶梯水平；描述跨周期 level，不是连续通用趋势 | 否 |",
        "| `recent_linear` | 最近96步斜率被外推至完整720步，出现非常大的末端锚定斜坡 | 否：明显边界伪影 |",
        "| `global_linear` | 没有周期泄漏和局部边界伪影，但只表达一个全局直线 | 是：作为最干净、最保守的趋势基线 |",
        "| `smooth_local` | 能追踪局部水平，但多个样本右端出现弯折/抬升，存在边界风险 | 否 |",
        "| `smooth_multiscale` | 短/长 Gaussian 差分，本质是中频残差，ETTm1 中仍保留明显周期 | 否：不符合纯趋势定义 |",
        "| `causal_ema` | 单侧平滑、无右侧 padding；比全局直线更能表示连续慢水平，周期泄漏较少 | 是：当前最有用的柔性慢趋势候选 |",
        "| `holt_local_linear` | ETTh1/Weather 尚平滑，但 ETTm1 图中仍有重复宽周期起伏 | 否：跨数据集纯趋势性不够稳定 |",
        "",
        "因此，`global_linear` 回答“最粗的全局漂移有没有用”，`causal_ema` 回答“平滑但可弯曲的慢水平有没有用”。两者构成了最有解释力的刚性/柔性趋势对照。", "",
        "## 3. 双向选样的量化概览", "",
        "正的 `Only-A − X-A` 表示 X-A 更好；负值表示 Only-A 更好。下表的数值是每组五个极端样本的均值，**用于描述代表性状态，不是总体平均效应或显著性检验**。`周期能量` 为 ETTh1/Weather 的24步、ETTm1的96步频率能量占比。", "",
        "| Dataset | A | 方向 | Only-A−X-A MAE | 最近96步水平变化 | 周期能量 | A 范围 |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("ETTh1", "Weather", "ETTm1"):
        for component in ("global_linear", "causal_ema"):
            for direction in ("x_minus_a_better", "only_a_better"):
                group = groups[dataset, component, direction]
                lines.append(f"| {dataset} | `{component}` | `{direction}` | {mean(group, 'only_minus_x_mae'):+.4f} | {mean(group, 'recent_level_shift_96'):+.4f} | {mean(group, 'cycle_energy_share'):.3f} | {mean(group, 'component_range'):.3f} |")
    lines += [
        "", "## 4. 有代表性的可视化证据与发现", "",
        "### 4.1 当 Only-A 更好：NLinear 使用 A 作为低频水平/偏置校正", "",
        "这类案例的共同点不是“任何趋势都够用”，而是历史中的未来相关部分主要是平滑、低频且持续的水平状态；A 本身可提供未来预测的整体高度与慢方向。", "",
        "- **Global-linear / Weather / origin 3111**：历史呈大尺度单向下行，Only-A MAE=`0.0605`，显著低于 X-A=`0.2690`。仅给全局漂移时，NLinear 的输出与缓慢下降的 GT 更接近。[图](figures/Weather__global_linear__only_a_better__01_origin3111_channel0.png)",
        "- **Global-linear / ETTm1 / origin 6745**：Only-A MAE=`0.2106`，低于 X-A=`0.5374`；图中未来主要是平滑水平恢复而不是要求精确追踪每个历史周期。[图](figures/ETTm1__global_linear__only_a_better__05_origin6745_channel0.png)",
        "- **Causal-EMA / ETTh1 / origin 2533**：Only-A MAE=`0.6223`，低于 X-A=`0.9395`。EMA A 给出明显的大尺度水平爬升，Only-A 分支更能校正预测的整体高度。[图](figures/ETTh1__causal_ema__only_a_better__01_origin2533_channel0.png)",
        "- **Causal-EMA / Weather / origin 2073**：Only-A MAE=`0.1523`，低于 X-A=`0.4242`；慢水平包络比局部残差更接近未来的总体走向。[图](figures/Weather__causal_ema__only_a_better__01_origin2073_channel0.png)",
        "",
        "跨数据集的辅助证据是：Only-A 更好组的周期能量均低于 X-A 更好组（ETTh1 EMA `0.549<0.642`；Weather EMA `0.00015<0.00041`；ETTm1 EMA `0.243<0.292`）。Global-linear 的 Only-A 优势在 ETTm1 和 ETTh1 还更常伴随近期负向水平变化。", "",
        "### 4.2 当 X-A 更好：NLinear 需要趋势之外的相位、幅度与局部动态", "",
        "在强周期、近期转折或局部状态快速变化时，Only-A 将 NLinear 的输入压缩为过平滑的 A，丢掉了对预测形状至关重要的残差信息；X-A 仍保留这些变化。", "",
        "- **Global-linear / Weather / origin 591**：Only-A MAE=`0.5098`，而 X-A 仅=`0.1386`。历史先快速上升再回落，全局直线无法表示这种弯曲；Only-A 预测近似错误的平滑偏置，X-A 更能跟随未来形状。[图](figures/Weather__global_linear__x_minus_a_better__01_origin591_channel0.png)",
        "- **Global-linear / ETTm1 / origin 3439**：存在强重复周期及后段状态变化，Only-A MAE=`1.2504`，高于 X-A=`0.7009`。[图](figures/ETTm1__global_linear__x_minus_a_better__01_origin3439_channel0.png)",
        "- **Causal-EMA / Weather / origin 3944**：在显著水平上升/转折后，X-A MAE=`0.0464`，Only-A=`0.4943`；说明此时保留局部剩余动态比单独慢趋势重要。[图](figures/Weather__causal_ema__x_minus_a_better__01_origin3944_channel0.png)",
        "- **Causal-EMA / ETTm1 / origin 9073**：未来仍由周期相位和幅度决定，Only-A MAE=`1.0297`，高于 X-A=`0.5145`；EMA 单独输入不能保留足够的周期定位信息。[图](figures/ETTm1__causal_ema__x_minus_a_better__01_origin9073_channel0.png)",
        "",
        "## 5. 对 A 在 NLinear 分支中作用的结论", "",
        "1. A 的作用是**条件性的低频状态校正**：在平滑水平、持续漂移或非周期包络主导时，Only-A 可优于 X-A，说明 NLinear 能将 A 转换为预测的 level/bias 校正。", "",
        "2. A 不是完整预测表征：在强周期、相位敏感、快速转折的样本中，Only-A 系统性失败，表明 NLinear 还需要趋势外残差所包含的短时形状和周期定位。", "",
        "3. Global-linear 的价值是诊断性的：它证明非常粗的线性漂移在少数单向状态中足够，但总体上过于刚性。Causal-EMA 是更合理的慢状态候选：它保留连续低频水平，又不会像 local smoothing 那样依赖右端 padding。", "",
        "4. 这不等于 PhaseFormer 没有使用 A。PhaseFormer 的输入在两条候选路由中恒为完整 X；证据只支持“NLinear 校正分支在何种状态下从 A 或 X-A 中获益”。", "",
        "## 6. 全部选例", "",
    ]
    for dataset in ("ETTh1", "Weather", "ETTm1"):
        for component in ("global_linear", "causal_ema"):
            lines += [f"### {dataset} / `{component}`", "", "| 方向 | rank | origin | X-A MAE | Only-A MAE | Only-A−X-A | 图 |", "|---|---:|---:|---:|---:|---:|---|"]
            for direction in ("x_minus_a_better", "only_a_better"):
                for row in groups[dataset, component, direction]:
                    lines.append(f"| `{direction}` | {row['rank']} | {row['origin']} | {float(row['x_minus_a_mae']):.4f} | {float(row['only_a_mae']):.4f} | {float(row['only_minus_x_mae']):+.4f} | [figure](figures/{figure(row)}) |")
            lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(OUT / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(REPORT, REPORT.name)
        for path in sorted((OUT / "figures").glob("*.png")): archive.write(path, f"figures/{path.name}")
    print(REPORT)


if __name__ == "__main__":
    main()
