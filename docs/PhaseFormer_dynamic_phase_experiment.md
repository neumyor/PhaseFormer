# 动态相位与自适应残差实验：完整闭环

## 1. 实验要验证的设想

弱周期序列存在相位提前/延迟和相位速度变化；动态 phase trajectory、圆周几何交互及自适应 residual gate 可能分别改善相位对齐、token 交互和 phase/residual 融合。

## 2. 实验的整体计划

依次比较 baseline、phase correction、phase velocity、circular geometry、phase rotation、harmonic modulation、adaptive residual 及最终组合；每个模块单独启用并做完整消融，同时记录 peak-shift、轨迹和 gate 分析，最终以 full-budget test 和 matched baseline 判定。

## 3. 每个实验的实现方式和结果

`PhaseVelocity` 用速度编码与轨迹积分生成动态相位；`Circular Attention Bias` 用圆周距离修正 QK 交互；`AdaptiveResidualGate` 学习逐通道融合权重。结果显示动态相位在 ETTh2 等 setting 有局部改善，但多数 setting 接近 baseline；几何交互和组合模型未表现出稳定的跨数据集增益。自适应 gate 的内部量是活跃的，但总体收益仍受 residual 拓扑和数据集差异限制。

完整数值、peak-shift、消融和分析见 `docs/PhaseFormer_dynamic_phase_report.md`、`docs/PhaseFormer_dynamic_phase_results.md` 及 `docs/PhaseFormer_adaptive_residual_results.md`。

## 4. 最终结论

动态相位假设得到局部而非普遍支持；自适应融合是更有前景的方向，但必须与相位可靠度和输出端残差进行严格归因。当前不把动态相位组合直接作为全局默认模型。
