# Weak Residual 非对称输入成分实验计划

> 状态：设计已冻结，尚未实现或运行。目标分支：`weak_residual_exploration`。

## 1. 问题

检验一个时序成分 A 是否对 Weak Residual 中的 NLinear-style 残差校正路径具有**增量价值**：即使
PhaseFormer 主分支始终看到完整输入 `X`，NLinear 分支失去 A 后，整模型是否仍会显著退化。

首个冻结候选为 D3 `cycle-levels`：对 `P=24`、`K=30` 个周期，

```text
level_k = mean_p X[k,p]
A[k,p] = level_k - level_(K-1)
B = X - A
```

因此 `A` 在最后时刻为零，保留 NLinear 的最后值 persistence anchor。该选择来自 D7：周期水平波动和
最后周期相对历史的水平偏移，是 NLinear 修正 phase residual 收益最强的两个描述量。

## 2. 固定模型条件

仅比较 `weak_residual` 的两种输入可见性；模型容量、初始化、损失、优化器、学习率、batch size、训练
epoch 上限、early stopping 与 checkpoint 选择规则必须完全一致，并继承执行前冻结的基础训练 setting。

|条件|PhaseFormer 分支输入|NLinear residual 分支输入|
|---|---|---|
|Baseline-full|`X`|`X`|
|Asymmetric-cycle-levels|`X`|`B=X-A`|

训练、validation、最终 test 均保持相同的非对称输入可见性。目标 `y` 与时间标记保持不变。

本轮按用户要求**不加入 sham / matched control**。因此结果回答的是“该确定性 A 对 NLinear 残差路径的
增量价值”，不能把差异表述为已经完全排除任意分布变化影响的成分专属因果效应。

## 3. 强制的共享 RevIN 归一化约束

这是本实验的必要实现条件。不得让两个分支根据各自输入独立估计 RevIN 统计量。

对每个 batch：

```text
(X_norm, stats) = RevIN.normalize(X)
B_norm          = RevIN.normalize_with_stats(B, stats)
phase_hat        = PhaseFormerPath(X_norm)
residual_hat     = NLinearPath(B_norm)
fused_norm       = (1-gate) * phase_hat + gate * residual_hat
prediction       = RevIN.denormalize(fused_norm, stats)
```

其中 `stats` 只能由完整 `X` 估计一次，并同时用于两个路径及最终反归一化。这样唯一改变的是 NLinear
分支是否能读取 A；不会同时引入另一套均值/方差坐标系，避免把归一化偏移误判为 A 的作用。

Baseline-full 也必须遵循同一代码路径：令 `B=X`，则 `B_norm=X_norm`，以确认 flag-off 与历史
`weak_residual` 数值等价。

## 4. 评估范围与阶段

发现阶段固定：

- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- horizon：96、192；lookback=720、period_len=24；
- seed=2021；共 `5 × 2 × 2 = 20` 次 full training；
- 使用 validation 选择 checkpoint；A 的公式和所有超参数在运行前冻结；不根据 test 改动。

每个 setting 报告 Baseline-full 与 Asymmetric-cycle-levels 的 validation/test MSE、MAE，以及相对
变化。输出还应记录 phase forecast、residual forecast、fused forecast、静态 gate 和每路径误差，供
后续判断退化是否确实来自残差校正路径。

若 Asymmetric-cycle-levels 在多个 setting 上稳定双指标退化，才以相同冻结公式扩展 seed=2022、2023；
若无稳定信号，则停止，不扩大数据集/horizon 或更换 A 后追逐 test 数值。

## 5. 可支持的结论边界

若非对称条件比 baseline 更差，可表述为：

> 当 PhaseFormer 已拥有完整 X 时，CycleLevels 仍向 NLinear-style 残差校正提供增量预测信息。

不能表述为：

> PhaseFormer 完全没有使用 CycleLevels；或该实验已用 matched control 排除所有扰动/分布变化解释。
