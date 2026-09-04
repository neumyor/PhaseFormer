# Weak Residual 非对称输入成分实验计划

> 状态：实现与 validation 发现阶段已启动。目标分支：`weak_residual_exploration`。

## 1. 问题

检验一个时序成分 A 是否对 Weak Residual 中的 NLinear-style 残差校正路径具有**增量价值**：即使
PhaseFormer 主分支始终看到完整输入 `X`，NLinear 分支失去 A 后，整模型是否仍会显著退化。

候选 A 均标记为**趋势性成分**。它们从完整的、已经过数据集训练集 scaler 的历史窗口
`X∈R^(L×C)` 中逐样本逐变量确定性提取；不使用预测目标，也不按数据集、horizon 或结果调参。
所有 A 均满足 `A[L-1]=0`，因此 `B=X-A` 保留 NLinear 的末值 persistence anchor。

|编号|趋势性成分 A|冻结提取公式（`L=720`）|所检验的信息|
|---|---|---|---|
|A1|`cycle_levels`|令 `P=24,K=L/P`，`level_k=mean_p X[k,p]`，`A[k,p]=level_k-level_(K-1)`|周期之间的平均水平轨迹及最后周期相对历史的水平偏移|
|A2|`recent_linear`|对最后 96 个点做逐变量 OLS，斜率为 `b`，`A[t]=(t-(L-1))b`|近期线性上升/下降趋势|
|A3|`global_linear`|对完整 720 点做逐变量 OLS，斜率为 `b`，`A[t]=(t-(L-1))b`|全窗口长期线性漂移|
|A4|`smooth_local`|`S_24(X)` 为 replicate-pad 的高斯平滑（标准差 24、半径 72），`A[t]=S_24(X)[t]-S_24(X)[L-1]`|不依赖参数拟合的局部平滑趋势|
|A5|`smooth_multiscale`|`A[t]=[S_24(X)[t]-S_72(X)[t]]-[S_24(X)[L-1]-S_72(X)[L-1]]`|近期平滑趋势相对更长期平滑趋势的偏离/转向|

这里 `S_σ` 的卷积核为 `exp(-u²/(2σ²))` 后归一化，`u∈[-ceil(3σ),ceil(3σ)]`；边界采用
replicate padding。A4、A5 均是基于平滑获取的趋势性信息，而非二次曲线或曲率拟合。

此前的首个候选 A1 `cycle-levels`：对 `P=24`、`K=30` 个周期，

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
|Asymmetric-A1…A5|`X`|`B=X-A_i`|

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
- seed=2021；Baseline-full 有 `5 × 2 = 10` 次 full training，五个非对称条件各有 10 次，
  共 60 次 full training；
- 使用 validation 选择 checkpoint，并只在 validation 上比较/排序五类 A；A 的公式和所有超参数在
  运行前冻结；发现阶段不读取 test。

每个 setting 报告 Baseline-full 与 Asymmetric-cycle-levels 的 validation/test MSE、MAE，以及相对
变化。输出还应记录 phase forecast、residual forecast、fused forecast、静态 gate 和每路径误差，供
后续判断退化是否确实来自残差校正路径。

发现阶段结束后，只有在多个 setting 上相对 Baseline-full 出现稳定双指标退化的 A，才进入确认阶段：
使用同一冻结公式，在 seed=2021、2022、2023 上训练，并在所有选择已结束后读取 test。若没有稳定
信号，则停止，不根据 test 更换 A 或调参。

## 5. 实施与运行

实现位于 `src/models/asymmetric_trend_components.py`。模型先执行 `RevIN.normalize(X)`，随后仅对
残差分支执行 `RevIN.normalize_with_stats(X-A, stats)`；PhaseFormer 路径始终使用完整的 `X_norm`。
`weak_residual_asymmetric_component=none` 时残差路径直接复用 `X_norm`，保留历史弱残差实现。

发现阶段由下列命令顺序执行，断点续跑会跳过已完成的确定性 run：

```bash
/home/wangjing/miniconda3/envs/raft/bin/python scripts/run_weak_residual_asymmetric_trend.py --require-cuda --resume
```

运行记录、搜索子任务输出和监控日志均位于
`research_runs/weak_residual_asymmetric_trend_discovery/`，不写入 `tmp/`。每项训练为完整训练、
best-validation checkpoint；该阶段产生的指标只能用于候选发现，不是 test 结论。

## 6. 可支持的结论边界

若非对称条件比 baseline 更差，可表述为：

> 当 PhaseFormer 已拥有完整 X 时，某个冻结的趋势性成分 A 仍向 NLinear-style 残差校正提供增量预测信息。

不能表述为：

> PhaseFormer 完全没有使用 CycleLevels；或该实验已用 matched control 排除所有扰动/分布变化解释。
