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
|A6|`trend_filter`|令 `f=argmin_f 0.5∑_t(X[t]-f[t])²+λ∑_t|f[t]-2f[t+1]+f[t+2]|`，`A[t]=f[t]-f[L-1]`；`λ=100·std(X)·(1 hour/Δt)²`|允许数据驱动折点的连续分段线性中尺度趋势；ETTh1/Weather 取 `Δt=1h`，ETTm1 取 `.25h`|

这里 `S_σ` 的卷积核为 `exp(-u²/(2σ²))` 后归一化，`u∈[-ceil(3σ),ceil(3σ)]`；边界采用
replicate padding。A4、A5 均是基于平滑获取的趋势性信息，而非二次曲线或曲率拟合。

A6 使用一阶 trend filtering（对二阶差分施加 L1 惩罚）。训练实现采用固定 256 步的 GPU 批量
Chambolle--Pock primal--dual 求解，优化目标与上式相同；先按逐样本逐变量 `std(X)` 缩放、在标准化空间
固定惩罚后缩放回原空间，严格等价于表中的 `λ` 规则。它不在每个 forward 执行 CPU ADMM 或逐变量 CPU
线性代数。`κ=100` 由六个固定 validation 历史窗口的独立可视诊断预先冻结，未使用预测标签或 test。

### 尚未进入训练的单侧局部平滑候选

为排查 A4/A5 的右端 replicate-padding 伪影，额外实现但**尚未纳入 X-A 或 Only-A 训练**三种单侧候选：

|名称|冻结提取|当前可视诊断结论|
|---|---|---|
|`causal_ema`|`T[t]=0.08X[t]+0.92T[t-1]`，`A[t]=T[t]-T[L-1]`|无右端填充，但 ETTh1/ETTm1 上仍保留明显周期形状|
|`causal_local_linear`|仅用最近 72 步、权重 `exp(-i²/(2·24²))` 的加权 OLS，取当前截距 `T[t]`|无右端填充、局部转折灵敏，但当前尺度明显跟随主周期|
|`holt_local_linear`|`l[t]=.15X[t]+.85(l[t-1]+b[t-1])`，`b[t]=.03(l[t]-l[t-1])+.97b[t-1]`，`A[t]=l[t]-l[L-1]`|无右端填充，但比 EMA 更明显地保留周期振幅|

三者均只使用当前及过去的历史点；由于末点锚定，改变未来历史会共同平移此前的 A 值，但不会改变此前的
相对轨迹。固定 validation 样本的图形比较位于
`research_runs/causal_trend_component_visual_probe/`。该诊断没有训练预测模型、没有读取 test，不能据此作出
预测性能或分支利用结论。

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

## 7. 补充探针：NLinear 只保留趋势成分 A（only-A）

为检验残差分支在没有其余历史细节时能否仅依靠某项趋势性信息作出有用校正，补充固定的
`component_only` 条件。PhaseFormer 分支继续接收完整 `X`；NLinear 分支接收 `A` 本身，而非此前的
`X-A`。两条分支仍共享由完整 `X` 计算的 RevIN 统计量。A 保持既有末点锚定定义，因此该条件只提供
趋势的相对历史形状，而不额外泄漏原始末值水平。

固定运行范围为 ETTh1、Weather、ETTm1，`L=720 → H=96`、period=24、seed=2021、Huber、最多30 epoch，
以 validation 最优 checkpoint 汇总 MAE/MSE；不读取 test，也不做样本选择。比较对象复用上一阶段在同一
设置、同一 seed 训练的 Baseline-full weak-residual checkpoint。该对照回答的是“仅有 A 是否足以给
NLinear 分支提供增量校正线索”，不能单独证明因果利用或与 `X-A` 的强弱关系。

```bash
/home/wangjing/miniconda3/envs/raft/bin/python \
  scripts/run_weak_residual_asymmetric_only_trend.py --require-cuda --resume
```

原始训练日志与 checkpoint 位于
`research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch/`；只保留整体统计的审计包将写到
`research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96/`。

## 8. 补充探针：A6 trend filter 的 X-A 与 Only-A

固定 A6 后，在 ETTh1、Weather、ETTm1 上各运行 `L=720→H=96`、seed 2021、最多 30 epoch 的完整训练。
两种路由均复用已有同协议 Baseline-full，不重新训练 baseline；训练、checkpoint 选择和比较只读取
validation，绝不实例化 test loader。原始日志、checkpoint 和监控记录存入
`research_runs/weak_residual_asymmetric_trend_filter_h96_scratch/`，最终审计包单独保留在
`research_runs/weak_residual_asymmetric_trend_filter_h96_audit/`。

```bash
/home/wangjing/miniconda3/envs/raft/bin/python \
  scripts/run_weak_residual_asymmetric_trend_filter.py --require-cuda --resume
```

`minus_component` 表示 NLinear residual 接收 `X-A6`；`component_only` 表示其仅接收 `A6`。两者都共享
完整 `X` 的 RevIN 统计量，PhaseFormer 始终接收完整 `X`。
