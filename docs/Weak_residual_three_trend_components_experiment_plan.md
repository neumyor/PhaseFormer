# Weak Residual：三种去周期趋势成分的非对称输入实验

> 状态：已准备实现与 launcher；尚未启动完整训练。所有比较仅使用 validation，绝不读取 test。

## 1. 目标与结论边界

检验三种预先冻结、并通过历史频谱周期泄漏筛查的趋势成分，是否向 Weak Residual 的 NLinear 残差路径提供
增量信息。PhaseFormer 主分支始终接收完整历史 `X`；唯一改变的是 NLinear 分支的可见输入。

这项实验可支持“在完整 X 已提供给 PhaseFormer 时，A 对 NLinear-style 校正有无增量价值”的配对观察；它不含
sham/matched residual control，不能单独证明严格成分因果效应或“PhaseFormer 完全没有使用 A”。

## 2. 冻结候选与参数

历史频谱筛查只使用输入窗口、不使用预测标签。筛选约束是趋势在输入主周期及相邻频点的功率比不超过 0.10；
在满足约束的设置中选取最大更新增益。`causal_local_linear` 未通过 ETTm1 的约96步主周期泄漏约束，明确排除。

|成分|公式/实现|ETTh1|Weather|ETTm1|
|---|---|---:|---:|---:|
|`trend_filter`|256步 GPU Chambolle--Pock 近似：`min 0.5||X-f||²+λ||D²f||₁`；`λ=100·std(X)·(1hour/Δt)²`|`Δt=1h`|`Δt=1h`|`Δt=.25h`|
|`causal_ema`|`T[t]=αX[t]+(1-α)T[t-1]`|`α=.024`|`α=.024`|`α=.006`|
|`holt_local_linear`|`l[t]=αX[t]+(1-α)(l[t-1]+b[t-1])`；`b[t]=β(l[t]-l[t-1])+(1-β)b[t-1]`|`α=.024,β=.006`|`α=.024,β=.006`|`α=.006,β=.0015`|

所有成分均采用 `A[t]=T[t]-T[L-1]`，因此 `A[L-1]=0`。A6 是固定迭代近似，不得称为逐窗口精确 trend-filter 解。

## 3. 固定训练与路由协议

- 数据集：ETTh1、Weather、ETTm1。
- 设置：`L=720 → H=96`、seed=2021、Huber、最多30 epoch、best-validation checkpoint。
- 参照：复用此前同 setting/seed 的 Baseline-full Weak Residual；PhaseFormer 与 NLinear 均输入完整 `X`。
- 归一化：每个 batch 的 RevIN stats 只由完整 `X` 估计一次，并同时用于 PhaseFormer、NLinear 和反归一化。
- 不训练新的 baseline，不读取 test，不用预测结果调整候选参数。

|路由|PhaseFormer 分支|NLinear 分支|
|---|---|---|
|X-A (`minus_component`)|完整 `X`|`X-A`|
|Only-A (`component_only`)|完整 `X`|`A`|

完整矩阵为 `3 数据集 × 3 成分 × 2 路由 = 18` 项 candidate full trainings。X-A 与 Only-A 是独立重训，均相对同一
Baseline-full 比较；二者的直接差异只能说明 A 与 A 外信息在该实验内的相对充分性，不能替代因果对照。

## 4. 启动与产物路径

完整训练命令（支持断点续跑）：

```bash
/home/wangjing/miniconda3/envs/raft/bin/python \
  /home/wangjing/PhaseFormer/scripts/run_weak_residual_trend_comparison.py \
  --require-cuda --resume
```

仅打印 18 项计划、不启动训练：

```bash
/home/wangjing/miniconda3/envs/raft/bin/python \
  /home/wangjing/PhaseFormer/scripts/run_weak_residual_trend_comparison.py \
  --dry-run
```

原始训练日志、checkpoint 和监控记录必须写入：

```text
/home/wangjing/PhaseFormer/research_runs/weak_residual_trend_comparison_h96_scratch/
```

完整训练结束后，严格六文件审计包应单独生成到：

```text
/home/wangjing/PhaseFormer/research_runs/weak_residual_trend_comparison_h96_audit/
```

审计包需报告每个 setting 的 Baseline-full、6 个 candidate 指标、样本级误差和程序化案例；训练中间产物不得混入
审计目录。

## 5. 待填实验结果表

`Δ = candidate − Baseline-full`；正数表示变差。所有值均应为 validation MSE/MAE。

|数据集|Baseline MSE / MAE|成分|X-A MSE / MAE|X-A ΔMSE / ΔMAE|Only-A MSE / MAE|Only-A ΔMSE / ΔMAE|
|---|---:|---|---:|---:|---:|---:|
|ETTh1|—|trend_filter|—|—|—|—|
|ETTh1|—|causal_ema|—|—|—|—|
|ETTh1|—|holt_local_linear|—|—|—|—|
|Weather|—|trend_filter|—|—|—|—|
|Weather|—|causal_ema|—|—|—|—|
|Weather|—|holt_local_linear|—|—|—|—|
|ETTm1|—|trend_filter|—|—|—|—|
|ETTm1|—|causal_ema|—|—|—|—|
|ETTm1|—|holt_local_linear|—|—|—|—|

## 6. 实施前验证

- [x] 三种 candidate 均通过组件 shape、末点锚定、flag-off 等价和真实 PhaseFormer forward 单测。
- [x] launcher dry-run 必须列出恰好18项训练。
- [ ] CUDA 1-epoch smoke：在驱动恢复后运行一项 `trend_filter` 与一项 `causal_ema`，分别覆盖两种路由；当前
  主机的 `nvidia-smi` 无法连接 driver，未虚报 GPU smoke 通过。
- [ ] 完整训练和最终审计包。
