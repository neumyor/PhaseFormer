# PhaseFormer 跨数据集 Golden 组合机制实验计划

> 信息锚点：`gold_combo_stability_v1`。本文件在实现和实验前冻结；后续选择、结果与失败结论均按本计划解释。

## 1. 目标与依据

目标是在 **ETTh2-720、ETTm2-96、Electricity-336** 上验证一个共同结构是否能稳定双指标超过固定 Golden。三者分别已有可复现信号：输出凸残差、相位不确定性+电平+高频修正、自适应输出残差。纯相位动态、轨迹解码和层内/多层残差未显示稳定增益，不再叠加。

固定 Golden（MSE/MAE）：ETTh2-720 `0.402/0.436`，ETTm2-96 `0.163/0.256`，Electricity-336 `0.165/0.257`。matched rerun 只作诊断，不替代 Golden。

## 2. 假设与候选

缺陷假设：同相位历史可靠时应保留相位预测；同相位噪声大时应更多采用直接时序残差。固定门无法在不同数据特性间切换，既有 MLP 门又未显式复用相位不确定性证据。

新增 **Reliability-Coupled Residual Fusion（RCRF）**：

```text
r = Var_l(mean_k x_lk) / (Var_l(mean_k x_lk) + mean_l Var_k(x_lk) + eps)
s = s_max * tanh(s_raw)
alpha = sigmoid(logit(alpha_0) + s * (1 - r))
y = (1 - alpha) * y_phase + alpha * y_residual
```

`r` 为样本×通道可靠度；`alpha` 为样本×通道凸融合门。门的灵敏度有界，残差头沿用共享 NLinear 弱周期头。可靠度从 shrinkage 前的原始相位序列计算，避免修正模块改变自身门控证据。

所有组合候选共享相位栈：uncertainty min `0.2`、trend gate `0.05`、period-level gate `0.2`/slope gate `0.05`、high-frequency strength `0.8`/threshold `0.5`/window `7`；共享残差门先验 `alpha_0=0.5`。

| mode | 相位栈 | 输出融合 |
|---|---|---|
| `original` | 无 | 无 |
| `latest` | 当前数据集策略 | 当前数据集策略 |
| `gold_combo_fixed` | 开 | 固定门 0.5 |
| `gold_combo_adaptive` | 开 | 既有三特征 MLP 门 0.5 |
| `gold_combo_reliability_s0` | 开 | RCRF，初始灵敏度 0 |
| `gold_combo_reliability_s2` | 开 | RCRF，初始灵敏度 2 |

## 3. 协议与选择隔离

- 公共设置：lookback `720`、period `24`、标准数据划分/缩放、best-validation checkpoint。
- ETTh2-720：Huber、lr `1e-3`；ETTm2-96 与 Electricity-336：MAE、lr `3e-4`；批量沿用目标正式配置。
- Stage A：seed `2021`，训练集 `30%`，最多 `8` epochs，**只计算 validation**。运行上述 6 个 mode。
- Stage A 排名：先对每个 setting 计算相对 `original` 的 `val_mse` 与 `val_mae` 比率，再取 6 个比率的均值；只在四个 `gold_combo_*` 中选择最小者。平局依次取参数更少、初始灵敏度更小者。
- 无论筛选是否正向，都冻结且仅冻结排名第一的组合进入 Stage B；筛选结果只决定候选，不决定是否报告失败。
- Stage B：`original`、`latest`、冻结候选；seeds `2021/2022/2023`；全数据、正式 epoch/patience、best-validation checkpoint；冻结后才允许读取 test。
- 不用 Golden 或 test 选择 mode、epoch、seed 或超参数。

## 4. 成功标准

单 setting 的“稳定双指标超过 Golden”同时要求：

1. 三个 seed 的 MSE、MAE 均低于 Golden；
2. 对 MSE、MAE 都满足 `mean + sample_std < Golden`。

跨数据集成功要求至少 `2/3` settings 达到上述标准，且剩余 setting 的三 seed 均值相对 Golden 的 MSE、MAE退化均不超过 `1%`。另报告相对 matched `original` 和 `latest` 的均值差、标准差，不把三位小数 Golden 的舍入差异包装为强结论。

## 5. 测试与审计

- 单元测试：可靠度极限、门形状/范围、初始门、灵敏度方向、互斥开关、前向/反向、flag-off 回归、preset/search 可达、seed 传递。
- smoke：三类数据至少各一次短训练；检查有限 loss、checkpoint、validation-only 不创建 test loader。
- 最终样本级分析以 `latest` 为 baseline、冻结组合为 candidate，在每个 setting×seed 上重算 sample×channel MSE/MAE；程序化选择 baseline 高误差、candidate 退化、candidate 改善各 top-10。
- 规范审计目录 `research_runs/gold_combo_stability_v1/` 最终只含：`run.yaml`、`results.csv`、`sample_errors.csv`、`selected_cases.npz`、`objective_error_analysis.md`、`objective_error_analysis.zip`、`figures/`。原始训练目录另存于被忽略的临时路径。
- 汇总结果写入 `docs/PhaseFormer_gold_combo_results.md`；命令、提交、选择理由与失败项写入 `docs/agent-log.md`。

## 6. 停止规则

只执行本表四个组合候选，不根据 Stage A 临时追加超参数。Stage B 后不再回看 test 修改结构；若未达成功标准，结论即为“本轮没有找到跨数据集稳定超过 Golden 的组合”，并保留全部负结果。
