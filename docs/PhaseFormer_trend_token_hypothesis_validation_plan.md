# B2 单机制架构假设验证计划

## 目标

以 `trend_token_direct`（B2）为唯一 reference，在相同数据划分、lookback、horizon、seed、loss、epoch 和 checkpoint 规则下，逐项验证 B2 样本级错误分析提出的架构假设。每个候选只打开一个机制字段，禁止把多个候选组合成新的 arm。

## 机制与唯一变化

| arm | B2 之外的唯一变化 | 设计假设 |
|---|---|---|
| `trend_token_dynamic_gate` | `trend_gate_mode=reliability` | gate 根据 token、phase attention entropy 和近期波动动态调整 |
| `trend_token_smooth` | `trend_direct_smooth_window=9` | direct correction 的 horizon 形状需要低频约束 |
| `trend_token_multi_token` | `trend_token_count=2` | 一个 token 不能同时表达 level、slope 和 curvature |
| `trend_token_selective_attention` | `trend_phase_attention_mode=null` | trend token 需要能拒绝不可靠的 phase interaction |
| `trend_token_factorized` | `trend_readout_type=factorized` | level 与 slope/shape 应使用可识别的分离 head |
| `trend_token_multiscale` | `trend_history_multiscale=true` | history encoder 需要局部和长程两个尺度 |
| `trend_token_robust` | `trend_history_robust=true` | 历史尖峰不应直接决定趋势 token |

## 实验矩阵

- 数据：ETTh1、ETTh2、ETTm1、ETTm2、Weather。
- Horizon：96、192、720；seed：2021、2022。
- 每个 setting：phase-only、B2 和 7 个候选，共 270 个 job。
- 训练：8 epochs，Huber，validation-only，8×A800，`--resume`。
- 指标：validation MAE/MSE、paired delta、胜负次数、参数量、trend diagnostics。
- 选择：不读取 test，不依据 test 或单个样本选择机制；所有候选固定进入汇总。

## 归因检查

`scripts/run_trend_token_validation.py` 在生成 manifest 前检查候选相对于 B2 只改变声明字段；`scripts/summarize_trend_hypothesis_matrix.py` 在读取 checkpoint 配置后再次逐 setting 核验。任一控制检查失败，矩阵结果不得用于架构结论。

## 判定规则

候选只有在 paired validation MAE 和 MSE 均不劣于 B2、且至少在两个数据集方向一致时，才进入下一轮多 seed 或 test-free robustness 验证。若总体均值改善但高误差组恶化，保留该机制作为条件性候选，不宣称统一提升。
