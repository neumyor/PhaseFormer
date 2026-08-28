# TriAxis-Former 自验证三轴融合实验

> 状态：已完成并在 Stage A 停止。模型设想来自已经暴露的历史 test 结果；本轮新配置只用
> validation 筛选。T0/T1/T2 均未通过冻结门槛，因此没有读取新 test，也没有更新 incumbent。

## 0. 实际结论（2026-08-28）

- 20 个 Stage A run 全部完成；168 项仓库测试通过。
- 三专家逐点 oracle 相对最佳单专家的 8 个 validation 指标宏平均改善 47.80%，证明专家有
  样本级互补性；但这是使用真实标签的不可部署上界。
- T2 相对每个 setting 的 A1/I0 较优指标：ETTh2、ETTm2 回退，Weather、Electricity 改善；
  8 指标宏平均比值 1.0005，最差 1.0426，仅 2/4 setting 双指标改善，冻结失败。
- 替补 T0/T1 也失败；按预注册协议不运行 Stage B/C，不产生新的 Golden/test 提升结论。
- 完整审计位于
  `research_runs/triaxis_self_validating_v1/objective_error_analysis.md`，便携包为同目录
  `objective_error_analysis.zip`。

| dataset | A1 MSE/MAE | I0 MSE/MAE | T0 MSE/MAE | T1 MSE/MAE | T2 MSE/MAE |
|---|---:|---:|---:|---:|---:|
| ETTh2 | .211136/.322032 | .229856/.339966 | .218336/.329535 | .214192/.320545 | .214768/.324585 |
| ETTm2 | .119617/.235593 | .140011/.258591 | .126302/.244074 | .120257/.236566 | .124714/.240781 |
| Weather | .418749/.293928 | .405543/.299233 | .394877/.284316 | .404460/.294693 | .396218/.286173 |
| Electricity | .114288/.209123 | .114085/.206621 | .111814/.205205 | .111172/.203637 | .111350/.204047 |

## 1. 假设

长度为 `L=K×P` 的历史可重排为周期矩阵 `X∈R^(K×P)`。当前证据支持三个互补视角：

- PhaseFormer 沿相位轴聚合同相位的跨周期证据，适合稳定相位结构；
- NLinear 沿原始时间轴直接映射近期轨迹，适合水平和趋势漂移；
- ICPT 沿周期间轴建模完整周期的形状、幅值和水平演化。

简单三分支平均或黑盒门控不是本实验的主张。核心候选使用历史内最后一个已知周期作为
伪预测目标，分别计算相位模板、局部线性趋势和周期间增量延续的无未来泄漏风险，再按
sample×channel×future-cycle×phase-slot 生成三专家权重。训练集未来标签只用于监督路由器
逼近三个专家的相对误差；推理时路由器只能读取输入历史。

## 2. 模型与对照

共享 PhaseFormer phase stack、RevIN、数据划分、loss、学习率、batch 和 best-validation
checkpoint 协议。候选均使用 `P=24`，ICPT 使用已正式测试的 future-query decoder、无 PE。

| ID | preset | 作用 |
|---|---|---|
| A1 | `gold_combo_reliability_s2` | PhaseFormer + NLinear，当前直接基线 |
| I0 | `rcrf_icpt_none` | PhaseFormer + ICPT，周期间专家对照 |
| T0 | `triaxis_uniform` | phase/NLinear/ICPT 固定均匀融合 |
| T1 | `triaxis_structural` | 只用相位可靠度、漂移和周期间创新量的学习路由 |
| T2 | `triaxis_self_validating` | 完整历史自验证路由 + 专家/路由辅助损失 |

T0–T2 在同一模型中并行计算三个原子专家，不对已经融合过的 A1/I0 输出再次集成。
feature flag 关闭时不得改变任何已有 preset 的初始化、state dict 或前向结果。

## 3. 分阶段协议

### Stage 0：实现与互补性门槛

- 单元测试：shape/finite、三权重归一化、历史风险不读取未来 marks/target、future-cycle/phase
  因子化、辅助损失梯度、旧 preset flag-off 等价。
- 使用已有 checkpoint 和新 Stage A 输出计算 sample×channel×horizon 的专家误差相关、获胜率、
  oracle top-1/convex 上界。若 oracle 相对每格最佳单专家的 8 指标宏平均改善 `<1%`，停止主线。

### Stage A：validation-only 筛选

- setting：ETTh2-96、ETTm2-96、Weather-96、Electricity-96。
- seed 2021、30% train、最多 8 epoch；不实例化 test loader。
- 执行 A1/I0/T0/T1/T2，共 20 run，保留全部结果。
- T2 只有满足以下任一条件才冻结：
  1. 相对每个 setting 中 A1/I0 较优者，8 个 MSE/MAE 比值宏平均 `<0.995`，最差 `≤1.005`；或
  2. 至少 3/4 setting 双指标改善，剩余 setting 最大回退 `≤0.5%`。
- T2 未通过时允许在不读取 test 的前提下判断 T0/T1 是否满足同一门槛；均失败则停止。

### Stage B：短/中 horizon 正式确认

冻结单一候选后，运行 ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity 的
`720→96/192`，seeds 2021/2022/2023，full train，最低 validation loss checkpoint，随后
一次性读取 test。对照为 A1、I0 和固定 Golden；已存在且协议完全一致的 checkpoint 可复用，
否则成对重跑。

进入全 horizon Stage C 必须同时满足：

- 12 个 setting 中至少 8 个相对 A1/I0 较优者双指标改善；
- 24 指标宏平均至少改善 0.5%；
- 任一平均指标回退不超过 0.5%；
- 至少 6 个 setting 按 `all seeds < Golden` 且 `mean+sample_std < Golden` 稳定超过 Golden。

### Stage C：完整数据集矩阵

Stage B 通过后，补齐 ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic 的
96/192/336/720，共 28 setting、三 seed。Exchange 作为 matched rerun 附加结果，不声称
超过缺失的 Golden。七个数据集不得使用 dataset ID、数据集专属分支开关或按 horizon 切换专家。

## 4. 训练目标与诊断

完整候选训练目标为：

`L = L_fused + λ_expert·mean(L_phase,L_linear,L_icpt) + λ_route·KL(w_oracle || w_history)`。

固定 `λ_expert=0.2`、`λ_route=0.1`、oracle temperature `0.2`；这些值只能在 Stage A
validation 上改变并保留所有尝试。诊断至少记录三类历史伪风险、三专家权重、权重熵、
专家实际误差、预测步分段误差和 oracle headroom。权重若在所有 setting 上都选择同一专家
超过 95%，判为路由塌缩，即使汇总指标改善也不能支持机制主张。

## 5. 样本级审计与产物

正式实验编号为 `triaxis_self_validating_v1`。按项目 Skill 只保留六个审计文件和
`figures/`：`run.yaml`、`results.csv`、`sample_errors.csv`、`selected_cases.npz`、
`objective_error_analysis.md`、`objective_error_analysis.zip`。逐 setting 程序化选择 baseline
高误差、candidate 显著退化、candidate 显著改善各最多 8 个非重叠案例；图中同时显示 history、
truth、baseline、candidate 和三专家预测，并在文字中报告对应权重与历史风险。

## 6. 结论边界

本轮首先验证“三类归纳偏置是否形成可预测的样本级互补”，而不是预设三分支必然优于所有
数据集。只有 Stage B/C 的统一模型通过冻结门槛，才能更新 incumbent；未通过也必须完整报告
oracle headroom、路由校准和失败样本，不能改成按数据集硬切换后声称统一提升。
