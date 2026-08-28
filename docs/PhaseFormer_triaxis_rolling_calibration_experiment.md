# TriAxis-Former v2：多截点历史校准实验

> 状态：预注册，尚未读取本轮任何新 test。v1 结果已经暴露，因此 v2 的机制、候选和门槛必须
> 先固定在本文；所有候选只用 validation 排序，通过后才能进入正式 test。

## 1. v1 缺陷与修正假设

v1 只在历史最后一个周期上做一次一步伪预测，却把该误差用于未来四个周期；同时路由器末层
零初始化使伪风险最初完全不参与权重。其三专家 oracle 宏平均空间为 47.80%，但可部署路由的
最佳专家命中率只有 34.54%–39.27%，说明主要问题是代理任务和路由约束，而非专家不互补。

v2 固定三项修正：

1. **多截点、等 horizon 回测**：对历史最后四个可观测目标周期做 rolling-origin 回测，并分别
   估计未来第 1/2/3/4 个周期的 phase/trajectory/cycle 风险；近期截点权重更高。
2. **风险单调先验**：先将三个风险在专家维标准化，再显式加入 `-softplus(β)·risk`；因此初始时
   历史回测风险越低的专家权重越高，而不是依赖 MLP 从零重新发现这个方向。回测不一致时按
   origin 间方差收缩到可学习的全局专家 prior。
3. **周期级校准**：路由监督不再使用噪声较大的逐点 oracle，而以每个未来 24 步周期内的专家
   平均平方误差构造 soft oracle；推理仍只读取历史。

这些修正不增加第四个专家、不使用 dataset ID、不读取 future values/marks，也不改变已有 preset。

## 2. 候选与对照

| ID | preset | 目的 |
|---|---|---|
| A1 | `gold_combo_reliability_s2` | 当前 RCRF+NLinear 配对基线 |
| T2-v1 | `triaxis_self_validating` | 单截点历史自验证父模型 |
| R0 | `triaxis_rolling_features` | 仅替换为多截点风险，仍由零初始化 MLP 学习 |
| R1 | `triaxis_rolling_prior` | R0 + 显式风险单调先验 |
| R2 | `triaxis_rolling_calibrated` | R1 + 周期级 oracle KL，完整候选 |

R0/R1/R2 共享三专家、参数量、优化器、`λ_expert=0.2`；R0/R1 的 `λ_route=0`，R2 为 0.1。
风险 prior 初始强度固定为 1.0，MLP 修正幅度上限固定为 0.5，不进行网格调参。

## 3. 分阶段测试

### Stage 0：实现测试

- rolling-origin 的每个目标严格位于对应伪起点之后，但全部位于输入窗口内；修改 decoder value、
  future mark 或真实 target 不得改变路由权重。
- 检查 1–4 周期风险 shape/finite、origin 数、风险单调性、低置信度收缩、权重归一化、梯度、
  H96/H192 前向和已有 preset state-dict 隔离。
- 仓库完整单元测试必须通过。

### Stage A：validation-only 修正筛选

- ETTh2-96、ETTm2-96、Weather-96、Electricity-96；L=720、P=24、seed 2021、30% train、
  最多 8 epoch、最低 validation loss checkpoint。
- 执行 R0/R1/R2 共 12 个新 run。A1 和 T2-v1 可复用 v1 的完全相同 run，但必须用本轮分析器
  复算 validation 指标并验证误差 `<1e-5`。
- 先按 8 个 MSE/MAE 比值宏平均选择 R0/R1/R2，保留全部尝试。冠军只有满足以下任一条件才冻结：
  1. 相对每个 setting 中 A1/I0 较优指标，宏平均比值 `<0.995` 且最差 `≤1.005`；或
  2. 至少 3/4 setting 双指标改善，剩余最大回退 `≤0.5%`。
- 若全部失败，立即停止，不读取 test；不得按数据集硬切换候选。

### Stage B：仅门槛通过后

冻结单一候选后，运行 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity 的 H96/H192、full train、
seeds 2021/2022/2023，并与 A1、I0、Golden 比较。沿用 v1 的正式扩展门槛；未通过 Stage A 时
本节标记“不适用”。

## 4. 三专家相对优势区间

只在 validation 上，用最终 Stage-A 冠军的三个原子专家计算，不能解释为 test 泛化规律：

- 最细粒度为 `sample×channel×future-cycle(24 steps)`；以周期段 MSE 最低者为 winner。
- 报告四个 horizon 区间 `[1,24]`、`[25,48]`、`[49,72]`、`[73,96]` 的专家 MSE、胜率和
  相对第二名优势。
- 对 `lag-24 correlation`、近期水平漂移、相位可靠度、周期形状创新量、rolling 风险 margin
  分 setting 做十分位分箱。某专家的优势区间定义为：样本数不少于 200、该专家条件胜率相对其
  全局胜率 lift `≥1.15`，且 1000 次 bootstrap 95% 置信区间下界高于其全局胜率；相邻合格箱合并。
- 若没有满足条件的区间，明确报告“没有稳定优势区间”，不得只挑有利阈值。

## 5. 审计产物与结论边界

实验编号为 `triaxis_rolling_calibration_v2`。严格生成一组 `run.yaml`、`results.csv`、
`sample_errors.csv`、`selected_cases.npz`、`objective_error_analysis.md`、
`objective_error_analysis.zip` 和 `figures/`。案例按程序规则覆盖 baseline 高误差、候选退化和
候选改善，总数不超过 10。只有 Stage B 通过才能更新 incumbent 或声称新的 Golden 提升。
