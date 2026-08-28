# Safe-Regret TriAxis：A1 锚定与可拒绝集成实验

> 状态：已完成，最终 validation gate 失败，未读取 test。本文的第 1–7 节为结果产生前的
> 预注册协议；第 8 节追加实际结果，不回写原假设和门槛。

## 1. 问题与可证伪假设

TriAxis v2 的三个专家存在条件优势，但 R0 必须在三个专家中混合，没有“保持 A1”的动作；其
监督学习三专家内部 winner，而不是相对 A1 的可兑现收益。结果是 Weather/Electricity 获益，
ETTm2 却回退。新假设是：**完整 A1 作为冻结锚点，专家只提供有置信度的相对修正；没有专家
显著优于 A1 时拒绝修正。** 若这种嵌套结构仍不能稳定超过原始包络，则不能再把问题归因于
“缺少 fallback”，应拒绝当前三专家集成主线。

候选输出固定为：

\[
\hat y=\hat y_{A1}+a(x,q)\sum_{e\in\{phase,trajectory,cycle\}}
\pi_e(x,q)\,\operatorname{clip}(\hat y_e-\hat y_{A1}).
\]

`a=0` 时逐元素严格等于加载的 A1 checkpoint。A1 的 phase、NLinear、RCRF 和全部校准参数
在候选训练中冻结；只有 cycle expert 与 safe router 可训练。路由只读取 encoder history。

## 2. 候选与原始参照

原始包络逐 setting、逐指标取以下三者的最小值：

| ID | preset | 角色 |
|---|---|---|
| A1 | `gold_combo_reliability_s2` | 当前最强 RCRF+NLinear；也是候选冻结锚点 |
| I0 | `rcrf_icpt_none` | RCRF+ICPT 原始周期模型 |
| R0 | `triaxis_rolling_features` | 上一轮最好的强制三专家集成 |

四个新候选共享同一个 A1 checkpoint、cycle expert、router 容量和 correction clip：

| ID | preset | 唯一新增机制 |
|---|---|---|
| S0 | `safe_triaxis_anchor` | 精确 A1 fallback；只用预测损失，检验嵌套结构本身 |
| S1 | `safe_triaxis_regret` | S0 + 周期级相对 A1 regret 路由监督 + cycle 辅助损失 |
| S2 | `safe_triaxis_guarded` | S1 + 平均正 regret 与最差 10% regret 的单边惩罚 |
| S3 | `safe_triaxis_monotone` | S2 + phase 随距离增强、cycle 随距离衰减的单调 horizon prior |

固定参数：P=24、rolling origins=4、router hidden=16、relative-gain margin=2%、temperature=0.1、
route aux=0.1、cycle aux=0.1、mean non-regret=0.05、CVaR=0.01、correction clip=2 个历史标准差。
不搜索阈值、损失权重或容量。S3 的 horizon prior 从近零单调强度开始，不使用 dataset ID。

## 3. Stage 0：实现与嵌套性测试

- 从 A1 checkpoint 加载时，只允许缺少 `safe_triaxis_*` 参数，不允许旧参数 missing/unexpected。
- `global_accept=0` 时，S0/S1/S2/S3 与该 A1 checkpoint 的 validation forward 逐元素一致。
- freeze 后只有 `safe_triaxis_cycle_expert` 和 `safe_triaxis_router` 可训练；A1 参数和 buffer 不变。
- 检查 H96/H192 shape、finite、权重和为 1、修正 clip、无 future value/mark 泄漏、拒绝动作、
  relative-regret target、CVaR 梯度、S3 horizon 单调方向和 flag-off state-dict 隔离。
- 完整仓库单元测试和 ETTm2 5%/1 epoch GPU smoke 必须通过后才进入 Stage A。

## 4. Stage A：六数据集 H96 机制筛选

- 数据：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity。
- 设置：L720→H96、P24、30% train、seed 2021、最多 8 epoch、Huber、最低 validation loss。
- 运行 A1/I0/R0/S0/S1/S2/S3，共 `6×7=42` 个 run。每个 S* 必须加载同 setting 的 A1
  best checkpoint，冻结 A1 后再训练。
- 候选按相对原始包络的 12 个 MSE/MAE 比值宏平均排序；统一选择一个 S*，不得按数据集切换。
- 严格通过条件：12 个比值全部 `<1.0`，且宏平均 `≤0.995`。若无人通过，仍选择宏平均最小者
  进入 Stage B 作跨 horizon 诊断，但标记为“未冻结诊断候选”。

## 5. Stage B：六数据集 H192 外推

- 使用 Stage A 统一候选，不再调整任何结构或权重；运行同六数据集的 H192。
- 每个 setting 运行 A1/I0/R0/统一候选，共 `6×4=24` 个 run；协议仍为 30%、8 epoch、seed
  2021、validation-only，候选加载同 setting A1 checkpoint。
- 最终 validation gate 同时覆盖 H96/H192：候选相对原始包络的 24 个 MSE/MAE 比值必须全部
  `<1.0`，宏平均 `≤0.995`。任何一个单元回退即失败，不允许平均收益掩盖。

## 6. Stage C：仅最终 gate 通过后

若且仅若 Stage B 最终 gate 通过，才在相同 12 个 setting 上进行 full train、seeds
2021/2022/2023 的 A1/I0/R0/候选正式 test，并与固定 Golden 同设置结果比较。若 gate 失败，
本阶段不适用且不得读取 test。Exchange 的周期尺度不同、Traffic 成本显著更高，只有 Stage C
通过后才作为额外外部数据扩展，不能用于挽救失败候选。

## 7. 诊断与错误分析

无论 gate 是否通过，都对 Stage A 统一候选在全部 12 个 H96/H192 validation setting 做：

- sample×channel 的 A1/候选 MSE、MAE、delta 和时间范围；另记录该 setting 最强原始参照。
- 每个 setting 程序化筛选 baseline high error、candidate regression、candidate improvement，
  跨类别对相同 channel、起点相距小于 horizon 的窗口去重；最终案例总数不超过 10。
- 报告 acceptance coverage、拒绝率、被接受样本的条件收益、正 regret 比例、最差 10% regret、
  三专家条件权重和四个等长 horizon 段误差。
- 区分事实与假设：门控值、误差和覆盖率是观察；“代理错配”“专家不足”只能作为待验证归因。

实验编号为 `safe_regret_triaxis_v1`。canonical 目录严格只包含 `run.yaml`、`results.csv`、
`sample_errors.csv`、`selected_cases.npz`、`objective_error_analysis.md`、
`objective_error_analysis.zip` 和被 Markdown 引用的 `figures/`。大 CSV、checkpoint、日志和图片
均保持本地忽略，不加入 Git。

## 8. 实际结果（2026-08-29）

Stage 0 通过 182 个仓库测试、H96/H192 前向/梯度/冻结检查和 ETTm2 5%/1 epoch GPU smoke；
候选加载 A1 时 `unexpected=0`，初始输出与 A1 最大绝对差为 0。Stage A/B 共完成 66 次
validation-only 训练，无失败 run。

### 8.1 Stage A

| 候选 | H96 宏比值 | 最差比值 | 结论 |
|---|---:|---:|---|
| S2 guarded | 1.013521 | 1.049856 | 最优但失败 |
| S3 monotone | 1.013539 | 1.049946 | 失败 |
| S0 anchor | 1.013775 | 1.056246 | 失败 |
| S1 regret | 1.014921 | 1.051123 | 失败 |

按预注册规则，S2 作为诊断候选原样进入 H192；没有二次调参。

### 8.2 最终 H96+H192 gate

- 相对 A1 的 24 个指标宏平均比值：`0.996263`，即平均改善 `0.37%`。
- 相对 A1/I0/R0 逐 setting、逐指标最优包络：宏平均 `1.010499`，即平均退化 `1.05%`。
- 最差单元：Weather-H192 MSE，退化 `5.32%`。
- 仅 ETTh2-H192、ETTm2-H96 同时改善 MSE/MAE；其余 setting 至少一个指标未超过原始最优。
- 最终 gate 失败，Stage C 不适用；`test_accessed=false`，因此没有新的 Golden 结论。

样本回放覆盖 12 个 setting、`2,208,464` 个 sample×channel。相对各 setting 最强原始模型，
显著改善（相对 MSE ≤-10%）占 `15.36%`，显著退化（≥+10%）占 `22.04%`。门控解决了
“不能退回 A1”的问题，却没有解决“某些 setting 的最强原始模型是 I0/R0”的问题；其训练
目标是相对 A1 regret，而科研门槛是相对三模型包络 regret，两者错位。

完整本地审计见 `research_runs/safe_regret_triaxis_v1/objective_error_analysis.md`。下一步若继续，
应改为 `A1/I0/R0 + no-op` 的多锚点选择，或先把三个完整模型蒸馏为统一强锚点；不应继续只在
A1 周围增加局部修正。
