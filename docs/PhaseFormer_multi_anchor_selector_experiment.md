# Multi-Anchor Selector：完整模型多锚点选择实验

> 状态：已完成，Stage-A gate 失败（2026-08-29）。本节的机制、候选、训练时间切分、筛选
> 门槛和停止条件在产生结果前固定；参数选择只使用 validation，全程未读取 test。

## 1. 上轮证据与新假设

Safe-Regret S2 相对 A1 平均改善 0.37%，但相对 A1/I0/R0 原始包络平均退化 1.05%。这说明
“精确退回 A1”已实现，却不是正确的安全目标：Weather/Electricity/部分 ETTh 的最强完整模型
分别来自 R0 或 I0。

本轮假设是：**把 A1、I0、R0 三个完整模型都作为锚点，以训练时间轴末端的 out-of-fold
预测学习选择规则；按样本×通道×未来周期选择完整锚点，再对相对 oracle envelope 的 regret
施加单边约束。** 这不是平均三个分支，而是学习“哪个完整建模视角在当前历史与预测形态下更
可靠”。

正式锚点：

| ID | preset | 视角 |
|---|---|---|
| A1 | `gold_combo_reliability_s2` | phase + NLinear/RCRF |
| I0 | `rcrf_icpt_none` | phase + 周期间 ICPT |
| R0 | `triaxis_rolling_features` | phase/trajectory/cycle 三轴融合 |

## 2. 避免 stacking 训练泄漏

每个 dataset×horizon 使用两个时间层次：

1. **影子锚点**：只用训练时间轴前 24% 训练 A1/I0/R0；
2. **路由校准段**：训练时间轴 24%–30% 的窗口。其预测目标没有被影子锚点训练过；
3. **正式验证**：路由结构冻结后，替换为上一轮同协议、用前 30% 训练的正式 A1/I0/R0，
   在原 validation split 评估。

影子锚点与正式锚点结构相同，仅训练截止点不同。路由不读取 dataset ID、future truth 或
decoder future value；只读取 encoder history 和三个锚点已经产生的未来预测描述符。

## 3. 路由结构

对每个样本、通道和未来 24 步周期，路由输入：近期漂移、lag-24 相关、差分波动、相位可靠度，
以及每个锚点预测相对最后值的偏离、周期内粗糙度、锚点间分歧和预测斜率。输出 A1/I0/R0
三个动作的 logits。

硬选择采用 straight-through one-hot：前向严格复制某一个完整锚点的该周期预测，反向使用
softmax 梯度；软选择作为消融。硬候选初始化全部选择 A1，加载审计后初始输出与 A1 逐元素
一致。所有六个影子/正式锚点冻结，只训练小型 router。

## 4. 候选消融

| ID | preset | 作用 |
|---|---|---|
| M0 | `multi_anchor_global_hard` | 只学习全局三锚点先验；验证 OOF 排名能否找回强锚点 |
| M1 | `multi_anchor_structural_hard` | M0 + 样本/通道/周期结构路由，硬选择完整锚点 |
| M2 | `multi_anchor_guarded_hard` | M1 + 相对三锚点 oracle 的均值正 regret 与最差 10% CVaR |
| M3 | `multi_anchor_structural_soft` | M1 的 soft convex 输出；检验硬选择是否必要 |

固定参数：P24、router hidden=24、temperature=0.2、oracle temperature=0.1、route CE=0.1；
M2 mean regret=0.05、CVaR=0.01。影子锚点 24%/8 epoch；router 校准段 24%–30%/最多
20 epoch；seed 2021、Huber、最低 validation loss checkpoint。不搜索阈值、容量或数据集专用
超参数。

## 5. Stage 0 与实验网格

Stage 0：

- 三类正式/影子 checkpoint 必须与 setting 匹配并 strict load；所有锚点参数冻结且训练前后不变。
- 硬路由初始化逐元素等于 A1；权重 one-hot 且和为 1；soft 消融权重为凸组合。
- H96/H192 shape/finite、周期展开、route oracle、regret/CVaR 梯度、无 future truth/mark 泄漏。
- 完整仓库测试和 ETTm2 5% smoke 通过后才训练正式候选。

阶段化成本控制：

- Pilot：ETTh1、ETTm2、Weather，H96，四个候选。若无人满足宏比值 `<1.01` 且最差单元
  `<1.03`，停止并做错误分析。
- Stage A：Pilot 通过后补齐 ETTh2、ETTm1、Electricity H96；按相对 A1/I0/R0 逐指标包络的
  12 个比值选择一个统一机制。严格通过仍要求全部 `<1` 且 macro `≤0.995`。
- Stage B：统一机制不调参外推六数据集 H192。最终 24 个比值必须全部 `<1` 且 macro
  `≤0.995`；否则失败。
- 正式 30% 锚点及其指标复用 `safe_regret_triaxis_v1_scratch` 的同协议结果；影子锚点和
  multi-anchor candidate 全部新跑，不能复用 candidate 结果。

只有最终 gate 通过才允许 full train/多 seed/test/Golden。失败时不读 test。

## 6. 样本级分析与产物

最终统一候选无论成败，都在所有实际到达的 setting 上相对单一最强完整锚点计算
sample×channel MSE/MAE、delta、锚点选择率、oracle 一致率、regret 和历史特征。程序化选取
baseline high error、candidate regression、candidate improvement，跨连续窗口去重，总案例
不超过 10。

实验编号：`multi_anchor_selector_v1`。canonical 本地目录严格为六个审计文件加 `figures/`；
checkpoint、日志、缓存与大 CSV 不提交 Git。

## 7. 执行与结果

Stage 0 的 190 项仓库测试全部通过；ETTm2 4%→5% 端到端 smoke 中，路由器只有 498 个
可训练参数，六个锚点均冻结，硬路由初始相对影子/正式 A1 的最大绝对差均为 0，校准段
244 个窗口的 future-target overlap 为 0。

Pilot 的最佳候选为 M3：相对三锚点逐指标包络，宏比值 `0.990870`、最差比值
`1.005759`，通过宽松晋级线。六数据集 H96 补齐后的统一排名如下：

| 排名 | 候选 | 12 指标宏比值 | 最差比值 | 全部 <1 |
|---:|---|---:|---:|---|
| 1 | M3 structural soft | 0.992072 | 1.005759 | 否 |
| 2 | M0 global hard | 1.000881 | 1.010575 | 否 |
| 3 | M2 guarded hard | 1.003836 | 1.020964 | 否 |
| 4 | M1 structural hard | 1.005283 | 1.021797 | 否 |

M3 的逐数据集结果：

| 数据集 | 包络 MSE | M3 MSE | 变化 | 包络 MAE | M3 MAE | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.693317 | 0.683372 | -1.43% | 0.570566 | 0.573852 | +0.58% |
| ETTh2 | 0.211136 | 0.207061 | -1.93% | 0.318662 | 0.318130 | -0.17% |
| ETTm1 | 0.419504 | 0.419063 | -0.11% | 0.433450 | 0.433507 | +0.01% |
| ETTm2 | 0.119617 | 0.119488 | -0.11% | 0.235593 | 0.235377 | -0.09% |
| Weather | 0.395253 | 0.386628 | -2.18% | 0.286409 | 0.280000 | -2.24% |
| Electricity | 0.110991 | 0.109648 | -1.21% | 0.202801 | 0.201510 | -0.64% |

因此 M3 平均改善 0.79%，并在 10/12 个指标上超过原始包络，但 ETTh1-MAE 和
ETTm1-MAE 未严格改善，Stage-A gate 失败；按预注册停止，没有运行 H192 或 test，也不能据此
宣称超过 Golden。

## 8. 样本级结论

validation 回放覆盖 1,121,992 个 sample×channel。相对每个 setting 的单一联合最强完整锚点，
显著改善（相对 MSE ≤-10%）占 14.24%，显著退化（≥+10%）占 10.19%，其余占 75.57%。
改善组的 lag-24 相关均值为 0.7616，退化组为 0.6861；这只是相关描述，不能证明周期性导致
收益。

实际 soft 平均权重显示：ETT 更依赖 A1（ETTm1 为 0.970，ETTm2 为 0.860），Weather 更均匀
（A1/I0/R0=0.323/0.371/0.306），Electricity 更偏 R0（0.411）。但这些权重与真实周期 oracle
占比差距明显，解释了为什么 hard one-hot 路由整体退化，而 soft 插值能利用预测误差抵消。

完整本地审计位于 `research_runs/multi_anchor_selector_v1/`：严格六文件加 `figures/`；包含
60 个正式训练结果、全量样本误差、逐 setting 程序化选出的 90 个案例和仅含报告/引用图的 ZIP。

## 9. 客观判断

多锚点修复了“只能退回 A1”的目标错位，并首次让统一集成在六数据集 H96 上相对原始包络
取得正的平均收益；但它仍不是稳定强模型。当前证据支持 soft 凸组合，不支持周期级 hard 专家
识别。若继续这一方向，最小合理改动是多折 rolling-origin OOF 与 shadow→full 权重校准，而
不是增加专家或继续强化 argmax。
