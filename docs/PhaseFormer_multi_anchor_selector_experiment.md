# Multi-Anchor Selector：完整模型多锚点选择实验

> 状态：预注册（2026-08-29）。本节在产生新结果前固定机制、候选、训练时间切分、筛选门槛
> 和停止条件。参数选择只使用 validation；最终 gate 通过前不读取 test。

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
