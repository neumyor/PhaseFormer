# PhaseFormer-L M1：中心化乘性调制 的设计思想与理念

记录日期：2026-10-10。本文是**设计意图与证据边界**记录，不新增训练或测试，不授权任何新实验。

## 0. 决策与定位

用户于 2026-10-10 指定 **M1（T→P 中心化乘性调制，代码名 `tp_mod_centered`）** 作为当前首要的模型结构，后续研究以 M1 为主线展开。

这是**研究方向上的选择**，不是性能结论。六任务 test 汇总（见 [交互结果记录](PhaseFormer_interaction_results_record_20261010.md) 第6节）中，M1 是交互实验里 MSE 最低的一行，但相对 B0 的优势在噪声量级：

| | MSE | MAE | ΔMSE vs Golden | ΔMAE vs Golden | ΔMSE vs B0 | ΔMAE vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.2933 | 0.3332 | -0.060% | +0.264% | — | — |
| M1 | 0.2930 | 0.3333 | -0.155% | +0.281% | -0.095% | +0.017% |

此前的计划文档曾基于 H720 validation 先导**拒绝**把 M1 升级为默认架构（见 [实验计划](PhaseFormer_trend_token_interaction_experiment_plan.md) 的“Centered-Mod 正式结论与关闭”）。本文不推翻该结论，只说明：把 M1 作为主线，是因为它的**设计动机最清晰、可归因、可证明地约束了副作用**，值得继续追问；其收益是否真实，仍需第7节列出的检验。

### 0.1 规格更新（2026-10-10）

用户决定：为严守 ICLR 版“低参数、高效率”的 claim，期刊稿中的 M1 一律使用**压缩到很小的低秩水平分支**。分支读周期均值、秩为3，具体规格由 [VLDBJ 大纲](PhaseFormer_VLDBJ_extension_outline.md) 第4.1节的 validation 先导选定。下文第2、4节描述并报告的是 rank=H/8 版本，它们是压缩版的先导证据，不是期刊稿主表配置。

## 1. 一句话设计思想

> **水平（周期均值）与波形（周期内形状）分工；trend 信息只被允许去调节波形，且保证不会改动 phase 直接写出的周期水平。**

M1 = B0 + 一个零初始化的、对 phase 表征做“中心化乘性调制”的小模块。

## 2. 结构

### 2.1 与 B0 共同的骨架（M1 不改动的部分）

- lookback 720，RevIN 归一化；融合在归一化坐标内完成，最后还原尺度。
- **phase 分支**：按 P=24 折叠成 phase token，经 cross-phase routing 层得到 `Z ∈ R^{B×C×P×d}`（P 个相位槽，每槽 d 维），再经共享线性 predictor 解码成未来各周期。
- **trend / 低秩分支**：原 rank-r（r=H/8）低秩编码器 E 把历史压成 `t ∈ R^{B×C×r}`，原解码器 D 与 last-value anchor 给出 trend 预测。
- **静态双分量输出融合**：每变量两个静态权重（水平 `g_level`、波形 `g_shape`，初始0.2，`shape_bias_delta` 零初始化），对每个 P24 周期的**均值**与**去均值波形**分别混合 phase 与 trend 预测。训练后固定，不随样本变化（代码：[lowrank_fusion.py](../src/models/lowrank_fusion.py)）。

### 2.2 M1 新增的唯一模块

令 `z̄ = mean_p Z_p`，`S_p = Z_p − z̄`（各相位槽相对周期均值的偏离），

```text
q   = B · A · t                      A: r→k 线性，B: k→d 线性，k=min(2,r,d)，均无 bias
Z'_p = Z_p + S_p ⊙ q                 ⊙ 为逐特征乘；q 对同一窗口内所有相位槽共享
```

- 插入位置：phase 路由之后、最终 phase predictor 之前；trend 分支本身（t 与其解码）**不被修改**，这是单向 T→P 写入。
- B 零初始化 ⇒ 初始时 `q=0`、`Z'=Z`，M1 与 B0 **逐位等价**；模型只在对训练有利时才偏离 B0。
- 额外参数 `k(r+d)`：H720 为 244/196/196（ETTh1/ETTm2/Weather），占基线 0.13%–0.15%；无新增 P×P 或 r×d 稠密矩阵，不改 r。
- 实现：[model.py](../scripts/centered_mod_interaction/model.py) 中 `Communication(mode='tp_mod_centered')`。

## 3. 设计理念（为什么这样设计）

### 3.1 水平与波形分责

预测误差可按每个未来周期分解为 水平 MSE（周期均值误差）+ 波形 MSE（去均值后的形状误差）。B0 已经用 **静态双分量融合** 把两者的 trend/phase 配比分开；低秩分支加 last-value anchor 本来就擅长水平。M1 沿用这一分工：水平继续由原有路径与融合负责，trend→phase 的新通道**只给波形用**。

### 3.2 为什么不写水平：吸取 TP-Add / L3 的教训

TP-Add（`Z' = Z + 广播消息`）、L3 等对所有相位槽加同一个消息，在最终线性 predictor 下等价于直接移动预测周期均值。这与 trend 分支、anchor 已经负责的水平通道**重叠**，既无新信息，又容易在联合训练中和已有水平预测互相抵消或造成过修正。已有激活审计也表明 TP-Add 的直接消息纯属周期水平层面（最大波形能量占比 1.05e-12）。M1 把这条冗余通道**在结构上堵死**。

### 3.3 为什么用乘性调制：让 trend “调幅”而非“平移”

`S_p ⊙ q` 的含义是：trend 状态决定当前窗口里周期形状的**每个潜在特征被放大或衰减多少**，而不是改变形状的位置。直觉上，趋势/水平状态（近期漂移、幅度 regime）会影响周期形态的显著程度，这类“幅度调节”无法由静态线性头吸收（相比简单 phase 线性 adapter，q 依赖 t，含输入条件化乘积）。

### 3.4 为什么要中心化：可证明的约束，而不是经验正则

若直接 `Z' = Z + Z ⊙ q`（未中心化 TP-Mod），调制项含 `z̄ ⊙ q` 这一周期均值成分，会连带改动水平。中心化后 `Σ_p S_p = 0`，在共享线性 predictor 与完整周期重排下

```text
Σ_p S_p = 0  ⇒  Σ_p (S_p ⊙ q) = 0  ⇒  Σ_p W(S_p ⊙ q) = 0
```

即**直接写入对每个未来周期的均值贡献恒为0**。这是代数事实，不是统计倾向。六个 H720 setting 的最高精度核验最大差值 5.96e-8–1.44e-7，通过按输出幅度×32 个 float32 eps 的数值容差。

适用条件与边界（必须同时记住）：

- 要求 predictor 为共享线性（`use_mlp=False`，代码里是硬检查）、预测按完整周期重排、静态逐变量分量融合。
- 只保证**直接写入**不改变水平；**独立联合训练**会通过改变 phase 主干间接改变水平预测，不能说整个训练结果水平不变。
- 推理时精确；训练时 dropout 可能破坏逐次前向的精确恒等式。
- 中心化不提供收缩或稳定性保证，大 q 仍可放大异常。

### 3.5 保守、可归因的最小改动

- 零初始化 + 严格退化为 B0，使 M1 vs B0 成为**单因素对照**。
- 不引入 LayerNorm、softmax、动态输出 Gate、新 loss、更大 k 或额外 phase 内容输入，避免多因素混淆。
- 低秩性只约束 trend→调制向量的映射 `BA`，**不能**宣称 slot×feature 或未来周期×phase 的最终波形更新秩 ≤ k。

### 3.6 不声称的东西

- 不声称全新调制方法：它是对既有 TP-Mod 的“中心化”结构约束消融。
- 不声称完全无门控含义：含乘法调制，具有广义门控的性质。
- 不声称 trend 与 phase 的“信息融合”已被证明有益（见第4节）。

## 4. 现有证据（正反两面）

**支持动机的**

- 波形误差确有可消除空间：B0 的 H720 validation 中，波形 MSE 占 MSE 的 16%–26%（ETTh1 ≈19%、ETTm2 ≈16%、Weather ≈26%）；这是以 label 作离线诊断的**理想上限**，不是可实现收益。
- ETTm2 上直接波形写入本身有用。
- 六任务 test：M1 MSE 0.2930，交互实验行中最低，相对 B0 -0.095%。

**不支持或尚未支持的**

- 相对 B0 仅 4/12 个 seed-setting MSE 胜出，MAE 反高 0.017%，不构成稳定双指标提升。
- H720 validation 先导（2 seed × 3 dataset）：相对 B0 的 setting 等权平均相对 MSE/MAE +0.052%/+0.157%，2/6 胜出；未达预登记门槛（对 B0 与原 TP-Mod 均 MSE ≤ -1%、≥4/6 胜出、MAE 不升）。
- 联合训练后水平误差可抵消波形增益（ETTm2），ETTh1 上直接波形修正本身变差。
- 效率：ETTm2 batch1 延迟 +6.2%，超过 5% 门槛；独立 epoch 成本未证实。
- **同对照下，phase-only 的 C2（无 t 输入）MSE 0.2931，与 M1 几乎相同**，说明 M1 的微小 MSE 优势目前无法归因于 trend→phase 的跨分支信息。这是对 M1 核心理念最直接的挑战。

**逐 task test（两 seed 均值，4 位显示，相对 B0，近似）**

| Setting | ΔMSE vs B0 | ΔMAE vs B0 |
|---|---:|---:|
| ETTh1_H96 | ≈ -0.68% | ≈ -0.63% |
| ETTh1_H720 | ≈ +0.07% | ≈ -0.09% |
| ETTm2_H96 | ≈ +0.06% | ≈ +0.20% |
| ETTm2_H720 | ≈ -0.43% | ≈ -0.11% |
| Weather_H96 | ≈ +0.89% | ≈ +0.72% |
| Weather_H720 | ≈ +0.19% | ≈ +0.58% |

收益集中于 ETTh1_H96 与 ETTm2_H720，Weather 上两个 horizon 均退化。此表由文档第7节的四位数值重算，仅作方向参考。

## 5. 与相邻方案的关系

| 方案 | 与 M1 的差别 | 回答的问题 |
|---|---|---|
| B0 | 无消息 | M1 的直接对照参照 |
| TP-Add / L3 | 写周期水平 | 水平通道是否冗余（M1 的反面） |
| 原 TP-Mod / L4 | 乘 `Z` 而非 `Z−z̄` | 中心化是否减轻水平副作用 |
| C1 | 加入 phase 上下文，写周期水平 | 条件化上下文是否有益 |
| C2 | 仅 phase 自身均值，无 t | 排除跨分支信息后还剩多少 |

## 6. 研究路线中的位置与假设

M1 的**核心假设**可表述为：

> H-M1：trend token 携带的、phase 主干未充分利用的状态信息，能通过逐特征“调幅”的方式改善周期内波形预测；这种改善在联合训练中不被水平误差抵消。

它是可证伪的。以下情况之一出现，应下调对该假设的信心：

1. 相同预算下，phase-only 对照（C2 类）持续达到与 M1 相同的收益 ⇒ 增益来自 phase 自身读出而非 trend。
2. trend 与 phase 的时间错配对照（同变量、时间不重叠的 t）同样有效 ⇒ 增益与 t 的内容无关。
3. 波形收益被联合训练引起的水平误差抵消，且在更多 seed / task 上稳定复现。
4. 增大 k 或增加 seed 后，相对 B0 的胜出比例仍不过半。

## 7. 作为主线推进前建议明确的检验（仅记录，未启动）

均需用户另行授权，且必须从头联合训练、共同初始化、匹配预算与 seed：

1. **phase-only 与错配控制**：与 M1 同参数量，区分跨分支贡献与 phase 自身读出（现有 C2 只是前者的一个实例）。
2. **补足 seed 与任务覆盖**：当前仅两 seed，4/12 的胜出比例不具备显著性；Weather 的退化需要定位。
3. **水平抵消的归因**：沿用 level/shape 分解，在 M1 联合训练 checkpoint 上分别报告 level MSE 与 shape MSE 相对 B0 的变化。
4. **效率复测**：在隔离 GPU 上重测延迟，判断 ETTm2 的 +6% 是否为并发混杂。

阈值若设立，应在训练前固定登记，不随结果调节。

## 8. 解释边界

- 本记录中的 test 结果来自已披露的 test 读取；此前 test 已经启发研究方向，validation 也被重复用于 checkpoint 与机制选择。选择 M1 作为主线本身依据了 test 排名，属于 **test-set selection**，M1 的任何后续读数都不能称为盲测或无偏泛化估计。
- 与 Golden 的差值（-0.155% MSE / +0.281% MAE）是单指标读数，不是双指标提升；两个 seed 不足以支持显著性，重叠窗口并不独立。
- 原 dense PhaseFormer-L（P1）的同 task 均值 0.2928/0.3321 仍低于 M1，但 seed 数与训练协议不同，仅为读数对照。
- 理想 oracle 与 label 辅助的上限诊断不证明收益可实现。

## 9. 来源

| 内容 | 位置 |
|---|---|
| 六任务与逐 task test | [交互结果记录](PhaseFormer_interaction_results_record_20261010.md) |
| M1 假设、预登记门槛与 validation 先导结论 | [实验计划](PhaseFormer_trend_token_interaction_experiment_plan.md)（Centered-Mod 各节） |
| 实现 | [centered_mod_interaction/model.py](../scripts/centered_mod_interaction/model.py) |
| 静态双分量融合 | [lowrank_fusion.py](../src/models/lowrank_fusion.py) |
| 先导审计产物 | `research_runs/centered_mod_pilot_20261009/` |
| 完整 test 扩展 | `research_runs/interaction_test_extension_20261009/` |
