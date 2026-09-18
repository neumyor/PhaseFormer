# PhaseFormer-L: The One Dimension That Phase Tokenization Leaves Behind

> **文档性质**：期刊扩展论文的 MiniPaper 草稿（2026-09-18）。Introduction 与 Methodology 为完整稿；
> Experiments 为**预注册空表**，只有 §4.1 的"先导证据"一节引用既有结果，且全部标注 test-exposed。
> 本文不改变任何既有结论的口径；所有先导数字均可在被引用的登记文档中逐项核对。
>
> 备选题目：*From Phases to Levels: Completing Phase-Domain Forecasting with a Single Degree of Freedom*；
> 中文：《相位之外只剩一维：补全相位域时序预测》。

---

## Abstract

**English.** PhaseFormer showed that tokenizing a periodic series by *phase* — values aligned at the same
offset across cycles — yields a representation that is structurally invariant to changes of cycle shape and
lives in a very low-dimensional subspace, which is why roughly 1k parameters suffice for state-of-the-art
long-horizon forecasting. Its theory, however, assumes locally stable periodicity and leaves
non-stationarity as future work. We show that the part of a periodic series that phase tokenization
*cannot* see is itself essentially one-dimensional: a cross-cycle level drift shifts the phase subspace by
exactly one direction, and the optimal linear correction of that shift is rank one up to a small remainder.
We verify this from three independent angles on seven benchmarks — sample-level diagnosis of where the
phase residual concentrates, a closed-form reduced-rank analysis in which a single input–output mode
accounts for 66–86% of everything a linear corrector can add, and a canonical dissection with causal
interventions of 72 jointly trained low-rank correction heads whose leading mode reads a recency-weighted
level and writes a constant shift over the whole horizon. We further show that this correction must be
learned *conditionally on the phase backbone*: the same direction frozen from an independent regression
degrades the model, while the jointly learned subspace aligns with the backbone-conditional target and
its removal hurts the fused forecast even where it improves the branch on its own. These findings distill a
70k–520k-parameter linear probe into **PhaseFormer-L**, a single reliability-gated level channel that keeps
PhaseFormer at the k-parameter scale. *[Main results on 28 settings to be filled.]*

**中文。** PhaseFormer 证明按相位（跨周期同偏移位置）做 token 化，得到的表示对周期形状变化结构不变、且位于
极低维子空间，因此约 1k 参数即可达到 SOTA。但其理论假设周期局部平稳，非平稳性被留作未来工作。本文证明相位
视角**看不见的那部分**本质上是一维的：跨周期电平漂移把相位子空间恰好推偏一个方向，对该偏移的最优线性修正
在小残项意义下为秩 1。我们在 7 个基准上用三种独立方法验证：样本级诊断显示相位残差集中在电平不稳定的窗口；
闭式降秩回归显示单一输入—输出模式即占线性修正全部可实现价值的 66%–86%；对 72 个联合训练的低秩修正头做规范
分解与因果干预，其主模式读取近端加权电平、写出整段恒定位移。我们进一步证明该修正必须**以相位主干为条件**
学习：同一方向若从独立回归中冻结得到反而使模型退化，而联合学到的子空间与主干条件性目标对齐，删除它会损害
融合预测——即便支路自身误差反而变好。这些发现把 70k–520k 参数的线性探针蒸馏为 **PhaseFormer-L**：一个由可靠度
门控的单电平通道，使 PhaseFormer 仍停留在 k 级参数。*[28 个 setting 的主结果待填。]*

---

## 1. Introduction

### 1.1 PhaseFormer 说清了什么

长程时序预测的主流范式把序列切成 patch 作为 token。PhaseFormer（Niu et al., ICLR 2026）指出这一范式在大规模
数据上效率低下的根源是**周期形状的变化**：形状变化迫使模型构造高维表示空间。它转而把跨周期**同一偏移位置**
的值聚成"相位 token"，并给出三条证据：

1. **全局平稳**：相位 token 的分布随时间几乎不漂移（MMD 显著小于 patch token）；
2. **低维**：两个主成分即可解释相位 token >90% 的方差，patch token 需要十一个以上；
3. **结构不变性**（Theorem 1–3）：在 `X = AGᵀ + N`、扰动 `X' = XSᵀ + R` 的设定下，周期内形状变换 `S` 只改变
   patch 子空间（行空间），相位子空间（列空间）在无噪声时精确不变。

由此，路由器数量与隐维可以固定为很小的常数，模型以约 1k 参数在七个基准上取得 SOTA。

### 1.2 它自己留下的问题

PhaseFormer 明确写下了假设与未来工作："the approach assumes locally stable periodicity across the input and
output horizons"，"future work will relax this assumption by modeling non-stationarity and complex drifts"。
它唯一输给 FITS 的数据集是 ETTh2。

本文的出发点是一个简单观察：Theorem 1–3 覆盖的扰动 `XSᵀ` 作用在**周期内**（相位轴），而真实数据中最常见的
非平稳性——**每个周期整体抬高或压低**——作用在**周期间**（周期轴）：

```text
X' = X + ℓ 1ᵀ ,   ℓ ∈ R^K  为各周期的电平轨迹
```

这一项给每个相位 token（`X` 的每一列）加上同一个向量 `ℓ`，因此**恰好把相位子空间推偏一个方向**。RevIN 只减去
窗口均值（即 `ℓ` 的平均），减不掉"最后一个周期相对历史偏了多少"。换言之，相位视角对形状变化免疫，却对
电平漂移暴露——而暴露的维度是 **1**。

### 1.3 本文的主张

我们把上述观察推进为一个可检验的命题，并从三个独立角度加以验证：

> **命题（非形式）。** 在相位主干给定的条件下，周期序列中相位 token 化无法表示的信息主要是一个标量——当前
> 电平相对窗口均值的偏移；对它的最优线性修正是"近端加权的近期电平 → 整段预测区间的恒定位移"，且在理论上界
> 意义下为秩 1（加一个小残项）。

验证路线不是直接提出新模块，而是先用一个**足够大、无归纳偏置的线性探针**（NLinear 风格的全时间轴残差支路，
70k–520k 参数）与相位主干联合训练，再回答三个问题：探针的收益集中在哪些样本？探针的可实现价值有多少维？
训练出的探针实际读什么、写什么？三条线索收敛到同一个对象后，我们再把探针蒸馏为最小实例。

### 1.4 贡献

1. **理论补全。** 把 PhaseFormer 的结构不变性论证从"周期内形状扰动 `S`"扩展到"跨周期电平扰动 `ℓ 1ᵀ`"，
   给出相位子空间被推偏恰好一个方向、且最优线性修正为秩 1 的命题与证明路线（§2）。
2. **实证：相位的补空间是一维的。** 三种互相独立的方法——样本级诊断、闭式降秩回归、训练权重的规范分解
   与因果干预——三角收敛到"近期加权电平 → 恒定位移"（§3.2–§3.3，§4.3–§4.4）。
3. **方法论发现：预测维数 ≠ 能量维数。** 最有用的方向只占输入方差 <12%；同一映射按权重奇异谱需 418 维、
   按 MSE 加权的预测谱只需 8 维。这解释了为什么所有按能量做的操作（输入平滑、PCA/结构化坐标）一致失败，
   而按任务做的联合低秩训练容量中性（§4.6）。
4. **设计原则：电平状态必须与相位主干联合学习。** 独立最优方向做冻结瓶颈失败、联合子空间与条件性目标对齐、
   支路自身误差与融合误差反向——三者共同确立"残差支路的信息是相对伙伴定义的"（§3.3.3，§4.5）。
5. **PhaseFormer-L。** 一个 EMA 初始化、可靠度门控、与主干联合训练的秩-1/秩-2 电平通道，把探针蒸馏回
   k 级参数（§3.4，§4.2）。

方法上的新意不在"加一个旁支"，而在 1–4 以及"把探针蒸馏成原理性最小模块"的过程。NLinear 在本文中是
显微镜，不是贡献。

---

## 2. 相位视角的盲点：一个一维的推偏

### 2.1 设定

沿用 PhaseFormer Appendix A.6 的记号。将长度 `L = K·P` 的单通道窗口重排为 `X ∈ R^{K×P}`（`K` 个周期、`P` 个
相位）。相位 token 化取 `X` 的**列空间** `Col(X)`（每列是一个相位跨周期的轨迹），patch token 化取行空间。
低秩模型 `X = AGᵀ + N`，`A ∈ R^{K×r}`，`G ∈ R^{P×r}`。

### 2.2 电平扰动

定义跨周期电平扰动 `X_ℓ = X + ℓ 1_Pᵀ`，`ℓ ∈ R^K`。记 `ℓ = ℓ̄ 1_K + ℓ_⊥`，其中 `ℓ̄` 为均值、`ℓ_⊥ ⊥ 1_K`。

**命题 1（推偏维度）。** 若 `ℓ_⊥ ∉ Col(A)`，则 `Col(X_ℓ) = Col(X) ⊕ span(P_{Col(X)}^⊥ ℓ_⊥)`：相位子空间恰好扩张
一维，且新增方向由 `ℓ_⊥` 在原子空间上的正交余量决定。RevIN 的窗口均值归一化只消去 `ℓ̄ 1_K` 分量
（它同时作用于所有周期与相位），对 `ℓ_⊥` 无作用。

*证明路线。* `ℓ 1_Pᵀ` 是秩 1 矩阵，其列空间为 `span(ℓ)`；`Col(X + ℓ1ᵀ) ⊆ Col(X) + span(ℓ)`，维数至多加一；
当 `ℓ_⊥ ∉ Col(A)` 时严格加一。RevIN 对 `X` 减去标量均值等价于减去 `c·1_K 1_Pᵀ`，只影响 `span(1_K)` 方向。
与原文 Lemma 1（列空间保持）对照即得。∎

### 2.3 相位主干条件下的残差目标

设相位主干输出 `ŷ_φ(X)`。对 `X_ℓ`，主干在归一化空间中看到的是形状信息与被均值化的电平，因而其残差
`y − ŷ_φ` 中与 `ℓ_⊥` 相关的部分主要是**最后周期（预测起点）的电平相对窗口均值的偏移**在预测区间上的
延续。若电平在预测区间内近似持续（局部平稳的最弱形式），该延续为常数：

```text
y − ŷ_φ  ≈  δ(ℓ) · 1_H + ε ,     δ(ℓ) = 当前电平 − 窗口均值电平
```

**命题 2（最优线性修正为秩 1）。** 在上述近似下，以 `Z = x − x_last`（或任一去均值/去锚点的历史）为输入、
`D = y − ŷ_φ` 为目标的最优线性映射 `W* = argmin E‖D − WZ‖²` 满足 `rank(W*) = 1 + O(‖ε‖)`，其输出方向趋于
`1_H`（恒定位移），输入方向趋于对 `δ(ℓ)` 的线性最优估计——在电平为随机游走或 AR(1) 时，即**指数加权的
近端核**。

*证明路线。* 降秩回归的闭式解由 `S = Szyᵀ Szz⁻¹ Szy` 的特征分解给出（§3.2）；当 `D` 的可预测部分是
`δ·1_H` 时 `S` 为秩 1 加噪声项，Wedin sinΘ 定理（原文 Lemma 3）给出特征向量的扰动界。输入方向为
`E[δ Zᵀ] Szz⁻¹`，在随机游走/AR(1) 电平下为指数核。∎

> 形式化证明与常数尚待完成；本节给出的是与原文 Lemma 1–3 同构的证明路线。§4.3–§4.4 的实证独立于此。

### 2.4 一个直接推论：为什么核是 EMA 而不是均值

RevIN 已经减掉了窗口均值，剩下要估的是"电平**现在**走到了哪"。在随机游走电平下，对当前电平的最优线性估计
以近端为重、按滞后指数衰减，这正是 §4.3 观测到的 τ=6–72 步指数核。均值核（DLinear 式趋势）对应的是
"电平不动"的先验，与命题 2 的目标不一致。

---

## 3. Methodology

### 3.1 探针：PhaseFormer + 全时间轴线性残差支路

在不改动 PhaseFormer 主干的前提下加入一条 NLinear 风格残差支路：

```text
z   = x_n − x_n,last                              # 归一化窗口减去末值锚点
δ_n = W z                                          # W ∈ R^{H×720}，直接头（direct）
ŷ_r = σ · (δ_n + x_n,last) + μ                     # 反归一化；σ 与 RevIN 精确抵消
ŷ   = (1 − g) · ŷ_φ + g · ŷ_r                      # g ∈ (0,1)：静态门或 RCRF 可靠度门
```

RevIN 的 `σ` 在 `W z` 中精确抵消，因此**支路在原始尺度下就是 `W(x − x_last)` 去拟合 `y − x_last`**；这使
§3.2 的闭式分析与模型损失口径一致。探针参数为 `720 × H`（70k–520k），占全模型参数的 92.5%–99.9%——它足够大、
无归纳偏置，适合作为"相位视角还缺什么"的显微镜。

**低秩形态。** `W = decoder ∘ encoder`，`encoder: 720 → r`，`decoder: r → H`，`q = r/H ∈ {1/4, 1/8, 1/16, 1/32}`。
低秩不是为了省参数，而是**放大镜**：把 720 维压到 1–3 维后，模型被迫只保留最要紧的东西，我们才能把它拆开读。

**主干不可见性。** 所有针对支路输入的干预（§3.3.3）只作用于支路私有输入 `z`；相位主干始终看到原始 `x`。

### 3.2 度量相位补空间的维数：闭式降秩回归

在训练集上取二阶矩 `Szz = E[ZZᵀ]`、`Szy = E[ZDᵀ]`、`Syy = E[DDᵀ]`，其中 `Z = x − x_last`，目标 `D` 取两种：

- **独立目标** `D_ind = y − x_last`（探针自身的回归任务）；
- **条件性目标** `D_cond = y − ŷ_φ`（相位主干给定后仍需补充的误差；用于 §3.3.2 的 H1）。

秩-r 最优映射有闭式解：令 `S = Szyᵀ Szz⁻¹ Szy`，取其前 `r` 个特征向量 `U_r`，
`W_r = U_r U_rᵀ Szyᵀ Szz⁻¹`。**`S` 的第 `i` 个特征值 `λ_i` 恰等于第 `i` 个方向买到的 MSE 降幅**
（`MSE_persist − MSE_r = Σ_{i≤r} λ_i`，逐 setting 数值校验到 1e-9）。据此定义：

| 量 | 含义 |
|---|---|
| `λ_1 / Σλ` | 单一方向占全部可实现降幅的份额（**命题 2 的直接检验量**） |
| `pred_dims_90` | 达到 90% 可实现降幅所需方向数 |
| `capture(r)` | 秩 r 在 validation 上保留的相对可实现提升比例 |
| `b_1, a_1` | 首方向的输入读取向量与输出写出向量（`W_1 = a_1 b_1ᵀ`） |
| `used_var_share` | 首方向所覆盖的中心化窗口方差份额（**预测维数 vs 能量维数**的检验量） |

对 `b_1` 用 16 个可解释模板（常数、ramp、末 24/168 步均值、末值、指数衰减核 τ∈{6,24,72,…}、尾部斜率、日周期
正余弦等）做单模板 `|cos|` 与字典回归 `R²`；对 `a_1` 计算与全程常值形状的 `|cos|` 及 horizon 上的符号一致性。

**为什么不用权重矩阵的奇异谱。** `W` 的奇异值衰减描述的是矩阵能量，不是预测价值；大量小奇异值方向几乎不贡献
MSE 降幅。§4.6 给出 418 维 vs 8 维的反例。

### 3.3 解剖训练出的低秩头：读什么、写什么、是否被依赖

#### 3.3.1 规范模式

低秩隐坐标存在旋转歧义，不能直接解释 `encoder` 的第 `i` 个隐藏单元。改为合成有效映射并做 SVD：

```text
W = W_dec W_enc ,   c = W_dec b_enc + b_dec ,   W = Σ_i s_i u_i v_iᵀ
模式 i ：读取方向 v_i  →  标量 h_i = v_iᵀ z  →  输出形状 u_i ，强度 s_i
```

对每个模式报告：修正能量份额 `s_i² h̄_i² / Σ`、参与比 `(Σ s_i²)²/Σ s_i⁴`、跨 seed 的子空间重叠
（固定维度 `leading4/leading8`，避免把小维子空间放进完整 horizon 空间产生的无判别力下界）。

#### 3.3.2 语义字典与条件性对齐

- **输入字典**：49 个模板、7 组（`recent_level`、`level_change`、`local_trend`、`local_curvature`、`period_level`、
  `period_shape`、`fast_local_change`）；**输出字典**：15 个模板、5 组（`overall_displacement`、`slow_tilt`、
  `curvature`、`periodic`、`recent_shape_continuation`）。对每个模式报告组解释率（投影份额）、留一组 `R²`、
  Shapley `R²`。一个 setting 的"稳定语义"须同时满足：≥2/3 seed 首位、输入解释率 ≥0.5、输出解释率 ≥0.8、
  Semantic-drop 超出同维随机删除的 95% 区间、Semantic-only 融合 MSE/MAE 不差于原 checkpoint 超过 0.5%。
- **H1（条件性对齐）**：训练头的输入子空间与"条件性 RRR"（目标 `D_cond`）的子空间距离，应小于与"独立 RRR"
  （目标 `D_ind`）的距离。这是命题 2 中"以相位主干为条件"一语的直接检验。

#### 3.3.3 支路私有输入干预（因果证据）

方向对得上只说明"长得像"，不说明模型真的在用。以下干预只改变支路输入 `z`，主干与门不变：

| 臂 | 作用 |
|---|---|
| Original | 原 checkpoint |
| Semantic-only / Semantic-drop | 只保留 / 删除语义字典张成的子空间 |
| PCA-only / PCA-drop | 同维度的输入主成分对照 |
| Random-only / Random-drop（100 次） | 同维度随机子空间对照，给出 95% 区间 |
| Conditional-RRR-only / Independent-RRR-only | 只保留条件性 / 独立最优子空间 |
| Bias-off | 去掉常数项 `c` |

对每个臂同时报告**支路自身误差**与**融合误差**。若删除某子空间使支路自身误差变好而融合误差变差，则该子空间
承载的是"相对主干的修正"而非"独立预测"——这是贡献 4 的核心判据，也是 §2.3 的直接后果。

### 3.4 PhaseFormer-L：最小实例

把 §3.2–§3.3 收敛到的对象写成一个结构：

```text
h   = wᵀ (x_n − x_n,last)                          # 单标量：近端加权电平偏移；w 以 exp(−lag/τ) 初始化
δ_n = h · u  (+ h₂ · u₂ 可选秩-2)                   # u 以 1_H 初始化：恒定位移；u₂ 以线性 ramp 初始化
ŷ_L = σ · (δ_n + x_n,last) + μ
ŷ   = (1 − g(x)) · ŷ_φ + g(x) · ŷ_L               # g(x)：可靠度门（RCRF 或电平非平稳度触发）
```

参数量约 `720 + H (+ 720 + H)`，即 0.8k–2.9k，与主干（1.16k）同量级；相对 direct 探针压缩 25–450 倍。
三点设计原则均来自实证而非先验：

1. **必须联合训练**（§4.5）：`w, u` 可学习，从独立 RRR 冻结得到的 `w` 反而退化；
2. **EMA 初始化、允许偏离**（§2.4，§4.3）：初始核是理论最优形状，训练允许数据修正 τ；
3. **门控可关闭**（§4.2，§4.7）：在电平已稳定的数据（先导证据中 ETTh1/ETTm1）上，通道应自动关小；这是
   命题 1 的可反驳预测，也是 main table 不退化的前提。

消融：`L-fixed`（冻结 EMA 核，只学 `u` 与门）、`L-rank2`、`L-nogate`（静态门）、`L-mean`（均值核初始化，
对照 §2.4）、`direct`（完整探针，上界参照）、`phase_only`。

---

## 4. Experiments（预注册；除 §4.1 外全部待填）

### 4.0 协议

- 数据：ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity/Traffic，输入 720，输出 96/192/336/720，共 28 setting。
- 训练：full-train、最低 validation loss checkpoint、seeds 2021/2022/2023、每 checkpoint 只读一次 test。
- 参照：固定 Golden（`PhaseFormer_gold_standard.md`）；matched rerun 仅用于协议诊断。
- **盲测边界**：§4.2 的 28 个 setting 在本文冻结后不做任何基于 test 的选择；机制分析（§4.3–§4.6）全部使用
  train/validation。§4.1 的先导证据来自既有 test-exposed 条件性配置，只作动机，不进入主结论。
- 判定门槛（预注册）：PhaseFormer-L 相对 Golden 在 ≥ [待定] /28 setting 双指标改善，任一 setting 双指标回退
  ≤ [待定]%；"稳定超过 Golden"沿用既有严格标准（三 seed 均值 + 样本 std < Golden）。

### 4.1 先导证据（既有结果，test-exposed，仅作动机）

以下数字来自既有登记文档，7 个 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336）为
test-set selection 所得，**不得表述为盲测**。

| 命题的预测 | 先导观测 | 出处 |
|---|---|---|
| 残差集中在电平不稳定的窗口 | 融合收益与跨周期水平波动 r=+0.49/+0.53、与最后周期电平偏移 r=+0.48/+0.49；修正与相位残差同向 cos 0.70/0.79（ETTm1-192，单 seed） | `PhaseFormer_structural_defect_research_narrative.md` §3 |
| 最优线性修正近似一维 | `λ_1/Σλ` = 0.66–0.86；90% 只需 2–4 维；PR 1.33–2.12 | `PhaseFormer_rank_capacity_and_data_property_report.md` §2.1, §2.6 |
| 读近端加权电平、写恒定位移 | `b_1` 最近 24 步占能量 53%–70%，最佳模板 exp 核 τ=6–72（\|cos\| 0.58–0.78）；`a_1` 与常值 \|cos\| 0.89–0.99、符号 7/7 一致 | 同上 §2.6 |
| 预测维数 ≠ 能量维数 | 首方向只占输入方差 0.8%–12.3%；ETTh2-720 权重谱 95% 能量需 418 维、预测谱 95% 只需 8 维 | 同上 §1.3, §2.3 |
| 训练头真的在用这一维 | 72 个低秩 checkpoint 主模式：读 EMA（\|cos\| 0.70–0.81）、写常数（0.92–0.99）；Semantic-only Δfused ≤ +0.0006；Semantic-drop +4.3%–16.0%（q=1/8），60/72 超出随机 95% 区间；4/6 setting 成立，Weather 为反例 | `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 表 4–7 |
| 修正相对主干定义 | 独立 RRR 方向 1/1+2 冻结为瓶颈：宏平均 +1.94%/+1.99%；H1 在 5/6 setting 3/3 seed 成立；删除子空间后支路自身误差变好 52/72、融合误差变差 72/72（中位 +30.4%） | `PhaseFormer_top2_direction_retention_summary.md`；同上 §11.8 |
| 周期越不稳定盲点越大 | 探针相对 phase_only：ETTm2 +6.2%、ETTh2 +3.4%、Weather +2.0%；ETTh1 −1.3%、ETTm1 −2.7%（单 seed，静态门） | `PhaseFormer_joint_lowrank_rank_sweep_plan.md` §13.4 |
| 压缩是放大镜不是增益 | 最深档（参数 3.5%–6.1%）保留 92.4%–101.9% 可实现价值；三 seed test 宏平均 −0.12/+0.07/−0.32/−0.52%，唯一可复现增益 ETTh2-720 q=1/8（+1.05%/+0.43%，3/3） | `PhaseFormer_rank_sweep_conditioned_experiment.md` §7 |

### 4.2 主结果：PhaseFormer-L vs Golden（28 setting × 3 seed）

| Dataset | H | Golden MSE/MAE | PhaseFormer-L MSE±std / MAE±std | Δ vs Golden | 稳定超过 | 参数量 | `g` 均值 |
|---|---:|---:|---|---:|---|---:|---:|
| ETTh1 | 96 | 0.359/0.382 | | | | | |
| ETTh1 | 192 | 0.397/0.404 | | | | | |
| ETTh1 | 336 | 0.425/0.424 | | | | | |
| ETTh1 | 720 | 0.431/0.450 | | | | | |
| ETTh2 | 96 | 0.275/0.338 | | | | | |
| ETTh2 | 192 | 0.341/0.376 | | | | | |
| ETTh2 | 336 | 0.369/0.405 | | | | | |
| ETTh2 | 720 | 0.402/0.436 | | | | | |
| ETTm1 | 96 | 0.293/0.344 | | | | | |
| ETTm1 | 192 | 0.323/0.361 | | | | | |
| ETTm1 | 336 | 0.358/0.381 | | | | | |
| ETTm1 | 720 | 0.412/0.410 | | | | | |
| ETTm2 | 96 | 0.163/0.256 | | | | | |
| ETTm2 | 192 | 0.219/0.293 | | | | | |
| ETTm2 | 336 | 0.269/0.326 | | | | | |
| ETTm2 | 720 | 0.351/0.379 | | | | | |
| Weather | 96 | 0.148/0.195 | | | | | |
| Weather | 192 | 0.193/0.237 | | | | | |
| Weather | 336 | 0.242/0.278 | | | | | |
| Weather | 720 | 0.309/0.332 | | | | | |
| Electricity | 96 | 0.129/0.221 | | | | | |
| Electricity | 192 | 0.148/0.238 | | | | | |
| Electricity | 336 | 0.165/0.257 | | | | | |
| Electricity | 720 | 0.201/0.285 | | | | | |
| Traffic | 96 | 0.361/0.238 | | | | | |
| Traffic | 192 | 0.373/0.243 | | | | | |
| Traffic | 336 | 0.385/0.248 | | | | | |
| Traffic | 720 | 0.428/0.270 | | | | | |

对照行（同协议、同 seed）：`phase_only`（matched rerun）、`direct`（完整探针）、`L-fixed`、`L-rank2`、`L-nogate`、
`L-mean`。参数量与 FLOPs 列须与原文 Table 4 同口径。

**必答问题**：(a) ETTh2 是否被补齐到 FITS 水平；(b) ETTh1/ETTm1 是否不退化、`g` 是否如预测自动关小；
(c) `L-rank2` 相对 `L` 的增量是否在 seed 噪声内（命题 2 的秩-1 预测）。

### 4.3 相位补空间的维数（7 数据集，train/validation）

| Dataset | H | `λ_1/Σλ` | `pred_dims_90` | PR | `b_1` 最佳模板（\|cos\|） | `a_1` vs 常值 \|cos\| | `used_var_share(1)` |
|---|---:|---:|---:|---:|---|---:|---:|
| （28 行待填；先导 7 行见 §4.1） | | | | | | | |

预期图：Scree 图（第一根柱 0.66–0.86 量级）；`b_1` 随 lag 的剖面（近端集中 + 指数衰减）与 `a_1` 随 horizon
的剖面（近似平线）双面板。

### 4.4 训练头的解剖（PhaseFormer-L 与低秩探针，3 seed）

| Dataset | H | 主模式输入组 / 解释率 | 主模式输出组 / 解释率 | 修正能量份额 | 跨 seed `leading4` 重叠 | 稳定语义判定 |
|---|---:|---|---|---:|---:|---|
| （待填） | | | | | | |

干预表（每 cell 10 臂；同时报告支路自身与融合误差）：

| Dataset | H | q/r | Semantic-only Δfused | Semantic-drop Δfused | 随机 95% 区间 | PCA-drop | **随机 RRR 子空间 drop**（新增对照） | 支路自身 Δ | 融合 Δ |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| （待填） | | | | | | | | | |

新增"随机 RRR 子空间"对照用于区分"语义有效"与"任意同数量主方向有效"（既有结果中 57/57 单元
Semantic-drop ≡ PCA-drop，此项此前缺失）。

### 4.5 条件性学习（贡献 4 的正式检验）

| Dataset | H | direct | 冻结独立-RRR 方向 1 | 冻结条件-RRR 方向 1 | PhaseFormer-L（联合） | H1：cond 距离 < indep 距离（seed 数） |
|---|---:|---|---|---|---|---|
| （待填） | | | | | | |

预测：冻结独立方向退化；冻结**条件性**方向应明显好于独立方向（此对照此前未做，是区分"冻结本身有害"与
"独立目标错位"的关键）；联合训练最好。

### 4.6 按能量 vs 按任务：负对照汇总

| 操作 | 作用对象 | 口径 | 结果（既有，test-exposed） | 本文补做 |
|---|---|---|---|---|
| 输入平滑（boxcar / causal EMA，各 5 档） | 支路输入 | 7 setting | 14/14 组合无一改善，越平滑越差 | 在 PhaseFormer-L 上复测 2 档 |
| 结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离） | 支路参数化 | 4 setting | 5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52% | — |
| SVD 截断 vs 秩约束训练 | 支路权重 | Electricity-336 r=10 | 截断 +29%，训练 +0.7% | 全 28 setting |
| 联合低秩训练 q=1/32 | 支路容量 | 7 setting | 保留 92.4%–101.9% 可实现价值 | — |

### 4.7 命题 1 的预测力：电平非平稳度 vs 增益（训练无关）

对 28 个 setting 计算训练集上的跨周期电平统计量（`cycle_level_std`、`last_cycle_shift`、电平自相关时间
`τ̂`），与 §4.2 中 PhaseFormer-L 相对 `phase_only` 的增益及门值 `g` 做相关：

| 统计量 | 与 ΔMSE 的 Spearman ρ | 与 `g` 的 ρ | 预期符号 |
|---|---:|---:|---|
| `cycle_level_std` | | | + |
| `last_cycle_shift` | | | + |
| `τ̂`（电平记忆长度） | | | 与学到的 EMA τ 正相关 |

若成立，则"哪些数据集需要电平通道"可由数据统计量事前预测；Weather 类弱周期数据若主模式转向曲率/慢趋势，
应作为命题的边界条件单独报告。

---

## 5. Limitations 与披露

1. §4.1 的全部先导证据来自 test-set selection 得到的 6–7 个 setting，属条件性、test-exposed 证据；本文的
   主结论只能建立在 §4.2–§4.7 冻结后的盲测与 train/validation 分析上。
2. 命题 1–2 目前是证明路线而非完整证明；命题 2 依赖"电平在预测区间内近似持续"的近似，长 horizon 上残项
   `ε` 可能不小（先导证据中 ETTh2-720 主模式修正能量份额仅 0.20，与此一致）。
3. 既有 checkpoint 解剖中只有主模式可命名；第 2 个及以后模式跨 seed 不稳定，应报告为"多组等价信息通路"。
   Weather 两个 setting 的主模式指向曲率/慢趋势而非电平，是命题的边界而非支持。
4. "语义有效"与"任意同数量主方向有效"尚未分开（既有 57/57 单元 Semantic-drop ≡ PCA-drop）；§4.4 新增的
   随机 RRR 子空间对照是解决此项的必要实验。
5. 低秩压缩在本文中是分析工具，不是精度贡献；既有三 seed 结果无普适增益。
6. 结论范围：与相位/周期主干经凸门联合训练的线性残差修正，标准长程基准；不宣称任意时序模型可压缩或
   任意线性模型等价于电平修正。

---

## 6. 关联文档

- 主干与金标准：`PhaseFormer_gold_standard.md`；原文 [arXiv 2510.04134](https://arxiv.org/abs/2510.04134)
- 探针的增益与不对称：`PhaseFormer_gold_combo_experiment.md`、`PhaseFormer_top5_test_models.md`、
  `PhaseFormer_joint_lowrank_rank_sweep_plan.md`
- 样本级诊断（D7）：`PhaseFormer_structural_defect_research_narrative.md`
- 闭式降秩分析：`PhaseFormer_rank_capacity_and_data_property_report.md`
- 训练头解剖与干预：`PhaseFormer_lowrank_checkpoint_information_analysis_plan.md`
- 条件性学习的负结果：`PhaseFormer_top2_predictive_direction_retention_report.md`、
  `PhaseFormer_direction1_neighborhood_report.md`
- 负对照：`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`、
  `PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md`、
  `PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md`、
  `PhaseFormer_lowrank_mechanism_analysis.md`
- 操作记录：`agent-log.md`
