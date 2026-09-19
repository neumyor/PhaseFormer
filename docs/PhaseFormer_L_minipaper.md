# PhaseFormer-L: The One Dimension That Phase Tokenization Leaves Behind

> **文档性质**：期刊扩展论文的 MiniPaper 草稿（2026-09-18）。Introduction 与 Methodology 为完整稿；
> Experiments 为**预注册空表**，只有 §4.1 的"先导证据"一节引用既有结果，且全部标注 test-exposed。
> 本文不改变任何既有结论的口径；所有先导数字均可在被引用的登记文档中逐项核对。
>
> **执行入口与回写目标**：`docs/PhaseFormer_L_experiment_plan.md`（2026-09-18 登记）——本文 §4 空表的
> 唯一执行契约与结果回写目标。该文 §4 给出缺口核对结论：**G1–G18 全部确认存在**（既有证据最大只覆盖
> 7 个 setting，且全部为条件性数字），并列出 6 项登记不一致 D1–D6 待修。
>
> **D1/D2 复核结果（2026-09-18）**：用服务器 `research_runs/lowrank_checkpoint_information_v1/intervention_results.csv`
> 的单向 rsync 副本（720 行 = 72 cell × 10 臂）复算：Semantic-drop 下支路自身 MSE 变好 **52/72**、融合 MSE 变差
> **72/72**、相对上升中位 **+30.43%**（范围 +5.72%～+51.65%）——与 §4.1 引用一致；Semantic-only 的 Δfused MSE
> 全 72 格最大 **+0.0027**（q=1/8 格最大 +0.0010），此前"≤ +0.0006"为 3-seed 均值口径且偏小，**已按全格最大值
> 更正**。两项数字尚未写入 `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 的表 6 正文，对外引用前
> 应先补登记（执行计划 D1/D2）。
>
> **2026-09-18 修订（风险收敛）**：§3.4 / §4.0 / §4.2 / §5 按"只基于既有实现、不新增模型代码"重写。
> PhaseFormer-L 现定义为**既有** `weak_residual`（静态门）/ `rcrf_nlinear_plain`（可靠度门）+ `shared` 或
> `pooled_lowrank` 头的组合；**秩 1–2 不再作为模型工作点**，只保留为分析对象与边界消融（理由见 §3.4.2）。
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
its removal hurts the fused forecast even where it improves the branch on its own. These findings turn the
probe into **PhaseFormer-L**: PhaseFormer plus a gated linear level corrector whose rank is set by the
predictive spectrum rather than by parameter budget — compressible to 3.5–6.1% of the dense head with
≥92% of its value provably retained — and whose gain is predicted, from training data alone, to concentrate
on datasets with non-stationary cross-cycle level. *[Main results to be filled.]*

**中文。** PhaseFormer 证明按相位（跨周期同偏移位置）做 token 化，得到的表示对周期形状变化结构不变、且位于
极低维子空间，因此约 1k 参数即可达到 SOTA。但其理论假设周期局部平稳，非平稳性被留作未来工作。本文证明相位
视角**看不见的那部分**本质上是一维的：跨周期电平漂移把相位子空间恰好推偏一个方向，对该偏移的最优线性修正
在小残项意义下为秩 1。我们在 7 个基准上用三种独立方法验证：样本级诊断显示相位残差集中在电平不稳定的窗口；
闭式降秩回归显示单一输入—输出模式即占线性修正全部可实现价值的 66%–86%；对 72 个联合训练的低秩修正头做规范
分解与因果干预，其主模式读取近端加权电平、写出整段恒定位移。我们进一步证明该修正必须**以相位主干为条件**
学习：同一方向若从独立回归中冻结得到反而使模型退化，而联合学到的子空间与主干条件性目标对齐，删除它会损害
融合预测——即便支路自身误差反而变好。这些发现把探针收束为 **PhaseFormer-L**：PhaseFormer 加一条门控线性电平修正器，
其秩由预测谱而非参数预算决定——可压缩到稠密头参数的 3.5%–6.1% 且可证保留 ≥92% 的价值——并且其增益能仅凭训练
数据预测为集中在跨周期电平非平稳的数据集上。*[主结果待填。]*

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
5. **PhaseFormer-L。** PhaseFormer + 门控线性电平修正器：以已完成三 seed 验证的既有实现为工作点，秩由
   预测谱选定并给出容量上界；配一个仅用训练集统计量决定"是否启用"的数据驱动开关，使增益的条件性成为
   可预注册的预测而非事后解释（§3.4，§4.2，§4.7）。

方法上的新意不在"加一个旁支"，而在 1–4：把一条旁支的作用**测出维数、命名、并证明其条件性**。NLinear 在本文中
首先是显微镜；它作为修正器被保留，是因为证据显示它的价值确实是低维的、可压缩的、且可预测在哪里出现。

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

### 3.4 PhaseFormer-L：基于既有实现的定义与工作点选择

> 本节只使用仓库中**已经实现且已有三 seed 测试记录**的部件，不新增模型代码。每个组件对应的
> `mechanism` / 配置项在表中注明，§4 的全部空表都能用既有 runner 直接补齐。

#### 3.4.1 定义

```text
PhaseFormer-L  =  相位主干 ŷ_φ（原始 PhaseFormer，不改）
               +  线性电平修正器 ŷ_r = σ·(W(x_n − x_n,last) + x_n,last) + μ
               +  门 g：静态 per-channel sigmoid（主）或 RCRF 可靠度门（消融）
               +  修正器恒定启用（无开关）
```

> **2026-09-18 修订（用户裁定 D-5）**：早期草案曾把"数据驱动开关 `s`"写进模型定义。
> 现改为：**`s` 不进入模型**——PhaseFormer-L 就是修正器恒定启用的 `weak_residual`，§4.2 不再
> 有"含开关"与"always-on"两列，二者合并。`s` 降级为 §4.7 的**诊断列**：它把命题 1 关于
> "哪些数据集需要电平通道"的预测写成可判对/判错的事前陈述。这样做的代价是主张 A 变强
> （失去按构造在 ETTh1/ETTm1 上不劣化的能力），收益是 §4.2 的结果不再依赖一个由训练集
> 统计量驱动的数据集级分支，避免 `dataset-aware` 路由带来的投机性。

| 组件 | 既有实现 | 既有证据 |
|---|---|---|
| 修正器（稠密） | `weak_period_residual_head_type="shared"`，`WeakPeriodResidualHead` | 7 setting × 3 seed test：MSE 7/7、MAE 6/7 优于 Golden；12 setting × 3 seed（A1 栈） |
| 修正器（低秩） | `weak_period_residual_head_type="pooled_lowrank"`，`pool_factor=1`，`rank = qH` | 7 setting × 3 seed × q∈{1/4,1/8,1/16,1/32}，105/105 审计 |
| 静态门 | `mechanism="weak_residual"`，`weak_period_residual_gate_init` | 全部压缩/解剖/干预证据均在此栈上 |
| 可靠度门 | `mechanism="rcrf_nlinear_plain"`（原始相位路径 + shared 头 + RCRF，无附加校准模块） | 已实现为正式对照；D0 validation 有单 seed 记录 |
| A1 incumbent | `mechanism="gold_combo_reliability_s2"` | 12 setting × 3 seed test（含 ETTm2-96/192 稳定超过 Golden） |

**主工作点**：`weak_residual` + `shared`（稠密）。理由：它是所有机制证据（§3.2–§3.3）实际分析的那个模型，
也是三 seed test 证据最完整的形态。低秩与 RCRF 作为受控变体报告，不作主张。

#### 3.4.2 为什么秩不是 1–2：工作点必须落在已测区间内

命题 2 说的是**主导模式**是一维的，不是"模型应当只有一维"。既有证据一致指出，把秩压到 1–2 会落在性能下行的区间：

| 证据 | 数字 | 含义 |
|---|---|---|
| 三 seed 压缩曲线（`PhaseFormer_rank_sweep_conditioned_experiment.md` §7.3） | 已测最深档 q=1/32（r=3/6/10/22）宏平均 ΔMSE −0.52%、ΔMAE −0.81%，0/7 setting 三 seed 双优；ETTh2-96 q=1/32 已劣于 Golden（0.2757 vs 0.275） | 越深越差的弱趋势在 r=3 就已出现；r=1–2 在网格之外、且在下行方向 |
| RRR 容量上界（`..._rank_capacity_..._report.md` 表 A） | `capture(1)` = 65.5%–86.2%，`capture(2)` = 81.6%–96.7%，`capture(3)` = 88.1%–99.3% | 秩 1 **必然**放弃 14%–35% 的支路价值，秩 2 放弃 3%–18%；ETTh2-96 相对 Golden 只有 0.7% 余量 |
| 训练头解剖（`..._checkpoint_information_analysis_plan.md` 表 2） | q=1/8 主模式修正能量份额：ETTh2-720 0.20、Weather 0.36–0.39、ETTm2/ETTh2-96 0.64–0.80；参与比 1.26–4.42 | 训练出的头在 3/6 setting 上实际使用 2–4 个以上模式 |
| ETTh2 的谱 | 95% 可实现降幅需 8–9 维；压缩档训练头与 `b_1` 对齐仅 0.53–0.72 | 该数据集依赖"若干次优但够用"的子空间，不是单方向 |
| 冻结秩-1（V1） | 6 setting 宏平均 ΔMSE +1.9% | 最接近"秩 1"的既有实验是负结果 |

因此本文的秩选择规则是：**由预测谱决定下界、由已测三 seed 曲线决定工作点**——

- 报告的低秩变体取 **q=1/4 与 q=1/8**（r=H/4、H/8；三 seed 宏平均在 ±0.3% 内，2–3/7 setting 双优），
  作为"容量中性"的实证；
- 深档 q=1/16、1/32 只作容量上界（≥92%）与效率下限（参数 3.5%–6.1%）的说明，明写其 −0.3%～−0.8% 的代价；
- **秩 1–2 只出现在两处**：§3.2 的 `capture(1/2)` 与 §3.3 的主模式（分析对象），以及 §4.6 一行预注册为
  "预期退化、用于量化残项 ε"的边界消融（`pooled_lowrank`，`rank∈{1,2}`，6 setting），其结果无论好坏都不影响主张。

"一维"因此是**对最优映射与学到的映射的主导模式的陈述**，工作点则诚实地停在预测谱的 90%–95% 维数附近。

#### 3.4.3 数据驱动开关：作为**诊断列**的预注册预测（不进入模型）

> **修订（D-5）**：开关**不是**模型的一部分。以下是 §4.7 的一个诊断列：它把"哪些数据集需要
> 电平通道"写成一个事前可判定的预测，从而让命题 1 可被反驳。它不参与任何训练或模型选择，
> 因此即使判错也不污染主结果。

先导证据（§4.1）显示修正器在 ETTh1/ETTm1 上不带来增益甚至略有退化。命题 1 预测这正是"电平已稳定、
相位视角无盲点"的数据。与其事后解释，不如把它写成可事先检验的预测：

```text
s(dataset) = 1[ ν_train(dataset) > ν* ] ,   ν_train ∈ {cycle_level_std, last_cycle_shift, τ̂}（只用训练集）
```

**已冻结的判定（2026-09-18，仅用训练集统计量，不看任何 test）**：三个候选统计量中只有
`τ̂`（电平记忆长度）能把已知增益符号的数据集分开，`cycle_level_std` 与 `last_cycle_shift`
都是**反序**的（ETTm1 高于 ETTm2），故

```text
ν = τ̂ (tau_hat_steps)        ν* = 57.35 步
```

`ν*` 取可分离区间 `(51.11, 63.58)` 的中点——这是一个规则式定义，只用到已知符号数据集的
训练集统计量，不涉及任何 test 数字。由此得到的事前预测是

```text
s = 1 :  ETTh2, ETTm2, Weather        （12 个 setting）
s = 0 :  ETTh1, ETTm1, Electricity, Traffic
```

**须披露的两点**：(i) Electricity 的 `τ̂ = 54.5` 步**落在可分离区间内部**，其判定对 `ν*` 的
位置敏感，是最不确定的一格；(ii) 已有 test-exposed 证据显示 Electricity-336 上修正器相对
matched `phase_only` 是 **+3.6%**（0.16768 → 0.1617，seed 2021），即**开关在 Electricity 上
判错**。按 minipaper §5 第 6 条的规则，判错须如实报告，不得调阈值迁就。

`τ̂` 的估计量本身有已知的有限样本偏差（K=30 个周期时低估真实记忆长度，见
`scripts/phaseformer_L/e19_predictive_stats.py` 的偏差表），因此 `τ̂` 只能作为**序数**仪器读，
不得当作绝对记忆长度；这一点须写入 §4.7 表注。§4.2 按数据集报告门值 `g` 的均值，与 `s` 并列。

#### 3.4.4 与既有失败设计的关系

| 已否定的设计 | 为什么 PhaseFormer-L 不重蹈 |
|---|---|
| 冻结独立 RRR 方向（V1/V2，+1.9%） | 修正器可学习、与主干联合训练（§3.3.3 的支路/融合反向证据要求如此） |
| 周期坐标电平头（`structured_level_shape`，−1.5%～−2.5%） | 保持时间轴坐标；电平读取由训练在 720 维上自行收敛为近端加权核 |
| 输入平滑（14/14 无改善） | 不做任何按能量的输入预处理 |
| 结构化基/共享基（5/5 退化） | 不更换坐标系；低秩只作受控变体且停在已测区间 |

---

## 4. Experiments（预注册）

> **填表状态（2026-09-19）**：§4.1 先导证据为既有结果；**§4.3 已填满**（28 行 + 三类图，E15）；
> **§4.7 的统计量与判定口径已填**（28 setting 的训练集统计量 + 冻结的 `ν*`，E19 阶段 1），
> 其两列 ρ 待 §4.2 的 test 读取完成后回填；**§4.2、§4.4、§4.5、§4.6 待填**——四者的表结构与
> 行数已固定（§4.2 为 24+4 setting × 6 个变体行，§4.4 为 21 行解剖 + 21 行干预，§4.5 为 7 行
> 四臂，§4.6 的三项"本文补做"），只等各自实验产出。

### 4.0 协议

- 数据：主表 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity × H96/192/336/720 = **24 setting**；Traffic 4 个
  setting 作为**探索性附录**（既有证据中无任何 Traffic 记录，不进入判定）。
- 训练：full-train、最低 validation loss checkpoint、seeds 2021/2022/2023、每 checkpoint 只读一次 test。
- 参照：固定 Golden（`PhaseFormer_gold_standard.md`）；matched rerun 仅用于协议诊断。
- **既有结果的复用规则**：主表中已有三 seed、同协议（720 输入、Huber、≤30 epoch、best-val、单次 test）记录的
  格子经逐格审计后**直接复用**，不重训；其余格子新训。2026-09-18 在服务器逐 run 核对后的实际复用范围是：
  `phase_only` 6 个 setting、`weak_residual`（= PhaseFormer-L）7 个 setting、`pooled_lowrank` q=1/4 与 q=1/8
  各 7 个 setting（每个 setting × 3 seed），合计 **81 格**；`L-rcrf` 与 A1 **无任何可复用格**（服务器全库没有
  `rcrf_nlinear_plain` 与 `gold_combo*` 的任何 run）。所有复用格在表中标注来源；曾参与 test-set selection 的
  7 个 setting 在表注中显式披露。
- **超参协议（2026-09-18 冻结）**：新格统一用 preset 默认 `gate_init=0.2`、`lr=1e-3`；复用格保留其 Stage-0
  冻结的 `(gate, lr)`。因此 §4.2 表内存在**三种** gate 先验：新格 0.2、`L-rcrf`/A1 由 preset 自持的 0.5、
  复用格的 Stage-0 冻结值（0.5 或 0.2，逐 setting 不同）。三者必须在表注逐行标注，且配对比较只在
  同一 setting 内成立。
- **盲测边界**：新训格子在本文冻结后不做任何基于 test 的选择；`τ̂`/`ν*` 与机制分析（§4.3–§4.6）只用
  train/validation。§4.1 的先导证据只作动机，不进入主结论。
- **判定门槛（预注册，2026-09-18 冻结）**：
  - 主张 A（不劣化）：PhaseFormer-L 相对 matched `phase_only` rerun 在 24 setting 上**无一**双指标
    回退超过 **1.0%**。注意：D-5 之后 PhaseFormer-L 恒定启用修正器，失去了在 ETTh1/ETTm1 上自动关闭的
    能力，而先导证据在这两个数据集上是 **−1.3% / −2.7%**，因此**本主张有可能不达标**——若如此，按下方
    报告规则如实报告为"未达预注册门槛"。
  - 主张 B（条件性增益 / 诊断预测力）：在诊断列判 `s=1` 的数据集（ETTh2/ETTm2/Weather，共 12 个 setting）上，
    **≥ 3/4 的 setting** 双指标优于 `phase_only`，且这些数据集与 §4.7 的预测一致。判错的数据集
    （判 `s=1` 而无增益，或判 `s=0` 而有增益）须逐个列出。
  - 主张 C（相对 Golden）：按既有严格标准（三 seed 均值 + 样本 std < Golden）逐格报告"稳定超过"计数，**不设
    最低数目门槛**——Golden 来自不同硬件环境，本文对 Golden 只做披露性比较。
    **两种定义须并列报告**：本文上述口径（**均值 + 样本 std**）与
    `PhaseFormer_gold_standard.md` §4 的既有口径（**三 seed 均值**）并不相同，
    故回填时**两个计数同时给出并各自注明口径**——否则同一张表里会出现两个"稳定超过"数字而无处解释其差异。
  - 主张 D（效率）：q=1/8 变体相对 direct 的三 seed 宏平均 |ΔMSE|、|ΔMAE| ≤ **0.5%**。
  - **报告规则**：上述数值一经冻结不得事后调整；若最终未达门槛，须如实报告为"未达预注册门槛"，
    不得改用其他统计口径重述。

### 4.1 先导证据（既有结果，test-exposed，仅作动机）

以下数字来自既有登记文档，7 个 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336）为
test-set selection 所得，**不得表述为盲测**。

| 命题的预测 | 先导观测 | 出处 |
|---|---|---|
| 残差集中在电平不稳定的窗口 | 融合收益与跨周期水平波动 r=+0.49/+0.53、与最后周期电平偏移 r=+0.48/+0.49；修正与相位残差同向 cos 0.70/0.79（ETTm1-192，单 seed） | `PhaseFormer_structural_defect_research_narrative.md` §3 |
| 最优线性修正近似一维 | `λ_1/Σλ` = 0.66–0.86；90% 只需 2–4 维；PR 1.33–2.12 | `PhaseFormer_rank_capacity_and_data_property_report.md` §2.1, §2.6 |
| 读近端加权电平、写恒定位移 | `b_1` 最近 24 步占能量 53%–70%，最佳模板 exp 核 τ=6–72（\|cos\| 0.58–0.78）；`a_1` 与常值 \|cos\| 0.89–0.99、符号 7/7 一致 | 同上 §2.6 |
| 预测维数 ≠ 能量维数 | 首方向只占输入方差 0.8%–12.3%；ETTh2-720 权重谱 95% 能量需 418 维、预测谱 95% 只需 8 维 | 同上 §1.3, §2.3 |
| 训练头真的在用这一维 | 72 个低秩 checkpoint 主模式：读 EMA（\|cos\| 0.70–0.81）、写常数（0.92–0.99）；Semantic-only Δfused MSE 全 72 格最大 +0.0027（q=1/8 格 ≤ +0.0010）；Semantic-drop +4.3%–16.0%（q=1/8），60/72 超出随机 95% 区间；4/6 setting 成立，Weather 为反例 | `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 表 4–7 |
| 修正相对主干定义 | 独立 RRR 方向 1/1+2 冻结为瓶颈：宏平均 +1.94%/+1.99%；H1 在 5/6 setting 3/3 seed 成立；删除子空间后支路自身误差变好 52/72、融合误差变差 72/72（中位 +30.4%） | `PhaseFormer_top2_direction_retention_summary.md`；同上 §11.8 |
| 周期越不稳定盲点越大 | 探针相对 phase_only：ETTm2 +6.2%、ETTh2 +3.4%、Weather +2.0%；ETTh1 −1.3%、ETTm1 −2.7%（单 seed，静态门） | `PhaseFormer_joint_lowrank_rank_sweep_plan.md` §13.4 |
| 压缩是放大镜不是增益 | 最深档（参数 3.5%–6.1%）保留 92.4%–101.9% 可实现价值；三 seed test 宏平均 −0.12/+0.07/−0.32/−0.52%，唯一可复现增益 ETTh2-720 q=1/8（+1.05%/+0.43%，3/3） | `PhaseFormer_rank_sweep_conditioned_experiment.md` §7 |

### 4.2 主结果：PhaseFormer-L vs matched PhaseFormer 与 Golden（24 setting × 3 seed；Traffic 附录）

| Dataset | H | Golden MSE/MAE | `phase_only`（matched） | PhaseFormer-L（恒定启用） | Δ vs phase_only | `g` 均值 | 诊断 `s` | 稳定超过 Golden | 来源/披露 |
|---|---:|---:|---|---:|---:|---|---|---|---|
| ETTh1 | 96 | 0.359/0.382 |  |  |  |  |  |  |  |
| ETTh1 | 192 | 0.397/0.404 |  |  |  |  |  |  |  |
| ETTh1 | 336 | 0.425/0.424 |  |  |  |  |  |  |  |
| ETTh1 | 720 | 0.431/0.450 |  |  |  |  |  |  |  |
| ETTh2 | 96 | 0.275/0.338 |  |  |  |  |  |  |  |
| ETTh2 | 192 | 0.341/0.376 |  |  |  |  |  |  |  |
| ETTh2 | 336 | 0.369/0.405 |  |  |  |  |  |  |  |
| ETTh2 | 720 | 0.402/0.436 |  |  |  |  |  |  |  |
| ETTm1 | 96 | 0.293/0.344 |  |  |  |  |  |  |  |
| ETTm1 | 192 | 0.323/0.361 |  |  |  |  |  |  |  |
| ETTm1 | 336 | 0.358/0.381 |  |  |  |  |  |  |  |
| ETTm1 | 720 | 0.412/0.410 |  |  |  |  |  |  |  |
| ETTm2 | 96 | 0.163/0.256 |  |  |  |  |  |  |  |
| ETTm2 | 192 | 0.219/0.293 |  |  |  |  |  |  |  |
| ETTm2 | 336 | 0.269/0.326 |  |  |  |  |  |  |  |
| ETTm2 | 720 | 0.351/0.379 |  |  |  |  |  |  |  |
| Weather | 96 | 0.148/0.195 |  |  |  |  |  |  |  |
| Weather | 192 | 0.193/0.237 |  |  |  |  |  |  |  |
| Weather | 336 | 0.242/0.278 |  |  |  |  |  |  |  |
| Weather | 720 | 0.309/0.332 |  |  |  |  |  |  |  |
| Electricity | 96 | 0.129/0.221 |  |  |  |  |  |  |  |
| Electricity | 192 | 0.148/0.238 |  |  |  |  |  |  |  |
| Electricity | 336 | 0.165/0.257 |  |  |  |  |  |  |  |
| Electricity | 720 | 0.201/0.285 |  |  |  |  |  |  |  |
| Traffic | 96 | 0.361/0.238 |  |  |  |  |  |  | 探索性附录，不进入判定 |
| Traffic | 192 | 0.373/0.243 |  |  |  |  |  |  | 探索性附录，不进入判定 |
| Traffic | 336 | 0.385/0.248 |  |  |  |  |  |  | 探索性附录，不进入判定 |
| Traffic | 720 | 0.428/0.270 |  |  |  |  |  |  | 探索性附录，不进入判定 |

**变体行（同协议、同 seed，全部为既有 preset）**：

| 行 | preset / 配置 | 作用 | 新训规模 |
|---|---|---|---|
| `phase_only` | `original`（matched rerun） | 配对基线 | 24 − 6 复用 = 18 setting × 3；Traffic 4 × 3 = 12 |
| PhaseFormer-L | `weak_residual`，`shared` 头，修正器**恒定启用**（D-5） | 主张 A/B | 24 − 7 复用 = 17 setting × 3；Traffic 4 × 3 = 12 |
| L-q1/4、L-q1/8 | `pooled_lowrank`，`rank=H/4`、`H/8` | 主张 D（效率） | 各 24 − 7 复用 = 17 × 3；Traffic 各 4 × 3 = 12 |
| L-rcrf | `rcrf_nlinear_plain` | 门的消融 | 28 × 3 = 84（无复用格） |
| A1 | `gold_combo_reliability_s2` | incumbent 参照 | 24 × 3 = 72（**无复用格**，见下） |

> **A1 行的勘误**：本文早期草案写"A1 有 12 格既有"。2026-09-18 审计在**本地与服务器两侧**都未找到
> 任何 `gold_combo*` 产物，且 `docs/agent-log.md` 记录的那批 A1 运行使用 **MAE loss、batch 256、lr 3e-4**，
> 而 §4.0 要求 **Huber**——即使找回也不满足"同协议"。因此 A1 **不存在可复用格**，按 §4.0 协议全训
> 24 × 3 = 72 runs。A1 是 incumbent 参照行，不进入主张 A–D。

参数量列按仓库 `metrics.csv` 的 `parameter_count` 口径报告（主干 + 修正器 + 门），并单列修正器参数；
FLOPs 不在本文口径内比较（原文 Table 4 口径未在本仓库复现）。

**必答问题**：(a) ETTh2 四个 horizon 相对 `phase_only` 与 Golden 的差距是否收窄、是否达到 FITS 的引用数字；
(b) 逐 dataset 报告门值 `g` 的均值；诊断列 `s` 是否把 ETTh1/ETTm1 判为 `s=0`，若判为 `s=1` 则如实报告其表现；
(c) q=1/8 与 direct 的三 seed 差是否在 ±0.5% 内。

### 4.3 相位补空间的维数（7 数据集，train/validation）

| Dataset | H | `λ_1/Σλ` | `pred_dims_90` | PR | `b_1` 最佳模板（\|cos\|） | `a_1` vs 常值 \|cos\| | `used_var_share(1)` |
|---|---:|---:|---:|---:|---|---:|---:|
| ETTh1 | 96 | 0.712 | 4 | 1.90 | exp τ=24 (0.737) | 0.950 | 0.051 |
| ETTh1 | 192 | 0.724 | 3 | 1.85 | exp τ=72 (0.767) | 0.958 | 0.064 |
| ETTh1 | 336 | 0.731 | 3 | 1.81 | exp τ=72 (0.786) | 0.963 | 0.069 |
| ETTh1 | 720 | 0.747 | 3 | 1.75 | exp τ=72 (0.793) | 0.968 | 0.078 |
| ETTh2 | 96 | 0.779 | 3 | 1.62 | exp τ=6 (0.575) | 0.967 | 0.011 |
| ETTh2 | 192 | 0.824 | 3 | 1.46 | exp τ=6 (0.576) | 0.976 | 0.022 |
| ETTh2 | 336 | 0.843 | 3 | 1.40 | exp τ=6 (0.581) | 0.982 | 0.031 |
| ETTh2 | 720 | 0.857 | 3 | 1.36 | exp τ=24 (0.576) | 0.989 | 0.046 |
| ETTm1 | 96 | 0.713 | 3 | 1.88 | exp τ=72 (0.553) | 0.923 | 0.071 |
| ETTm1 | 192 | 0.717 | 3 | 1.87 | exp τ=72 (0.657) | 0.939 | 0.102 |
| ETTm1 | 336 | 0.728 | 3 | 1.82 | exp τ=168 (0.639) | 0.949 | 0.125 |
| ETTm1 | 720 | 0.741 | 3 | 1.77 | exp τ=168 (0.657) | 0.957 | 0.157 |
| ETTm2 | 96 | 0.725 | 3 | 1.83 | exp τ=6 (0.670) | 0.947 | 0.008 |
| ETTm2 | 192 | 0.746 | 3 | 1.75 | exp τ=6 (0.670) | 0.962 | 0.012 |
| ETTm2 | 336 | 0.791 | 3 | 1.58 | exp τ=6 (0.647) | 0.965 | 0.023 |
| ETTm2 | 720 | 0.837 | 3 | 1.42 | exp τ=6 (0.604) | 0.976 | 0.043 |
| Weather | 96 | 0.862 | 2 | 1.33 | exp τ=6 (0.662) | 0.943 | 0.072 |
| Weather | 192 | 0.783 | 2 | 1.57 | exp τ=24 (0.740) | 0.957 | 0.079 |
| Weather | 336 | 0.792 | 2 | 1.55 | exp τ=72 (0.738) | 0.964 | 0.123 |
| Weather | 720 | 0.811 | 2 | 1.49 | exp τ=72 (0.723) | 0.970 | 0.210 |
| Electricity | 96 | 0.661 | 4 | 2.12 | exp τ=24 (0.660) | 0.891 | 0.077 |
| Electricity | 192 | 0.659 | 4 | 2.13 | exp τ=72 (0.757) | 0.890 | 0.102 |
| Electricity | 336 | 0.662 | 4 | 2.12 | exp τ=72 (0.784) | 0.892 | 0.123 |
| Electricity | 720 | 0.666 | 4 | 2.09 | exp τ=72 (0.790) | 0.895 | 0.142 |
| Traffic | 96 | 0.643 | 5 | 2.27 | exp τ=168 (0.709) | 0.931 | 0.177 |
| Traffic | 192 | 0.642 | 7 | 2.29 | exp τ=168 (0.833) | 0.928 | 0.257 |
| Traffic | 336 | 0.643 | 7 | 2.28 | exp τ=168 (0.864) | 0.931 | 0.313 |
| Traffic | 720 | 0.649 | 6 | 2.25 | exp τ=168 (0.862) | 0.932 | 0.353 |

**表注与披露（E15，2026-09-19 回填）**：

- **口径**：全部在**训练/验证**划分上计算，**从不读 test**；标准化为训练集均值/总体标准差。与 §4.2 的
  test 增益列**不同源**，不可直接相乘或相减。
- **覆盖来源**：28 行中 **21 行为新增 coverage**（`source=new_28_minus_7`），7 行沿用既有
  `lowrank_data_property_v2` 产物（`source=reused_v2_artifact`，即 §4.1 的 7 个先导 setting）；
  该 7 行在 `--verify-existing` 门下与既有产物逐字段一致（moments 相对差 0.0，23 列全等）。
- **`b_1` 最佳模板的定义**：在指数衰减核族 `exp(-lag/τ)`、τ ∈ {6, 24, 72, 168} 上取 `|cos|` 最大者
  （`docs/PhaseFormer_rank_capacity_and_data_property_report.md` §2.6(c) 的 16 模板族中含 4 个指数核）。
  该文档是 **2 位小数**参照，7 个先导 setting 的可复现性为 6/7 完全相同、ETTh2-96 差 0.005（0.575 vs 0.58，
  即舍入边界）；精确值见 `b1_template_detail.csv`。
- **与先导区间的关系**：`λ_1/Σλ` 实测 0.642–0.862（先导 0.66–0.86）、`a_1` 与常值 `|cos|` 0.890–0.989
  （先导 0.892–0.989）、`PR` 1.33–2.29（先导 1.33–2.12，Traffic 把上界推高）。**Traffic 的
  `used_var_share(1)` 最高（0.177–0.353）且 `b_1` 一致落在 τ=168**，是全部 28 个 setting 中最"长记忆电平"
  的一档——但按 §4.0 它只是探索性附录，不进入判定。
- **`λ_1/Σλ` 的分母是"可实现降幅"**，`Σλ = MSE_persist − MSE_rank→∞`，非总方差。
- **产物**：`research_runs/phaseformer_L_e15_dimension_v1/`（`dimension_table.csv`、`leading_direction.csv`、
  `optimal_rank_capture.csv`、`b1_template_detail.csv`、`figures/` 三类图、28 个 `moments_*.npz`）。

**图（实测，由 `e15_dimension.py` 在 28 个 setting 上生成）**：
`figures/scree_lambda_spectrum.png`（`λ_1/Σλ` 谱，首根 0.642–0.862）、
`figures/b1_lag_profile.png`（`b_1` 随 lag 的剖面：近端集中 + 指数衰减，Traffic 一格一致落在 τ=168）、
`figures/a1_horizon_profile.png`（`a_1` 随 horizon 的剖面：近似平线，常值 `|cos|` 0.890–0.989）。
（本条原为预注册写法"预期图：…第一根柱 0.66–0.86 量级"，即引用**先导**区间；
E15 完成后改为报告**实测**值与实际文件名，数值与上文条目一致，未引入新数字。）

### 4.4 训练头的解剖（PhaseFormer-L 与低秩探针，3 seed）

| 模型 | Dataset | H | 主模式输入组 / 解释率 | 主模式输出组 / 解释率 | 修正能量份额 | 跨 seed `leading4` 重叠 | 稳定语义判定 |
|---|---|---:|---|---|---:|---:|---|
| PhaseFormer-L | ETTh2 | 96 |  |  |  |  |  |
| PhaseFormer-L | ETTh2 | 720 |  |  |  |  |  |
| PhaseFormer-L | ETTm2 | 96 |  |  |  |  |  |
| PhaseFormer-L | ETTm2 | 192 |  |  |  |  |  |
| PhaseFormer-L | Weather | 96 |  |  |  |  |  |
| PhaseFormer-L | Weather | 192 |  |  |  |  |  |
| PhaseFormer-L | Electricity | 336 |  |  |  |  |  |
| L-q1/4 | ETTh2 | 96 |  |  |  |  |  |
| L-q1/4 | ETTh2 | 720 |  |  |  |  |  |
| L-q1/4 | ETTm2 | 96 |  |  |  |  |  |
| L-q1/4 | ETTm2 | 192 |  |  |  |  |  |
| L-q1/4 | Weather | 96 |  |  |  |  |  |
| L-q1/4 | Weather | 192 |  |  |  |  |  |
| L-q1/4 | Electricity | 336 |  |  |  |  |  |
| L-q1/8 | ETTh2 | 96 |  |  |  |  |  |
| L-q1/8 | ETTh2 | 720 |  |  |  |  |  |
| L-q1/8 | ETTm2 | 96 |  |  |  |  |  |
| L-q1/8 | ETTm2 | 192 |  |  |  |  |  |
| L-q1/8 | Weather | 96 |  |  |  |  |  |
| L-q1/8 | Weather | 192 |  |  |  |  |  |
| L-q1/8 | Electricity | 336 |  |  |  |  |  |

> **本表的行与口径（E16，2026-09-19 登记）**：行 = 3 个模型 × 7 个 test-selected setting
> （ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336）= **21 行**，每行 3 seed 聚合；
> 7 个 setting 由 test-set selection 得到，**不是盲测样本**，逐行披露。
> 全部解剖与干预**只用 validation 划分**（`evaluation_split=val`、`test_split_read=false`），
> 与 §4.2 的 test 增益列不同源。稠密 `PhaseFormer-L` 头的有效映射即其 `W`（H×720）；
> 低秩探针为 `W_dec·W_enc`。跨 seed `leading4` 重叠按 4 维主子空间的两两重叠计。
>
> **`reference_parity` 的适用范围（须在表注写明）**：与既有
> `lowrank_checkpoint_information_v1` 的逐字段比对**只对 6 个 setting 成立**——
> 该参照不包含 **Electricity-336**（其缺席源于 E10 因 **13.9 GiB OOM** 排除了这一格）。
> 对缺少参照的 cell，E16 **跳过而非判失败**（`e16_dissection.py:2337-2344`）。
> 因此 Electricity-336 的解剖是**本文新算**、不是复现，表注不得把它读作 parity 通过。
> 可比的 6 个 setting 上，`probe_cells ≈ 72` = 6 setting × 3 seed × 2 个低秩臂 × 2 个参照文件。

干预表（每 cell **10 个登记臂**——`Original` / `Semantic-only` / `Semantic-drop` /
`Semantic8-only` / `Semantic8-drop` / `Bias-off` / `PCA-only` / `PCA-drop` 共 8 个，
加同维 `PCA-matched-only` / `PCA-matched-drop` 共 10 个——之上再按该 cell 的可用基向量追加
`Independent-RRR-only`、`Conditional-RRR-only` 与**新增的 `RandomRRR-drop`**，
故实际为 **11–12 臂（11 为下界）**；同时报告支路自身与融合误差）：

| Dataset | H | q/r | Semantic-only Δfused | Semantic-drop Δfused | 随机 95% 区间 | PCA-drop | **随机 RRR 子空间 drop**（新增对照） | 支路自身 Δ | 融合 Δ |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| ETTh2 | 96 | dense（r=H） |  |  |  |  |  |  |  |
| ETTh2 | 720 | dense（r=H） |  |  |  |  |  |  |  |
| ETTm2 | 96 | dense（r=H） |  |  |  |  |  |  |  |
| ETTm2 | 192 | dense（r=H） |  |  |  |  |  |  |  |
| Weather | 96 | dense（r=H） |  |  |  |  |  |  |  |
| Weather | 192 | dense（r=H） |  |  |  |  |  |  |  |
| Electricity | 336 | dense（r=H） |  |  |  |  |  |  |  |
| ETTh2 | 96 | q=1/4（r=24） |  |  |  |  |  |  |  |
| ETTh2 | 720 | q=1/4（r=180） |  |  |  |  |  |  |  |
| ETTm2 | 96 | q=1/4（r=24） |  |  |  |  |  |  |  |
| ETTm2 | 192 | q=1/4（r=48） |  |  |  |  |  |  |  |
| Weather | 96 | q=1/4（r=24） |  |  |  |  |  |  |  |
| Weather | 192 | q=1/4（r=48） |  |  |  |  |  |  |  |
| Electricity | 336 | q=1/4（r=84） |  |  |  |  |  |  |  |
| ETTh2 | 96 | q=1/8（r=12） |  |  |  |  |  |  |  |
| ETTh2 | 720 | q=1/8（r=90） |  |  |  |  |  |  |  |
| ETTm2 | 96 | q=1/8（r=12） |  |  |  |  |  |  |  |
| ETTm2 | 192 | q=1/8（r=24） |  |  |  |  |  |  |  |
| Weather | 96 | q=1/8（r=12） |  |  |  |  |  |  |  |
| Weather | 192 | q=1/8（r=24） |  |  |  |  |  |  |  |
| Electricity | 336 | q=1/8（r=42） |  |  |  |  |  |  |  |

新增"随机 RRR 子空间"对照用于区分"语义有效"与"任意同数量主方向有效"（既有结果中 57/57 单元
Semantic-drop ≡ PCA-drop，此项此前缺失）。

### 4.5 条件性学习（贡献 4 的正式检验）

| Dataset | H | direct | 冻结独立-RRR 方向 1 | 冻结条件-RRR 方向 1 | PhaseFormer-L（联合） | H1：cond 距离 < indep 距离（seed 数） |
|---|---:|---|---|---|---|---|
| ETTh2 | 96 |  |  |  |  |  |
| ETTh2 | 720 |  |  |  |  |  |
| ETTm2 | 96 |  |  |  |  |  |
| ETTm2 | 192 |  |  |  |  |  |
| Weather | 96 |  |  |  |  |  |
| Weather | 192 |  |  |  |  |  |
| Electricity | 336 |  |  |  |  |  |

预测：冻结独立方向退化；冻结**条件性**方向应明显好于独立方向（此对照此前未做，是区分"冻结本身有害"与
"独立目标错位"的关键）；联合训练最好。

**本表的行与口径（E17，2026-09-19 登记，含一条重要前置发现）**：

- 行 = 7 个 test-selected setting × 3 seed（每格三 seed 聚合）；7 个 setting 由 test-set selection 得到，
  不是盲测样本。
- **`direct` 与 `PhaseFormer-L（联合）` 在本实现下是同一配置**：PhaseFormer-L 的修正器就是与主干
  联合训练、无瓶颈约束的头。两列数值相同是**构造使然**，不是两次独立实验——表注必须写明。
- 冻结臂的投影器由各 setting 的 **train split** 单独计算并冻结；投影发生在 `x_last` 中心化之后、
  线性层之前（沿用 V1/V2 的 `set_projection_basis` 机制）。
- **前置发现（E17 投影器阶段，2026-09-19）**：`D_cond = y − y_φ` 与 `D_ind = y − x_last` 的**首方向
  在 6/7 个 setting 上几乎相同**（`│cos│ ≥ 0.9991`：ETTh2-96 0.99962、ETTh2-720 0.99997、
  ETTm2-96 0.99909、ETTm2-192 0.99993、Weather-96 0.99938、Weather-192 0.99981），
  只有 **Electricity-336 为 0.0066（近正交）**。
  因此"冻结独立"与"冻结条件"两条臂在那 6 个 setting 上**按构造等价**（差异仅为浮点级投影器差别），
  **本对照的真实检验力集中在 Electricity-336 一格**。这条必须写进表注，否则该表的 7 行会被误读为
  7 次独立检验。
- 该发现本身**支持命题 2**：首方向是**数据的性质**，不是**目标定义的产物**；只有在 `λ_1/Σλ` 最低
  （Electricity-336 = 0.662，7 个 setting 中最低）且 `pred_dims_90 = 4` 的 setting 上，目标定义才足以改变首方向。
- 已核对的独立证据：独立路线对 E8 已发布的 6 个投影器复现 `│cos│ = 1.0`（浮点精度内完全一致）。
- H1 列沿用既有登记（`PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 表 5，72 行），
  判定口径为"**seed 内多数 rank 支持**"；该证据**不含 Electricity-336**，该格记 `evidence_missing`，
  不得推断。

### 4.6 按能量 vs 按任务：负对照汇总

| 操作 | 作用对象 | 口径 | 结果（既有，test-exposed） | 本文补做 |
|---|---|---|---|---|
| 输入平滑（boxcar / causal EMA，各 5 档） | 支路输入 | 7 setting | 14/14 组合无一改善，越平滑越差 | 在 PhaseFormer-L 上复测 2 档（见下注）：**42 runs** = 7 setting × 3 seed × 2 档 |
| 结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离） | 支路参数化 | 4 setting | 5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52% | — |
| SVD 截断 vs 秩约束训练 | 支路权重 | Electricity-336 r=10 | 截断 +29%，训练 +0.7% | **全 28 setting**（r=10）；**口径差异见下注**（原写在本格的披露已移至表下注，以免 §4.6 回填整行替换时丢失） |
| 联合低秩训练 q=1/32 | 支路容量 | 7 setting | 保留 92.4%–101.9% 可实现价值 | — |
| **边界消融：`pooled_lowrank` rank∈{1,2}** | 支路容量（网格之外） | 6 setting × 3 seed | 无 | **预注册预期：相对 direct 退化**，幅度上界由 `1−capture(1/2)`（14%–35% / 3%–18% 的支路价值）经 `g²` 折算；用于量化残项 ε，不影响主张；**36 runs** = 6 setting × 3 seed × rank∈{1,2}（**绝对**秩，非 `H/4`、`H/8` 的相对秩） |

> **§4.6 行 3 的口径差异（须披露）**：既有 E11 为 **test** 口径、单 seed、checkpoint 取自
> `rank_sweep_2_stage1`；本表补做部分为 **validation** 口径、checkpoint 取自 §4.2 的
> `l_main`/`l_q1_4`/`l_q1_8`——两者**只有截断代数与三段式比较结构相同**，
> 故行 3 的"既有"与"本文补做"两列**不可直接相减**。
> （本条原写在行 3 的"本文补做"单元格内；因 §4.6 的回填方式是**用回填工具产出的 5 行整行替换**，
> 留在单元格里会在回填时被静默删除，故**先移到此处**。）

> **§4.6 行 1 的「2 档」口径（2026-09-19 更正）**：先导实验的「两个算子」在 PhaseFormer-L 上**不可能都复现**——
> `smooth_ratio` 的算子取决于头：PhaseFormer-L 用的稠密 `shared` 头把它实现为与 **causal EMA** 的混合
> （`src/models/phase_adapters.py:113-115`），真正的 boxcar（`F.avg_pool1d`）只存在于 `pooled_lowrank`
> 头（`:168-179`）。因此 PhaseFormer-L 上的 `smooth_ratio` 是**强度**而非算子。据此把「2 档」定义为
> 同一算子（causal EMA，`alpha` 取算子默认 0.08）的两个**不同强度**：`smooth_ratio = 0.5` 与 `1.0`，
> 两者都落在先导扫描的网格 `s ∈ {0, 0.25, 0.5, 0.75, 1}` 内。
> 早期草案曾固定 `smooth_ratio=0.5` 只改 `alpha`，但 `alpha=0.08` 本就是算子默认值，那两个「档」是
> **数值上完全相同的配置**；该草案已作废。此口径在 E18 冒烟前冻结，不得事后调整。


### 4.7 命题 1 的预测力：电平非平稳度 vs 增益（训练无关）

对 28 个 setting 计算训练集上的跨周期电平统计量（`cycle_level_std`、`last_cycle_shift`、电平自相关时间
`τ̂`），与 §4.2 中 PhaseFormer-L 相对 `phase_only` 的增益及门值 `g` 做相关：

| 统计量 | 与 ΔMSE 的 Spearman ρ | 与 `g` 的 ρ | 预期符号 |
|---|---:|---:|---|
| `cycle_level_std` | | | + |
| `last_cycle_shift` | | | + |
| `τ̂`（电平记忆长度，`tau_hat_steps`） | | | 与学到的 EMA τ 正相关 |

**表注（必须写入）**：

1. **判定口径已冻结**（2026-09-18，仅用训练集统计量）：`ν = tau_hat_steps`，`ν* = 57.35` 步
   （取已知符号数据集的可分离区间 `(51.11, 63.58)` 的中点）。三候选的分隔能力实测为：

   | 候选 `ν` | 正类（Weather 96.9 / ETTh2 88.2 / ETTm2 63.6 步） | 负类（ETTh1 51.1 / ETTm1 34.9 步） | 可分离 |
   |---|---|---|---|
   | `cycle_level_std` | 0.507 / 0.418 / 0.348 | 0.371 / 0.489 | **否（反序）** |
   | `last_cycle_shift` | 0.467 / 0.398 / 0.336 | 0.363 / 0.434 | **否（反序）** |
   | `tau_hat_steps` | 96.9 / 88.2 / 63.6 | 51.1 / 34.9 | **是** |

2. **`τ̂` 有已知的有限样本偏差**：它由 K=30 个周期电平的 lag-1 自相关估计，
   `-1/ln(ρ)` 在 ρ→1 附近发散且在短窗下低估真实记忆。合成 AR(1) 上实测
   （真实 τ → 估计）：0.43→0.35、0.83→0.72、1.44→1.25、2.80→2.17、9.49→4.53 个周期。
   估计量**单调**，因此可用于排序与相关，但**不得**读作绝对记忆长度。
3. **诊断列 `s` 的判错须如实报告**：Electricity 的 `τ̂ = 54.5` 步落在可分离区间内部（判定最不确定），
   而已有 test-exposed 证据显示 Electricity-336 上修正器相对 matched `phase_only` 为 **+3.6%**，
   即该格**判错**。按 §4.0 报告规则列出全部判错格，不调阈值迁就。

若成立，则"哪些数据集需要电平通道"可由数据统计量事前预测；Weather 类弱周期数据若主模式转向曲率/慢趋势，
应作为命题的边界条件单独报告。

---

## 5. Limitations 与披露

1. §4.1 的全部先导证据来自 test-set selection 得到的 6–7 个 setting，属条件性、test-exposed 证据；主表中复用的
   7 个 direct 格与 6 个 `phase_only` 格同样来自这些 setting，表中逐格标注。本文的主张 A–D 只能建立在
   §4.2 冻结后的新训格与 train/validation 分析上。
2. 命题 1–2 目前是证明路线而非完整证明；命题 2 依赖"电平在预测区间内近似持续"的近似，长 horizon 上残项 `ε`
   不小（ETTh2-720 主模式修正能量份额仅 0.20）。**因此本文不主张模型应为秩 1**；"一维"是对主导模式的陈述，
   工作点停在预测谱 90%–95% 维数附近（§3.4.2）。
3. 既有 checkpoint 解剖中只有主模式可命名；第 2 个及以后模式跨 seed 不稳定，报告为"多组等价信息通路"。
   Weather 两个 setting 的主模式指向曲率/慢趋势而非电平，是命题的边界而非支持。
4. "语义有效"与"任意同数量主方向有效"尚未分开（既有 57/57 单元 Semantic-drop ≡ PCA-drop）；§4.4 新增的
   随机 RRR 子空间对照是解决此项的必要实验，需在 `evaluate_lowrank_semantic_interventions.py` 增加一个臂
   （分析侧代码，不涉及模型）。
5. 低秩压缩是分析工具与效率选项，不是精度贡献；既有三 seed 结果无普适增益，深档 q=1/16、1/32 有 −0.3%～−0.8%
   的代价，本文明写。
6. **诊断列 `s` 不是模型的一部分**（D-5）：PhaseFormer-L 恒定启用修正器，`s` 只是 §4.7 的一个事前
   预测列。阈值 `ν*` 由 5 个已知增益符号数据集的训练集统计量按规则式定义拟合，样本量极小；它是
   可预注册、可被 Traffic 与 H336/720 新格子反驳的预测，不是已验证的规律。若诊断在新格上判错
   （已知至少 Electricity 一格判错），按 §4.0 主张 B 如实报告为未达标并逐格列出。
   注意 §3.4.3 早期草案称"由 6 个已知 setting 拟合"，实际证据支持的是 **5 个已知符号的数据集**
   （ETTh2/ETTm2/Weather 为正、ETTh1/ETTm1 为负），此处以证据为准。
7. Golden 来自不同硬件环境；本文对 Golden 只做披露性比较，主张 A/B 的配对基线是同环境 matched rerun。
8. 结论范围：与相位主干经凸门联合训练的线性残差修正、标准长程基准；不宣称任意时序模型可压缩或任意线性模型
   等价于电平修正。

---

## 6. 关联文档

- **执行入口与结果回写目标**：`PhaseFormer_L_experiment_plan.md`（本文 §4 全部表格的施工图、
  缺口核对 G1–G18 与登记不一致 D1–D6）
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
