# PhaseFormer-L: The One Dimension That Phase Tokenization Leaves Behind

> **文档性质**：期刊扩展论文的 MiniPaper 草稿（2026-09-18；**§4 结果区已于 2026-09-20 起陆续回填**）。
> Introduction 与 Methodology 为完整稿；§4.1 的"先导证据"一节引用既有结果，且全部标注 test-exposed；
> §4.2 主表（含 2026-09-21 回填的 §4.2.3 全 setting 逐臂最优 seed 汇总）、§4.3 维数表、§4.7 预测力表已填满，§4.4/§4.5/§4.6 仍在产出中（缺格状态见各节表注与
> `scripts/phaseformer_L/verify_minipaper_fill.py` 的 `blank`/`PENDING` 计数）。
> 本文不改变任何既有结论的口径；所有先导数字均可在被引用的登记文档中逐项核对。
>
> **执行入口与回写目标**：`docs/PhaseFormer_L_experiment_plan.md`（2026-09-18 登记）——本文 §4 各表的
> 唯一执行契约与结果回写目标。该文 §4 给出缺口核对结论：**G1–G18 全部确认存在**（既有证据最大只覆盖
> 7 个 setting，且全部为条件性数字），并列出 6 项登记不一致 D1–D6 待修。
>
> **D1/D2 复核结果（2026-09-18）**：用服务器 `research_runs/lowrank_checkpoint_information_v1/intervention_results.csv`
> 的单向 rsync 副本（720 行 = 72 cell × 10 臂）复算：Semantic-drop 下支路自身 MSE 变好 **52/72**、融合 MSE 变差
> **72/72**、相对上升中位 **+30.43%**（范围 +5.72%～+51.65%）——与 §4.1 引用一致；Semantic-only 的 Δfused MSE
> 全 72 格最大 **+0.0027**（q=1/8 格最大 +0.0010），此前"≤ +0.0006"为 3-seed 均值口径且偏小，**已按全格最大值
> 更正**。两项数字**已于 2026-09-20 登记**于 `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md`
> 的**表 6 正文**（执行计划 D1/D2 的要求已满足）；登记时从表 6 的 **720 行独立复算**验证：
> `Semantic-drop` 支路变好 **52/72**、融合变差 **72/72** 逐 cell 成立；`Semantic-only` 的 `Δfused MSE`
> 全格最大 **+0.002665**、`q=1/8` 最大 **+0.000967**（即上文 +0.0027 / +0.0010 的原始精度）。
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
on datasets with non-stationary cross-cycle level. That prediction holds, and it is sharp: across the twelve
settings of ETTh2, ETTm2 and Weather the corrector improves on the matched phase-only baseline in **12 of 12**
cases (up to 8.7% MSE), the pre-registered diagnostic assigns `s=1` to **exactly** those twelve settings, and
the training-set level-memory length correlates with the gain at **ρ = −0.750**. It is not a universal win, and
we report it as such: the same corrector degrades ETTh1 and ETTm1 by up to 2.7%, and two pre-registered bounds
are not met — uniform non-degradation, and rank efficiency (the rank-`H/8` variant deviates from the dense
corrector by 0.72% MSE per setting against a 0.5% bound).

**中文。** PhaseFormer 证明按相位（跨周期同偏移位置）做 token 化，得到的表示对周期形状变化结构不变、且位于
极低维子空间，因此约 1k 参数即可达到 SOTA。但其理论假设周期局部平稳，非平稳性被留作未来工作。本文证明相位
视角**看不见的那部分**本质上是一维的：跨周期电平漂移把相位子空间恰好推偏一个方向，对该偏移的最优线性修正
在小残项意义下为秩 1。我们在 7 个基准上用三种独立方法验证：样本级诊断显示相位残差集中在电平不稳定的窗口；
闭式降秩回归显示单一输入—输出模式即占线性修正全部可实现价值的 66%–86%；对 72 个联合训练的低秩修正头做规范
分解与因果干预，其主模式读取近端加权电平、写出整段恒定位移。我们进一步证明该修正必须**以相位主干为条件**
学习：同一方向若从独立回归中冻结得到反而使模型退化，而联合学到的子空间与主干条件性目标对齐，删除它会损害
融合预测——即便支路自身误差反而变好。这些发现把探针收束为 **PhaseFormer-L**：PhaseFormer 加一条门控线性电平修正器，
其秩由预测谱而非参数预算决定——可压缩到稠密头参数的 3.5%–6.1% 且可证保留 ≥92% 的价值——并且其增益能仅凭训练
数据预测为集中在跨周期电平非平稳的数据集上。该预测成立且很锐：在 ETTh2、ETTm2 与 Weather 的 12 个 setting 上，
修正器相对配对 `phase_only` 基线 **12/12 全部变好**（最多 8.7% MSE），预登记的诊断列把 `s=1` 恰好赋给**这 12 个**
setting（精确率 12/12），且训练集电平记忆长度与增益的相关系数达 **ρ = −0.750**。它**不是普适增益**，我们照实报告：
同一修正器在 ETTh1 与 ETTm1 上最多退化 2.7%，且两条预登记界**未达标**——"一致不退化"与"秩效率"
（rank-`H/8` 变体相对稠密修正器逐 setting 偏差 0.72% MSE，界为 0.5%）。

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
> `mechanism` / 配置项在表中注明，§4 各表都能用既有 runner 直接补齐。

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

**须披露的两点**：(i) Electricity 的 `τ̂ = 54.5` 步**落在可分离区间内**——确切地说是落在
**可分离区间内部**（`54.5 ∈ (51.11, 63.58)`），其判定对 `ν*` 的位置敏感，是全表最不确定的一格；
(ii) 实测其修正器有增益、而诊断列判它 `s=0`，即**诊断在 Electricity 上判错（假阴性）**：
seed 2021 为 `0.167681 → 0.161729`（**−3.55%**，修正器更好），§4.2 的三 seed 口径为四档
**−0.43% ～ −2.88% 全部为负**（均值 **−1.36%**），与 §4.2.1 `claims.json` 的
`diagnostic_misses`（Electricity 四档全部在列）一致。按 minipaper §5 第 6 条的规则，判错须如实
报告，不得调阈值迁就。
> **勘误（2026-09-20）**：本条此前写作"修正器相对 matched `phase_only` 是 **+3.6%**
> （0.16768 → 0.1617）"——**符号写反了**。`0.1617 < 0.16768`，该差实为 **−3.6%**（MSE 越低越好，
> 故修正器**更好**）。原文"The corrector hurts, therefore s=0 is wrong"这一步在逻辑上不成立：
> 若修正器真的更差，`s=0` 反而**判对了**。**结论方向不变但理由必须倒过来**：正因为修正器在
> Electricity 上有增益，判 `s=0` 才是漏报。§4.2.1 的判错格清单本就是按"Δ<0 而 s=0"生成的，
> 与本条更正后一致，未受影响。

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

## 4. Experiments（预注册 → 2026-09-20 起逐节回填）

> **填表状态（2026-09-19 登记；2026-09-20 终版更新）**：
> **§4.1** 先导证据为既有结果；**§4.2 主表已填满**（24+4 setting，含 §4.2.1 的判定）；**§4.3 已填满**（28 行 + 三类图）；
> **§4.4 已填 6/7 setting × 3 模型 = 18/21 行**（含 §4.4.1 判定）——**Electricity-336 的 3 行未跑**
> （该 setting 单格实测 63.6 min，按指示停止），表内以 `未跑（按指示停止）` **逐格标注**；
> **§4.5 未跑**（E17 未执行，7 行 × 5 格全部逐格标注）；**§4.6 未跑**（E18 未执行，第五列"本文补做"逐行标注，
> **前四列的既有结果保持原文不动**）；**§4.7 已填**（28 setting 训练集统计量 + 冻结 `ν*` + 两列 ρ）。
> **凡标注 `未跑` 的格，是"没有数据"，不是"留白待补"**——不得据此推断任何结论；
> **已填格不得因后续实验而回改**，新数字按各自表注追加。

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
- **复用候选的拒绝，以及一个**污染批次**的排除（2026-09-20 补；实测自 `stage_a_reuse_audit.json` /
  `reuse_ambiguity.json`）**：复用审计在 81 个入选格之外**拒绝了 13 个候选**（分属 **6 个格子**），
  拒绝理由实测为：**7 个候选没有 test 指标**（既未内联记录、也不在已登记的外部证据表中）、
  **6 个候选 `learning_rate=0.5`**（远超审计网格的 `(0, 1e-2]`）、**6 个候选 `gate_init` 写成 2022 或 2023**
  ——即"**把 seed 写进了 gate init**"。`gate_init` 经 logit 进入模型，合法值必在开区间 `(0, 1)`。
  这批 run 的**结构字段**（mechanism / head / lookback / loss / epochs / percent / period）与合法 run
  **逐一相同**，因此**结构等价性检查抓不住它们**；而它们训练得明显不同
  （`val_mse` ≈ 0.29，合法 run ≈ 0.204）——**若不排除，会污染该 setting 的三 seed 均值**。
  修复后重出的 manifest 为现行版本，被排除的那一版归档在同目录
  （`stage_a_manifest.prelaunch_contaminated.json`），E3 审计对这批 run 的引用数为 **0**。
  这条是**可复现性**所需：读者若按"同一复用根"重放，必须知道其中 13 个候选已被判为非法。

- **超参协议（2026-09-18 冻结）**：新格统一用 preset 默认 `gate_init=0.2`、`lr=1e-3`；复用格保留其 Stage-0
  冻结的 `(gate, lr)`。因此 §4.2 表内存在**三种** gate 先验：新格 0.2、`L-rcrf`/A1 由 preset 自持的 0.5、
  复用格的 Stage-0 冻结值（0.5 或 0.2，逐 setting 不同）。三者必须在表注逐行标注，且配对比较只在
  同一 setting 内成立。
- **符号与量纲约定（2026-09-20 补；供 §4.2 / §4.4 / §4.7 的 `Δ` 列与 `ρ` 列解读，不声明则填完后无法读）**：
  - 凡 `Δ`（§4.2 的 `Δ vs phase_only`、§4.4 的 `Δfused` / `Δbranch`、§4.7 的 `ΔMSE`）一律定义为
    `100 × (候选 − 参照) / 参照` ⇒ **负值 = 候选更好**、正值 = 更差；产出者同时在代码里以 `Δ < 0`
    判定"更好"（`e14_writeback.pct_change`、`e19_predictive_power.py:166` 与 `:175`）。
  - Spearman `ρ` 的符号读法、以及 §4.7"预期符号"列**指哪一列、其 `+` 意味着什么**，见 §4.7 表注 4。
  - §4.4 的 `解释率` 与 `跨 seed leading4 重叠` 是 **0–1 小数**（不是百分数）：解释率的尺度由生产者的
    判据 `input_explanation >= 0.5` 与 `output_explanation >= 0.8` 佐证；`修正能量份额` 同为 0–1。
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

`g` 均值 = 该 setting 3 seed 的 **σ(gate) 均值**；新训格取自单次测试读取，**7 个复用格取自各自 checkpoint**
（口径与 2026-09-20 的修复见 §4.2.1"门列"一段）。`g` 为 **0.05 量级的格已全部更正**：旧值是把
`(臂, horizon)` 上的数据集合并后取最小值所致。

| Dataset | H | Golden MSE/MAE | `phase_only`（matched） | PhaseFormer-L（恒定启用） | Δ vs phase_only | `g` 均值 | 诊断 `s` | 稳定超过 Golden | 来源/披露 |
|---|---:|---:|---|---:|---:|---|---|---|---|
| ETTh1 | 96 | 0.359/0.382 | 0.361/0.387 | 0.369/0.397 | +1.98%/+2.78% | 0.211 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh1 | 192 | 0.397/0.404 | 0.405/0.411 | 0.410/0.421 | +1.21%/+2.35% | 0.213 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh1 | 336 | 0.425/0.424 | 0.442/0.435 | 0.438/0.438 | -0.91%/+0.81% | 0.219 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh1 | 720 | 0.431/0.450 | 0.423/0.442 | 0.421/0.449 | -0.41%/+1.42% | 0.201 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh2 | 96 | 0.275/0.338 | 0.282/0.343 | 0.273/0.333 | -3.06%/-2.92% | 0.492 | 1 | ✗ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_multiseed_stage1_20260914_v4/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| ETTh2 | 192 | 0.341/0.376 | 0.344/0.383 | 0.339/0.377 | -1.37%/-1.43% | 0.198 | 1 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh2 | 336 | 0.369/0.405 | 0.376/0.409 | 0.371/0.405 | -1.50%/-1.14% | 0.200 | 1 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTh2 | 720 | 0.402/0.436 | 0.416/0.449 | 0.392/0.429 | -5.68%/-4.59% | 0.505 | 1 | ✓ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| ETTm1 | 96 | 0.293/0.344 | 0.302/0.351 | 0.306/0.353 | +1.14%/+0.39% | 0.196 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTm1 | 192 | 0.323/0.361 | 0.330/0.363 | 0.338/0.369 | +2.40%/+1.61% | 0.194 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTm1 | 336 | 0.358/0.381 | 0.359/0.381 | 0.369/0.387 | +2.66%/+1.50% | 0.202 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTm1 | 720 | 0.412/0.410 | 0.415/0.413 | 0.417/0.414 | +0.40%/+0.20% | 0.196 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| ETTm2 | 96 | 0.163/0.256 | 0.174/0.265 | 0.159/0.248 | -8.73%/-6.32% | 0.507 | 1 | ✓ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| ETTm2 | 192 | 0.219/0.293 | 0.228/0.300 | 0.215/0.288 | -5.88%/-4.00% | 0.207 | 1 | ✓ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| ETTm2 | 336 | 0.269/0.326 | 0.276/0.331 | 0.267/0.324 | -3.12%/-2.29% | 0.205 | 1 | ✓ | phase_only=新训 3/3；L=新训 3/3 |
| ETTm2 | 720 | 0.351/0.379 | 0.352/0.380 | 0.347/0.376 | -1.26%/-0.93% | 0.203 | 1 | ✓ | phase_only=新训 3/3；L=新训 3/3 |
| Weather | 96 | 0.148/0.195 | 0.150/0.197 | 0.147/0.194 | -2.42%/-1.45% | 0.225 | 1 | ✓ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| Weather | 192 | 0.193/0.237 | 0.195/0.240 | 0.192/0.237 | -1.55%/-1.28% | 0.421 | 1 | ✗ | phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| Weather | 336 | 0.242/0.278 | 0.246/0.280 | 0.241/0.275 | -1.83%/-1.88% | 0.173 | 1 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| Weather | 720 | 0.309/0.332 | 0.316/0.332 | 0.314/0.328 | -0.52%/-1.36% | 0.145 | 1 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| Electricity | 96 | 0.129/0.221 | 0.130/0.223 | 0.129/0.223 | -0.96%/-0.03% | 0.197 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| Electricity | 192 | 0.148/0.238 | 0.146/0.236 | 0.146/0.237 | -0.43%/+0.29% | 0.183 | 0 | ✓ | phase_only=新训 3/3；L=新训 3/3 |
| Electricity | 336 | 0.165/0.257 | 0.167/0.260 | 0.162/0.256 | -2.88%/-1.45% | 0.334 | 0 | ✓ | phase_only=新训 3/3；L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_repair_v1/rank_sweep_2_multiseed_stage1_20260914_v3/rank_sweep_2_stage1）；L 属 test-selected 集合 |
| Electricity | 720 | 0.201/0.285 | 0.199/0.285 | 0.197/0.285 | -1.15%/+0.08% | 0.182 | 0 | ✗ | phase_only=新训 3/3；L=新训 3/3 |
| Traffic | 96 | 0.361/0.238 | 0.363/0.232 | 0.367/0.237 | +1.18%/+2.51% | 0.057 | 0 | ✗ | 探索性附录，不进入判定；phase_only=新训 3/3；L=新训 3/3 |
| Traffic | 192 | 0.373/0.243 | 0.380/0.242 | 0.385/0.246 | +1.38%/+1.75% | 0.070 | 0 | ✗ | 探索性附录，不进入判定；phase_only=新训 3/3；L=新训 3/3 |
| Traffic | 336 | 0.385/0.248 | 0.393/0.250 | 0.400/0.254 | +1.68%/+1.64% | 0.053 | 0 | ✗ | 探索性附录，不进入判定；phase_only=新训 3/3；L=新训 3/3 |
| Traffic | 720 | 0.428/0.270 | 0.433/0.270 | 0.441/0.278 | +1.75%/+2.73% | 0.066 | 0 | ✗ | 探索性附录，不进入判定；phase_only=新训 3/3；L=新训 3/3 |

**变体行（同协议、同 seed，全部为既有 preset）**：

| 行 | preset / 配置 | 作用 | 新训规模 |
|---|---|---|---|
| `phase_only` | `original`（matched rerun） | 配对基线 | 24 − 6 复用 = 18 setting × 3；Traffic 4 × 3 = 12 |
| PhaseFormer-L | `weak_residual`，`shared` 头，修正器**恒定启用**（D-5） | 主张 A/B | 24 − 7 复用 = 17 setting × 3；Traffic 4 × 3 = 12 |
| L-q1/4、L-q1/8 | `pooled_lowrank`，`rank=H/4`、`H/8` | 主张 D（效率） | 各 24 − 7 复用 = 17 × 3；Traffic 各 4 × 3 = 12 |
| L-rcrf | `rcrf_nlinear_plain` | 门的消融 | 28 × 3 = 84（无复用格） |
| A1 | `gold_combo_reliability_s2` | incumbent 参照 | 24 × 3 = 72（**无复用格**，见下） |

**复用格的**候选歧义**审计（2026-09-20 补，`reuse_ambiguity.json` 实测）**：81 个复用格中
**45 格存在多个合法候选 run**——若这些候选在超参或指标上不一致，"选哪一个"就会影响 §4.2 的数字。
实测结论是**不致影响**：**候选之间存在分歧的格子数为 0**，且**最大 `val_mse` 跨候选差为 0.0**
（即 45 格的多候选在 gate init、learning rate 与 `val_mse` 上完全一致，选谁都得同一个数）；
另有 **6 个格子**含被拒的候选（详见下条"候选拒绝"）、
**0 格的入选 run 落在扫描范围之外**。故 §4.2 的复用数字**不依赖任何 tie-break 规则**——
这条正是该审计自身声明的披露条件（"只有候选间存在分歧的格子才是真正的选择，且必须在 §4.2 表注中披露"），
实测满足"没有这类格子"。

> **A1 行的勘误**：本文早期草案写"A1 有 12 格既有"。2026-09-18 审计在**本地与服务器两侧**都未找到
> 任何 `gold_combo*` 产物，且 `docs/agent-log.md` 记录的那批 A1 运行使用 **MAE loss、batch 256、lr 3e-4**，
> 而 §4.0 要求 **Huber**——即使找回也不满足"同协议"。因此 A1 **不存在可复用格**，按 §4.0 协议全训
> 24 × 3 = 72 runs。A1 是 incumbent 参照行，不进入主张 A–D。

参数量列按仓库 `metrics.csv` 的 `parameter_count` 口径报告（主干 + 修正器 + 门），并单列修正器参数；
FLOPs 不在本文口径内比较（原文 Table 4 口径未在本仓库复现）。

**必答问题**：(a) ETTh2 四个 horizon 相对 `phase_only` 与 Golden 的差距是否收窄、是否达到 FITS 的引用数字；
(b) 逐 dataset 报告门值 `g` 的均值；诊断列 `s` 是否把 ETTh1/ETTm1 判为 `s=0`，若判为 `s=1` 则如实报告其表现；
(c) q=1/8 与 direct 的三 seed 差是否在 ±0.5% 内。

**门值列的两个来源，以及为什么不能跨臂比（2026-09-20 补）**：`g` 是**可训练的每格参数**（非固定超参），
本表报该 setting **3 seed 的 σ(g) 均值**。来源有两处——新训格取单次测试读取（`results.csv` 的 `gate_value`）、
复用格 7 格取各自 checkpoint——**逐格**在 `main_table.csv` 的 `l_main_gate_source` 列标明。
**L-rcrf 与 A1 的 `results.csv` 也有 `gate_value`，但那是另一个量**：这两个臂的模型**没有**
`weak_period_residual_gate` 参数（`parameter_table.csv` 的 `gate_param_present = False`，实测 84/84 与 72/72 行），
因此 `evaluate_once` 走到的是 `last_rcrf_alpha` 这条回退分支（`e14_read_test.py:1052-1056`），
得到的是**逐样本融合权重的均值**，不是静态门 ⇒ **不得与本表的 `g` 列横向比较**（这也是 §4.2.1 只对
`l_main` 做门值读数、不对 `l_rcrf`/A1 做对比的原因）。

**参数量列的口径（2026-09-20 补）**：`total_params_per_horizon` 现附 `total_params_reference_dataset`
（引用数据集）与 `total_params_per_horizon_range`（全数据集跨度），因相位主干随通道数变化；
`residual_params_per_horizon` 与数据集无关（实测跨数据集零差异），`params_constant_across_seeds`
的实测值为 **True**（84 个 (臂,数据集,horizon) 格在 3 seed 间完全不变）。

#### 4.2.1 实测判定（2026-09-20；数字取自本表，判定取自 `claims.json`，逐格由 `verify_minipaper_fill.py` 独立复核）

**必答 (a)：ETTh2 四个 horizon 全部收窄了相对 `phase_only` 的差距，但没有四档都"稳定超过 Golden"。**
Δ vs `phase_only` = **−3.06 / −1.37 / −1.50 / −5.68%**（四个 horizon 的 MSE 均为负，MAE 同向），
即 **4/4 收窄**；然而"稳定超过 Golden"列只有 **ETTh2-720 为 ✓**，其余三档为 ✗。
**这两件事不要混读**：收窄的是"相对配对基线"，没达标的是"相对 Golden 的三 seed 稳定超越"。

**必答 (b)：诊断列 `s` 恰好标出"修正器有增益"的那 12 个 setting，但它是单向判据。**
`s=1` 与 {ETTh2, ETTm2, Weather} × 4 horizon **完全重合**，而**这 12 格的 Δ 全部为负**（−0.52% ～ −8.73%）
⇒ **`s=1` ⇒ 有增益，12/12，精确率 100%**。**反向不成立**：`s=0` 的格里仍有 **6 格** Δ<0
（ETTh1-336、ETTh1-720 与 Electricity 四档，见 `claims.json.B.diagnostic_misses`）
⇒ `s` 是**高精确率、低召回**的诊断列，**不是双向判据**，§5 的限制里按此表述。

**门列（`g`）的口径与一处已修缺陷。** 本表 `g` 列优先取单次测试读取记录在案的值（`results.csv`），
但 **7 个复用 setting 的 Stage-0 证据里没有门值**（那批运行只写 MSE/MAE），故这 7 格改由**各自 checkpoint** 读取。
2026-09-20 复核发现该回退路径有缺陷并已修复：它按 `(臂, horizon)` 取值、**把数据集合并掉了**，于是这 7 格
拿到的是**该 horizon 上所有数据集门值的最小值**（即 Traffic 的门），而非本格自己的门。实测对照（checkpoint 直读为仲裁）：

| setting | 修复前（错） | 修复后（本表） | checkpoint 直读（仲裁） |
|---|---|---|---|
| ETTh2-96 | 0.052 | **0.492** | 0.492376 |
| ETTh2-720 | 0.047 | **0.505** | 0.505187 |
| ETTm2-96 | 0.052 | **0.507** | 0.507025 |
| ETTm2-192 | 0.055 | **0.207** | 0.207138 |
| Weather-96 | 0.052 | **0.225** | 0.224901 |
| Weather-192 | 0.055 | **0.421** | 0.420847 |
| Electricity-336 | 0.042 | **0.334** | 0.333684 |

另外两个同源缺陷一并修复：`total_params_per_horizon` 同样把数据集合并掉了（相位主干随通道数变化：
h192 在 7 通道数据集上 140 191，Electricity 411 913，Traffic 412 454），现已**注明所引用的数据集并同时给出全跨度**；
`params_constant_across_seeds` 因合并而错报 `False`，实测**所有 84 个 (臂,数据集,horizon) 格的参数量在 3 seed 间完全不变**。
`l_q1_4`/`l_q1_8` 的同 7 格也被同一缺陷命中（旧值 0.072–0.093 / 0.079–0.115），修复后与 checkpoint 逐格一致。
**全表 84 个有门格**（3 臂 × 28 setting）已逐格与 checkpoint 直读比对，偏差 0 格、最大相对差 < 1e-6。

**门值列的正确读法（替换原先"门接近 0 是修正器在干活"的读数）。** 修复后**main-24** 的逐 dataset `g` 均值为
ETTh1 0.211、ETTm1 0.197、ETTh2 0.349、ETTm2 0.281、Weather 0.241、Electricity 0.224
（Traffic 附录四格 0.062，不并入 main 口径）；
按"`delta_mse_pct` 是否为负"分组，**有增益的 18 格均值 0.2665，退化的 6 格均值 0.2019**——
即**有增益的格子门值反而偏高**，与原读数相反（原读数在**同一 main-24 口径**下是
**0.1367 vs 0.2019**，看似"有帮助的门更小"）。
**口径须写明**：若把 Traffic 附录四格计入（**all-28**，它们四格全部退化、g≈0.053–0.070），
退化的组变成 10 格、均值 0.1458，于是"有增益的门更高"这一对比**缩到近乎消失**
（all-28 下为 0.2665 vs 0.1458；修复前 all-28 为 0.1367 vs 0.1458）——
所以上面那组数字是 **main-24 口径**，不能当作全表口径。
> **一处易混的巧合（必须点明）**：修复前后 **all-28 的退化组均值都是 0.1458**——因为该组 10 格
> （main-24 的 6 格退化 + Traffic 4 格）**全是 `results.csv` 有门值的格子，门列修复一格未动它们**，
> 改的只有"有增益"那一组（0.1367→0.2665）。**两个 0.1458 同值但含义不同**：一个是**缺陷列**下的
> all-28 退化均值，另一个是**修复后**的 all-28 退化均值；引用时须连同"修复前/后"与口径一起写。
方向上这也自洽：`y = (1−σ(g))·y_phase + σ(g)·y_residual`，**门越大越走残差支路**，而残差支路正是修正器本身。
故此处**只报数字、不作因果解释**：同一 horizon 上各数据集的门值差异（main-24 内 0.145–0.505，
全 28 格 0.053–0.507）首先反映的是
**各格自身训练轨迹**（门是**可训练**参数，非冻结超参：新训格平均移动 17.7%、复用格 9.9%，见 §5 限制），
而非某个单一机制。

`s` 对 **ETTh1/ETTm1 判为 `s=0`**（与 §4.0 预登记一致），而这两组**实测确实变差**（见下），**如实报告**。

**必答 (c)：不在 ±0.5% 内，故主张 D 判定为 `false`。**
该主张的口径是**逐 setting 的 |L-q1/8 − L| 的平均值**（`e14_writeback.py:585-597`），
实测 **|ΔMSE| 平均 0.7183%、|ΔMAE| 平均 0.5343%**，界为 0.5% ⇒ **"rank=H/8 已足够"这一效率主张未达预登记界**。
**但它不否定参数量结论**：L-q1/8 的总参数确实只有 direct 的约 **17%**
（h192：**23 863 vs 140 191**，两者同取 **ETTh2 这一引用数据集**；变体表 `total_params_per_horizon`
现标注 `total_params_reference_dataset` 并附全跨度 `total_params_per_horizon_range`。
**该比值随 horizon 与数据集变化**：main-24 上 H=96 为 16.4%–20.8%、H=192 为 17.0%–19.0%、
H=336 为 19.1%–19.8%、H=720 为 25.2%–33.0%（宏平均 23.8%，H=720 因 rank=H/8 相对 dense 的 r=H
压缩比下降而变差），故 **17% 是 H=192 这一档的读数，不是全表常数**），
且它的**宏平均 ΔMSE 还略好于** direct（**−1.31% vs −1.00%**）——
即"低秩能省参数"成立，"**逐格都能压在 0.5% 以内**"不成立。

| 主张 | 预登记界 | 实测 | 判定 |
|---|---|---|---|
| **A** 24 setting 上相对 `phase_only` 的退化 ≤ 1% | 1.0% | 任一指标越界：**ETTh1-96/192/720、ETTm1-96/192/336**；双指标越界：ETTh1-96/192、ETTm1-192/336 | **✗ 不成立** |
| **B** {ETTh2, ETTm2, Weather} 的 12 个 setting 上**全部**变好 | ≥ 9/12 | **12/12 全部变好** | **✓ 成立** |
| **C** 相对 Golden 的"稳定超过"与双指标改善（描述性） | — | L **8** 个 setting 稳定超过 Golden（`phase_only` 仅 **2**）；双指标改善 L **12** vs `phase_only` **3** | 见文 |
| **D** 逐 setting \|L-q1/8 − L\| 的平均 ≤ 0.5% | 0.5% | \|ΔMSE\| **0.7183%**、\|ΔMAE\| **0.5343%** | **✗ 不成立** |

**本节的一句话结论（也是摘要那句话的出处）**：**PhaseFormer-L 不是普适增益**——
它在**电平非平稳**的 {ETTh2, ETTm2, Weather} 上 **12/12 稳定变好**（最多 **−8.73%** MSE），
在 **ETTh1/ETTm1 上反而变差**（最多 **+2.66%**），而**"哪里有用"可以由训练集上的电平记忆长度 `τ̂` 事前预测**
（§4.7：ρ(`τ̂`, ΔMSE) = **−0.750**）——这正是命题 1 所要的形态。


#### 4.2.2 定向调参搜索：8 个未达标 setting 的 test-set selection 搜索（2026-09-20 启动，三轮于 2026-09-21 收官）

> 结果已于 2026-09-21 回填完毕（三轮收官判定见本节末尾）；预注册口径与上界保留如下。

**动机与指令**：§4.2 的 24 格中，8 格未能双指标超 Golden。按用户 2026-09-20 明示指令，
对这 8 格做**以 test 指标为选择依据**的定向搜索（`docs/PhaseFormer_L_golden_search_plan.md`），
目标 ≥4/8 双指标超 Golden。**全部数字为 test-set selection**，不得表述为盲测或无偏泛化估计。

**搜索空间（冻结）**：`gate_init ∈ {0.02,0.05,0.1,0.2,0.35,0.5}` × `lr ∈ {1e-4,3e-4,1e-3}`
× head ∈ {dense, `pooled_lowrank` r=H/32,H/16,H/8,H/4}，第一轮 720 runs（seed 2021）；
第二轮按实测收窄后加 `mae` 损失与 `lr=3e-3`，边际 856 runs。

**预注册上界（写于第一轮完成之前，见计划 §1b）**：门压到近零时模型**就是** matched `phase_only`，
故 `phase_only` 的逐 seed 最好成绩是"门压小"类获胜格的天花板。用 E14 已有 3 seed（各指标各取最好 seed）
对照 Golden：**8 格中 0 格双指标胜出**；全 24 主 setting 上 `phase_only` **21/24 双指标落后 Golden**。
⇒ **"≥4/8" 在本次环境下结构性不可达；乐观上界为 1/8。** 该上界不改变搜索的执行，只约束结果的读法。

**已完成部分的实测（两个 setting，共 135 格；截至 2026-09-20 22:17）**

| setting | 已完成格 | 最优 MSE vs Golden | 最优 MAE vs Golden | 最优 vs `phase_only` | 双指标超 Golden |
|---|---:|---:|---:|---:|:--:|
| Electricity-96 | 90/90 | **−0.308%**（过） | **+0.079%**（未过） | −1.408% / −0.717% | **0/90** |
| ETTm1-720 | 45/90 | **−0.117%**（过） | **+0.279%**（未过） | −0.856% / −0.396% | **0/45** |

- **两个 setting 独立复现同一形态**：修正器相对 `phase_only` **两指标皆改善**，
  但相对 Golden **卡在 MAE 侧**（差 0.08% / 0.28%）。⇒ 该 MAE 缺口**跨 setting 共享且非修正器所致**，
  与 §1b 的"环境差"判断一致，并给出其方向。
- **MAE 口径已排除**：`metrics.MAE` 为普通 `mean|pred−true|`（无标准化/通道加权/mask），
  `test_step` 仅按 `pred_len` 截断且 `target_var_index = −1`（全通道）⇒ 差距是数值上的真实差距。
- **最优解倾向关闭修正器**：ETTm1-720 的前 10 名中 **6 格为 `gate=0.02`**（相位权重 98%），
  与该格上 `l_main` 相对 `phase_only` 为**负贡献**（+0.74%）一致 ⇒ 在这些 setting 上，
  "让修正器靠边站"确实是更好的选择，这正是 §1b 所指的上界路径。

**第一轮全量结果与判定（2026-09-21 03:00:57；720 格全部完成、0 失败，墙钟 ≈9.6 h）**

判定 = 逐 seed 取最优（Q6）、3 seed 全到齐（`final_selection.json`）：

| setting | 选中配置 | 最优 seed | vs Golden MSE | vs Golden MAE | gate-shrunk |
|---|---|---:|---:|---:|:--:|
| ETTh1-96 | g0.02 lr1e-3 r3 | 2022 | **−0.260%** ✓ | +1.411% ✗ | 是 |
| ETTh1-192 | g0.02 lr1e-3 r6 | 2023 | +0.771% | +2.267% | 是 |
| ETTh1-336 | g0.35 lr1e-3 r21 | 2022 | +0.917% | +1.989% | 否 |
| ETTm1-96 | g0.5 lr1e-3 r24 | 2021 | +1.942% | +1.441% | 否 |
| ETTm1-192 | g0.05 lr3e-4 r12 | 2021 | +2.449% | +1.804% | 是 |
| ETTm1-336 | g0.2 lr3e-4 r10 | 2021 | +0.329% | +0.548% | 否 |
| ETTm1-720 | g0.02 lr3e-4 r45 | 2023 | **−0.327%** ✓ | +0.499% ✗ | 是 |
| Electricity-96 | g0.2 lr1e-3 r12 | 2021 | **−0.308%** ✓ | +0.079% ✗ | 否 |

**结果：0/8 双指标超 Golden（目标 ≥4/8 未达）；4/8 双指标优于 matched `phase_only`。**

**这一夜搜索给出的五条结构性读数（比"未达标"本身更有信息量）**：

1. **8/8 全部败在 MAE 侧**；其中 **3/8 的 MSE 侧已胜 Golden**（ETTh1-96 −0.260%、ETTm1-720 −0.327%、
   Electricity-96 −0.308%）⇒ **瓶颈单侧集中在 MAE**，这是八格一致的事实而非单点。
2. **修正器的增益集中在 MSE 侧、MAE 侧中性到略负**，四处独立复现：
   ETTm1-192（0/71 格能追平 `phase_only`）、ETTh1-336（MSE **48/90** 格胜、MAE **0/90**）、
   ETTh1-192（MSE **14/87**、MAE **0/87**）、Electricity-96（MAE 缺口只补到 **90%**，MSE 补到 126%）。
   **与 §4.4 解剖不矛盾**：主模式写"整段恒定位移"，常数平移适合平方误差（压缩大偏差），
   不必然降低每点绝对偏差。
3. **符号翻转发生在同一数据集内部**：ETTm1 的四档给出三种结果（96/720 双改善、336 仅害 MAE、192 双损害）
   ⇒ **"条件性"不等于"数据集差异"，同一数据集仅 horizon 不同即可翻转**。
4. **换 seed 能翻转 MSE 侧、不能翻转 MAE 侧**：ETTh1-96 从 s2021 的 +0.284% 变为 s2022 的 −0.260%，
   而 MAE 仍差 1.4%；ETTm1-720 的 MAE 缺口在 s2023 反而从 +0.279% 变为 +0.499%。
5. **8 个 winner 全部是 `pooled_lowrank`（秩 R3–R45），无一稠密头** ⇒ 在这些 setting 上
   **低秩约束本身是净有利的**；**gate-shrunk 占 4/8**，其中两格的 MSE 胜出来自"门近乎关闭"，
   按 Q7 的规定**其胜利属于相位主干、不属于电平通道**。

**第二轮（已由守护按预注册自动启动，2026-09-21 03:01）**：`loss=mae` × lr{3e-4,1e-3,3e-3} × gate × head，
边际 856 新增 runs ≈6.8 h。其靶子是上表第 2 条——**若 MAE 目标训练出的修正器能在 MAE 侧转为正贡献，
则本表中"只差 MAE"的三格（Electricity-96 +0.079%、ETTm1-720 +0.499%、ETTm1-336 +0.548%）有翻盘空间**；
若不能，则"修正器天然偏 MSE"成为一条更强的机制结论。含 `gate=0.02+mae`（= 用 MAE 目标优化的 `phase_only`，
此前从未测量过）这一唯一未测方向。

**第二轮（`loss=mae` + lr 上移；2026-09-21 03:00 起由守护自动执行）**

第二轮的靶子是上表第 2 条：第一轮已证"修正器增益偏 MSE、MAE 侧普遍差最后 0.08%–0.5%"。
若 MAE 目标训练出的修正器能在 MAE 侧转为正贡献，则"只差 MAE"的几格有翻盘空间。
第二轮同时覆盖了 **`gate=0.02 + loss=mae`**——即"用 MAE 目标优化的 `phase_only`"，
这一组合在既有全部证据（含 E14 的 24 格 matched `phase_only` 均为 huber/mse）中**从未被测量**。

**stage-1（seed 2021）结果：达标 setting 从 0/8 升至 5/8**

| setting | 达标配置数 | 最好达标配置 | 最好 MSE / MAE vs Golden | 其中非 gate-shrunk |
|---|---:|---|---|---:|
| **ETTm1-336** | **19** | g0.2 lr3e-4 r10 | **−0.941% / −1.230%** | 17 |
| **ETTm1-720** | **13** | g0.1 lr3e-4 r45 | −0.494% / −0.714% | 8 |
| **ETTm1-96** | **8** | g0.1 lr1e-3 r12 | **−0.980% / −1.820%** | 6 |
| **Electricity-96** | **3** | g0.05 lr3e-3 dense | −0.395% / −1.687% | 1 |
| **ETTh1-96** | **2** | g0.02 lr1e-3 r3 | −0.626% / −0.066% | 1 |
| ETTm1-192 | 0 | — | 全网格 MSE 缺口 >0.769% | — |
| ETTh1-336 | 0 | — | MAE 0/90（缺口最小 +0.962%） | — |
| ETTh1-192 | 0 | — | MAE 0/90（MSE 却达 −2.480%） | — |

**第二轮给出的三条机制读数**（均为 stage-1 单 seed，test-set selection）：

1. **换目标函数能把 MAE 侧从"全败"翻成"多数胜"**：同一 setting 同协议下，
   ETTm1-720 的 MAE 胜出从 **0/90 → 60/90**、ETTm1-336 从 **0/90 → 88/90**、
   ETTh1-336 从 0/90 → **0/90**（不动）⇒ **该效应本身也是数据集条件性的**。
2. **同 horizon、不同数据集的决定性对照**：ETTm1-336 **88/90** vs ETTh1-336 **0/90**——
   只换数据集，MAE 响应从全胜变全败。⇒ 论文 §4.2 的"条件性"结论可再推进一层：
   **不仅"修正器是否有用"是条件性的，"用什么目标训练它"的收益也是条件性的。**
3. **达标不依赖关掉修正器**：43 套达标配置中 **33 套门值 >0.05**，
   即多数达标格的修正器仍有 10%–50% 权重 ⇒ 可作为**电平通道本身有正贡献**的证据；
   其余 10 套按 Q7 标注为"胜利来自相位主干"。

**最终判定（3 seed，逐 seed 取最优，Q6；2026-09-21 10:00:57）**：

> **5/8 setting 双指标超越 Golden，达到预设目标（≥4/8）。**
> 更强的一条：**8/8 setting 双指标超越 matched `phase_only`**。
> **5 个获胜格全部不是 gate-shrunk** ⇒ 胜利不来自"关掉修正器"（Q7 要求）。

| setting | 选中配置 | vs Golden MSE | MAE | vs `phase_only` MSE | MAE | 最优 seed | 3 seed 双指标胜 Golden |
|---|---|---:|---:|---:|---:|---:|---:|
| **ETTh1-96** | g0.2 lr3e-3 **mae** r24 | **−3.330%** | **−0.399%** | −3.973% | −1.606% | 2021 | 1/3 |
| **Electricity-96** | g0.2 lr3e-3 **huber** r24 | **−1.166%** | **−0.596%** | −2.257% | −1.386% | 2021 | **3/3** |
| **ETTm1-96** | g0.1 lr1e-3 **mae** r12 | −0.980% | **−1.820%** | −4.071% | −3.824% | 2021 | **3/3** |
| **ETTm1-336** | g0.2 lr3e-4 **mae** r10 | −0.941% | **−1.230%** | −1.300% | −1.271% | 2021 | **3/3** |
| **ETTm1-720** | g0.1 lr3e-4 **mae** r45 | −0.494% | −0.714% | −1.230% | −1.381% | 2021 | 1/3 |
| ETTh1-192 | g0.02 lr1e-3 mae r6 | **−2.480%** | +0.443% | −4.327% | −1.248% | 2021 | 0/3 |
| ETTh1-336 | g0.02 lr3e-3 mae r10 | −0.507% | +0.651% | −4.317% | −1.816% | 2023 | 0/3 |
| ETTm1-192 | g0.2 lr3e-4 mae r6 | +0.758% | +0.563% | −1.504% | −0.069% | 2023 | 0/3 |

**必须同时报告的三点口径**（否则结论会被读得过强）：

1. **3 个 setting 是 3/3 seed 稳定达标**（Electricity-96、ETTm1-96、ETTm1-336），
   **2 个只在 1/3 seed 上达标**（ETTh1-96、ETTm1-720）⇒ "5/8"应表述为
   **"3 格三 seed 稳定 + 2 格单 seed 达标"**，而不是"5 格同样稳固"。
2. **全部 8 格双指标优于 matched `phase_only`（8/8）**——这一条比 vs Golden 更强、也更干净：
   它不含环境差，说明**搜索确实在这 8 格上找到了比"不带修正器"更好的配置**。
3. **5 个获胜格全部 `gate > 0.05`**（0.1 或 0.2），即**修正器仍有实质权重** ⇒
   按 Q7，可以说"**电平通道本身有正贡献**"，无需"胜利来自相位主干"的标注。
   （三个未获胜格中，ETTh1-192 与 ETTh1-336 是 gate-shrunk，ETTm1-192 不是。）
4. **两批互补**：`Electricity-96` 的最优组合来自 `huber` 批（MSE 侧更强 −1.166% vs `mae` 批最好的 −0.395%），
   `ETTh1-96` 同样来自 `huber` 批（−3.330% vs `mae` 批的 −0.626%）⇒ **第二轮的两批都不可省**。

**第三轮（2026-09-21 13:56–16:09，用户指令："在剩下 3 个 setting 上继续搜索，每 setting 120–150 run"）**

靶子是三个仍未达标的 setting。前两轮已覆盖 `loss ∈ {huber, mae}` 两个端点、四档 lr、
六档 gate、五种 head，故第三轮只加**两个端点之间的连续谱与网格外的档位**：
`huber_delta ∈ {0.05,0.1,0.3,3.0}`、`max_epochs=60`、以及边界外 lr（ETTh1→1e-2、ETTm1-192→1e-4）。
每 setting 132 runs，共 396 runs。

| setting | 第三轮结果 | 与第一/二轮对比 |
|---|---|---|
| **ETTh1-336** | **2 套达标**：`g0.02 lr1e-2 r21 delta=0.05/0.1`，最好 **−1.687% / −0.206%** | **0/90 → 2/132 达标** |
| ETTh1-192 | 0 达标；MSE 侧已过（最好 −2.480% 来自第二轮），**MAE 侧 0/132** | 未改善 |
| ETTm1-192 | 0 达标；**MSE 侧最好仍差 0.957%**（MAE 侧 14/132 已过） | 未改善 |

- **决定性轴是 `lr=1e-2`**——第三轮新增的"网格外一档"。ETTh1-336 在第一/二轮 0/90 达标、
  最好 MSE +0.234%；加上这一档后 MSE 直接到 **−1.84%**。这验证了第三轮的设计判断
  （两格 winner 都压在 lr 网格边界上 ⇒ 甜点确实在网格外）。
- **`delta` 小档并不等价于 `loss=mae`**：归一化残差多落在 `delta` 内时，`HuberLoss(delta=0.05)`
  在线性段工作、梯度被压得很小 ⇒ 近似欠训练。实测在 ETTh1-336 上小 `delta` 让 MAE 更差
  （+1.113% vs 第二轮 +0.962%），在 ETTm1-192 上则略有帮助。**如实记录该轴未达设计意图。**
- **`ETTm1-192` 判为达不成**：三轮共 372 个 run，MSE 侧从未低于 +0.957%，
  与该格 `phase_only` 的 MSE 缺口 +1.784% 一致 ⇒ 结构性的，不是调参空间未覆盖。

**第三轮的达标计数：`ETTh1-336` 计入后为 6/8**，其中 1/3 seed 达标
（seed 2021 −0.507%/+0.651% 未过 MAE；seed 2023 +0.223%/+2.076%）。与 ETTh1-96、ETTm1-720 同属
"单 seed 达标"一类。**故最终表述为：6/8 达标，其中 3 格三 seed 稳定（Electricity-96、ETTm1-96、ETTm1-336）
+ 3 格单 seed（ETTh1-96、ETTh1-336、ETTm1-720）。**

**全部数字为 test-set selection**（以 test 指标选择组合，用户 2026-09-20 明示指令），
且 Golden 来自另一套硬件环境，只作披露性比较。第三轮使用 `max_epochs=60` 的格**与 E14 的 30-epoch
协议不可比**（本轮 396 格中含 60-epoch 档，须在引用时注明）。

#### 4.2.3 主表全 setting 逐臂最优 seed 汇总（2026-09-21 回填）

> **数据来源**：`docs/PhaseFormer_L_main_table_repro.md`（同日合并为唯一的复现手册，提交 `a05f7cd`；
> §1 的 164 行由生成器从**每格自己的 `config.json` 与同一 run 的 test 记录**读出，
> 行级校验器与命令校验器各报 `PROBLEMS: 0`）。口径与 §4.2.2 一致：**以 test 指标在 3 个 seed 中
> 取最优的一次**（按"两指标最差缺口"），全部为 test-set selection。本节把这些行**汇拢成可引用的
> 逐臂/逐数据集统计**；逐格的 gate/lr/head/rank、test 指标与 164 条复现命令一律查手册原文。

**（i）逐臂汇总（main-24；"达标格数"= 该臂 3 个 seed 中至少 1 个双指标胜 Golden 的 setting 数）**

| 臂 | 宏平均 vs Golden（MSE/MAE） | 最优 seed 双指标胜 Golden | 备注 |
|---|---:|---:|---|
| `phase_only` | +1.08% / +0.72% | **3/24** | matched 基线；全表唯一无残差支路的臂 |
| `l_main`（PhaseFormer-L，E14 固定协议） | **−0.13%** / +0.04% | **14/24** | 双指标计数最多的臂 |
| `l_q1_4` | −0.10% / +0.10% | 11/24 | rank=H/4 |
| `l_q1_8` | **−0.31%** / +0.11% | 13/24 | 宏平均 MSE 最好的臂 |
| `l_rcrf` | +0.27% / +0.21% | 10/24 | 可靠度门消融 |
| `a1`（incumbent） | +0.07% / +0.22% | 10/24 | 参照，不进主张 A–D |

同表条件下四个带修正器的臂全部把 `phase_only` 的 3/24 抬到 **10–14/24**；其中 `l_main` 与
`l_q1_8` 把宏平均 MSE 从 +1.08% 压到 −0.13% / −0.31%——**修正器对 Golden 的追赶是结构性的、
不是某个 setting 的单点**。四臂的 MAE 宏平均仍在 +0.04%～+0.21% ⇒ 与 §4.2.2 第一轮的读数一致：
**MAE 侧是共同的环境残差，不是某个臂的缺陷**。

> **把 14 拆开看，它与 §4.2 的"8 格稳定"并不矛盾**：`l_main` 的 **14 = 8 格 3/3 seed 全部达标**
> （ETTh2-720、ETTm2 四档、Weather-96、Electricity-192/336——**恰为 §4.2 表"稳定超过 Golden"列的
> 8 格**）+ **6 格 1–2/3 seed 部分达标**（ETTh1-720、ETTh2-96、ETTh2-336、Weather-192、Weather-336、
> Electricity-720）。两个数字是同一组证据的两种强度口径：**8 = "稳定赢"，14 = "存在一个能赢的 seed"**。
> 另外，低秩变体 `l_q1_4`/`l_q1_8` **不新增任何 setting**（其胜格全被 `l_main` 覆盖，合并仍 14/24）；
> 只有把门消融臂 `l_rcrf`/incumbent `a1` 也算上才到 **15/24**（多 ETTh2-192 一格，且仅 1/3 seed）。

**（ii）逐 setting：E14 协议下的最优臂（最优 seed 的最差缺口）**

| setting | 最优臂 | seed | vs Golden | 3-seed 双指标胜 | §4.2.2 调参后 | setting | 最优臂 | seed | vs Golden | 3-seed 双指标胜 | §4.2.2 调参后 |
|---|---|---:|---|---:|:--:|---|---|---:|---|---:|:--:|
| ETTh1-96 | `phase_only` | 2021 | +0.51%/+1.10% | 0/3 | ✓1/3 | ETTm2-96 | `l_main` | 2021 | −2.78%/−3.11% | 3/3 | — |
| ETTh1-192 | `phase_only` | 2021 | +1.77%/+1.31% | 0/3 | ✗ | ETTm2-192 | `a1` | 2022 | −2.01%/−2.31% | 3/3 | — |
| ETTh1-336 | `l_q1_4` | 2023 | +1.98%/+2.30% | 0/3 | ✓1/3 | ETTm2-336 | `l_main` | 2023 | −1.02%/−0.83% | 3/3 | — |
| ETTh1-720 | `phase_only` | 2023 | −2.84%/−2.38% | 3/3 | — | ETTm2-720 | `a1` | 2023 | −2.67%/−0.91% | 3/3 | — |
| ETTh2-96 | `a1` | 2023 | −1.53%/−1.60% | 3/3 | — | Weather-96 | `l_q1_8` | 2022 | −1.02%/−0.79% | 2/3 | — |
| ETTh2-192 | `a1` | 2022 | −0.48%/−0.49% | 1/3 | — | Weather-192 | `l_q1_8` | 2022 | −0.91%/−0.96% | 3/3 | — |
| ETTh2-336 | `l_q1_8` | 2023 | −0.84%/−0.32% | 2/3 | — | Weather-336 | `l_main` | 2021 | −0.87%/−1.52% | 2/3 | — |
| ETTh2-720 | `l_q1_8` | 2023 | −3.63%/−2.38% | 3/3 | — | Weather-720 | `l_q1_8` | 2022 | +0.55%/−1.75% | 0/3 | — |
| ETTm1-96 | `phase_only` | 2022 | +1.17%/+0.51% | 0/3 | ✓3/3 | Elec-96 | `l_q1_8` | 2021 | −0.31%/+0.08% | 0/3 | ✓3/3 |
| ETTm1-192 | `phase_only` | 2023 | +1.78%/+0.59% | 0/3 | ✗ | Elec-192 | `phase_only` | 2023 | −1.79%/−1.27% | 3/3 | — |
| ETTm1-336 | `phase_only` | 2021 | +0.14%/+0.08% | 0/3 | ✓3/3 | Elec-336 | `l_main` | 2021 | −1.98%/−0.89% | 3/3 | — |
| ETTm1-720 | `phase_only` | 2022 | +0.49%/+0.44% | 0/3 | ✓1/3 | Elec-720 | `l_q1_4` | 2023 | −2.43%/−0.41% | 2/3 | — |

"§4.2.2 调参后"列 = 定向搜索在同一 setting 达到的判定（✓n/3 = 双指标胜 Golden 的 seed 数，✗ = 三轮未达标）；
"—" = 该 setting 不在 8 格搜索范围内（E14 协议即为其最终数字）。

**逐格读数**：24 个 setting 中 **15/24** 在 E14 固定协议下已至少单 seed 双指标胜 Golden
（ETTh1-720、ETTh2 四档、ETTm2 四档、Weather-96/192/336、Electricity-192/336/720）；
未胜的 **9/24** 中 **8 格正是 §4.2.2 的搜索对象**（ETTh1-96/192/336、ETTm1-96/192/336/720、
Electricity-96），另 1 格是 Weather-720。两件事接起来读：**固定协议 + 定向调参合计，
24 个 setting 中 21 个至少单 seed 双指标胜 Golden**（调参补回 ETTh1-96/336、ETTm1-96/336/720、
Electricity-96 共 6 格）。**仍未胜的 3 格**：ETTh1-192（MAE 侧三轮 0/132，最好 +0.44%）与
ETTm1-192（MSE 底线 +0.76%，三轮 372 run 未破）已被判**结构性不可达**；Weather-720 不在
搜索范围内，最优 `l_q1_8` 的 MAE 侧已过线（−1.75%）、**MSE 侧仍 +0.55%**——它是 s=1 家族里
唯一未胜的一格：修正器相对 `phase_only` 仍有增益（`l_q1_8` 最优 seed −1.41%/−1.77%），
属于"有增益但补不上 MSE 侧环境差"的情形。

**（iii）`l_main` vs `phase_only` 的配对（各自最优 seed 的 MSE 侧）**：**17/24 L 更好**、7/24
`phase_only` 更好；`phase_only` 赢的 7 格是 ETTh1-96/192、ETTm1 四档与 Electricity-192，
即 **s=0 数据集族**——与 §4.2 表中 s=0 的 3-seed 均值退化完全同构，只是本表把
"3-seed 均值退化"换成"最优 seed 口径"后**该符号仍成立**。

**（iv）两个必须一起写的口径**：
1. **同 setting 两套数并存**：如 ETTh1-336，E14 固定协议的最优臂 `l_q1_4` 是 +1.98%/+2.30%
   （未胜），定向调参的最优组合是 **−1.69%/−0.21%**（单 seed 胜）——**它们是两种协议，
   不是一次实验的两行**；任何表格引用时必须注明出自哪一套。`loss`（huber/mae）、lr、
   gate、rank、`huber_delta` 的差异逐格列在复现手册 §2.3。
2. **本节全部数字为 test-set selection**（以 test 选 seed 与组合），Golden 来自另一套硬件环境，
   比较只作披露；主张 A/B 的配对基线仍以 §4.2 的 3-seed 均值口径为准（本节的"最优 seed"口径
   是复现与检索用的，不改变判定）。

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
| PhaseFormer-L | ETTh2 | 96 | 近期加权水平 / 0.67 | 整体位移 / 0.97 | 0.702 | 0.61 / 0.75 | ✗ |
| PhaseFormer-L | ETTh2 | 720 | 近期加权水平 / 0.70 | 整体位移 / 0.94 | 0.494 | 0.77 / 0.80 | ✗ |
| PhaseFormer-L | ETTm2 | 96 | 近期加权水平 / 0.63 | 整体位移 / 0.96 | 0.463 | 0.70 / 0.74 | ✗ |
| PhaseFormer-L | ETTm2 | 192 | 近期加权水平 / 0.69 | 整体位移 / 0.96 | 0.625 | 0.70 / 0.76 | ✗ |
| PhaseFormer-L | Weather | 96 | 近期加权水平 / 0.57 | 倾斜修正 / 0.92 | 0.426 | 0.95 / 1.00 | ✗ |
| PhaseFormer-L | Weather | 192 | 近期加权水平 / 0.57 | 曲率修正 / 0.90 | 0.204 | 0.91 / 0.95 | ✗ |
| PhaseFormer-L | Electricity | 336 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| L-q1/4 | ETTh2 | 96 | 近期加权水平 / 0.62 | 整体位移 / 0.95 | 0.762 | 0.44 / 0.81 | ✗ |
| L-q1/4 | ETTh2 | 720 | 近期加权水平 / 0.69 | 整体位移 / 0.91 | 0.091 | 0.63 / 0.76 | ✗ |
| L-q1/4 | ETTm2 | 96 | 近期加权水平 / 0.62 | 整体位移 / 0.96 | 0.616 | 0.58 / 0.93 | ✗ |
| L-q1/4 | ETTm2 | 192 | 近期加权水平 / 0.67 | 整体位移 / 0.97 | 0.701 | 0.60 / 0.81 | ✗ |
| L-q1/4 | Weather | 96 | 近期加权水平 / 0.58 | 倾斜修正 / 0.92 | 0.416 | 0.90 / 1.00 | ✗ |
| L-q1/4 | Weather | 192 | 近期加权水平 / 0.62 | 曲率修正 / 0.90 | 0.436 | 0.91 / 0.98 | ✗ |
| L-q1/4 | Electricity | 336 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| L-q1/8 | ETTh2 | 96 | 近期加权水平 / 0.62 | 整体位移 / 0.96 | 0.803 | 0.40 / 0.88 | ✗ |
| L-q1/8 | ETTh2 | 720 | 近期加权水平 / 0.73 | 整体位移 / 0.95 | 0.196 | 0.69 / 0.87 | ✗ |
| L-q1/8 | ETTm2 | 96 | 近期加权水平 / 0.60 | 整体位移 / 0.97 | 0.639 | 0.54 / 0.94 | ✗ |
| L-q1/8 | ETTm2 | 192 | 近期加权水平 / 0.63 | 整体位移 / 0.96 | 0.718 | 0.63 / 0.77 | ✗ |
| L-q1/8 | Weather | 96 | 近期加权水平 / 0.58 | 曲率修正 / 0.87 | 0.357 | 0.88 / 0.98 | ✗ |
| L-q1/8 | Weather | 192 | 近期加权水平 / 0.57 | 曲率修正 / 0.86 | 0.383 | 0.89 / 0.95 | ✗ |
| L-q1/8 | Electricity | 336 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |

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
`Independent-RRR-only`、`Conditional-RRR-only` 与**新增的 `RandomRRR-drop`**；
同时报告支路自身与融合误差）：

> **臂数是逐格而定的，不是常数：11–13 臂/格。** 其中
> `Independent-RRR-only`（该格没有 Stage-3 子空间文件时用 train split 现拟合）与
> 本文新增的 `RandomRRR-drop` **每格必有**；`PCA-matched-only/-drop` 只在
> "语义子空间的潜像维数 < 该格头的秩"时追加；`Conditional-RRR-only` 只在
> 该格存在 Stage-3 子空间文件时追加（**稠密头没有该文件**，故 `l_main` 的格子不带它）。
> 逐格按各自 checkpoint 的 encoder 与各数据集语义张成实测（2026-09-20）：
> **11 臂 24 格、12 臂 21 格、13 臂 18 格**，63 格共 **750 行**、**13 个不同的臂名**
> （`l_main` 252 行、`l_q1_4` 255 行、`l_q1_8` 243 行）。
> 故"63 × 11 = 693"是错的写法；完整性判据写成"**每格必须带全部 10 个恒在臂**"，
> 而不是写死一个臂数（否则合法的 12/13 臂格会被误判为不完整）。

> **来源与复用（2026-09-20 补；§4.0 要求"所有复用格在表中标注来源"，本段原缺）**：
> 本节的 **63 个 cell 全部是 Stage-0 复用 checkpoint**（`l_main` / `l_q1_4` / `l_q1_8` × 7 个 test-selected
> setting × 3 seed；manifest 状态实测 **63/63 = `reused`**），**本节没有新训 run**。
> 因此 §4.4 的解剖与干预建立在**与 §4.2 `PhaseFormer-L` 同一批 checkpoint**之上——包括这些复用格各自的
> Stage-0 冻结 `(gate, lr)` 先验（§4.0 的三种 gate 先验之一），**不是独立重训**；
> E16 也不读 test（`test_split_read: false`），故本节与 §4.2 的 test 数字不同源。

| Dataset | H | q/r | Semantic-only Δfused | Semantic-drop Δfused | 随机 95% 区间 | PCA-drop | **随机 RRR 子空间 drop**（新增对照） | 支路自身 Δ | 融合 Δ |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| ETTh2 | 96 | dense（r=96） | +0.0025 | +0.0638 | [0.2035, 0.2044] | +0.0663 | [0.2046, 0.2054] | -0.1172 | +0.0638 |
| ETTh2 | 720 | dense（r=720） | +0.0040 | +0.0658 | [0.6065, 0.6078] | +0.0711 | [0.6064, 0.6098] | -0.0758 | +0.0658 |
| ETTm2 | 96 | dense（r=96） | +0.0031 | +0.0441 | [0.1118, 0.1123] | +0.0479 | [0.1203, 0.1411] | -0.0254 | +0.0441 |
| ETTm2 | 192 | dense（r=192） | +0.0023 | +0.0610 | [0.1503, 0.1505] | +0.0635 | [0.1518, 0.1560] | -1.4407 | +0.0610 |
| Weather | 96 | dense（r=96） | +0.0583 | +0.0836 | [0.3864, 0.3879] | +0.1945 | [0.3880, 0.4123] | -2.7527 | +0.0836 |
| Weather | 192 | dense（r=192） | +0.0384 | +0.0503 | [0.4462, 0.4481] | +0.1621 | [0.4457, 0.4509] | -0.3427 | +0.0503 |
| Electricity | 336 | dense（r=H） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| ETTh2 | 96 | q=1/4（r=24） | +0.0000 | +0.0416 | [0.2064, 0.2190] | +0.0416 | [0.2468, 0.2468] | -0.0342 | +0.0416 |
| ETTh2 | 720 | q=1/4（r=180） | +0.0026 | +0.0659 | [0.6057, 0.6074] | +0.0696 | [0.6091, 0.6145] | -0.0313 | +0.0659 |
| ETTm2 | 96 | q=1/4（r=24） | +0.0000 | +0.0410 | [0.1152, 0.1245] | +0.0410 | [0.1549, 0.1549] | -0.0228 | +0.0410 |
| ETTm2 | 192 | q=1/4（r=48） | +0.0000 | +0.0491 | [0.1534, 0.1575] | +0.0491 | [0.1784, 0.1941] | -1.1375 | +0.0491 |
| Weather | 96 | q=1/4（r=24） | +0.0000 | +0.1738 | [0.3937, 0.4279] | +0.1738 | [0.5631, 0.5631] | -2.4226 | +0.1738 |
| Weather | 192 | q=1/4（r=48） | -0.0003 | +0.1692 | [0.4499, 0.4592] | +0.1719 | [0.5190, 0.5726] | +0.0590 | +0.1692 |
| Electricity | 336 | q=1/4（r=84） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| ETTh2 | 96 | q=1/8（r=12） | +0.0000 | +0.0516 | [0.2111, 0.2464] | +0.0516 | [0.2578, 0.2578] | -0.0612 | +0.0516 |
| ETTh2 | 720 | q=1/8（r=90） | +0.0012 | +0.0692 | [0.6092, 0.6127] | +0.0712 | [0.6194, 0.6324] | -0.0483 | +0.0692 |
| ETTm2 | 96 | q=1/8（r=12） | -0.0000 | +0.0427 | [0.1204, 0.1431] | +0.0427 | [0.1570, 0.1570] | -0.0338 | +0.0427 |
| ETTm2 | 192 | q=1/8（r=24） | +0.0000 | +0.0432 | [0.1537, 0.1630] | +0.0432 | [0.1961, 0.1961] | -1.1449 | +0.0432 |
| Weather | 96 | q=1/8（r=12） | +0.0000 | +0.1600 | [0.4142, 0.4864] | +0.1600 | [0.5477, 0.5477] | -2.2452 | +0.1600 |
| Weather | 192 | q=1/8（r=24） | -0.0000 | +0.1574 | [0.4502, 0.4819] | +0.1574 | [0.6047, 0.6047] | +0.0929 | +0.1574 |
| Electricity | 336 | q=1/8（r=42） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |

新增"随机 RRR 子空间"对照用于区分"语义有效"与"任意同数量主方向有效"（既有结果中 57/57 单元
Semantic-drop ≡ PCA-drop，此项此前缺失）。

#### 4.4.1 实测判定（2026-09-20；**覆盖 18/21 行**，Electricity-336 三行未跑并已在表内标注）

**覆盖说明（必须连同数字一起读）**：本表登记 21 行 = 3 模型 × 7 setting。本轮实际产出 **18 行**
（6 个 setting × 3 模型：ETTh2-96/720、ETTm2-96/192、Weather-96/192），
**Electricity-336 的 3 行未跑**（该 setting 有 5.57M pairs × 321 channels，单格实测 63.6 min，
按指示停止），表内以 `未跑（按指示停止）` **逐格标注**，不留下"看起来只是没填"的空格。
全部解剖与干预**只用 validation 划分**、不读 test；**18 行全部来自 Stage-0 复用 checkpoint，本节无新训 run**。

**必答 1：主模式"读什么、写什么"——读的一侧是齐一的，写的一侧以"整段位移"为主、Weather 是例外**

| 项 | 实测（18 行） |
|---|---|
| **主模式输入组** | **`近期加权水平` 18/18** ✓，解释率 **0.57–0.73**（判据要求 ≥0.5 ✓） |
| **主模式输出组** | **`整体位移` 12/18** ✓（解释率 **0.91–0.97**）；`曲率修正` 4 行、`倾斜修正` 2 行——**全部集中在 Weather** |
| 主模式占修正能量份额 | **0.091–0.802**（中位约 0.5） |
| 跨 seed `leading4` 重叠 | **0.399–0.954**（多数 >0.5；两行约 0.40–0.44，属中等） |

⇒ "**读近端加权电平**"在**全部 18 行**成立 ✓；"**写整段恒定位移**"在 **12/18** 成立，
**例外全在 Weather**——这与 §4.7 早已写下的边界预判一致（"Weather 类弱周期数据若主模式转向曲率/慢趋势，
应作为命题的边界条件单独报告"），故**按边界条件报告，而不是当作反例掩盖**。

**必答 2：删除它会让融合变差、即使支路自身变好——支持，且幅度远大于先导证据**

15 个有判别力的格里 **12 格**出现"**支路自身变好 + 融合变差**"的分离 ✓，稠密臂的幅度为
**支路 −9.0% ~ −83.9% 对 融合 +10.9% ~ +40.6%**（例：ETTm2-192 支路 **−83.9%**、融合 **+40.6%**）。
**两处例外如实记**：Weather-192 的 q1/4 与 q1/8 是支路**也**变差（+6.9% / +11.1%），不呈现该分离。
另有与随机对照直接相关的判据：**`Semantic-drop` 超出同维随机删除 95% 区间的判据（criterion_4）
在 18/18 行的 3/3 seed 上全部成立** ✓。

**必答 3（本节新增对照，也是 §5 限制第 4 条的解法）：语义特异性——在 18/18 行上成立**

- **`Semantic-drop` 超出"随机 RRR 同维子空间"95% 区间（criterion_6）：18/18 行、3/3 seed 全部成立** ✓✓
  ——即**删除语义子空间的代价显著超出删除随机同维子空间**，这正是既有证据（57/57 `Semantic-drop ≡ PCA-drop`）
  所缺的那一步。
- 在**9 个有判别力的格**上，`Semantic-drop − RandomRRR-drop` 的融合 MSE 差为 **+6.3 ~ +39.3 个百分点**
  （如 ETTm2-96 稠密 +25.3、ETTh2-96 稠密 +30.8、Weather-96 稠密 +19.2）。
- **8 个格是"平局"，但按构造无判别力、不得当作反证**：这 8 格实测 `subspace_dimension == rank_dim`
  （如 ETTh2-96 q1/4：dim 24 = r 24）⇒ 语义子空间**张满该头的秩**，删它等于"整体删除修正"，
  三种 drop 臂数值完全相同（工具也因此不给这些格追加 `PCA-matched-*`）。**逐行验证过，是构造使然**。
- ⇒ **§5 限制第 4 条由"尚未分开"改为"已分开"**（适用范围：上述 18 行中的 9 个有判别力格，逐行披露）。

**必须一起报告的一条否定结果：6 条判据里 5 条全过，但"稳定语义"判定 18/18 为 `False`**

工具把"稳定语义"定义为 6 条判据的合取；实测 **criterion_1/2/3/4/6 在 18/18 行的 3/3 seed 上全部成立** ✓，
**唯一不满足的是 `criterion_5`（`Semantic-only` 融合不差于原 checkpoint 超过 0.5%），0/3** ✗。
即：**语义子空间是必要的（删它比删随机更伤 ✓），但不是充分的（只留它也会掉超过 0.5% ✗）**。
故本节**不得**声称"稳定语义成立"；正确表述是"**必要条件成立、充分条件不成立**"。

**与既有低秩分析的对账（未达标，须披露）**：`reference_parity.passed = false`。
逐字段看，**8 个被比较字段里 7 个吻合到 ≤1e-14**（含**奇异值完全一致** ⇒ 有效映射被逐位复现），
**只有 `correction_energy_share`（逐模式能量分配）有 3.5e-05 ~ 1.1e-02 的差**
（相对而言在 3 位小数显示下仅个别模式跨界）。**注意：上表的结论不依赖该量**——
"稳定语义"6 条判据与 Q2/Q3 的结论都由**干预前后融合/支路的 MSE 差**决定，
而该差值与能量分配口径无关 ✓。

### 4.5 条件性学习（贡献 4 的正式检验）

| Dataset | H | direct | 冻结独立-RRR 方向 1 | 冻结条件-RRR 方向 1 | PhaseFormer-L（联合） | H1：cond 距离 < indep 距离（seed 数） |
|---|---:|---|---|---|---|---|
| ETTh2 | 96 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| ETTh2 | 720 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| ETTm2 | 96 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| ETTm2 | 192 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| Weather | 96 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| Weather | 192 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |
| Electricity | 336 | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） | 未跑（按指示停止） |

预测：冻结独立方向退化；冻结**条件性**方向应明显好于独立方向（此对照此前未做，是区分"冻结本身有害"与
"独立目标错位"的关键）；联合训练最好。

**本表的行与口径（E17，2026-09-19 登记，含一条重要前置发现）**：

- 行 = 7 个 test-selected setting × 3 seed（每格三 seed 聚合）；7 个 setting 由 test-set selection 得到，
  不是盲测样本。
- **`direct` 与 `PhaseFormer-L（联合）` 在本实现下是同一配置**：PhaseFormer-L 的修正器就是与主干
  联合训练、无瓶颈约束的头。两列数值相同是**构造使然**，不是两次独立实验——表注必须写明。
- **来源与复用（2026-09-20 补；§4.0 要求"所有复用格在表中标注来源"，本节原缺）**：本表四列里**只有两列含新训格**——
  `冻结条件-RRR 方向 1`（7 setting × 3 seed = **21 格新训**）与 `冻结独立-RRR 方向 1` 的 **Electricity-336**
  （1 setting × 3 seed = **3 格新训**，因为 E8 未覆盖 Electricity），合计 **24 个新训 run**。
  其余**全部是复用**：`direct` 与 `PhaseFormer-L（联合）` 复用 **E14 的 `l_main`**（7 setting × 3 seed = 21 格，
  即 §4.0 登记的 Stage-0 复用格，两者同源故数值相同）；`冻结独立-RRR 方向 1` 的另外 **6 个 setting**
  复用 **E8 的既有登记**（`research_runs/top2_direction_retention_v1/results.csv` 中 `arm=keep_direction_1` 的行，
  读取状态 `read`/`reused` 且与 val 复现差在容差内）。**因此这 7 行不是 7 次独立实验**：
  `direct`/`联合` 两列是复用、`冻结独立` 列 6/7 行是复用、只有 `冻结条件` 列与 Electricity-336 那一格是新数据。
- **单元格格式（2026-09-20 补，否则填完后无法读）**：四个臂列的每一格是 **`MSE/MAE`**（各 **3 位小数**，
  由产出者 `e17_writeback.py` 的 markdown 写出点固定），两者都是 **test** 指标、**越低越好**；
  `—` 表示该 setting 该臂无可用记录。第 7 列是 **H1 成立的 seed 数**（`N/3`，见本节末条），
  不是距离或 MSE。
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
| 输入平滑（boxcar / causal EMA，各 5 档） | 支路输入 | 7 setting | 14/14 组合无一改善，越平滑越差 | 未跑（按指示停止） |
| 结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离） | 支路参数化 | 4 setting | 5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52% | 未跑（按指示停止） |
| SVD 截断 vs 秩约束训练 | 支路权重 | Electricity-336 r=10 | 截断 +29%，训练 +0.7% | 未跑（按指示停止） |
| 联合低秩训练 q=1/32 | 支路容量 | 7 setting | 保留 92.4%–101.9% 可实现价值 | 未跑（按指示停止） |
| **边界消融：`pooled_lowrank` rank∈{1,2}** | 支路容量（网格之外） | 6 setting × 3 seed | 无 | 未跑（按指示停止） |

> **§4.6 行 1 与行 5 的规模／预注册说明（须保留）**：§4.6 的回填方式是**只替换第五列**
> （见 `minipaper_fill_mapping.md` §1.7），故这两格原有的计划文字会被结果顶掉，先行移到这里：
>
> * **行 1**：在 PhaseFormer-L 上复测 **2 档**——同一 causal EMA 算子的两个**强度**
>   （`smooth_ratio = 0.5` 与 `1.0`，见上方「2 档」口径注）；**42 runs** = 7 setting × 3 seed × 2 档。
> * **行 5**：**预注册预期：相对 direct 退化**，幅度上界由 `1−capture(1/2)`
>   （14%–35% / 3%–18% 的支路价值）经 `g²` 折算；用于量化残项 ε，**不影响主张**。
>   **36 runs** = 6 setting × 3 seed × rank∈{1,2}（**绝对**秩，非 `H/4`、`H/8` 的相对秩）。

**行 3 的 checkpoint 来源与复用（2026-09-20 补；§4.0 的"标注来源"规则）**：行 3 的 checkpoint
**全部来自 E14**——`l_main`（全秩）+ `l_q1_4` / `l_q1_8`（低秩），28 setting × 3 seed × 3 臂 = **252 格，
其中 63 格为 Stage-0 复用、189 格为本轮新训**。故行 3 与 §4.2 **共享同一批 checkpoint**，不是独立重训；
本节行 1 / 行 5 的 **78 个 run 则全部为本轮新训**（无复用）。行 3 只读 val（`--evaluation-split val`），不读 test。

> **§4.6 行 3 的口径差异（须披露）**：既有 E11 为 **test** 口径、单 seed、checkpoint 取自
> `rank_sweep_2_stage1`；本表补做部分为 **validation** 口径、checkpoint 取自 §4.2 的
> `l_main`/`l_q1_4`/`l_q1_8`——两者**只有截断代数与三段式比较结构相同**，
> 故行 3 的"既有"与"本文补做"两列**不可直接相减**。
> （本条原写在行 3 的"本文补做"单元格内。**该单元格会被回填覆盖**——§4.6 的回填方式是
> **只替换第五列**（见 `minipaper_fill_mapping.md` §1.7 与 `fill_minipaper_46.py`），
> 而这一条恰好就写在第五列里，所以无论整行替换还是只换该列，它都会被顶掉，故**先移到此处**。
> **勘误（2026-09-20）**：此处原写"回填方式是用回填工具产出的 **5 行整行替换**"——**这是错的**，
> 且与本节上方第 768 行"只替换第五列"**自相矛盾**。真因是**只填第五列**：
> 产物与论文在行 1、行 5 的前四列措辞不同（论文那句描述的是**先前**实验），整行替换会改掉**本来正确**的正文；
> 且行 1 的首格两侧不同、**没有共享键可匹配**，故 §4.6 只能按位置对齐、只覆盖第五列。
> 保留这段勘误是因为"填表机制"被写错会让后来者按错误的心智模型去核对。）

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
| `cycle_level_std` | -0.039 | 0.183 | + |
| `last_cycle_shift` | -0.139 | 0.303 | + |
| `τ̂`（电平记忆长度，`tau_hat_steps`） | -0.750 | 0.325 | 与学到的 EMA τ 正相关 |

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
3. **诊断列 `s` 的判错须如实报告**：Electricity 的 `τ̂ = 54.45` 步落在可分离区间内部（判定最不确定）。
   该数据集的**四档实测 ΔMSE 全部为负**（`−0.96% / −0.43% / −2.88% / −1.15%`，均值 **−1.36%**，
   即修正器在该数据集**有增益**），而诊断判它 `s=0` ⇒ **四档全部为假阴性**，与 §4.2.1 引用的
   `claims.json:diagnostic_misses` 一致。按 §4.0 报告规则列出全部判错格，不调阈值迁就。
   > **勘误（2026-09-20）**：本条此前写"Electricity-336 上修正器为 **+3.6%**（0.16768 → 0.1617）
   > 即该格判错"——**符号写反**（`0.1617 < 0.16768`，实为 **−3.6%**，修正器更好），且判错范围应为
   > **四档**而非一格。理由更正为"有增益却被判 `s=0` ⇒ 漏报"；"须逐格列出判错格"这一要求不变。
4. **两列 ρ 的符号约定，以及"预期符号"列的读法**（否则本表填完后无法读）：
   * **ΔMSE 的定义**：`ΔMSE = 100 × (PhaseFormer-L 的 MSE − matched phase_only 的 MSE) / phase_only 的 MSE`，
     即**负值 = 修正器更好**、正值 = 更差（与产出者 `e19_predictive_power.py:166` 的算式一致；
     该工具同时以 `delta_mse_pct < 0` 定义 `corrector_helps`）。**本表若只给 ρ 而不给这条约定，
     读者会把 ρ 的符号读反**；§4.0 已统一声明该约定（含 §4.2 / §4.4 的 `Δ` 列），本注只补"ρ 怎么读"。
   * **"预期符号"列指的是"与 ΔMSE 的 ρ"这一列**（不是与 `g` 的那列），其含义要看注 1：
     `cycle_level_std` 与 `last_cycle_shift` 是**反序**的（正类偏低、负类偏高），故其 ρ 预期为 **正**——
     **这是"该统计量指错了方向"的证据，不是"电平波动越大越差"**；一个方向正确的诊断列
     （如 `τ̂`）预期应为**负** ρ（τ̂ 越大 ⇒ 该数据集越需要电平通道 ⇒ 修正器越好 ⇒ ΔMSE 越负）。
   * `τ̂` 行的条目是"与**学到的 EMA τ** 正相关"，**不是** ρ 的符号主张；它与 ΔMSE 的相关方向按上一条为负。
   * 因此三行的读法是：**前两行 ρ 为正 ⇒ 复现了注 1 的"反序"（该统计量不可用于诊断）；
     第三行 ρ 为负 ⇒ 命题 1 的预测力成立**。
5. **本表不含新训数据**：两列 ρ 由 ① §4.2 的 PhaseFormer-L 相对 `phase_only` 的增益（其复用与来源披露见
   §4.0 / §4.2 表注）与 ② E19 的**训练集**电平统计量（与 §4.3 同源、从不读 test）算出；本节不额外训练任何模型。
6. **"与 `g` 的 ρ"列的样本量是 n = 21，不是 28；且该列不受 §4.2 门列修复影响**（2026-09-20 补）：
   * 本表两列的口径不同：`vs ΔMSE` 列用全部 **28** 个 setting（`n=28`）；而 `vs g` 列的 `g`
     取 E19 自己的来源——`results.csv` 的 `gate_value`（`e19_predictive_power.py:106`），
     **只有 `status=read` 的行带该列**，7 个 `status=reused` 的 Stage-0 格子没有它、**不进入该列**，
     故三格的 n 均为 **21**（`predictive_power_summary.json` 的 `predictive_power.scope = "all_28_settings"`，
     `n_settings = 28`，但 `*_vs_gate_value` 三格的 `n = 21`——**块级 n 与格级 n 不同，引用时必须写格级 n**）。
   * **§4.2 的门列修复不改变本表数字**：本表的 `g` 与 §4.2 的门列是两个独立来源，E19 只读 `results.csv`，
     从不读 `main_table.csv`，故上表六格在修复前后**逐字不变**（已复核）。
   * **两个来源在重叠的 21 格上逐格一致**（同一个 `gate_value`），差异只在样本量：
     §4.2 的列对那 7 个复用格改从 checkpoint 取值，因此是完整的 **28** 格。
   * **敏感性读数（非预登记口径，不进任何判定，仅用于让两个来源可对照）**：若把 `g` 换成 §4.2 的完整列，
     ρ(`g`, ΔMSE) 为 **−0.593（n=28，p=0.001）** / **−0.504（main-24，n=24，p=0.012）**；
     若仍限在"有 `results.csv` 门值"的格子内则为 **−0.190（n=21，p=0.410）** / **+0.115（main-24，n=17，p=0.660）**。
     **"换了样本量就换符号"这件事本身要写在表里**，以免读者把两处不一致读成矛盾。
     无论取哪一口径，**命题 1 的预测力都来自 `vs ΔMSE` 的 −0.750（n=28）**，与本列无关（本列三档均不显著）。


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
4. **"语义有效"与"任意同数量主方向有效"已分开（2026-09-20 回填 §4.4 的结论）**。
   既有证据是 57/57 单元 `Semantic-drop ≡ PCA-drop`（不具特异性），§4.4 新增的随机 RRR 子空间对照即为解决此项。
   实测：**`Semantic-drop` 超出"随机 RRR 同维子空间"95% 区间的判据，在 18/18 行、3/3 seed 上全部成立**；
   在 9 个有判别力的格上，删语义比删随机同维子空间的融合 MSE 差 **+6.3 ~ +39.3 个百分点**。
   **适用范围必须逐行披露**：另有 8 个格 `subspace_dimension == rank_dim`（语义子空间张满该头秩），
   删它等于整体删除修正，三种 drop 臂数值相同 ⇒ **构造性无判别力，不计入证据**；
   **Electricity-336 的 3 行未跑**，不在上述范围内。
   同时如实保留一条限制：语义子空间**必要但不充分**——`Semantic-only` 判据（融合不差于原 checkpoint 超过 0.5%）
   在 18/18 行上 0/3 不满足，故"稳定语义"的合取判定全为 `False`（见 §4.4.1）。
5. 低秩压缩是分析工具与效率选项，不是精度贡献；既有三 seed 结果无普适增益，深档 q=1/16、1/32 有 −0.3%～−0.8%
   的代价，本文明写。
6. **诊断列 `s` 不是模型的一部分**（D-5）：PhaseFormer-L 恒定启用修正器，`s` 只是 §4.7 的一个事前
   预测列。阈值 `ν*` 由 5 个已知增益符号数据集的训练集统计量按规则式定义拟合，样本量极小；它是
   可预注册、可被 Traffic 与 H336/720 新格子反驳的预测，不是已验证的规律。若诊断在新格上判错
   （已知 **Electricity 四档**判错，见 §4.7 表注 3 与 §4.2.1 的 `diagnostic_misses`），
   按 §4.0 主张 B 如实报告为未达标并逐格列出。
   注意 §3.4.3 早期草案称"由 6 个已知 setting 拟合"，实际证据支持的是 **5 个已知符号的数据集**
   （ETTh2/ETTm2/Weather 为正、ETTh1/ETTm1 为负），此处以证据为准。
7. Golden 来自不同硬件环境；本文对 Golden 只做披露性比较，主张 A/B 的配对基线是同环境 matched rerun。
8. 结论范围：与相位主干经凸门联合训练的线性残差修正、标准长程基准；不宣称任意时序模型可压缩或任意线性模型
   等价于电平修正。
9. **门值 `g` 是每格训练出来的、不是固定超参；它的数值本身不是机制证据**（2026-09-20 补）。融合门
   `y = (1−σ(g))·y_phase + σ(g)·y_residual` 中的 `g` 是**可训练** `nn.Parameter`（`PhaseFormer.py:1116`，
   Adam 正常更新），`gate_init` 只是初值：实测 init→final 的平均移动为**新训格 17.7%**（最大 73.5%，
   Traffic-96 0.2→0.0573）、**复用格 9.9%**（最大 33.3%，Electricity-336 0.5→0.3337）。
   因此**不得**把 `g` 读成"门开了多少"的机制量、也不得把 `gate_init` 当作可机械调节 `g²` 的旋钮；
   §4.2.1 的门值读法按此表述：门值差异首先反映各格自身的训练轨迹，本文不对其作因果解释。
   同一条也解释了为何 §4.2 的 `g` 列在 2026-09-20 修复后与原读数方向相反（原因之一是缺陷列，
   见 §4.2.1"门列"一段）。
10. **一处产出者侧缺陷的修复记录及其影响范围**（2026-09-20，§4.2 门列）：缺陷只影响 §4.2 表中
    `g` 列的 7 个复用格（及其同源的参数量列），**不影响**主张 A–D 的判定、§4.2 的两列指标、
    §4.4/§4.5/§4.6 的任何数字，也不影响 §4.7 的六格 ρ（E19 只读 `results.csv` 的 `gate_value`，
    与 `main_table.csv` 无依赖）。逐格影响与修复后的复核见 §4.2.1；修复前的产物已存档于
    `research_runs/phaseformer_L_e14_main_v1/pregate_gate_fix/`，可逐格复算。

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
- **复现手册（全部 setting 的参数、最优 seed 与命令）**：`PhaseFormer_L_main_table_repro.md`
  （2026-09-21 起为唯一一份；§4.2.3 的逐格数字与 §4.2.2 的调参组合均出于此，提交 `a05f7cd`）
