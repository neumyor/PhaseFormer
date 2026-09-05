# PhaseFormer + NLinear 渐进式信息瓶颈实验设计规范

**版本**：v1.0  
**用途**：后续实验执行的唯一事实源（Single Source of Truth）  
**起始阶段**：Stage 1  
**前置状态**：Stage 0 已完成，已确认 PhaseFormer + NLinear 相比 PhaseFormer 单模型具有稳定增益。  
**核心原则**：每个阶段只验证一个新增设计；GO 仅决定是否进入后续主线配置，不决定后续实验是否继续。

## 执行记录（2026-09-05）

本计划已在 `weak_residual_nlinear_bottleneck` 分支进入 Stage 1 实施。新增
`src/models/frozen_nlinear_correction.py` 与
`scripts/run_progressive_ib_stage1.py`，其实施约束为：

- 从原版 PhaseFormer checkpoint 加载 (P)，训练前和最低 validation-loss
  checkpoint 恢复后均计算 state hash；hash 不一致立即失败。
- `fusion` 是冻结 (P) 下的 Stage-0 配对 control；`target_residual` 使用同
  参数量的 centered-NLinear head 拟合 (Y-P(X))，并保留 learned fusion；
  `direct` 仅移除该 fusion，输出 (P(X)+\Delta(X))。
- 所有选择只读取 validation；正式运行固定 (L=720)、Huber、30 epochs、同一
  PhaseFormer checkpoint 和同 seed 配对。正式矩阵的每个 dataset/horizon 至少 3
  seeds；先完成 ETTh1-H96 的三 seed 链路校验，再扩展至 ETTh1/ETTh2/ETTm1/ETTm2/
  Weather 的 H96/H192。
- 训练 checkpoint、Lightning 日志及逐样本临时输出只写入
  `research_runs/progressive_ib_stage1_scratch/`；聚合完成后才写入符合六文件白名单
  的 `research_runs/progressive_ib_stage1_v1/`。

当前主机检查到 CUDA driver 不可用，因此正式命令必须附 `--require-cuda`，并在
GPU 恢复前保持未启动状态；CPU 结果不得进入正式矩阵。已使用 `raft` 环境在
ETTh1-H96、seed 2021、`direct`、1 epoch 完成 CPU smoke，仅验证数据、冻结 hash、
训练、最低 validation checkpoint 恢复和逐样本统计链路，不构成实验结果。

---

# 0. 文档目标与最终证据链

本研究不以“增加一个 IB 模块是否涨点”为最终目标，而是希望逐层建立以下证据链：

\[
\text{Correction Role}
\rightarrow
\text{Low-dimensional Sufficiency}
\rightarrow
\text{Statistical Structure}
\rightarrow
\text{Information Compression}
\rightarrow
\text{Conditional Complementarity}
\rightarrow
\text{Reliability-dependent Demand}
\rightarrow
\text{Adaptive Allocation}.
\]

逐层希望证明的最小结论如下：

1. **Stage 1：校正角色**
   - NLinear 的有效贡献可以被重新参数化为对 PhaseFormer 的 residual correction，而不显著损失原始增益。

2. **Stage 2：低维充分性**
   - 在保持 NLinear 原有 level anchor 的前提下，历史动态校正信息是否存在低维充分表示。

3. **Stage 3：统计结构**
   - 低维有效子空间是否与低阶趋势、低频成分等统计结构一致，而非任意随机低维投影均有效。

4. **Stage 4：Anchor 信息会计**
   - 将最后观测值 anchor 与历史动态信息的贡献分离，明确后续“信息压缩”到底压缩了哪部分。

5. **Stage 5：Variational Information Bottleneck**
   - 将“维度压缩”升级为“variational rate 压缩”，研究保存 correction 能力所需的最小编码率上界。

6. **Stage 6：Conditional Information Bottleneck**
   - 利用 PhaseFormer 表征作为 side information，研究能否进一步降低保存相同 correction 所需的 variational rate。

7. **Stage 7：周期可靠性与信息需求**
   - 在控制样本难度等混杂因素后，验证周期可靠性是否仍独立关联 correction information demand。

8. **Stage 8：Adaptive Information Bottleneck**
   - 验证基于周期可靠性的动态信息预算是否改善 rate–error Pareto frontier。

9. **Stage 9：Gate 竞争性解释**
   - 检验“不同样本只需要决定是否使用 NLinear”这一更简单解释是否已足够。

10. **Stage 10：Joint Fine-tuning**
    - 在机制结论已经独立成立后，评估联合优化能否进一步提升最终预测性能。

---

# 1. 全局实验原则

## 1.1 单变量原则

每个 Stage 只允许引入一个新的核心设计变量。

除当前 Stage 明确规定可以改变的因素外，下列内容必须保持不变：

- 数据集划分；
- input length；
- prediction horizon；
- 标准化方式；
- PhaseFormer 主干架构；
- NLinear 基础配置；
- optimizer；
- learning-rate schedule；
- batch size；
- epoch / early-stopping policy；
- random seed 集合；
- loss 定义；
- validation checkpoint selection rule；
- 测试指标。

若某项因实现原因不得不改变，必须在实验报告中明确标记为 **Protocol Deviation**，并说明为什么不会影响当前阶段的因果解释。

---

## 1.2 Test set 使用规则

所有：

- 超参数选择；
- rank 选择；
- \(\beta\) 选择；
- reliability threshold；
- operating point；
- GO / NO-GO 决策；

必须先依据 validation set 完成。

Test set 仅用于最终确认结果，不允许根据 test 表现重新修改配置。

---

## 1.3 随机性控制

基础要求：

\[
N_{\text{seed}}\ge 3.
\]

核心主结果建议：

\[
N_{\text{seed}}=5.
\]

所有成对比较必须使用相同 seed。

必须报告：

\[
\text{mean}\pm\text{std}.
\]

推荐额外报告 paired seed difference：

\[
\Delta D_s = D_{A,s} - D_{B,s}.
\]

---

# 2. 固定基准与核心指标

设：

- PhaseFormer 单模型为 \(M_P\)；
- Stage 0 已验证的 PhaseFormer + NLinear 模型为 \(M_{\mathrm{S0}}\)。

定义 PhaseFormer 误差：

\[
D_P.
\]

定义 Stage 0 完整组合模型误差：

\[
D_{\mathrm{full}}.
\]

某新模型误差：

\[
D_M.
\]

定义完整 NLinear correction gain：

\[
G_{\mathrm{full}}
=
D_P-D_{\mathrm{full}}.
\]

定义新模型 correction gain：

\[
G_M
=
D_P-D_M.
\]

定义 Correction Retention：

\[
\boxed{
CR(M)
=
\frac{D_P-D_M}
{D_P-D_{\mathrm{full}}}
}
\]

解释：

- \(CR=1\)：完整保留 Stage 0 中 NLinear 带来的增益；
- \(CR=0\)：完全失去该增益；
- \(CR>1\)：超过 Stage 0；
- \(CR<0\)：新设计反而损伤 PhaseFormer。

---

## 2.1 Correction Retention 的使用限制

CR 不能单独作为 GO 依据。

当：

\[
D_P-D_{\mathrm{full}}
\]

很小时，CR 的分母很小，数值会高度不稳定。

因此每次报告 CR 时，必须同步报告绝对误差差：

\[
\Delta D_M
=
D_M-D_{\mathrm{full}}.
\]

GO 需要综合考虑：

1. CR；
2. 绝对误差差；
3. cross-seed consistency；
4. cross-dataset / cross-horizon consistency。

---

# 3. 实验状态定义

所有 Stage 使用以下四种状态：

## GO

当前 Stage 的设计得到足够支持，可进入后续 Mainline Configuration。

## WEAK-GO

趋势支持假设，但 effect size、跨数据集一致性或统计稳定性不足。

进入 supplementary branch，不作为默认主线。

## NO-GO

当前设计没有得到支持。

该设计不进入后续主线，但后续 Stage 仍继续。

## INCONCLUSIVE

由于：

- 高方差；
- 训练不稳定；
- 样本量不足；
- 实现控制失败；
- 数据泄漏风险；
- protocol deviation；

暂时无法作出可信判断。

INCONCLUSIVE 不得解释为负结果。

---

# 4. GO 的统一含义

本研究中：

\[
\boxed{
NO\text{-}GO \neq STOP
}
\]

GO 仅表示：

> 当前设计是否获得“后续主线继承资格”。

NO-GO 表示：

> 不继承该设计，但后续问题仍通过预注册 Fallback 继续验证。

---

# Stage 1：验证 NLinear 的校正器角色

## 1. 实验目的

验证以下最小命题：

\[
\boxed{
\text{NLinear 的有效贡献可以被重新参数化为 PhaseFormer residual correction。}
}
\]

本阶段不证明：

- NLinear 天生只能建模 residual；
- NLinear 内部已经显式学习了误差；
- residual formulation 一定优于原始 fusion。

本阶段只研究：

> “原有增益是否可以通过 correction formulation 保留下来”。

---

## 2. 实验设定

固定已经训练完成的 PhaseFormer：

\[
\hat Y_P=P(X).
\]

整个 Stage 1：

\[
\boxed{P\text{ frozen}}
\]

定义：

\[
E=Y-\hat Y_P.
\]

### Control：Stage 0 原始路径

\[
N(X)\rightarrow Y.
\]

使用 Stage 0 原有 fusion。

### Treatment A：Target-only Residual

\[
N_R(X)\rightarrow E.
\]

再构造：

\[
\tilde Y_N=\hat Y_P+\Delta \hat Y.
\]

若 Stage 0 使用 learned fusion，则仍将 \(\tilde Y_N\) 输入原 fusion。

此实验只改变：

\[
\boxed{\text{NLinear learning target}}
\]

### Treatment B：Direct Residual Correction

\[
\hat Y=
\hat Y_P+\Delta\hat Y.
\]

此实验额外验证：

> 是否可以将 fusion 显式简化为 residual addition。

Treatment B 只有在 Treatment A 完成之后才能解释。

---

## 3. 控制变量规则

必须固定：

- PhaseFormer checkpoint；
- NLinear 输入；
- NLinear 参数规模；
- optimizer；
- learning rate；
- training epoch；
- normalization；
- random seed；
- prediction horizon；
- evaluation protocol。

PhaseFormer 参数必须被冻结，并在训练结束后验证权重 hash 或参数差异为零。

---

## 4. 额外机制指标：校正方向

定义每个样本的 PhaseFormer residual：

\[
e_P=Y-\hat Y_P.
\]

定义 NLinear correction：

\[
\Delta=\hat Y-\hat Y_P.
\]

计算：

\[
\langle e_P,\Delta\rangle
\]

以及：

\[
\cos(e_P,\Delta)
=
\frac{
\langle e_P,\Delta\rangle
}{
\|e_P\|_2\|\Delta\|_2
}.
\]

若 NLinear correction 确实朝减少 PhaseFormer error 的方向工作，则总体应更倾向于：

\[
\langle e_P,\Delta\rangle>0.
\]

推荐同时报告：

- 正内积样本比例；
- mean cosine；
- median cosine；
- correction magnitude：

\[
\|\Delta\|_1,\quad \|\Delta\|_2.
\]

---

## 5. 需要填写的结果

| Dataset | Horizon | Model | MSE | MAE | CR | \(\Delta D\) vs Stage 0 | Mean Cosine | Positive Dot % | Params |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| | | PhaseFormer | | | 0 | | | | |
| | | Stage 0 | | | 1.00 | 0 | | | |
| | | Target-only Residual | | | | | | | |
| | | Direct Residual | | | | | | | |

---

## 6. GO 门槛

推荐 operational threshold：

\[
CR_{\mathrm{Target-only}}\ge 0.95
\]

同时：

- paired seed 多数方向一致；
- 绝对 MSE degradation 不超过预注册 tolerance；
- 在多个 dataset / horizon 中具有一致趋势。

### Positive GO

若：

\[
CR>1
\]

且跨 seed 稳定。

### NO-GO

Residual target 导致明显、稳定的 correction gain 丢失。

---

## 7. Fallback

### Stage 1 GO

Stage 2 对 Residual NLinear 进行结构压缩。

### Stage 1 NO-GO

Stage 2 对 Stage 0 中原始 NLinear branch 进行结构压缩。

因此 Stage 2 的“低维性”问题不依赖 residual formulation 成立。

---

## 8. 提交前二次校验

- [ ] PhaseFormer checkpoint 完全一致；
- [ ] PhaseFormer 参数未被更新；
- [ ] residual target 使用对应样本 PhaseFormer prediction；
- [ ] train / val / test residual 没有跨 split 泄漏；
- [ ] 原始模型和 residual 模型 training budget 一致；
- [ ] 若 Stage 0 有 learned fusion，Treatment A 已优先完成；
- [ ] 未将 fusion simplification 收益错误归因于 residual target；
- [ ] 至少完成 3 seeds；
- [ ] CR 与 \(\Delta D\) 同时报告；
- [ ] correction direction 统计正确。

---

# Stage 2：验证非-anchor动态校正信息是否低维

## 1. 实验目的

验证：

\[
\boxed{
\text{在保留 NLinear level anchor 的情况下，历史动态校正信息是否具有低维充分表示。}
}
\]

注意：

Stage 2 **不能**直接声称“整个 NLinear correction 只需要 \(r\) 维”，因为 NLinear 的最后观测值 anchor：

\[
X_L
\]

仍可能作为旁路信息存在。

---

## 2. 实验设定

NLinear 先进行：

\[
\tilde X=X-X_L.
\]

保持原有 anchor 路径不变。

仅对：

\[
\tilde X
\]

进行结构瓶颈：

\[
\tilde X
\xrightarrow{A}
Z_r
\xrightarrow{B}
\Delta Y
\]

或原始 Stage 0 predictor branch 输出。

其中：

\[
Z_r\in\mathbb R^r.
\]

---

## 3. Rank 扫描

建议：

\[
r\in
\{
1,2,4,8,16,32,64
\}
\]

并补充：

\[
r=L
\]

或足够大的 full-capacity control。

若 \(L\) 较小，按 \(L\) 截断。

---

## 4. 实验规则

Stage 2 唯一允许改变的是：

\[
\boxed{r}
\]

不允许：

- 增加非线性；
- 增加 KL；
- 注入 stochastic noise；
- 改 loss；
- 改 PhaseFormer；
- 改 anchor；
- 加 reliability；
- 加 gating。

---

## 5. Full-capacity recovery control

必须验证 factorized full-capacity 模型能够基本恢复未压缩 NLinear。

若 full-capacity factorized model 本身明显劣化，则本 Stage 无法判断 rank effect，应标记 INCONCLUSIVE，而不是 NO-GO。

---

## 6. Parameter-matched control

在关键 rank 附近增加参数量匹配的 nonlinear MLP control，以排除：

> “性能变化只是参数减少或正则化带来的”。

---

## 7. 需要填写的结果

| \(r\) | Params | FLOPs | MSE | MAE | CR | \(\Delta D\) vs Full | Seed Std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | | | | | 1.00 | 0 | |
| 64 | | | | | | | |
| 32 | | | | | | | |
| 16 | | | | | | | |
| 8 | | | | | | | |
| 4 | | | | | | | |
| 2 | | | | | | | |
| 1 | | | | | | | |

必须绘制：

\[
CR(r)
\]

并记录：

\[
r_{95}
=
\min\{r:CR(r)\ge0.95\}.
\]

同时可记录：

\[
r_{99}.
\]

---

## 8. GO 门槛

推荐：

存在：

\[
r\le L/4
\]

且：

\[
CR(r)\ge0.95.
\]

Strong GO：

\[
r\le L/8
\quad\text{且}\quad
CR(r)\ge0.95.
\]

同时要求：

- paired seed 结果大多数一致；
- \(\Delta D\) 可接受；
- 多个 dataset / horizon 出现同向趋势。

---

## 9. Fallback

### Stage 2 GO

后续需要固定 latent capacity 时，优先使用：

\[
r_{95}.
\]

### Stage 2 NO-GO

后续 **不得**强行使用低 rank probe 作为主容量。

Stage 5 的 capacity 必须选择：

\[
\boxed{
r_{\text{full-correction}}
=
\text{能够恢复完整 correction 的最小表示容量}
}
\]

如果所有低 rank 都无法恢复完整 correction，则使用 full-rank / full-capacity latent。

目的：

> Stage 5 只测试 information-rate compression，不能让 capacity bottleneck 成为混杂因素。

Stage 3 仍可以使用预注册 probe rank 进行 structured basis 分析，但只能作为结构探索，不作为 Stage 5 的容量依据。

---

## 10. 二次校验

- [ ] anchor 路径完全一致；
- [ ] 只有 \(\tilde X=X-X_L\) 被压缩；
- [ ] full-capacity factorized control 可恢复基线；
- [ ] rank 之外无其他结构变化；
- [ ] 参数量和 FLOPs 计算正确；
- [ ] \(r_{95}\) 基于 validation 选择；
- [ ] test 不参与 rank selection；
- [ ] parameter-matched MLP 参数量误差处于预设范围；
- [ ] Stage 2 结论只针对 non-anchor dynamic correction。

---

# Stage 3：验证低维有效子空间是否具有低阶统计结构

## 1. 实验目的

区分：

### 命题 A

有效 correction information 是低维的。

### 命题 B

有效 correction information 与低阶趋势或低频统计一致。

Stage 2 最多支持命题 A。

Stage 3 专门验证命题 B。

---

## 2. 实验设定

固定 latent dimension：

### 若 Stage 2 GO

\[
r=r_{95}.
\]

### 若 Stage 2 NO-GO

使用预注册：

\[
r_{\text{probe}}
=
\min(8,L/4)
\]

用于 structured-basis 探索。

此时不得把 Stage 3 结果解释为“完整 correction 的低维充分性”。

---

## 3. Encoder 对比

### A. Learned Linear Projection

\[
Z=A_{\mathrm{learned}}\tilde X.
\]

作为 learned upper bound。

### B. Polynomial Statistics

使用时间方向多项式基：

\[
1,t,t^2,\ldots
\]

近似表示：

- level-relative offset；
- slope；
- curvature；
- higher-order slow variation。

### C. Low-frequency DCT / Fourier

保留前 \(r\) 个低频成分。

### D. Random Projection

\[
Z=R\tilde X.
\]

\(R\) 在训练过程中固定。

必须测试多个 random matrices，避免单一随机投影偶然偏差。

---

## 4. 统一控制规则

必须固定：

- latent dimension；
- decoder；
- loss；
- PhaseFormer；
- training budget；
- normalization。

Polynomial、DCT、Random basis 均不得训练。

---

## 5. 子空间几何分析

除了预测性能外，需要分析 learned subspace 与 handcrafted subspace 的几何相似性。

可使用：

- principal angles；
- subspace overlap；
- projection matrix similarity。

例如：

\[
\mathrm{Overlap}
=
\frac{
\mathrm{Tr}(P_{\mathrm{learned}}P_{\mathrm{stat}})
}{
r
}.
\]

其中：

\[
P_A=A^\top(AA^\top)^{-1}A.
\]

目的：

> 区分“两个表示碰巧性能相似”和“learned projection 确实学习到相似统计方向”。

---

## 6. 需要填写的结果

| Encoder | \(r\) | Learnable? | MSE | MAE | CR | vs Random | vs Learned | Subspace Overlap |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Learned | | Yes | | | | | 0 | 1.00 |
| Polynomial | | No | | | | | | |
| DCT | | No | | | | | | |
| Random | | No | | | | 0 | | |

另外建议报告 latent 与：

- historical mean；
- last-value-relative trend；
- local slope；
- variance；
- low-frequency energy；
- dominant-frequency strength；

之间的相关性。

---

## 7. 结论分级

### Level 1：仅低维成立

若：

\[
Learned\gg Random
\]

但 Polynomial / DCT 明显落后。

结论：

> correction 存在低维有效子空间，但其语义不一定是简单低阶统计。

### Level 2：低阶/低频统计成立

若：

\[
Polynomial\ \text{或}\ DCT
\approx Learned
\gg Random
\]

且 subspace overlap 较高。

可进一步支持：

> correction 的主要有效方向与低阶趋势 / 低频历史统计一致。

---

## 8. GO 门槛

推荐：

\[
CR_{\mathrm{structured}}
\ge
0.95
CR_{\mathrm{learned}}
\]

且显著优于 random projection。

如果只有 learned 有效，则 Stage 3 对“低阶统计结构”判 NO-GO，但不影响后续 IB。

---

## 9. Fallback

### Stage 3 GO

后续可以保留：

- Learned encoder；
- Best structured encoder。

### Stage 3 NO-GO

Stage 5 统一使用 learned encoder。

原因：

> “低阶统计假设失败”并不等于“不存在信息冗余”。

---

## 10. 二次校验

- [ ] 所有 encoder latent dimension 相同；
- [ ] decoder 完全一致；
- [ ] structured basis 未参与学习；
- [ ] Random projection 使用多个随机矩阵；
- [ ] 多项式 basis 做了数值归一化；
- [ ] DCT / Fourier 频率定义一致；
- [ ] 子空间相似度计算使用统一归一化；
- [ ] 没有把“性能相似”直接解释为“表示相同”；
- [ ] 没有将 Stage 3 GO 误写成真实信息瓶颈证据。

---

# Stage 4：拆分 NLinear Anchor 与历史动态信息贡献

## 1. 实验目的

明确：

\[
X_L
\]

即最后观测值 anchor 在 correction 中承担多少作用。

后续必须区分：

### Anchor-exempt compression

给定 \(X_L\) 作为免费 side information，研究还需要多少历史动态信息。

### Full-channel compression

连 \(X_L\) 也计入 correction channel 的信息预算。

---

## 2. 实验设定

定义：

\[
\tilde X=X-X_L.
\]

比较：

### Full

\[
(\tilde X,X_L)\rightarrow\Delta Y.
\]

### Anchor-only

\[
X_L\rightarrow\Delta Y.
\]

### History-only

\[
\tilde X\rightarrow Z\rightarrow\Delta Y
\]

decoder 不可访问 \(X_L\)。

### History + Explicit Anchor

\[
(Z(\tilde X),X_L)\rightarrow\Delta Y.
\]

---

## 3. 实验规则

本阶段只改变：

\[
\boxed{\text{anchor accessibility}}
\]

latent capacity、decoder capacity、PhaseFormer、训练预算均固定。

---

## 4. 需要填写的结果

| Model | Has \(X_L\)? | Has Dynamic History? | MSE | MAE | CR | \(\Delta D\) |
|---|---|---|---:|---:|---:|---:|
| Full | Yes | Yes | | | | |
| Anchor-only | Yes | No | | | | |
| History-only | No | Yes | | | | |
| History + Anchor | Yes | Yes | | | | |

建议额外记录近似 contribution：

\[
G_{\mathrm{anchor}},
\quad
G_{\mathrm{history}},
\quad
G_{\mathrm{interaction}}.
\]

注意这些量一般不严格可加，只作为 ablation-based accounting。

---

## 5. 结果类型

### Anchor-dominant

Anchor-only 已保存大部分 correction。

### Mixed

History + Anchor 明显优于二者单独。

### History-dominant

动态历史贡献主要 correction。

本 Stage 主要为机制信息会计，不要求性能 GO。

---

## 6. 后续规则

Stage 5 必须明确选择主问题：

### 主问题 A：Anchor-exempt

研究：

> 已知当前 level 后，还需多少历史动态信息？

### 主问题 B：Full-channel

研究：

> 整个 NLinear correction channel 总共需要多少信息？

至少一种作为主结果，另一种作为 control。

文档、图表、结论中必须显式说明 anchor 是否计入 rate。

---

## 7. 二次校验

- [ ] History-only decoder 无法通过 shortcut 获得 \(X_L\)；
- [ ] normalization 没有泄漏 absolute level；
- [ ] Anchor-only 参数量已控制；
- [ ] 各 variant decoder 容量一致；
- [ ] 明确后续 IB 中 anchor 是否免费；
- [ ] 不把 ablation contribution 当作严格加法分解。

---

# Stage 5：Variational Information Bottleneck

## 1. 实验目的

从：

\[
\text{representation dimension}
\]

升级到：

\[
\text{variational coding rate}.
\]

研究：

> 在不显著损失 correction 的条件下，可以把 correction channel 的 variational rate 压到多低？

---

## 2. 理论表述规范

使用：

\[
q_\phi(Z|X)
\]

以及 prior：

\[
p(Z).
\]

有：

\[
\mathbb E_X
KL(q(Z|X)||p(Z))
=
I(X;Z)
+
KL(q(Z)||p(Z)).
\]

因此：

\[
\mathbb E KL
\ge I(X;Z).
\]

所以：

\[
\frac{\mathbb E KL}{\ln2}
\]

只能称为：

\[
\boxed{\text{variational rate}}
\]

或：

\[
\boxed{\text{upper bound on information rate}}
\]

不得直接称为“真实 mutual information bits”。

---

## 3. Capacity 选择

### Stage 2 GO

使用：

\[
r=r_{95}.
\]

### Stage 2 NO-GO

必须使用：

\[
r=r_{\text{full-correction}}
\]

即能够恢复完整 correction 的最小容量。

若无低 rank 可恢复，则使用 full-capacity latent。

原则：

> Stage 5 只能测试 information-rate compression，不能让 representation capacity 成为前置瓶颈。

---

## 4. 实验设定

\[
q_\phi(Z|X)
=
\mathcal N(
\mu_\phi(X),
\sigma_\phi^2(X)
).
\]

\[
Z=\mu+\sigma\odot\epsilon.
\]

损失：

\[
\mathcal L
=
\mathcal L_{\mathrm{forecast}}
+
\beta
D_{KL}
[
q_\phi(Z|X)
\Vert
p(Z)
].
\]

默认：

\[
p(Z)=\mathcal N(0,I).
\]

---

## 5. \(\beta\) 扫描

预注册：

\[
\beta
\in
\{
0,
10^{-6},
10^{-5},
10^{-4},
10^{-3},
10^{-2},
10^{-1}
\}.
\]

若 transition region 位于 grid 间，可根据 validation 增加 refinement points。

新增点必须标记：

> refinement run

---

## 6. Noise-only control

必须包含：

\[
\beta=0
\]

但保留 stochastic sampling。

目的：

> 区分 stochastic noise regularization 和显式 rate penalty。

---

## 7. Rate 定义

\[
R_{\mathrm{var}}
=
\frac{
\mathbb E[D_{KL}(q||p)]
}{
\ln2
}.
\]

推荐报告：

- bit/sample；
- bit/channel；
- bit/prediction-step。

但必须统一说明：

\[
R_{\mathrm{var}}
\]

是 variational upper bound，不是真实 MI。

---

## 8. 核心曲线

绘制：

\[
CR(R_{\mathrm{var}})
\]

以及：

\[
D(R_{\mathrm{var}}).
\]

定义：

\[
R_{95}
=
\min
\{
R_{\mathrm{var}}:
CR\ge0.95
\}.
\]

推荐同时记录：

\[
R_{99}.
\]

---

## 9. 需要填写的结果

| \(\beta\) | Variational Rate | MSE | MAE | CR | \(\Delta D\) | KL Std | Latent Active Dims |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | | | | |
| | | | | | | | |

---

## 10. GO 门槛

重点不是单点 MSE，而是是否存在明确 compression–correction Pareto frontier。

推荐 operational rule：

在：

\[
CR\ge0.95
\]

条件下，相比弱约束模型 variational rate 能显著下降。

例如：

\[
R_{95}
\le0.5R_{\text{weak}}
\]

可视为 Strong GO 的一个参考标准。

---

## 11. Fallback

Stage 5 NO-GO 时 Stage 6 仍继续。

Stage 6 使用同一 stochastic architecture，只改变：

\[
p(Z)
\rightarrow
p(Z|H_P).
\]

理由：

> unconditional prior 压缩失败，并不能排除 PhaseFormer-conditioned side information 可以有效消除重复信息。

---

## 12. 二次校验

- [ ] KL 使用完整 evaluation set 平均；
- [ ] nat 转 bit 使用 \(/\ln2\)；
- [ ] 文档只称 variational rate / MI upper bound；
- [ ] 未把 latent dimension 等同于 bitrate；
- [ ] \(\beta=0\) stochastic control 已完成；
- [ ] posterior collapse 已检查；
- [ ] CR 与 rate 来自同一 checkpoint；
- [ ] Stage 2 NO-GO 时未强行使用低 rank；
- [ ] anchor 是否计入 rate 已明确。

---

# Stage 6：Conditional Information Bottleneck

## 1. 实验目的

验证：

> PhaseFormer 表征作为 side information 后，是否可以进一步减少 NLinear 为保存相同 correction 所需的增量 variational rate。

核心比较：

\[
R_{\mathrm{IB}}(CR)
\]

与：

\[
R_{\mathrm{CIB}}(CR).
\]

---

## 2. 实验设定

从固定 PhaseFormer 中提取：

\[
H_P.
\]

使用：

\[
q(Z|X,H_P)
\]

以及：

\[
p(Z|H_P).
\]

损失：

\[
\mathcal L
=
\mathcal L_{\mathrm{forecast}}
+
\beta
D_{KL}
[
q(Z|X,H_P)
\Vert
p(Z|H_P)
].
\]

---

## 3. 理论表述规范

conditional KL：

\[
\mathbb E
KL[
q(Z|X,H_P)
||
p(Z|H_P)
]
\]

应视为：

\[
I(Z;X|H_P)
\]

的 variational upper bound。

不得宣称：

> 已精确测量真实 conditional mutual information。

可以表述：

> PhaseFormer representation 作为 side information，降低了保存相同 correction 所需的 variational coding rate。

---

## 4. Control

必须比较：

### Ordinary IB

\[
p(Z)=N(0,I).
\]

### Conditional-prior IB

\[
p(Z|H_P).
\]

必须保持：

- latent capacity；
- decoder；
- \(\beta\) grid；
- training budget；
- optimizer；
- seed；

一致。

---

## 5. 参数量公平性

Conditional prior 增加参数。

必须报告：

- conditional prior params；
- total params；
- FLOPs。

建议增加 parameter-matched ordinary IB control。

---

## 6. 需要填写的结果

| Model | \(\beta\) | Variational Rate | MSE | MAE | CR | \(\Delta D\) | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| IB | | | | | | | |
| Conditional IB | | | | | | | |

关键：

\[
R_{95}^{IB}
\]

与：

\[
R_{95}^{CIB}.
\]

---

## 7. GO 门槛

推荐满足至少一项：

### Matched CR

\[
R_{95}^{CIB}
\le0.8R_{95}^{IB}.
\]

### Matched Rate

在相同 variational rate 下：

\[
CR_{CIB}>CR_{IB}.
\]

---

## 8. 可支持的结论

若 GO：

\[
\boxed{
\text{PhaseFormer side information 可降低 NLinear 保存相同 correction 所需的 variational rate。}
}
\]

可以进一步支持：

> NLinear 中一部分历史信息与 PhaseFormer 已有表征重复，而真正不可替代的是较少的增量 correction information。

但不得说：

> 已精确测得 \(I(Z;X|H_P)\)。

---

## 9. Fallback

若 Stage 6 NO-GO：

Stage 7 使用 Stage 5 Ordinary IB。

若 Stage 5 也 NO-GO：

Stage 7 使用 Stage 2 structural bottleneck 的最小容量需求：

\[
r_{\mathrm{required}}
\]

作为 information-demand proxy。

因此 reliability hypothesis 仍可继续独立测试。

---

## 10. 二次校验

- [ ] \(H_P\) 来自固定 PhaseFormer；
- [ ] \(H_P\) 不使用 future target；
- [ ] conditional prior 无法看到 \(Y\)；
- [ ] ordinary / conditional IB 使用相同 \(\beta\) grid；
- [ ] 参数量差异已报告；
- [ ] matched-rate 比较规则预先确定；
- [ ] 文档只使用 variational upper-bound 表述；
- [ ] 没有把更小 KL 直接等同于更小真实 MI。

---

# Stage 7：验证周期可靠性是否独立关联 correction information demand

## 1. 实验目的

验证：

\[
\boxed{
\text{周期可靠性是否在控制样本难度后，仍与 correction demand 独立相关。}
}
\]

这是非常关键的机制阶段。

简单观察：

\[
\text{low reliability}
\Rightarrow
\text{large correction gain}
\]

并不足够，因为低可靠性样本可能本来就更难。

---

## 2. Reliability 定义

定义：

\[
s(X)\in[0,1].
\]

必须只由历史 \(X\) 或 PhaseFormer 对历史的内部表征得到。

候选包括：

- ACF periodicity；
- spectral concentration；
- cycle-to-cycle similarity；
- phase consistency；
- PhaseFormer internal confidence。

主 reliability metric 必须在 test 前确定。

---

## 3. 基础分组分析

按照 reliability 分 quartile：

\[
Q_1,Q_2,Q_3,Q_4.
\]

其中：

\[
Q_1
\]

表示周期最不可靠。

分别计算：

\[
CR(R|Q_k)
\]

以及：

\[
R_{95}(Q_k).
\]

如果无 IB，则使用：

\[
r_{95}(Q_k)
\]

或最小 structural capacity。

---

## 4. 必须控制 baseline difficulty

定义 PhaseFormer sample-wise error：

\[
e_{P,i}.
\]

虽然该量使用未来 \(Y\)，但只允许用于**离线机制分析**，绝不允许进入部署模型或 reliability score。

推荐回归：

\[
G_i
=
\alpha
+
\beta_1s_i
+
\beta_2e_{P,i}
+
\beta_3Var(X_i)
+
\text{dataset controls}
+
\text{horizon controls}
+
\epsilon_i.
\]

关注：

\[
\beta_1.
\]

也可对 information demand 做：

\[
R_i
=
\alpha
+
\gamma_1s_i
+
\gamma_2e_{P,i}
+\cdots.
\]

---

## 5. Matched-difficulty analysis

额外构造：

> PhaseFormer error 相近，但 reliability 不同的样本对 / 样本组。

在 matched baseline difficulty 下比较：

- correction gain；
- required rate；
- required rank。

若差异仍存在，说明 reliability 的解释力不是纯粹由样本难度造成。

---

## 6. 需要填写的结果

| Reliability Bin | Sample # | PhaseFormer MSE | Full Gain | \(R_{95}\) / \(r_{95}\) | CR @ Fixed Budget |
|---|---:|---:|---:|---:|---:|
| Q1 | | | | | |
| Q2 | | | | | |
| Q3 | | | | | |
| Q4 | | | | | |

另外必须报告：

- Spearman correlation；
- difficulty-controlled regression coefficient；
- matched-difficulty effect size。

---

## 7. GO 门槛

至少需要同时满足：

1. reliability 与 correction demand 存在预期趋势；
2. 控制 baseline difficulty 后关系仍存在；
3. 该趋势在多个 dataset / horizon 上重复出现。

典型目标：

\[
R_{95}(Q_1)>R_{95}(Q_4).
\]

同时：

\[
G(Q_1)>G(Q_4).
\]

---

## 8. 可支持的结论

若 GO：

\[
\boxed{
\text{周期可靠性与 correction information demand 具有独立统计关联，而不仅仅是样本更难。}
}
\]

仍不得直接声称：

> 周期可靠性因果决定信息需求。

这是关联机制证据，而非严格因果识别。

---

## 9. Fallback

若 Stage 7 NO-GO：

Stage 8 仍可以运行。

但 Stage 8 只能表述为：

> reliability-conditioned allocation 的经验性能实验。

不得表述为：

> 基于已经验证的 reliability-information mechanism 设计 adaptive IB。

---

## 10. 二次校验

- [ ] reliability 不使用未来 \(Y\)；
- [ ] reliability threshold 不根据 test 调整；
- [ ] baseline difficulty 已控制；
- [ ] regression 和 matched analysis 至少完成一种，推荐两种都做；
- [ ] bin sample size 足够；
- [ ] dataset / horizon effect 已控制；
- [ ] 至少有一个替代 reliability metric robustness test；
- [ ] 不把统计相关写成因果结论。

---

# Stage 8：Adaptive Information Bottleneck

## 1. 实验目的

验证：

> 根据样本周期可靠性动态分配 correction variational rate，是否优于所有样本使用固定预算。

---

## 2. 实验设定

### Fixed IB

\[
\beta=\beta_0.
\]

### Adaptive IB

\[
\beta(X)=f(s(X)).
\]

基础设计：

\[
s(X)\uparrow
\Rightarrow
\beta(X)\uparrow
\]

即周期越可靠，压缩越强。

---

## 3. 渐进实验

### E8-A：Handcrafted monotonic mapping

固定：

\[
f
\]

不学习。

只验证：

> adaptive allocation principle。

### E8-B：Learnable mapping

只有 E8-A 完成后再增加 learnable allocator。

这样可以区分：

- adaptive principle；
- allocator network capacity。

---

## 4. 最关键公平性：Matched Average Rate

必须让：

\[
E[R_{\mathrm{adaptive}}]
\approx
E[R_{\mathrm{fixed}}].
\]

在相同 average variational rate 下比较：

\[
D_{\mathrm{adaptive}}
\]

与：

\[
D_{\mathrm{fixed}}.
\]

同时推荐反向比较：

### Matched Error

在相同 MSE 下比较：

\[
R_{\mathrm{adaptive}}
\]

与：

\[
R_{\mathrm{fixed}}.
\]

---

## 5. 核心结果不是单点，而是 Pareto Frontier

必须比较：

\[
D_{\mathrm{fixed}}(R)
\]

与：

\[
D_{\mathrm{adaptive}}(R).
\]

若 adaptive 有效，应体现为：

\[
D_{\mathrm{adaptive}}(R)
<
D_{\mathrm{fixed}}(R)
\]

在相同平均 rate 下成立。

或：

\[
R_{\mathrm{adaptive}}(D)
<
R_{\mathrm{fixed}}(D).
\]

---

## 6. 需要填写的结果

| Model | Avg Variational Rate | MSE | MAE | CR | Q1 Rate | Q4 Rate | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed IB | | | | | | | |
| Adaptive-Handcrafted | | | | | | | |
| Adaptive-Learned | | | | | | | |

---

## 7. GO 门槛

至少满足一项：

### Matched Rate

\[
D_{\mathrm{adaptive}}
<
D_{\mathrm{fixed}}.
\]

### Matched Error

\[
R_{\mathrm{adaptive}}
<
R_{\mathrm{fixed}}.
\]

并要求跨 seed / dataset / horizon 有稳定趋势。

---

## 8. Fallback

Stage 8 NO-GO 后 Stage 9 仍继续。

理由：

> Adaptive IB 失败可能意味着问题本质上只是 branch selection，而不是信息预算问题。

Stage 9 正是检验该竞争性解释。

---

## 9. 二次校验

- [ ] Fixed / Adaptive average rate 已匹配；
- [ ] allocator 不访问 \(Y\)；
- [ ] handcrafted 与 learned allocator 分开报告；
- [ ] Q1 / Q4 实际 rate 符合设计逻辑；
- [ ] adaptive 改进不是额外参数量造成；
- [ ] 至少完成 matched-rate；
- [ ] 推荐同时完成 matched-error；
- [ ] 结论基于 Pareto frontier，而不是单点 MSE。

---

# Stage 9：Adaptive IB 与 Simple Gate 的竞争性解释

## 1. 实验目的

检验一个更简单的假设：

> NLinear 并不需要“动态信息预算”，只需要根据样本决定是否启用 / 使用多少权重即可。

这是 Adaptive IB 的关键竞争性解释。

---

## 2. Gate Baseline

定义：

\[
\hat Y
=
\hat Y_P
+
g(X)\Delta Y.
\]

其中：

\[
g(X)\in[0,1].
\]

至少包含：

### Reliability Gate

\[
g(X)=f(s(X)).
\]

### Learnable Gate

从历史输入学习 \(g\)。

可额外测试 hard gate。

---

## 3. 必须比较的模型

1. PhaseFormer；
2. Stage 0 PhaseFormer + NLinear；
3. Fixed IB；
4. Adaptive IB；
5. Reliability Gate；
6. Learnable Gate。

---

## 4. 公平性

同时报告：

- MSE；
- MAE；
- CR；
- variational rate；
- branch usage；
- parameter count；
- FLOPs；
- correction magnitude。

Gate value 不能被直接解释成 bitrate。

---

## 5. 需要填写的结果

| Method | MSE | MAE | CR | Variational Rate | Branch Usage | Params | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 | | | | | | | |
| Fixed IB | | | | | | | |
| Adaptive IB | | | | | | | |
| Reliability Gate | | | | | | | |
| Learnable Gate | | | | | | | |

---

## 6. 结论解释

### 如果 Gate ≈ Adaptive IB

说明更简单的：

\[
\text{branch selection / scaling}
\]

可能已经足够。

此时 IB 可以保留为分析工具，但不宜强调为必要性能组件。

### 如果 Adaptive IB 稳定优于 Gate

特别是在 matched resource / matched usage 下：

\[
\boxed{
\text{NLinear complementarity 更像连续信息预算问题，而不只是旁路开关问题。}
}
\]

---

## 7. GO 门槛

Adaptive IB 必须在：

- matched compute；
- matched branch usage；
- 或合理资源约束；

下优于 Gate，才能支持“信息预算优于简单 gating”。

---

## 8. Fallback

无论 Gate 还是 IB 胜出，Stage 10 均继续。

Stage 10 至少保留：

- 最佳简单模型；
- 最佳 IB 模型；

分别进行 frozen / joint 比较。

---

## 9. 二次校验

- [ ] Gate 和 IB 使用相同 reliability signal；
- [ ] gate 参数量已报告；
- [ ] hard / soft gate 区分清楚；
- [ ] gate value 未被解释为信息量；
- [ ] matched-resource 比较成立；
- [ ] Adaptive IB 优势不是额外网络容量造成。

---

# Stage 10：Joint Fine-tuning

## 1. 实验目的

前面 Stage 1–9 主要用于：

\[
\text{mechanism identification}.
\]

Stage 10 才评估：

> PhaseFormer 与 correction branch 联合优化能否进一步提升最终预测性能。

---

## 2. 实验设定

### Frozen Mechanistic Model

\[
P\text{ frozen}.
\]

### Joint Performance Model

\[
P+N
\]

共同训练。

---

## 3. 重要解释规则

Joint training 后：

- PhaseFormer 可能接管原本属于 NLinear 的信息；
- NLinear correction 可能重新分工；
- 原有 rate / correction mechanism 可能变化。

因此：

\[
\boxed{
\text{Joint model 不能替代 frozen model 的机制证据。}
}
\]

若 joint model 更强，但机制发生明显变化，则最终必须区分：

### Mechanistic Model

Frozen。

### Performance Model

Joint-trained。

---

## 4. 机制重测

Joint training 后重新计算：

\[
CR(R)
\]

以及：

\[
R_{95}.
\]

同时检查：

- PhaseFormer representation drift；
- correction magnitude；
- NLinear branch usage；
- PhaseFormer 是否接管被压掉的信息。

---

## 5. 需要填写的结果

| Model | Frozen / Joint | MSE | MAE | CR | Variational Rate | \(R_{95}\) | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| Best simple model | Frozen | | | | | | |
| Best simple model | Joint | | | | | | |
| Best IB model | Frozen | | | | | | |
| Best IB model | Joint | | | | | | |

---

## 6. GO 判定

如果 joint training：

- 获得稳定性能提升；
- 且未完全破坏前面的 compression / correction mechanism；

则进入最终 performance model。

若 joint training 涨点但机制发生根本变化：

必须同时保留：

- frozen mechanistic result；
- joint performance result。

二者不得混写为同一个机制结论。

---

## 7. 二次校验

- [ ] frozen 模型完整保留；
- [ ] joint 结果未覆盖机制实验；
- [ ] joint 后重新测量 rate–correction curve；
- [ ] representation drift 已检查；
- [ ] correction branch 是否塌缩已检查；
- [ ] 最终报告明确区分机制模型与性能模型。

---

# 5. 阶段继承与 Fallback 总表

| 当前 Stage | GO 后主线继承 | NO-GO 后 Fallback |
|---|---|---|
| Stage 1 Residual | Residual NLinear | Stage 0 Original NLinear |
| Stage 2 Low-rank | \(r_{95}\) | Stage 5 使用 \(r_{\text{full-correction}}\)；Stage 3 允许 probe rank |
| Stage 3 Structured statistics | Best structured + learned | Learned projection |
| Stage 4 Anchor | 根据结果确定主 rate accounting | Anchor-exempt 与 Full-channel 至少保留一种 control |
| Stage 5 VIB | Best variational-rate operating region | 同一 stochastic architecture 进入 CIB |
| Stage 6 Conditional IB | Conditional IB | Ordinary IB；若 IB 也失败，则 structural demand proxy |
| Stage 7 Reliability relation | Reliability-guided mechanism | Stage 8 仅作为 empirical adaptive experiment |
| Stage 8 Adaptive IB | Best adaptive configuration | Fixed IB |
| Stage 9 Gate comparison | Winning mechanism + competitor | 两者均保留进入 final FT |
| Stage 10 Joint | Final performance model | Frozen mechanistic model |

---

# 6. 每阶段统一实验报告模板

## A. Hypothesis

本阶段唯一验证的假设：

> ______________________

## B. Independent Variable

本阶段唯一主动改变：

> ______________________

## C. Controlled Variables

固定：

> ______________________

## D. Control

> ______________________

## E. Treatment

> ______________________

## F. Primary Metric

> ______________________

## G. Secondary Metrics

> ______________________

## H. Operational GO Threshold

> ______________________

## I. Cross-seed Consistency

> ______________________

## J. Cross-dataset / Horizon Consistency

> ______________________

## K. Observed Result

> ______________________

## L. Decision

- [ ] GO
- [ ] WEAK-GO
- [ ] NO-GO
- [ ] INCONCLUSIVE

## M. Supported Conclusion

> ______________________

## N. Explicitly Unsupported Stronger Conclusion

> ______________________

## O. Mainline Configuration

> ______________________

## P. Fallback Configuration

> ______________________

## Q. Unexpected Observation

> ______________________

## R. Protocol Deviation

> ______________________

## S. Next Stage

固定：

> CONTINUE

除非发现：

- 代码错误；
- 数据泄漏；
- baseline 不可复现；
- protocol invalid。

---

# 7. 结论边界总表

| Stage | 可以支持的最小结论 | 不得越界声称 |
|---|---|---|
| 1 | NLinear 有效贡献可表述为 correction | NLinear 天生只建模 residual |
| 2 | 非-anchor动态 correction 存在低维充分表示 | 整个 NLinear 只需要 \(r\) 维 |
| 3 | 有效子空间与低阶趋势 / 低频统计一致 | 已经得到真实 IB |
| 4 | anchor 与动态历史贡献可以分离 | anchor 不消耗信息 |
| 5 | correction 可在低 variational rate 下保存 | KL 等于真实 MI |
| 6 | PhaseFormer side information 可降低 variational rate | 已精确得到 \(I(Z;X|H_P)\) |
| 7 | reliability 与 correction demand 在控制 difficulty 后独立相关 | reliability 因果决定需求 |
| 8 | adaptive allocation 改善 rate–error frontier | adaptive 一定优于 Gate |
| 9 | 可排除或接受“简单 gating 足够”的解释 | — |
| 10 | 联合优化可提高最终性能 | joint 模型自动继承 frozen 机制解释 |

---

# 8. 最终希望建立的完整科学叙事

若所有关键阶段均得到支持，则可以形成如下逐层结论：

首先，PhaseFormer 负责主要的周期 / phase structure forecasting。

在部分历史周期结构不可靠的情况下，PhaseFormer 会留下系统性预测残差。

NLinear 的主要有效贡献可以被重新参数化为：

\[
\Delta\hat Y
\]

形式的 correction signal。

进一步发现，该 correction 并不需要完整历史自由度，而可以由低容量动态表示保存。

若 Stage 3 进一步成立，则说明这些有效方向与：

- level-relative trend；
- slope；
- curvature；
- low-frequency structure；

等低阶统计一致。

Stage 4 将 level anchor 与动态历史显式分离。

Stage 5 进一步说明：

> 保存 correction 不仅不需要完整维度，而且只需要有限的 variational coding rate。

Stage 6 若成立，则说明：

> 利用 PhaseFormer 已有 representation 作为 side information 后，NLinear 只需传递更少的增量 correction information。

Stage 7 若在控制 baseline difficulty 后仍成立，则可以进一步支持：

> correction information demand 与历史周期可靠性具有独立关系。

Stage 8 将这一观察转化为：

\[
\text{reliability-aware information allocation}.
\]

Stage 9 再以 Gate 作为竞争性解释检验：

> 问题究竟是“旁路开关”，还是“连续信息预算”。

最终可能形成的核心科学结论为：

\[
\boxed{
\text{当 phase-based forecaster 的周期依据变得不可靠时，}
\text{并不一定需要另一个完整预测器重新理解整个历史；}
\text{只需补充少量、非冗余的 correction information。}
}
\]

Adaptive IB 只是该观察的算法化实现：

\[
\boxed{
\text{按需分配 correction information，而非始终保留完整旁路容量。}
}
\]

---

# 9. 最终执行原则

整个实验严格按照：

\[
S1
\rightarrow
S2
\rightarrow
S3
\rightarrow
S4
\rightarrow
S5
\rightarrow
S6
\rightarrow
S7
\rightarrow
S8
\rightarrow
S9
\rightarrow
S10
\]

执行。

每个 Stage 的结果只决定：

\[
\boxed{\text{下一阶段采用哪一种 Mainline Configuration}}
\]

而不决定：

\[
\boxed{\text{下一阶段是否执行}}
\]

因此：

\[
\boxed{
NO\text{-}GO
\Rightarrow
\text{切换 Fallback，而不是终止探索。}
}
\]

该规则的目的，是保证每个科学问题都能被独立验证，并防止前置假设失败后阻断后续潜在发现。
