# PCTF v3 锚点漂移归因与修复复测计划

## 一句话目的

这轮不再盲目更换融合网络，而是依次验证四件事：失败是否来自 A2 锚点被联合训练破坏、ICPT
是否学错了目标、门控是否学到了“何时修正真正有利”、以及 `H=period` 时 level 通路是否因
数学投影退化为零。本文只记录方案和已完成代码，**尚未运行任何训练，也没有新结果**。

## 已知证据与待验证归因

上一轮 132 个 validation-only 筛选中，最佳 `pctf_anchor_mlp` 相对 A2 的宏平均比为
1.001131，仅 4/12 个 setting 双指标改善，最差比为 1.021385；但 MLP 相对逐周期标量系数的
嵌套对照略好（0.999924）。这支持“历史证据可能有用，但训练对象或优化过程有问题”，不能
直接证明模型结构无效。

代码审计进一步得到四个可检验问题：

1. **锚点只在初始化时安全**：旧候选零初始化时严格等于 A2，但之后 PhaseFormer、NLinear、
   RCRF 和 ICPT 一起使用同一学习率更新，没有约束内部 A2 继续保持原性能。
2. **ICPT 辅助目标不匹配最终职责**：旧辅助损失让 ICPT 拟合完整真实序列的 level/shape，
   最终融合却只使用 `ICPT-NLinear` 和 `ICPT-PhaseFormer` 的创新；它没有直接学习 A2 尚未解释
   的残差。
3. **门控只靠最终预测损失间接学习**：历史 evidence MLP 没有明确的边际收益监督，因此可能
   学到修正幅度，却没有学到修正方向和适用样本。
4. **单周期 level 零空间**：旧定义先减去整个 horizon 的周期均值。当 `H=period` 时只有一个
   未来周期，level 分量恒为零；ETTm2-H96/period96 因而根本无法测试 ICPT 的周期级水平能力。

上述结论目前是结构归因，不是实验结论；下面的阶梯式复测用相邻消融逐项证伪。

## 修复后的统一模型

模型仍是一个端到端 PhaseFormer checkpoint，不是多个已训练模型的 ensemble。完整 A2
`A(x)` 由 PhaseFormer 相位通路、LFF-NLinear 轨迹通路和 RCRF 构成；ICPT 只提出两个有界创新：

\[
D_L=L_C-\operatorname{sg}(L_T),\qquad
D_S=S_C-\operatorname{sg}(S_P),
\]

\[
\hat y=A(x)+\beta_L D_L+\beta_S D_S,
\qquad |\beta_L|,|\beta_S|\le 0.25.
\]

`sg` 表示 stop-gradient。它不会改变前向数值，只阻止 ICPT 的残差辅助损失借参考分支“移动
靶子”。系数保持 tanh 有界、零初始化，所以加载 matched A2 后初始输出仍逐点等于 A2。

### 残差监督

令真实锚点残差为 `R = y - sg(A)`，将它投影到 level/shape 子空间：

\[
\mathcal L_{ICPT}=0.05\,\ell(D_L,\Pi_LR)
                    +0.05\,\ell(D_S,\Pi_SR).
\]

这使 ICPT 学习“补 A2 的缺口”，而不是再独立拟合一次完整预测。

### 锚点安全的联合优化

最终候选不是冻结模型。联合阶段增加内部锚点损失
`L_anchor = ℓ(A,y)`，并给锚点参数使用主学习率的 0.1 倍；ICPT 与融合器仍使用原学习率。
冻结版本只用于因果诊断“锚点漂移占多少”，不具备论文候选资格。

### 边际收益门控

对每个样本、变量和未来周期，根据训练标签计算一个 stop-gradient 的最优有界系数：

\[
\beta^*=\operatorname{clip}
\left(\frac{\langle D,R\rangle}{\lVert D\rVert_2^2+\epsilon},-0.25,0.25\right).
\]

evidence MLP 的实际系数用 Smooth-L1 拟合 `β*`，权重为 0.05。标签只在训练损失中使用；推理
仍只读取历史序列的相位可靠性、漂移、rolling regret 和分支分歧，不使用未来信息。

### 单周期 level 修复

旧 `horizon_centered` level 只保留周期间相对均值。新 `history_referenced` 定义保留
`ICPT-NLinear` 的直接周期均值差，并拆成：

- 多周期之间的相对 level：仍投影为 horizon 零均值；
- 全局 level：仅允许最多 0.05 的独立小信任域。

因此 `H=period` 时 level 不再恒为零，同时不会让 ICPT 无约束地覆盖 NLinear 的绝对水平。

## 阶梯式候选与因果问题

| preset | A2 是否冻结 | ICPT 目标 | gate 监督 | level 定义 | 回答的问题 |
|---|---|---|---|---|---|
| `pctf_anchor_mlp` | 否 | 完整序列 | 无 | 旧 | 复现当前联合训练控制 |
| `pctf_anchor_diag_frozen_absolute` | 是 | 完整序列 | 无 | 旧 | 只消除锚点漂移后是否改善 |
| `pctf_anchor_diag_frozen_residual` | 是 | A2 残差 | 无 | 旧 | 残差目标是否比绝对目标合理 |
| `pctf_anchor_repair_joint_residual` | 否，0.1× LR + anchor loss | A2 残差 | 无 | 旧 | 锚点安全联合训练能否保留冻结收益 |
| `pctf_anchor_repair_joint_marginal` | 否，0.1× LR + anchor loss | A2 残差 | 有 | 旧 | 问题是否主要在门控学习 |
| `pctf_anchor_repair_full` | 否，0.1× LR + anchor loss | A2 残差 | 有 | 新 | 修复单周期零空间是否带来额外收益 |

冻结诊断也只保存一个 checkpoint；它加载一次 A2 参数后冻结内部锚点、训练新增 composer，推理
不加载第二个模型。但论文方法选择只允许三个 `joint_*` 候选，冻结行不能作为最终方法。

## 复测矩阵

lookback 固定 720，PhaseFormer period 固定 24，训练集 30%，最多 12 epoch，Huber loss，
seeds 为 2021/2022，仅在 validation 比较。ICPT period 沿用上一轮已冻结选择：

| setting | ICPT period | 选择原因 |
|---|---:|---|
| ETTh2-H96/H192 | 48 | Stage P 已冻结 |
| ETTm2-H96/H192 | 96 | Stage P 已冻结，并直接覆盖单周期零空间 |
| Weather-H192 | 24 | Stage P 已冻结，覆盖高维气象序列 |
| Electricity-H96 | 12 | Stage P 已冻结，覆盖高维负荷序列 |

先运行 12 个 matched A2（6 settings × 2 seeds），再运行 72 个候选（6 × 2 × 6）。每个候选
必须从同 setting、同 seed 的 best-validation A2 checkpoint 初始化。所有命令强制 CUDA；
汇总器拒绝 test 字段、CPU 结果、环境/commit 混用、缺失/重复任务和非零初始锚点误差。

## 记录指标与判断规则

除最终 validation MSE/MAE 外，代码新增以下内部指标：

- `val_anchor_mse/mae`：候选训练后内部 A2 单独输出的性能；
- `val_*_ratio_vs_internal_anchor`：加入 ICPT 修正究竟帮助还是伤害当前锚点；
- `val_update_rms`：融合实际修正幅度；
- `val_confidence_regret_corr`：历史置信度与真实修正收益的相关性；
- `val_coefficient_regret_corr`：实际修正系数绝对值与真实收益的相关性。

预注册判据：

| 假设 | 支持条件 | 反证含义 |
|---|---|---|
| H1 冻结诊断有效 | 冻结行内部 anchor/A2 误差比在 `1±1e-8` | checkpoint 配对或评估实现错误，停止分析 |
| H2 目标错配存在 | frozen-residual 宏平均优于 frozen-absolute | 若不优于，残差目标不是主要瓶颈 |
| H3 漂移被控制 | joint-residual 内部锚点最差比 ≤1.01 | 若失败，0.1× LR/anchor loss 仍不够安全 |
| H4 gate 学习有问题 | joint-marginal 宏平均优于 joint-residual，且系数—收益相关性提高 | 若失败，不继续堆叠复杂 gate |
| H5 可进入正式候选 | full 宏平均/A2 ≤0.998，最差 ≤1.01，至少 8/12 行双指标改善 | 否则不运行 test，不声称超过 A2/Golden |

H5 只允许提出下一阶段正式复核，不自动授权读取 test。若以后运行全量正式实验，必须重新和
`docs/PhaseFormer_gold_standard.md` 的同 setting 金标准比较；本轮 matched A2 仅用于归因。

## 待填结果表

| 候选 | 宏平均/A2 | 最差/A2 | 双指标改善行 | 内部锚点/A2 | fused/内部锚点 | update RMS | 系数—收益相关 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| current-control | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| frozen-absolute | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| frozen-residual | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| joint-residual | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| joint-marginal | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| full-repair | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 代码与命令

- 模型：`src/models/anchored_phase_cycle_fusion.py`
- 损失、冻结与分组学习率：`src/models/PhaseFormer.py`
- presets：`src/models/phaseformer_presets.py`
- 单 run 入口和内部指标：`scripts/search_phaseformer.py`
- 成对复测与汇总：`scripts/run_pctf_anchor_attribution.py`

```bash
# 只检查计划命令，不训练
.venv/bin/python scripts/run_pctf_anchor_attribution.py --stage anchors-dry
.venv/bin/python scripts/run_pctf_anchor_attribution.py --stage candidates-dry

# 将来获准后才执行；本次没有运行
.venv/bin/python scripts/run_pctf_anchor_attribution.py --stage anchors
.venv/bin/python scripts/run_pctf_anchor_attribution.py --stage candidates
.venv/bin/python scripts/run_pctf_anchor_attribution.py --stage summarize
```

预计输出到 `research_runs/pctf_anchor_attribution_v3/`。当前状态：实现与非训练测试完成，所有
结果单元格保持待填。
