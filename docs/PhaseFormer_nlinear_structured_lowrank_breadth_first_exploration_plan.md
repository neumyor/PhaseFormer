# PhaseFormer NLinear 结构化低秩宽度优先探索计划

> 状态：**探索计划，2026-09-15 起执行**。本计划采用
> **test-oriented exploratory search**：候选训练完成后读取 test，并使用
> test MSE/MAE 选择下一轮探索方向。所有参与选择的候选、setting、指标和
> 选择轨迹必须保留。结果属于 **test-set selection / exploratory evidence**，
> 不得表述为盲测、无偏泛化估计或正式 benchmark 提升。
>
> 默认单 seed `2021`，优先横向比较不同技术路线；除非用户另行要求，
> 探索阶段不做多 seed 复核。多 seed 只属于后续确认阶段。

## 1. 研究问题

已有 `pooled_lowrank` 实验说明：普通低秩因子化能够大幅减少 NLinear
残差头参数，但没有稳定精度增益。现有证据支持：

1. 中等压缩通常近似中性，深度压缩略有害；
2. 不存在跨 setting 统一最优秩；
3. 训练低秩头会重组任务相关子空间，不能简单解释为全秩权重的 SVD 截断；
4. D7 与容量分析显示，NLinear 主要修正跨周期水平状态，主导方向接近
   “近端加权的当前水平 -> 预测区间恒定位移”。

因此本轮不再只问“原始 `720 x H` 映射应该使用多大的 rank”，而问：

> NLinear 的低维性是否存在于更合适的时间坐标系中，例如周期段轴、
> segment basis、周期级状态或稀疏的近期周期选择？

本计划借鉴 SparseTSF 的周期/跨周期稀疏思想和 TimeBase 的
segment-level basis 思想，但目标是把它们作为 **PhaseFormer 的 NLinear
residual adapter**，不是直接复现或替换 PhaseFormer。

## 2. 核心假设

### H0：普通低秩主要是压缩，不一定是正确坐标系

`W ≈ UV` 在原始时间点坐标中限制参数量，却没有显式利用周期分段、跨周期
重复或近期周期重要性。

### H1：周期轴低秩比时间点轴低秩更适合 NLinear

令 `L=720`、周期长度为 `P`，将末值锚定后的历史重排为：

```text
X - X_last -> S ∈ R^(K × P),  K = ceil(L / P)
```

只在 `K` 个周期之间施加低秩约束，同时保留周期内 phase slot。

### H2：NLinear 更适合预测 phase residual 的低维状态

PhaseFormer 主干已经负责周期模板和相位形状。NLinear 分支更可能应该学习：

```text
未来 level shift + 少量周期 shape correction
```

而不是独立重建完整未来波形。

### H3：有用的稀疏性在周期块上

已有主导方向偏向最近 `24--168` 个时间步。因此比较近期周期选择、全周期
加权和周期轴低秩，而不是直接对 720 个时间点做非结构化 top-k。

### H4：basis 的多样性可能比 rank 数量更重要

少量互补 basis 可能比更多但冗余的方向更有用。正交约束若在相同参数量下
有效，说明瓶颈问题不仅是 rank 大小，也包括 basis 冗余。

## 3. 探索边界与统一协议

### 3.1 Test-oriented 规则

本计划明确允许 test-oriented 搜索：

- 每个候选训练完成后读取一次 test；
- test MSE/MAE 可以用于候选排序、路线淘汰和下一轮方向选择；
- 所有参与选择的候选必须保留，不得只保留胜者；
- 每份结果摘要必须写明 `test-set selection`；
- 探索结果不得称为盲测、无偏泛化估计或最终 benchmark；
- `PhaseFormer_gold_standard.md` 仍是固定参照，不因探索结果修改；
- 只改善一个指标时，必须写成“单指标改善”。

### 3.2 默认训练协议

| 项 | 默认值 |
|---|---|
| lookback | 720 |
| horizon | H96、H192；首轮优先 H96 |
| loss | Huber |
| max epochs | 30 |
| checkpoint | lowest validation loss |
| seed | 2021 |
| split / normalization | 沿用当前仓库协议 |
| PhaseFormer path | 接收完整 `X` |
| residual path | 末值锚定的 `X - X_last` |
| test | 每个候选只读取一次 |

必须包含两个对照：

- `direct_nlinear`：当前未因子化 NLinear；
- `pooled_lowrank(q=1/8)`：现有非结构化低秩控制。

所有结构化候选还应尽量加入参数量匹配的 shallow linear control，避免把
“参数减少”误判成“坐标系利用”。

### 3.3 探索 setting

**Pilot 四个 setting：**

| setting | 选择理由 |
|---|---|
| ETTh2-H96 | 已有局部低秩信号，短 horizon |
| ETTh2-H720 | 三 seed 复核中唯一稳定受益的 setting |
| ETTm2-H192 | 深压缩退化明显的反例 |
| Electricity-H336 | 原始权重谱不低秩，但训练低秩能重组出好解 |

**Expansion 三个 setting：**

| setting | 选择理由 |
|---|---|
| Weather-H192 | 周期较弱，检验结构方法的适用边界 |
| ETTh1-H96 | 第一轮低秩表现不稳定，作为压力测试 |
| ETTm1-H96 | 与 ETTm2 对比，检验同频率族迁移性 |

不按 dataset/horizon 临时发明不同机制。只在单一 setting 有效的方案，
记录为 setting-specific 信号，不晋级为统一路线。

## 4. 宽度优先搜索树

每轮只改变一个结构轴，先比较路线，再在胜者内部做小范围参数探索：

```text
Round 0: 控制、参数预算和实现校验
    |
    +-- A. 周期轴低秩 / SparseTSF-style
    +-- B. segment basis / TimeBase-style
    +-- C. level-shape 状态分解
    +-- D. 近期周期稀疏选择
    +-- E. separable / Kronecker map
    |
Round 1: 保留最多两条路线
Round 2: 结构消融与参数预算对照
Round 3: 固定路线扩展到未参与 pilot 的 setting
Round 4: 可选效率评估
```

### 路线 A：周期轴低秩

将 `X - X_last` 重排为周期段。对每个相位位置共享一个跨周期映射：

```text
S[:, 1:K, p] -> future_segments[:, 1:K_y, p]
W_period = decoder(rank) @ encoder(K)
```

首轮配置：

- `P ∈ {24, 96}`；
- `r_period ∈ {1, 2, 4}`；
- 全部周期，不做近期裁剪；
- 输出按未来 segment 展开；
- 保留 NLinear last-value anchor。

### 路线 B：segment basis / TimeBase-style

```text
B = E(S),       E: K -> R
F = D(B),       D: R -> K_y
```

将少量历史 basis 映射为未来 segments。首轮配置：

- `P ∈ {24, 96}`；
- `R ∈ {2, 4, 8}`；
- `lambda_orth ∈ {0, 0.01}`；
- 不加额外 smoothing；
- basis 在变量间共享。

### 路线 C：level-shape split

每个周期分为：

```text
level_k = mean_p S[k,p]
shape_k,p = S[k,p] - level_k
```

分别建模：

```text
level path:  K -> K_y
shape path:  K × P -> K_y × P
```

首轮只比较：

1. `level_only`；
2. `level_lowrank + shape_direct`；
3. `level_lowrank + shape_lowrank`。

### 路线 D：近期周期稀疏选择

在周期坐标中只使用：

```text
recent_1, recent_3, recent_7, recent_15, all_30
```

首轮只测试 hard recent window 和 fixed exponential weighting，不加入可学习
gate，避免把额外门控参数误认为稀疏结构收益。

### 路线 E：separable / Kronecker map

将时间点映射限制为少量周期轴与相位轴的可分离项：

```text
W ≈ sum_j A_j(period) ⊗ B_j(phase)
```

该路线表达能力和实现成本都更高。只在 A--D 没有明确胜者、但结构诊断
显示周期/相位交互仍重要时启动。首轮仅测试 `J ∈ {1, 2}`。

## 5. Round 0：控制、参数预算和实现校验

### 实验目的

确认所有后续路线的指标、参数量、计算量和输出形状可公平比较，排除训练
失败或参数规模差异造成的假信号。

### 实验设置

- Pilot 四个 setting；
- seed 2021；
- `phase_only`、`direct_nlinear`、generic `q=1/8`；
- parameter-matched shallow linear control；
- 每个候选训练后读取一次 test；
- 记录 head params、total params、MACs、训练时间、推理时间和显存峰值。

### 代填充表：Round 0

| setting | phase_only MSE/MAE | direct MSE/MAE | generic q=1/8 MSE/MAE | matched control MSE/MAE | 最低参数量 | 备注 |
|---|---|---|---|---|---:|---|
| ETTh2-H96 | — | — | — | — | — | — |
| ETTh2-H720 | — | — | — | — | — | — |
| ETTm2-H192 | — | — | — | — | — | — |
| Electricity-H336 | — | — | — | — | — | — |

### 退出规则

若控制不能在四个 setting 全部完成，先修复协议或实现，不比较路线。若 generic
low-rank 在全部 pilot setting 都明显优于 direct，则后续结构候选必须采用
不高于 generic low-rank 的参数预算。

## 6. Round 1：五条路线宽搜

### 实验目的

不追求最佳超参数，只回答哪一种低维坐标系最值得继续：

- 周期轴；
- segment basis；
- level-shape；
- 近期周期稀疏；
- 周期-相位可分离映射。

### 实验设置

每条路线先使用一个低成本代表配置，总计最多 `5 x 4 = 20` 个候选训练，
另加 Round 0 控制。默认 seed 2021、test-oriented。

| 路线 | 代表配置 |
|---|---|
| A | `P=24, r_period=4` |
| B | `P=24, R=4, lambda_orth=0.01` |
| C | `level_lowrank + shape_direct, P=24` |
| D | `recent_7, P=24, fixed_exp` |
| E | `J=1, P=24` |

`P=96` 作为第二批结构条件，只有当 `P=24` 路线在至少两个 pilot setting
有正向 test 信号时才追加。

### 代填充表：Round 1 路线矩阵

| 路线 | setting | MSE | MAE | delta MSE% vs direct | delta MAE% vs direct | 参数量 | MACs | gate | 双指标改善 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A 周期轴低秩 | ETTh2-H96 | — | — | — | — | — | — | — | — | — |
| A 周期轴低秩 | ETTh2-H720 | — | — | — | — | — | — | — | — | — |
| A 周期轴低秩 | ETTm2-H192 | — | — | — | — | — | — | — | — | — |
| A 周期轴低秩 | Electricity-H336 | — | — | — | — | — | — | — | — | — |
| B segment basis | ETTh2-H96 | — | — | — | — | — | — | — | — | — |
| B segment basis | ETTh2-H720 | — | — | — | — | — | — | — | — | — |
| B segment basis | ETTm2-H192 | — | — | — | — | — | — | — | — | — |
| B segment basis | Electricity-H336 | — | — | — | — | — | — | — | — | — |
| C level-shape | ETTh2-H96 | — | — | — | — | — | — | — | — | — |
| C level-shape | ETTh2-H720 | — | — | — | — | — | — | — | — | — |
| C level-shape | ETTm2-H192 | — | — | — | — | — | — | — | — | — |
| C level-shape | Electricity-H336 | — | — | — | — | — | — | — | — | — |
| D recent sparse | ETTh2-H96 | — | — | — | — | — | — | — | — | — |
| D recent sparse | ETTh2-H720 | — | — | — | — | — | — | — | — | — |
| D recent sparse | ETTm2-H192 | — | — | — | — | — | — | — | — | — |
| D recent sparse | Electricity-H336 | — | — | — | — | — | — | — | — | — |
| E separable | ETTh2-H96 | — | — | — | — | — | — | — | — | — |
| E separable | ETTh2-H720 | — | — | — | — | — | — | — | — | — |
| E separable | ETTm2-H192 | — | — | — | — | — | — | — | — | — |
| E separable | Electricity-H336 | — | — | — | — | — | — | — | — | — |

### 路线晋级规则

路线进入 Round 2，满足以下任一条件：

1. 至少 2/4 个 pilot setting 同时改善 test MSE 和 MAE；
2. 至少 3/4 个 setting 的双指标平均变化不差于 `-0.3%`，且平均参数量
   至少减少 5 倍；
3. 在 ETTh2-H720 或 Electricity-H336 上出现清晰双指标改善，同时其余
   setting 没有超过 `1.5%` 的双指标退化。

未达到条件的路线保留为负结果，不追加该路线搜索。

## 7. Round 2：晋级路线结构消融

### 实验目的

判断 Round 1 的 test 信号来自真实结构，还是来自偶然 rank、参数量或周期
长度选择。

### 实验设置

最多保留两条路线。每条路线最多六个消融条件，仍使用四个 pilot setting、
seed 2021、每个候选读取 test。

| 消融轴 | 条件 |
|---|---|
| 周期长度 | `P=24` vs `P=96` |
| 历史范围 | all vs recent-7 |
| rank/basis | low vs medium |
| 状态结构 | level-only vs level+shape |
| 正交约束 | off vs on |
| 周期伪结构控制 | aligned vs shifted/random segmentation |

shifted/random segmentation 是必要控制。若其与 aligned 周期相同，不能
声称模型利用了真实周期结构。

### 代填充表：Round 2

| route | ablation | setting | MSE | MAE | delta MSE% vs base | delta MAE% vs base | 参数量 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | ETTh2-H96 | — | — | — | — | — | — |
| — | — | ETTh2-H720 | — | — | — | — | — | — |
| — | — | ETTm2-H192 | — | — | — | — | — | — |
| — | — | Electricity-H336 | — | — | — | — | — | — |

### 结构性证据门槛

路线只有同时满足以下条件，才可称为“利用结构化低秩性”：

- 对齐周期优于 shifted/random segmentation；
- 在相近参数量下优于 generic low-rank，或在相同性能下参数显著更少；
- 至少一个结构轴移除后出现 test 退化；
- gate、level/shape 输出或 basis 权重显示分支确实被使用，而非近似关闭。

若只满足“压缩后性能还可以”，结论只能是“压缩有效”。

## 8. Round 3：Expansion test-oriented scan

### 实验目的

将最多一条统一路线扩展到未参与 pilot 的 setting，检查跨数据集迁移。

### 实验设置

- 固定 Round 2 选出的结构和配置；
- 不再按 setting 调 rank；
- ETTh1-H96、ETTm1-H96、Weather-H192；
- seed 2021；
- 同时运行 `direct_nlinear` 和 generic `q=1/8`；
- 每个候选读取一次 test。

### 代填充表：Expansion

| setting | direct MSE/MAE | generic q=1/8 MSE/MAE | structured MSE/MAE | delta MSE% vs direct | delta MAE% vs direct | 双指标改善 | 严重退化 |
|---|---|---|---|---:|---:|---|---|
| ETTh1-H96 | — | — | — | — | — | — | — |
| ETTm1-H96 | — | — | — | — | — | — | — |
| Weather-H192 | — | — | — | — | — | — | — |

### Expansion 决策

- 至少 2/3 setting 双指标改善，且 pilot 无明显失败：列为值得有限确认；
- 只有 1/3 改善：记录为 dataset-specific，不升级为统一机制；
- 0/3 改善：停止路线扩展；
- 不因单个 setting 的 test 最优而改变已冻结结构。

## 9. 可选 Round 4：效率测试

只有 Round 1--3 找到结构性路线后才执行。

| setting | model | head params | total params | MACs | peak memory | train time/epoch | inference time | MSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| — | direct_nlinear | — | — | — | — | — | — | — | — |
| — | generic q=1/8 | — | — | — | — | — | — | — | — |
| — | structured winner | — | — | — | — | — | — | — | — |

## 10. Test-oriented 选择记录

每轮必须保留完整选择轨迹，不能只保留最终胜者：

| round | candidate id | setting | test MSE | test MAE | baseline | action | reason | test-exposed |
|---|---|---|---:|---:|---|---|---|---|
| — | — | — | — | — | — | keep/drop/expand/diagnose/stop | — | yes |

排序优先级为：

1. test MSE 与 MAE 是否同时改善；
2. 跨 pilot setting 的双指标改善数量；
3. 相对 generic low-rank 的改善；
4. 参数量、MACs、训练/推理成本；
5. bad-case 是否符合预期机制。

## 11. 样本级错误分析

Round 1 结束后，每条路线最多选择 8 个案例，覆盖：

- structured 双指标显著优于 direct；
- structured 双指标显著退化；
- 周期水平偏移；
- 周期相位错位；
- 突变或高频噪声；
- generic 与 structured 分歧最大。

每个案例至少记录 setting、sample/origin/channel、时间定位、真实值、三种
预测、MSE/MAE、level/shape/basis 或周期权重、归因和下一步动作。

若不能导出样本级预测，路线不得晋级为机制创新；先补充非侵入式预测导出。

## 12. 预算与停止规则

### 默认最大预算

| 阶段 | 最大新增训练 |
|---|---:|
| Round 0 | 4 settings x 4 controls = 16 |
| Round 1 | 5 routes x 4 settings = 20 |
| Round 2 | 最多 2 routes x 6 ablations x 4 settings = 48 |
| Round 3 | 1 route x 3 settings + controls = 9 |
| Round 4 | 仅效率评估，不要求重新训练 |

实际执行采用早停：Round 1 未晋级的路线不进入 Round 2。

### 立即停止

- 某路线在四个 pilot setting 全部双指标退化；
- 结构化路线不优于 generic low-rank，且没有显著效率优势；
- 所有路线都无法超过 direct，且没有路线在参数匹配下保持性能；
- 收益只来自一个 setting，Expansion 失败；
- 继续搜索只剩 dataset/horizon-specific 调参。

## 13. 统一计算口径

相对 `direct_nlinear`：

```text
delta MSE% = (MSE_direct - MSE_candidate) / MSE_direct * 100
delta MAE% = (MAE_direct - MAE_candidate) / MAE_direct * 100
```

相对 Golden：

```text
delta MSE% = (MSE_golden - MSE_candidate) / MSE_golden * 100
delta MAE% = (MAE_golden - MAE_candidate) / MAE_golden * 100
```

任何最终摘要必须同时报告 test MSE、test MAE、相对 direct、相对 Golden、
seed、参数量和 test-set selection 边界。

## 14. 实验产物

每个实验批次统一写入 `research_runs/<experiment_id>/`，至少包含：

- `run.yaml`：候选、setting、训练协议和 test-oriented 选择说明；
- `results.csv`：candidate x setting 的 val/test 指标；
- `sample_errors.csv`：样本级误差；
- `selected_cases.npz`：不超过 8 个/路线的案例；
- `objective_error_analysis.md`；
- `objective_error_analysis.zip`；
- `figures/`：报告实际引用的图。

不得只保存最优 checkpoint 或最终表格。候选淘汰轨迹本身就是结果。

## 15. 推荐执行顺序

1. 完成 Round 0 控制和参数/MACs 统计；
2. 一次性横向测试 A--E 代表配置；
3. 只保留最多两条路线；
4. 做周期长度、近期范围、level/shape、正交约束和错位控制；
5. 用固定结构扩展到 ETTh1、ETTm1、Weather；
6. 只有出现跨 setting 价值，才考虑后续少量多 seed 确认；
7. 若无路线通过，保留“结构化低秩未被证实”的负结果。

## 16. 结论模板

### 结构路线成立

> 在明确披露 test-set selection 的单 seed 探索中，`<route>` 在 `<x/y>`
> 个 setting 上相对 `direct_nlinear` 同时改善 test MSE/MAE，并且在参数
> 匹配 generic low-rank 和错位周期控制下仍保留优势。该结果支持
> “`<coordinate system>` 能更有效利用 NLinear 的低维性”，但不构成
> 无偏泛化结论。

### 只有压缩成立

> 结构化候选没有稳定优于 generic low-rank，但在相近 test 误差下显著减少
> 参数和计算量。因此证据支持低秩压缩的工程价值，不支持其作为精度改进机制。

### 路线不成立

> 在本次 test-oriented 单 seed 探索范围内，`<route>` 未相对 direct 或
> generic low-rank 显示一致优势，停止继续扩展。该结果不足以支持将其用于
> PhaseFormer。

## 17. 依据文档

- `PhaseFormer_pooled_lowrank_nlinear_experiment.md`
- `PhaseFormer_joint_lowrank_rank_sweep_plan.md`
- `PhaseFormer_rank_sweep_conditioned_experiment.md`
- `PhaseFormer_lowrank_mechanism_analysis.md`
- `PhaseFormer_rank_capacity_and_data_property_report.md`
- `PhaseFormer_input_component_D7_internal_path_report.md`

本计划不修改 preset，也不修改 `PhaseFormer_gold_standard.md`。
