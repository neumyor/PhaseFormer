# PhaseFormer 低秩 checkpoint 信息保留分析计划

> 状态：**待实现，尚未执行**
>
> 日期：2026-09-16
>
> 实验性质：复用既有低秩压缩 checkpoint 的事后机制分析；不新增模型训练，不重新读取 test
>
> 前置结果：
> - `docs/PhaseFormer_rank_sweep_conditioned_experiment.md`
> - `docs/PhaseFormer_rank_capacity_and_data_property_report.md`
> - `docs/PhaseFormer_top2_direction_retention_summary.md`
> - `docs/PhaseFormer_direction1_neighborhood_report.md`

## 0. 核心问题与总体思路

本阶段只回答一个问题：

> **已经训练好的低秩 NLinear 支路，实际从历史窗口中保留了什么信息，又把这些信息写成了
> 怎样的未来修正？**

现有证据已经说明：

1. 低秩压缩可以在大幅减少参数后保留大部分分支价值；
2. 独立 NLinear 回归的第一预测方向主要表现为“近期加权水平 → 未来整体位移”；
3. 但把该固定方向或前两个固定方向直接作为端到端输入瓶颈，不能普遍恢复完整 NLinear；
4. 因此不能继续把“独立 RRR 的前几个方向”直接等同于“端到端低秩模型实际保留的信息”。

本计划改为直接读取已经训练好的低秩 checkpoint。对每个 checkpoint，先把
`decoder @ encoder` 合成为真实有效映射，再用规范 SVD 分解成成对的：

```text
归一化输入读取方向 v_i  →  标量系数 h_i  →  输出修正形状 u_i
```

随后通过语义模板对齐、Phase 条件性目标对齐和残差支路私有输入干预，区分：

- 模型真正保留了哪些可解释信息；
- 哪些只是相关但非必要；
- 最终性能近中性有多少来自信息保留，又有多少来自 gate 稀释或主干补偿。

## 1. 预注册假设

### H1：低秩头保留的是条件性纠错信息

低秩支路实际需要预测的不是独立目标 `y - x_last`，而是 Phase 主干与 gate 给定后仍需补充的
误差。因此，训练 checkpoint 的输入子空间应当比独立 RRR 更接近“Phase 条件性 RRR”方向。

### H2：第一类稳定信息仍是近期水平校正

多数 setting 的首个主要规范方向预计读取多尺度近期加权水平或近期 level contrast，并在
输出侧生成接近恒定的预测区间位移。

### H3：第二类及后续信息是数据集条件性的形状校正

除第一方向外，其余有效方向可能分别对应局部变化率、曲率、周期 level、周期幅度或相位相关
修正，但不预设它们在 7 个 setting 上具有统一语义。

### H4：最终性能近中性不完全等于信息无损

部分 setting 可能出现低秩分支自身明显损失信息，但融合 gate 减小或 Phase 主干补偿，使最终
MSE/MAE 变化很小。此时结论必须写成“系统绕行/稀释了损失”，不能写成“低秩完整保留信息”。

## 2. checkpoint 范围

### 2.1 正式分析单元

复用条件性秩扫描的 7 个 setting：

| Setting | q=1/4 rank | q=1/8 rank | q=1/16 rank | q=1/32 rank |
|---|---:|---:|---:|---:|
| ETTh2-96 | 24 | 12 | 6 | 3 |
| ETTh2-720 | 180 | 90 | 45 | 22 |
| ETTm2-96 | 24 | 12 | 6 | 3 |
| ETTm2-192 | 48 | 24 | 12 | 6 |
| Weather-96 | 24 | 12 | 6 | 3 |
| Weather-192 | 48 | 24 | 12 | 6 |
| Electricity-336 | 84 | 42 | 21 | 10 |

每个 setting 使用 seeds `2021/2022/2023`：

- 低秩 checkpoint：`7 × 4 × 3 = 84`；
- 配对 direct checkpoint：`7 × 1 × 3 = 21`；
- 合计正式审计单元：**105 个 checkpoint**。

`q=1` 因子化满秩仅有 seed 2021，只作为参数化诊断，不进入跨 seed 正式结论。

### 2.2 远程产物位置

- seed 2021：`research_runs/rank_sweep_2_stage1/`
- seeds 2022/2023：
  - `research_runs/rank_sweep_2_multiseed_stage1_20260914_v4/`
  - `research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1/`
- 已有聚合：
  `research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/`

正式执行前按 `REMOTE_SERVER.md` 核对远端代码版本和 checkpoint 完整性。分析以远程
A800 服务器的 `time` 环境为准；矩阵分解可用 CPU，validation 前向使用空闲 GPU。

## 3. 规范化 checkpoint 映射

### 3.1 为什么不能直接解释 encoder 的第 i 个隐藏单元

低秩头为：

```text
delta = decoder(encoder(z))
```

对任意可逆矩阵 `R`，`decoder·R^{-1}` 与 `R·encoder` 产生相同有效映射。因此原始隐藏坐标
存在旋转不确定性，不能把 `encoder.weight[0]` 直接命名为“水平”、把第 2 行命名为“趋势”。

### 3.2 使用有效映射的 SVD

对每个 checkpoint 计算：

```text
W = decoder.weight @ encoder.weight
c = decoder.weight @ encoder.bias + decoder.bias
W = U diag(s) V^T
```

若 `pool_factor != 1`，必须把确定性的 pooling 算子并入 `W` 后再分解。本轮正式 checkpoint
预期均为 `pool_factor=1`，但仍需逐个审计，不允许默认假设。

第 i 个规范模式定义为：

```text
h_i = v_i^T z_n
delta_i = s_i · u_i · h_i
```

- `z_n`：残差支路实际接收的 RevIN 归一化、末值中心化历史；
- `v_i`：从该历史窗口读取什么；
- `h_i`：每个样本保留下来的标量；
- `u_i`：该标量在预测区间写成什么曲线；
- `s_i`：该模式的映射强度。

合成 bias `c` 在 RevIN 归一化空间中与样本无关，但反归一化后会被每个窗口的尺度
`sigma` 调制。它不读取历史时间形状，却可能利用 RevIN scale 形成样本相关修正，因此必须
作为独立的“scale-modulated bias”分析，不能混入输入方向语义。所有输入保留/删除干预
默认保持原始 `c` 不变，并另设 `bias-off` 诊断量化其贡献。

当相邻奇异值间隙过小，禁止解释单个方向，只解释对应子空间。

## 4. 语义字典

所有模板在历史长度 720 上构造并做 L2 单位化，但**不能统一对模板去均值**：近期均值和
输出常值模板的非零和正是 level/offset 语义的一部分。只对模板在数据样本上产生的激活值
做样本维中心化；组内共线模板通过 rank-revealing QR/SVD 得到稳定基。数据集的物理周期
按采样间隔换算，ETTm2 不得把 24 个点误写成一天。

### 4.1 输入侧语义组

| 语义组 | 候选模板 | 要回答的问题 |
|---|---|---|
| 近期水平 | 尾部均值 6/12/24/48/72/168；EMA τ=6/12/24/48/72/168 | 是否读取近期加权水平 |
| 水平变化 | 最近窗口与前一窗口的均值差；最近 24/72/168 与更早历史的 level contrast | 是否读取水平漂移 |
| 局部趋势 | 尾部 24/72/168 的常数、线性项；末点锚定 ramp | 是否读取上升/下降速度 |
| 局部曲率 | 尾部 24/72/168 的二次项及去线性残差模板 | 是否读取加速/拐弯 |
| 周期 level | 按日/周周期分块后的近期周期均值对比 | 是否读取周期之间的整体高度变化 |
| 周期形状 | 数据集物理日/周周期的 sin/cos、最近周期与历史周期差分模板 | 是否读取周期幅度或相位相关线索 |
| 快速局部变化 | 一阶差分、二阶差分、2/4/8 步局部对比模板 | 是否保留短时变化而非纯低通信息 |

同时设置两类非语义对照：

- 同维度训练集 PCA 子空间；
- 固定随机种子的同维度随机正交子空间，至少 100 次重复。

### 4.2 输出侧语义组

| 语义组 | 候选模板 |
|---|---|
| 整体位移 | 全 horizon 常值向量 |
| 缓慢倾斜 | 全 horizon 线性 ramp |
| 曲率修正 | 二次曲线及分段线性曲线 |
| 周期修正 | 物理日/周周期 sin/cos |
| 近期形状延续 | 最近周期复制、衰减复制及其差分 |

## 5. 实验阶段

### Stage 0：checkpoint 清单与等价性审计

1. 收集 105 个正式 checkpoint 的路径、配置、commit、seed、rank、pool factor 和文件哈希；
2. 检查每个 run 的 `status`、best-validation checkpoint 和已有指标；
3. 用固定 validation batch 验证：

```text
decoder(encoder(z)) == Wz + c
```

最大绝对误差要求 `<1e-6`；

4. 核对干预代码只改变 `weak_period_residual` 的私有中心化输入，Phase 主干仍读取原始完整输入；
5. 禁止读取 test，禁止覆盖原 checkpoint。

任一 checkpoint 无法通过等价性审计时，只排除对应 cell，不得静默替换成其他 run。

### Stage 1：规范模式、有效贡献和跨 seed 稳定性

对 84 个低秩 checkpoint：

1. 对有效映射 `W` 做 SVD；
2. 记录奇异值累计份额、participation ratio 和有效数值秩；
3. 在 validation 数据上计算每个模式的：
   - latent score 方差 `Var(h_i)`；
   - 输出修正能量 `E||delta_i||²`；
   - 单模式清零后的分支 MSE 与融合 MSE/MAE 变化；
   - 合成 bias 的修正能量及 `bias-off` 后的分支/融合指标变化；
4. 对同 setting、同 rank 的三个 seed 计算输入/输出子空间 principal angles、
   projection overlap 和 Procrustes 对齐；
5. 只有跨 seed 稳定且奇异值间隙充分的模式才允许单独命名，其余只报告子空间级语义。

主报告展示 `q=1/8` 与 `q=1/32`；其余两档用于检查语义是否随压缩强度改变。

### Stage 2：输入与输出语义归因

对每个规范输入方向 `v_i` 同时计算：

1. 与单个模板的欧氏 `|cos|`；
2. 在训练数据协方差度量下，`v_i^Tz_n` 与模板统计量的相关系数；
3. 各语义组对子空间的投影解释率；
4. 全字典回归 R²；
5. leave-one-group-out 与 Shapley R²，避免把高度相关的“近期均值”和“EMA”重复计功。

对输出方向 `u_i` 计算相同的模板对齐、组解释率和字典 R²。

最终必须以成对方式命名：

```text
读取：近期 EMA 水平
写入：预测区间整体位移
```

不能只根据输入方向或只根据输出方向命名完整机制。

### Stage 3：Phase 条件性目标分析

固定每个 checkpoint 的 Phase 主干和 gate，在训练 split 上记录：

```text
z_n = x_normalized - x_normalized_last
p_n = normalized Phase prediction
g = learned gate
mu, sigma = the exact RevIN statistics of this forward pass
```

直接在模型的真实归一化/反归一化路径上求解与融合 MSE 对齐的低秩仿射映射：

```text
r_n = x_normalized_last + W z_n + c
y_hat = RevIN_denorm((1-g)p_n + g r_n; mu, sigma)
min_{W,c} Σ ||y - y_hat||²
```

不得省略 RevIN，也不通过除以很小的 `g` 构造目标；应把 `g`、`sigma` 和 RevIN affine
参数直接并入加权最小二乘。Huber 口径作为附加 IRLS 复核，不替换主要 MSE 分析。

比较三个输入子空间：

1. checkpoint 实际学到的低秩子空间；
2. 独立目标 `y-x_last` 的既有 RRR 子空间；
3. Phase 条件性目标的 RRR 子空间。

如果实际 checkpoint 与条件性 RRR 的对齐稳定高于独立 RRR，才能支持“低秩保留的是主干
缺口信息，而不是独立未来预测信息”。

### Stage 4：残差支路私有输入的必要性/充分性干预

所有干预都只作用于低秩残差支路，Phase 主干输入保持不变。
除单独的 `bias-off` 诊断外，各输入干预均保持 checkpoint 原始合成 bias `c` 不变。

| 干预臂 | 残差支路看到的输入 | 作用 |
|---|---|---|
| Original | 原始归一化、末值中心化历史 `z_n` | checkpoint 基线 |
| Semantic-only | 只保留 `z_n` 中被识别的语义子空间 | 检验充分性 |
| Semantic-drop | 从 `z_n` 中删除该语义子空间 | 检验必要性 |
| PCA-only/drop | 同维度 PCA 保留/删除 | 方差对照 |
| Random-only/drop | 同维度随机正交保留/删除，100 次 | 随机子空间对照 |
| Conditional-RRR-only | 只保留条件性 RRR 子空间 | 检验条件性目标解释 |
| Independent-RRR-only | 只保留既有独立 RRR 子空间 | 与前置实验衔接 |
| Bias-off | 原始 `z_n`，仅删除合成 bias `c` | 区分尺度调制偏置与输入信息 |

记录：

- 原低秩修正曲线的重建 R²、余弦相似度和 RMSE；
- NLinear 分支单独的 validation MSE/MAE；
- 融合输出的 validation MSE/MAE；
- 相对 checkpoint 内 Phase-only 输出的贡献保留率；
- 每个模式、每个语义组被删除后的性能变化；
- 变化是否超过随机对照的 95% 区间。

这一步不重新训练。若后续需要验证模型能否在只保留语义信息时重新适应，应另立端到端训练
计划，不能与本轮 checkpoint 因果干预混在同一结论中。

### Stage 5：真实样本可视化

每个 setting 从 validation 中程序化选择：

- 1 个 Semantic-only 几乎完全复现原低秩修正的代表样本；
- 1 个 Semantic-drop 导致明显退化的必要性样本；
- 1 个语义解释失败或随机对照同样有效的反例。

每张图固定包含：

1. 历史窗口与真实未来；
2. Phase-only、原低秩 checkpoint、Semantic-only、Semantic-drop 的预测曲线；
3. 原低秩修正与各规范模式的分解曲线；
4. 主要输入统计量及其数值；
5. 样本级 MSE/MAE，不允许只凭视觉挑图。

## 6. 主要指标与判定

### 6.1 单个 setting 的稳定语义

一个语义组只有同时满足以下条件，才可命名为该 setting 的稳定保留信息：

1. 三个 seed 中至少 2 个把该语义组排在首位；
2. 该组对主要输入模式或主要输入子空间的解释率 `≥50%`；
3. 配对输出语义组解释率 `≥80%`；
4. Semantic-drop 的损失超过同维度随机删除的 95% 区间；
5. Semantic-only 的融合 MSE 与 MAE 均不比原 checkpoint 差超过 `0.5%`。

### 6.2 跨 setting 判定

| 结果 | 判定 |
|---|---|
| 同一“读取信息 → 输出修正”在至少 5/7 setting 成立 | 一致机制 |
| 在 3–4/7 setting 成立，或只在同类数据集成立 | 条件性机制 |
| 不超过 2/7，或不优于随机/PCA 对照 | 不支持该语义机制 |

### 6.3 最终结论必须区分三部分

最终报告分别量化：

1. **真实保留的信息**：Semantic-only/Conditional-RRR-only 能解释和保留多少；
2. **被删除但没有预测价值的信息**：删除后分支与融合指标均基本不变；
3. **被系统绕行的信息损失**：分支退化，但 gate 或 Phase 主干使融合指标近中性。

禁止把第 3 类写成“信息被完整保留”。

## 7. 待填充结果表

### 表 1：checkpoint 审计

| Setting | rank/q | seeds 完整 | pool factor | 映射等价误差 | checkpoint hash | QC |
|---|---:|---:|---:|---:|---|---|
| ETTh2-96 | | | | | | |
| ETTh2-720 | | | | | | |
| ETTm2-96 | | | | | | |
| ETTm2-192 | | | | | | |
| Weather-96 | | | | | | |
| Weather-192 | | | | | | |
| Electricity-336 | | | | | | |

### 表 2：checkpoint 规范模式

| Setting | q/rank | seed | mode/subspace | singular share | output energy share | zero-mode Δbranch MSE | zero-mode Δfused MSE |
|---|---|---:|---|---:|---:|---:|---:|
| | | | | | | | |

### 表 3：跨 seed 稳定性

| Setting | q/rank | input subspace overlap | output subspace overlap | stable individual modes | 结论 |
|---|---|---:|---:|---:|---|
| | | | | | |

### 表 4：输入—输出语义

| Setting | q/rank | canonical mode | 输入首要语义 | 输入解释率 | 输出首要语义 | 输出解释率 | 跨 seed 一致 |
|---|---|---:|---|---:|---|---:|---|
| | | | | | | | |

### 表 5：独立目标与条件性目标对齐

| Setting | q/rank | overlap with independent RRR | overlap with conditional RRR | 差值 | 支持 H1 |
|---|---|---:|---:|---:|---|
| | | | | | |

### 表 6：语义保留/删除干预

| Setting | q/rank | Arm | correction R² | Δbranch MSE | Δfused MSE | Δfused MAE | vs random 95% |
|---|---|---|---:|---:|---:|---:|---|
| | | Original | 1.000 | — | — | — | — |
| | | Semantic-only | | | | | |
| | | Semantic-drop | | | | | |
| | | PCA-only/drop | | | | | |
| | | Random-only/drop | | | | | |
| | | Conditional-RRR-only | | | | | |
| | | Independent-RRR-only | | | | | |
| | | Bias-off | | | | | |

### 表 7：最终机制裁定

| 候选机制 | 成立 setting | 反例 setting | 必要性证据 | 充分性证据 | 裁定 |
|---|---|---|---|---|---|
| 近期加权水平 → 整体位移 | | | | | |
| 水平变化/局部趋势 → 倾斜修正 | | | | | |
| 周期 level/幅度/相位 → 周期修正 | | | | | |
| 快速局部变化 → 局部形状修正 | | | | | |
| gate/主干绕行而非信息保留 | | | | | |

## 8. 待实现脚本

### `scripts/analyze_lowrank_checkpoint_information.py`

- 扫描并审计 105 个 checkpoint；
- 合成有效映射和 bias；
- SVD 规范化、模式贡献、跨 seed 子空间对齐；
- 计算输入/输出语义字典归因；
- 输出 `checkpoint_inventory.csv`、`canonical_modes.csv`、
  `semantic_alignment.csv`、`cross_seed_alignment.csv`。

### `scripts/compute_phase_conditional_rrr.py`

- 只读取 train split；
- 从冻结 checkpoint 导出 Phase 输出、gate 与中心化历史；
- 计算融合目标下的条件性 RRR；
- 与 checkpoint 子空间、独立 RRR 子空间比较；
- 输出 `conditional_rrr_alignment.csv` 和必要的矩文件。

### `scripts/evaluate_lowrank_semantic_interventions.py`

- 只读取 validation split；
- Phase 主干始终使用原始完整输入；
- 只替换残差支路私有中心化输入；
- 执行 Semantic/PCA/Random/Conditional-RRR/Independent-RRR 的 only/drop 干预；
- 输出逐 checkpoint 指标和样本级误差。

### `scripts/render_lowrank_checkpoint_information_report.py`

- 汇总表 1–7；
- 绘制规范输入方向、输出方向、语义解释率、跨 seed 对齐和真实样本预测曲线；
- 生成最终 Markdown 报告。

## 9. 输出目录

```text
research_runs/lowrank_checkpoint_information_v1/
  checkpoint_inventory.csv
  canonical_modes.csv
  cross_seed_alignment.csv
  semantic_alignment.csv
  conditional_rrr_alignment.csv
  intervention_results.csv
  sample_errors.csv
  analysis_summary.json
  report.md
  figures/
```

checkpoint 不复制进该目录，只记录原路径和哈希。

## 10. 边界与披露

1. 这些 checkpoint 来自既有 test-exposed 条件性配置，本分析不得表述为盲测或无偏泛化估计。
2. 正式语义选择只使用 train；性能与样本验证只使用 validation；本阶段不重新读取 test。
3. checkpoint 事后分析能回答“已经学到了什么”和“冻结模型依赖什么”，不能直接证明从头训练
   时该语义是唯一可学习机制。
4. 低秩隐藏坐标存在旋转不确定性，必须解释有效映射的规范模式或稳定子空间。
5. 相关性、模板相似度和频谱形状均不是因果证据；只有分支私有输入的保留/删除干预可用于
   必要性和充分性判断。
6. 如果不同 seed 学到低重合但等性能的子空间，应结论为“存在多组等价信息通路”，不能强行
   给出唯一语义。
7. 若 Semantic-only 失败而 Conditional-RRR-only 成功，应结论为“保留了任务相关线性混合，
   但当前人工语义字典不足”，而不是否定低秩信息集中。
8. 若分支明显退化而融合输出近中性，应把 gate 稀释和主干补偿作为独立机制报告。
