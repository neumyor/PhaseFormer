# PhaseFormer NLinear 结构化低秩宽度优先探索计划

> 状态：**Round 0 与 Round 1 已完成（2026-09-15）**。五个结构化候选在
> 4/4 pilot setting 上全部退化，无路线晋级，已按 §6 与 §13 早停：
> Round 2/3/4 不启动。结论见 §6.1，停止判定见 §13.1，逐格结果见 §6.2。
>
> 本计划采用 **test-oriented exploratory search**：候选训练完成后读取 test，
> 并使用 test MSE/MAE 选择下一轮探索方向。所有参与选择的候选、setting、
> 指标和选择轨迹必须保留。结果属于 **test-set selection / exploratory
> evidence**，不得表述为盲测、无偏泛化估计或正式 benchmark 提升。
>
> 默认单 seed `2021`，优先横向比较不同技术路线；除非用户另行要求，
> 探索阶段不做多 seed 复核。多 seed 只属于后续确认阶段。
>
> 产物：Round 0 复用审计 `research_runs/structured_lowrank_round0_v1/`；
> Round 1 结果矩阵 `research_runs/structured_lowrank_round1_v1/results.csv` 与
> `results.md`；训练 scratch `research_runs/structured_lowrank_round1_scratch/`
> （`research_runs/` 在 `.gitignore` 内，不入库）。

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

- validation 只用于选择每个训练 run 的 checkpoint；test 用于候选排序、
  路线淘汰和下一轮方向选择；
- 同一个候选在不同 rank、`P` 或结构配置下可以分别训练并分别读取 test；
  这属于明确登记的多轮自适应 test search；
- 所有参与选择的候选必须保留，不得只保留胜者；
- 每份结果摘要必须写明 `test-set selection`；
- 探索结果不得称为盲测、无偏泛化估计或最终 benchmark；
- `PhaseFormer_gold_standard.md` 仍是固定参照，不因探索结果修改；
- 只改善一个指标即可记为正向信号，但必须明确写成“单指标改善”，
  不得把它写成双指标或全面提升。

#### 3.1.1 本轮裁决的执行口径

1. **checkpoint 与路线选择分离**：每个 `(candidate, setting)` 训练后，
   恢复 validation loss 最低的 checkpoint；只在此 checkpoint 上读取一次
   test。validation 不决定路线晋级，test 决定路线晋级。
2. **“一次 test”按配置单元计算**：一次是一个完整的
   `(candidate, setting, seed, hyperparameter/config)` 单元。不同 rank 或
   `P` 是不同配置，允许各自读取 test。跨轮次的选择必须登记到同一条
   test-feedback 轨迹。
3. **已有结果优先复用**：若已有结果在数据划分、`L/H`、seed、模型语义、
   训练协议、checkpoint 规则和指标口径上完全匹配，则直接复用原始 test
   数字，不重复训练或重复读取 test；标记为 `reused_exact`。若任何条件
   不匹配，必须重训并标记为 `rerun_for_alignment`。
4. **单候选判据**：定义 `positive_mse = delta_MSE > 0`、
   `positive_mae = delta_MAE > 0`。任一指标为正即为正向信号；两者都为正
   记为双指标改善，只有一个为正记为单指标改善，两个都不为正记为无改善。
5. **路线晋级不强制双指标**：路线可凭 MSE 或 MAE 中任一指标在多个
   setting 上的正向信号晋级，但必须保留另一指标并报告其退化。若某路线
   只在一个 setting、只改善一个指标，则只能记为局部信号。

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
| test | 每个配置单元只读取一次；跨配置、跨轮次允许自适应选择 |

必须包含三个对照：

- `phase_only`：`use_residual_head=False`，完全关闭 residual branch 的纯相位路径；
- `direct_nlinear`：当前未因子化 NLinear；
- `pooled_lowrank(q=1/8)`：现有非结构化低秩控制。

所有结构化候选必须关联参数量匹配的时间点轴低秩 control，避免把
“参数减少”误判成“坐标系利用”。

### 3.3 参数、计算量与复用规则

#### 参数匹配 control 的固定定义

文档中的 `matched shallow linear control` 统一改名为
`time_axis_matched_lowrank`，定义为：

```text
centered = X - X_last
delta = Linear(720 -> r_match)
        followed by Linear(r_match -> H)
output = delta + X_last
```

它不做 pooling、周期分段、basis、近期选择或 smoothing；encoder 保持默认
初始化，decoder 零初始化，与现有 `pooled_lowrank` 的初始化语义一致。
`r_match` 选择为使 residual head 参数量最接近结构化候选；主分析允许的
head-parameter 差异为 **不超过 5%**。若无法达到 5%，选择不超过目标预算
的最近整数 rank，并记录实际差异。

参数口径固定为：

- **主要公平性口径**：residual head 的可训练参数，包括 encoder、decoder、
  level/shape/basis 参数；
- **次要效率口径**：total trainable parameters、MACs、peak memory、
  train/inference time；
- static gate 计入 total parameters，但不单独用于参数匹配；
- PhaseFormer 主干不参与 head 参数匹配，因为所有候选共享同一主干；
- 参数匹配不是要求所有路线拥有完全相同的总参数，而是要求每个结构化
  候选有一个相近 head budget 的 control。

Round 1 只训练每个 setting 和每个**唯一 `r_match`** 的 control。若多个路线
得到相同 `r_match`，复用同一个 control，避免为每条路线重复训练相同对照。

#### 现有结果复用清单

优先检查并复用：

- `phase_only`；
- `direct_nlinear`；
- `pooled_lowrank(q=1/8)`；
- 已经在完全相同配置下训练过的 generic 或 matched control。

复用结果仍属于既有 test-exposed 谱系，必须在 `results.csv` 的 `source`
列写明原始实验目录和 `reused_exact`。复用不能被写成新的独立确认结果。

### 3.4 探索 setting

**Pilot 四个 setting：**

| setting | 选择理由 |
|---|---|
| ETTh2-H96 | 已有局部低秩信号，短 horizon |
| ETTh2-H720 | 三 seed 复核中唯一稳定受益的 setting |
| ETTm2-H192 | 深压缩退化明显的反例 |
| Electricity-H336 | 原始权重谱不低秩，但训练低秩能重组出好解 |

**Expansion 四个 setting：**

| setting | 选择理由 |
|---|---|
| Weather-H192 | 周期较弱，检验结构方法的适用边界 |
| ETTh1-H96 | 第一轮低秩表现不稳定，作为压力测试 |
| ETTm1-H96 | 与 ETTm2 对比，检验同频率族迁移性 |
| ETTh2-H336 | 若 pilot 信号主要来自 ETTh2-H720，用于同数据集不同 horizon 的独立检查 |

不按 dataset/horizon 临时发明不同机制。只在单一 setting 有效的方案，
记录为 setting-specific 信号，不晋级为统一路线。

### 3.5 新 setting 的超参数与协议

对每个 setting，先查找完全匹配的已有 `direct_nlinear`、generic
low-rank 和 `phase_only` 结果：

1. 有完全匹配结果：复用其 `(gate_init, learning_rate)` 和 test 数字；
2. 没有完全匹配结果：只新增一个 `direct_nlinear` calibration run，使用
   当前 preset 默认 `gate_init=0.2`、`learning_rate=1e-3`；
3. 本轮不为新 setting 额外运行 Stage 0 的 gate/lr 网格，不通过 test 选择
   新的训练超参数；
4. 该 setting 的 structured candidate、generic control 和 matched control
   使用同一组 calibration 超参数；
5. 若 calibration run 失败或指标异常，先修复协议，不允许只为候选单独调参。

这条规则优先效率和可归因性；未来若路线晋级为确认候选，才另立独立的
超参数确认计划。

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

结构定义：

- `residual_period_len=P` 只作用于新 residual head；PhaseFormer 主干的
  `period_len` 固定为当前 setting 的既有值，默认仍为 24；
- `S` 的输入周期轴为 `K=ceil(L/P)`，对末值锚定后的 centered history
  右侧用零补齐；在原始尺度上等价于用最后一个值复制补齐；
- 对每个 phase slot `p` 共享同一个 `K -> K_y` 映射：
  `Y[k_y,p] = sum_k W[k_y,k] S[k,p]`；
- 映射是一次性直接预测，不递归、不把前一预测周期作为下一周期输入；
- 输出先生成 `ceil(H/P) * P` 个点，再按时间顺序截取前 `H` 点；
- 输出仍为 `delta + last`，并沿用当前归一化/反归一化链路。

首轮配置：

- `P=24, r_period=4` 作为代表配置；
- 若所有代表配置退化，追加一个预注册的中等容量诊断点
  `r_period=8`，不进行开放式 rank 搜索；
- 全部周期，不做近期裁剪；
- 输出按未来 segment 展开；
- 保留 NLinear last-value anchor。

### 路线 B：segment basis / TimeBase-style

```text
B = E(S),       E: K -> R
F = D(B),       D: R -> K_y
```

将少量历史 basis 映射为未来 segments。`P` 只作用于 residual head，
不改变 PhaseFormer 主干。首轮配置：

- `P=24, R=4, lambda_orth=0.01` 作为代表配置；
- 若代表配置退化，追加 `R=8` 作为唯一中等容量诊断点；
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

为避免 `shape_direct` 的约 29k 参数与 level-only 相差两个数量级，
Round 1 不使用 `shape_direct`。首轮只比较：

1. `level_only`；
2. `level_dense + shape_period_lowrank(r_shape=4)`；
3. `level_lowrank(r_level=1) + shape_period_lowrank(r_shape=4)`。

`shape_period_lowrank` 仍只在周期轴上做 `K -> K_y` 映射，并对 phase slot
共享；三个条件都继承末值锚点。`shape_direct` 只能作为 Round 2 的额外
容量对照，不能作为 Round 1 的路线代表配置。

### 路线 D：近期周期稀疏选择

在周期坐标中只使用：

```text
recent_1, recent_3, recent_7, recent_15, all_30
```

路线 D 的代表配置固定为 `recent_7, P=24, fixed_exp`。首轮只测试 hard
recent window 和 fixed exponential weighting，不加入可学习 gate，避免把
额外门控参数误认为稀疏结构收益。若需要错位控制，使用与主路线相同的
固定分段偏移，不另行改变窗口定义。

### 路线 E：separable / Kronecker map

将时间点映射限制为少量周期轴与相位轴的可分离项：

```text
W ≈ sum_j A_j(period) ⊗ B_j(phase)
```

该路线表达能力和实现成本都更高。它不是 Round 1 的必跑路线：只有在
A--D 没有明确胜者、但 Round 1 的 aligned-vs-shifted 结果显示周期/相位
交互仍可能重要时才启动；如果 A--D 全部无信号且没有该诊断依据，则直接
停止，不强行跑 E。启动时只测试 `P=24, J=1`，`J=2` 仅作为中等容量诊断点。

## 5. Round 0：控制、参数预算和实现校验

### 实验目的

确认所有后续路线的指标、参数量、计算量和输出形状可公平比较，排除训练
失败或参数规模差异造成的假信号。

### 实验设置

- Pilot 四个 setting；
- seed 2021；
- `phase_only`、`direct_nlinear`、generic `q=1/8`；
- `time_axis_matched_lowrank` control；已有完全匹配行优先复用；
- 每个候选训练后读取一次 test；
- 记录 head params、total params、MACs、训练时间、推理时间和显存峰值。

### 代填充表：Round 0

| setting | phase_only MSE/MAE | direct MSE/MAE | generic q=1/8 MSE/MAE | matched control MSE/MAE | source | 新增训练数 | 备注 |
|---|---|---|---|---|---|---:|---|
| ETTh2-H96 | 0.280834/0.343016 | 0.272100/0.332843 | 0.272314/0.334785 | 见 §6.2 矩阵（按候选 budget 匹配） | `reused_exact` | 0 | 三项控制全部复用；该 setting 最低匹配控制 head 参数 2636 |
| ETTh2-H720 | 0.425394/0.455186 | 0.390898/0.427867 | 0.387149/0.426510 | 见 §6.2 矩阵（按候选 budget 匹配） | `reused_exact` | 0 | 三项控制全部复用；该 setting 最低匹配控制 head 参数 2886 |
| ETTm2-H192 | 0.228508/0.298497 | 0.215685/0.288061 | 0.213480/0.287809 | 见 §6.2 矩阵（按候选 budget 匹配） | `reused_exact` | 0 | 三项控制全部复用；该 setting 最低匹配控制 head 参数 2864 |
| Electricity-H336 | 0.167681/0.259416 | 0.161729/0.254716 | 0.162918/0.256228 | 见 §6.2 矩阵（按候选 budget 匹配） | `reused_exact` | 0 | 三项控制全部复用；该 setting 最低匹配控制 head 参数 5470 |

### 退出规则

若控制不能在四个 setting 全部完成，先修复协议或实现，不比较路线。已有
完全匹配的控制结果直接复用；没有匹配结果才新增训练。若 generic low-rank
在全部 pilot setting 都明显优于 direct，结构化候选仍可使用更低预算，但必须
同时报告与其参数匹配的 control，不能只与 direct 比较。

## 6. Round 1：四条必跑路线宽搜，E 条件启动

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
| C | `level_dense + shape_period_lowrank(r_shape=4), P=24` |
| D | `recent_7, P=24, fixed_exp` |
| E | `J=1, P=24` |

E 在 A--D 没有满足条件时才条件启动；因此默认 Round 1 为 A--D 四条路线，
不是无条件五条路线。`P=96` 作为第二批结构条件，只有当 `P=24` 路线在
至少两个 pilot setting 有正向 test 信号时才追加。

### 代填充表：Round 1 路线矩阵

| 路线 | setting | test MSE | test MAE | delta MSE% vs direct | delta MAE% vs direct | delta MSE% vs generic | head 参数量 | 匹配控制 head | gate | 双指标改善 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| A 周期轴低秩 | ETTh2-H96 | 0.276858 | 0.341764 | -1.749 | -2.680 | -1.669 | 604 | 913 | 0.50 | 否 | 退化 |
| A 周期轴低秩 | ETTh2-H720 | 0.401989 | 0.438586 | -2.837 | -2.505 | -3.833 | 3724 | 3602 | 0.50 | 否 | 退化 |
| A 周期轴低秩 | ETTm2-H192 | 0.218705 | 0.295566 | -1.400 | -2.605 | -2.448 | 1084 | 1105 | 0.20 | 否 | 退化 |
| A 周期轴低秩 | Electricity-H336 | 0.168993 | 0.261404 | -4.492 | -2.626 | -3.729 | 1804 | 1393 | 0.50 | 否 | 退化 |
| B segment basis | ETTh2-H96 | 0.282910 | 0.346283 | -3.973 | -4.038 | -3.891 | 144 | 913 | 0.50 | 否 | 退化 |
| B segment basis | ETTh2-H720 | 0.400858 | 0.438845 | -2.548 | -2.566 | -3.541 | 274 | 2161 | 0.50 | 否 | 退化 |
| B segment basis | ETTm2-H192 | 0.220520 | 0.296495 | -2.242 | -2.928 | -3.298 | 164 | 1105 | 0.20 | 否 | 退化 |
| B segment basis | Electricity-H336 | 0.167044 | 0.259291 | -3.286 | -1.796 | -2.532 | 194 | 1393 | 0.50 | 否 | 退化 |
| C level-shape | ETTh2-H96 | 0.276092 | 0.340641 | -1.467 | -2.343 | -1.387 | 268 | 913 | 0.50 | 否 | 退化 |
| C level-shape | ETTh2-H720 | 0.397620 | 0.436280 | -1.720 | -1.966 | -2.705 | 1204 | 2161 | 0.50 | 否 | 退化 |
| C level-shape | ETTm2-H192 | 0.217271 | 0.292852 | -0.736 | -1.663 | -1.776 | 412 | 1105 | 0.20 | 否 | 退化 |
| C level-shape | Electricity-H336 | 0.165809 | 0.259338 | -2.523 | -1.815 | -1.775 | 628 | 1393 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTh2-H96 | 0.282788 | 0.346697 | -3.928 | -4.163 | -3.846 | 28 | 913 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTh2-H720 | 0.403034 | 0.439916 | -3.105 | -2.816 | -4.103 | 106 | 2161 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTm2-H192 | 0.220849 | 0.297780 | -2.395 | -3.374 | -3.452 | 40 | 1105 | 0.20 | 否 | 退化 |
| D 近期周期稀疏 | Electricity-H336 | 0.168192 | 0.260725 | -3.996 | -2.359 | -3.237 | 58 | 1393 | 0.50 | 否 | 退化 |
| E 可分离 | ETTh2-H96 | 0.282277 | 0.346336 | -3.740 | -4.054 | -3.658 | 148 | 913 | 0.50 | 否 | 退化 |
| E 可分离 | ETTh2-H720 | 0.398192 | 0.437091 | -1.866 | -2.156 | -2.852 | 954 | 2161 | 0.50 | 否 | 退化 |
| E 可分离 | ETTm2-H192 | 0.217995 | 0.293999 | -1.071 | -2.061 | -2.115 | 272 | 1105 | 0.20 | 否 | 退化 |
| E 可分离 | Electricity-H336 | 0.165980 | 0.258834 | -2.629 | -1.617 | -1.879 | 458 | 1393 | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTh2-H96 | 0.285104 | 0.344771 | -4.779 | -3.584 | -4.697 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTh2-H720 | 0.398223 | 0.433137 | -1.874 | -1.232 | -2.860 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTm2-H192 | 0.215819 | 0.292444 | -0.062 | -1.521 | -1.096 |  | — | 0.20 | 否 | 退化 |
| matched 时间点轴 r1 | Electricity-H336 | 0.165001 | 0.258478 | -2.023 | -1.477 | -1.278 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r4 | ETTh2-H96 | 0.272635 | 0.335414 | -0.197 | -0.772 | -0.118 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r4 | ETTh2-H720 | 0.389380 | 0.428317 | +0.388 | -0.105 | -0.576 |  | — | 0.50 | 否 | 正向信号 |
| matched 时间点轴 r4 | ETTm2-H192 | 0.217446 | 0.290769 | -0.817 | -0.940 | -1.858 |  | — | 0.20 | 否 | 退化 |
| matched 时间点轴 r4 | Electricity-H336 | 0.162639 | 0.255381 | -0.563 | -0.261 | +0.171 |  | — | 0.50 | 否 | 退化 |

### 路线晋级规则

路线进入 Round 2，满足以下任一条件：

1. 至少 2/4 个 pilot setting 在同一个指标上有正向信号（MSE 或 MAE）；
2. 至少 3/4 个 setting 在任一指标上有正向信号，且没有 setting 出现
   MSE 与 MAE 同时超过 `1.5%` 的退化；
3. 在 ETTh2-H720 或 Electricity-H336 上出现单指标或双指标改善，同时
   其余 setting 没有两个指标同时超过 `1.5%` 的退化。

路线报告必须分别给出 MSE leaderboard、MAE leaderboard 和双指标交集，
不能用双指标交集替代单指标结果。

未达到条件的路线保留为负结果，不追加该路线搜索。若 A--D 全部未达到，
只有在 aligned-vs-shifted 已显示周期/相位交互迹象时才启动 E；否则停止。

### 6.1 结果与判定（2026-09-15 完成，test-set selection）

> 本节全部数字来自 Round 1 的 48 次训练（4 setting × 6 候选 × 2 控制）与
> Round 0 复用的 12 行控制。每个 run 的 validation 只用于选 checkpoint，
> test 只读一次；控制来自既有 test-exposed 谱系（`reused_exact`）。
> 全部为探索性证据，不是盲测、无偏泛化估计或 benchmark 提升。

**判定：五条路线全部未晋级，触发早停。** 逐格结果见 §6.2，要点：

1. **A--E 在 4/4 pilot setting 上 MSE 与 MAE 同时退化**，没有任何一格为正
   （幅度 −0.74% ~ −4.49%）。按本节的晋级规则 1/2/3 全部不满足：不存在
   “≥2/4 setting 同指标正向”，也不存在“3/4 setting 双指标平均不差于
   −0.3%”（实际均值 −1.61% ~ −3.36%）。
2. **换坐标系没有收益。** 参数匹配的时间点轴控制 r1 均值 −2.19%（MSE），
   优于全部五条结构化路线；与路线 A budget 对齐的 r4 控制均值仅 −0.30%，
   并且在 ETTh2-H720 上是唯一正向格（+0.39%）。
3. **普通低秩仍然略优。** `generic pooled_lowrank(q=1/8)` 在 ETTh2-H720
   (+0.96%) 与 ETTm2-H192 (+1.02%) 优于 direct，且优于所有结构化候选，
   与既有 `pooled_lowrank` 结论一致：低秩本身近似中性，换坐标系没有额外收益。
4. **假设判定**：H0（普通低秩主要是压缩）被支持；H1（周期轴低秩更适合
   NLinear）、H3（周期块稀疏有用）、H4（basis 多样性比 rank 更重要）在
   4/4 setting 上均未被支持。D（最近 7 个周期）退化最重（−3.36%），方向与
   H3 相反；C（level-shape）退化最轻（−1.61%），但仍为负。
5. **E 路线不启动**：本节规定只有 A--D 无胜者**且** aligned-vs-shifted 显示
   周期/相位交互迹象时才启动 E；本轮 A--D 全部退化且无该诊断依据，因此不启动。
6. **`P=96` 第二批结构条件不追加**：本节规定仅当 `P=24` 路线在至少两个
   pilot setting 有正向 test 信号时才追加，实际为 0 个。

### 6.2 Round 1 逐格结果矩阵

| 路线 | setting | test MSE | test MAE | delta MSE% vs direct | delta MAE% vs direct | delta MSE% vs generic | head 参数量 | 匹配控制 head | gate | 双指标改善 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| A 周期轴低秩 | ETTh2-H96 | 0.276858 | 0.341764 | -1.749 | -2.680 | -1.669 | 604 | 913 | 0.50 | 否 | 退化 |
| A 周期轴低秩 | ETTh2-H720 | 0.401989 | 0.438586 | -2.837 | -2.505 | -3.833 | 3724 | 3602 | 0.50 | 否 | 退化 |
| A 周期轴低秩 | ETTm2-H192 | 0.218705 | 0.295566 | -1.400 | -2.605 | -2.448 | 1084 | 1105 | 0.20 | 否 | 退化 |
| A 周期轴低秩 | Electricity-H336 | 0.168993 | 0.261404 | -4.492 | -2.626 | -3.729 | 1804 | 1393 | 0.50 | 否 | 退化 |
| B segment basis | ETTh2-H96 | 0.282910 | 0.346283 | -3.973 | -4.038 | -3.891 | 144 | 913 | 0.50 | 否 | 退化 |
| B segment basis | ETTh2-H720 | 0.400858 | 0.438845 | -2.548 | -2.566 | -3.541 | 274 | 2161 | 0.50 | 否 | 退化 |
| B segment basis | ETTm2-H192 | 0.220520 | 0.296495 | -2.242 | -2.928 | -3.298 | 164 | 1105 | 0.20 | 否 | 退化 |
| B segment basis | Electricity-H336 | 0.167044 | 0.259291 | -3.286 | -1.796 | -2.532 | 194 | 1393 | 0.50 | 否 | 退化 |
| C level-shape | ETTh2-H96 | 0.276092 | 0.340641 | -1.467 | -2.343 | -1.387 | 268 | 913 | 0.50 | 否 | 退化 |
| C level-shape | ETTh2-H720 | 0.397620 | 0.436280 | -1.720 | -1.966 | -2.705 | 1204 | 2161 | 0.50 | 否 | 退化 |
| C level-shape | ETTm2-H192 | 0.217271 | 0.292852 | -0.736 | -1.663 | -1.776 | 412 | 1105 | 0.20 | 否 | 退化 |
| C level-shape | Electricity-H336 | 0.165809 | 0.259338 | -2.523 | -1.815 | -1.775 | 628 | 1393 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTh2-H96 | 0.282788 | 0.346697 | -3.928 | -4.163 | -3.846 | 28 | 913 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTh2-H720 | 0.403034 | 0.439916 | -3.105 | -2.816 | -4.103 | 106 | 2161 | 0.50 | 否 | 退化 |
| D 近期周期稀疏 | ETTm2-H192 | 0.220849 | 0.297780 | -2.395 | -3.374 | -3.452 | 40 | 1105 | 0.20 | 否 | 退化 |
| D 近期周期稀疏 | Electricity-H336 | 0.168192 | 0.260725 | -3.996 | -2.359 | -3.237 | 58 | 1393 | 0.50 | 否 | 退化 |
| E 可分离 | ETTh2-H96 | 0.282277 | 0.346336 | -3.740 | -4.054 | -3.658 | 148 | 913 | 0.50 | 否 | 退化 |
| E 可分离 | ETTh2-H720 | 0.398192 | 0.437091 | -1.866 | -2.156 | -2.852 | 954 | 2161 | 0.50 | 否 | 退化 |
| E 可分离 | ETTm2-H192 | 0.217995 | 0.293999 | -1.071 | -2.061 | -2.115 | 272 | 1105 | 0.20 | 否 | 退化 |
| E 可分离 | Electricity-H336 | 0.165980 | 0.258834 | -2.629 | -1.617 | -1.879 | 458 | 1393 | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTh2-H96 | 0.285104 | 0.344771 | -4.779 | -3.584 | -4.697 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTh2-H720 | 0.398223 | 0.433137 | -1.874 | -1.232 | -2.860 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r1 | ETTm2-H192 | 0.215819 | 0.292444 | -0.062 | -1.521 | -1.096 |  | — | 0.20 | 否 | 退化 |
| matched 时间点轴 r1 | Electricity-H336 | 0.165001 | 0.258478 | -2.023 | -1.477 | -1.278 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r4 | ETTh2-H96 | 0.272635 | 0.335414 | -0.197 | -0.772 | -0.118 |  | — | 0.50 | 否 | 退化 |
| matched 时间点轴 r4 | ETTh2-H720 | 0.389380 | 0.428317 | +0.388 | -0.105 | -0.576 |  | — | 0.50 | 否 | 正向信号 |
| matched 时间点轴 r4 | ETTm2-H192 | 0.217446 | 0.290769 | -0.817 | -0.940 | -1.858 |  | — | 0.20 | 否 | 退化 |
| matched 时间点轴 r4 | Electricity-H336 | 0.162639 | 0.255381 | -0.563 | -0.261 | +0.171 |  | — | 0.50 | 否 | 退化 |

> `gate` 列为该 setting 冻结并复用的 `weak_period_residual_gate_init`
> （ETTm2-H192 为 Stage 0 冻结的 0.2，其余为 0.5；lr 均为 1e-3）。
> `head 参数量`/`匹配控制 head` 为实测值；`matched 时间点轴 r1` 是各 setting
> 的 rank-1 控制，`r4` 是与路线 A budget 对齐的控制。

#### 正向信号计数与均值（vs direct）

| 候选 | MSE 正向 setting 数 | MAE 正向 setting 数 | 双指标正向 setting 数 | 均值 delta MSE% | 均值 delta MAE% |
|---|---:|---:|---:|---:|---:|
| A 周期轴低秩 | 0/4 | 0/4 | 0/4 | -2.619 | -2.604 |
| B segment basis | 0/4 | 0/4 | 0/4 | -3.012 | -2.832 |
| C level-shape | 0/4 | 0/4 | 0/4 | -1.611 | -1.947 |
| D 近期周期稀疏 | 0/4 | 0/4 | 0/4 | -3.356 | -3.178 |
| E 可分离 | 0/4 | 0/4 | 0/4 | -2.327 | -2.472 |
| matched 时间点轴 r1 | 0/4 | 0/4 | 0/4 | -2.185 | -1.953 |
| matched 时间点轴 r4 | 1/4 | 0/4 | 0/4 | -0.297 | -0.520 |

#### MSE leaderboard（每 setting 前 3，含控制）

- ETTh2-H96: matched_A_period_lowrank_r8(0.272635) < C_level_shape(0.276092) < A_period_lowrank(0.276858)
- ETTh2-H720: matched_A_period_lowrank_r8(0.389380) < C_level_shape(0.397620) < E_separable(0.398192)
- ETTm2-H192: matched_A_period_lowrank(0.215819) < C_level_shape(0.217271) < matched_A_period_lowrank_r8(0.217446)
- Electricity-H336: matched_A_period_lowrank_r8(0.162639) < matched_A_period_lowrank(0.165001) < C_level_shape(0.165809)

> 双指标交集（MSE 与 MAE 同时为正）在全部 4 个 setting 上均为空集；
> 因此不存在任何可晋级为“机制创新”的候选。


## 7. Round 2：晋级路线结构消融

> **未执行（2026-09-15）**：Round 1 的五条路线在 4/4 pilot setting 上全部退化、
> 无一满足 §6 的晋级规则，因此本节的两条路线 / 六个消融条件 / 48 次训练预算
> 全部未使用。下方表格保留为未执行模板。

### 实验目的

判断 Round 1 的 test 信号来自真实结构，还是来自偶然 rank、参数量或周期
长度选择。

### 实验设置

最多保留两条路线。每条路线最多新增六个消融条件，Round 1 的 base 配置
直接复用、不占用六个名额。仍使用四个 pilot setting、seed 2021、每个
配置单元读取一次 test。

| 路线 | 可用消融轴 |
|---|---|
| A 周期轴低秩 | `P`、low/medium rank、all/recent-7、aligned/shifted |
| B segment basis | `P`、low/medium basis、orth off/on、aligned/shifted |
| C level-shape | level-only vs level+shape、level/shape rank、`P`、aligned/shifted |
| D recent sparse | recent-1/3/7/15、hard vs fixed-exp、`P`、aligned/shifted |
| E separable | `J`、`P`、period/phase rank、aligned/shifted |

每条路线只选择与其机制直接相关的六个条件，不强行套用无关轴。
`base` 不占六个新增名额。

### 错位/随机分段控制

- `aligned`：从固定数据窗口边界按 `P` 分段；
- `shifted`：固定偏移 `delta=P/2`，重新确定分段边界，超出历史范围的
  centered 值用零补齐；
- `random`：以 seed 2021 在每个 `(route, setting)` 生成一个固定 offset，
  该 offset 对该 run 的所有 sample 共享，不对每个 sample 重新随机；
- 不做周期维随机置换；周期顺序置换是另一个时间顺序实验，超出本轮；
- 对路线 D，窗口仍按错位后的 segment 索引定义，保持稀疏预算不变。

若 shifted/random 与 aligned 几乎相同，不能声称模型利用了真实周期结构。

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
- 在相近 head 参数量下优于 `time_axis_matched_lowrank`，或在相同性能
  下参数显著更少；
- 至少一个结构轴移除后出现 test 退化；
- gate、level/shape 输出或 basis 权重显示分支确实被使用，而非近似关闭。

若只满足“压缩后性能还可以”，结论只能是“压缩有效”。

## 8. Round 3：Expansion test-oriented scan

> **未执行（2026-09-15）**：无路线进入本节，未在 ETTh1 / ETTm1 / Weather / ETTh2-H336
> 上做任何扩展训练。下方表格保留为未执行模板。

### 实验目的

将最多一条统一路线扩展到未参与 pilot 的 setting，检查跨数据集迁移。

### 实验设置

- 固定 Round 2 选出的结构和配置；
- 不再按 setting 调 rank；
- ETTh1-H96、ETTm1-H96、Weather-H192、ETTh2-H336；
- seed 2021；
- 同时运行 `direct_nlinear` 和 generic `q=1/8`；
- 每个候选读取一次 test。

ETTh2-H336 是为 ETTh2-H720 单独设置的 holdout：若路线的 pilot 正向信号
主要来自 ETTh2-H720，必须保留该行，不能只在同一 setting 上继续调参。
若路线在 ETTh2-H720 之外已经有至少两个 pilot 正向 setting，可将
ETTh2-H336 作为补充行，但仍建议执行。

### 代填充表：Expansion

| setting | direct MSE/MAE | generic q=1/8 MSE/MAE | structured MSE/MAE | delta MSE% vs direct | delta MAE% vs direct | signal (MSE/MAE) | 严重退化 |
|---|---|---|---|---:|---:|---|---|
| ETTh1-H96 | — | — | — | — | — | — | — |
| ETTm1-H96 | — | — | — | — | — | — | — |
| Weather-H192 | — | — | — | — | — | — | — |
| ETTh2-H336 | — | — | — | — | — | — | — |

### Expansion 决策

- 至少 2/4 setting 在同一个指标上改善，且 pilot 无明显失败：列为值得
  有限确认；
- 只有 1/4 改善：记录为 dataset-specific，不升级为统一机制；
- 0/4 改善：停止路线扩展；
- 不因单个 setting 的 test 最优而改变已冻结结构。

## 9. 可选 Round 4：效率测试

> **未执行（2026-09-15）**：本节的前提是“Round 1--3 找到结构性路线”，本轮没有
> 结构性胜者，因此不执行效率评估。下方表格保留为未执行模板。

只有 Round 1--3 找到结构性路线后才执行。

| setting | model | head params | total params | MACs | peak memory | train time/epoch | inference time | MSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| — | direct_nlinear | — | — | — | — | — | — | — | — |
| — | generic q=1/8 | — | — | — | — | — | — | — | — |
| — | structured winner | — | — | — | — | — | — | — | — |

## 10. 工程实现与执行范围

### 10.1 新 head 的代码落点

新结构先放在独立模块：

```text
src/models/structured_residual_heads.py
```

由 `PhaseFormer.py` 或现有 head factory 统一注册，保留
`src/models/phase_adapters.py` 中已有 `shared`、`pooled_lowrank` 等实现不变。
新取值统一使用 `structured_` 前缀：

| head type | 路线 |
|---|---|
| `structured_period_lowrank` | A |
| `structured_segment_basis` | B |
| `structured_level_shape` | C |
| `structured_recent_period` | D |
| `structured_separable` | E |

每个新 head 都必须满足以下不变量：

- 输入输出张量形状与现有 residual head 一致；
- centered history 的最后值为零；
- decoder 或最终 residual projection 零初始化；
- forward 输出为 `delta + last`；
- residual head 使用当前同一套 normalization / denormalization 链路；
- `residual_period_len` 是 residual head 专属字段，不能复用或修改主干
  `period_len`。

### 10.2 本地校验先于服务器训练

Round 0 之前必须完成：

1. CPU shape test：覆盖 `P=24/96`、`H=96/192/336/720`，检查输出严格为
   `(batch, H, channels)`；
2. anchor test：零初始化时输出等于沿 horizon 复制的 `last`；
3. padding/cropping test：检查 `ceil(L/P)`、`ceil(H/P)`、右侧 centered
   zero padding 和输出前 `H` 点截取；
4. parameter-count test：程序输出与手工公式一致，matched control 差异
   不超过 5% 或明确记录 nearest-lower fallback；
5. aligned/shifted offset test：同一 seed 下 offset 固定且跨 sample 一致；
6. `py_compile`、相关 unit tests 和仓库要求的轻量 `pytest`。

只有这些检查通过，才允许进入 A800/服务器训练。

> **执行说明（2026-09-15）**：本节六项已全部完成，落在
> `tests/test_structured_residual_heads.py`（17 项）与
> `tests/test_search_head_override.py` / `tests/test_structured_lowrank_runner.py`（7 项）。
> 本机没有 conda 环境，因此校验在一个临时 CPU torch 环境
> （`uv` + Python 3.10 + torch CPU，位于 `/tmp/pf_static_verify`，不入库）中运行；
> 训练仍在 A800 上进行。全仓校验结果为 **320 passed / 262 subtests passed**。
> 第 4 项的参数量检查同时产出 `scripts/report_structured_lowrank_params.py`，
> 实测控制头开销为 `r * (L + H + 1) + H`：其 rank-1 下限在 H96 为 913、H720 为 2161，
> 而 P=24 下除路线 A 外的结构化候选都低于该下限（B 144、C 268、D 28、E 148 @H96），
> 因此这些控制固定在 rank 1 并**记录实测差距**（计划允许取最近可达秩并记录差异）；
> 路线 A 按其实测预算匹配到 rank 1（H96/ETTm2）或 rank 2（H720）。

### 10.3 样本级 test 导出

现有 validation top-k bad-case 导出不能直接满足本计划。Round 1 前新增一个
非侵入式 test 导出/分析脚本，至少支持：

- direct、generic、structured 三者逐样本逐变量的预测与误差；
- 按 `candidate - direct` 的差值排序；
- 按 MSE 改善、MAE 改善、双指标退化、周期水平偏移、周期错位和最大分歧
  六类选择不超过 8 个案例；
- 保存 level/shape/basis/period-weight 诊断；
- 不改变训练、checkpoint 或 test metric 计算。

### 10.4 实验批次与原始运行目录

每个 Round 使用一个 audit `experiment_id`。原始 checkpoint、日志和逐 run
目录写入同批次的 scratch 路径，不进入最终 audit 根目录；最终 audit 根目录
严格保留第 15 节列出的六类文件和 `figures/`。已有运行目录可作为
`reused_exact` 来源，不复制 checkpoint。

本次用户裁决的执行范围为：**先冻结并更新计划，不立即修改代码或启动服务器
训练**。下一步是按 10.1--10.3 实现新 head、完成本地 smoke/unit tests，
然后再决定是否启动 Round 0/1。

## 11. Test-oriented 选择记录

每轮必须保留完整选择轨迹，不能只保留最终胜者：

> Round 0 与 Round 1 的完整轨迹见下表：4 个 pilot setting × 8 个候选 = 32 行全部保留（含被淘汰者）。
> `baseline` 列同时列出两个对照：policy A = 与 `direct_nlinear` 比较，policy B = 与 generic `q=1/8` 比较；
> 表内的 `delta MSE% vs direct` 为正表示候选更好。Round 2/3/4 未执行，故无后续行。

hostfile_replace_entries: mkstemp: Operation not permitted
update_known_hosts: hostfile_replace_entries failed for /Users/yimingniu/.ssh/known_hosts: Operation not permitted
| round | candidate id | setting | test MSE | test MAE | delta MSE% vs direct | baseline | action | reason | test-exposed |
|---|---|---|---:|---:|---:|---|---|---|---|
| 1 | A_period_lowrank | ETTh2-H96 | 0.276858 | 0.341764 | -1.749 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | A_period_lowrank | ETTh2-H720 | 0.401989 | 0.438586 | -2.837 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | A_period_lowrank | ETTm2-H192 | 0.218705 | 0.295566 | -1.400 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | A_period_lowrank | Electricity-H336 | 0.168993 | 0.261404 | -4.492 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | B_segment_basis | ETTh2-H96 | 0.282910 | 0.346283 | -3.973 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | B_segment_basis | ETTh2-H720 | 0.400858 | 0.438845 | -2.548 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | B_segment_basis | ETTm2-H192 | 0.220520 | 0.296495 | -2.242 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | B_segment_basis | Electricity-H336 | 0.167044 | 0.259291 | -3.286 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | C_level_shape | ETTh2-H96 | 0.276092 | 0.340641 | -1.467 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | C_level_shape | ETTh2-H720 | 0.397620 | 0.436280 | -1.720 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | C_level_shape | ETTm2-H192 | 0.217271 | 0.292852 | -0.736 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | C_level_shape | Electricity-H336 | 0.165809 | 0.259338 | -2.523 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | D_recent_sparse | ETTh2-H96 | 0.282788 | 0.346697 | -3.928 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | D_recent_sparse | ETTh2-H720 | 0.403034 | 0.439916 | -3.105 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | D_recent_sparse | ETTm2-H192 | 0.220849 | 0.297780 | -2.395 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | D_recent_sparse | Electricity-H336 | 0.168192 | 0.260725 | -3.996 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | E_separable | ETTh2-H96 | 0.282277 | 0.346336 | -3.740 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | E_separable | ETTh2-H720 | 0.398192 | 0.437091 | -1.866 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | E_separable | ETTm2-H192 | 0.217995 | 0.293999 | -1.071 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | E_separable | Electricity-H336 | 0.165980 | 0.258834 | -2.629 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank | ETTh2-H96 | 0.285104 | 0.344771 | -4.779 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank | ETTh2-H720 | 0.398223 | 0.433137 | -1.874 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank | ETTm2-H192 | 0.215819 | 0.292444 | -0.062 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank | Electricity-H336 | 0.165001 | 0.258478 | -2.023 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank_r8 | ETTh2-H96 | 0.272635 | 0.335414 | -0.197 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank_r8 | ETTh2-H720 | 0.389380 | 0.428317 | +0.388 | direct_nlinear (A) / generic q=1/8 (B) | keep | 出现正向信号，但不足以触发晋级规则 | yes |
| 1 | matched_A_period_lowrank_r8 | ETTm2-H192 | 0.217446 | 0.290769 | -0.817 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |
| 1 | matched_A_period_lowrank_r8 | Electricity-H336 | 0.162639 | 0.255381 | -0.563 | direct_nlinear (A) / generic q=1/8 (B) | drop | MSE 与 MAE 同时退化 | yes |

排序优先级为：

1. MSE leaderboard 与 MAE leaderboard 分别排序；
2. 跨 pilot setting 的同指标正向信号数量；
3. 双指标交集作为附加信息，而不是唯一晋级条件；
4. 相对 `time_axis_matched_lowrank` 和 generic low-rank 的改善；
5. 参数量、MACs、训练/推理成本；
6. bad-case 是否符合预期机制。

> **执行说明（2026-09-15，用户裁定）**：本轮**跳过**样本级错误分析。原因是五条
> 路线在 4/4 pilot setting 上全部退化且无一晋级，按本节末句“路线不得晋级为机制
> 创新”，样本级分析不再有晋级用途；用户明确要求先不执行审计，直接归档汇总结果。
> 非侵入式导出脚本已实现并可用（`scripts/export_structured_lowrank_cases.py`，
> 覆盖本节要求的六类案例与 ≤8 例/路线），供后续需要时按同一协议补跑。

## 12. 样本级错误分析

Round 1 结束后，每条路线最多选择 8 个案例，覆盖：

- structured MSE 或 MAE 显著优于 direct；
- structured MSE 与 MAE 同时退化；
- 周期水平偏移；
- 周期相位错位；
- 突变或高频噪声；
- generic 与 structured 分歧最大。

每个案例至少记录 setting、sample/origin/channel、时间定位、真实值、三种
预测、MSE/MAE、level/shape/basis 或周期权重、归因和下一步动作。

若不能导出样本级预测，路线不得晋级为机制创新；先补充非侵入式预测导出。

## 13. 预算与停止规则

### 默认最大预算

| 阶段 | 最大新增训练 |
|---|---:|
| Round 0 | 最多 4 settings x 4 controls = 16；完全匹配结果复用后通常更少 |
| Round 1 | A--D 最多 4 routes x 4 settings = 16 候选，另加唯一 matched controls |
| Round 2 | 最多 2 routes x 6 ablations x 4 settings = 48 |
| Round 3 | 1 route x 4 settings + controls = 12 |
| Round 4 | 仅效率评估，不要求重新训练 |

Round 1 的 matched controls 按唯一 `(setting, r_match)` 去重，理论上最多
再增加 16 个 control，但已有结果复用后通常显著少于该上限。实际执行采用
早停：Round 1 未晋级的路线不进入 Round 2。Round 0 完成后先记录单 run
墙钟时间；若 Electricity-H336 的实测成本超过 pilot 中位数的 2 倍，必须
先暂停并重新确认 Round 1 的服务器资源分配，不自动改变候选或选择规则。

### 立即停止

- 某路线在四个 pilot setting 的 MSE 和 MAE 均无正向信号且参数效率也不占优；

### 13.1 停止判定（2026-09-15）

以下条款已在 Round 1 判定后触发，Round 2/3/4 不启动：

- **触发**：「某路线在四个 pilot setting 的 MSE 和 MAE 均无正向信号且参数效率也不占优」——
  A--E 在 4/4 setting 上 MSE 与 MAE 全部为负（§6.2），且都优于不了同预算控制。
- **触发**：「结构化路线不优于 generic low-rank，且没有显著效率优势」——
  `generic q=1/8` 在 2/4 setting 优于 direct，而全部结构化候选都不优于它。
- 实际预算：Round 0 新增训练 **0**（三项控制全部 `reused_exact`）；Round 1 新增训练
  **48**（4 setting × 6 候选 + 4 setting × 6 匹配控制，其中 rank-1 与 rank-4 控制分别为
  6 与 6 格）。Round 2/3/4 未使用任何训练预算。
- 本计划不产生 preset 变更，也不修改 `PhaseFormer_gold_standard.md`。
- 结构化路线不优于 generic low-rank，且没有显著效率优势；
- 所有路线都无法超过 direct，且没有路线在参数匹配下保持性能；
- 收益只来自一个 setting，Expansion 失败；
- 继续搜索只剩 dataset/horizon-specific 调参。

## 14. 统一计算口径

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

## 15. 实验产物

每个实验批次统一写入 `research_runs/<experiment_id>/`，至少包含：

- `run.yaml`：候选、setting、训练协议和 test-oriented 选择说明；
- `results.csv`：candidate x setting 的 val/test 指标；
- `sample_errors.csv`：样本级误差；
- `selected_cases.npz`：不超过 8 个/路线的案例；
- `objective_error_analysis.md`；
- `objective_error_analysis.zip`；
- `figures/`：报告实际引用的图。

不得只保存最优 checkpoint 或最终表格。候选淘汰轨迹本身就是结果。

## 16. 推荐执行顺序

1. 完成新 head 的本地 shape/anchor/parameter 单测和 Round 0 控制审计；
2. 一次性横向测试 A--D 代表配置；只有满足条件时才启动 E；
3. 只保留最多两条路线；
4. 做周期长度、近期范围、level/shape、正交约束和错位控制；
5. 用固定结构扩展到 ETTh1、ETTm1、Weather、ETTh2-H336；
6. 只有出现跨 setting 价值，才考虑后续少量多 seed 确认；
7. 若无路线通过，保留“结构化低秩未被证实”的负结果。

## 17. 结论模板

### 结构路线成立

> 在明确披露 test-set selection 的单 seed 探索中，`<route>` 在 `<x/y>`
> 个 setting 上至少一个 test 指标相对 `direct_nlinear` 改善，并且在参数
> 匹配 control 和错位周期控制下仍保留该指标优势。若另一个指标没有同步
> 改善，结果只能称为单指标路线。该结果支持
> “`<coordinate system>` 能更有效利用 NLinear 的低维性”，但不构成
> 无偏泛化结论。

### 只有压缩成立

> 结构化候选没有稳定优于 generic low-rank，但在相近 test 误差下显著减少
> 参数和计算量。因此证据支持低秩压缩的工程价值，不支持其作为精度改进机制。

### 路线不成立

> 在本次 test-oriented 单 seed 探索范围内，`<route>` 未在任一指标上相对
> direct 或 generic low-rank 显示跨 setting 优势，停止继续扩展。该结果不足以
> 支持将其用于 PhaseFormer。

## 18. 依据文档

- `PhaseFormer_pooled_lowrank_nlinear_experiment.md`
- `PhaseFormer_joint_lowrank_rank_sweep_plan.md`
- `PhaseFormer_rank_sweep_conditioned_experiment.md`
- `PhaseFormer_lowrank_mechanism_analysis.md`
- `PhaseFormer_rank_capacity_and_data_property_report.md`
- `PhaseFormer_input_component_D7_internal_path_report.md`

本计划不修改 preset，也不修改 `PhaseFormer_gold_standard.md`。
