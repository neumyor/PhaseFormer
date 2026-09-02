# PhaseFormer 输入成分 H1/H3/H4 因果消融实验计划

> 状态：提取器、Track R/Track F、配对 CI 与 RCRF 反事实重组已经实现并完成受限 smoke。v1.1
> （2026-09-02）已恢复正式矩阵，正分批并行执行 D0（h192×seed2021）的 240 个 validation-only
> Track R（断点可恢复；控制与产物目录见 §7.3），D0 完成后接审计 → Track F → retrained test →
> D0 汇总。工程 smoke 曾读取 ETTm2-H96 的最多128个 test窗口，仅用于链路校验，未用于选择公式、
> 模型或参数；下列表格仍为空表，不能据此形成效果结论。自 D0 在 Stage 3a-F 首次读取 test 起公式
> 与门槛冻结（§14），此后不再修改。
>
> 修订（执行顺序 v1.1，2026-09-02）：正式执行改为“决策范围优先”。`horizon=192, seed=2021`
> （记 D0，8 数据集 × 3 模型 × 10 输入条件）不再是“只读 validation 的前置通过”，而是完整走完
> Track R(validation) → 审计 → Track F → retrained test → 汇总，先形成 h192×seed2021 下的完整
> 单 seed 结论；其余 horizon×seed 作为扩展范围 D1，在 D0 结论形成后按完全相同的冻结协议补跑并入
> 三 seed 宏平均。本修订只改变执行顺序与结论汇报层级，不改任何提取公式、模型、超参、QC 阈值或
> 判定门槛。D0 在 Stage 3a-F 首次读取 test —— 自该刻起全部公式与门槛即冻结（§14）；D1 不得对
> 已冻结项做任何修改（否则需新版本并披露 D0 的 test-set exposure）。

## 1. 研究问题

本实验验证 PhaseFormer 是否没有充分利用输入中的三类信息，同时确认这些信息能否被
NLinear 或 RCRF 路径利用：

- **H1：跨周期同相位残差**。稳定的 24-step 相位模板之外，各周期在同一 phase slot 上的偏离。
- **H3：近期水平漂移与局部趋势**。去除稳定 24-step 模板后，连续时间轴上的慢变水平与斜率。
- **H4：相位漂移/时间扭曲**。不同周期相对稳定模板的峰谷位置移动及其跨周期演化。

对每个假设都比较以下三个模型：

| 模型角色 | mechanism | 结构边界 |
|---|---|---|
| M0 | `original` | 原始 phase-only PhaseFormer |
| M1 | `weak_residual` | M0 + 共享 NLinear-style residual；普通静态 gate |
| M2 | `rcrf_nlinear_plain` | M0 + 同一 NLinear residual + RCRF；无额外 phase calibration |

M2 不使用 `gold_combo_reliability_s2`，因为后者同时启用 uncertainty shrinkage、period-level
calibration 和 high-frequency damping，无法把差异归因到 RCRF。

本实验不把“未达到统计显著”解释为“PhaseFormer 没利用 A”。“未利用”必须由等效性检验支持；
“A 有用”必须同时由删除效应、剂量响应和 matched disturbance 支持；residual probe 保留为可选的
增强诊断，不作为当前主判定的必要条件。

## 2. 共同数据与无泄漏约束

### 2.1 数据范围

- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather、Electricity、Traffic。
- lookback 固定为 720，horizon 为 96/192/336/720。
- `period_len=24`，因此每个输入窗口恰好包含 `K=30` 个周期。
- seeds：2021、2022、2023。
- **决策范围 D0（优先）**：先固定单个 seed `2021`、`horizon=192`，覆盖全部 8 个数据集、3 个模型
  和 10 个输入条件（Track R 共 240 个训练任务）。D0 是自足决策范围：Track R(validation) →
  validation 审计 → Track F（24 个 full 锚点 × 10 输入）→ retrained test（216 个非 full
  checkpoint）→ D0 汇总，形成 h192×seed2021 下的完整结论。由于仅 seed=2021，D0 结论为
  **单 seed 判定**（moving-block CI 仍按预测起点时间序列计算，故可行），并在汇报中显式标注
  single-seed provisional，最终分级以 D1 三 seed 宏平均为准。
- **扩展范围 D1**：D0 结论形成后，按与 D0 完全相同的冻结协议扩展到 `horizon∈{96,336,720} ×
  seed=2021` 与 `全部 horizon × seed∈{2022,2023}`，得到完整 2880 个 Track R、288 个 full
  锚点与 2592 个 retrained test，用于三 seed 宏平均与最终分级（含对 D0 单 seed 结论的复查）。
- full-train；按最低 validation loss 选择 checkpoint；冻结配置后每个 checkpoint×input-condition
  组合只评估一次 test，期间权重不更新。
- 七个有论文 Golden 的数据集同时报告 Golden 与 matched comparison；Exchange 只报告 matched
  comparison，不补造 Golden。

### 2.2 分解位置

提取器作用在数据集使用训练集统计量标准化后的 `x_enc` 上、模型内部 RevIN 之前。标准化器只在
原始训练集拟合一次；`half_A`、`minus_A`、`sham` 不重新拟合 scaler。由于标准化是逐通道仿射
变换，加性分解在该空间仍可精确重构，且不同数据集的稳健阈值具有一致量纲。

每个窗口只能读取自己的 720 步历史。提取器不得读取 `y`、未来时间戳、validation/test 指标或
其他窗口在预测起点之后的数据。时间标记 `x_mark/y_mark` 保持不变。

### 2.3 四种输入

每个假设生成四种输入，名称固定：

| input_variant | 含义 |
|---|---|
| `full` | 原输入 `X`；三个假设共享同一组训练 run |
| `half_A` | 删除 50% 的 A，或完成 50% 的几何干预 |
| `minus_A` | 完全删除 A，或把对应关系压平为无演化状态 |
| `sham` | 保留与 A 相近的能量/平滑度/位移量，但破坏正确时间对应关系的 matched disturbance |

`sham` 不是“保证无影响”的安慰剂；它用于估计一般平滑、重排、插值和分布漂移本身的影响。
所有需要随机重排的操作使用独立于训练 seed 的固定 `intervention_seed=9102`，并由
`dataset/split/window_start/hypothesis` 派生确定性子 seed，使三个模型看到逐元素相同的干预输入。
同一窗口所有通道共享置换，所以子 seed 故意不含 channel。

除 H4 的几何变换外，主分析都强制：

```text
transformed_x[-1, channel] == full_x[-1, channel]
```

这是因为 NLinear 显式以最后值为 persistence anchor；不保持末值会把“A 的历史信息”与“最后值
被破坏”混在一起。附加审计可报告不保末值版本，但不得替代主结果。

## 3. H1：跨周期同相位残差

### 3.1 提取

对每个窗口、每个通道，将 `X` reshape 为 `X[k,p]`，其中 `k=0..29`、`p=0..23`。逐 phase
使用跨周期中位数构造稳健模板：

```text
T[p] = median_k X[k,p]
B0[k,p] = T[p]
d = X[29,23] - B0[29,23]
B[k,p] = B0[k,p] + d
A[k,p] = X[k,p] - B[k,p]
```

中位数避免少数尖峰污染相位模板；常数 `d` 保证最后输入值不变。定义后必须满足 `X=A+B`
和 `A[29,23]=0`（数值容差 `1e-6`）。

### 3.2 四种输入

```text
full    = X
half_A  = X - 0.5 * A
minus_A = B
```

`sham` 使用固定周期置换 `pi` 重新排列完整残差周期，保留每周期内部形状与残差能量：

```text
A_perm0[k,p] = A[pi(k),p]
A_perm1      = A_perm0 - A_perm0[29,23]
A_perm       = A_perm1 * rms(A) / max(rms(A_perm1), eps)
sham         = B + A_perm
```

同一窗口的所有通道使用相同 `pi`，避免人为破坏跨通道同步关系；虽然当前三个模型不做
cross-channel fusion，也不应让干预附带无关的跨通道破坏。

### 3.3 提取质量检查

- `max_abs(X-(A+B)) <= 1e-6`。
- `max_abs(last(B)-last(X)) <= 1e-6`。
- `sham` 与 `full` 的 residual RMS 比位于 `[0.99,1.01]`。
- `minus_A` 的同相位跨周期方差相对 `full` 至少降低 95%。
- `half_A` 的残差 RMS 比应为 `0.5±0.01`。

## 4. H3：近期水平漂移与局部趋势

### 4.1 提取

先用 H1 的稳健模板 `T[p]` 去除稳定相位主体，但不使用 H1 的末值校正：

```text
R[t] = X[t] - T[t mod 24]
```

对 `R` 使用固定、因果 EMA 提取慢变轨迹。平滑窗口冻结为 `W=96`，
`alpha=2/(W+1)`：

```text
m[0] = median(R[0:24])
m[t] = alpha * R[t] + (1-alpha) * m[t-1]
A[t] = m[t] - m[719]
B[t] = X[t] - A[t]
```

减去 `m[719]` 使 `A[719]=0`：删除的是“如何走到当前水平”的历史轨迹，而不是当前水平本身。
`W=96` 在正式 test 前固定，不按数据集或 horizon 调整。

### 4.2 四种输入

```text
full    = X
half_A  = X - 0.5 * A
minus_A = B
```

`sham` 使用时间反转的低频轨迹，保留平滑度并匹配 RMS，但破坏其正确方向与时间位置：

```text
A_rev[t]  = A[719-t] - A[0]       # 保证 A_rev[719] = 0
A_sham    = A_rev * rms(A) / max(rms(A_rev), eps)
sham      = B + A_sham
```

实现不对 `rms(A)<1e-8` 做硬回退：这种 A 的删除改变量本身低于数据精度相关尺度，不影响主实验；
`sham` 的 RMS 匹配仅在候选 RMS 大于 `1e-12` 时缩放，否则候选置零，避免除以极小量制造伪干预。

### 4.3 提取质量检查

- `max_abs(X-(A+B)) <= 1e-6`，且 `last(B)==last(X)`。
- `half_A` 的 A-energy 比为 `0.5±0.01`。
- `sham` 与 A 的 RMS 比位于 `[0.99,1.01]`。
- 对最近 96 步做 OLS，`minus_A` 的 deseasonalized slope 绝对值相对 `full` 至少降低 80%；未达到
  时标记 `weak_intervention`，不能作为确认样本。
- 稳定 24-step 模板相关性变化不超过 1%。

## 5. H4：相位漂移/时间扭曲

### 5.1 位移估计

对每个周期先减去自身中位数并除以 MAD，以免位移估计被水平/振幅支配。稳健模板为标准化周期
在 `k` 轴上的逐 phase 中位数。对每周期枚举整数循环位移 `d in [-6,6]`，用归一化互相关选择
峰值，再用峰值及左右相邻点做三点抛物线插值，得到截断在 `[-6,6]` 的小数位移。实现冻结
`MAD epsilon=1e-6`、最低相关峰值 `0.15`：

```text
delta[k] = argmax_d corr(standardize(X[k]), circular_shift(T, d))
```

整数峰并列时依次选择 `abs(d)` 更小、再选择 `d` 更小者，确保确定性。位移只从该窗口历史估计。

### 5.2 四种输入

定义 `Shift(cycle,u)` 为基于 Fourier shift theorem 的循环位移；整数与小数 `u` 都可用，且应保持
周期均值和 L2 能量。令 `target_full[k]=delta[k]`：

```text
full target:    delta_full[k]  = delta[k]
half target:    delta_half[k]  = 0.5*delta[k] + 0.5*delta[29]
minus target:   delta_minus[k] = delta[29]
```

输入通过 `Shift(X[k], delta_variant[k]-delta[k])` 生成。因此 `minus_A` 将所有历史周期对齐到最新
周期的相位，删除 phase-shift 演化但保留预测起点处的最新相位。

`sham` 保留观测位移序列的分布和变化量，但用固定周期置换破坏正确时间顺序：

```text
delta_sham[k] = delta[pi(k)] - delta[pi(29)] + delta[29]
sham[k]       = Shift(X[k], delta_sham[k]-delta[k])
```

三个模型使用完全相同的 `pi`。所有 variant 的最后一个周期保持不动。

### 5.3 提取质量检查

- synthetic recovery：已知整数位移序列的恢复准确率至少 99%；已知半步位移平均绝对误差不超过
  0.25 step。
- 所有 variant 的最后周期逐元素相同。
- Fourier shift 前后每周期均值相对误差 `<1e-6`、L2-energy 相对误差 `<1e-5`。
- `minus_A` 重新估计的 shift 方差相对 `full` 至少降低 95%。
- `half_A` 的 shift 标准差比为 `0.5±0.05`。
- 若周期 MAD 不高于 `1e-6` 或相关峰值低于 `0.15`，该周期/通道保持为 `full`。只有最新周期
  可识别且至少 50% 历史周期可识别时，该窗口/通道才允许干预；否则该通道四种输入均退化为
  `full`。`sham` 的置换 donor 不可识别时，对应周期/通道也保持不动，不得用默认零位移强加干预。

### 5.4 Fourier 实值约束

小数循环位移对 `rfft` 系数乘单位模相位。偶数周期下 Nyquist 系数没有共轭伙伴，为同时保证输出
严格为实数且保持 L2 能量，该系数保持不变；其余频点只改变相位。因而本实验的 `Shift` 是
“除 Nyquist 模式外的 Fourier circular shift”：对含 Nyquist 能量的信号，奇数整数 shift 不等同于
逐元素 `roll`，这是主动保留能量的约定，不影响相位估计使用的整数 `roll`。QC 单独报告 Nyquist
能量占比；若该占比不可忽略，H4 只能解释为低于 Nyquist 的相位漂移。

## 6. 实现前必须通过的测试

已实现无模型依赖的输入干预模块和独立 runner；开始任何正式
训练前必须通过：

1. H1 人工模板+残差的精确重构、末值保持、置换能量测试。
2. H3 人工线性/分段趋势的方向、剂量、近零分量与末值保持测试。
3. H4 已知整数/半整数 shift 恢复、能量保持和不可识别窗口 fallback 测试。
4. 所有假设 `(B,L,C)` shape、dtype、device、NaN/Inf、单变量/多变量测试。
5. 相同窗口和 `intervention_seed` 的逐元素确定性；三个模型接收输入 hash 一致。
6. 代码级 no-future-access 测试：修改预测目标或输入窗口之后的数据，不得改变任何 variant。
7. dataset scaler 不重拟合、`x_mark/y_mark` 不改变。
8. 每个假设至少完成一个 CPU 1-batch forward smoke；随后完成 ETTm2-96、5% 数据、1 epoch、
   不读取 test 的训练 smoke。

任何质量检查失败都先修复提取器，不得通过跳过样本或放宽正式阈值继续训练。

实现文件冻结为：

- `src/dataset/input_component_ablation.py`：H1/H3/H4 提取、四输入和 dataset 只读包装器；
- `scripts/search_phaseformer.py`：单个 Track R 训练入口，配置哈希包含 hypothesis/variant/seed；
- `scripts/run_input_component_ablation.py`：去重后的 10 条输入条件矩阵生成/执行器；
- `scripts/evaluate_input_component_checkpoint.py`：同一 checkpoint 的 Track F 十条件配对评估、
  moving-block bootstrap 和 RCRF 分支诊断；
- `scripts/run_input_component_frozen_matrix.py`：从 Track R 的 `none/full` 结果发现并审计唯一
  checkpoint，生成/执行完整 Track F；
- `scripts/run_input_component_retrained_test_matrix.py`：审计完整 validation-only Track R 后，只对
  2592 个非 full checkpoint 做一次匹配输入 test；共享 full 结果直接复用 Track F；
- `scripts/evaluate_input_component_retrained_checkpoint.py`：单个 retrained checkpoint 的只读 test；
- `scripts/summarize_input_component_ablation.py`：矩阵完整性、重复行、checkpoint hash 审计，
  sham-adjusted 删除效应和相对 original 的 interaction 汇总。

## 7. 实验矩阵与执行阶段

### 7.1 两条互补证据链

**Track R：retrain without A**

每个模型分别在 `full/half_A/minus_A/sham` 上从头训练。它回答：当 A 不可用时，模型能否通过
其他成分补偿，以及完整训练后 A 的边际价值。

**Track F：frozen-checkpoint intervention**

只训练 `full` checkpoint，再在同一个 checkpoint 上测试九种 H1/H3/H4 干预输入。它回答已经
训练好的模型实际依赖 A 的程度。Track F 不允许在干预输入上继续训练或选择 checkpoint。

两条证据必须分别报告：retrain 效应小可能来自模型适应，frozen 效应大也可能包含输入分布漂移。

### 7.2 阶段

1. **Stage 0：提取器审计**。全数据集仅跑输入统计，不训练、不读取 test target。
2. **Stage 1：smoke**。按第6节执行，只验证链路，不形成效果结论。
3. **Stage 2：validation rehearsal**。在 ETTm2-96、ETTh2-720、Weather-96 上用完整训练集和
   seed 2021 对每个假设跑通四输入×三模型；只读 validation，用于发现训练/资源错误，不据此修改
   提取公式。
4. **Stage 3a：D0 Track R（决策范围训练）**。冻结代码和配置后，先运行 8 数据集 × `horizon=192`
   × 单 seed `2021`（240 个 validation-only Track R 任务）；完成后按 §6/§7.3 做 validation
   审计（完整性、无泄漏、健康度），审计通过才进入 Stage 3a-F。
5. **Stage 3a-F：D0 test 与机制分析**。对 D0 的 24 个 `none/full` 锚点 checkpoint 执行 Track F
   （每个×10 输入 = 240 次评估，其中 24 次 `none/full` 同时作为 Track R 基线），再对 D0 的 216
   个非 full checkpoint 执行单次 matched-input retrained test；随后运行 D0 汇总，填 §13.0 的
   D0 表并给出 **h192×seed2021 单 seed 结论（provisional）**。注意：此即 D0 首次读取 test ——
   读取后本计划全部公式与门槛即冻结（§14）。
6. **Stage 3b：D1 Track R（扩展范围训练）**。D0 结论形成后，扩展到其余
   `horizon{96,336,720}×seed2021` 与 `全部 horizon×seed{2022,2023}`（合计 2640 个
   validation-only Track R 任务），与 D0 的 240 个 run 构成完整 2880；先做 validation 审计。
7. **Stage 3b-F：D1 test 与全矩阵汇总**。对全部 288 个 full 锚点执行 Track F、对 2592 个非
   full checkpoint 执行 retrained test，运行全矩阵汇总，得到三 seed × 四 horizon 的宏平均与
   最终分级，并把 D0 单 seed 结论并入三 seed 复查。每个 checkpoint×input-condition 组合只评估
   一次 test、期间权重不更新；test 后不得回头修改 H1/H3/H4，任何后续版本使用新 experiment ID
   并披露 test-set selection。

三个假设共享 `full` run。唯一输入条件数为 `1 + 3 hypotheses × 3 interventions = 10`，所以
Track R 正式训练规模为：

```text
10 input conditions × 3 models × 32 settings × 3 seeds = 2880 training runs
```

Track F 对288个 full checkpoint 评估10种输入，其中288个 `none/full` 同时作为 Track R 基线，
其余为 `288×9=2592` 次干预评估。Track R 再评估2592个非 full checkpoint，因此共有5472个唯一
checkpoint×input-condition test。正式启动前必须记录 GPU、单 run 时间与预计总成本；可按数据集
分批调度，但不能缩减某个模型或干预造成不平衡矩阵。

按决策范围拆分，test 唯一单元为：**D0（h192×seed2021）= 24 锚点 full + 24×9 Track F 干预 +
24×9 retrained = 456**；**D1（其余）= 264 锚点 full + 264×9 ×2 干预/retrained = 5016**；
456 + 5016 = 5472。两个范围先后各自先完成 validation-only Track R 与审计、再读取本范围的
test；D1 必须复用与 D0 完全相同的冻结提取、模型与 runner（含 checkpoint 哈希审计）。

### 7.3 复现命令

validation-only rehearsal（默认只打印 30 条命令，加 `--execute` 才执行）：

```bash
.venv/bin/python scripts/run_input_component_ablation.py \
  --datasets ETTm2 --horizons 96 --seeds 2021 --max-epochs 30 --allow-cpu
```

正式 Track R 训练严格 validation-only，完整默认矩阵生成 2880 条唯一训练命令，三个假设共享
`none/full`；默认向每条命令传递 `--require-cuda`。

D0（h192×seed2021）共 `8×3×10=240` 个 Track R 训练任务，是 D1 全矩阵（§7.2 Stage 3b）的前缀
而不是独立调参实验；代码默认启用 `--priority-first`，即使直接执行完整矩阵也会先调度
`horizon=192, seed=2021`。D0 Track R（validation-only）：

```bash
python scripts/run_input_component_ablation.py \
  --horizons 192 --seeds 2021 \
  --output-dir research_runs/input_components_h134_scratch \
  --execute --resume
```

D0 Track R validation 审计通过后，进入 D0 下游（命令入口需按范围参数化，见下方实现说明）：

D0 Track F —— 只对 D0 的 24 个 `none/full` 锚点：

```bash
python scripts/run_input_component_frozen_matrix.py \
  --track-r-dir research_runs/input_components_h134_scratch \
  --output-dir research_runs/input_components_h134_frozen_d0 \
  --horizons 192 --seeds 2021 --expected-count 24 --execute
```

D0 retrained test —— 只对 D0 的 216 个非 full checkpoint：

```bash
python scripts/run_input_component_retrained_test_matrix.py \
  --track-r-dir research_runs/input_components_h134_scratch \
  --output-dir research_runs/input_components_h134_retrained_test_d0 \
  --horizons 192 --seeds 2021 --expected-count 216 --execute
```

D0 汇总 —— 输出单 seed h192 表与 provisional 结论（§13.0）：

```bash
python scripts/summarize_input_component_ablation.py \
  research_runs/input_components_h134_frozen_d0 \
  research_runs/input_components_h134_retrained_test_d0 \
  --horizons 192 --seeds 2021 --output /path/to/result_summary_d0.csv
```

> 实现说明（必做项，已在 D0 Track R 收尾前落地）：`run_input_component_frozen_matrix.py`、
> `run_input_component_retrained_test_matrix.py` 与 `summarize_input_component_ablation.py` 均
> 接受 `--horizons/--seeds` 范围过滤，`expected-count`/`expected-settings-per-track` 由范围自动
> 推导（D0 = 24 锚点 / 216 retrained；全矩阵 = 288 / 2592）。retrained 入口在范围子集上只校验
> 该范围内的 validation Track R 完整性、无泄漏与 100% 采样，源目录中出现范围外（未完成的 D1）
> 条件不会阻塞 D0 读取。D0 下游产物写入独立目录（`*_d0`），避免与 D1 全矩阵产物混用；D1 收尾
> 再写全矩阵目录（不带过滤参数即为全矩阵默认）。

D1 / 全矩阵 Track R（D0 结论形成后执行，对应 §7.2 Stage 3b）：余下
`horizon∈{96,336,720} × seed2021` 与全部 horizon × `seed∈{2022,2023}` 共 2640 个
validation-only 训练任务，与 D0 的 240 个合并成完整 2880：

```bash
python scripts/run_input_component_ablation.py \
  --output-dir research_runs/input_components_h134_scratch \
  --execute --resume
```

D1 Track R 完成后先做 validation 审计（§6），通过才进入 D1 下游 Track F。任意单 checkpoint 的
Track F（D0/D1 通用）：

```bash
python scripts/evaluate_input_component_checkpoint.py \
  --dataset ETTm2 --horizon 96 --model rcrf_nlinear_plain \
  --checkpoint /path/to/full/best.ckpt \
  --output-dir /path/to/frozen_eval --require-cuda
```

D1 完整 Track F —— 在确认 Track R 已有全部 288 个唯一 `none/full` checkpoint 后执行：

```bash
python scripts/run_input_component_frozen_matrix.py \
  --track-r-dir research_runs/input_components_h134_scratch \
  --output-dir research_runs/input_components_h134_frozen \
  --expected-count 288 --execute
```

D1 retrained test —— 随后只评估 2592 个 retrained 非 full checkpoint；该入口会先确认源目录
具有完整 2880 个条件且尚未包含任何 test 指标：

```bash
python scripts/run_input_component_retrained_test_matrix.py \
  --track-r-dir research_runs/input_components_h134_scratch \
  --output-dir research_runs/input_components_h134_retrained_test \
  --expected-count 2592 --execute
```

其中 `--max-samples` 和训练入口的 `--max-eval-samples` 只允许 smoke，正式结果必须保持默认值 0。
D1 全矩阵汇总命令会拒绝缺条件、重复条件或 frozen 条件混用不同 checkpoint：

```bash
.venv/bin/python scripts/summarize_input_component_ablation.py \
  research_runs/input_components_h134_frozen \
  research_runs/input_components_h134_retrained_test \
  --output /path/to/result_summary.csv
```

实验主日志、阶段日志和每30分钟监控记录统一保存在
`research_runs/input_components_h134_control/`，不得写入 `/tmp`。所有正式矩阵入口默认要求 CUDA、完整 checkpoint 数量、`percent=100`、训练期
`max_eval_samples=0`、测试期 `max_samples=0`，并拒绝已经含 test 指标的训练源。CPU 或部分样本
只能分别通过显式 `--allow-cpu`、`--smoke --max-samples N` 使用；正式汇总默认拒绝这些结果。

## 8. 指标、统计量与因果判定

### 8.1 基本效应

对 metric `L`（MSE/MAE，越低越好），定义每个模型的相对删除效应：

```text
Delta(M,H,V) = L(M,H,V) / L(M,full) - 1
```

其中 `V in {half_A,minus_A,sham}`。架构交互效应为：

```text
Interaction(M,H,V) = Delta(M,H,V) - Delta(M0,H,V)
```

正 Interaction 表示该增强模型比原始 PhaseFormer 更依赖该成分/对应关系。setting 宏平均对每个
dataset×horizon 等权，不能按测试窗口数加权，让 Traffic/Electricity 支配结论。

### 8.2 不确定性

- 报告三 seed mean和每 seed 明细；正式报告阶段再排版 sample std。
- MSE/MAE 的绝对效应与相对效应 CI 均使用连续预测起点的 moving-block bootstrap，block length
  固定为 `pred_len`，避免把
  高度重叠窗口当独立样本。
- setting 宏平均再对32个 setting 做分层 bootstrap；同时报告7个 Golden setting 家族与 Exchange
  的结果，不能只报告总体平均。
- 当前自动汇总不执行多重检验；主判定使用预注册效应门槛及原始 CI，不报告未经实现的 Holm 结果。
- **D0（h192×seed2021）provisional 语义**：D0 阶段只有单 seed，CI 仍是同一 moving-block
  bootstrap 规则，但不得声称跨 seed 稳定；D0 表格一律标 `provisional (seed2021 only)`。只有 D1
  三 seed 全矩阵汇总才能按本节正常规则给出最终分级。

### 8.3 判定规则

setting 级“PhaseFormer 对 A 等效不敏感”要求 M0 的 `minus_A` MSE 和 MAE 的 95% CI 都完全位于
`[-0.5%, +0.5%]`，而不是简单的 `p>0.05`。

增强模型“实质依赖 A”要求：

- `minus_A` 相对退化至少 1.0%，95% CI 下界大于0；
- `half_A` 的效应方向一致，并原则上位于 `full` 与 `minus_A` 之间；
- 相对 M0 的 `Interaction(minus_A)` 至少 +0.5%，CI 下界大于0；
- `minus_A` 的退化不能被 `sham` 的同等或更大退化完全解释。

结论分级：

| 等级 | 预注册含义 |
|---|---|
| Strong | M0 等效不敏感；M1/M2 都满足依赖、剂量和 Interaction；sham 不解释结果 |
| Partial | M0 等效不敏感；仅 M1 或 M2 满足依赖，或只有部分数据集家族复现 |
| Model-shared | M0、M1、M2 都明显退化；A 有用，但没有证据证明 PhaseFormer 特别未利用 |
| OOD/confounded | `sham` 与 `minus_A` 同等或更差，或质量检查显示大量分布破坏 |
| Rejected/null | M0 不满足等效性且没有稳定 Interaction，或三模型都无稳定删除效应 |

“Strong”不要求增强模型的 full 指标一定超过论文 Golden；本实验研究输入利用机制。任何性能提升
声明仍必须另外对固定 Golden 报告。

**D0（h192×seed2021）结论分级**：D0 阶段先按同一预注册门槛与分级表给出单 seed 判定，在 §13.0
记录为 `provisional (seed2021 only)`，并标明每个数据集家族在 D0 是否已有足够样本、哪些需等 D1
补足。D0 的任何 Strong/Partial 不得表述为跨 seed 稳健，须经 §7.2 Stage 3b-F 三 seed 复查确认后
才写入 §13.7 最终结论。

## 9. PhaseFormer residual probe（可选、当前未自动化）

为了直接检验 A 是否包含 PhaseFormer 尚未解释的信息，对 M0/full 的预测残差
`e=y-y_hat_M0` 建立简单线性 ridge probe：

- H1/H3 每通道将对应 A 的720步历史映射到该 horizon 的残差；H4 使用该通道30个周期的
  `delta[k]`、相邻差分及相对最新位移作为 probe 输入；均不做 cross-channel 输入。
- 在 validation origins 上按时间连续五折 cross-fitting，ridge 系数只在折内训练；lambda 网格在
  validation 内部冻结。
- 正式 test 前，用全部 validation 拟合一次 probe；test 只评估一次。
- 比较 `zero correction`、`sham-A probe` 与 `true-A probe` 的 MSE/MAE。

若后续实现，只有 true-A probe 在 held-out 数据稳定优于 zero 和 sham，才能把 A 称为
“PhaseFormer 残差中仍可预测的信息”。probe 不计入当前三个主模型的正式预测排名或主证据等级。

## 10. RCRF 机制拆解

M2 的 A 干预同时可能改变 PhaseFormer 分支、NLinear 分支、可靠度 `r` 和融合权重 `alpha`。正式
实现需导出每个 sample×channel 的：

```text
y_phase, y_nlinear, r, alpha, y_fused
```

固定 checkpoint 评估会流式重组以下四类反事实预测，不保存巨大的全量分支张量：

```text
F(P_variant, N_variant, alpha_full)     # 固定 gate，只看两个分支变化
F(P_full, N_full, alpha_variant)        # 固定分支，只看 gate 变化
F(P_variant, N_full, alpha_full)        # PhaseFormer 路径贡献
F(P_full, N_variant, alpha_full)        # NLinear 路径贡献
```

并校验 `F(P,N,alpha)` 与模型真实 fused 输出最大误差 `<2e-5`。若总体删除效应主要来自
`alpha_variant` 的变化，而 NLinear 分支本身没有利用 A，不得声称“NLinear 提取了 A”。

## 11. 样本级分析（正式报告阶段，当前 runner 未自动化）

每个 H×dataset×horizon 按程序规则选取以下案例，不人工挑图：

1. **underuse evidence**：M0 `minus_A-full` 接近0，但 M1/M2 退化最大的 Top-K。
2. **PhaseFormer-reliant counterexample**：M0 删除后退化最大的 Top-K。
3. **sham-sensitive/OOD**：sham 退化不小于 minus_A 的 Top-K。
4. **dose violation**：half_A 不位于 full 与 minus_A 之间的 Top-K。
5. **high-error**：M0/full 原始误差最大的 Top-K，防止只分析改善样本。

相同通道的案例预测区间不得重叠。图中同时显示输入、A、full/half/minus/sham、真值、三个模型预测；
H4 额外显示 `delta[k]`，H3 显示 `m[t]`，H1 显示模板与 residual energy。报告各类别总体占比，
不能把 Top-K 当作总体频率。

## 12. 审计产物

正式实验统一使用 `experiment_id=phaseformer_input_components_h134_v1`。最终目录遵循项目白名单：

```text
research_runs/phaseformer_input_components_h134_v1/
  run.yaml
  results.csv
  sample_errors.csv
  selected_cases.npz
  objective_error_analysis.md
  objective_error_analysis.zip
  figures/
```

多数据集、horizon、seed、hypothesis、input variant、model 和 track 通过显式字段区分，不按 setting
拆目录。checkpoint、训练日志和全量预测只放临时 scratch；汇总与校验完成后清理，不进入最终目录。
当前脚本负责生成和审计 scratch CSV/NPZ；严格六文件报告包属于正式报告阶段，尚未由本组 runner
自动生成，不能把 scratch 目录直接当作最终审计目录。

`results.csv` 至少包含：`setting,dataset,horizon,seed,hypothesis,input_variant,model,track,mse,mae,
delta_mse,delta_mae,interaction_mse,interaction_mae,component_energy,qc_status,selection_source`。

## 13. 空结果表

### 13.0 D0（h192×seed2021）决策范围表（provisional）

- 该范围 = 24 个 `none/full` 锚点 Track F（每个×10 输入，其中 24 次 `none/full` 兼作 Track R
  基线）+ 24 锚点的 9 项干预 + 216 个 retrained test，合计 456 个唯一 test 单元（§7.2）。D0 首次
  读取 test 发生在 Stage 3a-F，自该刻起公式与门槛冻结（§14）。
- D0 明细直接复用 §13.2/§13.3/§13.4 的表结构，但行只填 `horizon=192, seed=2021` 且每格加注
  `[D0/provisional]`；CI 仍为单 seed moving-block bootstrap（§8.2），不因样本更少而放宽容差，
  也不把单 seed 表现升级为跨 seed 结论。
- D0 汇总结论（逐 H、逐数据集家族）：

| H | M0 equiv (D0) | M1 dep (D0) | M2 dep (D0) | Interaction (D0) | families covered in D0 | provisional grade |
|---|---|---|---|---|---|---|
| H1 | — | — | — | — | — | — |
| H3 | — | — | — | — | — | — |
| H4 | — | — | — | — | — | — |

### 13.1 提取质量

| Hypothesis | Dataset | Variant | Reconstruction max err | Last-point max err | Target-stat reduction | Energy ratio | Invalid windows | QC |
|---|---|---|---:|---:|---:|---:|---:|---|
| H1 | — | half/minus/sham | — | — | — | — | — | — |
| H3 | — | half/minus/sham | — | — | — | — | — | — |
| H4 | — | half/minus/sham | — | — | — | — | — | — |

### 13.2 Track R：三 seed 正式 test 明细

| H | Dataset | Horizon | Model | Variant | MSE mean±std | MAE mean±std | ΔMSE | ΔMAE | 95% CI | Dose pass | Sham pass |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|---|
| — | — | — | — | — | — | — | — | — | — | — | — |

### 13.3 Track F：固定 checkpoint 干预

| H | Dataset | Horizon | Model | Variant | MSE | MAE | ΔMSE | ΔMAE | Block-bootstrap CI | Frozen/retrain agreement |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| — | — | — | — | — | — | — | — | — | — | — |

### 13.4 架构交互与等效性

| H | Scope | M0 equivalence | M1 Interaction MSE/MAE | M2 Interaction MSE/MAE | CI-based result | Evidence grade |
|---|---|---|---:|---:|---|---|
| H1 | — | — | — | — | — | — |
| H3 | — | — | — | — | — | — |
| H4 | — | — | — | — | — | — |

### 13.5 Residual probe

| H | Dataset | Horizon | Zero MSE/MAE | Sham-A MSE/MAE | True-A MSE/MAE | True vs zero | True vs sham | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H1 | — | — | — | — | — | — | — | — |
| H3 | — | — | — | — | — | — | — | — |
| H4 | — | — | — | — | — | — | — | — |

### 13.6 RCRF 路径归因

| H | Setting | Variant | Δ phase-only | Δ NLinear-only | Δ gate-only | Δ actual fused | mean r | mean alpha | Main pathway |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | — |

### 13.7 最终结论

| Hypothesis | PF underuses A? | A predictively useful? | NLinear uses A? | RCRF adds conditional use? | Cross-dataset stability | Decision |
|---|---|---|---|---|---|---|
| H1 | — | — | — | — | — | — |
| H3 | — | — | — | — | — | — |
| H4 | — | — | — | — | — | — |

## 14. 已知风险与停止条件

- H1 是总残差，H3/H4 是其中可能重叠的子结构；三者分别做干预，不把效应相加。
- H1/template-only 是最强分布改变，必须结合 half_A、sham 和 Track R/F 判断。
- H3 的 EMA 是操作性定义，不声称等于真实生成趋势；W=96 的结论只对该预注册尺度成立。
- H4 的循环 shift 假定周期边界可连接；能量保持不代表没有边界伪影，必须报告 sham。
- 若某假设超过20%的 test窗口为 `weak_intervention/unidentifiable`，该数据集只报告“提取不可识别”，
  不给肯定或否定机制结论。
- 若 smoke 或 Stage 2 显示输入 hash 不一致、test 被提前访问、QC 失败或融合无法重构，立即停止，
  修复后重新冻结 experiment ID；不能带缺陷进入正式矩阵。
- test 一旦读取，本计划的公式和门槛即冻结。后续任何改动必须新建版本并披露测试集暴露。
- **D0 优先使冻结时刻提前**：按 §7.2 Stage 3a-F，首次 test 读取出现在 D0（h192×seed2021）完成
  时，早于 D1 全矩阵。因此自 D0 test 读取起，提取公式、效应/等效门槛与 Track F/retrained/汇总
  逻辑即冻结；D1 不得为适配 D0 单 seed 结果调整任何门槛，只能报告三 seed 复查对 provisional
  结论的确认或推翻。
