# PhaseFormer 输入成分 H1/H3/H4 因果消融实验计划

> 状态：预注册草案，尚未实现提取器、启动训练或读取本实验 test。下列表格均为空表；任何修改
> 提取公式、阈值、数据范围或判定标准的行为都必须发生在正式 test 之前并留下记录。

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
“A 有用”必须同时由删除效应、剂量响应、matched disturbance 和 held-out residual probe 支持。

## 2. 共同数据与无泄漏约束

### 2.1 数据范围

- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather、Electricity、Traffic。
- lookback 固定为 720，horizon 为 96/192/336/720。
- `period_len=24`，因此每个输入窗口恰好包含 `K=30` 个周期。
- seeds：2021、2022、2023。
- full-train；按最低 validation loss 选择 checkpoint；冻结配置后每个 checkpoint 只读一次 test。
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
`dataset/split/window_start/channel/hypothesis` 派生确定性子 seed，使三个模型看到逐元素相同的
干预输入。

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

若 `rms(A)<1e-8`，该窗口四种输入均退化为 `full`，并在质量表记录为 `near_zero_component`，不得
通过除以极小量制造伪干预。

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

小数循环位移对 `rfft` 系数乘单位模相位。偶数周期下 Nyquist 系数没有共轭伙伴，为保证输出严格
为实数，该系数保持不变；其余频点只改变相位。因此均值与每周期 L2 能量保持，避免 H4 携带额外
的振幅删减。

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
4. **Stage 3：正式完整矩阵**。冻结代码和配置后，运行8数据集×4 horizon×3 seed。
5. **Stage 4：一次性 test 与机制/样本分析**。每个 best-validation checkpoint 只读一次 test；
   test 后不得回头修改 H1/H3/H4。任何后续版本使用新 experiment ID 并披露 test-set selection。

三个假设共享 `full` run。唯一输入条件数为 `1 + 3 hypotheses × 3 interventions = 10`，所以
Track R 正式训练规模为：

```text
10 input conditions × 3 models × 32 settings × 3 seeds = 2880 training runs
```

Track F 复用 288 个 full checkpoint，增加 `288 × 9 = 2592` 次无训练评估。正式启动前必须记录
GPU、单 run 时间与预计总成本；可按数据集分批调度，但不能缩减某个模型或干预造成不平衡矩阵。

### 7.3 复现命令

validation-only rehearsal（默认只打印 30 条命令，加 `--execute` 才执行）：

```bash
.venv/bin/python scripts/run_input_component_ablation.py \
  --datasets ETTm2 --horizons 96 --seeds 2021 --max-epochs 30
```

正式 Track R 在代码/配置冻结后显式加入 `--evaluate-test --execute`；不传 `--evaluate-test` 时绝不
构造 test loader。完整默认矩阵打印 2880 条唯一训练命令，三个假设共享 `none/full`。

单 checkpoint 的 Track F：

```bash
.venv/bin/python scripts/evaluate_input_component_checkpoint.py \
  --dataset ETTm2 --horizon 96 --model rcrf_nlinear_plain \
  --checkpoint /path/to/full/best.ckpt \
  --output-dir /path/to/frozen_eval
```

完整 Track F 在确认 Track R 有 288 个唯一 `none/full` checkpoint 后执行：

```bash
.venv/bin/python scripts/run_input_component_frozen_matrix.py \
  --track-r-dir research_runs/input_components_h134_scratch \
  --output-dir research_runs/input_components_h134_frozen \
  --expected-count 288 --execute
```

其中 `--max-samples` 和训练入口的 `--max-eval-samples` 只允许 smoke，正式结果必须保持默认值 0。
汇总命令会拒绝缺条件、重复条件或 frozen 条件混用不同 checkpoint：

```bash
.venv/bin/python scripts/summarize_input_component_ablation.py \
  /path/to/results --output /path/to/result_summary.csv
```

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

- 报告三 seed mean、sample std 和每 seed 明细。
- 样本级 CI 使用连续预测起点的 moving-block bootstrap，block length 至少为 `pred_len`，避免把
  高度重叠窗口当独立样本。
- setting 宏平均再对32个 setting 做分层 bootstrap；同时报告7个 Golden setting 家族与 Exchange
  的结果，不能只报告总体平均。
- 多个假设/模型/指标的确认性检验使用 Holm correction；效应量与原始 CI 同时保留。

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
| Strong | M0 等效不敏感；M1/M2 都满足依赖、剂量和 Interaction；probe 有增益；sham 不解释结果 |
| Partial | M0 等效不敏感；仅 M1 或 M2 满足依赖，或只有部分数据集家族复现 |
| Model-shared | M0、M1、M2 都明显退化；A 有用，但没有证据证明 PhaseFormer 特别未利用 |
| OOD/confounded | `sham` 与 `minus_A` 同等或更差，或质量检查显示大量分布破坏 |
| Rejected/null | M0 不满足等效性且没有稳定 Interaction，或三模型都无稳定删除效应 |

“Strong”不要求增强模型的 full 指标一定超过论文 Golden；本实验研究输入利用机制。任何性能提升
声明仍必须另外对固定 Golden 报告。

## 9. PhaseFormer residual probe

为了直接检验 A 是否包含 PhaseFormer 尚未解释的信息，对 M0/full 的预测残差
`e=y-y_hat_M0` 建立简单线性 ridge probe：

- H1/H3 每通道将对应 A 的720步历史映射到该 horizon 的残差；H4 使用该通道30个周期的
  `delta[k]`、相邻差分及相对最新位移作为 probe 输入；均不做 cross-channel 输入。
- 在 validation origins 上按时间连续五折 cross-fitting，ridge 系数只在折内训练；lambda 网格在
  validation 内部冻结。
- 正式 test 前，用全部 validation 拟合一次 probe；test 只评估一次。
- 比较 `zero correction`、`sham-A probe` 与 `true-A probe` 的 MSE/MAE。

只有 true-A probe 在 held-out 数据稳定优于 zero 和 sham，才能把 A 称为“PhaseFormer 残差中仍
可预测的信息”。probe 是诊断，不计入三个主模型的正式预测排名。

## 10. RCRF 机制拆解

M2 的 A 干预同时可能改变 PhaseFormer 分支、NLinear 分支、可靠度 `r` 和融合权重 `alpha`。正式
实现需导出每个 sample×channel 的：

```text
y_phase, y_nlinear, r, alpha, y_fused
```

除真实输出外，离线重组以下反事实预测：

```text
F(P_variant, N_variant, alpha_full)     # 固定 gate，只看两个分支变化
F(P_full, N_full, alpha_variant)        # 固定分支，只看 gate 变化
F(P_variant, N_full, alpha_full)        # PhaseFormer 路径贡献
F(P_full, N_variant, alpha_full)        # NLinear 路径贡献
```

并校验 `F(P,N,alpha)` 与模型真实 fused 输出最大误差 `<2e-5`。若总体删除效应主要来自
`alpha_variant` 的变化，而 NLinear 分支本身没有利用 A，不得声称“NLinear 提取了 A”。

## 11. 样本级分析

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

`results.csv` 至少包含：`setting,dataset,horizon,seed,hypothesis,input_variant,model,track,mse,mae,
delta_mse,delta_mae,interaction_mse,interaction_mae,component_energy,qc_status,selection_source`。

## 13. 空结果表

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

| H | Scope | M0 equivalence | M1 Interaction MSE/MAE | M2 Interaction MSE/MAE | Holm-adjusted result | Evidence grade |
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
