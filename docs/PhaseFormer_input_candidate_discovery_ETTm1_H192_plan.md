# PhaseFormer 输入盲区候选发现实验：ETTm1-H192

> 状态：S0、S1a、S1b 已完成并按早停规则结束；未读取本方案 test、未启动 S2 重训。
> 实验范围固定为 ETTm1、`horizon=192`、`seed=2021`。本实验不回写或修饰既有 H1/H3/H4 D0 结论。

## 1. 目标与结论边界

目标是寻找满足以下模式的输入成分 A：

1. A 对未来确实有预测信息；
2. 原版 PhaseFormer 对 A 的成分专属依赖接近零；
3. PhaseFormer+NLinear 或 PhaseFormer+NLinear+RCRF 对 A 有稳定正依赖；
4. 该差异不能由输入分布漂移、sham 扰动或 RCRF gate 自身变化解释。

本轮使用 validation 发现候选，候选定义、提取器、剂量、方向性预期、判定阈值和最多两个入围项
全部冻结后，才允许一次性读取 test 做确认。ETTm1-H192 test 已在旧 D0 中暴露，所以该步骤可称为
“独立于本轮 validation 选择的 test 确认”，但不能称为从未暴露的盲测或无偏泛化估计。更强的无偏
确认仍需使用尚未读取的 ETTm1-H336，或其他未暴露的 dataset×horizon。

## 2. 为什么先做发现而不直接重训

旧实验对每个候选直接做大幅删除并从头训练，成本高，而且 H1/H3/H4 的 sham 经常与
`minus_A` 同样有害，说明一般扰动效应压过了成分效应。本轮先用冻结 checkpoint、局部低剂量干预、
残差 probe 和分支反事实筛选；只有通过这些门槛的至多两个候选才进入重训。

原版 PhaseFormer 会把每个 phase slot 的 30 个历史周期值投影到低维 latent，因此并未天然丢弃
H1/H3。更可能的盲区是：被 phase folding 和低维压缩弱化、但能被 NLinear 的完整时间轴映射使用的
方向。ETTm1 为 15 分钟数据，`period_len=24` 只覆盖 6 小时；真实 96-step 日周期及更长结构是优先
检查对象。

## 3. 候选成分库

所有滤波器、模板和阈值只在 train 拟合，然后固定应用于 validation/test；test 在候选冻结前不可读。

| ID | 候选 A | 提取定义（概要） | 为什么可能形成模型差异 |
|---|---|---|---|
| C1 | 96-step 日周期增量 | train-fitted 96-step 模板减去其在 24-step 模板空间中的投影 | PhaseFormer 显式按24步折叠；NLinear保留完整时间轴 |
| C2 | 672-step 周内低频结构 | 因果低通/模板得到约一周尺度成分，再去除 C1 与常数项 | 720步窗口覆盖约一周，30→latent 的压缩可能损失慢序列 |
| C3 | 非24整齐频带 | 保留周期约32–80步的带通成分，并正交于24/96模板 | 非整齐频率在24步折叠后容易产生拍频和相位混叠 |
| C4 | 周期边界连续性 | 提取每24步边界附近的连续时间增量与短局部形状 | PhaseFormer先按phase重排，NLinear直接观察相邻时间点 |
| C5 | 周期间幅度包络 | 估计每24步块的稳健幅度，相对train模板构成加性调制项 | 跨周期幅度序列会先被压到低维；NLinear可直接拟合 |
| C6 | 平滑相位速度 | 估计24步相位位移的连续变化率，只施加小幅、连续几何干预 | RCRF可靠性可能在相位不稳定时增加NLinear权重 |
| C7 | 输入尾部近期创新 | 用仅在train拟合的因果预测器估计最后24步可由更早历史解释的轨迹，取实际尾部相对该轨迹的平滑残差 | NLinear显式以最后值为锚并直接映射完整时间轴，可能比按phase折叠的主干更利用近期偏离 |

C1–C7 是候选库，不预设其中一定存在正确 A。若全部未通过，结论是“该候选库未找到”，而不是选择
数值最大的候选强行进入正式结论。

### 3.1 C7 的预注册提取与近程指标

C7 测的不是“最后若干点是否重要”，而是最后一段中新出现、无法由较早历史充分预测的近期创新：

1. 固定尾部长为24步（ETTm1上为6小时，与当前 `period_len=24` 一致），不根据 validation/test
   结果搜索长度；
2. 在 train 上拟合因果一步预测器 `g`。对连续序列中的每个时点，`g` 只能读取该点之前的历史、
   同相位历史和已知日历标记，不能读取当前值或任何未来 target；由此先得到连续、可重用的创新
   序列 `e_t = x_t-g(x_{<t}, marks)`；
3. 切窗后只取输入最后24步的 `e_t`，并在该区间起点使用固定余弦 ramp 从0过渡到1，得到
   `A_recent`；令 `B=X-A_recent`，必须满足重建门。不同窗口对同一 `e_t` 的估计必须一致，只有
   窗口相对位置决定的尾部支撑 mask 可以不同；
4. `A_sham` 从 train 中按变量、时刻/星期、前缀水平和波动分层匹配的连续24步残差块抽取，保留
   跨变量同步关系与块内顺序；禁止逐点打乱。三个模型使用完全相同的匹配块与随机 seed；
5. 主终点预注册为预测步 `1–24` 的 MSE/MAE；`25–48` 为关键次终点，`49–96` 与 `97–192`
   用于检查效应是否随预测距离衰减。全192步聚合只作辅助，不能因其稀释近程效应而否定 C7。

这个定义与当前 NLinear 实现直接对应：`WeakPeriodResidualHead` 使用最后输入值作为 persistence anchor，
并将相对最后值中心化后的完整历史线性映射到预测窗口。因此，若增强模型的近程收益确实来自近期
依赖，C7 应优先影响 NLinear 分支以及预测窗口前24步，而不是只改变 RCRF gate。

## 4. 干预构造与分布约束

### 4.1 连续、因果、可复现

- 先在连续序列上构造 train-fitted 成分，再切成720步窗口；禁止每个重叠窗口重新拟合模板。
- C7 的连续基础成分是全序列创新 `e_t`；“只保留最后24步”的支撑 mask 是预注册的窗口相对操作，
  不属于重新拟合。
- 每个 origin 的成分提取只能使用该 origin 及之前的信息；不得读取对应 target。发现阶段不可读取 test。
- 加性候选必须满足 `X=B+A`；几何候选必须提供可逆的变换参数和重建误差。
- 随机 control 由固定 seed 派生，三个模型逐元素共享同一输入。

### 4.2 低剂量筛选输入

快速筛选不立即做100%删除。每个候选先生成：

```text
full
remove_025 = X - 0.25 A
remove_050 = X - 0.50 A
sham_025   = X - 0.25 A_sham
sham_050   = X - 0.50 A_sham
```

`A_sham` 使用连续 block shift、同 phase 条件重采样或 Fourier phase randomization 构造，按候选选择
最小破坏方式。它必须保留 A 的通道 RMS、差分尺度、功率谱和自相关，而只破坏 A 与预测目标之间的
正确时间对应。不得再使用整段时间反转或逐窗口任意周期置换作为默认 sham。

### 4.3 干预 QC 门

候选进入模型评分前必须全部通过：

- 加性重建最大误差 `<=1e-6`；几何重建最大误差 `<=2e-5`；
- 同一时间点在重叠窗口中的基础成分值一致，最大误差 `<=1e-6`；
- real/sham 成分 RMS 比在 `[0.90,1.10]`；
- 一阶差分 RMS 比在 `[0.90,1.10]`；
- 归一化功率谱余弦相似度 `>=0.95`；
- lag 1/24/96 自相关绝对差均 `<=0.05`；
- 24步边界 jump 的95分位比在 `[0.8,1.25]`；
- 所有输入有限，且 target/time split 未被修改。

QC 失败的候选只能修订构造后重新登记，不能进入排名。

## 5. 执行流程

### S0：三个 full 模型锚点

在完全一致的 ETTm1-H192-seed2021 设置下准备：

- M0：`original`；
- M1：`weak_residual`；
- M2：`rcrf_nlinear_plain`。

可复用同 commit、同配置且 validation-only 选择的 full checkpoint；否则各训练一次。NLinear-only
不作为第四个正式模型，优先直接读取 M2 已暴露的 NLinear branch，节省训练成本。

### S1a：validation 上的512 origins快速冻结筛选

在 validation 中按时间均匀取512个 origins。三个 full checkpoint 对7个候选的4种低剂量干预做
冻结前向：`3×7×4=84` 次条件评估，另加3次 full，共87次受限 validation 读，不重训。

逐候选计算：

1. **成分专属误差效应**

   ```text
   Specific_m(A,lambda) = Delta_remove_m - Delta_sham_m
   ```

2. **模型差异**

   ```text
   Interaction_m = Specific_m - Specific_M0
   ```

3. **预测敏感性**：干预前后预测本身的 MSE/MAE 距离，区分“模型没反应”和“反应但方向错误”。
4. **NLinear-only 损失反事实（实际利用的必要证据）**：保存 full 输入下的 PhaseFormer 分支输出和
   fusion gate，只把 NLinear 分支输出替换为干预输入下的输出，再按原融合公式重组预测。若 A 被
   NLinear 实际用于正确预测，该反事实相对 all-full 的 MSE/MAE 必须上升；仅观察到 NLinear 输出
   数值变化不算证据。phase-only、gate-only、branches-only 三类反事实继续保留为归因诊断，但不
   替代这个必要条件。
5. **PF 残差 probe**：用 A 的历史系数/能量/近期变化预测 `y-yhat_M0`。在 validation origins 上做
   时间连续五折 cross-fitting ridge；报告相对常数 probe 的 MAE 改善及 moving-block CI。C7 的
   probe 主终点只使用预测步1–24，并另外报告四个预注册预测区间。
6. **增强收益关联**：检验 A 强度是否与 `error_M0-error_M1/M2` 正相关，使用时间 block bootstrap。

按以下顺序筛选：QC硬门 → PF残差可预测性 → sham校正模型差异 → 分支反事实。只保留最多3个候选
进入 S1b。

### S1b：全 validation 候选冻结

对 S1a 入围的最多3个候选运行完整 validation origins，重复相同统计。所有候选选择、提取参数、
主/次终点、效应方向和阈值均在此阶段冻结；之后不得根据 test 改候选。入围 S2 至多2个候选，且
每个必须同时满足：

- PF 残差 probe 的 MAE 改善 `>=1%`，95% block CI 下界 `>0`；
- M0 在 `lambda=0.5` 的 `Specific` MSE 与 MAE CI 完全落在 `±0.5%` 内；
- M1 或 M2 的 `Specific` MSE、MAE均 `>=1%` 且CI下界 `>0`；
- 相对 M0 的 sham-adjusted Interaction `>=+0.5pp` 且CI下界 `>0`；
- `lambda=0.25→0.5` 方向一致，不能只在单一剂量跳变；
- NLinear-only 反事实的 MSE、MAE均恶化且95% block CI下界 `>0`；同时其损失效应至少为
  phase-only 的 `1.5×`，且 gate-only 变化不能解释超过一半总效应；
- 结果不能由单个通道或少于25个 origins 主导。

C7 使用同一门槛，但主判定仅针对预注册的预测步1–24；并要求效应在 `1–24` 最大，之后总体不增强。
若只在全192步或远期区间显著，则不能解释为“近程依赖”。

### S2：只对入围候选重训并一次性 test 确认

每个入围候选为三个模型训练 `half_A/minus_A/sham`；full checkpoint 复用。最多新增：

```text
2 candidates × 3 models × 3 non-full conditions = 18 training runs
```

训练仅使用 train，早停和 checkpoint 选择仅使用 validation。所有 checkpoint 与分析代码锁定后，
对 full/half_A/minus_A/sham 一次性读取 test，并按 S1b 预注册方向和阈值判定；看到 test 后不得修改
提取器、候选、剂量或选择 checkpoint 再重跑。若 frozen 与 retrain 方向相反，分别报告为“已有
模型依赖”和“重训可补偿”，不得平均成一个结论。

test 确认同时报告全192步与 `1–24/25–48/49–96/97–192` 四个固定区间。C1–C6 的主终点仍是
全192步；C7 的主终点是1–24步。由于旧 D0 已暴露 ETTm1-H192 test，报告必须显著标注
`test-set-exposed confirmation`，不能使用“盲测”措辞。若两个候选同时进入 test，主判定采用
Holm 校正控制 family-wise error rate；未经校正的区间只能作为描述性结果。

## 6. 预算与停止规则

- full 模型训练：最多3 runs；若锚点可合法复用则为0。
- 快速筛选：87个512-origin validation 条件读。
- 全 validation：最多3候选，约39个条件读（含共享 full）。
- 候选重训：最多18 runs。
- 总训练上限：从零开始最多21 runs，远低于旧 D0 的210 runs。

出现以下任一情况立即停止扩展：

- 所有候选都不能通过干预 QC；
- 没有候选能显著预测 M0 的未来残差；
- sham-adjusted Interaction 没有候选为正；
- NLinear-only 损失反事实不恶化，说明只能证明“分支响应”而不能证明“分支有效利用”。

## 7. 结果与审计产物

中间 checkpoint、逐条件日志和受限诊断放在：

```text
research_runs/input_candidate_discovery_ettm1_h192_v1_scratch/
research_runs/input_candidate_discovery_ettm1_h192_v1_control/
```

正式候选发现交付目录为：

```text
research_runs/input_candidate_discovery_ettm1_h192_v1/
```

最终按项目实验审计规范只保留 `run.yaml`、`results.csv`、`sample_errors.csv`、
`selected_cases.npz`、`objective_error_analysis.md`、`objective_error_analysis.zip` 和被报告引用的
`figures/`。所有实验日志及监控记录保存在 `research_runs/`，不得写入 `/tmp`。

## 8. 最终判定

只有同时满足“PF残差可预测”“M0 sham校正等效”“增强模型sham校正依赖”“只改变NLinear分支时
损失显著上升”，且 validation 发现与冻结后的 test 确认方向一致，才命名为“PhaseFormer未充分
利用、增强模型正在利用的候选成分”。C7 还必须呈现预注册的近程集中效应。本轮结论只限
ETTm1-H192、seed2021，并显著披露旧 D0 已造成 test exposure；更强泛化声明仍需未暴露范围复现。

## 9. S1 执行记录与早停（2026-09-03）

- S0：在 `raft` 环境 RTX 4090 上完成 original、weak_residual、rcrf_nlinear_plain 三个 ETTm1-H192
  full-input 锚点（30 epoch、seed2021）。
- S1a：7候选 × 4低剂量干预 × 3模型，在 validation 时间均匀的512 origins完成冻结读取；C2、C3、C7
  进入 S1b。S1b 对这三个候选完成全 validation（11,329 origins）复核，test 未读取。
- C2 的 remove-vs-sham MAE 均为显著负值（original `-7.92%`、weak `-8.64%`、RCRF `-7.94%`，移动块
  95% CI 均不跨0），表明 sham 比删除更有害，是 control/intervention mismatch，不能作为正确 A。
- C3 在 `lambda=0.5` 的 sham-adjusted MAE 为 original `+0.69%`、weak `+0.93%`、RCRF `+0.24%`；三个
  成对移动块95% CI均跨0，弱残差相对 original 的优势远低于预注册的 `+0.5pp` interaction 与 `>=1%`
  增强模型效应门。RCRF NLinear-only 重组虽有数值变化，但不能补足总效应和交互门。
- C7 在预测步1–24有可测响应，但不符合所需方向：`lambda=0.5` 时 original 的 remove/sham MAE 变化为
  `+0.822/+0.496pp`，weak 为 `+0.999/+1.073pp`，RCRF 为 `+0.316/+1.261pp`；原版并非等效不敏感，增强
  模型也未显示更大的 sham-adjusted 依赖。
- 因此没有候选通过 S1 的必要门，按 §6 停止扩展：不进行 S2 的 candidate retraining，也不读取已暴露的
  ETTm1-H192 test。正式审计产物位于 `research_runs/input_candidate_discovery_ettm1_h192_v1/`；它是
  validation-only negative candidate-discovery result，而非泛化效果结论。
