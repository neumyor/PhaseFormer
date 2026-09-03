# PhaseFormer 输入盲区候选发现实验：ETTm1-H192

> 状态：方案待确认，尚未实现、未训练、未读取新的 test。实验范围固定为 ETTm1、
> `horizon=192`、`seed=2021`。本实验用于候选发现，不回写或修饰既有 H1/H3/H4 D0 结论。

## 1. 目标与结论边界

目标是寻找满足以下模式的输入成分 A：

1. A 对未来确实有预测信息；
2. 原版 PhaseFormer 对 A 的成分专属依赖接近零；
3. PhaseFormer+NLinear 或 PhaseFormer+NLinear+RCRF 对 A 有稳定正依赖；
4. 该差异不能由输入分布漂移、sham 扰动或 RCRF gate 自身变化解释。

本轮只使用 train/validation。ETTm1-H192 test 已在旧 D0 中暴露，因此即使后续再次读取，也只能
标记为 test-set-selected 复核，不能作为新的无偏确认。真正的确认范围优先使用尚未读取的
ETTm1-H336，或其他未暴露的 dataset×horizon。

## 2. 为什么先做发现而不直接重训

旧实验对每个候选直接做大幅删除并从头训练，成本高，而且 H1/H3/H4 的 sham 经常与
`minus_A` 同样有害，说明一般扰动效应压过了成分效应。本轮先用冻结 checkpoint、局部低剂量干预、
残差 probe 和分支反事实筛选；只有通过这些门槛的至多两个候选才进入重训。

原版 PhaseFormer 会把每个 phase slot 的 30 个历史周期值投影到低维 latent，因此并未天然丢弃
H1/H3。更可能的盲区是：被 phase folding 和低维压缩弱化、但能被 NLinear 的完整时间轴映射使用的
方向。ETTm1 为 15 分钟数据，`period_len=24` 只覆盖 6 小时；真实 96-step 日周期及更长结构是优先
检查对象。

## 3. 候选成分库

所有滤波器、模板和阈值只在 train 拟合，然后固定应用于 validation。

| ID | 候选 A | 提取定义（概要） | 为什么可能形成模型差异 |
|---|---|---|---|
| C1 | 96-step 日周期增量 | train-fitted 96-step 模板减去其在 24-step 模板空间中的投影 | PhaseFormer 显式按24步折叠；NLinear保留完整时间轴 |
| C2 | 672-step 周内低频结构 | 因果低通/模板得到约一周尺度成分，再去除 C1 与常数项 | 720步窗口覆盖约一周，30→latent 的压缩可能损失慢序列 |
| C3 | 非24整齐频带 | 保留周期约32–80步的带通成分，并正交于24/96模板 | 非整齐频率在24步折叠后容易产生拍频和相位混叠 |
| C4 | 周期边界连续性 | 提取每24步边界附近的连续时间增量与短局部形状 | PhaseFormer先按phase重排，NLinear直接观察相邻时间点 |
| C5 | 周期间幅度包络 | 估计每24步块的稳健幅度，相对train模板构成加性调制项 | 跨周期幅度序列会先被压到低维；NLinear可直接拟合 |
| C6 | 平滑相位速度 | 估计24步相位位移的连续变化率，只施加小幅、连续几何干预 | RCRF可靠性可能在相位不稳定时增加NLinear权重 |

C1–C6 是候选库，不预设其中一定存在正确 A。若全部未通过，结论是“该候选库未找到”，而不是选择
数值最大的候选强行进入正式结论。

## 4. 干预构造与分布约束

### 4.1 连续、因果、可复现

- 先在连续序列上构造 train-fitted 成分，再切成720步窗口；禁止每个重叠窗口重新拟合模板。
- 每个 validation origin 只能使用该 origin 及之前的信息；不得读取 target 或 test。
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

### S1a：512 origins 快速冻结筛选

在 validation 中按时间均匀取512个 origins。三个 full checkpoint 对6个候选的4种低剂量干预做
冻结前向：`3×6×4=72` 次条件评估，另加3次 full，共75次受限 validation 读，不重训。

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
4. **RCRF 四类反事实**：phase-only、NLinear-only、gate-only、branches-only，判断变化来自哪个分支。
5. **PF 残差 probe**：用 A 的历史系数/能量/近期变化预测 `y-yhat_M0`。在 validation origins 上做
   时间连续五折 cross-fitting ridge；报告相对常数 probe 的 MAE 改善及 moving-block CI。
6. **增强收益关联**：检验 A 强度是否与 `error_M0-error_M1/M2` 正相关，使用时间 block bootstrap。

按以下顺序筛选：QC硬门 → PF残差可预测性 → sham校正模型差异 → 分支反事实。只保留最多3个候选
进入 S1b。

### S1b：全 validation 复核

对 S1a 入围的最多3个候选运行完整 validation origins，重复相同统计。入围 S2 至多2个候选，且每个
必须同时满足：

- PF 残差 probe 的 MAE 改善 `>=1%`，95% block CI 下界 `>0`；
- M0 在 `lambda=0.5` 的 `Specific` MSE 与 MAE CI 完全落在 `±0.5%` 内；
- M1 或 M2 的 `Specific` MSE、MAE均 `>=1%` 且CI下界 `>0`；
- 相对 M0 的 sham-adjusted Interaction `>=+0.5pp` 且CI下界 `>0`；
- `lambda=0.25→0.5` 方向一致，不能只在单一剂量跳变；
- NLinear-only 分支敏感性至少为 phase-only 的 `1.5×`，且 gate-only 变化不能解释超过一半总效应；
- 结果不能由单个通道或少于25个 origins 主导。

### S2：只对入围候选重训

每个入围候选为三个模型训练 `half_A/minus_A/sham`；full checkpoint 复用。最多新增：

```text
2 candidates × 3 models × 3 non-full conditions = 18 training runs
```

S2 仍只读 validation。若 frozen 与 retrain 方向相反，分别报告为“已有模型依赖”和“重训可补偿”，
不得平均成一个结论。

## 6. 预算与停止规则

- full 模型训练：最多3 runs；若锚点可合法复用则为0。
- 快速筛选：75个512-origin validation 条件读。
- 全 validation：最多3候选，约39个条件读（含共享 full）。
- 候选重训：最多18 runs。
- 总训练上限：从零开始最多21 runs，远低于旧 D0 的210 runs。

出现以下任一情况立即停止扩展：

- 所有候选都不能通过干预 QC；
- 没有候选能显著预测 M0 的未来残差；
- sham-adjusted Interaction 没有候选为正；
- 候选效应完全由 RCRF gate 改变而非 NLinear branch 使用解释。

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

只有同时满足“PF残差可预测”“M0 sham校正等效”“增强模型sham校正依赖”“NLinear分支反事实支持”
的候选，才命名为“PhaseFormer未充分利用、增强模型正在利用的候选成分”。本轮只产生 validation
候选，不产生最终泛化声明。入围候选须在未暴露范围重新预注册并确认；若使用 ETTm1-H192 test，必须
显著披露其已受旧 D0 test exposure 影响。
