# PhaseFormer 残差通路拓扑实验 — 结果反馈

> 状态：Stage A（验证集筛选）与 Stage B（全预算确认）已完成；A1/A2 输出逐层变体追加轮已完成（见 §3.2、§4 H6），输出×深度设计空间补齐，见下。
> 计划锚点：`docs/PhaseFormer_residual_topology_plan.md`；本文件是结果、失败、协议偏差与最终决策的回填。
>
> 分支：`weak-residual-phaseformer`
> 实验日期：2026-08-24（首轮）、2026-08-25（A1/A2 追加轮）
> 代码提交：`50a9b21`（计划）、`6f5b607`（实现）；追加轮代码提交见 git log（`a635948` 起的 A1/A2 接线 + 测试）。

## 0. 摘要

- **Stage A**（30% 数据、≤8 epoch、验证集、4 settings × 6 modes = 24 run）：R3/R4 在 ETTh2-h720（1 层）**数值等价**（val_mae=0.66184554、val_mse=0.82789717 完全相等），实现正确性验证通过。按 score（0.5×ΔMAE% + 0.5×ΔMSE%）前两名 = R1 `residual_output_convex`（15.55）与 R2 `residual_output_additive`（13.59）；4/4 settings 双指标无回退，满足冻结条件。
- **Stage B**（100% 数据、≤30 epoch、best checkpoint、测试集、3 ETT + Electricity = 12 run）：残差输出融合的收益**跨 setting 不一致**——
  - **ETTh2-h720（1 层）强正**：R1 ΔMSE −7.7% / ΔMAE −5.8%；R2 ΔMSE −7.6% / ΔMAE −5.7%。
  - **Electricity-h336（2 层）微正**：R1 ΔMSE −1.6% / ΔMAE −0.8%；R2 ΔMSE −1.3% / ΔMAE −0.4%。
  - **ETTh1-h336（3 层）中性/微负**：R1 ΔMSE +0.06% / ΔMAE +0.75%；R2 ΔMSE +1.0% / ΔMAE +1.1%。（§0 各 bullet 用候选减基线、负=改善；R1 在 ETTh1 两指标均微负 ≈ 中性）
  - **ETTm1-h720（2 层）中性/微负**：R1 ΔMSE +0.83% / ΔMAE +0.19%；R2 ΔMSE −0.01% / ΔMAE +0.61%。
- **主要结论**：残差拓扑问题在当前证据下收敛为——**输出层凸融合（R1，即现有设计）已是正确插入点**；加法修正（R2）、潜空间长跳连（R3）、逐层注入（R4）、混合（R5）均不优于 R1。收益与先前动态相位阶段一致地依赖数据集（ETTh2/Electricity 正、ETTh1/ETTm1 中性），**跨 setting 方向不一致，无冠军拓扑**；单 seed 结论，不更新 `_LATEST_POLICY`。
- **A1/A2 追加轮（输出×深度设计空间补齐）**：A1 = 逐层凸融合（R1 在每层施加）、A2 = 逐层加法修正（R2 在每层施加）。1 层 setting（ETTh2-h720）A1≡R1、A2≡R2 **数值完全相等**（实现退化解验证通过）。**测试集上层级融合不转移**：全部多层 setting 上 A1 ≤ R1、A2 ≤ R2（唯一例外 A2 在 Electricity +0.59/+1.83 优于 R2 +0.41/+1.32）；测试集平均 score R1 1.75 > R2 1.53 > A1 1.38 > A2 1.31。**与 Stage A 验证集信号相反**（验证集 A1 ≥ R1 在所有 setting）——再次确认验证集筛选不是测试集保证。**逐层级联设计判为不采纳**。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 模型 | PhaseFormer（weak-residual 分支），period 24，lookback 720 |
| 模式 | R0 original；R1 residual_output_convex；R2 residual_output_additive；R3 residual_latent_long；R4 residual_latent_layerwise；R5 residual_hybrid；**A1 residual_output_layerwise_convex；A2 residual_output_layerwise_additive** |
| Stage A | 30% 训练数据，≤8 epoch，仅验证集（`--evaluate-test` 关闭），seed 2021，huber |
| Stage B | 100% 数据，≤30 epoch，val early stop + best checkpoint，测试集，seed 2021 |
| 设置 | ETTh1-h336（3 层）、ETTh2-h720（1 层）、ETTm1-h720（2 层）、Electricity-h336（2 层） |
| Stage A 命令 | `scripts/search_phaseformer.py --stage mechanism_screen_1`（首轮 24 run + A1/A2 追加 8 run = 32 run） |
| Stage B 命令 | `scripts/benchmark_phaseformer_suite.py`（首轮 12 run + A1/A2 追加 8 run = 20 run，`--resume` 跳过已完成的 original） |
| 数据落盘 | `research_runs/residual_topology_screen_runs/`、`research_runs/residual_topology_full_runs/` |
| 单测 | `tests/` 全量 99/99 通过 |

**协议偏差**：
- 计划 Stage B 原定"先确认 ETTh1/ETTh2/ETTm1；Electricity 仅在候选通过前三项且仍有正向信号时运行"。前三项结果方向不一致（1 强正 + 2 中性/微负），门槛处于临界；因先前动态相位 full-budget 已观测 Electricity 残差稳定正收益，且该 setting 恰为检验"跨规模是否成立"而设，**补跑了 Electricity-h336**（额外 ~1 GPU·h），使结论覆盖全部 4 个计划 setting。此为判断性补充，不改变前三个 setting 的结论。
- ETTh2-h720 在 Stage A 的 R3/R4 参数仅 734（1 层小模型），与计划一致。

## 2. Stage A 结果（验证集，Δ% 正 = 改善）

| Setting | R1 convex | **A1 layerwise_convex** | R2 additive | **A2 layerwise_additive** | R5 hybrid | R3 latent_long | R4 latent_layerwise |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1-h336 | +12.81/+18.44 (15.63) | **+12.99/+18.44 (15.72)** | +9.39/+12.97 (11.18) | **+9.52/+13.02 (11.27)** | +9.24/+12.26 (10.75) | +0.52/+0.82 (0.67) | +0.53/+0.86 (0.70) |
| ETTh2-h720 | +15.14/+23.33 (19.23) | **+15.14/+23.33 (19.23) ≡R1** | +13.08/+20.36 (16.72) | **+13.08/+20.36 (16.72) ≡R2** | +13.10/+20.31 (16.70) | +0.39/+0.74 (0.57) | +0.39/+0.74 (0.57) |
| ETTm1-h720 | +13.91/+18.43 (16.17) | **+13.98/+18.91 (16.44)** | +13.15/+17.58 (15.37) | **+12.43/+17.07 (14.75)** | +13.34/+17.83 (15.58) | +2.01/+2.81 (2.41) | +2.01/+2.87 (2.44) |
| Electricity-h336 | +7.30/+15.05 (11.17) | **+7.61/+15.35 (11.48)** | +7.27/+14.92 (11.09) | **+7.06/+14.79 (10.93)** | +7.07/+14.67 (10.87) | +2.23/+5.00 (3.62) | +2.15/+4.57 (3.36) |

单元格 = ΔMAE%/ΔMSE%（score）。score = 0.5×ΔMAE + 0.5×ΔMSE。A1/A2 为追加轮（ETTh2-h720 是 1 层 → 无中间层 → A1≡R1、A2≡R2，实现退化解验证通过）。

**R3 ≡ R4 等价验证**（ETTh2-h720，1 层）：两者 val_mae=0.66184554、val_mse=0.82789717、params=734，**完全相等** → 通过。

**冻结决策（首轮）**：平均 score R1=15.55、R2=13.59、R5=13.48、R3=1.82、R4=1.77。全部候选 4/4 settings 双指标改善、无 >0.5% 回退。保留 **R0 + R1 + R2** 进入 Stage B。

**冻结决策（追加轮 A1/A2）**：平均 score A1=15.72（≥R1 15.55 全 setting）、A2=13.42（<R2 13.59）。按计划规则 top-2 = A1 + R1；因用户明确要求比较 R1/R2 的逐层变体，**追加轮同时将 A1 + A2 送入 Stage B**（偏离严格 top-2，目的为给出四种输出融合的完整 test 集对照；偏差已记录于 `stage_a_selection_notes.md`）。R1/R2 的 Stage B 结果沿用首轮。

## 3. Stage B 结果（测试集，Δ% 正 = 改善，matched rerun）

### 3.1 R1/R2（首轮）

| Setting | R1 convex ΔMAE/ΔMSE | R2 additive ΔMAE/ΔMSE | R1 epochs | R2 epochs |
|---|---:|---:|---:|---:|
| ETTh2-h720 | **+5.75 / +7.66** | **+5.69 / +7.56** | 17 | 17 |
| Electricity-h336 | +0.81 / +1.57 | +0.41 / +1.32 | 22 | 22 |
| ETTh1-h336 | −0.75 / −0.06 | −1.07 / −1.04 | 17 | 28 |
| ETTm1-h720 | −0.19 / −0.83 | −0.61 / +0.01 | 12 | 18 |

绝对测试指标（MAE / MSE）：

| Setting | original | R1 convex | R2 additive |
|---|---:|---:|---:|
| ETTh2-h720 | 0.4552 / 0.4254 | 0.4290 / 0.3928 | 0.4293 / 0.3932 |
| Electricity-h336 | 0.2591 / 0.1661 | 0.2570 / 0.1635 | 0.2580 / 0.1639 |
| ETTh1-h336 | 0.4314 / 0.4381 | 0.4347 / 0.4384 | 0.4361 / 0.4427 |
| ETTm1-h720 | 0.4127 / 0.4157 | 0.4135 / 0.4191 | 0.4152 / 0.4156 |

金标准 `docs/PhaseFormer_gold_standard.md` 为固定参照；本表为 matched rerun（lookback 720、period 24、seed 2021、huber），与金标准同协议可比但不替代。

### 3.2 A1/A2 追加轮（输出×深度 四形态完整对照，Δ% 正 = 改善）

| Setting | R1 convex | A1 layerwise_convex | R2 additive | A2 layerwise_additive |
|---|---:|---:|---:|---:|
| ETTh1-h336（3 层） | −0.75 / −0.06 (17ep) | −1.14 / +0.13 (15ep) | −1.07 / −1.04 (28ep) | −1.66 / −0.95 (16ep) |
| ETTh2-h720（1 层） | +5.75 / +7.66 (17ep) | **+5.75 / +7.66 (17ep) ≡R1** | +5.69 / +7.56 (17ep) | **+5.69 / +7.56 (17ep) ≡R2** |
| ETTm1-h720（2 层） | −0.19 / −0.83 (12ep) | −0.47 / −1.37 (10ep) | −0.61 / +0.01 (18ep) | −0.67 / −1.92 (10ep) |
| Electricity-h336（2 层） | +0.81 / +1.57 (22ep) | −0.27 / +0.71 (14ep) | +0.41 / +1.32 (22ep) | **+0.59 / +1.83 (30ep)** |
| **平均 score** | **1.75** | 1.38 | 1.53 | 1.31 |

绝对测试指标（MAE / MSE）：

| Setting | original | R1 | A1 | R2 | A2 |
|---|---:|---:|---:|---:|---:|
| ETTh1-h336 | 0.4314 / 0.4381 | 0.4347 / 0.4384 | 0.4363 / 0.4376 | 0.4361 / 0.4427 | 0.4386 / 0.4423 |
| ETTh2-h720 | 0.4552 / 0.4254 | 0.4290 / 0.3928 | 0.4290 / 0.3928 | 0.4293 / 0.3932 | 0.4293 / 0.3932 |
| ETTm1-h720 | 0.4127 / 0.4157 | 0.4135 / 0.4191 | 0.4146 / 0.4214 | 0.4152 / 0.4156 | 0.4154 / 0.4237 |
| Electricity-h336 | 0.2591 / 0.1661 | 0.2570 / 0.1635 | 0.2598 / 0.1649 | 0.2580 / 0.1639 | 0.2576 / 0.1631 |

新增参数（相对 original；Stage A 架构参数，与 Stage B 相同）：A1/A2 每个中间层 +`PhaseSlotResidualHead`（Linear(720→30)+gate）≈ 21.6K；ETTh1（2 中间层）+43.3K、ETTm1/Electricity（1 中间层）+21.6K、ETTh2（1 层）0。

**结论（测试集）**：
- **逐层级联不转移**：除 A2@Electricity（+0.59/+1.83 > R2 +0.41/+1.32）外，所有多层 setting 上层级变体 ≤ 单点变体；平均 score R1 1.75 > R2 1.53 > A1 1.38 > A2 1.31。
- **与 Stage A 验证集信号相反**：验证集 A1 ≥ R1 在所有 setting（15.72 vs 15.55），但测试集 A1 < R1 在全部 3 个多层 setting → **验证集筛选结果未转移**（与项目协议一致：验证集是筛网不是保证）。
- **1 层退化解通过**：ETTh2-h720 A1≡R1、A2≡R2 数值完全相等（无中间层 → 结构恒等），实现正确性验证通过。
- 凸融合逐层化（A1）在 Electricity 更早停（14ep vs R1 22ep）且 MAE 回退（−0.27）——早停选择了更差 checkpoint，进一步削弱逐层凸融合证据。

### 3.3 与金标准对比（`docs/PhaseFormer_gold_standard.md`，指示性）

按金标准规则：相对改善率 = (gold − new)/gold × 100（正 = 提升）；仅 MSE、MAE 均低于金标准才称"双指标提升"。**协议披露**：本实验为 matched rerun（lookback 720、period 24、seed 2021、huber、val early stop），与金标准论文协议（损失/早停/seed 数未披露）不一致——**本轮 4 个 setting 的 matched original 均差于金标准**（MSE −0.7%~−5.8%），协议偏移明显；注意该规律不适用于上一轮 10-setting（见下文"与上一轮的关系"：ETTh1-720 的 matched original 本身即优于金标准）。故相对金标准 Δ 混入协议差与机制收益，**仅作指示性参照，不构成相对论文的直接提升声明**（金标准规则 §4/§5）。

| Setting（金 MSE/MAE） | mode | test MSE/MAE | ΔMSE% | ΔMAE% | 判定 |
|---|---:|---:|---:|---:|---|
| ETTh1-336 (0.425/0.424) | original | 0.4381/0.4314 | −3.09 | −1.75 | 双指标退化 |
| | R1 | 0.4384/0.4347 | −3.15 | −2.51 | 双指标退化 |
| | R2 | 0.4427/0.4361 | −4.16 | −2.84 | 双指标退化 |
| | A1 | 0.4376/0.4363 | −2.96 | −2.91 | 双指标退化 |
| | A2 | 0.4423/0.4386 | −4.07 | −3.44 | 双指标退化 |
| ETTh2-720 (0.402/0.436) | original | 0.4254/0.4552 | −5.82 | −4.40 | 双指标退化 |
| | **R1 / A1**（1 层，≡） | 0.3928/0.4290 | **+2.29** | **+1.61** | **双指标提升** |
| | **R2 / A2**（1 层，≡） | 0.3932/0.4293 | **+2.18** | **+1.54** | **双指标提升** |
| ETTm1-720 (0.412/0.410) | original | 0.4157/0.4127 | −0.90 | −0.66 | 双指标退化 |
| | R1 | 0.4191/0.4135 | −1.74 | −0.85 | 双指标退化 |
| | R2 | 0.4156/0.4152 | −0.89 | −1.27 | 双指标退化 |
| | A1 | 0.4214/0.4146 | −2.28 | −1.13 | 双指标退化 |
| | A2 | 0.4237/0.4154 | −2.84 | −1.33 | 双指标退化 |
| Electricity-336 (0.165/0.257) | original | 0.1661/0.2591 | −0.67 | −0.81 | 双指标退化 |
| | R1 | 0.1635/0.2570 | **+0.91** | +0.00 | 双指标提升（MAE 临界≈舍入量级） |
| | R2 | 0.1639/0.2580 | +0.66 | −0.40 | 单指标提升（MSE） |
| | A1 | 0.1649/0.2598 | +0.05 | −1.08 | 单指标提升（MSE） |
| | A2 | 0.1631/0.2576 | **+1.18** | −0.22 | 单指标提升（MSE） |

解读：
- **仅 ETTh2-h720（1 层）R1/A1、R2/A2 双指标低于金标准**（+2.2~2.3% MSE、+1.5~1.6% MAE，幅度超舍入量级）——残差融合在此 setting 补上协议差（original −5.8%）并反超。
- **Electricity R1** 名义双指标提升，但 ΔMAE = +0.004% 处于金标准三位小数舍入量级 → 按金标准规则 §4 视为**单指标（MSE）提升**更稳妥；R2/A1/A2 亦仅单指标提升。
- ETTh1/ETTm1 无任何 mode 达到金标准（协议差吞掉残差收益）。金标准视角**不改变配对结论**：R1 仍最强、逐层级联不转移、跨 setting 无冠军、单 seed 不更新 `_LATEST_POLICY`。
- **与上一轮（动态相位 full-budget，`docs/PhaseFormer_dynamic_phase_report.md` §8）的关系**：上一轮的残差重构机制（residual_full / dyn_full）曾在 5 个 setting 双指标低于金标准——ETTh1-720（但 matched original 已 +3.0/+2.2 超金标准，属协议优势）、ETTh2-336、ETTh2-720、Electricity-336、Electricity-720（后两者 MAE 舍入量级）。本轮只重测了其中 ETTh2-720 与 Electricity-336，且**机制不同**（本轮 R1 = 输出凸融合头；上轮 residual_full = 残差重构 + 动态机制栈）：
  - **ETTh2-720 两轮一致超金标准**：R1 +2.29/+1.61 ≈ 上轮 residual_full +2.76/+1.87（方向与量级一致）；
  - **Electricity-336 本轮 R1 弱于上轮**：R1 仅 +0.91/+0.00（MAE 舍入级），上轮 residual_full +1.92/+0.74 双指标超 → 输出凸融合头在该 setting 弱于残差重构机制，不能重复上轮的金标准级优势。
  - 本轮未覆盖上轮超金标准的 ETTh1-720 / ETTh2-336 / Electricity-720（setting 不同，无矛盾）。

## 4. 假设核对（计划 §2 预注册）

| 假设 | 结论 | 证据 |
|---|---|---|
| H1 R1 对强趋势 setting 最强，但易污染已正确的 phase path | **部分成立** | ETTh2（强趋势）−7.7% 最强；ETTh1/ETTm1 中性/微负，未显著"污染" |
| H2 R2 加法修正比 R1 保守，在有害 setting 回退更小 | **不成立** | ETTh1 R2 −1.07% vs R1 −0.75%（MAE）、ETTm1 R2 −0.61% vs R1 −0.19%（MAE），R2 回退反而更大 |
| H3 R3 潜空间长跳连更适合弱误差 | **不成立** | Stage A R3 全 setting 仅 +0.4~5%，远低于输出融合；未被冻结 |
| H4 R4 逐层注入只在多层稳定优于 R3 | **不成立** | 多层 setting R4 与 R3 打平或略差（ETTh1 0.70 vs 0.67；ETTm1 2.44 vs 2.41；Electricity 3.36 vs 3.62） |
| H5 R5 混合须同时超过 R2/R4 才支持互补 | **不成立** | Stage A R5（13.48）≈ R2（13.59）< R1（15.55），判为冗余 |
| H6（追加轮）输出逐层级联（A1/A2 在每层施加 R1/R2 融合）优于单点输出融合 | **不成立** | 测试集除 A2@Electricity 外全部多层 setting A1≤R1、A2≤R2；平均 score R1 1.75 > A1 1.38、R2 1.53 > A2 1.31。验证集曾支持 A1≥R1，未转移 |

## 5. 判定标准核对（计划 §5）

- **首要（跨 setting 双指标方向一致）：不满足。** 仅 ETTh2/Electricity 正、ETTh1/ETTm1 中性/微负；单点大幅提升（ETTh2 −7.7%）被其他 setting 的中性/负抵消。
- **次要（相同精度下参数更少、更快）：** R1/R2 新增参数 ~242K–519K（ETT 小模型 +0~0.5K，Electricity +246K），均早停于 17–28 epoch，比 original 更快收敛；R1/R2 之间参数接近。
- **R4 vs R3（每层注入优于单一长跳连）：无证据。**
- **R2 vs R1（加法误差修正优于完整预测凸融合）：无证据。** R1 在 3/4 setting 优于或等于 R2。
- **R5 vs R2/R4（多级残差互补）：无证据，判为冗余。**
- **A1/A2 vs R1/R2（逐层级联优于单点输出融合，追加轮）：无证据。** 测试集除 A2@Electricity 外逐层级联不优于单点；验证集"逐层 > 单点"信号未转移。逐层级联新增参数（每中间层 ~21.6K）却无稳定收益 → **不采纳**。
- **三 seed 复核：未做。** 单 seed 只能称为配对证据；**不更新 `_LATEST_POLICY`，不宣称稳定泛化提升。**

## 6. 最终决策

- **无跨 setting 冠军拓扑。** 残差输出融合（凸融合 R1，即现有 `WeakPeriodResidualHead` 设计）在测试集上仍是正确的插入点；加法修正（R2）不构成改进。
- 残差收益与**数据集/模型规模相关**：对 1 层主干（ETTh2-h720）和 Electricity 稳定为正，对多层 ETT（ETTh1/ETTm1）中性/微负。这与先前动态相位阶段 full-budget 结论一致（residual 对 ETTh2/Electricity 正、ETTh1/ETTm1 中性、Traffic 负）。
- **A1/A2 逐层级联判为不采纳**（输出×深度设计空间补齐，结论与首轮一致）：单点输出融合（R1）仍是正确插入点；逐层中间融合不转移、多参数无稳定收益。**输出×深度格子已完整测试：R1/R2（单点）× A1/A2（逐层）**，该维度不再有未测的合法形态。
- 若后续要推进：可考虑仅在残差正收益的 setting 上启用残差（数据依赖门控），或对 1 层主干验证多 seed；不推荐继续探索 latent/layerwise 拓扑。

## 7. 数据落盘与复核

- Stage A：`research_runs/residual_topology_screen_runs/runs/*/metrics.csv`（首轮 24 个 + A1/A2 追加 8 个 = 32 个）+ `screen_summary.csv` + `stage_a_selection_notes.md`（含追加轮筛选与冻结决策）。
- Stage B：`research_runs/residual_topology_full_runs/*/metrics.csv`（首轮 12 个 + A1/A2 追加 8 个 = 20 个）+ 各 setting `*_summary.csv` + `full_summary.csv`。
- 复核：ETTh2 R3≡R4 数值相等（stage A 落盘）；ETTh1/ETTh2/ETTm1/Electricity 的 Δ 均从 metrics.csv 重算核对（一致）；ETTh2 A1≡R1、A2≡R2 数值完全相等（追加轮落盘）；§3.3 金标准相对改善率从 `full_summary.csv` + `docs/PhaseFormer_gold_standard.md` 重算核对（一致）；全部 run `resumed=false`（无断点污染，`--resume` 仅跳过已完成的 original）。

## 附录 A：run 明细（Stage B）

见 `research_runs/residual_topology_full_runs/full_summary.csv`（20 行：首轮 original+R1+R2 12 行 + A1/A2 追加 8 行，含 run_id、elapsed_sec；epochs 见各 setting `*_summary.csv`）。
