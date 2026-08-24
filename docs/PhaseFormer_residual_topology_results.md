# PhaseFormer 残差通路拓扑实验 — 结果反馈

> 状态：Stage A（验证集筛选）与 Stage B（全预算确认）已完成，见下。
> 计划锚点：`docs/PhaseFormer_residual_topology_plan.md`；本文件是结果、失败、协议偏差与最终决策的回填。
>
> 分支：`weak-residual-phaseformer`
> 实验日期：2026-08-24
> 代码提交：`50a9b21`（计划）、`6f5b607`（实现）；本实验未改代码。

## 0. 摘要

- **Stage A**（30% 数据、≤8 epoch、验证集、4 settings × 6 modes = 24 run）：R3/R4 在 ETTh2-h720（1 层）**数值等价**（val_mae=0.66184554、val_mse=0.82789717 完全相等），实现正确性验证通过。按 score（0.5×ΔMAE% + 0.5×ΔMSE%）前两名 = R1 `residual_output_convex`（15.55）与 R2 `residual_output_additive`（13.59）；4/4 settings 双指标无回退，满足冻结条件。
- **Stage B**（100% 数据、≤30 epoch、best checkpoint、测试集、3 ETT + Electricity = 12 run）：残差输出融合的收益**跨 setting 不一致**——
  - **ETTh2-h720（1 层）强正**：R1 ΔMSE −7.7% / ΔMAE −5.8%；R2 ΔMSE −7.6% / ΔMAE −5.7%。
  - **Electricity-h336（2 层）微正**：R1 ΔMSE −1.6% / ΔMAE −0.8%；R2 ΔMSE −1.3% / ΔMAE −0.4%。
  - **ETTh1-h336（3 层）中性/微负**：R1 ΔMSE −0.06% / ΔMAE +0.75%；R2 ΔMSE +1.0% / ΔMAE +1.1%。
  - **ETTm1-h720（2 层）中性/微负**：R1 ΔMSE +0.83% / ΔMAE +0.19%；R2 ΔMSE −0.01% / ΔMAE +0.61%。
- **主要结论**：残差拓扑问题在当前证据下收敛为——**输出层凸融合（R1，即现有设计）已是正确插入点**；加法修正（R2）、潜空间长跳连（R3）、逐层注入（R4）、混合（R5）均不优于 R1。收益与先前动态相位阶段一致地依赖数据集（ETTh2/Electricity 正、ETTh1/ETTm1 中性），**跨 setting 方向不一致，无冠军拓扑**；单 seed 结论，不更新 `_LATEST_POLICY`。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 模型 | PhaseFormer（weak-residual 分支），period 24，lookback 720 |
| 模式 | R0 original；R1 residual_output_convex；R2 residual_output_additive；R3 residual_latent_long；R4 residual_latent_layerwise；R5 residual_hybrid |
| Stage A | 30% 训练数据，≤8 epoch，仅验证集（`--evaluate-test` 关闭），seed 2021，huber |
| Stage B | 100% 数据，≤30 epoch，val early stop + best checkpoint，测试集，seed 2021 |
| 设置 | ETTh1-h336（3 层）、ETTh2-h720（1 层）、ETTm1-h720（2 层）、Electricity-h336（2 层） |
| Stage A 命令 | `scripts/search_phaseformer.py --stage mechanism_screen_1`（24 run） |
| Stage B 命令 | `scripts/benchmark_phaseformer_suite.py`（12 run） |
| 数据落盘 | `research_runs/residual_topology_screen_runs/`、`research_runs/residual_topology_full_runs/` |
| 单测 | `tests/` 全量 90/90 通过 |

**协议偏差**：
- 计划 Stage B 原定"先确认 ETTh1/ETTh2/ETTm1；Electricity 仅在候选通过前三项且仍有正向信号时运行"。前三项结果方向不一致（1 强正 + 2 中性/微负），门槛处于临界；因先前动态相位 full-budget 已观测 Electricity 残差稳定正收益，且该 setting 恰为检验"跨规模是否成立"而设，**补跑了 Electricity-h336**（额外 ~1 GPU·h），使结论覆盖全部 4 个计划 setting。此为判断性补充，不改变前三个 setting 的结论。
- ETTh2-h720 在 Stage A 的 R3/R4 参数仅 734（1 层小模型），与计划一致。

## 2. Stage A 结果（验证集，Δ% 正 = 改善）

| Setting | R1 convex | R2 additive | R5 hybrid | R3 latent_long | R4 latent_layerwise |
|---|---:|---:|---:|---:|---:|
| ETTh1-h336 | +12.81/+18.44 (15.63) | +9.39/+12.97 (11.18) | +9.24/+12.26 (10.75) | +0.52/+0.82 (0.67) | +0.53/+0.86 (0.70) |
| ETTh2-h720 | +15.14/+23.33 (19.23) | +13.08/+20.36 (16.72) | +13.10/+20.31 (16.70) | +0.39/+0.74 (0.57) | +0.39/+0.74 (0.57) |
| ETTm1-h720 | +13.91/+18.43 (16.17) | +13.15/+17.58 (15.37) | +13.34/+17.83 (15.58) | +2.01/+2.81 (2.41) | +2.01/+2.87 (2.44) |
| Electricity-h336 | +7.30/+15.05 (11.17) | +7.27/+14.92 (11.09) | +7.07/+14.67 (10.87) | +2.23/+5.00 (3.62) | +2.15/+4.57 (3.36) |

单元格 = ΔMAE%/ΔMSE%（score）。score = 0.5×ΔMAE + 0.5×ΔMSE。

**R3 ≡ R4 等价验证**（ETTh2-h720，1 层）：两者 val_mae=0.66184554、val_mse=0.82789717、params=734，**完全相等** → 通过。

**冻结决策**：平均 score R1=15.55、R2=13.59、R5=13.48、R3=1.82、R4=1.77。全部候选 4/4 settings 双指标改善、无 >0.5% 回退。保留 **R0 + R1 + R2** 进入 Stage B。

## 3. Stage B 结果（测试集，Δ% 正 = 改善，matched rerun）

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

## 4. 假设核对（计划 §2 预注册）

| 假设 | 结论 | 证据 |
|---|---|---|
| H1 R1 对强趋势 setting 最强，但易污染已正确的 phase path | **部分成立** | ETTh2（强趋势）−7.7% 最强；ETTh1/ETTm1 中性/微负，未显著"污染" |
| H2 R2 加法修正比 R1 保守，在有害 setting 回退更小 | **不成立** | ETTh1 R2 −1.07% vs R1 −0.75%（MAE）、ETTm1 R2 −0.61% vs R1 −0.19%（MAE），R2 回退反而更大 |
| H3 R3 潜空间长跳连更适合弱误差 | **不成立** | Stage A R3 全 setting 仅 +0.4~5%，远低于输出融合；未被冻结 |
| H4 R4 逐层注入只在多层稳定优于 R3 | **不成立** | 多层 setting R4 与 R3 打平或略差（ETTh1 0.70 vs 0.67；ETTm1 2.44 vs 2.41；Electricity 3.36 vs 3.62） |
| H5 R5 混合须同时超过 R2/R4 才支持互补 | **不成立** | Stage A R5（13.48）≈ R2（13.59）< R1（15.55），判为冗余 |

## 5. 判定标准核对（计划 §5）

- **首要（跨 setting 双指标方向一致）：不满足。** 仅 ETTh2/Electricity 正、ETTh1/ETTm1 中性/微负；单点大幅提升（ETTh2 −7.7%）被其他 setting 的中性/负抵消。
- **次要（相同精度下参数更少、更快）：** R1/R2 新增参数 ~242K–519K（ETT 小模型 +0~0.5K，Electricity +246K），均早停于 17–28 epoch，比 original 更快收敛；R1/R2 之间参数接近。
- **R4 vs R3（每层注入优于单一长跳连）：无证据。**
- **R2 vs R1（加法误差修正优于完整预测凸融合）：无证据。** R1 在 3/4 setting 优于或等于 R2。
- **R5 vs R2/R4（多级残差互补）：无证据，判为冗余。**
- **三 seed 复核：未做。** 单 seed 只能称为配对证据；**不更新 `_LATEST_POLICY`，不宣称稳定泛化提升。**

## 6. 最终决策

- **无跨 setting 冠军拓扑。** 残差输出融合（凸融合 R1，即现有 `WeakPeriodResidualHead` 设计）在测试集上仍是正确的插入点；加法修正（R2）不构成改进。
- 残差收益与**数据集/模型规模相关**：对 1 层主干（ETTh2-h720）和 Electricity 稳定为正，对多层 ETT（ETTh1/ETTm1）中性/微负。这与先前动态相位阶段 full-budget 结论一致（residual 对 ETTh2/Electricity 正、ETTh1/ETTm1 中性、Traffic 负）。
- 若后续要推进：可考虑仅在残差正收益的 setting 上启用残差（数据依赖门控），或对 1 层主干验证多 seed；不推荐继续探索 latent/layerwise 拓扑。

## 7. 数据落盘与复核

- Stage A：`research_runs/residual_topology_screen_runs/runs/*/metrics.csv`（24 个）+ `screen_summary.csv` + `stage_a_selection_notes.md`。
- Stage B：`research_runs/residual_topology_full_runs/*/metrics.csv`（12 个）+ 各 setting `*_summary.csv` + `full_summary.csv`。
- 复核：ETTh2 R3≡R4 数值相等（stage A 落盘）；ETTh1/ETTh2/ETTm1/Electricity 的 Δ 均从 metrics.csv 重算核对（一致）；全部 run `resumed=false`（无断点污染）。

## 附录 A：run 明细（Stage B）

见 `research_runs/residual_topology_full_runs/full_summary.csv`（12 行，含 run_id、epochs、elapsed_sec）。
