# PhaseFormer docs 索引（四结构保留版）

本仓库只保留四类模型结构。本文档把每一结构的**机制 preset、代码位置、完整实验结果与参数
组合、复现入口**汇总在一个地方，并指向对应的权威登记文档（数值一律以被指向的文档为准，
不在本文档复制数值表，避免多处漂移）。

> 清理记录：2026-09-02 移除冗余实验家族 docs（TriAxis、M3/multi-anchor、HPTC、ICPT
> 周期间 transformer 头、纯相位/动态相位/残差拓扑、PCTF v1/v2 早期谱系）。这些结构不属于
> 保留四结构，其 docs 已从 git 中删除（历史仍可恢复）；`src/` 与 `scripts/` 未改动。

## 保留结构与机制名

| # | 结构 | `mechanism`（phaseformer_presets） | 代码 | 权威结果/登记文档 |
|---|---|---|---|---|
| K1 | 原始 PhaseFormer | `original`（默认） | `src/models/PhaseFormer.py` | [`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md) |
| K2 | PhaseFormer + NLinear + RCRF（**LFF 周期位置编码**） | `rcrf_pe_lff` | `phaseformer_presets.py`（PERIODIC_RESIDUAL_PE_MODES） | `periodic_residual_pe_experiment` / `top5_test_models` / `periodic_residual_next_stage` |
| K3 | PhaseFormer + NLinear + RCRF（**无位置编码**） | `gold_combo_reliability_s2` | `phaseformer_presets.py`（gold-combo 族） | `gold_combo_experiment` / `top5_test_models` / `periodic_residual_next_stage` / `ETTm2_RCRF_sample_analysis` |
| K4 | strict-T28 扩展线（A2 锚点 + 有界 ICPT 修正，单 checkpoint） | `pctf_anchor_repair_strict_t28` | `phaseformer_presets.py`（PCTF anchored 族）、`anchored_phase_cycle_fusion.py` | [`PhaseFormer_strict_t28_global_golden_plan.md`](PhaseFormer_strict_t28_global_golden_plan.md) + `strict_t28_master_table_configs/` |

结构关系：**K2 = K3 + `use_periodic_residual_pe=True`（LFF 周期位置编码）**；
**K4 以 K2（文档中称 A2）的完整预测为锚点，只叠加有界的 ICPT 周期 level/shape 修正**，
融合器对 A2 输入完全 stop-gradient。

> **incumbent 状态注（2026-09-15）**：本索引此前把 K4 写成"当前最佳"，与两份登记文档
> 的表述不一致（`top5_test_models.md` / `periodic_residual_next_stage.md` 称 A2 为
> incumbent；K4 自己的计划写明 T28"尚未超过 two-stage Full Repair，是本轮的起点"）。
> 统一后的准确表述是：**A2（K2）是最后一个完成三 seed 正式验证的统一 incumbent；
> K4 是其后继的单-checkpoint 扩展线**，在 20 个已登记 setting 中 12 个双指标优于 Golden
> （其中 ETTh1/ETTm1/Weather 共 12 格为单 seed 的 test-set selection，Electricity/Traffic
> 已 CANCELLED），因此**尚未形成"全数据集最优"的既定结论**。

## 通用实验协议（所有结构，正式 test）

- 输入 720 → 输出 H96/192/336/720；full-train；最低 validation loss checkpoint；
  seeds 2021/2022/2023；每 checkpoint 只读一次 test。
  ⚠️ 例外：K4 的 20 个已登记 setting 中 12 个（ETTh1/ETTm1/Weather）为 **seed 2021 单 seed**
  （其中 Stage A/B/C 另用 30% train、8 epoch 作筛选）。
- 按 test 结果继续调参的所有实验都已按 `test-set selection` 披露，完整搜索轨迹保留在
  `agent-log.md` 与各登记文档；不得表述为盲测。
- 提升声明统一相对固定 Golden（`PhaseFormer_gold_standard.md`），matched rerun 只用于协议
  诊断（`MANAGE_RULES.md`「金标准优先」）。

## K1 — 原始 PhaseFormer

- 论文固定结果表：`PhaseFormer_gold_standard.md`（ETTh1/ETTh2/ETTm1/ETTm2/Weather/
  Electricity/Traffic × H96/192/336/720，28 settings；Exchange 无权威截图故不入主结论）。
- 该表是**固定参照**，不随候选调参更新；候选超越均相对此表判定。
- 复现：`phaseformer_presets.build_hyperparams` 默认 scheme（`original`）。matched rerun
  仅用于协议诊断，不静默替换金标准。

## K2 — NLinear + RCRF（LFF 周期位置编码），mechanism=`rcrf_pe_lff`

- 结构：PhaseFormer 相位主干预测周期主体；LFF-NLinear 残差头用可学习 Fourier Features 编码
  历史/未来位置，按位置相似度检索周期副本并与 NLinear 全 horizon 轨迹混合（逐 horizon
  `beta`）；外层 RCRF 按相位可靠度在相位分支与残差分支间连续融合。
- 代码：`get_ablation_overrides("rcrf_pe_lff")`（`phaseformer_presets.py`，由
  `gold_combo_reliability_s2` 派生并加 `weak_period_residual_head_type="periodic_pe"`、
  `use_periodic_residual_pe=True`、`type="lff"`、dim=16、temperature=0.1、
  cycle_decay=0.1、blend_init=0.1）。
- 结果文档：
  - [`PhaseFormer_periodic_residual_pe_experiment.md`](PhaseFormer_periodic_residual_pe_experiment.md)
    —— PE 消融闭环（本结构定义与完整结果）。
  - [`PhaseFormer_top5_test_models.md`](PhaseFormer_top5_test_models.md) —— 正式 5 模型
    矩阵中的 **A2** 行（6 数据集 × H96/H192 × 3 seed）。
  - [`PhaseFormer_periodic_residual_next_stage.md`](PhaseFormer_periodic_residual_next_stage.md)
    —— 该矩阵全部 3-seed 均值/sample std、参数量、训练时间与机制诊断的完整附录。
- 复现入口：`scripts/search_phaseformer.py --mechanism rcrf_pe_lff`（各 dataset/horizon/
  stage 的完整参数组合见上述登记文档）。

## K3 — NLinear + RCRF（无位置编码），mechanism=`gold_combo_reliability_s2`

- 结构：与 K2 相同但**不带周期位置编码**，NLinear 直接补近期水平/漂移/非周期变化；外层
  RCRF 融合。共享相位栈：uncertainty min 0.2/trend gate 0.05、period-level 0.2/
  slope gate 0.05、high-frequency 0.8-0.5-w7、residual gate α0=0.5、RCRF sensitivity 2.0/
  s_max 4.0。
- 结果文档：
  - [`PhaseFormer_gold_combo_experiment.md`](PhaseFormer_gold_combo_experiment.md)
    —— golden-combo 机制闭环（本结构定义与完整结果）。
  - [`PhaseFormer_top5_test_models.md`](PhaseFormer_top5_test_models.md) 中 **A1** 行 +
    `periodic_residual_next_stage.md` 附录。
  - [`ETTm2_RCRF_sample_analysis/ETTm2_RCRF_sample_analysis.md`](ETTm2_RCRF_sample_analysis/ETTm2_RCRF_sample_analysis.md)
    —— RCRF 公式在 ETTm2 的样本级证据。
- 复现入口：`scripts/search_phaseformer.py --mechanism gold_combo_reliability_s2`。

### K3 的因果消融对照：`rcrf_nlinear_plain`

- 该机制只保留原始 PhaseFormer 相位主干、共享 NLinear 残差头与 RCRF（`alpha_0=0.5`、
  `s_0=2`、`s_max=4`）；它明确关闭 uncertainty shrinkage、period-level calibration 与
  high-frequency damping。
- 它是检验 RCRF 独立贡献的正式可复现对照，不属于上表的独立保留结构或 incumbent。复现入口：
  `scripts/search_phaseformer.py --mechanism rcrf_nlinear_plain`。

## K4 — strict-T28 扩展线，mechanism=`pctf_anchor_repair_strict_t28`

> 状态：如上方"incumbent 状态注"所述，K4 **不是**被登记文档宣布的"当前最佳"；
> 它是 A2 之后的单-checkpoint 扩展线（12/20 setting 双指标优于 Golden，其中 12 格为
> 单 seed test-set selection）。

- 结构：完整 A2（=K2，见上）预测为**锚点**，单次 `Trainer.fit`、随机初始化、一个
  checkpoint；只叠加有界的 ICPT 周期 level/shape 修正；composer 对 A2 输入完全
  stop-gradient，A2 只由 anchor loss 训练。
- 冻结训练设置：lookback=720、Huber（ETTh1/ETTm1 用 MAE）、最多 30 epoch（best-val、
  ETTh1/ETTm1 为 50）、anchor/composer LR=1、anchor loss=1、shape/level/gate aux=0.05
  （ETTm1 为 0.01）、无 warm-up。
- 每数据集共享 cycle + trust-region 档位（同数据集四 horizon 共用一个配置，不按 horizon
  切换机制）：

  | 数据集 | cycle | correction/deformation/global-level | loss | lr× | 备注 |
  |---|---|---|---|---|---|
  | ETTh1 | 24 | 1.40 / 0.80 / 0.40 | MAE | 0.2 | 共享最优 `u_lr020`（其他机器搜索复制） |
  | ETTh2 | 48 | 0.25 / 0.10 / 0.05 | Huber | 1.0 | C 档，3-seed Stage D |
  | ETTm1 | 24 | 0.60 / 0.24 / 0.12 | MAE | 0.2 | 共享最优 `w_aux01`（aux=0.01） |
  | ETTm2 | 24 | 0.25 / 0.10 / 0.05 | Huber | 1.0 | C 档，3-seed Stage D |
  | Weather | 24 | 0.60 / 0.24 / 0.12 | MAE | 1.0 | W 档 |
  | Electricity / Traffic | — | CANCELLED（未运行） | | | 见登记表 |

  权威数值与完整 commands 见 `strict_t28_master_table_configs/<Dataset>/<h<horizon>>/
  {config.json,commands.sh}`（README 见 `strict_t28_master_table_configs/README.md`）。
- 结果文档：
  - [`PhaseFormer_strict_t28_global_golden_plan.md`](PhaseFormer_strict_t28_global_golden_plan.md)
    —— Stage A→D 流程 + **Stage D 完整登记表**（2026-09-02 权威）。
  - [`PhaseFormer_strict_t28_best_long_horizons.md`](PhaseFormer_strict_t28_best_long_horizons.md)
    —— H336/H720 扩展。
  - [`PhaseFormer_strict_t28_etth1_test.md`](PhaseFormer_strict_t28_etth1_test.md)、
    [`PhaseFormer_strict_t28_etth1_retune.md`](PhaseFormer_strict_t28_etth1_retune.md)、
    [`PhaseFormer_strict_t28_ett_golden_hunt.md`](PhaseFormer_strict_t28_ett_golden_hunt.md)
    —— ETTh1/ETTm1 的正式对比、重推导计划与 test-set selection 搜索轨迹（保留以披露选择过程）。
  - [`PhaseFormer_pctf_anchor_formal_etts.md`](PhaseFormer_pctf_anchor_formal_etts.md)
    —— 前身 two-stage Full Repair 与 A2 的 ETTh2/ETTm2 正式测试；strict-T28 的 master 计划
    明确以 Full Repair 为参照（注册时刻尚未超越），此文件保留该对照基线。
- 复现入口：
  1. `scripts/search_phaseformer.py --stage confirm --mechanism pctf_anchor_repair_strict_t28
     --dataset <ds> --horizon <h> ...`（参数照 `strict_t28_master_table_configs/` 的
     config.json/commands.sh）。
  2. `scripts/run_strict_t28_global_golden.py`（多数据集驱动）、
     `scripts/report_strict_t28_master_table.py` / `collect_strict_t28_configs.py`
     （登记表读写）。
  3. 选择轨迹核验：`verify_strict_t28_golden_goal.py` 等 `verify_*` 脚本。

## 机制消融：弱周期残差 / NLinear 支路（非保留结构）

- 该族测试 **PhaseFormer + NLinear 弱周期残差支路**，属机制消融，不属于上表四结构，
  **不产生可对外声明的提升**（该支路占全模型参数 92.5%–99.9%，但全部结果受
  test-set selection 与 seed 噪声约束，已按此披露）。
- 相关 mechanism（`phaseformer_presets.py`）：`weak_residual`（等价 `direct_nlinear` 头，
  全部压缩实验的对照）与 `weak_residual_asymmetric_trend`（H1/H3/H4 的 M1 臂）。

**演化链条（2026-09-10 → 09-15）**

| 阶段 | 文档 | 结论 |
|---|---|---|
| 前身探索 | [`PhaseFormer_pooled_lowrank_nlinear_experiment.md`](PhaseFormer_pooled_lowrank_nlinear_experiment.md) | H96 screen：无一候选双指标改善；其"Controlled Follow-up Plan"已被下一行取代 |
| 第一轮（全矩阵） | [`PhaseFormer_joint_lowrank_rank_sweep_plan.md`](PhaseFormer_joint_lowrank_rank_sweep_plan.md) | 70 runs / 10 setting：**无可检测的一致低秩效应**（null） |
| 第二轮（条件性） | [`PhaseFormer_rank_sweep_conditioned_plan.md`](PhaseFormer_rank_sweep_conditioned_plan.md)（预注册规则）→ [`PhaseFormer_rank_sweep_conditioned_experiment.md`](PhaseFormer_rank_sweep_conditioned_experiment.md)（**数值权威副本**） | Stage 0 28 + Stage 1 49 runs；单 seed 判 3/7"部分信号"；**§7 三 seed 复核（105/105 审计单元）修订为"中等压缩近中性、深压缩偏害、无统一最优秩"**，唯一可复现增益为 ETTh2-720 |
| 平滑（时间分辨率轴） | [`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`](PhaseFormer_residual_smooth_ratio_sweep_experiment.md)（boxcar）、[`PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md`](PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md)（causal EMA） | 各 35 runs；判 2/7 与 3/7，方向一律"越平滑越差"；两轮 14 个 (setting, 算子) 组合无一改善 |
| 机制解释 | [`PhaseFormer_lowrank_mechanism_analysis.md`](PhaseFormer_lowrank_mechanism_analysis.md) | SVD 谱 / SVD 截断 vs 训练低秩 / 基向量 FFT / 频段探针（探针未验证核心假设） |
| **容量与数据侧性质（最新）** | [`PhaseFormer_rank_capacity_and_data_property_report.md`](PhaseFormer_rank_capacity_and_data_property_report.md) | 最优秩-r 映射（RRR 上界）在**最深测试档**（参数 3.5%–6.1%）仍保留 **92.4%–101.9%** 的可实现提升；90% 只需 2–4 维；gate 只暴露 g²=4.3–25.7% 的分支误差；**§4 为论文投放版** |
| 弱残差前史（已结题） | [`Weak_residual_asymmetric_component_plan.md`](Weak_residual_asymmetric_component_plan.md)、[`Weak_residual_three_trend_components_experiment_plan.md`](Weak_residual_three_trend_components_experiment_plan.md)、[`Weak_residual_trend_component_study_closure.md`](Weak_residual_trend_component_study_closure.md) | A1–A6 趋势成分 + X-A/Only-A 路由；结论"A 是条件性校正而非完整表征"；2026-09-05 结题并交接给 NLinear 瓶颈研究 |
| 信息瓶颈父计划 | [`PhaseFormer_NLinear_Progressive_IB_Experiment_Plan_v1.0.md`](PhaseFormer_NLinear_Progressive_IB_Experiment_Plan_v1.0.md) | 形式化 Stage 1–10 计划；Stage 2/3 命题已由上一行的容量报告实质回答；**该文件状态块尚未同步更新** |

- **当前探索指导文档**：[`PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md`](PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md)
  —— 按用户授权采用 test-oriented、single-seed、宽度优先搜索，比较周期轴低秩、
  segment basis、level-shape、近期周期稀疏和可分离映射，并为每轮提供实验设置、
  test 选择规则、效率指标、bad-case 记录和代填充表格。

- 脚本：`scripts/plot_conditioned_rank_sweep.py`、
  `scripts/plot_3seed_conditioned_rank_sweep.py`、
  `scripts/analyze_conditioned_rank_sweep_multiseed.py`，
  以及容量分析 5 件套：`analyze_optimal_lowrank_capture.py`、
  `analyze_trained_gate_and_rank.py`、`verify_optimal_rank_identity.py`、
  `describe_leading_direction.py`、`align_trained_and_optimal_direction.py`。
  产物均在 `research_runs/`（gitignored）。

## 输入成分利用诊断（D0 已完成，D1 未收尾）

- 预注册计划：[`PhaseFormer_input_component_H1_H3_H4_plan.md`](PhaseFormer_input_component_H1_H3_H4_plan.md)
  —— 原始 PhaseFormer（M0）、`weak_residual`（M1）与 `rcrf_nlinear_plain`（M2）对
  H1 同相位残差、H3 近期漂移、H4 相位漂移的输入消融。
- **状态：D0（`horizon=192 × seed=2021 × 7 数据集`）已全链路完成**（2026-09-03）：
  Track R 210 runs → 审计 → Track F 210 读 → retrained 189 checkpoint →
  `result_summary_d0.csv` 420 行（qc 420/420）+ 宏表 60 行。结果见
  [`PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md`](PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md)
  （**provisional，单 seed**；三假设均未达 Strong/Partial）。
  **D1（三 seed）仍在进行中，尚无最终分级**（计划 §13.7 为空）。
- 候选发现（ETTm1-H192，已早停）：
  [`PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md`](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md)。
- 诊断步骤（ETTm1-H192、512 origins、validation-only、不读 test）：
  [`D4`](PhaseFormer_input_component_D4_complementary_frozen_report.md)、
  [`D5` 计划](PhaseFormer_input_component_D5_broad_frozen_plan.md) /
  [`D5` 报告](PhaseFormer_input_component_D5_broad_frozen_report.md)、
  [`D6` 计划](PhaseFormer_input_component_D6_structural_relation_plan.md) /
  [`D6` 报告](PhaseFormer_input_component_D6_structural_relation_report.md)、
  [`D7` 计划](PhaseFormer_input_component_D7_internal_path_plan.md) /
  [`D7` 报告](PhaseFormer_input_component_D7_internal_path_report.md)。
  **结论：不存在"M0 忽略、增强分支在用"的候选**；D7 把缺陷定位到"跨周期水平状态
  （尤其最后周期水平偏移）"。
- 全景汇总：[`PhaseFormer_input_component_evidence_summary.md`](PhaseFormer_input_component_evidence_summary.md)；
  叙事背景：[`PhaseFormer_structural_defect_research_narrative.md`](PhaseFormer_structural_defect_research_narrative.md)。
- ⚠️ 命名冲突：本节的 `D1/D2/D3`（候选发现线）与 H1/H3/H4 计划里的 `D1`（其余
  horizon×seed 扩展范围）**含义不同**，引用时必须带前缀。

## 288-run 正式矩阵的额外 mechanism（非保留结构）

`periodic_residual_next_stage.md`（12 setting × 8 mode × 3 seed）与
`top5_test_models.md` 的排名主体包含以下 mechanism，它们不属于上表四结构：

| 代号 | mechanism | 说明 |
|---|---|---|
| I0 | `rcrf_icpt_none` | 原始 future-query decoder ICPT（无 PE）；ETTh2-96 有 +6.5% 回退，未替代 A2 |
| I1 | `rcrf_icpt_horizon_none` | ordered full-horizon ICPT（无 PE） |
| D1 | `rcrf_phase_error_memory` | 相位模板误差的内容条件化周期记忆 + NLinear |
| D2 | `rcrf_dual_reliability_lff` | 双可靠度 LFF 融合 |
| D3 | `rcrf_multiperiod` | 多周期残差 |

## 实验文档审计与近期日志

- [`PhaseFormer_experiment_documentation_review.md`](PhaseFormer_experiment_documentation_review.md)
  —— 全量文档审计（链条、断点、校对清单）与**近期实验一览（2026-09-02 → 09-15）**。
- 逐条操作记录统一写 [`agent-log.md`](agent-log.md)（按时间追加；⚠️ 文件开头有一个
  2026-09-10 的补记块，其后才是顺序主体；对已删除文档的历史引用见该文件说明）。

## 登记表/结果文档的变更纪律

- `PhaseFormer_gold_standard.md` 只由固定论文参照更新；候选结果不写入。
- `PhaseFormer_strict_t28_global_golden_plan.md` 的「Stage D 完整登记表」是 K4 的权威
  结果登记，任何新增/更正实验必须在 `strict_t28_master_table_configs/` 补对应 config 与
  commands，并在 `agent-log.md` 记录验证命令与结果位置。
- 历史搜索与已完成实验的追加记录统一写 `agent-log.md`（按时间追加）。
