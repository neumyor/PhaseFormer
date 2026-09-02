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
| K4 | 当前最佳：strict-T28（A2 锚点 + 有界 ICPT 修正，单 checkpoint） | `pctf_anchor_repair_strict_t28` | `phaseformer_presets.py`（PCTF anchored 族）、`anchored_phase_cycle_fusion.py` | [`PhaseFormer_strict_t28_global_golden_plan.md`](PhaseFormer_strict_t28_global_golden_plan.md) + `strict_t28_master_table_configs/` |

结构关系：**K2 = K3 + `use_periodic_residual_pe=True`（LFF 周期位置编码）**；
**K4 以 K2（文档中称 A2）的完整预测为锚点，只叠加有界的 ICPT 周期 level/shape 修正**，
融合器对 A2 输入完全 stop-gradient。

## 通用实验协议（所有结构，正式 test）

- 输入 720 → 输出 H96/192/336/720；full-train；最低 validation loss checkpoint；
  seeds 2021/2022/2023；每 checkpoint 只读一次 test。
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

## K4 — 当前最佳 strict-T28，mechanism=`pctf_anchor_repair_strict_t28`

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

## 输入成分利用诊断（计划中）

- [`PhaseFormer_input_component_H1_H3_H4_plan.md`](PhaseFormer_input_component_H1_H3_H4_plan.md)
  预注册原始 PhaseFormer、`weak_residual` 与 `rcrf_nlinear_plain` 对 H1 同相位残差、H3 近期漂移、
  H4 相位漂移的四输入消融。当前仅有实验设计与空结果表，尚未实现或运行，不属于性能结论。

## 登记表/结果文档的变更纪律

- `PhaseFormer_gold_standard.md` 只由固定论文参照更新；候选结果不写入。
- `PhaseFormer_strict_t28_global_golden_plan.md` 的「Stage D 完整登记表」是 K4 的权威
  结果登记，任何新增/更正实验必须在 `strict_t28_master_table_configs/` 补对应 config 与
  commands，并在 `agent-log.md` 记录验证命令与结果位置。
- 历史搜索与已完成实验的追加记录统一写 `agent-log.md`（按时间追加）。
