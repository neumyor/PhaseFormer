# PhaseFormer 实验与超参数搜索计划 (EXPERIMENT_SEARCH_PLAN)

> **状态更新**：2026-09-10  
> **当前执行分支**：`weak_residual_nlinear_bottleneck`  
> **核心规范遵循**：[MANAGE_RULES.md](MANAGE_RULES.md)、[HOW_TO_DO_RESEARCH.md](HOW_TO_DO_RESEARCH.md) 及 [docs/PhaseFormer_gold_standard.md](docs/PhaseFormer_gold_standard.md)

---

## 1. 当前活跃实验看板 (Active Pipeline)

### 1.1 优先级 1：受控池化低秩 NLinear 因素实验 (Controlled Pooled Low-Rank Screen)
* **专项计划与文档**：[docs/PhaseFormer_pooled_lowrank_nlinear_experiment.md](docs/PhaseFormer_pooled_lowrank_nlinear_experiment.md)
* **核心研究问题**：
  1. 在固定池化倍率 $p$ 下，相对秩 $q$ 对验证集误差的主效应如何？
  2. 在固定相对秩 $q$ 下，时间池化 $p$ 对验证集误差的主效应如何？
  3. 平滑比例 $s$ 是否具有独立主效应，还是仅与池化/秩存在交互效应？
* **数据与任务**：ETTh1、ETTm1（$L=720, H=96$），Huber loss，最多 30 epoch，基于最低验证集损失 checkpoint。
* **执行阶段与预算**：
  * **Phase A（容量主效应，Validation-Only）**：
    * 因子：$p \in \{1, 2, 4\} \times q \in \{1/12, 1/3, 1\} \times \{s=0\}$
    * 对照组：`phase_only`、`direct_nlinear`
    * 规模：2 数据集 $\times$ 3 seeds (2021, 2022, 2023) $\times$ 11 个配置 = **66 次训练**
    * 专用 Runner：`scripts/run_joint_pooled_lowrank_phase_a.py`（严禁传 `--evaluate-test`）
    * 评估分析：`scripts/analyze_joint_pooled_lowrank_phase_a.py`
  * **Phase B（平滑交互，Validation-Only）**：
    * 因子：预注册 6 个 $p \times q$ cell $\times s \in \{0.25, 0.50\}$（严格复用 Phase A 的 $s=0$ 结果）
    * 规模：2 数据集 $\times$ 2 seeds (2021, 2022) $\times$ 12 个非零平滑配置 = **48 次训练**
    * 专用 Runner：`scripts/run_joint_pooled_lowrank_phase_b.py`
  * **Phase C（冻结确认，Test Confirmation）**：
    * 仅在 Phase A/B 完成统计分析并冻结最终最优候选后，进行 1 次 3-seed 测试集评估。

### 1.2 优先级 2：输入成分因果消融实验 (Input Component Causal Ablation)
* **专项计划与文档**：[docs/PhaseFormer_input_component_H1_H3_H4_plan.md](docs/PhaseFormer_input_component_H1_H3_H4_plan.md)
* **状态**：方案已预注册，待池化低秩主线推进完毕后执行。

---

## 2. 全局 32 任务全域搜索规范 (Global 32-Task Search Protocol)

### 2.1 任务网格定义
* **数据集 (8 个)**：ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange。
* **预测步长 (4 个)**：$H \in \{96, 192, 336, 720\}$，统一固定输入序列长度 $L=720$。
* **总任务数**：$8 \times 4 = 32$ 个独立预测任务。

### 2.2 评估与参照基准
* **固定金标准 (Gold Standard)**：提升声明统一以 [docs/PhaseFormer_gold_standard.md](docs/PhaseFormer_gold_standard.md) 为准；Matched Rerun 仅用于环境和协议偏差诊断。
* **综合评分指标**：
  $$\text{Score} = 0.5 \times \text{MAE 相对改善率} + 0.5 \times \text{MSE 相对改善率}$$
* **测试集暴露披露原则**：严格优先使用验证集调优与早停；凡涉及测试集信息反馈的搜索，必须在报告与配置中明确标明 `test-set selection`。

### 2.3 机制与超参数分层搜索协议

```
[阶段 1: 周期筛选] -> [阶段 2: 数据集共享机制筛选] -> [阶段 3: 任务级超参数搜索] -> [阶段 4: 三种子确认]
 (30%数据, 8 epoch)     (四 Horizon 综合验证评分)      (Successive Halving)      (Seeds 2021/2022/2023)
```

1. **阶段 1：周期筛选**
   - 候选周期：ETTh (12, 24, 48)、ETTm (24, 48, 96)、Weather/Electricity/Traffic (12, 24, 48)、Exchange (7, 14, 30)。
   - 在 $H \in \{96, 720\}$ 上用 30% 训练数据初筛，锁定数据集共享 period。
2. **阶段 2：数据集共享机制筛选**
   - 评估 phase-only, fixed weak residual, adaptive residual, phase uncertainty, trend filtering 等机制族。
   - 要求最差 horizon 回退不超过 0.5%，综合评分最优。
3. **阶段 3：超参数逐级减半 (Successive Halving)**
   - 12 组超参组合（Loss: Huber/MAE；LR: 0.3×/1×/3×；Capacity: base/compact）。
   - 初筛 12 组 (30% 数据, 8 epoch) $\to$ Top 4 (100% 数据, 15 epoch) $\to$ Top 2 正式预算 (30 epoch)。
4. **阶段 4：三种子确认与发布**
   - 使用 seeds 2021, 2022, 2023 运行 Baseline 与候选，计算均值与标准差。
   - 若改善小于跨种子标准差，默认保留更精简机制。

---

## 3. 架构与科研红线约束 (Architectural & Research Constraints)

1. **单模型端到端统一原则**：
   - 严禁通过 Stacking, Routing, Averaging 或 Mixture-of-Models 方式拼接多个独立训练的 PhaseFormer 模型。
   - 必须共享同一套 PhaseFormer 相位主干表示，新增分支必须是轻量、归因明确的结构偏置。
2. **审计产物严格 6 文件规范**：
   - 实验归档路径为 `research_runs/<experiment_id>/`，根目录只允许六个文件加 `figures/` 目录：
     `run.yaml`、`results.csv`、`sample_errors.csv`、`selected_cases.npz`、`objective_error_analysis.md`、`objective_error_analysis.zip`。
3. **环境与可复现性规范**：
   - 优先使用完整路径 Conda 解释器，记录 CUDA、Torch、Lightning 版本与 GPU 型号。

---

## 4. 历史实验与归档结论 (Historical Archive & Closures)

| 实验阶段 / 模块 | 核心机制与结论摘要 | 归档文档 |
| :--- | :--- | :--- |
| **Pooled Low-Rank 初筛 (H96)** | ETTh1/ETTm1 72 次训练，未观察到单调秩/池化效应，促成了当前 Phase A/B 受控实验。 | [docs/PhaseFormer_pooled_lowrank_nlinear_experiment.md](docs/PhaseFormer_pooled_lowrank_nlinear_experiment.md) |
| **PCTF 单阶段联合训练** | 测试一次性联合训练以消除 A2 两阶段预训练，50 策略矩阵未能超越两阶段 Full Repair。 | [docs/PhaseFormer_pctf_single_stage_h192_tuning.md](docs/PhaseFormer_pctf_single_stage_h192_tuning.md) |
| **PCTF v2/v3 锚点归因** | 明确 A2 锚点联合训练漂移与梯度干扰问题，完成了严格的配对消融。 | [docs/PhaseFormer_pctf_anchor_attribution_plan.md](docs/PhaseFormer_pctf_anchor_attribution_plan.md) |
| **Multi-Anchor Selector (M3)** | M3 soft routing 带来提升但依赖 3 个独立模型，依据论文红线转为诊断性上界，不再作为主线。 | [docs/PhaseFormer_multi_anchor_selector_experiment.md](docs/PhaseFormer_multi_anchor_selector_experiment.md) |
| **非对称弱残差与趋势滤波** | 探索了 SSA、Global Linear、Causal EMA 与 Trend Filtering 的残差分工与路由角色。 | [docs/Weak_residual_trend_component_study_closure.md](docs/Weak_residual_trend_component_study_closure.md) |
