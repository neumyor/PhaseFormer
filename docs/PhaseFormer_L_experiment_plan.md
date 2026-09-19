# PhaseFormer-L 实验计划（缺口补齐与主结果落地）

> 状态：**已登记，未执行（2026-09-18）。** 本文件是 PhaseFormer-L 线的**当前执行入口**，
> 也是 `docs/PhaseFormer_L_minipaper.md` §4 空表的**唯一结果回写契约**。
> 全部缺口经 `docs/agent-log.md`（2907 行全文）与 12 份被引登记文档逐项核对，结论见 §4：
> minipaper 声明的 18 项实验缺口**全部确认存在**，另有 6 项登记不一致（§4.2）需先修文档。
> 本文件本身**不运行任何实验、不读取 test、不修改任何既有结论口径**。

## 0. 本文件的地位

- **执行入口**：`docs/README.md` 机制消融小节的"当前探索入口"指向本文件；后续 PhaseFormer-L 的
  代码、脚本、训练矩阵、分析阶段一律以本文件的工作包编号（WP0–WP6）为准。
- **结果回写目标**：每个工作包产出的数字必须回写到 §7 指定的位置（minipaper 的哪一张表、
  `research_runs/` 的哪一份 CSV、`agent-log.md` 的哪一条）。**minipaper §4 的表格是回写目标，
  本文件是执行契约**：两者冲突时以本文件的协议与披露要求为准，数值以 `research_runs/` 原始产物为准。
- **谱系定位**：本线是"弱周期残差 / NLinear 支路"机制消融研究（非保留结构，不产生对外提升声明）
  的收束阶段。minipaper 是论文投放产物；本计划是其证据补齐的施工图。
- **Skill 边界**：按 `MANAGE_RULES.md` 第 15 条，若某工作包被执行方要求"实现设想 + 运行实验 +
  分析高误差或显著退化样本"三者同时满足，则该次运行必须触发 `experiment-and-error-analysis`
  Skill，并满足 `HOW_TO_DO_RESEARCH.md` §1.3 的六文件白名单。纯训练无关机制分析与仅有汇总结果的
  解读不触发。当前已可判定：**WP4（干预与解剖）若加入样本级高误差分析即触发 Skill**；WP1/WP2
  为训练无关分析，不触发。

---

## 1. 目的与范围

### 1.1 要回答的问题

minipaper 主张"相位 token 化看不见的部分本质上是一维的跨周期电平漂移，且该修正必须以相位主干为
条件学习"。该主张目前只有 §4.1 的 test-exposed 先导证据支撑。本计划的目标是把 §4.2–§4.7 的
预注册空表填成可发表的盲测与 train/validation 证据。

### 1.2 范围

| 项 | 取值 |
|---|---|
| 数据集 | ETTh1 / ETTh2 / ETTm1 / ETTm2 / Weather / Electricity / Traffic（7 个） |
| horizon | 96 / 192 / 336 / 720 |
| setting 总数 | 28（7 × 4） |
| 输入长度（lookback） | 720 |
| seed | 2021 / 2022 / 2023 |
| 参照 | 固定 Golden（`PhaseFormer_gold_standard.md`）；matched rerun 仅作协议诊断 |
| 盲测边界 | §4.2 主表冻结后不做任何基于 test 的选择；机制分析（§4.3–§4.7 的非增益列）全部用 train/validation |
| 不作范围 | 不宣称任意时序模型可压缩、不宣称任意线性模型等价于电平修正；Exchange 数据集不纳入（无金标准） |

---

## 2. 核对方法与可核对性边界

### 2.1 核对方法

缺口确认按下述三步完成，全部为只读操作：

1. 通读 `docs/agent-log.md` 全部 151 条条目（2907 行），提取与本文相关的 E1–E13 证据账本（§3）；
2. 逐项打开 minipaper §4.1 引用的 6 份登记文档，核对数字、覆盖范围、协议与判定；
3. 在本地 `research_runs/`（13 个目录）逐产物定位，确认哪些证据可在本地复核、哪些只在服务器。

### 2.2 可核对性边界（重要）

| 证据 | 本地可复核性 |
|---|---|
| E6/E7 的 RRR 与主导方向产物 | **可**：`research_runs/lowrank_data_property_v1/` 与 `..._v2/`（含 7 个 `moments_*.npz`、`optimal_rank_capture.csv`、`leading_direction.csv`、`trained_vs_optimal_alignment.csv`） |
| E1/E3/E4/E5/E8/E9 的 run 产物 | **可**：`research_runs/{joint_lowrank_rank_sweep_v1, rank_sweep_2_multiseed_*, smooth_ratio_sweep_v1, causal_ema_smooth_sweep_v1, top2_direction_retention_v1, direction1_neighborhood_v1}/` |
| E10 的 checkpoint 解剖产物 | **不可（仅服务器）**：`canonical_modes.csv`、`semantic_alignment.csv`、`intervention_results.csv` 等本地不存在，无法就地复算 |
| §4.2 参数量/FLOPs 同口径列、ETTh2 的 FITS 参照 | **不可**：仓库内无来源（见 G3、G4） |

因此 §4.2 的 D1/D2 两条登记不一致**必须在服务器侧产物上核对**，不能凭文档断言。

---

## 3. 既有证据账本（按 agent-log 整理）

下表是截至 2026-09-18 与本线相关的**全部**既有实验结果。这是判定缺口的事实基础。

| ID | 实验 | 覆盖范围 | 协议 | 判定 / 关键数字 | 出处 |
|---|---|---|---|---|---|
| E1 | Joint Low-Rank Rank Sweep v1 | {ETTh1,ETTh2,ETTm1,ETTm2,Weather} × {96,192} = 10 setting；臂 = `phase_only` / `direct_nlinear` / q∈{1,1/4,1/8,1/16,1/32}（70 runs） | 单 seed 2021 | **null**：无可检测的一致低秩效应（变差 4/10、变好 2/10、平坦 4/10） | `PhaseFormer_joint_lowrank_rank_sweep_plan.md`；log 09-10/09-11 |
| E2 | Conditioned Low-Rank Sweep Stage 0+1 | 7 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336）；7 档（49 runs） | 单 seed 2021，Stage 0 按 val 冻结配置 | 3/7 **部分信号**；test 最优档位高度分散 | `PhaseFormer_rank_sweep_conditioned_plan.md` / `..._experiment.md`（**数值权威副本**） |
| E3 | 同上三 seed 复核 | 7 setting × 3 seed × 5 臂 = **105/105** 审计单元 | 3 seed，test 一次 | 修订为"中等压缩近中性、深压缩偏害"；宏平均 ΔMSE −0.12/+0.07/−0.32/−0.52%；**唯一 3/3 增益 ETTh2-720 q=1/8** | log 09-14；`..._multiseed_stage1_20260914_summary/` |
| E4 | boxcar 平滑扫描 | 7 setting × `smooth_ratio`∈{0,.25,.5,.75,1} = 35 runs | 单 seed 2021 | 2/7，**无可检测效应**；最优档 4/7 聚在 s=0 | `PhaseFormer_residual_smooth_ratio_sweep_experiment.md` |
| E5 | causal-EMA 平滑扫描 | 7 setting × 5 档 = 35 runs（`alpha=0.08`） | 单 seed 2021 | 3/7 部分信号，方向一律"越平滑越差"；**两轮 14 个 (setting, 算子) 组合无一改善** | `PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md` |
| E6 | RRR 容量分析（训练无关） | 7 setting，val 口径 | 无训练、无 test | λ₁/Σλ = **0.662–0.862**；90% 提升只需 **2–4** 维；PR = **1.33–2.12**；最深档（参数 3.5%–6.1%）保留 **92.4%–101.9%**；ETTh2-720 权重谱 95% 能量需 **418** 维 vs 预测谱 **8** 维；`used_var_share(1)` = **0.8%–12.3%** | `PhaseFormer_rank_capacity_and_data_property_report.md`（§4 为论文投放版） |
| E7 | 主导方向精确刻画 | 7 setting + 363（后 570）checkpoint | 无训练、无 test | `b_1` 最后 24 步占能量 **53%–70%**（168 步 71%–90%）；`a_1` 与常值 \|cos\| = **0.892–0.989**、符号 **7/7** 一致；训练头对齐 \|cos\| = 0.81–0.98（5/7 setting） | 同上 §2.6；log 09-15 |
| E8 | 前两方向冻结保留实验（V1/V2） | 6 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192）× 3 seed；另 `phase_only` 三 seed 18 runs；新增训练 54 次 | 3 seed，test 一次性读取、**无 test-set selection** | 预注册判定 **不支持**：宏平均 ΔMSE **+1.9426%** / ΔMAE **+1.9901%**；V2 贡献保留率中位 61.5%/39.3% | `PhaseFormer_top2_predictive_direction_retention_report.md` + `..._summary.md` |
| E9 | 方向 1 邻域宽度 | 7 setting × 3 seed，98 runs（17.36 GPU·h） | **test-set selection**（宽度按 seed 2021 test 选） | 5 条判据 4 条成立（部分支持）；**主结论不支持"加宽邻域是普遍改进"**，仅 Weather-192 被选 Cone-4 3/3 seed 双指标改善 | `PhaseFormer_direction1_neighborhood_experiment_plan.md` / `..._report.md` |
| E10 | 低秩 checkpoint 信息保留分析 | 6 setting × 3 seed × 4 压缩档 = **72 cell**；10 臂 × 72 = **720 行**干预 | 不重训、不读 test、不改既有 checkpoint | 裁定 **条件性机制 4/6**；H1 支持见 §4.1；`Semantic-drop` 损害 60/72 超出同维随机 95% 区间 | `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` §11（表 1–7） |
| E11 | SVD 截断 vs 秩约束训练 | **7 setting**（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336） | 训练无关，读已有全秩 checkpoint | 6 个 setting 差 1–3%；**Electricity-336 为反例**：r=10 截断 MSE 0.2104 vs 训练低秩 0.1629（差 **29%**） | `PhaseFormer_lowrank_mechanism_analysis.md` §3.2 |
| E12 | 样本级结构缺陷诊断（D7） | ETTm1-192，单 seed | test-exposed | 融合收益与跨周期水平波动 r=+0.49/+0.53；修正与相位残差同向 cos 0.70/0.79 | `PhaseFormer_structural_defect_research_narrative.md` §3 |
| E13 | 结构化坐标负对照 | 4 setting | — | 5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52% | `PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md` |

**账本的三条结构性事实**（决定了缺口规模）：

1. **最大覆盖是 7 个 setting**（E2/E3/E4/E5/E6/E7），且这 7 个正是 test-set selection 挑出的集合；
   10 个 setting 的覆盖只有 E1，且限 H∈{96,192} 且单 seed。**28 个 setting 的完整覆盖在任何实验中都不存在。**
2. **PhaseFormer-L 尚不存在**：`src/models/` 无 §3.4 所述的"EMA 初始化 + 可靠度门控 + 联合训练的
   rank-1/2 电平通道"实现（仅有 `adaptive_residual_gate.py`、`frozen_nlinear_correction.py`、
   `structured_residual_heads.py` 等可复用积木）。§4.2/§4.4/§4.5 的联合列/§4.6 行 1 全部依赖它。
3. **全部既有 test 数字都是条件性的**：E1–E3、E8、E9 的 test 口径或来自 test-exposed 集合，
   或本身是 test-set selection（E9）。E8 是唯一"三 seed + test 一次 + 无选择"的实验，但只有 6 setting。

---

## 4. 缺口确认

### 4.1 实验缺口（G1–G18，全部确认存在）

判定列含义：**确认** = 缺口按 minipaper 描述确实存在；**确认（修正）** = 缺口存在但 minipaper
对现状的陈述需更正。

| # | minipaper 位置 | 声称的缺口 | 账本核对结论 | 现状 / 需补量 | 判定 |
|---|---|---|---|---|---|
| G1 | §4.2 PhaseFormer-L 列 | 28 setting × 3 seed 的 MSE±std / MAE±std | 无任何 PhaseFormer-L run；组件亦未实现 | 0 → **28 × 3** | **确认** |
| G2 | §4.2 对照行 6 变体 | `phase_only`、`direct`、`L-fixed`、`L-rank2`、`L-nogate`、`L-mean` 同协议同 seed | 最接近者：E1（10 setting，单 seed，且是探针）；E3（7 setting 仅 `direct`）；E8 有 6 setting × 3 seed `phase_only` | 0（L-* 全缺）→ **6 × 28 × 3 = 504 runs** | **确认** |
| G3 | §4.2 参数量 / FLOPs 列 | 须与原文 Table 4 同口径 | `PhaseFormer_gold_standard.md` 只有 28 行 MSE/MAE，**无参数量与 FLOPs**；仓库内只有 head 参数量（E1 计划 §4 表） | 口径定义缺失 → 需外部引用或明确本地口径 | **确认** |
| G4 | §4.2 必答 (a) | ETTh2 是否补齐到 FITS 水平 | 仓库内**无 FITS 数字**（金标准 grep 无 FITS） | 0 → 需补外部参照行 | **确认** |
| G5 | §4.2 必答 (b) | ETTh1/ETTm1 上 `g` 是否自动关小 | 探针侧 gate 实测存在（g=0.207–0.507；Electricity-336 唯一系统性关闭 0.433→0.332，Δg=−0.101±0.091）；**PhaseFormer-L 的 gate 不存在** | 部分 → 通道级 `g` 待测 | **确认** |
| G6 | §4.2 必答 (c) | `L-rank2` 相对 `L` 的增量是否在 seed 噪声内 | 无 PhaseFormer-L | 0 → 依赖 G1/G2 | **确认** |
| G7 | §4.3 表 | 28 行 `λ₁/Σλ`、`pred_dims_90`、PR、`b_1` 模板、`a_1` vs 常值、`used_var_share(1)` | 仅 **7 行**（E6/E7）。本地 `moments_*.npz` 恰 7 个 | 缺 **21 setting**：ETTh1×4、ETTh2-192/336、ETTm1×4、ETTm2-336/720、Weather-336/720、Electricity-96/192/720、Traffic×4 | **确认** |
| G8 | §4.3 预期图 | Scree 图 + `b_1` lag 剖面 + `a_1` horizon 剖面 | 容量报告 §4.4 有**图表方案**，未生成论文口径成图 | 0 → 3 类图 × 28 setting | **确认** |
| G9 | §4.4 解剖表 | PhaseFormer-L 与低秩探针的主模式输入/输出组、解释率、能量份额、跨 seed `leading4` 重叠 | E10 只覆盖**低秩探针**（6 setting × 3 seed × 4 档 = 72 cell）；PhaseFormer-L 无 checkpoint | 0（L 列）→ 待 PhaseFormer-L | **确认** |
| G10 | §4.4 干预表末列 | **随机 RRR 子空间 drop** | **文献级确认**：计划 §11.8.2 原文"本实验**没有设置**'随机 RRR 子空间'对照"；且 57/57 cell 的 `Semantic-drop` ≡ `PCA-drop`（差值恰 0.000000），15/72 cell 两"同维对照"维度不同（semantic 36 vs pca 180 等） | 0 → 每 cell 新增 100 次随机 RRR 零分布 | **确认（最高优先级）** |
| G11 | §4.5 冻结条件-RRR 行 | 冻结**条件性** RRR 方向 1 | E8 只做独立目标方向（V1/V2）；log 09-16 明确"当前实验未设置 phase/gate 冻结或支路独立训练对照…不能在信息不足、联合优化干扰与有限样本泛化之间做唯一归因" | 0 → 全新臂 | **确认（贡献 4 的关键）** |
| G12 | §4.5 冻结独立-RRR 行 | 同表 | E8 已有（6 setting × 3 seed，宏平均 +1.94%） | 已有 6 setting；表若要求 28 则需扩展 | **确认（部分已有）** |
| G13 | §4.5 H1 列 | cond 距离 < indep 距离（seed 数） | E10 表 5 已有 72 行；按"seed 内多数 rank 支持"读法为 5/6 setting 3/3 seed（Weather-192 全否；Weather-96 各 seed 为 3/4、2/4、4/4，无 seed 全 rank 支持） | 已有；**须在表注写明判定口径** | **确认（口径待注明）** |
| G14 | §4.6 行 1 | 输入平滑在 **PhaseFormer-L** 上复测 2 档 | E4+E5 共 14 个 (setting, 算子) 组合在**探针**上无一改善 | 0 → 2 档 × 28 × 3 | **确认** |
| G15 | §4.6 行 3 | SVD 截断 vs 秩约束训练扩展到全 28 setting | 现状**不是**只有 Electricity-336：E11 已覆盖 **7 setting**（其余 6 个差 1–3%），minipaper 只引用了反例 | 已有 7 → 缺 21 | **确认（修正：现状被低估）** |
| G16 | §4.6 行 2/行 4 | 表中标"—"（不补做） | 结构化坐标 4 setting（E13）、q=1/32 7 setting（E6）均已存在 | 无需补做 | **核对通过** |
| G17 | §4.7 全表 | 28 setting 的 `cycle_level_std`、`last_cycle_shift`、`τ̂` 及与 ΔMSE / `g` 的 Spearman ρ | **完全不存在**；最接近的是 E12（ETTm1-192 单 setting 内**样本级**相关），与跨 setting 的**setting 级**相关不是同一量，不可替代 | 0 → 28 × 3 统计量 + 2 列相关 | **确认** |
| G18 | §4.0 | 两个判定门槛 `[待定]` | 文档确认仍为待定 | 0 → 须在 WP3 开跑前冻结 | **确认** |

### 4.2 登记不一致（D1–D6，需先修文档，非实验缺口）

本轮核对在 minipaper 与其被引文档之间发现 6 处不一致。**这些不影响 §4 缺口的存在性，但会影响
论文数字的可追溯性**，应在 WP0 一并处理。

| # | 位置 | 问题 | 影响与处置 |
|---|---|---|---|
| D1 | minipaper §4.1 第 6 行 | "删除子空间后支路自身误差变好 **52/72**、融合误差变差 **72/72**（中位 **+30.4%**）"在它引用的两份文档（`PhaseFormer_top2_direction_retention_summary.md`、计划 §11.8）中**均不存在**；全仓库唯一出现处是**未跟踪脚本** `scripts/render_lowrank_semantic_readwrite_slide.py:270` 的硬编码字符串 | 该行是贡献 4 的核心判据，必须从 `intervention_results.csv` 重新导出并与登记文档对齐；本地无该 CSV，须服务器复核 |
| D2 | minipaper §4.1 第 5 行 | "`Semantic-only` Δfused ≤ **+0.0006**" 与计划 §11.8.2 "最大仅 **+0.0027**"不一致（表 6 中 ETTh2-720 q=1/4 `Semantic-only` Δfused MSE = **+0.002665**） | 口径矛盾，须回核后统一（疑为不同 scope 未写明） |
| D3 | minipaper §3.3.3 臂表 | 声称 10 臂并含 `Random-only` / `Random-drop`；但执行的表 6 实际 10 臂为 Original / Bias-off / Conditional-RRR-only / Independent-RRR-only / PCA-only / PCA-drop / Semantic-only / Semantic-drop / **Semantic8-only** / **Semantic8-drop**——随机臂**不是**臂（用作 95% 零分布），`Semantic8-*` 未在 §3.3.3 出现 | §3.3.3 的臂定义须与实际执行清单对齐，否则 §4.4 的复现清单不可执行 |
| D4 | minipaper §4.1 第 8 行 | 把 SVD 截断现状写成 Electricity-336 一例（见 G15） | 改为"7 setting 已测，Electricity-336 为反例" |
| D5 | 计划 §11.2 执行记录 | 记 `intervention_results.csv`（**576** 条），与 agent-log 09-17 修复记录"**720** 行 = 72 cell × 10 臂"不一致（576 = 72 × 8，疑为修复前计数） | 计划 §11.2 计数须更正为 720 |
| D6 | `docs/README.md` | 仍写低秩 checkpoint 信息保留计划为"**下一阶段计划（2026-09-16，待实现）**"，实际已于 **09-17 执行完成**（E10） | 状态行须更正，否则后续执行方会重复实现已完成的分析 |

---

## 5. 工作包

依赖关系：**WP0 → (WP1 ∥ WP2) → WP3 → (WP4 ∥ WP5 ∥ WP6)**。WP1/WP2 训练无关，可在等 GPU 时先做。

### WP0 前置冻结与文档勘误（零 GPU）

| 子项 | 内容 | 产出 |
|---|---|---|
| WP0-1 | **冻结 §4.0 判定门槛**（见 §6，需用户裁定后写入 minipaper §4.0） | minipaper §4.0 无 `[待定]` |
| WP0-2 | **实现 PhaseFormer-L**（minipaper §3.4）：`w` 以 `exp(−lag/τ)` 初始化、`u` 以 `1_H` 初始化、`u₂` 以 ramp 初始化、可靠度门 `g(x)`；并实现 4 个消融开关（`L-fixed` / `L-rank2` / `L-nogate` / `L-mean`） | `src/models/` 新模型 + preset + 单元测试；`py_compile` 与 `pytest tests/ -q` 通过 |
| WP0-3 | **参数量 / FLOPs 口径**（G3）与 **FITS 参照**（G4）：确定"与原文 Table 4 同口径"的可核对定义，补 FITS 在 ETTh2 的引用数字 | 口径段落写入 minipaper §4.2 表注；FITS 行入 §4.2 必答 (a) |
| WP0-4 | **修 D1–D6 六项登记不一致** | 相应文档勘误注；D1/D2 须先取得服务器 `intervention_results.csv` |
| WP0-5 | §4.3/§4.4/§4.5 表注写明判定口径（G13 的"seed 内多数 rank"读法） | minipaper 表注 |

### WP1 §4.3 相位补空间维数（训练无关；CPU）

- **目标**：把 G7 的 7 行扩到 28 行；产出 G8 的三类图。
- **方法**：沿用 `scripts/analyze_optimal_lowrank_capture.py`（`--save-moments` 落盘二阶矩）与
  `scripts/describe_leading_direction.py`；对 21 个新 setting 在 **train split** 上取二阶矩，
  RRR 闭式解 + 首方向模板拟合。
- **写回列来源**（已在既有产物中定位，便于直接复用）：`λ₁/Σλ` ← `leading_direction.csv:lambda1_share_of_achievable`；
  `b_1` 近端质量 ← `mass_last24/72/168`；`b_1` 模板 ← `cos_const`/`cos_ramp` 与 `optimal_rank_capture.csv:lead_dir_cos_*`；
  `a_1` ← `out_cos_const`、`out_sign_consistency`；`used_var_share` ← `optimal_rank_capture.csv:used_var_share`；
  `pred_dims_90` / PR 由 λ 谱导出。
- **技术约束（必须先解决）**：Electricity-336 的条件性 RRR 曾在 Gram 组装阶段超出单进程内存
  （`(17344, 336, 321)` 需 **13.9 GiB**），是 E10 排除该 setting 的原因。28 setting 含 Electricity-336/720
  与 Traffic-336/720，**必须先实现分片/流式二阶矩**，否则重复同一失败。
- **协议**：train/validation 口径，不读 test；报告中显式标注"与 §4.2 的 test 增益列不同源"。

### WP2 §4.7 命题 1 的预测力（训练无关；CPU）

- **目标**：G17 全表。
- **方法**：对 28 setting 在 train split 上计算 `cycle_level_std`、`last_cycle_shift`、电平自相关时间 `τ̂`
  （`τ̂` 的定义须在 WP0 冻结，不得事后调整）；与 WP3 的 ΔMSE 及门值 `g` 求 Spearman ρ。
- **依赖**：统计量列可立即产出；两列相关系数须等 WP3。
- **边界**：E12 的 setting 内样本级相关**不得**当作本表的先导证据（量纲不同，见 G17）。

### WP3 §4.2 主结果（GPU 主力）

- **目标**：G1/G2/G3/G5/G6；并回答必答 (a)(b)(c)。
- **矩阵**：7 变体（`L`、`L-fixed`、`L-rank2`、`L-nogate`、`L-mean`、`direct`、`phase_only`）× 28 setting × 3 seed
  = **588 runs**；`phase_only` 在 6 个 setting 上已有 E8 的三 seed 结果，**复用前须通过协议一致性审计**
  （同 lookback / 同 loss / 同 epochs / 同 best-val 口径），不一致则重训。
- **协议**：full-train、最低 validation loss checkpoint、每 checkpoint 只读一次 test、seeds 2021/2022/2023。
- **"稳定超过"判定**：三 seed 均值 + 样本 std 严格低于 Golden（沿用既有严格标准）。
- **必答项**：(a) ETTh2 对 FITS；(b) ETTh1/ETTm1 的 `g` 是否自动关小（同时报告逐 dataset 的 `g` 均值）；
  (c) `L-rank2` 相对 `L` 的增量是否落在 seed 噪声内。
- **披露**：主表冻结后不做任何基于 test 的选择；若发生任何按 test 的调整，必须在 `run.yaml`、
  agent-log 与 minipaper §5 明确标注 test-set selection。

### WP4 §4.4 解剖与干预

- **目标**：G9/G10；把 E10 的结构扩到 PhaseFormer-L 并补上缺失的对照。
- **新增对照（P0）**：**随机 RRR 子空间 drop**——每 cell 100 次同维度随机（同 RRR 谱）子空间，
  给出 95% 零分布，用于回答 §11.8.2 遗留的"语义有效 vs 任意同数量主方向有效"（D3/G10）。
- **同时报告**：支路自身误差与融合误差（贡献 4 的判据）；同维对照须**真正同维**
  （修正 15/72 cell 维度不匹配问题，或显式标注维度差异）。
- **技术约束**：沿用 E10 的分片/缓存机制；**启动长任务前必须做单 cell 端到端验证**
  （E10 的教训：4 个集成问题本应在启动 72-cell sweep 前一次发现）。
- **Skill 判定**：若本轮加入样本级高误差/退化分析，按 §0 触发 `experiment-and-error-analysis`。

### WP5 §4.5 条件性学习

- **目标**：G11/G12/G13。
- **四臂**：`direct` / 冻结独立-RRR 方向 1 / **冻结条件-RRR 方向 1（全新）** / PhaseFormer-L（联合训练）。
  冻结臂的投影器由各 setting 的 **train split** 单独计算并冻结，投影发生在 `x_last` 中心化之后、
  线性层之前（沿用 E8 的 `set_projection_basis` 机制）。
- **关键对照逻辑**：`冻结独立` 明显差于 `冻结条件` ⇒ 支持"独立目标错位"；两者都差于联合 ⇒ 支持
  "冻结本身有害"。只有同时具备两臂才能区分（这正是 E8 无法归因的原因）。
- **H1 列**：复用 E10 表 5，并在表注写明判定口径（G13）。

### WP6 §4.6 负对照补做

| 子项 | 内容 | 现状 → 需补 |
|---|---|---|
| WP6-1 | 输入平滑（boxcar / causal EMA 各 2 档）在 **PhaseFormer-L** 上复测 | 探针 14/14 无改善 → 2 档 × 28 × 3 |
| WP6-2 | SVD 截断 vs 秩约束训练扩展到 28 setting（G15） | 已有 7 setting → 缺 21；依赖 WP3 的 `direct` 全秩 checkpoint |
| WP6-3 | 行 2（结构化坐标）、行 4（q=1/32） | 已存在，**不补做**，minipaper 保持 "—" |

---

## 6. 预注册判定门槛（G18；需用户裁定后写入 minipaper §4.0）

**这些数值必须在 WP3 开跑前冻结**，否则"冻结后盲测"的声明不成立。下述为**建议值，待裁定**：

| 参数 | 含义 | 建议值 | 理由 |
|---|---|---|---|
| `K`（改善 setting 数） | PhaseFormer-L 相对 Golden **MSE 与 MAE 双指标**均改善的 setting 数下限 | **14 / 28**（过半）为"可声明提升"最低线；**20 / 28** 为"强结论" | 与金标准 §4"默认只有 MSE 与 MAE 都低于金标准才称双指标提升"一致 |
| `R`（回退上限） | 任一 setting 的双指标回退上限 | **1.0%** | 与 E1 既用的 ±1% 判定带一致；且金标准仅 3 位小数，R 必须显著大于舍入级 |
| 稳定超过 | 是否计入"稳定" | 三 seed 均值低于 Golden 且样本 std 不跨过 Golden | 沿用既有严格标准 |

**报告规则**：`K` 与 `R` 一经冻结不得事后调整；若最终未达 `K`，须如实报告为"未达预注册门槛"，
不得改用其他统计口径重述。

---

## 7. 结果回写契约

每个工作包的产出**必须**回写到下表指定位置。这是本文件作为"结果回写目标"的具体含义。

| WP | 回写目标（数值） | 原始产物 | 登记动作 | 披露要求 |
|---|---|---|---|---|
| WP1 | minipaper **§4.3 表**（28 行）+ §4.3 三类图 | `research_runs/phaseformer_L_dimension_v1/`（二阶矩 `.npz`、capture CSV、`leading_direction` CSV） | agent-log 新条目 | train/validation 口径；与 §4.2 test 增益**不同源**；21 个新 setting 为 coverage 扩展，非 test-selection |
| WP2 | minipaper **§4.7 表** | `research_runs/phaseformer_L_levelstats_v1/`（28 行统计量 + ρ） | agent-log 新条目 | `τ̂` 定义须在 WP0 冻结；Weather 边界条件单独报告 |
| WP3 | minipaper **§4.2 主表**（28 行）+ 必答 (a)(b)(c) + 参数量/FLOPs 列 | `research_runs/phaseformer_L_main_v1/results.csv`（含 `setting`、`seed`、`variant`、参数量、`g` 均值） | agent-log 新条目 + README | Golden 为固定参照；matched rerun 仅协议诊断；任何 test 选择须显式标注 |
| WP4 | minipaper **§4.4 两表** | `research_runs/phaseformer_L_dissection_v1/`（`canonical_modes`、`semantic_alignment`、`cross_seed_alignment`、`intervention_results.csv` 720+ 行） | agent-log 新条目 | 支路自身与融合误差同时报告；同维对照须真正同维 |
| WP5 | minipaper **§4.5 表** | `research_runs/phaseformer_L_conditional_v1/`（四臂 + 投影器审计） | agent-log 新条目 | 冻结投影器为 train-only；H1 判定口径写入表注 |
| WP6 | minipaper **§4.6 表**的"本文补做"两列 | 复用 WP3 产物 | agent-log 新条目 | 行 2/行 4 保持 "—" |

**通用回写规则**：

1. **数值权威副本**始终是 `research_runs/` 下的原始 CSV/JSON；文档表格必须逐项对回原始产物
   （沿用 E2 的"计划存规则、报告存数值、互相引用"惯例）。
2. 每个 WP 完成后**追加**一条 `docs/agent-log.md` 条目（日期、任务摘要、主要文件、关键命令、
   验证结果、已知风险），不覆盖历史。
3. 凡涉及 test 的选择，须同时写入 `run.yaml`、agent-log 与 minipaper §5 披露。
4. minipaper §4.1 的数字**不得**因本计划而改写口径（`PhaseFormer_L_minipaper.md` 头部已声明）；
   若 WP0-4 复核推翻了某个先导数字，须在 minipaper 加勘误注并同步 agent-log。
5. 未完成的表项保留为空白，不得用推断值或单一指标改述填充。

---

## 8. 预算与并行

按既有实测折算：E8 = 54 runs / 6.39 GPU·h ≈ 0.118 GPU·h/run；E9 = 98 runs / 17.36 GPU·h ≈ 0.177 GPU·h/run。

| WP | 新增训练 | GPU 估算 | 备注 |
|---|---:|---:|---|
| WP0 | 0 | 0 | 纯代码与文档 |
| WP1 | 0 | 0（CPU） | 需分片以避开 13.9 GiB 峰值 |
| WP2 | 0 | 0（CPU） | ρ 列等 WP3 |
| WP3 | 504（588 − 复用 `phase_only`） | **60–90 GPU·h** | 6 卡并行 wall-clock 约 12–20 h |
| WP4 | 0 | 评估为主 | 100× 随机 RRR/cell 的零分布 |
| WP5 | 168（2 冻结臂 × 28 × 3） | **20–30 GPU·h** | 投影器为 train-only，无须训练 |
| WP6-1 | 168（2 档 × 28 × 3） | **20–30 GPU·h** | 依赖 WP3 |
| WP6-2 | 0 | 评估为主 | 依赖 WP3 的全秩 checkpoint |
| **合计** | **约 840 runs** | **约 100–150 GPU·h** | 未含重训与返工余量 |

**资源约束**：A800 机器的 GPU 6/7 被他人的 vLLM 服务长期占用（E8/E9 记录），可用为 **0–5 号共 6 卡**，
调度须沿用"一卡一 run、互不重叠"的既有做法，并避免 `num_workers` 过载
（E3 记录：8 个 `num_workers=0` 任务曾把 load average 推到约 300，速度降到约 1 epoch/50 min）。

---

## 9. 边界与披露

1. **本文件不产生任何提升声明**：PhaseFormer-L 属机制消融线（非保留结构），最终对外表述
   必须遵守 minipaper §5 与 `PhaseFormer_rank_capacity_and_data_property_report.md` §4.5 的
   "安全表述"清单。
2. **§4.1 的先导证据继续为 test-exposed**：本计划不改变其性质；主结论只能建立在 WP3 的冻结后
   盲测与 WP1/WP2 的 train/validation 分析上。
3. **7 → 28 的 coverage 扩展不等于削弱选择问题**：既有 7 个 setting 是 test-selected 的集合，
   WP1/WP2/WP6-2 的扩展部分应明确标注为"新增 coverage"，与既有 7 行在来源上区分。
4. **Electricity-336 类的内存与计算约束**：高维 setting（Electricity/Traffic 的 336/720）计算量占优
   （E10 记录 Electricity-336 占 7 setting 总计算量的 42%），预算与进度须显式考虑。
5. **不得把 `L-rank2` 的增量解释为秩-2 必要**：若增量落在 seed 噪声内，只能支持命题 2 的秩-1
   预测，不能反向宣称秩-2 无效。
6. **命题 1–2 目前是证明路线而非完整证明**（minipaper §2.3 自注）；本计划的实证独立于此，
   但论文投稿前 §2 的形式化工作仍需完成（不在本计划工作包内，另立任务）。

---

## 10. 执行记录（追加式，不覆盖历史）

| 日期 | WP | 状态 | 摘要 / 产物 | 结论 |
|---|---|---|---|---|
| 2026-09-18 | — | **登记** | 建立本文件；核对 agent-log 全部 151 条条目与 12 份被引文档，产出证据账本（E1–E13）、缺口确认表（G1–G18 全部确认存在）、登记不一致（D1–D6） | 待 WP0 启动 |

- **2026-09-18（追加）**：用户裁定不新增模型头。minipaper §3.4 已重写为只基于既有 preset（`weak_residual` /
  `rcrf_nlinear_plain` / `pooled_lowrank` / `original` + 训练集统计量开关），**WP0-2 作废**；WP3 的
  `L`/`L-fixed`/`L-rank2`/`L-nogate`/`L-mean` 7 变体 × 28 × 3 = 588 runs 作废，应按 minipaper §4.2 的变体行
  （`phase_only` / PhaseFormer-L 含开关 / always-on / L-q1/4 / L-q1/8 / L-rcrf / A1）与 24+4 setting 重排，
  既有同协议三 seed 格子经审计复用。D1/D2 已用 rsync 副本复核（见 agent-log 同日条目），待写入
  低秩 checkpoint 计划表 6 正文。本条为追加记录，未改动 §1–§9 正文。

- **2026-09-20（追加，仅补指针，不改写上文）**：
  1. **上一条 09-18 追加里的变体行清单已过期**：它写的是"`phase_only` / PhaseFormer-L **含开关** /
     **always-on** / L-q1/4 / L-q1/8 / L-rcrf / A1"（7 项）。而 minipaper 的 **D-5 裁定（2026-09-19）**
     明确 `s` **不进入模型**——PhaseFormer-L 就是"修正器恒定启用"的 `weak_residual`，
     故「含开关」与「always-on」**合并为一行**。**当前变体口径以 minipaper §3.4.2 的 D-5 修订与 §4.2 的表为准**：
     §4.2 的臂级变体表为 5 行描述性文字；产物侧 `variant_table.csv` 为 **6 个臂**
     （`phase_only` / `l_main` / `l_q1_4` / `l_q1_8` / `l_rcrf` / `a1`）。
  2. **上一条的"D1/D2 待写入低秩 checkpoint 计划表 6 正文"已完成**：2026-09-20 已在
     `docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 的表 6 正文登记那两个数字
     （`Semantic-drop` 支路变好 **52/72**、融合变差 **72/72**；`Semantic-only` 的 `Δfused MSE` 全格最大
     **+0.002665**、`q=1/8` 最大 **+0.000967**），并且**从该表自身的 720 行独立复算验证过**。
     minipaper 顶部的对应披露已同步改为"已登记"。
  3. **一处命名冲突，须留意**：本计划 §"登记不一致"用的是 **D1–D6**（那一节里 **D5 = 一处计数不一致**），
     与 minipaper 的 **D-5（决定：`s` 不进入模型）**是**两个不同的东西**。本文件的历史条目同时出现两者，
     阅读时按上下文区分；后续引用建议写作"**决定 D-5**"与"**登记不一致 D5**"。
