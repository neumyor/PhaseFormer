# PhaseFormer-L MiniPaper：§4 明确要求的实验 — 现状核对

> 性质：**只读核对**（2026-09-18）。范围严格限定为 `docs/PhaseFormer_L_minipaper.md`
> **§3.2–§3.4 与 §4 中明确写出的实验/表格**；不引入 minipaper 未要求的实验。
> 对每一项回答同一组问题：minipaper 要求什么 → 本地 `research_runs/` 里有什么 →
> 还缺什么。不运行训练、不读 test、不改任何数值口径。
>
> 与 `PhaseFormer_L_experiment_plan.md` 的关系：该文是 §4 的执行契约（G1–G18 缺口编号、
> WP0–WP6 工作包）。本文只按 minipaper 的表格逐项对账，不重复其工作包定义。

---

## 1. minipaper 明确要求补充的实验（逐项）

### 1.1 §4.2 主结果表（24 setting × 3 seed + Traffic 附录）

**minipaper 的明确要求**

- 协议（§4.0）：主表 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity × H96/192/336/720
  = **24 setting**；Traffic 4 格为探索性附录、**不进入判定**；full-train、最低 validation loss
  checkpoint、seeds 2021/2022/2023、每 checkpoint 只读一次 test。
- 表列：`Golden MSE/MAE` / `phase_only`（matched）/ `PhaseFormer-L`（含开关 `s`）/ `always-on` /
  `Δ vs phase_only` / `稳定超过 Golden` / 来源披露。
- 变体行（§4.2 第二张表，"全部为既有 preset"）：

| 行 | preset / 配置 | 新训规模（minipaper 原文） |
|---|---|---|
| `phase_only` | `original`（matched rerun） | 24 − 6 复用 = **18 setting × 3** |
| PhaseFormer-L（主） | `weak_residual`，`shared` 头，逐数据集开关 `s` | 24 − 7 复用 = **17 setting × 3**（`s=0` 的数据集不需训练） |
| always-on | 同上，`s≡1` | 与上行共享 run |
| L-q1/4、L-q1/8 | `pooled_lowrank`，`rank=H/4`、`H/8` | 6 数据集 × 4 H × 2 × 3 − 复用 |
| L-rcrf | `rcrf_nlinear_plain` | 24 × 3（可先做 H96/192） |
| A1 | `gold_combo_reliability_s2` | 补 H336/720 可选 |

- 必答问题：(a) ETTh2 四个 horizon 相对 `phase_only` 与 Golden 的差距是否收窄、是否达到 FITS 的
  引用数字；(b) 开关是否把 ETTh1/ETTm1 判为 `s=0`，若判为 `s=1` 则 always-on 在这两个数据集上的
  表现（如实报告）；(c) q=1/8 与 direct 的三 seed 差是否在 ±0.5% 内。
- §4.2 末：参数量列按 `metrics.csv` 的 `parameter_count` 口径报告，并**单列修正器参数**。

**本地现状**

| 要求 | 本地已有 | 缺 |
|---|---|---|
| `phase_only` 三 seed | **6 setting**：`research_runs/top2_direction_retention_v1/results.csv` 的 `arm=phase_only`（18 行 = 6 setting × 3 seed） | 18 setting × 3 |
| `direct`（q=1/8 判定的比较底色）三 seed | **6 setting**：`rank_sweep_2_multiseed_stage1_20260914_summary/audited_results.csv`（105 单元中的 18 格，均标记为复用自 `joint_lowrank_rank_sweep_v1`） | 18 setting × 3 |
| PhaseFormer-L（`weak_residual`+`shared`）三 seed | **0**（既有只有 E1/E2 的 7 setting 单 seed） | 全部 |
| `always-on` | 与主行共享 run | 依赖主行 |
| `L-q1/4`、`L-q1/8` | 6 setting × 3 seed × 2 档 = **36 格**（同上 `audited_results.csv`） | 其余 18 setting × 2 档 × 3 |
| `L-rcrf` | **0 run**（`rcrf_nlinear_plain` 已实现于 `src/models/phaseformer_presets.py:793`） | 24 × 3（或先 H96/192 共 12 × 3） |
| A1 | 12 格既有 | H336/720 为"可选"，minipaper 未强制 |
| FITS 引用数字（必答 (a)） | 仓库内无 FITS 数字 | 需外部引用并注明来源 |
| 修正器参数量单列 | `metrics.csv` 有全模型 `parameter_count`；`joint_lowrank_rank_sweep_plan.md` §4 表有 head 参数量 | 需在产物中单列修正器参数量 |

**需要指出的表内不一致**：§4.2 变体行写"24 − 6 复用"（`phase_only`）与"24 − 7 复用"
（PhaseFormer-L）；§4.0 的复用规则写"`weak_residual` direct **7 格**；`phase_only` **6 格**……
直接复用，不重训"。核对本地产物：三 seed 的 `phase_only` 与三 seed 的 `direct` **各只有 6 setting**
（各 18 格）；"7 格"对应的是 E1/E2 的**单 seed**结果。即 §4.0 与 §4.2 的复用数不一致，
且 7 格不等于三 seed 证据。另外 `top2_direction_retention_v1/reuse_audit.json` 的 `rejected`
列表显示同批 `weak_residual` 单 seed checkpoint 曾因 `gate_init`、`learning_rate` 不匹配被拒——
"直接复用"须附加逐项协议审计条件。

### 1.2 §4.3 维数表 + 预期图

**minipaper 的明确要求**

- 表（28 行）：`λ_1/Σλ`、`pred_dims_90`、`PR`、`b_1` 最佳模板（│cos│）、`a_1` vs 常值 │cos│、
  `used_var_share(1)`；§4.0 盲测边界规定机制分析只用 **train/validation**。
- 预期图：Scree 图；`b_1` 随 lag 的剖面与 `a_1` 随 horizon 的剖面**双面板**。

**本地现状**：`research_runs/lowrank_data_property_v2/` 有 `moments_*.npz` **恰 7 个**
（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336）、`leading_direction.csv` 7 行、
`optimal_rank_capture.csv`；缺 **21 行**（ETTh1×4、ETTh2-192/336、ETTm1×4、ETTm2-336/720、
Weather-336/720、Electricity-96/192/720、Traffic×4）；三类成图未生成。

**技术前置（核对 `scripts/analyze_optimal_lowrank_capture.py` 后确认）**：脚本内 `DATASETS`
字典只有 `ETTh2 / ETTm2 / Weather / Electricity` 四个键，**ETTh1、ETTm1、Traffic 无法直接运行**；
且 Electricity-336 曾在 Gram 组装阶段因 `(17344, 336, 321)` 需 **13.9 GiB** 而失败（这也是 §4.4
的解剖实验排除该 setting 的原因）。28 行扩表须先扩数据集键并实现分片/流式二阶矩。

### 1.3 §4.4 解剖表 + 干预表

**minipaper 的明确要求**

- 解剖表（3 seed）：PhaseFormer-L **与**低秩探针的 主模式输入组/解释率、输出组/解释率、
  修正能量份额、跨 seed `leading4` 重叠、稳定语义判定。
- 干预表（每 cell 10 臂，同时报告支路自身与融合误差）：`Semantic-only` / `Semantic-drop` /
  随机 95% 区间 / `PCA-drop` / **随机 RRR 子空间 drop（新增对照）** / 支路自身 Δ / 融合 Δ。
- minipaper 自注该新增对照用于区分"语义有效"与"任意同数量主方向有效"，指出既有结果中
  57/57 单元 `Semantic-drop` ≡ `PCA-drop`，需在 `evaluate_lowrank_semantic_interventions.py`
  增加一个臂（"分析侧代码，不涉及模型"）。

**本地现状**：本地**无**这些产物（`canonical_modes.csv`、`semantic_alignment.csv`、
`intervention_results.csv` 仅在服务器；`PhaseFormer_L_experiment_plan.md` §2.2 已标注
"不可（仅服务器）"）。已登记覆盖为 **6 setting × 3 seed × 4 压缩档 = 72 cell**、10 臂 × 72 =
**720 行**（minipaper 头部与 agent-log 2026-09-17），且 Electricity-336 被排除、Traffic/ETTh1/ETTm1
无记录。缺：(i) **随机 RRR 子空间 drop 臂**（minipaper 明确"此前缺失"）；(ii) **PhaseFormer-L
的那一列**（依赖 §4.2 主行产物）。

### 1.4 §4.5 条件性学习表

**minipaper 的明确要求**：列 = `direct` / 冻结独立-RRR 方向 1 / **冻结条件-RRR 方向 1** /
PhaseFormer-L（联合）/ H1：cond 距离 < indep 距离（seed 数）。表下写明预测：冻结独立方向退化；
**冻结条件性方向应明显好于独立方向（此对照此前未做）**；联合训练最好。

**本地现状**：冻结独立-RRR 臂已有 **6 setting × 3 seed**（E8，宏平均 +1.94%/+1.99%）；
H1 有 72 行（6 setting × 3 seed × 4 档，按"seed 内多数 rank"读法 5/6 setting 3/3 seed）。
缺 **冻结条件-RRR 方向 1 臂（0）**；若要 24 setting 覆盖，H1 与冻结独立臂都需扩展。

### 1.5 §4.6 负对照汇总

**minipaper 明确要求"本文补做"的**：行 1 输入平滑在 **PhaseFormer-L** 上复测 **2 档**；
行 3 SVD 截断 vs 秩约束训练扩到**全 28 setting**；行 5 **边界消融 `pooled_lowrank`
rank∈{1,2}**（6 setting × 3 seed，预注册"预期退化、用于量化残项 ε，不影响主张"）；
行 2、行 4 标注"—"不补做。

**本地现状**：行 1 的既有证据是**探针**上 14/14 组合无改善（E4+E5，单 seed）→ PhaseFormer-L 上 0；
行 3 已有 **7 setting**（minipaper 行内只写了 Electricity-336 一例，其余 6 个差 1–3%，
Electricity-336 r=10 差 29%）→ 缺 21；行 5 rank∈{1,2} **完全不存在**（既有最浅档为 q=1/32
→ r=3/6/10/22）→ 6 × 2 × 3 = 36 runs。

### 1.6 §4.7 预测力表

**minipaper 的明确要求**：对 28 个 setting 计算**训练集**上的 `cycle_level_std`、
`last_cycle_shift`、电平自相关时间 `τ̂`，与 §4.2 的 ΔMSE 及门值 `g` 求 **Spearman ρ**；
预期符号 +、+、与学到的 EMA τ 正相关。表下说明 Weather 类弱周期数据若主模式转向曲率/慢趋势，
应作为边界条件单独报告。

**本地现状**：**0**。三个统计量从未计算；两列 ρ 依赖 §4.2 的 ΔMSE 与 `g`。

### 1.7 §3.4.3 开关（作为 §4.2 的一列，非独立实验）

minipaper 明确：`s(dataset) = 1[ν_train(dataset) > ν*]`，`ν` 定义与阈值 `ν*` **在 §4.7 冻结**
（由 6 个已知 setting 的训练集统计量拟合，不看 test）；`s=0` 时退化回 `original`，因此这些
数据集上"按构造不劣于"matched rerun；§4.2 同时报告 `s≡1` 的 always-on 列。
**本地现状**：`ν*` 与 `ν` 定义未冻结（依赖 §4.7）。

---

## 2. 汇总：minipaper 明确要求、目前完全为 0 的项

| 项 | minipaper 位置 | 现状 |
|---|---|---|
| PhaseFormer-L 主行（含开关） | §4.2 变体行 | 0 |
| always-on 列 | §4.2 表头 | 0（依赖主行） |
| `L-rcrf` 行 | §4.2 变体行 | 0 run（preset 已实现） |
| `L-q1/4`、`L-q1/8` 在新 setting 上 | §4.2 变体行 | 6/24 setting |
| 21 行 §4.3 维数 + 3 类图 | §4.3 | 7/28 行，图 0 |
| PhaseFormer-L 的解剖列 | §4.4 解剖表 | 0 |
| 随机 RRR 子空间 drop 臂 | §4.4 干预表末列 | 0 |
| 冻结条件-RRR 方向 1 臂 | §4.5 | 0 |
| 平滑在 PhaseFormer-L 上复测 2 档 | §4.6 行 1 | 0 |
| SVD 截断扩到 28 setting | §4.6 行 3 | 7/28 |
| 边界消融 rank∈{1,2} | §4.6 行 5 | 0 |
| §4.7 全表（3 统计量 + 2 列 ρ） | §4.7 | 0 |
| 修正器参数量单列 | §4.2 末 | 0（全模型 `parameter_count` 有） |
| FITS 引用数字（必答 (a)） | §4.2 | 仓库内无源 |
| ETTh2 四 horizon 对比（必答 (a)） | §4.2 | 依赖主行 |
| ETTh1/ETTm1 的 `s` 判定与 always-on（必答 (b)） | §4.2 | 依赖主行 |
| q=1/8 与 direct 的 ±0.5% 判定（必答 (c)） | §4.2 | 6 setting 可先算，其余依赖新训 |

---

## 3. 需要先在 minipaper 内裁定的三处（不动工理由）

1. **§4.0 与 §4.2 的复用数不一致**：§4.0 写 `weak_residual` direct 7 格 / `phase_only` 6 格，
   §4.2 写 24 − 6 / 24 − 7。本地三 seed 证据是 6 + 6。裁定后才知主表新训 17 还是 18 个 setting。
2. **§4.0 的判定门槛仍标"建议值，待用户裁定后冻结"**（主张 A ≤1.0%、主张 B ≥3/4、
   主张 C 不设门槛、主张 D |Δ|≤0.5%）；minipaper 自注须在开跑前冻结，否则"冻结后盲测"声明不成立。
3. **§4.6 行 3 的现状被低估**（已覆盖 7 setting，非 Electricity-336 一例），属表内事实性表述，
   需先改再作为施工依据。

---

## 4. 与执行契约文档的对应

本文每项对应 `PhaseFormer_L_experiment_plan.md` §4.1 的缺口编号与 §5 的工作包：
§4.2 → G1/G2/G3/G4/G5/G6 + WP3；§4.3 → G7/G8 + WP1；§4.4 → G9/G10 + WP4；
§4.5 → G11/G12/G13 + WP5；§4.6 → G14/G15/G16 + WP6；§4.7 → G17/G18 + WP2；
§3.4.3 开关 → G5。本文不替代该文档的执行契约地位。

---

## 5. 关联文档

- 核对对象：`PhaseFormer_L_minipaper.md` §3.2–§3.4、§4、§5
- 执行契约：`PhaseFormer_L_experiment_plan.md`
- 被核对的本地产物：`research_runs/{top2_direction_retention_v1, rank_sweep_2_multiseed_stage1_20260914_summary, lowrank_data_property_v1, lowrank_data_property_v2, joint_lowrank_rank_sweep_v1, smooth_ratio_sweep_v1, causal_ema_smooth_sweep_v1}`
- 覆盖与数字来源的登记文档：`PhaseFormer_rank_capacity_and_data_property_report.md`、
  `PhaseFormer_lowrank_checkpoint_information_analysis_plan.md`、`PhaseFormer_rank_sweep_conditioned_experiment.md`、
  `PhaseFormer_top2_direction_retention_summary.md`、`PhaseFormer_joint_lowrank_rank_sweep_plan.md`
