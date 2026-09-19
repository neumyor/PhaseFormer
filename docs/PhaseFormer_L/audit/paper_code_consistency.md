# 论文 ↔ 实现 一致性核对：§4 的每个冻结数字都对到了代码常量

> 方法：把 minipaper §4 里**写下的**数字/结构与代码中的常量逐项对照（AST 抽取，不靠记忆）
> 结论：**除一处过时表述外全部一致**；该处已修正并记录在下文 §3。
> 执行：2026-09-20（本地 AST 抽取 + 服务器实测值）

## 1. 为什么要单独做这一项

minipaper 是**交付物本身**。若它写下的门槛与实现里跑的常量不一致，
那么四个主张的判定就会**在无人察觉的情况下失去意义**——例如论文说回退门槛是 1.0%
而代码里是 2.0%，表照样填满、结论照样打印，但结论已经不对应论文所声明的东西。
这类漂移不会被任何单元测试发现（代码自洽、论文自洽），只能靠**两边对照**。

## 2. 逐项对照结果

### 2.1 §4.0 的四个冻结门槛

| §4.0 写法 | 实现常量 | 一致？ |
|---|---|---|
| 主张 A：24 setting 上无双指标回退超过 **1.0%** | `e14_writeback.REGRESSION_BOUND_PCT = 1.0` | ✓ |
| 主张 B：s=1 数据集上 **≥ 3/4** 的 setting 双指标优于 `phase_only` | `e14_writeback.CLAIM_B_FRACTION = 0.75` | ✓ |
| 主张 C：三 seed 均值 + 样本 std < Golden，**不设最低数目门槛** | 回填同时输出"均值+std"与"均值"两个计数（两份治理文档定义不同），无门槛判断 | ✓ |
| 主张 D：q=1/8 相对 direct 的宏平均 │ΔMSE│/│ΔMAE│ ≤ **0.5%** | `e14_writeback.CLAIM_D_BOUND_PCT = 0.5` | ✓ |

### 2.2 §4.2–§4.7 的结构数字

| 论文写法 | 实现 | 一致？ |
|---|---|---|
| §4.2「**24 setting** × 3 seed；Traffic 附录」 | 主表 28 行 = **24 core + 4 appendix**（由 `is_traffic_appendix` 标记）；6 个臂行（`MAIN_ARMS` 5 + `a1`） | ✓ |
| §4.3「**7 数据集**」、表 **28 行** | E15 已完成：28 行 = 7 数据集 × 4 horizon | ✓ |
| §4.4 解剖表 **21 行** | 21 = 3 模型（dense/q1·4/q1·8）× 7 setting | ✓ |
| §4.4 干预表「每 cell 10 臂」 | `build_arm_plan`：**8 个基础臂 + 2 个同维 `PCA-matched`** = 10 个登记臂，再按可用基向量追加 `Independent-RRR-only`/`Conditional-RRR-only`/**`RandomRRR-drop`** → 实测 **11–12 臂** | **表述过时，已修正**（见 §3） |
| §4.4 要求的随机 RRR 子空间对照 | `build_arm_plan` 末尾 `("RandomRRR-drop", bases["rrr_reference"], "drop")`；`e16_writeback.expected_arms = 11` | ✓ |
| §4.5「四臂」、**7 个 setting** | `e17_conditional.SETTINGS = 7`、`ARMS`/`FOUR_ARM_ORDER = 4` | ✓ |
| §4.6 平滑 **42 run** + 边界 **36 run** | 42 = 21 setting × 2 档（`causal_ema_mid`/`causal_ema_max`）；36 = 18 × 2 秩 | ✓ |
| §4.6 行 3 覆盖 **28 setting** | `e18_svd_truncation` 的 28 setting 表；冒烟实测 `rows=1`（限一格） | ✓ |
| §4.6 负对照表 **1–5 行** | 回填实测 `negative_table rows = ['1','2','3','4','5']` | ✓ |
| §4.7 三个候选统计量 | `e19_predictive_power.STATISTICS = ("cycle_level_std","last_cycle_shift","tau_hat_steps")` | ✓ |
| §4.7 冻结 **ν\* = 57.35** | `e14_writeback.NU_STAR = 57.35` **且** `e19_predictive_power.NU_STAR = 57.35`（两处独立声明，值相同） | ✓ |

## 3. 找到并修正的一处过时表述

**原文**（§4.4）：

```text
干预表（每 cell 10 臂；同时报告支路自身与融合误差）：
```

**问题**：同一段的表头里已经列出了"**随机 RRR 子空间 drop**（新增对照）"，
而紧随其后的正文也写明"新增'随机 RRR 子空间'对照用于区分…"——
即这篇论文自己**要求**了这个对照，但括号里的臂数没有随之更新。

**证据链**（三处独立一致，均来自实现而非记忆）：

1. `e16_dissection.build_arm_plan`：基础 8 臂（`Original`、`Semantic-only`、`Semantic-drop`、
   `Semantic8-only`、`Semantic8-drop`、`Bias-off`、`PCA-only`、`PCA-drop`）
   ＋ 同维 `PCA-matched-only`/`PCA-matched-drop` = **10 个登记臂**；
   再按可用基向量追加 `Independent-RRR-only`（需 `pool`）、`Conditional-RRR-only`（需 `conditional`）、
   **`RandomRRR-drop`**（需 `rrr_reference`）。
2. 真实产物实测：E16 冒烟回填报 `arms_per_cell_observed: [11, 12]`、
   `cells_with_fewer_arms: 0`——即 11 是**下界**、12 也合法，臂数是**按 cell 动态**的。
3. 回填工具的覆盖判据 `e16_writeback.expected_arms = 11`，与 §6.2 判据表的
   "每 cell 一行 × 11 臂（10 既有 + `RandomRRR-drop`）"一致。

**修正后**：把括号改成"每 cell **10 个登记臂**（列出 8+2）…之上再按该 cell 的可用基向量追加
`Independent-RRR-only`/`Conditional-RRR-only`/`RandomRRR-drop`，故实际为 **11–12 臂（11 为下界）**"。

**未改动**：文中另一处"10 臂"（开头对既有 rsync 副本的描述"720 行 = 72 cell × 10 臂"）
描述的是**既有历史数据**，那里 10 臂是对的，故保持原样——**只改过时的那一处，不做批量替换**。

## 4. 边界（本核对**没有**覆盖什么）

* 它核对的是**数字与结构**（门槛、计数、行/列数），**不**核对散文式论断是否正确；
* 它不验证任何结果的**数值**——那要等实验跑完后由各阶段的阶段 5 审校按冻结判据判定；
* 它依赖 AST 抽取与关键字匹配：**动态构造**的常量（如 `SMOOTH_LEVELS` 里引用
  `CAUSAL_EMA_ALPHA` 名）无法用 `literal_eval` 取到，需另法（本轮以"TUPLE 有 2 个元素 +
  两档的 level 名"另行确认）。**"我抽不到" ≠ "它不存在"**——这一点在
  `traceability_matrix.md` §5.2 已有一次自伤记录，此处再次适用。
