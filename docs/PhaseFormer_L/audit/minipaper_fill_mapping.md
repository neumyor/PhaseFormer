# minipaper §4 回填映射：**哪些表能直接复制、哪些必须组合**

> 目的：在**回填之前**把每张表的"论文列 ↔ 产物列"关系定下来，
> 因为"以为每个工具都会吐出 markdown"正是回填阶段最容易犯的错。
> 核对方式：逐一读**论文的表头**与**生产者的格式串**（都取自源码，不靠记忆）。
> 执行：2026-09-20（HEAD `0c073500` 附近）

## 0. 一句话结论

**八处需要填充的表位中，四处可以逐格直接复制、四处必须组合。**
两类在回填时的风险与校验手段完全不同，故必须在动手前分清。

## 1. 逐表映射

### 1.1 §4.2 主表（28 行 × 10 列）—— **直接复制** ✓

| 论文列 | 产物列 |
|---|---|
| Dataset / H | `dataset` / `horizon`（`%d`） |
| Golden MSE/MAE | `golden_mse:.3f` / `golden_mae:.3f` |
| `phase_only`（matched） | `phase_only_mse/mae`（`fmt` 成对） |
| PhaseFormer-L（恒定启用） | `l_main_mse/mae` |
| Δ vs phase_only | 预算好的 delta 串 |
| `g` 均值 | gate 均值 |
| 诊断 `s` | `diagnostic_s` |
| 稳定超过 Golden | `l_main_stable_beyond_golden` → `✓`/`✗` |
| 来源/披露 | `provenance_note` |

产物：`main_table.md`（**28 个数据行、无表头、无分隔线**）。
生成点 `e14_writeback.py:678-681`，格式串 `| %s | %d | %s/%s | %s | %s | %s | %s | %s | %s | %s |`。
**论文表头与之一一对应，顺序相同** → 可直接复制，且可用**字符串比较**校验（已实现并校准：见
`paper_code_consistency.md` §8）。

### 1.2 §4.2 的**臂级变体行**（4 行）—— **必须组合** ⚠

`e14_writeback` **只写 `variant_table.csv`，不写 `.md`**（对 `variant_table` 的处理见
`e14_writeback.py:9` 与 `:705`）。故论文里那 4 行（`weak_residual` shared；L-q1/4 与 L-q1/8；
`rcrf_nlinear_plain`；`gold_combo_reliability_s2`）需从 CSV 的**6 个臂行**组合而成——
其中 `l_q1_4` 与 `l_q1_8` 在论文里**合并为一行**。**这是组合，不是复制。**

### 1.3 §4.3 维数表（28 行 × 8 列）—— **映射**（已完成）✓

已回填并校验 168 格；映射见 `paper_code_consistency.md` §6.3。
注意论文把 `b_1` 模板与 `│cos│` 合成一格（`exp_tau=24` → `exp τ=24 (0.737)`）。

### 1.4 §4.4 解剖表（21 行 × 8 列）—— **必须组合** ⚠

`e16_writeback` **只写 `intervention_table_44.md`，不写解剖表的 markdown**（`e16_writeback.py:366` 是唯一的 `.md` 写出点）。故 8 列需从 `dissection_table_44.csv` 的 22+ 列组合，其中至少三处是"多列合一格"：

| 论文列 | 来源（组合） |
|---|---|
| 模型 | `model` |
| Dataset / H | `dataset` / `horizon` |
| **主模式输入组 / 解释率** | `input_group_label` **＋** `input_group_explanation` |
| **主模式输出组 / 解释率** | `output_group_label` **＋** `output_group_explanation` |
| 修正能量份额 | `correction_energy_share` |
| **跨 seed `leading4` 重叠** | `leading4_input_overlap` **＋** `leading4_output_overlap`（论文一格，产物两列） |
| 稳定语义判定 | `stable_semantics_verdict` 一类审计列 |

**组合方式需要先定下来再填**（例如两列如何并列、保留几位），否则同一张表里会出现两种写法。

### 1.5 §4.4 干预表（21 行 × 10 列）—— **直接复制、但要去掉第 1 列** ⚠→✓

产物 `intervention_table_44.md` 的行是 **11 格**：

```text
模型 | Dataset | H | q/r | Semantic-only | Semantic-drop | 随机带 | PCA-drop | 随机RRR带 | 支路自身Δ | 融合Δ
```

论文是 **10 列**（**没有**独立的"模型"列）。核对确认这不丢信息：
论文的 `q/r` 列本身就是 `dense（r=H）` / `q=1/4（r=24）` / `q=1/8（r=12）`，
**模型身份已含在其中**。故映射为：

```text
产物行[1:]  ==  论文行[0:]        # 丢掉产物行首格（模型），其余 10 格逐格相同
```

即"**去掉首列后可直接复制**"，校验仍可用字符串比较（只是先切掉一格）。

### 1.6 §4.5 四臂表（7 行 × 7 列）—— **直接复制** ✓

产物 `conditional_table.md`（`e17_writeback.py:188`，
`| %s | %d | %s | %s | %s | %s | %s |` = 7 格）与论文表头**列序逐一对应**：
`dataset | horizon | direct | 冻结独立 | 冻结条件 | joint | H1`。
**注意产物文件含表头与分隔线，复制时要跳过前两行。**

### 1.7 §4.6 负对照表（5 行 × 5 列）—— **直接复制** ✓

产物 `negative_table.md`（`e18_writeback.py:312`，`| %s | %s | %s | %s | %s |` = 5 格）
与论文表头逐一对应：`operation | target | scope | existing | addendum`
↔ `操作 | 作用对象 | 口径 | 结果（既有，test-exposed） | 本文补做`。
**同样含表头与分隔线，需跳过。**

### 1.8 §4.7 预测力表（3 行 × 4 列）—— **必须组合** ⚠

论文列：`统计量 | 与 ΔMSE 的 Spearman ρ | 与 g 的 ρ | 预期符号`（最后一列**已预填**）。
两个 ρ 列来自 `predictive_power_summary.json`（**不是** CSV）：

```text
predictive_power.spearman[f"{stat}_vs_delta_mse_pct"]["rho"]   -> 与 ΔMSE 的 ρ
predictive_power.spearman[f"{stat}_vs_gate_value"]["rho"]      -> 与 g 的 ρ
```

行序为 `STATISTICS = (cycle_level_std, last_cycle_shift, tau_hat_steps)`。
键名由 `e19_predictive_power.py:187-190` 生成，**不是**我推测的。

## 2. 这个区分对**校验**意味着什么

| 类别 | 表 | 可用校验 |
|---|---|---|
| **直接复制** | §4.2 主表、§4.4 干预表（去首列）、§4.5、§4.6 | **字符串比较**——最强：连列映射与中文披露列一起验。§4.2 已实现并五类校准 |
| **必须组合** | §4.2 变体行、§4.4 解剖表、§4.7 | 需要**映射感知**的比较（数值+精度，或显式键映射）；每节**填完当轮校准** |

**因此**：把"直接复制"的四张表用字符串比较验、把"必须组合"的三处用各自的映射比较验——
这条分工现在写下来了，回填时就不会出现"以为能直接粘、结果粘错了列"的情况。

## 3. 边界

* 本节只定**列映射**，不定**数值**是否正确——数值由各阶段 5 审校按冻结判据判定；
* "必须组合"的三处，**组合的具体写法（如何并列两列、保留几位小数）尚未决定**，
  需在回填时确定并写进本节，再据此写校验；
* 映射取自源码中的**格式串与生成点行号**，若这些工具被改动，本节须同步复核
  （三处"必须组合"尤其容易在工具改动后失效）。
