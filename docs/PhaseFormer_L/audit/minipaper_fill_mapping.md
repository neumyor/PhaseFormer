# minipaper §4 回填映射：**哪些表能直接复制、哪些必须组合**

> 目的：在**回填之前**把每张表的"论文列 ↔ 产物列"关系定下来，
> 因为"以为每个工具都会吐出 markdown"正是回填阶段最容易犯的错。
> 核对方式：逐一读**论文的表头**与**生产者的格式串**（都取自源码，不靠记忆）。
> 执行：2026-09-20（HEAD `0c073500` 附近）

## 0. 一句话结论

**§4 共有 10 张表；其中 3 张已经完整（§4.3、§4.2 的臂级变体表、§4.7 的第二张表），
7 张需要处理。** 需要处理的表里，**可直接复制**与**必须组合**两类在回填时的风险与校验手段完全不同，
故必须在动手前分清。

> **本节的更正（2026-09-20，填表前逐格清点后）**：初版把 §4.2 的臂级变体表列为"必须组合"，
> 并在 §4.6 那里写成"填空"。逐格清点后确认两处都不准确：**变体表 4 列 × 5 行全部已填**
> （它是"行 / preset / 作用 / 新训规模"的说明表，不来自 `variant_table.csv`）；
> **§4.6 的五列也全部已填**（第五列现在是**计划文字**，不是空）。详见 §1.2、§1.7 与 §1.9。

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

**关键操作细节：28 行都必须整行用产物替换（含已预填的格），不是只填空格。**
论文里已预填两类格子：主行的 `Dataset/H/Golden`（3 格），**Traffic 行还多预填 `来源/披露`**
（"探索性附录，不进入判定"）。两处都已核对**不会冲突**：

* **Golden 列**：28 格与 `PhaseFormer_gold_standard.md` 的 28 行**逐格一致、0 处不符**（2026-09-20 核验），
  故它会与产物 `f"{golden_mse:.3f}/{golden_mae:.3f}"` 的渲染相同 ✓；
* **Traffic 的 `来源/披露`**：产物的 `provenance_note` 是**先写"探索性附录，不进入判定"、
  再追加逐臂复用/新训披露**（`e14_writeback.py` 的 `notes` 构造：`if is_traffic_appendix:` 在臂循环**之前**）。
  即论文现有文字是产物字符串的**前缀** ✓ —— 所以**整行替换不会丢掉它**，反而补上了缺失的逐臂披露。

**反过来说**：若"只填 8 个空格、保留 Traffic 现有的短注"，那 4 行会与产物**不一致**，
校验器会报 4 处 MISMATCH —— 那是**填法错了**，不是校验器太严。**故规定：整行替换。**

### 1.2 §4.2 的**臂级变体表** —— **已完整，无需填充** ✓（初版此处有误，已更正）

该表的表头是 `| 行 | preset / 配置 | 作用 | 新训规模 |`——**是说明表，不是结果表**：
逐格清点得 **4 列 × 5 行 = 20 格，空格 0**。它描述"哪些臂、用什么 preset、进哪条主张、
新训多少格"，内容来自排期与复用范围（已核对：`phase_only` 18×3、PhaseFormer-L 17×3、
L-q1/4 与 L-q1/8 各 17×3、L-rcrf 28×3=84、A1 24×3=72）。

**因此它不来自 `variant_table.csv`**：那个 CSV 装的是逐臂**宏平均**
（`mean_mse`、`macro_delta_mse_pct`、`macro_delta_mse_pct_main24`、
`total_params_per_horizon`、`params_constant_across_seeds` 等），
服务的是 §4.2 主表的 Δ 列与正文表述，**不是**这张表。
（`e14_writeback` 不写 `variant_table.md` 这一事实仍然成立，见 `:9`/`:705`，
只是它对本表的填充没有影响。）

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

**组合方式需要先定下来再填**，否则同一张表里会出现两种写法。

**这一处与 §4.6 不同：**§4.6 我能从回填工具**硬编码的字符串**反推出设计意图（见 §1.7），
而这里**没有可反推的 ground truth**——`e16_writeback` 根本不渲染这张表。
更彻底地查过：§4.4 的正文在表格之前**只有标题、没有任何说明**，
表下注也只解释语义（"只用 validation 划分"、"跨 seed leading4 重叠按 4 维主子空间两两重叠计"），
**不规定单元格写法**。故这是**最后一处真正开放的格式问题**。

**建议的写法**（依据是 CSV 里实际可用的列名与含义，而非凭空规定）：

| 论文列 | 建议渲染 | 依据列 |
|---|---|---|
| 主模式输入组 / 解释率 | `{input_group_label}（{input_group_explanation:.2f}）` | `input_group_label`、`input_group_explanation` |
| 主模式输出组 / 解释率 | `{output_group_label}（{output_group_explanation:.2f}）` | `output_group_label`、`output_group_explanation` |
| 修正能量份额 | `{correction_energy_share:.3f}` | `correction_energy_share` |
| 跨 seed `leading4` 重叠 | `in {leading4_input_overlap:.2f} / out {leading4_output_overlap:.2f}` | `leading4_input_overlap`、`leading4_output_overlap` |
| 稳定语义判定 | 取判定列（`stable_semantics_verdict` 一类） | — |

**但两名小数/写法仍属表述取舍**，且**只有 E16 真正跑完后才能看到取值分布**
（例如解释率是 0–1 小数还是 0–100 百分数、`input_group_label` 的中文长度是否撑破表格）。
故建议：**E16 跑完后先打印几行真实取值，再据实定写法并回写本节**，
而不是现在把两位小数写死。**这样"最后一处开放格式"就有了确定的收口时点。**

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
该校验器**已实现并五类校准**（`paper_code_consistency.md` §10，正确配对 210/210）。

> **⚠️ 与 §1.1 同一条规则：必须整行替换（含已预填的 `q/r` 格）。**
> 论文的 **dense 行**把 `q/r` 预填为通用写法 **`dense（r=H）`**，而产物写**具体值**
> `dense（r=96）`（低秩行本就是具体值，如 `q=1/4（r=24）`）。
> 故若"只填 7 个空格、保留 `r=H`"，那 7 个 dense 行会与产物不一致而被报 MISMATCH。

### 1.6 §4.5 四臂表（7 行 × 7 列）—— **6 列直接复制＋第 7 列必须聚合** ⚠

产物 `conditional_table.md`（`e17_writeback.py:188`，
`| %s | %d | %s | %s | %s | %s | %s |` = 7 格）与论文表头**列序逐一对应**：
`dataset | horizon | direct | 冻结独立 | 冻结条件 | joint | H1`。
**注意产物文件含表头与分隔线，复制时要跳过前两行。**

**但第 7 列（H1）不是直接复制**——本轮逐字段核对时发现一处口径不一致：

| 项 | 内容 |
|---|---|
| 论文表头 | `H1：cond 距离 < indep 距离（**seed 数**）` → 要求一格**种子计数**（形如 `3/3`） |
| 产物该格 | `h1_seed_majority`，取值是 **`true` / `false` / `evidence_missing`**（逐个 seed 的判定，见 `e17_conditional.py:972-984`） |
| 可用的底层量 | 结果 CSV 里另有 `h1_ranks`（该 setting/seed 的 rank 行总数）与 `h1_ranks_supporting`（其中支持 H1 的行数）——但它们是**rank 级**计数，**不是 seed 级** |

即：直接把产物那格粘进去，会得到 `true` 而表头写着"（seed 数）"——**表头与内容对不上**。

**正解（有据可依）**：按 setting 把 3 个 seed 的 `h1_cond_gt_indep_seed_majority`
**聚合成种子计数**——数其中 `"true"` 的个数，渲染成 `N/3`；全为 `evidence_missing` 时写 `evidence_missing`。
这与本轮之前独立做过的 H1 汇总**完全一致**（当时报的是
"ETTh2-96/720、ETTm2-96/192 = **3/3 seed**；Weather-96 = **2/3**；Weather-192 = **0/3**；
Electricity-336 = `evidence_missing`"——那些数正是按 seed 分组数出来的）。

**故 §4.5 的映射修正为**：前 6 列直接复制；第 7 列由结果 CSV 按 setting 聚合 3 个 seed 得到，
并在回填时用**同一聚合口径**校验（不是与 `conditional_table.md` 的该格比字符串）。
该校验器**已实现并五类校准**（`paper_code_consistency.md` §9）。

> **⚠️ 回填该表时的操作警告（已实测踩到）**：`§4.5 的行文本` 是
> `§4.4 解剖表行文本` 的**后缀子串**（前者 7 格、后者 8 格，前者的内容正好等于后者去掉首格的部分）。
> 因此**任何按子串的 `replace` 都会命中文档中更早的 §4.4 行**，静默改坏 §4.4 而 §4.5 未变。
> **必须按行、且限定在 §4.5 的行区间内定位。**

### 1.7 §4.6 负对照表（5 行 × 5 列）—— **只填第五列**（前四列保持论文原文）✓

产物 `negative_table.md`（`e18_writeback.py:312`，`| %s | %s | %s | %s | %s |` = 5 格）
与论文表头逐一对应：`operation | target | scope | existing | addendum`
↔ `操作 | 作用对象 | 口径 | 结果（既有，test-exposed） | 本文补做`。
**同样含表头与分隔线，需跳过。**

**要点不在"填空"而在"替换"**：该表 **5 列 × 5 行 = 25 格、空格 0**；第五列现在装的是**计划文字**。

**我上一轮把它标成"需要决策"，本轮已用证据把它定下来**——把论文的「结果（既有）」列
与 `e18_writeback` 里**硬编码的 `existing` 参数**逐格对照：

| 行 | 论文「结果（既有）」 | 回填工具传入的 `existing` | 判定 |
|---|---|---|---|
| 1 | 14/14 组合无一改善，越平滑越差 | 同左（`summarise(row1, ...)`） | **逐字相同** |
| 2 | 5/5 双指标退化 1.6%–3.0%；同参数量时间轴对照仅 −0.30%/−0.52% | `KEPT_AS_DASH[0][3]` | **逐字相同** |
| 3 | 截断 +29%，训练 +0.7% | `截断 +29%，训练 +0.7%（Electricity-336, r=10）` | 论文是工具串的**前缀**（工具**多给了 setting 披露**） |
| 4 | 保留 92.4%–101.9% 可实现价值 | `KEPT_AS_DASH[1][3]` | **逐字相同** |
| 5 | 无 | `summarise(row5, "无")` | **逐字相同** |

**但"第五列逐字相同"不足以上升为"整表重建"**——本轮把**全部五行的前三列**也逐格对照后，
该推断被推翻：

| 行 | 操作 | 作用对象 | 口径 | 判定 |
|---|---|---|---|---|
| 2 | 结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离） | 支路参数化 | 4 setting | **逐字相同** ✓ |
| 3 | SVD 截断 vs 秩约束训练 | 支路权重 | Electricity-336 r=10 | **逐字相同** ✓ |
| 4 | 联合低秩训练 q=1/32 | 支路容量 | 7 setting | **逐字相同** ✓ |
| 1 | 论文 `输入平滑（boxcar / causal EMA，各 5 档）` | 支路输入 | 7 setting | **「操作」不同**（产物写 `输入平滑（causal EMA 两个强度）`） |
| 5 | 论文 `**边界消融：\`pooled_lowrank\` rank∈{1,2}**` | 论文 `支路容量（网格之外）` | 论文 `6 setting × 3 seed` | **三列都不同**（产物：`边界消融：pooled_lowrank 绝对秩 rank∈{1,2}` / `支路容量（低秩网格之外）` / `6 setting × 3 seed × 2 rank`） |

**3/5 行逐字相同、2/5 行措辞不同**。而行 1 的「操作」明确写"boxcar / causal EMA，各 5 档"——
那是在描述**先导实验**（即第 3、4 列所报告的既有实验）的算子网格，
**整行替换会把这段准确的既有描述改写掉**。

**修正后的回填方式**：

> **只替换第五列（本文补做）**，取产物的 `addendum`；**前四列保持论文原文**。
> 被顶掉的**计划/规模文字**（行 1 的 `42 runs`、行 5 的 `36 runs`、以及行 3 的 E11 口径差异）
> **移入表下注**——其中行 3 那条**已先行搬迁**（见 §1.7 前文与 `paper_code_consistency.md` §11）。

校验器据此实现：**第五列严格比较**，前四列不同则记 `INFO`（允许不同，理由写在 docstring），
并已五类校准（`paper_code_consistency.md` §11.2）。

### 1.8 §4.7 预测力表（3 行 × 4 列）—— **必须组合** ⚠

论文列：`统计量 | 与 ΔMSE 的 Spearman ρ | 与 g 的 ρ | 预期符号`（最后一列**已预填**）。
两个 ρ 列来自 `predictive_power_summary.json`（**不是** CSV）：

```text
predictive_power.spearman[f"{stat}_vs_delta_mse_pct"]["rho"]   -> 与 ΔMSE 的 ρ
predictive_power.spearman[f"{stat}_vs_gate_value"]["rho"]      -> 与 g 的 ρ
```

行序为 `STATISTICS = (cycle_level_std, last_cycle_shift, tau_hat_steps)`。
键名由 `e19_predictive_power.py:187-190` 生成，**不是**我推测的。

**本节的映射已在真实产物上核验**（2026-09-20）：用一个 schema 精确的合成结果表
（28 setting、`l_main`/`phase_only` × 3 seed = 168 行，统计量用**真实**的
`level_statistics.csv`）跑真实工具，观测到：

```text
predictive_power keys: ['n_settings', 'scope', 'spearman']
spearman entries: 6            # = 3 个统计量 × 2 个后缀
  cycle_level_std_vs_delta_mse_pct    present=True   (rho key exists)
  cycle_level_std_vs_gate_value       present=True
  last_cycle_shift_vs_delta_mse_pct   present=True
  last_cycle_shift_vs_gate_value      present=True
  tau_hat_steps_vs_delta_mse_pct      present=True
  tau_hat_steps_vs_gate_value         present=True
```

即**六条路径全部存在且都带 `rho` 键**——本节映射由"读源码推出"升级为"在真实产物上实测"。
（该次运行用的是合成结果，故 `rho` 全为 `nan`（合成值秩退化）；**验证的是结构、不是数值**。
这也再次印证前面记录过的边界：**summary 里可能出现裸 `NaN` 记号**，严格 JSON 解析器会拒绝。）

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

## 4. 逐表空格清点（填表前实测，2026-09-20）

对 §4 全部 10 张表逐格清点（表头/分隔行不计）：

| 表 | 行 × 列 | 空格数 | 处置 |
|---|---|---|---|
| §4.2 主表 | 28 × 10 | **192**（24 行 × 7 ＋ Traffic 4 行 × 6） | 直接复制（`main_table.md`）；Dataset/H/Golden 已预填，Traffic 行多预填 1 格 |
| §4.2 臂级变体表 | 5 × 4 | **0** | **已完整** |
| §4.3 维数表 | 28 × 8 | **0** | **已完整且已校验**（168 格） |
| §4.4 解剖表 | 21 × 8 | **105** | 组合（无 md 生成器） |
| §4.4 干预表 | 21 × 10 | **147** | 复制（去掉产物行首格后，填入 7 个空列） |
| §4.5 四臂表 | 7 × 7 | **35** | 直接复制（`conditional_table.md`） |
| §4.6 负对照表 | 5 × 5 | **0（但是计划文字）** | **整行替换**，且需先决策（见 §1.7） |
| §4.7 预测力表（主） | 3 × 4 | **6** | 组合（来自 summary JSON 的两个 ρ） |
| §4.7 第二张表（候选 `ν` 等） | 3 × 4 | **0** | **已完整** |

**需填总格数 = 192 + 105 + 147 + 35 + 6 = 485 格**（另加 §4.6 的 25 格替换决策）。

**这张清单本身也是一次校验**：它把"看起来还有很多空表"变成"**精确 485 格 + 1 项决策**"，
从而可以在回填完成后用同一个脚本重新清点、确认空格归零——即"填完没有"是可机检的，
不必靠人眼扫表格。

### 4.1 已做成可复用检查：`--inventory`

同一张清单已并入 `verify_minipaper_fill.py`，故**"填完了没有"可机检**：

```text
python scripts/phaseformer_L/verify_minipaper_fill.py --inventory
```

实测输出（2026-09-20）：

```text
=== §4.2 ===  table 1: 28 rows x 10 cols, empty cells = 192
              table 2:  5 rows x  4 cols, empty cells =   0
=== §4.3 ===  table 1: 28 rows x 12 cols, empty cells =   0
=== §4.4 ===  table 1: 21 rows x  8 cols, empty cells = 105
              table 2: 21 rows x 10 cols, empty cells = 147
=== §4.5 ===  table 1:  7 rows x  7 cols, empty cells =  35
=== §4.6 ===  table 1:  5 rows x  5 cols, empty cells =   0
=== §4.7 ===  table 1:  3 rows x  4 cols, empty cells =   6
              table 2:  3 rows x  4 cols, empty cells =   0
section 4 empty cells in total: 485
```

**完成判据：回填后重跑该命令，总数应为 0**（那三张"刻意带文字"的表本就是 0）。

**两处须记的技术细节**：

1. §4.3 表被解析成 **12 列**，而它实际是 8 列——因为表头与单元格里含**转义竖线**
   （`\|cos\|`）。这对"空格计数"无影响（空格的判定不受列数影响），但**任何按列索引取值的
   解析器都必须先处理转义**——否则 §4.3 的列会整体错位。本会话早先就因 `|cos|` 与 `\|cos\|`
   的转义问题反复修过文档表格，此处再次提醒。
2. 该 mode 的解析器**修过一次**：初版每节只开一张表，于是把 §4.2 的两张表、
   §4.4 的两张表各**合并**成一张——**总数仍是对的（485），但逐表明细是错的**。
   这正是"总数对不等于结构对"的一个实例，也正是这个 mode 存在的意义；
   修法是用**前瞻一行是否为分隔线**来判定表头（`|---|`），从而正确切分相邻表。

