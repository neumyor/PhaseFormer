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

## 5. 补充：全篇"过时设计"扫查，与 §4.3 的一处预注册残留

### 5.1 扫查：minipaper 里是否还留有被后续决策推翻的写法

用关键词在**全文**扫查早期草案中被推翻的设计（因 D-4/D-5/D-6 与两处勘误都改过论文）：

| 关键词 | 命中 | 判定 |
|---|---|---|
| `boxcar` | 2 | ✓ **合法**：一处是 §4.6 既有实验的算子说明（"boxcar / causal EMA，各 5 档"）与本次 2 档复测；一处是解释"PhaseFormer-L 的 `smooth_ratio` 是**强度**而非算子，真正的 boxcar 只在 `pooled_lowrank` 头" |
| `always-on` | 1 | ✓ **合法**：位于"D-5 修订"注中，明确写"早期草案曾有两列，现二者合并" |
| `12 格既有` | 1 | ✓ **合法**：位于 A1 行的**勘误**段，记录该前提已被 2026-09-18 审计推翻 |
| `s=0` | 2 | ✓ **合法**：主张 B 的定义与必答 (b) 里的**诊断列**用法（与 D-5"`s` 不进入模型"一致） |
| `5 个已知符号` | 1 | ✓ **合法**：记录把早期草案的"6 个 setting"更正为 5 个 |

即：**扫查未发现残留的过时设计**——所有命中都是**自证性的修订/勘误注记**，
说明论文的修订史是连贯的（每一处改动都留下了"早期怎么写、为何改"的记录）。

另外确认全文只剩 **2 个显式待填标记**，且都准确：
摘要的 `*[主结果待填。]*`，与 §4 开头的填充状态注（"§4.2、§4.4、§4.5、§4.6 待填，
§4.7 的两列 ρ 待 test 读取完成后回填"）。

### 5.2 修正：§4.3 的"预期图"是预注册残留

§4.3 是**唯一已完成**的实验小节（E15 全六阶段完成）。它已按实测回填了
`λ_1/Σλ` 0.642–0.862、`a_1│cos│` 0.890–0.989、`PR` 1.33–2.29 等，
但段末仍留着预注册写法：

```text
预期图：Scree 图（第一根柱 0.66–0.86 量级）；`b_1` 随 lag 的剖面（近端集中 + 指数衰减）与
`a_1` 随 horizon 的剖面（近似平线）双面板。
```

问题有两层：①"**预期**图"出现在**结果**小节里，读起来像未完成的草稿；
②括号里的 `0.66–0.86` 是引用的**先导**区间，而同一小节上文已经报告了**实测**的 `0.642–0.862`——
同一页里两个区间并列容易被误读为矛盾。

**修正**：改为报告实际图件与实测锚点，文件名逐一给出：

* `figures/scree_lambda_spectrum.png`（`λ_1/Σλ` 谱，首根 0.642–0.862）；
* `figures/b1_lag_profile.png`（`b_1` 随 lag 剖面：近端集中 + 指数衰减，Traffic 一格一致落在 τ=168）；
* `figures/a1_horizon_profile.png`（`a_1` 随 horizon 剖面：近似平线，常值 `│cos│` 0.890–0.989）。

**三个数字全部取自该小节上文已有的实测条目，未引入任何新数字**；
并在括注里说明"本条原为预注册写法、E15 完成后改为报告实测值与实际文件名"，
使改动本身可追溯。

**文件名的独立核实**（不靠记忆）：`e15_dimension.py` 确实写出这三个名字
（`:803` `scree_lambda_spectrum.png`、`:850` `b1_lag_profile.png`、`:857` `a1_horizon_profile.png`），
且 E15 的阶段 4/5 文档与阶段 5 审校（含"成图性验证"，240–344 KB）记录一致。

**顺带记下 E15 的既有验证强度**：该节的图不只是"文件存在"——阶段 5 已补做过**内容层面**的
成图性验证（`e15_dimension/05_audit.md` §6），故此处回填引用是安全的。

---

## 6. 回填校验：论文里**填进去的数**是否等于产物里的数

### 6.1 为什么这是最后一道没人守的门

六阶段契约的最后一环是**回填**（把数字写进论文）。前五项现在都有机器检查：
产物由工具生成，产物自身的验收判据由 `audit_phase2_outputs.py` 检查。
但"**把数填进论文**"这一步此前纯手工——而且**没有任何工具会写论文**：
唯一两个提到 minipaper 路径的脚本（`e15_dimension.py`、`e16_dissection.py`）
只是在 docstring 里引用它。

在五张表、几十个数值单元上手工转录，正是"**论文与自己的产物不一致**"的高发处；
而这种错误能躲过所有自动化检查——因为**两边各自都是对的**。

工具：`scripts/phaseformer_L/verify_minipaper_fill.py`。

### 6.2 判据：按**显示精度**比较，而不是比原始浮点

论文显示 `0.712`，产物存 `0.711673`（3 位小数），`1.90` 对 `1.8987`（2 位小数）。
故判据是"**论文显示的数值 == 产物值按该显示精度四舍五入后的值**"。

这条不是随手定的：**比原始浮点会把"正确的四舍五入"报成错误**——
E17 回填预演里我正是这样自伤了一次（拿 6 位小数的显示值去比未舍入的全精度值，
报了 7/7 MISMATCH，而工件其实完全正确）。所以这次把判据写死在工具里并注释了原因。

状态设计沿用审计器的三态思路，使它可以随时运行：
`match` / `MISMATCH`（**致命**，exit 1）/ `blank`（论文格还空着 → PENDING，**不**致命）/
`PENDING`（产物尚未生成）。

### 6.3 §4.3 的校准结果：**168 个单元全部一致**

§4.3 是目前**唯一已回填**的小节，因此它同时是理想的校准样本
（论文已填 + 产物真实存在）。实测：

```text
match: 168
OK: every filled section-4 cell matches its artifact at the displayed precision
exit=0
```

168 = 28 行 × 5 个数值列（`λ_1/Σλ`、`pred_dims_90`、`PR`、`a_1` vs 常值 `│cos│`、
`used_var_share(1)`）= 140，加 28 个 `b_1` 复合格（模板名 + `│cos│`）= 168。

**顺带的独立收获**：这等于**回头验证了此前 §4.3 的回填本身没有转录错误**——
那次回填是手工做的，此前只有"产物正确"（E15 阶段 5）与"表格已填"两类证据，
缺的正是"填进去的数与产物一致"这一环，现在补上了。

### 6.4 检测力校准（四项对照）

一个永远说 OK 的校验器等于没有，故做了四项对照：

| 对照 | 期望 | 实测 |
|---|---|---|
| 真实论文 | 168 match、exit 0 | ✓ |
| **扰动一个论文值**（`0.712`→`0.799`） | 报出并 exit 1 | ✓ `§4.3 / ETTh1-96 / lambda1_share_of_achievable: paper=0.799 artifact=0.711673 rounded=0.712 dp=3` |
| **清空一个论文格** | 记为 `blank`、**不**致命 | ✓ `still blank (1 cell(s))`、exit 0 |
| **删掉一行** | 行数不符 → exit 1 | ✓ `paper has 27 rows, artifact has 28` |

### 6.5 覆盖范围：**目前只覆盖 §4.3**（必须写明，不能含糊）

其余小节（§4.2、§4.4、§4.5、§4.6、§4.7）的表**现在还是空的**，故没有可校准的样本。
本轮**故意不为它们预先写比较逻辑**：在没有真实填充行可校准的情况下先写比较器，
正是本会话反复踩的坑（累计 9 次"搭台错误被误当真缺陷"）。
纪律是：**每填完一节，就为该节补上比较器并当轮校准**（对真实行 + 扰动对照），
而不是一次性写好五个未经校验的解析器。

**因此本工具现在的 pass 只意味着"§4.3 的填充与产物一致"**，
不代表其余小节已核对。这一句是本节最重要的边界说明。

---

## 7. 计划 ↔ 论文 覆盖核对：manifest 是否**恰好**覆盖 §4.2 要求的格子

### 7.1 这正好回到本任务最初的问题

这条任务最初的问题是"minipaper §4 还要求补哪些实验"。前几节核对的是"论文写下的数字"与
"实现里的常量/产物"是否一致；本节换一个方向核对：**实验计划是否恰好产出论文表格要求的每一个格子**
——**不多不少**。多了是浪费算力与事后解释负担，少了就是"论文有行、实验无格"，
而这正是最初要防的事。

数据来源是**真实 manifest**（`stage_a_manifest.json`，492 cells），不是我对它的记忆。

### 7.2 论文侧的要求（从表格读出）

§4.2 的表格经核对是**两张表**：

* 主表 **28 行** = ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity 各 4 个 horizon（**24**）
  ＋ **Traffic 附录 4 行** —— 即"**24+4**"；
* 其后另有一张**臂级变体小表**（4 行：`weak_residual` shared；L-q1/4 与 L-q1/8；
  `rcrf_nlinear_plain`；`gold_combo_reliability_s2`）。

（故对 §4.2 段落做整段 `grep '^| '` 会得到 33 行：28 + 变体表 1 行表头 + 4 行数据。
早先若不知有第二张表，容易把 33 误读为行数异常——此处记下以免重蹈。）

### 7.3 实现侧（真实 manifest）与判定

工具：`scripts/phaseformer_L/check_section42_coverage.py`（可复用、可指向 manifest 副本）。

```text
all settings: 28 = 24 main + 4 Traffic
arm         settings seeds reused  new  cells  expected
a1                24     3      0   72     72  24 settings
l_main            28     3     21   63     84  28 settings
l_q1_4            28     3     21   63     84  28 settings
l_q1_8            28     3     21   63     84  28 settings
l_rcrf            28     3      0   84     84  28 settings
phase_only        28     3     18   66     84  28 settings
total cells: 492 (expected 492)
PASS
```

四项都对上了，且**复用计数与各自声明的范围一致**（这是附带的一次交叉验证）：

| 臂 | 复用 | 应然 | 依据 |
|---|---:|---|---|
| `l_main`/`l_q1_4`/`l_q1_8` | 21 | 7 test-selected setting × 3 seed | `REUSE_SCOPE` = `REUSE_SETTINGS_FULL` |
| `phase_only` | 18 | **6** setting × 3 seed | `REUSE_SETTINGS_PHASE_ONLY`——独缺 Electricity-336（E8 未覆盖该格） |
| `l_rcrf` / `a1` | 0 | 0 | 审计确认二者无可复用格 |

且 `a1` **确实只覆盖 24 个主 setting、不含 Traffic**，与 `ARM_DATASET_EXCLUSIONS` 一致。

### 7.4 检测力校准

| 对照 | 期望 | 实测 |
|---|---|---|
| 真实 manifest | 全部一致、exit 0 | ✓ 492 = 492 |
| **从副本中删掉 `a1` 臂** | 报错并 exit 1 | ✓ `FAIL: total cell count differs` + `FAIL: arms absent from the manifest: ['a1']` |

### 7.5 为什么值得做成可复用脚本

manifest **已经被替换过一次**（2026-09-19 的复用污染事件：`gate_init` 被写入 seed、
`learning_rate=0.5`，原文件归档为 `stage_a_manifest.prelaunch_contaminated.json`）。
既然它是会被重建的产物，"计划是否仍恰好覆盖 §4.2"就应当是可随时复算的断言，
而不是一次性的口头核对。

### 7.6 边界

* 它核对**格子覆盖**（哪些 dataset×horizon×seed×arm 存在），**不**核对格子里的数值；
* 它不检查 `--verify`（复用解析正确性）——那是 `e14_reuse_audit.py` 的职责；
* 它只覆盖 §4.2（E14）。§4.4–§4.7 的覆盖由各自的 plan 门负责
  （E16 的 63 cell、E17 的 84/24、E18 的 78/28 均已在前文核对过）。

---

## 8. 回填校验扩展到 §4.2（主表），并完成五类对照校准

### 8.1 为什么 §4.2 这一节用**字符串比较**而不是数值比较

`e14_writeback` 产出的 markdown 行与论文 §4.2 主表**列序完全一致**（10 列）：
`dataset | horizon | golden_mse/golden_mae | phase_only | PhaseFormer-L | Δ |
g | s | stable | provenance_note`（生成点 `e14_writeback.py:678-681`，
格式串 `| %s | %d | %s/%s | %s | %s | %s | %s | %s | %s | %s |`）。

因此 §4.2 的校验**不需要任何数值格式假设**：直接逐格比较"论文行"与"产物行"的文本即可。
这比数值比较**更强**——它不仅验数字，还验**列映射**与**来源/披露列**（那一列是中文长文本，
手工转录最容易错，且数值比较根本覆盖不到）。

### 8.2 五类对照校准（用"由论文自身行构造的匹配对"）

校准的巧处：§4.2 主表的 **Golden 列本来就是预填的**（如 `0.359/0.382`），其余 9 列空着。
于是可以从论文自己的行**构造**出一对"正确匹配"（产物侧行 = 论文侧行填满），
再人为破坏，检验校验器是否会发现：

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 全部 match、exit 0 | ✅ **`match: 280`**（28 行 × 10 列）、exit 0 |
| 2 | **改论文一格**（`0.18x/0.28x` → `0.999/0.999`） | MISMATCH、exit 1 | ✅ `MISMATCH: 1`、exit 1 |
| 3 | **改产物一格** | MISMATCH、exit 1 | ✅ `MISMATCH: 1`、exit 1 |
| 4 | **删论文一行** | 行数不符、exit 1 | ✅ `MISMATCH: 1`（row count）、exit 1 |
| 5 | **留空一格** | 记 `blank`、**不**致命 | ✅ `blank: 1`、exit 0 |

即：该分支**在两个方向上都能发现不一致**，且对"尚未填"与"填错"给出不同处置。

### 8.3 校准时顺带得到的一条交叉验证

校准输出 `section 4.2 main-table rows found: 28`——
**论文侧**的 §4.2 主表恰为 **28 行**。这与 §7 从**计划侧**（真实 manifest）
得到的结论（28 = 24 main + 4 Traffic）**相互独立且一致**：
一边是"论文要求的行数"，一边是"计划覆盖的格子数"，两者都对上 28。

### 8.4 覆盖范围更新（仍是诚实的边界）

`verify_minipaper_fill.py` 现在覆盖 **§4.3（168 格，已填）** 与 **§4.2（280 格，待填；校验器已校准）**。

**仍未覆盖** §4.4、§4.5、§4.6、§4.7 的表。对这四节，纪律不变：
**先看它们的表头与产物行的列序是否逐列一致**（像 §4.2 这样能直接字符串比较最好），
**填完当轮校准**，而不是先写四个未经验证的解析器。

**因此 §4.2 分支现在的 pass 只意味着"（如果有填充行）它与产物逐格一致"**——
在 step 3 产出 `main_table.md` 并真正回填之前，它只会报 `PENDING`。

---

## 9. 回填校验扩展到 §4.5，并发现一处**行文本后缀撞车**的隐患

### 9.1 §4.5 的两条路径：六列复制 + 一列聚合

`verify_minipaper_fill.py` 新增 §4.5 分支：

* **前 6 列**（`direct` / 冻结独立 / 冻结条件 / 联合 等）与 `conditional_table.md`
  的前 6 格**逐格字符串比较**；
* **第 7 列（H1）不能复制**——论文表头写"（**seed 数**）"，而产物那格是
  **逐个 seed 的判定**（`true`/`false`/`evidence_missing`，`e17_conditional.py:972-984`）。
  故该列由 `results.with_test.csv` **按 setting 聚合 3 个 seed** 得到
  （数 `"true"` 的个数 → `N/3`；三个全为 `evidence_missing` → 照写），
  这正是本会话早先**独立**算出 H1 汇总时用的口径。
  比较时对数字**容错**（正则取 `N/M`），故写法写成 `3/3` 或 `3/3 seed` 都不会误报。

### 9.2 校准：五类对照，四种种子的聚合结果都被覆盖

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 全部 match、exit 0 | ✅ **`match: 49`**（7 行 × 7 列）、exit 0 |
| 2 | **改论文的 H1 计数** | MISMATCH、exit 1 | ✅ |
| 3 | **改论文的方法列** | MISMATCH、exit 1 | ✅ |
| 4 | **删论文一行** | 行数不符、exit 1 | ✅ |
| 5 | **H1 留空** | 记 `blank`、**不**致命 | ✅ exit 0 |

聚合的四种结果都被走到（构造的 7 行按 `3/3`、`2/3`、`0/3`、`evidence_missing` 循环）✓。

### 9.3 发现并记下的隐患：**§4.5 的行文本是 §4.4 解剖表行的后缀**

校准第一版"正确匹配对"竟然报 8 处 MISMATCH。逐项追查后确认**是校准脚本自己的错**：
它用 `str.replace(整行文本, …, 1)` 去填，而

```text
§4.5 的行：            | ETTh2 | 96 |  |  |  |  |  |
§4.4 解剖表的一行：| PhaseFormer-L | ETTh2 | 96 |  |  |  |  |  |
```

**前者是后者的后缀子串**——于是 `replace` 命中了**文档中更早的那一行**（§4.4），
把 §4.4 的一行改坏、而 §4.5 原封不动（`changed=True` 但 §4.5 未变，正是这个原因）。

**修法**：改为**按行定位**（只在该小节的行区间内、用"整行 strip 后相等"匹配），
并在校准里断言"7/7 行都被替换"，否则拒绝继续。

**这条隐患值得写下来，因为任何"自动填充/批量替换"都会踩它**：
本套件里同一张 7 列表的行文本可能是另一张表行文本的后缀，
**按子串替换会静默改错表**。结论：**回填脚本必须按行、且限定在目标小节的行区间内定位。**

### 9.4 覆盖范围更新

校验器现覆盖 **§4.3（168 格，已填）**、**§4.2（280 格，待填）**、**§4.5（49 格，待填）**。
**仍未覆盖** §4.4（解剖 105 格需组合、干预 147 格可复制）与 §4.6（25 格替换）、§4.7（6 格）。
纪律不变：**填完当轮校准**，且本轮的"后缀撞车"教训适用于上面每一节。

---

## 10. 回填校验扩展到 §4.4 干预表（147 格里可核对的部分）

### 10.1 映射：产物行**去掉首格**后与论文逐格相同

`e16_writeback` 的行是 **11 格**（`model | dataset | H | q/r | 七个值`），
论文 §4.4 干预表是 **10 列且没有独立的"模型"列**——因为模型身份**已含在 `q/r` 标签里**
（`dense（r=96）` / `q=1/4（r=24）` / `q=1/8（r=12）`，由 `e16_writeback.qr_label()` 生成）。
故映射为 `artifact_row[1:] == paper_row[0:]`，校验即十格逐格比较。

### 10.2 校准：五类对照

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 全部 match、exit 0 | ✅ **`match: 210`**（21 行 × 10 列） |
| 2 | 改论文一格 | MISMATCH、exit 1 | ✅ |
| 3 | 改产物一格 | MISMATCH、exit 1 | ✅ |
| 4 | 删论文一行 | 行数不符、exit 1 | ✅ |
| 5 | 留空一格 | 记 `blank`、**不**致命 | ✅ exit 0 |

### 10.3 顺带查实的一处预填差异（**必须整行替换**）

论文的 **dense 行**把 `q/r` 预填成通用写法 `dense（r=H）`，而产物按 `qr_label()` 写**具体值**
`dense（r=96）`（低秩行本就写具体值，如 `q=1/4（r=24）`）。二者**不一致**，
故回填必须**整行替换（含 `q/r` 格）**——否则那 7 个 dense 行会被校验器报 MISMATCH，
而那是**填法**问题、不是判据太严。这与 §1.1（§4.2）同一条规则：**预填格也要被产物覆盖。**

### 10.4 覆盖范围更新

校验器现覆盖 **§4.3（168 格，已填）**、**§4.2（280 格，待填）**、**§4.5（49 格，待填）**、
**§4.4 干预表（210 格，待填）**。**仍未覆盖**：§4.4 解剖表（105 格，写法待定）、
§4.6（25 格替换）、§4.7（6 格）。纪律不变：填完当轮校准；且**按行定位**（§9.3 的后缀撞车教训）。

---

## 11. §4.6 的回填范围**比我先前判断的窄**，并已据此实现校验器

### 11.1 修正：只有第五列需要填，不是整行替换

我在 §1.7 曾据"`existing` 列 4/5 逐字相同"推断"`negative_table.md` 的设计意图是**整表重建**"。
本轮把**全部五行的前三列**也逐格对照后，这个推断**站不住**：

| 行 | 操作 | 作用对象 | 口径 | 判定 |
|---|---|---|---|---|
| 2 | 结构化坐标（周期低秩、共享基、水平/形状、近期周期、可分离） | 支路参数化 | 4 setting | **逐字相同** ✓ |
| 3 | SVD 截断 vs 秩约束训练 | 支路权重 | Electricity-336 r=10 | **逐字相同** ✓ |
| 4 | 联合低秩训练 q=1/32 | 支路容量 | 7 setting | **逐字相同** ✓ |
| 1 | paper `输入平滑（boxcar / causal EMA，各 5 档）` | 支路输入 | 7 setting | **操作不同**（产物写 `输入平滑（causal EMA 两个强度）`） |
| 5 | paper `**边界消融：\`pooled_lowrank\` rank∈{1,2}**` | paper `支路容量（网格之外）` | paper `6 setting × 3 seed` | **三列都不同**（产物：`边界消融：pooled_lowrank 绝对秩 rank∈{1,2}` / `支路容量（低秩网格之外）` / `6 setting × 3 seed × 2 rank`） |

**3/5 行逐字相同、2/5 行措辞不同**。而不同的那两处，论文的措辞是在描述**先导实验**
（第 1 行明确写"boxcar / causal EMA，各 5 档"——那正是第 3、4 列所报告的既有实验的算子网格）。
故**整行替换会把准确的既有描述改写成产物的说法**。

**结论（修正 §1.7）**：§4.6 的回填范围是**第五列（本文补做）**，取产物的 `addendum`；
被它顶掉的**计划/规模文字**（`42 runs`、`36 runs`、以及第 3 行的 E11 口径差异——后者**已先行搬迁**）
应移入表下注。**前四列保持论文原文。**

### 11.2 校验器与校准

`verify_minipaper_fill.py` 新增 §4.6 分支：**第五列严格比较**；
前四列若与产物不同则记为 **`INFO`**（允许不同，理由写在 docstring 里），相同则记 `match`。

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 第五列 5 处 match、exit 0 | ✅ `match: 21`（含前四列的相同项） |
| 2 | **改论文的第五列** | MISMATCH、exit 1 | ✅ |
| 3 | **改产物的 `addendum`** | MISMATCH、exit 1 | ✅ |
| 4 | **删一行** | 行数不符、exit 1 | ✅ |
| 5 | **第五列留空** | 记 `blank`、**不**致命 | ✅ exit 0 |

### 11.3 由此发现的一个**判据盲区**（重要）

`--inventory` 的"空格数"**看不到 §4.6 的回填**：该表 25 格**本来就非空**（装的是计划文字），
回填是**替换文本**而非填空——所以**即使 §4.6 完全没填，总数也仍然是 0**。

**故完成判据必须两条并用**：

1. `--inventory` 总数 → 0（覆盖 §4.2/§4.4/§4.5/§4.7 的**填空型**回填）；
2. `verify_minipaper_fill.py` 无 MISMATCH（覆盖 §4.6 这种**替换型**回填，以及两类的一致性）。

只跑第 1 条会误以为"§4.6 已完成"。这条已补进执行排期的 §10.4/§10.5。

---

## 12. 回填校验扩展到 §4.7 —— **六个小节全部覆盖**（只剩 §4.4 解剖表的写法待定）

### 12.1 §4.7 的两个 ρ 列

论文 §4.7 第一张表有 3 行（三个候选统计量，顺序取自生产者的 `STATISTICS`）与两列 ρ；
数据来自 `predictive_power_summary.json` 的
`predictive_power.spearman[f"{stat}_vs_delta_mse_pct"]["rho"]` 与 `..._vs_gate_value`
（该键结构此前已**实测核验**，见 §7.6）。第 4 列"预期符号"是设计文字，**不校验**。
数值按论文的显示精度比较。

**NaN 的处理**：产物里的 `rho` 可能是**非有限值**（合成/退化输入下会出现，见 §1.8 的记录）。
校验器把它记为 **`PENDING` 而非 `MISMATCH`**——因为那是"产物未给出可比值"，不是"论文与产物矛盾"。

### 12.2 校准：五类对照

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 6 处 match、exit 0 | ✅ `match: 6`（3 行 × 2 列） |
| 2 | **改论文的 ρ** | MISMATCH、exit 1 | ✅ |
| 3 | **改产物的 ρ** | MISMATCH、exit 1 | ✅ |
| 4 | **ρ 留空** | 记 `blank`、**不**致命 | ✅ exit 0 |
| 5 | **产物的 ρ 为 NaN** | 记 `PENDING`、**不**致命 | ✅ exit 0（5 match + 1 PENDING） |

### 12.3 校准里我自己的一处脆性（已修，且教训通用）

首轮校准报"`line not found in 4.7`"。原因是我的校准**重建行文本去匹配**：

```text
论文 §4.7 的空格写法：| `cycle_level_std` | | | + |      ← 单空格
其它表的空格写法：    | ETTh2 | 96 |  |  |  |  |  |      ← 双空格
```

`render()` 统一按 `" | "` 拼接，于是**在单空格写法上匹配不上**。
（校验器本身没问题——它按 `strip()` 取格，两种写法都能解析。）

**修法**：改为**按键格匹配**（`§4.7` 内第一格即统计量名，唯一），不再重建整行文本。
**教训**：markdown 表的空格外写法在各表之间**并不统一**，任何"重建行文本再匹配"的做法都脆；
**应按格/按键定位**。这与 §9.3 的"后缀撞车"是同一类问题的另一面。

### 12.4 覆盖总表（截至本轮）

| 小节 | 格数 | 校验器状态 |
|---|---:|---|
| §4.3 | 168 | ✅ 已填、已验证 |
| §4.2 主表 | 280 | ✅ 已实现 + 五类校准 |
| §4.4 干预表 | 210 | ✅ 已实现 + 五类校准 |
| §4.5 | 49 | ✅ 已实现 + 五类校准 |
| §4.6 | 5（第五列） | ✅ 已实现 + 五类校准（替换型） |
| §4.7 | 6 | ✅ 已实现 + 五类校准 |
| **§4.4 解剖表** | **105** | ⏳ **写法待 E16 跑完后据真实取值定**，届时实现并校准 |

即：**七处表位中六处已有经校准的校验器**，仅剩解剖表一处——而它的收口时点已定（E16 之后），
且届时的做法是"先打印真实取值、定写法、再实现并校准"，不提前写未经验证的解析器。

---

## 13. §4.4 解剖表校验器：**七处表位全部覆盖**，最后一处开放项就此收口

### 13.1 映射（取自生产者 `build_dissection`，不是猜的）

| 论文列 | 来源列 |
|---|---|
| 模型 / Dataset / H | `model` / `dataset` / `horizon`（`model` 取自 `ARM_DISPLAY`，与论文行标签**逐字相同**） |
| **主模式输入组 / 解释率** | `leading_input_group_label` **＋** `mean_input_group_explanation` |
| **主模式输出组 / 解释率** | `output_group_label` **＋** `mean_output_group_explanation` |
| 修正能量份额 | `leading_correction_energy_share` |
| **跨 seed `leading4` 重叠** | `leading4_input_overlap` **＋** `leading4_output_overlap`（论文一格、产物两列） |
| 稳定语义判定 | `stable_semantics_verdict`（**bool**） |

### 13.2 写法的收口依据（**不必等 E16 跑完**）

上一轮我把"单元格写法"列为唯一开放项、打算等 E16 出数据后再定。本轮改为**从代码与表头直接定**，
依据有三，无需运行：

1. **论文表头自己给了分隔约定**：该列写作 `主模式输入组 / 解释率`——**斜杠**就是"组名 / 解释率"的写法；
2. **解释率的量纲有旁证**：生产者的判据之一是 `criterion_2_input_explanation_ge_0p5`，
   即解释率是 **0–1 的小数**（≥0.5 的阈值只有在 0–1 尺度下才讲得通），故按两位小数渲染；
3. **行标签可信**：`model` 与论文的 `PhaseFormer-L` / `L-q1/4` / `L-q1/8` **逐字相同**（同取自 `ARM_DISPLAY`）。

**采用的渲染**：`{组名} / {解释率:.2f}`、修正能量份额 `{:.3f}`、跨 seed 重叠 `{in:.2f} / {out:.2f}`、
判定用 `✓/✗`。

**一处刻意的容错**：组名**本身可能含斜杠**（真实标签里有 `周期形状/相位`），
故校验器**不做"按斜杠切分"**，而是用"**单元格以组名开头** + 末尾数字等于四舍五入后的解释率"来判，
判定列也接受 `✓/✗` 与 `true/false` 等多种写法。这样**写法上的小差异不会制造假报警**，
而数值与标签仍被严格校验。

### 13.3 校准：**七类对照**

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 正确匹配对 | 全部 match、exit 0 | ✅ **`match: 105`**（21 行 × 5 个被校验列） |
| 2 | **改解释率** | MISMATCH、exit 1 | ✅ |
| 3 | **改组名** | MISMATCH、exit 1 | ✅ |
| 4 | **改重叠对** | MISMATCH、exit 1 | ✅ |
| 5 | **翻转判定** | MISMATCH、exit 1 | ✅ |
| 6 | **产物少 3 行** | MISMATCH（"no artifact row"）、exit 1 | ✅ 3 处 |
| 7 | **组合格留空** | 记 `blank`、**不**致命 | ✅ exit 0 |

校准里的合成标签**故意含一个带斜杠的**（`周期形状/相位`），以验证容错路径真的被走到 ✓。

### 13.4 覆盖总表：**七处全部覆盖**

| 小节 | 格数 | 校验器 |
|---|---:|---|
| §4.3 | 168 | ✅ 已填、已验证 |
| §4.2 主表 | 280 | ✅ 五类校准 |
| §4.4 干预表 | 210 | ✅ 五类校准 |
| §4.4 解剖表 | 105 | ✅ **本轮**：七类校准（组合格分部件校验） |
| §4.5 | 49 | ✅ 五类校准 |
| §4.6 | 5（第五列） | ✅ 五类校准（替换型） |
| §4.7 | 6 | ✅ 五类校准（含 NaN → PENDING） |

**回填阶段的机器判据已完备**：填完后 ①`--inventory` 总数 → 0（填空型）、
②`verify_minipaper_fill.py` 无 MISMATCH（含替换型的 §4.6）。**无待定项、无未校准的解析器。**

---

## 14. 由"防线地图"反查审计器：**四个过弱的判据**已加强并校准

§10.1.3 的防线地图指出一个结构性事实：**分析/回填层是"报告后继续"（exit 0），
真正把关的是第 7 步验收审计**。于是应当反问一句——**审计器的判据是否真的覆盖了那些"报告出来的东西"？**
逐条对照后发现**四个判据过弱**（用"非空/存在"代替了真正的计数或一致性判据），已全部加强：

| 原判据 | 问题 | 加强为 | 依据 |
|---|---|---|---|
| `parameter_table.csv present` + **`non-empty`** | `e14_params.py` 是**报告型**（把解析不到的格列进 `unresolved` 并**照样 exit 0`）；"非空"会在**半张矩阵缺失**时通过 | **行数 = 492**（每个 cell 一行，与 `cells_with_parameters + unresolved = 492` 一致） | 源码 + dry-run 实测 175+317=492 |
| （无） | `e14_params` 会与 run 自身的 `metrics.csv:parameter_count` 交叉校验，但审计器没看 | **每行 `total_matches_metrics` 不得为 False** | 源码 |
| （无） | §4.2 的 `g` 均值列要求"每个 checkpoint 都能恢复门值"（E3 系 `metrics.csv` 无门值列，必须从 checkpoint 读） | **每行 `gate_value_from_checkpoint` 非空** | `e14_params` 的门值回退路径 |
| `results non-empty`（E18） | 同上：半跑的 E18 也能通过 | **行数 = 78** = 42 平滑（7 setting × 3 seed × 2 档）+ 36 边界（6 setting × 3 seed × 2 秩） | 计划计数与**我先前预演的行数算术**两条独立一致 |

**六类校准全部通过**：①492 行且一致 → 三项全 OK；②491 行 → `FAIL parameter table covers all 492 cells: 491 rows`；
③一行 `total_matches_metrics=False` → `FAIL … 1 row(s) mismatched e.g. ETTh1-96-0/l_main`；
④一行缺门值 → `FAIL gate value recovered from every checkpoint …`；
⑤E18 78 行 → `OK results rows = 78`；⑥E18 77 行 → `FAIL results rows = 78: 77 rows`（且 exit 1）。

**这条链值得记下**：它不是我事先想到的，而是**先建立"谁把关"的结构认识（防线地图），
再据此反查"把关者的判据是否够严"**才发现的。若只逐个检查判据，很容易停在"这条判据看起来没问题"；
而一旦问"**这条判据要挡住什么**"，就会发现"非空"挡不住"半张矩阵"。

---

## 15. 判据覆盖矩阵：**每个"被报告出来的缺口"，由哪条审计判据接住**

§14 的教训是"要问判据**要挡住什么**"。本节把它系统化：把六个工具的**全部缺口计数**列出来，
逐条指出**是哪条审计判据在接住它**（或**为何有意不查**）——这是"审计器是否完整"的直接证据。

| 工具 | 它报告的缺口 | 接住它的审计判据 |
|---|---|---|
| `e14_read_test.py` | `problems`、`failed_workers`、`evidence_rejected_rows` | E14：「results 行数 = 492」+「每行 `test_mse`/`test_mae` 非空」；**且该步自身 fail-closed** ✓ |
| `e14_params.py` | `unresolved` | E14：「参数表覆盖全部 492 格」（**§14 新增**） |
| `e14_params.py` | `mismatches`（与 `metrics.csv:parameter_count` 不一致） | E14：「参数计数与 `metrics.csv` 一致」（**§14 新增**） |
| `e14_params.py` | 门值缺失 | E14：「每个 checkpoint 都能恢复门值」（**§14 新增**） |
| `e16_dissection.py` | `algebra_failures` | E16：`algebra_failures == 0` ✓ |
| `e16_dissection.py` | `run_metric_failures`、`run_metric_not_comparable` | E16：两者均为 0 ✓（**故每 cell 的 `run_metric_state == "skipped"` 已被间接接住**：全量运行时它必须是 `ok`） |
| `e16_dissection.py` | `parity = {"skipped": True}`（传了 `--skip-reference-parity`） | E16：`reference_parity_passed is True` ✓——该键会缺失 → 判 FAIL（"key absent"）✓ |
| `e17_conditional.py` | `failed`（训练失败格） | E17：「24 个新训 cell 都在」+「每个新 cell 都有 test 指标」✓ |
| `e17_conditional.py` | `missing_projectors` | E17：**「无缺失的冻结投影器」**（**§15 新增**）——缺失投影器是**基础设施故障**、不是良性跳过，故值得显式报出，否则症状只剩"0 个新 cell" |
| `e17_conditional.py` | `settings_without_evidence` | **有意不查** ✓：`evidence_missing` 是**已披露的合法结果**（Electricity-336 无 H1 证据，见 §4.5 表注），把它当失败会误报 |
| `e18_negative.py` | `failed` | E18：「results 行数 = 78」+「每 cell 有 test 指标」✓ |
| `e18_svd_truncation.py` | `problems`、`unresolved` | E18：`e18_svd_truncation_summary.json` 的 `problems` 须为空 + 「行 3 覆盖 28 setting」✓ |
| `e19_predictive_power.py` | `settings_without_data` | E19：「预测力表覆盖 28 setting」✓ |

**结论**：**没有"被报告却无人接住"的缺口**；唯一"有意不查"的一项是 §4.5 的 `evidence_missing`，
且理由已写明（它本身是被披露的合法结果）。**这就是"审计器完整"的可核查形式**——
不是"我觉得判据够全"，而是逐条把"工具报告的缺口"对到"审计的判据"上。

### 15.1 §15 新增判据的校准（八类对照）

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 参数表 492 行且一致 | 三项全 OK、exit 0 | ✅ |
| 2 | 参数表 491 行 | FAIL、exit 1 | ✅ `FAIL parameter table covers all 492 cells: 491 rows` |
| 3 | 一行 `total_matches_metrics=False` | FAIL、exit 1 | ✅ `… 1 row(s) mismatched e.g. ETTh1-96-0/l_main` |
| 4 | 一行缺门值 | FAIL、exit 1 | ✅ `FAIL gate value recovered from every checkpoint …` |
| 5 | E18 78 行 | OK、exit 0 | ✅ `OK results rows = 78: 78 rows` |
| 6 | E18 77 行 | FAIL、exit 1 | ✅ |
| 7 | E17 已标审计、无缺失投影器 | OK、exit 0 | ✅ `OK no missing frozen projectors: 0 missing` |
| 8 | E17 缺一个投影器 | FAIL、exit 1 | ✅ `FAIL no missing frozen projectors: 1 missing e.g. ['ETTh2_96_Q1.npy']` |

## 16. 「消费者 ↔ 产物」集成校验：把静态提取器换成真实消费者试跑，并修掉查出的两处真缺陷

§15 收口的是"审计判据是否接得住工具报告的缺口"。本节问的是**更上游的一问**：那些工具**读得进 E14 的产物吗**？
读不进时的症状不是崩溃而是**静默降级**——步骤 exit 0、表照样写，只是**列是空的**。这正是 §4 六阶段契约里
"静态检查"要挡的东西，而它此前**没有**判据。

### 16.1 第一次尝试是错的：静态提取器"看不见接收者"

我先用正则扫了五个消费者源码里所有 `X.get("<key>")`，对真 manifest 报出 4 个"缺口"
（E14 缺 9 键、E16 缺 34 键、E17 缺 5 键、E18 缺 7 键）。**全部是假警报**：这些键的接收者不是 manifest cell，
而是脚本内部构造的 `config`/`hyper`/`cell`/CSV row 等 dict。教训与 §14 同源——
**判据必须问"它实际读的是哪个对象"**，而不是"源码里有没有出现过这个字符串"。

改法：**直接调用消费者自己的 loader**（`load_manifest_cells`、`build_e14_index`、`load_baseline_index`、
`arm_cells`、`build_cell_plan`），对**真产物**跑一遍。这样报出来的缺口才可能是真缺口。

### 16.2 实测的 manifest 结构：492 格、两种形状

| 形状 | 格数 | 键 |
|---|---|---|
| `new` | 411 | `{arm, command, dataset, horizon, key, seed, source: null, status}` |
| `reused` | 81 | `{arm, command: null, dataset, horizon, key, seed, source: {...}, status}` |

`source` 只在 reused 上有值：`{config_hash, gate_init, learning_rate, root, run_dir, test_evidence}`，
且 81/81 的 `run_dir` 都真实存在并带 `config.json`。**`command` 在 reused 格上是 null**——
stage A 没有启动它们，它们是复用审计"收编"来的。这一条就是下面缺陷 1 的根因。

### 16.3 缺陷 1（真）：E18 的基线出处列 **78/78 行会全空**

- **现象（实测）**：`load_baseline_index` → `resolved=63, rejected=21`，21 条拒绝理由全是
  `overrides do not implement l_main`。
- **根因链**：E18 用 argv 里的 `--overrides` 重新推导 `_arm_match`；reused 格没有 command ⇒
  `overrides={}` ⇒ `weak_period_residual_head_type` 取不到 ⇒ 与 `l_main` 的 `shared` 头不符 ⇒ 拒绝。
- **为何不是边角**：E18 的 `SMOOTH_SETTINGS = REUSE_SETTINGS_FULL`（7 个 setting）**正好等于**被复用的那 7 个，
  而 `RANK12_SETTINGS` 的 6 个 setting 也全在其中 ⇒ E18 的 **78 行（42 smooth + 36 rank12）无一例外**。
- **影响面（关键的一步）**：**数字不受影响**。`e18_writeback.build_row1` 的 baseline 取自 E14 的
  `results.csv`（`index_by_setting(e14_rows, arm_filter="l_main")`），不是这些出处列。受损的是**审计链**：
  产物会一边宣称 21 条拒绝、78 行基线为空，一边在 manifest 里写着 `arm_match_used_for_baselines`。
- **修法**：为 reused 格增加**第二条准入路径**——从它实际指向的 run 的 `config.json` 用**同一把尺子**
  （`_arm_match`，即 E14 复用审计当初用的那个）重推指纹，`gate_init`/`learning_rate` 取自该 config，
  `eval_root` 取 manifest 记下的 `source.root`。config 缺失、或 config 不满足 `l_main` ⇒ **照样拒绝**，
  不凭空信任 `source`。new 格的 argv 路径保持原样（防手改）。
- **验证**：真 manifest 上 `resolved=84 (new=63, reused=21), rejected=0`，smooth 覆盖 **21/21**；
  新增 `tests/test_phaseformer_L_reuse_baselines.py` 12 个单测（含 4 个否定对照：头不对、平滑比非 0、
  config 缺失、既无 command 又无 `source.run_dir`）。

### 16.4 缺陷 2（真）：每个 new 格记录的 `--output-dir` 是模板值 `/tmp/e14fix`

- **实测**：411 个 new 格**全部**如此；而 `grep -rn e14fix` 在仓库里**零命中** ⇒ 是"用
  `--output-root /tmp/...` 重建 manifest"留下的模板残留，不是真实评测根。
- **影响面（逐条核过，不是推断）**：
  - **E14 stage B 不受影响**：`dispatch_new_cells` 自己用 `--cell` 重新调用 worker，run dir 由
    `locate_run(output_root/runs)` 按 config 指纹定位，**从不读 `command`**。
  - **E18 SVD 不致命**：`resolve_run_dir` 在 `out_root/runs` 之后**还会**扫 `e14_root/runs`（同一函数内的
    双候选设计），所以真实 run 仍能被找到。
  - **E18 的 `baseline_eval_root` 会**把这个假路径写进产物 ⇒ 已改为：new 格取 **manifest 所在 root**
    （new 格的 run dir 在训练结束前无法命名，root 是唯一诚实的取值）。
- **未采用"重建 manifest"**：E14 正在跑，manifest 是 stage B 的权威输入；为一个出处列去重建，
  风险（键序、复用解析、与在跑批次的一致性）远大于收益。

### 16.5 缺陷 3（运维）：我自己的同步命令**一直是空操作**

`git fetch origin && git merge --ff-only FETCH_HEAD` 这次报 "Already up to date"，但服务器上的 bundle
`git bundle list-heads` 明明含新提交。根因：`remote.origin.fetch` 是 `+refs/heads/*:refs/remotes/origin/*`，
而 `git bundle create <file> HEAD` 只写**一个 `HEAD` ref** ⇒ 裸 `git fetch origin` 匹配不到任何 ref，
**连 `FETCH_HEAD` 都不生成**。正确形式是 `git fetch origin HEAD`（本次已用）。
判据：`git ls-remote origin` 有值 **≠** `FETCH_HEAD` 有值——同步后必须**实测**服务器 HEAD 变了，而不是看退出码。

### 16.6 已固化为 pre-flight：`check_phase2_consumers.py`

新增 `scripts/phaseformer_L/check_phase2_consumers.py`，并接进 `run_phase2_after_e14.sh` 的既有 pre-flight
（在**任何昂贵阶段之前**，与列契约、调用元数两项并列）：

| 判据 | 严重度 | 必须成立的事 | 实测（真产物，2026-09-20 05:1x） |
|---|---|---|---|
| C1 E14 loader schema | 失败即停 | loader 接受 manifest；492 格形状唯一；`counts.total` 自洽 | ✅ 492 cells, one schema |
| C2 E17 条件臂索引 | 失败即停 | 声明的 21 个 `l_main` 格全部解析、0 拒绝 | ✅ 21/21，0 拒绝 |
| C3 E18 基线出处 | 失败即停 | smooth 的每个 (setting, seed) 都有基线、0 拒绝 | ✅ 21/21，`resolved=84`，0 拒绝 |
| C4 E16 解剖计划 | 信息 | 当前可解析多少格（其 scope 由 E16 自己的默认值决定） | ℹ️ 63 格 |
| C5 E18 截断计划 | 信息 | 计划可构造；未解析格属**预期**（stage A 未跑完） | ℹ️ 28 setting，13 个带问题 |

C3 就是**能在 E18 那 3–5 小时之前**抓住缺陷 1 的那条判据。八类对照中本轮新增两类：
本地合成 fixture 的**正对照**（21 个完整 reused 格 → exit 0）与**两类负对照**
（删掉一个被收编 run 的 `config.json` → `rejected=1`；手改一个 new 格的 overrides → `rejected=1`，
理由为 `overrides do not implement l_main`），三类行为均符合预期。
C4/C5 在缺 numpy/torch 时降级为 SKIP 而不是失败（E16 是唯一在导入期需要 numpy/torch 的消费者）。

## 17. 第 7 步验收审计的**两个静默缺陷**：一个会**误判失败**、一个**从不写文件**

§14 的教训是"四个判据**过弱**"（用"非空/存在"代替真正的计数）。本节是它的**镜像**：
一个判据**过严**（把合法的行当成缺失），一个**参数被解析但从不使用**。两者都发生在
**第 7 步——链条上唯一承重的把关者**，也都是靠"拿真产物喂它"才暴露的。

### 17.1 缺陷 A（严重）：门值判据会在**全部 411 个 run 跑完之后**误判整链失败

原判据（`audit_phase2_outputs.py`）：

```python
no_gate = [r for r in rows if not str(r.get("gate_value_from_checkpoint", "")).strip()]
report.add(..., "PASS" if not no_gate else "FAIL", ...)
```

**门参数只存在于三个弱残差臂**（`l_main`/`l_q1_4`/`l_q1_8`，机制 `weak_residual`）；
`phase_only`（`no_residual`）、`l_rcrf`（`rcrf_nlinear_plain`）、`a1`（`gold_combo_reliability_s2`）
**根本没有这个参数**，所以 492 行里有 **85 行合法地没有门值**。

**用真产物实测（不是推断）**：把 `e14_params.py` 在真实 partial 数据上的输出（220 行）放进
合成根，用 `--root` 跑**旧版**审计器：

```text
FAIL gate value recovered from every checkpoint: 85 row(s) without a gate value e.g. Traffic-96/l_rcrf
```

即：**第 7 步会在 411 个 run 全部训完、所有表都填好之后，报告"链条失败"**。
代价不是重跑（审计是纯 IO），而是**在最容易被误信的时点给出一个假警报**——
而假警报与真结论在报告里长得一模一样。

**修法**：判据按行**作用域**收窄，用表自己的 `gate_param_present` 列做判别式，并**另立一条**
判据断言"臂 ↔ 是否有门参数"的一致（否则一整列恒为 `False` 就能让前一条判据**因为没东西可查而通过**）。
再加一条"无门臂却带着门值 = 表被写错"（这条是**校准器**发现的，见 17.3）。修后同一份真产物：

```text
OK  gate value recovered from every gated checkpoint: 0 gated row(s) without a gate value (of 135 gated rows; 85 rows have no gate parameter)
OK  gate presence matches the arm table: 0 row(s) whose arm and gate_param_present disagree, 0 gateless row(s) carrying a gate value
```

旁证：这 220 行的 `total_matches_metrics` **0 处不一致**、`total_params` 无一为空、
`present but no value = 0`、`value but not present = 0`——**真产物本来是对的，错的是判据**。

### 17.2 缺陷 B（轻微但真实）：`--json` 被声明、被解析，**从未被使用**

`run_phase2_after_e14.sh` 的第 7 步是：

```bash
"$PY" scripts/phaseformer_L/audit_phase2_outputs.py --json "$LOGDIR/phase2_acceptance_audit.json"
```

而 `main()` 里 `args.json` **一次都没出现过** ⇒ 收尾时那个"验收报告 JSON"**静默不存在**。
没有任何机器消费者依赖它（所以不会失败），但这正是"文档承诺了的产物没落地"：
我在收尾清单里要读它，却会找不到文件。**修法**：`Report.to_dict()/write_json()`
写出 `criteria / counts / failing / total_criteria / root`，并由校准器断言文件确实出现且自洽。

**顺带澄清一个数字**：判据总数**不是常数**——24 条是"什么产物都没有"时的条数；
每多一张存在的表，条件分支里就多登记若干条（如参数表存在时 +3）。故"24 条判据"只应理解为**基线**。

### 17.3 把校准**固化**成脚本：`rehearse_audit_controls.py`（13 类对照）

§15.1 的八类对照此前是**临时跑**的（脚本在 `/tmp`，随会话消失）。本轮把它写成仓库内脚本，
用审计器**自己的 `--root`**（其 docstring 明写"an audit script that can only ever report
'fine' proves nothing, and this is what lets the checks be calibrated"）对合成树断言逐条判决：

| # | 对照（合成树） | 期望 | 实测 |
|---|---|---|---|
| 1 | 492 行、135 有门值 / 85 无门（真实混形） | 三条判据全 PASS | ✅ |
| 2 | 一行**有门臂**的门值被清空 | 门值判据 FAIL | ✅ |
| 3 | 一行**无门臂**却带着门值 | 一致性判据 FAIL | ✅（**此条最初漏网**，见下） |
| 4 | 一行**有门臂**被标成 `gate_param_present=False` 且无值 | 一致性判据 FAIL | ✅（防"整列恒 False 就能通过"） |
| 5 | 491 行 | 行数判据 FAIL | ✅ |
| 6 | 一行 `total_matches_metrics=False` | 交叉校验判据 FAIL | ✅ |
| 7 | **全部 492 行都是无门臂** | 门值与一致性判据均 PASS | ✅（缺陷 A 的最小复现） |
| 8 | `--json` 到嵌套路径 | 文件写出、`total_criteria` 与 `criteria` 长度一致、含 `failing` 列表 | ✅（28 条） |

**校准器当场抓出了我自己修法的漏洞**：#3 第一次跑是 `PASS`（"0 rows disagree"）——
因为我只比较了"臂 ↔ 是否有门参数"，**没管无门臂上是否残留了一个门值**。
补上 `stray` 检查后 #3 才按预期 FAIL。**这就是把校准写成脚本而不是靠记忆的价值**：
它证明的不只是"判据能挡住已知的坏输入"，而是"**我这次改动自己有没有留下缝**"。

### 17.4 与 §14 合起来的方法论

| 类别 | 症状 | 触发它的做法 |
|---|---|---|
| §14 过弱（4 条） | "非空/存在"放过了半张矩阵、缺门值、交叉校验失败 | 拿**扰动过的**产物喂判据 |
| §17 过严（1 条） | 合法缺失被当成缺失 → **链尾误报失败** | 拿**真实**产物喂判据 |
| §17 空转（1 条） | 参数被解析却从不使用 → 承诺的产物不存在 | 核对"**文档/调用方声明的每个产物**是否真的被写出" |

三条做法合起来才是完整的："**真产物**证明判据不误杀，**扰动产物**证明判据不放过，"
**调用方声明**证明产物不空转。三者缺一，就会剩下一个只在特定输入下才现形的静默缺陷。

## 18. §4.4 的臂数是**逐格而定的**（11/12/13），而审计器、回填与论文都写死了它

§17 的教训是"判据的数必须来自产物结构"。本节是它的第三次复现，而且这次**同一处错误出现在三个地方**
（审计器判据、回填期望值、论文正文），且**审计器那一处会在 E16 那 3–5 小时跑完之后判整链失败**。

### 18.1 三处写死的"11"

| 位置 | 写法 | 后果 |
|---|---|---|
| `audit_phase2_outputs.py` | `intervention rows == 63 × 11 = 693`；`len(arms) == 11` | **第 7 步 FAIL**（真实产物既不是 693 行也不是 11 个臂名） |
| `e16_writeback.py` | `expected_arms = 11`，并以 `len(names) < 11` 判"薄" | 不误报（真实 11–13 ≥ 11），但写进 `arm_coverage.expected_arms_per_cell` 的**期望值是错的** |
| `PhaseFormer_L_minipaper.md` §4.4 | "10 个登记臂 … 之上再追加 `Independent-RRR-only`、`Conditional-RRR-only` 与新增的 `RandomRRR-drop`，故实际为 **11–12 臂**" | 算术本身就不自洽：10 + 3 = 13，不是 12 |

### 18.2 实测：逐格读该格自己的 checkpoint

臂数由 `build_arm_plan` 的**条件**决定：

| 追加项 | 条件 | 实测 |
|---|---|---|
| `PCA-matched-only` / `-drop`（+2） | `semantic_dimension < rank_dim` | **63/63 格成立**（语义张成 36–39，而最小秩为 42、稠密头为 720） |
| `Independent-RRR-only`（+1） | 恒成立（无 Stage-3 文件时用 train split 现拟合） | 63/63 |
| `RandomRRR-drop`（+1） | `--random-rrr` 且 `rrr_dimension ≥ 1` | 63/63（该 flag 默认 `True`） |
| `Conditional-RRR-only`（+1） | 该格存在带 `conditional_basis` 的 Stage-3 文件 | **42/42 低秩格成立**（84 个 `.npz` 全部含该键；稠密头无此文件） |

故 `8 + 2 + 1 + 1 + (0 或 1)` ⇒ **稠密 12 臂、低秩 12 或 13 臂**。逐格（各自 checkpoint 的 encoder、
各数据集真实语义张成，直接 import `build_bases` 用的 `semantic_basis`/`latent_image`，并复刻其秩与条件）：

| 每格臂数 | 格数 | | 按 E14 臂的行数 |
|---|---|---|---|
| 11 | 24 | | `l_main` 252 |
| 12 | 21 | | `l_q1_4` 255 |
| 13 | 18 | | `l_q1_8` 243 |

**合计 750 行、13 个不同的臂名**（不是 693 行 / 11 个臂名；`l_main` 的格子没有 `Conditional-RRR-only`）。

### 18.3 我在这条路上先给错了两个数，如实记下

1. **798 行**：第一版探针取"每个臂一个代表 checkpoint"，把该 checkpoint 的秩套到**所有** setting 上
   （于是 `l_q1_8 ETTh2-96` 被当成 r=42，实际 r=12=96/8），并**假设**低秩格都有 conditional 文件。
   两个错叠加得到 12/13 臂、798 行。
2. **750 行**：改成逐格读**该格自己的** checkpoint，并**实际检查** 42 个 Stage-3 文件后得到的数。
3. 期间还差一点把"11/12/13"写成"12/13"——因为漏了 `semantic_dimension == rank_dim` 的格子
   （r=12 或 24 且语义张成 36–39 时，两者相等 ⇒ **不**追加 PCA-matched）。

**这与 §9.1.2、§9.1.4 是同一类错误**：拿一个**代表量**（中位 run 时长 / 一个 checkpoint）
去套一组**构成在变**的对象。判据是：**当结论依赖"每个对象自己的属性"时，就必须逐个对象取，不能取代表。**

### 18.4 修法：判据改成**结构式**，不再写死基数

审计器（`audit_e16`）：

| 判据 | 形式 | 为什么这样写 |
|---|---|---|
| `intervention table covers 63 cells` | 63 个 `(arm, setting, seed)` 组 | 基数来自产物自己的分组 |
| `every cell carries the always-present arms` | 每格必须含 **10 个恒在臂**（8 登记 + `Independent-RRR-only` + `RandomRRR-drop`） | 真正的缺口是"少了一个恒在臂"，而不是"臂数不是 11" |
| `intervention rows match the runner's count` | 与 `e16_summary.json` 的 `counts.intervention_rows` 相等 | **两件产物互相印证**，不含任何常数 |
| `per-cell arm counts match the runner's` | 与 `counts.intervention_arms_per_cell` 相等 | 同上；缺该键时降级为 INFO 而不是 FAIL |
| `intervention arms per cell` | **INFO**：报出臂名数、逐格计数分布、总行数 | 把"数是多少"记录成事实，而不是判据 |

回填（`e16_writeback`）：`expected_arms = len(ALWAYS_PRESENT_ARMS)`（= 10），并把
`always_present_arms` 写进 `arm_coverage`，`note` 说明"11/12/13 臂的格子都是完整的"。

论文 §4.4：把"11–12 臂（11 为下界）"改成逐格而定的说明，并附实测分布（11/12/13、750 行、13 个臂名）
与"判据按恒在臂写"的理由。**只改这一处**：§4.4 表格的 21 行 × 10 列结构不受臂数影响。

### 18.5 校准：新增 7 类 E16 对照（合计 20 类，全部符合预期）

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1–4 | 合成"真实混形"表（3 臂 × 7 setting × 3 seed = 63 格；稠密 12 臂、低秩 11/13 臂） | 四条判据全 PASS | ✅ |
| 5 | 某格删掉一个**恒在臂**（`PCA-drop`） | 恒在臂判据 FAIL | ✅ |
| 6 | 表的总行数与 summary 声明不符 | 互印判据 FAIL | ✅ |
| 7 | 只有 62 格（缺一格） | 63 格判据 FAIL | ✅ |

**校准器又抓出我搭台的两个错**：合成格键最初用 `ETT-96/192/336` 这类**重名** setting，
`(arm, setting, seed)` 三元组互相碰撞，63 格被算成 **21** 格（第一次）与 **36** 格（第二次），
导致两条对照"看起来失败"。换成真实的 7 个 setting 名后才是 63 ✅。
**这与 §17.3 同类**：校准器的失败同样要先分清"判据错了"还是"我的 fixture 错了"。
