# E14 · §4.2 主结果矩阵 — 阶段 5：结果审校

> 状态：**进行中**（早期审校已完成；全量审校待 411 个 cell 跑完 + 阶段 B 单次 test 读取）。

## 1. 早期审校（7 个已完成 cell，2026-09-19 20:18）

在只完成 7/411 个 cell 时提前审校，目的是**尽早发现系统性缺陷**——若等到 411 个 cell 跑完
（约 15 h）再发现，返工代价是数十 GPU·h。

方法：对每个已产出顶层 `metrics.csv` 的 run，逐项校验 8 条阶段 A 不变量。

| # | 不变量 | 判据 | 结果 |
|---|---|---|---|
| 1 | epoch 在预算内（含早停） | `0 < epochs_completed ≤ epochs_requested == 30` | 7/7 通过（实测 18–30，中位 28） |
| 2 | **阶段 A 绝不读 test** | `test_mse` 与 `test_mae` 均为空 | **7/7 通过** |
| 3 | `val_mse` 有限 | 非空、非 NaN | 7/7 |
| 4 | 记录了 best checkpoint | `checkpoint` 字段非空 | 7/7 |
| 5 | 跑在 GPU 上 | `device_type == cuda` | 7/7 |
| 6 | config hash 存在 | `config_hash` 非空 | 7/7 |
| 7 | 协议：lookback | `lookback == 720` | 7/7 |
| 8 | 协议：loss | `loss == huber` | 7/7 |

**结论：8/8 不变量在 7/7 个 cell 上成立，无系统性缺陷。**

## 2. 早期审校中的一次**误判**及其纠正（记录以避免重犯）

首版审校脚本把"`epochs_completed == epochs_requested`"当作不变量，据此报出 5/7 失败
（实测 18/21/22/28/28 个 epoch 对 30 的预算）。逐项追查后确认为**误判**：

- 训练使用了**早停**（`make_exp_args` 的 `patience = hyperparams.get("patience", 8)`）；
- Traffic 的验证损失在 18–28 个 epoch 处进入平台，因此早停触发；
- 这不是本实验特有的：**复用格也普遍早停**（E3 系 `shared` 臂实测完成 10/12/13/15/16/19/30 个 epoch）。

**协议一致性核对**：新建格与复用格的 `config.json` 里 `patience` 均为 `None`（即都走同一代码默认值 8），
`max_epochs` 均为 30。因此"≤30 epoch + best-val + patience=8"对两类格子**完全同协议**，
早停触发的早晚只是数据差异，不构成协议分歧。

据此把不变量改为 `0 < epochs_completed ≤ epochs_requested`，重跑后 7/7 全通过。

## 3. 全量审校计划（待执行）

1. **规模对账**：顶层 `metrics.csv` 数 = 411；`stage_a_summary.json` 的 `failed` 为空；无 `retry` 剩余项。
2. **逐格不变量**：本文件 §1 的 8 条推广到全部 411 格。
3. **协议抽查**：按臂抽查新格的 `mechanism`/`head_type`/`rank`/`pool_factor`/`gate_init`/`lr`，
   与 `01_plan.md` §2、§3 的表格逐字段比对；并确认**三种 gate 先验**的分布（新格 0.2 /
   `l_rcrf`+`a1` 0.5 / 复用格 Stage-0 值）。
4. **阶段 B 后**：用 `e14_writeback.py` 生成 `main_table.csv`、`variant_table.csv`、`claims.json`
   与 `audit.json`；核对
   - 28 行无缺（`settings_incomplete` 为空）；
   - 每格 3 seed（`n == 3`）；
   - 主张 A/B/C/D 的判定与逐格名单；
   - 复用格的 test 数值与其来源登记值**逐位相同**（复用格不得被重读改写）。
5. **异常值检查**：`test_mse`/`test_mae` 非有限值、恰好为 0、或比 Golden 好 10 倍以上的格逐个人工核。

## 4. 已具备的工具

| 工具 | 用途 |
|---|---|
| `scripts/phaseformer_L/e14_params.py` | §4.2 的**参数量列**：读每个 cell checkpoint 的参数**形状**（`torch.load(mmap=True)`，不materialize 权重），拆成 `residual_params`（`weak_period_residual.*`，含门）与 `backbone_params`，并与 `metrics.csv:parameter_count` **交叉校验**。FLOPs 明确**不报**（原文 Table 4 口径未在本仓库复现） |
| `scripts/phaseformer_L/run_phase2_after_e14.sh` | E14 之后的六步流水线（单次 test 读取 → §4.7 ρ → §4.2 回填 → E16 → E17 → E18），带**完成守卫**（拒绝在 E14 未完成时启动）与逐步退出码 |
| `scripts/phaseformer_L/e14_writeback.py` | 全量聚合 + 4 项主张判定 + 生成 §4.2 表行；已用**合成**数据跑通全部代码路径（28 行、6 个变体行、4 项主张均产出），合成产物已删除 |
| `scripts/phaseformer_L/e14_read_test.py` | 阶段 B 的单次 test 读取（先 val 门、后 test 读；幂等） |
| 本文件 §1 的 8 条不变量 | 全量审校的判据 |

### 4.1 参数量列的口径与已验证结果（2026-09-19）

minipaper §4.2 要求"参数量列按仓库 `metrics.csv` 的 `parameter_count` 口径报告（主干 + 修正器 + 门），
并单列修正器参数"。初版回填工具**只**输出指标列，漏了参数量列——已补齐并验证：

- 精确性：`e14_params.py` 由 checkpoint 形状求和得到的总参数与 `metrics.csv:parameter_count`
  在**全部已解析的 94 个 cell 上逐一相等**（`total_mismatches: []`）。
- 算术抽查：`l_main`/Traffic-96 的 `residual_params = 69,216 = 720×96 + 96`，
  Traffic-192 `= 138,432 = 720×192 + 192`，与设计一致。
- 恒定性：回填按 `(arm, horizon)` 聚合，并校验**同一 horizon 内 3 个 seed 的参数完全一致**
  （`params_constant_across_seeds`）。参数**随 horizon 变化**是设计使然（`rank=H/4` 的桥接层在
  H=720 时是 H=96 的 7.5 倍），因此不做跨 horizon 的恒定性断言。
- 全 28 setting 未解析时该列记为 `null` 并在 `audit.json` 中留因，**不用部分数据填充**。

> `e14_writeback.py` 的合成冒烟挡下 1 个缺陷：Golden 表的解析正则原为 `[A-Za-z]+`，
> 无法匹配含数字的 `ETTh1/ETTh2/ETTm1/ETTm2`，只解析出 12/28 行（Weather/Electricity/Traffic）；
> 已改为 `[A-Za-z0-9]+` 并复测为 28 行。

---

## 5. 主张 C 的两个 Golden 口径（2026-09-19 追加）

两份治理文档对"相对 Golden 的提升"给了**不同**定义，回填工具现在**同时报两个计数**，
避免把两者混为一谈：

| 口径 | 定义 | 出处 | 用途 |
|---|---|---|---|
| **`stable_beyond_golden`** | 三 seed **均值 + 样本 std** 在 **MSE 与 MAE 上都**严格低于 Golden | minipaper §4.0 主张 C（"既有严格标准"） | 主张 C 的**正式计数** |
| **`double_metric_improvement`** | 三 seed **均值**在 MSE 与 MAE 上都低于 Golden | `docs/PhaseFormer_gold_standard.md` §4（"默认只有 MSE 和 MAE 都低于对应金标准时，才能称为双指标提升"） | 金标准的基准口径，**报告性对照** |

金标准 §4 还特别提示"由于金标准只保留三位小数，差异非常小时还应结合多 seed 方差判断，
不能把舍入误差当成稳定收益"——`stable_beyond_golden`（加 std）正是对这一提示的回应，
因此取它作为主张 C 的判据是合适的，**且比金标准本来的口径更严**。

**已用合成数据验证两者确实可区分**（12 vs 13 个 setting，严格口径给出更少的计数），
说明实现没有把两个定义写成同一个条件。

**表注须同时给出两个计数**并注明各自的定义，否则读者无法判断"稳定超过"是按哪一个口径数的。

---

## 6. 复用污染事件（2026-09-19 发现并修复）——**本实验最重要的一次审校**

### 6.1 现象

在给 §4.2 补"门值 `g` 列"时（`e14_params.py` 从 checkpoint 读 `sigmoid(weak_period_residual_gate)`），
发现 `ETTh2-96` 的 seed-2023 三个低秩/稠密臂的门值是 **0.000114**，而另两个 seed 是 **0.497 / 0.492**。
逐 run 追查后在 `rank_sweep_2_multiseed_stage1_20260914_v3` 里找到一批**配置本身损坏**的运行：

```text
confirm_etth2_h96_weak_residual_p24_base_none-full_huber_lr0.5_pct100_e30_s2023_0af49db62e7c
   gate_init = 2023        <-- 把 SEED 写进了 gate init
   learning_rate = 0.5     <-- 审计网格的 500 倍（应为 0.001 或 0.0003）
   val_mse = 0.2939        <-- 正常运行约 0.2040
```

`gate_init = 2023` 经 `logit` 前被 clamp 到 `1-1e-4`，`sigmoid` 后 ≈ **1e-4**，
即**修正器被实际关闭**；再叠加 500 倍学习率，该 run 与设计完全不是同一个模型。

### 6.2 影响面（已全部查清）

| 项 | 结论 |
|---|---|
| E14 的复用清单 | **受影响**：`ETTh2-96` seed-2023 的 `l_main` / `l_q1_4` / `l_q1_8` **三个臂**都解析到了损坏 run（其余 78 个复用格干净） |
| E14 的新训格 | **不受影响**：修复前后 `new/reused` 划分完全一致（63/21、63/21、63/21、84/0、66/18、72/0，总 492），**没有任何训练格被增减或改派** |
| **E3 的权威审计** | **不受污染**：`audited_results.csv` 中 ETTh2-96 的 15 个 cell（3 seed × 5 臂）**全部**指向 `lr0.001` 的 run，对 `lr0.5` 批次的引用数为 **0**——即 §4.1/§3.4.2 引用的既有数字从未被污染 |
| E16 / E17 / E18 | 三者都从 manifest 解析 checkpoint，**已随修复一并纠正**（E16 的 dry-run 先前确实列出过损坏 run） |

### 6.3 根因

`e14_main_matrix._protocol_ok` 只校验 `lookback / loss / max_epochs / percent / period`
五个**结构性**字段，`_arm_match` 也只校验 `mechanism / head_type / rank / pool_factor` 与
"无冻结投影"。**两者都没有校验冻结超参的合法性**，于是一个结构上完全匹配、
但 `gate_init` 与 `learning_rate` 荒谬的 run 被当成合法复用格接受。
`--verify` 也拦不住：它只断言"该格能解析出来"，不断言"解析到的东西是对的"。

### 6.4 修复

1. `_protocol_ok` 增加两条合法性判据：

   ```text
   gate_init ∈ (0, 1)        开区间；超出即非法（它要过 logit），并给出"疑似把 seed 写进 gate"的提示
   learning_rate ∈ (0, 1e-2] 审计网格只用 1e-3 与 3e-4，0.5 会立刻被拒
   ```

2. 复用索引新增 `gate_init` / `learning_rate` 两个字段**逐格落盘**，供 §4.2 表注逐行披露，
   而不是笼统声称"一种协议"。
3. 重新解析并**替换运行中的 manifest**；替换前把原文件归档为
   `stage_a_manifest.prelaunch_contaminated.json`，并在新 manifest 内嵌 `correction_note`
   （列出 3 个被改的格、前后来源、以及"为何在训练进行中替换是安全的"）。

### 6.5 修复的自校验（最强的一条证据）

修复后 `ETTh2-96` seed-2023 的三格解析到：

| 臂 | 修复后 run（尾部哈希） |
|---|---|
| `l_main` | `..._lr0.001_pct100_e30_s2023_d86e6a7cfcdb` |
| `l_q1_4` | `..._lr0.001_pct100_e30_s2023_2cf889fbbff1` |
| `l_q1_8` | `..._lr0.001_pct100_e30_s2023_503f014f0124` |

而 E3 权威审计中对应的 `direct` / `q=0.25` / `q=0.125` 三行**哈希逐字相同**。
即：修复不是"换了一个能用的 run"，而是**恢复到既有权威审计所用的那三个 run**。
另确认运行时 manifest 的 `remaining suspicious reused cells = 0`。

### 6.6 对流程的修订

本轮之前，E14 的复用审计被当作"已完成"（`verify_ok`、81/81 解析）。这次事件说明
**"能解析" ≠ "解析正确"**，因此把合法性校验补进了 `_protocol_ok`，并在 §2.4 的审计清单里
加入"冻结超参合法性"一条。该缺陷若未被门值列顺带发现，会在 §4.2 表里表现为
**ETTh2-96 的三 seed 均值被一个关闭了修正器的 run 拉低、std 被放大**，
而单看指标很难察觉是数据问题而非模型问题。

---

## 7. 复用歧义审计：81 个复用格逐个枚举全部候选（2026-09-19）

§6 的污染事件暴露了一个更一般的问题：**解析器的"取第一个匹配"是一个选择，不是一次查表**。
因此补一个审计工具 `scripts/phaseformer_L/e14_reuse_audit.py`，对每个复用格枚举白名单根下
**全部**通过校验的候选 run，回答两个问题：有多少格其实有多个合法候选？这些候选彼此一致吗？

### 7.1 结果

```json
{"cells_audited": 81, "multiple_valid": 45, "disagreeing": 0,
 "not_in_scan": 0, "invalid_rejected_cells": 6, "worst_val_mse_spread": 0.0}
```

| 指标 | 值 | 含义 |
|---|---:|---|
| 审计格数 | **81** | 与 manifest 的复用格数一致 ✓ |
| 有多个**合法**候选 | **45 / 81** | 超过一半的复用格其实需要在候选间做选择 |
| 合法候选之间存在**分歧** | **0** | 关键结论：没有任何一格的选择会影响结果 |
| 被选中的 run 不在扫描结果中 | **0** | manifest 的选择全部落在受校验的候选集内 ✓ |
| 含**非法**候选（被新判据拒绝）的格 | **6** | 即 §6 的 ETTh2-96 三格 × 两 seed |
| `val_mse` 最大离散度 | **0.0** | 同一格的候选在 8 位小数上完全相同 |

### 7.2 逐例验证（确认审计不是空转）

```text
l_main ETTh2-96 s2022   n_valid = 3
   rank_sweep_2_multiseed_stage1_20260914_v3   gate=0.5 lr=0.001 val_mse=0.20330916785720965
   rank_sweep_2_multiseed_stage1_20260914_v4   gate=0.5 lr=0.001 val_mse=0.20330916785720965
   rank_sweep_2_multiseed_stage1_20260914_v5   gate=0.5 lr=0.001 val_mse=0.20330916785720965
```

三个来自不同批次根目录的候选在 `gate_init`、`learning_rate` 与 `val_mse`（**16 位有效数字**）
上完全一致——即它们是**同一配置在多次批次中被重复产出的同一结果**（确定性 seeding 下逐位可复现），
只是存放在不同根目录。因此"取第一个匹配"这一 tie-break **对这些格无实质影响**。

### 7.3 可以写进 §4.2 表注的结论

> 81 个复用格中 **45 格存在多个合法候选**；审计确认这些候选在 `gate_init`、`learning_rate`
> 与 `val_mse` 上**完全一致**（`val_mse` 离散度 0.0），因此复用集合**不依赖任何 tie-break 规则**。
> 另有 **6 个非法候选**（`gate_init` 为 seed 值、`learning_rate=0.5`）被合法性判据拒绝，
> 其所在的三格已改选到与既有权威审计哈希一致的 run（§6.5）。

这条把"我们取了第一个匹配"从**实现细节**升级为**已被验证的无关性结论**——这是本次污染事件
之后应有的补强，否则同类问题只能靠偶然发现。

### 7.4 该审计的边界

- 只覆盖 `REUSE_SCOPE` 声明的 81 格；新训格不在范围内（它们没有"候选"概念）。
- "一致"的口径是 `gate_init` / `learning_rate` / `val_mse`（8 位小数）+ 是否存在 test 指标；
  不比较 checkpoint 文件本体（哈希级比较只对 §6.5 的 3 格做过）。
- 若将来新增白名单根目录，须重跑本审计。

---

## 8. 一条工程注意（2026-09-19 记录，避免后续误读）

`run_id` **只编码 mechanism，不编码臂**：`l_main`、`l_q1_4`、`l_q1_8` 三者的目录名都是
`confirm_<dataset>_h<H>_weak_residual_p24_...`，区别只在 `config.json` 的
`weak_period_residual_head_type` 与 `weak_period_residual_rank`。
因此：

- **不能用目录名推断"某个臂跑到第几个"**。本次据此一度误判调度器停滞
  （看到 `traffic_h720_weak_residual` 仍在跑，就以为已完成的 `l_main` 又跑了一遍）；
  实际是 `l_q1_8` 的 h720 三格尚未完成。用 `done` 事件统计才是权威口径。
- 任何"按名找 run"的逻辑都必须回到 `config.json` 做指纹匹配（本线的解析器都遵守这一点，
  包括 `_arm_match` / `find_run_dir` / E16–E18 的解析器）。

记账口径（本次核对）：**41 个 run 目录 = 33 个已完成（有 `metrics.csv`）+ 8 个在跑**，
与 `done` 事件数严格相等，无重复、无孤儿目录。

---

## 9. §4.2 门值列的双实现交叉验证（2026-09-19）

门值 `g` 是用两种**互相独立**的方式得到的：

| 路线 | 做法 | 出处 |
|---|---|---|
| **A** | 从 checkpoint 的 `state_dict["weak_period_residual_gate"]` 直接 `sigmoid(...).mean()` | `e14_params.py`（`mmap` 只读形状/参数，不跑数据） |
| **B** | 加载同一 checkpoint 建模型，在**训练集**上前向一遍，读模型自身的 `learned_residual_gate()` 并取均值 | `e17_conditional_projectors.py` 的 `revin` 路线（记录于 `projector_audit.json`） |

对 7 个 test-selected setting（seed 2021）逐一比对，并**先核对两条路线用的是不是同一个 checkpoint**：

```text
setting          A(ckpt)     B(forward)  abs_diff  same checkpoint
ETTh2-96         0.497043    0.497043    1.3e-08   True
ETTh2-720        0.505212    0.505212    0.0e+00   True
ETTm2-96         0.508954    0.508954    4.3e-09   True
ETTm2-192        0.208258    0.208258    1.9e-08   True
Electricity-336  0.323671    0.323671    2.0e-08   True
Weather-96       0.224903    0.224903    6.6e-09   True
Weather-192      0.392532    0.392533    3.2e-09   True
compared: 7  worst|diff|: 1.98e-08  same checkpoint: 7/7
```

- **7/7 用的是同一个 checkpoint**（路径 `realpath` 级相等），因此这是对**同一数值**的两次独立计算，
  不是两次不同实验的比较；
- 两路结果的**最差绝对差 1.98e-08**，即 float32 舍入量级。

结论：§4.2 的 `g` 列既可以从 checkpoint 直接取（无需数据、无需 test），也可以由前向重算，
两者一致；且它同时给 §4.7 的 ρ-vs-`g` 提供了可靠的 `g` 序列。

**边界（如实记录）**：`lowrank_checkpoint_information_v1/conditional_rrr_alignment.csv` 也带
`gate_mean` 列，可作第三个来源，但它的 cell 是**秩约束**的 q∈{1/16,1/8,1/4} 运行（`checkpoint_path`
指向 `rank_sweep_2_stage1` 的对应 run），与 `l_main` 稠密头**不是同一个 checkpoint**，
因此它佐证的是**方法**而非本列的具体数值，不纳入上面的比对。

---

## 10. 逐格臂指纹审计（2026-09-19，53 个已完成/在跑的 cell）

§1 的早期审校只覆盖前 7 个 cell 与 8 条不变量。本轮把审计扩到**全部已产出 config 的 cell**
（53 个 = 45 已完成 + 8 在跑），并新增一条更强的判据：**每个 config 必须恰好匹配一个声明的臂**。

### 10.1 结果

| 检查项 | 结果 |
|---|---|
| 扫描的 config 数 | **53**（新训格共 411） |
| **匹配到 ≠ 1 个臂的 config** | **0** —— 指纹无歧义 |
| 未通过 `_protocol_ok` 的 config | **0**（含本轮新加的 `gate_init ∈ (0,1)` 与 `lr ≤ 1e-2` 判据） |
| **在阶段 A 读了 test 的 cell** | **0** |

### 10.2 逐臂实测超参（与设计逐字段一致）

```text
l_main      [('shared',        None, None, 0.2, 0.001)]
l_q1_4      [('pooled_lowrank',  24, 1, 0.2, 0.001), ('pooled_lowrank',  48, 1, 0.2, 0.001),
             ('pooled_lowrank',  84, 1, 0.2, 0.001), ('pooled_lowrank', 180, 1, 0.2, 0.001)]
l_q1_8      [('pooled_lowrank',  12, 1, 0.2, 0.001), ('pooled_lowrank',  24, 1, 0.2, 0.001),
             ('pooled_lowrank',  42, 1, 0.2, 0.001), ('pooled_lowrank',  90, 1, 0.2, 0.001)]
l_rcrf      [('shared',        None, None, 0.5, 0.001)]
phase_only  [(None,            None, None, None, 0.001)]
```

逐项判读：

1. **秩正是 `H/4` 与 `H/8`**：`l_q1_4` 的 24/48/84/180 对应 H=96/192/336/720；`l_q1_8` 的
   12/24/42/90 同构。`pool_factor` 恒为 1。**没有任何一格用了绝对秩**（绝对秩只属于 §4.6 行 5）。
2. **三种 gate 先验在实测中同时出现**：`l_main`/`l_q1_4`/`l_q1_8` 为 **0.2**（D-2 新格默认），
   `l_rcrf` 为 **0.5**（`rcrf_nlinear_plain` preset 自持），`phase_only` 无门。
   这正是 §4.2 表注必须写明的三档，现在有**实测证据**而非仅有设计说明。
3. `l_rcrf` 的 `head=shared` 来自 preset 自身（`rcrf_nlinear_plain` 内部定义 `shared`），
   而非 `arm_command` 注入——与冒烟时的观察一致。

### 10.3 仍未覆盖的部分

- 53/411 格；`a1` 与 `phase_only` 的行尚未开始（按成本降序，它们在 Traffic 的最后）。
- 「恰好匹配一个臂」只证明指纹不歧义；它**不**证明超参是"正确"的那一套（那由 §10.2 的
  逐臂对照与 §6 的污染事件共同覆盖）。
- 尚未审计 `a1`（`gold_combo_reliability_s2`）的 gate 先验是否为 preset 的 0.5——该臂
  训练开始后须补一条同样的检查。

---

## 11. §4.2 的「来源/披露」列实现（2026-09-19）

minipaper §4.2 表头的最后一列是**来源/披露**，要求逐格标注复用来源并显式披露
test-set selection。首版回填工具把这一列留空——是一个实打实的缺口（表列存在但无内容）。
现已补齐：由 manifest 的 `source.root` 与 `status` 逐格生成，markdown 与 CSV 共用**同一个字符串**
（在行装配处算一次，避免两处不一致）。

实测输出（真实 manifest + 合成指标，仅验证文字逻辑）：

```text
ETTh2-96       -> phase_only=复用 3/3（rank_sweep_2_stage1/top2_direction_retention_v1）；phase_only 属 test-selected 集合；
                  L=复用 3/3（rank_sweep_2_multiseed_stage1_20260914_v3/…_v4/…）；L 属 test-selected 集合
ETTh2-192      -> phase_only=新训 3/3；L=新训 3/3
Electricity-336-> phase_only=新训 3/3；L=复用 3/3（…_repair_v1/…_v3/rank_sweep_2_stage1）；L 属 test-selected 集合
Traffic-96     -> 探索性附录，不进入判定；phase_only=新训 3/3；L=新训 3/3
```

三条判读：

1. **复用的 setting 逐个列出**其来源根目录，读者可自行追溯（而不是笼统写"部分复用"）；
2. **"属 test-selected 集合"只在真正复用该臂时出现**——`Electricity-336` 的 `phase_only` 是新训
   （E8 从未覆盖该格），因此**不**带该标注，而它的 `L` 是复用，**带**该标注。这正是 per-arm 粒度
   披露的意义；
3. Traffic 行同时带"探索性附录，不进入判定"。

**边界**：本轮只验证了文字生成逻辑（指标为合成数据）；真实的复用来源与状态取自真实 manifest，
是可信的；最终表内数字仍来自 E14 的正式运行。

---

## 12. A1 臂的门先验核对（2026-09-19 补做，关闭 §10.3 的遗留项）

§10.3 登记了一项待办：`a1`（`gold_combo_reliability_s2`）训练开始后须补一条与其它臂同类的
逐臂检查。该臂已于非 Traffic 阶段开始训练，现核对：

```text
a1 cells with configs: 2
    ('Electricity', 336)  [(gate=0.5, lr=0.001, head=None, mechanism='gold_combo_reliability_s2')]
    ('Electricity', 720)  [(gate=0.5, lr=0.001, head=None, mechanism='gold_combo_reliability_s2')]
```

判读：

1. **gate 为 0.5，不是 0.2**——`gold_combo_reliability_s2` 由 preset 自持门先验，
   `arm_command` 的 `gate_init=0.2` 注入只对 `mechanism == "weak_residual"` 生效。
   这与 D-2 的"三种 gate 先验"完全一致，**第四种先验不存在**。
2. `head=None`：A1 不使用 `weak_period_residual_head_type` 覆盖（其融合由 RCRF 决定），
   因此不吃 `shared`/`pooled_lowrank` 的任一分支。
3. `lr=0.001` 与 D-2 的新格默认一致。

至此 §4.2 表内**六种臂的实测超参全部核对完毕**：`phase_only`（无门）、
`l_main`/`l_q1_4`/`l_q1_8`（gate 0.2）、`l_rcrf`（gate 0.5，preset）、`a1`（gate 0.5，preset）。

## 13. 滚动式阶段 A 不变量审计（2026-09-19，72 个已完成格）

Traffic 阶段（60 格）全部结束后、非 Traffic 阶段进行中做一次滚动审计：

```text
new cells declared: 411 | completed: 72 | running: 8
by arm: {'a1': 3, 'l_rcrf': 13, 'l_q1_4': 15, 'l_main': 15, 'l_q1_8': 14, 'phase_only': 12}
fingerprint/protocol failures: 0
invariant violations: NONE
```

即：72 个格上**指纹无歧义（0 失败）、协议字段全合法（0 失败）**，且 7 条不变量
（epoch 在预算内、**阶段 A 未读 test**、`val_mse` 有限、记录了 best checkpoint、跑在 CUDA 上、
`lookback=720`、`loss=huber`）**无一条违反**。六个臂至此都已开始产出。

该审计是**滚动**的：随着 run 完成可随时重跑，用于在长任务中途持续确认而非只做首尾两次。

---

## 14. 跨阶段产物命名契约：一处会导致阶段 2 第 6 步失败的缺陷（已修）

阶段 2 是**无人值守**的长链（E14 阶段 B → E19 → E14 汇总 → E16 → E17 → E18），
因此本轮在等待阶段 A 的空档做了两件事：**逐条校核每个调用的实参**，以及
**校核阶段之间传递的文件名**。第二项查出一个真实缺陷。

### 14.1 缺陷

两个"读 test"脚本的产物命名**不对称**：

| 脚本 | 产物 | 位置 |
|---|---|---|
| `e14_read_test.py` | **就地**填充 `results.csv` 的 `test_mse`/`test_mae` | `<output_root>/results.csv` |
| `read_test_generic.py`（E17/E18 用） | 另写**同级副本** `<results>.with_test.csv` | `results.with_test.csv` |

而 `run_phase2_after_e14.sh` 第 6 步把 `--e14-results '$E14_ROOT/results.with_test.csv'`
交给 `e18_writeback.py`。经核实：

* `grep -c with_test scripts/phaseformer_L/e14_read_test.py` → **0**，该文件从不生产此名；
* `e14_read_test.py` 的写盘路径是 `results_path = output_root / "results.csv"`（第 1572 行）；
* `read_test_generic.py` 的输出是 `results_path.with_suffix(".with_test.csv")`（第 358 行）。

即第 6 步实参指向一个**永远不会存在**的文件。后果不是"缺一列"，而是
`read_csv` 抛 `FileNotFoundError` 使第 6 步整体失败——**在第 1–5 步已经耗时数小时之后**。

同一误解还留在了 `e18_writeback.py` 的 `Usage` 示例（第 31 行），一并修正为 `results.csv`。

### 14.2 修复：把契约声明一次，而不是重复三处

根因是同一个路径在**第 2、3、6 步各写了一遍**，因此可以互相漂移。修法是引入单一来源：

```bash
# e14_read_test.py 就地填充 results.csv，而 read_test_generic.py 另写
# results.with_test.csv —— 这是契约，不是风格选择。
E14_TEST_CSV="$E14_ROOT/results.csv"
```

第 2、3、6 步全部改用 `$E14_TEST_CSV`。E17/E18 自己的 `--results` 仍为
`results.with_test.csv`（它们确实由 `read_test_generic.py` 生产），保持正确。

### 14.3 顺带加上的快速失败闸门

既然该缺陷的代价"在最后一步才显现"，就在**最便宜的第 1 步**加闸门：读 test 是纯推理，
几分钟即完成；而第 4–6 步要跑 63+24+78 个 run。因此让第 1 步在产物缺少
`test_mse` 时立即中止，远优于在数小时后才发现。

闸门逻辑（对每个 fixture 实测通过，**在服务器 gawk 上复测通过**，见 §14.4）：

```bash
test -s "$E14_TEST_CSV" && awk -F, '...c==0||t==0||n!=t -> exit 1' "$E14_TEST_CSV"
```

四类 fixture 的实测结果：

| fixture | 期望 | 实测 |
|---|---|---|
| 文件不存在 | 失败 | exit 1 ✓ |
| 全部行 `test_mse` 非空 | 通过 | exit 0，`rows=2 with_test=2` ✓ |
| 一行 `test_mse` 为空 | 失败 | exit 1，`rows=2 with_test=1` ✓ |
| 整列 `test_mse` 缺失 | 失败 | exit 1 ✓ |

### 14.4 为什么要在服务器上再测一次

闸门是 `awk` + shell 转义写的，而**转义错误在静态检查里完全看不见**——`bash -n` 通过、
`python -m compile` 也通过，只有真正执行才会暴露（本会话已多次被转义/引号问题咬到）。
本地是 BSD awk、服务器是 **gawk**，两者对 `\$`、`\"` 的处理路径不同，所以
用**从已同步脚本里抽出来的同一段原文**在服务器上跑同一组 fixture，结果与本地一致。

### 14.5 实参校核：19 个调用、0 个未定义参数

对阶段 2 的每个 `$PY scripts/…` 调用，抽取其 `--flag` 并与目标脚本 `add_argument`
的定义集求差：

```
OK  e14_params.py          invocations=1 flags=2 defined=4
OK  e14_reuse_audit.py     invocations=1 flags=2 defined=3
OK  e14_writeback.py       invocations=1 flags=5 defined=6
OK  check_builder_outputs  invocations=4 flags=2 defined=4
OK  e16_dissection.py      invocations=2 flags=6 defined=28
OK  e16_writeback.py       invocations=1 flags=3 defined=3
OK  e17_conditional.py     invocations=2 flags=4 defined=24
OK  read_test_generic.py   invocations=2 flags=2 defined=13
OK  e17_writeback.py       invocations=1 flags=3 defined=3
OK  e18_negative.py        invocations=2 flags=5 defined=13
OK  e18_svd_truncation.py  invocations=1 flags=6 defined=14
OK  e18_writeback.py       invocations=1 flags=4 defined=4
未定义参数：0/12
```

并逐脚本核对了**必填参数**（`required=True`）是否都已提供：
`e14_read_test`(manifest) / `read_test_generic`(results) / `e14_writeback`(manifest,results,golden,output-root)
/ `e18_writeback`(results,e14-results,truncation,output-root) / `e19_predictive_power`(stats,results,output-root)
/ `e16_writeback`(intervention,dissection,output-root) / `e17_writeback`(results,output-root)
/ `e14_params`,`e14_reuse_audit`(manifest,output-root) / `check_builder_outputs`(csv) —— 全部已提供。

**一个例外**：`test -s` 那类"参数存在但语义需人来判断"的问题（如 `--verify` 在各脚本中
含义不同，见脚本文末注释）无法由本项校核覆盖，只能靠逐条语义核对，已在脚本内以注释固化。

### 14.6 全部阶段边界的文件名核对（同一缺陷类别的系统排查）

§14.1 查出的缺陷属于**一类**问题（"上游产物的名字 ≠ 下游引用的名字"），因此对阶段 2
的**每一处**跨阶段引用都做了生产者↔消费者核对，而不只是修掉那一处：

| 被消费的文件 | 消费者（步） | 生产者 | 写入点 | 结论 |
|---|---|---|---|---|
| `$E14_ROOT/results.csv` | 2, 3, 6 | `e14_read_test.py` | `output_root / "results.csv"` L1572 | ✓（本轮修正点） |
| `$E19_ROOT/level_statistics.csv` | 2, 3 | `e19_predictive_stats.py` | E19 阶段 1 已产出（本会话已验证） | ✓ |
| `docs/PhaseFormer_gold_standard.md` | 3 | —（手工文档） | 存在，3599 B | ✓ |
| `$E16_ROOT/dissection_table.csv` | 4 | `e16_dissection.py` | `output_dir / …` L2674 | ✓ |
| `$E16_ROOT/intervention_table.csv` | 4 | `e16_dissection.py` | `output_dir / …` L2675 | ✓ |
| `$E16_ROOT/*_44.csv` | 4 | `e16_writeback.py` | L323–324 | ✓ |
| `$E17_ROOT/results.csv` | 5 | `e17_conditional.py` | `--results-name` 默认 `results.csv` | ✓ |
| `$E17_ROOT/results.with_test.csv` | 5 | `read_test_generic.py` | `results_path.with_suffix(".with_test.csv")` L358 | ✓ |
| `$E17_ROOT/projectors/projector_audit.json` | 5 | `e17_conditional_projectors.py` | `output_dir / …` L1003 | ✓ **已存在**（33572 B） |
| `$E17_ROOT/conditional_table.csv` | 5 | `e17_writeback.py` | L204 | ✓ |
| `$E18_ROOT/results.csv` | 6 | `e18_negative.py` | `out_root / "results.csv"` L871 | ✓ |
| `$E18_ROOT/svd_truncation_table_28.csv` | 6 | `e18_svd_truncation.py` | `output_root / …` L932 | ✓ |
| `$E18_ROOT/negative_table.csv` | 6 | `e18_writeback.py` | L295 | ✓ |

两点与 E14 那处不同的、需要写下来的结论：

1. **`projector_audit.json` 没有生产者被编入流水线**，这是**有意**的而非遗漏：
   `e17_conditional_projectors.py` 属于**冻结阶段**，其产物（7 个 setting 的
   `Q1*/Q1COND/Q1INDREVIN` 基向量 + `projector_audit.json`）是 §4.5 要**冻结**的对象，
   训练期不得重算。已在服务器核实该目录 9 月 19 日 19:57 生成、`projector_audit.json`
   33572 B 存在，路径与第 5 步实参**逐字一致**，故契约成立。
   （代价是这条依赖是隐式的：若该目录被删，第 5 步会在训练之后才失败。已记于此表备查。）

2. **第 5 步的 `--stage a` 之后又显式跑一次 `--stage assemble`** 是**冗余但无害**的：
   按 `--stage` 的 help，`a` 的定义就是"train the 24 new cells, **then assemble**"。
   重复 assemble 是幂等的，且顺序上位于 `read_test_generic.py` **之前**，
   因此不会覆盖带 test 的 `results.with_test.csv`。保留它的理由是：即使 `a` 在
   部分 run 失败时仍装配出部分矩阵（脚本 L1343 注释），这一步也能给出确定的失败点。

**未覆盖项（需诚实说明）**：本项核对的只是"名字对得上"。它不能证明**内容**语义正确
（例如某个 CSV 的行键与下游 join 键是否一致、单位是否相同）。这一层只能靠各实验的
阶段 5 审校按设定的验收阈值逐项判断，本表不替代它。

---

## 15. 列名读写契约的静态检查（把"手工发现的一个 bug"升级为一类 bug 的检测器）

§14 的缺陷类别是"**消费者点名的列，生产者从来没写过**"。这不是孤例而是**一类**缺陷，
手工核对不可扩展，因此新增 `scripts/phaseformer_L/check_column_contracts.py`，
并在阶段二加为**预检**（`--strict`，静止、亚秒级、在任何耗时步骤之前）。

E16 那个真实的 `majority_input_group_label` 缺陷（生产者写的是
`leading_input_group_label`）后果是**静默清空** §4.4 的一列，而且只在解剖跑完之后才暴露——
正是这个检测器要挡的东西。

### 15.1 方法

* **生产者列集从其自身源码导出**，而不是从文档或我的记忆：要么取模块声明的
  `RESULTS_FIELDS`/`TABLE_FIELDS`，要么取**真正构造那些行的函数里的字面量键**。
  因此它无法与代码漂移。
* **消费者列集取自它真正读取的名字**（`row["x"]` 的 load、`row.get("x")`）。
* 报告"被点名但没有任何生产者提供的列"。

### 15.2 三次精度修正（都是为了不"狼来了"）

工具第一版报了 **49** 个"缺口"，其中绝大多数是假阳性；一个会喊狼来了的检查器等于没有。
逐条查清并修掉：

| 现象 | 真实原因 | 修正 |
|---|---|---|
| `mse`/`mae`/`verdict`/`row`/`target` 等被报缺 | 这些是回填工具**写出**的列，我却在统计读数 | 按 AST 上下文区分 **load / store**，并用"字典字面量的键 = 写出"剔除 |
| `direct_mse`/`joint_mae` 等 8 个被报缺 | 它们是**内部枢轴**列，用 f-string 构造（`entry[f"{arm}_mse"]`），字面量扫描看不见 | 把 f-string 写键收集为**模式**，匹配到的读数判为"内部构造" |
| `revin` 被报缺 | 它由 `record["revin"] = {...}` 定义，是**下标赋值**而非字典字面量 | 生产者侧同时收集下标赋值键 |

修正后：**3 个契约全部 0 缺口**。

### 15.3 检测器本身的验证（正对照）

只在"已修好的代码"上跑出 0 缺口**不能说明任何事**——一个永远返回 0 的脚本也能做到。
因此在服务器上做了**双向**正对照：

| 对照 | 做法 | 期望 | 实测 |
|---|---|---|---|
| **真树** | 原样 | 0 缺口、exit 0 | 0 缺口、exit 0 ✓ |
| **消费者突变** | 把 `leading_input_group_label` 改回历史上那个错误名 | 报出该列 | `MISSING: majority_input_group_label` ✓ |
| **生产者突变** | 把生产者的一列改名 | 报出被点名的列 | `MISSING: leading_output_group_label` ✓ |
| **门的退出码** | 上述两例加 `--strict` | exit 1 才能挡住阶段二 | 突变树 exit 1、真树 exit 0 ✓ |

即：该检测器**确实能抓住它被写出来要抓的那个真实缺陷**，且方向双向。

### 15.4 诚实边界（必须写清楚）

本项只核**名字**。它**不能**证明：

* 两列语义/单位一致，或 join 键能对上（名字对而含义错，它照样通过）；
* 某列虽然存在但**恒为空**——那是 `check_builder_outputs.py`（空列扫描）与各实验
  阶段 5 审校的职责；
* 动态拼出的 JSON 结构里除已知模式外的键。

因此它是**防线中的一道**，不替代阶段 5 审校。写明这一点，是为了避免后续把
"contract 检查通过"误读成"产物正确"。
