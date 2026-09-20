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

### 15.5 生产者列集是"超集"带来的精度风险，以及一次针对性收口

生产者的列集用"**包含标记行的那些函数**里的全部字面量键"来取，这是**超集**近似：
若某个被点名的列恰好与同一函数里**无关结构**的键同名，检查器会误判为"已被生产"，
即**漏报**（这是危险方向），而不是误报。

E16 的列集因此从 129 涨到 215 列，所以要针对 E16 实际被点名的 **7 列**逐一收口，
确认它们真的是**行**的列而不是别处同名：

| 被点名列 | 生产证据 | 判定 |
|---|---|---|
| `intervention_arm` | 字典字面量键 L1907（干预行） | ✓ 真列 |
| `leading_input_group` | 字典字面量键 L2224 | ✓ 真列 |
| `leading_input_group_label` | 字典字面量键 L2225 | ✓ 真列 |
| `leading_output_group_label` | 字典字面量键 L2232 | ✓ 真列 |
| `majority_input_group` | 字典字面量键 L2249 | ✓ 真列 |
| `seed` | 多处字典字面量键（含 L1902 干预行、L2215 解剖行） | ✓ 真列 |
| `worse_than_random_rrr_95pct_fused_mse` | **无**字典字面量键，仅 L2178 一处 | ✓ 真列——来自 `band_summary` 的 f-string 模板 `f"worse_than_{name}_95pct_fused_mse"` 以 `name="random_rrr"` 展开 |

7/7 都是真实的行列，因此本例中"超集"没有掩盖任何缺口。

**仍需记住的残余风险**：对**其它**列（尤其是只在单个函数里出现一次的列），
超集近似理论上可能掩盖缺口。彻底消除需要对每个表精确识别"被 append/write 的那个 dict 字面量"
（数据流级分析）；当前实现是可复算、可复核的近似，本节的收口表格即为对关键列的补偿。

### 15.6 扩展到 §4.2 主表，并**实测确认**一个已知盲区

检测器新增第 4 组契约——**`e14_writeback.py`（§4.2 主表，本实验的中心产物）**，
其生产者是 E14 结果表（18 列）、`e14_params.py` 的 `parameter_table.csv`、以及
E19 阶段 1 的 `level_statistics.csv`（§4.7 的 τ̂ 诊断列）。四组契约现在**全部 0 缺口**。

新增该契约后重跑了全套正对照（改动规则后**必须**重跑）：

| 对照 | 目标 | 实测 |
|---|---|---|
| 真树 | 0 缺口、exit 0 | ✓ exit 0 |
| 消费者突变（e16 历史错误列名） | 报出 | `MISSING: majority_input_group_label` ✓ |
| 生产者突变（改名一列） | 报出 | `MISSING: leading_output_group_label` ✓ |
| **新契约**消费者突变（`gate_value`→`gate_value_TYPO`） | 报出 | `MISSING: gate_value_TYPO` ✓ |

#### 15.6.1 一次差点误报为"真实缺陷"的排查（值得记下结论）

把规则放开后，检查器曾报 `e16_writeback.py` 缺 4 列：`random_band_low_fused_mse`、
`random_band_high_fused_mse`、`random_rrr_band_{low,high}_fused_mse`（且是 `row[...]`
直接下标，缺列会**抛 KeyError** 而非静默留白，看起来很像严重缺陷）。逐处核对后的结论是
**误报**：

* 生产者的 band 名取自 `bases["bands"]` 的键（`random` / `random_ambient_matched` /
  `random_rrr`），因此它产出的是 **`random_low_fused_mse`**；
* `random_band_low_fused_mse` 是**回填工具自己算出来的输出列**（`entry[...] = ...`，
  源自 `mean_of(band, "random_low_fused_mse")`），随后它遍历自己构建的表再读回来格式化 markdown。

即"输入列 `random_low_fused_mse` → 输出列 `random_band_low_fused_mse`"这个**改名是有意的**，
两侧名字不同是设计，不是缺陷。因此规则恢复为"字典字面量键 + 下标赋值键都算写出"。

#### 15.6.2 已知盲区：**pass-through 列**，以及"有别的机制兜着"这句话要说到什么程度

一个列若**从输入读出、又用同名写进输出**（pass-through，例如 `tau_hat_steps`、`test_mse`、
`gate_value`），会同时落进"读集"和"写集"而被减掉，因此**生产者漏写它时本检查器不会报告**。
这是**故意**保留的盲区：把它算进来会让上面那 4 个输出列重新变成误报，而一个会喊狼来了的
检查器无法当作硬门使用。

**该盲区已实测确认**，不是纸面推断：把 `NU_STAT` 从 `tau_hat_steps` 改成 `tau_hat_steps_TYPO`
后，检查器**没有**报出该列（同一轮里作为对照的 `gate_value_TYPO` 正常报出）。

那么谁兜住它？**需要把话说准，否则就是过度承诺**：
`check_builder_outputs.py`（空列扫描）确实能**发现**——pass-through 读不到就写不出，
输出列会整列变空而被记入 `empty_column_report.json`。但它**不会**中止流水线：

* 该脚本的 `main()` 没有任何 `sys.exit`，**永远 exit 0**，是**报告型**工具；
* 流水线对它的调用还额外带 `|| true`。

所以真实情况是"**会被记录、能被阶段 5 审校读到，但不是自动失败**"，而不是"自动兜住"。
若要让 pass-through 漏写变成硬失败，需要把空列扫描改成对**指定必须非空**的表生效
（不能一刀切为致命——§4.2–§4.7 的表里确有**按设计**合法的空格，例如只在部分 setting 成立的
诊断列）。这一步本轮**没有做**，理由是：本检查器会在 E14 收尾时**无人值守自动**作为阶段二
预检运行，任何引入误报的改动都会在我不在场时把整条链挡在门口；当前版本已在真树上验证
exit 0、且双向突变可捕获，因此**优先保持它已验证的状态**，把盲区**如实写明**而不是
用未经校验的复杂分析去换一点覆盖率。

**若后续要收口该盲区**，正确做法是**按函数做局部数据流**：在同一个函数内，
被赋值的基名（`entry`）属输出、`csv.DictReader` 的循环变量（`row`）属输入，
即可在**无跨函数同名遮蔽**干扰的前提下精确区分读写——即本轮评估后**主动推迟**的方案。

### 15.7 生产者模型的修正：单次读取器实际盖 **7** 列，而我只建模了 2 列

把工具与 `traceability_matrix.md` §5.2 的**手工**表格对表时发现一处**工具侧**的错误：
手工表格写 `read_test_generic.py` 增加 7 列，而我的检查器只把 `TEST_COLUMNS = ("test_mse","test_mae")`
（2 列）并入生产者集合。**表格是对的，我的模型是错的**——该脚本实际盖章 7 列：

```
test_mse  test_mae  nlinear_mse  nlinear_mae  gate_value  test_read_status  val_relative_difference
```

即 `TEST_COLUMNS`(2) + `BRANCH_COLUMNS`(2) + 3 列其它。后果：当前 4 组契约仍报 0 缺口
（没有消费者读那另外 5 列），但**只要将来有消费者读 `test_read_status` 等列，检查器就会误报缺口**——
对一个将**无人值守自动运行**的门来说，这是必须消掉的隐患，而不是等它发生。

修法与前几次一致：**从源码导出，而不是手写**。新增 `_subscript_store_keys()` 收集
`row["x"] = ...` 这类下标赋值键，作为"该脚本盖上去的列"，于是模型自动跟随源码
（现得 12 列，含 `checkpoint`、`recorded_val_mse`、`recomputed_val_mse` 等）。
同时过滤两类非列名：私有键（`_index`）与**全大写的环境变量赋值**
（`env["CUDA_VISIBLE_DEVICES"] = ...` 是字典赋值但不是任何表的列；这些表用 snake_case，
故全大写即环境变量）。

### 15.8 六点验证套件（每次改模型后重跑）

改动检测逻辑后**必须**重跑对照，否则"已校准"这句话就失效了。本轮的完整套件与实测：

| # | 对照 | 期望 | 实测 |
|---|---|---|---|
| 1 | 真树 | exit 0 | **exit 0** ✓ |
| 2 | e16 消费者突变（历史错误列名） | 报出 | `MISSING: majority_input_group_label` ✓ |
| 3 | e16 生产者突变（改名一列） | 报出 | `MISSING: leading_output_group_label` ✓ |
| 4 | e14_writeback 消费者突变（`gate_value`） | 报出 | `MISSING: gate_value_TYPO` ✓ |
| 5 | e17 消费者突变（改读取器盖的列） | 报出 | `MISSING: h1_cond_gt_indep_seed_majority_TYPO` ✓ |
| 6 | **盲区**：pass-through `tau_hat_steps` 改名 | **不报出**（已声明） | **无命中** ✓（与 §15.6.2 一致） |

第 5 项是新增的：它同时证明"生产者集合确实包含了读取器盖的列"与"消费者侧的改名能被抓住"。
第 6 项是**反向**对照——它确认的是工具**已知不足**的边界，与第 1–5 项同等重要：
一个只在应当报警时报警、且在应当沉默时沉默的工具，才是可依赖的门。

### 15.9 矩阵与工具的关系（避免两份真相）

`traceability_matrix.md` §5.1–§5.2 是**手工**核对的记录，§5.4 起为**工具**覆盖。
两者的关系已写清：手工表是**首次验证的历史证据**，工具是**持续保证**；
当两者不一致时**先怀疑工具**（本轮即如此：表格对、模型错，见 §15.7），
因为手工表是逐个人工确认过的，而工具的近似规则更容易出错。

---

## 16. 阶段二实参的**元数（arity）**缺陷：`--seeds 2021 2022 2023`（已修，并加了门）

### 16.1 缺陷

在 E14 尚未跑完时，用真实输入逐条预跑流水线各步的门（见 `e16_dissection/02_03_static_check_smoke.md` §4），
其中第 6 步的 SVD 阶段直接报：

```text
e18_svd_truncation.py: error: unrecognized arguments: 2022 2023     (exit 2)
```

原因是 `e18_svd_truncation.py` 的 `--seeds` **不是 `nargs` 参数**，它用
`parse_list(args.seeds, int)` 解析，而 `parse_list` 是**按逗号切分**的
（`str(raw).split(",")`）。流水线写的是 `--seeds 2021 2022 2023`，
argparse 吃掉 `2021` 后把 `2022 2023` 当成无法识别的位置参数。

**后果的严重性在于时点**：这一行在第 6 步，前面已经跑完 E14 阶段 A（411 runs）、
E14 单次 test 读取、E16（63 cells）、E17（24 runs）、E18 训练（78 runs）。
它会**在 165 个 run 之后**才炸，且是"整条链停在一个纯参数写法错误上"。

修法：改为 `--seeds 2021,2022,2023`（与 `SEEDS = (2021, 2022, 2023)` 一致）。
修后在服务器上实测：

| 形式 | 结果 |
|---|---|
| 修前 `--seeds 2021 2022 2023` | **exit 2**，`unrecognized arguments: 2022 2023` |
| 修后 `--seeds 2021,2022,2023`（配 `--verify --dry-run`） | 通过 argparse，正确报 `verify failed: 135 unresolved cells; refusing to evaluate`（exit 1）——即在 E14 未完成时**正确地拒绝**评估，跑完即可通过 |

### 16.2 为什么**之前那轮实参校核没有抓到它**——这才是要记的部分

§14.5 那轮已经逐条核对了阶段二的调用：**19 个调用、未定义参数 0 个、必填参数齐全**。
它仍然漏掉了这一条，原因很明确：

> **"这个 flag 存在"与"这个 flag 能接收这么多个值"是两件事。**

当时校核的是 *flag 存在性* 与 *required 参数是否给出*，**没有校核元数**。
这是一个方法论缺口，不是一次疏忽——所以修法不能只是改那一行，而要**把缺的那一维补进检查**。

### 16.3 新增门：`check_pipeline_invocations.py`

新增静态检查并在阶段二**加为预检**（与列名契约门并列，同样亚秒级、任何耗时步骤之前）：

* 逐个调用核对：每个 `--flag` 都在目标脚本的 `add_argument` 中声明；
* **元数**：若某 flag 后面跟了 **≥2 个裸值**，则它必须在目标脚本中声明了 `nargs`，
  否则报错并给出"请用单个逗号分隔值"的提示。

**检测力用正对照校准**（与前几轮同一纪律）：

| 对照 | 期望 | 实测 |
|---|---|---|
| 修后的真流水线（66 个 flag） | 通过 | **OK，exit 0** ✓ |
| 把 `--seeds` 那一行还原成修前的空格分隔写法 | 报错 | 精确报出 `--seeds got 3 bare values (['2021','2022','2023']) but is not declared with nargs`，**exit 1** ✓ |

并用同一条规则**扫描了整条流水线**，确认没有第二处同类问题
（扫出的其它匹配全部是注释与散文，非真实调用）。

### 16.4 边界

该门只能判断"值的**个数**与写法是否合法"，**不能**判断单个值在语义上是否正确——
例如 `--e14-results` 指向一个存在但错误的文件，它照样通过；那由 §14 的命名契约、
§15 的列名契约与各实验阶段 5 审校负责。三者覆盖的维度不同，不能互相替代。

## 17. §4.2 门值列的**产出者缺陷**：回退路径把数据集合并掉了（2026-09-20 发现并修复）

### 17.1 缺陷

§4.2 的 `g` 列优先取 `results.csv` 的 `gate_value`；**7 个复用格没有该值**（Stage-0 那批运行只写
MSE/MAE），故按设计从 `parameter_table.csv` 的 `gate_value_from_checkpoint` 回退。回退的查表键是

```python
params.get((g_arm, horizon))        # read_parameters() 的键：(臂, horizon)
```

**horizon 里没有数据集**。而 `read_parameters` 是**按 `(臂, horizon)` 聚合**的，最后取

```python
"gate_from_checkpoint": sorted(entry["gates"])[0]   # 取最小值
```

于是那 7 格拿到的是**同一 horizon 上所有数据集门值的最小者**——实测那正是 **Traffic 的门**，
而不是本格自己的门。

### 17.2 实测对照（三路取证，checkpoint 直读为仲裁）

修复前产物与修复后产物各读一遍，再**直读每个复用 run 的 checkpoint**（`metrics.csv:checkpoint` →
`torch.load(..., mmap=True)` → `sigmoid(weak_period_residual_gate).mean()` → 3 seed 取均值）：

| setting | 修复前（错） | 直读 checkpoint（真值） | 修复后产物 | 判 |
|---|---|---|---|---|
| ETTh2-96 | 0.051692 | 0.492376 | 0.492376 | ✓ |
| ETTh2-720 | 0.047175 | 0.505187 | 0.505187 | ✓ |
| ETTm2-96 | 0.051692 | 0.507025 | 0.507025 | ✓ |
| ETTm2-192 | 0.055337 | 0.207138 | 0.207138 | ✓ |
| Weather-96 | 0.051692 | 0.224901 | 0.224901 | ✓ |
| Weather-192 | 0.055337 | 0.420847 | 0.420847 | ✓ |
| Electricity-336 | 0.042203 | 0.333684 | 0.333684 | ✓ |

**7 格全错 → 7 格全对，且与 checkpoint 逐格相等**。同一缺陷也命中 `l_q1_4`/`l_q1_8` 的同 7 格
（旧值 0.072–0.093 / 0.079–0.115，修复后与 checkpoint 一致），共 **21 格**。

**反向对照（证明缺陷被准确定位，而非"顺手改对了"）**：把**修复后的预演**（`rehearse_e14_writeback.py`
的 case D/E）指向**修复前的 `e14_writeback.py`**，它精确报出：

```text
fallback-gate cells checked: 8; wrong: 24; e.g. [('ETTh1-96', "l_main='0.301'", 0.302), ...]
  <-- FAIL: the checkpoint fallback is not reading THIS cell's own gate
```

即新夹具对旧实现**报错**、对新实现**通过**——修复与检测互为正负对照。

**同一路径的其它量（一并核对）**：
- `total_params_per_horizon` 也被同一聚合影响。相位主干随通道数变化（H=192：7 通道 140191、
  Electricity 411913、Traffic 412454），故"每 horizon 一个数"本身无归属；现已**注明引用数据集**
  并同时给出全跨度；`params_constant_across_seeds` 因此前错报 `False`，实测为 **True**
  （84 个 (臂,数据集,horizon) 格在 3 seed 间零差异；`residual_params` 跨数据集也零差异）。
- `l_q1_4`/`l_q1_8` 的旧值**碰巧有时正确**（例：ETTh2-96 的池化最小恰好等于该格自身值）；
  **"部分正确"比"全错"更危险**——它不产生可见异常。

### 17.3 缺陷的**影响范围**（必须写清，避免过度撤回结论）

| 对象 | 是否受影响 |
|---|---|
| §4.2 的 `g` 列 7 格（+`l_q1_4`/`l_q1_8` 各 7 格） | **受影响，已修** |
| §4.2 的两列指标、`Δ vs phase_only`、`s` 列、`稳定超过 Golden` 列 | **不受影响**（来自 `results.csv` 的 MSE/MAE，与该键无关） |
| 主张 A–D 的判定（`claims.json`） | **不受影响**（不消费门值） |
| §4.7 的六格 ρ | **不受影响**：E19 只读 `results.csv` 的 `gate_value`（`e19_predictive_power.py:106`），与 `main_table.csv` 无依赖；修复前后逐字不变（已复核） |
| §4.4 / §4.5 / §4.6 | **不受影响**（不消费门值） |

**但有一处"读法"受影响**：§4.2.1 原先据错列写"门值接近 0 是修正器在干活的签名"（举
ETTh2-96/ETTm2-96/Weather-96 = 0.052）。用真值重算，**有增益的 18 格均值 0.2665、退化的 6 格均值 0.2019**
（main-24 口径），方向与原读数**相反**（**同一 main-24 口径**下错列为 0.1367 vs 0.2019，看似"有帮助的门更小"）。
正文已改为只报数字、不作因果解释。

> **一处必须点明的巧合（否则会被读成没改）**：**修复前后 all-28 的"退化组"均值都是 0.1458**。
> 原因是该组 10 格（main-24 的 6 格退化 + Traffic 4 格）**全部是 `results.csv` 有门值的格子**，
> 门列修复**一格都没动它们**；被改的只有"有增益"那一组（all-28：0.1367 → 0.2665，
> main-24：0.1367 → 0.2665）。两个 0.1458 **同值但含义不同**（缺陷列 vs 修复后），
> 引用时必须带上"修复前/后"与口径（main-24 / all-28）两件事。

### 17.4 修法

1. `read_parameters()` 的键改为 **`(臂, 数据集, horizon)`**，回退不再可能取到别的数据集的值；
2. 门值由 `sorted(...)[0]` 改为 **对 3 个 seed 取均值**——与 `results.csv` 路径（`np.mean(bucket["gate"])`）
   一致；即便同一格内也不再取最小值（同一格 3 seed 的门本来就不同，取最小在原理上就是错的）；
3. 参数量列加 `total_params_reference_dataset` / `total_params_per_horizon_by_dataset` /
   `total_params_per_horizon_range` 三个归属列；
4. **加了门（防止回归）**：
   - 新增 `tests/test_phaseformer_L_e14_gate_column.py`（7 个用例）——**已实测对旧实现失败**（用
     旧函数体配同一夹具跑一遍，4 条断言如实报错），故这些测试是检测器而非描述；
   - `rehearse_e14_writeback.py` 新增 case D（两数据集无 `results.csv` 门值 ⇒ 必须回退到**本格**门）与
     case E（参数表**没有** dataset 列 ⇒ 必须**不报**门值，而不是借用别的数据集的值）；
   - `audit_phase2_outputs.py` 新增两条**架构级**判据（参数表带非空 `dataset` 列、且跨 **7** 个数据集）
     与一条变体表判据（`total_params_reference_dataset` 必须填）；
5. 复算与回填：服务器上重跑步骤 3（`e14_params.py` → `e14_reuse_audit.py` → `e14_writeback.py`），
   再用 `fill_minipaper_table.py --write` 把 §4.2 主表重新回填；修复前产物存档于
   `research_runs/phaseformer_L_e14_main_v1/pregate_gate_fix/`（6 个文件），可逐格复算。

### 17.5 为什么既有检查**全都没抓到**——本轮要记的部分

| 既有检查 | 为什么无效 |
|---|---|
| `verify_minipaper_fill.py` 的 §4.2 检查 | 它做的是**论文行 vs 产物行的字符串比较**。两边同源于 `main_table.md`，**同一个错数当然"相等"** ⇒ 通过 |
| `check_column_contracts.py` | 它比的是**列名集合**。`gate_value_from_checkpoint` 这一列既在"读"集也在"写"集，被当作 pass-through 相减掉了 ⇒ 不报 |
| `audit_phase2_outputs.py`（修前） | 它只判"有门的行是否都有门值"（252 行全有）；**值本身错不错不在其判据内** ⇒ 通过 |
| `check_builder_outputs.py` | 只找**全空/几乎全空/恒定**的列；错值非空、非常量 ⇒ 通过 |
| 人工复核 §4.2 表格 | 7 个错值**都是 0.04–0.06 这个量级的合法小数**；它们与表内其它 0.14–0.22 的门值并无形态差别 ⇒ 肉眼不可见 |

**结论（方法论）**：这一类的缺陷**不能靠"值与产物是否一致"抓住**，因为产物本身就是错的。
必须有一条**独立于产出链**的取值路径（本轮是 checkpoint 直读）作为仲裁，并把判据落在
**使正确取值成为可能的架构性质**上（键里有没有数据集、样本量是多少）。这与 §16.2 的教训同型：
"flag 存在"≠"flag 能收多个值"；此处则是**"列非空"≠"列取对了格子"**。

### 17.6 边界（如实记录）

- 本轮**没有**对 §4.2 的 28 行 `g` 列做**全表**的第三路（前向重算）复核，只对 84 个有门格做了
  **checkpoint 直读 vs 产物**的逐格比对（偏差 0 格、最大相对差 < 1e-6）；§9 的两路交叉验证覆盖的是
  7 个 setting 的 seed 2021。若要更强的保证，需补一次全表前向重算——**当前未做，不声称已做**。
- `l_rcrf` 与 A1 的 `results.csv` 也带 `gate_value`，但那是 `last_rcrf_alpha` 的逐样本均值，
  **不是**本列的静态门（两臂模型无 `weak_period_residual_gate` 参数）；本文**不**把这两臂的该列纳入任何比较。
- 修复后 §4.7 的 `vs g` 列仍为 n=21（来源未变），与 §4.2 完整 28 格口径的差异已在 §4.7 表注 6 写明。

## 附：`audit_e14_stage_a.py` —— 阶段 5 的可复跑工具（2026-09-20）

阶段 A 的八条不变量**最初是手工过的前 7 个 cell**（`execution_schedule.md` 2026-09-19），
那个校核脚本当时写在临时目录、随会话消失。本轮把它落成仓库内工具，并对**整张矩阵**复跑：

```bash
python scripts/phaseformer_L/audit_e14_stage_a.py \
  --e14-root research_runs/phaseformer_L_e14_main_v1 [--json OUT.json]
```

逐格（用 `e14_read_test.locate_run` 解析——即阶段 B 读取器用的**同一把臂指纹尺子**）：

| # | 不变量 | 为什么 |
|---|---|---|
| 1 | run 目录能解析，且**唯一**（多于一个 = 两格争同一 run） | 歧义匹配会让阶段 B 读错 checkpoint |
| 2 | `metrics.csv` 存在 | 该格已完成 |
| 3 | `metrics.csv` 记了 `checkpoint` 且该文件存在 | 阶段 B 要恢复它 |
| 4 | `val_mse` 非空可解析 | 阶段 B 的复现门要用 |
| 5 | **`test_mse`/`test_mae` 为空** | 阶段 A **绝不读 test**——整个"只读一次"协议靠这条 |
| 6 | `1 ≤ epochs_completed ≤ 30` | **刻意不写成"等于请求轮数"**：早停 `patience=8` 对新格与复用格一视同仁，写成相等是早期审校脚本的误判 |
| 7 | `parameter_count` 非空 | 参数表要拿它交叉校验 |
| 8 | run 的 `config.json` 未置 `evaluate_test` | 协议面防线 |

**未跑完的格报 PENDING（exit 0），已完成格违反任何一条才 exit 1** —— 与审计器的三态约定一致。

**实测（2026-09-20 05:3x，E14 仍在跑）**：

```text
cells: 492 (declared total 492)   new: 411  reused: 81
stage A, per cell:  ok 155   pending 256   fail 0
test split read during stage A: 0 cell(s) (must be 0)
Stage A is incomplete (256 cell(s) still pending) but every finished cell satisfies all eight invariants.
```

**校准（`--self-test`，5 条断言全部成立）** —— 用自己的合成根注入四种情形：

```text
[OK] self-test clean: ok (expected ok)
[OK] self-test reads_test: fail (expected fail) -- ['test_mse is populated: stage A must never read test']
[OK] self-test zero_epochs: fail (expected fail) -- ['epochs_completed=0.0 outside 1..30']
[OK] self-test no_parameter_count: fail (expected fail) -- ['parameter_count missing']
[OK] exactly one cell flagged for reading test: 1
```

第 2 条尤其重要：它**证明"没读 test"这条判据不是空话**（真实矩阵上读出 0 是因为确实没读，
而不是因为判据写错了列）；真实 `metrics.csv` 的头 50 列里 `test_mse`/`test_mae`/`parameter_count` 都存在（已核）。

> **一次我自己的 fixture 错误**：`--self-test` 首跑时四格共用模板的 `key`，
> 于是四条判决**塌进同一个字典键**、每格都打印"最后一格"的结果（看起来像"clean 也 fail"）。
> 修法是给每个合成格一个独立 `key`。与 §17.3、§15.1 同类：**先分清是工件坏了还是我的搭台错了**。

## 附 2：完成 run 的**统计体检**（2026-09-20 05:5x，E14 仍在跑，n=176）

在阶段 A 尚未跑完时先对**已完成**的 run 做一次分布体检——若此刻发现系统性异常（例如全部早早停、
或 val 量级整体错），还来得及在阶段二消费这些数字之前处理。用的是与阶段 B 同一把尺子
（`e14_read_test.locate_run`）逐格解析。

| 检查 | 结果 |
|---|---|
| 完成 run 数 | **176** |
| `epochs_completed` 分布 | 13–30，中位 **18**；**133 格早停**、**43 格跑到 30 上限**（= 24%） |
| `val_mse` 按数据集（中位 [min, max], n） | Traffic 0.337 [0.314, 0.395] n=60；Electricity 0.122 [0.108, 0.169] n=63；Weather 0.518 [0.390, 0.610] n=47；ETTm1 0.937 [0.930, 0.945] n=6 |
| 同 setting 跨 seed 的 `parameter_count` 是否一致 | **0 处不一致**（该不一致会暴露配置写错或 run 目录错配） |
| 缺失/非有限/非正的 `val_mse` | **0** |
| 已带 test 指标的行（阶段 A 必须为 0） | **0** ✓（与阶段 A 审计独立地再次确认） |

**结论：未发现异常**。两项额外的结构性证据值得记下：①**同 setting 的三个 seed 参数量完全相同**
（这是"run 目录与 cell 正确配对"的一个强旁证，与 `locate_run` 的指纹匹配相互独立）；
②**没有任何一格在阶段 A 读过 test**，与 `audit_e14_stage_a.py` 的结论一致。

> **顺带更正我自己此前的一处措辞**：我在 §9.1.4 讨论 `COST_HINT` 为何高估 30–40% 时写的是
> "**早停主导**"。实测分布是**中位 18 轮、133/176 早停、另有 43 格（24%）跑满 30 轮**——
> "早停解释大部分高估"成立（中位远低于上限），但"主导"是**过强的说法**。以本节实测为准。

## 附 3：配对方向的**早期信号**（val-only，2026-09-20 05:5x，**不是判定**）

§4.2 的主张 A 是"PhaseFormer-L（`l_main`）相对 matched `phase_only` rerun 在 24 setting 上**不劣化**"。
在阶段二花掉十小时之前，先对**两个臂都已完成**的格子做一个配对方向的体检——**目的只有一个**：
排除"两臂接错/配对错"这类系统性接线错误（那会表现为整体、单向的偏移）。

实测（`val_mse`，逐 (setting, seed) 配对；正值 = 比 `phase_only` 差）：

| 统计量 | 值 |
|---|---|
| 可配对格数（两臂都完成） | **27**（另有 `phase_only` 30 格、`l_main` 29 格已完成，其余待跑） |
| 中位 Δ | **−0.70%**（略优于 `phase_only`） |
| 范围 | −1.90% ～ **+2.29%** |
| abs(Δ) ≤ 1% | **17/27** |
| 差于 1% 以上 | **1/27** |
| 优于 1% 以上 | 9/27 |

**读法与边界（必须与判定分开）**：

* **结论只能是"方向正常"**：中位略优、多数落在 ±1% 内 ⇒ **没有接线错误的迹象**；若两臂接错，
  应看到整体单向的大幅偏移，而这里没有。
* **它不能替代判定**：主张 A 用的是**test** 指标、**两项**指标（MSE 与 MAE）、**冻结的 1% 界**，
  且按 setting 聚合三 seed；本表是 val、单指标、未聚合。**正式判定只能由 `e14_writeback.claims.json`
  与第 7 步审计给出**，且**必须先完成阶段 B 的单次 test 读取**。
* 其中 **1/27 差于 1% 以上**是最值得盯的一项：若在 test 上仍如此，按 §4.0 应**如实报告为未达标并逐格列出**
  （主张 A 的口径本来就是"无一双指标退化"），而不是解释掉。
