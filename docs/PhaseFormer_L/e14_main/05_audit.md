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
