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
| `scripts/phaseformer_L/e14_writeback.py` | 全量聚合 + 4 项主张判定 + 生成 §4.2 表行；已用**合成**数据跑通全部代码路径（28 行、6 个变体行、4 项主张均产出），合成产物已删除 |
| `scripts/phaseformer_L/e14_read_test.py` | 阶段 B 的单次 test 读取（先 val 门、后 test 读；幂等） |
| 本文件 §1 的 8 条不变量 | 全量审校的判据 |

> `e14_writeback.py` 的合成冒烟挡下 1 个缺陷：Golden 表的解析正则原为 `[A-Za-z]+`，
> 无法匹配含数字的 `ETTh1/ETTh2/ETTm1/ETTm2`，只解析出 12/28 行（Weather/Electricity/Traffic）；
> 已改为 `[A-Za-z0-9]+` 并复测为 28 行。
