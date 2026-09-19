# E17 · §4.5 条件性学习 — 阶段 2/3：静态检查与冒烟（投影器部分）

> 代码：`scripts/phaseformer_L/e17_conditional_projectors.py`｜产物：`research_runs/phaseformer_L_e17_conditional_v1/projectors/`
> 状态：**投影器部分已完成**（训练部分待 E14 让出 GPU）

## 1. 阶段 2：静态检查

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `compile()` | 通过 |
| 2 | `--dry-run` 不导入 numpy/torch，可打印 7 个 setting / 14 个待写文件 | 通过（`{"event":"dry_run","settings":7,"artifact_files":14}`） |
| 3 | `reference_exists` 判定 | 6 个 E8 setting 为 `true`，Electricity-336 为 `false`（E8 未覆盖）——**符合预期** |
| 4 | 投影器文件约定与 E8 产物一致 | 通过（本地读取 `ETTh2_96_Q1.npy` 的 numpy header：`<f8`、C-order、shape `(720, 1)`） |
| 5 | 从不读 test | 通过（`{"reads_test_split": false}`；标准�化路线读取止于验证边界） |
| 6 | 复用格的 checkpoint 解析 | **首次运行失败并修复**，见 §3 |

## 2. 阶段 3：冒烟/正式运行结果（投影器）

```bash
python scripts/phaseformer_L/e17_conditional_projectors.py \
  --output-dir research_runs/phaseformer_L_e17_conditional_v1/projectors \
  --e14-root research_runs/phaseformer_L_e14_main_v1 \
  --reference-dir research_runs/top2_direction_retention_v1/projectors --num-workers 4
```

结果：**7/7 完成，`E17_PROJ_EXIT=0`，`reproduction_failures: []`**，wall-clock 约 10 分钟（CPU）。

**独立路线的复现门（对 E8 已发布的 6 个投影器）**：`abs_cos = 1.0`（6/6，最后一个为 1.0000000000000002，
即浮点精度内的完全一致）。这验证了独立 RRR 方向 1 的实现与 E8 完全同源。

## 3. 阶段 2 挡下的缺陷（若直接开跑会全 7 格失败）

`resolve_e14_run_dir` 只从 `e14_root/runs/*` 里找 `l_main`，但 **E14 对复用格不复制产物**——
它在原地复用 E3 系的 run，`stage_a_manifest.json` 里记的是
`status="reused"` + `source.run_dir` 指向 `rank_sweep_2_stage1/...`。
而 E17 需要的 7 个 setting **全部是复用格**，因此首次运行在第一个 setting 就报
`no E14 l_main run directory ...`（`E17_PROJ_EXIT=1`）。

修复：解析器**先查 manifest 的 `source.run_dir`**（并仍用 `e14_main_matrix._arm_match` 校验，
且优先于 glob 结果），再退回 glob。修复后 7/7 解析成功。

## 4. 本轮最重要的科学发现：`D_cond` 与 `D_ind` 的首方向**几乎相同**

每个 setting 同时记录了条件 RRR 与独立 RRR 首方向的 `|cos|`：

| setting | `pairs` | `gate_mean` | **`abs_cos_cond_vs_indep`** |
|---|---:|---:|---:|
| ETTh2-96 | 53,760 | 0.4970 | **0.99962** |
| ETTh2-720 | 50,176 | 0.5052 | **0.99997** |
| ETTm2-96 | 234,752 | 0.5090 | **0.99909** |
| ETTm2-192 | 234,752 | 0.2083 | **0.99993** |
| Weather-96 | 756,672 | 0.2249 | **0.99938** |
| Weather-192 | 755,328 | 0.3925 | **0.99981** |
| **Electricity-336** | 5,567,424 | 0.3237 | **0.00659** |

即：把残差目标从"独立"（`y − x_last`）换成"条件性"（`y − y_φ`）之后，
**6/7 个 setting 的首方向几乎不动**（`|cos| ≥ 0.9991`），
而 **Electricity-336 的首方向几乎正交**（`|cos| = 0.0066`）。

### 4.1 对 §4.5 的直接后果（必须写进表注）

§4.5 的预测是"冻结**条件性**方向应明显好于冻结独立方向"。由于两个方向在 6/7 个 setting 上
**就是同一个方向**，这两条冻结臂在那 6 个 setting 上**按构造等价**——它们的差异是
浮点级投影器差别，不是"目标定义"的差别。因此：

1. **该对照的真实检验力集中在 Electricity-336 一个 setting 上**（两个方向近正交）；
2. 其余 6 个 setting 的结果**不能**用来支持或否定"独立目标错位"假设——它们是同一臂的两次实现；
3. 这本身是一个**支持命题 2 的正面发现**：首方向是**数据的性质**，而不是**目标定义的产物**；
   只有在 `λ_1/Σλ` 最低（Electricity-336 为 0.662，7 个 setting 中最低）且 `pred_dims_90 = 4`
   的 setting 上，目标定义才足以改变首方向。

### 4.2 与既有证据的一致性核对（通过）

| 核对 | 本项目实测 | 既有登记 | 结论 |
|---|---|---|---|
| `gate_mean`（ETTm2-192） | 0.2083 | Stage-0 冻结 gate **0.2** | 一致 |
| `gate_mean`（Weather-96） | 0.2249 | Stage-0 冻结 gate **0.2** | 一致 |
| `gate_mean`（ETTh2-96 / ETTh2-720） | 0.4970 / 0.5052 | Stage-0 冻结 gate **0.5** | 一致 |
| `gate_mean`（ETTm2-96 / Weather-192） | 0.5090 / 0.3925 | Stage-0 冻结 **0.5 / 0.5** | ETTm2-96 一致；Weather-192 偏低（0.393 vs 0.5） |
| `gate_mean`（Electricity-336） | 0.3237 | 先导记录"0.433→**0.332**"（唯一系统性关闭） | **一致（0.324 vs 0.332）** |

`gate_mean` 是**用训练好的 `l_main` checkpoint 在训练集上前向得到的每样本门值均值**，
与 Stage-0 的**初始化** gate 不必相等（训练会移动它），因此上表是"初始化吻合 + 训练位移"的
合理图景；Weather-192 的位移较大（0.5 → 0.393）已如实记录。

## 5. 结论

投影器部分阶段 2/3 通过（并修复 1 个会使 7/7 格全灭的缺陷）。训练部分（24 runs）待 E14 让出 GPU
后启动；届时可直接复用这 7 个投影器与已解析的复用格。

---

## 6. 训练侧静态检查（`--stage plan --verify`，2026-09-19）

```bash
python scripts/phaseformer_L/e17_conditional.py --stage plan --verify \
  --output-root research_runs/phaseformer_L_e17_plan
# → {"event": "plan_only", "cells": 84, "new_runs": 24}
```

| # | 检查项 | 结果 |
|---|---|---|
| 1 | cell 总数 / 新训数 | **84 / 24**（21 冻结条件 + 3 冻结独立补 Electricity-336），与计划一致 ✓ |
| 2 | 投影器齐备 | 7/7 `Q1COND.npy` 与 7/7 `Q1.npy`（含 Electricity-336 的新建 `Q1`） |
| 3 | 走正确的 runner | 24 条命令全部经 `scripts/run_top2_direction_retention.py`——因为 `search_phaseformer.py` **没有** `--basis`，冻结子空间只能由该 wrapper 安装（同时必须给 `weak_residual_projection=frozen_subspace` override，否则 `PhaseFormer.__init__` 抛错） |
| 4 | 协议常数 | 24/24 为 `--lookback 720 --period 24 --max-epochs 30 --loss huber --percent 100 --require-cuda --resume` |
| 5 | **逐 setting 冻结超参** | 与 E8 的 `FROZEN` 表逐格一致：ETTh2-96 **0.5/1e-3**、ETTh2-720 **0.5/1e-3**、ETTm2-96 **0.5/3e-4**、ETTm2-192 **0.2/1e-3**、Weather-96 **0.2/3e-4**、Weather-192 **0.5/1e-3**；Electricity-336 为**新格**故用 D-2 默认 **0.2/1e-3** |
| 6 | 不读 test | 24/24 命令**无** `--evaluate-test` |
| 7 | 臂标识 | 用新的 `weak_residual_projection_arm` 标签（`e17_frozen_conditional_direction_1` / `e17_frozen_conditional_direction_1`），**不复用** E8 的 `keep_direction_1`，避免两个不同投影器共用一个臂名（E8 的 config 不记录 basis 路径或 sha256） |

### 6.1 需要披露的口径裂缝（已知，不阻塞）

`direct` 与 `PhaseFormer-L（联合）` 两列来自 E14 的 `l_main`；其中**新格**用 D-2 默认
`(0.2, 1e-3)`，而两条冻结臂用上表的 per-setting 冻结值。因此"冻结 vs 联合"的对比在
**超参不完全相同**这一层上不是严格配对的（§4.2 的 D-2 披露已覆盖该问题，此处为同一裂缝）。
代码在每格的 `protocol.frozen_hyperparams_source` 与 `note` 列记录了来源，供表注使用。

## 7. 结论

投影器与训练两侧的静态检查均通过；24 个新 run 具备进入正式运行的条件（待 E14 让出 GPU）。
