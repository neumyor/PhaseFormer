# E16 · §4.4 解剖与干预 — 阶段 2/3：静态检查与冒烟

> 代码：`scripts/phaseformer_L/e16_dissection.py`
> 状态：**阶段 3 未通过**（发现 2 个真实缺陷，已退回修复）；阶段 2 通过。

## 1. 阶段 2：静态检查（通过）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `compile()` | 通过 |
| 2 | 作用域 cell 数 | **63** = 3 臂（`l_main`/`l_q1_4`/`l_q1_8`）× 7 个 test-selected setting × 3 seed |
| 3 | 逐臂计数 | `l_main` 21 + `l_q1_4` 21 + `l_q1_8` 21 |
| 4 | 复用解析 | **63/63 全部 `status=reused`**，checkpoint 全部解析到真实 E3 系 run（含一个 `attempts/002`） |
| 5 | 低秩臂的秩 | `l_q1_4` → H96:24 / H192:48 / H336:84 / H720:180；`l_q1_8` → 12/24/42/90，**与 `H/4`、`H/8` 一致** |
| 6 | 不读 test | `evaluation_split: val`、`test_split_read: false`；日志逐 setting 打印 `test rows never read` |
| 7 | 零分布规模 | `random_repeats=100`、`random_rrr_repeats=100`、`random_rrr=true`（默认） |

### 1.1 阶段 2 挡下的缺陷：作用域被张成**笛卡尔积**

原实现在未显式给 `--datasets/--horizons` 时，用"数据集集合 × horizon 集合"构造 cell，
而 7 个 test-selected setting 是**不规则**的（ETTh2 只有 96/720；Electricity 只有 336）。
交集运行会把 7 个 setting 静默放大成 **16 个**（4 数据集 × 4 horizon），
引入 9 个登记范围外的 cell（`ETTh2-192/336`、`ETTm2-336/720`、`Weather-336/720`、
`Electricity-96/192/720`）——它们既不在 §4.1 的先导集合里，又依赖当时尚未训练的 E14 格。

修复：作用域改为**显式 pair 列表**，`--datasets/--horizons` 只做**收窄**、不做扩张；
若过滤后为空则报错退出而不是静默跑空。修复后 dry-run 恰好 63 cell、全部可解析。

## 2. 阶段 3：冒烟（**未通过**，2 个缺陷）

```bash
python scripts/phaseformer_L/e16_dissection.py --e14-root research_runs/phaseformer_L_e14_main_v1 \
  --output-root research_runs/phaseformer_L_e16_smoke \
  --datasets ETTh2 --horizons 96 --seeds 2021 --arms l_main,l_q1_8 \
  --max-batches 2 --random-repeats 10 --random-rrr-repeats 10 --mem-budget-mb 2048
```

### 2.1 缺陷 1：稠密头的"未干预臂不变量"失败

```text
[train] ETTh2-96: pairs=54775 channels=7 channel_block=7 windows=7825 rows parsed=11520 (test rows never read)
[cell] l_main   ETTh2-96  seed=2021 head=dense_shared  rank_dim=720  pairs=3584 invariant=FAILED gap=4.42e-02 (588.1s)
```

该不变量要求"未干预臂必须复现模型自己记录的 fused 输出"。计划文档 §8(d) 记载了
注册代数会把 `W_dec @ encoder_bias` 重复计入，并声明对**稠密头**该 gap 为 0——
**实测为 4.42e-02，不为 0**。因此稠密头的映射（`encoder=I`、`encoder_bias=0`、
`decoder=linear.weight`）实现有误，`l_main` 的解剖与干预数值在修复前不可用。

### 2.2 缺陷 2：头的类型没有按 cell 从 checkpoint 推导

`l_q1_8`（应为 `pooled_lowrank`）被按 `shared` 稠密头构造：

```text
RuntimeError: Error(s) in loading state_dict for PhaseFormer:
  Missing key(s) in state_dict: "weak_period_residual.linear.weight", "weak_period_residual.linear.bias".
  Unexpected key(s) in state_dict: "weak_period_residual.encoder.weight", "weak_period_residual.encoder.bias",
                             "weak_period_residual.decoder.weight", "weak_period_residual.decoder.bias".
```

即 `l_q1_8` 的 21 个 cell 全部无法加载。修复方向：从该 run 的 `config.json` 的
`weak_period_residual_head_type`（或直接从 checkpoint 的键集合）推导头类型，并在加载前断言。

### 2.3 缺陷 3（工程）：设备选择与 E14 争用

日志显示 `device: cuda:0`，而当时 8 张卡正被 E14 主表矩阵占满；单 cell 在该条件下耗时 588 s。
冒烟应能显式指定设备、并在未指定时**默认不抢占**正在使用的卡。

## 3. 结论与后续

阶段 3 **不通过**：3 个缺陷（1 个数值正确性、1 个加载正确性、1 个工程）已连同完整报错回退给实现方修复。
**在修复并通过冒烟之前，E16 的 63 个 cell 不得进入正式运行**——这 63 个 cell 是 §4.4 两表（解剖表与
10 臂干预表，含新增的"随机 RRR 子空间 drop"对照）的唯一来源。

价值说明：本次冒烟用**单 cell、2 个 batch** 就发现了会污染整张 §4.4 表的两处缺陷；
若直接提交 63-cell 正式运行，会在数小时后产出全部错误数值。

---

## 4. 阶段 3 复测（修复后，2026-09-19 20:41）—— **通过**

修复内容（仅改 `e16_dissection.py` 与 `01_plan.md`）：

| 缺陷 | 真实根因 | 修复 |
|---|---|---|
| 1「稠密头不变量失败 gap=4.42e-02」 | **不是代数错误，是我（审校方）的判据定义错误**：被 gate 的 `gap` 是"闭式未干预臂在**实际跑过的 2 个 batch**（512/2785 个验证窗口）上的 fused MSE"与 run 记录的**全划分** `val_mse` 之比，子集 ≠ 全划分，该数字不证明任何事；真正做"对拍"的那一列（`..._vs_recorded_fused`）既没有被打印、也没有被 gate | 拆成**两个独立不变量**：①代数证书 = 闭式 `Original` 臂 vs **模型自己的** fused 张量（同一批窗口，逐元素），硬失败即 `SystemExit`；②对 run 指标的核对，在 `--max-batches>0` 或窗口数不等时记为 `comparable=false` + 原因，不再报警 |
| 2「`l_q1_8` 按 `shared` 头构造」 | **模型 bundle 的缓存键是 `(dataset, horizon, batch_size)`**，导致第一个臂（`l_main`）的 hyperparams 被复用到 `l_q1_8` | 缓存键改为 `(dataset, horizon, arm, batch_size)`；新增 `resolve_expected_head()` 从**该 cell 自己的** `config.json` 读取头类型（plan 阶段即校验）；`describe_branch()` 在 `load_state_dict` 前断言"构造头 = 期望头 = checkpoint 键集合"、低秩臂还断言 `head.rank == cell r` |
| 3 设备争用 | `--gpus` 默认取 `cuda:0` | 默认改为**空 = CPU**；显式索引在可见设备命名空间内解析并做越界检查；CUDA 不可用时报错而非静默回落 CPU |

另新增一项**只需数秒**的阶段 2 检查：`--dry-run --verify-checkpoint-heads`，用 `mmap` 只读 checkpoint 的键名。
本次实测输出：

```text
l_main  ETTh2-96 seed=2021 head=dense_shared   r=0  ckpt=head=dense_shared keys=1
l_q1_8  ETTh2-96 seed=2021 head=pooled_lowrank r=12 ckpt=head=pooled_lowrank keys=2
```

即缺陷 2 现在**在跑任何前向之前**就能被抓到。

### 4.1 复测结果

```text
[cell] l_main ETTh2-96 seed=2021 head=dense_shared   rank_dim=720 pairs=3584 windows=512/2785 algebra=ok(rel_gap=2.15e-10, max_abs=0.00e+00) run_metric=skipped(--max-batches 2 evaluated a subset: 512 of 2785 validation windows) (604.7s)
[cell] l_q1_8 ETTh2-96 seed=2021 head=pooled_lowrank rank_dim=12  pairs=3584 windows=512/2785 algebra=ok(rel_gap=2.97e-05, max_abs=0.00e+00) run_metric=skipped(...) (4.8s)
{"event":"finished","cells":2,"intervention_rows":23,"dissection_rows":2,"algebra_failures":0,"run_metric_failures":0,"run_metric_not_comparable":2,...}
E16_SMOKE_EXIT=0
```

- **`algebra_failures: 0`**：稠密头的 `rel_gap=2.15e-10`、`max_abs=0`，即
  `encoder=I, encoder_bias=0, decoder=linear.weight` 的映射**精确**；低秩头 `rel_gap=2.97e-05`，
  即注册代数里 `W_dec @ encoder_bias` 的重复计入，量级与既有登记（中位 6.3e-07、最差 5.5e-06）同阶。
- `run_metric=skipped` 是正确行为（`--max-batches 2` 只跑了子集），不是失败。
- `device: cpu`，未占用正在跑 E14 的 8 张卡。

### 4.2 `reference_parity_passed: false` 的解读（**不是缺陷**）

`reference_parity.json` 把 2 个 cell 的重算列与既有 `lowrank_checkpoint_information_v1` 逐字段比对：

| 字段 | 最大绝对差 | 判读 |
|---|---:|---|
| `singular_value`、`singular_value_share` | **0.0** | 精确一致 |
| `input_group_explanation`、`output_group_explanation` | 3.3e-16 / **0.0** | 精确一致 |
| `input_dictionary_r2`、`output_dictionary_r2` | 1.1e-16 / **0.0** | 精确一致 |
| `latent_variance` | **26.5** | 子集伪影 |
| `correction_energy_share` | **0.0276** | 子集伪影 |

分界很清楚：**依赖 `W`（checkpoint 的映射）的 6 个字段逐位一致**（说明解剖代数完全正确）；
而 `latent_variance` 与 `correction_energy_share` 是在**被处理窗口**上累加的二阶矩，
`--max-batches 2` 只覆盖 512/2785 个验证窗口，**必然**不等于全划分的登记值。

因此该 flag 在冒烟下为 false 属预期。据此确定**正式运行的验收判据**：

> 全量运行（`--max-batches 0`）必须在**全部 8 个字段**上给出 `reference_parity_passed: true`。
> 若仍有字段不一致，才是真缺陷，须逐一追查。

### 4.3 另一处需在正式运行中核对的项

`verdicts` 两个 cell 均为 `not_supported`。这是"跨 seed 稳定语义判定"的输出，
在**单 seed** 冒烟下必然无法满足"≥2/3 seed 第一名"的判据，属预期；
正式运行覆盖 3 seed 后才有判别意义。

## 5. 结论（更新）

阶段 2/3 **通过**。冒烟用 2 个 cell、2 个 batch 就抓出 2 个会污染整张 §4.4 表的缺陷
（其中一个是**缓存键**导致 21 个低秩 cell 全部无法加载），并新增了一项数秒级的零成本前置检查。
E16 具备进入正式运行（63 cell）的条件；因 8 卡正被 E14 占用，排期在 E14 之后。

---

## 6. 正式运行的精确验收判据（2026-09-19 追加）

### 6.1 参照产物的**实际覆盖只有 6 个 setting**

服务器核对 `research_runs/lowrank_checkpoint_information_v1/`：

| 文件 | 行数 | 覆盖的 setting |
|---|---:|---|
| `canonical_modes.csv` | 501 | **6**：ETTh2-96/720、ETTm2-96/192、Weather-96/192 |
| `semantic_alignment.csv` | 576 | 同上 **6** |

**Electricity-336 不在参照产物中**——这正是 E10 因
`Unable to allocate 13.9 GiB for shape (17344, 336, 321)` 而排除它的后果（见
`docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` §11.1）。
因此 E16 的 7 个 setting 中，**只有 6 个有可对拍的登记值**。

代码行为已核对（`e16_dissection.py:2337-2344`）：parity 遍历**参照行**并回查本次算出的 payload，
`payload is None` 时 `continue`——即缺参照的 cell 被**跳过**而不是判失败。这是正确的处理，
但必须如实披露：

> **§4.4 表注须写明**：`reference_parity` 只对 6 个 setting 成立；
> **Electricity-336 的解剖数值没有登记对照**，属于**新算而非复现**，
> 其可信度只由 §6.3 的两条不变量（代数证书 + 与 run 指标核对）支撑。

### 6.2 全量运行的验收判据

| 判据 | 期望值 | 依据 |
|---|---|---|
| cell 数 | **63**（3 臂 × 7 setting × 3 seed） | `--event plan` |
| `algebra_failures` | **0** | 不变量 1（硬失败） |
| `run_metric_failures` | **0** | 不变量 2（`--max-batches 0` 时须为 `ok` 而非 `skipped`） |
| `run_metric_not_comparable` | **0** | 同上；冒烟时为 2 是 `--max-batches 2` 的子集效应 |
| `reference_parity_passed` | **true** | 对 6 个可比 setting |
| `reference_parity.probe_cells` | **≈ 72** = 6 setting × 3 seed × 2 个低秩臂（`l_q1_4`、`l_q1_8`）× 2 个参照文件 | `canonical_modes.csv` 与 `semantic_alignment.csv` 各计一次 |
| `checkpoint_path_mismatches` | **[]** | 说明本轮的 checkpoint 与登记表的 `checkpoint_inventory.csv` 指向同一文件 |
| `intervention_rows` | 每 cell 一行 × 11 臂（10 既有 + `RandomRRR-drop`） | §4.4 干预表 |
| `test_split_read` | **false** | `evaluation_split` 只接受 `val/validation` |

> 冒烟的 `probe_cells: 2` 正是"1 个低秩 cell × 2 个参照文件"，
> 与上式的计数方式一致——这本身就是对判据表的一次数值验证。

### 6.3 其余需在阶段 5 审校的项

1. `dissection_table.csv` 的 21 行（3 模型 × 7 setting）是否齐全、`stable_semantics` 判定是否按
   "≥2/3 seed 第一名 + 输入解释率 ≥0.5 + 输出 ≥0.8 + Semantic-drop 超出同维随机 95% 区间"四条合取；
2. `RandomRRR-drop` 臂的 95% 零分布是否每 cell 抽出（`random_rrr_repeats=100`），
   以及 `random_rrr_dimension_match` 是否为真（同维对照必须真正同维）；
3. `l_main`（稠密头）与低秩探针的解剖列是否分开标注（`head_kind`）；
4. `mapped_encoder_bias_absmax` 是否随 cell 记录（低秩头注册代数里 `W_dec @ encoder_bias`
   重复计入的规模，用于解释 2.97e-05 的代数残差）。
