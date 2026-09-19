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

---

## 7. §4.4 回填兼容性核对：产物 schema 已捕获（2026-09-19）

§4.4 的两张表要能被**机械回填**，前提是 E16 的产物列覆盖 minipaper 要求的每一项。
本轮用一个 2-cell 的 schema 冒烟（`--datasets ETTh2 --horizons 96 --seeds 2021 --arms l_main,l_q1_8
--max-batches 2`，CPU，约 13.6 min）把两张表的**真实列名**取了出来并逐项对照。

### 7.1 `intervention_table.csv`（98 列）

| minipaper §4.4 干预表要求 | E16 对应列 | 核对 |
|---|---|---|
| `q/r` | `q_or_r`、`lowrank_rank`、`rank_dim` | ✓ |
| 10 臂 + 新增 `RandomRRR-drop` | `intervention_arm`（含 `RandomRRR-drop`） | ✓ |
| `Semantic-only Δfused` | `intervention_arm=="Semantic-only"` 行的 `delta_fused_mse_vs_checkpoint` | ✓ |
| `Semantic-drop Δfused` | 同上（`Semantic-drop` 行） | ✓ |
| `随机 95% 区间` | `random_low_fused_mse` / `random_high_fused_mse` | ✓ |
| `PCA-drop` | `intervention_arm=="PCA-drop"` 行 | ✓ |
| **`随机 RRR 子空间 drop`（新增对照）** | `intervention_arm=="RandomRRR-drop"` + 独立零分布带：`random_rrr_mean/low/high_fused_mse`、`random_rrr_fused_mse_percentile_of_arm`、`worse_than_random_rrr_95pct_fused_mse`、`reference_matches_random_rrr_band` | ✓ |
| **支路自身 Δ** | `delta_branch_mse_vs_checkpoint`、`delta_branch_mae_vs_checkpoint` | ✓ |
| **融合 Δ** | `delta_fused_mse_vs_checkpoint`、`delta_fused_mae_vs_checkpoint` | ✓ |
| 同维对照是否真同维 | `semantic_pca_dimension_match`、`semantic8_pca_dimension_match`、`random_rrr_dimension_match`、`same_dimension_controls_available` | ✓（显式标注而非隐藏） |

另有两组审计列：不变量 1（`untouched_arm_reproduces_model_fused`、`..._gap_vs_model_fused_relative`、
`..._fused_max_abs_vs_model`）与不变量 2（`untouched_arm_reproduces_run_metric`、
`run_metric_check_comparable`、`run_metric_check_note`、`validation_windows_processed`），
以及 `head_kind`、`test_split_read`。

### 7.2 `dissection_table.csv`（48 列）

| minipaper §4.4 解剖表要求 | E16 对应列 | 核对 |
|---|---|---|
| 主模式输入组 / 解释率 | `leading_input_group`、`leading_input_group_label`、`leading_input_group_explanation` | ✓ |
| 主模式输出组 / 解释率 | `leading_output_group`、`leading_output_group_label`、`leading_output_group_explanation` | ✓ |
| 修正能量份额 | `leading_correction_energy_share`（另存绝对值 `leading_correction_energy`） | ✓ |
| 跨 seed `leading4` 重叠 | `cross_seed_leading4_input_overlap`、`cross_seed_leading4_output_overlap`、`cross_seed_pairs` | ✓ |
| **稳定语义判定** | `stable_semantics_verdict` + 6 条判据列 | ✓ |

**6 条判据**：`criterion_1_group_stable`、`criterion_2_input_explanation_ge_0p5`、
`criterion_3_output_explanation_ge_0p8`、`criterion_4_drop_beyond_random_95pct`、
`criterion_5_only_within_0p5pct`、**`criterion_6_drop_beyond_random_rrr_95pct`（本轮新增对照的判据）**。

即新增的"随机 RRR 子空间"对照不只产生一张表，而是**接进了稳定语义判定的第 6 条判据**——
正是 §4.4 表格注释所说"用于区分'语义有效'与'任意同数量主方向有效'"的那个区分。

### 7.3 `cross_seed_alignment.csv` 在单 seed 冒烟下缺失（符合预期）

该文件由跨 seed 配对写出，1 个 seed 时无配对，故不存在。全量运行（3 seed）应产出；
列为 §4.6 复核项之一（见 §6.3）。

### 7.4 结论

§4.4 的两张表**可以被机械回填**，无需人工映射；每一项 minipaper 要求的列都有对应产物列，
且新增对照既出现在干预表也进入了判定判据。schema 冒烟同时再次确认 `algebra_failures: 0`
与两个头的 `algebra=ok`（dense `rel_gap=2.15e-10`、低秩 `2.97e-05`）。

---

## 8. §4.4 回填工具（`e16_writeback.py`）与 schema 对拍

新增 `scripts/phaseformer_L/e16_writeback.py`，把 E16 的产物重塑成 §4.4 的两张表
（按 `(臂, setting)` 聚合 3 个 seed）。其内置的 `missing_columns` 检查**对拍真实列名**，
并在对拍中**抓出一个真缺陷**：

### 8.1 抓到的缺陷：`majority_input_group_label` 这一列不存在

首版按"输入组 / 解释率"取 `majority_input_group_label`，但真实 48 列表头里**没有这一列**——
E16 写的是每 seed 的 `leading_input_group` + `leading_input_group_label`（cols 13–14），
以及一个 `majority_input_group` 键（col 33）与它的 `majority_input_group_votes`（col 34）。
若照首版提交，§4.4 解剖表的"主模式输入组"整列会**静默空白**。

修复：改为对**每 seed 的 `leading_input_group_label` 投票**，并把 E16 自己的
`majority_input_group` 记入 `e16_majority_input_group` 作为**交叉核对**（两个口径都落盘）。

同时修掉一个 markdown 元组数不匹配（10 个 `%s` 对 11 个参数）——会在生成表行时直接抛
`TypeError`。

### 8.2 用真实列名做的端到端验证

构造与真实表头**逐列相同**的合成数据（干预表 693 行 = 3 臂 × 7 setting × 3 seed × 11 臂；
解剖表 63 行）跑通全路径：

```text
{"event": "finished", "intervention_rows": 21, "dissection_rows": 21,
 "missing_columns_intervention": [], "missing_columns_dissection": []}
```

| 检查 | 结果 |
|---|---|
| 行数 | 干预表 **21**、解剖表 **21**（3 臂 × 7 setting）✓ |
| 列数 | 解剖表 33 列 |
| 输入组标签 | `近端电平/EMA`，票数 `3/3` ✓（修复后不再空白） |
| 交叉核对列 | `e16_majority_input_group = recent_level` ✓ |
| 稳定语义判定 | 按 seed 多数：`0/3 → False`、`3/3 → True`，并单列每 seed 判对计数 ✓ |
| **第 6 条判据** | `criterion_6_seeds_true = 2/3`（新增的随机 RRR 子空间判据被单独追踪）✓ |
| 随机带 | 同维随机带 `[-0.0040, 0.0040]` 与 RRR 带 `[-0.0060, 0.0060]` 分列 ✓ |

### 8.3 聚合规则（已冻结，保证可复现）

- 数值列取可用 seed 的均值，并**同时给出 std 与 seed 数**，避免 1-seed 格子冒充 3-seed；
- 稳定语义判定取 **seed 多数（≥2）**，另列每 seed 判对计数与 6 条判据各自的 seed 计数；
- 随机带的 low/high 取各 seed 边界的均值；"是否超出 95% 零分布"按**百分位**判（另列给出
  `Semantic-drop` 超出 RRR 带的 seed 数与总数）。

**边界（如实记录）**：合成数据只用于验证代码路径，其数值无意义且已删除；
真实数值须由 E16 的 63-cell 正式运行产生。

---

## 9. 干预表的"臂覆盖"审计（2026-09-19）

§4.4 规定干预表**每 cell 10 臂**，本文再加一个 `RandomRRR-drop`，即 **11**。回填工具原先
只按需取 4 个命名臂，**不检查**其余臂是否存在——若产物少了一臂，表会"静默变薄"而不报警。
现已加入覆盖审计，并在验证中连修两处：

### 9.1 第一版按 `(臂, setting)` 聚合 → **漏掉单 seed 的缺口**

我故意从 `Weather-96` 的 **seed 2023** 删掉 `RandomRRR-drop`，第一版审计**没有报警**：
按 `(臂, setting)` 聚合时，另两个 seed 仍有该臂，集合看起来是完整的 11。而 §4.4 的表正是
`(臂, setting)` 三 seed 聚合——**一臂在某一个 seed 缺失只会静默缩小该臂的 n，不会让集合不完整**。

修复：同时审计**两个粒度**。修复后同一份数据给出

```text
aggregated_arms_observed          [11]        <- 聚合视图看起来完整（正是漏报的原因）
cells_with_fewer_arms             []
per_seed_arms_observed            [10, 11]    <- 暴露真实缺口
per_seed_cells_with_fewer_arms    ['l_main__Weather-96-s2023',
                                   'l_q1_4__Weather-96-s2023',
                                   'l_q1_8__Weather-96-s2023']   <- 精确定位
```

### 9.2 第二版有个**误导性字段**

`named_arms_missing_from_some_seed` 在同样的数据上返回 `[]`——因为该臂在别的 seed 里还在，
"从某个 seed 完全缺失"的表述掩盖了正是要找的部分缺口。改为**逐臂的"缺席 cell 计数"**
（`{arm: 缺席 cell 数}`），只在非零时列出。

### 9.3 该审计的价值

这两个修正都不是运行期错误，而是**"检查本身不够严"**——第一版会给出"覆盖完整"的结论而
实际存在 3 个格子缺臂。审计工具的严格性与被审计对象的粒度必须匹配：既然表是按 (臂, setting)
聚合的，就必须同时检查被聚合掉的 seed 维度。该教训与 §7 的 `missing_columns`、E14 §6 的
"能解析 ≠ 解析正确"是同一类。

---

## 4. 在 E14 尚未跑完时对**真实 manifest** 复跑本阶段的 dry-run 门（一个"看起来像 bug、其实正确"的结果）

阶段 2 的静态检查此前是在代码修完后做的。本轮在 E14 阶段 A 仍在训练（当时 92/411）时，
用**真实 manifest** 复跑了正式运行要过的同一道门：

```bash
python scripts/phaseformer_L/e16_dissection.py --dry-run --verify-checkpoint-heads \
  --e14-root research_runs/phaseformer_L_e14_main_v1 --output-root /tmp/e16_dryrun
```

**结果：exit 0 —— 门在 E14 只有 92/411 个 run 时就通过了。**

这个结果**第一眼像是严重的门缺陷**（本该"任一 cell 的 checkpoint 解析不到就拒绝"，
而 E14 明明还没跑完）。逐项核对后确认是**正确行为**，理由是：

```json
{"event": "plan", "cells": 63,
 "by_arm": {"l_main": 21, "l_q1_4": 21, "l_q1_8": 21},
 "by_e14_status": {"reused": 63},
 "settings": ["ETTh2-720","ETTh2-96","ETTm2-192","ETTm2-96",
              "Electricity-336","Weather-192","Weather-96"],
 "random_repeats": 100, "random_rrr": true, "random_rrr_repeats": 100}
```

* 63 个 cell 的 `status` **全部是 `reused`**（`by_e14_status: {"reused": 63}`，无一 `new`）；
* 原因是 §4.4 的 7 个 setting 正是 §4.5 的 **test-selected** 集合，而这三个臂
  （`l_main`/`l_q1_4`/`l_q1_8`）在这 7 个 setting × 3 seed 上的 21 格**全部来自既有复用链**
  （`rank_sweep_2_*` 各批次根），21 = 7 × 3 恰好对上；
* 因此 E16 解析的是**旧的复用 checkpoint**，与 E14 阶段 A 新训的 411 个 run **无关**。

**所以门没有偷懒**：它断言的是"我声明的每个 cell 都能解析到 checkpoint"，而这 63 个 cell
确实现在就能全部解析。这也是"门通过"与"上游跑完"是两件事的一个实例——
**门的正确性不取决于上游进度，取决于该阶段自己的输入是否齐备**。

### 4.1 由此得到的一个排期事实（但**本轮不作排期改动**）

既然 E16 不依赖 E14 阶段 A，它在流水线里串行排在第 4 步就**白等了约 6 小时**，
且其本身约 2 h（单卡、单进程）。理论上现在就跑可把 E14 之后的串行链缩短约 2 h。

**但不改**，理由是**算了才发现的**：本阶段**没有 resume/跳过逻辑**——它的产物
（`dissection_table.csv`、`intervention_table.csv`、`canonical_modes.csv`、
`semantic_alignment.csv`、`cross_seed_alignment.csv`、`e16_summary.json`）是**末尾一次性写出**的
（`e16_dissection.py:2674-2678`、`:2880`），没有按 cell 的断点续跑。因此现在跑一遍，
流水线第 4 步还会**再跑一遍**，净收益为零、反而多占 2 h 的 GPU 0 与 E14 抢资源
（E14 才是关键路径）。若要利用这一事实，需要给第 4 步加"产物已存在则跳过"的判定——
那是对**无人值守**链路的改动，收益约 2 h，风险与收益不成比例，故**明确记录而非实施**。

### 4.2 顺带确认的一项 §4.4 要求已被接线

plan 里 `random_rrr: true`、`random_rrr_repeats: 100`，即 minipaper §4.4 要求的
**随机 RRR 子空间对照**（`RandomRRR-drop` 臂，判据
`criterion_6_drop_beyond_random_rrr_95pct`）确实在计划内，且重复次数为 100（与 `random` 带一致）。

## 4. 上线前追加：用**真 manifest** 跑 E16 自己的前置门（2026-09-20 05:07，通过）

E16 是全链**唯一的昂贵重跑**（无 `--resume`，3–5 h）。而它的 63 个 cell **全部是 `reused` 格**，
依赖的 E14 checkpoint 早已定稿——所以它的前置门**不必等 E14 跑完就能先跑一遍**。
在 E14 仍在训练（137/411）时执行：

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python scripts/phaseformer_L/e16_dissection.py \
  --dry-run --verify-checkpoint-heads \
  --e14-root research_runs/phaseformer_L_e14_main_v1 \
  --output-root research_runs/phaseformer_L_e16_dissection_v1
```

**结果：exit 0**，日志 `~/niuyiming/logs/e16_pregate.log`。计划事件逐字如下：

```json
{"event": "plan", "cells": 63, "by_arm": {"l_main": 21, "l_q1_4": 21, "l_q1_8": 21},
 "by_e14_status": {"reused": 63}, "seeds": [2021, 2022, 2023],
 "settings": ["ETTh2-720","ETTh2-96","ETTm2-192","ETTm2-96","Electricity-336","Weather-192","Weather-96"],
 "evaluation_split": "val", "test_split_read": false,
 "random_repeats": 100, "random_rrr_repeats": 100, "random_rrr": true, "mem_budget_mb": 512.0}
```

三点值得记下：

1. **63 格全部解析、63 个 checkpoint 全部读通**，头类型逐格核对一致：
   `l_main` → `dense_shared`（keys=1）、`l_q1_4`/`l_q1_8` → `pooled_lowrank`（keys=2），
   秩为该 setting 的 `horizon/4` 与 `horizon/8`（实测 r = 12/24/42/48/84/90）。
2. **§4.4 要求的随机 RRR 子空间对照已确实接线**：`random_rrr: true`、`random_repeats: 100`、
   `random_rrr_repeats: 100`（不是"计划里有、代码里没有"）。
3. **协议护栏在位**：`evaluation_split: "val"`、`test_split_read: false` —— E16 自身不读 test。
4. `--dry-run` **不写任何产物**（跑后 `research_runs/phaseformer_L_e16_dissection_v1/` 仍不存在），
   即这次预先验证**没有污染**该实验的输出根。

因此第 4 步启动时不会卡在自己的前置门上；剩余不确定性只在训练/评估本身（GPU 时间），
而这部分无法在 E14 让出卡之前推进。
