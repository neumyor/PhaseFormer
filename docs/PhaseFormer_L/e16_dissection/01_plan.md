# E16 / minipaper §4.4 — 训练头解剖 + 支路私有输入干预（plan）

Artifact: `scripts/phaseformer_L/e16_dissection.py`（新，单脚本）+
`scripts/evaluate_lowrank_semantic_interventions.py` 的一处**纯增补**（新增
`RandomRRR-drop` 臂与 `random_rrr_*` 零假设带）。
Output root: `research_runs/phaseformer_L_e16_dissection_v1/`（`--output-root` 显式传入，
不与其它 E 单元共用根目录）。

## 1. §4.4 要求什么

`docs/PhaseFormer_L_minipaper.md:432-446`：

1. **解剖表**（PhaseFormer-L 与低秩探针，3 seed）：列 = dataset、H、
   `主模式输入组 / 解释率`、`主模式输出组 / 解释率`、`修正能量份额`、`跨 seed leading4 重叠`、
   `稳定语义判定`。
2. **干预表**（每 cell 10 臂）：列 = dataset、H、`q/r`、`Semantic-only Δfused`、
   `Semantic-drop Δfused`、`随机 95% 区间`、`PCA-drop`、**`随机 RRR 子空间 drop`（新增对照）**、
   `支路自身 Δ`、`融合 Δ`；即**每条臂都必须同时给出支路自身误差与融合误差**。

§5.4 与 `低秩分析计划 §11.8.2` verbatim 指出：既有结果中 **57/57 个同维 cell 上
`Semantic-drop` 与 `PCA-drop` 数值完全相同**，因此"语义有效"与"任意同数量的主要方向同样有效"
**无法区分**；解决办法是"在 `evaluate_lowrank_semantic_interventions.py` 增加一个臂
（分析侧代码，不涉及模型）"。

**范围**：7 个 test-selected setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192、
Electricity-336）× 3 seed（2021/2022/2023）× 3 个支路族（下表）= **63 cell**。
这 7 个 setting 来自 test-set selection（schedule §2.1），本单元只报告其 train/validation 量，
但仍须逐表披露（这是条件性证据，不是盲测）。

| `--arms` | 训练实现 | 有效映射 |
|---|---|---|
| `l_main`（PhaseFormer-L 主行） | `weak_residual` + `weak_period_residual_head_type=shared`（dense `WeakPeriodResidualHead`） | `W` (H×720) 本身 |
| `l_q1_4` | `weak_residual` + `pooled_lowrank`，`rank=H/4`，`pool_factor=1` | `W_dec @ W_enc` (H×720) |
| `l_q1_8` | 同上，`rank=H/8` | `W_dec @ W_enc` |

`phase_only`（无残差支路）、`l_rcrf`（`rcrf` 可靠性融合，不是两路凸门）、`a1`（本地无产物）
均**不在解剖范围内**，`--arms` 会显式拒绝并说明原因。

## 2. 继承的定义（不重新发明）

| 量 | 公式 / 来源 |
|---|---|
| 有效映射 `M` | `scripts/lowrank_checkpoint_core.py::effective_map(encoder, decoder, pooled_len)`（plan §3.2） |
| **dense 头** | `encoder = I_720`、`encoder_bias = 0`、`decoder = linear.weight` (H×720)、`decoder_bias = linear.bias`、`pooled_len = 720` ⇒ `M = W`，且**支路私有输入就是它的隐状态**（`(B,C,720)` 的中心化归一化历史） |
| 规范模式 / 谱 | `np.linalg.svd(M)`；份额 = `s²/Σs²`；`participation_ratio`、`numerical_rank`、gap：`analyze_lowrank_checkpoint_information.py` 同名实现 |
| 模式能量 | 解码器自身隐帧：`contribution_k = (dec_u[:,k]·dec_s[k]) ⊗ ⟨dec_vt[k], h⟩ ⊗ σ`，`mode_energy_k = mean(contribution_k²)`；dense 头两帧重合（decoder == M），探针不重合（**照抄既有口径**，见 `低秩计划 §11.4`） |
| 语义字典 | `input_templates(720, period)` / `output_templates(H, period)` + `build_groups`；归因 = `analyze_lowrank_checkpoint_information.py::align_direction`（best template / best group / explanation / dictionary R² / Shapley） |
| 输入侧度量 | train split 的 `z = x − x_last` 二阶矩（`CenteredMoments`，ddof=1，与 `centered_moments` 同代数） |
| 输出侧度量 | 本 cell validation 残差 `target − phase` 的 `(H,H)` 协方差 + ridge（`trace/H × 1e-6`），与 `output_residual_moments` 同一估计量，但**按 cell 而非按 setting 汇总**（见 §8 歧义 6） |
| 干预代数 | `scripts/evaluate_lowrank_semantic_interventions.py::arm_metrics`（`only`/`drop`/`identity`/`bias`）、`latent_image`、`semantic_basis`、`latent_input_pca_basis` 的等价 `eigh`（PCA） |
| 零假设带统计 | 同文件 `band_summary`（`random_*` 键名逐字保留）+ `random_drop_band` |
| 判定阈值 | `低秩计划 §6.1`：解释率 ≥50%、配对输出 ≥80%、随机 95% 区间、Semantic-only 双指标 ≤+0.5%、3 seed 中 ≥2 一致 |

**未改动任何既有数值口径**：探针行的解剖量与干预量由本脚本用同一批函数重算，
并可与已填的 `research_runs/lowrank_checkpoint_information_v1/` 产物逐字段对账（`reference_parity.json`）。

## 3. `RandomRRR-drop` 的精确定义（最高价值项）

### 3.1 新增臂

对每个 cell：

```
pool            = latent_image(npz["independent_basis"], encoder, rank_dim)
                  # Stage 3 的独立 RRR 输入子空间在隐空间的像，就是
                  # 既有 Independent-RRR-only 臂用的同一个对象；
                  # dense cell 没有该文件 → E16 用 train split 现场拟合（§3.4）
r               = min(#Semantic-drop 子空间维数, dim(pool))        # 与被检验臂同维
RandomRRR-drop  = random_rrr_basis(pool, r, default_rng(RANDOM_SEED + 2))   # 一次确定性抽样
band            = {random_rrr_basis(pool, r, default_rng(RANDOM_SEED + 1)) × --random-rrr-repeats}
```

`random_rrr_basis(pool, r, rng) = qr(pool @ N(0,1)^(dim(pool)×r))` 的前 `r` 列：即
**`span(pool)` 内均匀随机的 `r` 维子空间**（Haar 均匀），与 `drop` 模式组合后删除该子空间。
臂自身的行取**该族的一次确定性抽样**（独立随机流），band 是同族多次抽样的分布；
因此 `random_rrr_fused_mse_percentile_of_arm` 是该臂在零假设带中的位置，
`worse_than_random_rrr_95pct_*` 是"必要性超出同维可达随机方向"的判定位。

### 3.2 与既有"随机带"的区别（两条带都保留）

| | 既有 `random_*` 带 | 新增 `random_rrr_*` 带 |
|---|---|---|
| 抽样空间 | **整个隐空间**（`random_orthogonal_basis`） | **RRR 可达子空间 `span(pool)`**（`random_rrr_basis`） |
| 维数 | `max(1, min(rank_dim, 6))`（既有约定，逐字未改） | 被检验臂 `Semantic-drop` 的维数（= `#semantic_full`） |
| 回答的问题 | "删掉任意一个 6 维子空间有多痛" | "删掉任意一个**与被检验臂同维、且落在支路可读族内**的子空间有多痛" |
| 键名 | `random_mean_fused_mse` … `reference_matches_random_band` | `random_rrr_mean_fused_mse` … `reference_matches_random_rrr_band` |

既有带的维数 `min(rank_dim,6)` **并不等于** `Semantic-drop` 的维数（后者在 15/72 个既有 cell
上只有 36 而 `PCA-drop` 是 180），所以它不能作为"同维对照"；这一点正是 §11.8.1 记录的问题。
两条带因此**不是同维对照关系**，各自回答不同问题，都必须保留（既有 表 6 引用的就是既有带）。

### 3.3 dense 头为什么也能用同一代数

dense 头 `encoder = I`、无 encoder bias、`decoder = linear.weight`，因此
`hidden ≡ 中心化输入`，`arm_metrics` 的 `only/drop` 投影恰好作用在支路私有输入上，
`bias-off` 删的就是 `linear.bias`；`M = W_dec @ I @ pool(720) = W`。
所以在 `l_main` 上，"解剖"就是 `W` 的 SVD，"干预"就是同一套臂，无需第二套实现；
`l_main` 的谱与模式帧重合，其 `correction_energy_share` 没有帧歧义。

### 3.4 dense cell 的 RRR 池从哪来

Stage 3 的 `subspaces/*.npz` 只覆盖低秩 cell，dense cell 没有。E16 用**同一个估计量**在
train split 上现场拟合（`lowrank_checkpoint_core.independent_rrr`，ridge `1e-6`，
秩 = `min(H,720)`，即 dense 头有效映射的行空间秩）：

* 空间取 **RevIN 归一化空间**：支路自己的 `z = (x − x_last)/σ_window`，故等价于在
  训练集标准化窗口上做 `1/σ_window²` 加权的最小二乘；这与
  `scripts/compute_phase_conditional_rrr.py` 的口径一致（该脚本用模型记录得到同一空间）。
  `--rrr-pool-space standardized` 可切换到无权版本（诊断用）。
* 数值上是 float64 直接算（既有脚本走 float32 模型记录），因此同 setting 两个来源的池
  会有 ~1e-6 量级的旋转差异；`random_rrr_pool_source` 逐 cell 记录用了哪个来源。

### 3.5 已知退化：池可能覆盖整个隐空间

`pool` 的维数是"支路自身秩"，对**所有** rank bottleneck 探针 cell 而言其隐空间像就是整个隐空间
（`random_rrr_pool_covers_ambient = True`），此时 RRR 族与"整个隐空间"族重合，
`RandomRRR-drop` 与同维的 `random_ambient_matched` 带同分布；真正有区分力的是 dense cell
（`pool` 维数 `min(H,720)`）。这一点逐 cell 记录，不隐藏。E16 另外算了
`random_ambient_matched_*`（同维、整个隐空间）带，使"族"与"维数"两个因素都能被分离。

## 4. 复用 vs 重算（逐项）

| 对象 | 来源 | 说明 |
|---|---|---|
| checkpoint | E14 `stage_a_manifest.json` | `status=reused` 的 cell 用 `source.run_dir`（E3 系）；`status=new` 的 cell 在 `<e14-root>/runs/*` 内按 `e14_main_matrix._arm_match` 匹配（不同 tie-break 会产生候选，逐 cell 记录 `alternative_run_dirs`）。run 内的 checkpoint 以 `metrics.csv` 的 `checkpoint` 列（run 自己恢复用于 validation 的那一份）为准，glob `attempts/*/checkpoints/best.ckpt` 仅作回退；规则逐 cell 记为 `checkpoint_source`，因为"未干预臂 vs run 记录 `val_mse`"的不变量核对必须对同一份文件 |
| 解剖量（模式/语义/能量） | **全部重算** | 用 §2 的注册函数；探针行与既有产物做 `checkpoint_inventory.csv` 路径对账，路径一致的 cell 逐字段比较（`reference_parity.json`，容差 1e-6） |
| 干预臂（探针 9 臂 + 语义臂） | **全部重算** | 同一 `arm_metrics`，因此与既有表 6 可直接对账；重算的理由是 E14 的复用解析与低秩清单**选点规则不同**，可能指向不同的重复训练产物，而 §4.4 必须解剖 §4.2 主表真正使用的那一份 |
| 干预臂（新增 `RandomRRR-drop`） | 本单元 + 既有 root 的 evaluator 重放 | 两处用同一函数、同一随机流、同一 RRR 池，探针 cell 上二者应逐位一致 |
| `subspaces/*.npz`（RRR 池） | 复用既有 Stage 3 产物 | 探针 cell 直接读；缺失时回退到 §3.4 的现场拟合 |
| 7 setting 的 E14 主表读数 | 不读 | 本单元不重算精度、不读 test；`metrics.csv` 只取 `val_mse/val_mae` 用于不变量核对 |

## 5. test-split 保证

* `--evaluation-split` 只接受 `val` / `validation`（`test` 会被 argparse 直接拒绝）；
* 只构造 validation loader（`data_provider(..., "val")`），代码里没有 test loader；
* train 统计经 `scripts/phaseformer_L/e15_dimension.py::load_split` 读取，该函数用
  `nrows = validation border` 解析 CSV，**测试段的行根本不会被读入**（`cell_plan.json`
  与 `e16_summary.json` 记录 `rows parsed` / `test border row` / `test_split_read: false`）；
* 每个产物都有 `test_split_read: False` 列。

## 6. 内存策略（dense 头的 720 维隐状态）

dense 头的隐状态是 720 维中心化输入，一个 cell 的 `(samples, channels, 720)` float64
就是 13.9 GiB 事故的同族分配（Electricity-336 单是隐状态就要 ~2.9 GiB），所以：

* 按 `(batch × channel block)` 流式处理，`--mem-budget-mb`（默认 512）同时约束
  ① 统计/臂块 `(batch, channel_block, rank_dim)` 及其参考修正与误差张量、
  ② 随机带的 `(batch, channel_block, rank, arms)` 收缩；`--channel-block` 可强制覆盖；
* 随机带的基**每 cell 只抽一次**，所有 block 重放同一组基（否则按块抽基会把不同臂混在一起、
  低估带的宽度）；`random_drop_band` 自身的 `block_elements` 也按预算传入；
* 只保留标量 / 小矩阵累加器：模式能量用 `s_k²·mean(score_k²σ²)`（与
  `mean(contribution²)` 代数等价），隐状态二阶矩（≤720²）供 PCA 与 `latent_variance`，
  臂指标由**逐块调用注册 `arm_metrics`** 后按 pair 数加权汇总，
  `correction_reconstruction_r2` 由全局参考能量重新装配（分子来自块内 `correction_rmse`）；
* 每个 cell 走**两遍** validation：第一遍只累计"一阶基所需统计量"，第二遍带全部基评估臂与带
  （PCA 控制需要隐状态协方差，而缓存隐状态正是被禁止的分配）。

## 7. CLI 与产物

```
--output-root --e14-root --checkpoints --arms --datasets --horizons --seeds
--evaluation-split {val,validation} --random-repeats --random-rrr-repeats
--random-rrr/--no-random-rrr --rrr-pool-space {revin,standardized}
--semantic-rank --modes --mem-budget-mb --channel-block --max-batches --num-workers
--gpus --data-root --reference-root --skip-reference-parity --allow-missing-cells --dry-run
```

`<output-root>/`：

* `dissection_table.csv` — 63 行（7 setting × 3 arm × 3 seed），含 §4.4 六列 +
  判定的 6 个判据列 + 每 seed 的 `leading4` 输入/输出重叠 + 来源列；
* `intervention_table.csv` — 每 cell × 每条臂一行，含 `branch_mse/mae`、`fused_mse/mae`、
  两支路 Δ、`correction_*`、维数与同维标志、三条零假设带的全部列；
* `canonical_modes.csv` / `semantic_alignment.csv` — 与 `lowrank_checkpoint_information_v1`
  同名文件的**同名列**（外加 `arm` / `head_kind` / `rank_dim` 等来源列）；
* `cross_seed_alignment.csv` — `leading4` / `leading8` 的逐 seed 对重叠；
* `e16_summary.json` — 范围、协议、逐 cell 来源、随机控制定义、内存块大小、
  不变量、判定计数、披露清单、环境；
* `cell_plan.json` — 解析后的 cell ↔ checkpoint 对照（stage 2/4/5 的对账依据）；
* `reference_parity.json` — 与既有低秩产物的逐字段对账（路径不一致的 cell 单独列出，不比较）。

## 8. 判定口径与歧义（必须披露）

**判定**（`stable_semantics_verdict`，设置在 (setting, arm) 层，重复写在 3 个 seed 行上）：

| 判据 | 内容 | 阈值 |
|---|---|---|
| 1 | 3 seed 中 ≥2 把同一输入组排首位 | ≥2 |
| 2 | 该组平均解释率 | ≥0.50 |
| 3 | 该组配对的输出组平均解释率 | ≥0.80 |
| 4 | `Semantic-drop` 超出**既有整空间**随机带 95% | ≥2/3 seed |
| 5 | `Semantic-only` 双指标不差于基线 +0.5% | ≥2/3 seed |
| 6 | `Semantic-drop` 超出**RRR 可达族**随机带 95%（§4.4 新增） | ≥2/3 seed |

`verdict = not_supported`（1,2,3,5 不全）/ `stable_without_necessity`（缺 4）/
`stable_not_specific`（有 4 无 6，即"语义并不比同族随机方向更特殊"）/ `stable_specific`（全满足）。

**歧义与已做的选择**（未解决的一律列出）：

1. **"同维"到底指哪一维**：任务文本既说新带"与被检验臂同维"，又说既有带是"同维随机带"。
   既有带实际是 `min(rank_dim,6)` 维（**不等于** `Semantic-drop` 的维数），二者不可能同时成立。
   选择：新带与 `Semantic-drop` 同维（因为它是被检验的那条臂），并**保留**既有带原样、
   另加同维整空间带 `random_ambient_matched_*`，使"族"与"维数"可分离。
2. **"RRR 可达子空间"的读法**：任务文本"top-r RRR 方向组成随机组合，r = 臂的删除维数"
   若按字面执行是**退化**的（一个 r 维跨度内的随机 r 维子空间就是它自身）。选择：
   `pool` = 该 cell 的独立 RRR 子空间（注册的 `Independent-RRR-only` 臂用的同一对象），
   删除维数另取 `min(#semantic, dim(pool))`。若作者本意是别的族（例如条件 RRR 或
   `learned_basis`），换一个 `pool` 即可，键名与流程不变。
3. **探针 cell 的 RRR 族覆盖整个隐空间**（§3.5）：RRR 带在这些 cell 上退化为同维整空间带，
   真正有区分力的是 dense cell；已在逐 cell 标志与 summary 中披露。
4. **dense cell 的 RRR 池没有注册产物**：由 E16 现场拟合（§3.4）。空间取 RevIN 加权以对齐
   Stage 3 口径；若审校认为应改为未加权，`--rrr-pool-space standardized` 一条命令即可复算。
5. **同维对照缺失的既有 cell**：选择"记录 `dimension_match=False` 而不是重定义 `PCA-drop`"，
   因为既有 720 行干预表已被 表 6 与 §11.8 引用，改口径会静默改变已公布数字。
   E16 自己的表另加 `PCA-matched-only/drop`（与被检验臂同维的 PCA 控制），使该问题在
   新表内可直接回答。
6. **输出侧协方差度量按 cell 估计**（既有分析按 setting 汇总 seed × rank 的缓存残差）：
   只影响 `output_covariance_correlation_max` 一列；E16 不缓存张量，无法跨 cell 汇总。
7. **注册代数把映射后的 encoder bias 计了两次**（隐状态里已含 `encoder_bias`，
   `arm_metrics` 又加了一次 `W_dec @ encoder_bias`）：E16 **照抄**该约定（因为它喂已被引用的表），
   并逐 cell 记录 `mapped_encoder_bias_absmax` 与
   `untouched_arm_gap_vs_recorded_fused`（"未干预臂 vs 模型自身输出"的偏差）。
   dense 头 `encoder_bias = 0`，该偏差为 0。
8. **解码器帧 vs 有效映射帧**：探针 cell 的 `singular_value` 来自有效映射，而模式能量在解码器帧
   （照抄 `低秩计划 §11.4`）；dense 头两帧相同。表内两列都给出，不混用。
9. **哪些 cell 是主表读数**：`l_main`/`l_q1_4`/`l_q1_8` 的 7 setting 在 E14 中是 D-2 的
   **复用格**（E3 系 Stage-0 冻结 gate/lr），与 §4.2 新格不是同一超参协议，表注须逐格披露。
10. **Traffic 不在范围内**：§4.4 只要求 7 个 test-selected setting；脚本是逐通道分块的，
    若要扩展只需 `--datasets Traffic`，但会产生与 §4.4 规格不同的行。

## 9. Stage 2 静态检查清单（下一阶段逐条执行）

1. `PYTHONDONTWRITEBYTECODE=1 python3 -c "compile(open(<path>).read(), <path>, 'exec')"`
   对本单元的两个文件均通过（已做）。
2. `python3 -m pyflakes` 无新增告警（已做）。
3. `--help` 退出 0，且列出 §7 的全部必需 flag（已做；`--evaluation-split test` 必须被拒绝）。
4. `--dry-run` 打印的 cell 数 = `--arms` × 7 setting × 3 seed（默认 63），
   并逐个打印解析到的 checkpoint 路径；dry-run 不写任何产物（已做，见 §10）。
5. `--arms phase_only` / `--arms l_rcrf` 报错退出并说明原因（已做）。
6. 每个 cell 的 `untouched_arm_reproduces_run_metric` 为真、`validation_pairs` 与 loader 一致；
   `untouched_arm_gap_vs_recorded_fused` 在 dense 上应为 float32 量级（~1e-7），
   在探针上等于 §8.7 的 bias 项的效应。
7. `dissection_table.csv` 行数 = 63；`intervention_table.csv` 行数 = Σ 每 cell 臂数
   （dense 12、探针 10–12，取决于 npz 是否含 `conditional_basis`）；
   `reference_parity.json.passed` 在路径一致的 cell 上为真，容差 1e-6。
8. `--random-repeats 10 --max-batches 2` 的冒烟跑通，并记录单 cell 秒数。
9. `--no-random-rrr` 与默认输出的差异**只**是 `RandomRRR-drop` 行与 `random_rrr_*` 列。
10. 抽查 1 个 dense cell 的 `Sum_k correction_energy_k ≤ total_correction_energy`，
    以及 `correction_energy_share ∈ [0,1]`。

## 10. 本机已做的验证（无 torch / 无 checkpoint / 不读 test）

* 语法：两个文件 `compile()` 通过；`pyflakes`（含 `flake8 -select=F,E9`）无新增告警。
* `evaluate_lowrank_semantic_interventions.py` 的增补用 numpy 桩（torch 被 stub）
  实跑 `evaluate_arms`：默认臂表 = 既有 10 臂 + `RandomRRR-drop`；
  `--no-random-rrr` 时臂表**逐字**等于既有 10 臂；既有 11 个 `random_*` 键的数值与
  增补前实现**逐位相同**；`random_drop_band` 的默认路径与增补前实现**逐位相同**
  （5 组随机夹具）；新参数 `dimension` / `pool_basis` / `bases` 的行为与
  `random_rrr_basis` 的正交性、族内包含性与维数均验证通过。
* `e16_dissection.py`：
  * 用 numpy 桩 + 自写的 numpy 版假 torch（ndarray 子类提供 `permute/expand/...`）
    **完整跑通 `run_cell`**（dense 与 pooled 两种头、两遍流式、分块、基构造、臂表、
    三条带、行装配）：未干预臂的 `fused_mse/mae`、`correction_energy`、
    两遍之间的 `recorded_fused_mse` 都与独立重算一致到 float32 量级（dense 相对差 ~3e-8，
    探针 ~2e-8 记录值）；`--no-random-rrr` 正确移除臂与带；
  * 累加器与注册代数的一致性：模式能量 = `mean(contribution²)`、`latent_variance` = `np.var`、
    全部臂指标 = `arm_metrics`、三条带 = 未分块的 `random_drop_band`，且对分块数不变
    （1 块 vs 4 块，容差 1e-10）；
  * `train_statistics` 与直接重算逐字段一致（`szz/szy/加权 szz/加权 szy/中心化协方差`），
    且只解析到 validation 边界（3000 行 CSV 只读 2400 行）；
  * cell 解析：manifest 复用格 → E3 run 目录、新格 → E14 run 目录、`--checkpoints` 显式解析、
    缺格报错与 `--allow-missing-cells`、无法归属的 checkpoint 报错；
  * 判定：四条 verdict 分支、2/3 seed 多数、`leading4` 重叠（同基=1.0、旋转基=0.75）、
    对账报告（一致/路径不一致两种情形）、CSV 写出；
  * CLI：`--dry-run`（不写产物）、`--evaluation-split test` 被拒、`--arms phase_only` 被拒、
    `--help` 列全 flag。
* **未能验证**：真实 torch/CUDA 前向、真实 checkpoint、真实 dataset（本机无 torch、无数据）、
  真实 `stage_a_manifest.json`（E14 尚未在本地产出）、真实运行时间与内存峰值。
  这些只能由服务器上的 stage 3（冒烟）与 stage 4（正式跑）确认。
