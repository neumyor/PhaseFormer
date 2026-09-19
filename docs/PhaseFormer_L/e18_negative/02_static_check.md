# E18 · §4.6 负对照 — 阶段 2：静态检查

> 代码：`scripts/phaseformer_L/e18_negative.py`（行 1、行 5）、`scripts/phaseformer_L/e18_svd_truncation.py`（行 3）
> 产物根：`research_runs/phaseformer_L_e18_negative_v1/`
> 服务器：`yyk03@11.11.18.3`，conda `time`

## 1. 静态检查结果（全部通过）

| # | 检查项 | 判据 | 结果 |
|---|---|---|---|
| 1 | `compile()` | 两个脚本均可编译 | **通过** |
| 2 | 计划 cell 数 | 行 1 = 42、行 5 = 36，合计 **78** | **78**（`{"event":"plan_only","cells":78}`） |
| 3 | 行 1 分档计数 | `causal_ema_mid_s0.5` 21 + `causal_ema_max_s1` 21 | **21 / 21** |
| 4 | 行 5 分档计数 | `absolute_rank1` 18 + `absolute_rank2` 18 | **18 / 18** |
| 5 | **两个平滑档数值上确实不同** | 两份 overrides 的 JSON 必须不等 | **`levels numerically distinct: True`**（`smooth_ratio` 0.5 vs 1.0） |
| 6 | 协议常数 | 每条命令 `--lookback 720 --period 24 --max-epochs 30 --loss huber --percent 100` | **78/78 一致** |
| 7 | mechanism / head | 行 1 为 `weak_residual` + `shared`；行 5 为 `weak_residual` + `pooled_lowrank`（`pool_factor=1`，`smooth_ratio=0.0`） | **一致** |
| 8 | **D-2 新格超参** | `gate_init=0.2`、`lr=1e-3` | 78/78 一致 |
| 9 | **不读 test** | 78 条命令中不得出现 `--evaluate-test` | **0 条**；manifest `reads_test: False` |
| 10 | 行 5 的秩为**绝对**秩 | `absolute_rank1` → `rank=1`；`absolute_rank2` → `rank=2` | **一致**（与 §4.2 的相对秩 `H/4`、`H/8` 区分） |
| 11 | 覆盖设定 | 行 1 = 7 个 test-selected setting × 3 seed × 2 档；行 5 = 6 个 E8 setting × 3 seed × 2 秩 | **一致** |

## 2. 本阶段发现并处理的实质问题

### 2.1 D-4 的「2 档」原本是**退化**的（已在阶段 2 前修正）

首版把两档写成"boxcar `smooth_ratio=0.5`"与"causal EMA `alpha=0.08`"。核对源码后确认二者**数值上完全相同**：

- PhaseFormer-L 用的稠密 `shared` 头把 `smooth_ratio` 实现为与 **causal EMA** 的混合
  （`src/models/phase_adapters.py:113-115`）；真正的 boxcar（`F.avg_pool1d`）只存在于
  `pooled_lowrank` 头（`:168-179`）——所以 PhaseFormer-L 上 **没有** boxcar 这一算子；
- `alpha=0.08` 既是 `_causal_ema` 的默认（`src/models/asymmetric_trend_components.py:116`），
  也是读取键的默认（`src/models/PhaseFormer.py:1058`），因此"显式写 0.08"与"不写"是同一次计算。

即：原设计会在 42 个 run 上把同一条命令跑两遍（仅 config hash 不同）。已改为同算子的两个**不同强度**
`smooth_ratio ∈ {0.5, 1.0}`（含算子默认 `alpha=0.08`），并把上表第 5 项作为**固定卡点**：
两份 overrides 必须不等。minipaper §4.6 表注与本文档 D-4 已同步更正。

### 2.2 行 3 与既有 E11 的**口径差异**（必须披露）

E11 的截断对比是 **test-based**、单 seed、7 setting；E18 行 3 是 **validation-based**、
checkpoint 来自 E14 的 `l_main`/`l_q1_4`/`l_q1_8`。只有截断代数与三段式比较结构相同。
该差异由 `e18_svd_truncation_summary.json.e11_comparability` 逐条记录，供 §4.6 表注使用；
行 3 的输出每行都带 `records_test=false` / `split=val`。

### 2.3 行 3 的截断秩选择

主秩 **r = 10，全 28 setting**——r=10 正是 E11 测到 Electricity-336 反例的那一秩
（截断 +29% vs 训练 +0.7%），因此 minipaper 引用的唯一数字保持同秩可比。每行另报该 setting 的
原生训练秩（`H/4`、`H/8`）与同 seed 最接近 r=10 的训练低秩臂，并记录
`trained_lowrank_rank`，因为 r=10 在 H=96 上比训练网格更深、在 H=720 上却浅约 9 倍。

## 3. 明确不做的事

| 项 | 说明 |
|---|---|
| 行 2（结构化坐标） | minipaper 标「—」，**不补做**（既有 E13 覆盖 4 setting） |
| 行 4（q=1/32 容量） | minipaper 标「—」，**不补做**（既有 E6/E3 覆盖 7 setting） |
| 行 5 用于论证秩-2 必要性 | **禁止**：结果无论好坏都不得读作"秩-2 必要/不必要"的证据（minipaper §5 第 5 条） |

## 4. 结论

阶段 2 通过（11/11 项），并挡下 1 个会使 42 个 run 空转的**退化设计**。允许进入阶段 3
（冒烟：按 `--max-batches`/短 epoch 在 Electricity-336 与 ETTh2-96 上跑通两类格）。

---

## 5. 行 3（SVD 截断 vs 秩约束训练）的静态检查

```bash
python scripts/phaseformer_L/e18_svd_truncation.py \
  --e14-root research_runs/phaseformer_L_e14_main_v1 --output-root /tmp/e18svd \
  --ranks 10 --seeds 2021 --evaluation-split val --dry-run
```

| # | 检查项 | 结果 |
|---|---|---|
| 1 | setting 数 | **28**（7 数据集 × 4 horizon） |
| 2 | Traffic 在范围内 | 是（4 个 setting） |
| 3 | 不读 test | `--evaluation-split` 只接受 `val/validation`，`test` 被 argparse 拒绝；输出每行带 `records_test=false` / `split=val` |
| 4 | 复用的 7 个 setting | **全部解析为 `reused`**（ETTh2-96/720、ETTm2-96/192、Weather-96/192、Electricity-336），指向 E3 系真实 run |
| 5 | 秩选择 | 主秩 **r=10**（E11 测到 Electricity-336 反例的那一秩，保持同秩可比）；每 setting 另报原生训练秩（`H/4`、`H/8`） |

### 5.1 静态检查挡下的缺陷：默认范围漏掉 Traffic（24 ≠ 28）

首版 `--datasets` 默认取 `MAIN_DATASETS`（6 个数据集，不含 Traffic），dry-run 计划出 **24 个
setting**，而 minipaper §4.6 行 3 明确要求 **全 28 setting**（24 主表 + 4 Traffic 附录）。
这会产出"声称 28、实际 24"的静默缺失——整段 Traffic 被漏掉。已改为
`ALL_DATASETS = MAIN_DATASETS + TRAFFIC_DATASETS`，复测计划为 **28**，Traffic 4 行在位。

### 5.2 当前未解析项及其性质（**不是缺陷**）

dry-run 报 57 条未解析（`l_main` 17、`l_q1_4` 19、`l_q1_8` 21）。原因是 E14 仍在训练：
按成本降序，先跑 Traffic 的 `l_main`，因此部分 Traffic 格已可解析、其余臂与 setting 尚未产出。
**验收判据**：E14 阶段 A 全部结束后重跑，应报 `problems: 0`（28 个 setting × 3 个臂全部解析）。

---

## 6. §4.6 回填工具（`e18_writeback.py`）

§4.6 是五行汇总表，最后一列"本文补做"才是本文的工作。新增
`scripts/phaseformer_L/e18_writeback.py`，只填**行 1/3/5**，并把**行 2/4 固定为 "—"**
（minipaper 明确说不补做）——代码里以常量 `KEPT_AS_DASH` 落地，防止后来者顺手补上。

### 6.1 数据来源与配对口径

| 行 | 来源 | 配对基线 |
|---|---|---|
| 行 1（平滑 2 档） | E18 的 `results.with_test.csv` 中 `stage=="smooth"` 的行 | 同一 setting 的 **E14 `l_main`** test MSE/MAE |
| 行 3（SVD 截断） | E18 的 `svd_truncation_table_28.csv` | 同一 setting 的**全秩**与**秩约束训练**结果 |
| 行 5（绝对秩 1/2） | E18 的 `results.with_test.csv` 中 `stage=="rank12"` 的行 | 同上（E14 `l_main`） |

**所有补做数字都是同 setting、同 seed 的配对比较**（基线自身的超参不同，见 D-2 披露），
因此表内可比，但不能与金标准直接比较。

### 6.2 用真实列名做的端到端验证

构造与真实表头相同的合成数据（行 1+5 共 78 行 = 42+36；基线 21 行；截断表 9 行）跑通全路径：

```text
{"event": "finished", "row1": "no cell improves both metrics",
 "row1_mean_delta_mse_pct": 1.3081, "row3_mean_gap_pct": 4.821,
 "row3_worst": "Electricity-336", "row5_mean_delta_mse_pct": 17.6312,
 "row5_degraded_cells": 12}
```

| 检查 | 结果 |
|---|---|
| 行数 | **5** 行 ✓ |
| 行 2/4 | 均为 `—` ✓ |
| 行 1 判定 | 逐 cell 判"是否**双指标**同时改善"，输出改善/未改善清单 ✓ |
| 行 3 最差 setting | 正确挑出 **Electricity-336**（既有反例锚点）✓ |
| 行 5 退化 cell 数 | 逐个列出 ✓ |

### 6.3 已写入结果的披露

- 行 3 为 **validation 口径**，而既有 E11 为 **test 口径**——只有截断代数与三段式结构相同；
- 行 5 **不得**读作"秩-2 必要/不必要"的证据（minipaper §5 第 5 条）；
- 行 1/5 的基线来自 E14，其超参与被比格不同（D-2），配对只在同 setting 内成立。

**边界**：合成数据仅用于验证代码路径，数值无意义且已删除；真实数值须由 E18 的 78 个正式 run
与 28-setting 截断分析产生。

---

## 附：row 3（绝对秩截断）**评估路径**的冒烟 —— 通过

> 执行：2026-09-20 03:58（服务器，**CPU**，避免与正在训练的 E14 争 GPU）｜退出码 **0**

### 1. 为什么补做

`e18_svd_truncation.py` 此前只被执行到 `--verify`（解析 28 个 setting 的 checkpoint、
在 E14 未跑完时**正确拒绝**）。而"**真的去截断一个正确器、并在 val 上评估**"这条路径
从未跑过——它正是 §4.6 行 3 的数字来源，且排在步骤 6（E18 的 78 个 run 之后）。

### 2. 做法与结果

用**已有的复用 checkpoint**（ETTh2-96 属 test-selected 集合，其 3 个臂的 run 来自既有复用链，
故现在就可解析），把范围限到一格、秩 10、**子集评估 300 个样本**、CPU：

```text
{"event": "planned",   "settings": 1, "seeds": 1, "ranks": [10], "problems": 0}
{"event": "verify_ok", "settings": 1, "seeds": 1, "cells": 1}
{"event": "evaluated", "setting": "ETTh2-96", "seed": 2021}
{"event": "finished",  "settings": 1, "table_rows": 1,
 "table_csv": "/tmp/e18svd_smoke/svd_truncation_table_28.csv"}
```

`svd_truncation_table_28.csv` 的那一行确实把行 3 需要的三个量都算了出来：

| 列 | 值（本冒烟，**子集**评估） |
|---|---|
| `truncated_mse` | 0.16258476 |
| `trained_lowrank_mse` | 0.20631718 |
| `full_rank_mse` | 0.16108628 |
| `gap_truncated_vs_trained_mse_pct` | −21.1967 |
| `gap_trained_vs_full_mse_pct` | 28.0787 |
| `records_test` | **False** ✓（协议：不读 test） |

四个产物齐全：`e18_svd_truncation_plan.json`、`e18_svd_truncation_summary.json`、
`svd_truncation_per_rank.csv`、`svd_truncation_table_28.csv`；
summary 的键含 `test_split_read`、`records_test`、`protocol_mirrored_from_e14`、
`worst_truncation_gaps` 等。

### 3. 这一行数字**不是**行 3 的结果（必须写明）

冒烟用了 `--max-eval-samples 300`，即**只评估了 val 的一个子集**，
所以上表的 MSE 与 gap **不构成 §4.6 行 3 的任何结论**，也不能与论文中的
"截断 +29%、训练 +0.7%（Electricity-336, r=10）"相比。它证明的只是
**这条评估路径能跑通、并把三个量写进表里**。正式数字要等 28 个 setting × 3 seed 的
全量评估（步骤 6）由阶段 5 审校按冻结判据判定。

## 7. 阶段 2 追加：基线出处列**曾会 78/78 行全空**（已修）

§4.6 的行 1/行 5 要与 E14 的 `l_main`（PhaseFormer-L）配对，配对出处由
`e18_negative.load_baseline_index` 从 E14 的 `stage_a_manifest.json` 建立。用**真 manifest**
跑一遍该函数（而不是读它的源码）时发现：`resolved=63, rejected=21`，21 条拒绝理由全是
`overrides do not implement l_main`。

**根因**：E18 用 manifest 格记录的 argv 里的 `--overrides` 重推 `_arm_match`，而 **reused 格的 `command` 是 null**
（stage A 没启动它们，是 E14 复用审计收编来的），于是 `overrides={}`、`head_type` 取不到 `shared` ⇒ 拒绝。

**为何这不是边角**：`SMOOTH_SETTINGS = REUSE_SETTINGS_FULL`（7 个 setting）**正好等于**被复用的那 7 个，
`RANK12_SETTINGS` 的 6 个也全在其中 ⇒ E18 的 **78 行（42 smooth + 36 rank12）无一例外**。

**影响面**：**§4.6 的数字不受影响**——`e18_writeback.build_row1/build_row5` 的 baseline 取自 E14 的
`results.csv`（`index_by_setting(e14_rows, arm_filter="l_main")`），与本出处列无关。受损的是**审计链**：
产物会一边记录 21 条拒绝、一边把 78 行基线出处留空，而 manifest 自己写着 `arm_match_used_for_baselines`。

**修法**（`e18_negative.py`）：为 reused 格加**第二条准入路径**——从它实际指向的 run 的 `config.json`
用**同一把尺子**（`_arm_match`，E14 复用审计当初用的那个）重推指纹；`gate_init`/`learning_rate` 取自
该 config，`eval_root` 取 manifest 记下的 `source.root`。config 缺失或 config 不满足 `l_main` ⇒ **照样拒绝**
（不凭空信任 `source`）；new 格的 argv 路径保持原样，防手改。同时 **new 格的 `eval_root` 改为 manifest
所在 root**：实测 411 个 new 格记录的 `--output-dir` 全是模板值 `/tmp/e14fix`，若照抄会写进产物。

**验证**（真产物，2026-09-20）：`resolved=84 (new=63, reused=21), rejected=0`，smooth 覆盖 **21/21**；
12 个单测（`tests/test_phaseformer_L_reuse_baselines.py`，含 4 个否定对照）；全套 400 passed。
并已固化为阶段 2 pre-flight 判据 C3（`check_phase2_consumers.py`），在 E18 那 3–5 小时**之前**就会失败即停。
详见 `docs/PhaseFormer_L/audit/paper_code_consistency.md` §16。
