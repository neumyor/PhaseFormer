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
