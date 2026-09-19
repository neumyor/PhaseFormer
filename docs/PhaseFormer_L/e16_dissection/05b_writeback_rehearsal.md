# E16 · §4.4 解剖与干预 — 阶段 6 前置：回填工具与**真实生产者产物**的联调

> 方法：先用 E16 自身的冒烟参数**重新生成**真实产物，再把它们直接喂给回填工具
> 状态：**通过**（两个工具 exit 0）｜2026-09-20 03:56–04:08（服务器）

## 1. 为什么这一份用"真实产物"而不是合成 fixture

前三份预演（E14/E17/E18 回填）用的是**按生产者声明的 schema 合成**的输入，因为那些输入
（结果表、参数表）在跑完之前不存在。但 E16 有一个更好的选择：它的**生产者自己是可冒烟的**
（`--datasets ETTh2 --horizons 96 --seeds 2021 --arms l_main,l_q1_8 --max-batches 2 ...`，
约 9 分钟、纯 CPU），产出的 `dissection_table.csv` / `intervention_table.csv` 是**真实的**。

而 E16 的生产者行长字典由累加器**动态构造**（含 `**band_summary(...)` 的 f-string 展开），
手写 fixture 极易与真实形态漂移——本会话已因这类漂移产生多次假报警。
因此这里改为：**先生成真实产物，再直接喂给回填工具**，一次 fixture 都不写。

顺带也重新确认了冒烟本身仍然通过（此前记录的结果被清理过，故重新生成）：

```text
[cell] l_main  ETTh2-96 seed=2021 head=dense_shared   rank_dim=720
       algebra=ok(rel_gap=2.15e-10, max_abs=0.00e+00)  (551.9 s)
[cell] l_q1_8  ETTh2-96 seed=2021 head=pooled_lowrank rank_dim=12
       algebra=ok(rel_gap=2.97e-05, max_abs=0.00e+00)  (3.3 s)
{"event":"finished","cells":2,"intervention_rows":23,"dissection_rows":2,
 "algebra_failures":0,"run_metric_failures":0,"run_metric_not_comparable":2,
 "reference_parity_passed":false,"elapsed_seconds":557.2}   exit 0
```

与 `02_03_static_check_smoke.md` §6.2 事先写下的冒烟预期**逐项一致**
（`cells=2`、`intervention_rows` 同量级、`algebra_failures=0`、
而 `run_metric_not_comparable=2` 与 `reference_parity_passed=false` 正是
`--max-batches 2` 子集效应的预期结果，非缺陷）。

## 2. 回填工具：**通过**，且列名契约在真实数据上成立

```text
{"event": "finished", "intervention_rows": 2, "dissection_rows": 2,
 "missing_columns_intervention": [], "missing_columns_dissection": [],
 "arms_per_cell_observed": [11, 12], "cells_with_fewer_arms": 0,
 "per_seed_arms_observed": [11, 12], "per_seed_cells_with_fewer_arms": 0,
 "named_arms_missing": []}                                  exit 0
```

四个产物齐全：`intervention_table_44.csv`、`dissection_table_44.csv`
（及同名 `.md`）、`e16_writeback_summary.json`。

**最重要的两项**：`missing_columns_intervention: []` 与 `missing_columns_dissection: []`——
即回填工具点名的每一列都在**真实生产者产物**里存在。这把列名契约从"静态推导"
升级为"**在真实数据上实测成立**"（静态检查见 `e14_main/05_audit.md` §15，
那里报告的也是 0 缺口，但静态检查无法排除"名字对而结构不对"）。

`arms_per_cell_observed: [11, 12]`：一个 cell 观测到 11 个干预臂、另一个 12 个
（多出的应是"未干预/模型自身"那一行）。覆盖率审计**如实报告**而不是崩掉，
且 `cells_with_fewer_arms: 0` 说明按 11 臂的判据没有薄格。

## 3. §4.4 表格文字的实样（回填产物的样子）

`intervention_table_44.md`（这就是要进 minipaper 的那一行）：

| 行 | 内容 |
|---|---|
| 1 | `\| PhaseFormer-L \| ETTh2 \| 96 \| dense（r=96） \| -0.0051 \| +0.0481 \| [0.1594, 0.1604] \| +0.0436 \| [0.1618, 0.1632] \| +0.0244 \| +0.0481 \|` |
| 2 | `\| L-q1/8 \| ETTh2 \| 96 \| q=1/8（r=12） \| +0.0000 \| +0.0303 \| [0.1656, 0.1793] \| +0.0303 \| [0.1921, 0.1921] \| -0.0004 \| +0.0303 \|` |

列序与源码注释一致（11 格：model、dataset、H、q/r、Semantic-only、Semantic-drop、
**random band**、PCA-drop、**random-RRR band**、branch delta、fused delta）。

**值得记下的一点**：第 9 格（random-RRR band）确实是一个**区间**（`[0.1618, 0.1632]`），
即 §4.4 要求的**随机 RRR 子空间对照**贡献了真实的 95% 零分布带，
而不是空值或占位符——这张表的骨架与随机 RRR 对照都通了。

## 4. 边界（本联调**没有**证明什么）

* 冒烟只有 2 个 cell，因此**不能**验证 §4.4 的 21 行结构（3 模型 × 7 setting）与
  63 cell × 11 臂的完整性——那要等正式运行；
* `--max-batches 2` 使 `run_metric_*` 类判据处于 `skipped`，故本联调**不**验证
  "不变量 2"（与 run 记录值比对）在全划分下的行为；
* 数值本身来自**子集评估**，**不构成任何 §4.4 科学结论**；
* `reference_parity_passed: false` 是冒烟规模的预期结果，正式运行的验收判据仍是
  `02_03_static_check_smoke.md` §6.2 那张表（须为 `true`、`probe_cells ≈ 72`）。

## 附：`rehearse_e16_writeback.py` —— 补上仓库里缺失的那个同侪（2026-09-20）

仓库本来有 `rehearse_e14_writeback.py`、`rehearse_e17_writeback.py`、`rehearse_e18_writeback.py`，
**独缺 E16**。这不是形式问题：E16 是**唯一**没有预演脚本的回填工具，而第 4 步的调用是
`... && check_builder_outputs.py ... || true`——**回填失败不会中断链条**，
只会让 §4.4 的两张 44 表缺失，直到回填阶段才现形。本轮改了它的臂数判据（§18），正好把这个缺口补上。

脚本用**实测的逐格臂结构**（11/12/13 臂、63 格 = 3 臂 × 7 setting × 3 seed、共 762 行）合成干预表，
解剖表 21 行（3 臂 × 7 setting），只读它实际需要的列，然后：

**正对照（10 条断言全部成立）**

```text
fixture: 762 rows over 63 cells, arm counts [11, 12, 13]
[OK  ] positive control: write-back exit 0
      always_present_arms: 10 entries
      expected_arms_per_cell: 10
      observed arm counts: [11, 12, 13]
      cells_with_fewer_arms: []
      [OK  ] intervention_table_44.csv written
      [OK  ] dissection_table_44.csv written
      [OK  ] expected_arms_per_cell is the always-present count
      [OK  ] 11/12/13-arm cells are all complete
      [OK  ] per-seed cells are all complete
      [OK  ] observed counts cover the measured shape
      [OK  ] section 4.4 table = 21 rows (got 21)
```

**两个否定对照**（分别对应写回工具刻意保留的两级粒度）

```text
[OK] (a) 只删掉某一 seed 的 PCA-drop：per_seed_cells_with_fewer_arms = ['l_q1_4__ETTh2-96-s2021']，
     cells_with_fewer_arms（按 (arm, setting) 聚合）= [] —— 聚合是"各 seed 的并集"，
     单个 seed 的缺口**只能**在 per-seed 一级看到
[OK] (b) 删掉该 cell **三个 seed** 的 PCA-drop：cells_with_fewer_arms = ['l_q1_4__ETTh2-96'] ✓
```

> **我第一版否定对照的断言是错的**：我断言"删一个 seed 的臂 ⇒ 聚合也应报出来"。
> 实测聚合为空、per-seed 报出——**写回工具是对的**（它的注释恰好写明"an arm missing from one seed
> would leave the aggregate complete while quietly shrinking that arm's n"）。
> 这与本会话其它几次同类：**先分清是工件坏了还是我的判据错了**。
> 现在两个粒度各有一个否定对照，把"为什么必须同时保留两级"变成可复跑的证据。
