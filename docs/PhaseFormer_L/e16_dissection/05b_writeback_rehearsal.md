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
