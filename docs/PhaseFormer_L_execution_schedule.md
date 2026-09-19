# PhaseFormer-L 实验总排期（远程 8 卡执行契约）

> 状态：**已登记，执行中（2026-09-18）。**
> 本文是 `docs/PhaseFormer_L_experiment_plan.md`（工作包 WP0–WP6 与缺陷 G1–G18/D1–D6）的
> **排期与作业化**版本：把 WP 拆成可独立执行的实验单元 E14–E19，规定每个单元的六阶段流程、
> 文档命名、产物路径、8 卡分配与依赖关系。
> 冲突时的优先级：`minipaper §4 的表格` ＞ 本文的排期 ＞ `experiment_plan` 的 WP 编号。
>
> **本地不运行任何实验。** 全部训练、分析、汇总、作图脚本都在远程 A800 上执行；
> 本地只做代码编写、静态检查（语法/单元测试可本地跑）与文档回填。

---

## 1. 已冻结的决策（2026-09-18 用户裁定）

| # | 决策 | 取值 | 影响 |
|---|---|---|---|
| D-1 | §4.0 判定门槛 A–D | **按 minipaper 建议值冻结**：A 无一 setting 双指标回退 >1.0%；B s=1 数据集上 ≥3/4 setting 双优；C 不设最低数目；D q=1/8 vs direct 宏平均 │ΔMSE│、│ΔMAE│ ≤0.5% | 写入 minipaper §4.0，去掉"待定"字样 |
| D-2 | §4.2 新格子超参 | **统一用 preset 默认 `gate_init=0.2`、`lr=1e-3`**；复用格保留其 Stage-0 冻结值 | 省下 Stage-0 扩展（约 84 runs）；**表内出现两套超参协议，须在 §4.2 表注显式披露** |
| D-3 | §4.5 条件性学习范围 | **7 个 test-selected setting** | 新训 24 runs；须披露这 7 个 setting 的来源 |
| D-4 | §4.6 行 1 平滑复测范围与档位 | **7 个 test-selected setting；2 档 = causal EMA `smooth_ratio ∈ {0.5, 1.0}`（`alpha=0.08`）** | 新训 42 runs；**2026-09-19 更正**：早期草案写成 boxcar 0.5 加 causal EMA alpha=0.08，但 PhaseFormer-L 的 `shared` 头只实现 causal EMA（boxcar 仅存在于 `pooled_lowrank`），且 `alpha=0.08` 是算子默认值，两个档位数值上完全相同；已改为同算子的两个不同强度 |
| D-5 | §3.4.3 的开关 `s` 与阈值 `ν*` | **`s` 不进入模型定义**：PhaseFormer-L = 修正器**恒定启用**；`s` 仅作 §4.2/§4.7 的**诊断列**（预测该数据集是否需要电平通道，并如实报告判对/判错） | §4.2 的"PhaseFormer-L（含开关）"与"always-on"两列**合并为一列**；主张 B 由"模型性质"改述为"开关预测力"；**主张 A 变强**（失去在 ETTh1/ETTm1 上自动关闭的能力） |
| D-6 | A1 行范围 | **按 §4.0 协议全训 24 × 3 = 72 runs** | 见 §2.4：minipaper 的"12 格既有"前提被审计推翻 |

**D-2 的必然后果（必须在表注写明）：§4.2 表内存在三种 gate 先验，不是两种。**

| 来源 | `weak_period_residual_gate_init` | 适用格 |
|---|---|---|
| 新格（`l_main`/`l_q1_4`/`l_q1_8`） | **0.2**（D-2 preset 默认） | 51 + 51 + 51 + Traffic 36 |
| preset 自持（`l_rcrf`/`a1`） | **0.5**（`rcrf_nlinear_plain` 与 `gold_combo_*` 在 preset 内部定义，`arm_command` 不注入 0.2） | 84 + 72 |
| 复用格（Stage-0 冻结） | **0.5 或 0.2**（逐 setting，见 §2.1） | 81 格 |

冒烟实测已确认这三种取值同时出现在产物里（`Electricity-336`：`l_main` gate=0.2、`l_rcrf` gate=0.5、`a1` gate=0.5）。

**D-5 的关键后果：主张 A 变为实质性检验。** 去掉开关后，PhaseFormer-L 在 ETTh1/ETTm1 上不再能"按构造不劣于" matched rerun。§4.1 的先导证据在这两个数据集上是 **−1.3% / −2.7%**，均超过主张 A 的 1.0% 回退上限，因此**主张 A 有可能不达标**；按 §4.0 的报告规则，须如实报告为"未达预注册门槛"，不得改用其他口径重述。这正是 D-5 的代价，也是它作为检验的价值。

**§3.4.3 的 `ν` 冻结结果（训练集，无 test）**：三个候选统计量中只有 `tau_hat_steps` 能分开已知符号的数据集——

| 候选 `ν` | 正类（Weather 96.9 / ETTh2 88.2 / ETTm2 63.6 步） | 负类（ETTh1 51.1 / ETTm1 34.9 步） | 可分离 |
|---|---|---|---|
| `cycle_level_std` | 0.507 / 0.418 / 0.348 | 0.371 / 0.489 | **否（反序）** |
| `last_cycle_shift` | 0.467 / 0.398 / 0.336 | 0.363 / 0.434 | **否（反序）** |
| `tau_hat_steps` | 96.9 / 88.2 / 63.6 | 51.1 / 34.9 | **是**，可分离区间 (51.11, 63.58) |

**冻结**：`ν = tau_hat_steps`，`ν* = 57.35`（区间中点，规则式定义、只用到训练集统计量）。
因为 `s` 不进入模型（D-5），`ν*` 只决定**诊断列**的取值，不影响任何训练或模型选择，因此
即使放错也不会污染主结果。预测结果为：`s=1` ∈ {ETTh2, ETTm2, Weather}（12 个 setting）、
`s=0` ∈ {ETTh1, ETTm1, Electricity, Traffic}。**Electricity（τ̂=54.5）落在可分离区间内部**，
其判定对 `ν*` 的位置敏感，须在 §4.7 单独披露；而实测 Electricity-336 的修正器相对 matched
`phase_only` 是 **+3.6%**（0.16768 → 0.1617，seed 2021），即**开关在 Electricity 上判错**——
按 minipaper §5.6 如实报告。

---

## 2. 实验单元登记（E14–E19）

编号续接既有证据账本 E1–E13（见 `experiment_plan` §3），一一对应 minipaper 的 §4.2–§4.7。

| ID | minipaper | 实验名 | 类型 | 新训 runs | 前置依赖 |
|---|---|---|---:|---:|---|
| **E14** | §4.2 | 主结果矩阵（24 主 setting + 4 Traffic 附录） | GPU 训练 | **363** | `ν*` 冻结、8 卡空闲 |
| **E15** | §4.3 | 相位补空间维数（28 行 + 三类图） | CPU 分析 | 0 | Traffic 数据 |
| **E16** | §4.4 | 训练头解剖 + 支路私有输入干预（含随机 RRR 子空间对照） | 评估/分析 | 0 | E14、E15、E17 |
| **E17** | §4.5 | 条件性学习四臂 | GPU 训练 | **24** | E15（条件 RRR 方向）、E14 |
| **E18** | §4.6 | 负对照补做（平滑 2 档 / SVD 截断 28 / 边界消融 rank∈{1,2}） | GPU + 评估 | **78** | E14 |
| **E19** | §4.7 | 电平非平稳度统计量 + 预测力表（28 行 + 2 列 ρ） | CPU 分析 | 0 | 阶段 1 无依赖；阶段 2 需 E14 |

**新训合计 465 runs。**

### 2.1 复用清单（经审计、不重训）

| 臂 | 复用范围 | 来源 | 审计依据 |
|---|---|---|---|
| `phase_only`（`no_residual`） | 6 setting × 3 seed = 18 cell | E8 `top2_direction_retention_v1` | 三 seed、720/Huber/30ep/best-val/单次 test |
| `weak_residual`+`shared` | 7 setting × 3 seed = 21 cell | E3 `rank_sweep_2_multiseed_stage1_20260914_summary` | E8 `reuse_audit.json` 已判 accepted（逐格 gate/lr 精确匹配） |
| `pooled_lowrank` q=1/4 | 7 setting × 3 seed = 21 cell | E3 同上 | 同上（`config=q=0.25`） |
| `pooled_lowrank` q=1/8 | 7 setting × 3 seed = 21 cell | E3 同上 | 同上（`config=q=0.125`） |

**7 个 test-selected setting**：ETTh2-96、ETTh2-720、ETTm2-96、ETTm2-192、Weather-96、
Weather-192、Electricity-336。其中 `phase_only` 只复用前 6 个（E8 未覆盖 Electricity-336）。

> 复用格是 test-set selection 所得的集合，不是盲测；在 §4.2/§4.5/§4.6 表注逐格披露。
> **曾被否定的批次**：`joint_lowrank_rank_sweep_v1`（E1）为单 seed 且用 preset 默认
> `gate_init=0.2`，E8 的 `reuse_audit.json` 已明确 `rejected`（`gate_init=0.2 != 0.5`、
> `learning_rate=0.001 != 0.0003`），**不得**进入任何复用清单。

### 2.4 复用审计结论（2026-09-18 服务器逐 run 核对）

审计方法：扫描服务器 `research_runs/*/runs/*/config.json`（共 **690** 个 run），
按 `dataset/horizon/seed` + `mechanism` + `weak_period_residual_head_type` + `rank`
+ `pool_factor` + **无 `weak_residual_projection`**（排除冻结子空间臂）匹配，并要求
`lookback=720 ∧ loss=huber ∧ max_epochs=30 ∧ percent=100 ∧ period=24 ∧ metrics.csv 存在`。

| 复用目标 | 审计结论 | 判定 |
|---|---|---|
| `phase_only`（`no_residual`） | **恰好 6 个 setting × 3 seed**（ETTh2-96/720、ETTm2-96/192、Weather-96/192），主要来自 E8 `top2_direction_retention_v1` | 与 §4.2 的 "24 − 6" **一致** |
| `weak_residual`+`shared`（`l_main`） | **7 个 setting × 3 seed**（上述 6 个 + Electricity-336），来源 E3 系 `rank_sweep_2_stage1` 与 `..._multiseed_stage1_20260914_{v3,v4,repair_v1}` | 与 §4.2 的 "24 − 7" **一致** |
| `pooled_lowrank` q=1/4（`rank=H/4`） | **7 个 setting × 3 seed** | **一致** |
| `pooled_lowrank` q=1/8（`rank=H/8`） | **7 个 setting × 3 seed** | **一致** |
| `rcrf_nlinear_plain`（L-rcrf） | **0 个 run**：服务器全库无此 mechanism 的任何 run | 与 §3.4.1 "已实现为正式对照；D0 validation 有单 seed 记录" 相容（实现存在、无三 seed 产物）→ 需 **28 × 3 = 84 全训** |
| `gold_combo_reliability_s2`（A1） | **0 个 run**：服务器全库无任何 `gold_combo*` mechanism，且 `research_runs/` 下无 `gold_combo_*` 目录 | **与 minipaper 冲突**，见下 |

**审计推翻的两项 minipaper 陈述（须在 §4.2 表注或勘误中处理）**：

1. **A1 的"12 格既有"在本分支不可得。** `gold_combo` 系产物在**本地与服务器两侧都不存在**
   （本地 `research_runs/` 13 个目录、服务器 43 个目录均无 `gold_combo_*`）。`docs/agent-log.md`
   把该证据登记在 `research_runs/gold_combo_stability_v1/`、`research_runs/gold_combo_screen_runs/`，
   两者都不在当前分支的工作副本中。更关键的是，agent-log 1292 行记录的那批 A1 运行使用
   **MAE loss、batch 256、lr 3e-4**，而 §4.0 规定 **Huber**——即便找回也不满足 §4.0 的"同协议"。
   因此 §4.2 的 A1 行**不存在可复用格**，"补 H336/720"的前提不成立：该行要么按 §4.0 协议全训
   （24 × 3 = 72 runs），要么整行留空并披露。
2. **`phase_only` 的"额外三 seed 格"不可用。** 服务器上 ETTh1-96、ETTm1-96 确实存在三 seed 的
   `no_residual` 记录，但它们来自 `joint_lowrank_rank_sweep_v1`（E1，已被 E8 审计 rejected）与
   `joint_pooled_lowrank_phase_a_scratch`（scratch 根目录，非登记实验），**不得**计入复用。

**审计同时确认**：`weak_residual_projection=frozen_subspace` 的 E8 `keep_direction_*` 臂（36 个 run，
含 18 个 `direct_nlinear` + 18 个冻结臂）必须按投影臂排除，只有其中标记
`weak_residual_projection_arm=direct_nlinear` 的 18 个是 E3 复用格的镜像，不重复计入。

### 2.2 E14 主表矩阵明细（**已在服务器按 `--verify` 对账通过**，2026-09-18）

`--stage plan --verify` 输出的 `total = 492` 个 cell，其中 **新训 411 runs、复用 81 格**，
`reuse_cells_resolved = 81`、`missing = []`。

| 行 | 配置 | setting 域 | 复用 | 新训 runs |
|---|---|---|---:|---:|
| `phase_only` | `--mechanism no_residual` | 24 + Traffic 4 | 6 | 18×3 + 4×3 = **66** |
| PhaseFormer-L（= always-on，D-5） | `weak_residual` + `head_type=shared`，gate 0.2 | 24 + Traffic 4 | 7 | 17×3 + 4×3 = **63** |
| L-q1/4 | `pooled_lowrank`，`pool_factor=1`，`rank=H/4` | 24 + Traffic 4 | 7 | 17×3 + 4×3 = **63** |
| L-q1/8 | `pooled_lowrank`，`pool_factor=1`，`rank=H/8` | 24 + Traffic 4 | 7 | 17×3 + 4×3 = **63** |
| L-rcrf | `rcrf_nlinear_plain` | 24 + Traffic 4 | 0 | 28×3 = **84** |
| A1（incumbent 参照，D-6） | `gold_combo_reliability_s2` | 24（不含 Traffic） | 0 | 24×3 = **72** |
| **合计** | | | **81** | **411** |

**`always-on` 列与 PhaseFormer-L 合并（D-5）**：`s` 不再进入模型定义，PhaseFormer-L 就是
修正器恒定启用的 `weak_residual`，因此 minipaper §4.2 原表的
"PhaseFormer-L（含开关 `s`）"与"always-on"两列数值相同，须合并为一列，并把 `s` 移到诊断列。
`--verify` 的 81 个复用格与 §2.1 的审计清单逐格一致。

**G8 的三类图与 G7 的 28 行由 E15 产出（CPU，无训练）**，不计入本表。

**必答项落点**：(a) ETTh2 四 horizon 对 `phase_only`/Golden 的差距 + FITS 引用数字（外部参照，仓库无源）；
(b) ETTh1/ETTm1 上的 `g` 均值（D-5 后不再有"开关关小"的能力，改为报告逐 dataset 的 `g` 均值 + 诊断列 `s`）；
(c) q=1/8 与 direct 的三 seed 差是否 ≤0.5%。

### 2.3 E17 / E18 明细

**E17（§4.5，7 个 test-selected setting × 3 seed）**

| 臂 | 来源 | 新训 runs |
|---|---|---:|
| `direct`（无瓶颈） | 复用 E3 的 7×3 = 21 cell | 0 |
| 冻结独立-RRR 方向 1 | 复用 E8 `keep_direction_1` 的 6×3 = 18 cell | 3（补 Electricity-336） |
| 冻结条件-RRR 方向 1（**全新臂**） | 投影器由各 setting **train split** 单独计算并冻结 | 21 |
| PhaseFormer-L（联合） | 复用 E14 的 `weak_residual` 7 格 | 0 |
| **合计** | | **24** |

> **须在表注声明的两处**：(1) `direct` 与 `PhaseFormer-L（联合）` 在本实现下是**同一配置**
> ——PhaseFormer-L 的修正器就是与主干联合训练、无瓶颈约束的头；两列数值相同是构造使然，
> 不是两次独立实验。(2) Electricity-336 的冻结独立臂为新增（E8 未覆盖），其 6/7 复用比
> 其余 3 列少，逐格标注。

**E18（§4.6）**

| 子项 | 内容 | 域 | 新训 runs |
|---|---|---|---:|
| 行 1 | 输入平滑 2 档复测：causal EMA `smooth_ratio ∈ {0.5, 1.0}`，`alpha=0.08`（见 §1 D-4 更正） | 7 setting × 3 seed × 2 档 | 42 |
| 行 3 | SVD 截断 vs 秩约束训练扩展到 **28 setting** | 读 E14 的全秩与低秩 checkpoint | 0（评估） |
| 行 5 | 边界消融 `pooled_lowrank` `rank∈{1,2}`（绝对秩） | 6 setting × 3 seed × 2 rank | 36 |
| 行 2 / 行 4 | minipaper 标 "—"，**不补做** | — | 0 |
| **合计** | | | **78** |

> `smooth_ratio=0.5` 与 `alpha=0.08` 的取值理由：E4/E5 的 5 档网格在 7 setting 上
> "越平滑越差"，取各自网格中最具代表性的伤害档（boxcar 中档、causal EMA 的 E5 固定档），
> 与既有 14 个 (setting, 算子) 组合同 setting 直接对照。**此口径在 E18 冒烟前冻结，不得事后调整。**

---

## 3. 六阶段流程（每个实验单元强制）

每个 E 单元必须依次完成六个阶段，**每阶段产出一份独立命名的文档**，缺一不可进入下一阶段。

| 阶段 | 文档 | 内容 | 通过判据（Gate） |
|---|---|---|---|
| 1 撰写代码 | `01_plan.md` | 实验设计、命令构造、复用审计规则、判定口径、预期产物 | 设计可追溯到 minipaper 的具体表项 |
| 2 静态检查 | `02_static_check.md` | `py_compile`、`pytest`、CLI `--help` 校验、配置 hash 复核、**dry-run 命令清单逐条校对** | 编译通过、单测通过、dry-run 的 run 数与 §2 明细逐一相符 |
| 3 冒烟测试 | `03_smoke.md` | 每类 setting（最小/最大/高通道）各 1 个 `--stage smoke --max-eval-samples` 跑通，记录 **per-epoch 实测秒数** | 链路跑通、指标量级合理、产出目录结构正确 |
| 4 正式实验 | `04_run.md` | 全量提交 8 卡、记录 HEAD sha、run manifest、实时进度与失败重试 | 全部 cell 产出 `metrics.csv` + `result.json`，无 silent failure |
| 5 结果审校 | `05_audit.md` | 逐 cell 校验：复用/新训来源、协议字段、NaN/异常值、与 §2 明细逐行对账、抽样复算 | 行数、范围、来源 100% 对账；异常项有结论 |
| 6 回填 | `06_writeback.md` | 回写 minipaper §4 表项 + `research_runs/` 原始产物路径 + `docs/agent-log.md` 条目 | 表内每个空位都有原始产物溯源；未完成项留白不用推断值 |

### 3.1 文档与产物命名规范

```text
docs/PhaseFormer_L/<EID>_<slug>/          # 每个实验独立目录，互不混放
  01_plan.md  02_static_check.md  03_smoke.md  04_run.md  05_audit.md  06_writeback.md

research_runs/phaseformer_L_<eid>_<slug>_v1/   # 原始产物（数值权威副本）
  runs/<run_id>/...                            # 训练 run
  results.csv                                  # 汇总（服务器上生成）
  figures/                                     # 服务器上生成
  run.yaml                                     # 协议与环境快照

scripts/phaseformer_L/<eid>_*.py               # 该实验的 runner/analyzer（服务器执行）
```

- `<EID>`：`e14` … `e19`；`<slug>`：`main` / `dimension` / `dissection` / `conditional` /
  `negative` / `predictive`。
- **禁止**跨实验共用输出根目录；`--output-dir` 必须显式传入（沿用 E8 的教训：
  runner 默认 `research_runs/search_v1` 会把 run 撒进无关根目录）。
- 汇总与分析脚本一律在服务器执行，产物落 `research_runs/phaseformer_L_<eid>_<slug>_v1/`
  后由 rsync 下行（`--exclude` 权重），本地不重算。

---

## 4. 8 卡排期

### 4.1 依赖图

```text
Traffic 数据补齐 ─┬─→ E15 (§4.3, CPU)  ──┐
                  │                       ├─→ ν* 冻结 ─→ E14 (§4.2, GPU 主力) ─┬─→ E16 (§4.4)
                  └─→ E19-阶段1 (CPU) ────┘                                      ├─→ E17 (§4.5)
                                                                                  ├─→ E18 (§4.6)
                                                                                  └─→ E19-阶段2
```

### 4.2 三波次排期

| 波次 | 内容 | 卡数 | 预计 wall-clock |
|---|---|---|---|
| **W1（即时，零训练）** | Traffic 数据补齐 → E19 阶段 1（6 已知 setting 训练集统计量）→ **冻结 `ν*`**；同时 E15 在 CPU 上跑 28 行二阶矩 | 0 GPU（CPU） | 1–3 h |
| **W2（GPU 主力）** | E14 主表矩阵 363 runs，8 卡轮转；E17 的 21 个冻结条件臂与 E18 的 78 runs 插空 | 0–7 | 12–20 h |
| **W3（分析）** | E16 解剖与干预、E18 行 3、E19 阶段 2、全部审校与回填 | 评估为主 | 3–6 h |

### 4.3 实测预算基线与 E14 估算

服务器 677 份 `metrics.csv` 的 `elapsed_sec` 中位数（服务器 torch 2.6.0 / Lightning 2.6.5）：

| setting | 中位耗时 | setting | 中位耗时 |
|---|---:|---|---:|
| ETTh1-96/192 | 85 / 74 s | Weather-96/192 | **1100 / 683 s** |
| ETTh2-96/192/720 | 47 / 37 / 57 s | Electricity-336 | **1717 s** |
| ETTm1-96/192 | 253 / 244 s | Traffic-* | **待冒烟实测** |
| ETTm2-96/192 | 185 / 114 s | | |

按上表外推（H336/720 按同数据集趋势插值/外推），**不含 Traffic 的 465 runs 合计约 61 GPU·h**。
Traffic 为唯一未知量（862 通道、batch 8），须先冒烟实测每 epoch 秒数再定档，见 §6。

**8 卡分配原则**（沿用既有做法，见 `experiment_plan` §8）：

1. **一卡一 run，互不重叠**；用 `CUDA_VISIBLE_DEVICES=N` 选卡，启动前 `nvidia-smi` 复核该卡空闲。
2. **长 run 优先占卡**（Weather/Electricity/Traffic），短 run（ETT 系）填缝，避免尾部空卡。
3. 卡 0–7 全用；本轮实测 8 卡全空闲（此前被他人 vLLM 占用的 6/7 已释放），若他方重新占用则退回 0–5 六卡并顺延。
4. `--num-workers` 统一 **4**，避免 E3 记录的 load average 失控；并发 run 数 ≤ 8。
5. 每波次用 `setsid nohup` 提交，日志写 `~/niuyiming/<eid>_<wave>.log`，**不随 SSH 断开而中断**。
6. 启动前把 `git log --oneline -1` 写入日志首行（版本可追溯，见 `REMOTE_SERVER.md`）。

---

## 5. 环境与协议固定项（全实验统一）

| 项 | 取值 |
|---|---|
| 解释器 | `/home/yyk/yyk03/miniconda3/envs/time/bin/python` |
| 仓库 | `~/niuyiming/PhaseFormer`，分支 `weak_residual_nlinear_bottleneck`，HEAD `6e6f900f` |
| lookback / period | 720 / 24 |
| stage / epochs | `confirm` / 30（`≤30` 与 §4.0 一致） |
| loss | huber |
| seed | 2021 / 2022 / 2023 |
| checkpoint | 最低 validation loss（best-val） |
| test 读取 | 每 checkpoint **只读一次**；`--evaluate-test` 仅 `--stage confirm` 允许 |
| 新格超参 | preset 默认 `gate_init=0.2`、`lr=1e-3`（D-2） |
| 复用格超参 | 保留其 Stage-0 冻结值（表注披露） |
| 数据 | `resources/all_datasets/{ETT,electricity,weather,traffic}/` |
| 环境差异披露 | 服务器 torch 2.6.0 + Lightning 2.6.5，与金标准的 RTX 4090 环境不同 |

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **Traffic 训练成本未知**（862 通道、batch 8） | 可能使总预算翻倍 | W2 前先做 Traffic 冒烟实测 per-epoch 秒数；若单 run > 3 h，则只保留 `phase_only` + PhaseFormer-L 两行共 24 runs，并据此收窄附录 |
| Traffic 数据源 | §4.3/§4.7 的 28 行、§4.2 附录全部阻塞 | 已在服务器后台拉取；若公开源不可达，改用本地下载后 scp 上行；再不可行则向用户报告并把 28 行收窄到 24 行（须用户裁定） |
| 混合超参协议（D-2） | 审稿人质疑复用格与新格不可比 | §4.2/§4.5/§4.6 表注逐格标注来源与协议；配对比较只在同 setting 内声明 |
| Electricity-336 内存 | 高维 setting 的分析阶段可能 OOM | 沿用 E10 的分片/缓存机制；分析脚本必须流式；单 cell 端到端验证通过再放大 |
| 长任务启动后才发现缺陷 | 浪费数十 GPU·h | 强制六阶段 Gate：阶段 2 的 dry-run 清单与 §2 明细逐条对账、阶段 3 冒烟必须覆盖"最大 setting" |
| code 同步中断运行中的任务 | 训练进程读到新旧混杂代码 | 同步前 `pgrep -af "python"` 确认空闲；任务期间**不做** `reset --hard` |
| 他人重新占用 GPU 6/7 | 可用卡数降到 6 | 排期按 8 卡设计但保留 6 卡退化路径；每波次启动前复核 `nvidia-smi` |

---

## 7. 与既有文档的关系

- `docs/PhaseFormer_L_minipaper.md` §4：**回填目标**（本文不修改其 §4.1 已定数字）。
- `docs/PhaseFormer_L_experiment_plan.md`：执行契约（WP0–WP6、G1–G18、D1–D6）；
  本文是其排期化。WP0-2 / WP3 的 588-run 矩阵已按 §10 记录**作废**，由本文 §2 取代。
- `docs/PhaseFormer_L_experiment_gap_audit_2026-09-18.md`：minipaper 明示缺口与本地产物对照。
- `docs/agent-log.md`：每个 E 单元完成阶段 6 后**追加**一条记录，不覆盖历史。
- `REMOTE_SERVER.md` / `MANAGE_RULES.md` / `HOW_TO_DO_RESEARCH.md`：上行 bundle、下行 rsync、
  环境与记录规范；本文全部作业遵守。

---

## 8. 执行记录（追加式）

| 日期 | 单元 | 阶段 | 摘要 |
|---|---|---|---|
| 2026-09-18 | — | 登记 | 建立本文；完成侦察（minipaper §4 需求、复用链审计、服务器 8 卡全空闲、6 数据集在服务器、Traffic 缺失）；冻结 D-1–D-4 四项决策；登记 E14–E19 与 465 runs 排期 |
| 2026-09-18 | E00 | 数据 | 补齐 Traffic：下载 `laiguokun/traffic.txt.gz` → 17,544×862；按仓库约定把末列改名 `OT`（`Dataset_Custom_Multi` 需要），加写出后自校验；`scripts/phaseformer_L/e00_prepare_traffic.py` |
| 2026-09-18 | 审计 | — | 服务器 690 个 run 全量清点：复用链确认 81 格；**推翻** minipaper 的 A1「12 格既有」（全库 0 个 `gold_combo*`，且旧批用 MAE loss）与「switch 由 6 个已知 setting 拟合」（证据支持 5 个已知符号数据集） |
| 2026-09-18 | 决策 | — | 用户裁定：门槛按建议值冻结；新格用 preset 默认超参；§4.5 与 §4.6-行1 取 7 个 test-selected setting；**A1 全训 24×3** |
| 2026-09-18 | E19-1 | 1–4 | 28 setting 训练集电平统计量完成（`phaseformer_L_e19_predictive_v1`）；**`tau_hat_steps` 是唯一能分开已知符号数据集的候选**（可分离区间 51.11–63.58），`cycle_level_std`/`last_cycle_shift` 均为反序 → 冻结 `ν=tau_hat_steps`、`ν*=57.35`；修复 τ̂ 结果侧封顶缺陷（近单位 ρ 曾报出 2176 步 > 720 步窗） |
| 2026-09-19 | 决策 | — | 用户裁定 **D-5：`s` 不进入模型定义**（PhaseFormer-L 恒定启用），`s` 降为 §4.7 诊断列；`ν*` 因此只影响诊断列、不污染主结果。已回填 minipaper §3.4.1/§3.4.3/§4.0/§4.2/§4.7 与 §5 |
| 2026-09-19 | E15 | **1–6 完成** | §4.3 的 **28 行表 + 三类图**回填 minipaper；`--verify-existing` 门通过（7/7，moments 相对差 0.0，111/111）；28/28 完成、exit 0、16m42s；阶段 5 审校 11/11 通过。关键新事实：`pred_dims_90` 上界 **7**（Traffic），Traffic 的 `b_1` 一致落在 τ=168 且 `used_var_share(1)` 最高（0.177–0.353） |
| 2026-09-19 | E14 | 1–4（运行中） | 静态门 `verify_ok`（492 cell / 新训 411 / 复用 81）；冒烟 **18/18** 通过（6 臂 × 3 设定，1 epoch）并实测 Traffic 单 epoch 138 s；19:10 起 8 卡正式运行，按成本降序先发 Traffic |
| 2026-09-19 | E17 | 2–3（投影器完成） | 7/7 投影器完成、exit 0、`reproduction_failures: []`；独立路线对 E8 的 6 个已发布投影器 `abs_cos = 1.0`。**关键发现：`D_cond` 与 `D_ind` 的首方向在 6/7 setting 上 `│cos│ ≥ 0.9991`（几乎同一方向），仅 Electricity-336 为 0.0066（近正交）** → §4.5 的对照检验力集中在 Electricity-336 一格 |
| 2026-09-19 | E16/E17‑训练/E18 | 1（代码就绪） | 代码与 `01_plan` 已提交；E16 的 63 cell dry-run 确认其依赖 E14 checkpoint；E17 训练 24 runs、E18 78 runs 待 E14 让出 GPU |
| 2026-09-19 | 修正 | — | D-4 的「2 档」被证伪并更正：PhaseFormer-L 的 `shared` 头只实现 causal EMA（boxcar 仅在 `pooled_lowrank`），且 `alpha=0.08` 是算子默认值 → 原两档数值上完全相同；改为 `smooth_ratio ∈ {0.5, 1.0}` |
| 2026-09-19 | E16 | 2–3 修复后通过 | 阶段 2 挡下**作用域张成笛卡尔积**缺陷（7→16 setting，改显式 pair 列表，验证 63 cell）；阶段 3 冒烟挡下 2 个缺陷并修复：①**模型 bundle 缓存键** `(dataset,horizon,batch_size)` 导致第一个臂的头类型被复用 → 21 个低秩 cell 全部加载失败；②不变量把"子集 512/2785 窗口"与"全划分 `val_mse`"相比（判据定义错误，非代数错误）。修复后复测：稠密头 `rel_gap=2.15e-10, max_abs=0`（映射精确）、低秩头 2.97e-05；新增数秒级前置检查 `--dry-run --verify-checkpoint-heads` |
| 2026-09-19 | E17 | 2–3 通过（两侧） | 投影器 7/7（独立路线对 E8 的 6 个已发布投影器 `abs_cos=1.0`）；训练侧 `--stage plan --verify` 为 84 cell / **24 新训**，逐 setting 冻结超参与 E8 `FROZEN` 表逐格一致，24/24 命令无 `--evaluate-test` |
| 2026-09-19 | E18 | 2–3 通过 | 行 1+行 5 共 **78 cell**（42 平滑 = 21×0.5 + 21×1.0；36 边界消融 = 18×rank1 + 18×rank2），两个平滑档**数值上确实不同**（卡点已固化）；行 3 挡下**默认范围漏 Traffic** 缺陷（24→**28** setting），7 个复用 setting 全部解析 |
| 2026-09-19 | E14 | 早期审校 | 前 7 个 cell 逐项过 **8/8 阶段 A 不变量**（含最关键的「阶段 A 绝不读 test」）；纠正审校脚本自身的误判（`epochs_completed == requested` 不是不变量：早停 `patience=8`，且复用格同样早停，两类格子协议一致） |
| 2026-09-19 | E19-1 | 1–6 完成 | 28 setting 统计量 + `ν*=57.35` 冻结已回填 §4.7 表注与 §3.4.3；审校 9/9 |
| 2026-09-19 | 工具 | — | `e14_writeback.py`（全量聚合 + 主张 A–D 判定 + §4.2 表行）经合成冒烟跑通全部代码路径，并挡下 Golden 表解析缺陷（正则 `[A-Za-z]+` 匹配不了含数字的 `ETTh1` → 只解析 12/28 行，已修） |
| 2026-09-19 | G4 闭合 | — | 必答 (a) 需要"FITS 的引用数字"，而仓库金标准无任何外部模型。已补齐 FITS（ICLR 2024 Spotlight, L=720）的 28 个 MSE 到 `docs/PhaseFormer_L_external_refs.md`（来源：官方仓库 `VEWOXIC/FITS` README 的 Result Update 表，2026-09-19 抓取），并接入 `e14_writeback.py` 输出 `fits_mse` 列与 `claims.json` 的 `must_answer_a` 块。**只比 MSE**（源表无 MAE）；FITS 不进入主张 A–D 判定。起点：Golden 的 ETTh2 四格 MSE 全部高于 FITS 1.5%–6.2% |
| 2026-09-19 | §4.2 参数量列 | — | 补 `e14_params.py`：由 checkpoint 参数**形状**（`mmap` 只读）拆出 `residual_params`/`backbone_params`，与 `metrics.csv:parameter_count` 交叉校验；已解析的 94 个 cell **全部相等**（`total_mismatches: []`）。回填工具按 `(arm, horizon)` 聚合并校验同 horizon 内 3 seed 一致；FLOPs 明确不报（原文 Table 4 口径未复现） |
| 2026-09-19 | E18 行 3 | — | 挡下默认范围漏 Traffic 的缺陷（24 → **28** setting）；7 个复用 setting 全部解析，其余待 E14 |
| 2026-09-19 | 阶段二流水线 | — | 新增 `scripts/phaseformer_L/run_phase2_after_e14.sh`：E14 之后的六步（单次 test 读取 → §4.7 ρ → §4.2 回填 → E16 → E17 → E18），带**完成守卫**（实测在 E14 运行时正确拒绝启动，exit 1）与逐步退出码 |
| 2026-09-19 | 阶段二修正 | — | 修正流水线步骤 6 的**顺序错误**：`e18_negative.py --verify` 的语义是"任一计划 cell 没有已完成 run 就失败"，即**阶段 5 的完整性审计**，而不是开跑前门（实测在训练前调用会 exit 1 并报 `78 of 78 cells have no matching completed run`）。若按原顺序执行，会在 E14 完成、跑完 78 个 run 之前就中止。现改为"先训练 → 再 `--verify --dry-run` 审计"。同时为 E16 加上其自身门（`--dry-run --verify-checkpoint-heads`，数秒，可提前抓出头类型错误） |
| 2026-09-19 | E16 验收判据 | — | 核对参照产物实际只覆盖 **6** 个 setting（`lowrank_checkpoint_information_v1` 无 Electricity-336，系 E10 因 13.9 GiB OOM 排除该格的后果）。E16 对缺参照的 cell **跳过而非判失败**（代码 `:2337-2344`），因此表注须写明"parity 只对 6 个 setting 成立、Electricity-336 的解剖为新算而非复现"，并给出全量运行的精确验收值（`probe_cells ≈ 72`、`checkpoint_path_mismatches: []`） |
| 2026-09-19 | 主张 C 口径 | — | 两份治理文档对"相对 Golden 的提升"定义不同：minipaper §4.0 主张 C 用"三 seed **均值+std**"，`PhaseFormer_gold_standard.md` §4 用"三 seed **均值**"。回填工具改为**同时输出两个计数**并注明各自定义（合成数据验证两者可区分：12 vs 13），表注须同时给出 |
| 2026-09-19 | E14 审校 | **复用污染事件** | 给 §4.2 补门值列时发现 `ETTh2-96` seed-2023 的门值是 0.000114（另两 seed 为 ~0.49）。追查确认 `..._20260914_v3` 批次含**配置损坏**的 run（`gate_init=2023` 把 SEED 写进了门初始化、`learning_rate=0.5` 为审计网格 500 倍、val_mse 0.294 vs 正常 0.204），而 E14 的复用解析把它接受为合法的 `l_main`/`l_q1_4`/`l_q1_8` 复用格。根因：`_protocol_ok` 只校验五个结构性字段，**不校验冻结超参的合法性**；`--verify` 只断言"能解析"，不断言"解析正确"。修复：新增 `gate_init ∈ (0,1)` 与 `learning_rate ∈ (0,1e-2]` 两条判据，复用索引逐格落盘 gate/lr；替换运行中的 manifest（原文件归档为 `stage_a_manifest.prelaunch_contaminated.json`）。**影响面**：仅 3 个复用格受影响，新训格与 `new/reused` 划分完全不变（总 492）；**E3 权威审计对 `lr0.5` 批次的引用数为 0**，故 §4.1/§3.4.2 的既有数字从未被污染；修复后三格恢复到与 E3 审计**哈希逐字相同**的 run |
| 2026-09-19 | E14 门值列 | — | `e14_params.py` 增加从 checkpoint 直接读 `sigmoid(weak_period_residual_gate)`（形状 `(1,1,enc_in)`，公式与 `PhaseFormer.learned_residual_gate()` 逐字一致）。**必要性**：E3 系的 `metrics.csv` 没有门值列，而它们正是 7 个 test-selected setting——若不从 checkpoint 恢复，§4.2 的 `g` 列与 §4.7 的 ρ-vs-g 会在最关键的格上留白。回填工具优先用 `results.csv` 的门值、缺失时回退到 checkpoint，并逐格记录来源。已交叉验证：Weather-96 0.2249、ETTm2-192 0.2083 与 E17 独立算出的投影器审计门值一致 |
| 2026-09-19 | E14 复用歧义审计 | — | 新增 `e14_reuse_audit.py`，对 81 个复用格**枚举全部合法候选**（而不只是取第一个）。结果：**45 格有多个合法候选，但 0 格存在分歧**——候选在 `gate_init`/`learning_rate`/`val_mse` 上完全一致（`val_mse` 离散度 **0.0**），逐例验证为同一确定性结果在 `_v3/_v4/_v5` 等多个批次根下重复产出。另有 **6 个非法候选**被新判据拒绝（§6 的污染事件）。因此可把"取了第一个匹配"从实现细节升级为**已验证的无关性结论**并写入 §4.2 表注 |
| 2026-09-20 | 排期细化 | — | 核算 E14 剩余工期：用 `COST_HINT` 对 360 个待完成格求和 = **66.53 GPU·h → 8.3 h（8 卡）**，并用已完成的 48 个 Traffic 格反向校准（估计 56.00 GPU·h vs 实测约 51）→ 估计偏高约 10%。**Electricity 占待完成量的 42%**（28.13/66.53），与 E10 记录的"Electricity-336 占 7 setting 总计算量 42%"同量级：高通道数设定是主要成本。E14 阶段 A 总量 ≈ 122 GPU·h / 15.3 h。另核对：运行中的格为 `electricity-h720`×4 + `traffic-h336/h720`×4，证明调度器的**成本降序**发车正确（60 个 Traffic 格已全部发车后才轮到 Electricity） |
| 2026-09-20 | E16 资源 | — | 核对 `--gpus` 语义：**只用第一个索引**（单进程分析），因此 E16 不需要 8 卡，排期上可与其它步骤错开而不必独占。按 CPU 冒烟的相对耗时外推（稠密格 808 s、低秩格 5 s），63 格在单卡上约 **2 h**；稠密格慢 60×的原因是稠密头的有效映射是 H×720（对 720×720 做 SVD）且干预带宽 720 维，而低秩格只有 rank 12 |
