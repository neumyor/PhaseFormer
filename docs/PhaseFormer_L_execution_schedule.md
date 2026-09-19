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
| **W2（GPU 主力）** | E14 主表矩阵 **411 个新训 run**（矩阵 492 = 411 新 + 81 复用，经 manifest 核验），8 卡轮转；E17 的 **24 个新训 run**（84 cell，其余复用）与 E18 的 78 runs 插空 | 0–7 | 12–20 h（**实测校准见 §9**） |
| **W3（分析）** | E16 解剖与干预、E18 行 3、E19 阶段 2、全部审校与回填 | 评估为主 | **见 §9 的实测投影**（E16 实测稠密格 551.9 s，是 W3 的主要成本，而非先前估计的 2 h） |

### 4.3 实测预算基线与 E14 估算

服务器 677 份 `metrics.csv` 的 `elapsed_sec` 中位数（服务器 torch 2.6.0 / Lightning 2.6.5）：

| setting | 中位耗时 | setting | 中位耗时 |
|---|---:|---|---:|
| ETTh1-96/192 | 85 / 74 s | Weather-96/192 | **1100 / 683 s** |
| ETTh2-96/192/720 | 47 / 37 / 57 s | Electricity-336 | **1717 s** |
| ETTm1-96/192 | 253 / 244 s | Traffic-96 / 192 / 336 / 720 | **4096 / 2972 / 2585 / 2640 s（实测，见下）** |
| ETTm2-96/192 | 185 / 114 s | | |

按上表外推（H336/720 按同数据集趋势插值/外推），不含 Traffic 的部分约 61 GPU·h。
**注意**：该外推基于旧口径（"465 runs"），最终矩阵经 manifest 核验为 **492 格 = 411 新训 + 81 复用**，
其中 Traffic 占 **60 个新训 run**（故不含 Traffic 为 432 格 = 351 新训 + 81 复用）；
矩阵总量与分工以 §4.2 与 `check_section42_coverage.py` 的核验结果为准。
Traffic 曾是唯一未知量（862 通道、batch 8）。**现已实测**：其 60 个 run 的 `elapsed_sec` 中位数为
4096 / 2972 / 2585 / 2640 s（h96/192/336/720），且 Traffic-96 与冒烟预测（138 s/epoch × 30 epoch
= 4140 s）吻合。一般地，**实测耗时比按 horizon 单调外推的 `COST_HINT` 低 30–40%**，
原因是早停主导（Traffic-96 反而最慢，因其跑满 26–30 epoch）——详见 `docs/PhaseFormer_L/e14_main/04_run.md` §8。

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
| 2026-09-20 | 排期修正 | — | 用已完成 81 格的 `elapsed_sec` 实测校准 `COST_HINT`：整体**偏高 30–40%**（Electricity-720 比值 0.64、Traffic-96 除外为 0.98）→ 剩余 66.53 → 约 **46 GPU·h ≈ 5.8 h**，阶段 A 预计约 08:50 结束（原估 10:30）。**原因不是估算粗糙而是模型形式错误**：耗时由**早停**主导而非 horizon——Traffic-96 跑满 26–30 epoch（最慢，与冒烟 138 s/epoch×30=4140 s 完全吻合），而 Traffic-720 多在 12–26 epoch 就停。另记工程陷阱：**已完成 run 的 `status.json` 被覆写并丢掉 `started_at`**，墙钟耗时只能取 `metrics.csv` 的 `elapsed_sec`（这是测量方法问题，不是产物缺失） |
| 2026-09-20 | 阶段二契约 | **挡下并修复** | 逐条核校阶段二跨阶段文件名，挡下一处会让**第 6 步整体失败**的缺陷：`e14_read_test.py` 是**就地**填 `results.csv`，而 `read_test_generic.py`（E17/E18 用）另写 `<results>.with_test.csv`；流水线第 6 步却把 `$E14_ROOT/results.with_test.csv` 交给 `e18_writeback.py`——该文件**永不生成**。后果出现在第 1–5 步已耗时数小时之后。根因是同一路径在第 2/3/6 步各写一遍、可互相漂移，故改为**单一声明** `E14_TEST_CSV`；`e18_writeback.py` 的 Usage 示例同错已修。并在**最便宜的第 1 步**加快速失败闸门（读 test 是纯推理、几分钟；第 4–6 步要跑 63+24+78 个 run），四类 fixture 本地与**服务器 gawk** 各测一遍（转义错误在静态检查中不可见，故必须实跑） |
| 2026-09-20 | 阶段二边界 | — | 对**每一处**跨阶段引用做生产者↔消费者核对（13 行，全部 ✓）。两点需记录的结论：①`projector_audit.json` **没有生产者被编入流水线**是**有意**的——`e17_conditional_projectors.py` 属冻结阶段，7 setting 的基向量与审计是 §4.5 要冻结的对象，训练期不得重算；已在服务器核实该文件 33572 B 存在且路径与第 5 步实参逐字一致。代价是该依赖为**隐式**（若目录被删会在训练之后才失败），已记备查。②第 5 步 `--stage a` 后再显式 `--stage assemble` 属**冗余但无害**（`a` 的定义本就含 assemble，且顺序在 test 读取之前，不会覆盖带 test 的产物）。诚实边界：本项只核"名字对得上"，不能证明内容语义（join 键、单位）正确，后者仍由各实验阶段 5 审校负责 |
| 2026-09-20 | 阶段二实参 | — | 对 19 个调用逐个抽取 `--flag` 与目标脚本 `add_argument` 求差：**未定义参数 0/12 脚本**；并逐脚本核对 `required=True` 参数均已提供。**未覆盖项**：`--verify` 一类"参数存在但语义随脚本而异"的问题（如 `e18_negative.py --verify` 实为阶段 5 完整性审计），只能靠逐条语义核对，已以脚本内注释固化 |
| 2026-09-20 | 阶段二接力 | — | 新增 `watch_e14_then_phase2.sh`：轮询等 E14 干净结束（`E14_MAIN_EXIT=0` **且** 411 个 run 目录，两个条件独立校验）后自动启动阶段二，日志 `phase2.log`、状态 `phase2_watch.status`。已用沙箱副本实测三种门：flag=no+411 → 不启动（exit 3）；flag=yes+410 → 不启动；flag=yes+411 → 启动并记 `PHASE2_OK`。**核实过标记确实会被写出**：E14 由 `~/niuyiming/run_e14_main.sh` 驱动，其末尾 `echo "E14_MAIN_EXIT=$?"`（该标记在仓库内只有读取方、没有写入方，若不核实会出现"守护永等"）。03:11 以 nohup 启动（pid 28152），此刻 `phase2.log` 不存在 = 未提前启动 ✓ |
| 2026-09-20 | 列名契约 | — | 新增 `check_column_contracts.py`：把 §14/E16 那类"消费者点名的列生产者没写过"从一个手工发现的 bug 升级为**一类 bug 的静态检测器**，并加为阶段二预检（`--strict`，亚秒级、在任何耗时步骤之前，不通过就拒绝花 GPU）。生产者列集**从其自身源码导出**（`RESULTS_FIELDS` 或真正构造行的函数里的字面量键），故不会与代码漂移。**三次精度修正**都是为了不喊狼来了：①按 AST load/store 区分读写（`mse`/`verdict` 等输出列曾被误报）；②f-string 写的内部枢轴键收为模式（`entry[f"{arm}_mse"]` → 8 个假阳性）；③生产者键也含**下标赋值**（`record["revin"] = {...}`）。修后 3 个契约 **0 缺口** |
| 2026-09-20 | 检测器验证 | — | 「在已修好的代码上跑出 0 缺口」**不能说明任何事**（永远返回 0 的脚本也能做到），故做了**双向正对照**：真树 0 缺口/exit 0；消费者突变（把列名改回历史错误名 `majority_input_group_label`）→ 报出该列；生产者突变（改名一列）→ 报出被点名的列；两者加 `--strict` 均 exit 1（这才是能挡住阶段二的路径）。**诚实边界**：只核名字，不能证明两列语义/单位一致或 join 键对得上，也不能发现"列存在但恒为空"——后者归 `check_builder_outputs.py` 与阶段 5 审校，故本项是防线中的一道，不替代审校 |
| 2026-09-20 | E17 回填预演 | **通过** | 新增 `rehearse_e17_writeback.py`：用**真实冻结的** `projector_audit.json`（33572 B，不造假）+ 按 `e17_conditional.RESULTS_FIELDS` 精确 schema 合成的结果行，预演 §4.5 回填。**为何必须预演**：回填按 `(dataset, horizon)` 做 join，而 results 的 horizon 从 CSV 读出是**字符串**、审计里是**JSON 数字**；一旦不一致，`cosines.get(...)` 对每个 setting 都返回 `None`，cos 列静默变空——**不抛异常**，而"只有 Electricity-336 能区分两臂"这一论断的全部证据就在该列。断言结果：7/7 setting 的 cos 与审计**逐格相等**、0 失配；`Electricity-336 = 0.006588`、其余 ≥0.999088；被标记可区分的 setting **恰好是 `['Electricity-336']`**，与文档一致。即——**§4.5 的检验力集中结论在回填这一侧被独立复现**（不只是投影阶段测过一次） |
| 2026-09-20 | E17 预演教训 | — | 预演**第一次运行报 7/7 MISMATCH**，看似回填工具有严重缺陷；逐格核对发现**数值本身完全相等**，失配来自我自己的断言：回填按设计把余弦四舍五入到 6 位，而我的断言拿**未舍入**的全精度审计值去比（`1e-9` 容差），于是把**正确的**回填判成错误。已改为与舍入后的审计值比较并写入注释。记录此事是因为：**验收断言写错会制造外观与真实缺陷相同的告警**——当时若直接相信"7/7 MISMATCH"就会去修一个没坏的工件；若反过来放宽成永远通过则失去检测力。纪律：**验收断言必须先对着已知正确的输入校准**（与 `check_column_contracts.py` 的双向突变正对照同一套做法）。预演文档：`docs/PhaseFormer_L/e17_conditional/05b_writeback_rehearsal.md` |
| 2026-09-20 | 列名契约扩到 §4.2 | — | 检测器新增第 4 组契约 **`e14_writeback.py`（§4.2 主表，中心产物）**，生产者 = E14 结果表(18 列) + `e14_params.py` 的 `parameter_table.csv` + E19 阶段 1 的 `level_statistics.csv`（§4.7 的 τ̂ 列）。四组契约**全部 0 缺口**。改规则后**重跑全套正对照**：真树 exit 0；e16 历史错误列名→报出；生产者改名一列→报出；**新契约** `gate_value`→`gate_value_TYPO`→报出 |
| 2026-09-20 | 一次差点误报的排查 | — | 放开规则后检查器报 `e16_writeback.py` 缺 4 列 `random_band_low/high_fused_mse`、`random_rrr_band_low/high_fused_mse`，且是 `row[...]` 下标（缺列会**抛 KeyError**）——外观很像严重缺陷。逐处核对后确认是**误报**：生产者的 band 名取自 `bases["bands"]` 的键（`random`/`random_ambient_matched`/`random_rrr`），故只产出 `random_low_fused_mse`；`random_band_low_fused_mse` 是回填工具**自己算的输出列**（`mean_of(band,"random_low_fused_mse")`），它随后遍历自建的表再读回来格式化 markdown。**两侧改名是有意设计，不是缺陷** |
| 2026-09-20 | 盲区实测与表述校正 | — | 如实记录并按实测校正一处过度承诺：pass-through 列（读自输入、同名写进输出，如 `tau_hat_steps`）会被读写集相减减掉，故生产者漏写时**本检查器不报告**——已把 `NU_STAT` 改成 `tau_hat_steps_TYPO` **实测确认未报出**（同轮对照 `gate_value_TYPO` 正常报出）。兜底说的是"`check_builder_outputs.py` 空列扫描会发现"——**但该脚本 `main()` 无 `sys.exit`、永远 exit 0，且流水线调用还带 `|| true`**，故真实情况是"**被记录、阶段 5 审校可读，但不是自动失败**"，已把措辞改准。**本轮主动不做**按函数局部数据流改造：该检查器将在 E14 收尾时**无人值守自动**作为阶段二预检运行，引入误报会把整条链挡在门外，故优先保持已验证版本（真树 exit 0 + 双向突变可捕获），把盲区写明而非用未校验的复杂分析换覆盖率 |
| 2026-09-20 | 工具侧模型修正 | — | 把检查器与 `traceability_matrix.md` §5.2 的**手工**表对表，发现**工具错、表格对**：手工表称 `read_test_generic.py` 增加 **7** 列，而工具只把 `TEST_COLUMNS`(2 列) 并入生产者集合。实测该脚本确实盖 7 列（`TEST_COLUMNS` 2 + `BRANCH_COLUMNS` 2 + 3）。当前 4 组契约仍 0 缺口（无消费者读那 5 列），但**将来一旦有消费者读 `test_read_status` 等列就会误报**——对无人值守自动运行的门这是必须提前消掉的隐患。修法与历次一致：**从源码导出而非手写**，新增 `_subscript_store_keys()` 收集 `row["x"] = ...`；并过滤私有键与**全大写环境变量赋值**（`env["CUDA_VISIBLE_DEVICES"]` 是字典赋值但不是列；这些表用 snake_case，故全大写即环境变量） |
| 2026-09-20 | 六点验证套件 | — | 每次改检测逻辑后**必须**重跑对照，否则"已校准"失效。本轮六点全部符合预期：①真树 exit 0；②e16 消费者突变→报出；③e16 生产者突变→报出；④e14_writeback 消费者突变→报出；⑤e17 消费者突变（读读取器盖的列）→报出；⑥**盲区反向对照**（pass-through `tau_hat_steps` 改名）→**无命中**，与 §15.6.2 声明一致。第 6 项与第 1–5 项同等重要：一个**在应报警时报警、在应沉默时沉默**的工具才是可依赖的门 |
| 2026-09-20 | 矩阵与工具互校 | — | `traceability_matrix.md` 增 §5.5，写明手工表与工具的分工：手工表是**首次验证的历史证据**（含"正则抽不到 ≠ 没生产"的自伤记录），工具是**持续保证**；两者不一致时**先怀疑工具**（本轮即如此）。并记录手工表 §5.1 的局限：它只列了 `e14_writeback` 读 E14 结果表的列，未含其另读的 `parameter_table.csv` 与 `level_statistics.csv`，工具版已补全——自动化覆盖现为 **4 组契约** |
| 2026-09-20 | E18 回填预演 | **发现并修掉一处缺陷** | 新增 `rehearse_e18_writeback.py`，针对阶段二**最后一步**（其后无补救机会）做纯 CPU 预演，检验静态检查覆盖不到的维度——**退化输入**。三个 case 全 exit 0 且行结构恒为 `['1','2','3','4','5']`。**首次运行暴露真实缺陷**：百分比算不出来时 `None` 被 f-string 渲染成字符串，输出 `平均 ΔMSE None%` ——**不崩溃、无异常提示，但会漏进 minipaper 表格文字**，成为审稿人可见的瑕疵。已收进 `pct_cell()` 把 `None` 渲染为 `—`（与 `KEPT_AS_DASH` 既有表示一致）；修后正常路径**数值不变**（仍 `5.2356%`），退化路径变 `—`；并重跑列名契约门确认仍 exit 0。文档：`docs/PhaseFormer_L/e18_negative/05b_writeback_rehearsal.md` |
| 2026-09-20 | E18 预演自伤 | — | 预演首次运行抛 `ValueError: dict contains fields not in fieldnames: test_mae, test_mse`。核对确认**不是产品缺陷而是我的 fixture 少建列**：E18 结果表记的是 `test_mse_recorded`，回填读的 `test_mse` 是 `read_test_generic.py` 合并时盖上去的列。修法不是手写补两列，而是**直接复用** `check_column_contracts._subscript_store_keys()` 源码抽取器，让 fixture 与检查器**共用同一份真相**。与 §5.2 记录的"我抽取不到 ≠ 它没生产"同源；记下它的理由是：**预演脚本自身出错时，报错外观与真实缺陷无法区分**，故每次报警都要先分清是工件坏了还是断言/fixture 写错 |
| 2026-09-20 | 阶段二元数缺陷 | **挡下并修复** | 在 E14 未跑完时用真实输入逐条预跑各步的门，第 6 步 SVD 阶段**直接报 argparse 错误**：`e18_svd_truncation.py: error: unrecognized arguments: 2022 2023 (exit 2)`。原因是 `--seeds` 非 `nargs` 参数、由 `parse_list` **按逗号切分**，而流水线写成空格分隔。**严重性在时点**：该行在第 6 步，前面已跑完 411+165 个 run，会在**全部昂贵步骤之后**才炸，且纯粹是参数写法错误。已改为 `--seeds 2021,2022,2023`；实测修前 exit 2、修后通过 argparse 并在 E14 未完成时**正确拒绝**（`135 unresolved cells`） |
| 2026-09-20 | 为何上轮漏掉 | — | §14.5 那轮已核对"19 个调用、未定义参数 0 个、必填参数齐全"，仍漏掉此条，原因明确：**"flag 存在"与"flag 能接收几个值"是两件事**。当时校核的是*存在性*与*required 是否给出*，**没校核元数**——这是方法论缺口而非一次疏忽，故修法不是改那一行，而是**把缺的那一维补进检查**。新增 `check_pipeline_invocations.py` 并加为阶段二预检（与列名契约门并列）：核对每个 flag 已声明 + 后跟 ≥2 个裸值的 flag 必须声明了 `nargs`。正对照：真流水线 66 个 flag → OK exit 0；还原修前写法 → 精确报出并 exit 1。同规则扫描整条流水线，确认无第二处同类问题 |
| 2026-09-20 | 阶段二门禁全量预跑 | **六步全部用真实输入跑过** | 把"启动前完成静态检查与协议校对"做到**不只跑静态检查，而是把每一步的门用真实输入实际跑一遍**（输出全部指向 `/tmp`，仅第 1 步因需定位 run 目录用真实根但以 `--dry-run` 运行并核实不写盘）。结果：①`e14_read_test --dry-run` **exit 1 正确拒绝**且 `wrote_outputs: False`；③a `e14_params --dry-run` **exit 0**，在 **175 个真实 checkpoint** 上 `total_mismatches: []`（checkpoint 形状读出的参数量与 `metrics.csv` 全一致）；③b `e14_reuse_audit` **exit 0** 且与既有审计逐项一致；④E16 门 **exit 0**（63 cell 全为复用）；⑤E17 plan **exit 0**（84 cell/24 新训）；⑥a E18 verify **exit 1 正确**（`--verify` 是跑完后的完整性审计）；⑥b SVD 门 **先 exit 2 → 修复后 exit 1 正确拒绝**。文档：`docs/PhaseFormer_L/audit/phase2_gate_sweep.md` |
| 2026-09-20 | 第 1 步滚动对账 | — | 第 1 步 dry-run 的恒等式**精确成立**：`by_status {planned:94, reused:81, missing_metrics:8, missing_run:309}` = 492；`accepted 175 + problems 317` = 492；`by_arm` 六项 = 492。即**每个 cell 恰好落入一个桶**，解析器无漏计/重计。另查实 **`missing_metrics: 8` 是"在飞"而非"卡死"**：8 个目录全为 `electricity_h192`、mtime 落在查询前几分钟内（`metrics.csv` 完成时才写）；这同时证明解析器**不会把在飞的 run 提前当可用**，正是"test 读取必须在训练全部结束后"这条协议要的行为。`fingerprint_check` 亦显示 `e14_read_test.py` 与 `e14_main_matrix.py` 常量相等、20 个 parity case **零失败** |
| 2026-09-20 | E14 回填预演 | **发现并修掉两处真实缺陷** | 新增 `rehearse_e14_writeback.py`（用**真实** manifest/Golden/电平统计，只合成结果行与参数量行），针对**论文中心产物** §4.2 主表。case A 全填满 → exit 0、主表 28 行（core 24 + appendix 4，按 `is_traffic_appendix` 标记分界）、variant 6 臂汇总、`provenance_note` 无空值；case C 无参数量表 → 正常退化。**两处"局部缺失→整表全丢"缺陷**：①`round(pct_change(...), 4)` 未防 None → `TypeError`（同文件的姊妹比较**已有**该保护，属不一致而非风格）；②`variant_rows[0].keys()` 未防空 → `IndexError`（相邻主表写入**已有** `if rows` 保护）。二者都只在结果表缺指标时触发——而那只差"一个 setting 的指标缺失"就成立，代价是**整张主表丢失**而非一格 `—`。修后 case B 亦 exit 0；**正常路径未变**（28/28 行仍算出 FITS 差值），列名契约门仍 exit 0。文档：`docs/PhaseFormer_L/e14_main/05c_writeback_rehearsal.md` |
| 2026-09-20 | E14 预演自伤 | — | 首轮另有 **3 个报警是我自己写错**，逐项核对后确认与产品无关：①误以为"24+4"是 `main_table` 24 行 + `variant_table` 4 行（实际是主表 28 行带 `is_traffic_appendix` 标记 + variant 为 6 臂汇总）；②在 `claims.json` 里找 `claim_A_either_metric` 等（实际在 **stdout** 的 `finished` 行，`claims.json` 顶层是 `A/B/C/D/must_answer_a/must_answer_b`）；③我把 Golden 文件名写成 `PhaseFormer_golden_standard.md`，规范名一直是 **`docs/PhaseFormer_gold_standard.md`**。**同一教训第三次出现**（E17 舍入断言、E18 少建列、本次文件名）：预演脚本自身出错时报错外观与真实缺陷无法区分。本轮 5 个报警中 **2 真 3 假**——不逐项核对，既可能去修没坏的东西，也可能因"工具跑通了"放过真缺陷 |
| 2026-09-20 | 阶段 5 自动化入口 | — | 新增 `audit_phase2_outputs.py`，把**文档里已写下的**验收判据（E16 §6.2 判据表、E18 `02_static_check` 等）编码成检查，并加为阶段二**第 7 步**（链尾），使审计报告在链结束时已生成。**三态设计**使其可随时运行：`PENDING`（产物未生成，非致命）/`FAIL`（产物存在但与判据矛盾，exit 1）/`INFO`（文档未固定期望值）。当前在 E14 仍在训练时实测 `PASS=1, PENDING=21`、exit 0（唯一 PASS 来自真实存在的冻结投影器审计 7 setting） |
| 2026-09-20 | 审计器检测力校准 | — | 一个只会报"没问题"的审计脚本等于没有，故为脚本加 `--root` 以指向**合成的**产物树（否则它只能对真实产物说 fine），并对一棵**故意做错**的树实测：植入 **12 个缺陷全部报出**（492→1 行、空 `test_mae`、主表 27 行、空 provenance、variant 5 臂、claims 缺 4 项、`algebra_failures=1`、`reference_parity_passed=False`、干预行 10、干预臂 10、**缺随机 RRR 臂**、负对照表仅 1–3 行），脚本 exit 1。与列名契约门、实参元数门同一纪律：**先校准再依赖** |
| 2026-09-20 | E17 训练路径冒烟 | **通过** | 核对"哪条路径**只做过静态检查**"，发现 E17 的**冻结子空间训练**从未真正执行过（其文档：必须走 `run_top2_direction_retention.py` 的 `--basis` + `weak_residual_projection=frozen_subspace`，缺该 override 时 `PhaseFormer.__init__` 抛错），而它的 24 个 run 排在**步骤 5**（前面已有 411+63 runs）。据 plan 的真实命令取最便宜一格（ETTm2-96-s2021）、只改 output-dir 与 max-epochs=1、**按 runner 方式启动**，实测：`{"event":"projection_installed","rank":1,"basis":".../ETTm2_96_Q1COND.npy","sha256":"f45bddb0…"}`、`epochs_completed: 1`、`val_mse 0.14325`、**`test_mse/test_mae 均为空`**（训练期不读 test ✓）、checkpoint 已写 |
| 2026-09-20 | 单次 test 读取 | **挡下严重缺陷并修复** | 用上述真实 1-epoch checkpoint 冒烟 `read_test_generic.py`（**步骤 5/6 的 test 读取**），worker 明明**成功**（`status: read`、`test_mse 0.19280`、`val_relative_difference 1.525e-05`）而父进程却全记 `worker_failed`、test 列留空。根因：父进程判定成功的条件是 `marker.is_file() and code == 0`（`:325`），而 **worker 分支只 `print` 记录、全文件从未写任何 `<marker_dir>/<key>.json`**（`:350-351`），尽管模块 docstring 第 28 行**承诺**"each freshly read cell leaves a per-cell JSON marker"——**承诺存在、实现缺失**。若不发现：24+78 个 run 训练完后 test 列全空、§4.5/§4.6 表没有数字，而脚本 **exit 0** 流水线继续。对照证据：`e14_read_test.py`（步骤 1）的 worker **确实写了** per-cell artifact（`:1600`/`:1210`），父进程用同一键去读（`:1331`）——两个读取器一个有写一个没有。已按文档既定设计补齐（新增 `--marker-file`，worker 写出、dispatcher 传路径） |
| 2026-09-20 | test 读取修后复测 | — | **两个方向都验**：正向（basis 正确）→ `status='read'`、`test_mse=0.19280`、`test_mae=0.28717`、`val_relative_difference=1.525e-05`（远低于容差）；**负向**（不传 basis）→ `status='build_failed'`、test 列保持空，即**test 分裂没有被读**。负向对照是刻意加的：只做正向无法证明"校验不过就不读 test"这条协议保护真的生效 |
| 2026-09-20 | 两条已记录未改动观察 | — | ①**`read_test_generic.py` 永远 exit 0**：即使有 cell 被拒或 worker 崩溃，只在汇总里列 `rejected`，进程仍返回 0，即"部分失败"不会让流水线停下；**补偿机制**是第 7 步审计（E17/E18 各有"每个 cell 都要有 test 指标"的判据）。若要改成硬失败应改在**流水线**（策略属流水线契约）而非藏在工具里，但收益有限（步骤 5/6 之后紧接第 7 步），故本轮只记录。②**被读取的 run 必须位于仓库根之下**（`:207` 的 `checkpoint_path.relative_to(ROOT)`），`/tmp` 下的 run 会让 worker 抛 `ValueError`；真实 run 全在 `research_runs/` 下故不影响流水线，但**冒烟必须把 run 放进仓库内**（本轮据此把冒烟根改为 `<REPO>\_smoke_e17`，跑完即删） |
| 2026-09-20 | 冒烟搭台自伤 | — | 冒烟脚本自身出错 **4 次**，每次报错外观都与真实缺陷相似：①用 shell 驱动命令 → `--overrides` 的 JSON 被重新分词，得到假的 argparse 错误；②`find` 取 run 目录多退一级（`p.parent.parent`）→ `config.json` 找不到；③断言比较了**未写进输出 CSV** 的列（`recomputed_val_mse` 只在 marker/记录里，落盘的是 `val_relative_difference`）；④`run_case` 成功返回 `0`，我却写 `if not ok: return 1`——**判反 sentinel**，正向 case 一通过就退出。**本会话同类自伤累计 6 次**。净结果：**1 个真缺陷 + 4 个我自己的错**——不逐项核对，最容易把假报警当成真问题去修没坏的东西。文档：`docs/PhaseFormer_L/audit/single_test_read_smoke.md` |
| 2026-09-20 | E14 读取路径冒烟 | **通过** | 按 §1 的"只做过静态检查"核对法继续排查，发现 **E14 阶段 B 的读路径**同样只有 `--dry-run`（只走解析器）被反复执行过，而"建模型→复现校验→读 test→落产物"从未跑过；其文档 `04b_test_read_plan.md` 通篇是**计划**、无执行证据，而它是**阶段二第一步**。故补做隔离式冒烟：把一个真实完成的 run 复制到仓库内临时根（因 `--output-root` 既是定位处也是写盘处，直接对真实根跑会提前生成阶段 B 产物），用 `--cells-file` 限定单格。实测：`accepted: 1, problems: 0, failed_workers: [], by_status {"read": 1}`，`test_mse=0.1474874`、`test_mae=0.2392384`、`gate=0.3572`、`nlinear_mse=0.6245`；启动指纹校验 `parity_cases 20 / failures []`。**并实测确认真实根未被写入**（无 `results.csv`、无 `test_read/`），跑完即删 |
| 2026-09-20 | E14 冒烟附带事实 | — | 确立三条操作事实：①`--cells-file` 行格式是 **`arm:dataset:horizon:seed`**（非日志 token `arm__Dataset-H-sSEED`）；②解析器**只搜 `<output-root>/runs`**；③**每个 `missing_run` cell 约耗一个 `--poll-seconds`（15 s）周期**，故不限定 cell 的冒烟会枚举 492 格而**跑不完**（≈2 h）——首次尝试因此超时并被手动终止（已确认真实根未被写入、终止干净）。同时说明**阶段 1 正式运行不会遇到该问题**：那时 411 个新格全部已有 run、81 个复用格走登记证据，`missing_run` 应为 0（dry-run 对账亦印证）。附带确认 **`a1` 臂确实在被真实训练**，门值 0.357 与 `l_*` 臂的 preset 默认不同 |
| 2026-09-20 | 冒烟搭台自伤（续） | — | 本轮又 3 项自伤：①按**目录名**匹配 run（目录名只编码 mechanism、不含臂名，`l_main`/`l_q1_4`/`l_q1_8` 都叫 `weak_residual`，匹配必失败）；②用日志 token 当 `--cells-file` 行；③未限定 cell 导致 492 格枚举超时。**本会话同类自伤累计 9 次**。本轮净结果：**0 个新缺陷**（E14 读路径本身正确），全部报警皆出在搭台 |
| 2026-09-20 | E19 阶段 2 冒烟 | **通过** | 继续排查"只做过静态检查"的路径：**E19 阶段 2（§4.7 ρ 列，步骤 2）从未执行过**（它要 E14 带 test 的 `results.csv`，而那要等第 1 步），且它紧跟第 1 步、一崩就把无人值守的链停在那里。用**真实** `level_statistics.csv`（28 setting）+ 列名取自 `e14_read_test.RESULTS_FIELDS` 的合成结果表冒烟：case A 完整 → exit 0、`settings_with_data: 28`、28 行；**case B 半退化**（一半 setting 只给 1 seed）→ exit 0、`settings_with_data: 14 / without: 14`、14 行——**完整的做相关、不足 3 seed 的被跳过而不是崩溃**。输出列 `tau_hat_steps`/`diagnostic_s`/`corrector_helps`/`s_prediction_hit`/`delta_mse_pct` 均在。边界：退化输入下汇总会出现裸 `NaN`（严格 JSON 不允许），正式运行非退化故不触发，**本轮只记录不改** |
| 2026-09-20 | E18 行 3 评估路径冒烟 | **通过** | `e18_svd_truncation.py` 此前只被跑到 `--verify`（解析 + 在 E14 未完成时正确拒绝），而"**真的截断正确器并在 val 上评估**"这条路径没跑过——它正是 §4.6 行 3 的数字来源。用**已有复用 checkpoint**（ETTh2-96 属 test-selected，其 3 臂来自复用链故现在即可解析）限到一格、秩 10、**子集 300 样本**、**CPU**（不与 E14 争 GPU）：`verify_ok` → `evaluated` → `finished`，exit 0，四个产物齐全，表里 `truncated_mse`/`trained_lowrank_mse`/`full_rank_mse`/两个 gap 全部算出且 `records_test: False`。**明确写明**：因用 `--max-eval-samples 300`（子集评估），这些数字**不构成行 3 的任何结论**，只证明该路径跑通 |
| 2026-09-20 | E16 回填联调 | **通过（用真实产物）** | 前三份回填预演（E14/E17/E18）用的是**按生产者 schema 合成**的输入，因为那些输入在跑完前不存在。E16 有更好的选择：**其生产者自身可冒烟**（ETTh2-96 × 2021 × l_main,l_q1_8，--max-batches 2，约 9 min，纯 CPU），于是改为**先生成真实产物、再直接喂给回填工具，一次 fixture 都不写**——这很关键，因为 E16 的行长字典由累加器**动态构造**（含 `**band_summary` 的 f-string 展开），手写 fixture 极易漂移（本会话已因此产生多次假报警）。冒烟复现与事先写下的预期**逐项一致**（`cells=2`、`algebra_failures=0`、稠密头 rel_gap 2.15e-10、低秩 2.97e-05；`run_metric_not_comparable=2` 与 `reference_parity_passed=false` 正是 `--max-batches 2` 子集效应的预期）。回填 **exit 0**，四个产物齐全，且 **`missing_columns_intervention: []`、`missing_columns_dissection: []`** ——把列名契约从"静态推导"升级为"**在真实数据上实测成立**"。`arms_per_cell_observed: [11, 12]`、`cells_with_fewer_arms: 0`。文档：`docs/PhaseFormer_L/e16_dissection/05b_writeback_rehearsal.md` |
| 2026-09-20 | §4.4 表骨架确认 | — | 回填产出的 `intervention_table_44.md` 实样与我们需要的 11 格列序一致（model/dataset/H/q·r/Semantic-only/Semantic-drop/**random band**/PCA-drop/**random-RRR band**/branch delta/fused delta），且第 9 格**确实是区间**（`[0.1618, 0.1632]`）——即 §4.4 要求的**随机 RRR 子空间对照**贡献了真实的 95% 零分布带，而非空值或占位符。边界（已写明）：仅 2 cell 故不验证 21 行结构与 63×11 完整性；`--max-batches 2` 使"不变量 2"处于 skipped；数值来自子集评估，**不构成任何 §4.4 科学结论** |
| 2026-09-20 | 论文↔实现一致性 | **找到并修正一处过时表述** | 新做一项此前没有的核对：把 minipaper §4 里**写下的**每个数字/结构与代码常量逐项对照（AST 抽取，不靠记忆）。理由：论文是**交付物本身**，若它声明的门槛与实现里跑的常量不一致，四个主张的判定会**在无人察觉下失去意义**（表照样填满、结论照样打印），而这类漂移不会被任何单元测试发现（代码自洽、论文自洽）。结果：§4.0 四个门槛全部一致（1.0% ↔ `REGRESSION_BOUND_PCT`、3/4 ↔ `CLAIM_B_FRACTION 0.75`、0.5% ↔ `CLAIM_D_BOUND_PCT`、C 无门槛）、ν* = 57.35 在**两处独立声明**且相同；§4.2 24+4、§4.3 28 行、§4.4 21 行、§4.5 四臂/7 setting、§4.6 42+36=78 run 与 5 行、§4.7 三个统计量均一致 |
| 2026-09-20 | §4.4 臂数修正 | — | 唯一不一致处：§4.4 括号写"每 cell **10 臂**"，但**同一段表头已列出"随机 RRR 子空间 drop（新增对照）"**、正文也写明"新增该对照用于区分…"——即论文自己要求了这个对照，括号里的臂数却没跟着更新。**三处独立证据**：①`build_arm_plan` = 8 个基础臂 + 2 个同维 `PCA-matched` = **10 登记臂**，再按可用基向量追加 `Independent-RRR-only`/`Conditional-RRR-only`/**`RandomRRR-drop`**；②真实产物实测 `arms_per_cell_observed: [11, 12]`、`cells_with_fewer_arms: 0`（11 是**下界**、12 也合法，臂数**按 cell 动态**）；③回填判据 `expected_arms = 11`。已改为"10 个登记臂（列出 8+2）…追加后实际 11–12 臂（11 为下界）"。**只改这一处**：文中另一处"10 臂"描述的是既有 rsync 副本（72 cell × 10 臂）的历史数据，那里是对的，故不做批量替换。文档：`docs/PhaseFormer_L/audit/paper_code_consistency.md` |
| 2026-09-20 | 全文过时设计扫查 | — | 用关键词全文扫查被后续决策（D-4/D-5/D-6 与两处勘误）推翻的早期写法：`boxcar`×2、`always-on`×1、`12 格既有`×1、`s=0`×2、`5 个已知符号`×1 —— **全部判定为合法**，因为每处都是**自证性的修订/勘误注记**（明确写"早期怎么写、为何改"），而非残留。即**未发现残留的过时设计**，论文修订史连贯。并确认全文**只剩 2 个显式待填标记**且都准确（摘要 `*[主结果待填。]*`、§4 开头填充状态注） |
| 2026-09-20 | §4.3 预注册残留修正 | — | §4.3 是**唯一已完成**的小节，已按实测回填（`λ_1/Σλ` 0.642–0.862、`a_1│cos│` 0.890–0.989、`PR` 1.33–2.29），但段末仍留预注册写法"**预期图**：Scree 图（第一根柱 **0.66–0.86** 量级）…"。两层问题：①"预期图"出现在**结果**小节里像未完成草稿；②括号里是先导区间，而同一小节上文已报告实测的 0.642–0.862，**同页两个区间并列易被误读为矛盾**。已改为报告实际图件与实测锚点、文件名逐一给出（`scree_lambda_spectrum.png` / `b1_lag_profile.png` / `a1_horizon_profile.png`），**三个数字全部取自该节上文已有条目、未引入新数字**，并括注说明"原为预注册写法、E15 完成后改报告实测"。文件名独立核实：`e15_dimension.py:803/850/857` 确实写出这三个名字，且 E15 阶段 4/5 文档与"成图性验证"（240–344 KB）一致 |
| 2026-09-20 | 回填校验工具 | — | 六阶段最后一环"**把数填进论文**"此前纯手工，且**没有任何工具会写论文**（唯二提到 minipaper 路径的 `e15_dimension.py`/`e16_dissection.py` 只是 docstring 引用）。在五张表、几十个数值单元上手工转录正是"论文与自己的产物不一致"的高发处，而这种错误能躲过所有自动化检查——**两边各自都是对的**。新增 `verify_minipaper_fill.py`，判据写死为"**论文显示值 == 产物值按该显示精度四舍五入**"（比原始浮点会把正确的舍入报成错误——E17 预演正是这样自伤过一次），状态三态化（`MISMATCH` 致命 / `blank` 不致命 / `PENDING`），故可随时运行 |
| 2026-09-20 | §4.3 回填校验 | **168 单元全一致** | §4.3 是唯一已回填的小节，故同时是理想校准样本。实测 **`match: 168`、exit 0**（= 28 行 × 5 个数值列 = 140，加 28 个 `b_1` 复合格 = 168）。**顺带的独立收获**：这回头验证了**此前手工回填本身没有转录错误**——那次回填此前只有"产物正确"（E15 阶段 5）与"表格已填"两类证据，缺的正是"填进去的数与产物一致"这一环。**四项检测力对照**：真实论文 168 match/exit 0；扰动一个值（0.712→0.799）→ 精确报出 `paper=0.799 artifact=0.711673 rounded=0.712 dp=3` 且 exit 1；清空一格 → 记 `blank` **不**致命；删掉一行 → `paper has 27 rows, artifact has 28` 且 exit 1 |
| 2026-09-20 | 校验覆盖范围声明 | — | **目前只覆盖 §4.3**：其余小节表仍为空、没有可校准样本，故**故意不为它们预先写比较逻辑**——在没有真实填充行可校准的情况下先写比较器，正是本会话反复踩的坑（累计 9 次"搭台错误被误当真缺陷"）。纪律：**每填完一节就为该节补比较器并当轮校准**（真实行 + 扰动对照），而非一次性写好五个未校验的解析器。**因此本工具现在的 pass 只意味着"§4.3 的填充与产物一致"**，不代表其余小节已核对 |
| 2026-09-20 | 计划↔论文覆盖核对 | **通过（恰好覆盖）** | 换一个方向核对：**实验计划是否恰好产出论文表格要求的每个格子**（不多不少）——多了浪费算力与事后解释，少了就是"论文有行、实验无格"，正是本任务最初要防的事。数据源是**真实 manifest**（492 cells）。§4.2 经核对是**两张表**：主表 **28 行 = 24 + Traffic 4**，其后另有**臂级变体小表 4 行**（故整段 grep 得 33 行 = 28 + 1 表头 + 4，早先易误读为行数异常，已recorded）。实测：28 setting = 24 main + 4 Traffic；五主臂各 28 setting × 3 seed = 84、`a1` = 24 × 3 = **72**；**总计 492 = 期望 492**、exit 0。**附带交叉验证**：复用计数与各自声明范围一致（`l_main`/`l_q1_4`/`l_q1_8` = 21 = 7×3；`phase_only` = **18 = 6×3**，独缺 Electricity-336；`l_rcrf`/`a1` = 0），且 `a1` 确不含 Traffic（与 `ARM_DATASET_EXCLUSIONS` 一致）。**正则对照**：从副本删掉 `a1` → 报 `total cell count differs` + `arms absent: ['a1']`、exit 1。已做成可复用脚本 `scripts/phaseformer_L/check_section42_coverage.py`——因为 manifest **已被替换过一次**（复用污染事件），既然它会重建，该断言就应可随时复算 |

---

| 2026-09-20 | 阶段二工期投影 | — | 用**实测**（而非外推）给出 E14 之后各步的工期：第 1 步约 1.3 h（实测单格全 test 读取 1–2 min × 411/8 卡）；第 4 步 E16 **3–5 h**（实测稠密格 **551.9 s**、低秩格 3.3 s → 21 个稠密格 ≈ 3.2 h，上界因全划分更贵、下界因实测为 CPU 而流水线跑 GPU 0）——**远比先前估计的 ~2 h 长**；第 5 步约 1.2 h（E14 实测单 run 中位 ~20 min × 24/8）；第 6 步约 4.5 h（78 run/8 卡 + 行 3 的 28×3 次截断评估）。**合计约 12 h**，故阶段二约在 **23:00 前后**收尾、整条链约 28 h。新增 §9 |
| 2026-09-20 | 依赖核对与不采用的优化 | — | 核对发现**第 4、5、6 步彼此独立**（E16 只依赖 E14 checkpoint；E17 只依赖冻结投影器；E18 只依赖 E14 checkpoint 与自身配置），而流水线串行；E16 又是**单进程**（`--gpus` 只取第一个索引），其 3–5 h 里另 **7 张卡空闲**，理论上与 E17 重叠可省 2–4 h。**本轮不实施**：①需给无人值守链路加并发与"两个都成功"的失败语义，而当前链路的价值恰在**简单**（任一非零即停、逐步日志、`--from/--only` 可重入）；②E16 **无 resume/跳过**（已核实），单独先跑会被第 4 步**重复跑一遍**、净收益为零；③收益是墙钟 2–4 h，代价是改动**已校准的关卡**。若日后提速，正解是先给 E16 加按 cell 断点续跑，再让第 4 步复用产物——**先做对再提速** |
| 2026-09-20 | 排期数字回锚 | — | §4 里几处旧口径与已核验事实冲突（同一文档里两套数字＝两份真相）：W2 写"363 runs"、W3 写"3–6 h"、§4.3 写"465 runs"与 Traffic"待冒烟实测"。已全部回锚：**411 新训**（492 = 411 新 + 81 复用，经 manifest 核验）、E17 **24 新训**（84 cell）、W3 指向 §9 实测投影、Traffic 四个 horizon 的**实测** 4096/2972/2585/2640 s，并写明"实测比按 horizon 单调外推的 `COST_HINT` 低 30–40%、原因是早停主导（Traffic-96 反而最慢）"。旧口径保留一处并注明其为何过时，避免后人再按它推算 |

| 2026-09-20 | 满规模解析实测 | — | 查证一个真实疑问：单格冒烟时 `<output-root>/runs` 只有 1 个 run，正式运行时有 411 个；若解析是"每 cell 扫一遍全部 run"，成本可能随时钟平方增长、把第 1 步的 1.3 h 变成数小时。**实测**（对真实根、492 cell、121 个 run 跑 `--dry-run`，只解析不读 test）：**`elapsed 23.75 s`、`maxRSS 18224 KB`** —— 即 **492 格全解析仅 23.75 s（≈0.05 s/cell）**。故第 1 步耗时由真实读取主导、1.3 h 成立，**该风险被实测排除**。**附带纠正一处理解**：此前观察到的"每个 `missing_run` cell 约 15 s"是**非 dry-run** 下 dispatch 的 `--poll-seconds 15` 节拍、**不是**解析成本（dry-run 跳过 dispatch，故 492 格只用 23.75 s） |
| 2026-09-20 | 在飞 run 存活核验 | — | 对 8 个在飞 run 逐个检查"最近文件 mtime + 最近记录的 epoch"：8 个的 mtime 均为**当时那一秒**、epoch 在 3–26 间正常推进，**无卡死**。这类"看起来在跑但实际挂住"的失败不产生日志错误，只会让链永远等下去（只有 watcher 的 24 h 上限兜住），故在长周期等待中值得定期抽查。已写成 §9.1.1(b) |

| 2026-09-20 | 回填校验扩到 §4.2 | — | 发现 `e14_writeback` 的 markdown 行与论文 §4.2 主表**列序逐列一致**（10 列，格式串 `e14_writeback.py:678-681`），因此 §4.2 用**字符串比较**而非数值比较——**更强**：不仅验数字，还验**列映射**与那个中文长文本的**来源/披露列**（手工转录最易错、数值比较覆盖不到）。**五类对照校准**（巧处：§4.2 的 Golden 列本就预填，可从论文自身行构造"正确匹配对"）：①正确配对 → **`match: 280`**（28 行 × 10 列）、exit 0；②改论文一格 → MISMATCH/exit 1；③改产物一格 → MISMATCH/exit 1；④删论文一行 → 行数不符/exit 1；⑤留空一格 → 记 `blank`、**不**致命/exit 0。即两方向都能发现不一致，且对"未填"与"填错"给出不同处置 |
| 2026-09-20 | §4.2 行数交叉验证 | — | 校准时输出 `section 4.2 main-table rows found: 28`——**论文侧**的 §4.2 主表恰为 **28 行**。这与第 34 轮从**计划侧**（真实 manifest）得到的 28 = 24 main + 4 Traffic **相互独立且一致**：一边是"论文要求的行数"、一边是"计划覆盖的格子数"，两边都落在 28。**覆盖范围更新**：校验器现覆盖 §4.3（168 格，已填）与 §4.2（280 格，待填；校验器已校准）；§4.4–§4.7 仍未覆盖，纪律不变——先看其表头与产物行列序是否逐列一致，**填完当轮校准** |

| 2026-09-20 | 回填映射（动手前定清） | — | 在**回填之前**把每张表的"论文列 ↔ 产物列"关系定下来——因为"以为每个工具都会吐出 markdown"正是回填阶段最易犯的错。逐一读**论文表头**与**生产者格式串**（均取源码）。**结论：八处表位中四处可直接复制、四处必须组合。** **可直接复制**：§4.2 主表（`main_table.md`，28 行无表头，列序逐列一致）、§4.4 干预表（`intervention_table_44.md` 是 **11 格**、论文 **10 列**，映射为"**去掉首格模型后逐格相同**"——不丢信息，因论文 `q/r` 列本身即含模型身份 `dense（r=H）`/`q=1/4（r=24）`）、§4.5（`conditional_table.md` 7 格 ↔ 论文 7 列，需跳过表头两行）、§4.6（`negative_table.md` 5 格 ↔ 论文 5 列，同样跳过表头）。**必须组合**：§4.2 臂级变体行（只写 `variant_table.csv`、**无 md**，且论文把 `l_q1_4`/`l_q1_8` **合并为一行**）、§4.4 解剖表（**无 md 写出点**，8 列需从 22+ 列组合，含"label＋explanation"与"input＋output overlap"两处多列合一格）、§4.7（两个 ρ 来自 `predictive_power_summary.json` 的 `predictive_power.spearman[f"{stat}_vs_delta_mse_pct"／"_vs_gate_value"]["rho"]`，键名取自 `e19_predictive_power.py:187-190` 而非推测）。**对校验的含义**：直接复制类用字符串比较（最强）；组合类需映射感知比较且**填完当轮校准**。文档：`docs/PhaseFormer_L/audit/minipaper_fill_mapping.md` |

| 2026-09-20 | §4.7 映射实测核验 | — | `minipaper_fill_mapping.md` §1.8 声称 §4.7 的两个 ρ 来自 `predictive_power.spearman[f"{stat}_vs_delta_mse_pct"]["rho"]` 与 `_vs_gate_value`——**这是我读源码推出来的，未观测过**。用 schema 精确的合成结果表（28 setting、`l_main`/`phase_only` × 3 seed = 168 行，统计量取**真实** `level_statistics.csv`）跑真实工具后观测：`predictive_power` 的键为 `['n_settings','scope','spearman']`，**`spearman` 恰有 6 项 = 3 统计量 × 2 后缀**，六条路径**全部存在且都带 `rho` 键** → 映射由"推理"升级为"实测"。附带再次确认边界：合成值秩退化时 `rho` 全为 `nan`，即 **summary 里可能出现裸 `NaN` 记号**（严格 JSON 解析器会拒绝），验证的是结构而非数值 |

| 2026-09-20 | E14 剩余工期双投影 | **纠正一处 4 倍误估** | E14 日志的 `done` 事件**无时间戳**，故完成率改从文件系统取（`metrics.csv` 的 mtime 即完成时刻）：115/411，近 15/30/60/120 分钟为 0.20/0.33/0.35/0.37 per min，整体 0.22/min。**投影 A（朴素）**按 0.35/min 外推剩 296 → **14.1 h、ETA 18:32**。**投影 B（按成本，推荐）**先由真实 manifest 逐数据集算"还剩什么"：Traffic 0、Electricity 8、Weather 48、ETTm1 72、ETTm2 48、ETTh1 72、ETTh2 48 = **296**（与实测脚本的 remaining **完全一致**，交叉验证 ✓），按实测/修正后单 run 成本合计 **≈23.1 GPU·h → 2.9 h → ETA ~07:20**。**差距 4 倍且朴素的那个错**：调度按**成本降序**发车，当前速率反映的是**最贵批次**（Traffic 0.7–1.1 h/run），而剩下的是**最便宜批次**（ETTh1/ETTh2 仅 37–85 s/run），把最贵批次的速率外推到最便宜批次必然严重高估。最大不确定性：Weather 占 9.1/23.1 ≈ **39%**，其成本用 `COST_HINT × 0.7`（外推），故区间 **2.5–4 h → ETA 07:00–08:20**；全链约 **19:00–20:00** 收尾，早于先前按"E14 到 11:00"的估计 |

| 2026-09-20 | 收尾作业清单 | — | 新增 §10：把"E14 收尾 → 全链结束"要做什么写成可执行清单，使收尾阶段**不靠记忆**。含：①**确认阶段二真的开始了**（watcher 的两个独立条件、`phase2_watch.status`、逐步日志、`pgrep` 检查）；②**失败重入方式**（`--from N`/`--only N`）与重入安全性（`--from>1` 跳过完成守卫；第 1 步读取器**幂等**；两道预检只在 `--from 1` 时跑）；③**逐步的"产物 → 机器检查 → 填哪张表"对照表**（第 3 步填 §4.2 主表并即时跑校验器；第 4 步填 §4.4 干预表 147 格 + 解剖表 105 格需组合；第 5 步 §4.5 35 格；第 6 步 §4.6 用 5 行**替换** 25 格）；④**唯一开放项**（§4.4 写法）的收口时点；⑤回填后的**两个机器判据**（`verify_minipaper_fill.py` 逐格一致 + `--inventory` 空格应为 0）与**五项人工审校**（表注披露、主张引用冻结门槛、C 的两种定义并列等）；⑥最后两件非机器可查的事（摘要只在 §4.2 判定后替换；各节补写阶段 5/6 文档） |
| 2026-09-20 | 清单自身核对 | — | 清单写完后**对着代码逐条核过**（清单错了比没清单更糟）：六个检查脚本 + 六个实验脚本**全部存在**；`--from`/`--only` 确由 `run_phase2_after_e14.sh` 实现、`--list` 输出 7 步；清单里写的 `main_table.{csv,md}`/`variant_table.csv`/`dissection_table_44.csv`/`intervention_table_44.{csv,md}`/`conditional_table.{csv,md}`/`negative_table.{csv,md}` **与各 write-back 的写出点逐字一致**。（其间我先用不带 `.py` 的路径自查，报了 2 个假的 MISSING——又是一个"我的检查写错、而非工件缺失"的例子，已用带扩展名的复核澄清。） |

| 2026-09-20 | 披露项逐条查证 | **查出两条缺失、一条位置危险** | §10.4 原先把五项披露写成"待核提醒"，本轮逐条查证（用关键词定位并读上下文），结论是"三项已存在、一项需新增、一项需搬迁"：①§4.2 三种门先验 ✅ 已存在（注中写明"存在三种 gate 先验…须在表注逐行标注"）；②**§4.4 的"`reference_parity` 只对 6 个 setting 成立"❌ 全文 grep = 0**（§4.5 里那处"6 个 setting"讲的是两条冻结臂按构造等价，是另一件事）→ **回填时须新增**；③**§4.6 的 E11 口径差异"存在但位置危险"**：该披露现在写在**表格单元格内**（第 3 行"本文补做"格），而 §4.6 的回填方式是**整行替换**——**照做就会把它删掉**，故必须**搬迁为表下注**；这条由此从"建议"升级为**有证据的必要动作**；④**主张 C 只写了一种定义**（§4.0 写"三均值+std"，而回填工具**同时输出两个计数**）→ 须补第二种定义，否则两个计数出现而无处解释；⑤§4.7 的 ν* 与 τ̂ 偏差 ✅ 已存在。**净动作仅第 2/3/4 三处**，第 1/5 项保持不动 |

| 2026-09-20 | §4.5 映射再核 | **查出最后一处口径不一致** | 复核 §4.5 的七列映射时发现第 7 列（H1）**不是直接复制**：论文表头写 `H1：cond 距离 < indep 距离（**seed 数**）`，要求一格**种子计数**（形如 `3/3`）；而产物 `conditional_table.md` 该格是 `h1_seed_majority`，取值 **`true`/`false`/`evidence_missing`**（**逐个 seed** 的判定，`e17_conditional.py:972-984`）。结果 CSV 里另有 `h1_ranks`/`h1_ranks_supporting`，但那是**rank 级**计数、不是 seed 级。→ 直接粘会把 `true` 贴到写着"（seed 数）"的表头下，**表头与内容对不上**。**正解**：按 setting 聚合 3 个 seed 的 `h1_cond_gt_indep_seed_majority`、数 `"true"` 的个数渲染成 `N/3`（全为 `evidence_missing` 时照写）——与本轮之前独立做过的 H1 汇总**完全一致**（当时报"ETTh2-96/720、ETTm2-96/192 = **3/3 seed**；Weather-96 = **2/3**；Weather-192 = **0/3**"正是按 seed 分组数出来的）。映射已修正为"前 6 列直接复制 + 第 7 列按 setting 聚合"，并规定回填时用**同一聚合口径**校验而非比字符串 |

| 2026-09-20 | §4.2 预填格预核 | — | §4.2 主表有两类**已预填**的格，若填法不当会与产物不一致，故动手前先核：①**Golden 列**——28 格与 `PhaseFormer_gold_standard.md` 的 28 行**逐格一致、0 处不符**（按 `f"{v:.3f}"` 渲染后比对），故它会与产物渲染相同 ✓；②**Traffic 行的 `来源/披露`** 也多预填了"探索性附录，不进入判定"——核到产物 `provenance_note` 的构造是**先写该句、再追加逐臂复用/新训披露**（`notes` 中 `if is_traffic_appendix:` 位于臂循环**之前**），即论文现有文字是产物串的**前缀** ✓。**故规定 §4.2 一律"整行用产物替换（含已预填的格）"**：整行替换既不丢那句附录说明，又补上缺失的逐臂披露；反之若"只填空格、保留短注"，那 4 行会被校验器报 MISMATCH——那是**填法错**，不是校验器太严 |

| 2026-09-20 | §4.5 校验器 + 校准 | **并查出一处会改错表的隐患** | 补齐 §4.5 的回填校验：前 6 列与 `conditional_table.md` 逐格比较；**第 7 列（H1）由结果 CSV 按 setting 聚合 3 个 seed 得到**（论文表头写"（seed 数）"，而产物那格是逐 seed 的 `true/false/evidence_missing`），比较时对数字容错（`N/M`），故写 `3/3` 或 `3/3 seed` 都不误报。**五类校准**：正确配对 → **`match: 49`**（7×7）、exit 0；改 H1 计数 / 改方法列 / 删一行 → 均 MISMATCH、exit 1；H1 留空 → 记 `blank`、不致命；聚合的四种结果（`3/3`、`2/3`、`0/3`、`evidence_missing`）都被走到。**顺带查出一处隐患**：校准首版"正确配对"竟报 8 处 MISMATCH，追查后确认**是校准脚本自己的错**——`§4.5 的行文本是 §4.4 解剖表行的后缀子串`（7 格 vs 8 格，内容正好是后者去掉首格），于是按子串 `replace` **命中了文档中更早的 §4.4 行**，改坏 §4.4 而 §4.5 未变（`changed=True` 却未生效正是此因）。已改为**按行、且限定在目标小节行区间内**定位，并断言"7/7 行都替换"否则拒绝继续。**该教训对任何自动填充/批量替换都适用，已写进映射文档作为操作警告** |

| 2026-09-20 | §4.4 干预表校验器 | — | 映射为 `artifact_row[1:] == paper_row[0:]`（产物 11 格含"模型"列，论文 10 列无该列——**模型身份已含在 `q/r` 标签里**，由 `qr_label()` 生成）。**五类校准**：正确配对 **`match: 210`**（21 行 × 10 列）、exit 0；改论文一格 / 改产物一格 / 删一行 → 均 MISMATCH、exit 1；留空一格 → `blank`、不致命。**顺带查实一处预填差异**：论文 dense 行的 `q/r` 预填为通用写法 **`dense（r=H）`**，产物写**具体值** `dense（r=96）`（低秩行本就具体）→ 故**必须整行替换（含 `q/r` 格）**，否则那 7 个 dense 行会被报 MISMATCH，而那是填法问题。覆盖更新：§4.3/§4.2/§4.5/§4.4 干预表已覆盖；剩 §4.4 解剖表（写法待定）、§4.6、§4.7 |
| 2026-09-20 | 一次"忘了同步" | — | §4.4 校验器的校准首轮**全项"通过"却是假的**：输出里**一行 §4.4 都没有**。查明是**我改了本地脚本却没 commit/sync**，服务器跑的还是旧版——**破绽是本地报 `PENDING: 4` 而服务器报 `PENDING: 3`**（差的那 1 项正是新加的 §4.4）。commit+sync 后重跑才得到真实结果（210/210 与四类拒绝）。**教训**：在服务器上验证任何脚本前，先确认"服务器上的版本 == 我改的那版"；三态计数（PASS/PENDING/未覆盖项数）本身就是廉价的版本指纹 |

| 2026-09-20 | §4.6 回填范围修正 | **推翻自己上一轮的结论** | 我先前据"`existing` 列 4/5 逐字相同"推断 `negative_table.md` 的设计意图是**整表重建**。本轮把**全部五行的前三列**也逐格对照后，该推断**站不住**：**行 2/3/4 三列逐字相同，但行 1 与行 5 措辞不同**——行 1「操作」论文写 `输入平滑（boxcar / causal EMA，各 5 档）`、产物写 `输入平滑（causal EMA 两个强度）`；行 5 三列都不同（论文 `支路容量（网格之外）`/`6 setting × 3 seed` vs 产物 `支路容量（低秩网格之外）`/`6 setting × 3 seed × 2 rank`）。而论文那两处措辞是在描述**先导实验**（行 1 明确写"boxcar / causal EMA，各 5 档"，正是第 3、4 列所报既有实验的算子网格）→ **整行替换会把准确的既有描述改写掉**。**修正规则**：只替换**第五列**（取产物 `addendum`），前四列保持论文原文；被顶掉的计划/规模文字（`42 runs`、`36 runs`）移入表下注（行 3 的 E11 差异**已先行搬迁**）。校验器据此实现并五类校准（第五列严格比较、前四列不同记 `INFO`） |
| 2026-09-20 | 判据盲区（重要） | — | 由 §4.6 的"替换型"回填发现：**`--inventory` 的空格数看不到 §4.6 的回填**——该表 25 格本来就非空（装的是计划文字），回填是**替换文字**，故**即使 §4.6 完全没填，空格总数仍是 0**。**故完成判据必须两条并用**：①`--inventory` 总数 → 0（覆盖**填空型**：§4.2/§4.4/§4.5/§4.7）；②`verify_minipaper_fill.py` 无 MISMATCH（覆盖**替换型**：§4.6，及两类的一致性）。只跑第 ① 条会误判"§4.6 已完成"。已补进映射文档 §4 与本节 §10.4 |

| 2026-09-20 | §4.7 校验器 + 校准 | — | 补齐 §4.7 的两个 ρ 列校验（3 行 × 2 列，来自 `predictive_power.spearman[…]["rho"]`，键结构此前已实测核验；第 4 列"预期符号"是设计文字不校验）。**五类校准**：正确配对 → `match: 6`、exit 0；改论文 ρ / 改产物 ρ → MISMATCH、exit 1；ρ 留空 → `blank`、不致命；**产物 ρ 为 NaN → 记 `PENDING` 而非 `MISMATCH`**（那是"产物未给出可比值"、不是"论文与产物矛盾"）。**校准里我自己一处脆性**：首轮报 `line not found`——我按"重建行文本"去匹配，而**§4.7 的空格写成单空格 `\| \|`、其它表是双空格 `\|  \|`**，故匹配不上（校验器本身没问题，它按 `strip()` 取格）。已改为**按键格匹配**。教训：markdown 表空格写法在各表间不统一，**凡"重建行文本再匹配"都脆，应按格/按键定位**——这与 §9.3 的"后缀撞车"是同一类问题的另一面 |
| 2026-09-20 | 校验器覆盖总表 | — | **七处表位中六处已有经校准的校验器**：§4.3（168 格，已填已验证）、§4.2 主表（280）、§4.4 干预表（210）、§4.5（49）、§4.6（5 格第五列，替换型）、§4.7（6）。**仅剩 §4.4 解剖表（105 格）**，其收口时点已定（E16 跑完后先打印真实取值、定写法、再实现并校准），**不提前写未经验证的解析器** |

| 2026-09-20 | §4.4 解剖表校验器 | **最后一处开放项收口** | 上一轮把"单元格写法"列为唯一开放项、打算等 E16 出数据后再定。本轮改为**从代码与表头直接定**（无需运行），依据有三：①论文表头自己写作 `主模式输入组 / 解释率`，**斜杠即约定**；②生产者判据 `criterion_2_input_explanation_ge_0p5` 说明解释率是 **0–1 小数**；③`model` 取 `ARM_DISPLAY`，与论文行标签 `PhaseFormer-L`/`L-q1/4`/`L-q1/8` **逐字相同**。映射（取自 `build_dissection`）：输入/输出组列 = `label ＋ mean_*_explanation`；修正能量份额 = `leading_correction_energy_share`；跨 seed 重叠 = `leading4_input/output_overlap`（论文一格、产物两列）；判定 = `stable_semantics_verdict`（**bool**）。**渲染**：`{组名} / {解释率:.2f}`、份额 `{:.3f}`、重叠 `{in:.2f} / {out:.2f}`、判定 `✓/✗`。**刻意的容错**：组名本身可能含斜杠（真实标签有 `周期形状/相位`），故**不按斜杠切分**，改用"以组名开头 + 末尾数字 == 两位小数"判定，判定列亦接受多种写法 → 写法小差异不造假报警，而数值与标签仍严格校验。**七类校准**：正确配对 `match: 105`、exit 0；改解释率/改组名/改重叠对/翻转判定/产物少 3 行 → 均 MISMATCH、exit 1；组合格留空 → `blank` 不致命（合成标签**故意含一个带斜杠的**以走到容错路径） |
| 2026-09-20 | 回填判据完备 | — | **七处表位全部有经校准的校验器**：§4.3（168，已填已验证）、§4.2 主表（280）、§4.4 干预表（210）、§4.4 解剖表（105）、§4.5（49）、§4.6（5 格第五列，替换型）、§4.7（6）。**回填阶段的机器判据至此完备**：①`--inventory` 总数 → 0（填空型）；②`verify_minipaper_fill.py` 无 MISMATCH（含替换型的 §4.6）。**无待定项、无未校准的解析器** |

| 2026-09-20 | 实测刷新 ETA | — | 从已完成 run 的 `elapsed_sec` 取实测中位数（不再是外推），发现两处先前估值偏高：①**Weather h720 实测 449 s**（n=2），而 `COST_HINT × 0.7` 给的是 ~840 s——比值约 **0.37**，比我统一采用的 0.7 折扣低得多；②**Electricity h192 实测 848 s**，比 h336（1589 s）与 h720（1285 s）都便宜，说明此前用"Electricity ≈1285 s"统一估是偏高的。按实测折算剩余 289 个 run ≈ **15.3 GPU·h → 1.9 h → ETA ≈ 06:35**，区间 **06:30–08:10**（较 §9.1.2 略提前）。**诚实的不确定性**：Weather 样本仅 n=2 且恰是 hint 最贵的 horizon、ETT 四数据集尚无实测，故只作区间表述 |

| 2026-09-20 | 规则符合性自查 | **补齐两项未做的要求** | 对照 `MANAGE_RULES.md` 逐条自查"规则要求我留下什么"，发现两项未做：①**「验证要求」明列的轻量验证**（`python -m pytest tests/ -q`）——本日改过**产品代码**（`read_test_generic.py` 的 marker 修复、`e18_writeback.py` 的 None 保护、`e14_writeback.py` 的 FITS/空 variant 保护、`evaluate_lowrank_semantic_interventions.py` 的 `RandomRRR-drop` 臂），却只做了针对性冒烟与预演、**没跑仓库既有测试**；②「操作与改动记录」要求的 `docs/agent-log.md`（已补三条 2026-09-20 条目）。**测试结果**：`tests/test_lowrank_checkpoint_information.py` **26 passed（14.35 s）**；全量 `tests/`（37 文件）**388 passed + 262 subtests passed, 18 warnings（151.55 s）, exit 0** → 本日对产品代码的全部改动**未破坏任何既有测试**。18 条 warning 均为 `e19_predictive_stats.py` 的 `np.nanmean` 空切片（退化输入下的预期行为，由测试用例主动构造）。**覆盖边界**：仅 1 个测试文件会触及我改过的模块，跑全量是为满足规则要求 |

| 2026-09-20 | 规则符合性自查（第二轮） | **全部合规，无新缺口** | 继续按 `MANAGE_RULES.md` 核对前两轮未查的条款，四项全部通过：①**新实验脚本的说明义务**（规则要求说明数据集路径/关键超参/运行命令/输出目录）——六个 E14–E19 脚本**都**在 docstring 里给了 `Usage::` + 完整解释器路径 + `--output-root` + 数据集引用 ✓；②**不提交机器绝对路径**——新脚本里出现的绝对路径**只有**运行命令里的解释器路径，而规则本身**要求**"运行命令使用完整解释器路径，避免依赖 `conda run` 或激活 shell" ✓（属合规且必需，非违规）；③**不提交大体积/临时/缓存文件**——`git ls-files` 中 `*.pyc`/`__pycache__`/`*.log`/`*.ckpt`/`*.pt` 计数为 **0** ✓；④**最大跟踪文件**仅 524 KB（`uv.lock`）、`agent-log.md` 308 KB，无大体积产物 ✓。**并记一次我自己的假报警**：我最初用 `grep -c 'python scripts/phaseformer_L/'` 判定 `e19_predictive_stats.py` 缺运行命令，实为**命令按反斜杠换行**导致单行模式匹配不到——该文件第 84–90 行确有 `Usage::`。又是"我的检查写错 ≠ 工件缺失" |

| 2026-09-20 | 失败语义核验 | **全链 fail-closed（含一处已记录例外）** | 无人值守链路最危险的失效是"**带着空洞继续走**"（链跑完、表填出，但某些格其实缺）。本轮把该语义**逐层核到源码**：训练 runner **任一格失败即非零退出**（`e14_main_matrix.py:702-703 if failed: raise SystemExit`；`e17_conditional.py:1407`、`e18_negative.py:890-891` 同构）→ `run_step` 在第一个非零退出处停止 → watcher 记 `PHASE2_FAILED` → E14 完成守卫要求 `E14_MAIN_EXIT=0` **且** `runs/` 恰 411 目录，故 E14 若有失败格，**watcher 不会启动阶段二** ✓。**一处例外及其兜底**：`read_test_generic.py` **永远 exit 0**（即使 cell 被拒或 worker 崩溃，只在汇总列 `rejected`）——由**第 7 步验收审计**兜住（E17/E18 各有"每 cell 都要有 test 指标"判据），且代价有限（真出问题只需重跑该步的读取+回填，24/78 个训练 run 仍有效），故**保留其 exit-0 语义不改**，但例外本身必须写明，不能默认"全链都 fail-closed" |

| 2026-09-20 | 恢复代价核验 | **全链唯一昂贵重跑是第 4 步** | 承接失败语义，核实"失败后重跑要付多少"（决定该不该中途干预）。关键在 `search_phaseformer.py:638-644`：若 `metrics.csv` 存在，**带 `--resume` 时打印 `RESUME completed` 并干净 return（退出码 0）**，不带时才 `raise FileExistsError`。而 E14/E17/E18 的命令**都带 `--resume`**（`:468`/`:525`/`:247`），E16 **完全没有**。故：第 1 步幂等、第 3 步分钟级、**第 5/6 步廉价**（dispatch 虽重新发车每格，但已完成的 run 立即 no-op，**不重训**）；**第 4 步（E16）昂贵**——无 `--resume`、无按 cell 检查点、产物末尾一次性写出 → 失败即需**整步重跑 3–5 h**，是全链唯一昂贵重跑；`--from N` 另可跳过节级 ✓。**并诚实记下我一个过早结论**：我先只读 dispatcher、见 `pending = list(cells)` 无发车前跳过，几乎把"重跑会重训所有 run"写进文档；往下多读一层才发现子进程带 `--resume`、结论**正好相反**——"从单层代码推断跨层行为会出错"，好在写进文档前发现 |

| 2026-09-20 | 防线地图（谁 fail-closed） | **第 7 步审计是承重的** | 逐工具核到源码，得到两类（**两类都是有意的，不是缺陷**）：**失败即停**——第 1 步 `e14_read_test.py`（写盘后 `if problems: raise SystemExit`；dry-run 更早拒绝）、第 5/6 步训练 runner（`if failed: raise`）、第 6 步行 3 的 `--verify`（实测 `135 unresolved cells; refusing`）、第 7 步审计（有 FAIL 即非零）；**报告后继续（exit 0）**——第 5/6 步 test 读取（已知例外）、第 2 步 `e19_predictive_power.py`（只在"一个 setting 都算不出"时才停；部分跳过仍 exit 0）、第 3 步 `e14_writeback.py`/`e14_params.py`（**无任何 `raise SystemExit`**，只在 `audit.json` 列 `settings_incomplete`）、第 4 步 `e16_dissection.py`（`algebra_failures` 只进 summary 不抛错）、三个回填工具。**结论**：链条在**入口与最贵的两步**是 fail-closed，在分析/回填层是"报告后继续"——而**第 7 步审计正是为这一层设计的**，它检查的恰是这些工具"报告而非中止"的东西（每 cell 有 test 指标、预测力表覆盖 28 setting、主表 492 行、`algebra_failures` 为 0、行 3 覆盖 28 setting）。**故第 7 步不是装饰、是承重**：若把这些工具改成"失败即停"，链会因一个**良性**跳过而过早停止；留在报告层、由审计统一判定，则跑完全链后一次性给出可信结论。审计跑在最后也因此是对的——它之后没有步骤，"最后才发现"的代价只是我看一眼报告 |

| 2026-09-20 | 反查审计器判据 | **发现并加强四个过弱判据** | 由 §10.1.3 的防线地图反查："**把关者的判据是否真能挡住它要挡的东西**"。逐条对照后发现四个判据用"非空/存在"代替了真正的计数或一致性判据：①E14 `parameter_table.csv` 原判"non-empty"——而 `e14_params.py` 是**报告型**（解析不到的格列进 `unresolved` 并**照样 exit 0**），半张矩阵缺失也会通过 → 加强为 **行数 = 492**（与 `cells_with_parameters + unresolved = 492` 一致）；②新增 **每行 `total_matches_metrics` 不得为 False**（该工具本会与 run 自身 `metrics.csv:parameter_count` 交叉校验，审计器此前没看）；③新增 **每行 `gate_value_from_checkpoint` 非空**（§4.2 的 `g` 列依赖从 checkpoint 恢复门值）；④E18 原判"results non-empty" → 加强为 **行数 = 78** = 42 平滑（7×3×2）+ 36 边界（6×3×2），由计划计数与**我先前预演的行数算术**两条独立一致。**六类校准全部通过**：492 行一致 → 三项 OK；491 行 → FAIL；一行交叉校验 False → FAIL；一行缺门值 → FAIL；E18 78 行 → OK；E18 77 行 → FAIL 且 exit 1。**方法论收获**：不是逐个看判据是否合理，而是先问"**这条判据要挡住什么**"——一问就发现"非空"挡不住"半张矩阵" |

| 2026-09-20 | 判据覆盖矩阵 | **无"报告却无人接住"的缺口** | 把 §14 的教训系统化：列出六个工具的**全部缺口计数**，逐条指出**哪条审计判据接住它**（或**为何有意不查**）。结果：`e14_read_test` 的 `problems`/`failed_workers` → E14 的"492 行 + 每行有 test"（且该步自身 fail-closed）；`e14_params` 的 `unresolved`/`mismatches`/门值缺失 → §14 新增的三条；`e16_dissection` 的 `algebra_failures`/`run_metric_*` → E16 三条（故每 cell 的 `run_metric_state == "skipped"` 已被间接接住：全量时必须为 `ok`）；其 `parity={"skipped":True}` → E16 的 `reference_parity_passed is True`（键缺失即 FAIL）；`e17_conditional` 的 `failed` → E17 的 24 新 cell + test 指标；`e18_negative`/`e18_svd_truncation`/`e19_predictive_power` 的缺口 → 各自判据。**唯一"有意不查"**是 §4.5 的 `settings_without_evidence`——`evidence_missing` 是**已披露的合法结果**（Electricity-336），当失败会误报。**新增一条**：E17「无缺失的冻结投影器」（投影器缺失属**基础设施故障**而非良性跳过，值得显式报出，否则症状只剩"0 个新 cell"）。**八类对照校准全部通过**（492/491 行、交叉校验 False、缺门值、E18 78/77 行、E17 有/无缺失投影器），详见 `paper_code_consistency.md` §15 |

| 2026-09-20 | 校验方法修正 | **静态提取器错报 4 个缺口** | 「阶段二消费者读得进 E14 的产物吗」此前无判据。先用正则扫五个消费者源码里的 `X.get("<key>")`，对真 manifest 报出 4 个缺口（E14 缺 9 键、E16 缺 34 键、E17 缺 5 键、E18 缺 7 键）——**全部是假警报**：这些键的接收者是脚本内部构造的 `config`/`hyper`/`cell`/CSV row，不是 manifest cell。改法：**直接调用消费者自己的 loader** 对真产物跑一遍（`load_manifest_cells`/`build_e14_index`/`load_baseline_index`/`arm_cells`/`build_cell_plan`）⇒ 报出来的缺口才可能是真缺口。与 §14 同源教训：判据要问"**它实际读的是哪个对象**" |
| 2026-09-20 | 集成校验 | **修掉 E18 基线出处列 78/78 行全空的缺陷** | 真 manifest 实测 `load_baseline_index` → `resolved=63, rejected=21`，21 条拒绝全是 `overrides do not implement l_main`。**根因**：manifest 里 **reused 格的 `command` 是 null**（stage A 没启动它们，是复用审计收编来的），E18 却从 argv 的 `--overrides` 重推 `_arm_match` ⇒ `overrides={}`、头类型取不到 `shared` ⇒ 拒绝。**不是边角**：`SMOOTH_SETTINGS = REUSE_SETTINGS_FULL`（7 setting）**正好等于**被复用的那 7 个，`RANK12_SETTINGS` 的 6 个也全在其中 ⇒ **78 行（42+36）无一例外**。**影响面**：**§4.6 数字不受影响**（`build_row1/row5` 的 baseline 取自 E14 `results.csv`，与出处列无关），受损的是**审计链**（产物会记录 21 条拒绝 + 78 行空出处，而 manifest 自称 `arm_match_used_for_baselines`）。**修法**：为 reused 格加第二条准入路径——从其指向的 run 的 `config.json` 用**同一把尺子**（`_arm_match`）重推指纹，`gate_init`/`learning_rate`/`eval_root` 取自该证据；config 缺失或不满足 `l_main` ⇒ **照样拒绝**（不凭空信任 `source`）。同时 new 格 `eval_root` 改为 manifest 所在 root——实测 411 个 new 格记录的 `--output-dir` **全是模板值 `/tmp/e14fix`**（仓库 `grep` 零命中，是重建 manifest 时的残留），照抄会写进产物。**验证**：真产物 `resolved=84 (new=63, reused=21), rejected=0`、smooth 覆盖 21/21；12 个单测（含 4 个否定对照）；全套 **400 passed** |
| 2026-09-20 | 判据固化为 pre-flight | **C1–C5 接进阶段二 pre-flight；C3 就是能在 3–5 小时之前抓住它的那条** | 新增 `scripts/phaseformer_L/check_phase2_consumers.py`（失败即停：C1 E14 loader 接受 manifest 且 492 格形状唯一、C2 E17 声明的 21 格全解析、C3 E18 smooth 每 (setting,seed) 都有基线且 0 拒绝；信息：C4 E16 当前可解析格数、C5 SVD 计划规模，**未解析格属预期**），接入 `run_phase2_after_e14.sh` 既有 pre-flight（与列契约、调用元数并列，**在任何昂贵阶段之前**）。真产物实测：C1 ✅ 492 cells、C2 ✅ 21/21、C3 ✅ 21/21（`resolved=84`）、C4 ℹ️ 63 格、C5 ℹ️ 28 setting/13 带问题，exit 0。**三类对照**：合成 fixture 正对照 exit 0；删一个被收编 run 的 `config.json` → `rejected=1` exit 1；手改 new 格 overrides → `rejected=1`、理由 `overrides do not implement l_main`。C4/C5 在缺 numpy/torch 时降级 SKIP 而非失败 |
| 2026-09-20 | 运维修正 | **同步命令一直是空操作** | `git fetch origin && git merge --ff-only FETCH_HEAD` 报 "Already up to date"，但服务器 bundle `list-heads` 明明含新提交。**根因**：`remote.origin.fetch = +refs/heads/*:refs/remotes/origin/*`，而 `git bundle create <file> HEAD` 只写**一个 `HEAD` ref** ⇒ 裸 `git fetch origin` 匹配不到任何 ref，**连 `FETCH_HEAD` 都不生成**。正确形式：`git fetch origin HEAD`（本次已用并实测服务器 HEAD 前移）。判据：`git ls-remote origin` 有值 **≠** `FETCH_HEAD` 有值——同步后必须实测服务器 HEAD 变了 |

| 2026-09-20 | 上线前校对（三道 pre-flight） | **按流水线的原样跑通，闸门会开** | 在服务器上按 `run_phase2_after_e14.sh` 的实际写法跑了三道静态检查：①列契约 `--strict` → `total missing consumer columns: 0`、rc=0（9 张表的产出列集合与 4 个 write-back 的消费列逐条对齐）；②调用元数 → 66 个 flag 全声明、无单值 flag 收到多值、rc=0；③消费者契约 → C1 ✅492 cells、C2 ✅21/21、C3 ✅21/21（`resolved=84`，0 拒绝）、C4 ℹ️63 格、C5 ℹ️28 setting、rc=0。即**链不会卡在 pre-flight 闸门上** |
| 2026-09-20 | E16 前置门**提前**跑（仍未等 E14 跑完） | **通过：63 格全解析、63 个 checkpoint 全读通、随机 RRR 已接线** | E16 是全链**唯一昂贵重跑**（无 `--resume`，3–5 h），但其 63 格**全是 reused 格**、依赖的 checkpoint 早已定稿 ⇒ 前置门不必等 E14。在 E14 仍训练时（137/411）执行 `e16_dissection.py --dry-run --verify-checkpoint-heads`：**exit 0**（日志 `~/niuyiming/logs/e16_pregate.log`）。计划事件实测 `cells=63`、`by_arm={l_main:21, l_q1_4:21, l_q1_8:21}`、`by_e14_status={reused:63}`、**`random_rrr: true` + `random_repeats: 100` + `random_rrr_repeats: 100`**（§4.4 要求的随机 RRR 对照**确实接线**，不是"计划里有、代码里没有"）、**`evaluation_split: "val"` + `test_split_read: false`**（协议护栏在位）。逐格头类型一致：`l_main`→`dense_shared`(keys=1)、`l_q1_4`/`l_q1_8`→`pooled_lowrank`(keys=2)，秩 = horizon/4 与 horizon/8（实测 12/24/42/48/84/90）。`--dry-run` **不写任何产物**（输出根跑后仍不存在）⇒ 预先验证**未污染**该实验 |

| 2026-09-20 | 第 7 步审计器的**静默缺陷 A** | **会在 411 个 run 全跑完后误判整链失败（已修）** | 拿**真产物**喂判据时暴露：门值判据原写"**每一行**都要有门值"，而门参数**只存在于三个弱残差臂**（`l_main`/`l_q1_4`/`l_q1_8`），`phase_only`/`l_rcrf`/`a1` **根本没有这个参数** ⇒ 492 行里 **85 行合法无门值**。实测（把 `e14_params.py` 真 partial 输出 220 行放进合成根、用 `--root` 跑旧版）：`FAIL gate value recovered from every checkpoint: 85 row(s) without a gate value e.g. Traffic-96/l_rcrf` ⇒ **第 7 步会在全部训完、表全填好之后报告"链条失败"**。代价不是重跑（纯 IO）而是**在最容易被误信的时点给出与真结论长得一样的假警报**。**修法**：按行收窄作用域（用表自己的 `gate_param_present` 列做判别式），**另立一条**判据断言"臂 ↔ 是否有门参数"一致（否则整列恒 `False` 会让前一条因"没东西可查"而通过），再加"无门臂带门值 = 写错表"。修后同一份真产物：`OK 0 gated row(s) without a gate value (of 135 gated rows; 85 rows have no gate parameter)`；旁证 `total_matches_metrics` 0 处不一致、`total_params` 无一为空 ⇒ **真产物本来是对的，错的是判据** |
| 2026-09-20 | 第 7 步审计器的**静默缺陷 B** | **`--json` 被解析但从不使用（已修）** | 第 7 步的调用是 `audit_phase2_outputs.py --json "$LOGDIR/phase2_acceptance_audit.json"`，而 `main()` 里 `args.json` **一次都没出现** ⇒ 收尾时"验收报告 JSON"**静默不存在**。没有机器消费者依赖它（故不会失败），但这正是"文档承诺的产物没落地"。**修法**：`Report.to_dict()/write_json()` 写出 `criteria/counts/failing/total_criteria/root`。顺带澄清：判据总数**不是常数**——24 条是"什么产物都没有"时的基线，每多一张存在的表就多登记若干条（参数表存在时 +3） |
| 2026-09-20 | 校准固化 | **`rehearse_audit_controls.py`：13 类对照写入仓库** | §15.1 的八类对照此前只在 `/tmp` 临时跑（随会话消失）。本轮写成仓库内脚本，用审计器**自己的 `--root`** 对合成树断言逐条判决：492 行真实混形 → 三条 PASS；有门臂门值被清空 → FAIL；**无门臂却带门值 → FAIL（此条最初漏网）**；有门臂被标 `False` → FAIL；491 行 → FAIL；一行交叉校验 False → FAIL；**492 行全为无门臂 → PASS**（缺陷 A 的最小复现）；`--json` 嵌套路径 → 写出且 `total_criteria` 与 `criteria` 长度一致（28 条）。**校准器当场抓出我自己修法的漏洞**：#3 首跑是 PASS，因为我只比了"臂↔是否有门参数"、没管无门臂上残留的门值；补 `stray` 检查后才 FAIL |
| 2026-09-20 | 方法论（§14+§17 合并） | **三种喂法必须都有** | ①**真产物**证明判据**不误杀**（§17：过严的判据会在链尾假报失败）；②**扰动产物**证明判据**不放过**（§14：过弱的判据放过半张矩阵）；③**调用方声明**证明产物**不空转**（§17-B：`--json` 从不写文件）。三者缺一，就会剩下一个只在特定输入下才现形的静默缺陷 |

| 2026-09-20 | §4.4 臂数**逐格而定**（第三次"写死基数"缺陷） | **审计器会在 E16 那 3–5 小时跑完后判整链失败（已修）** | 三处同时写死"11 臂"：①审计器 `intervention rows == 63×11 = 693` 且 `len(arms) == 11`；②回填 `expected_arms = 11`；③论文 §4.4 正文"10 登记臂 + 追加 3 个 ⇒ 11–12 臂"（算术就不自洽：10+3=13）。**逐格读该格自己的 checkpoint 实测**（直接 import `build_bases` 用的 `semantic_basis`/`latent_image`，复刻其条件）：`PCA-matched-only/-drop` 在 **63/63 格**成立（语义张成 36–39 < 最小秩 42、稠密 720）；`Independent-RRR-only` 与 `RandomRRR-drop` **恒在**（后者因 `--random-rrr` 默认 True）；`Conditional-RRR-only` 在 **42/42 低秩格**成立（84 个 Stage-3 `.npz` 全含 `conditional_basis`，稠密头无此文件）⇒ **11 臂 24 格、12 臂 21 格、13 臂 18 格，合计 750 行、13 个不同臂名**（`l_main` 252 / `l_q1_4` 255 / `l_q1_8` 243）。**修法**：判据改结构式——「63 个 (arm,setting,seed) 组」+「每格必须含 10 个**恒在臂**」+ 与 `e16_summary.json` 的 `counts.intervention_rows` / `intervention_arms_per_cell` **互相印证**（不含任何常数），臂数本身降级为 INFO |
| 2026-09-20 | 我在这条路上先给了**两个错的数** | 798 → 750（如实记录） | ①**798**：第一版探针取"每臂一个代表 checkpoint"，把该 checkpoint 的秩套到**所有** setting 上（`l_q1_8 ETTh2-96` 被当成 r=42，实际 r=12=96/8），且**假设**低秩格都有 conditional 文件，两个错叠加；②**750**：改成逐格读该格自己的 checkpoint、并实际检查 42 个 Stage-3 文件后的数。期间还差点把"11/12/13"写成"12/13"（漏了 `semantic_dimension == rank_dim` 的格子：r=12/24 且张成 36–39 时两者相等 ⇒ **不**追加 PCA-matched）。**与 §9.1.2、§9.1.4 同类**：拿一个**代表量**去套一组**构成在变**的对象；判据是"当结论依赖每个对象自己的属性时，必须逐个取，不能取代表" |
| 2026-09-20 | 校准器（第 3 次）再抓出我的架台错误 | **20 类对照，全部符合预期** | 新增 7 类 E16 对照（合成 63 格真实混形 → 四条判据 PASS；删一个恒在臂 → FAIL；总行数与 summary 不符 → FAIL；只有 62 格 → FAIL）。**校准器先报了两条"失败"，查明是 fixture 错**：合成格键最初用 `ETT-96/192/336` 这类**重名** setting，`(arm,setting,seed)` 三元组碰撞，63 格被算成 21 格、再算成 36 格；换成真实 7 个 setting 名后才是 63 ✅（与 §17.3 同类：先分清"判据错了"还是"fixture 错了"） |
| 2026-09-20 | 论文 §4.4 正文更正 | 臂数说明改为逐格而定 + 实测分布 | 把"故实际为 **11–12 臂（11 为下界）**"改为注明：`Independent-RRR-only` 与 `RandomRRR-drop` **每格必有**，`PCA-matched-only/-drop` 仅在语义潜像维数 < 该格头秩时追加，`Conditional-RRR-only` 仅在该格有 Stage-3 文件时追加（稠密头没有）；实测 11/12/13 臂、63 格 750 行、13 个臂名；并写明判据按"10 个恒在臂"而**不是**写死臂数。§4.4 表格 21 行 × 10 列结构不受影响 |

| 2026-09-20 | 第 1 步**两条分支**都在真产物上跑通 | new 格路径（含协议指纹）+ **全部 81 个 reused 格** | (a) `new` 分支（411 格）：对一个**已完成**的真实 run 跑 `--dry-run` —— `fingerprint_check` 报 `constants_equal: true`、`differences: []`、`parity_cases: 20`、`parity_failures: []`（⇒ 不会在 `preflight_new_cell` 处因 protocol drift 被拒）；该格 `status: planned`、`accepted: 1`、`problems: 0`，并把同 setting 的另外 4 个臂作为 **near-miss 带理由排除**（`no_residual` / `rcrf_nlinear_plain` / 两个 `pooled_lowrank`）——即"5 臂共用一个 setting-seed、靠 config 指纹而非目录名区分"的直接证据。(b) `reused` 分支：用 `--cells-file`（行格式是 `arm:dataset:horizon:seed`，**不是** manifest 的 `arm__Dataset-H-sSEED`）**一次验完全部 81 格** —— `accepted: 81, problems: 0, warnings: 0`，`by_arm={l_main:21,l_q1_4:21,l_q1_8:21,phase_only:18}`，且 `source.test_evidence` **无一为空**（外部证据 36 行、0 拒绝）⇒ 不存在"某复用格没有 test 来源 ⇒ 第 1 步 fail-closed 卡住"的风险。两次均 `wrote_outputs: false`。**剩余不确定性只有时长**（≈1.3 h，见 §9.1/§9.1.1a） |
| 2026-09-20 | 审计器读取列名 ↔ 产出者**逐一核对** | 未发现新缺陷（§19） | `check_column_contracts.py` 覆盖的是"回填工具需要什么列 ↔ 产出者写了什么列"，**不覆盖**"审计器读什么列"。逐项人工过一遍 16 行（E14 results/parameter/main/variant/claims、E16 summary/reference_parity/intervention/dissection、E17 conditional/results/summary、E18 results/svd/negative、E19 predictive）：**全部对得上**。其中 E16 `reference_parity.probe_cells` **实测 = 72**（两个参照文件各 36 个匹配键 = 6 setting × 3 seed × 2 低秩臂；Electricity-336 不在参照内）⇒ 审计器的 `72 ± 2` 判据成立。这条校验的价值是**排除**"审计器读一个不存在的列名 ⇒ 整列被判缺失（如 E18 的 78 行全部没读 test）⇒ 全链跑完后才现形"这一类静默失败 |

| 2026-09-20 | 三项收尾客观条件核对（磁盘 / §4.3 图 / 文件名链） | 全部通过 | ①**磁盘**：`research_runs` 现占 310 G，`/home`（并行盘）**可用 8.7 PB**（37% 已用）、`/` 可用 9.7 TB；E14 根 658 M / 159 runs ≈ **4.2 MB/run** ⇒ 411 runs 约 **1.7 G**，E16 的 750×215 表与 E17/E18 产物合计不过数 G ⇒ **容量无风险**（最大单 run 11 M = Traffic h720）。②**§4.3 的三类图确实存在**：`research_runs/phaseformer_L_e15_dimension_v1/figures/` 下 `scree_lambda_spectrum.png`(242 K)、`b1_lag_profile.png`(252 K)、`a1_horizon_profile.png`(344 K)，均为 09-19 19:43 E15 收尾时写出 ⇒ 论文 §4.3 对三张图的引用有真实文件支撑。③**文件名链**：`read_test_generic.py:377` 写 `Path(results).with_suffix(".with_test.csv")`，即 `X/results.csv` → **`X/results.with_test.csv`**——正是审计器（E17/E18 的 `results.with_test.csv` 判据）与 `e17_writeback`/`e18_writeback` 读的那个名字；第 5/6 步正是以 `--results '$E17_ROOT/results.csv'` / `'$E18_ROOT/results.csv'` 调用它 ⇒ 链条一致，不会出现"审计器读到的是 PENDING 而不是 FAIL"的静默缺口 |

| 2026-09-20 | 回填的**第三类判据**被发现（正文占位） | `/tmp` 判据覆盖不到的地方：**填格碰不到格子周围的句子** | 填格只写表格单元。全文有两处 `待填` 在**正文**里：摘要 `*[主结果待填。]*`（:58）与 §4 开头填表状态段（:369，写着"§4.2、§4.4、§4.5、§4.6 待填"）。不管它们，**填满的表会被一段自称"主结果待填"的文字包着**，而逐格比对**完全看不见**。**新增判据③**：`verify_minipaper_fill.py` 的 `check_placeholders()` 把每处 `待填` 报成 `blank`（报告而不失败）⇒ end-state = **`--inventory` 0 + 无 MISMATCH + `blank` 0** 三者同时成立。**双向校准**：现稿报 2（:58、:369）、把两处替换成"已填"的临时副本报 0 且 exit 0；服务器上（产物齐备）实测 `match: 168, blank: 2, PENDING: 6`、exit 0。另记一处**无标记**、必须人工改的正文：§5 限制第 4 条（:738-740）现在把随机 RRR 对照写成"**必要实验**"（将来时），§4.4 跑完后要改成**它实际分开了什么** |

| 2026-09-20 | 第 6 步行 3 的**结构性就绪度**（可证伪计数，非分类判断） | **0 处结构性缺失** ⇒ "会在 E16 之后才卡住"的风险排除 | 行 3 的 `e18_svd_truncation --verify` 在任一格 run dir 未解析时**拒绝评估**，故"某个 setting 永远解析不了"会让链条在**付掉 E16 那 3–5 小时之后**停下。直接调 `e18_svd_truncation.load_manifest`/`build_plan`，对每个未解析的 `(setting, seed, arm)` 回查 manifest 是否**声明**该格：**未训练 36 格 / manifest 中根本没有 0 格**，且 **28 个 setting 全部带 `l_main`+`l_q1_4`+`l_q1_8` 三臂**（`{3: 28}`）。即未解析项**全部**属"已声明、尚未训练"（计划条目随 E14 推进由 13 降到 12）；`--verify` 在 E14 收尾后必然全解析。记录在 `e18_negative/02_static_check.md` §5.3 |

| 2026-09-20 | 阶段 A 的**阶段 5 工具**落地并复跑全矩阵 | **155 ok / 256 pending / 0 fail；"阶段 A 没读 test" 155/155 成立** | 八条阶段 A 不变量原先只**手工过了前 7 个 cell**、脚本写在临时目录。新增 `scripts/phaseformer_L/audit_e14_stage_a.py`（逐格用 `e14_read_test.locate_run` 的**同一把臂指纹尺子**解析）：run 唯一可解析 / `metrics.csv` 存在 / 记了 `checkpoint` 且文件存在 / `val_mse` 可用 / **`test_mse`+`test_mae` 为空** / `1 ≤ epochs_completed ≤ 30`（**刻意不写"等于请求轮数"**：早停 `patience=8`） / `parameter_count` 非空 / `config.json` 未置 `evaluate_test`。未跑完报 PENDING 且 exit 0，已完成格违反才 exit 1。实测：`ok 155, pending 256, fail 0`，`test split read during stage A: 0 cell(s)`。**`--self-test` 5 条断言全成立**，其中"`test_mse` 被填 ⇒ fail"一条**证明"没读 test"不是空话**（真实矩阵上读出 0 是"确实没读"，不是"判据查错了列"；真实 metrics.csv 50 列里 `test_mse`/`test_mae`/`parameter_count` 都在，已核）。记录在 `e14_main/05_audit.md` |

| 2026-09-20 | **上线前一次性快照**（八道闸门同一次跑，05:34:17，E14 = 158/411） | **8/8 全绿** | ①列契约 `--strict` → rc 0；②调用元数 → 全声明、无单值多值；③消费者契约 → `C1 ✅492 cells, one schema` / `C2 ✅21/21, 0 rejected` / `C3 ✅21/21（resolved=84, 0 rejected）`、`OK: phase-2 consumers accept the live E14 manifest`；④审计器 20 类对照全符合预期；⑤阶段 A 审计 → `ok 158 / pending 253 / fail 0`，`test split read during stage A: 0 cell(s)`；⑥阶段 A 审计 `--self-test` → 5/5；⑦回填校验（产物齐备）→ `match 168 / blank 2 / PENDING 6`（`blank 2` = 两处正文占位，待回填后清零；`PENDING 6` = §4.7 两列 ρ）；⑧`--inventory` → **485**（回填前基线）。**含义**：链条启动时不存在任何已知的"闸门会误拒"或"闸门形同虚设"——四类缺陷（§17 门值判据、§18 臂数、`--json` 空转、E18 基线出处）都已修并有对照；三处未完成项（253 个待训格、两处正文占位、6 个 ρ）都是**按计划在后续阶段完成**的，不是缺陷 |

| 2026-09-20 | 排期文档自查：**旧投影缺前向指针** | 已补两处指针（不改写历史行） | 用"我后来更正过的数"反查全文，发现 §9.1 的 `12 h / 11:00 / 23:00 / 28 h` 与 §9.1.3 的 ETA 都**没有指向取代它们的 §9.1.4/§9.1.5**——读者只读 §9.1 就会拿到已被实测否掉的时刻。已在这两处各加一条**指针式**说明（原文保留），明确"E14 的 ETA 见 §9.1.4、阶段二时长见 §9.1.5，本节时刻以那两节为准"。§8 的历史日志行（含当时"11–12 臂"的结论）**不改写**——日志的价值就是"当时是怎么判断的"，最新一行已给出实测的 11/12/13 臂与 750 行

| 2026-09-20 | **判据④（覆盖反查）**：待填格是否都在校验器视野内 | 回填前**不可评估**（诚实结论）；回填后的期望已预注册 | 把 `--inventory` 的"每节空格数"与校验器报告的"每节比较格数"**反过来对**，想确认"没有任何待填格落在所有校验器视野之外"（否则那格的错值会永远 survive）。**实测发现此刻无法评估**：§4.2/§4.4/§4.5/§4.7 的产物尚未产生，校验器对每节只输出**一条 PENDING 行**（`§4.2 1 · §4.4 1 · §4.5 1 · §4.6 1 · §4.7 1`，只有已填的 §4.3 = 168 是真值）。故把**回填后的期望逐节预注册**（由各 checker 的表/列结构算出）：§4.2 280 = 28×10、§4.4 315 = 210（干预 21×10 去模型列）+ 105（解剖 21×5）、§4.5 49 = 7×7、§4.6 5（只第 5 列，替换型）、§4.7 6、§4.3 168，**合计 823 ≥ 485 空格**，逐行成立 ⇒ 判据④ = "每节应比较格数 ≥ 该节空格数，且回填后实测相符"。**顺带确认**：论文 §4.2 表头正好是 checker 断言的 10 列（`| Dataset | H | Golden MSE/MAE | … | 来源/披露 |`），故 §4.2 无未覆盖列；§4.6 的"看不见"是 §11.3 已记的盲区（25 格本就非空 ⇒ 由"无 MISMATCH"那条判据覆盖） |

| 2026-09-20 | 预检之一**一直在漏检一半调用**（已修） | `check_pipeline_invocations.py` 只吃单引号形式 ⇒ **漏掉 5 处**，含**第 2 步与四条预检自身** | 给它加新检查时发现 `flags inspected` 没变（66→66），顺查根因：`\$PY'?\s+` 只匹配 `'$PY'`（`bash -c` 体内），而顶层写 `"$PY"` ⇒ 实测 25 处调用只匹配到 **20 处**。漏检的 5 处正是 `check_column_contracts.py`/`check_pipeline_invocations.py`/`check_phase2_consumers.py`/`audit_e14_stage_a.py`（**预检自身**）与 **`e19_predictive_power.py`（第 2 步）**。**修法**：正则接受两种引号，并把尾部限制到行尾（续行已合并；否则会吞掉下一条注释产生"`--output-root` 收到 46 个裸值"的**假警报**）。修后 **66→76 flags、OK**。**四类对照**（改副本不动真文件）：①第 2 步塞假 flag → PROBLEM ✅；②预检里塞假 flag → PROBLEM ✅；③原文件 → OK ✅；④恢复 `--seeds 2021 2022 2023` → PROBLEM ✅。**并如实记下边界**（写进 `declared_flags` docstring）：只记 `nargs` 的有无 ⇒ 分不清 `store_true` 与"取一个值"，故判据是"≥2 个裸值才报"，**挡不住**单个裸值（`--verify yes`）；做成精确判据需补 `action`/`type`，**现在故意不做**（只收紧阈值会把 `--max-epochs 30` 全误报）。对照 3 我最初写错了期望，查明是该检查的**边界**而非缺陷 |
| 2026-09-20 | 阶段 A 审计接入阶段二预检 | **协议面失败即停**（新增 `--require-complete`） | watcher 只证明"411 个 run 目录 + `E14_MAIN_EXIT=0`"——**若某个 run 读过 test，这两个条件照样成立**，而"每 checkpoint 只读一次"正是论文盲测主张的支点，此前**没有任何闸门看它**。已把 `audit_e14_stage_a.py` 接进预检（`--require-complete`：预检处应已完成，未完成即 exit 1）。该工具 5/5 自检、真产物实测 `ok 158 / pending 253 / fail 0`、`test split read: 0` |

| 2026-09-20 | **§4.4 解剖表校验器读错四个列名**（本会话最严重的静默缺陷） | 修前 **63/105 格"假通过"** ⇒ 修后 `match 105`（已修） | checker 读 `leading_input_group_label` / `mean_input_group_explanation` / `mean_output_group_explanation` / `leading_correction_energy_share`——那是**原始** `dissection_table.csv` 的列名，而 `e16_writeback.build_dissection` 聚合写 44 表时**改了名**（`input_group_label` / `input_group_explanation` / `output_group_explanation` / `correction_energy_share`）。**后果比 PENDING 更坏**：`label_and_rate` 对"artifact 侧为空"是**跳过**，于是报 **match**——**从未比较却记为通过**；份额列则 PENDING。故 105 格里 **63 格没被真正验证**，而三条回填判据（inventory 0 / 无 MISMATCH / blank 0）**都看不见**。**对照（同一 fixture，仅校验器版本不同）**：修前 `match 84 / PENDING 27`、把某格 rate 由 0.75 改成 0.11 **毫无反应（MISMATCH 0）**；修后 `match 105 / PENDING 6`、扰动 → `MISMATCH 1` ✅、把产物改回原始列名 → **列 MISMATCH**（`lacks ['input_group_explanation']`）✅。**修法三处**：①四个真名；②**加"列缺失即 MISMATCH"守卫**（根因级：将来任何改名都不会再退化成静默通过）；③`label_and_rate` 在 artifact 侧 label 与 rate **皆空**时报 MISMATCH 而非跳过。**顺带固化** `rehearse_minipaper_fill.py`（AST 取生产者列名 + 按规定格式填论文临时副本）：同时钉住**列名契约**与**§4.4 填写格式**，4/4 断言成立。**这次 before/after 是被"我忘了同步"意外成全的**：首跑服务器仍是旧版，于是原样复现缺陷；sync 后重跑才得到右列 ⇒ 再次印证"验证前先确认服务器版本就是我改的那版" |

| 2026-09-20 | 收官判据缺一条：**产物缺失时可全绿**（已补判据⑤） | 发现一条**真实的假阳性通道** | 审计 `check_builder_outputs.py`（唯一逐步报告"整列为空"的工具）时发现：它对**缺失**的 CSV 会 `exit 1`，但第 3/4/6 步带 `|| true` ⇒ **被掩盖**；第 7 步审计器对**缺失产物**报 `PENDING` 而非 `FAIL`（实测产物全缺时 `PASS=3, PENDING=22`、**exit 0**）。于是判据①（inventory 0）是唯一会发现的——**但对 §4.6 无效**（其 25 格本就非空=计划文字）；判据②（无 MISMATCH）在产物缺失时只得到 `PENDING`（artifact value empty）⇒ **也不报错**。⇒ 若某个 write-back 失败（被 `|| true` 掩盖），链条仍打印 `PHASE2_OK`，而 §4.6 的表**从未产生**、三条判据全不报错。**补判据⑤**：收官时跑 `audit_phase2_outputs.py` 并要求 `PENDING` = 0（其 `PENDING` 唯一来源就是"产物不存在"）。**五条收官判据**：①inventory 0；②无 MISMATCH；③blank 0；④每节比较格数 ≥ 空格数；⑤审计器 PENDING 0 |

| 2026-09-20 | 改动后的**回归扫全闸门**（05:48:11，E14 = 166/411） | **9/9 全绿** | 本轮连续改了验证层（调用检查器正则 §21、解剖/overlap/cosine 静默跳过 §22.1、收官判据⑤ §23、阶段 A 审计接入预检），故把**所有**闸门再跑一遍确认没有连带破坏：①列契约 rc=0；②调用元数 OK（`flags inspected` 由 66 升到 **76**）；③消费者契约 `OK`；④审计器 20/20；⑤阶段 A 审计 `ok 166 / fail 0` + `--self-test` 5/5；⑥回填预演 7/7；⑦回填校验 `match 168 / blank 2 / PENDING 6`；⑧审计器 `PASS=3, PENDING=22`（此刻应非零——产物未产生）；⑨E14 回填预演 `PASS`（24+4 结构 + 退化输入存活）。另：全量 `tests/` 在同一批改动后 **400 passed**。**这是"改完验证层要自己再验一遍"的固定动作** |

| 2026-09-20 | 编排器**逐行复核**（无人值守那段代码） | 未发现缺陷：`run_step` 是 fail-closed，且日志可核 | 此前只通过**输出**观察过 `run_phase2_after_e14.sh`，本轮把它读了一遍。`run_step`：`--only` 不匹配即跳过、`n < FROM` 即跳过；每步先打印 `=== step n: name` 与**当时的 HEAD**（便于事后确认"这一步跑的是哪版代码"）；命令输出重定向到 `logs/phase2_step${n}.log`；捕获 `rc` 并打印，**`rc≠0` 时打印日志尾部并以该 rc 退出**（⇒ 后续阶段不会在坏输入上启动）；成功则打印末尾 3 行。脚本用 `set -uo pipefail` 而**刻意不用 `-e`**——失败由 `run_step` 显式处理，预检块则各自 `|| exit 1`（已核）。`guard_e14_done` 三道判据：无 `search_phaseformer.py` 进程、`e14_main.log` 含 `E14_MAIN_EXIT=0`、`runs/` 恰 411 个目录——与 watcher 的两个条件**相互独立**（双保险）。`--list` 由脚本自身的用法注释 `sed` 抽取（实测 7 步）。**结论：无人值守路径的"失败即停 + 可事后核对"成立**；本轮未改动它（改动风险大于收益） |

| 2026-09-20 | 完成 run 的**统计体检**（E14 仍在跑，n=176） | **未发现异常**；两项独立结构证据 | 在阶段二消费这些数字之前先体检已完成 run。`epochs_completed` 13–30、中位 **18**（**133 早停 / 43 跑满 30**，即 24%）；`val_mse` 量级合理（Traffic 0.337、Electricity 0.122、Weather 0.518、ETTm1 0.937，均无 0/NaN）；**同 setting 跨 3 seed 的 `parameter_count` 0 处不一致**（⇒ "run 目录与 cell 正确配对"的强旁证，与 `locate_run` 的指纹匹配**相互独立**）；**0 行已带 test 指标**（⇒ 与 `audit_e14_stage_a.py` 独立地再次确认"阶段 A 没读 test"）。**顺带更正我一处过强措辞**：§9.1.4 说 `COST_HINT` 高估"因早停主导"——实测中位 18 轮、24% 跑满，故"解释大部分高估"成立但"主导"过强，以实测为准 |

| 2026-09-20 | 主张 A 的**配对方向早期信号**（val-only，非判定） | 方向正常：中位 **−0.70%**、17/27 落在 ±1% 内 | 在阶段二花十小时之前先排除"两臂接错/配对错"这类**系统性接线错误**（那会表现为整体单向偏移）。对**两臂都已完成**的 27 个 (setting, seed) 做配对：中位 Δ **−0.70%**（略优于 `phase_only`）、范围 −1.90%～**+2.29%**、abs≤1% 者 **17/27**、差于 1% 以上者 **1/27**、优于 1% 以上者 9/27 ⇒ **无接线错误迹象**。**明确不是判定**：主张 A 用 test 指标 + 两项指标 + 冻结 1% 界 + 按 setting 聚合，正式裁决只由 `e14_writeback.claims.json` 与第 7 步审计给出；其中"1/27 差于 1%"是最需盯的一项，若 test 上仍如此则按 §4.0 **如实报告未达标并逐格列出**，不得解释掉 |

| 2026-09-20 | 行 5 的**预注册上界**只有文本形态（记下待办，不代为实现） | 阶段 5 必须显式对照，否则预注册从未被检验 | 核对 §4.6 行 5 的"预注册预期"落地情况：paper 写的是一个**定量上界**（"退化幅度上界由 `1−capture(1/2)`（14%–35% / 3%–18% 的支路价值）**经 `g²` 折算**"），而 `e18_writeback.build_row5` **只把它当文本记录**（`preregistered_expectation`），另算实测 `mean_delta_mse_pct` 与 `cells_degraded_vs_dense`，**不算上界、也不对照** ⇒ 若不处理，预注册会停留在"写了但从未被检验"。**已写进 `e18_negative/01_plan.md` 第 7bis 条**：阶段 5 必须把实测与上界显式并列（逐 setting 或按均值），或写明为何不逐格对照。**刻意不代为实现该公式**：`g²` 折算是 paper 的**文字**表述，由我实现成算式等于替研究决定一个解释——那属于研究判断而非工具修正；若日后要自动化，应先冻结公式定义、再配正负对照 |

| 2026-09-20 | 回填源与论文表的**列数/列序对齐**（读产出者格式串实测） | 四个"直接复制"源全部对齐；唯一位移是 §4.4 干预表的"去第 1 列" | 直接复制类回填的可行性取决于"源 md 行的单元格数 = 论文列数"——不等就会整体错位（虽会被校验器判 MISMATCH，但那时已耗掉一轮回填）。实测：§4.2 `main_table.md` 格式串 **10** 个单元格 = 论文 10 列 ✅；§4.4 `intervention_table_44.md` **11** 个 = 论文 10 列 **+1** ⇒ 按 §1.5「去掉第 1 列」得 10，**该规则与格式串实测一致、不是猜的** ✅；§4.5 `conditional_table.md` **7** = 论文 7 ✅；§4.6 `negative_table.md` **5** = 论文 5 且只替换第 5 列 ✅。记录于 `minipaper_fill_mapping.md` §2.1 |

| 2026-09-20 | 六节"哪一列已非空"的**机械普查**：覆盖式丢文字的风险**完全收口** | 除 §4.6 与 §4.2 的 4 个 Traffic 披露格外，**结果列全空** | §4.6 是**替换型**（25 格本来就装计划文字）——该风险早已处理（文字已移到表下注）；§4.2 有 **4 格**非空（Traffic 的 `来源/披露` = "探索性附录，不进入判定"）——已**在源码处**核实：`e14_writeback.py:414` 把同一句作为 `provenance_note` 的**第一个元素**追加 ⇒ 论文文字是产物 note 的**前缀** ⇒ 整行替换后**保留**该披露。其余各表（§4.2 主表六列 168 格、§4.4 解剖 105 格 + 干预 147 格、§4.5 35 格、§4.7 两列 ρ 6 格）的**结果列当前全部为空**，非空列一律是键列或设计文字列 ⇒ **没有任何待填格会被覆盖**。记录于 `minipaper_fill_mapping.md` §2.1（含一次我自己的计数瑕疵：脚本把分隔行当数据行、每列多计 1，已更正） |

## 9. 阶段二工期投影（基于**实测**，而非外推）

### 9.1 各步的实测/推导依据

| 步 | 内容 | 估计 | 依据 |
|---|---|---|---|
| 1 | 411 个新格的单次 test 读取 | **约 1.3 h** | 实测单格（Electricity-192 全 test 分裂）约 1–2 min；411 / 8 卡 |
| 2–3 | §4.7 ρ、参数量表、复用审计、§4.2 回填 | 分钟级 | `e14_params` 在 175 个真实 checkpoint 上为秒级；其余为纯 IO |
| 4 | E16 解剖 + 干预（63 cell） | **3–5 h** | 实测：**稠密格 551.9 s**（CPU、`--max-batches 2`），低秩格 3.3 s；21 个稠密格 ≈ 3.2 h。上界来自"全划分比 max-batches 2 更贵"，下界来自"实测为 CPU 而流水线跑 GPU 0" |
| 5 | E17 训练 24 run + 装配 + test 读取 | **约 1.2 h** | E14 实测单 run 中位 ~20 min（同协议）；24 / 8 卡 |
| 6 | E18 训练 78 run + test 读取 + 行 3（28 setting 的 SVD 评估） | **约 4.5 h** | 78 / 8 卡 × ~20 min ≈ 3.3 h；行 3 的 28×3 次截断评估另计 |
| 7 | 验收审计 | 秒级 | — |

**合计约 12 h（E14 之后）**。若 E14 按当前速率在 **11:00 前后**结束，
则阶段二约在 **23:00 前后**收尾，整条链（含 E14）约 28 h。

> **本节已被取代，保留原文仅供对照**：表中的 12 h 用的是 E14 的**全局中位 run 时长**作代理量，
> 而第 5/6 步训练的是 7 个复用 setting（无 Traffic），代理量偏高。
> **E14 的 ETA 见 §9.1.4**（逐 setting 实测剩余成本，≈06:38 结束）；
> **阶段二的时长见 §9.1.5**（逐 setting 实测重估，≈6.7–9.7 h）。本节所有时刻请以那两节为准。

### 9.1.1 两项实测补充（本轮）

**（a）第 1 步的"解析开销"在满规模下可忽略**——这是先前的一个真实疑问：单格冒烟时
`<output-root>/runs` 下只有 1 个 run，而正式运行时该目录有 411 个，
若解析是"每个 cell 扫一遍全部 run"，成本可能随时钟规模平方增长、把 1.3 h 变成数小时。
**实测**：对**真实根**（492 cell、121 个 run 已存在）跑 `--dry-run`（只解析、不读 test）：

```text
elapsed 23.75 s   maxRSS 18224 KB
```

即 **492 个 cell 全部解析一遍仅 23.75 s（≈0.05 s/cell）**，内存 18 MB。
故第 1 步的耗时由**真实的 test 读取**主导，1.3 h 的估计成立，
"解析可能在满规模下变成瓶颈"这一风险**被实测排除**。

> 附带纠正一处理解：此前观察到"每个 `missing_run` cell 约耗 15 s"，
> 那是**非 dry-run** 模式下 dispatch 循环的 `--poll-seconds 15` 节拍，
> **不是**解析成本（dry-run 跳过 dispatch，故 492 格只用 23.75 s）。

**（b）在飞 run 的存活核验**：第 35 轮末对 8 个在飞 run 逐个检查
"最近文件 mtime + 最近记录的 epoch"，结果 8 个 run 的 mtime 均为**当时那一秒**、
epoch 在 3–26 之间正常推进——即**没有卡死的 run**。
这类"看起来在跑但实际挂住"的失败不会产生日志错误，只会让整条链永远等下去
（watcher 的 24 h 上限才会兜住），故值得在长周期等待中定期抽查。

### 9.1.2 E14 剩余工期的两种投影，以及**朴素外推为何会错 4 倍**（2026-09-20 04:26 实测）

E14 日志的 `done` 事件**不带时间戳**，故完成率只能从文件系统取：
每个 run 完成时写 `metrics.csv`，其 mtime 即完成时刻。实测：

```text
completed runs with metrics: 115 / 411
  last  15 min:   3 -> 0.20/min
  last  30 min:  10 -> 0.33/min
  last  60 min:  21 -> 0.35/min
  last 120 min:  44 -> 0.37/min
  overall: 115 completions over 519 min = 0.22/min
  newest completion: 04:26:41 (0.1 min ago)
```

**投影 A（朴素）**：按近 60 分钟的 0.35/min 外推剩 296 个 run → **846 min = 14.1 h → ETA 18:32**。

**投影 B（按成本，推荐）**：先算出"还剩什么"。由**真实 manifest** 逐数据集统计新训需求，
再减去已完成者：

| 数据集 | 需新训 | 已完成 | 待完成 | 单 run 成本依据 | 待完成 GPU·h |
|---|---:|---:|---:|---|---:|
| Traffic | 60 | 60 | **0** | 实测 4096/2972/2585/2640 s | 0 |
| Electricity | 63 | 55 | **8** | 实测 ~1285–1717 s | 2.9 |
| Weather | 48 | 0 | **48** | `COST_HINT` × 0.7（已知偏高 30–40%）≈ 680 s 均 | 9.1 |
| ETTm1 | 72 | 0 | **72** | 实测 253/244 s，h336/720 外推 | 5.7 |
| ETTm2 | 48 | 0 | **48** | 实测 185/114 s，h336/720 外推 | 2.8 |
| ETTh1 | 72 | 0 | **72** | 实测 85/74 s，h336/720 外推 | 1.9 |
| ETTh2 | 48 | 0 | **48** | 实测 47/37/57 s | 0.7 |
| **合计** | **411** | **115** | **296** | | **≈ 23.1 GPU·h** |

23.1 GPU·h ÷ 8 卡 = **约 2.9 h → ETA 约 07:20**。

**为什么两者差 4 倍——而且朴素的那个是错的**：调度器按**成本降序**发车，
所以"当前速率"反映的是**最贵的那一批**（Traffic 实测 0.7–1.1 h/run、Electricity 约 21 min/run），
而**剩下的是最便宜的一批**（ETTh1/ETTh2 单 run 仅 37–85 s，比 Electricity 便宜 1–2 个数量级）。
把最贵批次的速率外推到最便宜批次，必然严重高估。
这也解释了为什么"跑了 8.6 h 才完成 115 个"看起来很慢：那 8.6 h 花在了 60 个 Traffic + 55 个 Electricity 上。

**两次独立交叉验证**：投影 B 的"待完成 296"与实测脚本报的 `remaining 296` **完全一致**；
而 §9.1.1(a) 又已实测排除"解析在满规模下变慢"。故 296 这个数是可信的。

**投影 B 的最大不确定性**：Weather 一项占 9.1/23.1 ≈ **39%**，而它的成本用的是
`COST_HINT × 0.7`（**外推**，不是本数据集实测）。若 Weather 与 Electricity 一样被高估，
总时长会更短；故区间可写为 **约 2.5–4 h → ETA 07:00–08:20**。

> **本节的 ETA 已被 §9.1.4（E14，≈06:38）与 §9.1.5（阶段二，≈6.7–9.7 h）取代。**

**对全链路的影响**：E14 若在 07:20 前后收尾，叠加 §9.1 的阶段二投影（约 12 h），
整条链约在 **19:00–20:00** 收尾，早于先前按"E14 到 11:00"给出的估计。

### 9.1.3 实测刷新：Weather 比外推的**更便宜**，ETA 再提前（2026-09-20 04:41）

从已完成 run 的 `metrics.csv:elapsed_sec` 取实测中位数（**不再是外推**）：

| dataset / horizon | n | 实测中位 | 先前依据 |
|---|---:|---:|---|
| Traffic h96 / h192 / h336 / h720 | 15 / 15 / 15 / 15 | 4096 / 2972 / 2585 / 2640 s | 同（实测） |
| Electricity h96 | 16 | 1481 s | — |
| Electricity h192 | 18 | **848 s** | 注意：比 h336/h720 都便宜，此前用"Electricity ≈1285 s"统一估是偏高的 |
| Electricity h336 / h720 | 9 / 18 | 1589 / 1285 s | 同（实测） |
| **Weather h720** | **2** | **449 s** | 先前 `COST_HINT × 0.7` ≈ **840 s** |

**关键点**：`COST_HINT` 对 Weather h720 的估值（1200 s）明显偏高——实测 449 s，比值约 **0.37**，
比我先前统一采用的 0.7 折扣还要低得多。按此比例折算 Weather 其余 horizon（h96/h192/h336）
后，剩余工作量的构成变为：

| 剩余项 | 估法 | GPU·h |
|---|---|---:|
| Weather（余 46） | 实测 h720 外推其余 horizon（~380 s 均） | ≈4.9 |
| ETTm1（72） | `COST_HINT × 0.7`（尚无实测） | ≈5.0 |
| ETTm2（48） | 同上 | ≈2.0 |
| ETTh1（72） | 同上 | ≈2.0 |
| ETTh2（48） | 同上 | ≈0.8 |
| Electricity（余 ~2） | 实测 | ≈0.6 |
| **合计** | | **≈15.3 GPU·h → 1.9 h → ETA ≈ 06:35** |

**诚实的不确定性**：Weather 的样本只有 **n=2**，且恰是 hint 最贵的 horizon（h720），
故"0.37×"这个比例**证据偏弱**；ETT 四个数据集**尚无任何实测**。
故仍以区间表述：**约 1.8–3.5 h → ETA 约 06:30–08:10**（较 §9.1.2 的 07:00–08:20 略提前）。

### 9.1.4 实测刷新（2026-09-20 05:05）：137/411 完成，**ETA ≈ 06:38**，全链 ≈ 09:08

用 **E14 自己的匹配器**（`e14_read_test.locate_run`，即阶段 B 判定"这一格跑完了吗"的同一把尺子）
逐格核过 411 个 new 格：**137 已完成、273 待跑**，在飞 7–8 个，**0 重试 / 0 失败**。

剩余成本的估法：有实测的 setting 用实测中位数；没有的用 `COST_HINT × r`，
其中 **r = 0.642** 是 9 个已实测 setting 的 `elapsed/COST_HINT` 中位数（ETTh1/ETTm1 无 hint，
镜像 ETTh2/ETTm2 家族）：

| 已完成 setting | n | 实测中位 elapsed | COST_HINT | 比值 |
|---|---|---|---|---|
| Traffic-96 | 15 | 4096 s | 4200 | 0.975 |
| Traffic-192 | 15 | 2972 s | 4200 | 0.708 |
| Traffic-336 | 15 | 2585 s | 4200 | 0.615 |
| Traffic-720 | 15 | 2640 s | 4200 | 0.629 |
| Electricity-96 | 18 | 1472 s | 1500 | 0.981 |
| Electricity-192 | 18 | 848 s | 1600 | 0.530 |
| Electricity-336 | 9 | 1589 s | 1717 | 0.925 |
| Electricity-720 | 18 | 1285 s | 2000 | 0.642 |
| Weather-720 | 15 | 674 s | 1200 | 0.561 |

剩余 **12.4 GPU·h → 1.55 h 墙钟（8 卡）**，**E14 约 06:38 结束**；与 §9.1.3 的投影（06:30–08:10）一致。
> **05:53 刷新**（同一逐 setting 实测法，剩余集合已变成以 ETT 尾巴为主）：剩余 **7.3 GPU·h → 0.91 h 墙钟 ⇒ ETA 约 06:47**；接 §9.1.5 的阶段二（6.7–9.7 h）⇒ 全链约 **13:30–16:30**。

> **本节初稿有一处错，当天即改正**：初稿写"接上阶段二 ~2–3 h ⇒ 全链约 09:08"。
> 那个 2–3 h **不是** §9.1 的投影，而是我 ETA 脚本里写死的假设值，我把它当成"实测投影"抄进了文档。
> §9.1 的投影是**约 12 h**，而下面 §9.1.5 用实测数据重估后是 **约 6.5–9.5 h**。
> 三个数字（2–3 h / 12 h / 6.5–9.5 h）里，**只有 §9.1.5 那个有逐 setting 的实测依据**。
> 教训：ETA 里的每一段都要能指到它的依据，否则假设值会伪装成测量值。

> **又一次"朴素外推会错"的实例（与 §9.1.2 同类，记下来是因为我差点又信了它）**：
> 我先用"最近 1 h 完成 28 个 run"直接外推 → 273/28 ≈ **9.8 h**，比真实值（1.55 h）**大 6 倍**。
> 原因与 §9.1.2 完全相同：**剩余格子的构成在变**——已跑掉的是 Traffic/Electricity/Weather 这些贵 setting，
> 剩下的尾巴是 ETT 系列（ETTh2-96 = 30 s、ETTh2-192 = 24 s、ETTm2-336 = 96 s…），
> 按"当前速率"外推等于用最贵阶段的速率去套最便宜的尾巴。
> **正确做法是按 setting 逐类估成本**（本节即如此），而不是按 run/h 单点外推。
> 另记一处我自己的**检查**错误：第一版脚本按 `(dataset, horizon, seed)` 去重"已完成格"，
> 把 137 个 run 认成 27 个格——**格的身份是 `(arm, dataset, horizon, seed)`**，
> 同一 setting-seed 有 5 个臂在训。改用 E14 自己的 `locate_run` 后数字才对上（137）。

### 9.1.5 用**实测 duration 逐 setting 重估**第 5、6 步：§9.1 的 12 h 偏保守（2026-09-20 05:2x）

§9.1 估第 5 步"24/8 卡 × ~20 min ≈ 1.2 h"、第 6 步"78/8 卡 × ~20 min ≈ 3.3 h"，用的是
**E14 全局的单 run 中位 ~20 min**。这个代理量是错的——**E18/E17 训练的是 7 个复用 setting
（外加 rank12 的 6 个 ETT/Weather setting），其中根本没有 Traffic**，而全局中位被 Traffic
（2585–4096 s/run）与 Electricity 拉高了。又是"构成在变"这一类错误（§9.1.2、§9.1.4 同源）。

改用**同 setting 的真实 run 实测 `elapsed_sec`** 逐格估（15 个 setting 有实测；E18 平滑档的
42 格 = 7 setting × 3 seed × 2 档，rank12 的 36 格 = 6 setting × 3 seed × 2 档；E17 的格 = 7 setting × 3 seed）：

| 步 | 内容 | §9.1 旧估 | **实测重估** | 依据 |
|---|---|---|---|---|
| 5 | E17 24 个新训 run | 1.2 h | **≈ 0.6 h** | 21 格（7 setting × 3 seed）实测中位之和 3.07 GPU·h + 3 个 `frozen_independent` 新格（Electricity-336，E8 未覆盖，实测中位 1672 s → 1.39 GPU·h），合计 4.46 / 8 |
| 6 | E18 78 个新训 run | 3.3 h | **≈ 1.2 h** | 平滑 42 格 6.15 GPU·h + rank12 36 格 3.36 GPU·h → /8 |
| 6 | 行 3（28 setting 的 SVD 截断评估，仅 val） | 另计 | **≈ 1.5–2.5 h**（2026-09-20 更正） | 原估 0.2–0.5 h 犯了两个错：①**按 1 个 seed 估**，而流水线传 `--seeds 2021,2022,2023`（E18 计划第 7 条写明其**默认是单 seed**、扩到 3 seed 会把 validation 前向次数**乘 3**）⇒ 实际 28 setting × **3 seed** × （1 全秩 + 1 个秩）= **84 条记录**；②**以为能并行**，而 `e18_svd_truncation.py` **没有 `--gpus`**、只有 `--device` ⇒ **在单卡上顺序跑**。按每条记录「加载 checkpoint + 建模型 + 两次 val 前向（全秩与截断）」约 40–80 s 计 ⇒ ≈1.5–2.5 h（仍属**估算**，跑完以实测为准） |
| 1 | 411 次 test 读取 | 1.3 h | **≈ 1.3 h**（不变） | 单格实测 1–2 min，且满规模解析已实测仅 23.75 s（§9.1.1a） |
| 4 | E16 63 格解剖 + 干预 | 3–5 h | **3–5 h（不变，仍是最贵的一步）** | 稠密格实测 551.9 s（CPU、`--max-batches 2`） |
| 2–3, 7 | ρ 列 / 参数表 / 复用审计 / 回填 / 验收审计 | 分钟级 | 分钟级 | 纯 IO 与秒级 checkpoint 读取 |

**阶段二合计 ≈ 8.6–10.1 h**（2026-09-20 更正：原写 6.7–9.7 h，因第 6 步行 3 的成本被低估，见上行）——区间宽度几乎全部来自第 4 步 E16（3–5 h）与行 3（1.5–2.5 h），**而非 §9.1 的 12 h**。
配合 §9.1.4 的 E14 ETA（约 06:38）⇒ **全链约 13:00–16:00 收尾**。

**这条估算的诚实边界**（不写清楚就会变成下一个"假设值伪装成测量值"）：

* E17/E18 的新格**不是**被测量的那些 run：E18 平滑档改 `smooth_ratio`、rank12 用 `pooled_lowrank`
  头（参数更少）、E17 用冻结子空间投影。同 setting 同规模的实测时长是**代理量**，
  不是这批格子的实测时长；真正的数字要等它们跑完。
* E16 的 3–5 h 区间来自"实测在 CPU 上、流水线跑 GPU 0"的换算，未在新硬件配置上复测。
* 低估的风险在**评估**而不在训练：test 读取与行 3 都是"模型构建 + 一遍数据"，若某个 setting
  的 test 分裂远大于 val，第 1 步会比 1.3 h 长。第 1 步是唯一"读错了就浪费"的一步（协议只允许读一次），
  故它的时长不确定性只能靠**跑完后的实测**收敛，不能靠外推。

### 9.2 一处**不采用**的排期优化（记下供决策）

核对依赖后发现：**第 4、5、6 步彼此独立**（E16 只依赖 E14 的 checkpoint；
E17 只依赖冻结投影器；E18 只依赖 E14 checkpoint 与自身配置），
而流水线是**串行**的。由于 E16 是**单进程**（`--gpus` 只取第一个索引），
它的 3–5 h 里另外 **7 张卡处于空闲**。理论上把 E16 与 E17 重叠可省下约 2–4 h。

**本轮不实施**，理由：

1. 它需要给无人值守的链路加入并发与"两个都成功才算过"的失败语义，
   而当前链路的价值恰恰在于**简单**：任一非零退出即停、逐步日志、`--from/--only` 可重入；
2. E16 **没有 resume/跳过逻辑**（产物末尾一次性写出，已核实），
   所以"先单独跑 E16"会被第 4 步**重复跑一遍**，净收益为零、还多占一张卡（该结论已在
   `e16_dissection/02_03_static_check_smoke.md` §4.1 记录过一次）；
3. 收益是**墙钟时间**（约 2–4 h），代价是**改动一个已校准的无人值守关卡**——
   按本会话的既定纪律（不为边际收益改动已验证的门禁），不做。

**若日后要提速**，正确的做法是给 E16 加按 cell 的断点续跑，再让第 4 步复用已完成的产物；
那是"先做对再提速"的顺序，而不是并发化一条尚未跑完的链。

---

## 10. 收尾作业清单（E14 收尾 → 全链结束）

> 目的：把"接下来要做什么、用什么查、填哪张表"写成可执行清单，
> 使收尾阶段**不靠记忆**，每步都有对应的机器检查。
> 本节的每条路径/判据都取自本轮之前已核验的产物与工具。

### 10.1 阶段二会自动开始；先确认它**确实**开始了

watcher（`watch_e14_then_phase2.sh`）在 E14 干净结束时（`E14_MAIN_EXIT=0` **且** `runs/` 恰 411 个目录）
自动启动阶段二，两者条件独立校验。确认方式：

```bash
cat ~/niuyiming/logs/phase2_watch.status     # 应从 "waiting" 变为 "E14 complete ...; starting phase 2"
tail -20 ~/niuyiming/logs/phase2.log          # 每步的 === step N: ... / --- step N exit=0
ls ~/niuyiming/logs/phase2_step*.log          # 逐步日志
pgrep -af run_phase2_after_e14.sh             # 仍在跑？结束则看 status
```

**若某步失败**：`run_step` 在第一个非零退出处停止并打印该步日志尾部；
status 会写 `PHASE2_FAILED rc=N`。重入方式（不需要重跑已成功的步骤）：

```bash
bash scripts/phaseformer_L/run_phase2_after_e14.sh --from N   # 从第 N 步接着跑
bash scripts/phaseformer_L/run_phase2_after_e14.sh --only N   # 只跑某一步
```

重入安全性：①`--from >1` 会跳过 E14 完成守卫；②第 1 步的读取器**幂等**
（已带 test 指标的行走"原样复制"，除非显式传 `--all-rows`）；③每步的预检
（列名契约、实参元数、**消费者契约**）只在 `--from 1` 时执行——**若改动过脚本，先手动跑一次三道预检**：

```bash
"$PY" scripts/phaseformer_L/check_column_contracts.py --strict
"$PY" scripts/phaseformer_L/check_pipeline_invocations.py
"$PY" scripts/phaseformer_L/check_phase2_consumers.py --e14-root research_runs/phaseformer_L_e14_main_v1
```

第三道是 2026-09-20 新增的（详见 `paper_code_consistency.md` §16）：**直接调用阶段二消费者自己的 loader**
对 E14 真 manifest 跑一遍。它挡的是"**静默降级**"——消费者读不进产物时步骤仍 exit 0、表照样写，只是**列是空的**。
E18 的基线出处列曾会因此 **78/78 行全空**（reused 格没有 command），而该缺陷没有任何既有判据能看见。

### 10.1.1 已核验的失败语义：整条链**失败即停（fail-closed）**

无人值守链路最危险的失效模式是"**带着空洞继续往下走**"——链跑完了、表填出来了，
但某些格其实是缺的。本轮把这条语义**逐层核到源码**：

| 层 | 行为 | 位置 |
|---|---|---|
| 训练 runner | **任一格失败即非零退出** | `e14_main_matrix.py:702-703` `if failed: raise SystemExit(...)`；`e17_conditional.py:1407`；`e18_negative.py:890-891` 同构 |
| 训练 runner 的前置门 | 复用解析失败 / 预检失败即拒绝训练 | `e14_main_matrix.py:655`、`e17_conditional.py:1281` |
| 流水线 | `run_step` 在第一个非零退出处停止并打印该步日志尾部 | `run_phase2_after_e14.sh` 的 `run_step` |
| watcher | 记录 `PHASE2_FAILED rc=N` 到 `phase2_watch.status` | `watch_e14_then_phase2.sh` |
| E14 完成守卫 | 要求 `E14_MAIN_EXIT=0` **且** `runs/` 恰 411 个目录 | 同上 |
| 阶段二 pre-flight | 三道静态检查任一失败即**拒绝启动整条链**：列名契约、调用实参元数、**消费者契约**（C1/C2/C3 失败即停；C4/C5 仅信息） | `run_phase2_after_e14.sh` pre-flight 块；`check_phase2_consumers.py` |

即**从 runner 到 watcher 全链 fail-closed**：E14 若有失败格，`E14_MAIN_EXIT` 不会是 0，
watcher **不会**启动阶段二 ✓。

**一个已知且已记录的例外**：`read_test_generic.py`（步骤 5/6 的 test 读取）**永远 exit 0**，
即使有 cell 被拒或 worker 崩溃——它只在汇总里列 `rejected`（`e14_main/05_audit.md` §16 附近有记录）。
该例外**由第 7 步验收审计兜住**（E17/E18 各有一条"每个 cell 都要有 test 指标"的判据会判 FAIL），
且它的代价有限：真出问题时只需**重跑该步的读取与回填**（24/78 个训练 run 仍然有效），
不必重训。故**保留其 exit-0 语义不改**，但这条例外必须写在文档里，不能默认"全链都 fail-closed"。

### 10.2 逐步：期待什么产物 → 用什么查 → 填哪张表

| 步 | 主要产物 | 机器检查 | 对应回填 |
|---|---|---|---|
| 1 | `e14_main_v1/results.csv`（492 行）、`test_read_summary.json`、`test_read/` | 管道内 awk 守卫（每行 `test_mse` 非空）+ `test_read_summary.json` 的 `problems` 与 `by_status` | — |
| 2 | `e19_predictive_v1/predictive_power.csv`、`predictive_power_summary.json` | `check_builder_outputs`（空列）| **§4.7**：3 行 × 2 个 ρ ← `predictive_power.spearman[…]["rho"]`（映射已实测核验） |
| 3 | `e14_params`→`parameter_table.csv`；`e14_writeback`→`main_table.csv/.md`、`variant_table.csv`、`claims.json`、`audit.json` | `check_builder_outputs`（两次）；`audit.json.settings_incomplete` 应为空；`claims.json` 含 A–D 与两个必答块 | **§4.2 主表**：28 行 ← `main_table.md`，填完立即跑 `verify_minipaper_fill.py`（该类已五类校准） |
| 4 | `e16_dissection_v1/{dissection,intervention}_table.csv`、`e16_summary.json`、`reference_parity.json`；`e16_writeback`→`*_44.csv`、`intervention_table_44.md` | **§6.2 判据表**：cells 63、`algebra_failures`/`run_metric_failures`/`run_metric_not_comparable` 均 0、`reference_parity_passed` true、`checkpoint_path_mismatches` []、`test_split_read` false、`probe_cells ≈ 72` | **§4.4**：干预表 147 格 ← `intervention_table_44.md` 去掉首格；**解剖表 105 格需组合**，先打印真实取值定写法（见 §10.3） |
| 5 | `e17_conditional_v1/results.csv` → `results.with_test.csv` → `conditional_table.csv/.md` | 24 个新 cell 均有 test 指标；§4.5 表 7 行且 cos 列非空 | **§4.5**：35 格 ← `conditional_table.md`（跳过其表头两行） |
| 6 | `e18_negative_v1/results.csv` → `results.with_test.csv`；`svd_truncation_table_28.csv`；`negative_table.csv/.md` | `e18_negative_verify.json` 与 `e18_svd_truncation_summary.json` 的 `problems` 为空 | **§4.6**：用 `negative_table.md` 的 5 行**替换**（25 格）；计划/规模文字建议保留为表下注 |
| 7 | `phase2_acceptance_audit.json` | **审计器**：不得有 `FAIL`；**且收官时 `PENDING` 必须为 0**（判据⑤，见 §10.4——该 `PENDING` 的唯一来源是「产物不存在」，故它非零意味着某张表从未产生；假阳性通道记于审计文档 §23） | — |

### 10.3 §4.4 解剖表的单元格写法：**已收口，不再是开放项**（2026-09-20）

本节原先写着"唯一开放项、收口时点 = 第 4 步跑完之后"，因为"只有看到取值分布才知道解释率是 0–1 还是百分数"。
**这个问题已从另一条路解决**：不问取值的**观感**，而是从**生产者**读单元格的构造，再用**对照**把写法钉住。
当前写法与依据：

| 论文列 | 产物列 | 渲染 | 依据 |
|---|---|---|---|
| 主模式输入组 / 解释率 | `input_group_label` + `input_group_explanation` | `标签 / 两位小数` | 表头自己写作 `组 / 解释率`（斜杠即约定）；生产者判据 `input_explanation >= 0.5` 说明是 **0–1 小数**而非百分数 |
| 主模式输出组 / 解释率 | `output_group_label` + `output_group_explanation` | 同上 | 同上 |
| 修正能量份额 | `correction_energy_share` | 三位小数 | 与论文其余能量份额列一致 |
| 跨 seed `leading4` 重叠 | `leading4_input_overlap` + `leading4_output_overlap` | `in / out` 各两位小数 | 论文一格、产物两列 |
| 稳定语义判定 | `stable_semantics_verdict` | ✓ / ✗ | 产物为 bool |

**这两件事都由对照钉住，不再依赖人眼**：`rehearse_minipaper_fill.py` 按规定格式填一份论文临时副本 →
**7/7 断言成立**，其中"把某个 rate 由 0.75 改成 0.11 ⇒ MISMATCH"与"产物改回原始列名 ⇒ **列缺失 MISMATCH**"
正是写法与列名契约的对照（后者对应审计文档 §22 修掉的"105 格里 63 格假通过"）。

**若真实取值与上述约定不符**（例如某格解释率确实 > 1），那属于**产物与设计不符**，应按缺陷处理，
**而不是**改写法去迁就它。

### 10.4 回填完成后：**五条**机器判据 + 一轮人工审校

```bash
python scripts/phaseformer_L/verify_minipaper_fill.py             # 逐格一致；有 MISMATCH 则 exit 1
python scripts/phaseformer_L/verify_minipaper_fill.py --inventory  # ①空格总数应为 0（当前 485）
# ②无 MISMATCH（同上第一条命令的退出码）③blank 计数为 0（连同下面的正文占位）
```

**判据③是 2026-09-20 新增的**：填格**不会**碰到格子周围的句子。全文有两处 `待填` 出现在**正文**里——
摘要的 `*[主结果待填。]*`（第 58 行）与 §4 开头的填表状态段（第 369 行，写着"§4.2、§4.4、§4.5、§4.6 待填"）。
不管它们，**填满的表会被一段自称"主结果待填"的文字包着**，而逐格比对**完全看不见**这类矛盾。
`verify_minipaper_fill.py` 新增 `check_placeholders()` 把每处 `待填` 报成 `blank`（**报告而不失败**，
故 end-state 判据是"blank 计数 = 0"）。**双向校准**：现稿报 2（第 58、369 行）、把两处替换后报 0 且 exit 0；
服务器上（产物齐备）为 `match: 168, blank: 2, PENDING: 6`，exit 0。

**回填后必须同步改写的正文**（机器判据只覆盖带 `待填` 的两处，第 3 项没有标记、需人工盯）：

| # | 位置 | 为什么必须改 | 由什么触发 |
|---|---|---|---|
| 1 | §4 开头「填表状态（2026-09-19）」（:365-371） | 它逐节声明哪些还是"待填"；填完后就**变成假话** | 判据③覆盖（含 `待填`） |
| 2 | 摘要 `*[主结果待填。]*`（:58） | 要换成 §4.2 的**判定结论** | 判据③覆盖；内容等 §4.2 的 claim 判定（A–D）出来再写 |
| 3 | §5 限制第 4 条（:738-740） | 现写"语义有效与任意同数量主方向有效**尚未分开**…随机 RRR 对照是解决的**必要实验**"（将来时）；§4.4 跑完后要改成**它实际分开了什么** | **无标记、人工盯** |

**人工审校（机器查不到的判断项）**——本节原先把它们写成待核提醒，现已**逐条查证**，
结论是"三项已存在、一项需新增、一项需搬迁"：

| # | 应披露项 | 现状（2026-09-20 查证） |
|---|---|---|
| 1 | §4.2 的**三种门先验**（新格 0.2；`L-rcrf`/A1 preset 自持 0.5；复用格 Stage-0 冻结值 0.5 或 0.2） | ✅ **已存在**：§4.2 注已写明"存在**三种** gate 先验…三者必须在表注逐行标注，且配对比较只在同一 setting 内成立" |
| 2 | §4.4 的「`reference_parity` 只对 **6 个 setting** 成立、Electricity-336 为新算」 | ✅ **已补（2026-09-20）**：原为缺失（全文 `grep reference_parity` = 0；§4.5 里那处「6 个 setting」讲的是另一件事）。已在 §4.4 表注中新增，写明 6 setting 的由来（参照不含 Electricity-336，源于 E10 的 13.9 GiB OOM）、缺参照时**跳过而非判失败**（`:2337-2344`），以及 `probe_cells ≈ 72` 的构成 |
| 3 | §4.6 的 E11 口径差异（既有为 test 口径、单 seed、`rank_sweep_2_stage1` checkpoint）**以及行 1／行 5 的规模与预注册文字** | ✅ **已搬迁（2026-09-20）**：原写在**表格单元格内**，而 §4.6 的回填方式是**整行替换**——照做会**静默删除**这条披露。已把完整文字移到表下注，单元格留一句「**口径差异见下注**」的指针（故空格数仍为 485，未因搬迁而新增空格）。**同理处理了行 1 与行 5**：§4.6 的回填方式是**只替换第五列**，故那两格的计划文字（行 1 的 `42 runs = 7 setting × 3 seed × 2 档`、行 5 的预注册预期与 `36 runs = 6 setting × 3 seed × rank∈{1,2}`）也会被结果顶掉，已一并先移到表下注。**至此 §4.6 的三处需保留文字（行 1/3/5）都已在注中。** |
| 4 | 主张 C 的**两种定义**（三 seed 均值+std、与三 seed 均值）并列 | ✅ **已补（2026-09-20）**：原只写了「均值 + 样本 std」这一种，而回填工具**同时输出两个计数**（本文口径与 `PhaseFormer_gold_standard.md` §4 口径不同）。已在 §4.0 主张 C 处补上第二种定义，并要求回填时**两个计数同时给出并各自注明口径** |
| 5 | §4.7 的 ν* 冻结与 τ̂ 有限样本偏差 | ✅ **已存在**：表注给出 `ν = tau_hat_steps`、`ν* = 57.35`、可分离区间，并单独一条写"`τ̂` 有已知的有限样本偏差" |

**判据④（派生、回填后才可评估）：覆盖数 ≥ 空格数（逐节）**——把 `--inventory` 的"每节空格数"与
校验器报告的"每节比较了多少格"**反过来对**，确认**没有任何待填格落在所有校验器的视野之外**
（否则那一格的错值会永远survive）。**回填前实测：这条判据此刻不可评估**——
四张产物（§4.2/§4.4/§4.5/§4.7）尚未产生，校验器对每节只输出**一条 PENDING 行**：

```text
--inventory:            §4.2 192 · §4.3 0 · §4.4 252 · §4.5 35 · §4.6 0 · §4.7 6   （共 485）
校验器比较格数（现在）:  §4.2 1   · §4.3 168 · §4.4 1   · §4.5 1   · §4.6 1   · §4.7 1
```

**回填后的预注册期望**（覆盖数由各 checker 的表/列结构算出，回填后逐节核对）：

| 节 | 空格（回填前） | 应比较的格数 | 依据 |
|---|---:|---:|---|
| §4.2 | 192 | **280** | 主表 28 × 10 列（表的 10 列与 checker 断言一致）；变体表为说明文字 |
| §4.4 | 252 | **315** | 干预表 21 × 10 = 210（去模型列）+ 解剖表 21 × 5 = 105 |
| §4.5 | 35 | **49** | 7 × 7 |
| §4.6 | 0（**替换型**） | **5** | 只校验第五列（计划文字本就在格里，见 §11.3 的盲区） |
| §4.7 | 6 | **6** | 第一表 3 行 × 2 列 ρ；第二表为设计文字 |
| §4.3 | 0 | **168** | 28 × 6（已填，现即为此数 ✓） |
| **合计** | **485** | **823** | |

判据④成立的条件是"**每节 应比较格数 ≥ 该节空格数**"（上表逐行成立），且回填后实测值与之相符。

**判据①覆盖面的核查（2026-09-20）**：`--inventory` 的节起于 §4.2，故有必要确认"**待填内容全在其中**"。
逐行扫过 §4.1（:408–423）：它有一张表（`| 命题的预测 | 先导观测 | 出处 |`），**空格数 0**——
§4.1 是**先导证据**（既有结果），不属本文补做，且已填满。故 **§4 的待填内容 = §4.2–§4.7 的 485 格**，
**没有"在 §4 里但 inventory 数不到"的格子**；§4.0（冻结门槛）与 §4.1（先导证据）都是已成文的文字。

**披露项（上表）这三处已于 2026-09-20 全部处理完毕**（见上表）。它们都是**可事前完成的披露动作**，
故不留给回填阶段临时补写——**回填时披露方面已无待办**；第 1、5 项本就已就位。

**一个附带的自检**：搬迁第 3 项时有意在单元格里留下指针而不是清空，
故 `--inventory` 仍报 **485** —— 即「这次编辑没有改变待填总量」，
可作为「编辑只动了说明文字、没动待填结构」的证据。

### 10.5 最后两件非机器可查的事

1. **摘要**：`*[主结果待填。]*` 需按 §4.2 的结论替换——**只在 §4.2 判定完成后**动，避免先写出与门槛不符的话；
2. **各实验的独立文档**：每节在回填后补写"阶段 5 审校"与"阶段 6 回填"两段
   （E14/E16/E17/E18 各自目录下），使"每个实验文档独立命名存放"这一要求对**全部**小节成立。

### 10.6 本清单自身的核对

清单里的每条引用都在写完后**对着代码逐条核过**，以免清单本身出错（清单错了比没有清单更糟）：

* **脚本存在性**：`verify_minipaper_fill.py`、`check_builder_outputs.py`、`check_column_contracts.py`、
  `check_pipeline_invocations.py`、`audit_phase2_outputs.py`、`check_section42_coverage.py`
  与六个实验脚本**全部存在**；
* **CLI 选项**：`--from` / `--only` 确实由 `run_phase2_after_e14.sh` 实现，`--list` 输出 7 步；
* **产物文件名**：清单里写的 `main_table.{csv,md}`、`variant_table.csv`、
  `dissection_table_44.csv`、`intervention_table_44.{csv,md}`、`conditional_table.{csv,md}`、
  `negative_table.{csv,md}` **与各 write-back 的写出点逐字一致**
  （`e14_writeback.py`、`e16_writeback.py:366`、`e17_writeback.py`、`e18_writeback.py`）。

### 10.1.2 已核验的**恢复代价**：重跑一步是便宜还是昂贵

承接 §10.1.1（失败语义），本节核实"失败后重跑要付多少"——这决定了我该不该在无人值守的中途干预。

**关键代码**（`scripts/search_phaseformer.py:638-644`）：

```python
complete = run_dir / "metrics.csv"
if complete.exists():
    if args.resume:
        print(f"RESUME completed: {rid}")
        return                      # ← 已完成的 run 干净地 no-op、退出码 0
    raise FileExistsError(f"completed experiment already exists: {rid}")
```

而 E14 / E17 / E18 的命令**都带 `--resume`**（`e14_main_matrix.py:468`、`e17_conditional.py:525`、
`e18_negative.py:247`），E16 则**完全没有** `--resume`（计数 0）。

| 步 | 失败后重跑的实际代价 |
|---|---|
| 1（411 次 test 读取） | 廉价：读取器幂等（已带 test 指标的行走"原样复制"） |
| 3（三个回填/审计工具） | 分钟级：纯 IO，直接覆盖写出 |
| 5 / 6（E17 24 run / E18 78 run） | **廉价**：dispatch 会重新发车每个 cell，但 **已完成的 run 会打印 `RESUME completed` 并立即以 0 退出**，故只付"重新解析+逐格 no-op"的时间，**不重训** ✓ |
| **4（E16）** | **昂贵**：无 `--resume`、也无按 cell 的 checkpoint，产物还是**末尾一次性写出** → 失败即需**整步重跑（3–5 h）**，这是全链唯一的昂贵重跑 ✗ |
| `--from N` | 跳过节级：已成功的步不必重跑 ✓ |

**因此无人值守中若第 4 步失败，代价是 3–5 h 而非全链重来**；其余步失败都可在分钟级恢复。

> **诚实记下我自己的一个过早结论**：我先只读了 dispatcher，看到 `pending = list(cells)`
> （**没有**发车前跳过），就准备把"重跑一步会重训所有 run"写进文档。往下多读一层才发现
> 子进程带 `--resume`、对已完成的 run 是干净 no-op——**结论正好相反**。
> 这与本会话前几次"我的检查写错"同类：**从单层代码推断跨层行为会出错**，
> 好在这次是在写进文档之前发现的（`--from` 与恢复流程都依赖这个结论）。

### 10.1.3 防线地图：谁**失败即停**、谁**报告后继续**、以及第 7 步审计为何是**承重**的

把 §10.1.1（失败语义）与 §10.1.2（恢复代价）合起来，逐工具核到源码后得到这样一张图
（**注意：两个类别都是有意设计，不是缺陷**）：

| 层 | 行为 | 依据 |
|---|---|---|
| **第 1 步** `e14_read_test.py` | **失败即停**：写盘后 `if problems: raise SystemExit(f"E14 stage B had problems: {problems}")`；dry-run 则更早拒绝（`dry run found problems; refusing to proceed`） | 源码 tail |
| **第 5/6 步 训练 runner** | **失败即停**：`if failed: raise SystemExit(...)` | `e14_main_matrix.py:702`、`e17_conditional.py:1407`、`e18_negative.py:890` |
| **第 6 步行 3** `e18_svd_truncation.py` | **失败即停**：`--verify` 对未解析的格子直接拒绝（实测 `135 unresolved cells; refusing to evaluate`） | 实测 |
| **第 7 步** `audit_phase2_outputs.py` | **失败即停**：有 `FAIL` 则非零退出 | 12 缺陷正对照 |
| 第 5/6 步 test 读取 `read_test_generic.py` | **exit 0 并报告**（`rejected` 列在 summary 里） | 已知例外，见 §10.1.1 |
| 第 2 步 `e19_predictive_power.py` | **exit 0 并报告**：只在"**一个 setting 都算不出**"时才停；部分 setting 被跳过（实测半退化输入 → 14 完成 / 14 跳过）仍退出 0 | 实测 + 源码 |
| 第 3 步 `e14_writeback.py` / `e14_params.py` | **exit 0 并报告**：`audit.json` 里列 `settings_incomplete`，但**不**抛错 | 源码（二者无任何 `raise SystemExit`） |
| 第 4 步 `e16_dissection.py` | **exit 0 并报告**：summary 里列 `algebra_failures`，但**不**据此抛错 | 源码（`algebra_failures` 只进 summary） |
| 第 4/5/6 步 回填工具 | exit 0 并报告（未供给的列列为 `None`） | 同上 |

**结论**：链条在**最贵的两步**（训练）与**入口**（第 1 步）是 **fail-closed** 的，
但在**分析/回填**这些工具上是"报告后继续"。这**不是**疏漏——因为
**第 7 步验收审计正是为这一层设计的**：它检查的恰是这些工具"报告而不是中止"的东西
（每 cell 是否有 test 指标、预测力表是否覆盖 28 setting、主表是否 492 行、`algebra_failures` 是否为 0、
行 3 是否覆盖 28 setting…）。**换句话说，第 7 步不是装饰性的，它是承重的**：
把这些工具换成"失败即停"会让链因为一个**良性**的跳过而过早停止，
而把它们留在"报告"层、再由审计统一判定，则**跑完全链后一次性给出可信结论**。

**这也解释了为什么审计跑在最后是对的**：它之后没有别的步骤，所以"最后才发现"的代价只是
"我要看一眼报告"，而不是"浪费了后续算力"。

**唯一的真例外**仍是 `read_test_generic.py`：它不是"良性跳过"，而是**真的会坏**
（worker 崩溃时它照样 exit 0，只在 summary 列 `rejected`）——那属于**故障被静默**，
正是本会话修掉的那个 marker 缺陷所暴露的。故它的兜底是审计里"每 cell 都要有 test 指标"那条判据，
且重跑代价低（读取 + 回填，不重训）。

### 10.7 同步到服务器的**正确形式**（我此前的写法是空操作）

```bash
# 本地（提交后）
git bundle create /tmp/pf.bundle HEAD
scp -q -o ConnectTimeout=25 /tmp/pf.bundle yyk03@11.11.18.3:~/niuyiming/phaseformer_weak_residual.bundle

# 服务器
cd ~/niuyiming/PhaseFormer
git fetch origin HEAD          # ← 必须显式写 HEAD，见下
git merge --ff-only FETCH_HEAD
git log --oneline -1           # ← 实测 HEAD 是否前移，而不是看退出码
```

**为什么裸 `git fetch origin` 是空操作**：服务器的 `remote.origin.fetch` 是
`+refs/heads/*:refs/remotes/origin/*`，而 `git bundle create <file> HEAD` 只写**一个 `HEAD` ref**
（不写 `refs/heads/*`）⇒ 裸 fetch 匹配不到任何 ref，**连 `.git/FETCH_HEAD` 都不生成**，
随后 `git merge --ff-only FETCH_HEAD` 自然报 "Already up to date" 而**什么都不做**。
`git ls-remote origin` 有值 **≠** `FETCH_HEAD` 有值——**判据必须是"服务器 HEAD 是否前移"**。

**同步前后各一条纪律**（承接 §295 行与 2026-09-20 那条"忘了同步"的记录）：

* 同步前确认改动**不触碰 E14 训练路径**：`git diff <旧 HEAD>..<新 HEAD> --name-only | grep -E '^(src/|scripts/search_phaseformer\.py)'`
  必须**无输出**（E14 正从工作树训练）；
* 在服务器上验证任何脚本前，先确认"服务器上的版本 == 我改的那版"（比对 HEAD 哈希，或看三态计数是否含新判据）。

### 10.8 阶段二结束后要补写的文档（清单**已对着现存文件核过**）

六阶段契约要求每个实验的文档独立命名存放。现状盘点（`ls docs/PhaseFormer_L/*/`，2026-09-20）：

| 实验 | 目录 | 已有 | **待补** |
|---|---|---|---|
| E14 §4.2 主表 | `e14_main/` | `01_plan` `02_static_check` `03_smoke` `04_run` `04b_test_read_plan` `05_audit` `05c_writeback_rehearsal` | 阶段 A 收尾 + **阶段 B 测试读取**的 4 记录（追加进 `04_run`）、最终 5 审校、6 回填 |
| E15 §4.3 维数表 | `e15_dimension/` | `01`–`06` **齐全** | — |
| E16 §4.4 解剖/干预 | `e16_dissection/` | `01_plan` `02_03_static_check_smoke` `05b_writeback_rehearsal` | **`04_run`**、**`05_audit`**、**`06_writeback`** |
| E17 §4.5 四臂 | `e17_conditional/` | `01_plan` `02_03_projectors` `05b_writeback_rehearsal` | **`04_run`**、**`05_audit`**、**`06_writeback`** |
| E18 §4.6 负对照 | `e18_negative/` | `01_plan` `02_static_check` `05b_writeback_rehearsal` | **`04_run`**、**`05_audit`**、**`06_writeback`** |
| E19 §4.7 预测力 | `e19_predictive/` | `01`–`06`（阶段 1） | **阶段 2**（第 2 步的 ρ 列）的 4/5/6 记录，追加进现有 `04_run`/`05_audit`/`06_writeback` |

共 **12 处文档产出**（E14 三处、E16 三处、E17 三处、E18 三处，另有 E19 阶段 2 三处追加）。
每处都写**实测数字**，不写"占位"——这些是最后一批必须落地的交付物，故在此列明，避免收尾时漏写。
