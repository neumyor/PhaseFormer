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
| D-4 | §4.6 行 1 平滑复测范围 | **7 个 test-selected setting，2 档** | 新训 42 runs |

**D-2 的必然后果（必须在表注写明）**：`weak_residual` 行内，7 个复用格用 Stage-0 冻结
`(gate, lr)`，其余 17 个新格用 preset 默认；两者不可声明为同一超参协议，配对比较只在
"同一 setting 内 PhaseFormer-L vs `phase_only`"这一层成立。`phase_only` 行同理（6 复用 / 18 新）。

**§3.4.3 的 `ν*` 冻结次序**：`ν*` 只由 6 个已知 setting 的**训练集**统计量拟合，不需要任何
§4.2 结果。因此 E19 阶段 1（train-only 统计量）必须先于 E14（主表矩阵）完成并冻结 `ν*`，
以保住"冻结后盲测"的时序声明。这一步是分钟级 CPU 作业，不占用关键路径。

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

### 2.2 E14 主表矩阵明细

| 行 | 配置 | setting 域 | 复用 | 新训 runs |
|---|---|---|---:|---:|
| `phase_only` | `--mechanism no_residual` | 24 | 6 | 18×3 = 54 |
| PhaseFormer-L（主）/ always-on | `weak_residual` + `weak_period_residual_head_type=shared` | 24 | 7 | 17×3 = 51 |
| L-q1/4 | `pooled_lowrank`，`pool_factor=1`，`rank=H/4` | 24 | 7 | 17×3 = 51 |
| L-q1/8 | `pooled_lowrank`，`pool_factor=1`，`rank=H/8` | 24 | 7 | 17×3 = 51 |
| L-rcrf | `rcrf_nlinear_plain` | 24 | 0 | 24×3 = 72 |
| A1（incumbent 参照） | `gold_combo_reliability_s2` | 12（6 数据集 × H336/720） | 0 | 12×3 = 36 |
| Traffic 附录 `phase_only` | 同上行配置 | 4 | 0 | 12 |
| Traffic 附录 PhaseFormer-L | 同上行配置 | 4 | 0 | 12 |
| Traffic 附录 L-q1/4 + q1/8 | 同上 | 4 | 0 | 24 |
| **合计** | | | | **363** |

**`always-on` 与 PhaseFormer-L（主）共用同一批 run**：`always-on` 就是 `weak_residual`
在 24 个 setting 上的读数（17 新 + 7 复用）；PhaseFormer-L（主）是其中 `s=1` 的 setting
取 `weak_residual`、`s=0` 的 setting 取同 setting 的 `phase_only` 读数。因此"24 − 7 = 17"
计数的是 always-on 的 17 个新 setting，与 minipaper §4.2 的脚注一致。

**必答项落点**：(a) ETTh2 四 horizon 对 `phase_only`/Golden 的差距 + FITS 引用数字（外部参照，仓库无源）；
(b) 开关对 ETTh1/ETTm1 的判定 + always-on 在这两个数据集上的读数；(c) q=1/8 与 direct 的三 seed 差是否 ≤0.5%。

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
| 行 1 | 输入平滑 2 档复测（boxcar `smooth_ratio=0.5`；causal EMA `alpha=0.08`） | 7 setting × 3 seed × 2 档 | 42 |
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
