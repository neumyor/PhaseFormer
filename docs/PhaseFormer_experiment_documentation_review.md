# 实验文档审计与近期实验一览（2026-09-15）

> 目的：回答"现有实验文档能否串起来"，校对 `agent-log.md`，并给出一份可直接使用的
> **近期实验日志**（做了什么、结论是什么、证据在哪）。
>
> 方法：通读 `docs/` 下全部 41 份 Markdown（含子目录）与 `agent-log.md`（151 条），
> 按四条线做族内与跨族一致性核查，并对可疑数值回到 `research_runs/` 原始产物
> （Stage-0/Stage-1 CSV、checkpoint、`metrics.csv`）复核。本次**未运行任何训练**。
>
> 本次已修正两处文档缺陷（见 §2.1），其余为待办建议（§2.2、§4）。

---

## 1. 结论：能串起来，但索引层断了

**三条实验线各自是自洽的，且存在一次明确的交接；能串成一条主线，但"从索引出发"串不起来。**

### 1.1 三条线 + 一次交接

```text
[A] incumbent 线（K1→K4，08-24 ~ 09-02，已冻结）
    gold_standard(K1) → gold_combo(K3) → rcrf_pe_lff(K2=A2) → 288-run 正式矩阵
      → A2 = 3-seed 统一 incumbent；strict-T28(K4) 为后续单 checkpoint 扩展

[B] 输入成分诊断线（09-02 ~ 09-03，D1 未收尾）
    H1/H3/H4 计划 → D0 完成（210/210/189）→ D4/D5/D6/D7（ETTm1-H192, 512 origins）
      → 结论：不存在"M0 忽略、增强分支在用"的候选；D7 把缺陷定位到"跨周期水平状态"

[C] 弱残差趋势成分线（09-04 ~ 09-05，已结题）
    narrative D7 → asymmetric A1–A6 → X-A/Only-A 路由审计 → 结题（09-05）
      → 交接：转入 NLinear 信息瓶颈 / 低秩压缩研究

[D] 低秩压缩主线（09-10 ~ 09-15，当前活跃）
    pooled screen → 第一轮 joint sweep（null）→ 第二轮 conditioned（单 seed 3/7）
      → causal-EMA/boxcar 平滑扫描（时间分辨率轴）→ 三 seed 复核（结论修订）
      → 机制分析 → RRR 容量分析（容量轴）→ 主导方向刻画 → 论文投放版 §4
```

交接点只有两处、且都写在文档里：**B→C**（D7 结论驱动 A1 选择）、**C→D**（结题文档
§5 声明转入 NLinear 瓶颈研究，causal-EMA 计划 §1 显式引用结题文档）。

### 1.2 四个断点（按影响排序）

| # | 断点 | 具体表现 |
|---|---|---|
| 1 | **索引未覆盖最新一半工作** | `docs/README.md`（最后更新 09-11）只链接低秩线的 3 份文档（conditioned plan/experiment、joint plan）。磁盘上同族但**完全未被索引**的还有 9 份，其中包含**最新的、最权威的两份结论**：`PhaseFormer_rank_capacity_and_data_property_report.md`、`PhaseFormer_lowrank_mechanism_analysis.md`。后者在本次审计前是**真孤儿**（除 agent-log 外无任何文档指向它）。输入成分线的 11 份文档与 `PhaseFormer_structural_defect_research_narrative.md` 同样不在索引中 |
| 2 | **D1 未收尾** | 输入成分线的三 seed 最终分级（计划 §13.7）仍为空；D0 报告 §9 的四项待办（修 interaction 聚合、D1 全链路、计划计数、回填 §13.0）均无完成记录 |
| 3 | **孤儿 SSOT** | `PhaseFormer_NLinear_Progressive_IB_Experiment_Plan_v1.0.md`（2673 行，自称 SSOT）零引用：不在 README、不在 agent-log（日志只写分支名与脚本名）、无兄弟引用；其状态块仍写"Stage 1 起始 / GPU 未启动"，而 09-10/09-11/09-14 的压缩实验实质已回答其 Stage 2 命题 |
| 4 | **交接只存在一句话** | C→D 的交接仅存在于结题文档 §5 的一段文字，没有任何文档从趋势成分线**前向链接**到低秩压缩线；反向（低秩线引用趋势线）只有 causal-EMA 计划 §1 一处 |

---

## 2. 校对结果

### 2.1 本次已修正（2 项 + 1 项补拉）

**(1) 表 1 的 MAE 列在 Weather-96 与 Electricity-336 两行之间被互换 —— 已修正**

- 影响文档：`PhaseFormer_rank_sweep_conditioned_experiment.md` 表 1（数值权威副本）、
  `PhaseFormer_rank_sweep_conditioned_plan.md` §9 表 1 及紧随其后的逐 setting 明细段。
- 复核依据：服务器 `research_runs/rank_sweep_2_stage0/stage0_<Dataset>_<H>_validation.csv`
  的四个配置逐行原始值。
- 修正内容：

  | Setting | 原表 MAE（4 格） | 正确 MAE（4 格） |
  |---|---|---|
  | Weather-96 | 0.2290 / 0.2280 / 0.2274 / 0.2278 | **0.2700 / 0.2713 / 0.2721 / 0.2745** |
  | Electricity-336 | 0.2700 / 0.2713 / 0.2721 / 0.2745 | **0.2290 / 0.2280 / 0.2274 / 0.2278** |

- 两条独立旁证：① 原表 Weather-96 的 4 个 MAE 值 = Electricity-336 的真实值（四舍五入后
  逐格一致：0.228966→0.2290、0.228000→0.2280、0.227445→0.2274、0.227834→0.2278）；
  ② Stage 1 CSV 中冻结配置下 `direct_nlinear` 的 `val_mae` 为 Weather-96 `0.2713`、
  Electricity-336 `0.2274`，与还原后的值一致。
- **下游不受影响**：两行 MSE 列正确，两个 setting 的最优格与冻结配置不变，Stage 1 / 三 seed
  / 全部结论均不改变。该错误只影响"表 1 的可读性"，以及 plan §9 中一段
  "0.2721 vs 0.2713 冻结 MAE 更低者"的自相矛盾表述（已一并改写）。
- 附带澄清：原表把 Electricity-336 的两格 MSE 显示为 `0.1374/0.1374`（并列假象）是因为
  全表统一按**截断**而非四舍五入取 4 位小数；全精度下 `g0.5_lrdefault` = `0.137442`
  < `g0.2_lr3e-4` = `0.137456`，**从未触发并列规则**。冻结结论仍正确。

**(2) `agent-log.md` 两条条目日期错误 —— 已修正**

- `## 2026-09-14 — 主导预测方向的精确刻画与早期表述修正`
- `## 2026-09-14 — 论文用强结论与图表方案写入报告（§4）`

两条实际提交时间为 **2026-09-15 09:51 / 09:55**（commit `d7173a4`、`522eb25`），已改为
`2026-09-15`。逐条比对 151 个条目与其首次提交日期后，**其余条目的标题日期与提交日期均一致**，
仅此两条错。

**(3) 补拉 09-12 日志记录的"未落地"原始产物**

`agent-log.md` 2026-09-12 条目记录：因当时 SSH 会话的 stdout 读取限制，boxcar 与 causal-EMA
两轮共 14 份 `*_results.csv` 只留在服务器。本次已 rsync 到本地
`research_runs/smooth_ratio_sweep_v1/`（7 份）与 `research_runs/causal_ema_smooth_sweep_v1/`
（7 份），该限制已不复现；建议在日志中标注已解除。

### 2.2 待修清单（按优先级）

**P0 — 会误导读者对"当前状态"的判断**

1. `docs/README.md` 重写"机制消融"与"输入成分"两节：补 9 份低秩/弱残差线文档（含最新的
   容量报告与机制分析）、11 份输入成分文档、1 份叙事文档；补 I0/I1/D1/D2/D3 五个 mechanism
   与 `weak_residual`/`weak_residual_asymmetric_trend` 的索引条目。
2. `docs/README.md` §"输入成分利用诊断（计划中）"称"尚未实现或运行"——**与事实相反**：
   D0 全链路已于 2026-09-03 完成（210/210/189、`result_summary_d0.csv` 420 行、已读 test），
   D4–D7 报告均已存在。应改为"D0 已完成（provisional），D1 未收尾"。
3. 两篇平滑扫描文档第 3 行仍写"计划已冻结，结果待训练完成后回填"，而 §6 已写"已完成并回填"——
   头部状态行过期（读者会以为实验还没跑）。
4. `PhaseFormer_rank_sweep_conditioned_experiment.md` §5（单 seed 主实验结论）第 2/4 条仍写
   "部分信号（3/7）"与"压缩整体偏有害"，未指向 §7 的三 seed 修订结论
   （"中等压缩近中性、深压缩偏害、无统一最优 q"）。§5 第 6 条已加指针，第 2/4 条应同样处理。
5. `PhaseFormer_lowrank_mechanism_analysis.md` §3.3/§4.1 关于"保留下来的模式仍具高频/局部化
   敏感性"的表述建立在"谱质心位于中高段"这一弱判据上；最新报告 §2.3/§2.6 已明确该判据为弱判据
   并给出严格分带结果。建议给该节加限定语并指向修正声明（两者角度不同，不构成硬矛盾）。

**P1 — 影响可追溯性/合规性**

6. `PhaseFormer_rank_sweep_conditioned_plan.md` 头部与 §10 的"部分信号（3/7）"未标注
   "为 seed 2021 单 seed 结果，三 seed 下未复现"。
7. `PhaseFormer_residual_smooth_ratio_sweep_experiment.md` §1/§2.1 以"3/7 部分信号"作为
   平滑扫描的 rank 选择依据；该依据在三 seed 下已不成立（ETTm2-192 q=1/8 三 seed 均值
   +0.036%/−0.409%，Weather-192 q=1/32 +0.160%/−0.545%）。建议改为"沿用第二轮用户指定的
   test 最优 q（已披露 test-set selection）"。
8. `PhaseFormer_pooled_lowrank_nlinear_experiment.md` 内"Controlled Follow-up Plan"已被
   joint 计划取代，但文件头**无 superseded 标注**，读者无法从该文件得知。
9. 输入成分线：计划计数 `8 数据集/24 锚点/216 retrained/456 单元` 应更新为 v1.2 实际
   `7/21/189/399`（D0 报告 §8-2 已标记但未改计划）。**已修正（2026-09-15）**。
   *（更正本条初稿的一处过强表述：计划 §12 已明确写出"严格六文件报告包属于正式报告阶段，
   尚未由本组 runner 自动生成，不能把 scratch 目录直接当作最终审计目录"，因此该交付目录
   不是"悬空引用"，而是**已披露的已知限制**；真正待修的只有 v1.1 计数。）*
10. 输入成分线：`summarize_input_component_ablation.py` 的 aggregate interaction 列与 §8.1
    口径不符（frozen H1 minus_A 的 M1−M0：长表重算 +2.4pp vs aggregate 记 +36.1pp），
    导致 D0 的"Interaction ≥ +0.5pp 且 CI 下界>0"门槛**从未被正式判定**；且 `D1/D2/D3`
    在本仓库有两套完全不同的含义（候选发现线 vs H1/H3/H4 的 horizon×seed 扩展），
    引用时必须带前缀。
    **已修复代码层（2026-09-15）**：根因是 interaction 建在 sham-adjusted 差值上，即
    `Interaction(§8.1) − ShamInteraction`，该恒等式精确复现观测偏差
    （+2.4 − (−33.7) = +36.1 pp）。修复落在新建的
    `src/dataset/input_component_contrasts.py` + 5 项单元测试
    （`tests/test_input_component_contrasts.py`，本地通过，无需 torch）；宏平均同时改为
    逐 dataset×horizon 等权。**残留动作**：用修复后的脚本在 GPU 机器上重生成
    `result_summary_d0_aggregate.csv`，之后 Interaction 门槛才可正式判定
    （命令见 D0 报告 §8-1）。
11. incumbent 线"当前最佳"冲突：`docs/README.md` 称 K4（strict-T28）为当前最佳，而
    `top5_test_models.md`/`periodic_residual_next_stage.md`（09-02）称 A2 为 incumbent，
    且 K4 自己的登记文档写明"尚未超过 two-stage Full Repair，是本轮起点"。建议统一为
    一句话状态（例：A2 = 最后的 3-seed 统一 incumbent；K4 在 20 个已登记 setting 中 12 个
    双指标优于 Golden，其中 12 格为单 seed test-set selection，Electricity/Traffic 取消）。
12. `docs/strict_t28_master_table_configs/README.md` 与 K4 计划称"12 cells carry on-disk
    config/command files"，磁盘实为 **20 个 cell**（5 数据集 × 4 horizon，Weather 4 格齐全）。

**P2 — 记录整洁度**

13. `agent-log.md` 有 33 处指向已不存在文档的引用：25 处属 2026-09-02 docs 清理（README 已说明
    删除了哪些家族）与 08-26 的"合并为闭环文档"，5 处为从未放在 `docs/` 下的产物/审计文件
    （`ALL_COMPONENT_ROUTE_VALIDATION_METRICS.md` 等），3 处为已删除计划/笔误
    （`EXPERIMENT_SEARCH_PLAN.md`、`log.md`、`periodic_residual_next_stage.md`）。建议在文件头部
    加一段"历史引用说明"，避免后续读者逐条追查死链。
14. `agent-log.md` 中 D4–D7 与 IB 计划从未按**文件名**登记（只按主题描述），
    导致无法从日志直接定位文档。
15. 文件开头存在一个日期回跳块：前 6 条为 `2026-09-10`，第 7 条起回到 `2026-09-05` 并顺时
    递增到 09-15。日志事实上是"补记块 + 顺序主体"，但文件内无任何说明。
16. 服务器上 `research_runs/rank_sweep_2_multiseed_stage1_20260914_v5/`（5 个 run、
    与 v4 同名 run_id、数值逐位相同）**未在任何文档中登记或排除**；报告 §7.5 只列了 v4 与
    repair_v1（v3 已显式排除）。建议补一句"v5 为重复产物，不参与统计"。
17. `PhaseFormer_top5_test_models.md` 称"前五"选自 288-run 矩阵的八个模型，但排名表只列 5 行
    （I1 1.0038、D3 1.0003、A0 1.0072 只在 next_stage §3.3 出现）。
18. ETTm2-96 的 RCRF MSE 在文档间有三处小数差异（`0.159755` / `0.159762` / `0.159761`；
    Electricity-336 `0.164113` vs `0.164114`）；README 把 K3 同时指向两份文档但未指明权威值。
19. `docs/README.md` 的通用协议声明"所有结构…seeds 2021/2022/2023"，但 K4 的 20 个已登记
    setting 中 12 个为单 seed（ETTh1/ETTm1/Weather）。

### 2.3 本轮修复记录（2026-09-15）

| 类别 | 改动 | 提交 |
|---|---|---|
| 数值权威副本 | conditioned 报告与计划表 1 的 Weather-96 / Electricity-336 MAE 列互换 → 还原并加勘误注 | `aaaa062` |
| 日志 | 两条 09-14 标题改为 09-15（实际提交日）；头部加阅读说明（补记块、33 处历史引用的来源） | `39727d5`、`d836662` |
| 产物 | 补拉 09-12 未落地的 14 份原始 CSV（boxcar 7 + causal-EMA 7） | `aaaa062` |
| 代码 | input-component interaction 口径修复（新建 `src/dataset/input_component_contrasts.py`；`summarize_input_component_ablation.py` 改用 §8.1 定义并把宏平均改为逐 setting 等权）+ 5 项单元测试 | `e000cac` |
| 文档（P0/P1） | README 四节重写；平滑两篇状态行；conditioned 报告 §5 / 计划头部与 §10；pooled follow-up 标注被取代；机制分析频率判据限定；输入成分计划计数 v1.2；D0 报告 §8-1 根因记录 + 命名提醒；strict-T28 registry 与计划计数 12→20；incumbent 状态统一 | `d836662` |

> 所有改动均**不改动任何结论或数值**（表 1 勘误属纠错，interaction 修复属纠正聚合口径；
> 其余为状态标注、索引补全与计数同步）。interaction 修复后仍需在 GPU 机器上重生成 D0
> aggregate 文件，门槛判定才生效。

---

## 3. 基本日志：近期实验一览

口径说明：`seed 2021` 单 seed 实验一律标注；"test"指标均为**每 checkpoint 只读一次**；
7 setting = {ETTh2-96, ETTh2-720, ETTm2-96, ETTm2-192, Weather-96, Weather-192, Electricity-336}
（按 test 名义双优事后挑选，受 test-set selection 约束）。

### 3.1 低秩压缩主线（2026-09-10 → 09-15）

| 日期 | 实验 | 规模 | 主要结论（关键数字） | 文档 / 产物 |
|---|---|---|---:|---|
| 09-10 | pooled low-rank 筛选（H96 探索） | ETTh1/ETTm1 各若干候选 | 无一候选双指标改善：ETTh1 `pool2/rank8/s=0.75` test `0.362479/0.390818`（对 Golden +0.97%/+2.31%）；ETTm1 `pool1/rank16/s=0.25` `0.299431/0.350440`（+2.19%/+1.87%） | `PhaseFormer_pooled_lowrank_nlinear_experiment.md` |
| 09-10 → 09-11 | **第一轮 joint low-rank rank sweep**（默认配置全矩阵） | 70 runs = 10 setting × {phase_only, direct, q=1,1/4,1/8,1/16,1/32}，seed 2021 | **null result**：压缩变差 4/10、变好 2/10、平坦 4/10；q=1/32 在 MSE 7/10、MAE 8/10 变差但中位幅度 <1%；因子化满秩 vs direct 差异 ±1.9% 内；决策：不引入 preset、保留 `direct_nlinear` | `PhaseFormer_joint_lowrank_rank_sweep_plan.md`（§13 判定）、`research_runs/joint_lowrank_rank_sweep_v1/` |
| 09-11 | **第二轮 conditioned sweep**（用户质疑第一轮配置不当） | Stage 0：28 runs（7 setting × {gate 0.2/0.5} × {lr 1e-3/3e-4}，validation-only）；Stage 1：49 runs（7 setting × 7 档，读一次 test） | Stage 0：gate_init 支配 lr，**5/7 冻结配置 ≠ 第一轮**，但同 setting 内 4 配置 val MSE 极差 ≤0.9%（与单 seed 噪声同阶）；Stage 1（单 seed）判 **3/7 部分信号**（ETTh2-720 q=1 +1.46%、ETTm2-192 q=1/8 +1.02%、Weather-192 q=1/32 +0.52%），最优档位分散 | `PhaseFormer_rank_sweep_conditioned_plan.md`、`..._experiment.md`（数值权威副本） |
| 09-14 | **三 seed 复核（结论修订）** | +70 runs（7 setting × seeds 2022/2023 × {direct, q=1/4,1/8,1/16,1/32}）；审计 **105/105** 单元无缺失无重复 | 单 seed 的 3/7 未复现；跨 21 个 setting-seed 单元宏平均 ΔMSE/ΔMAE：q=1/4 `−0.119/−0.292%`、q=1/8 `+0.068/−0.186%`、q=1/16 `−0.320/−0.462%`、q=1/32 `−0.523/−0.810%`；**唯一可复现增益：ETTh2-720 q=1/4 与 q=1/8（3/3 seeds；q=1/8 +1.046%/+0.432%）**；结论：**中等压缩近中性、深压缩偏害、无统一最优秩** | `..._experiment.md` §7、`research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/`（`audited_results.csv` 等） |
| 09-14 | 三 seed 压缩曲线图表 | 10 张图 + 35 行图数据审计副本 | MSE/MAE 分面图、逐 setting 双面板；35 个（7×5）三 seed 均值单元中 30 个双指标低于 Golden | `scripts/plot_3seed_conditioned_rank_sweep.py`、`..._summary/figures/` |
| 09-14 | **机制分析（4 个事后实验，无训练）** | 复用已有 checkpoint | ① 全秩权重谱衰减快（6/7 setting 90% 能量秩 4–35；Electricity-336 例外 106、PR 160）；② Electricity-336 rank=10：SVD 截断 `0.2104` vs 训练低秩 `0.1629`（差 29%）→ 秩约束下训练能重组更优解；③ 低秩基向量仍含高频（encoder 高频占比 0.18–0.46）；④ 频段探针**未验证**核心假设（探针打在公共输入而非分支私有输入） | `PhaseFormer_lowrank_mechanism_analysis.md` |
| 09-14 | **RRR 容量分析（训练无关，本次新增）** | 7 setting，CPU | 最优秩-r 映射（容量上界）：最深测试档 r=3/6/10/22（参数为满秩头的 3.5–6.1%）保留 **92.4–101.9%** 的"相对 persistence 锚点的可实现提升"；r=1 保留 65.5–86.2%；90% 只需 **2–4 维**、PR **1.33–2.12**；λ 恒等式核对到 **4.7e-9**（26 单元）；**权重谱 418 维 ≠ 预测谱 8 维** | `PhaseFormer_rank_capacity_and_data_property_report.md` §1–§3；`scripts/analyze_optimal_lowrank_capture.py` 等 5 个脚本 |
| 09-14/15 | 主导方向刻画 + 训练对齐 | 7 setting + **534 checkpoint** / 363 checkpoint | b₁ 近端集中（末 24 步占能量 53–70%）、最佳单模板为指数衰减核（τ=6–72，\|cos\| 0.58–0.78）、输出为**恒定水平位移**（\|cos\| 0.892–0.989，符号 7/7 一致）；融合 gate `g`=0.207–0.507 ⇒ 指标只看得到 **g²=4.3–25.7%** 的分支误差；Electricity-336 在 r=10 时 gate 主动关闭（0.433→0.332，g² −43%）；训练头与最优方向 \|cos\| 在 5/7 setting ≥0.81；**"低频/局部趋势"表述已撤回**，改为"近端指数加权的近期平均水平 → 恒定位移" | 同上 §2.4/§2.6、§4（论文投放版）；`scripts/describe_leading_direction.py`、`scripts/align_trained_and_optimal_direction.py` |

### 3.2 平滑（时间分辨率轴）两条线（2026-09-12）

| 日期 | 实验 | 规模 | 主要结论 | 文档 |
|---|---|---|---:|---|
| 09-12 | boxcar `smooth_ratio ∈ {0,0.25,0.5,0.75,1}`（同 7 setting、同冻结 gate/lr、单 seed） | 35 runs | **2/7 检测到效应**（ETTh2-96 s=1 −1.11%/−1.21%；ETTh2-720 s=1 −2.98%/−2.27%，方向均为**劣化**）→ 判"无可检测效应"；最优 s 分布：s=0×4、0.25×2、1×1 | `PhaseFormer_residual_smooth_ratio_sweep_experiment.md` |
| 09-12 | causal-EMA 平滑（去掉低秩压缩，改用 X-A/Only-A 研究中的因果 EMA 算子；α=0.08） | 35 runs | **3/7 部分信号**（ETTh2-96 −1.81%/−1.42%；ETTm2-96 −1.24%/−1.04%；Electricity-336 −1.50%/−1.04%），方向全部"平滑越强越差"；两轮合计 14 个 (setting, 算子) 组合**无一出现"平滑显著改善"** | `PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md` |

机制二分（来自 09-14 报告 §4.1）：**平滑 = 沿"时间分辨率轴"直接删掉输入信号；秩压缩 = 沿
"容量轴"限制模式数量、不预先过滤输入**，因此前者一致偏害、后者容量中性。

### 3.3 输入成分诊断线（2026-09-02 → 09-03，D1 未收尾）

| 日期 | 实验 | 规模 | 主要结论 | 文档 / 产物 |
|---|---|---|---:|---|
| 09-02 | H1/H3/H4 干预流程实现与校验 | 提取器 + runners + tests（276→280 passed） | 工程就绪；2880-run 矩阵启动后被用户改为"决策范围优先 v1.1"，v1.2 剔除 Traffic（7 数据集）+ D0→D1 串行编排 | agent-log 09-02；`src/dataset/input_component_ablation.py` |
| 09-03 | **D0 全链路完成（H1/H3/H4）** | Track R 210 runs → 审计 → Track F 210 读 → retrained 189 checkpoint → `result_summary_d0.csv` 420 行（qc 420/420）+ 宏表 60 行 | 三假设均**未达** Strong/Partial；H1 出现 sham 混淆（M0 `sham` MAE +63.1% > `minus_A` +31.9%）；结论一律标记 `provisional (seed2021 only)`；**D1（三 seed）未收尾** | `PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md` |
| 09-03 | D4/D5/D6/D7 诊断（ETTm1/H192、512 origins、validation-only、无 test） | D4 全 validation 11,329 origins；D5 15 项；D6 3 项结构关系；D7 六描述量 + 五折 OOF | D4：分支确实使用 A（recent-linear `X-A` NLinear-only CF M1 +222.3%/M2 +31.2%），但 M0 也 +140.1% → 不是"原版没用"；D5：**15/15 无一项**满足"M0 近零 + 增强显著"；D6：结构关系亦无候选；D7（收敛点）：校正收益与**跨周期水平波动**相关最高（+0.490/+0.534），与**最后周期水平偏移**次之（+0.483/+0.488），六特征 OOF R² 0.205/0.299 | `PhaseFormer_input_component_D4...D7_*.md`、`..._evidence_summary.md` |

**与低秩主线的接口**：D7 的"水平状态"结论与容量报告 §2.6 的"近端水平 → 恒定位移"方向一致
（可互引）；但 D4/D5 同时显示 M0 也强依赖尾部观测（尾部置零使 M0 +140.1%、D2-192 +56.25%），
因此**不能**把"低秩只砍用不上的容量"写成"相位分支不需要这些输入"。

### 3.4 弱残差趋势成分线（2026-09-04 → 09-05，已结题）

| 日期 | 实验 | 规模 | 主要结论 | 文档 |
|---|---|---|---:|---|
| 09-04/05 | A1–A6 六种趋势提取（末点锚定、频谱泄漏 ≤0.10 冻结）+ X-A / Only-A 双路由审计 + 样本级案例 | 18 条模型结果、1,040,725 条 sample×channel 行、150 案例、60/32 双向角色样本 | A 是**条件性校正**而非完整表征；X-A 路由效果随数据集/成分反号（Weather CycleLevels −14.18%/−24.59%；ETTm1 RecentLinear −2.30%/−3.99%）；发现高斯平滑的右侧 padding 伪影 → 改用单侧 EMA/Holt | `Weak_residual_asymmetric_component_plan.md`、`Weak_residual_three_trend_components_experiment_plan.md`、`Weak_residual_trend_component_study_closure.md`（09-05 结题，§5 交接给 NLinear 瓶颈研究） |

### 3.5 更早的 incumbent 线（2026-08-24 → 09-02，摘要）

RCRF 融合（K3 `gold_combo_reliability_s2`）→ 周期位置编码 LFF（K2 `rcrf_pe_lff` = A2）→
**288-run 正式矩阵**（12 setting × 8 mode × 3 seed）→ 无候选满足替代 A2 的统一条件，
**A2 = 3-seed 统一 incumbent**；随后 strict-T28（K4：A2 锚点 + 有界 ICPT 修正、单 checkpoint）
在 20 个已登记 setting 中 12 个双指标优于 Golden（其中 ETTh1/ETTm1/Weather 共 12 格为单 seed
的 test-set selection，Electricity/Traffic 取消）；其登记文档自述"尚未超过 two-stage
Full Repair，是本轮起点"。当代结论：**A2 是最后的 3-seed 统一 incumbent；K4 是后续
单-checkpoint 扩展线，尚未形成"全数据集最优"的既定结论**。

---

## 4. 建议的最小修复集（可直接执行）

| 优先级 | 动作 | 影响面 |
|---|---|---|
| ✅ 已完成 | 表 1 MAE 勘误 + 两条日志日期 + 补拉 14 份原始 CSV（`aaaa062`、`39727d5`） | 数值权威副本正确性 |
| ✅ 已完成 | 重写 `docs/README.md` 的机制消融/输入成分/288-run mechanism/审计索引四节（新增索引 12 份文档 + 5 个 mechanism） | 决定新读者能否找到最新结论 |
| ✅ 已完成 | 状态行/指针：两篇平滑文档头部、conditioned 报告 §5 第 2/4 条、conditioned 计划头部与 §10、pooled 文档 follow-up、机制分析 §3.3/§4.1 | 消除"实验还没跑"与"旧结论仍是当前结论"的误读 |
| ✅ 已完成 | 输入成分线：计划计数 → v1.2（7/21/189/399、2520 Track R、全矩阵 252/2268）+ `--expected-count 189`；interaction 口径修复（代码 + 5 项测试）+ D0 报告 §8-1 记录根因与残留动作 | 该线唯一的量化门槛可判定性 |
| ✅ 已完成 | incumbent 状态统一（README 注 + 两份登记文档状态注）；registry README 12→20 cells 并补 Weather；`agent-log.md` 头部加阅读说明（补记块 + 33 处历史引用来源） | 记录整洁度与可追溯性 |
| 待办（本轮未做） | 服务器 `..._v5/` 目录登记为"重复产物、不参与统计"；`top5` 只列 5 行却称"前八模型"；ETTm2-96 RCRF MSE 三处小数差异 | P2：记录整洁度 |
