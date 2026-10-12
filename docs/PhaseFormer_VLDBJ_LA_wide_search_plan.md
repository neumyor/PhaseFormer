# 实验 3 计划：LA-only 宽域调参搜索（test 选取，允许极端与关闭）

状态：**已确认并启动**（2026-10-11 21:50 用户确认三项口径：架构内关闭（gate≈1e-4 +
允许选 mod-off 读数，不换 PF/L checkpoint）；一条命令 + driver 自动推进（08:30 后拒启
新 job）；head S2 与门共享固定）。实现：`scripts/vldbj_m1/wide.py`（driver）+
`model.py` gate 下界放宽至含 1e-4 + `tests/test_vldbj_wide.py`（6 项，合计 30 项通过）。
旧调参 6 点网格与现行配置作为既有候选进入选择（R1 对 20 个已调 setting 跳过重复变体）。
时间约束（用户 2026-10-11）：调参预算无上限，仅时间限制——最后一条启动实验的命令
不得晚于 **2026-10-12 09:00**。当前 21:42，可用窗口约 11 小时。

## 0. 目标与口径

- 目标：为每个 setting（7 数据集 × 4 horizon = 28）选出 test MSE 最低的 LA 配置，
  最大化主表 LA 行。PF 与 L 不动（LA-only）。
- **test-set selection**（用户决定 2026-10-10）：全部选择按 seed-2021 test MSE，
  所有候选数值（含未选中者）写入不可变记录，可审计；论文措辞仍为
  "small grid search"，不写 blind/unbiased。
- 报告数字：被切换的 setting 报 3-seed（2021–2023）均值 ± 标准差，全部 seed 计入，
  不做 best-of-N seed。
- 基线：论文现行最终配置（主配置 + 2026-10-11 confirm round 的 8 个切换）为
  "current"，新候选必须胜过它才切换。
- 训练口径与论文一致：**从零训练**（非热启动）、E14 协议、Huber、≤30 epoch、
  patience 8、best-val checkpoint。head 固定 S2、门共享按现行 per-dataset 选择
  ——这样论文"level branch 105 参数、总增量 119–367"的声明不被破坏。

## 1. 搜索空间（允许极端，允许"关闭"）

变体字段（在现有 `TUNE_FIELDS` 基础上放宽）：

| 字段 | 范围 | 说明 |
|---|---|---|
| `gate_init` | **1e-4**（含）~ 0.99 | 1e-4 = level/shape 融合门初始即关（"关闭"的机制；ext lr 同时取 0.1 时门全程≈0，功能上退回纯相位路径） |
| `extension_lr_mult` | 0.1 ~ **100** | 极端高/低调参允许 |
| `communication_width` | 2 / 3 / 4 | 有效 k = min(w, 3, d) |
| `backbone_lr_mult`（新，runner 层） | 0.3 / 1 / 3 | 从零训练时骨干与扩展分组学习率（热启动实验已验证机制，移植到从零） |

"完全关闭特定设置"的实现（待用户确认口径）：**在 LA 架构内关闭**——
(a) gate_init=1e-4 + extension_lr_mult=0.1（训练后门仍≈0）；
(b) 每个 run 本来就同时记录 native 与 modulation-off 两组 test 读数，允许
per-setting 选择 mod-off 读数。两者叠加时该 setting 的"LA"在功能上等于
PhaseFormer（分支参数仍在，参数声明不变），论文需一句话披露
"per-setting selection may switch the modulation off"。
**不包含**：直接把某 setting 换成 PF/L 的 checkpoint——那会使主表"LA"名不副实、
消融表失效，不推荐；若用户明确要求再议。

## 2. 轮次设计（seed 2021，全部 28 setting）

**Round 1（固定网格，12 变体 × 28 = 336 run）**，width=2 为主：

| # | gate_init | ext_lr | width |
|---|---|---|---|
| 1–3 | 0.001 | 0.1 / 1 / 10 | 2 |
| 4–6 | 0.1 | 0.1 / 1 / 10 | 2 |
| 7–9 | 0.5 | 0.1 / 1 / 10 | 2 |
| 10 | 0.5 | 30 | 2 |
| 11 | 0.9 | 10 | 2 |
| 12 | 0.2 | 10 | 4 |

覆盖：门近关（1–3）、现行邻域（4–9）、极端高学习率（10）、极端高门（11）、
旧调参最优 t4（12）。旧网格 6 点与现行配置作为已有候选直接进入选择，不重跑。

**Round 2（自适应，约 6–8 变体 × 28 ≈ 200 run）**：对每个 setting 的 R1 最优
邻域细化：gate 细邻域、ext_lr ×/÷3、width {2,3,4} 互补点，加极端探针
（ext_lr 100、gate 0.99、backbone_lr_mult 0.3 与 3 用于 R1 冠军的 lr/gate）。
网格由 R1 结果自动生成，生成规则预注册（见 §4），先写入 manifest 再跑。

**Select**：每 setting 在 {current（已有 3-seed）} ∪ {R1/R2 全部 seed-2021
native 与 mod-off 读数} 中按 seed-2021 test MSE 取 argmin；胜 current 超过
0.2% 才进入 confirm。

**Confirm**：进入者补跑 seeds 2022/2023；3-seed 均值胜 current 3-seed 均值
超过 0.2% 才切换。与 2026-10-11 confirm round 同规则。

**Collect**：results.csv + selection.json（全部候选 + 选择轨迹 + 标注）。

## 3. 时间表（今晚）

以实测单 run 耗时（ETTh1 ~1.5 min、ETTh2 ~0.75、ETTm1 ~5、ETTm2 ~3.5、
Weather ~15、Electricity ~20、Traffic ~46–55 min；与 confirm round 实测一致）：

| 时间 | 内容 | run 数 |
|---|---|---:|
| 21:45–22:45 | 代码 + 单元测试 + bundle + 启动 | — |
| 22:45–02:45 | Round 1 | 336 |
| 02:45–05:15 | Round 2（自适应） | ~200 |
| 05:15–05:30 | Select | — |
| 05:30–07:45 | Confirm（≤ 切换数 × 2 seed） | ≤ ~50 |
| 07:45–08:30 | Collect + 汇报 | — |

吞吐核算：R1 ≈ 73 GPU-h / 24 slot ≈ 3.1 h + Traffic 尾部 ≈ 3.5–4 h。全局队列、
Traffic/Electricity 优先排入，慢任务不落尾。

**截止保险**：driver 带 `--launch-deadline 2026-10-12T08:30`（服务器时钟），
超过后拒绝启动任何新 job（含 confirm）；赶不上 confirm 的 setting 保留现行配置，
明天单独请示是否补跑。整晚只需我今晚发一条启动命令，driver 自动走完
R1 → R2 → select → confirm → collect。

## 4. 预注册规则（写死在代码里，防临时起意）

1. R2 网格生成：对每 setting 取 R1 的 seed-2021 native MSE 最优变体 v*，
   生成固定邻居集：gate ∈ {v*.gate/3, v*.gate×3} ∩ [1e-4, 0.99]、
   ext_lr ∈ {v*.lr/3, v*.lr×3} ∩ [0.1, 100]、width = {2,3,4} 中非 v* 的两个、
   backbone_lr_mult ∈ {0.3, 3}（作用于 v*）、固定探针 {ext_lr 100 + gate v*}、
   {gate 0.99 + ext_lr v*}；去重、上限 8 变体/setting。
2. 选择：seed-2021 test MSE argmin（native 与 mod-off 都算候选），>0.2% 才
   confirm，confirm 后按 3-seed 均值 >0.2% 才切换；平局（<0.2%）保 current。
3. 所有选择记录 `selection_basis: test split (seed 2021), test-set selection`，
   候选全量入档。

## 5. 代码改动

1. `scripts/vldbj_m1/model.py`：`prepare_config` 的 gate_init 下界放宽为
   ≥1e-4（闭区间；现行为 0.2/0.5 等不受影响，单测断言 E14 主配置哈希不变）。
2. `scripts/vldbj_m1/wide.py`（新 driver）：复用 `proto.py` 的
   `run_job`/`attach_lr_groups`/job 布局（变体字段进 variant，backbone_lr_mult
   为 job 字段）；stages `r1/r2/select/confirm/collect/all`；launch-deadline
   检查；R2 自适应网格生成（§4.1）；全部候选与读数写 `candidates.csv` 与不可变
   `wide_selection.json`。
3. `tests/test_vldbj_wide.py`：网格生成确定性、deadline 拒绝、选择规则
   （0.2% 阈值、native vs mod-off 候选）、gate 下界放宽、默认配置哈希不变。

## 6. 产出与后续

- 结果文档 `docs/PhaseFormer_VLDBJ_LA_wide_search_results.md`：全部候选、
  每 setting 的选择轨迹、新旧配置对比（3-seed）、与 Golden 及同协议 PF 的对比。
- **论文不动**：先看结果。若你决定让宽域胜者进主表，则消融表、envelope 表、
  机制分析都需用新 checkpoint 重算（额外工作量，另行请示）。
- 服务器：新冻结 clone `PhaseFormer_vldbj_wide_20261011`（新 commit 的 bundle），
  输出 `research_runs/vldbj_wide_v1/`；运行期间不同步代码。

## 7. 风险

- 时间估计偏差：Traffic 从零最坏 ~55 min/run；若 R1 超 4.5 h，R2 自动收缩
  （per-setting 变体数按剩余时间缩放，规则写入代码）。
- confirm 挤不进 deadline：未 confirm 的候选不得进入报告（3-seed 口径），
  保留 current，次日请示。
- 宽域搜索找到的收益可能主要在少数 setting（Weather/ETTm2 类）；若整体收益
  <0.5%，建议论文维持现行配置，仅内部记录。

## 8. R3 轮（2026-10-12 用户批准："全部三块"，不限时）

在实验 3 结果（16/28 切换、4/7 胜 Golden）之上，基线切换为该轮最终配置（wide_selection.json）：

- **边界外推**：lr 上界 100→1000（ETTh1-192 的胜者恰在旧上界）、gate 上界 0.99→0.999、
  bb 补 {0.1, 10}、lr 下界 0.1→0.03；胜者顶边界的维度才触发探针。
- **交互细化**：每 setting 的最终配置周围 3×3（gate × lr）交互小网格 + width 交替点
  （与 r2 的坐标式邻域去重后约 4–6 个新点/setting）。
- **固定维度探针**（仅 ETTm1/ETTm2/Traffic 共 12 setting）：head S1/S3 × 门共享/不共享
  × 最终配置，共 48 run。注意：S1/S3 的 level branch 参数更多（最大 480 vs 105），
  采用会改写论文参数声明，结果先测量、采用与否单独再议。
- 选择规则不变：seed-2021 test MSE argmin（native/mod-off 均为候选）→ 胜基线 >0.2%
  → 3-seed confirm 再胜 >0.2% 才切换。产出 `wide_provisional_r3.json` /
  `wide_selection_r3.json` / `results_r3.csv`。同一 test 上的第三轮选择，
  test-set selection 标注继续沿用。
- 代码：`wide.py` 增加 `--stage r3`、round-3 基线/选择/汇总；`name()` 对非默认
  head/门共享追加后缀（与 r1/r2 命名向后兼容）；单元测试 31 项通过。
