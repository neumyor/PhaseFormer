# PhaseFormer-L Golden-Search（test-set selection，用户指令 2026-09-20）

> **性质**：**test-set selection 搜索**（用户明示指令），非盲测。
> 本文是该搜索的预注册计划与过程记录；所有产物必须带 `test-set selection` 标注，
> 任何后续引用不得表述为盲测或无偏泛化估计。
>
> 执行脚本：`scripts/phaseformer_L/golden_search.py`
> 产物：`research_runs/phaseformer_L_golden_search_v1/`

## 0. 用户指令（2026-09-20，逐条记录）

| # | 问题 | 用户裁定 |
|---|---|---|
| Q1 | 目标锚点 | **两条都报（vs Golden 与 vs phase_only），以"双指标超过 Golden"为达标判据** |
| Q2 | 搜索深度 | **深度搜索**（720 runs 级） |
| Q3 | 允许动的范围 | **超参数（gate_init、lr）+ `pooled_lowrank` 的不同 rank**；不改模型代码 |
| Q4 | 选择依据 | **明确要求按 test 集选择最佳组合** |
| Q5 | 基线参照 | **直接用 E14 已有的 phase_only / l_main**（其 test 指标已在 E14 阶段 B 读过一次，属已暴露） |
| Q6 | 达标口径 | **三个 seed 中的 best 即可**（逐 seed 取最优，非 3-seed 均值） |
| Q7 | 门压小后达标 | **算数，标注即可**（`gate ≤ 0.05` 判为 gate-shrunk；此时修正器近似关闭，赢的是相位主干） |

**目标**：8 个 setting 中 **≥ 4 个**实现双指标（MSE 与 MAE）超越 Golden。

## 1. 搜索对象：8 个未胜过 Golden 的 setting

来自 §4.2 主表（main-24，Traffic 附录除外）。差距为 E14 `l_main` 相对 Golden 的三 seed 均值：

| setting | ΔMSE | ΔMAE | 备注 |
|---|---:|---:|---|
| ETTh1-96 | +2.66% | +4.04% | s=0；phase_only 自身 +0.67%/+1.23% |
| ETTh1-192 | +3.16% | +4.10% | s=0；phase_only +1.93%/+1.71% |
| ETTh1-336 | +3.03% | +3.34% | s=0；phase_only **+3.98%**/+2.51% |
| ETTm1-96 | +4.40% | +2.48% | s=0；phase_only +3.22%/+2.08% |
| ETTm1-192 | +4.75% | +2.25% | s=0；phase_only +2.30%/+0.63% |
| ETTm1-336 | +3.03% | +1.54% | s=0；phase_only +0.36%/+0.04% |
| ETTm1-720 | +1.14% | +0.88% | s=0；phase_only +0.74%/+0.68% |
| Electricity-96 | +0.14% | +0.77% | s=0；**l_main 优于 phase_only**（−0.96%/−0.03%），最接近 Golden |

**预判（写在前，事后对账）**：
1. Electricity-96 最可能达标（差距 <0.8%）。
2. ETTh1/ETTm1 的 7 格中，phase_only 自身落后 Golden 0.36%–3.98%（环境差），
   且 l_main 在这些格**劣于** phase_only。若环境差无法被搜索补上，
   这些格唯一可能赢的组合是"门压到近零"（Q7 情形）——但那相当于 phase_only，
   而 phase_only 本身落后 ⇒ **4/8 目标在这些格上可能结构性不可达**。
   这不是搜索失败，是差距不在"修正器超参"里。如实报告。

## 2. 搜索空间（冻结）

| 轴 | 取值 |
|---|---|
| `weak_period_residual_gate_init` | 0.02, 0.05, 0.10, 0.20, 0.35, 0.50 |
| `learning_rate` | 1e-4, 3e-4, 1e-3 |
| head | `shared`（稠密）；`pooled_lowrank` rank = H/32, H/16, H/8, H/4 |
| 固定项 | lookback 720、period 24、Huber、30 epochs（best-val 早停）、percent 100、mechanism `weak_residual` |

- 单 setting 组合数 = 6 × 3 × 5 = **90**；8 setting × 90 = **720 runs**（seed 2021）。
- **选择依据 = test 指标**（Q4）：每个 run 训练后由 runner 的 `--evaluate-test`
  恢复 best-val checkpoint 并读一次 test——与 E14 阶段 B 同一机制。
- 阶段 2：每个 setting 的 stage-1 最优组合补 seed 2022/2023（≤16 runs），
  最终取 **3 seed 中的 best**（Q6）。
- 成本（E14 §12 实测）：ETTh1 ≈70 s/run、ETTm1 ≈250–390 s/run、
  Electricity-96 ≈1310 s/run ⇒ stage-1 ≈ **69 GPU·h ≈ 8.6 h 墙钟（8 卡）**，
  stage-2 ≈ 1.5 GPU·h。

## 2b. 第二轮预算（2026-09-20 用户授权："预算耗尽后允许再加同样规模"）

**预注册（在第一轮结果出来之前写下，避免事后凑格）**。第一轮 = 720 runs（huber）。若未达 4/8，
第二轮按下列**已冻结**的改动重跑，规模同为 720 runs：

| 轴 | 第一轮 | 第二轮 | 依据 |
|---|---|---|---|
| loss | huber | huber + **mae** | 8 格中 **5 格 MAE 缺口 ≥ MSE 缺口**（ETTh1-96/192 的 MAE 缺 4.04/4.10% vs MSE 2.66/3.16%）；runner 已实现并暴露 `--loss {mse,mae,smae,huber,smape}`，属训练超参，不改模型代码 |
| lr | 1e-4, 3e-4, 1e-3 | 3e-4, 1e-3, **3e-3** | Electricity-96 首 48 格实测：lr=1e-3 组（0.1285–0.1292）**一致优于** lr=1e-4 组（0.1320–0.1328）约 **3%**，1e-4 在全部 5 个 head 上都是最差档 ⇒ 真正的甜点可能在 ≥1e-3 |
| gate | 6 档含 0.02/0.05 | 同 6 档 | 保留 gate-shrunk 通道（Q7） |
| head | dense + r∈{H/32,H/16,H/8,H/4} | 同 5 档 | r24/r12 在 Electricity-96 上未显示优势，但 ETT 系列证据不足 |

两轮**互补而非重复**：第二轮把 lr 上界推到 3e-3、并加入 MAE 目标，覆盖第一轮
"lr 偏低 + 只优化 MSE" 这两个已被首 48 格实测暴露的盲区。cell_id 对 huber 保持原样
（无后缀）、mae 加 `_mae` 后缀，故两轮产物不冲突，`--stage search` 幂等续跑。

**报告规则（第二轮同样适用）**：若最终达标格来自第二轮，表中必须同时标注
`loss=mae` 与 `test-set selection`；`gate ≤ 0.05` 仍按 §4.2 标注 `gate_shrunk`。

## 3. 阶段与命令

```bash
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
# 0) 静态核对
$PY scripts/phaseformer_L/golden_search.py --stage plan
# 1) 冒烟（2 个便宜格，1 epoch，含 --evaluate-test 全链路）
$PY scripts/phaseformer_L/golden_search.py --stage smoke
# 2) 全网格（seed 2021，8 卡）
$PY scripts/phaseformer_L/golden_search.py --stage search --gpus 0,1,2,3,4,5,6,7
# 3) 选择 + 补 seed
$PY scripts/phaseformer_L/golden_search.py --stage select
$PY scripts/phaseformer_L/golden_search.py --stage confirm --gpus 0,1,2,3,4,5,6,7
# 4) 终表与判定
$PY scripts/phaseformer_L/golden_search.py --stage final
```

产物（均在 `research_runs/phaseformer_L_golden_search_v1/`）：
`runs/<cell_id>/`（含 metrics.csv）、`_logs/*.log`、
`stage1_all_rows.csv`（720 行全量选择轨迹）、`stage1_winners.json`、
`final_selection.json|csv`（含 `gate_shrunk` 标注与达标计数）。

## 4. 判定与报告规则（冻结）

1. 达标 = 该 setting 的最优组合在**某一 seed** 上 test MSE 与 test MAE **同时**低于 Golden（Q6）。
2. `gate ≤ 0.05` 的获胜组合标记 `gate_shrunk=true`，报告时必须注明
   "该组合的修正器近似关闭，胜利来自相位主干"（Q7）。
3. 同时报告相对 E14 `phase_only` / `l_main` 的差距（Q1 两条都报）。
4. **全部数字标注 test-set selection**；`stage1_all_rows.csv` 保留完整选择轨迹。
5. 未达 4/8 时如实报告"未达目标"，并按 §1 预判归因（环境差 vs 模型差）。

## 5. 服务器纪律（REMOTE_SERVER.md 摘要）

- 同步：bundle 上行 → `git fetch origin` → `checkout` → `reset --hard FETCH_HEAD`；
  **启动前确认无运行中任务**；GPU 只用全部 8 卡时需先 `nvidia-smi` 确认空闲。
- 解释器：`/home/yyk/yyk03/miniconda3/envs/time/bin/python`。
- 产物只写 `research_runs/phaseformer_L_golden_search_v1/`，不碰他人目录。
- 中断恢复：`--stage search` 幂等（已有 metrics.csv 的 run 自动跳过）。

## 6. 执行记录（追加式）

| 时间 | 事件 |
|---|---|
| 2026-09-20 16:2x | 计划与脚本登记（本文件 + `golden_search.py`）；本地 `--stage plan` 通过：720 runs、69.1 GPU·h、8.6 h 墙钟预估 |
| 2026-09-20 16:53 | 首次冒烟 **失败**：驱动把 runner 输出目录传错（runner 会在 `--output-dir` 下再建一层 `runs/<run_id>/`），metrics 找不到 → 修复为"每格一个输出目录 + 两级 glob"（`90a5544`、`a14a8b4`）；**overrides 落地已核**（冒烟 config 实测：gate 0.05/shared、gate 0.35/pooled_lowrank r=6、lr 均正确） |
| 2026-09-20 16:59 | 二次冒烟：两个 1-epoch 格全链路（训练 + `--evaluate-test`）跑通、metrics 可读、幂等跳过生效；**但 1-epoch 产物必须清除**，否则会被当成已完成格污染正式搜索 |
| 2026-09-20 17:01 | **正式启动**：删除冒烟 runs → `nohup ... --stage search --gpus 0..7`，HEAD `a14a8b4` 记入 `~/niuyiming/logs/golden_search_HEAD.txt`；启动后核验：进程在飞、8 卡各 ~2 GB/30–50% 利用率、stage1.log 首批 8 格全部为 Electricity-h96（按"贵者先行"调度） |
| | （待补：阶段完成时间与实测墙钟、stage-1 选择结果、confirm 与 final 判定） |
