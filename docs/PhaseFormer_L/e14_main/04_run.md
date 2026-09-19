# E14 · §4.2 主结果矩阵 — 阶段 4：正式实验

> 状态：**运行中**（本条在启动时写入，收尾另行追加）。

## 1. 启动记录

| 项 | 值 |
|---|---|
| 服务器 | `yyk03@11.11.18.3`，`~/niuyiming/PhaseFormer` |
| 代码版本 | `f870b1aed Include the A1 row and the Traffic appendix in the E14 default matrix` |
| 提交脚本 | `~/niuyiming/run_e14_main.sh` → 日志 `~/niuyiming/logs/e14_main.log` |
| 启动时间 | 2026-09-19 19:10:42 (+0800) |
| 解释器 | `/home/yyk/yyk03/miniconda3/envs/time/bin/python`（torch 2.6.0 + Lightning 2.6.5） |
| 卡 | `--gpus 0,1,2,3,4,5,6,7`（启动前 `nvidia-smi` 复核 8 卡全空闲，0 MiB / 0%） |
| 命令 | `python scripts/phaseformer_L/e14_main_matrix.py --stage a --verify --gpus 0..7 --output-root research_runs/phaseformer_L_e14_main_v1` |
| 计划 cell | **492**（新训 **411**、复用 **81**） |
| 门 | `{"event": "verify_ok", "reuse_cells_resolved": 81}` |

## 2. 启动时的计划输出（原始日志）

```json
{"event": "planned", "total": 492, "by_arm": {"l_main": {"new": 63, "reused": 21},
 "l_q1_4": {"new": 63, "reused": 21}, "l_q1_8": {"new": 63, "reused": 21},
 "l_rcrf": {"new": 84, "reused": 0}, "phase_only": {"new": 66, "reused": 18},
 "a1": {"new": 72, "reused": 0}}, "gpus": [0,1,2,3,4,5,6,7]}
{"event": "verify_ok", "reuse_cells_resolved": 81}
```

按 `COST_HINT` 降序发车，因此**首批 8 个 cell 全是 Traffic**（最贵先发，避免尾部空卡）：
`l_main__Traffic-{96,192,336}-s{2021,2022,2023}` 等。

## 3. 启动阶段观察到的现象（已判明，非故障）

| 现象 | 判读 |
|---|---|
| 启动后约 3 分钟内 8 个进程处于 `D`（disk sleep）、GPU 占用 1 MiB / 0% | 8 个进程并发读 136 MB 的 Traffic CSV（共享 GPFS `/hpcgpufs/hpchome`），I/O 争用导致初始化慢；`VmRSS ≈ 470 MB` 且持续增长，说明在正常加载 |
| 约 4 分钟后 GPU 显存 1887–3177 MiB、利用率 5–37% | 已进入训练，链路正常 |
| 未出现 `retry` 或 `failed` 事件 | 无失败 |

该现象已在 `REMOTE_SERVER.md` 注意事项的同一类问题里（E3 曾记录 `num_workers` 过载把 load 推到 ~300）；
本轮 `--num-workers 4`、并发 ≤ 8，load average 约 7.7，处于安全范围。

## 4. 收尾约定

1. 全部 cell 结束后，日志末行应为 `E14_MAIN_EXIT=0`，`stage_a_summary.json` 的 `failed` 为空；
2. 逐 run 复核 `metrics.csv` 存在且 `epochs_completed == 30`（非 silent failure）；
3. **阶段 B**（`e14_read_test.py`）在所有 checkpoint 冻结后执行**单次** test 读取，
   写入 `results.csv` 与 `test_read_summary.json`；
4. 之后进入阶段 5（审校）与阶段 6（回填）。

---

## 5. 中断恢复路径（2026-09-19 追加）

调度器（`e14_main_matrix.py --stage a`）的 pending 队列在**内存**中，因此若调度进程本身被杀，
未发车的 cell 不会自动重排。恢复方式：

```bash
cd ~/niuyiming/PhaseFormer
# 直接重跑同一命令即可：runner 带 --resume，已完成格会打印
#   RESUME completed: <run_id>
# 并立即返回，不会重复训练。
setsid nohup ~/niuyiming/run_e14_main.sh > ~/niuyiming/logs/e14_main_retry.log 2>&1 < /dev/null &
```

设计上支持这一点的依据（`scripts/search_phaseformer.py:644-650`）：
完成判据是 `runs/<run_id>/metrics.csv` **存在**；带 `--resume` 时命中即返回，
且 `run_id` 由 config hash 决定、与调度顺序无关。因此重跑是**幂等**的，
只会补齐缺失的 cell。

**运行期间禁止**：对 training 进程会导入的路径做 `git reset --hard`（`src/**`、
`scripts/search_phaseformer.py`）。本轮期间的所有同步都先核对 `git diff --name-status`
不涉及这两处；只新增 `scripts/phaseformer_L/**` 与 `docs/**` 是安全的。

## 6. 实测吞吐（用于外推 ETA）

| 时刻 | done | launched | failed |
|---|---:|---:|---:|
| 19:13（首批发车） | 0 | 8 | 0 |
| 20:19 | **8** | 15 | **0** |

首批 8 个 Traffic cell 在约 **69 分钟**内全部完成，与冒烟实测的单 epoch 138 s × 30 epoch ≈ 69 min 一致。
按此吞吐外推：Traffic 60 个 run ≈ 8.6 h，其余 351 个 run ≈ 6.8 h，E14 阶段 A 合计约 **15 h**。

---

## 7. 基于实测的剩余工期（2026-09-20 01:35 重新核算）

用 `COST_HINT`（服务器 677 份 `metrics.csv` 的历史中位数）对**尚未完成**的 360 个新训格求和，
并对照已完成的 Traffic 格核对量级：

| 数据集 | 待完成格数 | 估计 GPU·h |
|---|---:|---:|
| Electricity | 60 | **28.13** |
| Traffic | 12 | **14.00** |
| Weather | 48 | **13.47** |
| ETTm1 | 72 | 5.88 |
| ETTm2 | 48 | 2.25 |
| ETTh1 | 72 | 2.15 |
| ETTh2 | 48 | 0.66 |
| **合计** | **360** | **66.53** |

→ **约 8.3 h（8 卡并行）**。

**量级核对**：已完成的 48 个 Traffic 格按同一张表估计为 **56.00 GPU·h**，
而实测（19:10 起，01:35 完成 55 个 Traffic）为约 **6.4 h × 8 卡 ≈ 51 GPU·h**，
即估计偏高约 10%，属可接受范围（`COST_HINT` 的 Traffic 值取的是单次冒烟实测外推，
未含并发下的轻微 I/O 争用）。

**E14 阶段 A 总量估计**：56 + 66.5 ≈ **122 GPU·h → 约 15.3 h**，
与阶段 A 启动时（19:10）的估算一致，预计 **次日约 10:30** 结束。

**注意**：`Electricity` 占待完成量的 42%（28.13/66.53），与 E10 记录的
"Electricity-336 占 7 setting 总计算量 42%" 同量级——高通道数设定的计算代价是这一阶段的主要成本。

---

## 8. 实测耗时 vs `COST_HINT`（2026-09-20 03:05，81 个已完成格）

已完成的 81 个格（Traffic 60 + Electricity 21）的 `elapsed_sec` 实测中位数与 `COST_HINT` 对比：

| setting | n | 实测中位 | = 小时 | `COST_HINT` | 比值 |
|---|---:|---:|---:|---:|---:|
| Traffic-96 | 15 | 4095.6 s | 1.138 | 4200 | **0.98** |
| Traffic-192 | 15 | 2972.1 s | 0.826 | 4200 | 0.71 |
| Traffic-336 | 15 | 2584.8 s | 0.718 | 4200 | 0.62 |
| Traffic-720 | 15 | 2639.9 s | 0.733 | 4200 | 0.63 |
| Electricity-336 | 3 | 1273.2 s | 0.354 | 1717 | 0.74 |
| Electricity-720 | 18 | 1284.5 s | 0.357 | 2000 | **0.64** |

**`COST_HINT` 整体偏高约 30–40%**（Traffic-96 除外，比值 0.98）。

### 8.1 原因：早停使 horizon 成为耗时的**弱**预测子

各类 run 的 `epochs_completed` 分布（实测）：

| setting | 观察到的 epoch 数 |
|---|---|
| Traffic-96 | 26, 28, 29, 30 |
| Traffic-192 | 18, 21, 23, 24, 25, 27, 28, 29, 30 |
| Traffic-336 | 16, 19, 20, 21, 22, 24, 28, 30 |
| Traffic-720 | **12–26**, 30 |
| Electricity-336 | 16, 22, 26 |
| Electricity-720 | **11–24**, 30 |

即：**horizon 越小反而越慢**——Traffic-96 普遍跑满 26–30 个 epoch（最慢），
而 Traffic-720 有相当一部分在 12–17 个 epoch 就因验证损失不再改善而早停。
`COST_HINT` 是按 horizon 单调递增假设的（来自历史中位数，且当时未区分早停），
因此对高 horizon 系统性高估。

Traffic-96 的 4096 s 与冒烟实测（138 s/epoch × 30 epoch = 4140 s）**完全吻合**，
说明该档没有早停、`elapsed_sec` 的计量可靠。

### 8.2 对剩余工期的影响

用**实测中位数**替换 `COST_HINT` 中的 Electricity 值（1300 s/run 量级），剩余工作量从
66.53 GPU·h 降到约 **46 GPU·h → 约 5.8 h（8 卡）**，即阶段 A 预计在 **约 08:50** 结束
（而非先前估计的 10:30）。

未测档（Weather、四个 ETT 系）仍沿用 `COST_HINT`，因此该数字仍有向上偏差的余地——
但方向是"更快"，不是"更慢"。

### 8.3 一条工程注意

**已完成的 run 其 `status.json` 会被覆写**为 `{"status": "completed", "completed_at": ...}`，
**丢掉 `started_at`**。因此事后无法从 `status.json` 反推墙钟耗时；
`metrics.csv` 的 `elapsed_sec`（第 33 列）才是可靠来源。本轮最初的测量脚本因依赖
`started_at` 而错误地报告"0 个已完成格"——是**测量方法**的问题，不是产物缺失。
