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
