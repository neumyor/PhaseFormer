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

## 9. 运行中的两项体检与进度（2026-09-20 06:1x 追加，细节见指针）

本节只是**索引**，避免读阶段 4 文档的人不知道那两项体检的存在；细节与数据在 `05_audit.md` 与排期文档里，
**不在此重复**。

| 项 | 内容 | 详情位置 |
|---|---|---|
| 完成 run 的**统计体检**（n=176） | `epochs_completed` 13–30、中位 18（133 早停 / 43 跑满 30）；各数据集 `val_mse` 量级合理；**同 setting 跨 3 seed 的 `parameter_count` 0 处不一致**（"run 与 cell 配对正确"的独立旁证）；**0 行已带 test 指标** | `05_audit.md` **附 2** |
| 主张 A 的**配对方向早期信号**（val-only，**不是判定**） | 27 个两臂都完成的 (setting, seed)：中位 **−0.70%**、17/27 落在 ±1% 内、1/27 差于 1% 以上 ⇒ **无系统性接线错误的迹象**；判定仍只由 `claims.json` + 第 7 步审计给出 | `05_audit.md` **附 3** |
| **当前进度与投影** | 06:13 实测 210/411 完成、218 个 run 目录（与已发布 cell **严格 1:1**）、**0 retry / 0 failed**；剩余 4.8 GPU·h ⇒ **ETA 约 06:52**；接阶段二（§9.1.5 更正后 8.6–10.1 h）⇒ 全链约 **15:28–16:58** | 排期 §9.1.4 刷新行、§9.1.5 |

**收官已补（2026-09-20 07:1x）**：

- **`E14_MAIN_EXIT=0`**；`END: 2026-09-20T07:09:11+0800`（`START` 为 2026-09-19T19:10:42+0800
  ⇒ 墙钟 **11 h 58 min 29 s**）。启动时 `HEAD: f870b1ae`。
- **411/411 全部完成**：411 个 `new` 格全部解析成功；run 目录**恰为 411**、日志 `done` 事件 411、
  **0 retry / 0 failed**。`dirs == 411` 这条**精确等式**同时是阶段二自动放行的判据（排期 §10.2）；
  阶段二于 **07:11:04** 放行、**07:12:33** 进入第 1 步，四项预检（列契约 / 调用元数 / 阶段 A 不变量 / 消费者契约）全部通过。
- **`audit_e14_stage_a.py --require-complete` 判决：`states {'ok': 411}`、`failures: 0`**
  （411 个 new 格逐一满足八条不变量，另有 81 个复用格按设计不计入）。
  **判决取自阶段二预检自己跑出的 `phase2_stage_a_audit.json`**（07:11:06 那次），原因见下面的"有效性窗口"。
- §4 各格最终的 `来源/披露` 取值分布**仍待第 3 步**（`e14_writeback.py`）产出后回填，此处不预填。

> **一条有效性窗口（重要，免得后来者把它当成重大事故）**：`audit_e14_stage_a.py` 的八条不变量里有一条是
> "**阶段 A 期间没有任何 run 读 test**"。**第 1 步（阶段 B 的单次 test read）一启动，这条就必然被违反**——
> 那不是协议被破坏，而是该工具的**判定窗口只到第 1 步启动为止**。
> 因此"收官后再跑一次拿最终判决"是**错的做法**，我一度正打算这么做；正确做法是引用**预检那一次**的输出。
> 若在阶段 B 期间或之后重跑，它会因新出现的 test 指标而报失败，**看起来像重大事故，实际是时点问题**。

## 12. 收官成本模型：411/411 实测的 `elapsed_sec` 与 `epochs_completed`（2026-09-20 07:13）

§8 用的是 81 格的局部样本；这里是**全部 411 格**（每格都有 `metrics.csv`）的最终实测。
字段取自 `metrics.csv` 的 `elapsed_sec` 与 `epochs_completed`（**不用 mtime**，见 §8.3）：

| dataset | n | 实测中位 elapsed | 中位 epoch | **秒/epoch** | epoch 区间 |
|---|---:|---:|---:|---:|---|
| Traffic | 60 | 3047 s | 24 | **124.4** | 12–30 |
| Electricity | 63 | 1311 s | 21 | 62.4 | 11–30 |
| Weather | 48 | 705 s | 18 | 39.1 | 10–30 |
| ETTm1 | 72 | 220 s | 22 | 10.0 | 9–30 |
| ETTm2 | 48 | 92 s | 10 | 8.8 | 9–30 |
| ETTh1 | 72 | 67 s | 19 | 3.5 | 12–30 |
| ETTh2 | 48 | 40 s | 15 | 2.6 | 11–30 |

**两个可直接引用的结论**：

1. **`epochs_requested` 全为 30，而 `epochs_completed` 从 9 到 30 不等 ⇒ 早停是耗时差异的主要来源**，
   §8.1 在 81 格上得到的这条结论在 411 格上依然成立（并解释了 §8 里"高 horizon 反而更快"的现象）。
2. **秒/epoch 与 `train_size` 同量级递增**（ETTh 约 7201–7825 行 → 2.6–3.5 s；
   ETTm 约 33121–33745 行 → 8.8–10.0 s；两对**同规模的数据集秒/epoch 几乎相同、总时长却差 2.3 倍**
   （ETTm1 220 s vs ETTm2 92 s），差别**只在** `epochs_completed`（22 vs 10）——
   这是"**用行数估总时长会错、按 epoch 计价才对**"的直接证据。
   （注意反例：Traffic 训练集只有约 1.1 万行却达 124.4 s/epoch，可见秒/epoch 还随**通道数**走，
   故秒/epoch 应作为**逐 setting 常数**使用，不要外推到别的数据集。）

**对后续步骤的用途**：第 4/5/6 步（E16/E17/E18）的排期用的是各自从 E14 `COST_HINT` 镜像来的成本表；
本表给出的是**真实完成的 E14 成本**，可作为"E16 的 63 格前向、E17 的 24 次训练、E18 的 78 次训练"
耗时预估的**校准依据**，但**不能**直接替代——那些是不同的工作量（E16 只做前向）。


## 10. 训练代码在整张矩阵上**逐字节未变**（2026-09-20 核实，可供论文协议部分引用）

一个审稿式的问题："这 411 个格子是**同一版代码**训出来的吗？"本轮把它核到字节级：

| 检查 | 结果 |
|---|---|
| 启动提交（取 `e14_main.log` 里启动器自己打印的 `HEAD:` 行） | `f870b1a` — "Include the A1 row and the Traffic appendix in the E14 default matrix" |
| 启动至今变动的文件总数 | **77**（`docs/` 43、`scripts/` 33、`tests/` 1） |
| 其中落在**训练入口或模型代码**上的 | **`scripts/search_phaseformer.py` 与 `src/**`：0 个**（`git diff --name-only f870b1a..HEAD -- scripts/search_phaseformer.py src/` 输出为空） |
| 训练入口是否从 `scripts/` 里导入任何模块 | **没有**（`grep -E "^(from\|import) scripts"` 无命中；它只导入 stdlib、第三方与 `src.*`） |
| 唯一落在 E14 代码路径上的改动 | `scripts/phaseformer_L/e14_main_matrix.py`（**+42 行、无删改**：两个合法性常量、`_protocol_ok` 的两条 gate/lr 越界检查、复用索引里多记 `gate_init`/`learning_rate`）。它对 **stage B 的指纹门**做过**实跑**核验：`constants_equal: true, differences: [], parity_cases: 20, parity_failures: []`，且该步 dry-run `accepted: 1, problems: 0, wrote_outputs: false` ⇒ **不影响**；对**正在跑**的训练亦无影响（启动器在启动时已导入其内存版本，且它写出的 manifest 本就排除了那批污染 run） |

**结论**：**411 个格子由同一份字节相同的训练代码（`scripts/search_phaseformer.py` + `src/**`）训出**
——启动后改动的 33 个脚本**全部属于阶段二/分析层**（它们只在 E14 结束后运行）与预检/回填工具。
这条可以作为论文协议部分的事实陈述，而不只是"我们注意过"。

> **方法论（本轮的价值所在）**：这不是"顺手看一眼"，而是先问"**有没有哪个我改过的文件其实在跑着的实验的代码路径上**"，
> 再按"启动提交 → HEAD"的**字节级 diff** 回答。发现一个（`e14_main_matrix.py`）之后**没有靠推理**下结论，
> 而是把它的消费者（stage B 的指纹门）**实跑**一遍。若当时只推理"应该是纯增量、应该没事"，
> 就会漏掉"万一改了既有常量 ⇒ 第 1 步拒绝启动"这一支。

## 11. 协议在 411 个命令上**逐项一致**（2026-09-20 06:18 实测，可直接引用）

§10 回答"是不是同一版**代码**"；§11 回答"是不是同一套**协议**"——取 manifest 记录的 411 条命令逐项统计：

| flag | 取值分布 | 判读 |
|---|---|---|
| `--stage` | `confirm` × 411 | 一致 ✓ |
| `--lookback` | `720` × 411 | 一致 ✓ |
| `--period` | `24` × 411 | 一致 ✓ |
| `--max-epochs` | `30` × 411 | 一致 ✓ |
| `--loss` | `huber` × 411 | 一致 ✓ |
| `--percent` | `100` × 411 | 一致 ✓ |
| `--num-workers` | `4` × 411 | 一致 ✓ |
| **`--evaluate-test`** | **缺失 × 411** | **阶段 A 一格都没读 test** ✓（与 `audit_e14_stage_a.py` 的逐格结论、以及 176 格实测的 `test_mse` 全空三处独立一致） |
| `--seed` | 2021/2022/2023 **各 137** | 三 seed 完全均衡（411 = 3 × 137）✓ |
| `--mechanism` | `weak_residual` 189、`rcrf_nlinear_plain` 84、`no_residual` 66、`gold_combo_reliability_s2` 72 | 与 manifest 的按臂 new 计数**逐项吻合**（189 = 3 臂 × 63）✓ |
| `--require-cuda` / `--resume` | 两者都在 411 条里出现 | ✓（本表把它们显示成"取到下一个 flag"是我脚本对 store_true 的读法所致，非产物问题） |

**从 `--overrides` 反推的臂超参**（三类的计数与 manifest 的按臂计数完全一致）：

```text
x222  head=None            gate=None  lr=1e-3     # phase_only 66 + l_rcrf 84 + a1 72
x126  head=pooled_lowrank  gate=0.2   lr=1e-3     # l_q1_4 63 + l_q1_8 63
x 63  head=shared          gate=0.2   lr=1e-3     # l_main 63
```

⇒ **所有新训格一律 gate 0.2 / lr 1e-3**（§4.0 冻结的 D-2 新格超参）✓；
即 §4.0 表注所述"**三种 gate 先验**"中，0.5 那一种**只可能来自复用格**——与"81 个复用格保留 Stage-0 冻结值
（0.5 或 0.2）"的披露相互印证（实测复用格的 `source.gate_init` 为 0.5 共 57 格、0.2 共 24 格）。
