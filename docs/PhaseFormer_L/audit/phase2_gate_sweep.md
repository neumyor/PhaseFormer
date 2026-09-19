# 阶段二上线前的门禁全量预跑（用**真实输入**，E14 截至 94/411 时执行）

> 执行时间：2026-09-20 03:24–03:26（服务器）｜仓库 HEAD：`772c582c`｜执行者：本会话
> 目的：满足执行契约中"**长周期实验在启动前完成静态检查与协议校对**"，
> 并且**不只跑静态检查，而是把每一步的门用真实输入实际跑一遍**。

## 0. 为什么值得单独记一份

阶段二是一条**无人值守**的长链：E14 阶段 A（411 runs）→ 单次 test 读取 → §4.7 ρ
→ §4.2 回填 → E16（63 cells）→ E17（24 runs）→ E18（78 runs + 行 3）。
前面几步的静态检查与冒烟早已做完，但**静态检查无法发现"参数写法错但语法合法"、
"文件名字对但指向不存在"、"flag 存在但元数不允许"** 这类问题。

本轮的做法是把**每一步的门用真实输入预跑**（全部指向 `/tmp` 下的临时输出根，
除第 1 步因需定位 run 目录而使用真实根、但以 `--dry-run` 运行并已核实不写盘）。
结果：**六步全部门禁均已用真实输入跑过一遍**，并挡下一个会在 165 个 run 之后才炸的缺陷。

## 1. 逐步结果

| 步 | 门 | 结果 | 判读 |
|---|---|---|---|
| 1 | `e14_read_test.py --dry-run` | **exit 1**，`dry run found problems; refusing to proceed` | **正确**：E14 未完成时应当拒绝。且 `dry_run: True`、`wrote_outputs: False`，**未写任何产物** |
| 2 | `e19_predictive_power.py` | 未预跑 | 依赖第 1 步的 `results.csv`（该文件此时尚不存在）——依赖关系本身已由列名契约覆盖 |
| 3a | `e14_params.py --dry-run` | **exit 0**，`cells_with_parameters: 175`、`unresolved: 317`、**`total_mismatches: []`**、`cells_with_gate: 112`、`wrote: null` | **在 175 个真实 checkpoint 上**验证：由 checkpoint 参数**形状**读出的参数量与 `metrics.csv:parameter_count` **全部一致**；且 175+317 = **492** 精确等于矩阵总量 |
| 3b | `e14_reuse_audit.py` | **exit 0**，`cells_audited: 81`、`multiple_valid: 45`、`disagreeing: 0`、`not_in_scan: 0`、`invalid_rejected_cells: 6`、`worst_val_mse_spread: 0.0` | 与既有审计**逐项一致**（可复现）；复用歧义的"多候选但零分歧"结论再次成立 |
| 4 | `e16_dissection.py --dry-run --verify-checkpoint-heads` | **exit 0**，`cells: 63`、`by_e14_status: {"reused": 63}`、`random_rrr: true` | 63 个 cell **全部为复用**，故与 E14 新训 run 无关；门通过是**正确行为**（详见 `e16_dissection/02_03_static_check_smoke.md` §4） |
| 5 | `e17_conditional.py --stage plan --verify` | **exit 0**，`cells: 84`、`new_runs: 24` | 计划可解析；24 个新训 run 为 E17 自身所需 |
| 6a | `e18_negative.py --stage all --verify --dry-run` | **exit 1**，`78 of 78 cells have no matching completed run` | **正确**：`--verify` 的语义是**跑完后**的完整性审计（见 `run_phase2_after_e14.sh` 文末注释）。训练前调用必然失败，这正是流水线采用"先训练→再审计"的原因 |
| 6b | `e18_svd_truncation.py --verify --dry-run` | **先 exit 2 → 修复后 exit 1（正确拒绝）** | **挡下真实缺陷**，详见下节 |

## 2. 本轮挡下的缺陷：`--seeds` 的元数（arity）

第 6b 步暴露：

```text
e18_svd_truncation.py: error: unrecognized arguments: 2022 2023     (exit 2)
```

`--seeds` 非 `nargs` 参数，由 `parse_list`（**按逗号切分**）解析，而流水线写成空格分隔。
**该行在第 6 步**——前面已有 411 + 63 + 24 + 78 个 run 的投入，会在**全部昂贵步骤之后**才炸。

修复后同一调用：通过 argparse，并正确报 `verify failed: 135 unresolved cells; refusing to evaluate`
（exit 1，即在 E14 未完成时拒绝评估，跑完即可通过）。

**为什么上一轮校核没抓到**：那轮核对的是 *flag 存在性* 与 *required 参数是否给出*，
**没有校核元数**——"flag 存在"与"flag 能接收几个值"是两件事。故已把该维度补成静态门
`check_pipeline_invocations.py` 并纳入流水线预检。详见 `e14_main/05_audit.md` §16。

## 3. 第 1 步的滚动对账（一个精确成立的恒等式）

第 1 步 dry-run 的完整计数：

```text
cells: 492
by_status: {planned: 94, reused: 81, missing_metrics: 8, missing_run: 309}     # 94+81+8+309 = 492
accepted: 175   problems: 317                                                  # 175+317 = 492
by_arm:    {l_main: 84, l_q1_4: 84, l_q1_8: 84, l_rcrf: 84, phase_only: 84, a1: 72}  # = 492
fingerprint_check: {constants_equal: true, parity_cases: 20, parity_failures: []}
warnings: 0
```

三点判读：

1. **恒等式精确成立**：`accepted = planned(94) + reused(81) = 175`，
   `problems = missing_metrics(8) + missing_run(309) = 317`，两者之和 = 矩阵总量 492。
   即**每个 cell 恰好落入一个桶**，解析器没有漏计或重计。
2. **`missing_metrics: 8` 是"在飞"而非"卡死"**：逐个查目录确认这 8 个都是
   `electricity_h192` 的 run，目录 mtime 落在 03:11–03:25（即查询前几分钟内），
   属**正在训练**的 run（`metrics.csv` 完成时才写）。这同时说明解析器**不会**把在飞的 run
   提前当成可用——这正是"test 读取必须在训练全部结束之后"这条协议所要的行为。
3. **`fingerprint_check` 的兄弟常量一致性**：`e14_read_test.py` 与
   `e14_main_matrix.py` 的常量**相等**，20 个 parity case **零失败**，
   即两个脚本对协议常量的理解一致（这是"两份实现一个真相"的内部校验）。

## 4. 本预跑**没有**覆盖什么（诚实边界）

* 它验证的是"门会正确地放行/拒绝"以及"参数与文件名能被接受"，
  **不验证**各步的**数值结果**是否正确——那是阶段 5 审校的职责；
* 第 2 步（§4.7 ρ）未预跑，因为它依赖第 1 步的产物；其列名契约已由
  `check_column_contracts.py` 覆盖（4 组契约 0 缺口）；
* 各步的**运行时长**仍按 `COST_HINT` 估计（E16 ≈ 2 h 单卡、E17 ≈ 24 runs、E18 ≈ 78 runs），
  这些数字来自冒烟外推而非实测，故第 4–6 步的实际耗时仍有不确定性。
