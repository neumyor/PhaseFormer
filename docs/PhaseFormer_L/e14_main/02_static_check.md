# E14 · §4.2 主结果矩阵 — 阶段 2：静态检查

> 对应 `01_plan.md` §7 的清单。全部在**服务器**上执行（本地无 torch/numpy）。
> 服务器：`yyk03@11.11.18.3`，仓库 `~/niuyiming/PhaseFormer`，conda 环境 `time`。

## 1. 逐项结果

| # | 检查项 | 命令 / 依据 | 结果 |
|---|---|---|---|
| 1 | 语法编译 | `PYTHONDONTWRITEBYTECODE=1 python3 -c "compile(open(p).read(), p, 'exec')"` | **通过**（`e14_main_matrix.py`、`e14_read_test.py`、`e19_predictive_stats.py`、`e15_dimension.py`） |
| 2 | 计划与复用硬门 | `--stage plan --verify` | **通过**：`total=492`、`reuse_cells_resolved=81`、`missing=[]`、`event=verify_ok` |
| 3 | 三条复用链解析数 | manifest `reuse_summary` | **一致**：`phase_only` 18/18、`l_main` 21/21、`l_q1_4` 21/21、`l_q1_8` 21/21；`missing=[]` |
| 4 | 抽查复用格协议 | `config.json` 与白名单逐字段比对 | **一致**（见 §2） |
| 5 | `--dry-run` 状态 | 前 20 个 cell 打印 | **一致**：`l_rcrf__Electricity-720-*`、`phase_only__Electricity-720-*` 为 `new`；`l_main/l_q1_4__Electricity-336-*` 为 `reused` |
| 6 | 阶段 A 无 test 读取 | `arm_command()` 源码审查 + manifest 每条 `command` | **确认**：无 `--evaluate-test`，manifest 声明 `reads_test: false` |

## 2. 复用抽查明细（阶段 2 第 4 项）

| 复用链 | 运行根目录 | 抽查到的关键字段 |
|---|---|---|
| `phase_only` | `top2_direction_retention_v1` | `mechanism=no_residual`、`lookback=720`、`loss=huber`、`max_epochs=30`、`percent=100`、`period=24`；test 证据来自 `results.csv`（`test_read_status=read`） |
| `l_main` | `rank_sweep_2_stage1` 等 E3 系 | `mechanism=weak_residual`、`head_type=shared`、无 `weak_residual_projection`；test 证据内联于 `metrics.csv` |
| `l_q1_4` / `l_q1_8` | 同上 | `head_type=pooled_lowrank`、`pool_factor=1`、`rank=H/4` 或 `H/8`（实测 Electricity-336 → 84/42，ETTh2-96 → 24/12） |

## 3. 本轮静态检查**挡下并修复**的缺陷

| # | 缺陷 | 发现方式 | 修复 |
|---|---|---|---|
| 1 | `build_cells` 对没有复用域的臂（`l_rcrf`/`a1`）用 `reuse[arm]` 直接取值 → `KeyError` | 首次 `--stage plan` | 改为 `reuse.get(arm, {}).get(key)` |
| 2 | `phase_only` 的 12 个复用格解析失败：E8 分阶段协议把 test 读数写在实验汇总 CSV 而非 run 的 `metrics.csv`，原判据 `_has_test_metrics` 过严 | `--verify` 硬门拦截（`missing` 12 格） | 引入 `EXTERNAL_TEST_EVIDENCE`：接受登记在册的外部 test 读数，并要求 `test_read_status ∈ {read, reused}` 与 `val_relative_difference ≤ 1e-3`；解析数 6 → **18** |
| 3 | D-2 只描述了"两种"超参协议 | `e14_read_test` 子代理审查 | 实测确认为**三种** gate 先验（新格 0.2 / `l_rcrf`+`a1` preset 0.5 / 复用格 Stage-0 值），已写入 §4.0 与 `01_plan.md` §3 |
| 4 | minipaper 声称 A1 有 "12 格既有" | 服务器 690 个 run 的 mechanism 全量清点 | 全库 `gold_combo*` **0 个 run**；审计推翻该前提，A1 改为全训 72 runs（用户裁定 D-6） |

## 4. 结论

阶段 2 全部 6 项通过，且硬门在本轮实际拦截了 2 个会在开跑后才暴露的缺陷（复用格解析失败、
`KeyError`）。允许进入阶段 3。

---

## 附录：阶段 B（单次 test 读取）的静态检查

> 代码：`scripts/phaseformer_L/e14_read_test.py`｜计划文档：`04b_test_read_plan.md`

### B.1 计划对账（`--dry-run`，不写任何文件、不读 test）

```bash
python scripts/phaseformer_L/e14_read_test.py \
  --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \
  --output-root research_runs/phaseformer_L_e14_main_v1 --dry-run
```

| 项 | 值 | 判定 |
|---|---|---|
| `cells_in_manifest` | **492** | 与阶段 A 的 `total` 一致 ✓ |
| `cells_selected` | 492 | ✓ |
| `manifest_counts` | `{"new": 411, "reused": 81}` | 与阶段 A 逐项一致 ✓ |
| `reused_cells` | **81** | 全部解析成功，复用格走"核对并复制、**不重读**"路径 ✓ |
| `external_evidence_rows` | **36** | E8 登记的外部 test 读数（18 `phase_only` + 18 `direct_nlinear`）✓ |
| `external_evidence_rejected` | **0** | 无一条证据被拒 ✓ |

### B.2 门行为核对（**非循环依赖**，重要）

首次运行时退出码为 1，报告 `missing_run`（约 800 条）与 `missing_metrics`（16 条）。逐条核对结论：

- `missing_run`：阶段 A **尚未训练**的 `new` 格（E14 仍在跑第一批 Traffic），符合预期；
- `missing_metrics`：`run_dir/metrics.csv` 缺失或为空 ⇒ 该格阶段 A **未完成**。
  源码判据见 `e14_read_test.py:904-908`。

**关键核对：该门不是循环依赖。** 阶段 B 只要求 `metrics.csv` **存在**（阶段 A 的产物），
**不**要求其中已有 test 指标——因为 test 指标正是阶段 B 要写入的东西。若某格已带 test 指标
（意味着阶段 A 误传了 `--evaluate-test`），脚本会复制并**标记告警**，而不是拒绝。

因此：阶段 A 全部结束后该门应转为通过；当前的退出码 1 是"阶段 A 未完成"的正确表现。

### B.3 与阶段 A 的指纹一致性（由实现方在阶段 1 记录）

`fingerprint_report()` 导入 `e14_main_matrix` 并断言 `ARMS`、`EXTERNAL_TEST_EVIDENCE` 与
5 个协议常数**逐项相同**，另跑 20 个合成 config 的 `arm_match` 对拍（0 失败）。
即"哪个 run 属于哪个臂"只在 `e14_main_matrix` 定义一次，阶段 B 不会与阶段 A 漂移。

### B.4 阶段 B 相对 E8 模板修正的 4 个缺陷

| # | E8 的做法 | 为什么有问题 | 阶段 B 的做法 |
|---|---|---|---|
| 1 | 先读 test，再校验 validation 复现 | 会在 val 不匹配时**白白消耗一次 test 读取** | **先 val 门后 test 读**；不通过则 status=`rejected` 且**不读 test**、不报数字 |
| 2 | 幂等判据用了自己不会写的列 | 重跑会**重复读 test** | 每格 `test_read/<key>.json` + 已消费状态白名单，重跑不重读 |
| 3 | 硬编码 `attempts/001` | 复用链上有 `attempts/002` 的真实 run（已确认） | 改为 glob `attempts/*/checkpoints/` |
| 4 | 未记录 `val_mse` 的 run 被静默接受 | 无法校验复现 | 视为 `rejected`，另拒 NaN/0 的 val 与非有限复算值 |

---

## 附录 B.5：阶段 B 的解析器进度对账（滚动核对，2026-09-20）

阶段 B 的 `--dry-run` 会报告它能否解析每一个 cell。它在 E14 运行期间自然"失败"
（未训练的新格当然找不到 run），因此不能只看退出码——要看**它是否恰好识别出全部未完成格**。

在阶段 A 完成 88/411 时实测：

```json
{"event": "finished", "dry_run": true, "cells": 492, "accepted": 161, "problems": 331,
 "by_status": {"planned": 80, "missing_metrics": 8, "reused": 81, "missing_run": 323},
 "by_arm": {"l_main": 84, "l_q1_4": 84, "l_q1_8": 84, "l_rcrf": 84, "phase_only": 84, "a1": 72}}
```

对账：

| 量 | 值 | 含义 |
|---|---:|---|
| `cells` | 492 | 与阶段 A manifest 一致 ✓ |
| `accepted` | **161** | = 80（`planned`，已完成待读 test）+ 81（`reused`） |
| `problems` | **331** | = 8（`missing_metrics`，run 目录已建但 `metrics.csv` 未落盘，即在跑）+ 323（`missing_run`，尚未发车） |
| **80 + 8 + 323** | **411** | **恰好等于声明的 411 个新训格** ✓ |
| `by_arm` | 84×5 + 72 | `a1` 为 72（不含 Traffic），其余各 84 ✓ |

即解析器**完整覆盖**每个新格：已完成的可解析、在跑的识别为"未落盘"、未发车的识别为"无 run"，
三者相加不重不漏。这是"阶段 B 能在阶段 A 一结束时正确接手"的直接证据，
而不是等它跑完再看退出码。

另一项同时被确认的事：`--dry-run` 先跑**指纹一致性检查**，输出
`constants_equal: true, parity_cases: 20, parity_failures: [], differences: []`——
即阶段 B 与阶段 A 的 `ARMS`、`EXTERNAL_TEST_EVIDENCE` 与 5 个协议常数**在真实仓库上逐项相同**，
20 个合成 config 的 `arm_match` 对拍全部一致。
