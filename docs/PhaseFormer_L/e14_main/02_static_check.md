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
