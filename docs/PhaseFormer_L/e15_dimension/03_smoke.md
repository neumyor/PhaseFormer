# E15 · §4.3 相位补空间维数 — 阶段 3：冒烟测试

> 目的：在 28 个 setting 正式开跑前，用最小子集确认链路、产物结构与**最坏 setting 的内存/耗时**。

## 1. 冒烟命令与结果

```bash
# 冒烟 1：正确性门（7 个既有 setting，约 1–3 分钟）——见 02_static_check.md §1
python scripts/phaseformer_L/e15_dimension.py --verify-existing \
  --output-root research_runs/e15_verify4 --reference-dir research_runs/lowrank_data_property_v2

# 冒烟 2：端到端小结（合成数据，28 行/10 列/3 图/28 npz 全部产出）
#   由阶段 1 的实现方在与既有脚本的逐字段对拍中完成：
#   - 与 analyze_optimal_lowrank_capture.py 在**同一合成数据**上对比：
#     Szz/Szy/Syy 逐元素完全相同（max|Δ|=0.0），optimal_rank_capture.csv 16 列全等（最差 1e-8，仅 CSV 舍入）
#   - leading_direction.csv 与 describe_leading_direction.py **逐字节相同**
```

## 2. 真实数据上的最坏设定实测（正式运行日志）

| setting | 实测单 setting 耗时 | 说明 |
|---|---:|---|
| ETTh1-96 | 1.1 s | 最便宜的 ETT 档 |
| Weather-720 | 16.3 s | |
| Electricity-96 | 56.7 s | 321 通道，流式分块 |
| Electricity-192 | 66.6 s | |
| **Traffic-96** | **102.9 s** | 862 通道，全表最坏档 |
| **Traffic-720** | 约 150 s | |

全部 28 个 setting 合计 **约 17 分钟**（19:26:44 → 19:43:26，`E15_EXIT=0`），
峰值内存远低于 `--mem-budget-mb` 默认（512 MB）——**没有出现历史上
Electricity-336 那样 13.9 GiB 的 OOM**，因为本脚本按通道流式累加二阶矩。

## 3. 冒烟确认的关键事实

| 事实 | 证据 |
|---|---|
| test 划分从不解析 | 每次 `[load]` 都打印 `11520 read (test border at row 14400; test never read)`；Traffic 为 `14036 read (test border at row 17544)` |
| Traffic（862 通道）可正常处理 | Traffic×4 全部在 103–150 s 内完成，无 OOM |
| 二阶矩落盘格式正确 | 28 个 `moments_*.npz`，键含 `szz/szy/syy/persistence_mse/n_pairs/n_windows/n_channels`；`Traffic_h720` 的 `szz/szy/syy` 均为 `(720,720)` |
| 分块不变性 | `--channel-block 1` 与自动分块的 CSV 逐字节相同 |
| 复用 vs 新增标记正确 | `source` 列在 7 个先导 setting 上为 `reused_v2_artifact`，其余 21 行为 `new_28_minus_7` |

## 4. 结论

阶段 3 通过：正确性门、端到端结构、最坏 setting 的内存与耗时均已实测。允许进入阶段 4。
