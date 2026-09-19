# E15 · §4.3 相位补空间维数 — 阶段 4：正式实验

> 状态：**已完成**（2026-09-19）。

## 1. 启动记录

| 项 | 值 |
|---|---|
| 代码版本 | `08018a7cb Fix the stale fine-grid template variable in E15` |
| 提交脚本 | `~/niuyiming/run_e15.sh` → 日志 `~/niuyiming/logs/e15.log` |
| 命令 | `python scripts/phaseformer_L/e15_dimension.py --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Traffic --horizons 96,192,336,720 --seq-len 720 --save-moments --output-root research_runs/phaseformer_L_e15_dimension_v1` |
| 线程限制 | `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`（CPU 作业，与同时段 8 卡 GPU 训练共存，避免争用） |
| 开始 / 结束 | 2026-09-19 19:26:44 → 19:43:26 (+0800)，wall-clock 约 **16 分 42 秒** |
| 退出码 | **`E15_EXIT=0`** |

## 2. 逐 setting 完成情况

28/28 全部完成，无一失败、无重试。日志逐行 `[done] <dataset>-<H>: lam1=… dims90=… PR=… exp_tau=…(cos) a1|cos|=… uvs(1)=… [source] <秒>`。

| 数据集 | 各 horizon 耗时（s） |
|---|---|
| ETTh1 | 1.1 / 1.2 / 1.3 / 1.6 |
| ETTh2 | 1.2 / 1.3 / 1.5 / 2.0 |
| ETTm1 | 1.6 / 2.5 / 3.9 / 7.4 |
| ETTm2 | 1.4 / 1.5 / 1.9 / 2.6 |
| Weather | 3.6 / 6.5 / 10.0 / 16.3 |
| Electricity | 56.7 / 66.6 / 78.4 / 140.6 |
| Traffic | 102.9 / 108.4 / 118.0 / 约 150 |

（Electricity/Traffic 的耗时代价来自 321/862 通道的逐通道流式累加。）

## 3. 产物

```text
research_runs/phaseformer_L_e15_dimension_v1/
  dimension_table.csv          28 行 × 10 列（§4.3 的 6 列 + dataset/horizon/source）
  b1_template_detail.csv       精确 |cos| 与细网格对照（不污染 §4.3 表头）
  leading_direction.csv        与既有 v2 产物同 schema
  optimal_rank_capture.csv     与既有 v2 产物同 schema
  leading_directions.npz       各 setting 的 b_1/a_1 向量
  moments_*.npz                28 个二阶矩文件（szz/szy/syy/…）
  dimension_summary.json       机器可读汇总
  figures/scree_lambda_spectrum.png
  figures/b1_lag_profile.png
  figures/a1_horizon_profile.png
```

三类图（§4.3 要求的 Scree / `b_1` lag 剖面 / `a_1` horizon 剖面）各 240–344 KB，均为有效成图。

## 4. 收尾

阶段 4 无失败项，进入阶段 5 审校。
