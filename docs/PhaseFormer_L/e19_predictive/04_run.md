# E19 · §4.7 预测力 — 阶段 4：正式实验（阶段 1 统计量）

> 状态：**阶段 1 已完成**；阶段 2（ρ 列）待 E14 的单次 test 读取。

## 1. 启动记录

| 项 | 值 |
|---|---|
| 代码版本 | `0af43e3f Restore the unit-rho branch and keep the result-side cap`（含封顶修复） |
| 脚本 | `scripts/phaseformer_L/e19_predictive_stats.py` |
| 提交方式 | `setsid nohup`，日志 `~/niuyiming/logs/e19_stats.log`；**CPU 作业**（无 GPU） |
| 命令 | `--datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Traffic --horizons 96,192,336,720 --num-workers 4` |
| 产物根 | `research_runs/phaseformer_L_e19_predictive_v1/` |
| 退出码 | **`E19_STATS_EXIT=0`**，`{"event":"finished","rows":28}` |

## 2. 逐 setting 完成情况

28/28 全部完成。训练窗口数（`n_windows`）随 horizon 略变，因为窗口数 = `len(seg) − 720 − H + 1`：

| 数据集 | n_windows（H96 → H720） |
|---|---|
| ETTh1 / ETTh2 | 8,640 → 8,016 |
| ETTm1 / ETTm2 | 33,536 → 33,024 |
| Weather | 35,840 → 35,328 |
| Electricity | 17,584 → 16,960 |
| Traffic | 11,464 → 11,224 |

## 3. 关键产出：三候选的可分离性诊断

`nu_star_diagnostic.json` 与 `dataset_level_statistics.csv` 的结论：

| 候选 `ν` | 正类（Weather / ETTh2 / ETTm2） | 负类（ETTm1 / ETTh1） | 完全分离 |
|---|---|---|---|
| `cycle_level_std` | 0.507 / 0.418 / 0.348 | 0.489 / 0.371 | **否（反序）** |
| `last_cycle_shift` | 0.467 / 0.398 / 0.336 | 0.434 / 0.363 | **否（反序）** |
| **`tau_hat_steps`** | **96.9 / 88.2 / 63.6** | **34.9 / 51.1** | **是**，区间 (51.11, 63.58) |

→ 冻结 `ν = tau_hat_steps`、`ν* = 57.35`（区间中点）。预测：`s=1` ∈ {ETTh2, ETTm2, Weather}（12 setting）、
`s=0` ∈ {ETTh1, ETTm1, Electricity, Traffic}。

## 4. 全 28 setting 的统计量（回填依据）

| dataset | τ̂ (步) | `cycle_level_std` | `last_cycle_shift` | `tau_capped_frac` |
|---|---:|---:|---:|---:|
| Weather | 96.92 | 0.5074 | 0.4675 | 0.0154 |
| ETTh2 | 88.17 | 0.4180 | 0.3984 | 0.0017 |
| ETTm2 | 63.58 | 0.3481 | 0.3365 | 0.0232 |
| Electricity | 54.46 | 0.2350 | 0.2192 | 0.0007 |
| ETTh1 | 51.11 | 0.3715 | 0.3635 | 0.0 |
| ETTm1 | 34.86 | 0.4887 | 0.4344 | 0.0 |
| Traffic | 25.10 | 0.2560 | 0.2124 | ~0 |

（`tau_capped_frac` 为触顶通道占比；Weather 的 1.5% 与 ETTm2 的 2.3% 说明长记忆档有少量通道触顶，
不影响排序。）

## 5. 未完成的部分

**阶段 2（§4.7 的两列 ρ）尚未执行**：它需要 E14 阶段 B 产出的 `results.csv`（每个 setting 的
ΔMSE 与门值 `g`）。脚本 `e19_predictive_power.py` 已就绪并通过合成冒烟（28 行、6 个 ρ、诊断准确率）。

## 6. 收尾

阶段 1 完成、无失败项；阶段 2 待 E14。进入阶段 5/6 时两者一并审校与回填。
