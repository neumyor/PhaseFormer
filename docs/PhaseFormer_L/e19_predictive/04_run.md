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

## 5. 阶段 2：§4.7 的两列 ρ（2026-09-20 07:49，**已完成**）

阶段 2 需要 E14 阶段 B 产出的 `results.csv`（每 setting 的 ΔMSE 与门值 `g`），故排在阶段二第 2 步。
**观测运行时间：`07:49:07 → 07:49:08`（1 秒），`exit 0`**（第 1 步 07:12:33 起、36.5 min 完成，故第 2 步一起步就拿到了输入）。

**实测（`predictive_power_summary.json`）**：

| 统计量 | vs ΔMSE | n | p | 与 `g` | n | p |
|---|---:|---:|---:|---:|---:|---:|
| `cycle_level_std` | **−0.039** | 28 | 0.842 | 0.183 | 21 | 0.427 |
| `last_cycle_shift` | **−0.139** | 28 | 0.480 | 0.303 | 21 | 0.182 |
| `tau_hat_steps`（`τ̂`） | **−0.750** | 28 | **4.3e−06** | 0.325 | 21 | 0.151 |

**结论**：三候选里**只有 `τ̂` 与增益显著相关**（ρ = −0.750，p = 4.3e−06），
且**符号与 §4.7 注 4 的预登记一致**（τ̂ 越长 ⇒ 越需要电平通道 ⇒ ΔMSE 越负）；
另两个候选在注 1 里已被判"否（反序）"，其与 ΔMSE 的相关也**接近 0 或不显著**（p 均 > 0.4）
⇒ "**不可用作诊断列**"在**两条独立口径上一致**（注 1 的类别可分离性 + 本节的 28-setting 相关）。

**两处口径必须写清，否则数字会被错读**：

1. **`vs g` 三列的 n = 21，不是 28。** 原因已核到具体格子：门值来自 E14 `results.csv` 里 `l_main` 行的
   `gate_value` 字段，而**恰好 7 个 setting 的 `l_main` 行是 Stage-0 复用格、其来源证据没有记门值**——
   这 7 个正是 §4.2 表里"**L=复用 3/3**"的那 7 行（ETTh2-96、ETTh2-720、ETTm2-96、ETTm2-192、
   Weather-96、Weather-192、Electricity-336），28 − 7 = 21 ✓ **与论文表可逐行对上**。
   故 `vs g` 与 `vs ΔMSE` 两组**样本量不同，不可直接比较**。
2. **`settings_without_data` 为空**（28/28 都有数据）⇒ 阶段 2 **没有**因缺数据而丢格子。

**一条独立交叉验证（顺带得到）**：summary 里的 `diagnostic_accuracy` 报
**28 setting / hits 22 / misses 6**（`hit_rate 0.7857`），其 `missed_settings` 是
ETTh1-336、ETTh1-720、Electricity-96/192/336/720 —— 与 §4.2 的 `claims.json.B.diagnostic_misses`
**逐项相同**。两个**不同工具**（`e19_predictive_power` 与 `e14_writeback`）独立算出同一组 6 个 miss ✓。

## 6. 收尾

阶段 1 与阶段 2 **均已完成、无失败项**；阶段 5 审校见 `05_audit.md`，阶段 6 回填见 `06_writeback.md`。
