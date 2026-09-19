# E19 · §4.7 命题 1 的预测力 — 阶段 1：计划

> 实验单元：**E19**｜回填目标：minipaper **§4.7 表（28 setting × 3 统计量 + 2 列 ρ）**
> 代码：`scripts/phaseformer_L/e19_predictive_stats.py`（阶段 1，训练无关）、
> `scripts/phaseformer_L/e19_predictive_power.py`（阶段 2，需 E14 结果）
> 产物：`research_runs/phaseformer_L_e19_predictive_v1/`

## 1. 目标

§4.7 要求对 **28 个 setting** 计算**训练集**上的跨周期电平统计量
（`cycle_level_std`、`last_cycle_shift`、`τ̂`），并与 §4.2 的 PhaseFormer-L 增益（ΔMSE）及门值 `g`
求 Spearman ρ。其中统计量列可立即产出；两列 ρ 必须等 E14 的 test 读取完成后才能填。

## 2. 为什么阶段 1 必须先于 E14（原设计）

§3.4.3 的开关阈值 `ν*` 只由**已知增益符号的数据集**的**训练集**统计量拟合，不需要任何 §4.2 结果。
但原设计把 `s` 写进**模型定义**，因此 `ν*` 必须在 E14 开跑前冻结。
**2026-09-18 用户裁定 D-5 改变了这一点**：`s` 不进入模型，`ν*` 只决定 §4.7 的**诊断列**，
不影响任何训练或模型选择，因此即使判错也不污染主结果。阶段 1 仍在 E14 之前完成，
以保住"先冻结、后看结果"的时序。

## 3. 定义（关键：继承而非新造）

| 量 | 定义 | 来源 |
|---|---|---|
| `cycle_level_std` | `mean_c std_k(mean_p X[k,p,c])` | D7 描述量，`docs/PhaseFormer_structural_defect_research_narrative.md` §A.6 登记；实现逐字取自 `scripts/run_d7_internal_path_probe.py::features` |
| `cycle_amplitude_std` | `mean_c std_k(std_p X[k,p,c])` | 同上 |
| `last_cycle_shift` | `mean_c │l(K-1,c) − mean_{k<K-1} l(k,c)│` | 同上 |
| `local_diff` / `recent_deviation` / `daily_lag_change` | D7 的另三个窗口描述量 | 同上（本表不用于 §4.7，作为附带产物） |
| **`τ̂`（新增）** | 对每通道的周期电平序列 `l(k,c)` 求 lag-1 自相关 `ρ_c`，`τ_c = −1/ln(ρ_c)`（`0<ρ<1`）、`0`（`ρ≤0`）、封顶 30 周期（`ρ≥1` 或结果超窗）；`τ̂ = P · mean_c τ_c`，单位**步** | 本实验冻结（minipaper 只给了名字，无既有实现） |

- 统计量在 **loader 归一化后的输入张量**上计算（训练集拟合的全局 StandardScaler 之后、模型自身 RevIN 之前），
  与 D7 完全同口径；不做任何再缩放。
- 实现为**逐窗口**计算再对训练窗口取均值；`K = 720/24 = 30` 个周期。

## 4. τ̂ 的已知偏差（**已在代码中冻结并披露**）

`ρ` 只由 K=30 个周期电平估计，向下偏。合成 AR(1) 实测（真实 τ → 估计，单位周期）：

| φ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| 真实 τ | 0.434 | 0.831 | 1.443 | 2.804 | 9.491 |
| 实测 τ̂ | 0.352 | 0.725 | 1.245 | 2.169 | 4.530 |
| 比值 | 0.81 | 0.87 | 0.86 | 0.77 | **0.48** |

结论：`τ̂` 是**单调**仪器，可用于排序与相关（这正是 §3.4.3 与 §4.7 的用途），
但**不得**读作绝对记忆长度。该表已写入 minipaper §4.7 表注。

## 5. 判定口径与阈值

- `ν` 取 `tau_hat_steps`：三个候选里**唯一**能分开已知符号数据集的（见阶段 2 的可分离性诊断）。
- `ν* = 57.35` 步 = 可分离区间 `(51.11, 63.58)` 的**中点**；规则式定义、只用训练集统计量、不看 test。
- 诊断列 `s = 1[tau_hat_steps > ν*]`，**不进入模型**（D-5）。

## 6. 产物

```text
research_runs/phaseformer_L_e19_predictive_v1/
  level_statistics.csv          28 行 × (3+3 统计量及其 sd) + n_windows/train_size
  dataset_level_statistics.csv  7 行（按数据集聚合；s() 在 §3.4.3 中按数据集索引）
  nu_star_diagnostic.json       三候选的可分离性诊断（正类/负类取值、是否完全分离）
  run.yaml                      协议与环境快照（含 reads_test: false）
  predictive_power.csv          阶段 2 产出
  predictive_power_summary.json 阶段 2 产出（6 个 ρ、诊断准确率）
```

## 7. 静态检查清单（移交阶段 2）

1. `compile()` 通过。
2. 单元测试 `tests/test_phaseformer_L_e19_stats.py` 全绿（含 D7 公式对拍、τ̂ 的封顶/无记忆/AR(1) 单调性）。
3. `--dry-run` 打印 28 个 setting 且不读数据。
4. 统计量的**只读训练集**：源码与运行时日志均无 test loader。
5. 28 行无缺、无空值；`run.yaml` 的 `reads_test` 为 false。
6. 与既有 D7（E12，ETTm1-192 单 setting）的口径一致性：三公式逐字一致。
