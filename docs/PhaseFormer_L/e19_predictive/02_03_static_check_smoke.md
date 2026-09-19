# E19 · §4.7 预测力 — 阶段 2/3：静态检查与冒烟

> 阶段 2 与阶段 3 的判据见 `01_plan.md` §7。全部在服务器（`conda time`）执行。

## 1. 阶段 2：静态检查

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `compile()` | 通过 |
| 2 | 单元测试 | **14/14 通过**（`tests/test_phaseformer_L_e19_stats.py`） |
| 3 | `--dry-run` | 打印 4 / 28 个 setting 计划，不加载数据 |
| 4 | 只读训练集 | `data_provider(..., "train")` 唯一入口；产出 `run.yaml` 记录 `reads_test: false` |
| 5 | 28 行完整性 | 见 `04_run.md`（28/28，无空值） |
| 6 | 与 D7 口径一致 | `window_descriptors()` 逐字复刻 `run_d7_internal_path_probe.py::features` 的三个公式；单测对拍通过 |

### 1.1 单元测试覆盖的内容

| 测试 | 断言的性质 |
|---|---|
| `test_keys_and_shapes` / `test_batch_dimension_is_preserved` | 输出契约 |
| `test_rejects_wrong_lookback` | 输入校验（720 必须整除 24） |
| `test_cycle_level_std_matches_registered_formula` | 与 §A.6 注册公式逐值一致（6 位小数） |
| `test_last_cycle_shift_matches_registered_formula` | 同上 |
| `test_zero_level_variation_gives_zero_cycle_level_std` | 退化输入 |
| `test_linear_ramp_saturates_the_cap` | 单调线性电平必须**封顶**而不是发散 |
| `test_near_unit_rho_is_also_capped` | **回归测试**：近单位 ρ（0.99999）也必须封顶 |
| `test_tau_hat_never_exceeds_the_window_in_steps` | τ̂ ≤ 720 步（φ ∈ {0.5,0.9,0.99}） |
| `test_alternating_level_has_no_memory` | ρ=−1 ⇒ τ̂=0 |
| `test_ar1_level_recovers_the_known_timescale` | 有偏但落在已知带宽内 |
| `test_ar1_tau_is_monotone_in_the_true_memory` | **单调性**（§4.7 排序所依赖的性质） |
| `test_flat_level_is_reported_as_non_finite_not_crashing` | 零方差电平返回 NaN 而非崩溃 |

## 2. 阶段 3：冒烟

```bash
python scripts/phaseformer_L/e19_predictive_stats.py \
  --output-root research_runs/phaseformer_L_e19_smoke \
  --datasets ETTh2,Traffic --horizons 96,720 --max-windows 64
```

结果：4 个 setting 全部产出，wall-clock **8.26 s**，峰值内存 951 MB。用于验证：
最高通道数（Traffic 862）与最贵档（H=720）都能在单进程内跑通。

## 3. 阶段 2/3 挡下的缺陷（**2 个**）

### 3.1 τ̂ 的封顶只作用于 `ρ ≥ 1`，近单位 ρ 会溢出

首轮冒烟报出 `ETTh2-720 τ̂ = 2176.6 步`，**超过 720 步的窗口长度**。
原因：`-1/ln(ρ)` 在 `ρ → 1⁻` 时发散，而原实现只对 `ρ ≥ 1` 封顶，
`ρ = 0.99999` 会给出约 90 个周期的"记忆"。

修复：把封顶施加到**结果**而非只施加在 `ρ` 分支上，并同时保留 `ρ ≥ 1` 的显式饱和分支；
另加两条测试（近单位 ρ 封顶、τ̂ ≤ 窗口步数）。修复后同数据为 3.34 周期（80.1 步）。

> 这条缺陷要是没被冒烟拦下，`τ̂` 会在 Traffic/ETTh2 上给出远超窗口的值，
> 直接毁掉 §4.7 的三个 ρ 里最关键的 `τ̂` 那一列。

### 3.2 修复时误删单位 ρ 分支（被测试立刻抓回）

第一次修复把 `ρ ≥ 1` 的显式饱和删除后，`ρ` 恰为 1.0（完全线性电平）会保持 NaN，
两条测试立即失败（`nan not less than or equal to 30`）。已恢复显式分支 + 结果侧封顶，
两条路径各有独立测试覆盖。

## 4. 结论

阶段 2/3 通过（14/14 单测、4 setting 冒烟 8.26 s）。允许进入阶段 4 的 28-setting 正式运行。
