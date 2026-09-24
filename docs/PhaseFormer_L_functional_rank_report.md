# PhaseFormer-L：低秩 checkpoint 的 functional rank 与 mode 可解释性

> 实验编号：`lowrank_functional_rank_v1`
> 执行日期：2026-09-24
> 计划依据：`low_rank_checkpoint_analysis_experiment_plan.md`（§5–§13）
> 性质：**既有 checkpoint 的事后分析**。不训练、不改 checkpoint、**不读取 test**。

## 0. 生效范围与披露

- 生效 setting：**6 个**（ETTh2-96、ETTh2-720、ETTm2-96、ETTm2-192、Weather-96、Weather-192），
  满足 §4.1 的检查清单，且与 §2.1 的低秩 checkpoint 家族完全对应。
- 压缩档：`q=1/4`、`q=1/8`、`q=1/16`、`q=1/32`；seed 2021/2022/2023。
  正式单元 = 6 × 4 × 3 = **72** 个 cell。
- Electricity-336 不在本计划 §4.1 范围内，且已有 feature cache 不完整（12 个中 6 个），
  **排除在全部裁定之外**；§4.1 的 ETTm1/Traffic 负对照需要新训练，本阶段未做。
- **test-set selection 披露**：这些 checkpoint 来自既有的条件性秩扫描，其 setting 曾按 test
  表现挑选。本阶段的全部结论因此是**条件性证据**，不得表述为盲测或无偏泛化估计。

## 1. 质量检查

| 检查 | 结果 | 含义 |
|---|---|---|
| 加性恒等式 `|Σ I_i − (MSE(ŷ₀) − MSE(ŷ))|` | **5.12e-10**（阈值 1e-6） | 模式分解与 fused 目标一致 |
| 闭式重建 vs 模型自身 `fused` | **7.39e-06** | 代数口径与模型前向一致（float32 往返量级） |
| 真实前向验证（§8.3） | **120** 次对照，最大相对误差 **3.3e-09** | 解析 `I_i` 在模型真实前向中成立 |

§8.3 的验证在模型内部删除 mode（从 hidden 中减去 `(decoder⁺u_i)·s_i·(v_iᵀz)`），
比较实测与解析的 fused MSE；`drop-all` 子集同时复现 `MSE(ŷ₀)`，因此也验证了干预机制本身。
覆盖 6 个 setting × 4 个压缩档（seed 2021 的 24 个 cell）。

## 2. 表 1：functional rank（计划 §9.3）

`r90/r95/r99` = 恢复完整低秩 checkpoint 所实现改善的 90/95/99% 所需的最小 mode 数，
base 取 `W=0`（计划 §9.3 的两种 base 之一，此处固定使用 `W=0`）。

| Setting | Nominal rank | Numerical rank | r90 (contrib) | r95 (contrib) | r99 (contrib) | r95 (singular) | r95 (activation) | negative modes | top-1 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh2-96 | 3 | 3.0 ± 0.0 | 1.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 0.0 ± 0.0 | 91% |
| ETTh2-96 | 6 | 6.0 ± 0.0 | 1.3 ± 0.5 | 2.0 ± 0.8 | 3.7 ± 0.5 | 2.0 ± 0.8 | 2.0 ± 0.8 | 0.0 ± 0.0 | 93% |
| ETTh2-96 | 12 | 12.0 ± 0.0 | 1.3 ± 0.5 | 3.0 ± 0.0 | 4.0 ± 0.0 | 3.3 ± 0.5 | 3.0 ± 0.0 | 0.3 ± 0.5 | 90% |
| ETTh2-96 | 24 | 24.0 ± 0.0 | 2.0 ± 0.0 | 2.7 ± 0.5 | 5.3 ± 0.5 | 3.0 ± 0.8 | 2.7 ± 0.5 | 1.3 ± 0.5 | 87% |
| ETTh2-720 | 22 | 22.0 ± 0.0 | 3.0 ± 0.0 | 4.0 ± 0.8 | 6.3 ± 0.9 | 4.3 ± 1.2 | 4.0 ± 0.8 | 5.0 ± 2.4 | 70% |
| ETTh2-720 | 45 | 45.0 ± 0.0 | 3.3 ± 0.5 | 4.7 ± 0.5 | 8.0 ± 1.6 | 5.7 ± 0.5 | 4.7 ± 0.5 | 12.3 ± 2.1 | 68% |
| ETTh2-720 | 90 | 90.0 ± 0.0 | 3.0 ± 0.0 | 4.7 ± 0.5 | 8.3 ± 0.5 | 5.7 ± 0.5 | 5.3 ± 1.2 | 24.7 ± 4.8 | 69% |
| ETTh2-720 | 180 | 180.0 ± 0.0 | 4.7 ± 0.5 | 7.3 ± 1.7 | 11.3 ± 2.5 | 7.7 ± 1.2 | 8.3 ± 1.7 | 62.0 ± 3.6 | 60% |
| ETTm2-96 | 3 | 3.0 ± 0.0 | 1.7 ± 0.5 | 2.3 ± 0.5 | 2.7 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 0.0 ± 0.0 | 87% |
| ETTm2-96 | 6 | 6.0 ± 0.0 | 2.7 ± 0.5 | 3.0 ± 0.0 | 4.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.0 ± 0.0 | 67% |
| ETTm2-96 | 12 | 12.0 ± 0.0 | 3.0 ± 0.0 | 3.3 ± 0.5 | 4.3 ± 0.5 | 3.3 ± 0.5 | 3.3 ± 0.5 | 0.0 ± 0.0 | 65% |
| ETTm2-96 | 24 | 24.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 5.3 ± 0.5 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.3 ± 0.5 | 63% |
| ETTm2-192 | 6 | 6.0 ± 0.0 | 2.3 ± 0.5 | 3.0 ± 0.0 | 3.7 ± 0.5 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.7 ± 0.5 | 80% |
| ETTm2-192 | 12 | 12.0 ± 0.0 | 2.3 ± 0.5 | 3.0 ± 0.0 | 3.7 ± 0.9 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.3 ± 0.5 | 78% |
| ETTm2-192 | 24 | 24.0 ± 0.0 | 2.7 ± 0.5 | 3.0 ± 0.0 | 4.3 ± 0.5 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.3 ± 0.5 | 73% |
| ETTm2-192 | 48 | 48.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 5.3 ± 1.2 | 3.0 ± 0.0 | 3.0 ± 0.0 | 2.3 ± 2.6 | 74% |
| Weather-96 | 3 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.0 ± 0.0 | 49% |
| Weather-96 | 6 | 6.0 ± 0.0 | 4.0 ± 0.0 | 4.7 ± 0.5 | 5.0 ± 0.0 | 4.7 ± 0.5 | 4.7 ± 0.5 | 0.0 ± 0.0 | 45% |
| Weather-96 | 12 | 12.0 ± 0.0 | 4.0 ± 0.0 | 5.0 ± 0.0 | 5.3 ± 0.5 | 5.0 ± 0.0 | 5.0 ± 0.0 | 1.0 ± 1.4 | 40% |
| Weather-96 | 24 | 24.0 ± 0.0 | 3.3 ± 0.5 | 4.3 ± 0.5 | 6.0 ± 0.0 | 4.3 ± 0.5 | 4.3 ± 0.5 | 6.7 ± 1.7 | 43% |
| Weather-192 | 6 | 6.0 ± 0.0 | 4.3 ± 0.5 | 5.0 ± 0.0 | 6.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 0.0 ± 0.0 | 29% |
| Weather-192 | 12 | 12.0 ± 0.0 | 4.0 ± 0.0 | 5.0 ± 0.0 | 6.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 0.3 ± 0.5 | 30% |
| Weather-192 | 24 | 24.0 ± 0.0 | 4.7 ± 0.5 | 5.0 ± 0.0 | 7.7 ± 0.5 | 5.0 ± 0.0 | 5.0 ± 0.0 | 2.3 ± 0.5 | 28% |
| Weather-192 | 48 | 48.0 ± 0.0 | 4.0 ± 0.0 | 5.0 ± 0.0 | 6.7 ± 0.5 | 5.0 ± 0.0 | 5.0 ± 0.0 | 6.0 ± 0.8 | 29% |

## 3. 表 2：四种排序的对比（计划 §9.1）

| Setting | rank | r95 contribution | r95 singular | r95 activation | r95 weight |
|---|---:|---:|---:|---:|---:|
| ETTh2-720 | 22 | 4.0 ± 0.8 | 4.3 ± 1.2 | 4.0 ± 0.8 | 4.3 ± 1.2 |
| ETTh2-720 | 45 | 4.7 ± 0.5 | 5.7 ± 0.5 | 4.7 ± 0.5 | 5.7 ± 0.5 |
| ETTh2-720 | 90 | 4.7 ± 0.5 | 5.7 ± 0.5 | 5.3 ± 1.2 | 5.7 ± 0.5 |
| ETTh2-720 | 180 | 7.3 ± 1.7 | 7.7 ± 1.2 | 8.3 ± 1.7 | 7.7 ± 1.2 |
| ETTh2-96 | 3 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 |
| ETTh2-96 | 6 | 2.0 ± 0.8 | 2.0 ± 0.8 | 2.0 ± 0.8 | 2.0 ± 0.8 |
| ETTh2-96 | 12 | 3.0 ± 0.0 | 3.3 ± 0.5 | 3.0 ± 0.0 | 3.3 ± 0.5 |
| ETTh2-96 | 24 | 2.7 ± 0.5 | 3.0 ± 0.8 | 2.7 ± 0.5 | 3.0 ± 0.8 |
| ETTm2-192 | 6 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| ETTm2-192 | 12 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| ETTm2-192 | 24 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| ETTm2-192 | 48 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| ETTm2-96 | 3 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 | 2.3 ± 0.5 |
| ETTm2-96 | 6 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| ETTm2-96 | 12 | 3.3 ± 0.5 | 3.3 ± 0.5 | 3.3 ± 0.5 | 3.3 ± 0.5 |
| ETTm2-96 | 24 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| Weather-192 | 6 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 |
| Weather-192 | 12 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 |
| Weather-192 | 24 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 |
| Weather-192 | 48 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 |
| Weather-96 | 3 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 |
| Weather-96 | 6 | 4.7 ± 0.5 | 4.7 ± 0.5 | 4.7 ± 0.5 | 4.7 ± 0.5 |
| Weather-96 | 12 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 | 5.0 ± 0.0 |
| Weather-96 | 24 | 4.3 ± 0.5 | 4.3 ± 0.5 | 4.3 ± 0.5 | 4.3 ± 0.5 |

## 4. 表 3：zero-shot mode pruning（计划 §11.1）

`random` 为同规模随机删除的对照带；`negative_contribution` 删除全部 `I_i<0` 的 mode。

| Setting | rank | criterion | fraction | dropped | Δfused MSE |
|---|---:|---|---:|---:|---:|
| ETTh2-720 | 22 | activation_energy | 0.1 | 2.0 ± 0.0 | -0.000003 |
| ETTh2-720 | 22 | activation_energy | 0.25 | 6.0 ± 0.0 | +0.000025 |
| ETTh2-720 | 22 | activation_energy | 0.5 | 11.0 ± 0.0 | +0.000152 |
| ETTh2-720 | 22 | contribution | 0.1 | 2.0 ± 0.0 | -0.000313 |
| ETTh2-720 | 22 | contribution | 0.25 | 6.0 ± 0.0 | -0.000359 |
| ETTh2-720 | 22 | contribution | 0.5 | 11.0 ± 0.0 | -0.000312 |
| ETTh2-720 | 22 | negative_contribution | — | 5.0 ± 2.4 | -0.000368 |
| ETTh2-720 | 22 | random | 0.1 | 2.0 ± 0.0 | +0.004257 |
| ETTh2-720 | 22 | random | 0.25 | 6.0 ± 0.0 | +0.013475 |
| ETTh2-720 | 22 | random | 0.5 | 11.0 ± 0.0 | +0.023162 |
| ETTh2-720 | 22 | singular | 0.1 | 2.0 ± 0.0 | -0.000002 |
| ETTh2-720 | 22 | singular | 0.25 | 6.0 ± 0.0 | +0.000015 |
| ETTh2-720 | 22 | singular | 0.5 | 11.0 ± 0.0 | +0.000296 |
| ETTh2-720 | 22 | weight_energy | 0.1 | 2.0 ± 0.0 | -0.000002 |
| ETTh2-720 | 22 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000015 |
| ETTh2-720 | 22 | weight_energy | 0.5 | 11.0 ± 0.0 | +0.000296 |
| ETTh2-720 | 45 | activation_energy | 0.1 | 4.0 ± 0.0 | +0.000001 |
| ETTh2-720 | 45 | activation_energy | 0.25 | 11.0 ± 0.0 | +0.000005 |
| ETTh2-720 | 45 | activation_energy | 0.5 | 22.0 ± 0.0 | +0.000067 |
| ETTh2-720 | 45 | contribution | 0.1 | 4.0 ± 0.0 | -0.000531 |
| ETTh2-720 | 45 | contribution | 0.25 | 11.0 ± 0.0 | -0.000645 |
| ETTh2-720 | 45 | contribution | 0.5 | 22.0 ± 0.0 | -0.000632 |
| ETTh2-720 | 45 | negative_contribution | — | 12.3 ± 2.1 | -0.000646 |
| ETTh2-720 | 45 | random | 0.1 | 4.0 ± 0.0 | +0.004688 |
| ETTh2-720 | 45 | random | 0.25 | 11.0 ± 0.0 | +0.012491 |
| ETTh2-720 | 45 | random | 0.5 | 22.0 ± 0.0 | +0.026244 |
| ETTh2-720 | 45 | singular | 0.1 | 4.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 45 | singular | 0.25 | 11.0 ± 0.0 | +0.000004 |
| ETTh2-720 | 45 | singular | 0.5 | 22.0 ± 0.0 | +0.000063 |
| ETTh2-720 | 45 | weight_energy | 0.1 | 4.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 45 | weight_energy | 0.25 | 11.0 ± 0.0 | +0.000004 |
| ETTh2-720 | 45 | weight_energy | 0.5 | 22.0 ± 0.0 | +0.000063 |
| ETTh2-720 | 90 | activation_energy | 0.1 | 9.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 90 | activation_energy | 0.25 | 22.0 ± 0.0 | +0.000002 |
| ETTh2-720 | 90 | activation_energy | 0.5 | 45.0 ± 0.0 | +0.000011 |
| ETTh2-720 | 90 | contribution | 0.1 | 9.0 ± 0.0 | -0.000727 |
| ETTh2-720 | 90 | contribution | 0.25 | 22.0 ± 0.0 | -0.000732 |
| ETTh2-720 | 90 | contribution | 0.5 | 45.0 ± 0.0 | -0.000728 |
| ETTh2-720 | 90 | negative_contribution | — | 24.7 ± 4.8 | -0.000733 |
| ETTh2-720 | 90 | random | 0.1 | 9.0 ± 0.0 | +0.007009 |
| ETTh2-720 | 90 | random | 0.25 | 22.0 ± 0.0 | +0.015579 |
| ETTh2-720 | 90 | random | 0.5 | 45.0 ± 0.0 | +0.031559 |
| ETTh2-720 | 90 | singular | 0.1 | 9.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 90 | singular | 0.25 | 22.0 ± 0.0 | +0.000002 |
| ETTh2-720 | 90 | singular | 0.5 | 45.0 ± 0.0 | +0.000014 |
| ETTh2-720 | 90 | weight_energy | 0.1 | 9.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 90 | weight_energy | 0.25 | 22.0 ± 0.0 | +0.000002 |
| ETTh2-720 | 90 | weight_energy | 0.5 | 45.0 ± 0.0 | +0.000014 |
| ETTh2-720 | 180 | activation_energy | 0.1 | 18.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 180 | activation_energy | 0.25 | 45.0 ± 0.0 | +0.000001 |
| ETTh2-720 | 180 | activation_energy | 0.5 | 90.0 ± 0.0 | +0.000005 |
| ETTh2-720 | 180 | contribution | 0.1 | 18.0 ± 0.0 | -0.001047 |
| ETTh2-720 | 180 | contribution | 0.25 | 45.0 ± 0.0 | -0.001058 |
| ETTh2-720 | 180 | contribution | 0.5 | 90.0 ± 0.0 | -0.001057 |
| ETTh2-720 | 180 | negative_contribution | — | 62.0 ± 3.6 | -0.001058 |
| ETTh2-720 | 180 | random | 0.1 | 18.0 ± 0.0 | +0.006086 |
| ETTh2-720 | 180 | random | 0.25 | 45.0 ± 0.0 | +0.016576 |
| ETTh2-720 | 180 | random | 0.5 | 90.0 ± 0.0 | +0.030895 |
| ETTh2-720 | 180 | singular | 0.1 | 18.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 180 | singular | 0.25 | 45.0 ± 0.0 | +0.000001 |
| ETTh2-720 | 180 | singular | 0.5 | 90.0 ± 0.0 | +0.000006 |
| ETTh2-720 | 180 | weight_energy | 0.1 | 18.0 ± 0.0 | +0.000000 |
| ETTh2-720 | 180 | weight_energy | 0.25 | 45.0 ± 0.0 | +0.000001 |
| ETTh2-720 | 180 | weight_energy | 0.5 | 90.0 ± 0.0 | +0.000006 |
| ETTh2-96 | 3 | activation_energy | 0.25 | 1.0 ± 0.0 | +0.000655 |
| ETTh2-96 | 3 | activation_energy | 0.5 | 2.0 ± 0.0 | +0.002978 |
| ETTh2-96 | 3 | contribution | 0.25 | 1.0 ± 0.0 | +0.000625 |
| ETTh2-96 | 3 | contribution | 0.5 | 2.0 ± 0.0 | +0.002978 |
| ETTh2-96 | 3 | random | 0.25 | 1.0 ± 0.0 | +0.011595 |
| ETTh2-96 | 3 | random | 0.5 | 2.0 ± 0.0 | +0.022294 |
| ETTh2-96 | 3 | singular | 0.25 | 1.0 ± 0.0 | +0.000655 |
| ETTh2-96 | 3 | singular | 0.5 | 2.0 ± 0.0 | +0.002978 |
| ETTh2-96 | 3 | weight_energy | 0.25 | 1.0 ± 0.0 | +0.000655 |
| ETTh2-96 | 3 | weight_energy | 0.5 | 2.0 ± 0.0 | +0.002978 |
| ETTh2-96 | 6 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000010 |
| ETTh2-96 | 6 | activation_energy | 0.25 | 2.0 ± 0.0 | +0.000170 |
| ETTh2-96 | 6 | activation_energy | 0.5 | 3.0 ± 0.0 | +0.001285 |
| ETTh2-96 | 6 | contribution | 0.1 | 1.0 ± 0.0 | +0.000010 |
| ETTh2-96 | 6 | contribution | 0.25 | 2.0 ± 0.0 | +0.000170 |
| ETTh2-96 | 6 | contribution | 0.5 | 3.0 ± 0.0 | +0.000983 |
| ETTh2-96 | 6 | random | 0.1 | 1.0 ± 0.0 | +0.009599 |
| ETTh2-96 | 6 | random | 0.25 | 2.0 ± 0.0 | +0.021040 |
| ETTh2-96 | 6 | random | 0.5 | 3.0 ± 0.0 | +0.031407 |
| ETTh2-96 | 6 | singular | 0.1 | 1.0 ± 0.0 | +0.000010 |
| ETTh2-96 | 6 | singular | 0.25 | 2.0 ± 0.0 | +0.000170 |
| ETTh2-96 | 6 | singular | 0.5 | 3.0 ± 0.0 | +0.001363 |
| ETTh2-96 | 6 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000010 |
| ETTh2-96 | 6 | weight_energy | 0.25 | 2.0 ± 0.0 | +0.000170 |
| ETTh2-96 | 6 | weight_energy | 0.5 | 3.0 ± 0.0 | +0.001363 |
| ETTh2-96 | 12 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTh2-96 | 12 | activation_energy | 0.25 | 3.0 ± 0.0 | +0.000003 |
| ETTh2-96 | 12 | activation_energy | 0.5 | 6.0 ± 0.0 | +0.000043 |
| ETTh2-96 | 12 | contribution | 0.1 | 1.0 ± 0.0 | +0.000000 |
| ETTh2-96 | 12 | contribution | 0.25 | 3.0 ± 0.0 | +0.000002 |
| ETTh2-96 | 12 | contribution | 0.5 | 6.0 ± 0.0 | +0.000029 |
| ETTh2-96 | 12 | negative_contribution | — | 1.0 | -0.000000 |
| ETTh2-96 | 12 | random | 0.1 | 1.0 ± 0.0 | +0.004301 |
| ETTh2-96 | 12 | random | 0.25 | 3.0 ± 0.0 | +0.011834 |
| ETTh2-96 | 12 | random | 0.5 | 6.0 ± 0.0 | +0.025938 |
| ETTh2-96 | 12 | singular | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTh2-96 | 12 | singular | 0.25 | 3.0 ± 0.0 | +0.000003 |
| ETTh2-96 | 12 | singular | 0.5 | 6.0 ± 0.0 | +0.000029 |
| ETTh2-96 | 12 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTh2-96 | 12 | weight_energy | 0.25 | 3.0 ± 0.0 | +0.000003 |
| ETTh2-96 | 12 | weight_energy | 0.5 | 6.0 ± 0.0 | +0.000029 |
| ETTh2-96 | 24 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTh2-96 | 24 | activation_energy | 0.25 | 6.0 ± 0.0 | +0.000002 |
| ETTh2-96 | 24 | activation_energy | 0.5 | 12.0 ± 0.0 | +0.000013 |
| ETTh2-96 | 24 | contribution | 0.1 | 2.0 ± 0.0 | -0.000000 |
| ETTh2-96 | 24 | contribution | 0.25 | 6.0 ± 0.0 | +0.000001 |
| ETTh2-96 | 24 | contribution | 0.5 | 12.0 ± 0.0 | +0.000011 |
| ETTh2-96 | 24 | negative_contribution | — | 1.3 ± 0.5 | -0.000001 |
| ETTh2-96 | 24 | random | 0.1 | 2.0 ± 0.0 | +0.003478 |
| ETTh2-96 | 24 | random | 0.25 | 6.0 ± 0.0 | +0.009987 |
| ETTh2-96 | 24 | random | 0.5 | 12.0 ± 0.0 | +0.020964 |
| ETTh2-96 | 24 | singular | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTh2-96 | 24 | singular | 0.25 | 6.0 ± 0.0 | +0.000002 |
| ETTh2-96 | 24 | singular | 0.5 | 12.0 ± 0.0 | +0.000017 |
| ETTh2-96 | 24 | weight_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTh2-96 | 24 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000002 |
| ETTh2-96 | 24 | weight_energy | 0.5 | 12.0 ± 0.0 | +0.000017 |
| ETTm2-192 | 6 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000005 |
| ETTm2-192 | 6 | activation_energy | 0.25 | 2.0 ± 0.0 | -0.000003 |
| ETTm2-192 | 6 | activation_energy | 0.5 | 3.0 ± 0.0 | +0.000509 |
| ETTm2-192 | 6 | contribution | 0.1 | 1.0 ± 0.0 | -0.000009 |
| ETTm2-192 | 6 | contribution | 0.25 | 2.0 ± 0.0 | -0.000003 |
| ETTm2-192 | 6 | contribution | 0.5 | 3.0 ± 0.0 | +0.000509 |
| ETTm2-192 | 6 | negative_contribution | — | 1.0 ± 0.0 | -0.000014 |
| ETTm2-192 | 6 | random | 0.1 | 1.0 ± 0.0 | +0.007902 |
| ETTm2-192 | 6 | random | 0.25 | 2.0 ± 0.0 | +0.015031 |
| ETTm2-192 | 6 | random | 0.5 | 3.0 ± 0.0 | +0.021461 |
| ETTm2-192 | 6 | singular | 0.1 | 1.0 ± 0.0 | +0.000005 |
| ETTm2-192 | 6 | singular | 0.25 | 2.0 ± 0.0 | -0.000003 |
| ETTm2-192 | 6 | singular | 0.5 | 3.0 ± 0.0 | +0.000509 |
| ETTm2-192 | 6 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000005 |
| ETTm2-192 | 6 | weight_energy | 0.25 | 2.0 ± 0.0 | -0.000003 |
| ETTm2-192 | 6 | weight_energy | 0.5 | 3.0 ± 0.0 | +0.000509 |
| ETTm2-192 | 12 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 12 | activation_energy | 0.25 | 3.0 ± 0.0 | +0.000006 |
| ETTm2-192 | 12 | activation_energy | 0.5 | 6.0 ± 0.0 | +0.000081 |
| ETTm2-192 | 12 | contribution | 0.1 | 1.0 ± 0.0 | -0.000025 |
| ETTm2-192 | 12 | contribution | 0.25 | 3.0 ± 0.0 | -0.000023 |
| ETTm2-192 | 12 | contribution | 0.5 | 6.0 ± 0.0 | +0.000038 |
| ETTm2-192 | 12 | negative_contribution | — | 1.0 | -0.000076 |
| ETTm2-192 | 12 | random | 0.1 | 1.0 ± 0.0 | +0.004506 |
| ETTm2-192 | 12 | random | 0.25 | 3.0 ± 0.0 | +0.013131 |
| ETTm2-192 | 12 | random | 0.5 | 6.0 ± 0.0 | +0.024945 |
| ETTm2-192 | 12 | singular | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 12 | singular | 0.25 | 3.0 ± 0.0 | +0.000006 |
| ETTm2-192 | 12 | singular | 0.5 | 6.0 ± 0.0 | +0.000081 |
| ETTm2-192 | 12 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 12 | weight_energy | 0.25 | 3.0 ± 0.0 | +0.000006 |
| ETTm2-192 | 12 | weight_energy | 0.5 | 6.0 ± 0.0 | +0.000081 |
| ETTm2-192 | 24 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | activation_energy | 0.25 | 6.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | activation_energy | 0.5 | 12.0 ± 0.0 | +0.000008 |
| ETTm2-192 | 24 | contribution | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | contribution | 0.25 | 6.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | contribution | 0.5 | 12.0 ± 0.0 | +0.000007 |
| ETTm2-192 | 24 | negative_contribution | — | 1.0 | -0.000000 |
| ETTm2-192 | 24 | random | 0.1 | 2.0 ± 0.0 | +0.003328 |
| ETTm2-192 | 24 | random | 0.25 | 6.0 ± 0.0 | +0.010477 |
| ETTm2-192 | 24 | random | 0.5 | 12.0 ± 0.0 | +0.022581 |
| ETTm2-192 | 24 | singular | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | singular | 0.25 | 6.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | singular | 0.5 | 12.0 ± 0.0 | +0.000007 |
| ETTm2-192 | 24 | weight_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 24 | weight_energy | 0.5 | 12.0 ± 0.0 | +0.000007 |
| ETTm2-192 | 48 | activation_energy | 0.1 | 5.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | activation_energy | 0.25 | 12.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | activation_energy | 0.5 | 24.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 48 | contribution | 0.1 | 5.0 ± 0.0 | -0.000000 |
| ETTm2-192 | 48 | contribution | 0.25 | 12.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | contribution | 0.5 | 24.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 48 | negative_contribution | — | 3.5 ± 2.5 | -0.000000 |
| ETTm2-192 | 48 | random | 0.1 | 5.0 ± 0.0 | +0.004657 |
| ETTm2-192 | 48 | random | 0.25 | 12.0 ± 0.0 | +0.012169 |
| ETTm2-192 | 48 | random | 0.5 | 24.0 ± 0.0 | +0.023687 |
| ETTm2-192 | 48 | singular | 0.1 | 5.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | singular | 0.25 | 12.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | singular | 0.5 | 24.0 ± 0.0 | +0.000001 |
| ETTm2-192 | 48 | weight_energy | 0.1 | 5.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | weight_energy | 0.25 | 12.0 ± 0.0 | +0.000000 |
| ETTm2-192 | 48 | weight_energy | 0.5 | 24.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 3 | activation_energy | 0.25 | 1.0 ± 0.0 | +0.000990 |
| ETTm2-96 | 3 | activation_energy | 0.5 | 2.0 ± 0.0 | +0.003748 |
| ETTm2-96 | 3 | contribution | 0.25 | 1.0 ± 0.0 | +0.000990 |
| ETTm2-96 | 3 | contribution | 0.5 | 2.0 ± 0.0 | +0.003748 |
| ETTm2-96 | 3 | random | 0.25 | 1.0 ± 0.0 | +0.010452 |
| ETTm2-96 | 3 | random | 0.5 | 2.0 ± 0.0 | +0.019844 |
| ETTm2-96 | 3 | singular | 0.25 | 1.0 ± 0.0 | +0.000990 |
| ETTm2-96 | 3 | singular | 0.5 | 2.0 ± 0.0 | +0.003748 |
| ETTm2-96 | 3 | weight_energy | 0.25 | 1.0 ± 0.0 | +0.000990 |
| ETTm2-96 | 3 | weight_energy | 0.5 | 2.0 ± 0.0 | +0.003748 |
| ETTm2-96 | 6 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 6 | activation_energy | 0.25 | 2.0 ± 0.0 | +0.000029 |
| ETTm2-96 | 6 | activation_energy | 0.5 | 3.0 ± 0.0 | +0.000773 |
| ETTm2-96 | 6 | contribution | 0.1 | 1.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 6 | contribution | 0.25 | 2.0 ± 0.0 | +0.000029 |
| ETTm2-96 | 6 | contribution | 0.5 | 3.0 ± 0.0 | +0.000773 |
| ETTm2-96 | 6 | random | 0.1 | 1.0 ± 0.0 | +0.005952 |
| ETTm2-96 | 6 | random | 0.25 | 2.0 ± 0.0 | +0.011901 |
| ETTm2-96 | 6 | random | 0.5 | 3.0 ± 0.0 | +0.017688 |
| ETTm2-96 | 6 | singular | 0.1 | 1.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 6 | singular | 0.25 | 2.0 ± 0.0 | +0.000029 |
| ETTm2-96 | 6 | singular | 0.5 | 3.0 ± 0.0 | +0.000773 |
| ETTm2-96 | 6 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 6 | weight_energy | 0.25 | 2.0 ± 0.0 | +0.000029 |
| ETTm2-96 | 6 | weight_energy | 0.5 | 3.0 ± 0.0 | +0.000773 |
| ETTm2-96 | 12 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 12 | activation_energy | 0.25 | 3.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 12 | activation_energy | 0.5 | 6.0 ± 0.0 | +0.000032 |
| ETTm2-96 | 12 | contribution | 0.1 | 1.0 ± 0.0 | +0.000000 |
| ETTm2-96 | 12 | contribution | 0.25 | 3.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 12 | contribution | 0.5 | 6.0 ± 0.0 | +0.000031 |
| ETTm2-96 | 12 | random | 0.1 | 1.0 ± 0.0 | +0.003294 |
| ETTm2-96 | 12 | random | 0.25 | 3.0 ± 0.0 | +0.010485 |
| ETTm2-96 | 12 | random | 0.5 | 6.0 ± 0.0 | +0.021318 |
| ETTm2-96 | 12 | singular | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 12 | singular | 0.25 | 3.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 12 | singular | 0.5 | 6.0 ± 0.0 | +0.000031 |
| ETTm2-96 | 12 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 12 | weight_energy | 0.25 | 3.0 ± 0.0 | +0.000004 |
| ETTm2-96 | 12 | weight_energy | 0.5 | 6.0 ± 0.0 | +0.000031 |
| ETTm2-96 | 24 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-96 | 24 | activation_energy | 0.25 | 6.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 24 | activation_energy | 0.5 | 12.0 ± 0.0 | +0.000010 |
| ETTm2-96 | 24 | contribution | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-96 | 24 | contribution | 0.25 | 6.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 24 | contribution | 0.5 | 12.0 ± 0.0 | +0.000009 |
| ETTm2-96 | 24 | negative_contribution | — | 1.0 | -0.000000 |
| ETTm2-96 | 24 | random | 0.1 | 2.0 ± 0.0 | +0.003567 |
| ETTm2-96 | 24 | random | 0.25 | 6.0 ± 0.0 | +0.010823 |
| ETTm2-96 | 24 | random | 0.5 | 12.0 ± 0.0 | +0.019866 |
| ETTm2-96 | 24 | singular | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-96 | 24 | singular | 0.25 | 6.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 24 | singular | 0.5 | 12.0 ± 0.0 | +0.000009 |
| ETTm2-96 | 24 | weight_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| ETTm2-96 | 24 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000001 |
| ETTm2-96 | 24 | weight_energy | 0.5 | 12.0 ± 0.0 | +0.000009 |
| Electricity-336 | 10 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000120 |
| Electricity-336 | 10 | activation_energy | 0.25 | 2.0 ± 0.0 | +0.000711 |
| Electricity-336 | 10 | activation_energy | 0.5 | 5.0 ± 0.0 | +0.003053 |
| Electricity-336 | 10 | contribution | 0.1 | 1.0 ± 0.0 | +0.000120 |
| Electricity-336 | 10 | contribution | 0.25 | 2.0 ± 0.0 | +0.000623 |
| Electricity-336 | 10 | contribution | 0.5 | 5.0 ± 0.0 | +0.003041 |
| Electricity-336 | 10 | random | 0.1 | 1.0 ± 0.0 | +0.006366 |
| Electricity-336 | 10 | random | 0.25 | 2.0 ± 0.0 | +0.014837 |
| Electricity-336 | 10 | random | 0.5 | 5.0 ± 0.0 | +0.034865 |
| Electricity-336 | 10 | singular | 0.1 | 1.0 ± 0.0 | +0.001825 |
| Electricity-336 | 10 | singular | 0.25 | 2.0 ± 0.0 | +0.002282 |
| Electricity-336 | 10 | singular | 0.5 | 5.0 ± 0.0 | +0.006826 |
| Electricity-336 | 10 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.001825 |
| Electricity-336 | 10 | weight_energy | 0.25 | 2.0 ± 0.0 | +0.002282 |
| Electricity-336 | 10 | weight_energy | 0.5 | 5.0 ± 0.0 | +0.006826 |
| Electricity-336 | 21 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000104 |
| Electricity-336 | 21 | activation_energy | 0.25 | 5.0 ± 0.0 | +0.000549 |
| Electricity-336 | 21 | activation_energy | 0.5 | 10.0 ± 0.0 | +0.001945 |
| Electricity-336 | 21 | contribution | 0.1 | 2.0 ± 0.0 | +0.000104 |
| Electricity-336 | 21 | contribution | 0.25 | 5.0 ± 0.0 | +0.000519 |
| Electricity-336 | 21 | contribution | 0.5 | 10.0 ± 0.0 | +0.001848 |
| Electricity-336 | 21 | random | 0.1 | 2.0 ± 0.0 | +0.004002 |
| Electricity-336 | 21 | random | 0.25 | 5.0 ± 0.0 | +0.011748 |
| Electricity-336 | 21 | random | 0.5 | 10.0 ± 0.0 | +0.022583 |
| Electricity-336 | 21 | singular | 0.1 | 2.0 ± 0.0 | +0.000401 |
| Electricity-336 | 21 | singular | 0.25 | 5.0 ± 0.0 | +0.000985 |
| Electricity-336 | 21 | singular | 0.5 | 10.0 ± 0.0 | +0.002464 |
| Electricity-336 | 21 | weight_energy | 0.1 | 2.0 ± 0.0 | +0.000401 |
| Electricity-336 | 21 | weight_energy | 0.25 | 5.0 ± 0.0 | +0.000985 |
| Electricity-336 | 21 | weight_energy | 0.5 | 10.0 ± 0.0 | +0.002464 |
| Electricity-336 | 84 | activation_energy | 0.1 | 8.0 ± 0.0 | +0.000070 |
| Electricity-336 | 84 | activation_energy | 0.25 | 21.0 ± 0.0 | +0.000596 |
| Electricity-336 | 84 | activation_energy | 0.5 | 42.0 ± 0.0 | +0.002620 |
| Electricity-336 | 84 | contribution | 0.1 | 8.0 ± 0.0 | +0.000055 |
| Electricity-336 | 84 | contribution | 0.25 | 21.0 ± 0.0 | +0.000533 |
| Electricity-336 | 84 | contribution | 0.5 | 42.0 ± 0.0 | +0.002512 |
| Electricity-336 | 84 | negative_contribution | — | 1.0 | -0.000002 |
| Electricity-336 | 84 | random | 0.1 | 8.0 ± 0.0 | +0.014072 |
| Electricity-336 | 84 | random | 0.25 | 21.0 ± 0.0 | +0.032129 |
| Electricity-336 | 84 | random | 0.5 | 42.0 ± 0.0 | +0.070924 |
| Electricity-336 | 84 | singular | 0.1 | 8.0 ± 0.0 | +0.000077 |
| Electricity-336 | 84 | singular | 0.25 | 21.0 ± 0.0 | +0.000935 |
| Electricity-336 | 84 | singular | 0.5 | 42.0 ± 0.0 | +0.003266 |
| Electricity-336 | 84 | weight_energy | 0.1 | 8.0 ± 0.0 | +0.000077 |
| Electricity-336 | 84 | weight_energy | 0.25 | 21.0 ± 0.0 | +0.000935 |
| Electricity-336 | 84 | weight_energy | 0.5 | 42.0 ± 0.0 | +0.003266 |
| Weather-192 | 6 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.006770 |
| Weather-192 | 6 | activation_energy | 0.25 | 2.0 ± 0.0 | +0.014063 |
| Weather-192 | 6 | activation_energy | 0.5 | 3.0 ± 0.0 | +0.029452 |
| Weather-192 | 6 | contribution | 0.1 | 1.0 ± 0.0 | +0.005169 |
| Weather-192 | 6 | contribution | 0.25 | 2.0 ± 0.0 | +0.014063 |
| Weather-192 | 6 | contribution | 0.5 | 3.0 ± 0.0 | +0.029452 |
| Weather-192 | 6 | random | 0.1 | 1.0 ± 0.0 | +0.025816 |
| Weather-192 | 6 | random | 0.25 | 2.0 ± 0.0 | +0.053314 |
| Weather-192 | 6 | random | 0.5 | 3.0 ± 0.0 | +0.080210 |
| Weather-192 | 6 | singular | 0.1 | 1.0 ± 0.0 | +0.005169 |
| Weather-192 | 6 | singular | 0.25 | 2.0 ± 0.0 | +0.015682 |
| Weather-192 | 6 | singular | 0.5 | 3.0 ± 0.0 | +0.029452 |
| Weather-192 | 6 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.005169 |
| Weather-192 | 6 | weight_energy | 0.25 | 2.0 ± 0.0 | +0.015682 |
| Weather-192 | 6 | weight_energy | 0.5 | 3.0 ± 0.0 | +0.029452 |
| Weather-192 | 12 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000025 |
| Weather-192 | 12 | activation_energy | 0.25 | 3.0 ± 0.0 | +0.000158 |
| Weather-192 | 12 | activation_energy | 0.5 | 6.0 ± 0.0 | +0.001110 |
| Weather-192 | 12 | contribution | 0.1 | 1.0 ± 0.0 | -0.000005 |
| Weather-192 | 12 | contribution | 0.25 | 3.0 ± 0.0 | +0.000129 |
| Weather-192 | 12 | contribution | 0.5 | 6.0 ± 0.0 | +0.001110 |
| Weather-192 | 12 | negative_contribution | — | 1.0 | -0.000017 |
| Weather-192 | 12 | random | 0.1 | 1.0 ± 0.0 | +0.012513 |
| Weather-192 | 12 | random | 0.25 | 3.0 ± 0.0 | +0.038233 |
| Weather-192 | 12 | random | 0.5 | 6.0 ± 0.0 | +0.076021 |
| Weather-192 | 12 | singular | 0.1 | 1.0 ± 0.0 | +0.000025 |
| Weather-192 | 12 | singular | 0.25 | 3.0 ± 0.0 | +0.000158 |
| Weather-192 | 12 | singular | 0.5 | 6.0 ± 0.0 | +0.005582 |
| Weather-192 | 12 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000025 |
| Weather-192 | 12 | weight_energy | 0.25 | 3.0 ± 0.0 | +0.000158 |
| Weather-192 | 12 | weight_energy | 0.5 | 6.0 ± 0.0 | +0.005582 |
| Weather-192 | 24 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000001 |
| Weather-192 | 24 | activation_energy | 0.25 | 6.0 ± 0.0 | +0.000003 |
| Weather-192 | 24 | activation_energy | 0.5 | 12.0 ± 0.0 | +0.000051 |
| Weather-192 | 24 | contribution | 0.1 | 2.0 ± 0.0 | -0.000008 |
| Weather-192 | 24 | contribution | 0.25 | 6.0 ± 0.0 | -0.000006 |
| Weather-192 | 24 | contribution | 0.5 | 12.0 ± 0.0 | +0.000036 |
| Weather-192 | 24 | negative_contribution | — | 2.3 ± 0.5 | -0.000008 |
| Weather-192 | 24 | random | 0.1 | 2.0 ± 0.0 | +0.013490 |
| Weather-192 | 24 | random | 0.25 | 6.0 ± 0.0 | +0.039716 |
| Weather-192 | 24 | random | 0.5 | 12.0 ± 0.0 | +0.075291 |
| Weather-192 | 24 | singular | 0.1 | 2.0 ± 0.0 | +0.000001 |
| Weather-192 | 24 | singular | 0.25 | 6.0 ± 0.0 | +0.000004 |
| Weather-192 | 24 | singular | 0.5 | 12.0 ± 0.0 | +0.000051 |
| Weather-192 | 24 | weight_energy | 0.1 | 2.0 ± 0.0 | +0.000001 |
| Weather-192 | 24 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000004 |
| Weather-192 | 24 | weight_energy | 0.5 | 12.0 ± 0.0 | +0.000051 |
| Weather-192 | 48 | activation_energy | 0.1 | 5.0 ± 0.0 | +0.000000 |
| Weather-192 | 48 | activation_energy | 0.25 | 12.0 ± 0.0 | +0.000001 |
| Weather-192 | 48 | activation_energy | 0.5 | 24.0 ± 0.0 | +0.000003 |
| Weather-192 | 48 | contribution | 0.1 | 5.0 ± 0.0 | -0.000000 |
| Weather-192 | 48 | contribution | 0.25 | 12.0 ± 0.0 | -0.000000 |
| Weather-192 | 48 | contribution | 0.5 | 24.0 ± 0.0 | +0.000003 |
| Weather-192 | 48 | negative_contribution | — | 6.0 ± 0.8 | -0.000000 |
| Weather-192 | 48 | random | 0.1 | 5.0 ± 0.0 | +0.018352 |
| Weather-192 | 48 | random | 0.25 | 12.0 ± 0.0 | +0.044130 |
| Weather-192 | 48 | random | 0.5 | 24.0 ± 0.0 | +0.083519 |
| Weather-192 | 48 | singular | 0.1 | 5.0 ± 0.0 | +0.000001 |
| Weather-192 | 48 | singular | 0.25 | 12.0 ± 0.0 | +0.000001 |
| Weather-192 | 48 | singular | 0.5 | 24.0 ± 0.0 | +0.000004 |
| Weather-192 | 48 | weight_energy | 0.1 | 5.0 ± 0.0 | +0.000001 |
| Weather-192 | 48 | weight_energy | 0.25 | 12.0 ± 0.0 | +0.000001 |
| Weather-192 | 48 | weight_energy | 0.5 | 24.0 ± 0.0 | +0.000004 |
| Weather-96 | 3 | activation_energy | 0.25 | 1.0 ± 0.0 | +0.015564 |
| Weather-96 | 3 | activation_energy | 0.5 | 2.0 ± 0.0 | +0.062657 |
| Weather-96 | 3 | contribution | 0.25 | 1.0 ± 0.0 | +0.015564 |
| Weather-96 | 3 | contribution | 0.5 | 2.0 ± 0.0 | +0.062657 |
| Weather-96 | 3 | random | 0.25 | 1.0 ± 0.0 | +0.039538 |
| Weather-96 | 3 | random | 0.5 | 2.0 ± 0.0 | +0.081885 |
| Weather-96 | 3 | singular | 0.25 | 1.0 ± 0.0 | +0.015564 |
| Weather-96 | 3 | singular | 0.5 | 2.0 ± 0.0 | +0.062657 |
| Weather-96 | 3 | weight_energy | 0.25 | 1.0 ± 0.0 | +0.015564 |
| Weather-96 | 3 | weight_energy | 0.5 | 2.0 ± 0.0 | +0.062657 |
| Weather-96 | 6 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000136 |
| Weather-96 | 6 | activation_energy | 0.25 | 2.0 ± 0.0 | +0.007583 |
| Weather-96 | 6 | activation_energy | 0.5 | 3.0 ± 0.0 | +0.018352 |
| Weather-96 | 6 | contribution | 0.1 | 1.0 ± 0.0 | +0.000136 |
| Weather-96 | 6 | contribution | 0.25 | 2.0 ± 0.0 | +0.007583 |
| Weather-96 | 6 | contribution | 0.5 | 3.0 ± 0.0 | +0.018352 |
| Weather-96 | 6 | random | 0.1 | 1.0 ± 0.0 | +0.027304 |
| Weather-96 | 6 | random | 0.25 | 2.0 ± 0.0 | +0.053593 |
| Weather-96 | 6 | random | 0.5 | 3.0 ± 0.0 | +0.080553 |
| Weather-96 | 6 | singular | 0.1 | 1.0 ± 0.0 | +0.000136 |
| Weather-96 | 6 | singular | 0.25 | 2.0 ± 0.0 | +0.007583 |
| Weather-96 | 6 | singular | 0.5 | 3.0 ± 0.0 | +0.018352 |
| Weather-96 | 6 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000136 |
| Weather-96 | 6 | weight_energy | 0.25 | 2.0 ± 0.0 | +0.007583 |
| Weather-96 | 6 | weight_energy | 0.5 | 3.0 ± 0.0 | +0.018352 |
| Weather-96 | 12 | activation_energy | 0.1 | 1.0 ± 0.0 | +0.000000 |
| Weather-96 | 12 | activation_energy | 0.25 | 3.0 ± 0.0 | +0.000001 |
| Weather-96 | 12 | activation_energy | 0.5 | 6.0 ± 0.0 | +0.000314 |
| Weather-96 | 12 | contribution | 0.1 | 1.0 ± 0.0 | -0.000000 |
| Weather-96 | 12 | contribution | 0.25 | 3.0 ± 0.0 | +0.000001 |
| Weather-96 | 12 | contribution | 0.5 | 6.0 ± 0.0 | +0.000314 |
| Weather-96 | 12 | negative_contribution | — | 3.0 | -0.000000 |
| Weather-96 | 12 | random | 0.1 | 1.0 ± 0.0 | +0.013999 |
| Weather-96 | 12 | random | 0.25 | 3.0 ± 0.0 | +0.038148 |
| Weather-96 | 12 | random | 0.5 | 6.0 ± 0.0 | +0.081767 |
| Weather-96 | 12 | singular | 0.1 | 1.0 ± 0.0 | +0.000000 |
| Weather-96 | 12 | singular | 0.25 | 3.0 ± 0.0 | +0.000001 |
| Weather-96 | 12 | singular | 0.5 | 6.0 ± 0.0 | +0.000314 |
| Weather-96 | 12 | weight_energy | 0.1 | 1.0 ± 0.0 | +0.000000 |
| Weather-96 | 12 | weight_energy | 0.25 | 3.0 ± 0.0 | +0.000001 |
| Weather-96 | 12 | weight_energy | 0.5 | 6.0 ± 0.0 | +0.000314 |
| Weather-96 | 24 | activation_energy | 0.1 | 2.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | activation_energy | 0.25 | 6.0 ± 0.0 | -0.000000 |
| Weather-96 | 24 | activation_energy | 0.5 | 12.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | contribution | 0.1 | 2.0 ± 0.0 | -0.000000 |
| Weather-96 | 24 | contribution | 0.25 | 6.0 ± 0.0 | -0.000000 |
| Weather-96 | 24 | contribution | 0.5 | 12.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | negative_contribution | — | 6.7 ± 1.7 | -0.000000 |
| Weather-96 | 24 | random | 0.1 | 2.0 ± 0.0 | +0.013535 |
| Weather-96 | 24 | random | 0.25 | 6.0 ± 0.0 | +0.043582 |
| Weather-96 | 24 | random | 0.5 | 12.0 ± 0.0 | +0.084605 |
| Weather-96 | 24 | singular | 0.1 | 2.0 ± 0.0 | -0.000000 |
| Weather-96 | 24 | singular | 0.25 | 6.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | singular | 0.5 | 12.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | weight_energy | 0.1 | 2.0 ± 0.0 | -0.000000 |
| Weather-96 | 24 | weight_energy | 0.25 | 6.0 ± 0.0 | +0.000000 |
| Weather-96 | 24 | weight_energy | 0.5 | 12.0 ± 0.0 | +0.000000 |

## 5. 表 4：low-rank 与 dense head 的子空间对齐（计划 §10）

`dense_singular` = 按 dense 的普通奇异值排序；`dense_functional` = 按 dense 的实测预测贡献排序。

| Setting | Low-rank cell | dim | dense reference | input overlap | output overlap |
|---|---|---:|---|---:|---:|
| ETTh2-720 | q=1/4 | 2 | dense_functional | 0.837 | 0.934 |
| ETTh2-720 | q=1/4 | 2 | dense_singular | 0.837 | 0.934 |
| ETTh2-720 | q=1/4 | 4 | dense_functional | 0.658 | 0.733 |
| ETTh2-720 | q=1/4 | 4 | dense_singular | 0.638 | 0.711 |
| ETTh2-720 | q=1/4 | 8 | dense_functional | 0.575 | 0.640 |
| ETTh2-720 | q=1/4 | 8 | dense_singular | 0.628 | 0.696 |
| ETTh2-720 | q=1/4 | 16 | dense_functional | 0.535 | 0.630 |
| ETTh2-720 | q=1/4 | 16 | dense_singular | 0.576 | 0.679 |
| ETTh2-720 | q=1/8 | 2 | dense_functional | 0.808 | 0.921 |
| ETTh2-720 | q=1/8 | 2 | dense_singular | 0.808 | 0.921 |
| ETTh2-720 | q=1/8 | 4 | dense_functional | 0.623 | 0.700 |
| ETTh2-720 | q=1/8 | 4 | dense_singular | 0.623 | 0.703 |
| ETTh2-720 | q=1/8 | 8 | dense_functional | 0.557 | 0.651 |
| ETTh2-720 | q=1/8 | 8 | dense_singular | 0.590 | 0.684 |
| ETTh2-720 | q=1/8 | 16 | dense_functional | 0.405 | 0.546 |
| ETTh2-720 | q=1/8 | 16 | dense_singular | 0.399 | 0.534 |
| ETTh2-96 | q=1/4 | 2 | dense_functional | 0.461 | 0.598 |
| ETTh2-96 | q=1/4 | 2 | dense_singular | 0.461 | 0.598 |
| ETTh2-96 | q=1/4 | 4 | dense_functional | 0.460 | 0.772 |
| ETTh2-96 | q=1/4 | 4 | dense_singular | 0.437 | 0.713 |
| ETTh2-96 | q=1/4 | 8 | dense_functional | 0.299 | 0.667 |
| ETTh2-96 | q=1/4 | 8 | dense_singular | 0.297 | 0.657 |
| ETTh2-96 | q=1/4 | 16 | dense_functional | 0.183 | 0.654 |
| ETTh2-96 | q=1/4 | 16 | dense_singular | 0.188 | 0.676 |
| ETTh2-96 | q=1/8 | 2 | dense_functional | 0.474 | 0.654 |
| ETTh2-96 | q=1/8 | 2 | dense_singular | 0.474 | 0.654 |
| ETTh2-96 | q=1/8 | 4 | dense_functional | 0.429 | 0.798 |
| ETTh2-96 | q=1/8 | 4 | dense_singular | 0.429 | 0.793 |
| ETTh2-96 | q=1/8 | 8 | dense_functional | 0.252 | 0.649 |
| ETTh2-96 | q=1/8 | 8 | dense_singular | 0.247 | 0.632 |
| ETTh2-96 | q=1/8 | 12 | dense_functional | 0.184 | 0.607 |
| ETTh2-96 | q=1/8 | 12 | dense_singular | 0.185 | 0.633 |
| ETTm2-192 | q=1/4 | 2 | dense_functional | 0.745 | 0.812 |
| ETTm2-192 | q=1/4 | 2 | dense_singular | 0.745 | 0.812 |
| ETTm2-192 | q=1/4 | 4 | dense_functional | 0.617 | 0.686 |
| ETTm2-192 | q=1/4 | 4 | dense_singular | 0.630 | 0.713 |
| ETTm2-192 | q=1/4 | 8 | dense_functional | 0.404 | 0.575 |
| ETTm2-192 | q=1/4 | 8 | dense_singular | 0.391 | 0.543 |
| ETTm2-192 | q=1/4 | 16 | dense_functional | 0.309 | 0.701 |
| ETTm2-192 | q=1/4 | 16 | dense_singular | 0.309 | 0.692 |
| ETTm2-192 | q=1/8 | 2 | dense_functional | 0.769 | 0.848 |
| ETTm2-192 | q=1/8 | 2 | dense_singular | 0.769 | 0.848 |
| ETTm2-192 | q=1/8 | 4 | dense_functional | 0.642 | 0.744 |
| ETTm2-192 | q=1/8 | 4 | dense_singular | 0.636 | 0.730 |
| ETTm2-192 | q=1/8 | 8 | dense_functional | 0.386 | 0.586 |
| ETTm2-192 | q=1/8 | 8 | dense_singular | 0.377 | 0.566 |
| ETTm2-192 | q=1/8 | 16 | dense_functional | 0.253 | 0.703 |
| ETTm2-192 | q=1/8 | 16 | dense_singular | 0.249 | 0.691 |
| ETTm2-96 | q=1/4 | 2 | dense_functional | 0.644 | 0.733 |
| ETTm2-96 | q=1/4 | 2 | dense_singular | 0.680 | 0.812 |
| ETTm2-96 | q=1/4 | 4 | dense_functional | 0.612 | 0.818 |
| ETTm2-96 | q=1/4 | 4 | dense_singular | 0.591 | 0.765 |
| ETTm2-96 | q=1/4 | 8 | dense_functional | 0.364 | 0.716 |
| ETTm2-96 | q=1/4 | 8 | dense_singular | 0.359 | 0.648 |
| ETTm2-96 | q=1/4 | 16 | dense_functional | 0.225 | 0.839 |
| ETTm2-96 | q=1/4 | 16 | dense_singular | 0.224 | 0.827 |
| ETTm2-96 | q=1/8 | 2 | dense_functional | 0.555 | 0.618 |
| ETTm2-96 | q=1/8 | 2 | dense_singular | 0.625 | 0.718 |
| ETTm2-96 | q=1/8 | 4 | dense_functional | 0.602 | 0.801 |
| ETTm2-96 | q=1/8 | 4 | dense_singular | 0.588 | 0.745 |
| ETTm2-96 | q=1/8 | 8 | dense_functional | 0.340 | 0.679 |
| ETTm2-96 | q=1/8 | 8 | dense_singular | 0.328 | 0.605 |
| ETTm2-96 | q=1/8 | 12 | dense_functional | 0.244 | 0.715 |
| ETTm2-96 | q=1/8 | 12 | dense_singular | 0.243 | 0.686 |
| Weather-192 | q=1/4 | 2 | dense_functional | 0.736 | 0.819 |
| Weather-192 | q=1/4 | 2 | dense_singular | 0.751 | 0.828 |
| Weather-192 | q=1/4 | 4 | dense_functional | 0.871 | 0.943 |
| Weather-192 | q=1/4 | 4 | dense_singular | 0.871 | 0.943 |
| Weather-192 | q=1/4 | 8 | dense_functional | 0.846 | 0.945 |
| Weather-192 | q=1/4 | 8 | dense_singular | 0.882 | 0.985 |
| Weather-192 | q=1/4 | 16 | dense_functional | 0.558 | 0.680 |
| Weather-192 | q=1/4 | 16 | dense_singular | 0.571 | 0.710 |
| Weather-192 | q=1/8 | 2 | dense_functional | 0.760 | 0.819 |
| Weather-192 | q=1/8 | 2 | dense_singular | 0.914 | 0.983 |
| Weather-192 | q=1/8 | 4 | dense_functional | 0.858 | 0.909 |
| Weather-192 | q=1/8 | 4 | dense_singular | 0.858 | 0.909 |
| Weather-192 | q=1/8 | 8 | dense_functional | 0.788 | 0.868 |
| Weather-192 | q=1/8 | 8 | dense_singular | 0.826 | 0.908 |
| Weather-192 | q=1/8 | 16 | dense_functional | 0.569 | 0.668 |
| Weather-192 | q=1/8 | 16 | dense_singular | 0.580 | 0.707 |
| Weather-96 | q=1/4 | 2 | dense_functional | 0.844 | 0.939 |
| Weather-96 | q=1/4 | 2 | dense_singular | 0.844 | 0.939 |
| Weather-96 | q=1/4 | 4 | dense_functional | 0.918 | 0.992 |
| Weather-96 | q=1/4 | 4 | dense_singular | 0.918 | 0.992 |
| Weather-96 | q=1/4 | 8 | dense_functional | 0.696 | 0.854 |
| Weather-96 | q=1/4 | 8 | dense_singular | 0.696 | 0.854 |
| Weather-96 | q=1/4 | 16 | dense_functional | 0.364 | 0.639 |
| Weather-96 | q=1/4 | 16 | dense_singular | 0.365 | 0.684 |
| Weather-96 | q=1/8 | 2 | dense_functional | 0.862 | 0.947 |
| Weather-96 | q=1/8 | 2 | dense_singular | 0.862 | 0.947 |
| Weather-96 | q=1/8 | 4 | dense_functional | 0.763 | 0.838 |
| Weather-96 | q=1/8 | 4 | dense_singular | 0.763 | 0.838 |
| Weather-96 | q=1/8 | 8 | dense_functional | 0.592 | 0.841 |
| Weather-96 | q=1/8 | 8 | dense_singular | 0.592 | 0.841 |
| Weather-96 | q=1/8 | 12 | dense_functional | 0.404 | 0.682 |
| Weather-96 | q=1/8 | 12 | dense_singular | 0.405 | 0.710 |

### 5.1 dense 模型可及改善中由低秩子空间保留的比例（计划 §10 的操作化形式）

把 dense 的有效映射限制到低秩输入子空间后，仍保留的 dense 改善份额：

| Setting | Low-rank cell | dim 2 | dim 4 | dim 8 | dim 16 |
|---|---|---:|---:|---:|---:|
| ETTh2-720 | q=1/4 | 72% | 85% | 89% | 92% |
| ETTh2-720 | q=1/8 | 71% | 83% | 87% | 89% |
| ETTh2-96 | q=1/4 | 72% | 79% | 81% | 81% |
| ETTh2-96 | q=1/8 | 76% | 85% | 85% | — |
| ETTm2-192 | q=1/4 | 71% | 82% | 84% | 85% |
| ETTm2-192 | q=1/8 | 68% | 80% | 80% | 81% |
| ETTm2-96 | q=1/4 | 63% | 80% | 81% | 82% |
| ETTm2-96 | q=1/8 | 59% | 75% | 78% | — |
| Weather-192 | q=1/4 | 44% | 86% | 98% | 98% |
| Weather-192 | q=1/8 | 42% | 83% | 95% | 96% |
| Weather-96 | q=1/4 | 72% | 92% | 96% | 96% |
| Weather-96 | q=1/8 | 74% | 90% | 94% | — |

### 5.2 dense head 的两种排序是否可分（计划 §10.4）

若 dense 的普通权重能量顺序与预测贡献顺序本身不一致，"低秩保留的是 functional 而非
energetic 子空间"这个问题才可分。

| Setting | seed | dense rank corr (singular vs functional) | top-16 input subspace overlap |
|---|---:|---:|---:|
| ETTh2-720 | 2021 | 0.00 | 0.88 |
| ETTh2-720 | 2022 | -0.20 | 0.81 |
| ETTh2-720 | 2023 | 0.03 | 0.87 |
| ETTh2-96 | 2021 | 0.81 | 0.81 |
| ETTh2-96 | 2022 | 0.84 | 0.87 |
| ETTh2-96 | 2023 | 0.80 | 0.81 |
| ETTm2-192 | 2021 | 0.91 | 0.94 |
| ETTm2-192 | 2022 | 0.87 | 1.00 |
| ETTm2-192 | 2023 | 0.92 | 0.87 |
| ETTm2-96 | 2021 | 0.91 | 0.94 |
| ETTm2-96 | 2022 | 0.94 | 0.94 |
| ETTm2-96 | 2023 | 0.90 | 0.94 |
| Weather-192 | 2021 | 0.51 | 0.62 |
| Weather-192 | 2022 | 0.42 | 0.75 |
| Weather-192 | 2023 | 0.29 | 0.69 |
| Weather-96 | 2021 | 0.76 | 0.69 |
| Weather-96 | 2022 | 0.80 | 0.69 |
| Weather-96 | 2023 | 0.83 | 0.75 |

## 6. 表 5：单个 mode 内部的进一步稀疏化（计划 §12）

`fused MSE increase` 为逐 mode 的**精确**融合代价（mode 输出方向在输入侧稀疏化下保持正交，
因此仍是标量运算）；`dense` 行恒等于 0 是整条链路的自洽性检查。

| Variant | modes | reconstruction R² | fused MSE increase | retained lags | atoms |
|---|---:|---:|---:|---:|---:|
| dense | 72 | 1.000 | +2.99e-17 | 720 | — |
| group_select_24_keep10 | 72 | 0.332 | +6.85e-03 | 72 | — |
| group_select_24_keep25 | 72 | 0.552 | +4.53e-03 | 192 | — |
| group_select_96_keep10 | 72 | 0.286 | +8.28e-03 | 81 | — |
| group_select_96_keep25 | 72 | 0.445 | +6.38e-03 | 170 | — |
| hard_90 | 72 | 0.487 | +4.86e-03 | 72 | — |
| hard_95 | 72 | 0.344 | +6.55e-03 | 36 | — |
| hard_99 | 72 | 0.147 | +1.01e-02 | 7 | — |
| semantic_lasso | 72 | 0.524 | +2.00e-03 | 720 | 39.0 |
| tv_0.05 | 72 | 0.769 | +7.99e-05 | 720 | — |
| tv_0.2 | 72 | 0.522 | +1.07e-03 | 720 | — |
| tv_0.5 | 72 | 0.324 | +3.47e-03 | 720 | — |

## 6b. 逐 mode 语义归因："这 3–5 个 mode 分别是什么"

上表回答了"需要几个 mode"，这一节回答"分别是哪个方向"。按**实测预测贡献**排序，
把每个 mode 的输入方向匹配到输入字典、输出方向匹配到输出字典。

None

### 配对机制（`q=1/8`，三个 seed 的众数标签与均值解释率）

| Setting | rank position | contribution share | input reads | group expl. | output writes | group expl. | paired mechanism |
|---|---:|---:|---|---:|---|---:|---|
| ETTh2-96 | 1 | 90.1% | recent level | 0.63 | displacement | 0.96 | recent level -> displacement |
| ETTh2-96 | 2 | 4.5% | recent level | 0.30 | tilt | 0.80 | recent level -> tilt |
| ETTh2-96 | 3 | 3.0% | periodic shape | 0.41 | periodic correction | 0.81 | periodic shape -> periodic correction |
| ETTh2-96 | 4 | 1.7% | periodic shape | 0.47 | periodic correction | 0.76 | periodic shape -> periodic correction |
| ETTh2-96 | 5 | 0.4% | recent level | 0.15 | tilt | 0.44 | recent level -> tilt |
| **ETTh2-96** | *(r95 = 3.0 of 12)* | | | | | | |
| ETTh2-720 | 1 | 68.7% | recent level | 0.73 | displacement | 0.95 | recent level -> displacement |
| ETTh2-720 | 2 | 11.9% | periodic shape | 0.72 | periodic correction | 0.95 | periodic shape -> periodic correction |
| ETTh2-720 | 3 | 10.2% | periodic shape | 0.75 | periodic correction | 0.93 | periodic shape -> periodic correction |
| ETTh2-720 | 4 | 2.8% | level change | 0.36 | curvature | 0.52 | level change -> curvature |
| ETTh2-720 | 5 | 1.5% | recent level | 0.41 | curvature | 0.55 | recent level -> curvature |
| **ETTh2-720** | *(r95 = 4.7 of 90)* | | | | | | |
| ETTm2-96 | 1 | 65.3% | recent level | 0.60 | displacement | 0.97 | recent level -> displacement |
| ETTm2-96 | 2 | 18.8% | periodic shape | 0.61 | periodic correction | 0.96 | periodic shape -> periodic correction |
| ETTm2-96 | 3 | 12.3% | periodic shape | 0.63 | periodic correction | 0.96 | periodic shape -> periodic correction |
| ETTm2-96 | 4 | 2.6% | recent level | 0.26 | periodic correction | 0.71 | recent level -> tilt |
| ETTm2-96 | 5 | 0.7% | recent level | 0.11 | periodic correction | 0.19 | recent level -> periodic correction |
| **ETTm2-96** | *(r95 = 3.3 of 12)* | | | | | | |
| ETTm2-192 | 1 | 72.8% | recent level | 0.63 | displacement | 0.97 | recent level -> displacement |
| ETTm2-192 | 2 | 15.1% | periodic shape | 0.67 | periodic correction | 0.94 | periodic shape -> periodic correction |
| ETTm2-192 | 3 | 10.2% | periodic shape | 0.70 | periodic correction | 0.96 | periodic shape -> periodic correction |
| ETTm2-192 | 4 | 0.9% | recent level | 0.39 | curvature | 0.57 | recent level -> curvature |
| ETTm2-192 | 5 | 0.5% | recent level | 0.30 | curvature | 0.54 | recent level -> curvature |
| **ETTm2-192** | *(r95 = 3.0 of 24)* | | | | | | |
| Weather-96 | 1 | 40.2% | recent level | 0.52 | curvature | 0.96 | recent level -> curvature |
| Weather-96 | 2 | 34.6% | recent level | 0.47 | curvature | 0.95 | recent level -> curvature |
| Weather-96 | 3 | 14.2% | recent level | 0.15 | curvature | 0.49 | recent level -> curvature |
| Weather-96 | 4 | 5.6% | recent level | 0.22 | curvature | 0.50 | recent level -> curvature |
| Weather-96 | 5 | 4.2% | local curvature | 0.15 | tilt | 0.46 | local curvature -> tilt |
| **Weather-96** | *(r95 = 5.0 of 12)* | | | | | | |
| Weather-192 | 1 | 27.9% | recent level | 0.44 | curvature | 0.52 | recent level -> curvature |
| Weather-192 | 2 | 24.8% | recent level | 0.57 | curvature | 0.92 | recent level -> curvature |
| Weather-192 | 3 | 22.2% | local trend | 0.24 | tilt | 0.42 | local trend -> tilt |
| Weather-192 | 4 | 14.5% | recent level | 0.25 | tilt | 0.33 | recent level -> tilt |
| Weather-192 | 5 | 6.9% | recent level | 0.21 | tilt | 0.10 | local curvature -> tilt |
| **Weather-192** | *(r95 = 5.0 of 24)* | | | | | | |

**解释率不是划分**：输入字典内 EMA24 / EMA48 / tail_mean_24 等模板近乎共线（计划 §4 已指出），
各语义组对同一方向都有非零投影，因此一行内的组解释率**不要求和为 1**。要看"成分比例"
应读下一节的逐组数值，而不是把 best match 当成唯一成分。

### 逐组解释率（`q=1/8`）

| Setting | position | recent level | periodic shape | fast local change | local curvature | other | output displacement | output tilt | output curvature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh2-96 | 1 | 0.62 | 0.60 | 0.07 | 0.58 | 1.53 | 0.96 | 0.94 | 0.91 |
| ETTh2-96 | 2 | 0.27 | 0.15 | 0.02 | 0.22 | 0.50 | 0.02 | 0.68 | 0.71 |
| ETTh2-96 | 3 | 0.09 | 0.41 | 0.02 | 0.04 | 0.07 | 0.00 | 0.17 | 0.13 |
| ETTh2-720 | 1 | 0.73 | 0.57 | 0.10 | 0.68 | 1.76 | 0.95 | 0.90 | 0.88 |
| ETTh2-720 | 2 | 0.18 | 0.72 | 0.05 | 0.06 | 0.09 | 0.00 | 0.00 | 0.00 |
| ETTh2-720 | 3 | 0.05 | 0.75 | 0.05 | 0.06 | 0.01 | 0.00 | 0.00 | 0.00 |
| ETTm2-96 | 1 | 0.60 | 0.23 | 0.15 | 0.56 | 1.16 | 0.97 | 0.96 | 0.93 |
| ETTm2-96 | 2 | 0.34 | 0.61 | 0.02 | 0.28 | 0.67 | 0.02 | 0.59 | 0.81 |
| ETTm2-96 | 3 | 0.40 | 0.63 | 0.02 | 0.34 | 0.66 | 0.01 | 0.53 | 0.90 |
| ETTm2-192 | 1 | 0.63 | 0.19 | 0.18 | 0.57 | 1.17 | 0.96 | 0.96 | 0.87 |
| ETTm2-192 | 2 | 0.35 | 0.67 | 0.01 | 0.29 | 0.66 | 0.01 | 0.34 | 0.13 |
| ETTm2-192 | 3 | 0.40 | 0.70 | 0.02 | 0.34 | 0.65 | 0.00 | 0.16 | 0.05 |
| Weather-96 | 1 | 0.52 | 0.17 | 0.27 | 0.44 | 0.92 | 0.62 | 0.93 | 0.95 |
| Weather-96 | 2 | 0.47 | 0.09 | 0.14 | 0.34 | 0.87 | 0.35 | 0.89 | 0.95 |
| Weather-96 | 3 | 0.15 | 0.00 | 0.00 | 0.05 | 0.21 | 0.00 | 0.07 | 0.49 |
| Weather-192 | 1 | 0.44 | 0.22 | 0.03 | 0.38 | 1.17 | 0.10 | 0.24 | 0.52 |
| Weather-192 | 2 | 0.57 | 0.21 | 0.49 | 0.55 | 0.92 | 0.86 | 0.81 | 0.92 |
| Weather-192 | 3 | 0.24 | 0.04 | 0.02 | 0.20 | 0.51 | 0.00 | 0.42 | 0.30 |

## 7. 表 6：跨 seed 的模式稳定性（计划 §13）

按 `|v_i·v_j'|·|u_i·u_j'|` 做 Hungarian 匹配后的成对余弦。

| Setting | cell | seed pair | input cos mean | output cos mean | matched >0.7 | rank correlation |
|---|---|---|---:|---:|---:|---:|
| ETTh2-720 | q=1/16 | 2021–2022 | 0.184 | 0.352 | 0.04 | 0.58 |
| ETTh2-720 | q=1/16 | 2022–2023 | 0.190 | 0.348 | 0.07 | 0.42 |
| ETTh2-720 | q=1/32 | 2021–2022 | 0.238 | 0.435 | 0.09 | 0.46 |
| ETTh2-720 | q=1/32 | 2022–2023 | 0.238 | 0.439 | 0.14 | 0.79 |
| ETTh2-720 | q=1/4 | 2021–2022 | 0.133 | 0.199 | 0.02 | 0.41 |
| ETTh2-720 | q=1/4 | 2022–2023 | 0.128 | 0.200 | 0.02 | 0.44 |
| ETTh2-720 | q=1/8 | 2021–2022 | 0.143 | 0.263 | 0.03 | 0.48 |
| ETTh2-720 | q=1/8 | 2022–2023 | 0.146 | 0.267 | 0.06 | 0.29 |
| ETTh2-96 | q=1/16 | 2021–2022 | 0.328 | 0.704 | 0.17 | 0.94 |
| ETTh2-96 | q=1/16 | 2022–2023 | 0.428 | 0.731 | 0.17 | 1.00 |
| ETTh2-96 | q=1/32 | 2021–2022 | 0.443 | 0.766 | 0.33 | 0.50 |
| ETTh2-96 | q=1/32 | 2022–2023 | 0.456 | 0.595 | 0.33 | 0.50 |
| ETTh2-96 | q=1/4 | 2021–2022 | 0.151 | 0.440 | 0.04 | 0.71 |
| ETTh2-96 | q=1/4 | 2022–2023 | 0.156 | 0.426 | 0.04 | 0.81 |
| ETTh2-96 | q=1/8 | 2021–2022 | 0.224 | 0.525 | 0.08 | 0.78 |
| ETTh2-96 | q=1/8 | 2022–2023 | 0.238 | 0.516 | 0.08 | 0.82 |
| ETTm2-192 | q=1/16 | 2021–2022 | 0.251 | 0.476 | 0.08 | 0.74 |
| ETTm2-192 | q=1/16 | 2022–2023 | 0.259 | 0.520 | 0.08 | 0.57 |
| ETTm2-192 | q=1/32 | 2021–2022 | 0.430 | 0.586 | 0.33 | 0.94 |
| ETTm2-192 | q=1/32 | 2022–2023 | 0.418 | 0.674 | 0.33 | 0.83 |
| ETTm2-192 | q=1/4 | 2021–2022 | 0.134 | 0.346 | 0.06 | 0.82 |
| ETTm2-192 | q=1/4 | 2022–2023 | 0.137 | 0.332 | 0.06 | 0.86 |
| ETTm2-192 | q=1/8 | 2021–2022 | 0.177 | 0.429 | 0.04 | 0.85 |
| ETTm2-192 | q=1/8 | 2022–2023 | 0.187 | 0.403 | 0.12 | 0.77 |
| ETTm2-96 | q=1/16 | 2021–2022 | 0.374 | 0.660 | 0.17 | 1.00 |
| ETTm2-96 | q=1/16 | 2022–2023 | 0.417 | 0.671 | 0.17 | 0.94 |
| ETTm2-96 | q=1/32 | 2021–2022 | 0.478 | 0.818 | 0.33 | 1.00 |
| ETTm2-96 | q=1/32 | 2022–2023 | 0.620 | 0.980 | 0.33 | 1.00 |
| ETTm2-96 | q=1/4 | 2021–2022 | 0.178 | 0.487 | 0.12 | 0.85 |
| ETTm2-96 | q=1/4 | 2022–2023 | 0.159 | 0.454 | 0.04 | 0.83 |
| ETTm2-96 | q=1/8 | 2021–2022 | 0.256 | 0.552 | 0.08 | 0.85 |
| ETTm2-96 | q=1/8 | 2022–2023 | 0.240 | 0.588 | 0.08 | 0.90 |
| Weather-192 | q=1/16 | 2021–2022 | 0.589 | 0.782 | 0.42 | 0.98 |
| Weather-192 | q=1/16 | 2022–2023 | 0.591 | 0.785 | 0.58 | 0.94 |
| Weather-192 | q=1/32 | 2021–2022 | 0.868 | 0.952 | 1.00 | 0.89 |
| Weather-192 | q=1/32 | 2022–2023 | 0.862 | 0.949 | 1.00 | 0.89 |
| Weather-192 | q=1/4 | 2021–2022 | 0.247 | 0.410 | 0.17 | 0.76 |
| Weather-192 | q=1/4 | 2022–2023 | 0.231 | 0.386 | 0.15 | 0.75 |
| Weather-192 | q=1/8 | 2021–2022 | 0.407 | 0.570 | 0.33 | 0.82 |
| Weather-192 | q=1/8 | 2022–2023 | 0.448 | 0.635 | 0.42 | 0.83 |
| Weather-96 | q=1/16 | 2021–2022 | 0.732 | 0.820 | 0.83 | 1.00 |
| Weather-96 | q=1/16 | 2022–2023 | 0.753 | 0.908 | 0.83 | 1.00 |
| Weather-96 | q=1/32 | 2021–2022 | 0.907 | 0.985 | 1.00 | 1.00 |
| Weather-96 | q=1/32 | 2022–2023 | 0.906 | 0.991 | 1.00 | 1.00 |
| Weather-96 | q=1/4 | 2021–2022 | 0.274 | 0.487 | 0.21 | 0.79 |
| Weather-96 | q=1/4 | 2022–2023 | 0.285 | 0.506 | 0.25 | 0.75 |
| Weather-96 | q=1/8 | 2021–2022 | 0.459 | 0.681 | 0.42 | 0.96 |
| Weather-96 | q=1/8 | 2022–2023 | 0.454 | 0.722 | 0.42 | 0.92 |

## 8. 关键可视化（计划 §15）

- `figures/fig1_functional_rank_curves.png` — 恢复曲线（四种排序）
- `figures/fig2_importance_mismatch.png` — 三种"重要性"不一致
- `figures/fig3_negative_contribution_share.png` — 负贡献比例随 nominal rank 增长
- `figures/fig4_dense_alignment.png` — 与 dense 两种排序的子空间重叠
- `figures/fig5_canonical_modes.png` — 代表 setting 的 top-5 read-write modes
- `figures/fig6_sparsity_tradeoff.png` — 稀疏化的 R²–代价权衡

## 9. 结论分层（计划 §22）

**Q1 低秩保留了哪些方向？** 由 canonical SVD modes 与语义归因回答；本阶段补充了
activation energy（§6.4）与四种排序的对比。最重要的差异是：`r95` 在四种排序下高度接近，
说明在该模型家族中 **singular 排序与 contribution 排序给出几乎相同的前缀**
（见 §3 表 2），与计划 H3 预期的强不一致并不一致。

**Q2 各自贡献多少？** 由 `I_i`、leave-one-mode-out 与真实前向验证回答；加性与前向两重
验证均通过（§1）。

**Q3 还能否进一步压缩？** 由 functional-rank 曲线、pruning 与 §12 的稀疏化回答。

## 10. 限制

1. 全部结论依赖既有的 test-exposed checkpoint；不构成盲测证据。
2. §8.3 的前向验证覆盖 seed 2021；代数量级结论（~1e-9）不随 seed 改变，但未逐 seed 复核。
3. §12 的稀疏化是**事后**施加在已冻结 mode 上的；它回答"该 mode 的 kernel 能否被压缩"，
   不能回答"从头训练时稀疏 kernel 是否同样可学到"。
4. 跨 seed 匹配在近退化的奇异子空间上仍可能给出低余弦；表 6 的低值应结合
   `cross_seed_alignment.csv` 的子空间级重叠一起读，不能单独作为"机制不稳定"的证据。
5. dense 对齐只在 6 个 setting 上成立；dense checkpoint 家族与低秩家族同源
   （`rank_sweep_2_stage1`），因此该对齐是**同协议**的，但不覆盖 §4.1 之外的 setting。
