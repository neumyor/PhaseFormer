# ICPT 正式 Test 结果

日期：2026-08-27。数据集仅包含 ETTh2-720 与 ETTm2-96；每个模型使用
full-train、seed 2021/2022/2023、最低 validation loss checkpoint，然后读取一次
test。指标为 `MSE / MAE`，均值后括号内为跨 seed sample std。

| Setting | 模型 | MSE mean ± std | MAE mean ± std |
|---|---|---:|---:|
| ETTh2-720 | RCRF + NLinear | 0.394228 ± 0.005051 | 0.429443 ± 0.002123 |
| ETTh2-720 | 旧 ICPT decoder | 0.436310 ± 0.016184 | 0.448457 ± 0.006985 |
| ETTh2-720 | full-horizon ICPT | 0.418833 ± 0.006797 | 0.446292 ± 0.003387 |
| ETTm2-96 | RCRF + NLinear | 0.159761 ± 0.000182 | 0.245333 ± 0.000274 |
| ETTm2-96 | 旧 ICPT decoder | 0.162994 ± 0.001651 | 0.246979 ± 0.001310 |
| ETTm2-96 | full-horizon ICPT | 0.160597 ± 0.000642 | 0.245405 ± 0.000405 |

固定金标准 PhaseFormer 为：ETTh2-720 `0.402 / 0.436`，ETTm2-96
`0.163 / 0.256`。金标准来自 `docs/PhaseFormer_gold_standard.md`，不是本次重训。

结论：full-horizon ICPT 在两个 setting 上都优于旧 ICPT decoder，但 ETTh2 仍明显
差于 RCRF + NLinear；ETTm2 接近但 MSE 仍略差。因此当前 ICPT 不能替代 NLinear，
也没有超过金标准。

完整逐 run 原始输出（含 checkpoint，未纳入 Git）位于
`research_runs/icpt_etth2_ettm2_full_20260827/`。
