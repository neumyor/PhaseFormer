# Golden 组合机制实验：完整闭环

## 1. 实验要验证的设想

相位证据可靠时应依赖相位预测，相位不可靠时应依赖近期轨迹残差；固定门控无法适应跨数据集差异。因此验证 Reliability-Coupled Residual Fusion（RCRF）能否把已有的相位修正、电平/高频修正与输出残差统一起来，并稳定超过固定 Golden。

## 2. 实验的整体计划

setting 为 ETTh2-720、ETTm2-96、Electricity-336。Stage A 以 validation-only、30% 数据、8 epoch 筛选 `fixed`、`adaptive`、`reliability_s0`、`reliability_s2` 四个组合；按 3 setting×2 metric 的平均比值冻结一个候选。Stage B 对 `original`、`latest` 和冻结候选做 full-budget、best-validation checkpoint、seed 2021/2022/2023 测试，并进行 sample×channel 审计。预注册成功标准为至少 2/3 setting 稳定双指标超过 Golden，剩余 setting 相对 Golden 回退不超过 1%。

## 3. 每个实验的实现方式和结果

RCRF 使用 `r=Var(mean phase)/(Var(mean phase)+mean within-phase Var+eps)`，`s=4*tanh(s_raw)`，`alpha=sigmoid(logit(0.5)+s(1-r))`，最终 `y=(1-alpha)y_phase+alpha y_residual`。Stage A 18 run 中 `gold_combo_reliability_s2` 胜出（score 0.80473），冻结前未读取测试集。

Stage B 三 seed 均值：

| setting | original MSE/MAE | latest MSE/MAE | RCRF MSE/MAE |
|---|---:|---:|---:|
| ETTh2-720 | 0.416091 / 0.449137 | 0.397183 / 0.429125 | **0.394228 / 0.429443** |
| ETTm2-96 | 0.167986 / 0.256188 | 0.160479 / 0.248705 | **0.159755 / 0.245331** |
| Electricity-336 | 0.168789 / 0.258626 | **0.163346 / 0.253320** | 0.164113 / 0.254625 |

RCRF 相对 matched original 三个 setting 均改善；ETTh2 与 ETTm2 满足稳定双指标标准，Electricity 的 MSE `mean+std=0.16516` 略越 Golden 0.165，故不称为稳定提升。

## 4. 最终结论

RCRF 通过了预注册的跨数据集标准（2/3 稳定，剩余 setting 未超过 1% 回退），但其收益不能归因于单一组件，且对 Electricity 的当前 dataset policy 略有退化。后续应优先研究更细粒度、可回退的门控，而不是继续堆叠残差分支。完整表格和审计包见 `docs/PhaseFormer_gold_combo_results.md` 与 `research_runs/gold_combo_stability_v1/`。
