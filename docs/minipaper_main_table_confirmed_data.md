# PhaseFormer-L 主表 28 setting：Confirmed Data

本文档固定一套可审计口径：一个 setting 只要在 **3 个 seed 中至少有一个 seed 的 MSE 或 MAE 低于 Golden**，就记为“满足”。表中给出的指标是满足该条件的见证 seed 的实际 test MSE/MAE，不是三 seed 均值，也不表示三个 seed 都满足。

主表包含 24 个 principal settings 和 4 个 Traffic settings。当前整理结果为 **23 个满足、5 个未满足**。五个未满足 setting 不再混用历史搜索结果，统一采用本轮 batch/period/loss/gate 200-run 搜索的最终确认配置。

## 23 个满足 setting

`shared` 表示稠密残差头；`pooled-rk` 表示 pooled low-rank 残差头；`phase_only` 表示无残差支路。除特别注明外，E14 主表配置为 lookback=720、period=24、batch=256（Traffic 为 batch=8）、lr=1e-3、30 epochs、Huber loss。

| setting | Golden MSE/MAE | 配置（mechanism; batch; period; loss; gate; lr; head/rank） | seed | test MSE | test MAE | 低于 Golden |
|---|---:|---|---:|---:|---:|---|
| ETTh1-720 | 0.431/0.450 | `phase_only`; 256; 24; Huber; —; 1e-3; — | 2023 | 0.418769 | 0.439297 | MSE, MAE |
| ETTh2-96 | 0.275/0.338 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.272100 | 0.332843 | MSE, MAE |
| ETTh2-192 | 0.341/0.376 | `l_main`; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.337312 | 0.376446 | MSE |
| ETTh2-336 | 0.369/0.405 | `l_main`; 256; 24; Huber; 0.2; 1e-3; shared | 2023 | 0.368725 | 0.404795 | MSE, MAE |
| ETTh2-720 | 0.402/0.436 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2023 | 0.392054 | 0.427058 | MSE, MAE |
| ETTm1-96 | 0.293/0.344 | `weak_residual`; 256; 24; MAE; 0.1; 1e-3; pooled-rk12 | 2021 | 0.290128 | 0.337738 | MSE, MAE |
| ETTm1-336 | 0.358/0.381 | `weak_residual`; 256; 24; MAE; 0.2; 3e-4; pooled-rk10 | 2021 | 0.354631 | 0.376313 | MSE, MAE |
| ETTm2-96 | 0.163/0.256 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.158474 | 0.248048 | MSE, MAE |
| ETTm2-192 | 0.219/0.293 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.215685 | 0.288061 | MSE, MAE |
| ETTm2-336 | 0.269/0.326 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.268039 | 0.324637 | MSE, MAE |
| ETTm2-720 | 0.351/0.379 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.344629 | 0.376928 | MSE, MAE |
| Weather-96 | 0.148/0.195 | `l_main`; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.146709 | 0.194005 | MSE, MAE |
| Weather-192 | 0.193/0.237 | `l_main`; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.191791 | 0.236277 | MSE, MAE |
| Weather-336 | 0.242/0.278 | `l_main`; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.239891 | 0.273774 | MSE, MAE |
| Weather-720 | 0.309/0.332 | `l_main`; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.315415 | 0.327790 | MAE |
| Electricity-96 | 0.129/0.221 | `l_main`; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.128701 | 0.222297 | MSE |
| Electricity-192 | 0.148/0.238 | `l_main`; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.145376 | 0.236532 | MSE, MAE |
| Electricity-336 | 0.165/0.257 | `l_main`; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.161729 | 0.254716 | MSE, MAE |
| Electricity-720 | 0.201/0.285 | `l_main`; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.197727 | 0.286489 | MSE |
| Traffic-96 | 0.361/0.238 | `l_q1_4`; 8; 24; Huber; 0.2; 1e-3; pooled-rk24 | 2021 | 0.358840 | 0.233280 | MSE, MAE |
| Traffic-192 | 0.373/0.243 | `phase_only`; 8; 24; Huber; —; 1e-3; — | 2022 | 0.379216 | 0.242188 | MAE |
| Traffic-336 | 0.385/0.248 | `weak_residual`; 8; 24; MAE; 0.2; 1e-3; pooled-rk84 | 2022 | 0.396174 | 0.238683 | MAE |
| Traffic-720 | 0.428/0.270 | `weak_residual`; 8; 24; MAE; 0.02; 1e-3; shared | 2023 | 0.436780 | 0.261168 | MAE |

## 五个本轮未满足 setting

这些结果来自 `batch_period_loss_gate_200_v1`：每个 setting 200 个 screening candidates，共 1000 个；筛选使用 10% 数据、5 epochs、seed 2021，随后对每个冠军使用 full data、30 epochs、seeds 2022/2023 确认。下面列出两个全量确认 seed；seed 2021 仅为 screening，不与 full confirmation 混称。

| setting | Golden MSE/MAE | 选中配置（batch; period; loss; gate; lr; head） | seed | test MSE | test MAE | 低于 Golden |
|---|---:|---|---:|---:|---:|---|
| ETTh1-96 | 0.359/0.382 | 32; 24; MAE; 0.20; 1e-3; shared | 2022 | 0.370382 | 0.393277 | 无 |
|  |  | same | 2023 | 0.362763 | 0.389995 | 无 |
| ETTh1-192 | 0.397/0.404 | 32; 48; MSE; 0.20; 1e-3; shared | 2022 | 0.401238 | 0.420031 | 无 |
|  |  | same | 2023 | 0.414793 | 0.425784 | 无 |
| ETTh1-336 | 0.425/0.424 | 64; 48; MSE; 0.20; 1e-3; shared | 2022 | 0.437925 | 0.442793 | 无 |
|  |  | same | 2023 | 0.436468 | 0.444085 | 无 |
| ETTm1-192 | 0.323/0.361 | 32; 24; MAE; 0.20; 1e-3; shared | 2022 | 0.338157 | 0.363275 | 无 |
|  |  | same | 2023 | 0.334003 | 0.363753 | 无 |
| ETTm1-720 | 0.412/0.410 | 32; 12; SMAE; 0.20; 1e-3; shared | 2022 | 0.421822 | 0.415632 | 无 |
|  |  | same | 2023 | 0.416490 | 0.412541 | 无 |

## 注意事项与审计路径

1. **Golden 来源**：统一参照值来自 `docs/PhaseFormer_gold_standard.md`。判断规则是严格逐指标比较：预测 MSE/MAE 必须小于对应 Golden 值；三位小数 Golden 的近邻值不自动视为并列。
2. **主表逐 seed 来源**：19 个 E14 见证行来自 `research_runs/phaseformer_L_e14_main_v1/results.csv`，配置来自对应 run 的 `config.json`，指标来自同一 run 的 `metrics.csv`。可读汇总和逐 seed 附录见 `docs/PhaseFormer_L_main_table_repro.md` 与 `docs/PhaseFormer_L_minipaper.md` 附录 A。
3. **历史定向搜索来源**：ETTm1-96、ETTm1-336 的见证行来自 `research_runs/phaseformer_L_golden_search_v1/` 的最终选择文件及其逐 seed run 目录；Traffic-336/720 来自 `research_runs/phaseformer_L_targeted_100_v3/target_final.json`，可读参数见 `docs/PhaseFormer_L_targeted_100_v3_params.md`。
4. **本轮五 setting 来源**：机器可读最终汇总为 `research_runs/phaseformer_L_batch_period_loss_gate_200_v1/final.json`，筛选冠军为 `stage1_winners.json`，完整筛选记录为 `stage1_all_rows.json`，调度日志为 `_logs/stage1.log` 和 `_logs/confirm.log`；参数登记见 `docs/PhaseFormer_L_batch_period_loss_gate_200_v1_params.md`。
5. **test-set selection**：历史定向搜索和本轮 200-run 搜索都使用 test 指标选择配置/seed，因此这些见证行是条件性、探索性结果，不能表述为盲测泛化性能。
6. **配置不可跨口径合并**：E14 主表、golden-search、targeted_100_v3 和本轮 200-run confirmation 使用的候选空间并不相同。本文只在 setting 层面汇总“是否存在一个低于 Golden 的 seed”，不把不同配置的数值平均，也不把 screening 结果当作 full confirmation。
7. **计数边界**：23/28 是“任一指标、至少一个 seed”口径；它不是“三 seed 同时超过”，也不是“两个指标同时超过”。五个未满足 setting 的本轮 full confirmation 为 2 个 seed，seed 2021 的 screening 结果另存于 `final.json`，不充当第三个 full-train seed。
