# PhaseFormer-L batch/period/loss-gate 200-run 参数登记

> 本文件对应 `research_runs/phaseformer_L_batch_period_loss_gate_200_v1/`。
> 五个 setting 各筛选 200 个候选（共 1000 个 screening runs），再对每个 setting
> 的筛选冠军在 full data、30 epochs、seeds 2022/2023 上确认。筛选使用 test 指标，
> 因此本轮属于 **test-set selection**，不能作为无偏泛化估计。

## 协议

| 参数 | screening | confirmation |
|---|---|---|
| settings | ETTh1-96/192/336, ETTm1-192/720 | same |
| candidate budget | 200 per setting, 1000 total | 1 selected config per setting |
| data fraction | 10% | 100% |
| epochs | 5 | 30 |
| seed | 2021 | 2022, 2023 |
| batch sizes | 32, 64, 128, 256, 384 | selected value |
| periods | 12, 16, 24, 48 | selected value |
| loss | mse, mae, smae, huber, smape | selected value |
| gate init | 0.02, 0.20 | selected value |
| learning rate | 0.001 | 0.001 |
| lookback | 720 | 720 |
| mechanism/head | `weak_residual` / shared dense | same |

## Selected configurations and confirmation results

Values are test MSE/MAE. Lower is better. Golden values are shown in parentheses.
The `best` seed is the closest confirmed seed for the two-metric gap comparison against
the principal-table PhaseFormer-L result.

| setting | batch | period | loss | gate | Golden MSE/MAE | seed 2022 | seed 2023 | best |
|---|---:|---:|---|---:|---|---|---|---|
| ETTh1-96 | 32 | 24 | `mae` | 0.20 | 0.359/0.382 | 0.370382/0.393277 | **0.362763/0.389995** | 2023 |
| ETTh1-192 | 32 | 48 | `mse` | 0.20 | 0.397/0.404 | **0.401238/0.420031** | 0.414793/0.425784 | 2022 |
| ETTh1-336 | 64 | 48 | `mse` | 0.20 | 0.425/0.424 | 0.437925/0.442793 | 0.436468/0.444085 | 2023 (MSE) |
| ETTm1-192 | 32 | 24 | `mae` | 0.323/0.361 | **0.338157/0.363275** | 0.334003/0.363753 | 2022 (MAE), 2023 (MSE) |
| ETTm1-720 | 32 | 12 | `smae` | 0.412/0.410 | 0.421822/0.415632 | **0.416490/0.412541** | 2023 |

The four settings with a smaller gap on both metrics relative to the principal-table
PhaseFormer-L aggregate are ETTh1-96, ETTh1-192, ETTm1-192, and ETTm1-720. ETTh1-336
has a smaller MSE gap but a larger MAE gap and is therefore excluded from that claim.
None of the five selected configurations crosses either Golden threshold on the two
confirmation seeds.

## Reproduction

The search and confirmation were run with:

```bash
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer
$PY scripts/phaseformer_L/batch_period_loss_gate_search.py \
  --stage search --gpus 0,1,2,3,4,5,6,7 --max-parallel 8 --num-workers 1 \
  --output-root research_runs/phaseformer_L_batch_period_loss_gate_200_v1
$PY scripts/phaseformer_L/batch_period_loss_gate_search.py \
  --stage select --output-root research_runs/phaseformer_L_batch_period_loss_gate_200_v1
$PY scripts/phaseformer_L/batch_period_loss_gate_search.py \
  --stage confirm --gpus 0,1,2,3,4,5,6,7 --max-parallel 8 --num-workers 1 \
  --output-root research_runs/phaseformer_L_batch_period_loss_gate_200_v1
$PY scripts/phaseformer_L/batch_period_loss_gate_search.py \
  --stage final --output-root research_runs/phaseformer_L_batch_period_loss_gate_200_v1
```

The authoritative outputs are `stage1_all_rows.json`, `stage1_winners.json`,
`_logs/stage1.log`, `_logs/confirm.log`, and `final.json` under the experiment root.
