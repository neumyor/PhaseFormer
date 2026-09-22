# PhaseFormer-L targeted_100_v3 参数登记

> 本文件是 `research_runs/phaseformer_L_targeted_100_v3/target_final.json` 的可读参数记录。
> 每个 setting 使用 100 个候选配置进行筛选；随后对选中的配置在 full train、30 epochs、
> seeds 2021/2022/2023 上做完整确认。候选和确认均属于 **test-set selection**，不能作为
> 无偏盲测估计。Golden 仅作披露性参照。

## 公共确认协议

| 参数 | 值 |
|---|---|
| lookback | 720 |
| period | 24 |
| percent | 100 |
| max epochs | 30 |
| checkpoint | lowest validation loss |
| evaluation | single test read per checkpoint |
| mechanism | `weak_residual` |
| residual period | 24 |
| seeds | 2021, 2022, 2023 |
| candidate budget | 100 runs per setting |
| candidate screening | 10% train data, 5 epochs |

## 最终确认配置

`gate_init` 是 `weak_period_residual_gate_init`；`head=shared` 表示稠密头，
`head=pooled_r84` 表示 `weak_period_residual_head_type=pooled_lowrank`、
`pool_factor=1`、`rank=84`。

| setting | gate_init | lr | loss | head | rank | 达标 seed | 达标指标 | 达标 seed 的 MSE / MAE |
|---|---:|---:|---|---|---:|---|---|---|
| ETTm1-96 | 0.5 | 0.001 | `mae` | `shared` | — | 2022, 2023 | MAE | 2022: 0.300107 / 0.341759; 2023: 0.300731 / 0.341392 |
| Traffic-336 | 0.2 | 0.001 | `mae` | `pooled_lowrank` | 84 | 2022, 2023 | MAE | 2022: 0.396174 / 0.238683; 2023: 0.395962 / 0.239510 |
| Traffic-720 | 0.02 | 0.001 | `mae` | `shared` | — | 2022, 2023 | MAE | 2022: 0.438427 / 0.261801; 2023: 0.436780 / 0.261168 |

Golden references are respectively `0.293/0.344`, `0.385/0.248`, and `0.428/0.270`
(MSE/MAE). Thus all successful comparisons are MAE-only; no confirmed seed beats Golden
on MSE for these three settings.

## Reproduction commands

The full confirmation commands use the following common arguments, with the setting-specific
arguments in the table above and `--seed` set to each listed seed:

```bash
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer
$PY scripts/search_phaseformer.py \
  --dataset <DATASET> --horizon <H> --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed <SEED> --loss mae --percent 100 \
  --require-cuda --resume --num-workers 1 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '<SETTING_OVERRIDES>' --evaluate-test
```

The authoritative per-seed metrics and run identifiers remain in
`research_runs/phaseformer_L_targeted_100_v3/target_final.json` and
`target_stage1_all_rows.json`.
