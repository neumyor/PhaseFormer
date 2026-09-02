# Strict-T28 最优共享配置：H336/H720 扩展

## 固定配置

以已完成的 H96/H192 test-selection ledger 中、各数据集四项 Golden 相对指标平均值最低的**共享配置**为准：

| 数据集 | 固定配置 | 选择依据 |
|---|---|---|
| ETTh1 | `u_lr020`：cycle=24，correction/deformation/level=`1.40/0.80/0.40`，MAE，LR multiplier=0.20 | H96/H192 四指标平均比值 0.9942 |
| ETTm1 | `w_aux01`：cycle=24，`0.60/0.24/0.12`，MAE，LR multiplier=0.20，shape/level/gate aux 均为 0.01 | H96/H192 四指标平均比值 0.9957 |

模型拓扑固定为“完整 A2 预测为锚点 + 两个受限周期修正”。每个 setting 重新完整训练，lookback=720、seed=2021、最多 50 epoch、best-validation checkpoint、RTX 4090/CUDA，并仅在训练结束后读取一次 test。

## H336/H720 结果

| 数据集 | Horizon | Golden MSE/MAE | Strict-T28 MSE/MAE | MSE Δ | MAE Δ |
|---|---:|---|---|---:|---:|
| ETTh1 | 336 | 0.425 / 0.424 | 待填 | 待填 | 待填 |
| ETTh1 | 720 | 0.431 / 0.450 | 待填 | 待填 | 待填 |
| ETTm1 | 336 | 0.358 / 0.381 | 待填 | 待填 | 待填 |
| ETTm1 | 720 | 0.412 / 0.410 | 待填 | 待填 | 待填 |

运行：

```bash
/home/wangjing/miniconda3/envs/raft/bin/python \
  scripts/run_strict_t28_best_long_horizons.py --dataset all
```

结果写入被忽略的 `research_runs/strict_t28_best_long_horizons/test_selection_results.csv`。由于配置已使用 H96/H192 的 test 选择，该长 horizon 扩展也必须标注为 test-set selection，不能表述为盲测。
