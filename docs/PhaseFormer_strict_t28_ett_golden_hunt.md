# ETTh1/ETTm1 单 seed Golden 定向搜索

## 用户目标与评价

用户要求 ETTh1、ETTm1 在 H96、H192 都至少优于 Golden 0.5%，只按 seed=2021 判断，以尽快完成
定向搜索。阈值为：候选 MSE、MAE 必须分别不高于 Golden 的 `99.5%`。这是明确允许的
**test-set selection**，因此最终结果只能作为目标导向的选择轨迹，不能表述为盲测或泛化估计。

## 搜索与重跑策略

入口 `scripts/run_strict_t28_golden_hunt.py` 固定 strict-T28 单模型结构，仅搜索：

- cycle：ETTh1 为 24/48；ETTm1 为 24/48/96；
- trust region：off=`.02/.01/.005`、C=`.25/.10/.05`、W=`.60/.24/.12`、X=`.95/.50/.25`；
- loss：Huber/MAE；learning-rate multiplier：0.3/1/3。

每个配置在 H96、H192 依次 full-train（100%、最多 30 epoch、best-validation checkpoint）并读取一次
test，seed 固定为 2021。脚本把每项 test 与 Golden 的差和是否达到 0.5% 门槛写入紧凑 CSV。每条失败
命令自动重试最多 3 次；底层 `--resume` 保证中断后不会重复训练已完成实验，CSV 以配置 key 去重。

示例：

```bash
.venv/bin/python scripts/run_strict_t28_golden_hunt.py --dataset ETTh1
.venv/bin/python scripts/run_strict_t28_golden_hunt.py --dataset ETTm1
```

初始空间分别为 48/72 个配置（每个配置有两个 horizon）。达到两个 horizon 的双指标阈值后，停止该
数据集的后续搜索；若搜索空间耗尽仍无通过项，如实报告失败，不使用未记录的手调参数。
