# PCTF v2 复测结果（阶段性）

本轮按 `pctf_anchor_fusion_v2` 协议执行。Stage P 已完整完成；Stage S 已完成 111/132 个
validation-only setting，尚有 21 个 setting 未闭合，因此没有执行 Stage F，也没有读取 test。

## 已完成结果

| 阶段 | 计划 | 已完成 | test 是否读取 | 状态 |
|---|---:|---:|---|---|
| Stage P 周期筛选 | 48 | 48 | 否 | 完成 |
| Stage S 策略筛选 | 132 | 111 | 否 | 未完成 |
| Stage F 三 seed 确认 | 144 | 0 | 否 | 按协议阻断 |

Stage P 冻结的 ICPT 周期如下：

| 数据集 | 冻结周期 |
|---|---:|
| ETTh1 | 48 |
| ETTh2 | 48 |
| ETTm1 | 48 |
| ETTm2 | 96 |
| Weather | 24 |
| Electricity | 12 |

所有已落盘结果均为 validation-only，环境签名一致：RTX 4090、PyTorch 2.7.1+cu126、CUDA
12.6、Lightning 2.6.5。Stage S 中曾出现一次文件落盘竞态和一次 CUDA OOM；受影响的任务已
保留状态记录，未被填充为成功结果。完整逐 run 指标位于：
`research_runs/pctf_anchor_fusion_v2/screen/runs/*/metrics.csv`。

## 结论边界

当前不能声称任何候选策略通过预设门槛，也不能声称优于 A2 或 Golden。原因是 Stage S 矩阵
尚未完整，汇总器会拒绝缺失 setting；在矩阵闭合并通过筛选门槛前，协议禁止进入 test 阶段。

## 复现与继续执行

继续执行时使用：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen-summarize
```

建议在独占 GPU 的持久终端或作业调度器中运行，避免多个 runner 并发写同一 `runs/` 目录。
