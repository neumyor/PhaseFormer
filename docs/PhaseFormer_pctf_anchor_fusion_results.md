# PCTF v2 复测结果（阶段性）

本轮按 `pctf_anchor_fusion_v2` 协议执行。Stage P 与 Stage S 均已完整完成；Stage F 按协议被
筛选门槛阻断，没有读取 test。

## 已完成结果

| 阶段 | 计划 | 已完成 | test 是否读取 | 状态 |
|---|---:|---:|---|---|
| Stage P 周期筛选 | 48 | 48 | 否 | 完成 |
| Stage S 策略筛选 | 132 | 132 | 否 | 完成 |
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

## Stage S 汇总（validation-only）

| 候选 | 宏平均 / A2 | 双指标改善 setting | 最差 / A2 | 宏平均 / 参考包络 | 通过 |
|---|---:|---:|---:|---:|---|
| anchor-shape-only（消融） | 1.001169 | 4/12 | 1.022089 | 1.007398 | 否 |
| anchor-level-only（消融） | 1.002013 | 3/12 | 1.022192 | 1.008246 | 否 |
| component-scalar | 1.002222 | 4/12 | 1.022081 | 1.008461 | 否 |
| component-cycle | 1.001223 | 3/12 | 1.022138 | 1.007449 | 否 |
| monotonic evidence | 1.001778 | 4/12 | 1.022241 | 1.008017 | 否 |
| MLP evidence | **1.001131** | 4/12 | **1.021385** | **1.007364** | 否 |
| phase modulation | 1.002373 | 4/12 | 1.023734 | 1.008615 | 否 |

最好的论文候选是 `pctf_anchor_mlp`，但仍未达到宏平均 ≤0.998、至少 8/12 个 setting 双指标
改善、最差 ≤1.01 和参考包络 ≤1.005 的联合门槛。相对 A2 的差异约为 +0.113%；最差 setting
仍约退化 2.14%。因此不能进入 Stage F，也不能宣称超过 A2 或 Golden。

设计对照显示：MLP 相对 component-cycle 的宏平均比为 0.999924，且有 9/12 个 setting 双指标
改善，说明证据特征可能有价值；但其最差比为 1.010829，仍不够稳定。component-cycle 相对
scalar 为 0.999001，说明逐周期系数略有帮助；phase modulation 则明显较差（1.001164）。

所有已落盘结果均为 validation-only，环境签名一致：RTX 4090、PyTorch 2.7.1+cu126、CUDA
12.6、Lightning 2.6.5。Stage S 中曾出现一次文件落盘竞态和一次 CUDA OOM；受影响的任务已
保留状态记录，未被填充为成功结果。完整逐 run 指标位于：
`research_runs/pctf_anchor_fusion_v2/screen/runs/*/metrics.csv`。

## 结论边界

当前不能声称任何候选策略通过预设门槛，也不能声称优于 A2 或 Golden。尽管 Stage S 已闭合，
所有候选仍未通过联合门槛，因此协议阻断 Stage F/test。

## 复现与继续执行

继续执行时使用：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen-summarize
```

建议在独占 GPU 的持久终端或作业调度器中运行，避免多个 runner 并发写同一 `runs/` 目录。
