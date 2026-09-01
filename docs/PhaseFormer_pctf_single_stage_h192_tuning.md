# 严格单阶段 PCTF：H192 五十次调参

## 目标与边界

本轮只优化随机初始化、一次 `Trainer.fit` 的三分支 PCTF，目标是在 ETTh2、ETTm2 的
L720→H192 上相对已有 two-stage `pctf_anchor_repair_full` 至少降低 **0.5%** 的 MSE/MAE
联合宏平均误差。这里的参照是已完成正式矩阵的 Full Repair validation 指标，而不是 A2 或
Golden；因此它直接回答“取消预训练+微调后能否超过 Full Repair”。

两阶段 Full Repair 的 test 已经历史性暴露。本轮五十个策略的选择严格只读取 validation，所有
候选不启动 test loader。若有且仅有一个策略通过下方预注册门槛，才可在未参与调参的 seed 2023
上运行一次 test 确认；该确认仍须在最终报告中披露既有 test 暴露，不能称为盲测。

## 归因与修复

`decoupled_protected` 已将最终 fused loss 对 A2 主输出的梯度置零，但融合器仍以 A2 的 phase、
trajectory 和 anchor 为输入。若这些输入未 detach，修正分支仍可能沿“修正依赖的 A2 特征”回传，
使 A2 不再只由自身 anchor loss 更新。这与上一轮内部 A2 相对独立 A2 仍有 `+0.066%` 联合
退化的现象一致。

因此本轮基准 T0 增加**严格输入梯度隔离**：融合器读取 A2 分支当前数值的 detached 副本，仍生成
`A2 + bounded ICPT correction`；A2 仅接受权重 1.0 的 anchor loss，ICPT/融合器仅接受 fused 和
component auxiliary losses。结构、推理信息流、参数量和单 checkpoint 形式都不变；变动仅是训练
时禁止融合器通过其输入改写 A2。

## 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | ETTh2、ETTm2 |
| 输入/输出 | 720→192 |
| cycle period | ETTh2=48、ETTm2=96 |
| 训练 | full train、Huber、最多 30 epoch；收敛组单独测试 36/45/60 epoch、best-validation checkpoint |
| seeds | 2021、2022；与 Full Repair validation 逐 seed 配对 |
| 训练方式 | 所有分支随机初始化、单次 `Trainer.fit`、无 checkpoint 初始化、无 warm-up |
| 预算口径 | 50 个共享策略；每策略覆盖两数据集×两 seed，共 200 个训练运行 |
| 设备 | 强制 CUDA；结果中审计 GPU/软件版本 |
| 测试集 | 不读取 |

## 五十个预注册策略

所有策略均启用严格隔离、A2 初始学习率 1×、anchor loss=1。前 35 个是单因素或残差子空间的
诊断，随后 9 个为预先固定的互补组合，最后 6 个检查收敛过程。除了最后的 warm-3，均无
correction warm-up。

| 策略组 | 数量 | 取值/改动 | 目的 |
|---|---|---|
| T00 | 1 | strict base | 修复未完全隔离的梯度路径 |
| T01–T08 | 8 | composer LR=`0.25, 0.40, 0.60, 0.80, 1.20, 1.50, 2.00, 2.50`× | 定位 ICPT 修正的优化速度 |
| T09–T15 | 7 | shape=level aux=`0, .01, .025, .075, .10, .15, .20` | 平衡 fused 和周期分量监督 |
| T16–T21 | 6 | gate aux=`0, .01, .025, .075, .10, .15` | 评估周期级系数监督强度 |
| T22–T28 | 7 | correction max=`.10, .15, .20, .30, .35, .45, .60`（其余边界同比缩放） | H192 的保守/充分修正边界 |
| T29–T34 | 6 | shape/level aux 非对称组合 | 判断长预测主要缺形状还是周期水平 |
| T35–T43 | 9 | LR×aux、LR×窄/宽边界、LR×gate 的预注册组合 | 验证互补而非事后拼接 |
| T44–T49 | 6 | epoch=`36/45/60`、patience=`12/16`、warm-up=3 | 检查是否为收敛时间而非结构瓶颈 |

## 选择与停止规则

以四个 dataset×seed×metric 的相对 Full Repair validation 比值计算联合宏平均。唯一冠军必须同时满足：

1. 联合宏平均比值 `≤0.995`（至少 0.5% 改善）；
2. 最差单个配对指标比值 `≤1.005`；
3. 不读取 test 后才冻结。

未满足则完整报告五十项结果并停止；不能用单项最好、单 seed 波动或已有 test 数值宣称达成目标。

## 运行

```bash
.venv/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage dry
.venv/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage smoke
.venv/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage run
.venv/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage summarize
```

`dry` 生成完整的 200 条正式训练命令；`smoke` 仅执行 4 个真实 CUDA 小任务（T00/T49 × ETTh2/ETTm2，
30% 数据、1 epoch、无 test）以覆盖两种 cycle 几何和最不同的训练路径。30% 是必要下限：ETTh2 在
batch=256 时 5% 数据没有完整 batch，无法通过训练入口的审计；`run` 才执行完整矩阵。

输出在 ignored 目录 `research_runs/pctf_single_stage_h192_tuning_v2/`。汇总器会拒绝缺失、重复、CPU
或任何带 test 指标的候选结果，并只接收 `candidates/` 中 `percent=100` 的正式记录；smoke 输出隔离
在 `smoke/`，不会混入排序。其余运行产物保持本仓库既有的实验忽略规则。

## Smoke 验证（2026-09-01）

首次 5% smoke 在 ETTh2 上发现训练集小于一个 batch，入口审计取 batch 时触发 `StopIteration`；已将
smoke 固定为 30%。修复后，T00 strict base 与 T49 warm-3 分别在 ETTh2-H192（cycle=48）和
ETTm2-H192（cycle=96）完成 1 epoch CUDA 训练、验证和 checkpoint 恢复：

| smoke | 数据 | 参数量 | 训练秒 | 峰值显存 | test 指标 |
|---|---|---:|---:|---:|---|
| T00 | ETTh2 | 165,297 | 0.88 | 475.6 MiB | 空 |
| T00 | ETTm2 | 168,349 | 2.47 | 356.5 MiB | 空 |
| T49 | ETTh2 | 165,297 | 0.99 | 475.6 MiB | 空 |
| T49 | ETTm2 | 168,349 | 2.34 | 356.5 MiB | 空 |

环境为 RTX 4090、PyTorch 2.7.1+cu126、Lightning 2.6.5。该 smoke 不参与正式汇总，也不是性能结论。

## 当前结果与默认参数更新

50 策略 H192 validation-only 筛选已完成。最佳是 T28（`trust_060`），即将 correction trust region
从 `0.25 / 0.10 / 0.05` 扩大为 `0.60 / 0.24 / 0.12`（分别为 correction、deformation、global-level
上限）；其余严格单阶段设置保持为 anchor/composer LR=1、anchor loss=1、完全梯度隔离、无 warm-up，
shape/level/gate auxiliary weight 均为 0.05。

| 配置 | 联合宏平均 vs Full Repair | 最差配对比值 | ETTh2 s2021 | ETTh2 s2022 | ETTm2 s2021 | ETTm2 s2022 |
|---|---:|---:|---:|---:|---:|---:|
| Full Repair（两阶段参照） | 1.0000 | 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| T00 strict base（0.25） | 1.0022 | 1.0078 | 0.9937 / 0.9948 | 1.0058 / 1.0044 | 1.0078 / 1.0051 | 1.0025 / 1.0039 |
| T28 trust_060（0.60） | **1.0019** | 1.0078 | **0.9930 / 0.9938** | 1.0078 / 1.0043 | 1.0069 / 1.0043 | 1.0021 / 1.0031 |

比值为 `candidate val / Full Repair val`，小于 1 表示更好。T28 相对 T00 仅改善 0.03 个百分点，主要
来自 ETTm2 两个 seed 的小幅下降；ETTh2 s2022 反而退化。因此 T28 是当前已知最佳**参数**，但仍未达到
`≤0.995` 的预注册门槛，不能声称单阶段超过 Full Repair 0.5%，也不读取新的 test。

对应实际 validation 指标（MSE / MAE）：

| 配置 | ETTh2 s2021 | ETTh2 s2022 | ETTm2 s2021 | ETTm2 s2022 |
|---|---:|---:|---:|---:|
| Full Repair | 0.2648 / 0.3528 | 0.2628 / 0.3506 | 0.1486 / 0.2624 | 0.1489 / 0.2628 |
| T00 | 0.2632 / 0.3510 | 0.2644 / 0.3522 | 0.1497 / 0.2637 | 0.1493 / 0.2638 |
| T28 | 0.2630 / 0.3506 | 0.2649 / 0.3522 | 0.1496 / 0.2635 | 0.1492 / 0.2636 |

`pctf_anchor_repair_full` preset 已更新为 T28 的三项 trust-region 参数。搜索 runner 显式固定 T00 的
历史 `0.25 / 0.10 / 0.05`，确保已有 T00–T49 表格在未来仍可复现而不会被新默认值污染。
