# 严格单阶段 PCTF：H192 十次调参

## 目标与边界

本轮只优化随机初始化、一次 `Trainer.fit` 的三分支 PCTF，目标是在 ETTh2、ETTm2 的
L720→H192 上相对已有 two-stage `pctf_anchor_repair_full` 至少降低 **0.5%** 的 MSE/MAE
联合宏平均误差。这里的参照是已完成正式矩阵的 Full Repair validation 指标，而不是 A2 或
Golden；因此它直接回答“取消预训练+微调后能否超过 Full Repair”。

两阶段 Full Repair 的 test 已经历史性暴露。本轮十个策略的选择严格只读取 validation，所有
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
| 训练 | full train、Huber、最多 30 epoch（T9 为 45）、best-validation checkpoint |
| seeds | 2021、2022；与 Full Repair validation 逐 seed 配对 |
| 训练方式 | 所有分支随机初始化、单次 `Trainer.fit`、无 checkpoint 初始化、无 warm-up |
| 预算口径 | 10 个共享策略；每策略覆盖两数据集×两 seed，共 40 个训练运行 |
| 设备 | 强制 CUDA；结果中审计 GPU/软件版本 |
| 测试集 | 不读取 |

## 十个预注册策略

所有策略均启用严格隔离、A2/composer 初始学习率 1×、anchor loss=1、无 warm-up；每行只改变一个
可解释的训练因素。

| ID | 改动 | 目的 |
|---|---|---|
| T0 | strict base | 修复未完全隔离的梯度路径 |
| T1 | composer LR=0.5× | 防止 ICPT 早期过快破坏长预测 |
| T2 | composer LR=1.5× | 加快周期修正学习 |
| T3 | composer LR=2.0× | 检验更强的修正优化是否必要 |
| T4 | shape/level aux=0.025 | 减少辅助目标压制 fused 目标 |
| T5 | shape/level aux=0.10 | 强化零初始化 ICPT 的早期可学习性 |
| T6 | gate aux=0.10 | 加强周期级边际系数的监督 |
| T7 | 窄 trust region | 降低罕见大误差造成的 MSE 风险 |
| T8 | 宽 trust region | 允许较大跨周期水平/形状修正 |
| T9 | 最多 45 epoch | 检验单阶段是否主要受收敛时间限制 |

## 选择与停止规则

以四个 dataset×seed×metric 的相对 Full Repair validation 比值计算联合宏平均。唯一冠军必须同时满足：

1. 联合宏平均比值 `≤0.995`（至少 0.5% 改善）；
2. 最差单个配对指标比值 `≤1.005`；
3. 不读取 test 后才冻结。

未满足则完整报告十项结果并停止；不能用单项最好、单 seed 波动或已有 test 数值宣称达成目标。

## 运行

```bash
/home/niuyiming/.conda/envs/py310/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage dry
/home/niuyiming/.conda/envs/py310/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage run
/home/niuyiming/.conda/envs/py310/bin/python scripts/run_pctf_single_stage_h192_tuning.py --stage summarize
```

输出在 ignored 目录 `research_runs/pctf_single_stage_h192_tuning_v1/`。汇总器会拒绝缺失、重复、CPU
或任何带 test 指标的候选结果；其余运行产物保持本仓库既有的实验忽略规则。

## 结果（待运行）

| Rank | 策略 | 联合比值 vs Full Repair | 改善 | 最差比值 | 双指标改善 runs | 结论 |
|---:|---|---:|---:|---:|---:|---|
| — | — | — | — | — | — | 待运行 |
