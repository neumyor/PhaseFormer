# PCTF 单阶段联合训练实验

## 目的

当前 Full Repair 的正式结果来自“A2 预训练→加载 A2→联合微调”两阶段流程。该流程在
ETTh2/ETTm2 上有效，但总训练时间约为 A2 的 2.77–3.45 倍，而且已有归因表明部分收益来自
第二阶段继续训练 A2。本实验不改变三分支结构，只研究能否从随机初始化开始，在一次
`Trainer.fit` 中同时训练 PhaseFormer、LFF-NLinear、ICPT 和融合器，去掉预训练 checkpoint。

## 优化问题与假设

直接删除 A2 checkpoint 但沿用旧 preset 并不公平：旧 Full Repair 将 A2 主干学习率设为
0.1×，这是为了防止已训练锚点漂移；从零训练时会让尚未成形的 PhaseFormer/NLinear 主干
欠训练。同时，ICPT 的监督目标是 `y-stopgrad(A2)`，从零训练早期这个目标变化很快。

因此固定模型结构，只比较三项训练控制：

1. A2 主干学习率恢复为与 ICPT 相同的 1.0×；
2. 内部 A2 保护损失使用 0、0.25 或 1.0；
3. 可选 5 epoch correction warm-up：所有模块从第一个 epoch 就训练，ICPT 继续接受 level、
   shape 和 gate 辅助监督，但其修正对最终输出的比例从 0 线性升至 1。这仍是一次训练，不读取、
   保存或加载中间 A2 checkpoint。

warm-up 比例是 checkpoint 中的持久状态；历史 checkpoint 缺少该字段时按完整比例 1 兼容。

## Validation-only 筛选

固定 L720、PhaseFormer period 24、ETTh2 ICPT period 48、ETTm2 ICPT period 96，使用官方完整
训练划分、Huber、最多 30 epoch、best-validation checkpoint。筛选覆盖 ETTh2/ETTm2 的
H96/H192 和 seeds 2021/2022。第一轮包含 8 个 matched A2 与 48 个单阶段候选；不读取 test。

| policy | A2 LR | A2 loss | correction warm-up | 作用 |
|---|---:|---:|---:|---|
| `legacy_safe` | 0.1× | 1.0 | 0 | 直接删除预训练的朴素控制 |
| `uniform_unprotected` | 1.0× | 0 | 0 | 只恢复主干学习率 |
| `uniform_mild` | 1.0× | 0.25 | 0 | 轻度保护锚点 |
| `uniform_protected` | 1.0× | 1.0 | 0 | 完整保护锚点 |
| `warm5_mild` | 1.0× | 0.25 | 5 | 轻保护+平滑启用修正 |
| `warm5_protected` | 1.0× | 1.0 | 5 | 强保护+平滑启用修正 |

### 第一轮结果与归因

第一轮在提交 `7cb64cc`、RTX 4090、PyTorch 2.7.1+cu126、Lightning 2.6.5 上完成。以下均为
validation-only，`联合比` 是 8 个 setting×seed 上 16 个 MSE/MAE 比值的宏平均：

| policy | MSE/A2 | MAE/A2 | 联合比 | 最差比 | 双改善 | 内部A2/A2 | fused/内部A2 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| legacy_safe | 0.99950 | 0.99800 | 0.99875 | 1.01355 | 5/8 | 1.00078 | 0.99797 | 平均最好但尾部失败 |
| uniform_unprotected | 1.00000 | 0.99977 | 0.99988 | 1.00660 | 3/8 | 1.00358 | 0.99631 | 主干明显被拉偏 |
| uniform_mild | 0.99982 | 0.99975 | 0.99979 | 1.00628 | 3/8 | 1.00283 | 0.99696 | 保护不足 |
| uniform_protected | 0.99947 | 0.99891 | 0.99919 | 1.00476 | 3/8 | 1.00142 | 0.99777 | 统一LR中最好，仍未过门槛 |
| warm5_mild | 0.99980 | 0.99942 | 0.99961 | 1.00508 | 3/8 | 1.00148 | 0.99814 | 最佳点未完整启用修正 |
| warm5_protected | 0.99947 | 0.99911 | 0.99929 | 1.00467 | 3/8 | 1.00118 | 0.99812 | 最佳点未完整启用修正 |

六种策略均未通过预设门槛。0.1× LR 使候选平均训练 37.8 秒，1.0× 策略约 22–25 秒，证明
预训练式低 LR 不适合从零训练；但统一 LR 下内部 A2 仍比 matched A2 差 0.12%–0.36%，而
融合相对这个受损锚点通常改善，说明主要矛盾是 fused loss 和 anchor loss 对同一 A2 参数的
梯度干扰。warm-up 多次选择 correction scale 为 0、0.25 或 0.75 的 checkpoint，也说明只延迟
开启修正没有解决职责冲突。

据此追加一个验证集上的因果复测 `decoupled_protected`，不是事后扩大参数网格：前向计算和
部署模型完全不变，仍在一次 `Trainer.fit` 中同时训练全部模块；反向时 fused loss 只更新
ICPT/融合器，A2 仅由权重 1.0 的独立 anchor loss 更新。其 A2 LR 为 1.0×、无 correction
warm-up。该设计应在保留 ICPT 学习信号的同时，使内部 A2 接近 matched A2。追加 8 个候选后
按原门槛统一重算；若仍失败，停止且不读取 test。

为避免跨 commit 比较，复测在独立输出目录重跑同 commit 的 8 个 matched A2，只选择 8 个
`decoupled_protected` 候选：

```bash
.venv/bin/python scripts/run_pctf_single_stage_training.py \
  --stage screen-baselines \
  --output-root research_runs/pctf_single_stage_training_decoupled_v1 \
  --policies decoupled_protected
.venv/bin/python scripts/run_pctf_single_stage_training.py \
  --stage screen-candidates \
  --output-root research_runs/pctf_single_stage_training_decoupled_v1 \
  --policies decoupled_protected
.venv/bin/python scripts/run_pctf_single_stage_training.py \
  --stage screen-summarize \
  --output-root research_runs/pctf_single_stage_training_decoupled_v1 \
  --policies decoupled_protected
```

共享训练策略只有同时满足以下条件才冻结进入正式 test：16 个 MSE/MAE 比值宏平均 `<0.998`，
8 个 setting×seed 中至少 6 个双指标改善，最差比值 `≤1.01`，且入选 checkpoint 的 correction
scale 已达到 1。若无策略通过，不读取新 test，而是根据内部锚点/A2 和 fused/内部锚点比值判断
问题来自主干欠拟合还是 ICPT 修正。

## 正式确认

冻结的统一策略在相同四个 setting 上运行 seeds 2021/2022/2023；每个 run 都从随机初始化开始，
只执行一次最多 30 epoch 的训练。重新训练 12 个同 commit matched A2，并对两者 best-validation
checkpoint 各读取一次 test。正式局部替换门槛与上一轮一致：八个 setting-level MSE/MAE 比值
宏平均 `<0.998`，至少 3/4 setting 双指标改善，最差回退不超过 0.5%。同时报告固定 Golden、
参数量、训练时间与峰值显存，并和历史两阶段 Full Repair 对照。

虽然训练策略只由 validation 选择，但 ETTh2/ETTm2 test 已在上一轮暴露，因此正式结果不能称为
首次盲测；本轮正式 test 之后不得再用这些数值选择单阶段策略。

## 命令与状态

```bash
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage screen-baselines-dry
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage screen-candidates-dry
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage screen-baselines
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage screen-candidates
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage screen-summarize
```

若验证门槛通过，再执行：

```bash
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage formal-baselines-dry
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage formal-candidates-dry
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage formal-baselines
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage formal-candidates
.venv/bin/python scripts/run_pctf_single_stage_training.py --stage formal-summarize
```

第一轮与解耦复测输出分别位于 `research_runs/pctf_single_stage_training_v1/` 和
`research_runs/pctf_single_stage_training_decoupled_v1/`。逐 run 指标保留在这些 ignored
目录中，不提交 checkpoint；复测未过门槛，所以上述正式确认命令没有执行。

## 梯度解耦复测结果

复测在提交 `5bf0534`、同一 RTX 4090/软件环境完成，8 个 matched A2 与 8 个候选均来自随机
初始化且只调用一次训练；所有 `test_mse/test_mae` 字段为空。

| policy | MSE/A2 | MAE/A2 | 联合比 | 最差比 | 双改善 | 内部A2/A2 | fused/内部A2 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decoupled_protected | 0.99914 | 0.99902 | 0.99908 | 1.00537 | 3/8 | 1.00066 | 0.99842 | 未过门槛，不进 test |

| setting | MSE/A2 | MAE/A2 | 内部A2/A2（联合） | fused/内部A2（联合） |
|---|---:|---:|---:|---:|
| ETTh2-H96 | 0.99837 | 0.99832 | 0.99946 | 0.99889 |
| ETTh2-H192 | 1.00340 | 1.00148 | 1.00542 | 0.99703 |
| ETTm2-H96 | 0.99349 | 0.99529 | 0.99523 | 0.99915 |
| ETTm2-H192 | 1.00132 | 1.00101 | 1.00255 | 0.99862 |

解耦使内部 A2/A2 从 `uniform_protected` 的 1.00142 降至 1.00066，并使融合相对内部 A2
在 8/8 行改善 MAE、7/8 行改善 MSE；因此“融合损失干扰主干”的归因得到支持。但候选仍仅
3/8 双指标优于 matched A2：其 best-fused checkpoint 与独立 A2 的 best-anchor epoch 不总是
重合，ETTh2-H192 的内部锚点联合退化 0.54%，虽然融合修正追回约 0.30%，仍不足以填平差距。
这说明硬解耦修复了梯度职责，却没有解决两个子目标收敛时间不同和单 checkpoint 选择的问题。

候选平均训练 22.98 秒，matched A2 为 12.13 秒，即单阶段成本约为 A2 的 1.90 倍；历史两阶段
流程为 A2 的 2.77–3.45 倍，因此单阶段节省约 32%–45% 的总训练时间，但目前以稳定精度为代价。

## 决策：单阶段最好如何训练

在必须一次训练时，当前最合理的配置是 `decoupled_protected`：A2 与 ICPT/融合器都从 epoch 0
训练且使用 1.0× LR；前向始终输出 `A2 + bounded ICPT correction`；A2 只接受权重 1.0 的
anchor loss，fused loss 与 level/shape/gate 辅助损失只训练 ICPT/融合器；不使用 correction
warm-up，并按 fused validation loss 选择唯一 checkpoint。它比低 LR、无保护或 warm-up 更能
保持锚点，同时成本最低且职责清楚。

但这只是“被迫单阶段时的首选训练法”，不能替换当前两阶段 Full Repair：它未达到 0.2% 平均
改善和 6/8 稳定性门槛，所以未运行正式 test，也不能声明相对 Golden 有新提升。下一轮若继续，
应预注册研究单 checkpoint 的多目标选择或使两条优化轨迹同步收敛，而不是继续扫描 anchor loss
权重；任何新策略仍须先过 validation 门槛才读取 test。
