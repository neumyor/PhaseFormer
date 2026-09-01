# Strict-T28 全数据集 Golden 对比计划

## 目标与冻结对象

本计划测试一个统一的、随机初始化且单次 `Trainer.fit` 训练的模型：
`pctf_anchor_repair_strict_t28`。它将 A2 的完整相位—NLinear/LFF 预测作为锚点，只增加有界的
ICPT 周期 level/shape 修正；融合器对 A2 输入完全 stop-gradient，A2 仅由 anchor loss 学习。

已冻结的通用训练设置为：lookback=720、Huber、最多 30 epoch、best-validation checkpoint、
anchor/composer LR=1、anchor loss=1、shape/level/gate auxiliary weight 均为 0.05、无 warm-up。
T28 在 ETTh2/ETTm2 H192 的小范围 validation 搜索中使 correction/deformation/global-level 边界
取 `0.60/0.24/0.12`，但尚未超过 two-stage Full Repair；它是本轮的**起点**，不是全数据集最优的
既定结论。

固定 Golden 覆盖 ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic 的四个 horizon，
共 28 个 setting。Exchange 没有权威 Golden，因此不混入本轮主结论。

## 为什么不能直接跑 84 个 test

不同数据集的周期长度与可接受修正幅度可能不同；但同一数据集的四个 horizon 不应随意切换机制。
更关键的是，当前 composer 要求 `pred_len % cycle_period == 0`：历史 ETTm2 的 cycle=96 不适用于
H336。因此必须先基于 validation 为每个数据集冻结一个能覆盖四个 horizon 的周期和 trust-region
档位，再读取 test。

## Stage A：低成本周期可行性筛选

使用 strict-T28 的原始边界、30% train、8 epoch、seed=2021、validation-only，在 H96/H336 两端筛选：

| 数据集 | 候选 cycle | run 数 | 原因 |
|---|---|---:|---|
| ETTm2 | 24、48 | 4 | 96 无法整除 H336；需替代历史 96 |
| Traffic | 12、24、48 | 6 | 尚无 anchored-PCTF 冻结周期 |

其它数据集沿用已有训练期冻结且四 horizon 均整除的周期：ETTh1=48、ETTh2=48、ETTm1=48、Weather=24、
Electricity=12。每个数据集按 H96/H336 的 MSE/MAE 相对本数据集各周期最优值的联合均值排序；若前二
差小于 0.2%，优先较短周期，降低模型复杂度。此阶段不读取 test。

## Stage B：数据集级 trust-region 推导

周期冻结后，在所有 7 个数据集、H96/H336、30% train、8 epoch、seed=2021 上比较三个统一档位：

| 档位 | correction / deformation / global-level |
|---|---|
| C（保守） | 0.25 / 0.10 / 0.05 |
| M（中等） | 0.40 / 0.16 / 0.08 |
| S（平滑插值） | 0.50 / 0.20 / 0.10 |
| W（T28） | 0.60 / 0.24 / 0.12 |

共 `7 × 2 × 4 = 56` 个 validation-only run。每个数据集只冻结一个档位，并原样外推至其 H192/H720；
不按 horizon 挑选不同档位。选择分数为四个 MSE/MAE 比值相对 C 的平均，附加约束为任一端点单指标
退化不超过 0.5%。不通过时仍保守选 C 并标记该数据集“无可信 correction 扩张收益”。

## Stage C：确认性 validation

对 7 个冻结的“周期+档位”组合，在四个 horizon、100% train、seeds 2021/2022 做 validation-only
复核，共 `7 × 4 × 2 = 56` 个 run。该阶段只验证跨 horizon 稳定性；不再改变周期、档位、损失或
容量。若某数据集四 horizon 的 16 个 MSE/MAE 比值中任一值相对 C 回退超过 0.5%，该数据集回退到 C。

## Stage D：正式 Golden 对比

冻结后，strict-T28 在 28 个 Golden setting 上使用 full train、seeds 2021/2022/2023、best-validation
checkpoint、一次 test，共 **84** 个 candidate run；不需要重跑 Golden。

每格报告 MSE/MAE mean±sample std、相对 Golden 的绝对/百分比变化、训练时间、峰值显存、参数量。
只有每个 seed 的 MSE/MAE 都低于 Golden，且 `mean + std < Golden`，才称为该 setting 稳定双指标超过
Golden。由于历史 two-stage Full Repair 的 test 已暴露，ETTh2/ETTm2 的结果须披露相关 test exposure，
不得描述为完全盲测。

## 当前 Stage A/B 进度（2026-09-01 更新）

Stage A 全部完成。ETTm2 由先导探测冻结 cycle=24；Traffic 在本轮 4×A100 补跑
（30% train、8 epoch、seed 2021、validation-only，输出
`research_runs/pctf_strict_t28_global_golden_v1/`）。

Traffic Stage A 结果（validation MSE/MAE）：

| Horizon | cycle=12 | cycle=24 | cycle=48 |
|---|---:|---:|---:|
| 96 | 0.328997 / 0.217733 | 0.329051 / 0.218076 | 0.328870 / 0.218076 |
| 336 | 0.346189 / 0.217846 | 0.345601 / 0.218260 | 0.345822 / 0.218510 |

Traffic 各 cycle 的联合相对分数（四个单元格相对各单元格最优值的均值）：cp12=1.000522、
cp24=1.001006、cp48=1.001315。cp12 为最优且同时是最短周期，规则无歧义：**冻结 Traffic cycle=12**。
决策文件：`research_runs/pctf_strict_t28_global_golden_v1/frozen_decisions.json`。

Stage B（7 数据集 × H96/H336 × 4 档，30% train、8 epoch、seed 2021，共 56 run）完成。
每数据集按”四个 MSE/MAE 比值相对 C 的均值”选档；约束为任一端点单指标相对 C 退化 ≤0.5%，
且该档平均上必须相对 C 有可信改进（mean<1.0）；不满足则保守冻结 C。

| 数据集 | cycle | C(0.25/0.10/0.05) | M(0.40/0.16/0.08) | S(0.50/0.20/0.10) | W(0.60/0.24/0.12) | 冻结档位 |
|---|---|---:|---:|---:|---:|---|
| ETTh1 | 48 | 0.70410/0.58372/1.28985/0.79255 | 0.99701 | 0.99519 | 0.99349 | W |
| ETTh2 | 48 | 0.20995/0.31898/0.37094/0.42765 | 0.99706 | 0.99516 | 0.99342 | W |
| ETTm1 | 48 | 0.42305/0.43538/0.65035/0.53597 | 1.00028 | 1.00039 | 0.99909 | W |
| ETTm2 | 24 | 0.12039/0.23733/0.19901/0.30257 | 0.99942 | 0.99962 | 0.99977 | M |
| Weather | 24 | 0.41908/0.29968/0.53018/0.37861 | 0.99426 | 0.99298 | 0.99800 | S |
| Electricity | 12 | 0.11269/0.20624/0.14082/0.23427 | 0.99909 | 0.99940 | 0.99959 | M |
| Traffic | 12 | 0.32862/0.21771/0.34548/0.21777 | 1.00130 | 1.00138 | 1.00091 | C（无可信扩张收益） |

表中 C 列为 H96/H336 的 `mse/mae/mse/mae`；M/S/W 列为四个比值相对 C 的均值。

## Stage C 确认结果与主研究者决策（2026-09-01）

Stage C 对 ETTh1/ETTh2/ETTm1/ETTm2/Weather 五个数据集完成（7 冻结组合 × 4 horizon ×
seeds 2021/2022，100% train，validation-only）。Electricity 与 Traffic 的 Stage C 由主研究者
指令跳过：Electricity 直接确认 M（不执行回退检查），Traffic 保持 Stage B 的 C。

已执行数据集的 16 比值回退检查（任一 frozen/C 比值 >1.005 回退到 C）：

| 数据集 | 冻结档位 | worst 比值 | 触发 cell | 结论 |
|---|---|---:|---|---|
| ETTh1 | W | 1.0074 | mse@H336·s2021 | **回退到 C** |
| ETTh2 | W | 1.0068 | mse@H720·s2022 | **回退到 C** |
| ETTm1 | W | 1.0018 | 无 | 保持 W |
| ETTm2 | M | 1.0055 | mse@H720·s2021 | **回退到 C** |
| Weather | S | 1.0047 | 无 | 保持 S |

三个触发回退的数据集各仅有一个 MSE cell 略超 1.005（超 0.05%–0.74%），其余 15 个比值均低于
阈值、16 比值均值均在 1.0 附近；但按预注册规则严格回退到保守档 C。最终冻结档位：

| 数据集 | cycle | 档位 |
|---|---|---:|---|
| ETTh1 | 48 | C |
| ETTh2 | 48 | C |
| ETTm1 | 48 | W |
| ETTm2 | 24 | C |
| Weather | 24 | S |
| Electricity | 12 | M（主研究者确认） |
| Traffic | 12 | C |

Stage D（28 setting × 3 seed，full train + test）已据此启动。

## 可复现执行与参数治理

所有筛选通过 `scripts/search_phaseformer.py` 执行，必须添加 `--require-cuda`，且不添加 `--test`；
每个 run 的 `metrics.csv` 是唯一比较来源。Stage A 的命令模板如下（替换数据集、horizon 与
`--cycle-period`）：

```bash
.venv/bin/python scripts/search_phaseformer.py \
  --dataset ETTm2 --horizon 96 --stage period_screen \
  --mechanism pctf_anchor_repair_strict_t28 --period 24 --cycle-period 24 \
  --lookback 720 --percent 30 --max-epochs 8 --seed 2021 --loss huber \
  --num-workers 0 --bad-case-limit 0 \
  --output-dir research_runs/pctf_strict_t28_global_pilot_v1/period \
  --require-cuda --resume
```

参数的可选择范围严格限于本计划的 dataset-shared `cycle_period` 和 trust-region 档位；学习率、
loss、容量、辅助权重、seed 与训练协议不随数据集改变。这样可以检验“周期尺度和修正幅度是否随
数据生成频率不同”这一假设，而不是以大量数据集专属旋钮追逐指标。每次冻结均需在本文件写入
validation 分数、tie-break 及失败/OOM 状态，随后才允许进入下一阶段。
