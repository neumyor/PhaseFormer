# ETTh1 Strict-T28 参数重推导计划

## 目标、已知证据与边界

目标是在 ETTh1 的 H96/H192 上寻找一个**数据集共享**的 strict-T28 配置，优先通过 validation
验证，再与固定 Golden 做一次三 seed test 对比。模型结构不变：完整 A2 为预测锚点，ICPT 只加入
有界的周期水平/形状修正；全程是一个 checkpoint、一次性联合训练。

现有一次冻结配置的 test 已经暴露：T28-W、cycle=48、Huber 的 H96/H192 均低于 Golden，详见
`docs/PhaseFormer_strict_t28_etth1_test.md`。更重要的是，同协议历史 A2 在 ETTh1-H96/H192 约为
`0.3663/0.3966` 与 `0.4027/0.4191`（MSE/MAE），也低于 Golden `0.359/0.382` 与
`0.397/0.404`。因此本轮检验的不只是“放大/缩小修正是否有用”，还包括训练目标是否是 ETTh1
的主要失配来源；不得承诺一定超过 Golden。

用户已要求以 Golden 为目标，故本计划允许最终读取新的 candidate test；任何以此后 test 数字继续
改变参数的动作均须标为 **test-set selection**，不能称盲测。

## 统一搜索空间

每个候选同时用于 H96 和 H192，禁止按 horizon 选参数：

| 因子 | 取值 | 理由 |
|---|---|---|
| ICPT cycle | 24、48 | 均可整除两个 horizon；24 是更短的日内循环，48 复核原 T28 设置。 |
| trust region | C=`0.25/0.10/0.05`，M=`0.40/0.16/0.08`，S=`0.50/0.20/0.10`，W=`0.60/0.24/0.12` | 依次为 correction/deformation/global-level，检验“ETTh1 是否需要更弱修正”。 |
| loss | Huber、MAE | Huber 保持 T28 原设置；MAE 直接检验大误差是否主导参数更新。 |

其他 strict 约束固定：lookback=720、A2 输入对 composer 全 stop-gradient、fused/anchor 梯度解耦、
anchor/composer LR=1、anchor loss=1、shape/level/gate 辅助权重均 0.05、无 warm-up、base capacity
和数据集默认 learning rate。总计 `2 × 4 × 2 = 16` 个数据集级候选。

## Stage A：低成本 validation 筛选

16 个配置 × H96/H192，30% train、8 epoch、seed=2021、best-validation checkpoint、**不读取 test**，
共 32 runs。对每个 horizon 的 MSE/MAE 分别除以该 horizon 的候选最小值，再在四项指标上求均值。

晋级规则：保留综合前二，且候选相对 Huber+W 的两个 horizon 均没有任一验证指标退化超过 0.5%。
如果无人满足，停止：说明周期尺度、边界与 loss 的该小空间无法修复 ETTh1，不以 test 反复搜寻。

## Stage B：全数据 validation 确认

两名晋级者 × H96/H192 × seeds 2021/2022，100% train、最多 30 epoch、validation-only，共 8 runs。
以八项相对 Huber+W 对照的 MSE/MAE 平均选出唯一配置；若前两名差小于 0.2%，选择更短 cycle，之后
选择 Huber（保持更平滑的主损失）。仍不读取 test。

## Stage C：用户授权的 Golden test

将唯一冻结配置在 H96/H192 × seeds 2021/2022/2023 各训练一次，best-validation checkpoint 后各读取
一次 test，共 6 runs。报告 mean±sample std 与固定 Golden；Delta 定义为 candidate minus Golden。
只有两项均低于 Golden，且 `mean + std < Golden`，才称为稳定双指标超过。

若 Stage C 未通过，不再按 test 数值继续调参；保留完整的失败搜索轨迹，并将结论限定为该结构/训练空间
没有支持 ETTh1 超越 Golden 的证据。

## 可复现入口

使用 `scripts/search_phaseformer.py`，Stage A 通过 JSON overrides 指定 trust region。例如 cycle=24、
M、MAE 的 H96 run：

```bash
.venv/bin/python scripts/search_phaseformer.py \
  --dataset ETTh1 --horizon 96 --stage hp_low \
  --mechanism pctf_anchor_repair_strict_t28 --period 24 --cycle-period 24 \
  --loss mae --percent 30 --max-epochs 8 --seed 2021 --num-workers 0 \
  --overrides '{"anchored_pctf_correction_max":0.40,"anchored_pctf_deformation_max":0.16,"anchored_pctf_global_level_max":0.08}' \
  --output-dir research_runs/pctf_strict_t28_etth1_retune_v1 --require-cuda --resume
```
