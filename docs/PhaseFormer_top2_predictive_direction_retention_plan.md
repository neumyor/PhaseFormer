# PhaseFormer 前两预测方向数据保留实验计划

> 状态：预注册计划，尚未实现、尚未训练、尚无结果
>
> 日期：2026-09-16
>
> 目标：在数据层面只保留已识别的前一或前两个预测方向，端到端训练 PhaseFormer，
> 测量它们能够保留多少完整 NLinear 支路性能

## 1. 要回答的问题

已有 reduced-rank regression（RRR）分析表明，NLinear 支路的可实现收益高度集中：

- 方向 1 对应“近端指数加权的近期平均水平 → 预测区间整体位移”；
- 方向 2 是去除整体位移后的第一种主要形状校正，在 ETT 中主要表现为周期相位/幅度或
  缓慢趋势，在 Weather 中主要表现为缓慢趋势和曲率；
- 前两个方向在当前 6 个 ETT/Weather setting 上，理论上可保留完整线性映射
  **85.8%–96.7%** 的可实现收益。

本实验不再改变 NLinear 的参数量，也不手工近似“均值、趋势或周期”特征，而是直接在
NLinear 已中心化的输入数据上保留训练集 RRR 得到的精确方向。实验只回答：

1. 只给 NLinear 方向 1，端到端模型能保留多少性能？
2. 再加入方向 2，能否稳定恢复方向 1 丢失的主要性能？
3. 前两个方向是否足以让最终模型接近完整输入 NLinear？

## 2. 预注册假设

### H1：方向 1 是主要信息

只保留方向 1 的模型应保留完整 NLinear 相对 phase-only 的大部分收益，但在依赖周期形状
或趋势变化的 setting 上允许出现可测退化。

### H2：方向 2 提供主要的形状补充

保留方向 1+2 后，模型应稳定优于只保留方向 1，并接近完整输入 NLinear。该增量应在
方向 2 理论贡献较大的 setting 上更明显。

### H3：如果方向解释正确，性能顺序应具有单调性

在相同训练协议下，预期总体满足：

```text
phase-only < keep-direction-1 <= keep-direction-1-and-2 ≈ direct-NLinear
```

这里的顺序指 NLinear 所带来的有效收益，不要求每个随机种子的 MSE 和 MAE 都严格单调。

## 3. 实验范围

本计划中的“ETT 和 Weather”严格限定为已经完成 RRR 方向分析的 6 个 setting，不临时扩展
到未计算方向的其他数据集或 horizon：

| Setting | 方向 1 理论 capture | 方向 1+2 理论 capture |
|---|---:|---:|
| ETTh2-96 | 75.6% | 85.8% |
| ETTh2-720 | 78.7% | 89.3% |
| ETTm2-96 | 73.9% | 86.0% |
| ETTm2-192 | 75.0% | 86.2% |
| Weather-96 | 86.2% | 96.7% |
| Weather-192 | 79.4% | 91.3% |

上表是已有最优秩映射在 validation 上的理论参照，不是本次端到端训练结果，也不作为
结果回填值。

## 4. 两个模型变体

### V1：仅保留方向 1

NLinear 只能看到中心化历史在第一预测输入方向上的投影：

```text
z = x - x_last
z_keep = Q1 Q1^T z
```

其中 `Q1` 是方向 1 的单位正交基。

### V2：保留方向 1 和方向 2

NLinear 只能看到中心化历史在前两个预测输入方向张成子空间中的投影：

```text
z = x - x_last
z_keep = Q12 Q12^T z
```

其中 `Q12` 是方向 1、2 输入函数经过正交化得到的二维基。

两种变体均使用：

```text
y_nlinear = Linear(z_keep) + x_last
```

投影放在 NLinear 完成 `x_last` 中心化之后、线性层之前。投影器固定，不参与梯度更新；
NLinear 线性层、PhaseFormer 主干、融合 gate 和其他原有参数全部正常端到端训练。

## 5. 方向的计算与冻结

每个 dataset×horizon 独立使用其训练 split 计算一次方向：

```text
Z = x - x_last
D = y - x_last
Szz = E[ZZ^T]
Szy = E[ZD^T]
S = Szy^T (Szz + εI)^(-1) Szy
```

对 `S` 做特征分解，按特征值从大到小取 `u1,u2`，对应输入函数为：

```text
b_i = u_i^T Szy^T (Szz + εI)^(-1)
```

随后分别对 `{b1}` 和 `{b1,b2}` 正交化，得到 `Q1` 和 `Q12`。

约束：

- 只读取训练 split，不读取 validation/test 的输入或标签；
- 同一 setting 的三个随机种子共享同一个固定投影器；
- 保存 `Szz/Szy` 来源、样本数、ridge、特征值、基矩阵形状和文件哈希；
- 检查 `Q^TQ≈I`、投影幂等性和方向信息保持误差；
- 记录 `λ2/Σλ`、`λ3/Σλ` 与 `(λ2-λ3)/λ2`。若第二、第三特征值接近，明确标注
  “方向 2 的单独朝向可能不稳定”，但不在看到结果后改成保留三维。

## 6. 对照与控制变量

正式比较包含四个实验臂，其中只有 V1、V2 是新增变体：

| 实验臂 | NLinear 可见信息 | 作用 |
|---|---|---|
| `phase_only` | 无 NLinear | 测量 NLinear 的总贡献 |
| `direct_nlinear` | 完整 720 步中心化历史 | 完整信息对照 |
| `keep_direction_1` | 方向 1 | 变体 V1 |
| `keep_direction_1_2` | 方向 1+2 | 变体 V2 |

固定不变：

- 数据划分、lookback、horizon、scaler 和通道共享方式；
- PhaseFormer 主干、NLinear 线性层尺寸和可训练参数量；
- gate 结构及初始化、loss、optimizer、学习率、batch size、epoch 和 early stopping；
- 每个 setting 既有条件性秩实验已经冻结的训练配置；
- seed 为 2021、2022、2023；
- best-validation checkpoint 选择规则和 test 评估代码。

允许复用已有 `phase_only` 与 `direct_nlinear` 结果，但必须同时满足：setting、seed、数据划分、
训练配置、代码语义和指标实现完全一致，且 checkpoint 审计通过。任一项不一致即配对重训，
不得混用金标准或其他实验协议的数字。

## 7. 分阶段执行

### Stage 0：无训练方向审计

对 6 个 setting 生成 `Q1/Q12`，完成以下检查后才能训练：

1. 训练样本数与既有数据划分一致；
2. 特征值降序、方向范数、正交误差和投影幂等误差通过；
3. V1 投影后只能恢复 `b1^Tz`，V2 能同时恢复 `b1^Tz`、`b2^Tz`；
4. validation/test 标签未参与矩估计；
5. 对一个固定 batch 验证 V1、V2 和 direct 的输入确实不同；
6. 投影器在不同 seed 运行中哈希一致。

### Stage A：单随机种子链路验证

- setting：全部 6 个；
- seed：2021；
- 新训练：V1、V2，共 12 次；
- 只读取 validation 指标；
- 目的：确认训练稳定、投影生效、指标与 gate 记录完整。

Stage A 的结果不用于修改方向、不新增第三个变体，也不按数据集调整超参数。只要实现审计
通过，无论结果正负都进入 Stage B。

### Stage B：三随机种子正式验证

- 在相同 6 个 setting 上补 seed 2022、2023，共新增 24 次训练；
- 汇总 2021/2022/2023 的 validation 均值和 sample standard deviation；
- 所有训练与审计通过后，对四个实验臂的冻结 checkpoint 各读取一次 test；
- 不依据 test 结果修改 projector、训练配置或模型结构。

预计新增训练总量为 **36 次**。若已有两个对照不能严格复用，最多增加 36 次配对对照训练。

## 8. 主要指标

### 8.1 最终预测指标

- test MSE，三 seed 均值 ± sample standard deviation；
- test MAE，三 seed 均值 ± sample standard deviation；
- 相对完整 NLinear 的变化：

```text
delta_direct = (metric_variant / metric_direct - 1) × 100%
```

### 8.2 NLinear 贡献保留率

分别对 MSE 和 MAE 计算：

```text
retention(V) =
    (metric_phase_only - metric_V)
    / (metric_phase_only - metric_direct) × 100%
```

解释：

- `100%`：完整保留 direct NLinear 的有效贡献；
- `0%`：与 phase-only 相同；
- `>100%`：优于 direct；
- `<0%`：比 phase-only 更差。

若 `direct_nlinear` 未优于 `phase_only`，该 setting 的贡献保留率记为 `N/A`，只报告绝对
MSE/MAE 和相对 direct 的变化，避免用接近零或反号的分母制造误导。

### 8.3 方向 2 的增量价值

```text
direction2_recovery =
    (metric_V1 - metric_V2)
    / (metric_V1 - metric_direct) × 100%
```

仅在 V1 差于 direct 时计算。它直接回答：加入方向 2 后，恢复了 V1 与完整输入之间多少差距。

### 8.4 诊断指标

- learned gate 的三 seed 均值和标准差；
- NLinear 分支单独输出的 MSE/MAE；
- V1/V2 的投影输入方差占原中心化输入方差比例；
- `b1^Tz`、`b2^Tz` 的样本标准差；
- 最佳 epoch、训练时间、参数量和峰值显存；
- 三 seed 的方向一致性和异常终止情况。

## 9. 预注册判定

### 强支持“前两个方向基本足够”

同时满足：

1. V2 相对 direct 的六 setting 宏平均 MSE 和 MAE 均不劣于 `+0.5%`；
2. V2 的 MSE/MAE 贡献保留率中位数均不低于 `90%`；
3. V2 相对 V1 在至少 4/6 setting 的三 seed 均值上双指标更好；
4. 没有 setting 的 V2 相对 direct 出现超过 `2%` 的 MSE 或 MAE 退化。

### 部分支持

V2 明显优于 V1，但未达到上述接近 direct 的门槛；结论写为“方向 2 有稳定增量价值，但前两个
方向不足以完整替代原输入”。

### 不支持

出现任一情况：

- V2 在至少 3/6 setting 上不优于 V1；
- V2 的贡献保留率中位数低于 `80%`；
- V2 在至少 2 个 setting 上相对 direct 的 MSE 或 MAE 退化超过 `2%`。

此时结论应是：RRR 的前两方向能够解释固定数据分布上的线性最优收益，但不足以约束
端到端训练时 NLinear 所需的全部输入信息。

## 10. 待填充表格

### 表 1：投影器审计

| Setting | train windows | λ1 share | λ2 share | λ3 share | λ2/λ3 gap | Q1 正交误差 | Q12 正交误差 | 幂等误差 | projector hash | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh2-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 2：Stage A validation

| Setting | direct MSE/MAE | phase-only MSE/MAE | V1 MSE/MAE | V2 MSE/MAE | V1 retention | V2 retention | V2−V1 | QC |
|---|---|---|---|---|---:|---:|---:|---|
| ETTh2-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-96 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 3：三 seed 正式 test 结果

| Setting | Model | MSE mean±std | MAE mean±std | ΔMSE vs direct | ΔMAE vs direct | MSE retention | MAE retention |
|---|---|---|---|---:|---:|---:|---:|
| ETTh2-96 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-192 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-96 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-192 | direct / phase-only / V1 / V2 | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 4：方向 2 增量

| Setting | λ2 share | V1→V2 ΔMSE | V1→V2 ΔMAE | MSE gap recovery | MAE gap recovery | 三 seed 方向一致 |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-192 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-192 | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 5：最终决策

| 判定项 | 预注册门槛 | 实测 | 通过 |
|---|---|---|---|
| V2 宏平均 MSE vs direct | ≤ +0.5% | TBD | TBD |
| V2 宏平均 MAE vs direct | ≤ +0.5% | TBD | TBD |
| V2 MSE retention 中位数 | ≥ 90% | TBD | TBD |
| V2 MAE retention 中位数 | ≥ 90% | TBD | TBD |
| V2 双指标优于 V1 | ≥ 4/6 settings | TBD | TBD |
| V2 最坏单格退化 | ≤ 2% | TBD | TBD |
| 最终结论 | 强支持 / 部分支持 / 不支持 | TBD | TBD |

## 11. 后续可视化

训练完成后再补充，不在计划阶段预造结果：

1. 六个 setting 的 direct、V1、V2 MSE/MAE 对比图；
2. V1/V2 的 NLinear 贡献保留率图；
3. 每个 setting 选取 V2 相对 V1 改善最大和退化最大的真实样本预测曲线；
4. 在曲线中同时显示历史、真实未来、direct、V1、V2 和 phase-only；
5. 样本选择必须由程序化误差排序产生，并记录 sample/channel 索引，不能人工挑图。

## 12. 结果解释边界

- 本实验验证的是“训练集 RRR 找到的前两个任务相关方向，作为固定数据瓶颈时是否足以支撑
  端到端模型”，不是证明人类给方向 2 起的语义名称绝对正确。
- V1/V2 与 direct 参数量相同，因此结果差异应归因于可见输入信息，而不是模型容量。
- 方向提取使用训练标签，属于训练期监督信息；必须与 validation/test 严格隔离。
- 方向 2 与方向 3 接近时，单独的方向 2 可能对矩估计扰动敏感；该风险通过 eigengap 记录，
  不通过事后增加维度规避。
- 所有性能结论先以本实验内配对 direct 为准；相对原始 PhaseFormer 金标准的比较只能作为
  背景，不得替代本实验的因果对照。
