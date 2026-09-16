# PhaseFormer 方向 1 邻域宽度实验计划

> 状态：**代码与待填表格已完成，尚未生成投影器，尚未训练**
>
> 日期：2026-09-16
>
> 前置实验：`docs/PhaseFormer_top2_predictive_direction_retention_plan.md`
>
> 协议属性：**明确允许 test-set selection，并允许按数据集选择邻域宽度**

## 1. 实验目的

前置实验把 NLinear 输入严格限制在方向 1 或方向 1+2：

```text
V1: z_keep = Q1 Q1^T z
V2: z_keep = Q12 Q12^T z
```

结果表明严格方向 1 没有稳定恢复完整 NLinear，而直接加入全局方向 2 也没有稳定增量。
本轮验证新的解释：

> 方向 1 可能是有效机制的中心轴，但严格秩 1 投影过窄。训练分布中的有效滤波器可能在
> 方向 1 周围形成一个低维邻域；保留这个邻域可能比加入全局方向 2 更有效。

本实验回答：

1. 方向 1 在训练数据重采样下是否形成稳定、低维的局部变化空间？
2. 邻域宽度从 1 增加到 2、4、8 时，输入方差、独立预测收益和端到端性能如何变化？
3. 方向 1 邻域是否优于同为二维的全局 RRR 方向 1+2？
4. 不同数据集是否需要不同邻域宽度？

## 2. “扇形邻域”的实现

### 2.1 为什么不用固定夹角

在线性模型中，多条靠近方向 1 的射线一旦同时输入线性层，就等价于保留它们张成的完整
子空间。因此“5°/10°扇形”不能持续约束后续线性层。

本实验把扇形宽度定义为：

> **保留方向 1，再增加多少个由训练集重复估计得到的局部切向方向。**

实际控制参数是邻域维度 `k∈{1,2,4,8}`。

### 2.2 连续区块 bootstrap

每个 setting 只读取训练 split：

1. 按既有 RRR 口径计算全训练集方向 `q1`；
2. 将训练窗口起点划分成 16 个连续区块；
3. 固定随机种子 `20260916`，进行 64 次区块有放回重采样；
4. 每次重新计算方向 1，并与全训练集 `q1` 对齐符号；
5. 去除平行于 `q1` 的分量：

```text
r_m = (I - q1 q1^T) q1_bootstrap_m
```

6. 对 `{r_m}` 做 PCA，得到切向方向 `v1,v2,...`。

连续区块重采样保留了重叠窗口和时序区段之间的相关结构。

### 2.3 邻域投影

```text
Qcone1 = [q1]
Qcone2 = [q1, v1]
Qcone4 = [q1, v1, v2, v3]
Qcone8 = [q1, v1, ..., v7]

z = x - x_last
z_keep = Qconek Qconek^T z
y_nlinear = Linear(z_keep) + x_last
```

所有基均由训练集生成并冻结。`k` 增加只改变 NLinear 可见输入子空间，不改变 NLinear
线性层尺寸、模型参数量、phase 主干或融合 gate。

本实验**不要求**新生成的 `Qcone1` 与前置实验保存的 Q1 哈希或投影矩阵严格相等。
所有关键对照均在本实验中按相同代码路径重新训练。

## 3. 七个 setting

| 数据集 | Horizon |
|---|---|
| ETTh2 | 96, 720 |
| ETTm2 | 96, 192 |
| Weather | 96, 192 |
| Electricity | 336 |

Electricity-336 已有匹配训练与 RRR 分析，本轮将其纳入后共 7 个 setting。

## 4. 实验臂

| 实验臂 | NLinear 可见信息 | 作用 |
|---|---|---|
| direct | 完整 720 步中心化历史 | 完整输入对照 |
| RRR-2 | 全局 RRR 方向 1+2 | 与上一轮 V2 同类的方向对照 |
| Cone-1 | bootstrap 邻域中心轴 q1 | 严格秩 1 对照 |
| Cone-2 | q1 + 1 个局部切向方向 | 最窄邻域 |
| Cone-4 | q1 + 3 个局部切向方向 | 中等邻域 |
| Cone-8 | q1 + 7 个局部切向方向 | 较宽邻域 |

全部实验臂使用相同 NLinear 线性层和参数量。

## 5. 固定训练设置

- lookback：720
- period：24
- loss：Huber
- max epochs：30
- checkpoint：最低 validation loss
- head：shared NLinear
- 每个 setting 的 learning rate 和 residual gate 初始化沿用条件秩实验：

| Setting | gate init | learning rate |
|---|---:|---:|
| ETTh2-96 | 0.5 | 1e-3 |
| ETTh2-720 | 0.5 | 1e-3 |
| ETTm2-96 | 0.5 | 3e-4 |
| ETTm2-192 | 0.2 | 1e-3 |
| Weather-96 | 0.2 | 3e-4 |
| Weather-192 | 0.5 | 1e-3 |
| Electricity-336 | 0.5 | 1e-3 |

## 6. 执行阶段

### Stage 0：投影器与数据几何审计

在 7 个 setting 上生成：

```text
Qrrr2, Qcone1, Qcone2, Qcone4, Qcone8
```

必须检查：

1. 只读取训练 split；
2. 所有基正交，投影矩阵幂等；
3. `Qcone1 ⊂ Qcone2 ⊂ Qcone4 ⊂ Qcone8`；
4. 所有 Cone-k 都能恢复本轮全训练集方向 1；
5. bootstrap 角度和切向谱为有限值；
6. 记录可见输入方差、独立线性预测收益以及 Cone-k 与全局方向 2 的重叠。

旧 V1 与本轮 Cone-1 的差异只作为诊断记录，不作为阻断条件。

### Stage T：七 setting 单 seed test sweep

直接运行：

```text
7 settings × 6 arms × seed 2021 = 42 runs
```

每个 run 仍按 validation 选择 checkpoint，但 checkpoint 训练完成后立即读取一次 test。
宽度选择使用 test 指标，因此本阶段和后续结果均属于 **test-set selection**。

### Stage T 宽度选择

每个数据集选择一个共享宽度，不按 horizon 单独选择：

- ETTh2：H96/H720 共享一个 `k`
- ETTm2：H96/H192 共享一个 `k`
- Weather：H96/H192 共享一个 `k`
- Electricity：H336 单独一个 `k`

选择规则：

1. 对每个 `k` 计算该数据集所有 horizon 相对 direct 的宏平均 test ΔMSE；
2. 保留距离最优 MSE 不超过 0.10 个百分点的候选；
3. 在这些候选中选择宏平均 test ΔMAE 最小者；
4. 若仍相同，选择更小的 `k`。

选择结果写入 `test_selection.json`。该文件必须明确包含
`"test_set_selection": true`。

### Stage S：选择后稳定性复核

按 Stage T 得到的每数据集宽度，补充 seeds 2022/2023：

```text
7 settings × (3 or 4 arms) × 2 new seeds = 42–56 runs
```

实验臂为：

- direct
- RRR-2
- Cone-1
- 该数据集选择的 Cone-k

当选择结果为 `k=1` 时，Cone-1 与 selected Cone-k 自动去重，因此该 setting 每个新 seed
运行 3 个实验臂；否则运行 4 个。seed 2021 复用 Stage T。最终形成三 seed 结果，但由于
结构和宽度已经依据 seed 2021 test 选择，这些结果只能解释为**选择后稳定性复核**，
不能称为盲测或无偏泛化估计。

## 7. 结果判读

### 支持方向 1 邻域假设

满足以下现象时，可以认为假设得到支持：

1. 至少 3/4 个数据集选择 `k>1`；
2. 选择后的 Cone-k 在数据集宏平均 MSE 与 MAE上优于 Cone-1；
3. 在至少 4/7 setting 上，Cone-k 三 seed MSE 优于 Cone-1；
4. Cone-k 在至少 4/7 setting 上优于或持平 RRR-2；
5. 宽度增加后的收益不是只来自单个异常 seed。

### 部分支持

只有部分数据集稳定选择 `k>1`，或 MSE 改善但 MAE、跨 seed 一致性不足。此时只能说
“方向 1 邻域具有数据集条件性价值”。

### 不支持

多数数据集选择 `k=1`，或更宽邻域增加大量可见方差但没有改善 test MSE/MAE。此时结论为：

> 方向 1 周围存在额外数据能量，但当前证据不支持这些邻域成分具有额外端到端预测价值。

## 8. Test-set Selection 披露

本实验明确允许：

- 使用 seed 2021 test 指标选择邻域宽度；
- 不同数据集使用不同宽度；
- 根据 test 结果决定是否进入多 seed 稳定性复核。

因此：

- 必须保留全部 `k=1/2/4/8` 的结果；
- 不得只报告每个数据集的最优宽度；
- 不得将 Stage S 描述为独立确认集；
- 不得用本实验结果声明无偏泛化提升；
- 对外汇报必须同时给出完整选择轨迹和未选择宽度的结果。

## 9. 实验脚本

### 9.1 生成投影器

脚本：`scripts/compute_direction1_neighborhood_projectors.py`

```bash
python scripts/compute_direction1_neighborhood_projectors.py \
  --output-dir research_runs/direction1_neighborhood_v1/projectors \
  --bootstrap-blocks 16 \
  --bootstrap-replicates 64 \
  --bootstrap-seed 20260916 \
  --widths 1,2,4,8
```

该脚本同时生成 `Qrrr2` 与四个 Cone-k，只读取 train split，不训练模型。

### 9.2 运行七 setting test sweep

脚本：`scripts/run_direction1_neighborhood_matrix.py`

```bash
python scripts/run_direction1_neighborhood_matrix.py \
  --stage sweep \
  --gpus 0,1,2,3,4,5 \
  --output-root research_runs/direction1_neighborhood_v1
```

脚本生成 42 个任务，并在 manifest 中逐格标记 `test_set_selection=true`。

### 9.3 按数据集选择宽度

脚本：`scripts/select_direction1_neighborhood_width.py`

```bash
python scripts/select_direction1_neighborhood_width.py \
  --root research_runs/direction1_neighborhood_v1
```

输出：

```text
test_selection.json
test_selection.md
```

### 9.4 多 seed 稳定性复核

```bash
python scripts/run_direction1_neighborhood_matrix.py \
  --stage confirm \
  --selection-file research_runs/direction1_neighborhood_v1/test_selection.json \
  --gpus 0,1,2,3,4,5 \
  --output-root research_runs/direction1_neighborhood_v1
```

## 10. 待填充实验表格

### 表 1：Stage 0 bootstrap 几何

| Setting | blocks | replicates | angle median | angle p90 | tangent effective rank | QC |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-192 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-96 | TBD | TBD | TBD | TBD | TBD | TBD |
| Weather-192 | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | TBD | TBD | TBD | TBD | TBD | TBD |

### 表 2：Stage 0 邻域解析性质

| Setting | k | tangent variation explained | visible variance | independent predictive capture | overlap with direction 2 |
|---|---:|---:|---:|---:|---:|
| ETTh2-96 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTh2-720 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTm2-96 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTm2-192 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| Weather-96 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| Weather-192 | 1/2/4/8 | TBD | TBD | TBD | TBD |
| Electricity-336 | 1/2/4/8 | TBD | TBD | TBD | TBD |

### 表 3：Stage T 全量 test sweep

| Setting | Arm | test MSE | test MAE | ΔMSE vs direct | ΔMAE vs direct |
|---|---|---:|---:|---:|---:|
| ETTh2-96 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTh2-720 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTm2-96 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| ETTm2-192 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| Weather-96 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| Weather-192 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |
| Electricity-336 | direct / RRR-2 / Cone-1/2/4/8 | TBD | TBD | TBD | TBD |

### 表 4：按数据集宽度选择

| Dataset | k=1 ΔMSE/ΔMAE | k=2 ΔMSE/ΔMAE | k=4 ΔMSE/ΔMAE | k=8 ΔMSE/ΔMAE | selected k |
|---|---:|---:|---:|---:|---:|
| ETTh2 | TBD | TBD | TBD | TBD | TBD |
| ETTm2 | TBD | TBD | TBD | TBD | TBD |
| Weather | TBD | TBD | TBD | TBD | TBD |
| Electricity | TBD | TBD | TBD | TBD | TBD |

### 表 5：三 seed 稳定性结果

| Setting | Arm | MSE mean±std | MAE mean±std | ΔMSE vs direct | ΔMAE vs direct | MSE wins vs direct | MSE wins vs RRR-2 |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh2-96 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| ETTh2-720 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| ETTm2-96 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| ETTm2-192 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| Weather-96 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| Weather-192 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |
| Electricity-336 | direct / RRR-2 / Cone-1 / selected Cone-k | TBD | TBD | TBD | TBD | TBD/3 | TBD/3 |

### 表 6：最终解释

| 判定项 | 实测 |
|---|---|
| 选择 k>1 的数据集数 | TBD/4 |
| Cone-k 三 seed MSE 优于 Cone-1 的 setting | TBD/7 |
| Cone-k 三 seed MAE 优于 Cone-1 的 setting | TBD/7 |
| Cone-k 优于或持平 RRR-2 的 setting | TBD/7 |
| 宽度收益是否跨 seed 稳定 | TBD |
| 最终结论 | 支持 / 部分支持 / 不支持 |

## 11. 当前状态

本次只完成代码和实验计划：

- 未生成任何真实投影器；
- 未读取本地或远程数据集；
- 未启动训练；
- 未读取 test；
- 所有结果表均保持待填写。
