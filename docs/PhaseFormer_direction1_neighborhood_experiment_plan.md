# PhaseFormer 方向 1 邻域宽度实验计划

> 状态：**已全量执行并回填结果（判定：部分支持）**
>
> 执行日期：2026-09-16
>
> 结果报告：`docs/PhaseFormer_direction1_neighborhood_report.md`
>
> 前置实验：`docs/PhaseFormer_top2_predictive_direction_retention_plan.md`
>
> 协议属性：**明确允许 test-set selection，并允许按数据集选择邻域宽度**
>
> 执行摘要：Stage 0 在 7 个 setting 上生成 `Qrrr2` 与 `Qcone1/2/4/8` 共 35 个冻结基，
> 28 项审计全部 PASS；Stage T 完成 42 次训练（seed 2021 × 7 setting × 6 臂，0 失败）；
> 按数据集选出 `k`：ETTh2=8、ETTm2=2、Weather=4、Electricity=2；
> Stage S 完成 56 次训练（seeds 2022/2023 × 7 setting × 4 臂，0 失败）。
> 新增训练合计 **98 次**，累计 GPU 时间 **17.36 小时**，在远程 A800 的 GPU 0–5 上以
> “一卡一 run”完成（GPU 6/7 被他人任务占用，未触碰）。

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

## 10. 实验表格（已回填）

> 下列六张表的数值与 `docs/PhaseFormer_direction1_neighborhood_report.md` 一致，
> 由 `scripts/aggregate_direction1_neighborhood.py` 从 `research_runs/direction1_neighborhood_v1/`
> 的 Stage 0 审计与 98 个 run 的 `metrics.csv` 直接生成；建议与 §11 的审计和 §12 的
> 披露限制一起阅读。

### 表 1：Stage 0 bootstrap 几何

| Setting | blocks | replicates | angle median | angle p90 | tangent effective rank | QC |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-96 | 16 | 64 | 27.04° | 35.08° | 64 | PASS |
| ETTh2-720 | 16 | 64 | 33.57° | 46.44° | 64 | PASS |
| ETTm2-96 | 16 | 64 | 15.44° | 20.49° | 64 | PASS |
| ETTm2-192 | 16 | 64 | 16.11° | 20.97° | 64 | PASS |
| Weather-96 | 16 | 64 | 9.84° | 12.28° | 64 | PASS |
| Weather-192 | 16 | 64 | 9.63° | 11.93° | 64 | PASS |
| Electricity-336 | 16 | 64 | 6.26° | 8.94° | 64 | PASS |

补充：每个 setting 的 16 个连续区块覆盖 450–2,255 个训练窗口起点（每区块、每通道；
逐 setting 不同），参与的窗口×通道对为 54,775（ETTh2-96）至 5,571,597（Electricity-336）。
bootstrap 方向 1 与全训练集方向 1 的夹角中位数为 6.26°–33.57°，p90 为 8.94°–46.44°。
`tangent effective rank` 为 64，等于 replicate 数（切向谱在 720 维中没有更小的有效维度）。

### 表 2：Stage 0 邻域解析性质

| Setting | k | tangent variation explained | visible variance | independent predictive capture | overlap with direction 2 |
|---|---:|---:|---:|---:|---:|
| ETTh2-96 | 1 | 0.0% | 1.08% | 77.9% | 0.00% |
| ETTh2-96 | 2 | 30.8% | 1.45% | 78.3% | 0.34% |
| ETTh2-96 | 4 | 50.3% | 2.16% | 78.6% | 0.37% |
| ETTh2-96 | 8 | 73.2% | 3.31% | 79.5% | 0.78% |
| ETTh2-720 | 1 | 0.0% | 4.57% | 85.7% | 0.00% |
| ETTh2-720 | 2 | 39.3% | 5.45% | 86.0% | 0.00% |
| ETTh2-720 | 4 | 62.8% | 8.81% | 87.0% | 2.35% |
| ETTh2-720 | 8 | 81.1% | 12.12% | 88.0% | 2.42% |
| ETTm2-96 | 1 | 0.0% | 0.76% | 72.5% | 0.00% |
| ETTm2-96 | 2 | 33.5% | 2.22% | 72.9% | 0.11% |
| ETTm2-96 | 4 | 58.2% | 3.64% | 73.9% | 1.42% |
| ETTm2-96 | 8 | 78.5% | 4.85% | 75.1% | 2.46% |
| ETTm2-192 | 1 | 0.0% | 1.19% | 74.6% | 0.00% |
| ETTm2-192 | 2 | 37.0% | 1.93% | 74.8% | 0.79% |
| ETTm2-192 | 4 | 65.4% | 4.80% | 75.5% | 0.82% |
| ETTm2-192 | 8 | 82.0% | 6.21% | 78.4% | 1.20% |
| Weather-96 | 1 | 0.0% | 7.18% | 86.2% | 0.00% |
| Weather-96 | 2 | 32.8% | 8.63% | 87.8% | 8.48% |
| Weather-96 | 4 | 61.2% | 9.78% | 88.2% | 13.40% |
| Weather-96 | 8 | 84.9% | 15.78% | 90.0% | 22.26% |
| Weather-192 | 1 | 0.0% | 7.86% | 78.3% | 0.00% |
| Weather-192 | 2 | 32.8% | 8.81% | 82.4% | 0.79% |
| Weather-192 | 4 | 63.2% | 10.63% | 83.8% | 2.41% |
| Weather-192 | 8 | 85.7% | 28.94% | 86.2% | 6.17% |
| Electricity-336 | 1 | 0.0% | 12.34% | 66.2% | 0.00% |
| Electricity-336 | 2 | 39.7% | 14.14% | 66.7% | 0.01% |
| Electricity-336 | 4 | 77.0% | 23.70% | 68.4% | 0.01% |
| Electricity-336 | 8 | 93.4% | 47.27% | 69.8% | 0.20% |

读法：`tangent variation explained` 是切向 PCA 前 k-1 个主成分占 bootstrap 偏离能量的比例；
`visible variance` 是该子空间承载的中心化输入方差比例；`independent predictive capture`
是该子空间在训练集上的最优线性预测所捕获的可实现 MSE 收益比例；`overlap with direction 2`
是切向成分与全局 RRR 方向 2 的重叠度。

三条值得记录的解析事实：

1. **加宽邻域确实带来可见输入能量。** 从 k=1 到 k=8，可见方差在所有 setting 上单调上升，
   Electricity-336 从 12.34% 升到 47.27%（+34.93 个百分点），Weather-192 从 7.86% 升到
   28.94%。
2. **可见能量的增量没有转化为预测收益。** 独立线性预测捕获在同样区间只从
   66.2%→69.8%（Electricity-336）、78.3%→86.2%（Weather-192）等小幅上升，多数 setting
   的增量在 1–5 个百分点内。
3. **邻域成分与全局方向 2 基本不同。** 除 Weather-96 在 k=8 时达到 22.26% 外，
   其余 setting 的切向成分与方向 2 重叠都在 0%–6% 量级，说明本轮 Cone-k 不是方向 2 的
   重参数化。

### 表 3：Stage T 全量 test sweep（seed 2021）

| Setting | Arm | test MSE | test MAE | ΔMSE vs direct | ΔMAE vs direct |
|---|---|---:|---:|---:|---:|
| ETTh2-96 | direct | 0.2721 | 0.3328 | — | — |
| ETTh2-96 | RRR-2 | 0.2715 | 0.3360 | -0.229% | +0.960% |
| ETTh2-96 | Cone-1 | 0.2722 | 0.3367 | +0.048% | +1.163% |
| ETTh2-96 | Cone-2 | 0.2765 | 0.3405 | +1.632% | +2.312% |
| ETTh2-96 | Cone-4 | 0.2746 | 0.3387 | +0.932% | +1.747% |
| ETTh2-96 | Cone-8 | 0.2744 | 0.3377 | +0.848% | +1.458% |
| ETTh2-720 | direct | 0.3909 | 0.4279 | — | — |
| ETTh2-720 | RRR-2 | 0.4020 | 0.4399 | +2.835% | +2.824% |
| ETTh2-720 | Cone-1 | 0.4053 | 0.4427 | +3.683% | +3.462% |
| ETTh2-720 | Cone-2 | 0.4034 | 0.4419 | +3.208% | +3.284% |
| ETTh2-720 | Cone-4 | 0.4017 | 0.4376 | +2.753% | +2.270% |
| ETTh2-720 | Cone-8 | 0.3980 | 0.4343 | +1.813% | +1.495% |
| ETTm2-96 | direct | 0.1585 | 0.2480 | — | — |
| ETTm2-96 | RRR-2 | 0.1652 | 0.2570 | +4.263% | +3.607% |
| ETTm2-96 | Cone-1 | 0.1653 | 0.2576 | +4.338% | +3.833% |
| ETTm2-96 | Cone-2 | 0.1640 | 0.2564 | +3.517% | +3.379% |
| ETTm2-96 | Cone-4 | 0.1636 | 0.2556 | +3.248% | +3.052% |
| ETTm2-96 | Cone-8 | 0.1640 | 0.2556 | +3.479% | +3.044% |
| ETTm2-192 | direct | 0.2157 | 0.2881 | — | — |
| ETTm2-192 | RRR-2 | 0.2187 | 0.2948 | +1.403% | +2.345% |
| ETTm2-192 | Cone-1 | 0.2190 | 0.2962 | +1.534% | +2.813% |
| ETTm2-192 | Cone-2 | 0.2189 | 0.2962 | +1.501% | +2.828% |
| ETTm2-192 | Cone-4 | 0.2267 | 0.2991 | +5.097% | +3.847% |
| ETTm2-192 | Cone-8 | 0.2242 | 0.2989 | +3.969% | +3.757% |
| Weather-96 | direct | 0.1467 | 0.1940 | — | — |
| Weather-96 | RRR-2 | 0.1521 | 0.2000 | +3.645% | +3.112% |
| Weather-96 | Cone-1 | 0.1486 | 0.1966 | +1.260% | +1.314% |
| Weather-96 | Cone-2 | 0.1475 | 0.1947 | +0.515% | +0.340% |
| Weather-96 | Cone-4 | 0.1470 | 0.1938 | +0.221% | -0.130% |
| Weather-96 | Cone-8 | 0.1466 | 0.1940 | -0.087% | +0.018% |
| Weather-192 | direct | 0.1918 | 0.2363 | — | — |
| Weather-192 | RRR-2 | 0.1915 | 0.2376 | -0.162% | +0.581% |
| Weather-192 | Cone-1 | 0.1919 | 0.2375 | +0.073% | +0.498% |
| Weather-192 | Cone-2 | 0.1915 | 0.2378 | -0.158% | +0.624% |
| Weather-192 | Cone-4 | 0.1885 | 0.2342 | -1.696% | -0.869% |
| Weather-192 | Cone-8 | 0.1896 | 0.2356 | -1.139% | -0.308% |
| Electricity-336 | direct | 0.1617 | 0.2547 | — | — |
| Electricity-336 | RRR-2 | 0.1670 | 0.2606 | +3.258% | +2.309% |
| Electricity-336 | Cone-1 | 0.1657 | 0.2586 | +2.447% | +1.518% |
| Electricity-336 | Cone-2 | 0.1654 | 0.2583 | +2.272% | +1.415% |
| Electricity-336 | Cone-4 | 0.1666 | 0.2594 | +2.984% | +1.857% |
| Electricity-336 | Cone-8 | 0.1665 | 0.2592 | +2.975% | +1.776% |

结论要点（seed 2021）：35 个投影 cell 中，**只有 2 个 cell 双指标优于 direct**，
且都在 Weather-192（Cone-4：ΔMSE -1.696% / ΔMAE -0.869%；Cone-8：ΔMSE -1.139% /
ΔMAE -0.308%）；MSE 单项优于 direct 的有 6/35，MAE 单项优于 direct 的有 3/35。
其余全部投影臂在 ETT 与 Electricity 上劣于 direct；ETTh2-720 与 ETTm2-96 上可以观察到
“k 增大 → 退化收窄”的趋势（ETTh2-720：+3.683% → +1.813%），但收窄后仍未回到 direct。

### 表 4：按数据集宽度选择

| Dataset | k=1 ΔMSE/ΔMAE | k=2 ΔMSE/ΔMAE | k=4 ΔMSE/ΔMAE | k=8 ΔMSE/ΔMAE | selected k |
|---|---:|---:|---:|---:|---:|
| ETTh2 | +1.866% / +2.312% | +2.420% / +2.798% | +1.843% / +2.008% | **+1.330% / +1.477%** | 8 |
| ETTm2 | +2.936% / +3.323% | **+2.509% / +3.104%** | +4.172% / +3.449% | +3.724% / +3.401% | 2 |
| Weather | +0.667% / +0.906% | +0.178% / +0.482% | **-0.738% / -0.500%** | -0.613% / -0.145% | 4 |
| Electricity | +2.447% / +1.518% | **+2.272% / +1.415%** | +2.984% / +1.857% | +2.975% / +1.776% | 2 |

选择规则：先取数据集宏平均 test ΔMSE 最优，再把与其差距不超过 0.10 个百分点的宽度纳入
候选，在候选中取宏平均 test ΔMAE 最小者，仍相同取更小的 k。本轮四个数据集的候选池都只
含单个宽度（`selection_pool` 分别为 [8]、[2]、[4]、[2]），因此 MAE 平手规则未被触发。

该选择直接读取 seed 2021 的 test 指标，属于 **test-set selection**。

需要注意的读法：选择规则的“最优”是**在四个投影宽度之间**取最优，而不是“相对 direct 有改善”。
ETTh2、ETTm2、Electricity 三个数据集所选的 k，其宏平均 ΔMSE/ΔMAE 仍为正（即仍劣于
direct）；只有 Weather 选出的 k=4 在宏平均上优于 direct。因此“4/4 选择 k>1”这一事实
**不能**读成“4/4 数据集从更宽邻域获益”。

### 表 5：三 seed 稳定性结果

三 seed = 2021（Stage T）+ 2022/2023（Stage S，仅 direct / RRR-2 / Cone-1 / 该数据集所选 Cone-k，
所以未选宽度只有 seed 2021 一列，见上表）。std 为样本标准差（n-1）。

| Setting | Arm | MSE mean±std | MAE mean±std | ΔMSE vs direct | ΔMAE vs direct | MSE wins vs direct | MSE wins vs RRR-2 |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh2-96 | direct | 0.2732 ± 0.0030 | 0.3334 ± 0.0014 | — | — | — | 2/3 |
| ETTh2-96 | RRR-2 | 0.2740 ± 0.0031 | 0.3388 ± 0.0031 | +0.293% | +1.635% | 1/3 | — |
| ETTh2-96 | Cone-1 | 0.2733 ± 0.0017 | 0.3385 ± 0.0022 | +0.020% | +1.533% | 1/3 | 2/3 |
| ETTh2-96 | Cone-8（selected） | 0.2749 ± 0.0005 | 0.3377 ± 0.0002 | +0.603% | +1.309% | 1/3 | 1/3 |
| ETTh2-720 | direct | 0.3925 ± 0.0018 | 0.4285 ± 0.0019 | — | — | — | 3/3 |
| ETTh2-720 | RRR-2 | 0.4079 ± 0.0059 | 0.4428 ± 0.0039 | +3.942% | +3.345% | 0/3 | — |
| ETTh2-720 | Cone-1 | 0.4060 ± 0.0027 | 0.4431 ± 0.0014 | +3.456% | +3.394% | 0/3 | 2/3 |
| ETTh2-720 | Cone-8（selected） | 0.4001 ± 0.0033 | 0.4358 ± 0.0017 | +1.943% | +1.705% | 0/3 | 3/3 |
| ETTm2-96 | direct | 0.1591 ± 0.0005 | 0.2485 ± 0.0005 | — | — | — | 3/3 |
| ETTm2-96 | RRR-2 | 0.1642 ± 0.0009 | 0.2557 ± 0.0013 | +3.247% | +2.889% | 0/3 | — |
| ETTm2-96 | Cone-1 | 0.1641 ± 0.0014 | 0.2559 ± 0.0014 | +3.193% | +2.977% | 0/3 | 1/3 |
| ETTm2-96 | Cone-2（selected） | 0.1637 ± 0.0006 | 0.2551 ± 0.0011 | +2.884% | +2.675% | 0/3 | 2/3 |
| ETTm2-192 | direct | 0.2148 ± 0.0008 | 0.2878 ± 0.0002 | — | — | — | 3/3 |
| ETTm2-192 | RRR-2 | 0.2206 ± 0.0023 | 0.2959 ± 0.0010 | +2.693% | +2.794% | 0/3 | — |
| ETTm2-192 | Cone-1 | 0.2204 ± 0.0014 | 0.2969 ± 0.0016 | +2.631% | +3.142% | 0/3 | 1/3 |
| ETTm2-192 | Cone-2（selected） | 0.2205 ± 0.0014 | 0.2965 ± 0.0012 | +2.649% | +3.025% | 0/3 | 1/3 |
| Weather-96 | direct | 0.1465 ± 0.0004 | 0.1939 ± 0.0001 | — | — | — | 3/3 |
| Weather-96 | RRR-2 | 0.1499 ± 0.0020 | 0.1968 ± 0.0028 | +2.303% | +1.463% | 0/3 | — |
| Weather-96 | Cone-1 | 0.1479 ± 0.0008 | 0.1955 ± 0.0019 | +0.942% | +0.825% | 0/3 | 3/3 |
| Weather-96 | Cone-4（selected） | 0.1471 ± 0.0014 | 0.1943 ± 0.0021 | +0.393% | +0.196% | 1/3 | 3/3 |
| Weather-192 | direct | 0.1921 ± 0.0013 | 0.2370 ± 0.0011 | — | — | — | 0/3 |
| Weather-192 | RRR-2 | 0.1906 ± 0.0008 | 0.2366 ± 0.0009 | -0.822% | -0.186% | 3/3 | — |
| Weather-192 | Cone-1 | 0.1912 ± 0.0007 | 0.2372 ± 0.0008 | -0.474% | +0.066% | 2/3 | 0/3 |
| Weather-192 | Cone-4（selected） | 0.1891 ± 0.0006 | 0.2347 ± 0.0004 | -1.594% | -1.000% | 3/3 | 3/3 |
| Electricity-336 | direct | 0.1625 ± 0.0007 | 0.2558 ± 0.0011 | — | — | — | 3/3 |
| Electricity-336 | RRR-2 | 0.1674 ± 0.0003 | 0.2605 ± 0.0001 | +3.029% | +1.860% | 0/3 | — |
| Electricity-336 | Cone-1 | 0.1666 ± 0.0009 | 0.2595 ± 0.0010 | +2.558% | +1.445% | 0/3 | 3/3 |
| Electricity-336 | Cone-2（selected） | 0.1667 ± 0.0012 | 0.2596 ± 0.0011 | +2.643% | +1.494% | 0/3 | 2/3 |

逐 seed 的 Cone-k vs Cone-1 对照（仅列被选宽度与被选数据集）：

| Setting | selected k | seed 2021 | seed 2022 | seed 2023 | MSE 胜出 seed 数 |
|---|---:|---|---|---|---:|
| ETTh2-96 | 8 | 输 (0.2744 vs 0.2722) | 输 (0.2753 vs 0.2724) | 赢 (0.2749 vs 0.2752) | 1/3 |
| ETTh2-720 | 8 | 赢 (0.3980 vs 0.4053) | 输 (0.4039 vs 0.4038) | 赢 (0.3984 vs 0.4090) | 2/3 |
| ETTm2-96 | 2 | 赢 (0.1640 vs 0.1653) | 赢 (0.1639 vs 0.1645) | 输 (0.1630 vs 0.1626) | 2/3 |
| ETTm2-192 | 2 | 赢 (0.2189 vs 0.2190) | 赢 (0.2208 vs 0.2219) | 输 (0.2217 vs 0.2204) | 2/3 |
| Weather-96 | 4 | 赢 (0.1470 vs 0.1486) | 赢 (0.1457 vs 0.1470) | 输 (0.1485 vs 0.1481) | 2/3 |
| Weather-192 | 4 | 赢 (0.1885 vs 0.1919) | 赢 (0.1896 vs 0.1912) | 赢 (0.1891 vs 0.1906) | 3/3 |
| Electricity-336 | 2 | 赢 (0.1654 vs 0.1657) | 输 (0.1676 vs 0.1675) | 输 (0.1673 vs 0.1666) | 1/3 |

唯一在所有三个 seed 上都优于 Cone-1 的是 **Weather-192 的 Cone-4**；只在一个 seed 上
胜出的是 ETTh2-96（Cone-8）与 Electricity-336（Cone-2）。

### 表 6：最终解释

| 判定项 | 实测 |
|---|---|
| 选择 k>1 的数据集数 | 4/4（ETTh2=8, ETTm2=2, Weather=4, Electricity=2） |
| 选择后 Cone-k 数据集宏平均 MSE 与 MAE 均优于 Cone-1 的数据集 | 3/4（ETTh2、ETTm2、Weather） |
| Cone-k 三 seed MSE 优于 Cone-1 的 setting | 4/7（ETTh2-720、ETTm2-96、Weather-96、Weather-192） |
| Cone-k 三 seed MAE 优于 Cone-1 的 setting | 6/7（除 Electricity-336 外全部） |
| Cone-k 优于或持平 RRR-2 的 setting（MSE） | 6/7（仅 ETTh2-96 除外） |
| Cone-k 优于或持平 RRR-2 的 setting（MAE，附加口径） | 6/7（仅 ETTm2-192 除外） |
| 宽度收益是否跨 seed 稳定 | 三 seed 均值口径下 Cone-k 优于 Cone-1 的 4 个 setting 中，4 个都在至少 2/3 个 seed 上同向成立，**没有**任何均值收益由单个异常 seed 造成 |
| 最终结论 | **部分支持**（5 条判据中 4 条成立；第 2 条不成立） |

逐条判据（对应 §7 的 1–5）：

| # | 判据 | 结果 |
|---|---|---|
| 1 | 至少 3/4 个数据集选择 k>1 | ✅ 4/4 |
| 2 | 选择后的 Cone-k 在数据集宏平均 MSE 与 MAE 上优于 Cone-1 | ❌ 3/4，Electricity-336 上宏平均反而变差（MSE +0.083%、MAE +0.049%） |
| 3 | 至少 4/7 setting 上 Cone-k 三 seed MSE 优于 Cone-1 | ✅ 4/7 |
| 4 | Cone-k 在至少 4/7 setting 上优于或持平 RRR-2 | ✅ 6/7 |
| 5 | 宽度收益不是只来自单个异常 seed | ✅ 4 个均值收益 setting 全部满足 ≥2/3 seed 同向 |

§7 对“部分支持”的定义是“只有部分数据集稳定选择 k>1，或 MSE 改善但 MAE、跨 seed
一致性不足”。本轮的实际形态是：**全部数据集都选了 k>1，但只有在 Weather 上这个选择同时
对应相对 direct 的真实改善；在 ETT 与 Electricity 上，更宽的邻域只是“退得更少”，并没有
补回被投影丢掉的信息。**因此结论落在“部分支持”，且只能表述为
**方向 1 邻域具有数据集条件性价值**，不能表述为统一的宽度收益。

## 11. 执行状态与审计

阶段状态（2026-09-16 全部完成）：

| 阶段 | 计划 | 实际 | 结果 |
|---|---|---|---|
| Stage 0 投影器 + 六项审计 | 7 setting × 4 宽度 + Qrrr2 | 35 个冻结基 + 28 项审计记录 | 28/28 PASS |
| Stage T test sweep | 7 × 6 × seed 2021 = 42 runs | 42 runs | 42/42 完成，0 失败，0 重试 |
| Stage T 宽度选择 | 每数据集一个共享宽度 | `test_selection.json` | 4/4 数据集选中 k>1 |
| Stage S 稳定性复核 | 42–56 runs | 56 runs（4 个数据集都非 k=1，故每格 4 臂） | 56/56 完成，0 失败 |
| 合计新增训练 | — | **98 runs** | 累计 GPU 时间 **17.36 小时** |

运行环境与产物：

- 远程 A800-SXM4-80GB ×6（GPU 0–5，一卡一 run；GPU 6/7 全程被他人任务占用，未触碰）。
- `/home/yyk/yyk03/miniconda3/envs/time`，Python 3.10、torch 2.6.0+cu124、
  pytorch-lightning 2.6.5（与仓库记录的 torch 2.7.1 / Lightning 2.1.0 不同，正式对比须注明）。
- 产物根目录 `research_runs/direction1_neighborhood_v1/`（gitignored）：
  `projectors/`（35 个 `.npy` + `stage0_audit.{json,csv,md}` + `projectors.json`）、
  `runs/`（98 个 run 目录）、`sweep_manifest.json`、`sweep_summary.json`、
  `confirm_manifest.json`、`confirm_summary.json`、`test_selection.{json,md}`、
  `aggregation/`（`table1`–`table7`、`results.csv`、`aggregate.json`）。

审计结果：

1. Stage 0 六项检查（train-only、正交、幂等、包含全训练集方向 1、嵌套、数值有限）
   在 28 个 (setting, k) 记录上全部 PASS；`Qcone1 ⊂ Qcone2 ⊂ Qcone4 ⊂ Qcone8` 的
   嵌套误差与方向 1 恢复误差均为浮点级（≤2.5e-16）。
2. 98 个 run 的 `status.json` 全为 `completed`，无 `failed`；每个 run 的 test 只在其
   checkpoint 冻结后读取一次（`--evaluate-test`，随训练进程写回 `metrics.csv`）。
3. **参数量逐格相等**：在 21 个 (setting, seed) 组合内，6（或 4）个实验臂的
   `parameter_count` / `trainable_parameter_count` 完全一致，排除容量解释。
4. `results.csv` 的 98 行与聚合表逐格可回溯到 `runs/<run_id>/metrics.csv`。

诊断记录（非阻断项）：上一轮 `top2_direction_retention_v1` 的 `Q1` 与本轮
`Qcone1` 的子空间主夹角在 6 个可比较 setting 上最大 **1.21e-06°**（ETTh2-96、
ETTm2-96、Weather-96 上为 0.0°，其余三格为 8.5e-07°–1.21e-06° 的浮点级差异），
即本轮重新估计的方向 1 与上一轮保存的 Q1 落在同一个一维子空间内，本轮 Cone-1
与前置实验的 V1 在几何上等价（ETTm2-192 的基向量整体反号，但投影矩阵 `Q1 Q1ᵀ`
与符号无关）。Electricity-336 上一轮没有 Q1（不在其 setting 内），本轮新增。
按 §6 的规定，该差异只作诊断记录，不构成阻断条件。

## 12. 披露合规与读法限制

本实验按 §8 的要求执行并披露：

1. **全部宽度都保留。** 表 3 给出全部 6 个实验臂（含未选中的 k），表 4 给出每个数据集
   全部 4 个宽度的宏平均 ΔMSE/ΔMAE，`aggregation/results.csv` 保留 98 个 run 的逐格原始值。
   没有只报告各数据集的最优宽度。
2. **Stage S 不是独立确认集。** 结构与宽度都依据 seed 2021 的 test 指标选定，
   seeds 2022/2023 只能解释为**选择后稳定性复核**。
3. **不声明无偏泛化提升。** 本实验的任何数字都不得表述为盲测或无偏泛化估计。
4. **唯一相对 direct 的改善是数据集条件性的。** 三 seed 口径下，只有 Weather 所选
   Cone-4 同时在 MSE（-1.594%）与 MAE（-1.000%）上优于 direct，并且这一优势在 3/3 个
   seed 上成立；ETTh2、ETTm2、Electricity 上所选宽度仍劣于 direct。
5. **宽度收益的合理表述**是：方向 1 周围确实存在额外数据能量（表 2 的 visible variance
   随 k 单调上升，Electricity-336 从 12.34% 升到 47.27%），但在三 seed 均值口径下，
   7 个 setting 中只有 Weather-192 的被选宽度相对 direct 同时改善 MSE 与 MAE
   （-1.594% / -1.000%），另外 6 个 setting 即使换到各自最优的 k 仍劣于 direct。
   因此本实验只支持“**方向 1 邻域具有数据集条件性价值**”，
   不支持“加宽方向 1 邻域是普遍改进”。
