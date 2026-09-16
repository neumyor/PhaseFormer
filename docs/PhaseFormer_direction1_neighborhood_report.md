# PhaseFormer 方向 1 邻域宽度实验结果报告

> 状态：**已全量执行并回填结果**（2026-09-16）
>
> 计划：`docs/PhaseFormer_direction1_neighborhood_experiment_plan.md`（本报告与其 §10 表格一致）
>
> 前置实验：`docs/PhaseFormer_top2_predictive_direction_retention_plan.md`
>
> 协议属性：**明确披露的 test-set selection**；不构成盲测或无偏泛化估计

---

## 0. 结论（先说结果）

**判定：部分支持（5 条预注册判据中 4 条成立，第 2 条不成立）。**

- **方向 1 周围确实有额外数据能量。** 从 k=1 加宽到 k=8，被投影子空间承载的中心化输入方差
  在 7 个 setting 上单调上升：Electricity-336 从 12.34% 升到 47.27%，Weather-192 从
  7.86% 升到 28.94%。这些新增成分与全局 RRR 方向 2 基本不重合（多数 setting 重叠 <6%），
  所以它们不是方向 2 的重参数化。
- **但这部分能量没有转化为端到端性能。** 三 seed 均值口径下，7 个 setting 中只有
  **Weather-192** 的被选宽度（Cone-4）相对 direct 同时改善 MSE（-1.594%）与 MAE
  （-1.000%），且这一改善在 **3/3 个 seed** 上成立。其余 6 个 setting 即使各自换到最优的
  k 仍劣于 direct，退化幅度从 +0.196%（Weather-96 MAE）到 +3.025%（ETTm2-192 MAE）。
- **加宽邻域的作用是“退得更少”，不是“补回来”。** 在 ETT 上可以看到清晰的单调趋势
  （ETTh2-720：ΔMSE 从 k=1 的 +3.456% 收窄到 k=8 的 +1.943%；ETTm2-96：从 +3.193%
  收窄到 +2.884%），但收窄后仍未回到 direct。这与前置实验“严格方向 1 不能恢复完整
  NLinear”的结论方向一致，只是把“过窄”这一解释进一步压低了。
- **按数据集选择宽度不改变这个结论。** 4/4 个数据集都选了 k>1，但那是“在四个投影宽度里
  选最优”，不是“相对 direct 有改善”。ETTh2、ETTm2、Electricity 所选宽度的宏平均
  ΔMSE/ΔMAE 仍为正。
- **可以对外说的**：方向 1 邻域具有**数据集条件性价值**，在 Weather 上出现了 MSE、MAE
  与跨 seed 一致性三者同时成立的正向信号。**不可以说的**：加宽方向 1 邻域是普遍改进。

---

## 1. 实验执行与协议合规

### 1.1 执行规模

| 阶段 | 内容 | 计划 | 实际 | 失败 |
|---|---|---|---|---|
| Stage 0 | 生成 `Qrrr2` 与 `Qcone1/2/4/8` 并做六项审计 | 7 setting | 35 个冻结基，28 条审计记录，28/28 PASS | 0 |
| Stage T | seed 2021 × 7 setting × 6 臂 | 42 runs | 42 runs | 0 |
| 宽度选择 | 每数据集一个共享宽度 | 4 个数据集 | 4/4 选中 k>1 | 0 |
| Stage S | seeds 2022/2023 × 7 setting × 4 臂 | 42–56 runs | 56 runs | 0 |
| **合计** | | | **98 runs，累计 GPU 时间 17.36 小时** | **0** |

98 个 run 的 `status.json` 全部为 `completed`，无 `failed`，无重试。

### 1.2 运行环境

- 远程 A800-SXM4-80GB ×6：GPU 0–5，一卡一 run，六路并发；GPU 6/7 全程被他人任务占用，
  未触碰。
- `/home/yyk/yyk03/miniconda3/envs/time`：Python 3.10、torch 2.6.0+cu124、
  pytorch-lightning 2.6.5。
- 与仓库记录的 torch 2.7.1 + Lightning 2.1.0（RTX 4090）不同，**正式跨环境对比须注明**。
  本报告的所有结论都是本实验内部的配对比较（同一代码路径、同一环境、同一协议），
  不使用 `docs/PhaseFormer_gold_standard.md` 或其它协议的数字。

### 1.3 固定训练设置（与计划 §5 一致）

lookback 720、period 24、Huber、max epochs 30、最低 validation loss checkpoint、
shared NLinear head，learning rate 与 residual gate 初始化逐 setting 沿用条件秩实验：

| Setting | gate init | learning rate |
|---|---:|---:|
| ETTh2-96 | 0.5 | 1e-3 |
| ETTh2-720 | 0.5 | 1e-3 |
| ETTm2-96 | 0.5 | 3e-4 |
| ETTm2-192 | 0.2 | 1e-3 |
| Weather-96 | 0.2 | 3e-4 |
| Weather-192 | 0.5 | 1e-3 |
| Electricity-336 | 0.5 | 1e-3 |

所有实验臂共用同一 NLinear 线性层；投影器是训练集生成的冻结数据工件，不接收梯度。
**参数量审计**：在 21 个 (setting, seed) 组合内，各实验臂的 `parameter_count` 与
`trainable_parameter_count` 完全一致，因此任何差异都不能用容量不同解释。

### 1.4 对照与隔离

- 每个 setting 的 `direct` 对照都在**本实验内按同一代码路径重训**（未复用其它实验的
  数字），因此 ΔMSE/ΔMAE 是严格配对的。
- `RRR-2` 使用本轮重新计算的 `Qrrr2`（全局 RRR 方向 1+2），与 Cone 族共用同一投影安装点
  （`x_last` 中心化之后、线性层之前），差异只来自被保留的子空间。
- test 的读取：每个 run 在 checkpoint 冻结后立即读一次 test（`--evaluate-test`），
  结果写回该 run 自己的 `metrics.csv`；没有事后重读。
- 诊断（非阻断）：本轮 `Qcone1` 与上一轮 `top2_direction_retention_v1` 的 `Q1` 的子空间
  主夹角最大 1.21e-06°（其中 3 个 setting 为 0.0°），即本轮 Cone-1 与前置实验的 V1
  在几何上等价。Electricity-336 上一轮没有 Q1，本轮新增。

---

## 2. Stage 0：方向 1 的 bootstrap 邻域几何（计划表 1、表 2）

### 2.1 构造方法

每个 setting 只读训练 split：按既有 RRR 口径算全训练集方向 1（`q1`）；把训练窗口起点切成
16 个连续区块；固定种子 `20260916` 做 64 次区块有放回重采样，每次重算方向 1 并与 `q1`
对齐符号；去掉平行于 `q1` 的分量（`r_m = (I - q1 q1ᵀ) q1_bootstrap_m`）；对 `{r_m}` 做
PCA 得到切向方向，再拼成嵌套基：

```text
Qcone1 = [q1]              (k=1)
Qcone2 = [q1, v1]          (k=2)
Qcone4 = [q1, v1, v2, v3]  (k=4)
Qcone8 = [q1, v1, ..., v7] (k=8)
```

宽度参数是**保留的切向秩 k-1**，不是字面角度阈值：线性层一旦同时看到多条靠近方向 1 的
射线，就等价于保留它们张成的整个子空间，所以“5°/10°扇形”无法持续约束后续线性层。

### 2.2 表 1：bootstrap 几何

| Setting | blocks | replicates | angle median | angle p90 | tangent effective rank | QC |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-96 | 16 | 64 | 27.04° | 35.08° | 64 | PASS |
| ETTh2-720 | 16 | 64 | 33.57° | 46.44° | 64 | PASS |
| ETTm2-96 | 16 | 64 | 15.44° | 20.49° | 64 | PASS |
| ETTm2-192 | 16 | 64 | 16.11° | 20.97° | 64 | PASS |
| Weather-96 | 16 | 64 | 9.84° | 12.28° | 64 | PASS |
| Weather-192 | 16 | 64 | 9.63° | 11.93° | 64 | PASS |
| Electricity-336 | 16 | 64 | 6.26° | 8.94° | 64 | PASS |

每个 setting 的 16 个连续区块覆盖 450–2,255 个训练窗口起点（每区块、每通道），参与的
窗口×通道对从 54,775（ETTh2-96）到 5,571,597（Electricity-336）。
`tangent effective rank` 等于 replicate 数 64，说明切向谱在 720 维中没有更小的有效维度，
因此 k=8 远未触到 bootstrap 云自身的维度上限。

**一个重要的几何事实**：bootstrap 方向 1 与全训练集方向 1 的夹角中位数在 ETT 上是
15°–34°，只有 Weather 与 Electricity 在 6°–10°。也就是说，在 ETT 上“方向 1”本身在
重采样下并不稳定；这为“方向 1 可能只是一个中心轴、有效机制在其邻域内”提供了动机，
但同时也意味着在 ETT 上这个邻域可能很宽而难以用低维子空间覆盖。

### 2.3 表 2：邻域解析性质

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

- `tangent variation explained`：切向 PCA 前 k-1 个主成分占 bootstrap 偏离能量的比例。
- `visible variance`：该子空间承载的中心化输入方差比例。
- `independent predictive capture`：该子空间在训练集上的最优线性预测所捕获的可实现 MSE
  收益比例（与前置实验同一口径）。
- `overlap with direction 2`：切向成分与全局 RRR 方向 2 的重叠度。

**解析侧的关键不对称**：加宽邻域时，`visible variance` 增长很大（Electricity-336 +34.93
个百分点、Weather-192 +21.08 个百分点），而 `independent predictive capture` 只增长很小
（Electricity-336 +3.6 个百分点、ETTh2-96 +1.6 个百分点）。也就是说，新增的切向维度主要是
**输入能量大但线性预测价值低**的方向。这正是后面端到端结果没有改善的解析对应物。

### 2.4 Stage 0 六项审计

| 检查 | 结果 |
|---|---|
| 只读取训练 split | 28/28 PASS（`splits_read == ["train"]`） |
| 所有基正交 | 最大 `orthogonality_error` 2.2e-16 |
| 投影矩阵幂等 | 最大 `projector_idempotence_error` 1.1e-16 |
| `Qcone1 ⊂ Qcone2 ⊂ Qcone4 ⊂ Qcone8` | `nested_previous_error` 全 0.0 |
| 所有 Cone-k 恢复全训练集方向 1 | 最大 `recover_direction1_error` 2.5e-16 |
| bootstrap 角度与切向谱为有限值 | 28/28 PASS |
| `Qrrr2` 有效 | 正交 1.0e-16 / 幂等 2.8e-17 |

---

## 3. Stage T：seed 2021 全量 test sweep（计划表 3）

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

读法要点：

- 35 个投影 cell 中，**只有 2 个双指标优于 direct**，且都在 Weather-192（Cone-4 与
  Cone-8）。MSE 单项优于 direct 的 6/35，MAE 单项优于 direct 的 3/35。
- **k 的单调性只在部分 setting 成立。** ETTh2-720（+3.683% → +1.813%）与 ETTm2-96
  （+4.338% → +3.248% 后回升）表现得最接近单调；ETTh2-96 与 ETTm2-192 上 k=2 或 k=4
  反而是局部最差。
- **RRR-2 并没有比 Cone 族更好。** 在 6/7 个 setting 上，被选 Cone-k 的 MSE 不劣于
  RRR-2；而 RRR-2 自己相对 direct 也只在 Weather-192 与 ETTh2-96 上略优（且 MAE 更差）。
  这与前置实验“方向 2 没有稳定增量”的结论一致。

---

## 4. 按数据集宽度选择（计划表 4）

规则（计划 §6）：先取数据集宏平均 test ΔMSE 最优；保留与其差距不超过 0.10 个百分点的
候选；在候选中取宏平均 test ΔMAE 最小者；仍相同取更小的 k。

| Dataset | k=1 ΔMSE/ΔMAE | k=2 ΔMSE/ΔMAE | k=4 ΔMSE/ΔMAE | k=8 ΔMSE/ΔMAE | selected k |
|---|---:|---:|---:|---:|---:|
| ETTh2 | +1.866% / +2.312% | +2.420% / +2.798% | +1.843% / +2.008% | **+1.330% / +1.477%** | 8 |
| ETTm2 | +2.936% / +3.323% | **+2.509% / +3.104%** | +4.172% / +3.449% | +3.724% / +3.401% | 2 |
| Weather | +0.667% / +0.906% | +0.178% / +0.482% | **-0.738% / -0.500%** | -0.613% / -0.145% | 4 |
| Electricity | +2.447% / +1.518% | **+2.272% / +1.415%** | +2.984% / +1.857% | +2.975% / +1.776% | 2 |

- 四个数据集的候选池都只含一个宽度（`selection_pool` 分别为 [8]、[2]、[4]、[2]），
  MAE 平手规则没有被触发。
- **必须避免的误读**：这里的“最优”是在四个投影宽度之间取最优，不是“相对 direct 有改善”。
  ETTh2、ETTm2、Electricity 所选宽度的宏平均仍为正（仍劣于 direct）；只有 Weather 的
  k=4 在宏平均上优于 direct。因此“4/4 数据集选择 k>1”不能读成“4/4 数据集从更宽邻域获益”。
- 选择依据是 seed 2021 的 test 指标，属于 **test-set selection**。`test_selection.json`
  中含 `"test_set_selection": true`。

---

## 5. Stage S：三 seed 稳定性（计划表 5）

三 seed = 2021（Stage T）+ 2022/2023（Stage S）。Stage S 只训练 direct / RRR-2 / Cone-1 /
该数据集被选的 Cone-k 四个臂，所以未选宽度只有 seed 2021 一列（见表 3）。std 为样本标准差。

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

逐 seed 的 selected Cone-k vs Cone-1（同为三 seed）：

| Setting | selected k | seed 2021 | seed 2022 | seed 2023 | MSE 胜出 |
|---|---:|---|---|---|---:|
| ETTh2-96 | 8 | 输 (0.2744 vs 0.2722) | 输 (0.2753 vs 0.2724) | 赢 (0.2749 vs 0.2752) | 1/3 |
| ETTh2-720 | 8 | 赢 (0.3980 vs 0.4053) | 输 (0.4039 vs 0.4038) | 赢 (0.3984 vs 0.4090) | 2/3 |
| ETTm2-96 | 2 | 赢 (0.1640 vs 0.1653) | 赢 (0.1639 vs 0.1645) | 输 (0.1630 vs 0.1626) | 2/3 |
| ETTm2-192 | 2 | 赢 (0.2189 vs 0.2190) | 赢 (0.2208 vs 0.2219) | 输 (0.2217 vs 0.2204) | 2/3 |
| Weather-96 | 4 | 赢 (0.1470 vs 0.1486) | 赢 (0.1457 vs 0.1470) | 输 (0.1485 vs 0.1481) | 2/3 |
| Weather-192 | 4 | 赢 (0.1885 vs 0.1919) | 赢 (0.1896 vs 0.1912) | 赢 (0.1891 vs 0.1906) | 3/3 |
| Electricity-336 | 2 | 赢 (0.1654 vs 0.1657) | 输 (0.1676 vs 0.1675) | 输 (0.1673 vs 0.1666) | 1/3 |

- 唯一在 3/3 个 seed 上都优于 Cone-1 的是 **Weather-192 的 Cone-4**。
- 只用单个 seed 胜出的是 ETTh2-96（Cone-8）与 Electricity-336（Cone-2）；
  这两格也正是判据 2 不成立、以及 Electricity 被排除在“宽度收益”之外的原因。
- MAE 侧的信号比 MSE 侧更一致：6/7 个 setting 上被选 Cone-k 的 MAE 优于 Cone-1。

---

## 6. 判定汇总（计划表 6）

| 判定项 | 实测 |
|---|---|
| 选择 k>1 的数据集数 | 4/4（ETTh2=8、ETTm2=2、Weather=4、Electricity=2） |
| 选择后 Cone-k 数据集宏平均 MSE 与 MAE 均优于 Cone-1 的数据集 | 3/4（ETTh2、ETTm2、Weather） |
| Cone-k 三 seed MSE 优于 Cone-1 的 setting | 4/7（ETTh2-720、ETTm2-96、Weather-96、Weather-192） |
| Cone-k 三 seed MAE 优于 Cone-1 的 setting | 6/7（除 Electricity-336 外全部） |
| Cone-k 优于或持平 RRR-2 的 setting（MSE） | 6/7（仅 ETTh2-96 除外） |
| Cone-k 优于或持平 RRR-2 的 setting（MAE，附加口径） | 6/7（仅 ETTm2-192 除外） |
| 宽度收益是否跨 seed 稳定 | 有均值收益的 4 个 setting 全部在 ≥2/3 个 seed 上同向；**没有**任何均值收益由单个异常 seed 造成 |
| 最终结论 | **部分支持** |

逐条判据（计划 §7）：

| # | 判据 | 结果 |
|---|---|---|
| 1 | 至少 3/4 个数据集选择 k>1 | ✅ 4/4 |
| 2 | 选择后的 Cone-k 在数据集宏平均 MSE 与 MAE 上优于 Cone-1 | ❌ 3/4；Electricity-336 上宏平均反而变差（MSE +0.083%、MAE +0.049%） |
| 3 | 至少 4/7 setting 上 Cone-k 三 seed MSE 优于 Cone-1 | ✅ 4/7 |
| 4 | Cone-k 在至少 4/7 setting 上优于或持平 RRR-2 | ✅ 6/7 |
| 5 | 宽度收益不是只来自单个异常 seed | ✅ 4 个均值收益 setting 全部 ≥2/3 seed 同向 |

---

## 7. 假设裁定

计划 §1 提出的假设是：

> 方向 1 可能是有效机制的中心轴，但严格秩 1 投影过窄；训练分布中的有效滤波器可能在方向 1
> 周围形成一个低维邻域；保留这个邻域可能比加入全局方向 2 更有效。

拆成两个可直接检验的子命题：

| 子命题 | 证据 | 裁定 |
|---|---|---|
| 方向 1 周围存在稳定、低维、且带额外预测价值的局部变化空间 | 局部变化空间确实存在且可估计（表 2 的 `visible variance` 随 k 单调上升）；但它的 `independent predictive capture` 增量很小，且与方向 2 基本不重叠 | **部分成立**：空间存在，额外预测价值微弱 |
| 保留该邻域**优于**加全局方向 2 | 被选 Cone-k 在 6/7 setting 的 MSE 上不劣于 RRR-2（且 RRR-2 自己多数格劣于 direct） | **成立（但双方都基本不优于 direct）** |
| 保留该邻域能恢复完整 NLinear | 三 seed 均值口径下只有 1/7 setting（Weather-192）双指标优于 direct | **不成立** |

综合结论：**“严格秩 1 过窄”这个解释被部分压低但没有被推翻。** 证据显示的是：
方向 1 邻域里确实装着大量输入能量，但这些能量对当前 NLinear 读出头几乎不可用——
把 k 从 1 加到 8 只是让投影臂的退化从 +3.5% 收窄到 +1.7%，而不是把退化消掉。
唯一出现正向信号的是 Weather-192（以及 MAE 口径上的 Weather-96），是一个**数据集条件性**
而不是普遍性的结果。

### 为什么这可能仍然是有价值的结论

1. 它与前置实验构成一个闭环：严格秩 1（V1）不能恢复 direct，加上全局方向 2（V2/RRR-2）
   也没有增量，本轮加上“方向 1 的局部邻域”同样没有稳定增量。**三种“保留少数方向”的
   做法在 6–7 个 setting 上都打不过完整 720 维输入**，说明瓶颈很可能不在“保留了哪几个
   方向”，而在“用固定秩的线性子空间读历史”这个约束本身。
2. 表 2 给出了一个可检验的机理线索：新增切向维度的 `visible variance / predictive capture`
   比值很差。这把下一轮的候选假设从“换方向”推向“换读出方式”，例如按样本自适应选择子空间、
   或在残差头里显式建模被丢弃的补空间成分（而不是继续加宽固定基）。
3. Weather 上的正向信号（尤其 Weather-192 的 3/3 seed 一致）说明邻域机制并非全程无效，
   而是一个有适用域的效果，这与 `docs/README.md` 中记录的“平滑/低秩效应均为
   数据集条件性”的长期结论一致。

---

## 8. 审计与偏差披露

**已执行的审计**

| 审计项 | 结果 |
|---|---|
| Stage 0 六项检查 | 28/28 PASS |
| 运行完成度 | 98/98 `completed`，0 `failed`，0 重试 |
| 参数量一致性 | 21 个 (setting, seed) 内所有臂的 `parameter_count` 与 `trainable_parameter_count` 相同 |
| test 读取 | 每个 run 在 checkpoint 冻结后只读一次，写入自己的 `metrics.csv`；无事后重读 |
| 聚合可追溯 | `aggregation/results.csv` 的 98 行可逐行回溯到 `runs/<run_id>/metrics.csv` |
| 方向 1 跨轮一致性 | 本轮 `Qcone1` 与上一轮 `Q1` 子空间主夹角 ≤1.21e-06° |

**协议边界（必须随结论一起引用）**

1. 本实验**明确允许**用 seed 2021 的 test 指标选择邻域宽度，并允许按数据集选择不同宽度。
2. Stage S 的 seeds 2022/2023 **不是独立确认集**：结构与宽度都已依据 seed 2021 的 test
   结果选定。Stage S 只能解释为**选择后稳定性复核**。
3. 不得用本实验的数字声明无偏泛化提升。所有数字都是本实验内部的配对比较。
4. 全部 `k=1/2/4/8` 的结果都已保留（表 3、表 4、`results.csv`），没有只报告最优宽度。
5. 本报告的 test 数字来自本轮重训的 `direct` 对照，**不是**金标准数字，也不与之混用。
6. 运行环境（torch 2.6.0 / Lightning 2.6.5 / A800）与仓库记录的 4090 环境不同。

**已知未做**

- 未做样本级高误差/退化分析（本计划未要求）。
- 未对 Electricity-336 的 bootstrap 邻域做与训练后有效滤波器的对齐比较
  （`scripts/align_trained_and_optimal_direction.py` 支持该分析，但未在本轮扩到 Cone 族）。
- 未做“自适应子空间 / 补空间建模”的候选实现——这是本轮结论指向的下一轮方向。

---

## 9. 复现方式

```bash
# 0) Stage 0：生成 Qrrr2 与 Qcone1/2/4/8（只读训练 split，不训练模型）
python scripts/compute_direction1_neighborhood_projectors.py \
  --output-dir research_runs/direction1_neighborhood_v1/projectors \
  --bootstrap-blocks 16 --bootstrap-replicates 64 \
  --bootstrap-seed 20260916 --widths 1,2,4,8

# 1) Stage T：7 setting × 6 臂 × seed 2021 = 42 runs（一卡一 run，六路并发）
python scripts/run_direction1_neighborhood_matrix.py \
  --stage sweep --gpus 0,1,2,3,4,5 \
  --output-root research_runs/direction1_neighborhood_v1

# 2) 按数据集选择宽度（读 seed 2021 test，属于 test-set selection）
python scripts/select_direction1_neighborhood_width.py \
  --root research_runs/direction1_neighborhood_v1

# 3) Stage S：seeds 2022/2023 × direct/RRR-2/Cone-1/selected Cone-k = 56 runs
python scripts/run_direction1_neighborhood_matrix.py \
  --stage confirm \
  --selection-file research_runs/direction1_neighborhood_v1/test_selection.json \
  --gpus 0,1,2,3,4,5 \
  --output-root research_runs/direction1_neighborhood_v1

# 4) 汇总成计划表 1–6（+ 上一轮 Q1 漂移诊断表）
python scripts/aggregate_direction1_neighborhood.py \
  --root research_runs/direction1_neighborhood_v1
```

产物：

- `research_runs/direction1_neighborhood_v1/projectors/`：35 个冻结基 `.npy`、
  `stage0_audit.{json,csv,md}`、`projectors.json`
- `research_runs/direction1_neighborhood_v1/runs/`：98 个 run（`metrics.csv`、`config.json`、
  `status.json`、`environment.json`、Lightning 日志）
- `research_runs/direction1_neighborhood_v1/{sweep,confirm}_manifest.json`、
  `{sweep,confirm}_summary.json`
- `research_runs/direction1_neighborhood_v1/test_selection.{json,md}`
- `research_runs/direction1_neighborhood_v1/aggregation/`：`table1`–`table7*.md`、
  `results.csv`、`aggregate.json`

`research_runs/` 已 gitignore，不入库。
