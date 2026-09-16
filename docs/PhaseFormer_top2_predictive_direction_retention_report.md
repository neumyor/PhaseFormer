# PhaseFormer 前两预测方向数据保留实验结果报告

> 状态：**已全部完成**（Stage 0 / Stage A / Stage B / 一次性 test 读取）
>
> 日期：2026-09-16
>
> 计划：`docs/PhaseFormer_top2_predictive_direction_retention_plan.md`
>
> 产物根目录：`research_runs/top2_direction_retention_v1/`（本机与服务器同名同结构）
>
> 代码提交：`26359ff`（Stage 0 与训练链路）、`da3cf10`（报告生成）
>
> 运行环境：远程 A800 服务器，Python 3.10 + torch 2.6.0+cu124 +
> pytorch-lightning 2.6.5，单卡 A800-SXM4-80GB，`CUDA_VISIBLE_DEVICES` 一卡一 run

## 0. 结论（先说结果）

**预注册判定：不支持。**

V1（只保留方向 1）与 V2（保留方向 1+2）都**没有**恢复到完整输入 NLinear 的
大部分贡献，而且**方向 2 的增量价值在 test 上几乎不存在**：

- V2 相对完整 `direct_nlinear` 的三 seed 平均 test 指标：六个 setting 宏平均
  MSE **+1.94%**、MAE **+1.99%**（门槛是 ≤ +0.5%）；
- V2 的 NLinear 贡献保留率中位数：MSE **61.5%**、MAE **39.3%**（门槛是 ≥ 90%，
  低于 80% 即判不支持）；
- V2 相对 V1 三 seed 双指标更好：**1/6** setting（门槛是 ≥ 4/6）；
- V2 最坏单格退化 MSE **+3.94%**、MAE **+3.35%**（均出现在 ETTh2-720，门槛 ≤ 2%）；
  六格中有 **4** 格相对 direct 的 MSE 或 MAE 退化超过 2%。

按计划 §9 的“不支持”条款，三项硬性触发条件（V2 在 ≥3/6 setting 上不优于 V1、
V2 贡献保留率中位数 <80%、≥2 个 setting 退化 >2%）**全部命中**。

计划 §9 对“不支持”预设的结论因此成立，并在本实验中得到了完整证据：

> RRR 的前两方向能够解释固定数据分布上的线性最优收益，但不足以约束端到端
> 训练时 NLinear 所需的全部输入信息。

## 1. 实验执行与协议合规

| 阶段 | 内容 | 结果 |
|---|---|---|
| Stage 0 | 6 个 setting 的 `Q1/Q12` 投影器 + 6 项审计 | 6/6 全部 PASS，无训练 |
| Stage A | seed 2021 的 V1/V2/phase-only，18 次训练 | 18/18 完成，无异常终止 |
| Stage B | seed 2022/2023 的 V1/V2/phase-only，36 次训练 | 36/36 完成，无异常终止 |
| 一次性 test | 冻结 checkpoint 后统一读取 test | 72 格全部读出，0 个问题 |

**新增训练 54 次**（计划预估 36 次 + phase_only 三 seed 对照 18 次），
累计 GPU 时间 **6.39 小时**，全部在 6 张空闲卡（GPU 0–5）上以“一卡一 run”
完成，未占用被 vLLM 服务占用的 GPU 6/7。

### 1.1 对照复用

`direct_nlinear` 对照**未重训**，而是按计划 §6 复用既有 run。
`scripts/audit_top2_direction_retention_reuse.py` 逐个核对 setting、seed、
`lookback=720`、`period=24`、Huber、30 epoch、`percent=100`、该 setting 冻结的
`(gate_init, learning_rate)`、best-validation checkpoint 存在性、test 指标存在性
以及“非作废批次”：

- **18/18 全部通过并复用**（`reuse_audit.md`）；
- 4 个候选被拒（gate/lr 与冻结值不一致，来自早期批次）；
- 复用 run 的原始目录被就地读取，产物**没有**复制进本实验目录。

`phase_only` 则在本实验内用同一套代码与协议**重新训练了三个 seed**，以保证与
V1/V2 完全同源可比。

### 1.2 变异点隔离

四个实验臂的可训练参数量与主干完全一致，V1/V2 与 direct 的参数量逐格相等
（例如 ETTh2-96 均为 70,939，Weather-192 均为 142,155）。
冻结投影器是非训练属性，不进入 `state_dict`，也不进入优化器与参数量统计
（`results.csv` 的 `parameter_count` 可直接核对）。因此本实验的性能差异只能归因于
**NLinear 可见的输入信息**，不是模型容量。

## 2. Stage 0：投影器审计（计划表 1）

方向按计划 §5 在**训练 split** 上用通道池化矩估计求解：

```text
Z = x_window - x_last,  D = y_horizon - x_last
S = Szy^T (Szz + eps I)^-1 Szy,  eps = 1e-6
b_i = u_i^T Szy^T (Szz + eps I)^-1,  {b1} -> Q1, {b1,b2} -> Q12
```

| Setting | train windows | λ1 share | λ2 share | λ3 share | λ2/λ3 gap | Q1 正交误差 | Q12 正交误差 | 幂等误差 | projector hash | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh2-96 | 7825 | 0.7793 | 0.0669 | 0.0566 | 0.153 | 0.00e+00 | 2.22e-16 | 5.55e-17 | `1c95a680d7afcdb8` | PASS |
| ETTh2-720 | 7201 | 0.8570 | 0.0373 | 0.0350 | **0.063** | 2.22e-16 | 2.22e-16 | 1.11e-16 | `56157e7328048c83` | PASS |
| ETTm2-96 | 33745 | 0.7251 | 0.1267 | 0.0579 | 0.543 | 0.00e+00 | 1.11e-16 | 5.55e-17 | `cabb2b9e879b636a` | PASS |
| ETTm2-192 | 33649 | 0.7462 | 0.0959 | 0.0620 | 0.354 | 2.22e-16 | 8.88e-16 | 1.11e-16 | `112ea213cc9f5ac8` | PASS |
| Weather-96 | 36072 | 0.8617 | 0.1065 | 0.0244 | 0.771 | 2.22e-16 | 1.11e-16 | 1.11e-16 | `44f3d4774d9423ca` | PASS |
| Weather-192 | 35976 | 0.7833 | 0.1300 | 0.0673 | 0.482 | 2.22e-16 | 2.22e-16 | 5.55e-17 | `cd74cda3ca572425` | PASS |

六项审计全部通过：训练样本数与既有划分一致；正交/幂等误差为机器精度；V1 只能恢复
`b1`、V12 同时恢复 `b1` 与 `b2`、且 V1 **无法**重构 `b2`；只读取训练 split；
`b1^Tz`、`b2^Tz` 的样本标准差为正；同一 setting 的三个 seed 共享同一投影器
（basename 指向同一文件，sha256 一致，已逐 run 记录在训练日志中）。

**方向 2 单独朝向的稳定性标注**（计划 §5 要求，不做事后加维）：

- **ETTh2-720（λ2/λ3 gap = 0.063）与 ETTh2-96（gap = 0.153）**：
  第二、第三特征值接近，**方向 2 的单独朝向可能不稳定**，这两个 setting 上
  “保留方向 2”的语义本身带噪；
- ETTm2-96（0.543）、ETTm2-192（0.354）、Weather-96（0.771）、Weather-192（0.482）
  的 gap 较宽，方向 2 的朝向相对稳定。

值得注意：ETTh2-720 恰好也是 V2 相对 direct 退化最严重的 setting（MSE +3.94%），
这与“方向 2 在该 setting 朝向不稳定”的先验标注一致，但本实验不做因果归因。

### 2.1 独立数值交叉校验

Stage 0 的特征值用仓库既有的 `research_runs/lowrank_data_property_v2/moments_*.npz`
独立复算过：本脚本从原始 split 重新累积矩得到的 top-5 特征值与该分析存档的
`pred_eigvals_top5` 最大相对差 **3.8e-06**（同为 ridge 1e-6），说明方向求解与
既有 RRR 实现一致，不是新引入的数值口径。窗口数亦与既有分析的
`n_train_windows`（ETTh2-96 = 54775、ETTh2-720 = 50407 等）逐格相等。

## 3. Stage A：单 seed 链路验证（计划表 2）

setting 全部 6 个，seed 2021，**只读 validation**，test 未参与。

| Setting | direct MSE/MAE | phase-only MSE/MAE | V1 MSE/MAE | V2 MSE/MAE | V1 retention | V2 retention | V2−V1 | QC |
|---|---|---|---|---|---|---:|---:|---:|---|
| ETTh2-96 | 0.2040 / 0.3116 | 0.2169 / 0.3238 | 0.2084 / 0.3185 | 0.2080 / 0.3179 | 66.0% | 69.4% | -0.000442 | validation: retention defined |
| ETTh2-720 | 0.6095 / 0.5424 | 0.6787 / 0.5853 | 0.6298 / 0.5590 | 0.6255 / 0.5583 | 70.7% | 76.8% | -0.004259 | validation: retention defined |
| ETTm2-96 | 0.1119 / 0.2281 | 0.1199 / 0.2402 | 0.1165 / 0.2361 | 0.1160 / 0.2351 | 42.7% | 48.6% | -0.000473 | validation: retention defined |
| ETTm2-192 | 0.1499 / 0.2649 | 0.1545 / 0.2711 | 0.1577 / 0.2748 | 0.1576 / 0.2735 | N/A | N/A | -0.000058 | validation: direct 未优于 phase-only → retention N/A |
| Weather-96 | 0.3869 / 0.2713 | 0.3916 / 0.2739 | 0.3828 / 0.2696 | 0.3926 / 0.2780 | 186.5% | -21.6% | +0.009823 | validation: retention defined |
| Weather-192 | 0.4454 / 0.3109 | 0.4538 / 0.3157 | 0.4460 / 0.3137 | 0.4448 / 0.3129 | 92.8% | 106.8% | -0.001183 | validation: retention defined |

QC 记录：链路本身完全正常——18/18 训练稳定收敛，投影器逐 run 哈希一致，
`TOPDIR_AUDIT` 证明“投影后 NLinear 输入与 direct 输入不同、head 输出也不同”，
且 V1/V2 的保留能量（0.0095 / 0.0245，ETTh2-96）与 Stage 0 解析值
（0.0108 / 0.0261）量级一致。**Stage A 的结果没有被用来修改方向、增维或调参**，
按计划 §7 无论正负都进入 Stage B。

ETTm2-192 在 validation 上出现 `direct` 未优于 `phase-only` 的情况，
按计划 §8.2 该 setting 的 validation 保留率记为 **N/A**，只报绝对值与相对 direct 的变化。

## 4. Stage B：三 seed 正式 test 结果（计划表 3）

所有训练与审计通过后，四个实验臂的冻结 checkpoint**各读取一次** test；
36 个新 checkpoint 由独立的 `read_top2_direction_retention_test.py` 读出，
该步骤**不写回** `metrics.csv`。

| Setting | Model | MSE mean±std | MAE mean±std | ΔMSE vs direct | ΔMAE vs direct | MSE retention | MAE retention |
|---|---|---|---:|---:|---:|---:|---:|
| ETTh2-96 | phase-only | 0.2818 ± 0.0009 | 0.3434 ± 0.0011 | 3.15% | 3.01% | 0.0% | 0.0% |
| ETTh2-96 | direct | 0.2732 ± 0.0030 | 0.3334 ± 0.0014 | 0.00% | 0.00% | 100.0% | 100.0% |
| ETTh2-96 | V1 | 0.2733 ± 0.0017 | 0.3385 ± 0.0022 | 0.02% | 1.53% | 99.4% | 49.0% |
| ETTh2-96 | V2 | 0.2740 ± 0.0031 | 0.3388 ± 0.0031 | 0.29% | 1.64% | 90.7% | 45.6% |
| ETTh2-720 | phase-only | 0.4161 ± 0.0085 | 0.4491 ± 0.0057 | 6.02% | 4.81% | 0.0% | 0.0% |
| ETTh2-720 | direct | 0.3925 ± 0.0018 | 0.4285 ± 0.0019 | 0.00% | 0.00% | 100.0% | 100.0% |
| ETTh2-720 | V1 | 0.4060 ± 0.0027 | 0.4431 ± 0.0014 | 3.46% | 3.39% | 42.6% | 29.5% |
| ETTh2-720 | V2 | 0.4079 ± 0.0059 | 0.4428 ± 0.0039 | **3.94%** | **3.35%** | 34.5% | 30.5% |
| ETTm2-96 | phase-only | 0.1743 ± 0.0018 | 0.2653 ± 0.0012 | 9.57% | 6.75% | 0.0% | 0.0% |
| ETTm2-96 | direct | 0.1591 ± 0.0005 | 0.2485 ± 0.0005 | 0.00% | 0.00% | 100.0% | 100.0% |
| ETTm2-96 | V1 | 0.1641 ± 0.0014 | 0.2559 ± 0.0014 | 3.19% | 2.98% | 66.6% | 55.9% |
| ETTm2-96 | V2 | 0.1642 ± 0.0009 | 0.2557 ± 0.0013 | 3.25% | 2.89% | 66.1% | 57.2% |
| ETTm2-192 | phase-only | 0.2282 ± 0.0005 | 0.2998 ± 0.0019 | 6.25% | 4.16% | 0.0% | 0.0% |
| ETTm2-192 | direct | 0.2148 ± 0.0008 | 0.2878 ± 0.0002 | 0.00% | 0.00% | 100.0% | 100.0% |
| ETTm2-192 | V1 | 0.2204 ± 0.0014 | 0.2969 ± 0.0016 | 2.63% | 3.14% | 57.9% | 24.6% |
| ETTm2-192 | V2 | 0.2206 ± 0.0023 | 0.2959 ± 0.0010 | 2.69% | 2.79% | 56.9% | 32.9% |
| Weather-96 | phase-only | 0.1501 ± 0.0004 | 0.1968 ± 0.0012 | 2.48% | 1.47% | 0.0% | 0.0% |
| Weather-96 | direct | 0.1465 ± 0.0004 | 0.1939 ± 0.0001 | 0.00% | 0.00% | 100.0% | 100.0% |
| Weather-96 | V1 | 0.1479 ± 0.0008 | 0.1955 ± 0.0019 | 0.94% | 0.82% | 62.0% | 43.9% |
| Weather-96 | V2 | 0.1499 ± 0.0020 | 0.1968 ± 0.0028 | 2.30% | 1.46% | 7.2% | 0.4% |
| Weather-192 | phase-only | 0.1952 ± 0.0016 | 0.2401 ± 0.0016 | 1.57% | 1.30% | 0.0% | 0.0% |
| Weather-192 | direct | 0.1921 ± 0.0013 | 0.2370 ± 0.0011 | 0.00% | 0.00% | 100.0% | 100.0% |
| Weather-192 | V1 | 0.1912 ± 0.0007 | 0.2372 ± 0.0008 | -0.47% | 0.07% | 130.1% | 94.9% |
| Weather-192 | V2 | 0.1906 ± 0.0008 | 0.2366 ± 0.0009 | **-0.82%** | **-0.19%** | 152.2% | 114.4% |

读数要点：

1. **只有 Weather-192 的 V2 真正追平或略超 direct**（MSE −0.82%，保留率 152.2%）；
2. **ETTh2-96 的 V1 在 MSE 上接近追平**（+0.02%，保留率 99.4%），但同一模型的
   MAE 只保留 49.0%——即“看起来追平”发生在 MSE 口径而非 MAE 口径；
3. 其余四个 setting（ETTh2-720、ETTm2-96、ETTm2-192、Weather-96）的 V1/V2
   都明显落后于 direct，保留率 7%–67%；
4. 计划 §3 给出的理论 capture（方向 1：73.9%–86.2%；方向 1+2：85.8%–96.7%）
   是**固定分布上的最优秩映射**在 validation 上的理论参照；端到端训练后实际只留下
   其中一部分，这本身就是本实验要测的量。

## 5. 方向 2 的增量价值（计划表 4）

```text
MSE gap recovery = (metric_V1 - metric_V2) / (metric_V1 - metric_direct) × 100%
```

仅在 V1 差于 direct 时计算；V1 已不差于 direct 时记 N/A（无 gap 可恢复）。

| Setting | λ2 share | V1→V2 ΔMSE | V1→V2 ΔMAE | MSE gap recovery | MAE gap recovery | 三 seed 方向一致 |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-96 | 0.0669 | -0.000744 | -0.000339 | -1348.6% | -6.6% | 1/3 |
| ETTh2-720 | 0.0373 | -0.001910 | +0.000208 | -14.1% | 1.4% | 1/3 |
| ETTm2-96 | 0.1267 | -0.000088 | +0.000218 | -1.7% | 2.9% | 2/3 |
| ETTm2-192 | 0.0959 | -0.000133 | +0.001001 | -2.4% | 11.1% | 1/3 |
| Weather-96 | 0.1065 | -0.001992 | -0.001239 | -144.4% | -77.5% | 0/3 |
| Weather-192 | 0.1300 | +0.000668 | +0.000600 | N/A（V1 不差于 direct） | 381.0% | 1/3 |

（`V1→V2 ΔMSE` 为正表示 V2 更优。）

**这是本次实验最清晰的一条负结果：加入方向 2 基本没有帮助，甚至常常有害。**

- 三 seed 均值上，V2 的 **MSE 只在 1/6** setting 优于 V1（Weather-192），
  其余 5/6 setting 反而被 V1 反超；MAE 上 V2 赢 4/6，但两处优势极小
  （ETTh2-720 为 −0.000208、ETTm2-96 为 −0.000218）；
- **双指标同时更好的只有 1/6**（Weather-192），这正是计划 §9 判定的口径；
- 在 ETTh2-96、ETTh2-720、Weather-96 上，V2 的 MSE **双双更差**，其中
  Weather-96 的 V2 把 V1 已有的收益几乎全部抹掉（retention 62.0% → 7.2%）；
- “三 seed 方向一致”最高只有 2/3（ETTm2-96），Weather-96 为 0/3，说明 V2 相对
  V1 的差异在多数 setting 上不具备跨 seed 稳定性——即使用最好的口径看，
  方向 2 的增量也在噪声量级。

λ2 share 与增量之间**没有**单调关系：λ2 share 最大的 Weather-192（0.1300）确实
拿到了唯一的正向增量，但 λ2 share 第二大的 ETTm2-96（0.1267）几乎没有变化，
而 λ2 share 最小的 ETTh2-720（0.0373）反而退化明显。按计划 §5 的先验标注，
ETTh2-720 与 ETTh2-96 的方向 2 朝向本身不稳定；这两个 setting 的 V2 也都没有
正增量，与该标注方向一致。

## 6. 诊断指标（计划表 6 / §8.4）

完整逐格诊断见 `report_tables.md` 表 6，这里只摘录关键读数：

| Setting | Arm | learned gate (mean±std) | NLinear 支路 test MSE | 可见中心化输入方差占比 | 最佳 epoch (mean±std) |
|---|---|---|---:|---:|---|
| ETTh2-96 | V1 | 0.5115 ± 0.0048 | 0.3268 | 0.0108 | 22.7 ± 1.5 |
| ETTh2-96 | V2 | 0.4954 ± 0.0060 | 0.3378 | 0.0261 | 24.0 ± 1.0 |
| ETTh2-720 | V1 | 0.5120 ± 0.0118 | 0.4601 | 0.0457 | 25.0 ± 2.0 |
| ETTh2-720 | V2 | 0.5117 ± 0.0108 | 0.4521 | 0.0579 | 24.0 ± 1.0 |
| Weather-96 | V1 | 0.2639 ± 0.0043 | 0.3811 | 0.0718 | 30.0 ± 0.0 |
| Weather-96 | V2 | 0.2548 ± 0.0091 | 0.4933 | 0.1612 | 30.0 ± 0.0 |
| Weather-192 | V1 | 0.3848 ± 0.0592 | 0.3036 | 0.0786 | 23.7 ± 7.8 |
| Weather-192 | V2 | 0.3315 ± 0.0219 | 0.3378 | 0.1573 | 27.0 ± 2.6 |

解释：

- **学习到的 gate 没有被“关掉”**。V1/V2 的 gate 都停在 0.23–0.54，说明融合层
  仍然给 NLinear 支路分配了实质性权重——退化不是因为 gate 学会了忽略该分支，
  而是因为该分支本身看不到足够的信息（或看到了误导性信息）。
  Weather-96 的 V2 gate（0.2548）低于 V1（0.2639），与 V2 的 NLinear 支路
  test MSE 反而更差（0.4933 vs 0.3811）一致：支路输出变坏，融合结果随之变坏。
- **可见方差占比极低**。V1 只看得到中心化输入方差的 0.76%–7.86%，
  V2 为 2.6%–16.1%。这与 λ1 share 高达 0.73–0.86 形成鲜明对比：
  第一方向承担了绝大部分**可实现收益**，却只占**输入方差**的极小部分。
  这说明 RRR 的方向是“任务相关”的低能量方向，用“保留多少方差”来判断信息是否
  足够会严重高估，也解释了为什么按方向裁剪输入会丢掉大量 NLinear 所需要的信号。
- `b1^Tz`、`b2^Tz` 的样本标准差在六个 setting 上都显著为正（如 ETTh2-96 为
  3.23 / 1.14），投影器确实读到了有量纲的信号，退化不是“投影后输入恒为零”这类
  实现故障。
- 最佳 epoch 全部在 10–30 之间、无异常早停；训练时间与显存开销与 direct 同量级
  （峰值显存 143–196 MB，共 6.39 GPU 小时），说明 V1/V2 不存在训练不稳定或
  资源异常。

## 7. 判定汇总（计划表 5）

| 判定项 | 预注册门槛 | 实测 | 通过 |
|---|---|---|---|
| V2 宏平均 MSE vs direct | ≤ +0.5% | 1.94% | **FAIL** |
| V2 宏平均 MAE vs direct | ≤ +0.5% | 1.99% | **FAIL** |
| V2 MSE retention 中位数 | ≥ 90% | 61.5% | **FAIL** |
| V2 MAE retention 中位数 | ≥ 90% | 39.3% | **FAIL** |
| V2 双指标优于 V1 | ≥ 4/6 settings | 1/6 | **FAIL** |
| V2 最坏单格退化 | ≤ 2.0% | MSE 3.94%（ETTh2-720）、MAE 3.35%（ETTh2-720） | **FAIL** |
| 最终结论 | 强支持 / 部分支持 / 不支持 | **不支持** | — |

“不支持”的三条硬性触发（计划 §9）全部命中：

- V2 在 ≥3/6 setting 上不优于 V1 → 实测 5/6 不优于；
- V2 贡献保留率中位数 <80% → 实测 MSE 61.5%、MAE 39.3%；
- V2 在 ≥2 个 setting 上相对 direct 的 MSE 或 MAE 退化超过 2% →
  实测 4 格（ETTh2-720、ETTm2-96、ETTm2-192、Weather-96）。

## 8. 假设裁定

| 假设 | 内容 | 裁定 |
|---|---|---|
| H1 | 方向 1 是主要信息，应保留完整 NLinear 相对 phase-only 的大部分收益 | **部分成立**。方向 1 确实拿到了相对更大的份额（V1 普遍优于 V2 之外的任何裁剪），但只在 ETTh2-96 的 MSE 口径接近追平 direct（99.4%），其余 setting 只保留 7%–67%；MAE 口径全面不足（最高 94.9%，最低 24.6%） |
| H2 | 方向 2 提供主要的形状补充，V2 应稳定优于 V1 并接近 direct | **不成立**。V2 仅在 1/6 setting 上双指标优于 V1，三格上反而更差，跨 seed 一致性最高只有 2/3 |
| H3 | 性能顺序应满足 phase-only < keep-1 ≤ keep-1-and-2 ≈ direct | **不成立**。`keep-1-and-2 ≈ direct` 只在 Weather-192 成立（MSE −0.82%）；`keep-1 ≤ keep-1-and-2` 在三 seed 均值上**只在 MAE 口径部分成立**（MSE 1/6、MAE 4/6），双指标同时更好只有 1/6；且 `keep-1-and-2` 在 5/6 setting 上明显差于 direct（+0.29%～+3.94%），整体不满足单调假设的实质内容 |

## 9. 为什么这仍然是有价值的结论

1. **理论 capture ≠ 端到端可用信息。** 计划 §3 的理论参照说前两方向能保留
   85.8%–96.7% 的可实现收益；端到端训练后实际只剩 7%–152%（中位数 61.5%）。
   RRR 方向是对**固定数据分布**上的线性最优映射做的分解，而端到端训练中
   NLinear 与 phase 主干、融合 gate 是联合优化的，其所需输入并不等于单独最优
   线性映射的输入子空间。
2. **“任务相关方向”可以是低能量方向。** λ1 占可实现收益的 73%–86%，却只占输入
   方差的 0.8%–7.9%。这解释了为什么简单的降维/投影直觉会失效：按能量裁剪与按
   任务相关性裁剪是两件不同的事。
3. **方向 2 的“形状补充”作用没有在端到端里兑现。** 无论它在固定分布上解释了
   多少剩余收益，把它单独作为第二个可见维度都不可靠——这也与 ETTh2 两格
   λ2/λ3 gap 偏小的稳定性标注互相印证。
4. **结论方向明确、成本可控。** 54 次训练、6.39 GPU 小时就给出了一个六 setting、
   三 seed、预注册门槛下的清晰负结果，且四个实验臂的参数量严格相同，排除了容量
   解释。

## 10. 可视化

`figures/`（由 `scripts/plot_top2_direction_retention.py` 从未手工录入的
`results.csv` 与 Stage 0 索引直接生成）：

| 图 | 内容 |
|---|---|
| `arms_test_metrics.png` | 六个 setting 上四个实验臂的三 seed 平均 test MSE/MAE |
| `retention.png` | V1/V2 的 NLinear 贡献保留率（含 100% 与 80% 参考线） |
| `spectrum_vs_retention.png` | 方向 2 的 λ2 share / 可见输入方差占比 vs 端到端保留率 |

**计划 §11.3–11.4（改善/退化最大的真实样本预测曲线）未执行**，理由与边界：

- 这两项需要额外导出 test 集逐样本预测，属于计划里排在“训练完成后再补充”的
  可视化条目，不属于预注册判定的一部分；
- 本次判定由三项硬性触发条件直接给出，聚合证据已经足以定论；样本级曲线在此
  负结果下主要会呈现 V1/V2/direct 三条曲线高度重合于 phase-only 之上，
  边际信息量低；
- 因此这里**明确标注为未做**，而不是用聚合图冒充样本级证据。若需要补齐，
  可以在冻结 checkpoint 上一次性导出（已有 `test_read.json` 与 checkpoint 路径，
  不需要重新训练），但按计划 §7 这不会改变任何已冻结的结论。

## 11. 复现方式

```bash
# 0) Stage 0：投影器与审计（只读训练 split，CPU）
python scripts/compute_top2_direction_projectors.py \
  --output-dir research_runs/top2_direction_retention_v1/projectors

# 1) direct_nlinear 对照复用审计
python scripts/audit_top2_direction_retention_reuse.py \
  --root research_runs/top2_direction_retention_v1

# 2) Stage A / Stage B：一卡一 run 的矩阵调度（可断点续跑）
python scripts/run_top2_direction_retention_matrix.py \
  --stage a --gpus 0,1,2,3,4,5 --skip-reused \
  --output-root research_runs/top2_direction_retention_v1
python scripts/run_top2_direction_retention_matrix.py \
  --stage b --gpus 0,1,2,3,4,5 \
  --output-root research_runs/top2_direction_retention_v1

# 3) 冻结后一次性读取 test（每 checkpoint 一次）
python scripts/read_top2_direction_retention_test.py \
  --root research_runs/top2_direction_retention_v1 --gpus 0,1,2,3,4,5

# 4) 汇总成计划表 1–6
python scripts/aggregate_top2_direction_retention.py \
  --root research_runs/top2_direction_retention_v1

# 5) 图
python scripts/plot_top2_direction_retention.py \
  --root research_runs/top2_direction_retention_v1
```

单个 run 也可复现，例如：

```bash
python scripts/run_top2_direction_retention.py \
  --dataset ETTh2 --horizon 96 --stage confirm --lookback 720 --period 24 \
  --max-epochs 30 --seed 2021 --loss huber --learning-rate 0.001 --require-cuda \
  --basis research_runs/top2_direction_retention_v1/projectors/ETTh2_96_Q1.npy \
  --overrides '{"weak_period_residual_head_type":"shared","weak_period_residual_gate_init":0.5,"learning_rate":0.001,"weak_residual_projection":"frozen_subspace","weak_residual_projection_arm":"keep_direction_1"}' \
  --output-dir research_runs/top2_direction_retention_v1
```

## 12. 审计与偏差披露

- **test 只读一次**：36 个新 checkpoint 的 test 由独立步骤读出，
  `metrics.csv` 未被该步骤改写；18 个复用对照保留其原 run 记录的 test 数值。
  没有任何 test 结果被用来修改投影器、训练配置或模型结构。
- **test 读取前的可复现性守卫**：每个新 checkpoint 在读出 test 之前，先在
  validation 上重算一次并与该 run 记录的 `val_mse` 比对。72 格中最大相对偏差
  **3.8e-05**，全部落在浮点级（dataloader worker 数不同导致的求和顺序差异）以内，
  没有出现 `val_mismatch`。这保证读出的 test 数值确实属于被审计的那套协议。
- **训练期异常**：54/54 完成，无异常终止、无 NaN、无早停异常；
  18 个 Stage A 格子在一次误写错输出目录后被中断并**全部重跑**，
  最终 54 个 run 全部位于 `research_runs/top2_direction_retention_v1/runs/`。
- **对照来源**：`direct_nlinear` 为复用（审计通过 18/18），
  `phase_only` 为本实验重训（3 seed × 6 setting）。
  报告中所有性能结论均以**本实验内配对 direct** 为准；
  未使用 `docs/PhaseFormer_gold_standard.md` 金标准数字，
  也未与 `structured_lowrank_round1` 等其它协议的数字混用。
- **未做 test-set selection**：本实验的方向、投影器、超参与模型结构均在读取 test
  之前冻结，判定按预注册门槛机械执行。
- **环境差异**：本实验在远程服务器 torch 2.6.0 + Lightning 2.6.5 上完成，
  与仓库记录的 torch 2.7.1 + Lightning 2.1.0（RTX 4090）不同；
  复用对照的 run 亦在服务器环境产生，两者环境一致。
- **方向语义**：本实验只验证“训练集 RRR 前两个任务相关方向作为固定数据瓶颈是否
  足以支撑端到端模型”，不主张方向 2 的人类可读语义命名绝对正确。
