# PCTF：相位—周期—轨迹统一模型与实验计划

> 状态：模型、preset、实验 runner 和结构测试已完成，**等待确认具体实现**；尚未训练，下面
> 所有 PCTF 结果格均为空。实验编号：`pctf_v1`。

## 一句话方案

只训练一个 PhaseFormer：PhaseFormer 建模相位，NLinear 建模绝对水平、近期漂移和完整未来
轨迹，ICPT 不再输出第三份完整预测，而只修正 NLinear 漏掉的“周期内形状”和“周期间相对
水平”；最后仍由 RCRF 根据相位可靠度连续融合相位预测与统一残差预测。

## 1. 为什么这样整合

| 已有模型/实验 | 真实结果 | 对新结构的直接约束 |
|---|---|---|
| A1：RCRF+NLinear | 正式 test 相对 A2 宏平均比 1.0002，最差回退 0.52%；ETTh2-H96 MSE 最好 | NLinear 是稳定的完整轨迹锚点，不能轻易删掉 |
| A2：RCRF+NLinear+LFF | 当前统一单模型正式 test incumbent | 新模型必须与它公平比较，而不是只比原始 PhaseFormer |
| I0：RCRF+ICPT | 正式 test 相对 A2 宏平均比 0.9969，8/12 setting 双指标改善；但 ETTh2-H96 MSE 回退 6.47% | ICPT 的周期间关系有价值，但直接替换 NLinear 风险很大 |
| HPTC-H4 | H96 validation 相对 A1 宏平均改善 0.29%，最差回退 0.34%，但仅 3/6 双指标改善；相对三模型包络仍差 1.47% | 正交小修正比完整专家路由安全，但“每周期严格零均值”过强 |
| TriAxis/M3 | 历史路由命中率约 31%–42%；M3 是三个冻结 checkpoint 的 ensemble | 不再做完整预测的 hard/soft expert ensemble |

因此，本实验不是把 PhaseFormer、ICPT、NLinear 三个模型的输出做加权平均，而是把它们放进
一个可识别的生成过程：相位主干负责相位，NLinear 负责完整轨迹，ICPT 只进入两个受约束的
周期修正子空间。

## 2. 具体实现

给定归一化历史 `X`：

- `P = PhaseFormer(X)`：相位分支的完整预测；
- `T = NLinear(X)`：轨迹分支的完整预测；
- `C = ICPT(X)`：将 720 步按 24 步切成 cycle patch 后得到的周期候选；ICPT 使用已有 I0
  的 no-PE future-query decoder，`d_model=32`、4 heads、1 层 encoder/decoder；
- `D = C - T`：ICPT 相对 NLinear 真正新增的信息。

把未来第 `j` 个 24 步周期的差值写为 `D_j`，定义：

`D_shape,j = D_j - mean_time(D_j)`

`D_level,j = mean_time(D_j) - mean_all_future(D)`

其中 `D_level,j` 在该周期的 24 个位置上广播。最终统一残差分支为：

`R = T + g_shape(X,j)·D_shape + Π_H0[g_level(X,j)·D_level]`

`Π_H0` 表示在 gate 之后再次减去完整 horizon 均值；这是必要的，因为不同未来周期的 gate
不同时，仅约束原始 `D_level` 的均值并不足以约束实际加权更新。

外层保持 A1 的 RCRF：

`Y = (1-α(X))·P + α(X)·R`

这样有三条可以直接检查的性质：

1. `D_shape` 每个未来周期均值为零，只改变周期内波形；
2. `D_level` 在完整 horizon 上均值为零，只允许不同未来周期之间重新分配水平，不改变绝对
   预测水平；
3. 两个修正正交，NLinear 独占 horizon-wide 绝对水平和整体漂移，ICPT 不会变成隐藏的第三个
   完整预测器。

`g_shape`、`g_level` 是逐未来周期的可学习 sigmoid gate，初值均为 0.10；可选历史置信度只
能把 gate 连续缩小，不能选择完整专家。外层 `α` 仍由原始 phase series 的同相位一致性给出，
所以周期修正不能修改决定自身权重的证据。

### 历史置信度

使用最近两个历史周期作为伪目标。对每个伪目标，只保留它之前的历史，左侧用最早周期补齐到
720 步，然后让同一个 NLinear 和 ICPT 预测下一周期。整个过程不读取真实未来，也不反向传播
置信度梯度。

- `fixed`：置信度恒为 1，只学习逐未来周期 gate；
- `masked_absolute`：按 ICPT 的历史重建误差与跨伪起点方差指数衰减修正；
- `masked_regret`：只在 ICPT 历史误差高于 NLinear 时衰减，避免“ICPT 绝对误差大但相对
  NLinear 仍更好”被误杀。

置信度下限为 0.05，`risk_scale=1.0`，方差权重 0.5。masked 模式把两个伪起点合并为一个
batch，因此没有 Python 逐样本循环，但 residual path 的 ICPT 计算量约为 fixed 模式的 3 倍；
正式结果必须同时报告参数量、显存和推理耗时。

## 3. 五个候选与所回答的问题

| ID | preset | shape | relative level | history confidence | 要回答的问题 |
|---|---|---:|---:|---|---|
| P0 | `pctf_shape_fixed` | ✓ | — | fixed | HPTC 的纯形状修正能否复现？ |
| P1 | `pctf_level_fixed` | — | ✓ | fixed | 放松每周期零均值本身是否有效？ |
| P2 | `pctf_dual_fixed` | ✓ | ✓ | fixed | 两个正交周期子空间是否有组合增益？ |
| P3 | `pctf_dual_masked` | ✓ | ✓ | absolute | ICPT 自身历史误差能否安全收缩？ |
| P4 | `pctf_dual_regret` | ✓ | ✓ | regret vs NLinear | 相对优势证据是否优于绝对误差？ |

所有候选继承 A1 的相位校正栈、RCRF 公式、训练损失和 NLinear；除表中变量外保持一致。构造
ICPT 时隔离随机数状态，已校验同一 seed 下 A1 与 PCTF 的 PhaseFormer embedding、routing
layers 和 predictor 初始化逐参数一致。

## 4. 预注册实验协议

### 阶段 A：validation-only 筛选

- 数据：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity；
- setting：输入 720、输出 96、period 24；
- seed 2021，30% train，最多 8 epoch，Huber，最低 validation loss checkpoint；
- 同协议重跑 A1、A2、I0 和 P0–P4，共 `6×8=48` 个 model run；
- runner 明确禁止生成或读取 test 指标，汇总器发现 test 字段非空会直接报错。

候选相对 A2 的晋级门槛同时满足：12 个 MSE/MAE 比值宏平均不高于 0.998、至少 4/6 数据集
双指标改善、最差单指标回退不超过 1.0%；此外，相对 A1/A2/I0 逐指标包络的宏平均比不能高于
1.005。若无人通过，实验停止，不访问 test。

| candidate | macro / A2 | 双指标改善 / 6 | worst / A2 | macro / reference envelope | 决策 |
|---|---:|---:|---:|---:|---|
| P0 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P1 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P2 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P3 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P4 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 阶段 B：冻结后的正式确认

只有阶段 A 通过后，冻结唯一冠军，在六数据集、H96/H192、seeds 2021/2022/2023 上对 A1、
A2、I0 和冠军统一重跑，共 `6×2×3×4=144` 个 model run。每个训练只按 validation 选 checkpoint，
训练结束后才读取一次 test；不再根据 test 换候选或调参。

| test setting | Golden | A1 | A2 incumbent | I0 | PCTF champion | 相对 A2 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1-96 | 0.359 / 0.382 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh1-192 | 0.397 / 0.404 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh2-96 | 0.275 / 0.338 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh2-192 | 0.341 / 0.376 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm1-96 | 0.293 / 0.344 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm1-192 | 0.323 / 0.361 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm2-96 | 0.163 / 0.256 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm2-192 | 0.219 / 0.293 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Weather-96 | 0.148 / 0.195 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Weather-192 | 0.193 / 0.237 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Electricity-96 | 0.129 / 0.221 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Electricity-192 | 0.148 / 0.238 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |

正式替换 A2 还需：24 个指标宏平均比 `<0.998`、至少 8/12 setting 双指标改善、最差单指标
回退 `≤0.5%`。同时报告三 seed 均值/sample std、严格稳定超过 Golden 的 setting 数，以及
相对 A1/A2/I0 逐指标包络的距离。

## 5. 执行方式

仅预览命令，不训练：

```bash
uv run python scripts/run_pctf_experiment.py --stage screen-dry
uv run python scripts/run_pctf_experiment.py --stage confirm-dry --champion pctf_dual_masked
```

确认后才执行：

```bash
uv run python scripts/run_pctf_experiment.py --stage screen --progress
uv run python scripts/run_pctf_experiment.py --stage screen-summarize
uv run python scripts/run_pctf_experiment.py --stage confirm --progress
uv run python scripts/run_pctf_experiment.py --stage confirm-summarize
```

`confirm` 不接受命令行临时冠军，必须读取通过门槛后写出的 `screen_decision.json`；这保证 test
阶段无法绕过 validation 冻结。历史 ETTh2/ETTm2 test 已被多次查看，因此即使本轮严格冻结，
最终论文仍需披露整个项目层面的 test exposure，不能称为完全盲测。

## 6. 已完成的代码验证

- 全仓 `208 passed`，另有 `187 subtests passed`；现有警告仅为无 Trainer 的测试日志和
  测试环境 NVML，不影响断言；
- PCTF 头支持 H96/H192/H336/H720，forward 均为有限值；
- 数值验证 shape 的逐周期零均值、level 的全 horizon 零均值及两者正交；
- 构造型测试验证两个 masked origin 严格只使用各自目标之前的数据；
- fixed、absolute、regret 三种置信度均有正确 shape 和 `[0.05,1]` 边界；
- NLinear、ICPT、shape gate、level gate 在非退化学习状态均能获得梯度；
- 五个 preset 都是单 PhaseFormer、单 checkpoint，启用 RCRF 且不启用 TriAxis/SafeTriAxis；
- dry-run 校验筛选为 48 runs、正式确认是 144 runs，且只有冻结确认命令含 test 开关；
- 汇总器用合成矩阵验证冠军冻结，并验证 validation 文件一旦出现 test 数值就拒绝汇总。

注意：ICPT 零初始化为 RepeatLastCycle，初始时所有未来周期的相对 level 恰好相同，所以
`level gate` 在第一个优化步可能没有梯度；ICPT 接受非平稳未来监督更新后，相对 level 立即成为
可学习量。该 warm start 保留了旧 ICPT 的稳定初始化，不应伪装成实现错误，但训练时需记录
level correction/gate 是否真正离开零；若 P1/P2/P3/P4 的 level correction 长期接近零，则应
客观判定该机制未被数据使用。
