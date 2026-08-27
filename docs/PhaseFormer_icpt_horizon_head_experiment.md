# ICPT Full-Horizon Head 与位置编码实验

> 状态：**Stage 0 与 validation-only Stage A 已完成；没有候选通过预注册门槛，Stage B 未运行，test 从未读取。** 本实验使用新实验编号，不修改或覆盖上一轮 ICPT 失败结论。

## 1. 实验要验证的设想

上一轮 ICPT 使用 future-query decoder，并要求无位置编码的 attention 架构先通过筛选；但 `ICPT-none` 对历史周期顺序近似置换不变，无法真正表达周期演化。同时，它以最近完整周期而不是最后观测值作为 warm start，与 RCRF 已知的近期漂移补偿需求不匹配。

本轮采用类似 PatchTST 的 full-horizon prediction head：将按时间顺序排列的全部历史周期 token 展平，一次性映射到完整预测 horizon。周期 encoder 负责提取跨周期关系，展平 head 保留周期顺序并为每个未来步提供独立权重；输入以最后观测值中心化，输出再加回最后值。主假设是：该结构既保留 NLinear 的直接全 horizon 映射和漂移锚点，又能通过周期 token 与位置编码增加可学习的周期归纳偏置。

## 2. 实验的整体计划

### 2.1 固定结构与公平性

输入长度 720，周期 patch `P=24`，30 个不重叠 token；`d_model=24`、4 heads、FFN=48、一层 encoder、dropout=0。预测头为 `Linear(30×24, H)`，因此主 head 与 NLinear 的 `Linear(720,H)` 参数量一致；ICPT 仅额外增加小型 patch encoder。输出层零初始化，初始结果严格等于 last-value persistence。

固定 PhaseFormer、RCRF 门控、数据切分、loss、learning rate、batch 和 checkpoint 规则。旧 ICPT decoder 路径保持不变并通过 flag-off 回归测试。

### 2.2 候选与对照

| ID | 模型 | 作用 |
|---|---|---|
| A2 | `RCRF + NLinear` | 直接 baseline |
| C0 | Full-horizon + last-cycle anchor + none | 单独检查 anchor 修正的作用 |
| P0 | Full-horizon + last-value anchor + none | 新架构无显式 PE 对照 |
| P1 | P0 + fixed sin/cos | 绝对周期序号 |
| P2 | P0 + learned absolute | 可学习绝对周期序号 |
| P3 | P0 + Time2Vec | 线性时间与可学习周期 |
| P4 | P0 + RoPE | attention 中的相对周期距离 |
| P5 | P0 + relative bias | 直接学习周期 lag 偏置 |
| P6 | P0 + ALiBi | 确定性近期偏置 |
| P7 | P0 + LFF | 可学习 Fourier 周期距离 |
| P8 | P0 + sin/cos + relative | 绝对与相对位置组合 |
| P9 | P0 + calendar | 只使用历史 timestamp，单独排名 |

### 2.3 Stage A：validation-only 广筛

四个 setting：ETTh2-720、ETTm2-96、Electricity-336、Weather-336；seed 2021、30% train、最多 8 epoch，不构造 test loader。共 48 个 run，所有 P0–P9 均执行，避免再次用 P0 的失败阻止位置编码实验。

每个 index 候选 P0–P8 相对 A2 满足任一条件即合格：

1. 八项 MSE/MAE 比值均值 `<1`，最差 `≤1.01`；或
2. 至少 3/4 setting 双指标改善，其余最大回退 `≤0.5%`。

合格者依次按平均比值、最差比值、参数量和运行时间冻结。P9 calendar 使用同一门槛但单列，不与纯 index PE 混排。若无候选合格，则停止，不读取 test。

### 2.4 Stage B：正式确认

只有 Stage A 合格后才运行。冻结候选与 A2 在上述四个 setting 上进行 seeds 2021/2022/2023、full train、best-validation checkpoint、一次性 test；同时报告固定 Golden。正式有效要求至少 3/4 setting 的三 seed 平均 MSE/MAE 同时优于 A2，剩余回退不超过 0.5%，且宏平均改善为正。Golden 结论逐 setting 检查三个 seed 全胜和 `mean+sample_std < Golden`，不得以 validation 结果声明超过 Golden。

## 3. 实现方式和实验结果

### 3.1 Stage 0：实现与测试

| 检查 | 结果 |
|---|---|
| old ICPT / RCRF flag-off 完全不变 | 通过；旧 decoder 仍是默认路径，新增 full-horizon 路径仅由新 preset 开启 |
| P0–P9 shape、finite forward/backward | 通过；全部候选输出形状正确、数值有限且梯度可回传 |
| 零初始化严格等于 last-value persistence | 通过；full-horizon head 初始增量为零 |
| calendar 不读取未来 timestamp/target | 通过；只消费 encoder/history timestamp，改变未来 mark 不改变输出 |
| 参数量倍率与 full-horizon head 维度 | head 均为 `720→H`；residual head 相对 NLinear 在 H=96/192/336/720 时为 1.0807×/1.0403×/1.0231×/1.0108× |
| ETTm2 5% / 1 epoch smoke | 通过；none validation 0.15927/0.28294，calendar 0.16082/0.28479，均未构造 test loader |

完整回归测试共 **146 项全部通过**。Stage A 的 48 条命令也全部完成，无 OOM、无缺失 run。

### 3.2 Stage A：validation-only 广筛结果

下表均为单 seed、30% 训练数据、最多 8 epoch 的 validation MSE/MAE；不是正式 test 结果。

| 候选 | ETTh2-720 | ETTm2-96 | Electricity-336 | Weather-336 | 8 项均值/最差比 | 合格/排名 |
|---|---|---|---|---|---|---|
| A2 NLinear | 0.63206/0.55913 | **0.11817/0.23265** | **0.13936/0.23060** | **0.54069**/0.36654 | 1.00000/1.00000 | baseline |
| C0 cycle anchor | 0.58392/0.54703 | 0.12656/0.24286 | 0.14337/**0.23053** | 0.57538/0.37380 | 1.01619/1.07099 | control |
| P0 none | 0.57968/0.54782 | 0.12319/0.23788 | 0.14225/0.23464 | 0.55003/0.36122 | 1.00035/1.04241 | 不合格 |
| P1 sin/cos | 0.57847/0.54716 | 0.12312/0.23782 | 0.14220/0.23455 | 0.54956/0.36101 | 0.99960/1.04188 | 不合格 |
| P2 learned abs | **0.57205/0.54216** | 0.12328/0.23852 | **0.14168/0.23319** | 0.56207/0.36192 | 0.99975/1.04320 | 不合格 |
| P3 Time2Vec | 0.57401/0.54476 | 0.12372/0.23866 | 0.14262/0.23579 | 0.55522/0.36159 | 1.00181/1.04694 | 不合格 |
| P4 RoPE | 0.58005/0.54802 | 0.12309/0.23773 | 0.14236/0.23477 | 0.55942/0.36314 | 1.00328/1.04159 | 不合格 |
| P5 relative | 0.57967/0.54782 | 0.12320/0.23789 | 0.14224/0.23463 | 0.54988/0.36116 | 1.00030/1.04254 | 不合格 |
| P6 ALiBi | 0.58010/0.54811 | **0.12308**/0.23779 | 0.14225/0.23469 | 0.55239/0.36203 | 1.00118/1.04147 | 不合格 |
| P7 LFF | 0.57967/0.54782 | 0.12316/0.23785 | 0.14221/0.23461 | 0.55008/0.36115 | 1.00024/1.04221 | 不合格 |
| P8 sin/cos+relative | 0.57847/0.54716 | 0.12313/0.23783 | 0.14219/0.23455 | 0.54947/**0.36094** | **0.99954**/1.04191 | 不合格 |
| P9 calendar（单列） | 0.57614/0.54619 | 0.12318/0.23780 | 0.14228/0.23521 | 0.55665/0.36491 | 1.00236/1.04236 | 不合格 |

关键观察：

- 所有 full-horizon 候选都只在 ETTh2-720 上同时改善 MSE/MAE；ETTm2 两项均退化约 2.2%–4.7%，Electricity 两项均退化约 1.1%–2.3%，Weather 则是 MSE 退化、MAE 小幅改善。因此没有任何候选达到“至少 3/4 setting 双指标改善”的条件。
- P8 的八项平均比值最低，为 **0.999544**，即相对 A2 宏平均仅好约 **0.046%**；但其最差项仍回退 **4.191%**。它相对无 PE 的 P0，八项平均只改善约 **0.082%**，说明位置编码的独立贡献很小。
- P2 在 ETTh2 和 Electricity 最好，但在 Weather 的 MSE 相对 P0 反而退化 2.19%；P8 在 Weather 最好，却没有解决 ETTm2/Electricity 的退化。calendar 也不合格。没有证据支持某一种 PE 能跨数据集稳定提高 ICPT。
- last-value anchor 整体优于 cycle anchor：P0 的八项平均比为 1.00035，C0 为 1.01619；但 Electricity 的 cycle anchor MAE 略好，说明 anchor 不是所有数据集退化的唯一原因。

与上一轮 future-query decoder ICPT-none 相比，新 P0 的变化如下。旧结果只保留到 5 位小数，因此改善率为近似值：

| Setting | 旧 decoder ICPT | 新 full-horizon P0 | 新 head 相对旧 head |
|---|---:|---:|---:|
| ETTh2-720 | 0.60690/0.54418 | 0.57968/0.54782 | MSE 改善约 4.5%，MAE 退化约 0.7% |
| ETTm2-96 | 0.15099/0.27222 | 0.12319/0.23788 | MSE/MAE 改善约 18.4%/12.6% |
| Electricity-336 | 0.16461/0.24755 | 0.14225/0.23464 | MSE/MAE 改善约 13.6%/5.2% |
| Weather-336 | 0.68911/0.43357 | 0.55003/0.36122 | MSE/MAE 改善约 20.2%/16.7% |

这说明上一轮 ICPT 的主要问题确实包括 decoder、容量与锚点设计；改成 NLinear 同形状的全 horizon head 后，三个原本严重退化的数据集都明显恢复。但恢复仍不足以稳定超过 NLinear，位置编码也没有补上剩余差距。

### 3.3 Stage B：按门槛停止

| Setting | Golden MSE/MAE | A2 mean±std | 冻结候选 mean±std | 相对 A2 | 稳定超过 Golden |
|---|---|---|---|---|---|
| ETTh2-720 | 0.402 / 0.436 | 未运行 | 无冻结候选 | 不适用 | 不可判断 |
| ETTm2-96 | 0.163 / 0.256 | 未运行 | 无冻结候选 | 不适用 | 不可判断 |
| Electricity-336 | 0.165 / 0.257 | 未运行 | 无冻结候选 | 不适用 | 不可判断 |
| Weather-336 | 0.242 / 0.278 | 未运行 | 无冻结候选 | 不适用 | 不可判断 |

`freeze_record.json` 记录 `frozen_index_candidate=null`、`calendar_eligible=false`、`test_read_before_freeze=false`、`screen_passed=false`。因此没有运行三 seed full-train，也没有读取 test；Stage A 的 validation 数值不能与固定 Golden test 数值直接比较。

## 4. 最终结论

本轮结论是：**PatchTST 式 full-horizon head 是对旧 ICPT 的有效结构修正，但 ICPT 仍不能稳定替代 NLinear；位置编码没有成为决定性增益来源。** ETTh2-720 的强正向信号说明跨周期 token 在长 horizon 上可能有价值，但 ETTm2、Electricity 和 Weather 的不一致结果使任何“稳定超过 NLinear/Golden”的结论都不成立。

因此保留当前 RCRF-NLinear 默认方案，不冻结任何 ICPT PE，也不启动正式 test。若继续研究，优先级不应是继续枚举 PE，而应检查周期长度是否与数据真实周期匹配、共享通道模型是否损失变量特异性，以及 full-horizon Transformer 的优化/正则化是否适配短中 horizon；这些都应建立新的预注册实验，不能从本轮 validation 反复调参。
