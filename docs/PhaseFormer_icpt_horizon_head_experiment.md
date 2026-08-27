# ICPT Full-Horizon Head 与位置编码实验

> 状态：**方案已预注册，尚未产生本轮 validation/test 结果。** 本实验使用新实验编号，不修改或覆盖上一轮 ICPT 失败结论。

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

### 3.1 Stage 0 待填

| 检查 | 结果 |
|---|---|
| old ICPT / RCRF flag-off 完全不变 | 待填 |
| P0–P9 shape、finite forward/backward | 待填 |
| 零初始化严格等于 last-value persistence | 待填 |
| calendar 不读取未来 timestamp/target | 待填 |
| 参数量倍率与 full-horizon head 维度 | 待填 |
| ETTm2 5% / 1 epoch smoke | 待填 |

### 3.2 Stage A 待填

| 候选 | ETTh2-720 | ETTm2-96 | Electricity-336 | Weather-336 | 8 项均值/最差比 | 合格/排名 |
|---|---|---|---|---|---|---|
| A2 NLinear | 待填 | 待填 | 待填 | 待填 | 1 / 1 | baseline |
| C0 cycle anchor | 待填 | 待填 | 待填 | 待填 | 待填 | control |
| P0 none | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P1 sin/cos | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P2 learned abs | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P3 Time2Vec | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P4 RoPE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P5 relative | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P6 ALiBi | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P7 LFF | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P8 sin/cos+relative | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| P9 calendar（单列） | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 3.3 Stage B 待填

| Setting | Golden MSE/MAE | A2 mean±std | 冻结候选 mean±std | 相对 A2 | 稳定超过 Golden |
|---|---|---|---|---|---|
| ETTh2-720 | 0.402 / 0.436 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-96 | 0.163 / 0.256 | 待填 | 待填 | 待填 | 待填 |
| Electricity-336 | 0.165 / 0.257 | 待填 | 待填 | 待填 | 待填 |
| Weather-336 | 0.242 / 0.278 | 待填 | 待填 | 待填 | 待填 |

## 4. 最终结论

待 Stage A/B 按预注册门槛完成后填写。无论成功或失败，都必须明确区分相对 matched NLinear、相对固定 Golden、位置编码独立贡献和 anchor/head 结构贡献。
