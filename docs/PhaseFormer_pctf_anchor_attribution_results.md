# PCTF 锚点归因复测结果

## 结论先行

本轮 84 个 validation-only 运行已完成（12 个 A2 锚点 + 72 个候选，6 个数据设置、2 个 seed）。冻结诊断确认有效：冻结候选的内部锚点与 A2 相同（数值误差小于 `1e-6`）。在候选中，`pctf_anchor_repair_full` 的宏平均 MSE 比匹配 A2 低约 **0.64%**，但最差设置仍退化 **1.25%**，因此未达到预设的“稳定改进”门槛，也未授权读取 test。

## 实验协议与可审计性

- 设置：ETTh2 H96/H192、ETTm2 H96/H192、Weather H192、Electricity H96。
- 每个设置使用 seed 2021、2022；训练 30% 数据，最多 12 epoch，Huber 损失。
- 所有比较均为 validation；脚本拒绝包含 test 指标的结果，不能据此宣称超过 Golden 或正式 PhaseFormer test。
- 硬件：RTX 4090；PyTorch 2.7.1+cu126；协议版本见 `attribution_decision.json`。

## 聚合结果（相对匹配 A2，<1 表示更好）

|候选|宏 MSE 比值|最差 MSE 比值|同时改善(MSE/MAE)|内部锚点宏比值|更新 RMS|
|---|---:|---:|---:|---:|---:|
|anchor_mlp|1.0026|1.0325|3/12|1.0202|0.0442|
|frozen_absolute|0.9972|0.99996|12/12|1.0000|0.0204|
|frozen_residual|0.9970|0.99984|12/12|1.0000|0.0211|
|joint_residual|0.9940|1.0133|8/12|0.9957|0.0210|
|joint_marginal|0.9938|1.0126|8/12|0.9958|0.0182|
|repair_full|**0.9936**|1.0125|8/12|0.9957|0.0190|

冻结 residual 目标略优于冻结 absolute（宏 MSE 0.99699 vs 0.99723）。三种可训练 repair 均有约 0.6% 的宏观收益，但收益集中在部分设置；ETTh2 H192 seed 2022 是主要退化来源。

## 假设判定

|假设|结果|依据|
|---|---|---|
|H1 冻结是有效控制|支持|冻结内部锚点比值的最大偏差约 `2.4e-7`，属于浮点/归约误差。|
|H2 residual 目标优于 absolute|支持|冻结 residual 宏 MSE 比值更低。|
|H3 joint residual 能限制漂移|不支持|最差内部锚点比值约 1.0136，超过 1.01。|
|H4 marginal gate 优于 joint residual|支持|宏 MSE/MAE 比值均略低。|
|H5 full repair 通过预正式门槛|不支持|宏比值虽低于 0.998，但最差比值 1.0125>1.01，且仅 8/12 行同时改善（门槛要求至少 8 行，仍被最差约束否决）。|

## 解释与下一步

结果支持“残差目标和边界修复比自由 MLP 更稳”的方向，但也显示当前 repair 仍受长预测窗口和 seed 敏感性影响。下一轮应优先针对 ETTh2 H192 的漂移进行约束或按 horizon 自适应缩放，并在独立验证集确认后，才进行预注册的多 seed/test 正式实验。本诊断不包含 test 指标，不能替代 Golden 对比。

原始明细：`research_runs/pctf_anchor_attribution_v3/attribution_details.csv`；聚合：`attribution_aggregates.csv`；机器判定：`attribution_decision.json`。
