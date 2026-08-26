# Pure Phase Modeling 实验：完整闭环

## 1. 实验要验证的设想

当前 PhaseFormer 的 phase token 可能缺少多尺度、动态变形和周期几何表达；若只依赖纯相位预测，不使用 residual，也应能通过增强 phase representation、phase evolution、phase interaction 和 decoder 提升弱周期序列预测。

## 2. 实验的整体计划

按四阶段逐步验证：多尺度 phase representation、动态 phase deformation、geometry-aware interaction、pure-phase decoder；每阶段先低成本筛选，再做完整消融和分析可视化。最终比较 `original`、各单模块、组合模型及完整消融，检查跨数据集/预测长度稳定性。

## 3. 每个实验的实现方式和结果

实现包括多尺度 phase token、动态形变/轨迹模块、圆周距离 attention bias 及纯 phase decoder。各阶段均保留可关闭开关，并记录 phase trajectory、deformation、frequency consistency、smoothness 等内部量。结果反馈显示：部分模块在个别 ETTh2/ETTm2 setting 有小幅收益，但组合模型没有形成稳定、跨 setting 的双指标提升；若干 run 因训练中途停止而不纳入最终效果结论。

完整阶段表、消融矩阵、缺失 run 和复现实验路径见 `docs/PhaseFormer_pure_phase_results.md`；原始预注册方案见 `docs/PhaseFormer_pure_phase_plan.md`。

## 4. 最终结论

纯 phase 路线提供了有解释性的表示增强，但当前证据不足以替代带输出残差的方案。后续研究应把有效的相位修正作为受控组件，与输出端 residual fusion 做严格 matched 消融，而不继续无条件叠加 phase 模块。
