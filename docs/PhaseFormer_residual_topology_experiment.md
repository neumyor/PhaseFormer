# 残差通路拓扑实验：完整闭环

## 1. 实验要验证的设想

残差应放在预测输出端还是 Transformer 的中间层，决定其是否能直接修正预测轨迹。设想是比较输入端、层内、输出端及其深度变体，找出可归因且跨 setting 稳定的拓扑。

## 2. 实验的整体计划

先用 validation-only Stage A 筛选 24 个拓扑/深度组合，再冻结候选做 Stage B full-budget matched rerun；统一数据、随机种子、损失和 checkpoint 规则，报告 MSE/MAE、资源和 sample-level bad cases。只保留相对 Golden 和 matched baseline 的可比结论。

## 3. 每个实验的实现方式和结果

R1/R2 将弱周期 residual 分别接入输入或中间层；A1/A2 进一步比较输出端残差的单层/多层变体。Stage A 以验证集筛选，Stage B 在多 setting 测试。结果显示输出端凸融合是最稳定的正向拓扑；层内和多层残差没有稳定超过输出端方案，部分 setting 退化。

核心结果与判定、24-job 明细及复核路径见 `docs/PhaseFormer_residual_topology_results.md`；正式结论是保留输出端拓扑，淘汰层内/多层作为默认结构。

## 4. 最终结论

残差拓扑的收益主要来自输出端对最终预测的直接、可控修正，而不是把残差注入更多中间层。该实验支持 RCRF 的输出融合设计，也限制了后续结构搜索范围。
