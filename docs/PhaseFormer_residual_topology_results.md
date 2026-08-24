# PhaseFormer 残差通路拓扑实验结果

> 计划锚点：`docs/PhaseFormer_residual_topology_plan.md`  
> 状态：执行中  
> 建立日期：2026-08-24

## 0. 当前状态

- 实验拓扑与公平性约束已冻结。
- Stage A、Stage B 和样本级误差分析尚未回填。
- 在完整证据落盘前，不更新 `_LATEST_POLICY`，不声明新最优模型。

## 1. 实现验证

待回填：feature flag、shape、零初始化等价性、R3/R4 单层等价性、测试结果与 commit。

## 2. Stage A 验证集筛选

待回填：全部 4 settings × 6 modes 的 val MSE/MAE、参数量、耗时和选择轨迹。

## 3. Stage B 全预算确认

待回填：matched baseline、候选、相对 matched rerun 与相对金标准结果。

## 4. 客观误差分析

待回填：误差分布、horizon 分段、程序化案例和重复可测模式。

## 5. 结论与决策

待回填：输出凸融合、输出加法、latent 长跳连、逐层注入和混合拓扑的保留/淘汰结论。
