# 周期位置编码残差实验：完整闭环

## 1. 实验要验证的设想

ETTm2 样本分析表明，RCRF 在部分具有周期重复结构的窗口上收益更明显。因此设想：若在 NLinear 残差支路中显式建立历史位置到未来位置的周期匹配核，残差支路可以更稳定地复用周期结构，并改善当前 RCRF。

## 2. 实验的整体计划

固定 PhaseFormer 主干、RCRF 门控、数据划分和训练协议，只改变残差支路的位置表示。Stage A 使用 3 个 setting、seed 2021、30% 数据和最多 8 epoch，仅看 validation，从 7 种位置编码中冻结候选；Stage B 对当前 RCRF 与冻结候选进行 3 个 seed 的 full-budget 测试。成功标准是至少 2/3 setting 的双指标稳定超过 Golden，剩余 setting 回退不超过 0.5%。同时导出 sample×channel 误差、门控、匹配核和 bad cases。

## 3. 每个实验的实现方式和结果

统一结构为 `A[h,t]=softmax(cos(e_future,e_history)/temperature-cycle_decay*distance/P)`，用匹配后的历史残差与 NLinear 输出按预测步权重 `beta[h]` 融合；`temperature=0.1`、`cycle_decay=0.1`、`beta` 初值 0.1。候选包括 ST-Informer、单周期、谐波、Traffic hybrid、Time2Vec、LFF 和 calendar。

Stage A 共 24/24 个 validation-only run。`rcrf_pe_lff` 胜出，六项比值均值 `0.9995488`，最差比值 `1.0003643`，测试集读取发生在冻结之后。

Stage B（ETTh2-720、ETTm2-96、Electricity-336；3 seed）结果如下：

| setting | RCRF MSE/MAE | RCRF+LFF MSE/MAE | 相对 RCRF |
|---|---:|---:|---:|
| ETTh2-720 | 0.394228 / 0.429443 | **0.393591 / 0.428967** | +0.162% / +0.111% |
| ETTm2-96 | 0.159762 / 0.245333 | **0.159678 / 0.245196** | +0.052% / +0.056% |
| Electricity-336 | **0.164114 / 0.254625** | 0.164260 / 0.254876 | −0.089% / −0.099% |

LFF 在 ETTh2、ETTm2 稳定超过 Golden；Electricity 平均回退约 0.1%，未达到稳定提升门槛。样本级分析显示改善组的 lag-24 自相关仅略高于退化组（0.5785 vs 0.5513），说明周期检索假设得到弱支持但不是因果证明；`beta` 保持活跃，约为 0.10–0.185。

## 4. 最终结论

可学习 Fourier 位置匹配是小幅、条件性的残差增强：在 ETT 系列有效，在 Electricity 上不稳定，不能替换所有数据集的默认残差头。详细配置、复现命令和审计包见原始记录及 `research_runs/periodic_residual_pe_v1/`；本文件合并了原计划与结果，原文件保留作审计来源。
