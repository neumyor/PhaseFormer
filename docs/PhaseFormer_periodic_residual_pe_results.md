# PhaseFormer 周期位置编码残差实验结果

## 结论

在 NLinear 残差支路中加入“可学习 Fourier 位置匹配（LFF）”后，ETTh2-720 每个 seed
的 MSE/MAE 都改善；ETTm2-96 每个 seed 的 MAE 都改善，三 seed 平均 MSE/MAE 也改善。
Electricity-336 平均回退约 0.1%，因此该机制是小幅、条件性的周期增强，不是三数据集全胜。

相对固定 Golden，LFF 候选在 ETTh2、ETTm2 满足预注册的“全 seed 双指标低于 Golden，
且 mean+std 低于 Golden”；全部 18 个 dataset×seed×metric 组合中 17 个优于 Golden，
唯一例外是 Electricity seed 2022 MSE `0.165042`，略高于 Golden `0.165`。

## 模型设计

当前 RCRF 保留 PhaseFormer 作为相位主干，NLinear 作为补偿近期轨迹的残差支路。新模块
不是把位置向量直接加到标量输入，而是让位置编码显式决定 future→history 周期匹配：

```text
A[h,t] = softmax(cos(e(L+h), e(t))/temperature - cycle_decay*(L+h-t)/P)
dP[h]  = sum_t A[h,t] * (x[t]-x[last])
y_res  = x[last] + (1-beta[h])*d_NLinear[h] + beta[h]*dP[h]
```

外层 RCRF 的可靠度 `r` 和残差权重 `alpha` 不变，因而本轮只检验“残差分支如何显式使用
周期位置”。实现支持 ST-Informer、单周期、固定谐波、Traffic hybrid、Time2Vec、LFF 和
calendar 七种编码；RoPE 需要 query/key，会同时引入注意力结构，故没有混入受控比较。

## 实验结果

Stage A 使用 30% 数据、最多 8 epoch、seed 2021，只读取 validation。LFF 以六项
MSE/MAE 比值均值 `0.9995488`、最差比值 `1.0003643` 胜出并在测试前冻结；Time2Vec
排名第二。信号只有约 0.045%，正式实验预期本来就是小幅差异。

Stage B 使用全量训练、best-validation checkpoint、seeds 2021/2022/2023：

| Setting | 当前 RCRF mean MSE/MAE | RCRF+LFF mean MSE/MAE | LFF 相对 RCRF | LFF 相对 Golden | 稳定超 Golden |
|---|---:|---:|---:|---:|---|
| ETTh2-720 | 0.394228 / 0.429443 | **0.393591 / 0.428967** | +0.162% / +0.111% | +2.09% / +1.61% | 是 |
| ETTm2-96 | 0.159762 / 0.245333 | **0.159678 / 0.245196** | +0.052% / +0.056% | +2.04% / +4.22% | 是 |
| Electricity-336 | **0.164114 / 0.254625** | 0.164260 / 0.254876 | −0.089% / −0.099% | +0.45% / +0.83% | 否 |

预注册的“跨数据集有效”标准通过：2/3 settings 相对当前 RCRF 的平均双指标同时改善，
剩余 Electricity 的回退小于 0.5%，且 2/3 settings 稳定超过 Golden。但当前证据不足以
把 LFF 替换成所有数据集的默认 RCRF 残差头。

## 样本与内部量分析

- 全部 9 个 setting 共导出 `5,028,081` 条 `sample×channel` 误差，程序化选择 270 个
  baseline 高误差/显著改善/显著退化案例，没有人工挑例。
- 在 9 settings 的极端 Top-K 合并统计中，改善组相对退化组具有略高 lag-24 自相关
  （`0.5785 vs 0.5513`）和周期频带能量（`0.1446 vs 0.1432`）；差异很小，只与周期
  检索假设一致，不能作为因果证明。
- `beta` 并未失活：ETT 上约 `0.10–0.106`，Electricity 上约 `0.139–0.185`；LFF 前四个
  频率倍率多数接近初始化 `1.0`。Electricity 使用了更大的周期检索权重却没有平均收益，
  提示下一步应检验通道级/多周期异质性，而不是简单放大 `beta`。

完整审计证据位于 `research_runs/periodic_residual_pe_v1/`，包含逐 cell CSV、案例 NPZ、
44 张中文 matplotlib 图片、完整 Markdown 和只含报告/引用图片的 ZIP。原始 Stage A/Stage B
记录分别位于 `research_runs/periodic_residual_pe_screen/` 和
`research_runs/periodic_residual_pe_full/`。

## 复现与校验

```bash
MPLCONFIGDIR=/tmp/phaseformer_mpl /home/wangjing/miniconda3/bin/python \
  scripts/run_periodic_residual_pe.py --stage screen --num-workers 0
MPLCONFIGDIR=/tmp/phaseformer_mpl /home/wangjing/miniconda3/bin/python \
  scripts/run_periodic_residual_pe.py --stage freeze
MPLCONFIGDIR=/tmp/phaseformer_mpl /home/wangjing/miniconda3/bin/python \
  scripts/run_periodic_residual_pe.py --stage full --num-workers 0
MPLCONFIGDIR=/tmp/phaseformer_mpl /home/wangjing/miniconda3/bin/python \
  scripts/analyze_periodic_residual_pe.py --device cuda:0
```

本机缺少规则首选的 py310 环境，实际使用 base conda fallback（Python 3.13.5、
torch 2.7.1+cu126、RTX 4090）。18 个 checkpoint 的重放指标均在 `1e-5` 内匹配训练日志；
全部单元测试、逐 cell 汇总、setting 覆盖、Top-K、中文字体、Markdown 图片引用、目录白名单
及 ZIP 字节一致性均通过。
