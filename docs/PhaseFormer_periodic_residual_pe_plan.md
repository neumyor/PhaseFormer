# PhaseFormer 周期位置编码残差实验计划

## 1. 研究问题

当前 RCRF 将增强后的 PhaseFormer 相位分支与 NLinear 式残差分支融合。已有 ETTm2
样本分析显示，RCRF 的显著改善窗口具有更高的 lag-24 自相关（0.0935 vs 0.0438），
但差异较弱，且可能混有趋势影响。因此本轮检验一个更严格的假设：

> NLinear 负责按时间顺序外推近期轨迹；若显式告诉它“历史位置与未来位置在周期中的
> 对应关系”，则残差分支应更容易复用周期结构，并在多个数据集上比当前 RCRF 更稳定。

直接把位置向量加到标量序列后再接一个线性层，通常只会被吸收到线性权重或偏置中，
不能证明模型真的利用了周期。因此本轮统一使用“位置相似度生成历史—未来周期匹配核”，
保持其余 PhaseFormer、RCRF 门控和训练协议不变。

## 2. 调研依据与候选

- [ST-Informer](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289464/) 使用以 1000 为底的固定
  正余弦位置编码，并强调时序位置及周期信息。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
  给出以 10000 为底的多频率固定正余弦编码。
- [Traffic Transformer](https://par.nsf.gov/servlets/purl/10191796) 将连续位置、日周期和周周期
  位置共同编码，并建议以位置相似度修正时间点之间的匹配强度。
- [Time2Vec](https://arxiv.org/abs/1907.05321) 用可学习频率和相位的周期激活表示时间。
- [Learnable Fourier Features](https://proceedings.neurips.cc/paper_files/paper/2021/file/84c2d4860a0fc27bcf854c444fb8b400-Paper.pdf)
  通过成对的可学习 Fourier 特征得到依赖相对位置差的相似度。
- [RoFormer](https://arxiv.org/abs/2104.09864) 的 RoPE 依赖 query/key 旋转。本轮不单列 RoPE：
  NLinear 没有注意力 query/key，增加 RoPE 必须同时增加注意力层，会破坏“只比较位置编码”的归因。

统一比较以下七种编码，所有候选使用相同周期匹配头、温度、衰减和融合初始化：

| 配置 | 位置表示 | 主要归纳偏置 |
|---|---|---|
| `rcrf_pe_st` | ST-Informer 固定多频正余弦，base=1000 | 连续绝对位置与多尺度距离 |
| `rcrf_pe_cycle` | 周期 `P` 的单一 sin/cos | 平滑的同周期相位匹配 |
| `rcrf_pe_harmonic` | `P` 上 1–4 阶固定谐波 | 更尖锐的同相位匹配 |
| `rcrf_pe_traffic` | 连续位置编码 + 周期谐波编码 | 连续性与周期性联合匹配 |
| `rcrf_pe_time2vec` | 线性时间项 + 可学习正弦频率/相位 | 数据驱动的绝对时间与周期 |
| `rcrf_pe_lff` | 成对可学习 Fourier 频率 | 可学习、平移不变的相对周期 |
| `rcrf_pe_calendar` | 日内、周内、月内循环时间戳 | 真实日历周期，而非仅依赖 `P` |

## 3. 统一结构

令中心化历史为 `z_t=x_t-x_last`，普通 NLinear 输出为 `d_N=Wz`。位置编码 `e(t)`
先生成历史到未来的周期匹配权重：

```text
A[h,t] = softmax_t(cos(e(L+h), e(t))/temperature
                   - cycle_decay * (L+h-t)/P)
d_P[h] = sum_t A[h,t] * z_t
beta[h] = sigmoid(b[h])
y_res[h] = x_last + (1-beta[h])*d_N[h] + beta[h]*d_P[h]
```

`beta` 按预测步学习、跨样本和通道共享，初始化为 0.1；`temperature=0.1`，
`cycle_decay=0.1`。外层 RCRF 公式及其相位可靠度 `r`、残差权重 `alpha` 保持不变。
因此唯一实验变量是残差分支的位置表示。报告必须同时给出 `beta`、匹配核熵和主要 lag。

## 4. 实验协议

固定设置沿用已验证的 Golden-combo 协议：

| Setting | Golden MSE/MAE | loss | lr | batch | lookback / period |
|---|---:|---|---:|---:|---:|
| ETTh2-720 | 0.402 / 0.436 | Huber | 1e-3 | 256 | 720 / 24 |
| ETTm2-96 | 0.163 / 0.256 | MAE | 3e-4 | 256 | 720 / 24 |
| Electricity-336 | 0.165 / 0.257 | MAE | 3e-4 | 64 | 720 / 24 |

Golden 固定取自 `docs/PhaseFormer_gold_standard.md`；matched rerun 不替代 Golden。

### Stage A：validation-only 广筛

- 当前 `gold_combo_reliability_s2` 与七个 PE 候选；3 settings，seed 2021。
- 30% 训练数据，最多 8 epoch；最低 validation loss checkpoint；不创建 test loader。
- 对每个候选计算六项比值（3 settings × MSE/MAE）相对当前 RCRF 的均值。
- 候选资格：均值比值 `<1`，且任一 setting/metric 相对 RCRF 回退不超过 1%。
- 在合格候选中冻结均值比值最低者；并列时依次选择最差比值更低、参数更少者。
- 若无候选合格，则结论为位置编码当前无稳定验证收益，不读取测试集继续调参。

### Stage B：三 seed 正式确认

- 只运行当前 RCRF 与冻结候选；3 settings × seeds 2021/2022/2023；全量训练；
  validation early stopping，恢复 best checkpoint 后测试。
- 每个 setting 报告 MSE/MAE mean±sample std、相对当前 RCRF 和固定 Golden 的改善率。
- “稳定超过 Golden”：三个 seed 双指标均低于 Golden，且双指标均满足 `mean+std<Golden`。
- “位置编码跨数据集有效”：至少 2/3 settings 相对当前 RCRF 的 MSE、MAE 均值同时改善，
  剩余 setting 任一指标回退不超过 0.5%；同时至少 2/3 settings 稳定超过 Golden。

## 5. 样本级验证与审计

最终比较以当前 RCRF 为 baseline、冻结 PE 候选为 candidate，按 `sample×channel` 导出误差。
每个 setting/seed 程序化选择 baseline 高误差、candidate 显著改善和显著退化案例，不人工挑选。

重点验证：

1. 改善样本是否具有更高 lag-P 自相关、周期频带能量或更稳定的同相位重复；
2. 改善是否集中在较远预测步，还是仅复制最近周期改善短期；
3. `beta`、匹配核主要 lag 与 RCRF `alpha/r` 是否表现出真实活性；
4. 退化样本是否对应周期漂移、未来转向、异常波动或错误日历匹配。

审计目录固定为 `research_runs/periodic_residual_pe_v1/`，严格保留六个文件和
`figures/`；原始 checkpoint 与日志只存放在忽略目录，完成审计后不进入正式包。
