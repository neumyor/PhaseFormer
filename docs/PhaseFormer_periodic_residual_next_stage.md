# PhaseFormer 周期互补残差下一阶段实验

> 状态：**实验方案已预注册，代码实现与 Stage 0 测试待完成；尚未产生本轮训练、validation 或 test 结果。** 本轮不会用少量 setting 提前淘汰 ICPT，所有预注册模型都必须完成 12 个 setting 的三 seed 正式测试。

## 1. 实验要验证的设想

已有 `RCRF + NLinear + LFF` 在 ETTh2-720 和 ETTm2-96 上取得当前最佳均值，但 LFF 相对 NLinear 的增益只有约 0.05%–0.16%。其主要限制是：周期检索只由全局位置决定，不能按样本和通道判断周期残差是否真实存在；它直接检索原始中心化历史，与 PhaseFormer 和 NLinear 均有功能重叠；单一 `P=24` 也不足以表示多个时间尺度。

此前 ICPT 只在少量、间隔较大的 setting 上筛选，且 validation 结果与后来补充的正式 test 排名并不完全一致，因此现有证据只能否定当时 setting 上的默认替换，不能概括 ICPT 在短、中 horizon 上的表现。本轮在输入 720、输出 96/192 上覆盖 ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity，重新比较 ICPT 与三个更强调“相位—周期互补”的残差方案。

核心分工固定为：PhaseFormer 学习同相位跨周期的主体形状；NLinear 保留对近期漂移和非周期变化的直接映射；新增周期模块只建模 PhaseFormer/NLinear 尚未充分利用的周期证据；RCRF 根据原始相位可靠性完成外层融合。

## 2. 实验的整体计划

### 2.1 模型与对照

| ID | preset | 结构与作用 |
|---|---|---|
| A0 | `original` | matched 原始 PhaseFormer，诊断训练协议 |
| A1 | `gold_combo_reliability_s2` | `RCRF + NLinear` 直接基线 |
| A2 | `rcrf_pe_lff` | 当前最佳统一方案，主要比较对象 |
| I0 | `rcrf_icpt_none` | 原始 future-query decoder ICPT，无 PE |
| I1 | `rcrf_icpt_horizon_none` | ordered full-horizon ICPT，无 PE |
| D1 | `rcrf_phase_error_memory` | 相位模板误差的内容条件化周期记忆 + NLinear |
| D2 | `rcrf_dual_reliability_lff` | 相位可靠性控制外层融合，残差周期可靠性控制 NLinear/LFF 内层融合 |
| D3 | `rcrf_multiperiod` | `12/24/48/96` 多周期检索库按样本、通道自相关路由 + NLinear |

I0/I1 同时保留，是为了把“ICPT 本身”和“decoder/head 设计”分开。ICPT 本轮不搜索 PE，避免再次把 backbone、head 和 PE 混成一个不可归因的比较。

### 2.2 三个新方向的固定实现

#### D1：Phase-Error Periodic Memory

将完整历史按 `P=24` 划分成周期，减去每个相位跨周期均值，得到无未来泄漏的 phase-template error。用最近误差周期作为 query、较早误差周期作为 key/value，按内容相似度和周期间距检索一个未来误差周期；以零初始化、可正可负的 horizon gate 将该修正加到未改动的 NLinear 上。它不声称获得完整 PhaseFormer backcast，实验名称和结论均使用“phase-template error”这一准确口径。

#### D2：Dual-Reliability LFF

保留 LFF 的位置核，但不再使用全局 `beta[h]` 直接融合。外层仍用原始 RCRF 相位可靠性 `r_phase` 决定 PhaseFormer 与 residual candidate 的比例；内层根据相邻周期中 phase-template error 的一致性得到 `r_periodic`，逐样本、通道、horizon 决定 NLinear 与 LFF periodic copy 的比例。相位不可靠但周期误差也不可靠时，应回退 NLinear，而不是盲目增加 LFF。

#### D3：Adaptive Multi-Period Residual Bank

固定候选周期 `12/24/48/96`，每个周期使用“同余位置 + 距离衰减”产生轻量检索预测；根据每个样本、通道在对应 lag 的自相关进行 soft routing。聚合的周期修正通过零初始化 horizon gate 加到完整 NLinear 输出上，因此 flag-on 初始值严格等于 NLinear，且不会因周期候选错误而破坏 warm start。

三个方向都使用共享超参数，不按数据集或 horizon 选择结构。参考机制包括 TimesNet 的多周期视图、SparseTSF 的 cross-period sparse forecasting、CycleNet 的显式周期残差以及 TQNet 的周期 query；本轮的新变量是它们与 PhaseFormer 相位证据的职责分离，而不是复刻完整 backbone。

### 2.3 数据矩阵与正式协议

- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity。
- setting：每个数据集 horizon 96、192，共 12 个；lookback 固定 720，主周期固定 24。
- 每个 A0–D3 均使用 full train、seeds 2021/2022/2023、最低 validation loss checkpoint，然后一次性读取 test；不因中途 validation 或 test 排名删除任何预注册模型。
- loss、learning rate、batch、epoch、patience、数据划分和 normalization 使用 `phaseformer_presets.py` 中各 dataset×horizon 的 matched 配置；同一 setting 的全部模型必须一致。
- 共 `12 settings × 8 modes × 3 seeds = 288` 个正式 run。runner 必须支持逐 setting resume，失败/OOM 只允许以相同 baseline/candidate batch 成对重跑。
- 主指标为 MSE/MAE，同时记录参数量、训练时间、峰值显存；每个 setting 报告三 seed mean±sample std。
- 固定 Golden 只取 `docs/PhaseFormer_gold_standard.md` 对应的 12 个结果；A0 matched rerun 不替代 Golden。

### 2.4 选择与结论门槛

主要比较为 D1–D3、I0/I1 相对 A2。统一方案只有同时满足下列条件才可替代 A2：

1. 12 个 setting 中至少 8 个的平均 MSE/MAE 同时改善；
2. 24 个平均指标比值的宏平均 `<0.998`，即至少约 0.2% 的整体改善；
3. 任一 setting 的任一平均指标相对 A2 回退不超过 0.5%；
4. 若改善幅度小于跨 seed 波动，标记为“数值持平”，优先保留更简单模型。

逐 setting 声明“稳定超过 Golden”必须同时满足：三个 seed 的 MSE/MAE 都低于 Golden，且 `mean + sample_std < Golden`。ICPT 即使未达到统一替代门槛，也要报告短/中 horizon 的完整分布，不能用先前长 horizon 结论覆盖本轮证据。

### 2.5 test 暴露声明

本轮三个方向是在已知 ETTh2-720、ETTm2-96 test 结果后提出，I0/I1 也已有部分 test 暴露，因此本轮不是完全盲测。为限制进一步 test-set selection，本文件冻结后不得根据单个 test 结果修改候选、周期列表、门控初值或训练超参数；若之后修改，必须建立新实验编号并保留本轮完整结果。

## 3. 实现方式和待填结果

### 3.1 Stage 0

| 检查 | 结果 |
|---|---|
| A0/A1/A2/I0/I1 flag-off 回归 | 待填 |
| D1/D2/D3 96/192 shape 与 finite forward/backward | 待填 |
| D1/D3 零门控严格恢复 NLinear warm start | 待填 |
| D1 attention、D2 双可靠性、D3 周期权重归一化 | 待填 |
| D2 refactor 前后原 `rcrf_pe_lff` 输出一致 | 待填 |
| 6 数据集 preset 和 288-run dry-run 清单 | 待填 |
| ETTm2 与 Weather 5%/1 epoch smoke | 待填，本轮代码阶段不要求正式训练 |

### 3.2 三 seed 正式 test 表

每格填写 `MSE mean±std / MAE mean±std`。

| Setting | Golden | A0 | A1 | A2 | I0 | I1 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1-96 | 0.359/0.382 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTh1-192 | 0.397/0.404 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTh2-96 | 0.275/0.338 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTh2-192 | 0.341/0.376 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm1-96 | 0.293/0.344 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm1-192 | 0.323/0.361 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-96 | 0.163/0.256 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-192 | 0.219/0.293 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Weather-96 | 0.148/0.195 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Weather-192 | 0.193/0.237 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Electricity-96 | 0.129/0.221 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Electricity-192 | 0.148/0.238 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 3.3 汇总与机制诊断

| 模型 | 24 指标宏平均比/A2 | 双指标改善 setting | 最差回退 | 稳定超过 Golden 数 | 参数/时间/显存 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| A0 | 待填 | 待填 | 待填 | 待填 | 待填 | matched control |
| A1 | 待填 | 待填 | 待填 | 待填 | 待填 | direct baseline |
| A2 | 1.000 | 0 | 0 | 待填 | 待填 | incumbent |
| I0 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| I1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| D1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| D2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| D3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

另外必须报告 D1 内容检索熵与 correction gate、D2 `r_phase/r_periodic/alpha/rho` 分布、D3 每个候选周期的选择占比，并检查这些量是否随样本变化而非塌缩成常数。机制量只能用于解释结果，不能在读取 test 后反向选择新配置。

## 4. 最终结论

待 288 个预注册正式 run 全部完成后填写。结论必须分别回答：ICPT 在 96/192 上是否仍系统性弱于 NLinear；三个新方向是否带来超过 LFF 边际增益的实质改善；改善是否跨数据集稳定；以及是否满足逐 setting 的 Golden 稳定性标准。
