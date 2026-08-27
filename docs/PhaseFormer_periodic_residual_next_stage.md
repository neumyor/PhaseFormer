# PhaseFormer 周期互补残差下一阶段实验

> 状态：**288 个预注册正式 run 已全部完成（12 setting × 8 mode × 3 seed，lookback 720、period 24、full-train、best-val checkpoint、单次 test 读取），结果与决策已回填至 §3.2/§3.3。没有任何候选满足替换 A2 的统一门槛；I0 达到 8/12 双指标改善但被 ETTh2-96 的 +6.5% 回退挡在门外。** 本轮没有用少量 setting 提前淘汰任何预注册模型，正式汇总见 `research_runs/periodic_residual_next_stage_v1/formal_summary.csv` 与 `decision_summary.json`。

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

### 2.6 执行与汇总入口

先检查固定矩阵，应输出 `commands=36 model_runs=288`：

```bash
python scripts/run_periodic_residual_next_stage.py --stage dry-run
```

正式顺序执行并允许按 run 续跑：

```bash
python scripts/run_periodic_residual_next_stage.py --stage full --resume
```

只有 288 个 run 全部存在时，汇总器才会生成 `formal_summary.csv` 和 `decision_summary.json`；缺失或重复 run 会明确失败：

```bash
python scripts/run_periodic_residual_next_stage.py --stage summarize
```

## 3. 实现方式和待填结果

### 3.1 Stage 0

| 检查 | 结果 |
|---|---|
| A0/A1/A2/I0/I1 flag-off 回归 | 通过；全仓库 160 项测试全部成功 |
| D1/D2/D3 96/192 shape 与 finite forward/backward | 通过；完整 PhaseFormer 两个 horizon 前向有限，三个 head 在 `720→192` 上反向有限 |
| D1/D3 零门控严格恢复 NLinear warm start | 通过，逐元素精确相等 |
| D1 attention、D2 双可靠性、D3 周期权重归一化 | 通过；并验证 D1 按样本变化、D2 对重复误差提高周期权重、D3 对正弦输入选择正确周期 |
| D2 refactor 前后原 `rcrf_pe_lff` 输出一致 | 通过；component 重构与原 blend 逐元素一致，原 LFF 仍使用可学习 `beta` |
| 6 数据集 preset 和 288-run dry-run 清单 | 通过；36 个 setting×seed 命令，每条 8 个 mode，共 288 model runs |
| 正式汇总器 | 通过 synthetic complete-matrix 测试；三 seed sample std、A2 比值、Golden 稳定性和门槛判断均自动生成 |
| ETTm2 与 Weather 5%/1 epoch smoke | 未运行；本轮按用户要求只完成代码与计划，不启动训练 |

候选 residual head 的可训练参数量如下；D2 为 LFF head 与双可靠性 fusion 合计：

| Horizon | NLinear | D1 | D2 | D3 |
|---:|---:|---:|---:|---:|
| 96 | 69,216 | 69,728（1.0074×） | 69,323（1.0015×） | 69,316（1.0014×） |
| 192 | 138,432 | 139,040（1.0044×） | 138,635（1.0015×） | 138,628（1.0014×） |

实现提交为 `d1ab49e`。新增模块位于 `src/models/periodic_residual_experts.py`，preset/融合入口位于 `src/models/phaseformer_presets.py` 与 `src/models/PhaseFormer.py`，正式执行和严格汇总入口为 `scripts/run_periodic_residual_next_stage.py`。

### 3.2 三 seed 正式 test 表

每格填写 `MSE mean±std / MAE mean±std`。

| Setting | Golden | A0 | A1 | A2 | I0 | I1 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1-96 | 0.359/0.382 | 0.3614±0.0024/0.3867±0.0004 | 0.3678±0.0020/0.3971±0.0010 | 0.3663±0.0028/0.3966±0.0010 | 0.3591±0.0053/0.3890±0.0028 | 0.3737±0.0035/0.4012±0.0040 | 0.3669±0.0037/0.3958±0.0013 | 0.3650±0.0023/0.3949±0.0020 | 0.3679±0.0021/0.3971±0.0010 |
| ETTh1-192 | 0.397/0.404 | 0.4047±0.0022/0.4109±0.0014 | 0.4039±0.0015/0.4201±0.0025 | 0.4027±0.0019/0.4191±0.0020 | 0.3940±0.0051/0.4136±0.0025 | 0.4178±0.0090/0.4322±0.0054 | 0.4020±0.0016/0.4190±0.0006 | 0.4017±0.0020/0.4184±0.0019 | 0.4037±0.0014/0.4200±0.0024 |
| ETTh2-96 | 0.275/0.338 | 0.2818±0.0009/0.3434±0.0011 | 0.2722±0.0012/0.3328±0.0005 | 0.2735±0.0018/0.3332±0.0007 | 0.2912±0.0044/0.3415±0.0031 | 0.2780±0.0012/0.3379±0.0017 | 0.2745±0.0032/0.3324±0.0010 | 0.2754±0.0022/0.3342±0.0009 | 0.2722±0.0013/0.3328±0.0005 |
| ETTh2-192 | 0.341/0.376 | 0.3436±0.0009/0.3828±0.0013 | 0.3421±0.0026/0.3762±0.0021 | 0.3423±0.0024/0.3763±0.0023 | 0.3538±0.0021/0.3811±0.0019 | 0.3466±0.0020/0.3826±0.0009 | 0.3404±0.0023/0.3751±0.0004 | 0.3403±0.0029/0.3759±0.0025 | 0.3420±0.0026/0.3762±0.0020 |
| ETTm1-96 | 0.293/0.344 | 0.3024±0.0086/0.3512±0.0070 | 0.3044±0.0026/0.3511±0.0032 | 0.3038±0.0021/0.3505±0.0029 | 0.2961±0.0053/0.3462±0.0038 | 0.3059±0.0036/0.3510±0.0018 | 0.3039±0.0021/0.3502±0.0008 | 0.3044±0.0028/0.3508±0.0034 | 0.3046±0.0025/0.3511±0.0032 |
| ETTm1-192 | 0.323/0.361 | 0.3304±0.0022/0.3633±0.0017 | 0.3395±0.0023/0.3702±0.0011 | 0.3385±0.0023/0.3693±0.0004 | 0.3268±0.0013/0.3638±0.0016 | 0.3427±0.0063/0.3732±0.0004 | 0.3386±0.0030/0.3685±0.0015 | 0.3395±0.0035/0.3697±0.0012 | 0.3397±0.0023/0.3702±0.0011 |
| ETTm2-96 | 0.163/0.256 | 0.1719±0.0018/0.2627±0.0028 | 0.1608±0.0011/0.2492±0.0004 | 0.1608±0.0007/0.2490±0.0005 | 0.1657±0.0013/0.2555±0.0008 | 0.1628±0.0011/0.2515±0.0011 | 0.1602±0.0008/0.2495±0.0007 | 0.1606±0.0011/0.2492±0.0007 | 0.1608±0.0011/0.2492±0.0004 |
| ETTm2-192 | 0.219/0.293 | 0.2282±0.0005/0.2998±0.0019 | 0.2150±0.0004/0.2869±0.0008 | 0.2139±0.0009/0.2863±0.0007 | 0.2249±0.0018/0.2944±0.0010 | 0.2195±0.0010/0.2930±0.0013 | 0.2151±0.0006/0.2871±0.0006 | 0.2142±0.0009/0.2866±0.0008 | 0.2150±0.0004/0.2869±0.0008 |
| Weather-96 | 0.148/0.195 | 0.1499±0.0006/0.1968±0.0006 | 0.1501±0.0024/0.1965±0.0025 | 0.1505±0.0029/0.1973±0.0039 | 0.1437±0.0008/0.1899±0.0016 | 0.1455±0.0010/0.1898±0.0005 | 0.1484±0.0021/0.1950±0.0021 | 0.1500±0.0026/0.1967±0.0029 | 0.1501±0.0024/0.1965±0.0025 |
| Weather-192 | 0.193/0.237 | 0.1951±0.0016/0.2397±0.0019 | 0.1955±0.0017/0.2409±0.0016 | 0.1962±0.0020/0.2412±0.0022 | 0.1884±0.0004/0.2344±0.0006 | 0.1910±0.0009/0.2339±0.0005 | 0.1948±0.0019/0.2400±0.0020 | 0.1961±0.0026/0.2410±0.0028 | 0.1955±0.0017/0.2409±0.0016 |
| Electricity-96 | 0.129/0.221 | 0.1289±0.0002/0.2202±0.0001 | 0.1288±0.0007/0.2230±0.0016 | 0.1290±0.0005/0.2233±0.0014 | 0.1276±0.0002/0.2206±0.0001 | 0.1267±0.0002/0.2203±0.0004 | 0.1286±0.0002/0.2227±0.0008 | 0.1292±0.0002/0.2242±0.0009 | 0.1288±0.0007/0.2230±0.0016 |
| Electricity-192 | 0.148/0.238 | 0.1459±0.0006/0.2359±0.0004 | 0.1473±0.0027/0.2377±0.0018 | 0.1474±0.0032/0.2378±0.0023 | 0.1463±0.0006/0.2363±0.0006 | 0.1459±0.0006/0.2371±0.0002 | 0.1455±0.0011/0.2372±0.0009 | 0.1463±0.0008/0.2373±0.0006 | 0.1474±0.0028/0.2378±0.0020 |

### 3.3 汇总与机制诊断

| 模型 | 24 指标宏平均比/A2 | 双指标改善 setting | 最差回退 | 稳定超过 Golden 数 | 参数/时间 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| A0 | 1.0072 | 6 | +6.97% | 1 | 全模型 2,072 · ~9.2 min/run | matched control |
| A1 | 1.0002 | 6 | +0.52% | 3 | head 69,216 · ~8.8 min/run | direct baseline |
| A2 | 1.000 | 0 | 0.00% | 2 | head 69,320 · ~9.0 min/run | incumbent |
| I0 | 0.9969 | 8 | +6.47% | 4 | head 23,160 · ~10.8 min/run | 8/12 双指标改善但最差回退 6.5%，不替代 |
| I1 | 1.0038 | 4 | +3.74% | 4 | head 74,736 · ~10.3 min/run | 宏平均比 A2 差 0.38%，不替代 |
| D1 | 0.9973 | 6 | +0.56% | 2 | head 69,728 · ~8.8 min/run | 宏平均改善 0.27% 但双指标仅 6/12、最差回退 0.56%，不替代 |
| D2 | 0.9995 | 6 | +0.67% | 3 | head 69,224+99 · ~8.3 min/run | 宏平均改善 0.05%，双指标 6/12、最差回退 0.67%，不替代 |
| D3 | 1.0003 | 5 | +0.52% | 3 | head 69,316 · ~8.8 min/run | 宏平均比 A2 差 0.03%，不替代 |

> 峰值显存未由本轮协议（benchmark metrics.csv）记录，此列以平均单 run 训练时间代替；参数为 ETTh1-96 上各 mode 的 residual-head 参数（A0 为全模型），共享 PhaseFormer backbone 2,072 参数不随 mode 变化。

**机制诊断**（seed 2021、每个 setting 最多 48 个 test batch，加载 `best.ckpt` 前向捕获 `last_*`；数据见 `mechanism_diagnostics.csv`，仅用于解释冻结的 test 结果，未用于反向选择）：

- **D1 内容检索随样本变化、未塌缩**：attention entropy 跨 setting 为 1.46–2.20，且在同一 setting 的 batch 间 std 为 0.06–0.18（全部 12 个 setting `collapsed=False`），说明记忆确实按样本检索不同模板。但 correction gate 在 10/12 个 setting 上 `|mean| ≤ 0.06`，只在 Electricity 显著打开（h96 `0.40`、h192 `0.22`）——即模板误差记忆只在 Electricity 上实质参与预测，其余 setting 上 D1 实际退化为 A1 路径，这与 D1 的 MSE 与 A1 几乎重合一致。
- **D2 内层周期可靠性自适应、但 LFF 副本权重持续偏低**：`r_periodic` 跨 setting 为 0.53–0.75、cell 间 std 0.08–0.30（随样本变化，未塌缩）；内层 periodic gate 均值仅 0.11–0.33，始终远离 1，即 NLinear 在内层持续占主导，LFF 周期副本被系统性压低。外层 `r_phase` 与数据集相位可靠性吻合：ETTm/Weather 接近 0、ETTh 0.18–0.39、Electricity 0.76，外层 phase gate 相应为 0.15–0.87。LFF 位置核的 attention entropy 跨 batch 恒定（std=0）符合其输入无关的位置核设计。
- **D3 路由选到了数据集合适的周期、但 correction gate 几乎始终关闭**：argmax 周期占比为——ETTh1/ETTh2 以 P24 为主（0.57–0.80）、ETTm1 以 P96 为主（0.79–0.81）、ETTm2 在 P12/P96 间（0.43–0.57）、Weather 以 P12 为主（0.50–0.71）、Electricity 从 h96 的 P96（0.59）切到 h192 的 P24（0.93）；路由权重在 10/12 setting 随样本变化，Electricity 上转为近似 channel 依赖（per-sample 权重均值恒定）。但零初始化 correction gate 在除 Electricity（h96 `0.17`、h192 `0.09`）外的所有 setting 上 `|mean| ≤ 0.031`（Weather 为负），即精心路由的周期修正大部分被门控掉，D3 在效果上同样退化为 A1。

三个机制的可学习 gate 几乎一致地收敛到接近零（只在 Electricity 打开），是 D1/D2/D3 相对 A2 仅在 ±0.6% 内波动、无法跨越替换门槛的直接机制解释：新增周期模块在多数数据集上实际以 NLinear（A2 路径）输出为准，真正的改善来自 RCRF + NLinear 基线本身。

## 4. 最终结论

288 个预注册正式 run（12 setting × 8 mode × 3 seed）全部完成，逐 setting 结果见 §3.2，决策与机制诊断见 §3.3，原始逐 run 数据保留在 `research_runs/periodic_residual_next_stage_v1/`。按门槛检查，**没有任何候选满足替代 A2 的统一条件**，A2（`rcrf_pe_lff`）继续作为 incumbent。

1. **ICPT 在 96/192 上不是系统性弱于 NLinear，而是数据集相关。** `rcrf_icpt_none`（I0）是唯一达到"8/12 setting 双指标同时改善"的候选（在 ETTh1、ETTm1、Weather、Electricity 的 96/192 上全部双指标优于 A2），24 指标宏平均比 A2 为 0.9969（约整体改善 0.31%），且是 stable-below-Golden 数最多的模型（4 个：Weather 与 Electricity 的 96/192）。但它在 ETTh2-96 上 MSE 回退 +6.5%（最差回退 1.065，超过 0.5% 上限），被替换门槛挡住。`rcrf_icpt_horizon_none`（I1）整体更差（宏平均 1.0038），只在 Weather/Electricity 上优于 A2——加 horizon PE 的 ICPT 反而弱于无 PE 版本。因此先前"ICPT 系统性弱于 NLinear"的结论只在 ETTh2 上成立，不能泛化到 Weather/Electricity。
2. **三个新方向未带来超过 LFF 边际增益的实质改善。** D1/D2/D3 相对 A2 的 24 指标宏平均比分别为 0.9973 / 0.9995 / 1.0003，最佳 setting 改善 0.5%–1.4%、最差回退 0.5%–0.7%，全部落在跨 seed 波动量级内；机制诊断（§3.3）表明三个可学习 correction gate 几乎一致地收敛到接近零、只在 Electricity 打开，因此三个模块在多数数据集上实际退化为 NLinear/A2 路径。改善主要来自 RCRF + NLinear 基线本身，而非新增的周期证据建模。
3. **改善不跨数据集稳定。** 没有任何候选在所有 setting 上优于 A2：I0 的赢面集中在 Weather/Electricity/ETTh1/ETTm1，输面在 ETTh2；D1/D3 只在 Electricity 上打开 gate；D2 全 setting 都在 ±0.7% 以内。不存在可统一替换的周期互补方案。
4. **逐 setting Golden 稳定性按预注册标准判定**：A2 稳定超过 Golden 2/12（ETTm2-96/192），I0/I1 各 4/12（Weather/Electricity 的 96/192），D2/D3 3/12，其余更少。**两个数据集家族互为补充**——NLinear 族（A1/A2/D1-D3）在 ETTm2 上稳定超过 Golden，ICPT 族（I0/I1）在 Weather/Electricity 上稳定超过 Golden，但 12 个 setting 上没有任何单一统一模型能稳定超过 Golden 超过 4 个 setting。

综上，本轮是清晰的 null result：周期互补残差方向（D1/D2/D3）与无 PE 的 ICPT 在 96/192 上都没有达到统一替换 A2 的门槛；唯一有实质信号的是 I0 在 Weather/Electricity 上相对 NLinear 的一致改善，但被 ETTh2 的严重回退抵消。后续若继续，方向应是按数据集家族（如 ICPT 用于 Weather/Electricity、NLinear 族用于 ETTm/ETTh）而非统一替换，或先修复 ICPT 在 ETTh2 上的回退——两者都属于本文件冻结后的新实验，需新建编号并保留本轮完整结果。
