# PCTF 多融合策略实验计划

> 状态：模型、preset、实验 runner 与定向测试已完成；尚未启动训练或读取新的 test。
> 实验编号：`pctf_fusion_v1`。

## 1. 实验问题

当前 PCTF 已经把 NLinear 和 ICPT 做成结构化残差，但最外层仍由一个 RCRF 权重在两个完整
预测之间融合。它有三个潜在问题：

1. 同一个权重同时控制绝对水平、周期间水平和周期内形状；
2. 权重在整个 horizon 上共享，不能区分第几个未来周期；
3. 内层 ICPT gate 和外层 RCRF 分别训练，可能重复收缩同一个周期信号。

本轮固定 PhaseFormer、NLinear、no-PE ICPT、period 24、训练设置和初始化，只改变融合器，
验证收益究竟来自哪一种融合逻辑。所有论文候选都是一个端到端 PhaseFormer、一个 checkpoint；
均匀平均与 Softmax 只作为负对照，永远不能被 runner 冻结成论文冠军。

## 2. 三个公共分支

- `P`：PhaseFormer 相位预测，负责周期内相位模板；
- `T`：NLinear 轨迹预测，负责绝对水平和周期级轨迹；
- `C`：ICPT 周期预测，将 720 步切成 24 步 cycle patch，使用 no-PE future-query decoder、
  `d_model=32`、4 heads、1 层 encoder/decoder。

七个新融合器以相同顺序构造 NLinear 和 ICPT；额外随机模块均在隔离 RNG 中初始化。因此同一
seed 下，它们的三条分支以及 PhaseFormer embedding/routing/predictor 初始参数严格配对。

A1 的高频抑制在新模型中只校准 `P`，然后再进入融合器；不会在融合后改变 NLinear 所拥有的
绝对均值。这个位置调整是保证下面分量约束成立所必需的。旧模型和其他 preset 的执行顺序不变。

## 3. 正交分量表示

将任意未来预测按 `K=H/24` 个周期重排，对分支 `Z` 唯一分解：

`Z = μ_Z + L_Z + S_Z`

- `μ_Z`：完整 horizon 的均值；
- `L_Z`：每个周期的均值减去 `μ_Z`，在 horizon 上均值为零；
- `S_Z`：每个周期内部去均值后的形状，每个24步周期均值为零。

`μ`、`L`、`S` 分别对应绝对水平、周期间水平变化和周期内波形，互不重叠。结构化融合器均让
NLinear 独占 `μ`，因此 ICPT 或 PhaseFormer 无法变成隐藏的第三个完整轨迹锚点。

## 4. 融合候选

| ID | preset | 具体融合 | 论文候选 |
|---|---|---|---:|
| F0 | `pctf_dual_fixed` | 当前两级 PCTF：NLinear+ICPT 后再过 RCRF | 是，历史锚点 |
| F1a | `pctf_fusion_component_scalar` | 正交分量融合，全 horizon 共用两个 gate | 是 |
| F1b | `pctf_fusion_component_cycle` | 正交分量融合，每个未来周期独立 gate | 是 |
| F2a | `pctf_fusion_monotonic` | F1b + 有方向约束的历史证据门控 | 是 |
| F2b | `pctf_fusion_mlp` | F1b + 小型无约束证据 MLP | 是 |
| F3 | `pctf_fusion_phase_modulation` | ICPT 调制 PhaseFormer 相位模板 | 是 |
| C0 | `pctf_fusion_uniform_control` | 三个完整预测均匀平均 | **否** |
| C1 | `pctf_fusion_softmax_control` | 逐未来周期学习三路完整预测 Softmax | **否** |

### F1a/F1b：分量级投影融合

`Y = μ_T + [(1-g_L)L_T + g_L L_C] + [(1-g_S)S_P + g_S S_C]`

- 绝对水平只来自 NLinear；
- 周期间轨迹在 NLinear 与 ICPT 之间融合；
- 周期内形状在 PhaseFormer 与 ICPT 之间融合；
- F1a 的 `g_L/g_S` 是两个全局标量；F1b 为每个未来周期分别学习 gate；
- cycle-dependent level gate 之后再次减去 horizon 均值，避免门控重新引入绝对水平。

两个 gate 初值均为 0.10。F1a 对 F1b 直接回答“逐未来周期自由度是否必要”。

### F2a：单调证据门控

F2a 保持 F1b 的分量公式，但使 gate 随历史证据变化：

`g_S = sigmoid(b_S + w_u(1-r_phase) + w_q(q_shape-0.5))`

`g_L = sigmoid(b_L + w_q(q_level-0.5) - w_d·drift)`

所有 `w=softplus(raw)≥0`，所以方向不可被训练反转：相位越不可靠，越允许 ICPT 修正周期形状；
ICPT 历史 regret 越低，越相信对应 ICPT 分量；近期漂移越强，周期间水平越回退到更稳定的
NLinear。`q_shape/q_level` 来自最近两个严格历史伪起点的 ICPT-vs-NLinear 重建 regret，
不读取未来且不反向传播置信度梯度。

### F2b：证据 MLP

使用同一 F1b 公式，但让一个 `7→16→2K` MLP 输出 gate logit 修正。七个输入为：相位不可靠
度、近期漂移、shape/level 历史置信度，以及 `P-T`、`P-C`、`T-C` 三个归一化预测分歧。
特征全部 detach，防止三个分支通过操纵 gate 特征降低训练损失；末层零初始化，所以初始严格
等于 gate=0.10 的 F1b。

F2a 与 F2b 的比较回答：明确的时序先验是否比自由 MLP 更稳定。此前 TriAxis 路由命中率仅约
31%–42%，因此 F2a 是优先论文候选，F2b 主要检验单调约束是否过强。

### F3：相位模板调制

F3 使用 NLinear 的逐周期均值作为轨迹，然后用 ICPT 的周期形状对 PhaseFormer 形状进行
可微圆周对齐：

1. 枚举24个 circular shift，按 PhaseFormer/ICPT 形状相关性做 temperature=0.10 的 softmax；
2. 得到期望相位偏移和对齐后的 PhaseFormer 模板；
3. 用最小二乘幅度调制模板，幅度限制在 `[0.5,2.0]`；
4. 以初值0.10注入调制模板，以初值0.05注入 ICPT 剩余的零均值形变。

最终输出仍保持 NLinear 的完整 horizon 均值，形状部分逐周期零均值。该方法把 ICPT 解释为
“预测相位模板如何跨周期发生偏移、振幅变化和剩余形变”，而不是另一条完整预测，叙事最贴合
相位—周期中心。

### C0/C1：负对照

C0 计算 `(P+T+C)/3`；C1 学习逐未来周期的三路 Softmax，初始化同样为三等分。二者刻意保留
完整预测混合，用来检验结构化候选是否真的优于简单误差抵消。即使它们 validation 最好，
汇总器也会写出结果但拒绝将其冻结为正式冠军。

## 5. Validation-only 筛选

- 数据：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity；
- 输入720、输出96、period 24、30% train、最多8 epoch、seed 2021、Huber；
- 同协议运行 A1、A2、I0、F0–F3、C0/C1，共 `6×11=66` 个 model run；
- 只按最低 validation loss checkpoint 计算 validation MSE/MAE；任何非空 test 字段都会让
  汇总器报错停止。

论文候选须同时满足：相对 A2 的12指标宏平均比 `≤0.998`，至少4/6数据集双指标改善，最差
单指标比 `≤1.01`，相对 A1/A2/I0 逐指标包络宏平均比 `≤1.005`，并且相对 C0/C1 逐指标
包络宏平均比 `≤1.000`。最后一项保证结构化方法不能只靠三分支误差抵消解释收益。

| ID | macro/A2 | 双指标改善/6 | worst/A2 | macro/参考包络 | macro/负对照包络 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| F0 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| F1a | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| F1b | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| F2a | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| F2b | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| F3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| C0 | 待填 | 待填 | 待填 | 待填 | 1.000基线 | 仅诊断 |
| C1 | 待填 | 待填 | 待填 | 待填 | 1.000基线 | 仅诊断 |

必须额外报告四组嵌套对照：F1b/F1a、F2a/F1b、F2b/F1b、F3/F1b，不能仅给最终冠军。

## 6. 冻结后的正式确认

只有论文候选通过上述门槛后，runner 才允许冻结唯一冠军。在六数据集、H96/H192、seeds
2021/2022/2023 上统一重跑 A1、A2、I0 和冠军，共 `6×2×3×4=144` 个 model run。训练只按
validation 选 checkpoint，冻结后读取一次 test，不根据 test 更换融合器。

| Test setting | Golden | A1 | A2 | I0 | Fusion champion | 相对A2 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1-96 | 0.359 / 0.382 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh1-192 | 0.397 / 0.404 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh2-96 | 0.275 / 0.338 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTh2-192 | 0.341 / 0.376 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm1-96 | 0.293 / 0.344 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm1-192 | 0.323 / 0.361 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm2-96 | 0.163 / 0.256 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| ETTm2-192 | 0.219 / 0.293 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Weather-96 | 0.148 / 0.195 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Weather-192 | 0.193 / 0.237 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Electricity-96 | 0.129 / 0.221 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |
| Electricity-192 | 0.148 / 0.238 | 待重跑 | 待重跑 | 待重跑 | 待填 | 待填 |

正式替换 A2 仍要求：24指标宏平均比 `<0.998`、至少8/12 setting 双指标改善、最差单指标
回退 `≤0.5%`。同时报告三 seed 均值/sample std、稳定低于 Golden 的 setting 数、参数量、
显存以及推理时间。

## 7. 命令与输出

只检查命令矩阵：

```bash
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage screen-dry
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage confirm-dry \
  --champion pctf_fusion_monotonic
```

确认后执行：

```bash
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage screen --progress
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage screen-summarize
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage confirm --progress
.venv/bin/python scripts/run_pctf_fusion_strategies.py --stage confirm-summarize
```

scratch 输出为 `research_runs/pctf_fusion_v1/`；checkpoint 和训练日志不提交。正式结论后再按
仓库审计规则整理白名单报告包。本项目过去已经查看过 ETTh2/ETTm2 等 test，因此最终结果即使
遵守本轮冻结协议，也必须披露项目级 test exposure，不能描述为完全盲测。

## 8. 已知风险和停止判断

- F1/F2 让 NLinear 只贡献绝对均值和周期级水平，不使用其周期内形状；若显著弱于 F0，说明
  这个职责划分过强，而不是 gate 参数不足。
- F2 的两个历史伪起点只估计下一周期 regret，再广播到全部未来周期；F1b 的 per-cycle prior
  可以表达平均 horizon 差异，但样本级远期置信度仍可能失配。
- F3 的相位偏移来自 ICPT 输出与 PhaseFormer 输出的可微匹配，而不是 ICPT token 直接监督；
  若 shift 长期接近零或幅度卡在边界，应判定调制机制未被有效使用。
- F2b 比 F2a 好但不优于 C1 时，只能说明自由路由有工程收益，不能支持结构化融合创新。
- 若没有论文候选通过门槛，实验在 validation 停止，不访问正式 test，也不通过额外按数据集
  选择融合器来制造统一提升。

## 9. 实现验证

- 全仓 `223 passed`，另有 `229 subtests passed`；现有 warning 仅来自无 Trainer 的日志测试和
  测试环境 NVML；
- 七种新策略均完成有限值 forward/backward，PhaseFormer、NLinear、ICPT 都能获得梯度；
- F1/F2/F3 数值校验完整 horizon 均值等于 NLinear，周期形状均值绝对误差 `<2e-6`；
- F2a 校验相位不可靠度、ICPT 置信度和漂移对 gate 的单调方向；两个 masked context 严格因果；
- F3 校验期望 circular shift shape、幅度 `[0.5,2.0]` 边界和 H96/H192 完整 PhaseFormer forward；
- C0/C1 初始输出均严格等于三个完整预测的均值，branch weights 和为1；
- 同 seed 下七个 preset 的共享 PhaseFormer 参数均与 A1 逐参数一致；
- dry-run 为66个 validation-only runs和144个冻结 test runs；合成矩阵验证 test 泄漏拒绝、
  负对照不可晋级、冠军冻结和正式汇总。

ETTm2-H96 的参数量审计：A1为72,905；F0及F1/F2a/F3/C0/C1为96,063–96,075，F2b为
96,335。除F2b的小型 MLP 外，各融合策略的参数量几乎完全相同；F3虽然不增加明显参数，但
24次 circular alignment 会增加运行时间，必须在真实实验中单独报告。
