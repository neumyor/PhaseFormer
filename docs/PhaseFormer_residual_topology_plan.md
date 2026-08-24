# PhaseFormer 残差通路拓扑实验计划

> 状态：执行中  
> 分支：`weak-residual-phaseformer`  
> 建立日期：2026-08-24  
> 本文件是本轮实验的唯一计划锚点；实现、筛选、确认和结果解释均以此为准。

## 1. 研究问题

固定 PhaseFormer 主干、残差信号来源、训练协议和评估指标，只改变残差进入模型的位置与
融合形式，回答：

1. 输入到输出的长残差是否优于潜空间长跳连？
2. 输入信息在每个路由层重复注入，是否优于只在末层注入一次？
3. 完整预测的凸融合与零初始化的加法修正，哪一种更稳健？
4. 潜空间逐层残差与输出残差组合后是否互补，还是重复建模并引入退化？

这里不把 attention/MLP 内部已有的局部 skip connection 当作实验变量；它们在所有配置中
保持不变，用于维持 PhaseFormer 主干的训练稳定性。本轮比较的是跨阶段、跨深度或跨输入输出
的长程残差拓扑。

## 2. 统一残差拓扑

所有新增通路均默认关闭，并在开启时零初始化，使初始输出严格退化为原始 PhaseFormer。

| ID | 模式 | 残差路径 | 融合位置 | 目的 |
|---|---|---|---|---|
| R0 | `original` | 无新增长残差 | — | phase-only matched baseline |
| R1 | `residual_output_convex` | 归一化输入 → NLinear 完整预测 | 输出层凸融合 | 复核当前 residual 设计 |
| R2 | `residual_output_additive` | 中心化输入 → 零初始化线性修正 | 输出层加法 | 区分“完整预测替代”与“误差修正” |
| R3 | `residual_latent_long` | 初始 phase latent → 零初始化投影 | 最终 predictor 前一次注入 | 测试输入表征到输出表征的长跳连 |
| R4 | `residual_latent_layerwise` | 初始 phase latent → 独立零初始化投影 | 每个 routing layer 后注入 | 测试逐层残差；一层模型中与 R3 结构等价 |
| R5 | `residual_hybrid` | R4 + R2 | 每层 latent + 输出加法 | 测试多级残差互补性 |

公平性约束：

- R1 沿用当前共享 `WeakPeriodResidualHead`，固定 gate init=0.5；不叠加其他 phase/dynamic 模块。
- R2/R5 共用同一个零初始化输入→输出修正头，固定 gate init=0.5。
- R3/R4/R5 的 latent 投影均为 `D→D`、无 bias、零初始化；不搜索宽度和 gate。
- 不改变 period、loss、LR、batch size、数据划分、early stop 或 checkpoint 规则。
- 报告参数量和耗时；不能把参数更多的结构仅凭微小差异判为更优。

## 3. 实验设置

共同设置：lookback 720、period 24、Huber、base LR、validation early stop、最低
`val_loss` checkpoint、seed 2021；ETTh1 使用其正式 preset seed 2026 时必须单独标注。

代表性 setting：

| Setting | 层数 | 选择理由 |
|---|---:|---|
| ETTh1-h336 | 3 | 多层主干；历史输出残差近中性，可检验逐层注入 |
| ETTh2-h720 | 1 | 历史输出残差最强收益；R3/R4 等价性对照 |
| ETTm1-h720 | 2 | 历史输出残差有害；检验 latent 残差能否避免输出污染 |
| Electricity-h336 | 2 | 高维数据且输出残差有收益；检验拓扑是否跨规模成立 |

### Stage A：验证集筛选

- 4 settings × 6 modes，30% 训练数据，最多 8 epoch。
- 只读取验证集；禁止根据测试集选择拓扑。
- 每个 setting 计算 `score = 0.5×MAE相对改善率 + 0.5×MSE相对改善率`。
- 候选任一指标回退超过 0.5% 视为该 setting 明显退化。
- 冻结规则：保留平均 score 前两名；至少 3/4 settings 双指标不退化，且最差单指标回退
  不超过 1%。若无候选满足，则保留 R0，并把最有诊断价值的一个候选送入确认。
- R3 与 R4 在 ETTh2-h720 必须数值等价；否则先排查实现，不进入效果比较。

### Stage B：全预算确认

- 对 R0、Stage A 前两名运行 100% 数据、正式 epoch/patience、best checkpoint。
- 先确认 ETTh1-h336、ETTh2-h720、ETTm1-h720；Electricity-h336 仅在候选通过前三项
  且仍有正向信号时运行，控制单 GPU 成本。
- 正式结果同时报告 matched rerun 与 `docs/PhaseFormer_gold_standard.md`；matched rerun
  不替代金标准。
- 冻结冠军后再决定是否补 seeds 2022/2023；单 seed 只能称为配对证据。

## 4. 样本级误差分析

对最终 R0/冠军逐 setting 计算 sample×channel MSE/MAE，并程序化选择各最多 8 个：

1. baseline high error；
2. candidate regression；
3. candidate improvement。

分析 horizon 四分段误差、均值/标准差、range、斜率和 peak 位置。可测量观察与原因假设
分开表述。审计产物统一写入：

```text
research_runs/residual_topology_v1/
  run.yaml
  results.csv
  sample_errors.csv
  selected_cases.npz
  objective_error_analysis.md
  objective_error_analysis.zip
  figures/
```

该目录严格遵守六文件加 `figures/` 白名单；训练 checkpoint、日志与临时预测保存在独立
忽略目录，生成审计包后不复制进上述目录。

## 5. 判定标准

- 首要：跨 setting 双指标方向一致；单点大幅提升不能掩盖其他数据集退化。
- 次要：相同精度下优先参数更少、训练更快的拓扑。
- R4 只有在多层 setting 稳定优于 R3，才支持“每层注入优于单一长跳连”。
- R2 只有在稳定优于 R1，才支持“加法误差修正优于完整预测凸融合”。
- R5 只有在同时优于 R2 与 R4，才支持多级残差互补；否则判为冗余。
- 未经三 seed 复核，不更新 `_LATEST_POLICY`，不宣称稳定泛化提升。

## 6. 执行记录

结果、失败、协议偏差和最终决策统一回填
`docs/PhaseFormer_residual_topology_results.md`；本文件只在实验设计或范围发生变化时更新。
