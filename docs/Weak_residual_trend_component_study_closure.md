# Weak Residual 趋势性成分研究：阶段结题记录

> 状态：**阶段结束**。后续工作转入 `weak_residual_nlinear_bottleneck`，研究 NLinear 弱残差分支在信息瓶颈下能否保留校正能力。本文件冻结本阶段的实验边界、已交付证据与可支持结论；它不替代金标准模型比较。

## 1. 已完成的问题与固定协议

研究对象是 Weak Residual 中的 NLinear-style 校正分支。每一组对照均保持 PhaseFormer 主分支接收完整历史 `X`，只改变 NLinear 分支的可见输入：

| 路由 | PhaseFormer 输入 | NLinear 输入 |
|---|---|---|
| Baseline-full | `X` | `X` |
| X-A | `X` | `X-A` |
| Only-A | `X` | `A` |

两路径共享**由完整 X 估计的一组 RevIN 统计量**。所有已汇总的三数据集对照均为 ETTh1、Weather、ETTm1，`L=720 → H=96`、seed=2021、validation-only；没有把 test 用于候选选择或结论。

已比较的成分包括周期 level、近期/全局线性、两类 Gaussian 分量、causal EMA 和 Holt local linear。`smooth_multiscale=G_24(X)-G_72(X)` 是双尺度差分，不应称为全局平滑趋势。后续的深入样本审计额外识别了 `smooth_local` 的双侧平滑右端伪影，因此不将其作为纯趋势主张的核心证据。

## 2. 已交付的可审计结果

| 交付物 | 内容 |
|---|---|
| `research_runs/asymmetric_prediction_divergence_cases/ALL_COMPONENT_ROUTE_VALIDATION_METRICS.md` | 三数据集、七个已训练成分的 Baseline-full / X-A / Only-A validation 聚合指标。 |
| `research_runs/asymmetric_prediction_divergence_cases/EXTRACTION_PARAMETERS.md` | 图册实际使用的提取参数和 checkpoint 路径审计。 |
| `research_runs/global_ema_route_role_cases/objective_error_analysis.md` | `global_linear` 与慢参数 `causal_ema` 的60个双向误差角色样本、图与 Baseline-full MAE。 |
| `research_runs/etth1_smooth_route_role_cases/objective_error_analysis.md` | ETTh1 `smooth_local` / `smooth_multiscale` 的32个双向误差角色样本、图与边界伪影量化。 |

上述 `research_runs/` 目录是本地、不可提交的大型实验工件；每个正式审计目录均保留其对应六文件审计结构和便携 ZIP。

## 3. 本阶段证据支持的结论

### 3.1 趋势 A 的作用是条件性校正，而非完整预测表征

- 在平滑水平、持续漂移或历史周期不再可靠的样本中，Only-A 可优于 X-A。NLinear 可把低频/宽尺度 A 转换为预测的整体 level、偏置或振幅校正。
- 在强周期、相位敏感或快速转折的样本中，Only-A 会失败。NLinear 仍需要趋势外残差中的短时形状、周期相位和峰谷定位；此时 X-A 更合适。
- 因而 NLinear 分支的总体收益取决于这两类状态在数据集中的比例及其误差代价，不能由单个极端样本或一个趋势提取器概括。

### 3.2 对 PhaseFormer 的互补性

当前证据最适合支持以下分工，而不是“PhaseFormer 完全不使用趋势”的表述：

\[
\hat Y = \hat Y_{\mathrm{PhaseFormer}}(X) +
\Delta_{\mathrm{NLinear}}(Z),\qquad Z\in\{X, X-A, A\}.
\]

- PhaseFormer 用完整 `X` 形成预测的周期相位、局部形状、跨尺度关系和峰谷时序。
- NLinear 以较低复杂度提供数值校正，主要对应预测整体高度、慢漂移、偏置，以及周期振幅或 level 失配。
- 这解释了为何历史周期可靠时保留 `X-A` 有益，而周期误导时 Only-A 有时更稳：前者提供动态定位，后者避免把过时周期直接外推。

## 4. 不可超出的结论边界

1. 这些 X-A / Only-A 模型是独立端到端训练；最终误差不能严格分离为 NLinear 的单独因果贡献，因为 PhaseFormer 可随训练重新分工。
2. 因此结果能证明“某类 A 对 NLinear 校正路由具有可用价值和条件性充分性”，不能证明“PhaseFormer 从未利用 A”或“NLinear 已充分、最优地利用 A”。
3. 结果全部为 discovery validation 证据，不能作为盲测泛化或替代 `docs/PhaseFormer_gold_standard.md` 的金标准提升声明。

## 5. 阶段决策与下一步

趋势性成分发现和样本级机制解释在本分支结束。下一阶段不再扩张趋势提取器网格，而在新分支中固定已验证的 Weak Residual 协议，研究对 NLinear 分支输入或隐表示施加信息瓶颈后：

1. 哪类低阶校正信息仍能保留；
2. 分支性能可压缩到何种程度；
3. 压缩何时首先破坏 level/bias 校正，何时首先破坏相位/局部动态校正。

任何新实验均应另立计划，并与本阶段的 validation-only 发现结论清晰区分。
