---
name: experiment-and-error-analysis
description: >
  将用户给定的模型设想实现并按约定设置实验，统计 baseline/candidate 结果，
  筛选高误差与显著退化样本进行客观分析，并生成可审计的 Markdown/PDF 报告。
  当用户要求实现明确模型设想、运行对照实验并开展样本级高误差或退化分析时使用；
  不用于纯算法讨论、仅分析已有汇总结果、单纯 smoke test 或明确不运行实验的任务。
---

# Experiment and Error Analysis

## 全局原则

- 不重复询问用户已经明确的信息；普通未指定参数沿用 baseline。
- 仅在实现含义、baseline、数据、训练协议或评价目标存在重要歧义时确认。
- 用户未提供代码时，使实现局部、可关闭，并保持关闭后 baseline 路径不变。
- 允许根据 test 集结果调整模型、参数或机制；保留所有参与选择的实验，并在配置和报告中明确披露 test-set selection。不得将相关结果描述为盲测或无偏泛化估计。
- 高误差报告只陈述可直接测量的现象；如需讨论原因，将其与观察分开，并明确标记为待验证假设。
- 关键数字、样本和图必须能从落盘文件重新验证。
- 遵守仓库的 `MANAGE_RULES.md`、`HOW_TO_DO_RESEARCH.md` 和当前实验计划；发生冲突时，以用户当前明确要求和仓库规则为准，并记录采用的协议。

每次完整运行至少保留下列规范审计集合：

```text
research_runs/<experiment_id>/
├── run.yaml
├── results.csv
├── sample_errors.csv
├── selected_cases.npz
├── objective_error_analysis.md
└── objective_error_analysis.pdf
```

这些是最低产物，不是删除清单。继续保留仓库规则或实验计划要求的 checkpoint、命令、环境、日志、全量预测或其他可复现证据；不要删除既有用户产物。

## 1. 确认并记录实验

从用户设想、代码仓库和已有实验设置中确认：

- mechanism 如何进入计算流程；
- baseline 与 candidate；
- dataset、lookback、horizon、seed；
- optimizer、loss、learning rate、epoch、checkpoint rule；
- metrics 和参数选择方式。

创建 `run.yaml`，至少记录：

```yaml
experiment_id:
code:
  repository:
  branch:
  commit:
  modified_files:
mechanism:
  description:
  feature_flag:
experiment:
  baseline:
  candidate:
  datasets:
  horizons:
  seeds:
  training:
  metrics:
selection:
  source: test | validation | fixed
  selected_config:
  search_notes:
analysis:
  ranking_metric:
  top_k:
  dedup_rule:
```

若最终参数依据 test 结果选择，将 `selection.source` 标为 `test`，并在 `search_notes` 记录测试反馈如何影响后续配置。

## 2. 实现并检查 Candidate

用户未提供实现时：

1. 找到最小插入位置；
2. 实现机制并尽量增加 feature flag；
3. 完成 forward、shape 和 smoke test；
4. 检查 flag-off 是否保持 baseline 行为。

已有实现时，检查其是否符合设想并能正常运行。将 commit、修改文件、机制说明和验证状态写入 `run.yaml`。不要顺便加入与设想无关的 loss、residual、normalization 或其他模块。

## 3. 运行实验与参数搜索

运行 baseline、candidate 以及实际尝试的配置。每次运行至少记录：

- dataset、horizon、seed、config；
- MSE、MAE 和用户指定指标；
- 相对 baseline 的变化；
- 是否最终选中。

允许依据 test 结果修改参数并重新实验，但必须保留真正参与选择的全部配置，不能只留下最终最好结果。

写入 `results.csv`：

```text
config_id,dataset,horizon,seed,model,key_params,mse,mae,delta_mse,delta_mae,selected
```

## 4. 计算样本级误差

使用最终比较的 baseline 和 candidate 在 evaluation 数据上的输出，按 `sample × channel` 或任务适用的最细粒度计算 MSE、MAE、candidate-minus-baseline delta，以及必要的时间范围和 channel 信息。

写入 `sample_errors.csv`：

```text
sample_id,channel,time_range,baseline_mse,candidate_mse,delta_mse,baseline_mae,candidate_mae,delta_mae
```

该文件必须足以重新执行 high-error ranking。如果仓库规则需要保留全量预测，则同时保留，不因生成该表而删除。

## 5. 筛选案例

默认以程序化规则筛选三组：

1. **Baseline High Error**：baseline error 最大；
2. **Candidate Regression**：`candidate_error - baseline_error` 最大；
3. **Candidate Improvement**：candidate 相对 baseline 改善最大。

每组默认 Top 5–10。连续窗口或同 channel 高度重复时，按 `run.yaml` 中的规则去重。不要为了支持某种解释人工挑选案例。

在 `run.yaml` 记录选择结果：

```yaml
analysis:
  ranking_metric: mae
  top_k: 10
  dedup_rule:
  baseline_high_error: []
  candidate_regression: []
  candidate_improvement: []
```

## 6. 保存并分析选中案例

对选中的少量案例保存：

- historical input、ground truth；
- baseline prediction、candidate prediction；
- sample、channel 和 time metadata。

计算适合任务的描述性量，例如 MSE/MAE、mean/std/min/max、range、linear slope、peak/trough position、horizon segment error，必要时再计算 lag、频谱或 autocorrelation。生成 history/truth/baseline/candidate 对比图。

将足以重算案例指标和重绘图的数据写入 `selected_cases.npz`。

只把下面这样的内容写成客观观察：

```text
Candidate MAE 比 baseline 高 0.083。
Candidate peak 位于 step 71，truth peak 位于 step 48。
Candidate std = 0.82，truth std = 0.54。
```

不要把“没有学会趋势”“存在 phase shift”“机制导致 amplitude instability”等未经验证的解释写成事实。需要归因时，将其单列为假设，并给出后续验证方法。

## 7. 汇总客观缺陷

基于 `results.csv`、`sample_errors.csv` 和 `selected_cases.npz` 总结：

- 整体指标提升或退化；
- error distribution 和 horizon-wise error；
- dataset、horizon、channel 差异；
- high-error cases；
- 多案例重复出现的可测量模式。

例如记录 `Candidate std > truth std: 8 / 10`，而不是据此直接声称某个机制原因。将这些内容直接纳入最终报告，不创建重复摘要文件。

## 8. 生成报告

从 `run.yaml`、`results.csv`、`sample_errors.csv` 和 `selected_cases.npz` 生成内容一致的：

```text
objective_error_analysis.md
objective_error_analysis.pdf
```

Markdown 是 canonical report，至少包含：

```markdown
# Experiment and Objective Error Analysis
## 1. Experiment Setup
## 2. Experiment Results
## 3. Parameter / Configuration Search
## 4. Error Distribution
## 5. Horizon-wise Error
## 6. High-Error Selection
## 7. Case Analysis
## 8. Repeated Observable Patterns
## 9. Objective Defect Summary
## 10. Experiment Scope
```

可视化直接嵌入 Markdown；PDF 使用相同数据生成。若使用 test 调参，在报告显著位置写明：

> Final configuration was selected using test-set results.

同时列出 test-set selection 的配置范围和轮次。报告回答哪里更差或更好、差多少、哪些样本最明显、有哪些可测量差异以及出现次数；原因解释只能作为明确标记的假设。

## 9. 最终闭环校验

文件生成后执行以下校验。

### Results

- 确认 `run.yaml` 的 selected config 存在于 `results.csv`；
- 重算报告中的 MSE、MAE 和 delta；
- 确认所有参与选择的配置均已保留；
- 确认 test-set selection 已正确披露。

### Ranking

依据 `run.yaml` 的 metric、Top-K 和 dedup rule，从 `sample_errors.csv` 重新筛选，确认 sample ID 与 `run.yaml`、`selected_cases.npz` 和报告一致。

### Cases

从 `selected_cases.npz` 重新计算报告中的关键案例指标，确认数值一致。

### Reports

- 确认 Markdown 表格和图来自实际落盘数据；
- 确认 PDF 与 Markdown 的核心数字和结论一致；
- 渲染并检查 PDF，确保无缺图、乱码、截断或严重分页问题。

### Files

确认六个最低审计文件存在且非空，并确认仓库额外要求的证据产物也已保留。完成后写入：

```yaml
validation:
  results_checked: true
  case_ranking_checked: true
  case_metrics_checked: true
  markdown_checked: true
  pdf_render_checked: true
  status: passed
```

任一步骤未完成时写 `status: incomplete` 和具体 `issues`；未实际完成校验时不得写 `passed`。

## Workflow

```text
User Idea
→ Confirm Setup
→ Implement Candidate
→ Run/Search Experiments
→ results.csv
→ Sample Errors
→ sample_errors.csv
→ Select Cases
→ selected_cases.npz
→ Objective Analysis
→ MD + PDF
→ Recompute & Validate
→ validation: passed
```

用尽量少的规范产物保存完整证据链，使实验结果、高误差样本和报告都能反向复核，同时保留项目规则要求的必要中间证据。
