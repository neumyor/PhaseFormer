---
name: experiment-and-error-analysis
description: >
  实现用户给定的模型设想并运行 baseline/candidate 对照实验，筛选高误差与显著退化样本，
  生成可审计的 Markdown 报告及其 ZIP 图片包。当用户同时要求实现明确设想、运行实验和开展
  样本级错误分析时使用；不用于纯讨论、仅分析已有汇总结果、单纯 smoke test 或明确不运行实验的任务。
---

# Experiment and Error Analysis

## 原则与产物

- 遵守 `MANAGE_RULES.md`、`HOW_TO_DO_RESEARCH.md` 和当前实验计划；用户当前明确要求优先。
- 未指定参数沿用 baseline；仅在实现、baseline、数据、协议或目标存在重要歧义时询问。
- 实现应局部、可关闭，flag-off 保持 baseline 路径；不要夹带无关模块。
- 允许根据 test 集结果调参，但保留全部选择轨迹，并显著披露 test-set selection；不得称为盲测或无偏泛化估计。
- 报告区分可测量观察与原因假设；所有数字、案例和图均须可从落盘文件复核。

每个 `experiment_id` 严格只保留下列六个非空文件和一个图目录：

```text
research_runs/<experiment_id>/
├── run.yaml
├── results.csv
├── sample_errors.csv
├── selected_cases.npz
├── objective_error_analysis.md
├── objective_error_analysis.zip
└── figures/
    └── <setting>__<figure_name>.png
```

根目录不得有其他文件或目录；`figures/` 只保留 Markdown 引用的图。禁止保留 PDF、checkpoint、脚本、环境快照、日志、TensorBoard、全量预测和临时文件。临时产物放在仓库忽略的位置，完成后仅清理本次生成的内容，不删除来源不明的用户文件。

一次运行可含多个 setting。`setting` 是 dataset、horizon、seed、split 等评估条件的稳定唯一字符串（如 `ETTh1_h96_seed2021`）；baseline、candidate 和 config 不写入其名称。所有 setting 共用上述文件，靠显式 `setting` 字段区分；禁止 setting 子目录或按 setting 拆分 YAML/CSV/NPZ/Markdown/ZIP。

## 1. 确认、实现与运行

确认 mechanism、baseline/candidate、dataset、split、lookback、horizon、seed、optimizer、loss、learning rate、epoch、checkpoint rule、metrics 和参数选择方式。创建 `run.yaml`，至少记录：

```yaml
experiment_id:
code: {repository:, branch:, commit:, modified_files: []}
mechanism: {description:, feature_flag:}
experiment:
  baseline:
  candidate:
  settings:
    - {setting:, dataset:, split:, lookback:, horizon:, seed:}
  training:
  metrics:
execution:
  environment:
  settings:
    - {setting:, commands: [], runtime:}
selection:
  source: test | validation | fixed
  selected_configs:
    - {setting:, config_id:, search_notes:}
analysis: {ranking_metric: mae, top_k: 10, dedup_rule:}
```

`experiment.settings`、`execution.settings`、`selection.selected_configs` 必须覆盖同一 setting 集。若 test 结果影响最终参数，设 `selection.source: test`，并在 `search_notes` 记录配置、指标、尝试顺序和选择依据。

实现 candidate 时：找到最小插入点；增加 feature flag；检查 forward、shape、smoke test 和 flag-off baseline 等价性；把 commit、修改文件和验证状态写入 `run.yaml`。已有实现也需核对设想与可运行性。

运行 baseline、candidate 及实际尝试的全部配置。所有 setting 写入单一 `results.csv`：

```text
setting,config_id,dataset,horizon,seed,model,key_params,mse,mae,delta_mse,delta_mae,selected
```

至少记录 MSE、MAE、用户指定指标、相对 baseline 变化和是否入选；失败运行及原因也应在 `run.yaml` 或报告记录。不能只保留最优配置。

## 2. 样本误差与案例

用最终比较的 baseline/candidate evaluation 输出，按 `sample × channel` 或任务可用的最细粒度计算 MSE、MAE、candidate-minus-baseline delta、time range 和 channel。所有 setting 写入单一 `sample_errors.csv`：

```text
setting,baseline_config_id,candidate_config_id,sample_id,channel,time_range,baseline_mse,candidate_mse,delta_mse,baseline_mae,candidate_mae,delta_mae
```

每个 setting 只写最终 baseline/candidate 对；文件必须足以重新排名，不保留全量预测。

按程序化规则逐 setting 筛选，默认每组 Top 5–10，并依 `dedup_rule` 去除连续窗口或同 channel 高度重复案例：

1. **Baseline High Error**：baseline error 最大；
2. **Candidate Regression**：candidate − baseline 最大；
3. **Candidate Improvement**：baseline − candidate 最大。

不得为支持解释而人工挑选。将选择结果写入 `run.yaml`：

```yaml
analysis:
  ranking_metric: mae
  top_k: 10
  dedup_rule:
  selections:
    - setting:
      baseline_high_error: []
      candidate_regression: []
      candidate_improvement: []
```

把全部 setting 的入选案例保存在单一 `selected_cases.npz`：至少含对齐的 `setting`、sample/channel/time metadata、historical input、truth、baseline prediction 和 candidate prediction。可用统一数组或 setting 前缀键，但必须能逐 setting 重算指标和重绘图。

计算适用的描述统计，如 MSE/MAE、mean/std/min/max、range、linear slope、peak/trough position 和 horizon segment error；仅必要时计算 lag、频谱或 autocorrelation。图中对比 history/truth/baseline/candidate，保存为 `figures/<setting>__<figure_name>.png`，Markdown 仅用 `figures/<filename>` 相对路径；禁止绝对路径、`file://`、`..`、符号链接及未引用图。

只把可测量内容写成观察，例如“Candidate MAE 高 0.083”或“Candidate peak 在 step 71，truth 在 step 48”。“没有学会趋势”“phase shift”“机制导致 amplitude instability”等只能标为待验证假设，并附验证方法。

## 3. 汇总与报告

基于 `results.csv`、`sample_errors.csv`、`selected_cases.npz` 总结整体指标、误差分布、horizon-wise error、dataset/horizon/channel 差异、代表案例和跨案例重复的可测量模式（如 `Candidate std > truth std: 8/10`）。不创建额外摘要文件。

生成 canonical `objective_error_analysis.md`，至少包含：

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

按 setting 分组或明确标识所有结果与案例。报告回答哪里更好/更差、差多少、哪些样本最明显、有哪些可测量差异及出现次数；归因只能作为假设。若使用 test 调参，显著写明：

> Final configuration was selected using test-set results.

并列出参与选择的配置范围、轮次和依据。

生成 `objective_error_analysis.zip`，其根目录只能包含：

```text
objective_error_analysis.md
figures/<Markdown 实际引用的图片>
```

ZIP 中 Markdown/图片须与实验目录原件字节一致。解析 Markdown 得到图片白名单后逐项写入 ZIP，不得递归打包实验目录；排除其他审计文件、未引用图、父目录、绝对路径、符号链接、隐藏文件、`.DS_Store` 和 `__MACOSX/`。解压后应能直接打开 Markdown 并看到全部图片。

## 4. 必要校验

完成后进行简要但真实的闭环检查：

- 每个声明的 setting 在 `run.yaml`、`results.csv`、`sample_errors.csv` 和 `selected_cases.npz` 中一致；selected config 存在，所有搜索配置均保留，test selection 已披露。
- 按记录规则抽查重算汇总指标、Top-K 排名和关键案例指标。
- Markdown 引用图均存在；ZIP 只含 Markdown 及其引用图，解压后路径有效且文件与原件一致。
- 实验目录恰好符合六文件加 `figures/` 的白名单，无按 setting 拆分产物或 PDF。

在 `run.yaml` 记录：

```yaml
validation:
  results_checked: true
  ranking_and_cases_checked: true
  report_and_archive_checked: true
  directory_and_settings_checked: true
  status: passed
```

未完成的项目设为 `false`，并写 `status: incomplete` 与 `issues`；不得虚报 `passed`。

流程：确认设置 → 最小实现 → 对照/搜索实验 → 汇总结果 → 样本误差 → 程序化选例 → 客观分析 → Markdown + ZIP → 必要校验。始终以单组跨 setting 产物维持完整证据链。
