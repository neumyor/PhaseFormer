# PCTF Full Repair 与 A2 的 ETTh2/ETTm2 正式测试

## 状态与目的

本实验在 PCTF v3 validation-only 归因完成后冻结候选
`pctf_anchor_repair_full`，先在 ETTh2、ETTm2 的 H96/H192 上与正式 incumbent
`rcrf_pe_lff`（A2）进行三 seed full-train test 对比，并同时报告固定 Golden。本文建立时尚未
运行正式训练，结果表保持待填。

## 模型与公平性边界

- A2：PhaseFormer 相位预测 + LFF-NLinear 完整轨迹 + RCRF 可靠度融合。
- Full Repair：从同 setting、同 seed 的 best-validation A2 checkpoint 初始化，再加入 ICPT
  的 residual-target level/shape 修正、边际系数监督、单周期 level 修复和锚点安全联合优化。
- 两者推理都是一个模型、一个 checkpoint，不是多模型 ensemble。
- 候选包含额外的 A2 预训练与微调阶段，因此同时报告两阶段训练时间；结果不能用于声称同等
  训练成本。如果候选获胜，后续仍需补 continued-A2 对照来分离额外训练与结构贡献。
- 候选已由 validation 冻结，正式 test 只读取一次；本轮之后基于这些 test 数值继续调参必须
  标记为 test-set selection，不能描述为盲测。

## 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | ETTh2、ETTm2 |
| 输入长度 | 720 |
| 输出长度 | 96、192 |
| PhaseFormer period | 24 |
| ICPT period | ETTh2=48，ETTm2=96；同数据集跨 horizon 共享 |
| 训练数据 | 100% |
| seeds | 2021、2022、2023 |
| 预算 | 最多 30 epoch，best-validation checkpoint |
| loss | Huber |
| test | 每个冻结 checkpoint 一次 |
| 运行数 | 12 个 A2 + 12 个 Full Repair = 24 |

局部替换 A2 的判据预先固定为：8 个 MSE/MAE 比值的宏平均 `<0.998`，至少 3/4 setting
双指标改善，最差单指标回退不超过 0.5%。逐 setting 稳定超过 Golden 要求三个 seed 的 MSE、
MAE 全部低于 Golden，且 `mean + sample_std < Golden`。

## 待填正式结果

每格格式为 `MSE mean±std / MAE mean±std`。

| Setting | Golden | A2 | Full Repair | Full/A2 | Full 相对 Golden | 结论 |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-H96 | 0.275 / 0.338 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTh2-H192 | 0.341 / 0.376 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-H96 | 0.163 / 0.256 | 待填 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-H192 | 0.219 / 0.293 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 运行和审计

```bash
.venv/bin/python scripts/run_pctf_anchor_formal_etts.py --stage anchors-dry
.venv/bin/python scripts/run_pctf_anchor_formal_etts.py --stage candidates-dry
.venv/bin/python scripts/run_pctf_anchor_formal_etts.py --stage anchors
.venv/bin/python scripts/run_pctf_anchor_formal_etts.py --stage candidates
.venv/bin/python scripts/run_pctf_anchor_formal_etts.py --stage summarize
```

输出目录为 `research_runs/pctf_anchor_formal_etts_v1/`；汇总文件为 `formal_details.csv`、
`formal_summary.csv` 和 `formal_decision.json`。执行命令全部强制 CUDA，汇总器拒绝缺失 test、
CPU 结果、环境/commit 混用、重复或不完整矩阵以及候选初始输出不等于 matched A2。
