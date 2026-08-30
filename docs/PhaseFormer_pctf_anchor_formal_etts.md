# PCTF Full Repair 与 A2 的 ETTh2/ETTm2 正式测试

## 状态与目的

本实验在 PCTF v3 validation-only 归因完成后冻结候选
`pctf_anchor_repair_full`，在 ETTh2、ETTm2 的 H96/H192 上与正式 incumbent
`rcrf_pe_lff`（A2）进行三 seed full-train test 对比，并同时报告固定 Golden。24/24 个正式
运行已经完成；候选通过了预注册的**这两个数据集上的局部替换门槛**，但这不等价于已经证明
它能在全部数据集上替换 A2。

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

## 正式 test 结果

每格格式为 `MSE mean±std / MAE mean±std`。

| Setting | Golden | A2 | Full Repair | Full 相对 A2（MSE / MAE） | Full 相对 Golden（MSE / MAE） |
|---|---:|---:|---:|---:|---:|
| ETTh2-H96 | 0.275 / 0.338 | 0.273569±0.001677 / 0.333215±0.000692 | **0.272845±0.000404 / 0.332200±0.000353** | **+0.265% / +0.305%** | **+0.784% / +1.716%** |
| ETTh2-H192 | 0.341 / 0.376 | 0.342269±0.002436 / 0.376320±0.002288 | **0.336671±0.001985 / 0.373755±0.001785** | **+1.635% / +0.682%** | **+1.269% / +0.597%** |
| ETTm2-H96 | 0.163 / 0.256 | 0.160706±0.000677 / 0.249093±0.000498 | **0.158474±0.000190 / 0.247592±0.000827** | **+1.389% / +0.602%** | **+2.777% / +3.284%** |
| ETTm2-H192 | 0.219 / 0.293 | **0.213835±0.001000** / 0.286338±0.000939 | 0.214270±0.000676 / **0.285080±0.000234** | -0.203% / **+0.439%** | **+2.160% / +2.703%** |

这里“+”表示误差下降。四个 setting 的 MSE 比值宏平均为 `0.992285`，MAE 比值宏平均为
`0.994930`，即 Full Repair 相对 A2 平均降低 **0.772% MSE、0.507% MAE**；八个比值的联合
平均改善为 **0.639%**。它在 3/4 个 setting 上同时改善 MSE 和 MAE，最坏单指标是
ETTm2-H192 的 MSE 回退 0.203%，因此通过预注册门槛。

逐 seed 看，12 个 setting×seed 对比中有 8 个双指标获胜：ETTh2-H192、ETTm2-H96 均为
3/3，ETTh2-H96 为 2/3，ETTm2-H192 为 0/3（但后者三个 seed 的 MAE 都改善）。按“三个 seed
均低于 Golden，且 mean+sample std 仍低于 Golden”的严格定义，Full Repair 为 4/4，A2 为
2/4。

## 收益归因与边界

Full Repair 从 matched A2 checkpoint 开始第二阶段联合微调。候选 checkpoint 内部的 A2
输出相对原始 A2 平均降低约 0.599% MSE、0.230% MAE；最终 ICPT 融合相对这个已经微调过的
内部 A2 再降低约 0.174% MSE、0.278% MAE。因此当前结果证明“完整训练流程”胜过 A2，但不能
把全部 0.772%/0.507% 收益归因于 ICPT 结构；需要同预算的 continued-A2 才能严格分离额外
训练收益。

ETTm2-H192 进一步说明该候选不是全面支配：ICPT 修正稳定改善 MAE，却让 MSE 略高。这更像
减少普遍的小误差、同时没有压住少数较大误差，而不是稳定降低全部误差风险。由于本轮 test
已经读取，任何依据这些 test 数值进行的后续修改都属于 test-set selection，不能再称为盲测。

## 训练成本

候选总训练时间包含 matched A2 预训练和第二阶段 Full Repair 微调；A2 只含单阶段。时间是
本次 RTX 4090 运行的每 seed 均值，显存是三 seed 峰值。

| Setting | A2 / Full 参数量 | A2 / Full 总训练秒 | A2 / Full 峰值显存（MiB） |
|---|---:|---:|---:|
| ETTh2-H96 | 71,059 / 95,881 | 6.44 / 17.80 | 158.0 / 432.0 |
| ETTh2-H192 | 140,407 / 165,297 | 5.76 / 17.00 | 173.3 / 474.4 |
| ETTm2-H96 | 73,009 / 100,917 | 20.48 / 59.05 | 199.2 / 349.9 |
| ETTm2-H192 | 140,407 / 168,349 | 16.67 / 57.55 | 174.6 / 353.6 |

候选参数量增加约 17.7%–38.2%，端到端训练时间约为 A2 的 2.77–3.45 倍。推理仍只加载一个
checkpoint，但本轮没有单独测量推理延迟。

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

实际矩阵为 12 个 A2 + 12 个候选，全部使用 NVIDIA RTX 4090、PyTorch 2.7.1+cu126、CUDA
12.6、Lightning 2.6.5，以及实验冻结提交
`c8b61c4c9a0f4d6b637c3d4599d3c55cdc0da452`。审计确认候选初始化与 matched A2 的最大输出差
为 0、缺失的 55 个键全部属于新增 composer 参数、无 unexpected key，且没有失败或 CPU
结果混入。
