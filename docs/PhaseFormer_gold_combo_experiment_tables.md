# PhaseFormer Golden 组合实验待填表

> 实验 ID：`gold_combo_stability_v1`  
> 配套方案：`docs/PhaseFormer_gold_combo_plan.md`  
> 状态：**TBD / 未正式执行**。本文件中的 `TBD` 只能由对应阶段的落盘结果填写，不得凭历史结果或人工估计补值。

## 1. 已知依据（非本轮新结果）

| Setting | Golden MSE/MAE | 已知超过 Golden 的机制 | 历史代表结果 MSE/MAE | 相对 Golden MSE/MAE | 本轮用途 | 证据限制 |
|---|---|---|---|---|---|---|
| ETTh2-720 | 0.402/0.436 | 输出端凸残差/强残差 | 0.3901/0.4265（`dyn_full`） | +2.96%/+2.18% | 验证残差主导场景 | 单 seed；matched protocol 与 Golden 来源并非完全同源 |
| ETTm2-96 | 0.163/0.256 | 相位不确定性+电平+高频修正 | 0.160189/0.248220 | +1.72%/+3.04% | 验证相位修正主导场景 | 单 seed；best-validation checkpoint 修复结果 |
| Electricity-336 | 0.165/0.257 | 自适应输出残差+MAE 训练 | 0.163118/0.253083 | +1.14%/+1.52% | 验证高维自适应场景 | 单 seed；与三位小数 Golden 的差距较小 |

说明：上表只解释候选来源，不参与本轮候选排名，也不替代 Stage B 新结果。

## 2. 实验配置登记表

### 2.1 固定公共配置

| 项目 | 固定值 | 实际值 | 核验 |
|---|---|---|---|
| lookback | 720 | TBD | TBD |
| period | 24 | TBD | TBD |
| 数据划分/缩放 | 仓库标准协议 | TBD | TBD |
| checkpoint | validation loss 最优 | TBD | TBD |
| Stage A 数据比例/epoch/seed | 30% / 8 / 2021 | TBD | TBD |
| Stage B 数据比例/seeds | 100% / 2021,2022,2023 | TBD | TBD |
| test 隔离 | Stage A 不创建 test loader | TBD | TBD |

### 2.2 Setting 训练配置

| Setting | Loss | LR | Batch | 正式 epochs | Patience | 实际配置哈希 |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-720 | Huber | 1e-3 | 256 | 按 base preset | 按 base preset | TBD |
| ETTm2-96 | MAE | 3e-4 | 256 | 按 base preset | 按 base preset | TBD |
| Electricity-336 | MAE | 3e-4 | 64 | 按 target preset | 按 target preset | TBD |

### 2.3 候选机制配置

| Mode | 相位不确定性 | 电平校准 | 高频抑制 | 残差融合 | 门初值/灵敏度 | 参数量 | 配置哈希 |
|---|---|---|---|---|---|---:|---|
| `original` | 关 | 关 | 关 | 无 | — | TBD | TBD |
| `latest` | 当前 target policy | 当前 target policy | 当前 target policy | 当前 target policy | 当前 target policy | TBD | TBD |
| `gold_combo_fixed` | min=0.2 | level=0.2 | 0.8/0.5/w7 | 固定凸融合 | α₀=0.5 | TBD | TBD |
| `gold_combo_adaptive` | min=0.2 | level=0.2 | 0.8/0.5/w7 | 既有三特征 MLP 门 | α₀=0.5 | TBD | TBD |
| `gold_combo_reliability_s0` | min=0.2 | level=0.2 | 0.8/0.5/w7 | RCRF | α₀=0.5, s₀=0 | TBD | TBD |
| `gold_combo_reliability_s2` | min=0.2 | level=0.2 | 0.8/0.5/w7 | RCRF | α₀=0.5, s₀=2 | TBD | TBD |

## 3. Stage A：validation-only 筛选

### 3.1 原始结果（18 runs）

| Setting | Mode | val MSE | val MAE | MSE/original | MAE/original | epochs | test 字段为空 | run/config hash |
|---|---|---:|---:|---:|---:|---:|---|---|
| ETTh2-720 | `original` | TBD | TBD | 1.000000 | 1.000000 | TBD | TBD | TBD |
| ETTh2-720 | `latest` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | `gold_combo_fixed` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | `gold_combo_adaptive` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | `gold_combo_reliability_s0` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | `gold_combo_reliability_s2` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `original` | TBD | TBD | 1.000000 | 1.000000 | TBD | TBD | TBD |
| ETTm2-96 | `latest` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `gold_combo_fixed` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `gold_combo_adaptive` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `gold_combo_reliability_s0` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `gold_combo_reliability_s2` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `original` | TBD | TBD | 1.000000 | 1.000000 | TBD | TBD | TBD |
| Electricity-336 | `latest` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `gold_combo_fixed` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `gold_combo_adaptive` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `gold_combo_reliability_s0` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `gold_combo_reliability_s2` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 3.2 候选总分与冻结记录

总分：`score = mean_setting,metric(candidate_val / original_val)`，共 3 settings × 2 metrics = 6 项；越低越好。

| Rank | Candidate | 6 项均值 score | 最差单项 ratio | 参数量 tie-break | 灵敏度 tie-break | 入选 |
|---:|---|---:|---:|---:|---:|---|
| TBD | `gold_combo_fixed` | TBD | TBD | TBD | 0 | TBD |
| TBD | `gold_combo_adaptive` | TBD | TBD | TBD | 0 | TBD |
| TBD | `gold_combo_reliability_s0` | TBD | TBD | TBD | 0 | TBD |
| TBD | `gold_combo_reliability_s2` | TBD | TBD | TBD | 2 | TBD |

| 冻结项 | 待填内容 |
|---|---|
| 冻结候选 | TBD |
| 选择来源 | validation-only（必须核验） |
| 冻结时间/commit | TBD |
| test 是否在冻结前读取 | TBD（必须为否） |
| 未入选配置是否保留 | TBD |

## 4. Stage B：三 seed 正式测试

### 4.1 每 seed 原始结果（27 runs）

每个单元填写 `MSE / MAE`；括号填写相对 Golden 改善百分比 `ΔMSE% / ΔMAE%`，正数为改善。

| Setting | Seed | `original` | `latest` | 冻结候选 | Candidate run/config hash |
|---|---:|---|---|---|---|
| ETTh2-720 | 2021 | TBD | TBD | TBD | TBD |
| ETTh2-720 | 2022 | TBD | TBD | TBD | TBD |
| ETTh2-720 | 2023 | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2021 | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2022 | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2023 | TBD | TBD | TBD | TBD |
| Electricity-336 | 2021 | TBD | TBD | TBD | TBD |
| Electricity-336 | 2022 | TBD | TBD | TBD | TBD |
| Electricity-336 | 2023 | TBD | TBD | TBD | TBD |

### 4.2 三 seed 聚合

| Setting | Model | MSE mean±sample std | MAE mean±sample std | vs Golden MSE/MAE | vs matched original MSE/MAE | vs latest MSE/MAE |
|---|---|---|---|---|---|---|
| ETTh2-720 | `original` | TBD | TBD | TBD | — | TBD |
| ETTh2-720 | `latest` | TBD | TBD | TBD | TBD | — |
| ETTh2-720 | frozen candidate | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | `original` | TBD | TBD | TBD | — | TBD |
| ETTm2-96 | `latest` | TBD | TBD | TBD | TBD | — |
| ETTm2-96 | frozen candidate | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | `original` | TBD | TBD | TBD | — | TBD |
| Electricity-336 | `latest` | TBD | TBD | TBD | TBD | — |
| Electricity-336 | frozen candidate | TBD | TBD | TBD | TBD | TBD |

### 4.3 稳定性判定

| Setting | 3 seeds MSE 全低于 Golden | 3 seeds MAE 全低于 Golden | MSE mean+std < Golden | MAE mean+std < Golden | 稳定双指标提升 |
|---|---|---|---|---|---|
| ETTh2-720 | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | TBD | TBD | TBD | TBD | TBD |

| 跨数据集总判定 | 待填 |
|---|---|
| 稳定双指标提升 settings 数 | TBD / 3 |
| 剩余 setting 平均退化是否均 ≤1% | TBD |
| 是否满足预注册成功标准 | TBD |
| 可否表述为“稳定超过 Golden” | TBD |

## 5. 门控与误差分析待填表

### 5.1 RCRF 活性

| Setting | Seed | mean reliability r | mean gate α | gate std | sensitivity mean/range | 低可靠度是否对应更高 α |
|---|---:|---:|---:|---:|---|---|
| ETTh2-720 | 2021/22/23 | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2021/22/23 | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | 2021/22/23 | TBD | TBD | TBD | TBD | TBD |

### 5.2 sample×channel 误差分布（candidate 相对 latest）

| Setting | Seed | cells | improved % | regressed % | mean ΔMSE | mean ΔMAE | baseline high-error top-10 | regression top-10 | improvement top-10 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| ETTh2-720 | 2021 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | 2022 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTh2-720 | 2023 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2021 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2022 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ETTm2-96 | 2023 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | 2021 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | 2022 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Electricity-336 | 2023 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. 审计与复现检查表

| 检查项 | 要求 | 状态/证据 |
|---|---|---|
| 单元测试 | 可靠度、门控、互斥、前后向、flag-off、seed | TBD |
| smoke | 3 settings；有限 loss；有 best checkpoint | TBD |
| Stage A 隔离 | `test_mse/test_mae` 为空，不创建 test loader | TBD |
| Stage A 完整性 | 18/18 runs，配置哈希唯一 | TBD |
| 冻结记录 | selection.source=`validation_only` | TBD |
| Stage B 完整性 | 27/27 runs，3 seeds 均真实生效 | TBD |
| 指标重算 | 从预测重算结果与 `results.csv` 一致 | TBD |
| case 排名 | 三类 top-10 程序化选择且可复算 | TBD |
| NPZ 对齐 | setting/sample/channel/history/truth/baseline/candidate 齐全 | TBD |
| ZIP 一致性 | Markdown 与图逐字节一致，无未引用图 | TBD |
| 审计目录白名单 | 仅 six-file 协议文件与 `figures/` | TBD |
| git 状态 | 代码/方案/结果 commit 可追溯，工作树干净 | TBD |

## 7. 最终结论模板

```text
冻结候选：TBD（由 validation-only Stage A 选出）。
三 seed 下，稳定双指标超过 Golden 的 setting 为：TBD。
ETTh2-720：MSE TBD±TBD，MAE TBD±TBD，相对 Golden TBD%/TBD%。
ETTm2-96：MSE TBD±TBD，MAE TBD±TBD，相对 Golden TBD%/TBD%。
Electricity-336：MSE TBD±TBD，MAE TBD±TBD，相对 Golden TBD%/TBD%。
跨数据集成功标准：TBD（满足/不满足）。
限制：Golden 仅三位小数，来源协议与 matched rerun 的同源性为 TBD；不得把舍入级差异表述为稳定收益。
```
