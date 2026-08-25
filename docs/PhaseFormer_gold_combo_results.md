# PhaseFormer 跨数据集 Golden 组合机制实验 —— 结果反馈

> 本文件按实验计划 `docs/PhaseFormer_gold_combo_plan.md` 执行，给出结果与反馈。所有数值均可从
> 落盘文件逐行复核：Stage A `research_runs/gold_combo_screen_runs/*/metrics.csv`（18 run）、
> Stage B `research_runs/gold_combo_full_runs/*/metrics.csv`（27 run）、样本级审计
> `research_runs/gold_combo_stability_v1/`（six-file 协议包）。
>
> - 分支：`weak-residual-phaseformer`；实验日期：2026-08-25
> - 协议：Stage A validation-only（30% 数据、max 8 epoch、seed 2021，不创建 test loader）筛选 →
>   预注册 6 比值分数冻结一个候选 → Stage B full-budget（100% 数据、validation early stop +
>   best checkpoint，seed 2021/2022/2023）三 seed 正式测试
> - Golden（MSE/MAE，三位小数）：ETTh2-720 `0.402/0.436`；ETTm2-96 `0.163/0.256`；
>   Electricity-336 `0.165/0.257`（来源见 `docs/PhaseFormer_gold_standard.md`）
> - Δ% 约定：`(golden − new)/golden × 100`，正数 = 超过 Golden
> - 运行覆盖：Stage A 18 run（3 settings × 6 modes）；Stage B 27 run（original/latest/frozen ×
>   3 settings × 3 seeds）

---

## 一、冻结候选

Stage A 在 4 个组合候选中按预注册分数 `score = mean over 3 settings × 2 metrics of
(candidate_val/original_val)` 排名（越低越好；分数 tie-break 为参数量、再为初始灵敏度）：

| Rank | Candidate | 6 项均值 score | 最差单项 ratio | 入选 |
|---:|---|---:|---:|---|
| 1 | `gold_combo_reliability_s2` | **0.80473** | 0.85280 | ✓ |
| 2 | `gold_combo_adaptive` | 0.80720 | 0.86020 | |
| 3 | `gold_combo_reliability_s0` | 0.80739 | 0.85518 | |
| 4 | `gold_combo_fixed` | 0.80827 | 0.85686 | |

**冻结记录**（`research_runs/gold_combo_screen_runs/freeze_record.json`）：
candidate = `gold_combo_reliability_s2`（RCRF，α₀=0.5，s₀=2，s_max=4.0）；
selection source = `validation_only`；`test_read_before_freeze = False`（test 未在冻结前读取）。
未入选配置全量保留于 screen_summary.csv，未按 Stage A 临时追加超参数（遵守停止规则）。

RCRF 融合：`r = Var_l(mean_k x) / (Var_l(mean_k x) + mean_l Var_k x + eps)`（由收缩前相位序列计算，
数据确定性、与 seed 无关）；`s = s_max·tanh(s_raw)`；`α = sigmoid(logit(α₀) + s·(1−r))`，
`y = (1−α)·y_phase + α·y_residual`。s₀=0 时 α 恒为 0.5（固定门 warm-start），s₀=2 提供初始灵敏度。

## 二、Stage B 三 seed 测试结果（27 runs）

每格 `MSE / MAE`；括号为相对 Golden 的 `ΔMSE% / ΔMAE%`（正 = 改善）。

| Setting | Seed | `original` | `latest` | `gold_combo_reliability_s2` |
|---|---|---|---|---|
| ETTh2-720 | 2021 | 0.425394/0.455186 (−5.8%/−4.4%) | 0.402206/0.431116 (−0.1%/+1.1%) | 0.400050/0.431524 (+0.49%/+1.03%) |
| ETTh2-720 | 2022 | 0.414260/0.448377 (−3.0%/−2.8%) | 0.394293/0.427505 (+1.9%/+1.9%) | 0.391617/0.427280 (+2.58%/+2.00%) |
| ETTh2-720 | 2023 | 0.408618/0.443848 (−1.6%/−1.8%) | 0.395049/0.428755 (+1.7%/+1.7%) | 0.391018/0.429524 (+2.73%/+1.49%) |
| ETTm2-96 | 2021 | 0.169031/0.258217 (−3.7%/−0.9%) | 0.160697/0.249424 (+1.4%/+2.6%) | 0.159872/0.245121 (+1.92%/+4.25%) |
| ETTm2-96 | 2022 | 0.165805/0.252485 (−1.7%/+1.4%) | 0.161145/0.248931 (+1.1%/+2.8%) | 0.159846/0.245648 (+1.93%/+4.04%) |
| ETTm2-96 | 2023 | 0.169122/0.257861 (−3.8%/−0.7%) | 0.159594/0.247761 (+2.1%/+3.2%) | 0.159547/0.245222 (+2.12%/+4.21%) |
| Electricity-336 | 2021 | 0.169980/0.259650 (−3.0%/−1.0%) | 0.163211/0.252918 (+1.1%/+1.6%) | 0.162954/0.253420 (+1.24%/+1.39%) |
| Electricity-336 | 2022 | 0.168299/0.258164 (−2.0%/−0.5%) | 0.163062/0.252934 (+1.2%/+1.6%) | 0.164409/0.254921 (+0.36%/+0.81%) |
| Electricity-336 | 2023 | 0.168089/0.258064 (−1.9%/−0.4%) | 0.163765/0.254109 (+0.7%/+1.1%) | 0.164977/0.255533 (+0.01%/+0.57%) |

（完整 27 run 每 seed 行含 run/config hash 见 `docs/PhaseFormer_gold_combo_experiment_tables.md` §4.1。）

## 三、三 seed 聚合与稳定性判定

| Setting | Model | MSE mean±sample_std | MAE mean±sample_std | vs Golden MSE/MAE | vs matched original MSE/MAE | vs latest MSE/MAE |
|---|---|---|---|---|---|---|
| ETTh2-720 | `original` | 0.416091±0.008537 | 0.449137±0.005707 | −3.51%/−3.01% | — | +4.76%/+4.66% |
| ETTh2-720 | `latest` | 0.397183±0.004367 | 0.429125±0.001834 | +1.20%/+1.58% | −4.54%/−4.46% | — |
| ETTh2-720 | `s2` | 0.394228±0.005051 | 0.429443±0.002123 | **+1.93%/+1.50%** | −5.25%/−4.38% | −0.74%/+0.07% |
| ETTm2-96 | `original` | 0.167986±0.001889 | 0.256188±0.003211 | −3.06%/−0.07% | — | +4.68%/+3.01% |
| ETTm2-96 | `latest` | 0.160479±0.000798 | 0.248705±0.000854 | +1.55%/+2.85% | −4.47%/−2.92% | — |
| ETTm2-96 | `s2` | 0.159755±0.000180 | 0.245331±0.000280 | **+1.99%/+4.17%** | −4.90%/−4.24% | −0.45%/−1.36% |
| Electricity-336 | `original` | 0.168789±0.001036 | 0.258626±0.000888 | −2.30%/−0.63% | — | +3.33%/+2.09% |
| Electricity-336 | `latest` | 0.163346±0.000370 | 0.253320±0.000683 | +1.00%/+1.43% | −3.22%/−2.05% | — |
| Electricity-336 | `s2` | 0.164113±0.001044 | 0.254625±0.001087 | +0.54%/+0.92% | −2.77%/−1.55% | **+0.47%/+0.51%** |

**单 setting 稳定性判定**（要求：3 seed 的 MSE、MAE 均低于 Golden，且对两指标 `mean + sample_std < Golden`）：

| Setting | 3 seeds 全低于 Golden | mean+std < Golden | 稳定双指标提升 |
|---|---|---|---|
| ETTh2-720 | MSE ✓ / MAE ✓ | MSE 0.39928<0.402 ✓ / MAE 0.43157<0.436 ✓ | **是** |
| ETTm2-96 | MSE ✓ / MAE ✓ | MSE 0.15994<0.163 ✓ / MAE 0.24561<0.256 ✓ | **是** |
| Electricity-336 | MSE ✓ / MAE ✓ | MSE 0.16516>0.165 ✗ / MAE 0.25571<0.257 ✓ | **否** |

**跨数据集总判定**（计划 §4）：至少 `2/3` settings 稳定（ETTh2-720、ETTm2-96 = 2/3 ✓）；
剩余 setting（Electricity-336）三 seed 均值相对 Golden 的 MSE、MAE 退化分别
`(0.164113−0.165)/0.165 = −0.54%`、`(0.254625−0.257)/0.257 = −0.92%` —— 均为改善（负退化），
≤1% 满足 → **预注册跨数据集成功标准满足**。

**可表述为"跨数据集稳定超过 Golden"**，但需附以下限定：

1. **Electricity-336 不是稳定增益**。其三个 seed 的 MSE、MAE 虽均低于 Golden
   （MSE 0.162954/0.164409/0.164977 < 0.165；MAE 0.253420/0.254921/0.255533 < 0.257），但
   MSE 的 `mean + sample_std = 0.16516` 越过 Golden（0.165）。越过的幅度约 `0.00016`，恰在
   Golden 三位小数舍入范围内——**按计划"不得把舍入级差异表述为稳定收益"的要求，此处只报告
   数值，不作稳定收益结论**。
2. **Electricity-336 相对 `latest` 略有退化**：三 seed 均值 MSE +0.47%、MAE +0.51%
   （s2022/s2023 两个 seed 均 +0.0012~+0.0019）；RCRF 在高可靠度场景（r≈0.77）选择偏相位、
   关闭部分残差，代价是略低于当前 dataset policy。此退化绝对量 ~0.001，且仍在 Golden 之上，
   但方向如实记录。
3. 相对 matched `original`，冻结候选在三个 setting 上均为一致改善
   （ETTh2 −5.25%/−4.38%，ETTm2 −4.90%/−4.24%，Electricity −2.77%/−1.55%）。

## 四、RCRF 活性分析（`objective_error_analysis.md` §8，从落盘重算）

| Setting | mean reliability r | mean gate α | α std | sensitivity mean | r-α corr |
|---|---:|---:|---:|---:|---|
| ETTh2-720 | 0.193013 | 0.773–0.811 | 0.039 | 1.65–1.86 | −0.996~−0.998 |
| ETTm2-96 | 0.01857 | 0.872–0.878 | 0.0038 | 1.97–2.01 | −0.9998 |
| Electricity-336 | 0.771778 | 0.308–0.384 | 0.041–0.054 | 0.78–0.94 | −0.9995~−1.0 |

**可测量观察**：
- **r-α 相关性 ≈ −1.0（全部 9 个 setting×seed）**：低可靠度 → 高 α（偏残差），高可靠度 → 低 α
  （偏相位），与 RCRF 设计方向一致（α = sigmoid(logit α₀ + s·(1−r))）。
- **ETTm2-96 被判定为低可靠度**（r≈0.019）→ α≈0.87，融合强烈偏向残差分支；该 setting 也是
  相对 `latest` 双指标均改善最大的一格（MAE −1.36%），与"残差主导场景"一致。
- **Electricity-336 被判定为高可靠度**（r≈0.77）→ α≈0.31，融合偏相位；这正是其相对 `latest`
  轻微退化（残差被部分关闭）的直接机制来源。
- **ETTh2-720 居中**（r≈0.19）→ α≈0.77–0.81，偏残差；MSE 相对 `latest` 改善、MAE 几乎持平。
- r 跨 seed 恒等（数据确定性，seed 无关）；α 因训练的 sensitivity s 随 seed 变化
  （ETTh2 s2021→s2023 α 0.811→0.773 递减），敏感度范围 0.78–2.01。

以上为可测量观察；"低可靠度⇒偏残差是否带来收益"的因果归因仅作假设，证据是三个 setting 的
α 行为与各自相对 `latest` 的 MSE/MAE 方向基本一致，但样本数有限（9 格）。

## 五、sample×channel 误差分布（candidate 相对 `latest`，`sample_errors.csv`）

| Setting | Seed | cells | improved % | regressed % | mean ΔMSE | mean ΔMAE |
|---|---:|---:|---:|---:|---:|---|
| ETTh2-720 | 2021 | 15127 | 45.12% | 54.88% | −0.002155 | +0.000408 |
| ETTh2-720 | 2022 | 15127 | 49.03% | 50.97% | −0.002676 | −0.000225 |
| ETTh2-720 | 2023 | 15127 | 48.34% | 51.66% | −0.004030 | +0.000770 |
| ETTm2-96 | 2021 | 79975 | 58.38% | 41.62% | −0.000825 | −0.004303 |
| ETTm2-96 | 2022 | 79975 | 56.94% | 43.06% | −0.001297 | −0.003281 |
| ETTm2-96 | 2023 | 79975 | 56.52% | 43.48% | −0.000048 | −0.002540 |
| Electricity-336 | 2021 | 1580925 | 50.82% | 49.18% | −0.000257 | +0.000502 |
| Electricity-336 | 2022 | 1580925 | 45.45% | 54.55% | +0.001347 | +0.001987 |
| Electricity-336 | 2023 | 1580925 | 47.81% | 52.19% | +0.001212 | +0.001424 |

**可测量观察**：
- **ETTm2-96：全 3 seed 双指标均值均改善**（ΔMSE、ΔMAE 全负），改善 cell 占比 56–58%，与聚合
  双指标改善一致。
- **ETTh2-720：MSE 均值改善（ΔMSE 全负，幅度最大 −0.0040）但 MAE 均值略增**（ΔMAE
  +0.0004/+0.0008），cell 层面约 45–49% 改善——MSE 的改善集中在少数大误差 cell，MAE 层面
  局部有代价。
- **Electricity-336：s2021 双指标边际改善（ΔMSE −0.0003），s2022/s2023 双指标退化**
  （ΔMSE +0.0012~+0.0013、ΔMAE +0.0014~+0.0020）——与聚合表中"相对 `latest` +0.47%/+0.51%"
  对应，也与 α≈0.31 偏相位（关闭残差）一致。
- 程序化 top-10（baseline 高误差 / candidate 退化 / candidate 改善）逐 setting 落盘于
  `selected_cases.npz`（269 个对齐 cell：history/truth/baseline_pred/candidate_pred/original_pred，
  7 数组齐全），无人工挑选。

## 六、审计与复核（`docs/PhaseFormer_gold_combo_experiment_tables.md` §6）

- 单元测试全绿（含 RCRF 15 项 + gold_combo preset）；smoke 3 setting 有限 loss + best.ckpt +
  validation-only 无 test loader；Stage A 18/18、Stage B 27/27 全落盘，config 哈希唯一。
- 指标由 `scripts/analyze_gold_combo.py` 从 best.ckpt 测试预测重算，与 `results.csv` 一致
  （本文件 §二、§三数值与 `research_runs/gold_combo_full_runs/*/metrics.csv` 逐行核对）。
- 审计目录 `research_runs/gold_combo_stability_v1/` 仅含 six-file 协议文件 + `figures/`
  （18 张引用图全部被 Markdown 引用，ZIP 逐字节一致，无未引用图、无 PDF/checkpoint/全量预测）。
- `sample_errors.csv`（704MB，per-cell 粒度）足以重排序且不保留全量预测；npz 2.2MB 仅存选中
  case 切片。两者均位于被忽略的 `research_runs/`，不提交。
- git：实现 `a5f0b1f`、工具 `7694579`、本次结果文档提交见 `docs/agent-log.md`。

## 七、限制

1. Golden 仅三位小数，且与 matched rerun 的来源协议同源性有限（计划 §1 证据限制）。
2. Electricity-336 的 MSE `mean+std` 越线属舍入级幅度，**不表述为稳定收益**。
3. RCRF 的 r 是数据确定性先验；α 依赖训练的 sensitivity，三 seed 间 α 有 ~0.04 波动。
4. 结论基于 3 setting × 3 seed 的固定组合；跨数据集泛化结论限于预注册判据范围。

## 八、结论

- 冻结候选 `gold_combo_reliability_s2`（RCRF，α₀=0.5，s₀=2）由 validation-only Stage A 选出。
- **稳定双指标超过 Golden**：ETTh2-720（MSE 0.394228±0.005051，MAE 0.429443±0.002123，
  +1.93%/+1.50%）、ETTm2-96（MSE 0.159755±0.000180，MAE 0.245331±0.000280，+1.99%/+4.17%）。
- Electricity-336 三 seed 均值亦低于 Golden（+0.54%/+0.92%），但 MSE `mean+std` 越线（舍入级），
  按计划不列为稳定增益；相对 `latest` 轻微退化（+0.47%/+0.51%），如实报告。
- **预注册跨数据集成功标准满足**（2/3 稳定 + 剩余 setting 相对 Golden 无 ≥1% 退化）。
- RCRF 机制行为与设计一致（r-α corr ≈ −1.0）：ETTm2 低可靠度偏残差、Electricity 高可靠度偏相位、
  ETTh2 居中；未发现与设计相反的门控行为。
