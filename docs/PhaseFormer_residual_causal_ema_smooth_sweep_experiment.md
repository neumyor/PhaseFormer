# 残差支路 causal-EMA 平滑扫描（无低秩压缩）— 实验计划与结果登记

> 状态：**计划已冻结，结果待训练完成后回填。**

---

## 1. 背景与动机

`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`（boxcar 平滑扫描）在同
7 个 setting（各自用户指定的低秩 rank 下）扫描了残差支路的对称、双侧、有限窗口
（24 步）boxcar 平滑因子 `smooth_ratio`，判定"无可检测效应（2/7）"，且检测到的
两个效应（ETTh2-96、ETTh2-720）方向均为"平滑越强越差"。

讨论中提出一个假设：低秩压缩（沿"容量/秩"轴）近似中性，可能是因为它不改变分支
看到的**时间分辨率**；而平滑（沿"时间细节"轴）有害，可能是因为它把分支从
`X-A` 式（保留细节的残差校正）推向 `Only-A` 式（只剩低频趋势）行为——这与更早的
`Weak_residual_trend_component_study_closure.md` 中"`Only-A` 在强周期/相位敏感
样本上会失败"的发现方向一致。

用户现在要求：**去掉低秩压缩**（改用全秩残差头，不经过
`PooledLowRankWeakPeriodResidualHead` 的 rank 瓶颈），**改用之前 X-A/Only-A
趋势成分研究里使用的 causal EMA 方式**做平滑，在同 7 个 setting 上重新扫描一遍。
目的是把"压缩维度"和"平滑维度"解耦，并换一种平滑算子（单侧因果、指数衰减，而非
对称有限窗口）看结论是否变化。

本任务只要求"实现扫描、跑实验、确认影响"，不要求分析高误差/显著退化样本，按
`CLAUDE.md` 的界定不触发 `/experiment-and-error-analysis` skill；按
`HOW_TO_DO_RESEARCH.md` 的常规专项实验流程执行。

---

## 2. 关键前提披露（不可省略）

1. **全秩头，非低秩压缩**。本轮使用 `WeakPeriodResidualHead`
   （`src/models/phase_adapters.py`）——单层 `nn.Linear(seq_len, pred_len)`，
   零初始化，无池化无秩瓶颈，是历史 `direct_nlinear` 基线用的同一个头。不涉及
   任何 rank 选择，因此不再有"rank 由用户指定、dataset-aware"的披露问题；但
   **7 个 setting 本身仍是沿用早前 test 名义双优事后挑选的结果**，披露约束不变
   （见第 2 条）。
2. **Test-set selection 延续**。7 个 setting 早已按 test 名义双优事后挑选
   （见 `PhaseFormer_rank_sweep_conditioned_experiment.md` §2），且已被多轮读过
   test。本轮沿用同一批 setting、同一批 (gate_init, lr) 冻结配置，新增对
   causal-EMA `smooth_ratio` 维度的一次性 test 读取——不是新的盲测数据，同样不得
   表述为无偏泛化估计。
3. **单 seed 2021**，与此前所有轮一致；结果为 test-exposed 探索性证据。
4. **causal EMA 的 `alpha` 为 analogy 选择，非本轮调参/校准结果。** 固定
   `alpha = 2/(24+1) = 0.08`——用 EMA 半衰期近似对应 boxcar 实验里的
   `smooth_window=24`，使两轮"平滑强度量级"大致可比；该值同时恰好与
   `docs/Weak_residual_asymmetric_component_plan.md`（`α=0.08`）中已用过的一个
   causal EMA 校准值一致，但**不是针对本轮这 7 个 setting 重新校准或调过的**，
   也不是盲测选择。本轮只扫描 `smooth_ratio`，不扫描 `alpha`。
5. 本实验**不修改任何 preset 默认值**；无论结果如何，`smooth_ratio`/
   `causal_ema_alpha` 不会因本轮结果被写入默认配置。

---

## 3. 逐 setting 冻结配置

沿用 `PhaseFormer_rank_sweep_conditioned_experiment.md` 表 1（Stage 0 冻结的
gate_init/lr），**不含 rank 列**（全秩头无 rank 参数）：

| Setting | H | gate_init | learning_rate |
|---|---:|---:|---:|
| ETTh2-96 | 96 | 0.5 | 默认（1e-3） |
| ETTh2-720 | 720 | 0.5 | 默认（1e-3） |
| ETTm2-96 | 96 | 0.5 | 3e-4 |
| ETTm2-192 | 192 | 0.2 | 默认（1e-3） |
| Weather-96 | 96 | 0.2 | 3e-4 |
| Weather-192 | 192 | 0.5 | 默认（1e-3） |
| Electricity-336 | 336 | 0.5 | 默认（1e-3） |

其余训练协议（L720→H、period 24、Huber loss、≤30 epochs、best-val checkpoint、
`--require-cuda`）与两轮 rank sweep / boxcar 平滑扫描一致。

---

## 4. 平滑因子网格

`smooth_ratio ∈ {0, 0.25, 0.5, 0.75, 1.0}`（5 档，与 boxcar 扫描相同网格，便于
直接对比两种平滑算子）。`causal_ema_alpha=0.08` 全局固定（不扫描）。

`s=0` 档（等价于全秩、无平滑，即 `direct_nlinear`）**重新训练**，不复用历史任何
run——理由与 boxcar 扫描一致：确保配对可比，不依赖历史某个 run 的配置完全一致。

7 setting × 5 档 = **35 runs**，单 seed，每个 config 直接一次性
`--evaluate-test`（与此前各轮做法一致）。

---

## 5. 判定规则（预注册，回填结果前不得修改）

与 boxcar 扫描完全相同的规则（原文照搬，保持跨实验可比）：

对每个 setting，以该 setting 的 `smooth_ratio=0` 结果为参照，计算各非零档位的
`ΔMSE% = (ref − new) / ref × 100`、`ΔMAE%` 同理（正值 = 该档优于 `s=0`）。

1. **单 setting 判定**：若至少一个非零档位在 MSE 与 MAE 上同时同向且 `|Δ| ≥ 1%`，
   记为该 setting"检测到平滑效应"（方向不限）。
2. **跨 setting 计数**：
   - `≥5/7` 检测到效应 → "平滑因子对该结构有可检测影响"；
   - `3–4/7` → "部分信号"；
   - `≤2/7` → "无可检测效应"。
3. **最优档位聚集性**：额外统计 7 个 setting 各自 test 最优 `smooth_ratio` 的
   分布；若明显聚集，可指示性讨论，但仍受第 2 节披露约束，不作为可泛化结论。
4. **止损条款**：本轮结果不触发 preset 修改；若判定为"无可检测效应"或"部分
   信号"，不追加 seed 或维度。
5. **跨算子对比（指示性，非预注册统计检验）**：额外并排列出本轮（causal EMA）
   与 boxcar 扫描的判定结果、每个 setting 的效应方向与最优档位，讨论两种平滑
   算子的结论是否一致；这一对比不改变第 1–4 条的判定标准，只是描述性讨论。

---

## 6. 结果

> 状态：**已完成并回填（2026-09-12）。** 7 setting × 5 档，共 35 runs，seed 2021，
> 单次读 test。判定按 §5 预注册规则得出 **"部分信号（3/7）"**。

### 表 1：各 setting × 5 档 test MSE/MAE

| Setting | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|---|
| ETTh2-96 | 0.2721/0.3328 | 0.2730/0.3336 | 0.2743/0.3348 | 0.2758/0.3363 | 0.2770/0.3376 |
| ETTh2-720 | 0.3909/0.4279 | 0.3912/0.4279 | 0.3920/0.4284 | 0.3940/0.4297 | 0.3958/0.4311 |
| ETTm2-96 | 0.1585/0.2480 | 0.1585/0.2481 | 0.1587/0.2484 | 0.1599/0.2500 | 0.1604/0.2506 |
| ETTm2-192 | 0.2157/0.2881 | 0.2158/0.2883 | 0.2162/0.2888 | 0.2169/0.2895 | 0.2174/0.2901 |
| Weather-96 | 0.1467/0.1940 | 0.1467/0.1940 | 0.1469/0.1941 | 0.1468/0.1940 | 0.1469/0.1938 |
| Weather-192 | 0.1918/0.2363 | 0.1917/0.2363 | 0.1916/0.2361 | 0.1921/0.2366 | 0.1918/0.2366 |
| Electricity-336 | 0.1617/0.2547 | 0.1621/0.2544 | 0.1618/0.2547 | 0.1621/0.2554 | 0.1642/0.2574 |

`s=0` 一列为本轮重新训练的基线（未复用历史任何 run，理由见 §4）。

### 表 2：各档相对 `s=0` 的 ΔMSE% / ΔMAE%（正 = 优于 s=0）

| Setting | s=0.25 | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|
| ETTh2-96 | -0.34/-0.22 | -0.82/-0.60 | -1.36/-1.04 | -1.81/-1.42 |
| ETTh2-720 | -0.07/-0.00 | -0.28/-0.12 | -0.79/-0.44 | -1.25/-0.76 |
| ETTm2-96 | -0.02/-0.03 | -0.15/-0.15 | -0.89/-0.77 | -1.24/-1.04 |
| ETTm2-192 | -0.07/-0.08 | -0.26/-0.26 | -0.54/-0.50 | -0.79/-0.70 |
| Weather-96 | -0.02/-0.01 | -0.12/-0.06 | -0.06/+0.02 | -0.14/+0.09 |
| Weather-192 | +0.07/+0.00 | +0.11/+0.07 | -0.15/-0.13 | -0.00/-0.13 |
| Electricity-336 | -0.22/+0.11 | -0.07/-0.01 | -0.25/-0.28 | -1.50/-1.04 |

### 表 3：判定汇总（按 §5 规则：≥1 个非零档位双指标同向且 |Δ|≥1%）

| Setting | 检测到效应？ | 满足阈值的档位 | test 最优 `s`（MSE 最低，并列取 MAE） |
|---|---|---|---|
| ETTh2-96 | **是** | s=0.75（-1.36%/-1.04%）、s=1（-1.81%/-1.42%），均为劣化 | s=0 |
| ETTh2-720 | 否 | 无档位达到 \|Δ\|≥1% 双指标同向（s=1 最大，-1.25%/-0.76%，MAE 未过线） | s=0 |
| ETTm2-96 | **是** | s=1（-1.24%/-1.04%，同向劣化） | s=0 |
| ETTm2-192 | 否 | 无档位达到阈值 | s=0 |
| Weather-96 | 否 | 无档位达到阈值，且 s=0.75/1 两指标符号相反 | s=0 |
| Weather-192 | 否 | 无档位达到阈值 | s=0.5 |
| Electricity-336 | **是** | s=1（-1.50%/-1.04%，同向劣化） | s=0 |

**跨 setting 计数：3/7** → 按 §5 判定规则落入 **`3–4/7` → "部分信号"**。

**最优档位分布**（7 个 setting 各自 test 最优 `s`）：`s=0` × 6（ETTh2-96、
ETTh2-720、ETTm2-96、ETTm2-192、Weather-96、Electricity-336）、`s=0.5` × 1
（Weather-192，且 Δ 幅度 <0.11%，在噪声量级内）。**绝大多数（6/7）聚集在
`s=0`（关闭平滑最优）**，三个检测到效应的 setting（ETTh2-96、ETTm2-96、
Electricity-336）效应方向均一致地"平滑越强越差"（各自在 `s=1` 上达到最大劣化
幅度），未观察到任何 setting 上"平滑显著改善性能"的同等强度信号。

### 表 4：与 boxcar 平滑扫描的跨算子对比（指示性，非预注册统计检验）

| Setting | boxcar：检测到效应？ | causal EMA：检测到效应？ | 两者方向是否一致 |
|---|---|---|---|
| ETTh2-96 | 是（劣化） | 是（劣化） | 一致 |
| ETTh2-720 | 是（劣化） | 否 | 部分一致（同向但未过线） |
| ETTm2-96 | 否 | 是（劣化） | 未过线方向仍为劣化，一致 |
| ETTm2-192 | 否 | 否 | 一致（均无效应） |
| Weather-96 | 否 | 否 | 一致 |
| Weather-192 | 否 | 否 | 一致 |
| Electricity-336 | 否 | 是（劣化） | 未过线方向仍为劣化，一致 |

- **跨 setting 计数**：boxcar 2/7（无可检测效应）vs. causal EMA 3/7（部分信号）——
  同一量级，均未达到 `≥5/7` 的"一致效应"门槛。
- **方向完全一致**：两轮实验里，所有 7 个 setting 在所有非零档位上都没有出现
  "同向显著改善"的情形；凡是有方向性信号的 setting，方向都是"平滑越强越差"。
  即换用单侧因果、指数衰减的平滑算子后，结论方向未变。
- **具体 setting 不完全重叠**：ETTh2-720 在 boxcar 下过线（-2.98%/-2.27%），
  在 causal EMA 下未过线（-1.25%/-0.76%）；ETTm2-96、Electricity-336 则反过来，
  boxcar 下未过线、causal EMA 下过线。这提示"哪个 setting 对平滑更敏感"本身
  可能依赖平滑算子的具体形状（对称有限窗口 vs. 单侧指数衰减），而非仅由数据集/
  horizon 决定——但样本量（7 个 setting、单 seed、两种算子）过小，只作指示性
  观察，不构成机制性结论。

### 结论

1. **换用 causal EMA 后，"平滑对该全秩残差头有害、压缩本身相对中性"的方向性
   结论保持不变，且信号强度略有增强**（3/7 部分信号 vs. boxcar 的 2/7 无可
   检测效应），支持此前讨论中的假设：平滑（无论对称有限窗口还是单侧指数衰减）
   抹掉的是残差支路依赖的短时/相位细节，这与压缩容量（秩）是两个不同的、影响
   不对称的轴。
2. **没有观察到任何"平滑显著改善性能"的信号**，两轮实验、两种平滑算子、共
   14 个 (setting, 算子) 组合上全部一致。
3. **最优档位高度聚集在 `s=0`**（6/7），进一步支持"该全秩残差头的价值确实
   来自短时细节，抹平细节没有换来任何可测的正收益"。
4. **具体哪些 setting 过线依赖平滑算子形状**，样本太小不构成可泛化结论，仅作
   指示性观察（见表 4）。
5. **与本实验前提保持一致**：本轮为全秩头（无 rank 选择），但 7 个 setting 与
   其 (gate_init, lr) 组合仍是此前 test-exposed 事后挑选的结果；`alpha=0.08`
   为 analogy 选择，非本轮调参/校准结果（见 §2 第 4 条）。结果**不能**解释为
   "causal EMA 平滑在通用配置下有害"，只能表述为"在这 7 个用户指定 setting、
   固定 alpha 下，条件性扫描的结果"。
6. **止损条款生效**：结果为"部分信号"，按 §5 第 4 条不追加 seed 或维度；
   `smooth_ratio`/`causal_ema_alpha` 不写入任何 preset 默认值。

### 边界与披露（重申，见 §2）

- 7 个 setting 及其 (gate_init, lr) 组合均为用户指定的 test-exposed 条件性
  配置，非盲测、非无偏泛化估计。
- 单 seed 2021。
- `causal_ema_alpha=0.08` 为跨轮"平滑强度量级可比"的 analogy 选择，未针对本轮
  重新校准，也未扫描。
- 本结果**不构成**"causal EMA 平滑机制被证明有效或无效"的可对外声明证据，只是
  这一狭窄配置族上的探索性诊断，且与 boxcar 扫描共同构成的"平滑总体有害/压缩
  中性"图景，其推广范围同样受限于上述条件。

---

## 7. 复现

### 7.1 运行命令模板

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_causal_ema_smooth_sweep.py \
  --dataset ETTh2 --horizon 96 --seed 2021 \
  --gate-init 0.5 \
  --smooth-ratios 0,0.25,0.5,0.75,1.0 \
  --causal-ema-alpha 0.08 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/causal_ema_smooth_sweep_v1
```

`--learning-rate 0.0003` 在需要覆盖数据集默认 lr 的 setting（ETTm2-96、
Weather-96）上追加；其余 setting 使用数据集默认 lr，不传该参数。逐 setting 的
具体命令按第 3 节冻结配置表填入 `--gate-init`/`--learning-rate`。

### 7.2 结果与产物位置

- 每个 setting 一份
  `research_runs/causal_ema_smooth_sweep_v1/causal_ema_sweep_<Dataset>_h<H>_s2021_results.csv`。
- `research_runs/` 已 gitignore，产物仅本地/服务器保留，不入库。

### 7.3 复现环境

- 服务器：8× A800-80GB，conda env `time`
  （`/home/yyk/yyk03/miniconda3/envs/time/bin/python`）。
- 同步规范：见 `REMOTE_SERVER.md`。

---

## 8. 关联文档

- **同结构 boxcar 平滑扫描（对照算子）**：[`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`](PhaseFormer_residual_smooth_ratio_sweep_experiment.md)
- **上一轮压缩比扫描**：[`PhaseFormer_rank_sweep_conditioned_experiment.md`](PhaseFormer_rank_sweep_conditioned_experiment.md)
- **X-A/Only-A 趋势成分研究结题**：[`Weak_residual_trend_component_study_closure.md`](Weak_residual_trend_component_study_closure.md)
- **causal EMA 参数来源**：[`Weak_residual_asymmetric_component_plan.md`](Weak_residual_asymmetric_component_plan.md)
- **金标准参照**：[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)
- **操作记录**：[`agent-log.md`](agent-log.md)
