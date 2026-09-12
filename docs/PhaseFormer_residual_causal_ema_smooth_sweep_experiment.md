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

> 待训练完成后回填：表 1（各 setting × 5 档 test MSE/MAE）、表 2（各档相对
> `s=0` 的 ΔMSE%/ΔMAE%）、表 3（判定汇总）、表 4（与 boxcar 扫描的跨算子对比）、
> 结论、复现命令与产物位置。

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
