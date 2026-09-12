# 残差支路平滑因子（smooth_ratio）扫描 — 实验计划与结果登记

> 状态：**计划已冻结，结果待训练完成后回填。**

---

## 1. 背景与动机

`PhaseFormer_rank_sweep_conditioned_experiment.md`（第二轮 rank sweep）在
7 个"名义 test 双优于 Golden"的 setting 上扫描了 NLinear 弱周期残差支路的相对秩
`q`，结论是低秩压缩效应仅"部分信号"（3/7），且各 setting 的 test 最优档位高度分散
（q=1 ×3、q=1/8 ×2、q=1/4 ×1、q=1/32 ×1）。

用户在此基础上要求：**在这 7 个 setting 各自的（用户指定的）test 最优压缩比 q 下，
改为扫描残差支路的平滑因子 `smooth_ratio`，确认它对模型性能的影响。**

`smooth_ratio`（`src/models/phase_adapters.py:75-131`，`PooledLowRankWeakPeriodResidualHead`）
是残差支路在做时间池化/低秩因子化之前，对"去 last 值居中"的输入序列与其
`smooth_window`（固定 24）步 replicate-padded boxcar 平滑之间的凸组合：

```text
centered = (1 - smooth_ratio) * centered + smooth_ratio * smoothed
```

`smooth_ratio=0` 关闭平滑（等价于此前两轮 rank sweep 一直固定使用的设置）；
`smooth_ratio=1` 完全使用平滑后的序列。本轮只扫描 `smooth_ratio`，`smooth_window`
与 `pool_factor=1` 均保持固定，不引入新的自由维度。

本任务只要求"实现扫描、跑实验、确认影响"，不要求分析高误差/显著退化样本，按
`CLAUDE.md` 的界定不触发 `/experiment-and-error-analysis` skill；按
`HOW_TO_DO_RESEARCH.md` 的常规专项实验流程执行。

---

## 2. 关键前提披露（不可省略）

1. **Rank 选择由用户直接指定，非统一规则挑选。** 第二轮 rank sweep 的结论是
   "无统一最佳压缩比"（3/7 部分信号，最优档位分散）。用户在被明确告知这一结论后，
   选择"逐 setting 使用各自 test 最优 q"作为本轮固定 rank 的依据。这是
   dataset-aware 的选择，不满足 `HOW_TO_DO_RESEARCH.md` §4 关于统一机制、避免
   按数据集切换方案的默认要求；**本轮结果不能作为"存在可泛化的平滑机制"或
   "存在通用最佳 rank"的证据，只能解读为"在用户指定的这组特定 (setting, rank,
   gate_init, lr) 上，平滑因子的条件性影响"**。
2. **Test-set selection 延续。** 7 个 setting 本身早已是按 test 名义双优事后挑选
   （见 `PhaseFormer_rank_sweep_conditioned_experiment.md` §2），且在上一轮已
   逐档读过一次 test。本轮沿用同一批 setting、同一批 (gate_init, lr) 冻结配置，
   新增对 `smooth_ratio` 维度的一次性 test 读取——不是新的盲测数据，但同样不得
   表述为无偏泛化估计。
3. **单 seed 2021**，与两轮 rank sweep 一致；结果为 test-exposed 探索性证据。
4. 本实验**不修改任何 preset 默认值**；无论结果如何，`smooth_ratio` 不会因本轮
   结果被写入默认配置（除非用户后续另行要求并满足金标准/多 seed 复核条件）。

---

## 3. 逐 setting 冻结配置

沿用 `PhaseFormer_rank_sweep_conditioned_experiment.md` 表 1（Stage 0 冻结的
gate_init/lr）与表 2/表 5（Stage 1 test 最优 q，本轮由用户指定采用）：

| Setting | H | gate_init | learning_rate | 采用的 test 最优 q | rank r = round(q×H) |
|---|---:|---:|---:|---:|---:|
| ETTh2-96 | 96 | 0.5 | 默认（1e-3） | 1/8 | 12 |
| ETTh2-720 | 720 | 0.5 | 默认（1e-3） | 1 | 720 |
| ETTm2-96 | 96 | 0.5 | 3e-4 | 1 | 96 |
| ETTm2-192 | 192 | 0.2 | 默认（1e-3） | 1/8 | 24 |
| Weather-96 | 96 | 0.2 | 3e-4 | 1 | 96 |
| Weather-192 | 192 | 0.5 | 默认（1e-3） | 1/32 | 6 |
| Electricity-336 | 336 | 0.5 | 默认（1e-3） | 1/4 | 84 |

`pool_factor=1`、`smooth_window=24` 全局固定。其余训练协议（L720→H、period 24、
Huber loss、≤30 epochs、best-val checkpoint、`--require-cuda`）与两轮 rank sweep
一致。

---

## 4. 平滑因子网格

`smooth_ratio ∈ {0, 0.25, 0.5, 0.75, 1.0}`（5 档）。`s=0` 档**重新训练**，不复用
历史任何 run——本轮 (gate_init, lr) 是逐 setting 冻结值，不能保证与历史某个
`s=0` run 的配置完全一致，重新训练以确保配对可比。

7 setting × 5 档 = **35 runs**，单 seed，每个 config 直接一次性 `--evaluate-test`
（与 Stage 1 做法一致；这些 setting 已被多轮读过 test，本轮不存在"首次读 test"
的额外净新增暴露顾虑，故不再加验证集预筛阶段）。

---

## 5. 判定规则（预注册，回填结果前不得修改）

对每个 setting，以该 setting 的 `smooth_ratio=0` 结果为参照，计算各非零档位的
`ΔMSE% = (ref − new) / ref × 100`、`ΔMAE%` 同理（正值 = 该档优于 `s=0`）。

1. **单 setting 判定**：若至少一个非零档位在 MSE 与 MAE 上同时同向且 `|Δ| ≥ 1%`，
   记为该 setting"检测到平滑效应"（方向不限——变好或变差都算检测到效应，效应
   方向另行在结论中报告，不得只挑正向的档位而忽略负向档位同样满足条件的情形）。
2. **跨 setting 计数**：
   - `≥5/7` 检测到效应 → 判定"平滑因子对该结构有可检测影响"；
   - `3–4/7` → "部分信号"；
   - `≤2/7` → "无可检测效应"。
3. **最优档位聚集性**：额外统计 7 个 setting 各自 test 最优 `smooth_ratio`
   （MSE 最低，并列取 MAE 更低）的分布；若明显聚集于同一档，可指示性讨论"是否
   存在偏好的平滑强度"，但仍受第 2 节的披露约束，不作为可泛化结论。
4. **止损条款**：本轮结果不触发 preset 修改；若判定为"无可检测效应"或"部分
   信号"，不追加 seed 或维度。若判定为"一致效应"（≥5/7）且用户希望进一步验证，
   需征得用户同意后才追加 seed/setting。

---

## 6. 结果

> 待训练完成后回填：表 1（各 setting × 5 档 test MSE/MAE）、表 2（各档相对
> `s=0` 的 ΔMSE%/ΔMAE%）、表 3（判定汇总）、结论、复现命令与产物位置。

---

## 7. 复现

### 7.1 运行命令模板

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_smooth_ratio_sweep.py \
  --dataset ETTh2 --horizon 96 --seed 2021 \
  --rank 12 --gate-init 0.5 \
  --smooth-ratios 0,0.25,0.5,0.75,1.0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/smooth_ratio_sweep_v1
```

`--learning-rate 0.0003` 在需要覆盖数据集默认 lr 的 setting（ETTm2-96、
Weather-96）上追加；其余 setting 使用数据集默认 lr，不传该参数。逐 setting 的
具体命令按第 3 节冻结配置表填入 `--rank`/`--gate-init`/`--learning-rate`。

### 7.2 结果与产物位置

- 每个 setting 一份 `research_runs/smooth_ratio_sweep_v1/smooth_sweep_<Dataset>_h<H>_s2021_results.csv`。
- `research_runs/` 已 gitignore，产物仅本地/服务器保留，不入库。

### 7.3 复现环境

- 服务器：8× A800-80GB，conda env `time`
  （`/home/yyk/yyk03/miniconda3/envs/time/bin/python`）。
- 同步规范：见 `REMOTE_SERVER.md`。

---

## 8. 关联文档

- **上一轮压缩比扫描**：[`PhaseFormer_rank_sweep_conditioned_experiment.md`](PhaseFormer_rank_sweep_conditioned_experiment.md)
- **机制前身（原始 pooled low-rank 筛选）**：[`PhaseFormer_pooled_lowrank_nlinear_experiment.md`](PhaseFormer_pooled_lowrank_nlinear_experiment.md)
- **金标准参照**：[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)
- **操作记录**：[`agent-log.md`](agent-log.md)
