# Conditioned Low-Rank Rank Sweep 计划（第二轮，用户指定）

> 状态：**planned only，2026-09-11。方案先行登记，Stage 0/1 均未运行。**
> 用户于 2026-09-11 指定：围绕第一轮 sweep 中"名义双指标优于 Golden"的
> 7 个 setting 重新做低秩扫描，以核查第一轮 sweep 的配置合理性。
> 设计经确认：两阶段（Stage 0 配置核对 → Stage 1 冻结配置扫描），
> period_len 一律保持 24。

## 1. 背景与动机

第一轮 sweep（`docs/PhaseFormer_joint_lowrank_rank_sweep_plan.md`）在全
setting 上未发现一致低秩效应。用户质疑部分 setting 的参数配置不当，要求
在以下 7 个"名义双指标优于 Golden"的 setting 上重扫：

```text
ETTh2-96, ETTh2-720, ETTm2-96, ETTm2-192,
Weather-96, Weather-192, Electricity-336
```

## 2. 谱系与披露（不可省略）

1. **Test-set selection**：这 7 个 setting 是按 test 名义双优挑选的，
   本轮全部结果只能表述为"在名义双优 setting 上的条件性扫描"，不得称
   盲测或无偏泛化。
2. **循环性**：其中 5 个（ETTh2-96、ETTm2-96/192、Weather-96/192）的
   最好成绩来自第一轮 sweep 自带的 `direct_nlinear` 对照——即该基线本就
   在被核查的配置下胜过 Golden。ETTh2-720 与 Electricity-336 从未进过
   sweep（当时 runner 限定 H96/192 且无 Electricity），其"双优"来自
   residual topology benchmark（不同 runner/协议），本轮为首次真正的
   sweep 覆盖。
3. 单 seed 2021，结果为 test-exposed 探索性证据。

## 3. 第一轮实际配置（被核查对象）

| 项 | 值 | 来源 |
|---|---|---|
| 残差 gate | 静态 per-channel sigmoid，init **0.2** | preset `weak_residual` |
| lr | 数据集默认，7 个 setting 均为 **1e-3**（服务器实测确认） | `build_hyperparams(original)` |
| period_len | 24（本轮确认保持） | runner 固定 |
| batch | ETT 256 / Weather 64 / Electricity 64 | `PLANNED_BATCH_SIZE` |
| 训练 | Huber、30 epochs 上限、best-val checkpoint、seed 2021 | runner 固定 |

可核查假设：(a) gate_init=0.2 压制残差支路（对照：RCRF 的 alpha_0=0.5
且初始 s=2.0 是唯一稳定赢 Golden 的形态）；(b) lr 1e-3 是否偏大。
period_len 不在本轮变量范围内（已确认保持 24）。

## 4. Stage 0 — 配置核对（validation-only，不读 test）

- 对每个 setting 训练 `direct_nlinear`（未因子化 shared 头）× 网格
  **{gate_init 0.2, 0.5} × {lr 默认(1e-3), 3e-4}** = 4 runs/setting，
  共 **28 runs**。
- 其余协议与第一轮一致（L720、period 24、Huber、30 epochs、best-val
  checkpoint、seed 2021）。
- **冻结规则（预注册）**：按 validation MSE 最低冻结该 setting 的
  (gate_init, lr)；MSE 并列时取 MAE 更低者。全部 4 配置的 val 两指标
  记录在案（表 1）。

## 5. Stage 1 — 冻结配置下的低秩扫描（一次读 test）

- 每个 setting 使用 Stage 0 冻结的 (gate_init, lr)，训练
  **7 个 config**：`phase_only`、`direct_nlinear`、
  `q ∈ {1, 1/4, 1/8, 1/16, 1/32}`（pool=1、smooth=0），共 **49 runs**。
- 秩映射 `r = max(1, min(H, round(q × H)))`（pool=1 时 max_rank=H）。
  H720/H336 存在非整除档位，四舍五入结果钉死如下（Python banker's
  rounding）：

| q | r @ H96 | r @ H336 | r @ H720 |
|---|---:|---:|---:|
| 1 | 96 | 336 | 720 |
| 1/4 | 24 | 84 | 180 |
| 1/8 | 12 | 42 | 90 |
| 1/16 | 6 | 21 | 45 |
| 1/32 | 3 | 10（10.5→10） | 22（22.5→22） |

  head 参数量（encoder 720→r + decoder r→H）：H336 满秩 355,488、
  q=1/32 10,906；H720 满秩 1,038,240、q=1/32 32,422。

## 6. 分析与判定规则（预注册，回填前不得修改）

1. **配置问题判定**：统计冻结配置 ≠ 第一轮配置（gate 0.2 + lr 1e-3）
   的 setting 数；并比较冻结配置下 `direct_nlinear` 的 test 结果相对
   第一轮 direct 的位移。若多数 setting 冻结了不同配置且 test 位置
   移动 >1%，则第一轮"配置不当"的质疑成立。
2. **低秩效应判定（在冻结配置下）**：各 q<1 档位相对 `direct_nlinear`
   的配对 ΔMSE/ΔMAE——方向一致（双指标同时为正 = 优于 direct）且
   |Δ| 中位 ≥1% 的 setting 计数：**≥5/7** → 一致信号；**3–4/7** →
   部分信号；**≤2/7** → 无可检测效应。
3. 对 Golden 的 Δ 仅作指示性披露（test-set selection + 单 seed），
   措辞遵守金标准规则 §4。
4. 无信号时的止损：不追加 seed/维度；低秩瓶颈继续不引入 preset。
5. 若出现一致信号，后续以多 seed（2021/2022/2023）在信号最强的
   2 个 setting 复核，复核通过前不作结构性结论。

## 7. 准备事项与工程改动

1. **数据**：electricity.csv 已上传至服务器
   `resources/all_datasets/electricity/`（2026-09-11）。
2. **Runner 改动**（commit 随本方案）：
   - `run_joint_pooled_lowrank_phase_a.py`：`--dataset` 增加
     `Electricity`；`--horizon` 放开至 {96, 192, 336, 720}；
   - 新增 `run_rank_sweep_stage0_config_check.py`（Stage 0 网格 +
     冻结，validation-only）。
3. 服务器同步后先跑 `pytest tests/ -q` 与 Electricity 1-epoch smoke。

## 8. 运行命令模板

```bash
# Stage 0（validation-only，28 runs）
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_rank_sweep_stage0_config_check.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/rank_sweep_2_stage0

# Stage 1（按 frozen_configs.json 逐 setting 传 --overrides，49 runs）
# 例（假设 ETTh2-96 冻结 g0.5_lr0.0003）：
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_joint_pooled_lowrank_phase_a.py \
  --dataset ETTh2 --horizon 96 --seed 2021 \
  --pool-factors 1 --relative-ranks 1,0.25,0.125,0.0625,0.03125 \
  --exact-rank --evaluate-test \
  --overrides '{"weak_period_residual_gate_init": 0.5, "learning_rate": 0.0003}' \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/rank_sweep_2_stage1
```

## 9. 待填充结果表格

> 约定：`MSE / MAE`；Δ% = (ref − new)/ref × 100，正为优于参照。

### 表 1：Stage 0 配置网格（validation MSE/MAE，7 setting × 4 配置）

| Setting | g0.2_lr1e-3 | g0.2_lr3e-4 | g0.5_lr1e-3 | g0.5_lr3e-4 | 冻结配置 |
|---|---|---|---|---|---|
| ETTh2-96 | — | — | — | — | — |
| ETTh2-720 | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — |
| Weather-96 | — | — | — | — | — |
| Weather-192 | — | — | — | — | — |
| Electricity-336 | — | — | — | — | — |

### 表 2：Stage 1 主结果矩阵（test MSE/MAE，seed 2021，冻结配置）

| Setting | phase_only | direct_nlinear | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|---|
| ETTh2-96 | — | — | — | — | — | — | — |
| ETTh2-720 | — | — | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — | — | — |
| Weather-96 | — | — | — | — | — | — | — |
| Weather-192 | — | — | — | — | — | — | — |
| Electricity-336 | — | — | — | — | — | — | — |

### 表 3：相对 Golden 的 ΔMSE% / ΔMAE%（指示性）

| Setting | Golden MSE/MAE | direct_nlinear | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|---|
| ETTh2-96 | 0.275 / 0.338 | — | — | — | — | — | — |
| ETTh2-720 | 0.402 / 0.436 | — | — | — | — | — | — |
| ETTm2-96 | 0.163 / 0.256 | — | — | — | — | — | — |
| ETTm2-192 | 0.219 / 0.293 | — | — | — | — | — | — |
| Weather-96 | 0.148 / 0.195 | — | — | — | — | — | — |
| Weather-192 | 0.193 / 0.237 | — | — | — | — | — | — |
| Electricity-336 | 0.165 / 0.257 | — | — | — | — | — | — |

### 表 4：压缩效应（各 q 档位相对 direct_nlinear，ΔMSE% / ΔMAE%）

| Setting | q=1 vs direct | q=1/4 vs direct | q=1/8 vs direct | q=1/16 vs direct | q=1/32 vs direct |
|---|---|---|---|---|---|
| ETTh2-96 | — | — | — | — | — |
| ETTh2-720 | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — |
| Weather-96 | — | — | — | — | — |
| Weather-192 | — | — | — | — | — |
| Electricity-336 | — | — | — | — | — |

### 表 5：判定汇总

| Setting | 冻结配置 ≠ 第一轮？ | test 最优档位 | 压缩响应形状 | 备注 |
|---|---|---|---|---|
| ETTh2-96 | — | — | — | — |
| ETTh2-720 | — | — | — | — |
| ETTm2-96 | — | — | — | — |
| ETTm2-192 | — | — | — | — |
| Weather-96 | — | — | — | — |
| Weather-192 | — | — | — | — |
| Electricity-336 | — | — | — | — |

## 10. 边界

- 单 seed、test-exposed、setting 按 test 结果挑选——本实验回答的是
  "配置核对后低秩结论是否改变"，不产生可对外声明的提升。
- q=1（满秩两层因子化）与 direct 的差异仍属因子化参数化效应，与压缩
  效应分开解读。
- 第一轮 runner 的 `--exact-rank` 规则沿用；H336/H720 的非整除档位
  按 §5 钉死的舍入执行。
