# Conditioned Low-Rank Rank Sweep — 实验报告（第二轮，用户指定）

> 状态：**已完成并回填（2026-09-11）。** 7 setting × 7 档位，共 77 个 run
> （Stage 0 28 个 validation-only + Stage 1 49 个），seed 2021，单次读 test。
> 判定按预注册规则得出 **"部分信号（3/7）"**，未推翻第一轮"无一致低秩效应"。
>
> 本文件是**结果登记报告**（数值权威副本），完整预注册计划、判定规则与运行
> 命令见 [`PhaseFormer_rank_sweep_conditioned_plan.md`](PhaseFormer_rank_sweep_conditioned_plan.md)。
> 指标数值以本文档为准；如需追溯规则推导与冻结依据，回看计划文档。

---

## 1. 所测试的模型（务必先读）

本实验测试的是 **PhaseFormer + NLinear 弱周期残差支路** 结构，**不含 RCRF**，
**不含周期位置编码**。它是保留四结构（K1–K4）之外的**机制消融实验**，不属于
incumbent 结构，也不产生可对外声明的提升。

### 1.1 组成与数据流

| 部件 | 实现 | 说明 |
|---|---|---|
| 相位主干 | 原始 PhaseFormer 相位路由路径 | 论文固定结构，见 K1 |
| 残差支路 | `WeakPeriodResidualHead` / `PooledLowRankWeakPeriodResidualHead` | `src/models/phase_adapters.py` |
| 融合 | 静态 per-channel sigmoid gate（**无 RCRF**） | `torch.sigmoid(self.weak_period_residual_gate)` |

- 残差支路的 NLinear 语义：`X - X_last → （pool）→ rank-r 因子化 → horizon → + X_last`，
  即 last 值锚点保持在动态输入之外。
- 融合式（非 RCRF 路径，`src/models/PhaseFormer.py`）：
  `y_hat = (1 - gate) * y_hat_phase + gate * y_hat_residual`，
  其中 `gate = sigmoid(weak_period_residual_gate)`，`gate_init` 由
  `weak_period_residual_gate_init` 给出（本轮变量之一）。

### 1.2 两种残差头形态

| 头 | `weak_period_residual_head_type` | 参数化 |
|---|---|---|
| 未因子化 shared | `shared`（= `direct_nlinear` 对照） | 单层 `Linear(720 → H)` |
| 池化低秩因子化 | `pooled_lowrank` | `encoder: Linear(pooled_len → rank)` + `decoder: Linear(rank → pred_len)`，decoder 零初始化 |

本轮固定 `pool_factor=1`、`smooth_ratio=0`，**只改相对秩 q**，因此
`pooled_len = 720`、`max_rank = min(ceil(720/1), H) = H`、`q = rank / H`。

### 1.3 冻结的共享超参

| 项 | 值 |
|---|---|
| lookback → horizon | 720 → H（H ∈ {96, 192, 336, 720}） |
| period_len | 24（固定，非本轮变量） |
| 损失 | Huber |
| epochs | ≤30，best-val checkpoint |
| batch | ETT 256 / Weather 64 / Electricity 64 |
| seed | 2021（单 seed） |
| lr | Stage 0 冻结后逐 setting 固定（表 1） |
| `weak_period_residual_gate_init` | Stage 0 冻结后逐 setting 固定（表 1） |

---

## 2. 所测试的 7 个 setting 与选择披露（不可省略）

本轮只在以下 **7 个 setting** 上扫描：

```text
ETTh2-96, ETTh2-720, ETTm2-96, ETTm2-192,
Weather-96, Weather-192, Electricity-336
```

**为什么是这 7 个（test-set selection）**：它们是在第一轮以 **test 名义双指标
优于 Golden** 挑选出来的。因此本报告全部结果只能表述为"在名义双优 setting 上的
**条件性扫描**"，**不得**称盲测或无偏泛化。

**循环性提示**：其中 5 个（ETTh2-96、ETTm2-96/192、Weather-96/192）的最好成绩
本就来自第一轮 sweep 自带的 `direct_nlinear` 对照；ETTh2-720 与 Electricity-336
从未进过 sweep（当时 runner 限定 H96/192 且无 Electricity），其"双优"来自
residual topology benchmark（不同 runner/协议），本轮为首次真正的 sweep 覆盖。

**第一轮"配置不当"假设**：第一轮用 preset 默认 `gate_init=0.2` + lr 1e-3。
本轮用 Stage 0 重新核对配置（表 1），检验该假设。

---

## 3. 协议

两阶段，**Stage 0 validation-only 冻结配置，Stage 1 一次读 test**。

- **Stage 0**：对每个 setting 训练 `direct_nlinear` × 网格
  `{gate_init 0.2, 0.5} × {lr 默认 1e-3, 3e-4}` = 4 runs/setting，共 28 runs。
  按 **validation MSE 最低**冻结 (gate_init, lr)，MSE 并列取 MAE 更低者。
- **Stage 1**：每个 setting 用冻结配置训练 **7 个 config**：
  `phase_only`、`direct_nlinear`、`q ∈ {1, 1/4, 1/8, 1/16, 1/32}`（共 49 runs），
  每个 config 只读一次 test。

秩映射 `r = max(1, min(H, round(q × H)))`，H336/H720 的非整除档位按计划文档
§5 钉死（Python banker's rounding）：

| q | r @ H96 | r @ H336 | r @ H720 |
|---|---:|---:|---:|
| 1 | 96 | 336 | 720 |
| 1/4 | 24 | 84 | 180 |
| 1/8 | 12 | 42 | 90 |
| 1/16 | 6 | 21 | 45 |
| 1/32 | 3 | 10（10.5→10） | 22（22.5→22） |

**判定规则（预注册）**：压缩档位在 MSE 与 MAE 上**同时**优于 `direct_nlinear`
才算该 setting 支持压缩；计数 **≥5/7** → 一致信号；**3–4/7** → 部分信号；
**≤2/7** → 无可检测效应。Δ% = (ref − new)/ref × 100，正为优于参照。

---

## 4. 结果

> 约定：`MSE / MAE`；Δ% 正为优于参照。

### 表 1：Stage 0 配置网格（validation MSE/MAE，7 setting × 4 配置）

| Setting | g0.2_lr1e-3 | g0.2_lr3e-4 | g0.5_lr1e-3 | g0.5_lr3e-4 | 冻结配置 |
|---|---|---|---|---|---|
| ETTh2-96 | 0.2085 / 0.3165 | 0.2078 / 0.3157 | **0.2040 / 0.3116** | 0.2044 / 0.3118 | g0.5_lrdefault |
| ETTh2-720 | 0.6189 / 0.5478 | 0.6205 / 0.5489 | **0.6095 / 0.5424** | 0.6110 / 0.5441 | g0.5_lrdefault |
| ETTm2-96 | 0.1125 / 0.2300 | 0.1126 / 0.2293 | 0.1125 / 0.2286 | **0.1118 / 0.2281** | g0.5_lr0.0003 |
| ETTm2-192 | **0.1499 / 0.2649** | 0.1503 / 0.2656 | 0.1502 / 0.2646 | 0.1500 / 0.2649 | g0.2_lrdefault |
| Weather-96 | 0.3885 / 0.2290 | **0.3868 / 0.2280** | 0.3910 / 0.2274 | 0.3904 / 0.2278 | g0.2_lr0.0003 |
| Weather-192 | 0.4470 / 0.3097 | 0.4472 / 0.3134 | **0.4453 / 0.3109** | 0.4456 / 0.3117 | g0.5_lrdefault |
| Electricity-336 | 0.1376 / 0.2700 | 0.1374 / 0.2713 | 0.1374 / 0.2721 | 0.1375 / 0.2745 | g0.5_lrdefault |

加粗为各 setting 最优格。**读法**：gate_init 的影响远大于 lr（4 个 setting 最优
格来自 gate 0.5，无一格由 lr 单独决定）；7 个 setting 中 **5 个**冻结出 ≠ 第一轮
配置；但同 setting 内 4 配置 val MSE 极差全部 ≤0.9%，与单 seed 噪声同阶。

### 表 2：Stage 1 主结果矩阵（test MSE/MAE，seed 2021，冻结配置）

| Setting | phase_only | direct_nlinear | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|---|
| ETTh2-96 | 0.2808/0.3430 | 0.2721/0.3328 | 0.2758/0.3351 | 0.2741/0.3350 | 0.2723/0.3348 | 0.2747/0.3368 | 0.2755/0.3373 |
| ETTh2-720 | 0.4254/0.4552 | 0.3909/0.4279 | 0.3852/0.4247 | 0.3905/0.4277 | 0.3871/0.4265 | 0.3922/0.4291 | 0.3928/0.4298 |
| ETTm2-96 | 0.1724/0.2638 | 0.1585/0.2480 | 0.1598/0.2494 | 0.1605/0.2505 | 0.1603/0.2503 | 0.1618/0.2523 | 0.1621/0.2539 |
| ETTm2-192 | 0.2285/0.2985 | 0.2157/0.2881 | 0.2148/0.2898 | 0.2147/0.2898 | 0.2135/0.2878 | 0.2149/0.2904 | 0.2166/0.2909 |
| Weather-96 | 0.1498/0.1969 | 0.1467/0.1940 | 0.1463/0.1948 | 0.1487/0.1969 | 0.1474/0.1966 | 0.1469/0.1936 | 0.1486/0.1973 |
| Weather-192 | 0.1939/0.2383 | 0.1918/0.2363 | 0.1910/0.2359 | 0.1923/0.2377 | 0.1911/0.2365 | 0.1945/0.2398 | 0.1908/0.2361 |
| Electricity-336 | 0.1677/0.2594 | 0.1617/0.2547 | 0.1631/0.2567 | 0.1620/0.2550 | 0.1629/0.2562 | 0.1628/0.2560 | 0.1629/0.2567 |

### 表 3：相对 Golden 的 ΔMSE% / ΔMAE%（指示性披露）

| Setting | Golden MSE/MAE | direct_nlinear | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|---|
| ETTh2-96 | 0.275 / 0.338 | +1.05/+1.53 | -0.29/+0.85 | +0.32/+0.89 | +0.98/+0.95 | +0.09/+0.36 | -0.17/+0.21 |
| ETTh2-720 | 0.402 / 0.436 | +2.76/+1.86 | +4.18/+2.60 | +2.85/+1.90 | +3.69/+2.18 | +2.44/+1.58 | +2.30/+1.43 |
| ETTm2-96 | 0.163 / 0.256 | +2.78/+3.11 | +1.93/+2.57 | +1.52/+2.14 | +1.63/+2.23 | +0.75/+1.45 | +0.55/+0.81 |
| ETTm2-192 | 0.219 / 0.293 | +1.52/+1.69 | +1.90/+1.08 | +1.97/+1.10 | +2.52/+1.77 | +1.86/+0.88 | +1.11/+0.73 |
| Weather-96 | 0.148 / 0.195 | +0.87/+0.51 | +1.14/+0.12 | -0.44/-0.97 | +0.43/-0.81 | +0.73/+0.70 | -0.39/-1.18 |
| Weather-192 | 0.193 / 0.237 | +0.63/+0.30 | +1.01/+0.46 | +0.37/-0.28 | +0.98/+0.23 | -0.79/-1.17 | +1.15/+0.36 |
| Electricity-336 | 0.165 / 0.257 | +1.98/+0.89 | +1.16/+0.13 | +1.85/+0.77 | +1.26/+0.30 | +1.31/+0.40 | +1.28/+0.10 |

Golden 数值来自 [`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)。
表 3 仅作指示性披露（受 test-set selection + 单 seed 约束），不作提升声明。

### 表 4：压缩效应（各 q 档位相对 direct_nlinear，ΔMSE% / ΔMAE%）

| Setting | q=1 vs direct | q=1/4 vs direct | q=1/8 vs direct | q=1/16 vs direct | q=1/32 vs direct |
|---|---|---|---|---|---|
| ETTh2-96 | -1.36/-0.69 | -0.75/-0.65 | -0.08/-0.58 | -0.97/-1.19 | -1.24/-1.34 |
| ETTh2-720 | +1.46/+0.75 | +0.09/+0.04 | +0.96/+0.32 | -0.33/-0.29 | -0.48/-0.44 |
| ETTm2-96 | -0.87/-0.56 | -1.30/-0.99 | -1.19/-0.90 | -2.08/-1.71 | -2.30/-2.37 |
| ETTm2-192 | +0.39/-0.62 | +0.46/-0.60 | +1.02/+0.09 | +0.35/-0.82 | -0.41/-0.98 |
| Weather-96 | +0.27/-0.39 | -1.32/-1.49 | -0.44/-1.32 | -0.14/+0.19 | -1.27/-1.71 |
| Weather-192 | +0.39/+0.16 | -0.26/-0.59 | +0.36/-0.08 | -1.42/-1.48 | +0.52/+0.06 |
| Electricity-336 | -0.83/-0.77 | -0.14/-0.11 | -0.74/-0.59 | -0.69/-0.49 | -0.72/-0.80 |

### 表 5：判定汇总

| Setting | 冻结配置 ≠ 第一轮？ | test 最优档位 | 压缩响应形状 | 备注 |
|---|---|---|---|---|
| ETTh2-96 | 是（g0.5） | q=1/8 (0.2723) | 单调劣化（无档位持平） | 双指标未胜 direct；最优档仅 -0.08% MSE |
| ETTh2-720 | 是（g0.5） | q=1 (0.3852) | 满秩最优、压缩后单调劣化 | 双指标胜 direct；满秩纯因子化收益 +1.46% |
| ETTm2-96 | 是（g0.5/lr3e-4） | q=1 (0.1598) | 随压缩单调劣化 | 双指标未胜 direct；压缩全程有害 |
| ETTm2-192 | 否（g0.2/lr1e-3） | q=1/8 (0.2135) | 先降后升，q=1/8 峰 | 双指标胜 direct（+1.02% MSE） |
| Weather-96 | 是（lr3e-4） | q=1 (0.1463) | 无规律、噪声主导 | 双指标未胜 direct |
| Weather-192 | 是（g0.5） | q=1/32 (0.1908) | 无规律、噪声主导 | 双指标胜 direct（+0.52% MSE） |
| Electricity-336 | 是（g0.5） | q=1/4 (0.1620) | 压缩全程小幅有害 | 双指标未胜 direct |

---

## 5. 结论

1. **配置核对部分证实疑虑，但不改变结论方向。** 5/7 冻结配置 ≠ 第一轮，gate_init
   是支配因素；但配置网格内 val MSE 极差 ≤0.9%（与单 seed 噪声同阶），且冻结
   配置改变后 Stage 1 相对格局未出现方向性翻转。
2. **低秩压缩判为"部分信号"，未达一致。** 支持压缩者 **3/7**：ETTh2-720（q=1，
   +1.46%）、ETTm2-192（q=1/8，+1.02%）、Weather-192（q=1/32，+0.52%）。
   落在 3–4/7 区间 → **部分信号**，而非"一致低秩效应"（需 ≥5/7）。
3. **最优档位高度分散**（q=1 ×3、q=1/8 ×2、q=1/4 ×1、q=1/32 ×1），本身即噪声
   主导的证据，不支持"存在通用压缩档位"。
4. **压缩在该配置族下整体偏有害。** 表 4 中 ΔMSE% 负值（劣化）占比高于正值，
   尤其 ETTm2-96（全程 -0.87% 至 -2.30%）、Electricity-336（全程负）、
   ETTh2-96（q≥1/4 后转负）。
5. **与第一轮结论一致。** 第一轮"无一致低秩效应、因子化本身近中性、NLinear 支线
   数据集族方向在所有秩档下保持"三点判断**均未被推翻**。用户"配置不对"的假设
   方向性成立（5/7 配置确实非最优），但对低秩结论的实质影响不成立。
6. **可对外声明的部分。** `direct_nlinear`（不压缩）在 7/7 setting 上仍双指标优于
   Golden（表 3 首列全为正值）；这是两轮实验中唯一稳健、可复现的正向结果，但
   "7 setting 按 test 挑选 + 单 seed"的披露约束仍然适用。

---

## 6. 边界与披露（不可省略）

- **Test-set selection**：7 个 setting 按 test 名义双优事后挑选，本报告全部结论
  只能表述为"在名义双优 setting 上的条件性扫描"，**绝非**盲测或无偏泛化估计。
- **单 seed 2021**，结果为 test-exposed 探索性证据；不作为提升声明对外引用。
- q=1（满秩两层因子化）与 direct 的差异属**因子化参数化效应**，与**压缩效应**
  （q<1 vs q=1）需分开解读。
- 本实验**不改变**任何 preset 默认值；低秩瓶颈未引入 preset（符合计划文档 §6 止损条款）。
- 无信号情形下不追加 seed / 维度。

---

## 7. 复现与产物

### 7.1 运行命令

Stage 0（validation-only，28 runs）：

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_rank_sweep_stage0_config_check.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/rank_sweep_2_stage0
```

Stage 1（按 `frozen_configs.json` 逐 setting 传 `--overrides`，49 runs）：

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_joint_pooled_lowrank_phase_a.py \
  --dataset ETTh2 --horizon 96 --seed 2021 \
  --pool-factors 1 --relative-ranks 1,0.25,0.125,0.0625,0.03125 \
  --exact-rank --evaluate-test \
  --overrides '{"weak_period_residual_gate_init": 0.5, "learning_rate": 0.0003}' \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/rank_sweep_2_stage1
```

### 7.2 结果与图表位置

- **原始 test 结果 CSV**（数值权威来源，recompute 一切表格）：
  `research_runs/rank_sweep_2_stage1/phase_a_<Dataset>_h<H>_s2021_results.csv`
  （7 个 setting 各一份，含 `phase_only`、`direct_nlinear`、5 个 q 档位的
  `val_mse/val_mae/test_mse/test_mae` 与 run_id）。
- **冻结配置**：逐 setting 记录于本报告表 1 与计划文档 §9 表 1（Stage 0 由
  `scripts/run_rank_sweep_stage0_config_check.py` 产出 `frozen_configs.json`；
  Stage 1 的 CSV 中 `direct_nlinear` 的 `val_mse` 与冻结格逐一吻合，可用于核对，例如
  ETTh2-96 0.204035↔0.2040、ETTm2-192 0.149915↔0.1499、Weather-96 0.386875↔0.3868）。
- **折线图**（本次生成的 7 setting × 2 指标图；MSE 与 MAE 分轴绘制，Golden 为
  各指标色虚线基准）——绘制脚本 [`../scripts/plot_conditioned_rank_sweep.py`](../scripts/plot_conditioned_rank_sweep.py)：
  - 合并总图（7 行 × 2 列）：`research_runs/rank_sweep_2_stage1/figures/compression_lines.png`
  - 逐 setting 双面板：`research_runs/rank_sweep_2_stage1/figures/compression_<Dataset>_h<H>.png`

  重绘命令：

  ```bash
  python scripts/plot_conditioned_rank_sweep.py
  ```

  > 注：`research_runs/` 在 `.gitignore` 内，图表与原始 CSV 均为本地产物，
  > 不入库；入库的是脚本与本报告。

### 7.3 复现环境

- 服务器：8× A800-80GB，conda env `time`
  （`/home/yyk/yyk03/miniconda3/envs/time/bin/python`），单 GPU `CUDA_VISIBLE_DEVICES=N`。
- 同步规范：见 `REMOTE_SERVER.md`（代码 local→server 走 git bundle；远端
  `git status --porcelain` 必须为空，`git reset --hard FETCH_HEAD`，**禁止**
  `git merge FETCH_HEAD`；产 server→local 走 `rsync --partial --progress`）。

---

## 8. 关联文档

- **预注册计划与判定规则**：[`PhaseFormer_rank_sweep_conditioned_plan.md`](PhaseFormer_rank_sweep_conditioned_plan.md)
- **第一轮全矩阵 sweep**：[`PhaseFormer_joint_lowrank_rank_sweep_plan.md`](PhaseFormer_joint_lowrank_rank_sweep_plan.md)
- **金标准参照**：[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)
- **机制前身**：[`PhaseFormer_pooled_lowrank_nlinear_experiment.md`](PhaseFormer_pooled_lowrank_nlinear_experiment.md)
- **操作记录**：[`agent-log.md`](agent-log.md)（2026-09-11 条目）
