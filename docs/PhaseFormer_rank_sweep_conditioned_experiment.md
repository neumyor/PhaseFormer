# Conditioned Low-Rank Rank Sweep — 实验报告（第二轮，用户指定）

> 状态：**seed 2021 主实验已完成（2026-09-11），三 seed 复核已完成
> （2026-09-14）。** 原实验为 Stage 0 28 个 validation-only run + Stage 1
> 49 个 run；复核新增 70 个 run，使 `direct_nlinear` 与四个压缩档位在
> 7 个 setting 上均具有 seeds 2021/2022/2023。单 seed 的"部分信号（3/7）"
> 未在多 seed 下形成一致低秩收益，详见 §7。
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
| Weather-96 | 0.3885 / 0.2700 | **0.3868 / 0.2713** | 0.3910 / 0.2721 | 0.3904 / 0.2745 | g0.2_lr0.0003 |
| Weather-192 | 0.4470 / 0.3097 | 0.4472 / 0.3134 | **0.4453 / 0.3109** | 0.4456 / 0.3117 | g0.5_lrdefault |
| Electricity-336 | 0.1376 / 0.2290 | 0.1374 / 0.2280 | 0.1374 / 0.2274 | 0.1375 / 0.2278 | g0.5_lrdefault |

加粗为各 setting 最优格。**读法**：gate_init 的影响远大于 lr（4 个 setting 最优
格来自 gate 0.5，无一格由 lr 单独决定）；7 个 setting 中 **5 个**冻结出 ≠ 第一轮
配置；但同 setting 内 4 配置 val MSE 极差全部 ≤0.9%，与单 seed 噪声同阶。

> **勘误（2026-09-15，本次文档审计发现并修正）**：原表 1 的 **Weather-96 与
> Electricity-336 两行 MAE 被互换**（MSE 列无误）。已按
> `research_runs/rank_sweep_2_stage0/stage0_<Dataset>_<H>_validation.csv` 的原始
> 逐配置值还原：Weather-96 的 MAE 应为 `0.2700/0.2713/0.2721/0.2745`，
> Electricity-336 应为 `0.2290/0.2280/0.2274/0.2278`。独立旁证：Stage 1 CSV 中
> 冻结配置下 `direct_nlinear` 的 `val_mae` 为 Weather-96 `0.2713`、
> Electricity-336 `0.2274`，与还原后的值一致。**冻结配置与所有下游结果不受影响**
> （两行的 MSE 列正确，两个 setting 的最优格因此不变）。

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
2. **低秩压缩判为"部分信号"，未达一致。**（⚠️ **§1–§5 为 seed 2021 单 seed 主实验的结论；
三 seed 复核已修订，一律以 §7 为准**：3/7 未复现，结论为"中等压缩近中性、深压缩偏害、
无统一最优秩"。） 支持压缩者 **3/7**：ETTh2-720（q=1，
   +1.46%）、ETTm2-192（q=1/8，+1.02%）、Weather-192（q=1/32，+0.52%）。
   落在 3–4/7 区间 → **部分信号**，而非"一致低秩效应"（需 ≥5/7）。
3. **最优档位高度分散**（q=1 ×3、q=1/8 ×2、q=1/4 ×1、q=1/32 ×1），本身即噪声
   主导的证据，不支持"存在通用压缩档位"。
4. **压缩在该配置族下整体偏有害。**（⚠️ **单 seed 结论；三 seed 下修订为"中等压缩近中性、
深压缩偏害"，见 §7.4 第 4 条。**） 表 4 中 ΔMSE% 负值（劣化）占比高于正值，
   尤其 ETTm2-96（全程 -0.87% 至 -2.30%）、Electricity-336（全程负）、
   ETTh2-96（q≥1/4 后转负）。
5. **与第一轮结论一致。** 第一轮"无一致低秩效应、因子化本身近中性、NLinear 支线
   数据集族方向在所有秩档下保持"三点判断**均未被推翻**。用户"配置不对"的假设
   方向性成立（5/7 配置确实非最优），但对低秩结论的实质影响不成立。
6. **seed 2021 的 direct 观察。** `direct_nlinear`（不压缩）在原 Stage 1 的
   7/7 setting 上双指标优于 Golden（表 3 首列全为正值）。这只是单 seed、
   test-selected 的历史观察；三 seed 下的低秩复核结论以 §7 为准。

---

## 6. 边界与披露（不可省略）

- **Test-set selection**：7 个 setting 按 test 名义双优事后挑选，本报告全部结论
  只能表述为"在名义双优 setting 上的条件性扫描"，**绝非**盲测或无偏泛化估计。
- §1–§6 的原主实验为 **单 seed 2021**；§7 已追加 seeds 2022/2023，但仍为
  test-exposed 探索性证据，不作为无偏提升声明对外引用。
- q=1（满秩两层因子化）与 direct 的差异属**因子化参数化效应**，与**压缩效应**
  （q<1 vs q=1）需分开解读。
- 本实验**不改变**任何 preset 默认值；低秩瓶颈未引入 preset（符合计划文档 §6 止损条款）。
- 无信号情形下不追加 seed / 维度。

---

## 7. 三 seed 低秩复核（2026-09-14）

### 7.1 范围与审计口径

- 新增 **70 个 run**：7 setting × seeds {2022, 2023} × 5 个配置。
- 每个 setting 的 5 个配置为真正的 `direct_nlinear` 与
  `q ∈ {1/4, 1/8, 1/16, 1/32}`。`direct_nlinear` 明确定义为
  `X - X_last → Linear(720, H) → + X_last`；不再把两层因子化的 `q=1`
  当作 direct 对照。
- `q = rank / H`，不是相对输入长度 720 的压缩率。实际 rank 为：
  H96 `{24,12,6,3}`，H192 `{48,24,12,6}`，H336 `{84,42,21,10}`，
  H720 `{180,90,45,22}`。
- 沿用 seed 2021 Stage 0 冻结的逐 setting gate/lr；不重新训练
  `phase_only`，PhaseFormer 参照继续使用 Golden。
- seed 2021 从原 Stage 1 读取 35 个目标单元，seeds 2022/2023 从新增运行读取
  70 个单元。严格审计要求 7 × 3 × 5 = **105 个且每格恰好一个**完整结果；
  最终审计为 **105/105，无缺失、无重复**。
- 结果仍是 test-exposed：7 个 setting 来自既有 test 结果筛选，不能解释为
  无偏泛化估计。

### 7.2 三 seed 均值与标准差

下表为 test `MSE ± sample std / MAE ± sample std`：

| Setting | direct | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|
| ETTh2-96 | 0.273218±0.002951 / 0.333378±0.001410 | 0.272857±0.001110 / 0.334981±0.000403 | 0.272114±0.000339 / 0.334673±0.000097 | 0.272973±0.001615 / 0.335183±0.001537 | 0.275656±0.003260 / 0.336748±0.001784 |
| ETTh2-720 | 0.392474±0.001823 / 0.428511±0.001861 | 0.389231±0.001428 / 0.426862±0.000828 | 0.388369±0.001906 / 0.426659±0.001129 | 0.390217±0.002090 / 0.428194±0.000986 | 0.391004±0.002853 / 0.428545±0.001057 |
| ETTm2-96 | 0.159066±0.000516 / 0.248492±0.000472 | 0.160671±0.000265 / 0.250665±0.000374 | 0.160703±0.000308 / 0.251034±0.000678 | 0.160836±0.000808 / 0.251772±0.000460 | 0.161237±0.000770 / 0.252518±0.001235 |
| ETTm2-192 | 0.214788±0.000776 / 0.287837±0.000224 | 0.214712±0.000354 / 0.288913±0.000765 | 0.214706±0.001089 / 0.289014±0.001446 | 0.217465±0.002355 / 0.291235±0.000918 | 0.214821±0.001544 / 0.290285±0.000783 |
| Weather-96 | 0.146512±0.000446 / 0.193945±0.000086 | 0.148459±0.000250 / 0.196065±0.000739 | 0.147108±0.000534 / 0.194986±0.001560 | 0.147116±0.000332 / 0.194424±0.001333 | 0.148927±0.000713 / 0.196814±0.000670 |
| Weather-192 | 0.192146±0.001274 / 0.237048±0.001068 | 0.191536±0.000753 / 0.236621±0.000935 | 0.190976±0.000340 / 0.235400±0.000932 | 0.191941±0.002406 / 0.236834±0.002623 | 0.191834±0.000902 / 0.238340±0.001913 |
| Electricity-336 | 0.162455±0.000674 / 0.255757±0.001111 | 0.162114±0.000250 / 0.255193±0.000287 | 0.162735±0.000159 / 0.255931±0.000410 | 0.162817±0.000071 / 0.256018±0.000067 | 0.162876±0.000395 / 0.256148±0.000885 |

### 7.3 相对同 seed direct NLinear 的配对结果

下表为三 seed 配对 `ΔMSE% ± std / ΔMAE% ± std`，正值表示压缩头更好；
括号为两个指标同时优于 direct 的 seed 数。

| Setting | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|
| ETTh2-96 | +0.124±1.219 / -0.482±0.541 (1/3) | +0.397±0.995 / -0.390±0.435 (1/3) | +0.082±1.236 / -0.542±0.597 (0/3) | -0.908±2.234 / -1.013±0.960 (1/3) |
| ETTh2-720 | +0.824±0.823 / +0.383±0.581 (3/3) | +1.046±0.125 / +0.432±0.180 (3/3) | +0.573±0.785 / +0.073±0.416 (1/3) | +0.373±0.798 / -0.009±0.525 (1/3) |
| ETTm2-96 | -1.009±0.275 / -0.875±0.108 (0/3) | -1.029±0.142 / -1.023±0.208 (0/3) | -1.114±0.837 / -1.320±0.359 (0/3) | -1.366±0.813 / -1.621±0.676 (0/3) |
| ETTm2-192 | +0.035±0.410 / -0.374±0.194 (0/3) | +0.036±0.863 / -0.409±0.579 (1/3) | -1.249±1.445 / -1.181±0.397 (0/3) | -0.014±0.367 / -0.850±0.269 (0/3) |
| Weather-96 | -1.330±0.414 / -1.093±0.363 (0/3) | -0.407±0.060 / -0.537±0.798 (0/3) | -0.413±0.250 / -0.247±0.731 (0/3) | -1.648±0.358 / -1.480±0.358 (0/3) |
| Weather-192 | +0.314±0.970 / +0.178±0.714 (1/3) | +0.607±0.518 / +0.693±0.784 (2/3) | +0.105±1.335 / +0.087±1.361 (2/3) | +0.160±0.721 / -0.545±0.575 (1/3) |
| Electricity-336 | +0.209±0.406 / +0.219±0.451 (2/3) | -0.174±0.511 / -0.069±0.472 (1/3) | -0.224±0.415 / -0.103±0.431 (1/3) | -0.260±0.406 / -0.154±0.558 (1/3) |

跨全部 21 个 setting-seed 配对单元，四个压缩档汇总如下：

| q | 宏平均 ΔMSE% | 宏平均 ΔMAE% | 双指标胜 direct 的单元 | 三 seed 均值双优的 setting |
|---|---:|---:|---:|---:|
| 1/4 | -0.119 | -0.292 | 7/21 | 3/7 |
| 1/8 | +0.068 | -0.186 | 8/21 | 2/7 |
| 1/16 | -0.320 | -0.462 | 4/21 | 2/7 |
| 1/32 | -0.523 | -0.810 | 4/21 | 0/7 |

### 7.4 多 seed 结论

1. **单 seed 的局部最优确实受随机性影响。** seed 2021 中
   ETTm2-192 `q=1/8` 与 Weather-192 `q=1/32` 的双指标优势没有稳定复现：
   前者三 seed 均值为 `+0.036% / -0.409%`，后者为
   `+0.160% / -0.545%`。因此原来的 3/7 "部分信号"不能升级为稳定结论。
2. **没有跨 setting 的统一低秩增益。** 四个 q 中没有一个在超过 3/7 个
   setting 的三 seed 均值上双指标优于 direct；按逐 seed 单元计，最高也只有
   `q=1/8` 的 8/21。
3. **存在一个局部、可复现的例外。** ETTh2-720 的 `q=1/4` 与 `q=1/8`
   均在 3/3 seeds 上双指标优于 direct，其中 `q=1/8` 的平均收益为
   `+1.046% MSE / +0.432% MAE`。这是 setting-specific 信号，不足以证明
   通用低秩机制。
4. **更准确的总体描述是"中等压缩近中性，深压缩偏害"。** `q=1/4` 与
   `q=1/8` 的宏平均变化接近 0，但 MAE 均略差；压到 `q=1/16`、`q=1/32`
   后两指标均转为负，且 `q=1/32` 在 0/7 个 setting 上实现三 seed 均值双优。
   各 setting 的四个压缩档均值曲线跨度约为 MSE 0.36%–1.30%、MAE
   0.37%–1.24%，所以秩效应总体小，但并非整个扫描区间都严格性能中性。
5. **随机性会遮蔽弱趋势，但不是没有趋势的唯一原因。** 三 seed 降低了单点偶然性，
   揭示出压缩加深时整体逐渐变差的弱趋势；与此同时，效应量通常与 seed 标准差同阶，
   且最优 q 仍随 setting/指标变化。因此证据支持“低秩不是普适增益，过深压缩略有害”，
   不支持“存在一个统一最佳压缩比”。

### 7.5 复核产物

- 严格审计与汇总脚本：
  [`../scripts/analyze_conditioned_rank_sweep_multiseed.py`](../scripts/analyze_conditioned_rank_sweep_multiseed.py)
- seeds 2022/2023 有效运行：
  `research_runs/rank_sweep_2_multiseed_stage1_20260914_v4/`（58 个）与
  `research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1/`（12 个）。
- 汇总目录：
  `research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/`，包含
  `audited_results.csv`、`three_seed_summary.csv`、
  `three_seed_setting_diagnostics.csv`、`three_seed_aggregate.csv` 与
  `three_seed_summary.md`。
- `research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/` 因调度/结果完整性
  问题明确排除，不参与任何统计。

### 7.6 三 seed 图表（MSE / MAE 分图）

绘图脚本
[`../scripts/plot_3seed_conditioned_rank_sweep.py`](../scripts/plot_3seed_conditioned_rank_sweep.py)
读取 `three_seed_summary.csv`（3 seed 均值与样本标准差）与 `audited_results.csv`
（105 个逐 seed 单元，仅用于背景散点），Golden 基准取
[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)。产物位于
`research_runs/rank_sweep_2_multiseed_stage1_20260914_summary/figures/`：

| 文件 | 内容 |
|---|---|
| `three_seed_MSE_by_setting.png` | 7 setting × 1 指标（MSE）分面图；x 轴为压缩档位 `direct → q=1/32`（标注实际 rank），y 轴为 test MSE，折线为 3 seed 均值，半透明带为均值 ±1 样本标准差，虚线为该 setting 的 Golden MSE 基准 |
| `three_seed_MAE_by_setting.png` | 同上，指标为 test MAE 与 Golden MAE |
| `three_seed_<Dataset>_h<H>.png` | 逐 setting 双面板（左 MSE、右 MAE），共 7 张 |
| `three_seed_figure_data.csv` | 图中全部数值的审计副本（均值、标准差、rank、Golden、seed 数） |

重绘命令：`python scripts/plot_3seed_conditioned_rank_sweep.py`（`research_runs/`
在 `.gitignore` 内，图表为本地产物；入库的是脚本与本报告）。

**图读法（对 Golden，仅供参考）**：把 5 个档（含 `direct`）的 3 seed 均值与 Golden
逐 setting 对比，7 × 5 = 35 个档位单元中有 30 个在 MSE 与 MAE 上同时低于 Golden，
未达标者为 ETTh2-96 `q=1/32`、Weather-96 `q=1/4` 与 `q=1/32`、Weather-192 `direct`
与 `q=1/32`。多数差距（约 0.3%–3%）与 seed 标准差同阶，且本表与图同样受
test-set selection 约束，**只能作为条件性、test-exposed 的图示**，不构成提升声明。

**与 §5.6「direct 7/7 双指标优于 Golden」的关系（避免误读）**：该 7/7 是 seed 2021
单 seed 读法（表 3 首列）。三 seed 均值下 `direct` 仍然 **7/7 在 MSE 上优于 Golden**，
MAE 为 **6/7 严格优于 + 1 个舍入级持平**：Weather-192 `direct` 为
`0.237048 ± 0.001068` vs Golden `0.237`（ΔMAE = −0.02%，且 3 个 seed 中 2 个仍双优），
属于金标准仅保留三位小数导致的舍入差异，按
[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md) §4 不应计为退化。
30/35 中未通过的 5 个单元里，4 个出现在压缩档（尤其 H96 的 `q=1/32` 与 Weather-96
的 `q=1/4`），即压缩带来的相对退化；这**不改变**“不压缩的 `direct` 在该 7 个 setting
上超过 Golden”这一既有观察。

---

## 8. 复现与产物

### 8.1 运行命令

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

### 8.2 结果与图表位置

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

### 8.3 复现环境

- 服务器：8× A800-80GB，conda env `time`
  （`/home/yyk/yyk03/miniconda3/envs/time/bin/python`），单 GPU `CUDA_VISIBLE_DEVICES=N`。
- 同步规范：见 `REMOTE_SERVER.md`（代码 local→server 走 git bundle；远端
  `git status --porcelain` 必须为空，`git reset --hard FETCH_HEAD`，**禁止**
  `git merge FETCH_HEAD`；产 server→local 走 `rsync --partial --progress`）。

---

## 9. 关联文档

- **预注册计划与判定规则**：[`PhaseFormer_rank_sweep_conditioned_plan.md`](PhaseFormer_rank_sweep_conditioned_plan.md)
- **第一轮全矩阵 sweep**：[`PhaseFormer_joint_lowrank_rank_sweep_plan.md`](PhaseFormer_joint_lowrank_rank_sweep_plan.md)
- **金标准参照**：[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)
- **机制前身**：[`PhaseFormer_pooled_lowrank_nlinear_experiment.md`](PhaseFormer_pooled_lowrank_nlinear_experiment.md)
- **操作记录**：[`agent-log.md`](agent-log.md)（2026-09-11 条目）
