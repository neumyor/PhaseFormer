# PhaseFormer-L 主表全 setting 复现手册

> **范围**：**主表全部 setting × 全部臂 × 全部 seed** 的复现参数与实测值，
> 而不只是定向调参的那几个 setting。主表 = §4.2 的 24 个主 setting + 4 个 Traffic 附录 setting，
> 共 6 个臂（`phase_only` / `l_main` / `l_q1_4` / `l_q1_8` / `l_rcrf` / `a1`）× 3 个 seed。
>
> **每个格子的参数都从该格自己的 `config.json` 读出，指标从同一个 run 的 `metrics.csv` 读出**
> ——不引用任何汇总表，因此本手册与产物**按构造一致**。
>
> **口径**：仍然只关心 **3 个 seed 中最优的那一次**，且以 **test 指标**选优（用户 2026-09-20 明示的 test-set selection）。

## 0. 公共训练协议（除表中另注明者外，所有格子相同）

| 项 | 值 |
|---|---|
| lookback | 720 |
| period | 24 |
| loss | huber（preset 默认；E14 协议）|
| max_epochs | 30（best-val 早停，patience 8）|
| percent | 100（full-train）|
| checkpoint | 最低 validation loss |
| 评估 | 每 checkpoint **只读一次 test** |
| 融合 | `y = (1-g)·y_phase + g·y_residual`；`g` 由 `weak_period_residual_gate_init` 初始化后可训练 |
| 输入 | `x_last` 锚点保持在动态输入之外：`z = x_n − x_n,last` |

**六个臂各自的结构**：

| 臂 | mechanism | 结构 |
|---|---|---|
| `phase_only` | `no_residual` | 原始 PhaseFormer（无残差支路）|
| `l_main` | `weak_residual` | 残差支路 = `shared` 稠密头（`Linear(720,H)`）|
| `l_q1_4` | `weak_residual` | 残差支路 = `pooled_lowrank`，`rank = H/4` |
| `l_q1_8` | `weak_residual` | 残差支路 = `pooled_lowrank`，`rank = H/8` |
| `l_rcrf` | `rcrf_nlinear_plain` | 原始相位路径 + `shared` 头 + RCRF 可靠度门（无附加校准）|
| `a1` | `gold_combo_reliability_s2` | incumbent：RCRF + 共享 NLinear + 相位校准模块 |

> **三种 gate 先验**（§4.0 已披露）：新格 `gate_init = 0.2`；`l_rcrf`/`a1` 由 preset 自持 `0.5`；
> 复用格保留其 Stage-0 冻结值。下表的 `gate_init` 是**逐格实测值**，直接取自 `config.json`。

## 1. 主表 24 个 setting：逐臂的最佳 seed 与参数

> 每个 setting 6 行（一个臂一行）。**「最佳 seed」= 该臂在 3 个 seed 中按「两指标最差缺口」最优的那一次**；
> 「3-seed 双指标胜 Golden」给出该臂在几个 seed 上 MSE 与 MAE 同时低于 Golden。
> `rank` 为 `pooled_lowrank` 的实际中间维数，`—` 表示该臂不使用低秩瓶颈。

### ETTh1-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.360815 | 0.386209 | +0.51% / +1.10% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.365555 | 0.396061 | +1.83% / +3.68% | 0/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.362588 | 0.394088 | +1.00% / +3.16% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 12 | 0.371475 | 0.404374 | +3.47% / +5.86% | 0/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.365380 | 0.395888 | +1.78% / +3.64% | 0/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.365571 | 0.396260 | +1.83% / +3.73% | 0/3 | 新训 |

### ETTh1-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.404023 | 0.409278 | +1.77% / +1.31% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.409537 | 0.420528 | +3.16% / +4.09% | 0/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.402483 | 0.418342 | +1.38% / +3.55% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.401418 | 0.417742 | +1.11% / +3.40% | 0/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.404458 | 0.417522 | +1.88% / +3.35% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.404649 | 0.417392 | +1.93% / +3.31% | 0/3 | 新训 |

### ETTh1-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.438117 | 0.431423 | +3.09% / +1.75% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.438007 | 0.436807 | +3.06% / +3.02% | 0/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.433404 | 0.433743 | +1.98% / +2.30% | 0/3 | 新训 |
| `l_q1_8` | **2023** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.436096 | 0.436842 | +2.61% / +3.03% | 0/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.430867 | 0.434684 | +1.38% / +2.52% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.433016 | 0.436231 | +1.89% / +2.88% | 0/3 | 新训 |

### ETTh1-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.418769 | 0.439297 | -2.84% / -2.38% | 3/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.418169 | 0.447131 | -2.98% / -0.64% | 2/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.421956 | 0.446828 | -2.10% / -0.70% | 3/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.424441 | 0.445578 | -1.52% / -0.98% | 2/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.422851 | 0.448740 | -1.89% / -0.28% | 2/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.419263 | 0.447049 | -2.72% / -0.66% | 3/3 | 新训 |

### ETTh2-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | 0.5 | 0.001 | — | — | 0.280834 | 0.343016 | +2.12% / +1.48% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.270989 | 0.332315 | -1.46% / -1.68% | 2/3 | 复用 |
| `l_q1_4` | **2023** | 0.5 | 0.001 | pooled_lowrank | 24 | 0.272369 | 0.334569 | -0.96% / -1.01% | 3/3 | 复用 |
| `l_q1_8` | **2022** | 0.5 | 0.001 | pooled_lowrank | 12 | 0.271723 | 0.334621 | -1.19% / -1.00% | 3/3 | 复用 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.271441 | 0.332674 | -1.29% / -1.58% | 3/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.270798 | 0.332583 | -1.53% / -1.60% | 3/3 | 新训 |

### ETTh2-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.344240 | 0.381259 | +0.95% / +1.40% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.337312 | 0.376446 | -1.08% / +0.12% | 0/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.340129 | 0.377196 | -0.26% / +0.32% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.337877 | 0.376804 | -0.92% / +0.21% | 0/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.339436 | 0.374301 | -0.46% / -0.45% | 1/3 | 新训 |
| `a1` | **2022** | 0.5 | 0.001 | — | — | 0.339349 | 0.374162 | -0.48% / -0.49% | 1/3 | 新训 |

### ETTh2-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.372822 | 0.408478 | +1.04% / +0.86% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.368725 | 0.404795 | -0.07% / -0.05% | 1/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.370009 | 0.403771 | +0.27% / -0.30% | 0/3 | 新训 |
| `l_q1_8` | **2023** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.365907 | 0.403720 | -0.84% / -0.32% | 2/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.370158 | 0.402407 | +0.31% / -0.64% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.367826 | 0.402777 | -0.32% / -0.55% | 1/3 | 新训 |

### ETTh2-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | 0.5 | 0.001 | — | — | 0.408616 | 0.443848 | +1.65% / +1.80% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.392054 | 0.427058 | -2.47% / -2.05% | 3/3 | 复用 |
| `l_q1_4` | **2022** | 0.5 | 0.001 | pooled_lowrank | 180 | 0.387705 | 0.426072 | -3.56% / -2.28% | 3/3 | 复用 |
| `l_q1_8` | **2023** | 0.5 | 0.001 | pooled_lowrank | 90 | 0.387393 | 0.425611 | -3.63% / -2.38% | 3/3 | 复用 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.390707 | 0.426947 | -2.81% / -2.08% | 3/3 | 新训 |
| `a1` | **2022** | 0.5 | 0.001 | — | — | 0.391617 | 0.427280 | -2.58% / -2.00% | 3/3 | 新训 |

### ETTm1-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.296413 | 0.345745 | +1.17% / +0.51% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.305023 | 0.353522 | +4.10% / +2.77% | 0/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.301112 | 0.350402 | +2.77% / +1.86% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 12 | 0.297170 | 0.349666 | +1.42% / +1.65% | 0/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.306238 | 0.351275 | +4.52% / +2.11% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.301475 | 0.347780 | +2.89% / +1.10% | 0/3 | 新训 |

### ETTm1-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.328763 | 0.363129 | +1.78% / +0.59% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.335165 | 0.367483 | +3.77% / +1.80% | 0/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.336637 | 0.368904 | +4.22% / +2.19% | 0/3 | 新训 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.336736 | 0.368716 | +4.25% / +2.14% | 0/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.334692 | 0.366841 | +3.62% / +1.62% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.336961 | 0.369036 | +4.32% / +2.23% | 0/3 | 新训 |

### ETTm1-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.358508 | 0.381307 | +0.14% / +0.08% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.366170 | 0.385825 | +2.28% / +1.27% | 0/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.365835 | 0.384184 | +2.19% / +0.84% | 0/3 | 新训 |
| `l_q1_8` | **2023** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.363597 | 0.384863 | +1.56% / +1.01% | 0/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.363999 | 0.384548 | +1.68% / +0.93% | 0/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.363042 | 0.382969 | +1.41% / +0.52% | 0/3 | 新训 |

### ETTm1-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.414015 | 0.411819 | +0.49% / +0.44% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.415731 | 0.412364 | +0.91% / +0.58% | 0/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.416854 | 0.415509 | +1.18% / +1.34% | 0/3 | 新训 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.418249 | 0.414060 | +1.52% / +0.99% | 0/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.417493 | 0.414915 | +1.33% / +1.20% | 0/3 | 新训 |
| `a1` | **2022** | 0.5 | 0.001 | — | — | 0.417302 | 0.413835 | +1.29% / +0.94% | 0/3 | 新训 |

### ETTm2-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | 0.5 | 0.0003 | — | — | 0.172376 | 0.263844 | +5.75% / +3.06% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2021** | 0.5 | 0.0003 | shared (稠密) | — | 0.158474 | 0.248048 | -2.78% / -3.11% | 3/3 | 复用 |
| `l_q1_4` | **2022** | 0.5 | 0.0003 | pooled_lowrank | 24 | 0.160506 | 0.250390 | -1.53% / -2.19% | 3/3 | 复用 |
| `l_q1_8` | **2021** | 0.5 | 0.0003 | pooled_lowrank | 12 | 0.160349 | 0.250276 | -1.63% / -2.24% | 3/3 | 复用 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.159250 | 0.248384 | -2.30% / -2.97% | 3/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.160100 | 0.248717 | -1.78% / -2.84% | 3/3 | 新训 |

### ETTm2-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | 0.2 | 0.001 | — | — | 0.227672 | 0.299039 | +3.96% / +2.06% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.214320 | 0.287613 | -2.14% / -1.84% | 3/3 | 复用 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.215078 | 0.288337 | -1.79% / -1.59% | 3/3 | 复用 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.213480 | 0.287809 | -2.52% / -1.77% | 3/3 | 复用 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.214750 | 0.286238 | -1.94% / -2.31% | 3/3 | 新训 |
| `a1` | **2022** | 0.5 | 0.001 | — | — | 0.214597 | 0.286226 | -2.01% / -2.31% | 3/3 | 新训 |

### ETTm2-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.272467 | 0.331865 | +1.29% / +1.80% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.266250 | 0.323280 | -1.02% / -0.83% | 3/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.267050 | 0.324348 | -0.72% / -0.51% | 2/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.266544 | 0.325316 | -0.91% / -0.21% | 2/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.267089 | 0.321676 | -0.71% / -1.33% | 1/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.266950 | 0.321581 | -0.76% / -1.36% | 1/3 | 新训 |

### ETTm2-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.350952 | 0.379325 | -0.01% / +0.09% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.348071 | 0.375494 | -0.83% / -0.93% | 3/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.347200 | 0.376320 | -1.08% / -0.71% | 3/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.343484 | 0.377051 | -2.14% / -0.51% | 3/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.343179 | 0.375710 | -2.23% / -0.87% | 3/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.341635 | 0.375544 | -2.67% / -0.91% | 3/3 | 新训 |

### Weather-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | 0.2 | 0.0003 | — | — | 0.149795 | 0.196924 | +1.21% / +0.99% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.0003 | shared (稠密) | — | 0.146825 | 0.193847 | -0.79% / -0.59% | 3/3 | 复用 |
| `l_q1_4` | **2022** | 0.2 | 0.0003 | pooled_lowrank | 24 | 0.148555 | 0.195508 | +0.38% / +0.26% | 0/3 | 复用 |
| `l_q1_8` | **2022** | 0.2 | 0.0003 | pooled_lowrank | 12 | 0.146495 | 0.193457 | -1.02% / -0.79% | 2/3 | 复用 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.151256 | 0.196068 | +2.20% / +0.55% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.148589 | 0.194550 | +0.40% / -0.23% | 0/3 | 新训 |

### Weather-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | 0.5 | 0.001 | — | — | 0.193897 | 0.238277 | +0.46% / +0.54% | 0/3 | 复用 |
| `l_main` (PhaseFormer-L) | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.191791 | 0.236277 | -0.63% / -0.30% | 2/3 | 复用 |
| `l_q1_4` | **2023** | 0.5 | 0.001 | pooled_lowrank | 48 | 0.191531 | 0.235881 | -0.76% / -0.47% | 2/3 | 复用 |
| `l_q1_8` | **2022** | 0.5 | 0.001 | pooled_lowrank | 24 | 0.191234 | 0.234721 | -0.91% / -0.96% | 3/3 | 复用 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.193616 | 0.239143 | +0.32% / +0.90% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.193474 | 0.238716 | +0.25% / +0.72% | 0/3 | 新训 |

### Weather-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.244304 | 0.278613 | +0.95% / +0.22% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.239891 | 0.273774 | -0.87% / -1.52% | 2/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.242068 | 0.276161 | +0.03% / -0.66% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.240477 | 0.274327 | -0.63% / -1.32% | 3/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.244498 | 0.279674 | +1.03% / +0.60% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.245256 | 0.280548 | +1.35% / +0.92% | 0/3 | 新训 |

### Weather-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.315129 | 0.332051 | +1.98% / +0.02% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.312689 | 0.326608 | +1.19% / -1.62% | 0/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.313092 | 0.327724 | +1.32% / -1.29% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.310691 | 0.326189 | +0.55% / -1.75% | 0/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.319952 | 0.334094 | +3.54% / +0.63% | 0/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.311702 | 0.329657 | +0.87% / -0.71% | 0/3 | 新训 |

### Electricity-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.129780 | 0.221568 | +0.60% / +0.26% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.128701 | 0.222297 | -0.23% / +0.59% | 0/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.128698 | 0.221541 | -0.23% / +0.24% | 0/3 | 新训 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 12 | 0.128603 | 0.221174 | -0.31% / +0.08% | 0/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.129221 | 0.222108 | +0.17% / +0.50% | 0/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.130403 | 0.225134 | +1.09% / +1.87% | 0/3 | 新训 |

### Electricity-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.145346 | 0.234985 | -1.79% / -1.27% | 3/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.145376 | 0.236532 | -1.77% / -0.62% | 3/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.145333 | 0.236281 | -1.80% / -0.72% | 3/3 | 新训 |
| `l_q1_8` | **2023** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.144879 | 0.235674 | -2.11% / -0.98% | 2/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.145077 | 0.235473 | -1.98% / -1.06% | 2/3 | 新训 |
| `a1` | **2023** | 0.5 | 0.001 | — | — | 0.145710 | 0.237688 | -1.55% / -0.13% | 1/3 | 新训 |

### Electricity-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.166442 | 0.259232 | +0.87% / +0.87% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.161729 | 0.254716 | -1.98% / -0.89% | 3/3 | 复用 |
| `l_q1_4` | **2021** | 0.5 | 0.001 | pooled_lowrank | 84 | 0.161949 | 0.255010 | -1.85% / -0.77% | 3/3 | 复用 |
| `l_q1_8` | **2022** | 0.5 | 0.001 | pooled_lowrank | 42 | 0.162652 | 0.255463 | -1.42% / -0.60% | 3/3 | 复用 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.163254 | 0.255836 | -1.06% / -0.45% | 1/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.163872 | 0.257324 | -0.68% / +0.13% | 0/3 | 新训 |

### Electricity-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.198748 | 0.284326 | -1.12% / -0.24% | 2/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.196344 | 0.283933 | -2.32% / -0.37% | 2/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.196116 | 0.283824 | -2.43% / -0.41% | 2/3 | 新训 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.196483 | 0.285221 | -2.25% / +0.08% | 0/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.199599 | 0.286658 | -0.70% / +0.58% | 0/3 | 新训 |
| `a1` | **2021** | 0.5 | 0.001 | — | — | 0.199256 | 0.286940 | -0.87% / +0.68% | 0/3 | 新训 |

### Traffic-96

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2022** | — | 0.001 | — | — | 0.361093 | 0.230624 | +0.03% / -3.10% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2021** | 0.2 | 0.001 | shared (稠密) | — | 0.365329 | 0.236688 | +1.20% / -0.55% | 0/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.357544 | 0.232045 | -0.96% / -2.50% | 2/3 | 新训 |
| `l_q1_8` | **2021** | 0.2 | 0.001 | pooled_lowrank | 12 | 0.357959 | 0.233527 | -0.84% / -1.88% | 1/3 | 新训 |
| `l_rcrf` | **2023** | 0.5 | 0.001 | shared (稠密) | — | 0.360369 | 0.233320 | -0.17% / -1.97% | 2/3 | 新训 |

### Traffic-192

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.377812 | 0.239940 | +1.29% / -1.26% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.380649 | 0.244843 | +2.05% / +0.76% | 0/3 | 新训 |
| `l_q1_4` | **2021** | 0.2 | 0.001 | pooled_lowrank | 48 | 0.375466 | 0.243265 | +0.66% / +0.11% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 24 | 0.373360 | 0.241353 | +0.10% / -0.68% | 0/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.378217 | 0.245168 | +1.40% / +0.89% | 0/3 | 新训 |

### Traffic-336

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2023** | — | 0.001 | — | — | 0.391122 | 0.248667 | +1.59% / +0.27% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2022** | 0.2 | 0.001 | shared (稠密) | — | 0.391937 | 0.251512 | +1.80% / +1.42% | 0/3 | 新训 |
| `l_q1_4` | **2023** | 0.2 | 0.001 | pooled_lowrank | 84 | 0.392055 | 0.251322 | +1.83% / +1.34% | 0/3 | 新训 |
| `l_q1_8` | **2022** | 0.2 | 0.001 | pooled_lowrank | 42 | 0.389783 | 0.250043 | +1.24% / +0.82% | 0/3 | 新训 |
| `l_rcrf` | **2021** | 0.5 | 0.001 | shared (稠密) | — | 0.399493 | 0.252129 | +3.76% / +1.66% | 0/3 | 新训 |

### Traffic-720

| 臂 | 最佳 seed | gate_init | lr | head | rank | test MSE | test MAE | vs Golden | 3-seed 双指标胜 | 来源 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|
| `phase_only` | **2021** | — | 0.001 | — | — | 0.430241 | 0.270697 | +0.52% / +0.26% | 0/3 | 新训 |
| `l_main` (PhaseFormer-L) | **2023** | 0.2 | 0.001 | shared (稠密) | — | 0.439668 | 0.277599 | +2.73% / +2.81% | 0/3 | 新训 |
| `l_q1_4` | **2022** | 0.2 | 0.001 | pooled_lowrank | 180 | 0.431205 | 0.273391 | +0.75% / +1.26% | 0/3 | 新训 |
| `l_q1_8` | **2023** | 0.2 | 0.001 | pooled_lowrank | 90 | 0.433528 | 0.271775 | +1.29% / +0.66% | 0/3 | 新训 |
| `l_rcrf` | **2022** | 0.5 | 0.001 | shared (稠密) | — | 0.433429 | 0.273985 | +1.27% / +1.48% | 0/3 | 新训 |

## 2. 复现命令（每行一条，取该行的最佳 seed）

> 每条命令的 `--output-dir` 就是原始 run 目录；已有产物时 `--resume` 会跳过训练。
> `--overrides` 只给出**该格非默认**的超参（`gate_init` / `lr` / `head` / `rank`）。
> **`phase_only` 无残差支路**，故没有 gate / head / rank 参数。

```bash
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer
```

# [1] ETTh1-96 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_396d11b85321 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [2] ETTh1-96 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_e95e115f7fd1 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [3] ETTh1-96 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_a34c3a30df92 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [4] ETTh1-96 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_7c63ae6dfd92 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [5] ETTh1-96 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_8c5bb4a54f35 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [6] ETTh1-96 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_d8658d541fb5 \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [7] ETTh1-192 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_3d11eb0cee58 \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [8] ETTh1-192 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_d2463749831c \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [9] ETTh1-192 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_21a1fa0f88b3 \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [10] ETTh1-192 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_6571c21f6bfe \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [11] ETTh1-192 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_55949e5464f1 \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [12] ETTh1-192 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_3911a993f88a \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [13] ETTh1-336 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_4c124058794a \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [14] ETTh1-336 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_b7a738ad8cdd \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [15] ETTh1-336 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_be19148fd5eb \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [16] ETTh1-336 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_81eb95a6310e \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [17] ETTh1-336 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_1091da07e4e5 \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [18] ETTh1-336 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_c4a6365c1ed8 \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [19] ETTh1-720 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_b865d60c2b49 \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001}' --evaluate-test

# [20] ETTh1-720 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_d95dc976a572 \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [21] ETTh1-720 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_4b764c4d40a5 \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [22] ETTh1-720 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_0dba9cbab7a4 \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [23] ETTh1-720 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_58195bdf0813 \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [24] ETTh1-720 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth1_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_7d140428adab \
  --dataset ETTh1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"huber_delta": 0.3, "learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [25] ETTh2-96 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_etth2_h96_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_a35b6e3aaa3b \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [26] ETTh2-96 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_etth2_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_0522811aaf44 \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [27] ETTh2-96 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v4/runs/confirm_etth2_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_2cf889fbbff1 \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [28] ETTh2-96 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_etth2_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_45e82f5b5c6d \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [29] ETTh2-96 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_a686c30f0d49 \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [30] ETTh2-96 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_5bb5213e05c6 \
  --dataset ETTh2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [31] ETTh2-192 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_59f88dc9bcae \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [32] ETTh2-192 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_323796bab747 \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [33] ETTh2-192 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_545d28dedb9d \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [34] ETTh2-192 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_82af4fef409d \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [35] ETTh2-192 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_5671fda51755 \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [36] ETTh2-192 a1 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_446dffb24fdd \
  --dataset ETTh2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [37] ETTh2-336 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_12b4145c9e93 \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [38] ETTh2-336 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_c27461a39d18 \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [39] ETTh2-336 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_b6980753d15c \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [40] ETTh2-336 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_2c165158bd9d \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [41] ETTh2-336 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_f3427c548b6d \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [42] ETTh2-336 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_774ba17ce12a \
  --dataset ETTh2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [43] ETTh2-720 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/top2_direction_retention_v1/runs/confirm_etth2_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_a31cdf35ab8b \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [44] ETTh2-720 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_etth2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_fa021e3ef6e1 \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [45] ETTh2-720 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_etth2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_db9bf74fdbbf \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [46] ETTh2-720 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_etth2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_d97656e6e8b6 \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [47] ETTh2-720 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_a4e36afe12f9 \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [48] ETTh2-720 a1 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_etth2_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_cac39ea86902 \
  --dataset ETTh2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [49] ETTm1-96 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_e9ba2dceac17 \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [50] ETTm1-96 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_8b74609cc5e5 \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [51] ETTm1-96 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_871f23b96d51 \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [52] ETTm1-96 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_1becb2f03a0d \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [53] ETTm1-96 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_2ec88fb73806 \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [54] ETTm1-96 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_2c6f163d4dfe \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [55] ETTm1-192 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_84b194e30d99 \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [56] ETTm1-192 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_8a79f103069c \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [57] ETTm1-192 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_38ef7032bac9 \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [58] ETTm1-192 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_4bc773e8e3eb \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [59] ETTm1-192 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_6b5641c4f3a6 \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [60] ETTm1-192 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_655f68778c2b \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [61] ETTm1-336 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_6678a10551da \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [62] ETTm1-336 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_69df48cd6c8d \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [63] ETTm1-336 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_3d0cb4528bad \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [64] ETTm1-336 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_f307e6763fcf \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [65] ETTm1-336 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_7a9fac613e7f \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [66] ETTm1-336 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_96fbb72b30a0 \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [67] ETTm1-720 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_93a945bd024b \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [68] ETTm1-720 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_5b2f8b3e366c \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [69] ETTm1-720 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_cdf582b7da24 \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [70] ETTm1-720 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_0741eb740b9f \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [71] ETTm1-720 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_064762c11e04 \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [72] ETTm1-720 a1 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm1_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_8cdb8e618bc2 \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [73] ETTm2-96 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_ettm2_h96_no_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2021_fcaa438479bc \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003}' --evaluate-test

# [74] ETTm2-96 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_ettm2_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2021_bb0782c0b454 \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [75] ETTm2-96 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_ettm2_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2022_bd6a2028a1c2 \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [76] ETTm2-96 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_ettm2_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2021_c970a55e8e93 \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [77] ETTm2-96 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_0639325a1d04 \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [78] ETTm2-96 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_20cd7f90ddcd \
  --dataset ETTm2 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [79] ETTm2-192 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/top2_direction_retention_v1/runs/confirm_ettm2_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_9723e623851b \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [80] ETTm2-192 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_ettm2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_fcc357cb7434 \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [81] ETTm2-192 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_ettm2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_e1d1f9d2fd3c \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [82] ETTm2-192 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_ettm2_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_31f67687693d \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [83] ETTm2-192 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_9bf478563b24 \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [84] ETTm2-192 a1 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_9c0e3247ab50 \
  --dataset ETTm2 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [85] ETTm2-336 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_2dfde51368d1 \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [86] ETTm2-336 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_f0a3609541ef \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [87] ETTm2-336 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_a2848af36a53 \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [88] ETTm2-336 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_3f9677a6c927 \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [89] ETTm2-336 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_95a7609585b1 \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [90] ETTm2-336 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_b5bde13406df \
  --dataset ETTm2 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [91] ETTm2-720 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_cac75100504a \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [92] ETTm2-720 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_be02762bd310 \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [93] ETTm2-720 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_553cc5f03a1e \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [94] ETTm2-720 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_33f27a39e6bd \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [95] ETTm2-720 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_5fa55909fcbb \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [96] ETTm2-720 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_ettm2_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_67fde1546e87 \
  --dataset ETTm2 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [97] Weather-96 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_weather_h96_no_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2021_3af67e912320 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003}' --evaluate-test

# [98] Weather-96 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_weather_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2023_cbd74201d0b0 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [99] Weather-96 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_weather_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2022_33e355e3f042 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [100] Weather-96 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_weather_h96_weak_residual_p24_base_none-full_huber_lr0.0003_pct100_e30_s2022_92e1f25d3de4 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [101] Weather-96 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_4412443f9026 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [102] Weather-96 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_2f4c99857111 \
  --dataset Weather --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [103] Weather-192 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_weather_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_97bf0062075e \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [104] Weather-192 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_weather_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_fe99c0bb3da5 \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [105] Weather-192 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_weather_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_cc0634cb2134 \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [106] Weather-192 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_weather_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_5aad3ac57e7d \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [107] Weather-192 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_f52e02ff3e15 \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [108] Weather-192 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_ea751468d54c \
  --dataset Weather --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [109] Weather-336 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_79812608a9ad \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [110] Weather-336 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_0f0aa37c7dd6 \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [111] Weather-336 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_867709b8f03c \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [112] Weather-336 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_798198934c14 \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [113] Weather-336 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_5d28d77344f7 \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [114] Weather-336 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_85af9ab2d42c \
  --dataset Weather --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [115] Weather-720 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_6fc4bbcdcc14 \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [116] Weather-720 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_e81a3d88cab5 \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [117] Weather-720 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_075d98faab4c \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [118] Weather-720 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_025c953f964f \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [119] Weather-720 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_e11a5ee6bc20 \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [120] Weather-720 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_weather_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_10111fbb9709 \
  --dataset Weather --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [121] Electricity-96 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_4a7e2aeefb87 \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [122] Electricity-96 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_69bb94a0742b \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [123] Electricity-96 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_4178e8930fd5 \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [124] Electricity-96 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_57d6f128cb4e \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [125] Electricity-96 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_6fe301b5ddbd \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [126] Electricity-96 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h96_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_92090ee3fa23 \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [127] Electricity-192 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_366bb44a07e9 \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [128] Electricity-192 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_8a40b4ee356a \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [129] Electricity-192 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_41e0d4913405 \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [130] Electricity-192 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_fe33f4375850 \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [131] Electricity-192 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_216d7380f94d \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [132] Electricity-192 a1 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h192_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_b19b76dac258 \
  --dataset Electricity --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [133] Electricity-336 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_ab4afb6cb035 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [134] Electricity-336 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_electricity_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_f7af1dba67a4 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [135] Electricity-336 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_stage1/runs/confirm_electricity_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_13187e2bb000 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [136] Electricity-336 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/rank_sweep_2_multiseed_stage1_20260914_v3/runs/confirm_electricity_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_9b9ab1ad1616 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [137] Electricity-336 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_617c6a7717d6 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [138] Electricity-336 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h336_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_487fbda8cdc3 \
  --dataset Electricity --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [139] Electricity-720 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_792d1020d146 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [140] Electricity-720 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_981c92b157f5 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [141] Electricity-720 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_49587a73c526 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [142] Electricity-720 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_51f94a5da405 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [143] Electricity-720 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_c14b9bc31c88 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [144] Electricity-720 a1 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_electricity_h720_gold_combo_reliability_s2_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_55333ec9e7d5 \
  --dataset Electricity --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism gold_combo_reliability_s2 --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5}' --evaluate-test

# [145] Traffic-96 phase_only seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h96_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_b821d694da05 \
  --dataset Traffic --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [146] Traffic-96 l_main seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_61083eb5fe21 \
  --dataset Traffic --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [147] Traffic-96 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_1caa451566b0 \
  --dataset Traffic --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [148] Traffic-96 l_q1_8 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h96_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_e9e2ea44d753 \
  --dataset Traffic --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' --evaluate-test

# [149] Traffic-96 l_rcrf seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h96_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_c5475c442f5a \
  --dataset Traffic --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [150] Traffic-192 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h192_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_31fd574d099a \
  --dataset Traffic --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [151] Traffic-192 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_7e8efb67fe05 \
  --dataset Traffic --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [152] Traffic-192 l_q1_4 seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_89cf02c5b44c \
  --dataset Traffic --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 48}' --evaluate-test

# [153] Traffic-192 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h192_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_6fbb8db72979 \
  --dataset Traffic --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' --evaluate-test

# [154] Traffic-192 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h192_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_8e43666150e7 \
  --dataset Traffic --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [155] Traffic-336 phase_only seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h336_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_4b2218a8e858 \
  --dataset Traffic --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [156] Traffic-336 l_main seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_c8b839656916 \
  --dataset Traffic --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [157] Traffic-336 l_q1_4 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_712724a7935e \
  --dataset Traffic --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 84}' --evaluate-test

# [158] Traffic-336 l_q1_8 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h336_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_70de36b22201 \
  --dataset Traffic --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 42}' --evaluate-test

# [159] Traffic-336 l_rcrf seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h336_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_8f05c8ccc79e \
  --dataset Traffic --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [160] Traffic-720 phase_only seed 2021
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h720_no_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2021_9b5845e0805c \
  --dataset Traffic --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism no_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001}' --evaluate-test

# [161] Traffic-720 l_main seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_32c601fda0fb \
  --dataset Traffic --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "shared"}' --evaluate-test

# [162] Traffic-720 l_q1_4 seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_ccecb988542a \
  --dataset Traffic --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 180}' --evaluate-test

# [163] Traffic-720 l_q1_8 seed 2023
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h720_weak_residual_p24_base_none-full_huber_lr0.001_pct100_e30_s2023_e934d0207b53 \
  --dataset Traffic --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2023 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 90}' --evaluate-test

# [164] Traffic-720 l_rcrf seed 2022
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_e14_main_v1/runs/confirm_traffic_h720_rcrf_nlinear_plain_p24_base_none-full_huber_lr0.001_pct100_e30_s2022_abc4c01910bc \
  --dataset Traffic --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 --seed 2022 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism rcrf_nlinear_plain --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.5, "weak_period_residual_head_type": "shared"}' --evaluate-test

## 3. 口径与边界（引用本表时必须一并写明）

1. **test-set selection**：本表以 test 指标选 3 个 seed 中最优的一次（用户 2026-09-20 明示口径），
   属条件性证据，**不得表述为盲测或无偏泛化估计**。
2. **7 个 setting 来自既有 test-set selection**（ETTh2-96/720、ETTm2-96/192、Weather-96/192、
   Electricity-336），其 cell 中有复用/新训两类来源，表中「来源」列逐行标注。
3. **Golden 来自另一套硬件环境**（本服务器 torch 2.6.0 + Lightning 2.6.5），与 Golden 的比较只作披露；
   配对基线应为同环境的 matched `phase_only`。
4. **三种 gate 先验**：新格 0.2、`l_rcrf`/`a1` 由 preset 自持 0.5、复用格为其 Stage-0 冻结值；
   表中的 `gate_init` 是逐格实测值。
5. **Traffic 为探索性附录**（4 个 setting），不进入 §4.2 的判定。
6. **定向调参的 8 个 setting 另见** [`PhaseFormer_L_best_settings_repro.md`](PhaseFormer_L_best_settings_repro.md)，
   其中 `ETTh1-336` 用到第三轮的 `huber_delta=0.1` 与 `lr=1e-2`。