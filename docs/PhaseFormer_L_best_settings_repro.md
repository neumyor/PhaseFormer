# PhaseFormer-L 最佳 setting 复现手册

> **性质**：本文件列出 §4.2 主表中**经定向调参得到的最佳 setting**，含复现所需的全部参数与对应的最佳 seed。
> 选择口径为**用户在 2026-09-20 明示指定的 test-set selection**：以 **test 指标**选择组合，
> 并在 **3 个 seed 中取最优的那一次**（逐 seed 取最优，非 3-seed 均值）。
> **本文件的每一个数都从 `final_selection.json` / `final_with_delta.csv` 读取，并与 run 目录逐格核对过**
> （校验器 `verify_final.py` 报 `PROBLEMS: 0`）。

- 产物根目录：`research_runs/phaseformer_L_golden_search_v1/`
- 权威表：`final_selection.json`（判定）、`final_with_delta.csv`（含 `delta` 列的修正版）
- 每个 setting 的 3 个 seed 各自的 run 目录、`config.json`、`metrics.csv` 均保留

## 0. 复现协议（所有 setting 相同）

| 项 | 值 |
|---|---|
| 输入长度 | 720 |
| 周期 | 24 |
| loss | 见各 setting（`huber` 或 `mae`）|
| `max_epochs` | 30（第三轮另有 60 的档；**本表选中的 winner 全为 30**）|
| batch | ETT 系 256、Electricity 64（preset 默认，见各格 `config.json`）|
| 优化 | 逐 setting 的 `learning_rate`，见下表 |
| checkpoint | 最低 validation loss（best-val，早停 patience=8）|
| 评估 | 每 checkpoint **只读一次 test** |
| 融合 | `y = (1-g)·y_phase + g·y_residual`，`g = sigmoid(gate_init)` 起、可训练 |
| 种子 | 复现命令里的 `--seed` 即该格的最佳 seed（见下表）|

## 1. 最佳 setting 总表（8 个 setting 全覆盖）

| setting | 达标 | 最佳 seed | gate_init | lr | loss | head | rank | delta | test MSE | test MAE | vs Golden | vs phase_only | 3-seed 达标数 |
|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---|---|---:|
| **ETTh1-96** | ✓ | **2021** | 0.2 | 0.003 | mae | pooled_lowrank (r=24) | 24 | — | 0.347045 | 0.380476 | -3.33% / -0.40% | -3.97% / -1.61% | 1/3 |
| **ETTm1-96** | ✓ | **2021** | 0.1 | 0.001 | mae | pooled_lowrank (r=12) | 12 | — | 0.290128 | 0.337738 | -0.98% / -1.82% | -4.07% / -3.82% | 3/3 |
| **Electricity-96** | ✓ | **2021** | 0.2 | 0.003 | huber | pooled_lowrank (r=24) | 24 | — | 0.127495 | 0.219683 | -1.17% / -0.60% | -2.26% / -1.39% | 3/3 |
| **ETTh1-192** | ✗ | **2021** | 0.02 | 0.001 | mae | pooled_lowrank (r=6) | 6 | — | 0.387154 | 0.405789 | -2.48% / +0.44% | -4.33% / -1.25% | 0/3 |
| **ETTm1-192** | ✗ | **2023** | 0.2 | 0.0003 | mae | pooled_lowrank (r=6) | 6 | — | 0.325449 | 0.363033 | +0.76% / +0.56% | -1.50% / -0.07% | 0/3 |
| **ETTh1-336** | ✓ | **2021** | 0.02 | 0.01 | huber | pooled_lowrank (r=21) | 21 | 0.1 | 0.417830 | 0.423127 | -1.69% / -0.21% | -5.45% / -2.65% | 1/3 |
| **ETTm1-336** | ✓ | **2021** | 0.2 | 0.0003 | mae | pooled_lowrank (r=10) | 10 | — | 0.354631 | 0.376313 | -0.94% / -1.23% | -1.30% / -1.27% | 3/3 |
| **ETTm1-720** | ✓ | **2021** | 0.1 | 0.0003 | mae | pooled_lowrank (r=45) | 45 | — | 0.409963 | 0.407074 | -0.49% / -0.71% | -1.23% / -1.38% | 1/3 |

**达标计数 6/8**：其中 **3 格三 seed 稳定**（Electricity-96、ETTm1-96、ETTm1-336）、
**3 格单 seed**（ETTh1-96、ETTh1-336、ETTm1-720）。**8/8 双指标优于 matched `phase_only`。**

## 2. 逐 setting 复现命令与三个 seed 的实测值

> 命令中的 `--overrides` 为该格全部非默认超参。复现时**只跑表中所指的 seed** 即得到最佳那一格；
> 若要完整复核，把 `--seed` 换成 2022/2023 各跑一次（下表给出三者的实测值）。
> `--output-dir` 必须与原始目录一致，`--resume` 会在已有产物时跳过。

```bash
PY=/home/yyk/yyk03/miniconda3/envs/time/bin/python
cd ~/niuyiming/PhaseFormer   # 必须在仓库根目录运行
```

### ETTh1-96　（达标）

- **最佳 seed：2021**　test MSE **0.347045** / MAE **0.380476**（vs Golden -3.33% / -0.40%）
- 配置：`gate_init=0.2`、`lr=0.003`、`loss=mae`、`head=pooled_r24`、`rank=24`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTh1-h96_s2021_g0.2_lr0.003_r24_mae \
  --dataset ETTh1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.003 \
  --overrides '{"learning_rate": 0.003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.347045 | 0.380476 | -3.33% | -0.40% | ✓ | 29 |
| 2022 | 0.360326 | 0.387848 | +0.37% | +1.53% | ✗ | 30 |
| 2023 | 0.356931 | 0.388076 | -0.58% | +1.59% | ✗ | 17 |

### ETTm1-96　（达标）

- **最佳 seed：2021**　test MSE **0.290128** / MAE **0.337738**（vs Golden -0.98% / -1.82%）
- 配置：`gate_init=0.1`、`lr=0.001`、`loss=mae`、`head=pooled_r12`、`rank=12`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTm1-h96_s2021_g0.1_lr0.001_r12_mae \
  --dataset ETTm1 --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.1, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 12}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.290128 | 0.337738 | -0.98% | -1.82% | ✓ | 30 |
| 2022 | 0.291005 | 0.335844 | -0.68% | -2.37% | ✓ | 30 |
| 2023 | 0.291855 | 0.340201 | -0.39% | -1.10% | ✓ | 30 |

### Electricity-96　（达标）

- **最佳 seed：2021**　test MSE **0.127495** / MAE **0.219683**（vs Golden -1.17% / -0.60%）
- 配置：`gate_init=0.2`、`lr=0.003`、`loss=huber`、`head=pooled_r24`、`rank=24`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/Electricity-h96_s2021_g0.2_lr0.003_r24 \
  --dataset Electricity --horizon 96 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.003 \
  --overrides '{"learning_rate": 0.003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 24}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.127495 | 0.219683 | -1.17% | -0.60% | ✓ | 30 |
| 2022 | 0.128142 | 0.220502 | -0.66% | -0.23% | ✓ | 30 |
| 2023 | 0.128201 | 0.220604 | -0.62% | -0.18% | ✓ | 30 |

### ETTh1-192　（未达标，列出最优尝试）

- **最佳 seed：2021**　test MSE **0.387154** / MAE **0.405789**（vs Golden -2.48% / +0.44%）
- 配置：`gate_init=0.02`、`lr=0.001`、`loss=mae`、`head=pooled_r6`、`rank=6`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTh1-h192_s2021_g0.02_lr0.001_r6_mae \
  --dataset ETTh1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.001 \
  --overrides '{"learning_rate": 0.001, "weak_period_residual_gate_init": 0.02, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 6}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.387154 | 0.405789 | -2.48% | +0.44% | ✗ | 30 |
| 2022 | 0.401362 | 0.412448 | +1.10% | +2.09% | ✗ | 30 |
| 2023 | 0.398328 | 0.407044 | +0.33% | +0.75% | ✗ | 30 |

### ETTm1-192　（未达标，列出最优尝试）

- **最佳 seed：2023**　test MSE **0.325449** / MAE **0.363033**（vs Golden +0.76% / +0.56%）
- 配置：`gate_init=0.2`、`lr=0.0003`、`loss=mae`、`head=pooled_r6`、`rank=6`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTm1-h192_s2023_g0.2_lr0.0003_r6_mae \
  --dataset ETTm1 --horizon 192 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2023 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 6}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 | 0.325483 | 0.361057 | +0.77% | +0.02% | ✗ | 30 |
| 2022 | 0.326291 | 0.362049 | +1.02% | +0.29% | ✗ | 30 |
| 2023 ←**最佳** | 0.325449 | 0.363033 | +0.76% | +0.56% | ✗ | 30 |

### ETTh1-336　（达标）

- **最佳 seed：2021**　test MSE **0.417830** / MAE **0.423127**（vs Golden -1.69% / -0.21%）
- 配置：`gate_init=0.02`、`lr=0.01`、`loss=huber`、`head=pooled_r21`、`rank=21`、`huber_delta=0.1`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTh1-h336_s2021_g0.02_lr0.01_r21_d0.1 \
  --dataset ETTh1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss huber --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.01 \
  --overrides '{"huber_delta": 0.1, "learning_rate": 0.01, "weak_period_residual_gate_init": 0.02, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 21}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.417830 | 0.423127 | -1.69% | -0.21% | ✓ | 12 |
| 2022 | 0.433825 | 0.439404 | +2.08% | +3.63% | ✗ | 11 |
| 2023 | 0.425946 | 0.432803 | +0.22% | +2.08% | ✗ | 11 |

### ETTm1-336　（达标）

- **最佳 seed：2021**　test MSE **0.354631** / MAE **0.376313**（vs Golden -0.94% / -1.23%）
- 配置：`gate_init=0.2`、`lr=0.0003`、`loss=mae`、`head=pooled_r10`、`rank=10`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTm1-h336_s2021_g0.2_lr0.0003_r10_mae \
  --dataset ETTm1 --horizon 336 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.2, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 10}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.354631 | 0.376313 | -0.94% | -1.23% | ✓ | 30 |
| 2022 | 0.354837 | 0.377608 | -0.88% | -0.89% | ✓ | 30 |
| 2023 | 0.356078 | 0.377386 | -0.54% | -0.95% | ✓ | 30 |

### ETTm1-720　（达标）

- **最佳 seed：2021**　test MSE **0.409963** / MAE **0.407074**（vs Golden -0.49% / -0.71%）
- 配置：`gate_init=0.1`、`lr=0.0003`、`loss=mae`、`head=pooled_r45`、`rank=45`、`max_epochs=30`

```bash
$PY scripts/search_phaseformer.py \
  --output-dir research_runs/phaseformer_L_golden_search_v1/runs/ETTm1-h720_s2021_g0.1_lr0.0003_r45_mae \
  --dataset ETTm1 --horizon 720 --stage confirm \
  --lookback 720 --period 24 --max-epochs 30 \
  --seed 2021 --loss mae --percent 100 \
  --require-cuda --resume --num-workers 4 --bad-case-limit 0 \
  --mechanism weak_residual --learning-rate 0.0003 \
  --overrides '{"learning_rate": 0.0003, "weak_period_residual_gate_init": 0.1, "weak_period_residual_head_type": "pooled_lowrank", "weak_period_residual_pool_factor": 1, "weak_period_residual_rank": 45}' \
  --evaluate-test
```

| seed | test MSE | test MAE | vs Golden MSE | vs Golden MAE | 双指标胜 Golden | 实跑 epoch |
|---:|---:|---:|---:|---:|:--:|---:|
| 2021 ←**最佳** | 0.409963 | 0.407074 | -0.49% | -0.71% | ✓ | 30 |
| 2022 | 0.413274 | 0.408274 | +0.31% | -0.42% | ✗ | 30 |
| 2023 | 0.414482 | 0.408548 | +0.60% | -0.35% | ✗ | 29 |

## 3. 口径与边界（引用本表时必须一并写明）

1. **test-set selection**：本表以 test 指标选组合、并在 3 seed 中取最优，属条件性证据；
   **不得表述为盲测或无偏泛化估计**。
2. **单 seed 达标 vs 三 seed 稳定**：6 个达标格中仅 3 格是三 seed 全部达标，
   ETTh1-96、ETTh1-336、ETTm1-720 **只在 1/3 个 seed 上达标**，须分开表述。
3. **Golden 来自另一套硬件环境**（本服务器 torch 2.6.0 + Lightning 2.6.5），
   与 Golden 的比较只作披露；配对基线应为同环境的 matched `phase_only`。
4. **第三轮的 `delta`/`max_epochs` 轴**：`huber_delta` 只在第三轮出现（ETTh1-336 的 winner 用到 `delta=0.1`）；
   60-epoch 档**与 E14 的 30-epoch 协议不可比**，但**本表选中的 winner 全部为 30 epoch**。
5. **`gate_shrunk` 标注**：`gate ≤ 0.05` 的获胜格须注明"胜利来自相位主干、不是电平通道"。
   本表中 `ETTh1-336`（gate 0.02）与 `ETTh1-192`（gate 0.02，未达标）属该情形。
6. 每个 setting 的**完整搜索轨迹**（1576 + 396 = 1972 个 run）见 `stage1_all_rows.csv` 与各 `runs/*/config.json`。
