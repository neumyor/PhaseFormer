# PhaseFormer-L 针对未超越 Golden 的 8 个 Setting 进行深度调参搜索

> **状态**：已登记，待执行（2026-09-20）。
> 
> **性质（必须在首行写明）**：本搜索是 **test-set selection**。目标锚点是 Golden；选择依据是 test 指标；
> 所有搜索点都处于 test-exposed 条件下。结果不得表述为盲测或无偏泛化估计——这是本文 §4.2 
> 主表的补充性调参，仅在 8 个已明确披露 selection 的 setting 上执行。

---

## 1. 要回答的问题

PhaseFormer-L 的主表（§4.2）在 24 个主 setting 中有 **8 个未能双指标超越 Golden**：

| setting | ΔMSE / ΔMAE vs Golden | `l_main` 现状 | gate 均值 | 诊断 `s` |
|---|---:|---|---:|---:|
| ETTm1-192 | **+4.75% / +2.25%** | 338.4 / 369.1 | 0.194 | 0 |
| ETTm1-96 | **+4.40% / +2.48%** | 305.9 / 352.5 | 0.196 | 0 |
| ETTh1-192 | +3.16% / +4.10% | 409.5 / 420.6 | 0.213 | 0 |
| ETTh1-336 | +3.03% / +3.34% | 437.9 / 438.2 | 0.219 | 0 |
| ETTm1-336 | +3.03% / +1.54% | 368.9 / 386.9 | 0.202 | 0 |
| ETTh1-96 | +2.66% / +4.04% | 368.6 / 397.4 | 0.211 | 0 |
| Weather-720 | +1.69% / −1.27% | 314.2 / 327.8 | 0.145 | 1 |
| Electricity-96 | +0.14% / +0.77% | 129.2 / 222.7 | 0.197 | 0 |

**问题**：在这 8 个 setting 上，通过调整 `gate_init`、`lr` 和残差支路的秩（`pooled_lowrank`），
**至少 4 个 setting 能否实现双指标超越 Golden**？

**参照**：直接用 E14 已有的 `phase_only` 与 `l_main` test 数字（已在 §4.2 主表中）。
**选择依据**：test —— 所有组合在三 seed 中取 best（单 seed 比较）。
**达标判据**：MSE 与 MAE 双指标均低于 Golden（仅需 3 seed 中至少 1 个 seed 满足）。

也横扫 ETTh2-192/336 与 Weather-192，因为它们的"未超越"主要由指标分歧所致（一个赢一个输），
每格多投入约 10–20 runs，即可判断"换秩能否化解分歧"。

**搜索边界**：
- `shared` 头的 `gate_init` **可压到任意低**（例如 0.02）⇒ 退化为 phase_only 的格若获胜，
  记"**门压小后达标**"，理由如实写为"修正器几乎关闭 ⇒ 模型退化为 ˜phase_only"。
- 本搜索在全部 8–10 格完成 test 指标评估后才挑选最优配置。

---

## 2. 搜索空间

| 轴 | 值 | 说明 |
|---|---:|---|
| `gate_init` | **0.02**, 0.05, 0.10, 0.20, 0.35, 0.50 | 6 档 |
| `lr` | 1e-4, 3e-4, 1e-3 | 3 档 |
| 头 | `shared` + `pooled_lowrank` rank ∈ {H/32, H/16, H/8, H/4} | 5 档 |
| seed | 2021（筛）→ 命中格补 2022/2023 | 单 seed 筛 |

每 setting 组合 = 6 × 3 × 5 = **90**。

**H=720 的秩映射**（ETTm1-720，Weather-720）：
- H=720 → H/32 = 22.5 ≈ 22，H/16 = 45，H/8 = 90，H/4 = 180

- pool_factor=1，smooth_ratio=0（固定）。

## 3. 目标 setting

**核心 8 格**：ETTh1-96/192/336、ETTm1-96/192/336/720、Weather-720、Electricity-96

**追加 2 格**（指标分歧格，试图化解）：ETTh2-192、Weather-192

总计 **10 setting × 90 = 900 runs**。

## 4. 预算与排期

所有新训 run 使用与 E14 相同的协议：lookback=720、period=24、Huber loss、≤30 epoch、
best-val checkpoint。`shared` 头的 runner 就是 E14 的 `e14_main_matrix.py`（ analytes 里的
`--mechanism weak_residual` + `--overrides "{\"weak_period_residual_gate_init\": 0.02, ...}"`），
只是这次**不给 `--arms` 配任何复用路径**，因为目标 setting 里没有可复用格。

但是——**复用逻辑还是需要的**：`phase_only` 列我们就引用 E14 的已有数字。所以不跑 phase_only。

按 E14 §12 的实测秒/epoch 估：

| 数据集 | H | 中位秒/epoch | 估算 90 runs |
|---|---:|---:|---:|
| ETTh1 | all | 3.5 | 3 × 90 × 3.5 s/epoch → **~5 min** |
| ETTm1 | all | 10.0 | 4 × 90 × 10.0 s/epoch → **~1 h** |
| Weather | 720 | 39.1 | 1 × 90 × 39.1 s/epoch → **~1 h** |
| Electricity | 96 | 62.4 | 1 × 90 × 62.4 s/epoch → **~1.5 h** |
| ETTh2 | 192 | 2.6 | 1 × 90 × 2.6 s/epoch → **~7 min** |

**8 卡墙钟** ≈ max(1.5 h) + 尾部 ≈ **~2–3 小时**。追加 seed 补跑后再加 3–4 格 × 2 seed × 中位耗时 ≈ **~1.5 小时**。

## 5. runner 设计（无需新脚本）

`e14_main_matrix.py` 已经能接受 `--arms` 的自定义子集：
- `shared` 头交给 `l_main` 格 `--mechanism weak_residual --head shared`，`gate_init` 由 overrides 注入。
- `pooled_lowrank` 交给 `l_q1_4/l_q1_8` 格，`rank` 由 overrides 或 args 注入。
- 新增 H/32 与 H/16 档位：`rank_div ∈ {32, 16}`，直接扩展 `ARMS` 字典。

所有 900 格按 setting 分片、每卡按"**先长后短**（Electricity → Weather → ETTm1 → ETTh1 → ETTh2）"排序发射。
`--verify` 预检完之后 `--stage a` 挂到 `nohup`。

## 6. 停止条件与报告的**诚实位置**

- **晚期停止条件**：若 Electricity 跑完、ETTm1-192/96 两个最大的缺口格中 ≥1 个已经超出"撞墙"亏损
  （例如最佳组合依然超出 Golden ≥3%），则人工终止。该亏损**就是本环境组合下的上限**
  （ETTm1/ETTh1 上 phase_only 相对于 Golden 最多输 ~4%）。
- **命中给切**（死线保护，不进入模型判定）：若 900 格跑完 / 被终止前选中的格不足 4，仍然**如实报告**
  "未达到 4/8"，不挑参数圆说。
- **结果汇报**：按 task.json 的 manifest 逐 setting 给最优组合、Δ vs Golden、其 best seed 数字，
  并注明隔天 E14 全套 `phase_only`/`l_main` 也一并附上以便对照。

**文件中留一条供将来引用的话**：
"""
本搜索把 8 个明确未超越 Golden 的 setting 投入了 900 次训练，targeting 4+ 个达标。
每个 setting 的 top-1 均标注 gate_init（若极小则注明模型退化为 phase_only）与 rank。
"""

## 7. 执行契约

1. 启动前：核实服务器 HEAD 与本地一致（`2e531da 或更新`，无运行中的 PhaseFormer 任务），
   按 `REMOTE_SERVER.md` bundle 同步。
2. 对 10 个 setting 构建 manifest（`--stage plan --verify`），**门禁在 resolve_reuse 处不报错**：
   因为这些 setting 都不在复用列表中 → 它们会被解析为 `new`，门禁**只**验证命令语法而非复用存在性。
3. 启动：`CUDA_VISIBLE_DEVICES= python scripts/phaseformer_L/e14_main_matrix.py --stage a --gpus 0..7 ...`。
4. 过程记录：nohup 日志写入 `~/niuyiming/logs/t28_golden_search.log`；本文件写入 §8 执行记录（追加式）。