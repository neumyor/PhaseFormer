# Joint Low-Rank Rank Sweep 计划（用户指定）

> 状态：**planned only，2026-09-10。本方案先行登记，尚未训练任何模型、尚未读取任何 test 结果。**
> 本实验由用户于 2026-09-10 直接指定：固定 pooling 与 smoothing，仅改变 NLinear
> 因子化的相对秩，在 5 数据集 × 2 horizon 上测量低秩压缩的影响。它取代
> `docs/PhaseFormer_pooled_lowrank_nlinear_experiment.md` 中「Controlled Follow-up
> Plan」的活跃地位（后者保留为可选深化路径，不被废弃）。

## 1. 要回答的问题

前次 H96 screen（`PhaseFormer_pooled_lowrank_nlinear_experiment.md`）中 pooling 与
rank 同时变化，无法分离纯低秩效应，且只覆盖 ETTh1/ETTm1 的 H96。本实验固定
`pool_factor=1`、`smooth_ratio=0`，仅改变相对秩 `q`，回答：

1. 把 NLinear 残差分支的因子化映射压缩到 `H/4`、`H/8`、`H/16`、`H/32` 秩，
   相对不压缩（满秩因子化）如何改变误差？
2. 该低秩效应在 5 个数据集 × H96/H192 上方向是否一致？
3. 低秩压缩相对未因子化的普通 NLinear 头（`direct_nlinear`）处于什么位置？

## 2. 谱系披露（不可省略）

- 前次 screen 的 test 矩阵是 test-set-exposed 探索性证据，本方案与其共享机制和
  部分配置空间，因此本实验结果**不得描述为盲测或无偏泛化估计**。
- 本方案为全矩阵报告：所有档位结果一并呈现，**不存在基于 test 的档位选择**；
  每个配置只读取一次 test。

## 3. 结构定义

- PhaseFormer 相位路径 + `PooledLowRankWeakPeriodResidualHead`
  （`src/models/phase_adapters.py`），`pool_factor=1`、`smooth_ratio=0`；
- 融合：静态 per-channel sigmoid gate，与 PhaseFormer 路径、残差头**联合随机
  初始化训练**（与前次 screen 的联合训练协议完全一致）；
- last 值锚点保持在动态输入之外：`X - X_last -> pooled(=1, 恒等) -> rank-r -> horizon -> + X_last`；
- 相对秩定义：`q = r / min(ceil(720 / pool), H)`；在 `pool=1`、lookback 720 下
  即 `q = r / H`。

## 4. 档位与秩映射

用户指定的不压缩对照 + 四个压缩档（单 seed 2021）：

| q | 含义 | r @ H96 | r @ H192 | head 参数量 @ H96 | head 参数量 @ H192 |
|---|---|---:|---:|---:|---:|
| 1 | 不压缩（满秩因子化对照） | 96 | 192 | 78,528 | 175,488 |
| 1/4 | 压缩档 1 | 24 | 48 | 19,704 | 44,016 |
| 1/8 | 压缩档 2 | 12 | 24 | 9,900 | 22,104 |
| 1/16 | 压缩档 3 | 6 | 12 | 4,998 | 11,148 |
| 1/32 | 压缩档 4 | 3 | 6 | 2,547 | 5,670 |
| （参照）direct_nlinear | 未因子化单层 NLinear | — | — | 69,216 | 138,432 |

参数量 = encoder `Linear(720→r)` + decoder `Linear(r→H)`（含 bias）；decoder
零初始化，encoder 默认初始化。所有 H96 档位 `r` 与 H192 档位 `r` 均为整数，
无需舍入。

## 5. 控制组

| 控制组 | 作用 |
|---|---|
| `phase_only` | matched 联合训练 original PhaseFormer；A800 环境与 Golden 环境不同，Golden 只作参照不作环境对照 |
| `direct_nlinear` | 未因子化 shared 头；把「因子化参数化效应」（q=1 vs direct_nlinear）与「低秩压缩效应」（q<1 vs q=1）分开 |

ETTh1/ETTm1 的 H96 控制组：若前次 screen 中 `joint_pooled_lowrank_nlinear_h96_v1`
的 baseline / direct 对照与本协议完全一致（L720、seed 2021、Huber、30 epochs、
best-val checkpoint、联合训练），可复用并在表格中标注来源；否则重跑。

## 6. 实验矩阵与预算

- Settings：{ETTh1, ETTh2, ETTm1, ETTm2, Weather} × {96, 192}，共 10 个。
- 核心矩阵：5 档位 × 10 settings = **50 runs**；控制组 2 × 10 = **20 runs**；
  合计 ≤ **70 runs**（控制组可复用时相应减少）。
- 单 seed：2021。
- 预算估计：参照 288-run 矩阵与 screen 的单 run 时长（约 9–13 min/30 epochs），
  总计约 **10–15 GPU·h**；8 卡并行（每卡串行队列）wall-clock 约 1.5–2 h。

## 7. 训练与评估协议（所有配置严格一致）

- lookback 720、`period_len=24`、Huber loss、max epochs 30、最低 validation loss
  checkpoint、每配置一次 `trainer.test()`。
- 每配置记录：best-checkpoint 的 val MSE/MAE、test MSE/MAE、head 参数量、训练时长。
- 运行环境：A800 服务器 `time` conda 环境（Python 3.10、torch 2.6.0+cu124、
  pytorch-lightning 2.6.5），完整解释器路径调用，正式命令带 `--require-cuda`，
  用 `CUDA_VISIBLE_DEVICES`/`--gpus` 分卡；结果文档必须注明环境（见
  `REMOTE_SERVER.md` 注意事项 2）。
- 数据脚本必须在仓库根目录运行（`data_provider` 的 `root_path` 为相对路径）。

## 8. 准备事项

1. **数据上传**（当前服务器仅有 ETTh1）：
   - `ETTh2.csv`、`ETTm1.csv`、`ETTm2.csv` → 服务器 `resources/all_datasets/ETT/`；
   - `weather.csv` → 服务器 `resources/all_datasets/weather/`；
   - 来源为本地 `datasets.zip` 对应目录，上传后先用 1-epoch smoke 验证加载。
2. **Runner 扩展**（实施时的代码改动，单独 commit）：
   - `scripts/run_joint_pooled_lowrank_phase_a.py` 的 `--dataset` choices 当前
     硬编码 `["ETTh1", "ETTm1"]`，需扩展 `ETTh2`、`ETTm2`、`Weather`；
   - 该脚本为 validation-only（无 `trainer.test`），需新增 `--evaluate-test`
     透传开关（一次 test 读取），其余协议不变；
   - 控制组 `phase_only` / `direct_nlinear` 已在该脚本中实现，直接复用；
   - 验证：`py_compile` + `pytest tests/ -q` 全绿。
3. **服务器检查**：`nvidia-smi` 确认目标卡空闲；确认磁盘空间足够 70 个 run 的
   checkpoint/日志输出（输出根目录 `research_runs/joint_lowrank_rank_sweep_v1/`，
   已被 gitignore 覆盖）。

## 9. 运行命令模板（以扩展后 runner 为准）

```bash
cd /home/yyk/yyk03/niuyiming/PhaseFormer
nohup /home/yyk/yyk03/miniconda3/envs/time/bin/python \
  scripts/run_joint_pooled_lowrank_phase_a.py \
  --dataset ETTh2 --horizon 96 --seed 2021 \
  --pool-factors 1 \
  --relative-ranks 1,0.25,0.125,0.0625,0.03125 \
  --evaluate-test --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/joint_lowrank_rank_sweep_v1 \
  > sweep_etth2_h96.log 2>&1 &
```

每个 setting 一条命令；10 条命令可顺序或分批提交。`--relative-ranks` 的五个值
分别对应 q ∈ {1, 1/4, 1/8, 1/16, 1/32}。

## 10. 待填充结果表格

> 约定：所有单元格 `MSE / MAE`；`—` 表示待填充；Δ% = (ref − new)/ref × 100，
> 正为优于参照。

### 表 1：主结果矩阵（test MSE/MAE，seed 2021）

| Setting | phase_only | direct_nlinear | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|---|
| ETTh1-96 | — | — | — | — | — | — | — |
| ETTh1-192 | — | — | — | — | — | — | — |
| ETTh2-96 | — | — | — | — | — | — | — |
| ETTh2-192 | — | — | — | — | — | — | — |
| ETTm1-96 | — | — | — | — | — | — | — |
| ETTm1-192 | — | — | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — | — | — |
| Weather-96 | — | — | — | — | — | — | — |
| Weather-192 | — | — | — | — | — | — | — |

### 表 2：各档位相对 Golden 的 ΔMSE% / ΔMAE%（负值 = 低于 Golden）

Golden 取自 `docs/PhaseFormer_gold_standard.md`（三位小数，指示性参照）。

| Setting | Golden MSE/MAE | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|---|
| ETTh1-96 | 0.359 / 0.382 | — | — | — | — | — |
| ETTh1-192 | 0.397 / 0.404 | — | — | — | — | — |
| ETTh2-96 | 0.275 / 0.338 | — | — | — | — | — |
| ETTh2-192 | 0.341 / 0.376 | — | — | — | — | — |
| ETTm1-96 | 0.293 / 0.344 | — | — | — | — | — |
| ETTm1-192 | 0.323 / 0.361 | — | — | — | — | — |
| ETTm2-96 | 0.163 / 0.256 | — | — | — | — | — |
| ETTm2-192 | 0.219 / 0.293 | — | — | — | — | — |
| Weather-96 | 0.148 / 0.195 | — | — | — | — | — |
| Weather-192 | 0.193 / 0.237 | — | — | — | — | — |

### 表 3：各档位相对 matched `phase_only` 的 ΔMSE% / ΔMAE%

| Setting | q=1 | q=1/4 | q=1/8 | q=1/16 | q=1/32 |
|---|---|---|---|---|---|
| ETTh1-96 | — | — | — | — | — |
| ETTh1-192 | — | — | — | — | — |
| ETTh2-96 | — | — | — | — | — |
| ETTh2-192 | — | — | — | — | — |
| ETTm1-96 | — | — | — | — | — |
| ETTm1-192 | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — |
| Weather-96 | — | — | — | — | — |
| Weather-192 | — | — | — | — | — |

### 表 4：因子化效应与压缩效应（ΔMSE% / ΔMAE%）

- 因子化参数化效应 = q=1 相对 `direct_nlinear`；
- 低秩压缩效应 = 各压缩档相对 q=1（同结构、同协议的配对差）。

| Setting | 因子化效应 q=1 vs direct_nlinear | q=1/4 vs q=1 | q=1/8 vs q=1 | q=1/16 vs q=1 | q=1/32 vs q=1 |
|---|---|---|---|---|---|
| ETTh1-96 | — | — | — | — | — |
| ETTh1-192 | — | — | — | — | — |
| ETTh2-96 | — | — | — | — | — |
| ETTh2-192 | — | — | — | — | — |
| ETTm1-96 | — | — | — | — | — |
| ETTm1-192 | — | — | — | — | — |
| ETTm2-96 | — | — | — | — | — |
| ETTm2-192 | — | — | — | — | — |
| Weather-96 | — | — | — | — | — |
| Weather-192 | — | — | — | — | — |

### 表 5：响应曲线判定汇总（每 setting 一行，实验完成后填写）

| Setting | val 最优档位 | test 最优档位 | 压缩响应形状（单调升 / 单调降 / U 型 / 平坦） | 备注 |
|---|---|---|---|---|
| ETTh1-96 | — | — | — | — |
| ETTh1-192 | — | — | — | — |
| ETTh2-96 | — | — | — | — |
| ETTh2-192 | — | — | — | — |
| ETTm1-96 | — | — | — | — |
| ETTm1-192 | — | — | — | — |
| ETTm2-96 | — | — | — | — |
| ETTm2-192 | — | — | — | — |
| Weather-96 | — | — | — | — |
| Weather-192 | — | — | — | — |

## 11. 分析与判定规则（预注册，结果回填前不得修改）

1. 配对比较只在同一 setting 内进行（同数据、同 seed、同协议）；主响应为
   test MSE/MAE 相对 q=1 的 Δ，validation 响应作为旁证列。
2. 低秩压缩效应是否成立：
   - q<1 相对 q=1 的 ΔMSE（或 ΔMAE）方向在 **≥8/10 setting 一致**且中位
     |Δ| ≥ 1% → 记录为「一致低秩效应信号」；
   - 方向在 4–7/10 setting 一致 → 「数据集相关 / 部分信号」；
   - ≤3/10 → 「本预算下无可检测低秩效应」。
3. 单指标 vs 双指标措辞遵守金标准规则 §4；Golden 只保留三位小数，临界差异
   不得表述为稳定收益。
4. 全部结论必须标注：**single-seed、test-set-exposed exploratory**；不得宣称
   跨 seed 或跨 horizon 稳定性；不得据此修改任何 preset。
5. 若出现一致低秩效应信号，后续仅追加多 seed（2021/2022/2023）复核响应最陡的
   2 个 setting；复核通过前不进入任何结构性结论。
6. 若无信号，保留 `direct_nlinear` 路线，不再追加 pool/smoothing 维度（与
   前次受控计划的止损规则一致）。

## 12. 边界与已知风险

- 单 seed 无方差估计；任何两档位间 <1% 的差异都可能落在优化噪声内，解释时
  必须保守。
- q=1 的两层因子化头与 `direct_nlinear` 的参数量、初始化结构不同，二者差异
  属「因子化参数化效应」，与压缩效应分开报告（表 4 独立列），不得混入低秩结论。
- 前次 screen 在 ETTh1-H96/ETTm1-H96 的联合训练候选相对 Golden 双指标退化；
  若本实验复现该模式，应在结论中明确低秩调整无法修复联合训练形态的基线劣势。
- Weather 上 `period_len=24` 等默认策略沿用 runner 现有配置；若数据加载或
  策略差异导致协议不可比，先修复协议再回填表格，不得带病比较。
