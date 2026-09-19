# E16 · §4.4 解剖与干预 — 阶段 2/3：静态检查与冒烟

> 代码：`scripts/phaseformer_L/e16_dissection.py`
> 状态：**阶段 3 未通过**（发现 2 个真实缺陷，已退回修复）；阶段 2 通过。

## 1. 阶段 2：静态检查（通过）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `compile()` | 通过 |
| 2 | 作用域 cell 数 | **63** = 3 臂（`l_main`/`l_q1_4`/`l_q1_8`）× 7 个 test-selected setting × 3 seed |
| 3 | 逐臂计数 | `l_main` 21 + `l_q1_4` 21 + `l_q1_8` 21 |
| 4 | 复用解析 | **63/63 全部 `status=reused`**，checkpoint 全部解析到真实 E3 系 run（含一个 `attempts/002`） |
| 5 | 低秩臂的秩 | `l_q1_4` → H96:24 / H192:48 / H336:84 / H720:180；`l_q1_8` → 12/24/42/90，**与 `H/4`、`H/8` 一致** |
| 6 | 不读 test | `evaluation_split: val`、`test_split_read: false`；日志逐 setting 打印 `test rows never read` |
| 7 | 零分布规模 | `random_repeats=100`、`random_rrr_repeats=100`、`random_rrr=true`（默认） |

### 1.1 阶段 2 挡下的缺陷：作用域被张成**笛卡尔积**

原实现在未显式给 `--datasets/--horizons` 时，用"数据集集合 × horizon 集合"构造 cell，
而 7 个 test-selected setting 是**不规则**的（ETTh2 只有 96/720；Electricity 只有 336）。
交集运行会把 7 个 setting 静默放大成 **16 个**（4 数据集 × 4 horizon），
引入 9 个登记范围外的 cell（`ETTh2-192/336`、`ETTm2-336/720`、`Weather-336/720`、
`Electricity-96/192/720`）——它们既不在 §4.1 的先导集合里，又依赖当时尚未训练的 E14 格。

修复：作用域改为**显式 pair 列表**，`--datasets/--horizons` 只做**收窄**、不做扩张；
若过滤后为空则报错退出而不是静默跑空。修复后 dry-run 恰好 63 cell、全部可解析。

## 2. 阶段 3：冒烟（**未通过**，2 个缺陷）

```bash
python scripts/phaseformer_L/e16_dissection.py --e14-root research_runs/phaseformer_L_e14_main_v1 \
  --output-root research_runs/phaseformer_L_e16_smoke \
  --datasets ETTh2 --horizons 96 --seeds 2021 --arms l_main,l_q1_8 \
  --max-batches 2 --random-repeats 10 --random-rrr-repeats 10 --mem-budget-mb 2048
```

### 2.1 缺陷 1：稠密头的"未干预臂不变量"失败

```text
[train] ETTh2-96: pairs=54775 channels=7 channel_block=7 windows=7825 rows parsed=11520 (test rows never read)
[cell] l_main   ETTh2-96  seed=2021 head=dense_shared  rank_dim=720  pairs=3584 invariant=FAILED gap=4.42e-02 (588.1s)
```

该不变量要求"未干预臂必须复现模型自己记录的 fused 输出"。计划文档 §8(d) 记载了
注册代数会把 `W_dec @ encoder_bias` 重复计入，并声明对**稠密头**该 gap 为 0——
**实测为 4.42e-02，不为 0**。因此稠密头的映射（`encoder=I`、`encoder_bias=0`、
`decoder=linear.weight`）实现有误，`l_main` 的解剖与干预数值在修复前不可用。

### 2.2 缺陷 2：头的类型没有按 cell 从 checkpoint 推导

`l_q1_8`（应为 `pooled_lowrank`）被按 `shared` 稠密头构造：

```text
RuntimeError: Error(s) in loading state_dict for PhaseFormer:
  Missing key(s) in state_dict: "weak_period_residual.linear.weight", "weak_period_residual.linear.bias".
  Unexpected key(s) in state_dict: "weak_period_residual.encoder.weight", "weak_period_residual.encoder.bias",
                             "weak_period_residual.decoder.weight", "weak_period_residual.decoder.bias".
```

即 `l_q1_8` 的 21 个 cell 全部无法加载。修复方向：从该 run 的 `config.json` 的
`weak_period_residual_head_type`（或直接从 checkpoint 的键集合）推导头类型，并在加载前断言。

### 2.3 缺陷 3（工程）：设备选择与 E14 争用

日志显示 `device: cuda:0`，而当时 8 张卡正被 E14 主表矩阵占满；单 cell 在该条件下耗时 588 s。
冒烟应能显式指定设备、并在未指定时**默认不抢占**正在使用的卡。

## 3. 结论与后续

阶段 3 **不通过**：3 个缺陷（1 个数值正确性、1 个加载正确性、1 个工程）已连同完整报错回退给实现方修复。
**在修复并通过冒烟之前，E16 的 63 个 cell 不得进入正式运行**——这 63 个 cell 是 §4.4 两表（解剖表与
10 臂干预表，含新增的"随机 RRR 子空间 drop"对照）的唯一来源。

价值说明：本次冒烟用**单 cell、2 个 batch** 就发现了会污染整张 §4.4 表的两处缺陷；
若直接提交 63-cell 正式运行，会在数小时后产出全部错误数值。
