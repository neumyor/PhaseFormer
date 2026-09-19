# E14 · §4.2 主结果矩阵 — 阶段 1：计划

> 实验单元：**E14**（`docs/PhaseFormer_L_execution_schedule.md` §2）｜minipaper 回填目标：**§4.2 主表 + 变体行 + 必答 (a)(b)(c)**
> 产物根：`research_runs/phaseformer_L_e14_main_v1/`｜代码：`scripts/phaseformer_L/e14_main_matrix.py`（阶段 A）、
> `scripts/phaseformer_L/e14_read_test.py`（阶段 B）
> 六阶段文档：`01_plan.md`（本文）→ `02_static_check.md` → `03_smoke.md` → `04_run.md` → `05_audit.md` → `06_writeback.md`

## 1. 目标

用**既有 preset**（不新增模型代码）填满 minipaper §4.2 的主表：24 个主 setting
（ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity × H96/192/336/720）加 Traffic 4 个探索性附录
setting，每格三 seed（2021/2022/2023），并回答必答 (a)(b)(c)。

## 2. 矩阵（`--stage plan --verify` 已对账）

| 行 | mechanism / 配置 | setting 域 | 复用 | 新训 |
|---|---|---|---:|---:|
| `phase_only` | `no_residual` | 24 + Traffic 4 | 6 | 66 |
| PhaseFormer-L | `weak_residual` + `head_type=shared` | 24 + Traffic 4 | 7 | 63 |
| L-q1/4 | `pooled_lowrank`，`pool_factor=1`，`rank=H/4` | 24 + Traffic 4 | 7 | 63 |
| L-q1/8 | `pooled_lowrank`，`pool_factor=1`，`rank=H/8` | 24 + Traffic 4 | 7 | 63 |
| L-rcrf | `rcrf_nlinear_plain` | 24 + Traffic 4 | 0 | 84 |
| A1 | `gold_combo_reliability_s2` | 24（不含 Traffic） | 0 | 72 |
| **合计** | | | **81** | **411** |

总计 492 个 cell。

## 3. 协议（minipaper §4.0，已冻结）

- lookback 720、period 24、huber、30 epoch、percent 100、`--stage confirm`、best-validation checkpoint、
  seeds 2021/2022/2023、`--require-cuda`、`--num-workers 4`、每 checkpoint 只读一次 test。
- **阶段 A 绝不传 `--evaluate-test`**：测试集在阶段 B 由 `e14_read_test.py` 单独读一次。
- 新格超参：preset 默认 `gate_init=0.2`、`lr=1e-3`（D-2）。
- 复用格保留其 Stage-0 冻结 `(gate, lr)`。
- **§4.2 表内存在三种 gate 先验**：新格 0.2 / `l_rcrf` 与 `a1` 由 preset 自持 0.5 /
  复用格 Stage-0 冻结值（0.5 或 0.2）。表注必须逐行标注。

## 4. 复用审计规则（阶段 A 的 `resolve_reuse`）

- 只接受**白名单根目录**：`rank_sweep_2_stage1`、`..._multiseed_stage1_20260914_{v3,v4,v5,repair_v1}`、
  `top2_direction_retention_v1`。显式排除 E1（`joint_lowrank_rank_sweep_v1`）、scratch 根、
  选择类与干预类实验目录。
- 逐 run 校验：`dataset/horizon/seed` + `mechanism` + `head_type` + `rank` + `pool_factor` 匹配；
  **排除** `weak_residual_projection`（冻结子空间臂）、`input_hypothesis != none`、
  `smooth_ratio != 0`、有 `init_checkpoint` 的 run。
- 协议字段必须全等：`lookback=720 ∧ loss=huber ∧ max_epochs=30 ∧ percent=100 ∧ period=24`。
- **test 证据**：run 自身 `metrics.csv` 有非空 `test_mse/test_mae`，或来自登记的
  `EXTERNAL_TEST_EVIDENCE`（E8 的分阶段协议把 test 读数记在 `top2_direction_retention_v1/results.csv`，
  要求 `test_read_status ∈ {read, reused}` 且 `val_relative_difference ≤ 1e-3`）。
- `--verify` 是**硬门**：任何声称的复用格解析不出来即拒绝开跑。

## 5. 调度

- 一卡一 run、8 卡轮转；按 `COST_HINT` 的估算耗时**降序**发车（Traffic 先发），避免尾部空卡。
- `setsid nohup` 提交，日志 `~/niuyiming/logs/e14_main.log`；失败重试 1 次；`--poll-seconds 15`。
- 启动前 `pgrep -af "scripts/.*\.py"` 确认服务器空闲；运行期间**不做**代码同步中被训练进程导入的文件。

## 6. 预期产物

`stage_a_manifest.json`、`stage_a_reuse_audit.json`、`stage_a_summary.json`、`runs/<run_id>/`、
`_logs/<cell>.log`；阶段 B 追加 `results.csv`、`test_read_summary.json`、`test_read/<cell>.json`。

## 7. 静态检查清单（移交阶段 2）

1. `compile()` 通过。
2. `--stage plan --verify` 输出 `total = 492`、`by_arm` 与 §2 表格逐格相符、`reuse_cells_resolved = 81`、`missing = []`。
3. 三条复用链各自的解析数与 §2.1 审计清单一致（`phase_only` 18、`l_main` 21、`q1/4` 21、`q1/8` 21）。
4. 抽查一个复用格的 `config.json`：`mechanism`/`head`/`rank`/协议字段/冻结超参与白名单一致。
5. `--dry-run` 打印的 20 个 cell 的状态（new/reused）与预期一致。
6. 确认阶段 A 的 argv 中**没有** `--evaluate-test`。
