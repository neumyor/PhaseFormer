# E14 · §4.2 主结果矩阵 — 阶段 3：冒烟测试

> 目的：在 411 个 cell 开跑**之前**确认每个臂的**精确命令行**被 runner 接受，
> 并在最大通道数设定上实测单 epoch 耗时以校准预算。

## 1. 命令

```bash
cd ~/niuyiming/PhaseFormer
setsid nohup ~/niuyiming/run_e14_smoke.sh > ~/niuyiming/logs/e14_smoke.log 2>&1 < /dev/null &
# 内部：python scripts/phaseformer_L/e14_main_matrix.py --stage smoke \
#         --gpus 0,1,2,3,4,5,6,7 --smoke-epochs 1 \
#         --output-root research_runs/phaseformer_L_e14_main_v1
```

覆盖块 `SMOKE_BLOCK = (ETTh2-96, Electricity-336, Traffic-96)` × 全部 6 个臂 =
**18 个 cell**，seed 2021，1 epoch，产物写入独立根目录 `..._main_v1_smoke`（不污染正式产物）。

## 2. 结果：**18/18 通过，无失败**

```json
{"event": "smoke_finished", "ok": 18, "failed": []}
E14_SMOKE_EXIT=0
```

## 3. 逐臂产物核对（阶段 3 的核心目的）

从冒烟产物 `runs/*/config.json` 逐个读出，确认**每个臂的 mechanism / head / rank / gate / lr 与设计一致**：

| 臂 | mechanism | head | rank | pool | `gate_init` | `lr` |
|---|---|---|---|---:|---:|---:|---:|
| `phase_only` | `no_residual` | — | — | — | — | 1e-3 |
| `l_main` | `weak_residual` | `shared` | — | — | **0.2** | 1e-3 |
| `l_q1_4` | `weak_residual` | `pooled_lowrank` | **H/4**（336→84，96→24） | 1 | **0.2** | 1e-3 |
| `l_q1_8` | `weak_residual` | `pooled_lowrank` | **H/8**（336→42，96→12） | 1 | **0.2** | 1e-3 |
| `l_rcrf` | `rcrf_nlinear_plain` | `shared`（preset 自持） | — | — | **0.5**（preset） | 1e-3 |
| `a1` | `gold_combo_reliability_s2` | — | — | — | **0.5**（preset） | 1e-3 |

**这张表同时确认了 D-2 的"三种 gate 先验"**：0.2（新格）、0.5（`l_rcrf`/`a1` 由 preset 自持）
在同一次冒烟里同时出现，且与设计一致。若当时按"两种协议"的错误假设开跑，§4.2 表注会写错。

## 4. 单 epoch 实测与预算校准

| 设定 | 单 epoch | 30 epoch 外推 | 来源 |
|---|---:|---:|---|
| Traffic-96（862 通道，batch 8） | **138.2 s**（wall 152.3 s） | ≈ 1.15 h | `metrics.csv: elapsed_sec` |
| ETTh2-96 | 数秒 | ≈ 0.02 h | 历史中位数 47 s/run |
| Electricity-336 | 约 1 min | ≈ 0.48 h | 历史中位数 1717 s/run |

据此外推正式矩阵：非 Traffic 355 runs ≈ 54 GPU·h，Traffic 56 runs ≈ 65 GPU·h，
合计 **≈ 119 GPU·h**；8 卡 wall-clock 约 15 h。

## 5. 冒烟还捕获的缺陷与配置事实

| # | 事项 | 说明 |
|---|---|---|
| 1 | Traffic 数据格式 | 首次冒烟在 `Dataset_Custom_Multi` 的 `cols.remove(self.target)` 处失败：仓库约定 CSV **最后一列必须名为 `OT`**（Electricity 为 `...,319,OT`，Weather 为 `...,Tlog (degC),OT`），而解压出的 Traffic 列为 `0..861`。已在 `e00_prepare_traffic.py` 中把末列改名 `OT` 并加入写出后的自校验；重跑冒烟通过 |
| 2 | `l_rcrf`/`a1` 的 gate | 二者由 preset 自持 0.5，`arm_command` 的 `gate_init=0.2` 注入仅对 `mechanism == weak_residual` 生效——冒烟实测确认 |
| 3 | Traffic 训练成本 | 862 通道 + batch 8，单 run ≈ 1.1 h，是预算中最大的一项；据此保留 Traffic 附录的完整范围（48 runs）而非收窄 |

## 6. 结论

阶段 3 通过：**6 个臂 × 3 个代表设定全部跑通**，臂指纹与设计逐字段一致，Traffic 成本已实测。
允许进入阶段 4。
