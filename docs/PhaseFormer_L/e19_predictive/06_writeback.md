# E19 · §4.7 预测力 — 阶段 6：回填（阶段 1 部分）

> 回填目标：minipaper **§4.7 表**。阶段 1 回填的是"判定口径已冻结"的**方法与诊断**；
> 表内两列 ρ 与诊断准确率须等 E14 完成后回填。

## 1. 本阶段已回填

| 目标 | 动作 | 结果 |
|---|---|---|
| minipaper §4.7 表注 | 新增 3 条：①判定口径已冻结（`ν=tau_hat_steps`、`ν*=57.35`、三候选可分离性实测表）；②`τ̂` 的有限样本偏差表；③诊断列判错须如实报告（含 Electricity 已知判错） | **已写入** |
| minipaper §3.4.3 | 开关由"模型的一部分"改为**诊断列**（D-5），并写入冻结的 `ν`/`ν*` 与判错披露要求 | **已写入** |
| minipaper §5 第 6 条 | 改写为"诊断列不是模型的一部分"，并纠正"由 6 个已知 setting 拟合"为**证据支持的 5 个已知符号数据集** | **已写入** |
| `docs/agent-log.md` | 追加 E19 阶段 1 记录 | 见 §3 |

## 2. 数值权威副本

`research_runs/phaseformer_L_e19_predictive_v1/level_statistics.csv`（28 行）为数值权威副本；
`dataset_level_statistics.csv`（7 行）为其按数据集的聚合；`nu_star_diagnostic.json` 为阈值冻结依据。
§4.7 表内两列 ρ 将由 `predictive_power.csv` / `predictive_power_summary.json` 回填。

## 3. 阶段 2 的 ρ 已回填（2026-09-20；原先的"保持留白"已解除）

| 表项 | 状态 |
|---|---|
| `cycle_level_std` 与 ΔMSE 的 ρ | **已填 −0.039**（p=0.842） |
| `last_cycle_shift` 与 ΔMSE 的 ρ | **已填 −0.139**（p=0.480） |
| `τ̂` 与 ΔMSE 的 ρ | **已填 −0.750**（p=4.3e−06） |
| 三统计量分别与 `g` 的 ρ | **已填** 0.183 / 0.303 / 0.325（n=21，全部不显著） |
| 诊断 `s` 的逐格判对/判错 | 已进 `diagnostic_accuracy`（28 setting / 22 hits / 6 misses），论文按此叙述 |

**回填方式（脚本化，不手抄）**：`scripts/phaseformer_L/fill_minipaper_47.py`
从 `predictive_power_summary.json` 按 `STATISTICS` **下标**填两列，精度 **3 位小数**
（与校验器 `compare_number(..., 3)` 一致）；非有限 ρ 会写成字面 `NaN`，本次 `non-finite: 0`。
工具带一条**防漂移断言**：其 `STATISTICS` 必须与校验器逐项相等，否则拒绝运行。

**回填动作与校验（实测）**：

```text
dry run:  cells filled: 6   non-finite: 0 []
write  :  wrote docs/PhaseFormer_L_minipaper.md
verify :  verify_minipaper_fill.py → "every filled section-4 cell matches its artifact
          at the displayed precision", exit 0；§4.7 表 1 空格数 6 → 0
```

**与预登记不符之处如实记录**：注 4 预期前两行 ρ（vs ΔMSE）为正（"反序"），实测为 −0.039 / −0.139 且不显著
⇒ **"反序"在 ΔMSE 相关上未复现**，更准确的读法是这两行"**对增益无预测力**"。
三个数字**照实测写，未做任何阈值或口径迁移**（summary 的 `frozen_threshold` 亦明写未被事后重拟合）。

## 4. agent-log 追加条目（已写入）

```text
## 2026-09-19 — E19 阶段 1：§4.7 的 28-setting 训练集电平统计量与 ν* 冻结
实验：E19（docs/PhaseFormer_L/e19_predictive/）。代码 scripts/phaseformer_L/e19_predictive_stats.py。
命令：python scripts/phaseformer_L/e19_predictive_stats.py --datasets <7> --horizons 96,192,336,720
      --num-workers 4 --output-root research_runs/phaseformer_L_e19_predictive_v1
产物：level_statistics.csv(28) / dataset_level_statistics.csv(7) / nu_star_diagnostic.json / run.yaml
验证：单元测试 14/14；阶段 5 审校 9/9；E19_STATS_EXIT=0。
冻结：ν=tau_hat_steps，ν*=57.35（可分离区间 (51.11,63.58) 的中点）；另两候选在已知符号数据集上反序，
      不具判别力。τ̂ 有 0.48–0.87 的系统性低估，仅作序数仪器（偏差表已写入 §4.7 表注）。
已知判错：Electricity 的 τ̂=54.46 落在区间内，而实测其修正器相对 matched phase_only 为 +3.6%，
      即诊断在该格判错；按 §4.0 报告规则须逐格列出。
边界：只读训练集（run.yaml reads_test=false）、不改任何模型代码、不读 test。
```
