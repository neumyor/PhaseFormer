# E15 · §4.3 相位补空间维数 — 阶段 6：回填

> 状态：**已完成**。回填目标：minipaper **§4.3 表（28 行）+ §4.3 三类图**。

## 1. 回填动作

| 目标 | 动作 | 结果 |
|---|---|---|
| minipaper §4.3 表 | 把原占位行 `（28 行待填；先导 7 行见 §4.1）` 替换为 28 行实测值 | 已替换；表头沿用原文（含转义的 `\|cos\|`） |
| minipaper §4.3 表注 | 新增「表注与披露（E15，2026-09-19 回填）」6 条 | 已写入：train/validation 口径与 test 不同源、21/7 覆盖来源、模板定义与 2 位小数参照的分辨率、与先导区间的关系、`λ_1/Σλ` 分母语义、产物路径 |
| §4.3 三类图 | 图已生成于 `figures/`；论文排版时引用 | 已产出（scree / `b_1` lag 剖面 / `a_1` horizon 剖面） |
| `docs/agent-log.md` | 追加一条 E15 记录 | 见下节 |

## 2. 数值权威副本

回填的每一个数字都来自 `research_runs/phaseformer_L_e15_dimension_v1/dimension_table.csv`，
逐行对应；该 CSV 为数值权威副本。用于生成 markdown 的映射为：

| 表列 | CSV 列 | 格式 |
|---|---|---|
| `λ_1/Σλ` | `lambda1_share_of_achievable` | 3 位小数 |
| `pred_dims_90` | `pred_dims_90` | 整数 |
| PR | `PR` | 2 位小数 |
| `b_1` 最佳模板（\|cos\|） | `b_1_best_template` + `b_1_best_template_abs_cos` | `exp τ=<τ> (<cos>)`，τ 取自模板名 |
| `a_1` vs 常值 \|cos\| | `a1_vs_const_abs_cos` | 3 位小数 |
| `used_var_share(1)` | `used_var_share_r1` | 3 位小数 |

## 3. 披露要求（已满足）

- ✅ train/validation 口径，**不读 test**；
- ✅ 与 §4.2 的 test 增益列**不同源**；
- ✅ 21 个新 setting 标注为 coverage 扩展（`source=new_28_minus_7`），与既有 7 行来源区分。

## 4. 未回填 / 保持留白

- §4.2、§4.4–§4.7 的表**不**由本实验回填（各自等待 E14/E16/E17/E18/E19）。
- §4.3 表内**无留白**：28 行全部有实测值。

## 5. agent-log 追加条目（已写入）

```text
## 2026-09-19 — E15：§4.3 相位补空间维数（28 setting，train/validation）
实验：E15（docs/PhaseFormer_L/e15_dimension/）。代码 scripts/phaseformer_L/e15_dimension.py。
命令：python scripts/phaseformer_L/e15_dimension.py --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Traffic
      --horizons 96,192,336,720 --seq-len 720 --save-moments
      --output-root research_runs/phaseformer_L_e15_dimension_v1
产物：research_runs/phaseformer_L_e15_dimension_v1/{dimension_table.csv,b1_template_detail.csv,
      leading_direction.csv,optimal_rank_capture.csv,leading_directions.npz,moments_*.npz(28),figures/(3)}
验证：--verify-existing 门通过（7/7 setting，moments 相对差 0.0，111/111 项）；正式运行 28/28 完成，
      退出码 0，wall-clock 16m42s；阶段 5 审校 11/11 项通过。
回填：minipaper §4.3 的 28 行表 + 表注 6 条。
已知偏差：报告 §2.6(c) 是 2 位小数参照，ETTh2-96 位于舍入边界（0.5749 vs 0.58），已在表注披露；
      精确值另存 b1_template_detail.csv。
关键新事实：pred_dims_90 实测上界为 7（Traffic-192/336），先导的"2–4 维"未覆盖 Traffic；
      Traffic 的 b_1 一致落在 τ=168 且 used_var_share(1) 最高（0.177–0.353），
      与 E19 测得的最低 τ̂（25.1 步）方向相反，须在 §4.7 表注区分两个量。
```
