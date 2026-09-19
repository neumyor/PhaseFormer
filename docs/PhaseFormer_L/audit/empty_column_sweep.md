# 回填产物的「空列」机械扫描（2026-09-19）

## 1. 为什么需要这一步

本线反复出现的失败模式**不是崩溃，而是"表头有列、每一行都空"**——minipaper 于是静默丢掉一个
必需字段。已手工发现 4 次：

| # | 空列/缺失 | 发现方式 |
|---|---|---|
| 1 | Golden 表解析只覆盖 12/28 行（正则匹配不了 `ETTh1`） | 合成冒烟 |
| 2 | E16 的 `majority_input_group_label` 列根本不存在 | schema 对拍 |
| 3 | §4.2「来源/披露」列留空 | 逐列对照 minipaper 表头 |
| 4 | must-answer (b) 完全未实现 | 逐条对照 §4.2 的三个必答 |

因此把这次搜寻变成**机械检查**：`scripts/phaseformer_L/check_builder_outputs.py`。
它对每个 CSV 报告三类信号——**整列为空**（`always_empty`，真实产物上即缺陷）、
**填充率低于阈值的列**（`mostly_empty`，通常意味着某个 join 键对多数格失败）、
以及**取值恒定**的列（`constant`，需人眼确认是设计常量还是意外常量）。

用法：

```bash
python scripts/phaseformer_L/check_builder_outputs.py \
    --csv <builder output>.csv [--min-fill 0.9] [--expect-empty <column>] \
    --output <report>.json
```

## 2. 六个回填工具的输出扫描结果

用与真实表头一致的合成夹具逐个跑通并扫描：

| 产物 | 行×列 | `always_empty` | 判读 |
|---|---|---|---|
| E14 `main_table.csv` | 28×58 | **无** | `a1_*` 填充率 24/28 = 0.857：**符合设计**（`a1` 不在 Traffic 上训练，其域为 24 个 setting） |
| E14 `variant_table.csv` | 6×13 | 参数列 4 个为空 | **符合设计**：夹具未提供 `--params`，故参数量列按既定行为置空并在 `audit.json` 留因。**正式运行时 `e14_params.py` 先跑**（流水线步骤 3 已固定该顺序），故真实产物不会为空 |
| E17 `conditional_table.csv` | 7×36 | **无** | 无空列问题 ✓ |
| E18 `negative_table.csv` | 5×6 | **无** | 无空列问题 ✓ |
| E16 `intervention_table_44.csv` | 21×37 | **无** | 无空列问题 ✓ |
| E16 `dissection_table_44.csv` | 21×33 | 首轮报 `head_kind` 空 | **夹具缺陷而非工具缺陷**：夹具漏填该列。补填后输出为 `pooled_lowrank` 14 / `dense_shared` 7 = 21 ✓，扫描 0 问题。真实 `dissection_table.csv` 确有 `head_kind`（第 9 列，已与真实表头核对） |

`constant` 列检出的都是设计常量或结构性常量，例如 E14 的 `*_n` 恒为 3（三 seed）、
E17 的 `*_seeds` 与 `*_source`、E16 的随机带上下界（夹具里是常数）。**这些不是缺陷**，
但值得每次正式回填后过一眼。

## 3. 扫描本身的两点边界（如实记录）

1. **夹具的空列 ≠ 真实产物的空列**。本扫描的作用是在**真实数据到达之前**确认"如果输入齐全，
   输出就不会有空列"；它不能代替真实产物上的同一次扫描。因此流水线在每步回填后应再跑一次，
   并以 `--output` 落盘报告。
2. 扫描无法判断"某个非空值是否正确"，只能判断"是否有值"。数值正确性由各实验自己的
   验收判据覆盖（E14 的 §10 不变量与主张、E16 的 `algebra_failures` 与 `reference_parity`、
   E17 的 H1 来源、E18 的行 3 口径、E19 的 ρ）。

## 4. 建议的固定用法

每个 E 单元完成阶段 6 前，对其回填产物执行一次：

```bash
python scripts/phaseformer_L/check_builder_outputs.py \
  --csv research_runs/phaseformer_L_<eid>_<slug>_v1/<table>.csv \
  --output  research_runs/phaseformer_L_<eid>_<slug>_v1/empty_column_report.json
```

退出码非零即表示存在整列为空或填充率不足的列，须在回填前解释清楚（设计使然 or 缺陷）。
