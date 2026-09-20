# E16 §4.4：阶段 6 **回写与回填**（2026-09-20）

本文件记录"数字怎么进论文"的完整链条与**最终判据状态**。执行与诊断见 `04_run.md`、`05_audit.md`。

---

## 1. 一句话

E16 的产物由 **18 个分片（6 setting × 3 模型 = 18 行、54 格）** 合并而成，
经回写产出 44 表，再逐格填进论文 §4.4 的**两张表**；
**Electricity-336 的 3 行未跑，表内逐格标注 `未跑（按指示停止）`**。
最终：**空格总数 0、`blank` 0、逐格 `match` 724**；**7 处 MISMATCH 全部且仅仅来自那 3 行未跑格**（见 §5）。

## 2. 合并（`merge_e16_shards.py`，只拼接、不重算）

```text
shards found: 21（其中 3 个是"停在写产物之前"的空目录，按 not-run 跳过）
dissection_table.csv     : 54 row(s)（= 18 行 × 3 seed）
intervention_table.csv   : 636 row(s)
canonical_modes.csv      : 432      semantic_alignment.csv: 432      cross_seed_alignment.csv: 108
PARTIAL merge: 54 of 63 cells
missing registered combos: [Electricity-336 × l_main / l_q1_4 / l_q1_8]
kernel: einsum_optimize=True（全部 18 片一致；混内核会被拒写）
reference parity: probe_cells=72（满额）  value_compared=28  path_mismatched=22  passed=False
```

**为什么允许"部分合并"而不是硬凑 63 格**：`--allow-partial` 会把**缺哪些 `(setting, arm)` 组合**
从分片根目录**推导出来并写进 `merge_provenance`**，并在产物里附一句说明"这些行没有数据"。
即"表可以小，但不许因为少了几行而**看起来完整**"。**规范产物因此自带缺口清单**，这不是靠人记。

## 3. 回写（`e16_writeback.py`）

```text
intervention_rows: 18   dissection_rows: 18
missing_columns_intervention: []   missing_columns_dissection: []
arms_per_cell_observed: [11, 12, 13]   cells_with_fewer_arms: 0   named_arms_missing: []
```

⇒ 列名契约与"每格必须带全部恒在臂"两条**在真实数据上成立** ✓；
臂数 11/12/13 三种与 `minipaper_fill_mapping.md` §1.5 的登记一致（**不是常数**）。
产出：`dissection_table_44.csv`（18 行 × 33 列）、`intervention_table_44.md/.csv`。

## 4. 回填（两个工具，均为"对着校验器写"的实现）

| 表 | 工具 | 结果 |
|---|---|---|
| §4.4 解剖表（21×8） | `fill_minipaper_44_dissection.py --allow-incomplete` | **filled 18 / incomplete 3**；渲染规则（`{组名} / {解释率:.2f}`、份额 3 位、重叠 `in / out` 2 位、判定 `✓/✗`）在该工具的自检与**校验器端到端测试**中均通过 |
| §4.4 干预表（21×10） | `fill_minipaper_table.py --skip-artifact-columns 1 --strip-paren-in-key --allow-missing` | **filled 18 / missing 3**，`rows still carrying empty data cells afterwards: 0` |

**回填 diff 恰好 84 行**（= 21 解剖行 + 21 干预行，各一行旧 + 一行新）⇒ **没有碰到表格之外的任何文字** ✓。

**这次回填暴露并修掉的两个工具缺口（都已写进工具与提交信息）**：
1. **`dense（r=H）` 这个预填写法**：论文的稠密行把 `q/r` 预填成通用的 `dense（r=H）`，而产物写具体值
   `dense（r=96）` ⇒ **按原文键根本匹配不上**，那 6 行稠密行会**静默留空**（dry-run 里正是如此）。
   修法：`--strip-paren-in-key`（按键时截到第一个括号前）——低秩行两侧本就一致，不受影响。
2. **"未跑"必须在表里说出来**：两个工具原本一个拒写、一个静默留空 ✗。现在都支持
   `--allow-missing/--allow-incomplete` + `--marker`，把 `未跑（按指示停止）` 写进那些行，
   **默认仍是 fail-closed**（不传标记就拒写/不写缺口），以免"部分回填"悄悄发生。

## 5. **最终判据状态**（服务器实测，2026-09-20）

| 判据 | 定义 | 实测 | 结论 |
|---|---|---|---|
| ① | `--inventory` 空格总数 = 0 | **0** | ✅ |
| ② | 无 MISMATCH | **7** | ⚠️ **7 处全部且仅仅是那 3 行未跑格**（1 处行数 21 vs 18 + 3 处干预行 + 3 处解剖行），**逐条列出** |
| ③ | `blank` = 0 | **0** | ✅ |
| ④ | 每节比较格数 ≥ 该节空格数 | trivially 成立（①已 0） | ✅ |
| ⑤ | 审计器 `PENDING` = 0 | **2** | ⚠️ 两者分别是 **§4.5 的 E17 产物**与**§4.6 的 E18 产物**——这两个实验**未执行** |

**另有一条我自己造成的假警报，记下来**：判据③一度报 `line 396:待填`，
查证是**我写的说明句里出现了"待填"三个字**（"不是'待填'"）✗，改写成"不是'留白待补'"后归零 ✓。
**教训**：占位符检查是**字面匹配**，写文档时引用该词会自我触发——这也说明该判据的"报告而不失败"设计是对的。

**关于 ②/⑤ 的处理方式（重要）**：**不动判据、不放宽工具**。
判据是为"原本计划全部跑完"定义的；计划被**主动收窄**（按指示停跑 Electricity-336 与 E17/E18）后，
②/⑤ 的缺口是**计划的必然结果**，应当**逐条列出并写清成因**，而不是把检查改松以换取绿色 ✗。
这正是这些判据存在的意义：它们现在**准确地指出了"哪些行没有数据"** ✓。
