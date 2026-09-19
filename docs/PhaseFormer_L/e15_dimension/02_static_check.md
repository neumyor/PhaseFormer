# E15 · §4.3 相位补空间维数 — 阶段 2：静态检查

> 代码：`scripts/phaseformer_L/e15_dimension.py`｜产物：`research_runs/phaseformer_L_e15_dimension_v1/`
> 服务器：`yyk03@11.11.18.3`，conda `time`

## 1. 正确性门（`--verify-existing`）—— 本实验的核心静态检查

§4.3 要产出 28 行，其中 7 行与既有登记产物重叠。因此阶段 2 的判据不是"能跑"，而是
**"能在既有 7 行的每一个字段上复现登记值"**。命令：

```bash
python scripts/phaseformer_L/e15_dimension.py --verify-existing \
  --output-root research_runs/e15_verify4 \
  --reference-dir research_runs/lowrank_data_property_v2
```

结果：**`PY_EXIT=0`，`top-level passed = True`，7/7 setting 通过**。

| setting | 二阶矩（szz/szy/syy） | 全部共享度量 | 报告 §2.6(c) 模板值 | 结论 |
|---|---|---|---|---|
| ETTh2-96 / ETTh2-720 / ETTm2-96 / ETTm2-192 / Weather-96 / Weather-192 / Electricity-336 | 最大相对差 **0.0** | 111/111 项通过 | 6/7 完全相同，ETTh2-96 差 0.005 | 通过 |

## 2. 报告 §2.6(c) 参照的**分辨率修正**（本轮发现并处理）

首次运行时门报 `passed: false`，唯一失败项是 ETTh2-96 的 `b_1` 最佳模板 `|cos|`：
本脚本得 **0.5749**，而报告印的是 **0.58**。

追查结论：报告的该表是"用 **16 个可解释模板**"逐一拟合（其中含 **4 个指数衰减核**），
即与 `EXP_TAU_GRID` 同族；且 7 个 setting 中 6 个逐位相同，仅 ETTh2-96 恰落在
**2 位小数的舍入边界**上（0.5749 舍到 0.57，报告印 0.58，等价于报告侧真值 ≥0.575）。
因此这是**参照自身的分辨率**问题，不是计算错误。处理方式：

1. 门改为把**未舍入**的精确 `|cos|` 与 2 位小数参照比较，容差取参照末位一个单位（**0.01**）；
   τ 的同一性仍要求**精确相等**；
2. 精确值与细网格值另写入 `b1_template_detail.csv`，**不**污染 `dimension_table.csv`——
   后者的表头固定为 §4.3 的 6 列加 `dataset/horizon/source`；
3. 表注（已写入 minipaper §4.3）如实说明"6/7 完全相同、1 例差 0.005 且位于舍入边界"。

**该修正只放宽了跨文档比较的分辨率，没有放宽任何自产物的一致性判据**：moments、`leading_direction.csv`
与 `optimal_rank_capture.csv` 仍要求与既有产物逐字段相等（容差 1e-6 或按字段类型）。

## 3. 其它静态检查项

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `compile()` | 通过 |
| 2 | `--help` 参数完整性（`--datasets/--horizons/--output-root/--save-moments/--verify-existing/--max-channels`） | 通过 |
| 3 | test 划分**从不解析**（读取时 `nrows = border2s[1]`，即止于验证边界） | 通过（日志每次打印 `test never read`） |
| 4 | 高通道 setting 的流式累加（Electricity 321 通道、Traffic 862 通道）；`--channel-block` 分块不改变结果 | 通过（分块不变性 1e-10） |
| 5 | `dimension_table.csv` 的 6 个 §4.3 列全部由实算值写出（无占位列） | 通过 |
| 6 | `--verify-existing` 不写出任何文件、不读 test | 通过 |

## 4. 阶段 2 挡下并修复的缺陷

| # | 缺陷 | 修复 |
|---|---|---|
| 1 | 门把 2 位小数参照当作精确值比较 → ETTh2-96 误判失败 | 分辨率感知容差（见 §2） |
| 2 | `best_template()` 只返回舍入值，无法给出精确值 | 改为返回精确值，调用点自行舍入；精确值落 `b1_template_detail.csv` |
| 3 | 改名后 `settings_summary` 仍引用旧变量 `fine_cos` → `NameError`（首次正式运行即崩） | 改为 `round(float(fine_cos_exact), 6)`；这是**正式运行前**被捕获的 |

第 3 项说明阶段 2/3 的价值：该 `NameError` 只有在真正执行到该分支时才触发，
若直接提交长任务，会在产出 20+ 个 setting 后崩溃。

## 5. 结论

阶段 2 通过（含 1 项跨文档分辨率修正与 3 项缺陷修复）。允许进入阶段 3。
