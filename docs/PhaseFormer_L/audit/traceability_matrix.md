# PhaseFormer-L §4 需求 → 回填工具 → 审计 对照表

> 用途：确认 minipaper §4 的**每一项**要求都有（a）一个产出它的工具、（b）一个验证它的审计，
> 从而不留下"表列存在但会静默空缺"的口子。本文只做映射核对，不产生数值。
> 建立日期 2026-09-20；六个回填工具与全部审计脚本均已就绪。

## 1. 逐项对照

| minipaper 要求 | 产出工具 | 产物 | 验证它的审计 | 状态 |
|---|---|---|---|---|
| **§4.0** 协议与四条门槛 | — | `docs/PhaseFormer_L_minipaper.md` §4.0（已冻结） | `claims.json` 的 A/B/C/D 逐条判定 | 门槛**已冻结**，判定待结果 |
| **§4.1** 先导证据 | — | 既有登记文档 | 不重算、不改口径（minipaper 头部声明） | 既有 |
| **§4.2** 主表 24+4 setting | `e14_read_test.py` → `e14_writeback.py` | `main_table.csv/.md` | 逐格不变量（8 条，53 格已过）、臂指纹（0 歧义）、复用审计（81 格）、主张 A–D | 运行中 |
| **§4.2** 变体行 6 个臂 | 同上 + `e14_main_matrix.py` | `variant_table.csv` | 逐臂实测超参（含三种 gate 先验） | 运行中 |
| **§4.2** 参数量列 + 修正器单列 | `e14_params.py` | `parameter_table.csv` | 与 `metrics.csv:parameter_count` **逐格交叉校验**（114 格 0 不符）、同 horizon 内 3 seed 恒定 | 工具就绪 |
| **§4.2** 必答 (a) ETTh2 vs FITS | `e14_writeback.py` | `claims.json.must_answer_a` | FITS 为**外部引用**、只比 MSE（源表无 MAE）、不进入主张 A–D | 已实现并冒烟 |
| **§4.2** 必答 (b) 逐 dataset 门值 + `s` 诊断 | `e14_writeback.py` | `claims.json.must_answer_b` | gate 值的**双实现交叉验证**（7/7 同 checkpoint，最差 1.98e-08）；`etth1_ettm1_all_marked_s0` 显式判定 | 已实现并冒烟 |
| **§4.2** 必答 (c) q=1/8 vs direct ≤0.5% | `e14_writeback.py` | `claims.json.D` | 逐 setting 配对 | 已实现 |
| **§4.2** 来源/披露列 | `e14_writeback.py` | `main_table.csv:provenance_note` | per-arm 粒度：复用来源逐个列出 +「属 test-selected 集合」只在真正复用时出现 | 已实现并冒烟 |
| **§4.3** 28 行维数表 + 三类图 | `e15_dimension.py` | `dimension_table.csv`、`figures/`×3 | `--verify-existing` 门（7/7、111/111）、行数/网格/空值/来源 11 项审校、**图的成图性**（灰度 std 38.6–50.7） | **已回填** |
| **§4.4** 解剖表（21 行） | `e16_dissection.py` → `e16_writeback.py` | `dissection_table_44.csv` | 不变量 1（代数证书，硬失败）、臂覆盖（(臂,setting) 与 (臂,setting,seed) 双粒度）、schema 对拍 0 缺列、空列扫描 0 问题 | 工具就绪，待运行 |
| **§4.4** 干预表（21×11 臂，含随机 RRR） | 同上 | `intervention_table_44.csv` | 不变量 2（vs run 指标）、`RandomRRR-drop` 零分布带、同维对照显式标注、`criterion_6_drop_beyond_random_rrr_95pct` 进入判定 | 工具就绪，待运行 |
| **§4.4** 参照复现 | — | `reference_parity.json` | 6 个可比 setting 上 8 字段；**Electricity-336 无参照**（E10 排除），披露为"新算非复现" | 判据已定 |
| **§4.5** 四臂表（7 行） | `e17_conditional_projectors.py` → `e17_conditional.py` → `read_test_generic.py` → `e17_writeback.py` | `conditional_table.csv` | 投影器复现门（独立路线对 E8 的 6 个投影器 `abs_cos=1.0`）、H1 来源为 Stage-3 CSV、`direct==joint` 自查、|cos| 判定可区分性 | 工具就绪 |
| **§4.5** H1 列（seed 数） | `e17_conditional.py` | `h1_*` 四列 | 用文件自带 `seed` 列分组（不依赖位置约定）；Electricity-336 记 `evidence_missing` 不推断 | 已实测 4/6 为 3/3、Weather-192 为 0/3 |
| **§4.6** 行 1 平滑 2 档 | `e18_negative.py` → `read_test_generic.py` → `e18_writeback.py` | `negative_table.csv` 行 1 | 两档**数值上确实不同**（卡点已固化）；逐 cell 判"双指标是否同时改善" | 工具就绪 |
| **§4.6** 行 3 SVD 截断 28 setting | `e18_svd_truncation.py` | `svd_truncation_table_28.csv` | 范围 **28**（含 Traffic，曾被漏为 24）；**validation 口径**与既有 E11 的 test 口径差异显式披露 | 工具就绪 |
| **§4.6** 行 5 边界消融 rank∈{1,2} | `e18_negative.py` → `e18_writeback.py` | `negative_table.csv` 行 5 | 绝对秩（非 `H/4`、`H/8`）；"不得读作秩-2 必要性"写入输出 | 工具就绪 |
| **§4.6** 行 2/4 保持 "—" | `e18_writeback.py` | 同上 | `KEPT_AS_DASH` 常量，防止被顺手补上 | 已实现 |
| **§4.7** 28×3 统计量 | `e19_predictive_stats.py` | `level_statistics.csv` | 14/14 单测、9/9 审校、`reads_test:false`、τ̂ 上界 ≤720 步 | **已回填** |
| **§4.7** 两列 ρ | `e19_predictive_power.py` | `predictive_power.csv/.json` | 6 个 ρ 的 n 与 p 值；诊断准确率与逐格判错清单 | 工具就绪 |
| **§3.4.3** `s` 诊断（不进入模型） | `e14_writeback.py` + `e19_predictive_stats.py` | 主表 `诊断 s` 列 + `must_answer_b` | `ν*` 由训练集统计量按规则式定义冻结；判错逐格列出 | 已冻结并回填文档 |
| **§5** 八条限制与披露 | 各回填工具 | 各自 `*_summary.json` 的 `disclosures` | 每份披露清单可逐条核对（E14 3 条、E16 4 条、E17 5 条、E18 4 条） | 已实现 |

## 2. 全局性的审计（跨实验）

| 审计 | 覆盖 | 脚本 |
|---|---|---|
| 空列扫描（整列空 / 填充率低 / 常量列） | **全部 6 个回填产物** | `check_builder_outputs.py`（已接入流水线每步之后） |
| 复用歧义审计（45/81 多候选、0 分歧） | E14 的 81 个复用格 | `e14_reuse_audit.py`（已接入流水线步骤 3） |
| 冻结超参合法性（`gate_init ∈ (0,1)`、`lr ≤ 1e-2`） | 全部复用解析 | `e14_main_matrix._protocol_ok` |
| 单次 test 读取纪律 | E14（自身阶段 B）、E17/E18（`read_test_generic`） | 三个读取器的 `reads_test` 标志与 per-cell marker |
| 代码同步安全（不动 `src/` 与训练入口） | E14 运行期间的全部同步 | 每次同步前 `git diff --name-status` 核对 |

## 3. 尚未闭合的项（只有执行，没有设计缺口）

| 项 | 等待什么 |
|---|---|
| §4.2 / §4.4 / §4.5 / §4.6 / §4.7-ρ 的数值 | E14 阶段 A 完成 → 阶段二六步 |
| §4.4 的 Electricity-336 无登记对照 | 已定性披露为"新算非复现"，不补造参照 |
| §4.4 的 `cross_seed_alignment.csv` | 3 seed 全量运行后产出（单 seed 冒烟下无配对，属预期） |
| A1 臂的 gate 先验核对 | `a1` 训练开始后补一条与 §4.2 同类的逐臂检查（其 `gold_combo_reliability_s2` 预期为 preset 的 0.5） |
| §4.1 的 FITS 引用数字 | 已登记（外部引用，2026-09-19 抓取） |
| minipaper 摘要的 `[主结果待填]` | §4.2 回填后一并填写 |

## 4. 结论

minipaper §4 的**每一项**要求都映射到了具体的产出工具与验证审计；**没有设计缺口**。
剩余的全部是执行（E14 的计算时间与随后的阶段二六步），以及执行后按本表逐项回填。
EOF
