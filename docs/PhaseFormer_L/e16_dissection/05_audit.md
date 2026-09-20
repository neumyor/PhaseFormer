# E16 §4.4：阶段 5 **审校**（2026-09-20）

本文件记录**对产物的审校**：不变量判定、覆盖范围、与既有参照的对账，以及一次**工具缺陷诊断**的全过程。
执行背景见 `04_run.md`，计划见 `01_plan.md`。

---

## 1. 一句话

**7 个分片（21 格，全部低秩臂）被 `algebra` 不变量拒绝，拒绝本身是对的**——
闭式"未触碰臂"确实与模型自身的 fused 输出有**真实的**小偏差（RMS 9.3e-5）；
**但工具打印出来的诊断数据是错的**（恒为 0），而那个错误的数字**一度把我引向错误的结论**。
本文件把这个过程、证据与两条缺陷都记下来。

## 2. 缺陷 **D2**（报告层，会给出**虚假的安心**）：逐元素统计读的是**过期快照**

`run_cell` 的调用顺序（`e16_dissection.py`）：

```text
1689:  batches_first  = iterate("statistics")     # 第一遍：统计量（同时累计 recorded_fused_sq / elements）
1690:  statistics = accumulator.statistics()      # ← 快照在这里取
1702:  batches_second = iterate("arms")            # 第二遍：臂与干预（逐元素 algebra_* 只在这里累计）
1837:  model_fused_mse = float(statistics["recorded_fused_mse"])   # 读快照（此处尚可）
1886:  ... f"RMS {statistics['algebra_fused_rmse']:.3e}, max ..."  # 读快照 → 恒为 0
```

`algebra_sq` / `algebra_samples` / `algebra_absmax` **只在 `add_arm_block` 里累加**
（`e16_dissection.py:1254-1257`），即**第二遍**；而 `statistics` 快照在**第一遍之后**就取走了
⇒ 这三个字段在快照里**永远是 0** ⇒ 错误信息里的
「element-wise RMS **0.000e+00**, max **0.000e+00**」**不是测量结果，而是过期快照**。

**影响面不止错误信息**：同一快照被写进产物——
`untouched_arm_fused_rmse_vs_model` 与 `untouched_arm_fused_max_abs_vs_model`
（`:1941-1942`、`:2021-2022`）⇒ **每个 cell 的这两列恒为 0**。
任何依据它们判断"闭式是否复现模型"的读者都会得到"完全一致"的**错误结论**。

**实测证据（单格插桩）**：对失败格 `ETTh2-96 l_q1_4 s2022` 用包装后的累加器跑一遍：

```text
calls: statistics=11  arm=11  band=11
statistics: sum elements=1871520   sum recorded_fused_sq=3.853653e+05
arm       : sum algebra_samples=1871520   sum algebra_sq=1.623411e+01
⇒ 真实逐元素 RMS = sqrt(16.23411 / 1871520) = 9.31e-05      ← 不是 0
⇒ 工具信息里却打印 RMS 0.000e+00
```

## 3. 缺陷 **D1**（科学层）：闭式"未触碰臂"与模型确实有**真实**小偏差

**同一格的实测数值**（fused MSE，元素总数 1,871,520）：

| 量 | fused MSE | 与模型的相对差 |
|---|---:|---:|
| **模型自身**（`recorded_fused_mse`） | **2.0591031615e-01** | — |
| 闭式未触碰臂 = `Original`（**加上** mapped encoder bias） | 2.0615584266e-01 | **+1.192e-03** ← 越界（容差 1e-3） |
| `Bias-off`（**不加**该 bias 项） | 2.056227…e-01 | −1.475e-03 |

**读数**：模型**落在两种约定之间**（`Bias-off` < 模型 < `Original`），
且模型距 `Bias-off` 的距离约为两种约定之差의 **55%** ⇒ 说明**该 bias 项在模型里只"部分"出现**，
而闭式代数按注册约定把它整项加上/去掉 ⇒ **两种约定都不等于模型**，
残差 RMS ≈ **9.3e-5**（约为融合误差尺度的 0.1%）。

**与工具自身注释一致**：`:1586-1590` 写着
"The registered arm algebra adds the mapped encoder bias to a hidden state that **already contains**
the encoder bias, so the closed-form untouched arm is **offset from the model's own output by this term**.
It is recorded instead of silently corrected, because the probe intervention table is already filled with
the registered convention; **the term is exactly zero for the dense head**."

**结构性后果（与观测完全一致）**：稠密臂 `encoder = I`、无 encoder bias ⇒ 该项**恰为 0** ⇒
**6 个稠密分片全部通过** ✓；低秩臂该项非零 ⇒ 视其"相对融合误差尺度"的大小而越界 ✗
——这也解释了为什么**用 bias 的绝对量做预测会失败**（我实测过：通过格的最大 bias 比失败格更大，不可分离 ✓）。

## 4. 我在诊断中被误导，以及如何纠正（流程留档）

- **我先前写下的判断（错）**：因为错误信息里逐元素 RMS/max 都是 0，我推断
  "元素完全一致却均值不同 ⇒ 差异只可能来自元素集合/计数不同 ⇒ 是**聚合口径**问题，不是代数算错"，
  并把这条写进了 `04_run.md` §8.2 与 agent-log。
- **真实情况**：那个 0 是 **D2 的过期快照**，逐元素 RMS 实际是 **9.3e-5** ⇒ **偏差是真实的**，
  "聚合口径"的推断**不成立**。
- **教训（与项目里其它几条同族）**：**诊断数据本身也要被验证**。我把"工具打印的一个统计量"
  当成了"一个测量结果"，于是基于一个**从未被更新的字段**建立了推理链 ✗。
  正确做法是：**任何用来支撑结论的字段，先确认它在代码里何时被写、何时被读**——
  这与 §4.4 校验器当初读错四个列名、以及"`^|` 看不见缩进表"是同一类错误。
- 纠正方式：`04_run.md` §8.2 保留原文并**就地加勘误指针**（不改写历史，但让读者不会读到错结论）；
  本文件为最终结论。

## 5. 覆盖范围（本轮能报告什么）

21 行 = 3 模型 × 7 setting；**有效 11 行（33 格）**：

| 模型 | 有数据 | 缺 |
|---|---|---|
| PhaseFormer-L | ETTh2-96/720、ETTm2-96/192、Weather-96/192 = **6/7** | Electricity-336 |
| L-q1/4 | Weather-96/192 = **2/7** | ETTh2-96/720、ETTm2-96/192、Electricity-336 |
| L-q1/8 | ETTm2-96、Weather-96/192 = **3/7** | ETTh2-96/720、ETTm2-192、Electricity-336 |

⇒ 可判定：**Q1 机制**与 **Q2 因果必要性**在**稠密臂 6/7 setting** 上成立；
**Q3 语义特异性**（限制第 4 条）**只在稠密臂 6 setting + Weather 两行的范围内**可判。
不可判定：Electricity-336 全部；低秩臂的跨 setting 陈述。

## 6. 与既有参照的对账（**未达标，须披露**）

- 判据：`reference_parity.passed` 应为 `true`（`01_plan.md:272`；`02_03_static_check_smoke.md:133`
  要求全量运行在全部 8 个字段上为真），容差 1e-6。
- 实测：**`passed = false`**；逐字段最大绝对差 **2.33e-04 ~ 2.27e-03**（系统性偏正），
  且每 cell 只有 **1/3 seed** 可做值比较（另两个 seed 因复用 tie-break 解析到不同 run 目录，
  `checkpoint_path_mismatches = 2`，工具**如实列出而非隐藏**）。
- 量级参考：受影响的 `correction_energy_share` 是**逐模式**份额，其 8 个模式之和在两个实现里都≈1
  （E16 1.00023 / 参照 0.99991）；差 ~3e-4 相对，**在 3 位小数显示下只有 1 个模式跨界**
  （0.725592 → `0.726` vs 0.725359 → `0.725`）。
- **结论**：E16 自身自洽，但**不能声称与既有低秩分析逐字段等价**；§4.4 的正文若引用该参照，
  必须按"定性一致、定量有 ~3e-4 相对差"披露，且**不得**把该参照（test-exposed）当作 §4.4 的主证据。

## 7. 结论与建议的处置（待决策）

1. **D2 必修**（无误）：把 `statistics` 快照改为在**两遍都跑完之后**取（或校验时重新读取这三个字段）。
   否则信息与产物里的 `untouched_arm_fused_*_vs_model` 两列**恒为 0**，是又一个"假通过/假安心"通道。
2. **D1 需要口径决策**（研究判断，不宜我单方面定）：
   - **(a) 修约定**：在闭式里**减去被重复计入的 mapped encoder bias**，使未触碰臂与模型一致，
     并把该偏差**逐格披露**；代价是与既有低秩表的数值不再同口径。
   - **(b) 保约定、改判据**：保留注册约定，把"不变量"从"逐值相等"改成
     "**扣除已登记的 bias 项后**残差 ≪ 容差"，并逐格报告该项大小与扣除后残差。
   - **(c) 折中**：两种约定都算，正文报告 (b) 的残差，附表给出两约定之差（即 bias 项）。
3. **覆盖缺口**：修好 D2 后**先验证 D1 的残差量级**，再决定那 7 片是否重跑
   （它们的拒绝源于 D1 而非计算错误，故**很可能不需要重算实验数据**，只需按新口径报告）；
   Electricity-336 的 3 片属**未跑**，与 D1 无关。
