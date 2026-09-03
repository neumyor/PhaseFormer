# PhaseFormer 输入成分 H1/H3/H4 因果消融 —— D0 阶段性汇报

> 日期：2026-09-03
>
> 本文件是对《PhaseFormer_input_component_H1_H3_H4_plan.md》（下称「计划」，§编号均指该文）
> 的**阶段性汇报**，覆盖 D0（`horizon=192 × seed=2021 × 7 数据集`）已自动跑完的
> Track R → 审计 → Track F → retrained test → 汇总全链路。
>
> **D0 只有单 seed**：文中所有结论一律为 `provisional (seed2021 only)`，不得表述为跨 seed 稳健，
> 最终分级须待 D1 三 seed 全矩阵（§7.2 Stage 3b/3b-F）复查后按正常规则给出（§8.2/§8.3）。
> 自 Stage 3a-F 首次读取 test 起公式与门槛已冻结（§14），本文不修改任何冻结项；文中数字全部由
> `research_runs/result_summary_d0.csv` 长表按 §8.1 定义重算，标注明来源。

---

## 1. 执行范围与状态（v1.2）

数据范围：7 数据集 `ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Exchange/Weather`（Traffic 已按 v1.2
修订剔除，不计入训练调度与一切 test 读）。D0 = 单 setting 组：`7 dataset × horizon=192 ×
seed=2021`，每 setting 3 模型 × 10 输入条件。

| 阶段 | 规模 | 状态 |
|---|---:|---|
| D0 Track R（validation-only） | 7×3×10 = 210 runs | 完成（含 v1.1 期已完成的非 Traffic run，Weather 尾部已补齐） |
| D0 validation 审计 | — | 通过（计数口径修复见 §8-1，commit 6e39a14） |
| Track F（frozen 读，§7.1） | 21 个 `none/full` 锚点 ×10 输入 = 210 次读 | 完成；其中 21 次 `none/full` 兼作 Track R 基线 |
| retrained test | 21 锚点 × 9 项非 full 干预 = 189 个 checkpoint | 完成 |
| D0 汇总 | `result_summary_d0.csv`（长表 420 行，qc_status 全 `ok`）<br>`result_summary_d0_aggregate.csv`（宏表 60 行） | 完成（2026-09-03T02:41:20Z 编排器 stage=done） |
| D1（Stage 3b） | 2520 jobs = stage1 210（复用 D0 前缀）+ stage2 630（`h96/336/720 × s2021`）+ stage3 1680（三 horizon × `s2022/2023`） | 进行中：supervisor pid 730056，GPU2/3，`--max-stage 3`；快照 stage2 ≈216/630（`control/supervisor.json`） |

产物目录：Track R `research_runs/input_components_h134_scratch/`、Track F
`input_components_h134_frozen_d0/`、retrained `input_components_h134_retrained_test_d0/`、控制与
日志 `input_components_h134_control/`（§7.3）。

## 2. 方法与口径（引自计划 §7/§8）

- 模型：M0 `original`（phase-only PhaseFormer）、M1 `weak_residual`（M0+共享 NLinear residual+
  普通静态 gate）、M2 `rcrf_nlinear_plain`（M0+同一 residual+RCRF，无 phase calibration）。
- 输入：三个假设共享 `full`；每个假设另有 `half_A`（删 50% A / 半剂量几何干预）、`minus_A`
  （删 100% A）、`sham`（**matched disturbance**：保留与 A 相近的能量/平滑度/位移量分布，但用
  固定周期置换破坏正确时间对应——H1 置换残差周期、H3 时间反转低频轨迹、H4 置换位移序列；
  §3.2/4.2/5.2）。`sham` 不是安慰剂，它估计「一般平滑/重排/分布漂移本身」的影响（§2.3）。
- 证据链：Track R = 去掉 A 从头重训，回答「模型能否补偿、训练后 A 的边际价值」；Track F =
  冻结 full checkpoint 在干预输入上读，回答「已训模型实际依赖 A 的程度」。两者必须分别报告，
  且 frozen 效应大也可能来自输入分布漂移（§7.1）。
- 统计：`Delta(M,H,V)=L(M,H,V)/L(M,full)-1`，`Interaction=Delta(M,H,V)-Delta(M0,H,V)`（§8.1）。
  setting 宏平均对 7 个 dataset setting 等权。本文表格数值 = 长表按行级相对变化（= 变体 MSE/MAE
  相对该 setting 自身 `full` 基线的变化）再对 7 setting 宏平均。单位：Delta 为百分比，Interaction
  为百分点（pp）。
- 判定：§8.3 门槛；D0 只给 provisional 判定。

## 3. 结果总览：Delta 宏平均（MSE / MAE，%）

### 3.1 Track F —— frozen 读（full checkpoint 直接读干预输入）

| H | variant | M0 original | M1 weak_residual | M2 rcrf_nlinear_plain |
|---|---|---:|---:|---:|
| | | MSE | MAE | MSE | MAE | MSE | MAE |
| H1 | half_A | +17.3 | +8.2 | +21.5 | +10.4 | +20.7 | +9.9 |
| H1 | minus_A | +86.2 | +31.9 | +88.6 | +33.8 | +87.7 | +33.4 |
| H1 | sham | **+172.1** | +63.1 | +138.4 | +52.6 | +134.1 | +51.1 |
| H3 | half_A | +1.6 | +0.9 | +3.3 | +1.9 | +2.7 | +1.5 |
| H3 | minus_A | +11.3 | +6.5 | +12.3 | +7.1 | +11.4 | +6.5 |
| H3 | sham | +23.7 | +13.1 | +23.8 | +13.1 | +23.8 | +12.9 |
| H4 | half_A | +4.8 | +2.4 | +3.6 | +1.8 | +3.6 | +1.8 |
| H4 | minus_A | +9.9 | +4.7 | +8.5 | +4.0 | +8.5 | +4.0 |
| H4 | sham | +10.2 | +5.3 | +9.3 | +4.9 | +9.5 | +4.9 |

### 3.2 Track R —— retrain 读（去掉 A 重训后的 checkpoint）

| H | variant | M0 original | M1 weak_residual | M2 rcrf_nlinear_plain |
|---|---|---:|---:|---:|
| | | MSE | MAE | MSE | MAE | MSE | MAE |
| H1 | half_A | +1.1 | +0.9 | +0.6 | +0.4 | +0.3 | +0.1 |
| H1 | minus_A | +83.7 | +30.9 | +86.4 | +33.4 | +85.9 | +32.7 |
| H1 | sham | +67.6 | +29.8 | +67.7 | +31.0 | +66.4 | +30.3 |
| H3 | half_A | −2.5 | −1.1 | −0.0 | −0.1 | −0.4 | −0.2 |
| H3 | minus_A | +2.1 | +1.3 | +1.9 | +1.0 | +2.9 | +1.4 |
| H3 | sham | +3.6 | +3.1 | +6.7 | +4.2 | +4.1 | +3.0 |
| H4 | half_A | +2.6 | +1.8 | +1.8 | +1.2 | +2.3 | +1.4 |
| H4 | minus_A | +5.0 | +3.3 | +3.8 | +2.3 | +4.0 | +2.3 |
| H4 | sham | +5.1 | +3.4 | +4.5 | +2.9 | +4.7 | +2.7 |

**粗读**：frozen 侧三类输入扰动都造成大幅退化（M0 的 H1 sham 高达 +172%），且几乎处处
`sham ≥ minus_A`；retrain 侧 H1 的 `minus_A` 仍高达 ~84–86%，H3/H4 的 `minus_A` 大幅收缩到
~2–5%，但 `sham` 仍与之相当或更大。两个证据链方向一致指向：**模型对输入时序/分布扰动的敏感性
（robustness）主导了效应，成分专属的依赖信号很弱**（§7.1 对 frozen 效应的告诫 + §8.3 的
`sham` 控件）。

### 3.3 Interaction vs M0（百分点，长表配对重算）

`Interaction = Delta(M) − Delta(M0)`，逐 setting 配对再宏平均。正值 = 增强模型比 M0 更依赖该成分。

| track | H | variant | M1−M0 MSE | M1−M0 MAE | M2−M0 MSE | M2−M0 MAE |
|---|---|---|---:|---:|---:|---:|
| frozen | H1 | minus_A | +2.4 | +1.9 | +1.5 | +1.5 |
| frozen | H1 | sham | −33.7 | −10.6 | −38.0 | −12.1 |
| frozen | H3 | minus_A | +1.0 | +0.6 | +0.1 | −0.0 |
| frozen | H3 | sham | +0.2 | +0.0 | +0.1 | −0.2 |
| frozen | H4 | minus_A | −1.3 | −0.7 | −1.3 | −0.7 |
| frozen | H4 | sham | −0.9 | −0.5 | −0.7 | −0.4 |
| retrain | H1 | minus_A | +2.7 | +2.4 | +2.2 | +1.7 |
| retrain | H1 | sham | +0.1 | +1.2 | −1.2 | +0.5 |
| retrain | H3 | minus_A | −0.2 | −0.3 | +0.8 | +0.1 |
| retrain | H3 | sham | +3.1 | +1.1 | +0.5 | −0.1 |
| retrain | H4 | minus_A | −1.2 | −1.0 | −1.1 | −1.0 |
| retrain | H4 | sham | −0.6 | −0.5 | −0.4 | −0.7 |

要点：增强模型相对 M0 的 Interaction **几乎处处很小**；唯一稳定的正号只在 H1 `minus_A`
（MSE ~+2 pp，尺度 ≪ minus 主效应 ~85 pp）；H4 处处为负（增强模型对相位漂移演化的依赖反而
略低于 M0）；frozen 侧 H1 `sham` 的 Interaction 大幅为负（−34/−38 pp，增强模型对「打乱周期」的
脆弱性显著低于 M0，但这是 robustness 差异，不是成分依赖差异）。

### 3.4 `minus_A` 宏 Delta 与宏 CI（%，用于 §8.3 门槛）

CI = summarizer 的宏 block-bootstrap（settings 分层，§8.2）。M0 等效门槛要求 MSE 与 MAE 的 CI
完全落在 ±0.5% 内——下表显示**没有任何单元格接近该门槛**。

| track | H | M0 ΔMSE [lo, hi] | M0 ΔMAE [lo, hi] | M1 ΔMSE [lo, hi] | M1 ΔMAE [lo, hi] | M2 ΔMSE [lo, hi] | M2 ΔMAE [lo, hi] |
|---|---|---|---|---|---|---|---|
| frozen | H1 | 86.2 [28.8, 151.5] | 31.9 [10.7, 55.0] | 88.6 [36.6, 157.0] | 33.8 [15.7, 55.0] | 87.7 [35.7, 152.1] | 33.4 [15.7, 53.1] |
| frozen | H3 | 11.3 [4.7, 18.6] | 6.5 [2.4, 11.4] | 12.3 [6.8, 18.6] | 7.1 [3.6, 11.4] | 11.4 [5.5, 17.6] | 6.5 [2.7, 10.8] |
| frozen | H4 | 9.9 [2.3, 21.2] | 4.7 [1.3, 9.1] | 8.5 [1.9, 17.0] | 4.0 [1.2, 7.8] | 8.5 [2.0, 16.5] | 4.0 [1.2, 7.5] |
| retrain | H1 | 83.7 [29.5, 151.2] | 30.9 [10.7, 54.3] | 86.4 [39.1, 147.8] | 33.4 [16.0, 52.9] | 85.9 [38.3, 146.2] | 32.7 [15.5, 51.7] |
| retrain | H3 | 2.1 [−0.0, 3.9] | 1.3 [−0.7, 3.2] | 1.9 [1.0, 2.7] | 1.0 [0.5, 1.6] | 2.9 [1.5, 4.2] | 1.4 [0.6, 2.1] |
| retrain | H4 | 5.0 [1.9, 8.3] | 3.3 [1.4, 5.3] | 3.8 [1.2, 6.7] | 2.3 [0.7, 4.1] | 4.0 [0.6, 7.3] | 2.3 [0.5, 4.1] |

## 4. §8.3 判定要点（provisional）

- **M0 等效不敏感：对 H1/H3/H4 全部不成立。** `minus_A` 的宏 Delta 与 CI 远在 ±0.5% 外
  （retrain H3 最小也有 ~2%，H1 达 ~84%）。即「PhaseFormer 对这些成分不敏感」的前件在 D0 无法
  建立——模型确实对这些输入的变化有强反应，但需要判定这是否为「专属成分依赖」还是「一般扰动
  敏感」。
- **增强模型实质依赖四门槛（M1/M2）**，D0 逐项：
  - `minus_A ≥1% 且 CI 下界>0`：retrain 侧三假设 MSE 均满足（H1 ~85–87%、H3 1.9–2.9%、H4
    3.8–4.0%）；frozen 侧均满足。
  - `half_A` 剂量：H3/H4 retrain 的 `half_A` 为 −0.4..+2.6%（方向一致、介于 full 与 minus 间），
    H1 retrain `half_A` ≈ +0.3–1.1% 而 `minus_A` ≈ 85%（响应高度非线性地集中在完全删除）；
    剂量门槛形式通过但判别力弱。
  - `Interaction ≥ +0.5% 且 CI 下界>0`：仅 H1 `minus_A`（MSE ~+2 pp）与 retrain H3-M2（+0.8 pp）
    为正，H4 为负。**无稳定正 Interaction**（且 Interaction 宏 CI 待 §8-2 修复后才有，见 §8）。
  - `sham` 不解释：**在 D0 普遍失败**——frozen 侧三假设 `sham ≥ minus_A`（H1：172 vs 86；H3：
    24 vs 11；H4：10 vs 9.9）；retrain 侧 H3 `sham > minus_A`（M1：6.7 vs 1.9）、H4 `sham ≈
    minus_A`（5.1 vs 5.0）、H1 的 MSE 上 sham（~67%）低于 minus（~85%）但 MAE 上 sham≈minus
    （~30 vs 31–33）。
- **结论分级（D0 provisional）：三个假设均达不到 Strong/Partial。** 证据形态属 §8.3 的
  OOD/confounded（`sham` 与 `minus_A` 同等或更差，扰动由一般分布漂移主导）与近 null/Model-shared
  混合；retrain 侧 H3/H4 有真实但微弱的删除效应、却完全被 matched disturbance 复现。**不得据此
  D0 宣告 Rejected**——单 seed 且冻结读数受 robustness 污染，须 D1 三 seed 按同一冻结协议复查。

## 5. 逐假设解读（provisional）

### H1 跨周期同相位残差

- frozen：三模型对任意 H1 干预都极敏感；M0 `minus_A` +86%、`sham` +172%（sham 反而大近一倍）。
  M1/M2 的 `minus_A` ≈ M0（+88/+88），但 `sham` 明显低于 M0（+138/+134 vs +172）——增强模型在
  frozen 读下对「打乱周期次序」更稳健，这属于 robustness/输入分布差异，不是对残差内容的依赖。
- retrain：去掉 H1 重训后 `minus_A` 仍 ~84–86%（三模型几乎相同，MSE CI 下界 ~30%），说明模型
  无法通过重训补偿 H1 缺失；但 `sham`（~67%）同样巨大 → 该信号主体被 matched disturbance 混淆。
- 名义 Interaction（`minus_A`，retrain MSE +2.2..+2.7 pp；M1 在 7 family 中 5/7 为正，M2 仅
  3/7）尺度 ≪ 主效应，不构成「PhaseFormer 特别未利用 H1」的稳健证据。
- 家族异质性大：dataset 级 `minus_A`（M0, retrain）从 Exchange −19% 到 ETTm1 +261%。**Exchange
  反号**：M0 删掉残差后错误反而下降（−19/−23%，M1/M2 近 0）——该家族上 phase-only 的 M0 可能
  在残差模板里拟合了噪声。

### H3 近期水平漂移（EMA）

- frozen `minus_A` 11–12%（M0≈M1≈M2），`sham` ~24% > minus → 弱剂量、扰动主导。
- retrain `minus_A` 收缩到 1.9–2.9%（MSE；M1/M2 CI 下界>0，M0 的 CI 跨 0），`half_A`≈0，而
  `sham`（M0 3.6 / M1 6.7 / M2 4.1）均 ≥ `minus_A` → 可解释为一般水平/低频漂移扰动，非专属依赖。
- Interaction：retrain M1 −0.2 pp、M2 +0.8 pp；frozen 侧 ~0。无稳定正 Interaction。

### H4 相位漂移演化

- frozen `minus_A`：M0 +9.9%，M1/M2 +8.5%；`sham`≈`minus_A`。
- retrain `minus_A`：M0 +5.0%，M1 +3.8%，M2 +4.0%（CI>0），`half_A` +1.8..+2.6%；`sham`≈`minus_A`。
- Interaction 在 `minus_A` 处**为负**（−1.0..−1.3 pp）：增强模型对相位漂移演化的依赖并不高于 M0。
- 解读：D0 上 H4 主要表现为对「任意相位/时序扰动」的敏感性（robustness 议题），而非对 shift 演化
  信息的专属依赖；无增强依赖证据。

## 6. 数据质量与产物核查

- `result_summary_d0.csv` 420 行 = Track F 210（21 锚点 ×10 输入）+ Track R/retrain 210（189 个
  实际重训干预 checkpoint + 21 个与 Track F 共用的 `none/full` 基线行）；`qc_status=ok` 420/420。
- retrained checkpoint 文件 189/189、frozen 锚点 21/21、审计计数修复后与 runner main() 口径一致
  （commit 6e39a14）。
- 提取 QC 相关列（`input_change_rms`、`input_endpoint_max_abs`、`nyquist_energy_fraction`、
  `rcrf_alpha_mean`/`rcrf_reliability_mean`、RCRF `cf_*` 反事实重组列等）已落入长表，可衔接计划
  §13.1/§13.6 审计；本汇报未逐 setting 展开 QC 表。

## 7. 独立观察（非预注册判定，供 D1 参考）

- **全模型对输入扰动的敏感性**：frozen 读下三模型对三假设的干预输入都极脆弱（尤其 H1），并且
  `sham` 普遍 ≥ `minus_A`。这更像输入分布/时序结构稳健性问题，不是本计划判定的「成分专属依赖」。
  若 D1 复现，值得另立 robustness 结论，不与「是否 underuse A」混淆。
- **Exchange 在 H1 反号**、ETTm1/Electricity 驱动 H1 大效应：宏平均被少数家族主导，D1 三 seed
  汇总时须按 §8.2 报告各 dataset 家族而不能只报总体宏平均。

## 8. 已知问题 / 数据工程债务

1. **aggregate 的 interaction 列口径与 §8.1 不符（需修）**：该列不是「逐 setting 配对差再宏平均」。
   示例：frozen H1 `minus_A` 的 M1−M0 宏 Interaction ≈ **+2.4 pp**（长表重算），aggregate 该列却
   记 **+36.1 pp**；frozen H1 `sham` M1−M0 ≈ **−33.7 pp**，aggregate 亦异常。→ 本汇报所有
   Interaction 一律从长表按 §8.1 配对重算，**无宏 CI**；须在 D1 汇总前修复
   `summarize_input_component_ablation.py` 的 interaction 聚合，才能对「Interaction ≥ +0.5% 且
   CI>0」作正式判定。
2. **计划文档 §7.2/§7.3/§13.0 正文仍保留 v1.1 的 8 数据集计数**（24 锚点 / 216 retrained / 456
   单元，§13.0 表头亦如此），与 v1.2 实际（7 数据集 / 21 锚点 / 189 retrained / 399 唯一单元）
   不一致。本次未静默改写计划，仅在此标记，建议以独立维护性小提交统一到 v1.2。
3. retrain 轨道中的 21 个 `none/full` 基线行与 Track F 复用同一批读取（§7.1）；叙述与 §13.3 表格
   口径需要一致注明，避免把 420 误读为「420 次独立 test 读」。
4. `result_summary_d0.csv` 的 `selection_source` 列约 55% 行为空；若 D1 判定需要 selection
   provenance 请补查其语义后再用。
5. 本文件 CI 为 summarizer 宏 block-bootstrap；未对逐 setting 重跑。D0 provisional 阶段可接受。

## 9. D1 与下一步

- D1 supervisor pid 730056（GPU2/3，`--max-stage 3`）已运行：stage2（`h96/336/720 × s2021`，630
  jobs）快照 ≈216/630，stage3（`s2022/2023`，1680 jobs）随后。进度以
  `research_runs/input_components_h134_control/supervisor.json` 为准刷新。
- 待办顺序建议：① 修复 §8-1 的 interaction 聚合列（先于 D1 汇总，避免再次产生误导数字）；
  ② D1 全矩阵 Track R 完成后按冻结协议走审计 → Track F → retrained test → 全矩阵汇总，给出
  三 seed 最终分级并回填计划 §13.7；③ 计划正文 v1.1→v1.2 计数维护；④ 把本 D0 provisional 结论
  与判定表并入计划 §13.0 对应行。

## 10. D0 provisional 决策表（计划 §13.0 模板）

| H | M0 equiv (D0) | M1 dep (D0) | M2 dep (D0) | Interaction (D0) | families covered in D0 | provisional grade |
|---|---|---|---|---|---|---|
| H1 | No（retrain minus ~84%, CI 远出 ±0.5%） | 名义有（retrain minus ~86% MSE, CI>0），但 MAE 上 sham≈minus、frozen sham>minus | 同 M1（~86%） | 弱正（`minus_A` MSE +2.2..+2.7 pp, MAE +1.7..+2.4 pp；frozen `sham` 负 −34/−38 pp）；宏 CI 待修 | 7/7（Exchange 反号 −19%） | 无法分级（OOD/confounded 为主，Model-shared 倾向）；不作 underuse 证据 |
| H3 | No（retrain ~2.1%, frozen ~11.3%） | 边缘（retrain minus MSE ~1.9% CI>0），sham≥minus（6.7 vs 1.9） | 边缘（~2.9%），sham≥minus（4.1 vs 2.9） | 无稳定正（M1 −0.2 pp, M2 +0.8 pp retrain） | 7/7 | 无法分级（近 null + confound） |
| H4 | No（retrain ~5.0%, frozen ~9.9%） | 名义 minus（~3.8% CI>0），sham≈minus | 名义 minus（~4.0% CI>0），sham≈minus | 负（−1.2/−1.1 pp `minus_A`） | 7/7 | 无法分级（confound；增强依赖证据为负） |

> 定位声明：D0 单 seed、单 horizon（h192）的不通过并不等于跨 seed 的 Rejected，反过来说，D0
> 某些数值的 CI>0 也不构成 D1 稳健性；三 seed 复查前不写 §13.7 最终结论。
