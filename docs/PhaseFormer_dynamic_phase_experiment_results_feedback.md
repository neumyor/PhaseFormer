# PhaseFormer Weak-Residual 分支动态相位实验 —— 结果反馈

> 本文件按实验计划 `PhaseFormer_dynamic_phase_experiment_plan.md` 的章节结构（一~十三），逐节给出**执行情况、结果数据与反馈**。
> 所有数值均与运行目录 `metrics.csv` 逐行核对（full-budget 为 `research_runs/dyn_phase_full/dynphase_{ds}_{h}_{mode}_*`；筛选为 `research_runs/dyn_phase_screen/`；审计为 `research_runs/dyn_phase_audit_*`）。
>
> - 分支：`weak-residual-phaseformer`
> - 实验日期：2026-08-22 ~ 2026-08-23
> - 协议：Stage A 筛选（30% 数据、8 epoch、验证集）；full-budget 确认（100% 数据、≤30 epoch、测试集，全部数值以 full-budget 为准）
> - 设置：lookback 720，period 24，huber loss，lr 0.001 base，seed 2021（ETTh1 h720 为 2026），单 seed
> - 选择披露：全部 run 使用固定每数据集基础超参，未用测试集做超参搜索；结论为单 seed 配对比较

---

## 一、当前 PhaseFormer 架构分析

**计划要求**：梳理 `PhaseFormer.py` 中 phase representation → phase interaction → residual reconstruction 流程，作为后续增强的接入点。

**执行情况**：确认当前架构主流程为

```
Input → Phase Alignment → Phase tokens → Cross Phase Routing → Phase Prediction → Residual Reconstruction → Forecast
```

核心模块：`PhaseEmbedding`、`CrossPhaseRoutingLayer`、`PhasePredictor`、`WeakPeriodResidualHead`、`PhaseLocalTrendHead`、`PhaseNoiseHighFreqDamping`。

**反馈**：本分支在此架构基础上新增动态相位模块（全部 flag 可关闭、warm-start identity、flag-off 与 baseline 等价）：

| 模块 | 文件 | 对应计划阶段 |
|---|---|---|
| `PhaseCorrection`（Δφ 偏移 + phase_warp） | `src/models/phase_correction.py` | 阶段2 |
| `CircularPhaseEmbedding`（Fourier 周期位置嵌入） | `src/models/phase_geometry.py` | 阶段3 |
| `PhaseRotation`（2D θ 旋转） | `src/models/phase_rotation.py` | 阶段4 |
| `HarmonicModulation`（z′=γz+β） | `src/models/harmonic_modulation.py` | 阶段5 |

接入位置均在 `PhaseFormer.py` 的 phase 路径中，`phaseformer_presets.py` 的 `get_ablation_overrides` 负责模式接线。

---

## 二、研究假设

| 假设 | 计划内容 | 实测反馈（full-budget） |
|---|---|---|
| **Hypothesis 1**：动态相位变化建模能提升长期预测 | 为 phase token 增加 offset（相位提前/延迟/速度变化） | **部分成立，数据依赖**。A（Phase Correction）仅在 ETTh2 上稳定改善（h336 −1.5%、h720 −1.9%），其余 setting 落在 ±2% 噪声区间（见 §五）；但其 peak 定位校正在 Electricity 上明确（9–10/10 峰值更近，见 §五·Peak-shift） |
| **Hypothesis 2**：周期几何结构能改善 phase token 表达 | 以循环拓扑（P→1 连续）替换线性排列 | **微弱成立**。B（A + Circular）在多数 setting 优于 A（ETTh2 h720 −3.0%、ETTh1 h336 −2.0%），但幅度有限（见 §六） |
| **Hypothesis 3**：多周期特征动态调制能增强复杂周期预测 | 根据输入动态调整不同周期成分重要性 | **相对最一致，但幅度不足**。Harmonic 是四机制中 full-budget 信号最一致的正向（dyn_stack 在 Electricity 上 −1.9%/−1.2%），但远小于 residual 分支的贡献（见 §八） |

---

## 三、阶段0：Baseline 复现

**计划要求**：以未修改代码建立 5 数据集 × 4 horizons 的稳定 baseline。

**执行情况**：`original` 模式（无任何动态机制开关、无 residual head）full-budget 复现。96/192 来自 `weak_residual_matrix`（seed 2021），336/720 来自 `dyn_phase_full`（seed 2021；ETTh1 h720 为 2026），两批协议（lookback 720、seed 2021）已核对一致。

### 3.1 完整 baseline 表（test MSE / MAE）

| Dataset | h=96 | h=192 | h=336 | h=720 |
|---|---|---|---|---|
| ETTh1 | 0.3608 / 0.3862 | 0.4040 / 0.4093 | 0.4381 / 0.4314 | 0.4179 / 0.4403 |
| ETTh2 | 0.2808 / 0.3430 | 0.3440 / 0.3835 | 0.3735 / 0.4076 | 0.4254 / 0.4552 |
| ETTm1 | 0.2987 / 0.3486 | 0.3330 / 0.3651 | 0.3585 / 0.3813 | 0.4157 / 0.4127 |
| Electricity | 0.1290 / 0.2203 | 0.1459 / 0.2355 | 0.1661 / 0.2591 | 0.2010 / 0.2880 |
| Traffic | 0.3635 / 0.2322 | 0.3778 / 0.2399 | 0.3912 / 0.2503 | 0.4302 / 0.2707 |

### 3.2 与金标准对照（matched rerun，未静默替换金标准）

金标准 `docs/PHASEFORMER_GOLD_STANDARD.md` 为固定参照。本实验 matched rerun 与金标准偏差在单 seed 随机方差范围内（例：ETTh1 h96 MSE +0.5%/MAE +1.1%，ETTh2 h96 MSE +2.1%/MAE +1.5%，Traffic h96 MSE +0.7%/MAE −2.4%），不改变金标准权威性。

### 3.3 训练耗时（full-budget，original）

| Dataset | h336 | h720 | 备注 |
|---|---:|---:|---|
| ETTh1 | 1.4 min | 1.1 min | 30 ep |
| ETTh2 | 0.9 min | 0.9 min | 30 ep |
| ETTm1 | 4.2 min | 4.8 min | 30/23 ep |
| Electricity | 44.4 min | 26.3 min | 30/18 ep |
| Traffic | 40.6 min | 36.3 min | 16/13 ep |

训练量随数据集规模增长明显（ETT ~5k 样本 → Electricity/Traffic 百万级）。

---

## 四、阶段1：验证 Residual Branch 贡献

**计划要求**：比较"完整模型 vs 去除 residual head"。

**执行情况与设计澄清（修补）**：筛选协议中 baseline `original` 本身**未启用** residual head，因此 `no_residual`≡`original` 是**空实验**（参数与指标逐字节一致）。阶段1 的正确对比是：

- `original`（无 residual 分支）= 去除 residual
- `residual_full`（WeakPeriodResidualHead + PhaseLocalTrendHead）= 完整模型

full-budget 对 10 个 setting 均完成此对比。

### 结果（full-budget，test MSE 相对 original 变化 %）

| Dataset | h=336 | h=720 | 反馈 |
|---|---|---:|---|
| ETTh1 | +1.1% | −0.1% | 中性 |
| ETTh2 | −1.3% | **−8.1%** | **显著正** |
| ETTm1 | +2.2% | +1.1% | 负 |
| Electricity | −2.6% | −1.6% | 正 |
| Traffic | +2.9% | +2.4% | **明确负** |

**反馈**：residual 分支是**数据依赖**的——对 ETTh2（强趋势/低频）与 Electricity 稳定正收益，对 Traffic（通道稀疏）明确有害（+2.4%~+2.9%），对 ETTh1/ETTm1 中性。计划"后续改进保留 residual 分支"只在 ETTh2/Electricity 成立；Traffic 应关闭。

---

## 五、阶段2：Dynamic Phase Correction（核心实验）

**计划要求**：新增 `phase_correction.py`，`phase_embedding → phase_corrector(Δφ) → phase_warp → routing`；重点观察 h=336/720；分析指标含 MSE/MAE 与 **Peak shift error**（预测峰值位置 − 真实峰值位置）。

**执行情况**：`PhaseCorrection` 由 MLP 输出逐 phase-token 偏移 Δφ，warm-start identity（初始 Δφ≈0），flag-off 完全跳过。

### 结果（full-budget，test MSE 相对 original）

| Dataset | h=336 | h=720 |
|---|---:|---:|
| ETTh1 | −0.2% | +1.2% |
| ETTh2 | −1.5% | −1.9% |
| ETTm1 | +0.1% | −0.8% |
| Electricity | +0.3% | −0.1% |
| Traffic | −0.5% | +1.9% |

### Peak-shift error 分析（A 单独 vs dyn_full，h720）

计划阶段2 要求对 DPC 实验（A）单独观测峰值定位误差。补跑 A 单独审计（`research_runs/dyn_phase_audit_{electricity,etth2}_720_dpc`，从 `best.ckpt` 重算 test 预测）：

| Dataset | 分组 | A 单独 peak closer | dyn_full peak closer | A 单独 std closer | dyn_full std closer |
|---|---|---|---|---|---|
| Electricity | baseline_high_error | **9/10** | 10/10 | 10/10 | 5/10 |
| Electricity | candidate_regression | **10/10** | 9/10 | 9/10 | 3/10 |
| Electricity | candidate_improvement | **10/10** | 9/10 | 9/10 | 10/10 |
| ETTh2 | baseline_high_error | **0/10** | 0/10 | 10/10 | 10/10 |
| ETTh2 | candidate_regression | **7/10** | 4/10 | 2/10 | 10/10 |
| ETTh2 | candidate_improvement | **3/10** | 2/10 | 8/10 | 7/10 |

**反馈**：
- A 单独在 Electricity h720 即达 **9–10/10 峰值更近**，与 dyn_full（10/9/9）几乎相同 → **峰值定位校准可完全归因于 DPC**，非其他机制贡献。
- ETTh2 h720 上 A 单独 baseline_high_error 组 **0/10 峰值、10/10 std**，与 dyn_full 逐格一致 → ETTh2 上"峰值不校准、仅幅度/方差匹配"是 DPC 自身行为。
- 结论：peak-shift 收益**数据集依赖且可归因于 DPC**；DPC 的 MSE/MAE 增益微弱（弱信号、数据依赖），但在周期主导数据上有明确的峰值相位校准功能。

---

## 六、阶段3：Circular Phase Geometry

**计划要求**：以周期 Fourier embedding 替换 learnable 位置嵌入，使 phase token 感知周期边界连续性（P→1）。

**执行情况**：`CircularPhaseEmbedding` 以 `[sin(2πp/P), cos(2πp/P)]` 构建非持久 buffer，`PhaseEmbedding.forward` 在 `use_circular_pos` 时**替换**（而非叠加）可学习 `pos_embedding`（learnable 参数仅保留以维持 flag-off 初始化等价）。

### 结果（full-budget，B = A + Geometry，相对 original；括号内为 B 相对 A 的增量）

| Dataset | h=336 | h=720 |
|---|---:|---:|
| ETTh1 | −2.0%（ΔA −1.8pp） | +0.7%（ΔA −0.5pp） |
| ETTh2 | −0.8%（ΔA +0.7pp） | −3.0%（ΔA −1.1pp） |
| ETTm1 | +0.2%（ΔA +0.1pp） | −1.1%（ΔA −0.3pp） |
| Electricity | +0.7%（ΔA +0.4pp） | −0.5%（ΔA −0.4pp） |
| Traffic | +0.6%（ΔA +1.1pp） | +1.7%（ΔA −0.2pp） |

**反馈**：B 在多数 setting 优于 A（ETTh2 h720 −3.0%、ETTh1 h336 −2.0%、ETTm1 h720 −1.1%），**Circular Geometry 在 full-budget 上是微弱正增量**；但幅度有限，不足以单独支撑显著提升。

---

## 七、阶段4：Phase Rotation

**计划要求**：对 latent 特征做 2D 相位旋转（norm-preserving），增强相位轨迹表达能力。

**执行情况**：`PhaseRotation` 按 θ 对 latent 做旋转，warm-start θ≈0（identity）。

### 结果（full-budget，C = B + Rotation，相对 original；括号内为 C 相对 B 的增量）

| Dataset | h=336 | h=720 |
|---|---:|---:|
| ETTh1 | −1.0%（ΔB +1.0pp） | +1.0%（ΔB +0.3pp） |
| ETTh2 | +2.4%（ΔB +3.2pp） | +2.0%（ΔB +5.0pp） |
| ETTm1 | +3.4%（ΔB +3.2pp） | −0.7%（ΔB +0.4pp） |
| Electricity | +0.2%（ΔB −0.5pp） | −0.5%（ΔB 0.0pp） |
| Traffic | +2.1%（ΔB +1.5pp） | +1.5%（ΔB −0.2pp） |

**反馈**：**C 在 7/10 setting 上严格更差（C>B）**，仅 Electricity 336、Traffic 720 略优。**Phase Rotation 是四个机制中 full-budget 上最一致有害的**，建议关闭或重构（warm-start norm-preserving 旋转在本实现下不带来收益）。

---

## 八、阶段5：Harmonic Feature Modulation

**计划要求**：根据输入动态调制不同周期成分的重要性（z′=γz+β），增强多周期数据预测。

**执行情况**：`HarmonicModulation` 由输入周期特征生成 γ/β，warm-start γ=1、β=0（identity）。

### 结果（full-budget，D = dyn_stack = 四机制无 residual，相对 original）

| Dataset | h=336 | h=720 |
|---|---:|---:|
| ETTh1 | +1.0% | +2.0% |
| ETTh2 | +2.2% | +1.7% |
| ETTm1 | +1.7% | +0.1% |
| Electricity | **−1.9%** | **−1.2%** |
| Traffic | +2.9% | +0.8% |

筛选（30%/8ep）上 dyn_stack 是动态组合中最优（Electricity h336 −4.9%、ETTh2 h720 −1.7%、ETTm1 h336 −1.9%），但 full-budget 上仅 Electricity 稳定为负。

**反馈**：Harmonic 是四机制中信号最一致的正向（Electricity 上 dyn_stack 稳定 −1.9%/−1.2%），但幅度远小于 residual 分支；筛选系统高估其收益，以 full-budget 为准。

---

## 九、最终模型结构（dyn_full）

**计划要求**：按"九"接线最终模型：

```
Phase Alignment → Embedding → Dynamic Phase Correction → Geometry → Rotation → Cross Phase Routing → Harmonic Modulation → Residual Reconstruction → Forecast
```

**执行情况**：`dyn_full` 全开四个动态机制 + residual 双头，full-budget 确认。

### full-budget 结果（test MSE，相对 original）

| Dataset | h=336 | h=720 |
|---|---:|---:|
| ETTh1 | +0.9% | 0.0% |
| ETTh2 | −1.3% | **−8.3%** |
| ETTm1 | +1.6% | +2.2% |
| Electricity | −2.3% | **−2.7%** |
| Traffic | +1.7% | +3.0% |

### 深度审计（`analyze_experiment.py`，ETTh2/Electricity h720）

| 审计项 | ETTh2 h720 | Electricity h720 |
|---|---|---|
| dMSE / dMAE | −8.3% / −6.3% | −2.7% / −1.2% |
| 双指标优于金标准 | ✓（0.3901/0.4265 vs 0.402/0.436） | ✓（0.1955/0.2846 vs 0.201/0.285） |
| 改善单元占比 | 73.3% cells improved | sample_errors.csv 可复核 |
| 峰值更近 | 0/10（baseline_high_error） | 9–10/10 |
| std 更近 | 10/10 | 5/10、10/10 |
| 机制活动 \|δ\|,\|θ\|,\|γ−1\| | 0.119 / 0.314 / 0.447 | 0.145 / 0.256 / 0.602 |

**反馈**：dyn_full 的收益主体是 **residual 重构**（对比 §四：residual_full 与 dyn_full 在 ETTh2 −8.1% vs −8.3%、Electricity −2.6% vs −2.3% 数值接近）；动态机制自身的增量（residual_full → dyn_full）是微小的正或负。对 ETTh2/Electricity 有双指标金标准级表现（单 seed）；对 Traffic 是净负面。

---

## 十、完整消融实验

**计划要求**：Baseline / A（+Correction）/ B（+Geometry）/ C（+Rotation）/ D（+Harmonic）完整阶梯。

**执行情况**：筛选 + full-budget 双协议全部 10 setting 完成；full-budget 曾缺 A/B/C 中间阶梯，已补跑确认。

### 筛选（30%/8ep，val MSE 相对 original）

| Dataset | h | A +Corr | B +Geo | C +Rot | D dyn_stack |
|---|---|---|---:|---:|---:|
| ETTh1 | 336 | −0.1% | +1.1% | +1.2% | −1.0% |
| ETTh1 | 720 | 0.0% | +1.4% | +1.3% | +1.3% |
| ETTh2 | 336 | −0.2% | +0.4% | +0.4% | +0.4% |
| ETTh2 | 720 | −0.1% | +0.7% | +0.5% | −1.7% |
| ETTm1 | 336 | +0.9% | +1.0% | +2.0% | −1.9% |
| ETTm1 | 720 | −0.1% | −1.6% | +2.4% | +0.1% |
| Electricity | 336 | −0.3% | −0.0% | +0.4% | **−4.9%** |
| Electricity | 720 | −0.5% | +0.0% | −1.0% | −1.1% |
| Traffic | 336 | −1.0% | −0.7% | −1.0% | −1.1% |
| Traffic | 720 | +0.1% | +0.0% | +1.5% | +0.5% |

### Full-budget 消融（test MSE，相对 original）

| Dataset | h | A +Corr | B +Geo | C +Rot | D dyn_stack | dyn_full |
|---|---|---|---:|---:|---:|---:|---:|
| ETTh1 | 336 | −0.2% | −2.0% | −1.0% | +1.0% | +0.9% |
| ETTh1 | 720 | +1.2% | +0.7% | +1.0% | +2.0% | 0.0% |
| ETTh2 | 336 | −1.5% | −0.8% | +2.4% | +2.2% | −1.3% |
| ETTh2 | 720 | −1.9% | −3.0% | +2.0% | +1.7% | **−8.3%** |
| ETTm1 | 336 | +0.1% | +0.2% | +3.4% | +1.7% | +1.6% |
| ETTm1 | 720 | −0.8% | −1.1% | −0.7% | +0.1% | +2.2% |
| Electricity | 336 | +0.3% | +0.7% | +0.2% | **−1.9%** | **−2.3%** |
| Electricity | 720 | −0.1% | −0.5% | −0.5% | **−1.2%** | **−2.7%** |
| Traffic | 336 | −0.5% | +0.6% | +2.1% | +2.9% | +1.7% |
| Traffic | 720 | +1.9% | +1.7% | +1.5% | +0.8% | +3.0% |

### 消融结论

1. **四机制叠加（dyn_full）的收益几乎全部来自 residual 分支**；动态机制自身的增量（residual_full → dyn_full）是微小的正或负。
2. **A（Phase Correction）**：ETTh2 上稳定改善（−1.5%/−1.9%），其余 setting 弱信号、数据依赖，非普适提升。
3. **B（A + Geometry）**：多数 setting 优于 A，**full-budget 上正向增量**。
4. **C（B + Rotation）**：7/10 setting 上 C 严格更差（C>B），**最一致有害**。
5. **D（dyn_stack）**：full-budget 上几乎没有显著负值，唯一例外是 Electricity（−1.9%/−1.2%）。

---

## 十一、结果分析逻辑

**计划要求**：按三种情况判定瓶颈。

| 情况 | 计划判定 | 实测 |
|---|---|---|
| 情况1：Dynamic Correction 提升最大 → 固定 phase representation 是瓶颈 | **未出现** | A 仅 ETTh2 稳定改善，非提升最大机制 |
| 情况2：Geometry 提升明显 → 周期拓扑约束有效 | **部分出现** | B 在 ETTh2 h720（−3.0%）、ETTh1 h336（−2.0%）、ETTm1 h720（−1.1%）明显为负且多数优于 A，但整体不单调 |
| 情况3：Harmonic Modulation 提升 → 数据含明显多周期结构 | **部分出现** | dyn_stack 为筛选上最佳动态组合；full-budget 仅 Electricity 稳定负值；Harmonic 是四机制中信号最一致但幅度远小于 residual |

**可测观察汇总**：
1. 动态机制中**最一致有害的是 Rotation（C）**（7/10 setting C>B）；**Geometry（B）微弱正增量**；**A 仅 ETTh2 有效**；唯一稳定负值来自含 residual 的 dyn_full 在 ETTh2/Electricity，且主要由 residual 贡献。
2. 筛选系统性高估 residual_full 收益（−13%~−24%）而 full-budget 仅 ±3%，Traffic 反转为正（+2.4~2.9%）；**以 full-budget 为准**。
3. peak 校准收益数据集依赖（Electricity ✓、ETTh2 ✗），且**可归因于 DPC 单独**（§五）。

---

## 十二、实验优先级

**计划排序**：① Dynamic Correction ② Phase Rotation ③ Circular Geometry ④ Harmonic Modulation。

**实测排序（full-budget 为准）**：

1. **Residual reconstruction**（ETTh2/Electricity 显著，Traffic 应关闭）—— 当前主要贡献源，代码已是。
2. **Harmonic modulation**（信号最一致，Electricity 有效）。
3. **Circular Geometry**（微弱正增量，可保留）。
4. **Dynamic Phase Correction**（计划列第一）—— 实测仅 ETTh2 有效、其余中性/退化，但**是 Electricity peak 校准的唯一来源（§五）**，对周期主导数据仍值得保留。
5. **Phase Rotation**（最一致有害）—— 建议关闭或重构。

**反馈**：计划阶段12 的优先级假设**未获支持**，实测主贡献是 residual 重构而非动态相位机制；但 DPC 的峰值相位校准功能在周期主导数据上有独立价值。

---

## 十三、最终研究方向

**计划核心思想**：从静态 phase token 建模扩展到动态 phase trajectory 建模。

**实测反馈**：

- **动态相位机制在本实现（warm-start identity、默认超参、单 seed）下未带来超越 residual 重构的收益**。动态 trajectory 建模的潜力在 Electricity 的 peak 校准上得到部分验证（DPC 单独即达 9–10/10 峰值更近，§五），但在 ETTh2 等强趋势数据上未转化为 MSE/MAE 提升。
- **最终模型（dyn_full）的价值集中在 residual reconstruction + 适度 harmonic modulation**，对 ETTh2 与 Electricity 有双指标金标准级表现（单 seed）。
- **进一步方向建议**：a) 多 seed 确认 dyn_full 在 ETTh2/Electricity 的收益稳健性；b) 针对周期主导数据（Electricity 类）单独优化 dynamic correction（因其 peak 校准收益明确）；c) Traffic 上验证关闭 residual 的配置。

---

## 附录：数据落盘与复核

| 数据 | 位置 |
|---|---|
| Stage A 筛选 | `research_runs/dyn_phase_screen/runs/mechanism_screen_1_*/`（汇总 `screen_summary.csv`） |
| Full-budget | `research_runs/dyn_phase_full/dynphase_{ds}_{h}_{mode}_*/metrics.csv` |
| 审计（dyn_full） | `research_runs/dyn_phase_audit_{etth2,electricity}_720/`（six-file 协议） |
| 审计（DPC 单独） | `research_runs/dyn_phase_audit_{etth2,electricity}_720_dpc/`（six-file 协议） |

> 全部数值可由对应 `metrics.csv` 逐行复核；本反馈文件与 `docs/DYNAMIC_PHASE_EXPERIMENT_REPORT.md` 数据一致。
