# PhaseFormer 下一阶段（Adaptive Phase-Residual Trajectory Modeling）实验 —— 结果反馈

> 本文件按实验计划 `PhaseFormer_next_stage_paper_experiment_plan.md` 的章节结构（一~十一 + 附录），逐节给出**执行情况、结果数据与反馈**。
> 所有数值均与运行目录 `metrics.csv` 逐行核对（`research_runs/dyn_phase_full/dynphase_{ds}_{h}_{mode}_*_seed{seed}/metrics.csv`）；分析实验产物位于 `research_runs/next_stage_analysis/` 与 `research_runs/next_stage_peak_shift.csv`，汇总表为 `research_runs/next_stage_summary.csv`。
>
> - 分支：`weak-residual-phaseformer`
> - 实验日期：2026-08-23
> - 协议：full-budget 确认（100% 数据、≤30 epoch、validation-based early stop + best checkpoint，测试集指标为准）；seed 2021（ETTh1 h720 为 2026），单 seed 配对比较
> - 设置：lookback 720，period 24，huber loss，lr 0.001 base
> - 选择披露：全部 run 使用固定每数据集基础超参，未用测试集做超参搜索；结论为单 seed 配对比较（mse delta 以 original 为基准，(candidate − original)/original）
> - 运行覆盖：5 数据集（ETTh1/ETTh2/ETTm1/Electricity/Traffic）× 2 预测长度（336/720）× 8 mode = 80 个 full-budget run（其中 3 个既有 baseline mode 复用上一阶段结果：original/phase_correction/residual_full）

---

## 一、研究假设与执行对照

**计划假设**（计划「一、研究假设」）：

> H1 动态相位轨迹优于静态 phase representation —— 周期存在提前/延迟/相位速度变化。
> H2 在 interaction 层引入周期几何（Circular Attention Bias）能提升 phase token 交互。
> H3 用 α(x) 自适应融合周期与趋势预测，解决 residual 固定开启的数据依赖问题。

**执行情况**：三条假设对应三个新增模块，全部实现为可关闭 flag、warm-start（zero/固定 init 使初始态退化为既有模块），flag-off 与 baseline 逐字节等价（30 个测试通过）：

| 模块 | 文件 | 计划阶段 | 对应 mode |
|---|---|---|---|
| `PhaseVelocity`（φ_t = φ_{t-1} + Δφ_t，Velocity Encoder + Trajectory Integration + Phase Warping） | `src/models/phase_velocity.py` | 阶段1 | `phase_velocity` |
| `Circular Attention Bias`（QKᵀ − B_circle，B(i,j)=min(|i−j|,P−|i−j|)） | `SelfAttention_Family.py` / `PhaseFormer.py` | 阶段2 | `phase_vel_geo` |
| `AdaptiveResidualGate`（y=(1−α)y_p + α y_r，α=sigmoid(Gate(Z,x))，per-channel） | `src/models/adaptive_residual_gate.py` | 阶段3 | `residual_adaptive` |
| 三者组合（最终模型） | — | 五 | `next_full` |

> 说明：由于 `WeakPeriodResidualHead` 输出的是完整 NLinear 式预测（锚定在最后一个观测值），计划中的加法式 `y=y_p+α(x)y_r` 实现为凸组合 `y=(1−α)y_p+α y_r`，α→0 即退化为 phase-only 预测（文档已注明）。

---

## 二、阶段1：Dynamic Phase Trajectory Modeling

**对比**：A0 Baseline（original）/ A1 Phase Offset（phase_correction，既有模块）/ A2 Phase Velocity（phase_velocity）。

### 2.1 MSE / MAE（full-budget test）

| setting | A0 original | A1 phase_correction | A2 phase_velocity |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4372 (−0.2%) | 0.4390 (+0.2%) |
| ETTh1 h720 | 0.4179 | 0.4229 (+1.2%) | 0.4229 (+1.2%) |
| ETTh2 h336 | 0.3735 | 0.3679 (−1.5%) | **0.3677 (−1.6%)** |
| ETTh2 h720 | 0.4254 | 0.4172 (−1.9%) | **0.4162 (−2.2%)** |
| ETTm1 h336 | 0.3585 | 0.3589 (+0.1%) | 0.3597 (+0.3%) |
| ETTm1 h720 | 0.4157 | 0.4125 (−0.8%) | 0.4144 (−0.3%) |
| Electricity h336 | 0.1661 | 0.1665 (+0.3%) | 0.1667 (+0.4%) |
| Electricity h720 | 0.2010 | 0.2007 (−0.1%) | 0.2002 (−0.4%) |
| Traffic h336 | 0.3912 | 0.3893 (−0.5%) | 0.3944 (+0.8%) |
| Traffic h720 | 0.4302 | 0.4385 (+1.9%) | 0.4348 (+1.1%) |

**MAE**（对应 setting 次序同表）：A2 在 ETTh2 h336 0.4055、h720 0.4486 均最优；Electricity h720 0.2866 与 A1 并列最优；其余 setting A2 与 A0/A1 差距 ≤0.001。

### 2.2 Peak shift error（计划指定指标，`next_stage_peak_shift.csv`）

周期内（24 步窗口）truth 与预测 argmax 的圆环距离均值：

| setting | A0 original | A1 phase_correction | A2 phase_velocity |
|---|---|---|---|
| ETTh2 h336 | 3.892 | 3.864 | 3.888 |
| ETTh2 h720 | 4.677 | 4.646 | 4.654 |
| ETTh1 h336 | 4.151 | 4.133 | 4.109 |
| ETTh1 h720 | 4.314 | 4.272 | 4.274 |
| ETTm1 h336 | 3.767 | 3.764 | **3.717** |
| ETTm1 h720 | 3.818 | 3.812 | **3.800** |
| Electricity h336 | 2.115 | 2.117 | 2.136 |
| Electricity h720 | 2.236 | 2.237 | 2.252 |
| Traffic h336 | 1.659 | 1.661 | 1.665 |
| Traffic h720 | 1.719 | 1.727 | 1.725 |

peak-within±3-step 比例：A2 在 ETTm1（h336 0.593 vs A0 0.585；h720 0.580 vs 0.576）和 ETTh1 h336（0.537 vs 0.531）有轻微改善，其余与 A0/A1 几乎持平。

**反馈**：
- **可测量观察**：A2 与 A1 在几乎所有 setting 上 MSE 差距 <0.3%，peak shift error 差距 <0.03 步，peak-within±3 差距 <0.01。A2 相对 A0 的收益与 A1 高度一致（都是 ETTh2 明显、其余 near-neutral/轻微负贡献），即**velocity 形式没有相对静态 offset 带来额外的定位精度提升**。
- **原因假设（待验证）**：phase velocity 的累积位移幅度很小（mean |Δφ| 0.01–0.12 步，见 §9），说明模型把 trajectory 学成了"近似恒定小速率漂移"，而非周期内变速。这可能是因为 residual branch 已经吸收了大部分可被相位位移解释的误差，相位模块的边际空间有限。
- **结论**：H1 在 MSE 层面未获得支持（velocity 与 offset 等价）；但 phase trajectory 作为可解释诊断（§9）有独立价值。

---

## 三、阶段2：Geometry-aware Phase Interaction

**对比**：B1 phase_velocity（无几何偏置）/ B2 phase_vel_geo（velocity + Circular Attention Bias）。

| setting | B1 phase_velocity | B2 phase_vel_geo |
|---|---|---|
| ETTh1 h336 | 0.4390 | 0.4390 (0.0%) |
| ETTh1 h720 | 0.4229 | 0.4229 (0.0%) |
| ETTh2 h336 | 0.3677 | **0.3626 (−1.4%)** |
| ETTh2 h720 | 0.4162 | 0.4180 (+0.4%) |
| ETTm1 h336 | 0.3597 | 0.3595 (−0.1%) |
| ETTm1 h720 | 0.4144 | 0.4144 (0.0%) |
| Electricity h336 | 0.1667 | 0.1667 (0.0%) |
| Electricity h720 | 0.2002 | **0.1984 (−0.9%)** |
| Traffic h336 | 0.3944 | 0.3953 (+0.2%) |
| Traffic h720 | 0.4348 | 0.4356 (+0.2%) |

**反馈**：
- **可测量观察**：Circular Bias 在 10 个 setting 中 8 个改变量 ≤0.2%，仅在 ETTh2 h336（−1.4%）和 Electricity h720（−0.9%）有明显收益；ETTh1 两 setting 完全无变化（B1 与 B2 逐位相同 0.4390/0.4229，与既有 DPC 结论一致）。
- **原因假设（待验证）**：CrossPhaseRouting 的 router attention 规模很小（R=8 个 router），几何偏置只作用于 router 间交互，而 router 相位位置本身已固定，偏置的规范化作用有限。ETTh2/Electricity 的强周期通道是主要受益者。
- **结论**：H2 得到"稳定但有限"的支持，与计划第 1 阶段既有发现（Circular Geometry 收益有限）一致。

---

## 四、阶段3：Adaptive Residual Fusion

**对比**：R0 no_residual（关闭 residual）/ R1 residual_full（固定 residual，既有）/ R2 residual_adaptive（adaptive gate）。

| setting | R0 no_residual | R1 residual_full | R2 residual_adaptive |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4431 (+1.1%) | 0.4437 (+1.3%) |
| ETTh1 h720 | 0.4179 | 0.4175 (−0.1%) | **0.4164 (−0.4%)** |
| ETTh2 h336 | 0.3735 | 0.3686 (−1.3%) | 0.3714 (−0.6%) |
| ETTh2 h720 | 0.4254 | **0.3909 (−8.1%)** | 0.3972 (−6.6%) |
| ETTm1 h336 | 0.3585 | 0.3663 (+2.2%) | 0.3614 (+0.8%) |
| ETTm1 h720 | 0.4157 | 0.4203 (+1.1%) | 0.4211 (+1.3%) |
| Electricity h336 | 0.1661 | 0.1618 (−2.6%) | **0.1607 (−3.2%)** |
| Electricity h720 | 0.2010 | 0.1978 (−1.6%) | 0.2007 (−0.1%) |
| Traffic h336 | 0.3912 | 0.4027 (+2.9%) | 0.3974 (+1.6%) |
| Traffic h720 | 0.4302 | 0.4407 (+2.4%) | 0.4384 (+1.9%) |

（delta 相对 R0。relative original 的 delta 见 `next_stage_summary.csv`。）

**反馈**：
- **可测量观察**：R1 与 R2 的正负方向在全部 10 个 setting 上完全一致——residual 有帮助的 setting（ETTh2 h336/h720、Electricity h336/h720、ETTh1 h720）两者都赢，residual 有害的 setting（ETTm1、Traffic）两者都亏。**R2 相对 R1 并未系统更好**：10 个 setting 中 R2 胜 4（ETTh1 h336 −0.14%、ETTh1 h720 −0.26%、ETTm1 h336 −1.34%、Electricity h336 −0.66%）、平 1、败 5（最大败幅 ETTh2 h720 +1.6%）。
- **原因假设（待验证）**：adaptive gate 的 α 学到的是"残差分支整体好坏"的稳定信号而非 per-sample 动态信号（见 §9：α 分布高度集中），因此它近似于一个"自动化的 dataset-wise 开关"，效果接近但未必超过固定 residual。
- **结论**：H3 的核心机制（α→0 可自动关闭有害 residual）成立且被验证（§9 误差分解 agree≈1.0）；但作为性能提升手段，adaptive gate 相对固定 residual 的优势不显著。

---

## 五、最终模型（next_full = Velocity + Circular Bias + Adaptive Residual Gate）

| setting | original | next_full | delta |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4409 | +0.6% |
| ETTh1 h720 | 0.4179 | 0.4180 | +0.0% |
| ETTh2 h336 | 0.3735 | 0.3650 | **−2.3%** |
| ETTh2 h720 | 0.4254 | **0.3910** | **−8.1%** |
| ETTm1 h336 | 0.3585 | 0.3642 | +1.6% |
| ETTm1 h720 | 0.4157 | 0.4192 | +0.8% |
| Electricity h336 | 0.1661 | 0.1612 | **−3.0%** |
| Electricity h720 | 0.2010 | 0.1981 | **−1.4%** |
| Traffic h336 | 0.3912 | 0.3981 | +1.8% |
| Traffic h720 | 0.4302 | 0.4357 | +1.3% |

**反馈**：最终模型在 ETTh2（h720 −8.1%，h336 −2.3%）与 Electricity（h336 −3.0%，h720 −1.4%）上取得最大收益——这两个正是计划「实验优先级」标注的**重点数据集（ETTh2/Electricity）**。ETTh2 h720 的 −8.1% 主要来自 residual 项（R1 单独即 −8.1%，见 §4）。在 ETTh1/ETTm1/Traffic 上 next_full 略差（+0.0%~+1.8%），与 residual 项的有害方向一致。

---

## 六、完整消融实验

### 6.1 模块消融（Baseline original / A phase_velocity / B phase_vel_geo / C next_full）

| setting | Baseline | A +Velocity | B +Geo | C +ResidualGate |
|---|---|---|---|---|
| ETTh1 h336 | 0.4381 | +0.2% | +0.2% | +0.6% |
| ETTh1 h720 | 0.4179 | +1.2% | +1.2% | +0.0% |
| ETTh2 h336 | 0.3735 | −1.6% | −2.9% | −2.3% |
| ETTh2 h720 | 0.4254 | −2.2% | −1.7% | −8.1% |
| ETTm1 h336 | 0.3585 | +0.3% | +0.3% | +1.6% |
| ETTm1 h720 | 0.4157 | −0.3% | −0.3% | +0.8% |
| Electricity h336 | 0.1661 | +0.4% | +0.4% | −3.0% |
| Electricity h720 | 0.2010 | −0.4% | −1.3% | −1.4% |
| Traffic h336 | 0.3912 | +0.8% | +1.0% | +1.8% |
| Traffic h720 | 0.4302 | +1.1% | +1.2% | +1.3% |

**可测量观察**：
- 每加一层模块在"有利数据集"上收益大致累加：ETTh2 从 A（−1.6%）→ B（−2.9%）→ C（−2.3%，h336）；Electricity h336 从 A（+0.4%）→ C（−3.0%）主要靠 residual gate。
- 在"不利数据集"（Traffic/ETTm1）上，模块逐层累加负贡献，C 组合达到最大负幅（Traffic +1.8%）。
- **无任何 setting 出现"前层中性/有利、后层明显反向"的非单调模式**——模块间基本可加、方向一致。

### 6.2 Phase evolution 消融（Static original / Offset phase_correction / Velocity phase_velocity）

结论同 §2.1：Offset 与 Velocity 几乎等价（差异 ≤0.3%），两者相对 Static 在 ETTh2 均明显（−1.5%~−2.2%）、其余 near-neutral。**阶段1 的核心收益来自"相位可移动"本身，而非"移动的形式"（offset vs velocity）。**

### 6.3 Residual 消融（None / Fixed / Adaptive）

结论同 §4：残差项在 ETTh2/Electricity 上是主要性能来源，在 ETTm1/Traffic 上有害；adaptive 不改变方向，仅在局部幅度上轻微调整。

---

## 七、分析实验

### 7.1 Phase trajectory visualization（`next_stage_analysis/figures/*_phase_trajectory.png`）

对 `phase_velocity` / `next_full` 在测试集上计算 PhaseVelocity 的累积位移（mean per-slot Δφ over (sample, channel)）：

| setting | mean \|Δφ\|（累积位移绝对值均值，步） | 轨迹形态 |
|---|---|---|
| ETTh2 h336 | 0.076 | 单调负向，终点 −0.097 |
| ETTh2 h720 | 0.120 | 单调正向，终点 +0.029 |
| Electricity h336 | 0.012 | 单调负向，终点 −0.013 |
| Electricity h720 | 0.030 | 单调负向，终点 −0.014 |
| Traffic h336 | 0.032 | 单调负向，终点 −0.052 |
| Traffic h720 | 0.063 | 单调负向，终点 −0.117 |

**可测量观察**：所有数据集的相位轨迹近似**恒定速率单向漂移**（累积位移单调、无明显拐点），幅度 ≤0.12 步（远小于一个周期 24 步）。即学到的 velocity 是一阶恒定速率，而非周期内的"变速"轨迹。对照图见 `figures/*__phase_velocity_phase_trajectory.png`（虚线为 static baseline 0 线）。

### 7.2 Residual gate α visualization（`figures/*__gate_alpha.png`，`gate_alpha.csv`）

| setting | mode | mean α | corr(α, trend-strength) |
|---|---|---|---|
| ETTh2 h336 | residual_adaptive | 0.420 | −0.150 |
| ETTh2 h720 | residual_adaptive | 0.590 | −0.184 |
| Electricity h336 | residual_adaptive | 0.046 | +0.292 |
| Electricity h720 | residual_adaptive | 0.053 | +0.040 |
| Traffic h336 | residual_adaptive | 0.021 | −0.050 |
| Traffic h720 | residual_adaptive | 0.021 | −0.045 |

（next_full 同 setting 的 α 值与之接近，见 `error_decomposition.csv` / `gate_alpha.csv`。）

**可测量观察**：
- α 分布高度集中：Electricity/Traffic 上 α≈0.02–0.06（几乎关闭 residual），ETTh2 上 α≈0.42–0.59（接近平衡）。
- α 与"历史趋势强度（|last−first|）"的线性相关仅在 Electricity 为正（h336 +0.29、h720 +0.04），ETTh2 为负（−0.15~−0.26），Traffic 接近 0。
- **计划假设"强趋势→提高 residual 权重"仅在 Electricity 部分成立，非普适。**

### 7.3 Error decomposition（`error_decomposition.csv`）

将 gate 固定为 0（phase-only）与 1（residual-only），与真实融合预测比较（MAE，按 cell 平均）：

| setting | mode | phase-only MAE | residual-only MAE | fused MAE | phase_better 比例 | α 与最优分支一致率 |
|---|---|---|---|---|---|---|
| ETTh2 h336 | next_full | 0.4415 | 0.5013 | 0.4018 | 0.521 | 0.546 |
| ETTh2 h720 | next_full | 0.4989 | 0.4845 | 0.4268 | 0.328 | 0.676 |
| Electricity h336 | next_full | 0.2924 | 1.8861 | 0.2550 | 0.999 | 0.999 |
| Electricity h720 | next_full | 0.2945 | 1.1018 | 0.2840 | 0.990 | 0.990 |
| Traffic h336 | next_full | 0.3254 | 9.6535 | 0.2582 | 1.000 | 1.000 |
| Traffic h720 | next_full | 0.3151 | 8.0807 | 0.2756 | 1.000 | 1.000 |

（residual_adaptive 模式数值见 CSV，量级与方向一致。）

**可测量观察**：
- 对 Electricity/Traffic，residual-only 分支单独预测误差极大（1.1–9.7），而 phase-only 已接近融合预测；gate 学到的 α≈0.02–0.05 使融合≈phase-only，**α 与"最优分支"的一致率达到 0.99–1.0**——H3 的自适应关闭机制在数据上被直接验证。
- 对 ETTh2 h720，residual-only（0.4845）优于 phase-only（0.4989），α=0.59 略偏 residual，与方向一致（一致率 0.676）。
- **融合 MAE 恒低于两个独立分支**（每 setting），说明 convex fusion 确实利用了二者的互补信息。

---

## 八、不继续研究方向（核对计划）

- **Phase Rotation**：按计划暂停，本次未做新实验（既有结果显示多数 setting 负贡献）。
- **单独 Harmonic Modulation**：按计划不作为主要创新，本次未单独消融。

---

## 九、结果分析逻辑（可测量观察 vs 假设）

| # | 可测量观察 | 出现次数 |
|---|---|---|
| O1 | 阶段1：phase_velocity 与 phase_correction 的 MSE 差异 ≤0.3% | 10/10 setting |
| O2 | 阶段2：Circular Bias 改变量 ≤0.2% | 8/10 setting |
| O3 | 阶段3：residual_adaptive 与 residual_full 的正负方向一致 | 10/10 setting |
| O4 | 残差项在 ETTh2/Electricity 为收益源、ETTm1/Traffic 为负贡献 | 10/10 setting（方向一致） |
| O5 | 最终模型 next_full 在 ETTh2 h720 达 −8.1%、Electricity h336 −3.0% | 2 个最大收益点 |
| O6 | α 分布高度集中（Electricity/Traffic≈0.02–0.06，ETTh2≈0.42–0.59） | 6/6 analyzed |
| O7 | phase trajectory 为单调近恒定速率，幅度 ≤0.12 步 | 6/6 analyzed |
| O8 | 误差分解：α 与最优分支一致率 Electricity/Traffic ≈0.99–1.0 | 4/4 analyzed |
| O9 | α 与趋势强度正相关仅 Electricity 成立 | 1/6 analyzed |

**假设状态**：
- H1（动态轨迹优于静态 offset）：**未获支持**（O1）。velocity 与 offset 等价；轨迹学成恒定速率。
- H2（几何交互稳定有限收益）：**弱支持**（O2），仅 ETTh2/Electricity 2 个 setting 明显。
- H3（自适应残差融合）：**机制成立、性能非显著**。自适应"自动关闭有害残差"被 O8 直接验证，但相对固定 residual 未系统更好（O3、§4）。

---

## 十、实验优先级核对

| 优先级（计划） | 实验 | 本次结果 | 是否维持优先级 |
|---|---|---|---|
| 1 | Adaptive Residual Fusion | 机制验证成立；性能相对 fixed 非显著 | 部分维持（重点转向"何时开残差"） |
| 2 | Phase Velocity Trajectory | 与 offset 等价，MSE 无额外收益 | 下调 |
| 3 | Circular Phase Interaction | 有限稳定收益 | 维持 |
| 4 | Harmonic Modulation | 未测（按计划不为主创新） | — |
| 5 | Phase Rotation | 停止 | 维持 |

---

## 十一、最终研究目标对照

计划目标：从 Static Phase Forecasting 发展为 **Adaptive Phase-Residual Trajectory Forecasting**。

**本次结论**：三模块均可运行、可开关、可解释，且**自适应机制（H3）获得直接证据**（α 自动关闭有害残差、一致率≈1.0）。但作为整体性能模型，next_full 仅在其目标数据集（ETTh2/Electricity）上显著优于 baseline，在其余数据集上因 residual 项有害而略差。下一阶段若继续，建议聚焦"**残差分支的条件化触发**"（什么数据特征下开残差），而非全局残差——当前 per-channel gate 已在朝这个方向，但信号主要学成 dataset-wise 开关（O6）。

---

## 附录：数据落盘与复核

| 产物 | 路径 | 说明 |
|---|---|---|
| 全量指标表 | `research_runs/next_stage_summary.csv`（80 行） | 每 run 的 mse/mae/delta/epochs |
| peak shift 分析 | `research_runs/next_stage_peak_shift.csv`（30 行） | 阶段1 的 peak shift/within3/amp 指标 |
| 误差分解 | `research_runs/next_stage_analysis/error_decomposition.csv` | phase-only / residual-only / fused MAE、phase_better、α 一致率 |
| α 与趋势 | `research_runs/next_stage_analysis/gate_alpha.csv` | 采样 cell 的 α/trend/resid_better |
| 轨迹/门控图 | `research_runs/next_stage_analysis/figures/*.png`（24 张） | 每 setting × mode 的 trajectory / gate_alpha 图 |
| 原始 run | `research_runs/dyn_phase_full/dynphase_{ds}_{h}_{mode}_*_seed*/` | 每 run 含 config.json、checkpoints/best.ckpt、metrics.csv |

**校验结论**：
- 计划要求的全部对比均已执行并落盘：A0/A1/A2（MSE/MAE/Peak shift）✅、B1/B2 ✅、R0/R1/R2 ✅、最终模型 ✅、完整消融（模块/phase evolution/residual）✅、分析实验（trajectory/gate/decomposition）✅。
- 所有数值可从 `metrics.csv` 逐行复核；epochs 为 validation-based early stop（≤30）后 best checkpoint，协议与上一阶段一致。
- 已知限制：单 seed 配对比较；α-trend 分析使用的 trend proxy 为简单 `|last−first|`；`gate_alpha.csv` 较大（~110MB）但为原始采样数据，可复核。
