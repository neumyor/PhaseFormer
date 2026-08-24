# PhaseFormer Pure Phase Modeling 下一阶段实验结果反馈

> 计划文档：`docs/PhaseFormer_pure_phase_next_stage_experiment_plan.md`
> 对应提交：`1653cd1`（模块实现）、`00f09dc`（实验/分析工具链）
> 报告日期：2026-08-24

## 0. 执行摘要

本阶段按计划实现 4 个纯 phase-space 模块（Multi-scale Phase Representation、
Dynamic Phase Deformation、Phase Graph Interaction、Trajectory Decoder），
注册 7 个消融 mode（含最终模型 `pure_full`），并按既有协议跑 full-budget
（≤30 ep、val early stop + best ckpt、lookback 720、period 24、单 seed 配对）。

**核心结论（可测量）：**

| 模式 | 可用 setting 数 | 平均 ΔMSE% vs original | wins(<0) |
|---|---|---|---|
| multiscale_phase | 9 | **+0.53** | 2/9 |
| phase_deformation | 9 | **−0.09** | 5/9 |
| phase_geo | 9 | **−0.16** | 4/9 |
| phase_graph | 9 | **−0.10** | 4/9 |
| predictor_mlp | 9 | **+0.03** | 4/9 |
| trajectory_decoder | 8 | **+33.61** | 3/8 |
| pure_full（最终模型） | 8 | **+33.54** | 2/8 |

1. **阶段1–3 的三个表示/演化/交互模块均为与静态 baseline 持平**：单独消融的平均
   ΔMSE 在 −0.2% 到 +0.5% 之间，跨 setting 符号混合（各 4–5/9 有改进），无任何
   一致性收益。多尺度门控 `zeta` 确实被打开（99% 维度非零，mean|ζ|≈0.17），
   变形场学到压缩（s≈0.67）但累计位移 <0.1 步——即模块"被使用"但幅度不足以改变预测。
2. **Trajectory Decoder 是唯一强效应组件，但是灾难性负面**：在 ETTm1
   （+90.5% / +71.8%）、Electricity（+26%）、Traffic（+59%）上大幅退化，仅
   ETTh1/ETTh2 持平或略好。分析表明低阶多项式约束确实让输出更平滑（跨期
   smoothness 0.894 vs baseline 0.945，−5.4%），但**以显著劣化相位峰值对齐为代价**
   （peak shift 3.67 vs 3.24，within3 0.592 vs 0.646）。
3. **最终模型 `pure_full` 继承 trajectory decoder 的失败**：在 3/5 数据集灾难性退化
   （+25% 到 +91%），仅在 ETTh2 h720（−4.2%）等少数 setting 有改进。
4. **研究叙事不被数据支持**：纯 phase modeling 的提升（如果有）幅度远小于既有
   residual 收益，且主要来自 trajectory decoder 的错误方向。"Adaptive Phase
   Geometry Forecasting 的所有性能提升来自 phase 四方面、不依赖 residual"这一
   最终目标未达成。

**完成情况披露**：70 个计划 run 中完成 **61 个**（用户要求中途停止实验）。缺失 9 个：
`Traffic h720` 的 `trajectory_decoder`、`pure_full`（实验停止时正在跑），以及
`ETTh1 h720 (seed2026)` 整批 7 个 mode（batch 3 未启动）。所有结论基于已完成的
61 个 run；缺失项的表格以 `--` 标注。

---

## 1. 研究假设核对（计划"一"）

**Hypothesis 1：当前 phase token 表达缺少动态周期结构 → 需从 static 转向 adaptive
phase geometry representation。**

- 可测量观察：三个"adaptive geometry"模块（deformation / geo / graph）单独消融的
  平均 ΔMSE 为 −0.09% / −0.16% / −0.10%，均与静态 baseline 持平；相位对齐指标
  （peak shift）有微幅改善（3.11 / 3.11 / 3.08 vs 3.24），但不足以转化为 MSE/MAE 收益。
- **结论：假设未被证实。** 增加动态几何结构没有带来可测量的端到端精度提升；
  模块确实"学到了东西"（见分析实验），但学到的变形幅度远小于一个相位槽。

---

## 2. 阶段1：Multi-scale Phase Representation（计划"二"）

实现：`src/models/multiscale_phase.py`。长周期视图 = 相邻 `coarse=2` 组沿周期轴
平均（保留相位槽对齐，P_in 30→15），独立嵌入后经 `zeta ⊙ Z_long` 门控融合进短周期
嵌入；`zeta` init 0 → 精确 warm-start。

**结果（MSE，delta% 以 original 为 base，负=更优）：**

| setting | original | multiscale_phase | Δ% |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4443 | +1.4 |
| ETTh1 h720 | 0.4179 | -- | -- |
| ETTh2 h336 | 0.3735 | 0.3783 | +1.3 |
| ETTh2 h720 | 0.4254 | 0.4307 | +1.3 |
| ETTm1 h336 | 0.3585 | 0.3574 | −0.3 |
| ETTm1 h720 | 0.4157 | 0.4100 | −1.4 |
| Electricity h336 | 0.1661 | 0.1662 | +0.0 |
| Electricity h720 | 0.2010 | 0.2010 | +0.0 |
| Traffic h336 | 0.3912 | 0.3984 | +1.8 |
| Traffic h720 | 0.4302 | 0.4326 | +0.5 |

- 平均 ΔMSE **+0.53%**（2/9 wins）。仅在 ETTm1 两 setting 有小幅改进（−0.3%/−1.4%），
  ETTh1/ETTh2/Traffic 一致小幅退化（+1.3%~+1.8%）。
- **观察**：多尺度融合在多数数据上是略负面或中性，无系统性收益。

---

## 3. 阶段2：Dynamic Phase Deformation Modeling（计划"三"）

实现：`src/models/phase_deformation.py`。每槽速率 `v=scale·tanh(net_rate)`（scale 0.2）
+ 拉伸因子 `s=1+tanh(net_stretch)`，位移场 `delta=cumsum(v·s)`，k=2 scatter warp；
双头 zero-init → identity。

**结果（MSE，含阶段2参照 phase_velocity）：**

| setting | original | phase_velocity | phase_deformation |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4390 (+0.2) | 0.4419 (+0.9) |
| ETTh1 h720 | 0.4179 | 0.4229 (+1.2) | -- |
| ETTh2 h336 | 0.3735 | 0.3677 (−1.6) | 0.3670 (−1.7) |
| ETTh2 h720 | 0.4254 | 0.4162 (−2.2) | 0.4175 (−1.9) |
| ETTm1 h336 | 0.3585 | 0.3597 (+0.3) | 0.3613 (+0.8) |
| ETTm1 h720 | 0.4157 | 0.4144 (−0.3) | 0.4133 (−0.6) |
| Electricity h336 | 0.1661 | 0.1667 (+0.4) | 0.1658 (−0.2) |
| Electricity h720 | 0.2010 | 0.2002 (−0.4) | 0.1991 (−0.9) |
| Traffic h336 | 0.3912 | 0.3944 (+0.8) | 0.3957 (+1.1) |
| Traffic h720 | 0.4302 | 0.4348 (+1.1) | 0.4370 (+1.6) |

- 平均 ΔMSE **−0.09%**（5/9 wins）。deformation 与 velocity 高度接近（两者在 9 个
  setting 的平均 Δ 差 <0.2%），印证上阶段"velocity 与 offset 等价"的发现：更一般的
  非线性变形场并未带来超越恒定漂移的收益。
- 分析实验（§8.2）显示 deformation 学到的累计位移 |Δ|≈0.05 步，远小于一个周期
  （24 步），实际相位移动接近零。

---

## 4. 阶段3：Geometry-aware Phase Interaction（计划"四"）

实现：`phase_geo`（router attention 上的 circular bias，纯 geometry）、
`phase_graph`（`src/models/phase_graph.py`，相位槽环上 ±k 邻域消息传递，平移等变）。

**结果（MSE）：**

| setting | original | phase_geo | phase_graph |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4381 (+0.0) | 0.4400 (+0.4) |
| ETTh1 h720 | 0.4179 | -- | -- |
| ETTh2 h336 | 0.3735 | 0.3634 (−2.7) | 0.3765 (+0.8) |
| ETTh2 h720 | 0.4254 | 0.4241 (−0.3) | 0.4142 (−2.6) |
| ETTm1 h336 | 0.3585 | 0.3582 (−0.1) | 0.3613 (+0.8) |
| ETTm1 h720 | 0.4157 | 0.4157 (+0.0) | 0.4112 (−1.1) |
| Electricity h336 | 0.1661 | 0.1661 (+0.0) | 0.1647 (−0.8) |
| Electricity h720 | 0.2010 | 0.2002 (−0.4) | 0.1973 (−1.8) |
| Traffic h336 | 0.3912 | 0.3947 (+0.9) | 0.4025 (+2.9) |
| Traffic h720 | 0.4302 | 0.4348 (+1.1) | 0.4328 (+0.6) |

- phase_geo 平均 **−0.16%**（4/9），phase_graph 平均 **−0.10%**（4/9）。
- **可测量观察**：两个几何机制在相位对齐指标上是全部新模块中最好的
  （phase_graph peak shift 3.08、within3 0.667，全场最优；见 §8.3），但 MSE 收益
  零散：graph 在 ETTh2 h720 / Electricity 有 −1.8%~−2.6% 改进，却在 Traffic h336 退化
  +2.9%。

---

## 5. 阶段4：Pure Phase Forecasting Decoder（计划"五"）

实现：`src/models/phase_decoder.py`。每槽预测 (order+1)=3 个多项式系数，
`y[b,c,l,:]=Σ_m coef·t^m`，`t∈[-1,1]` 跨整个 P_out 未来轴，替换顶层 predictor；
`predictor_mlp`（capacity 匹配对照）用 MLP 顶层。

**结果（MSE）：**

| setting | original | predictor_mlp | trajectory_decoder |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4232 (−3.4) | 0.4322 (−1.4) |
| ETTh1 h720 | 0.4179 | -- | -- |
| ETTh2 h336 | 0.3735 | 0.3769 (+0.9) | 0.3715 (−0.5) |
| ETTh2 h720 | 0.4254 | 0.4177 (−1.8) | 0.4134 (−2.8) |
| ETTm1 h336 | 0.3585 | 0.3694 (+3.0) | **0.6830 (+90.5)** |
| ETTm1 h720 | 0.4157 | 0.4238 (+1.9) | **0.7142 (+71.8)** |
| Electricity h336 | 0.1661 | 0.1646 (−0.9) | **0.2094 (+26.1)** |
| Electricity h720 | 0.2010 | 0.1996 (−0.7) | **0.2527 (+25.8)** |
| Traffic h336 | 0.3912 | 0.3955 (+1.1) | **0.6237 (+59.4)** |
| Traffic h720 | 0.4302 | 0.4308 (+0.1) | -- |

- predictor_mlp 平均 **+0.03%**（4/9）：与线性顶层等价，确认该对比是公平的容量匹配
  （MLP 顶层不引入额外收益也不带来退化）。
- trajectory_decoder 平均 **+33.6%**（3/8 wins）：在 ETTh1/ETTh2 上 −0.5%~−2.8%
  小幅改进，在 ETTm1/Electricity/Traffic 上 **+26%~+90.5% 灾难性退化**。
- **可测量观察**：trajectory_decoder 的训练正常收敛（val early stop 在 15–30 ep 停止，
  非崩溃）；分析实验（§8.3–8.4）显示其输出显著更平滑（smoothness −5.4%）但相位
  峰值对齐显著变差（peak shift 3.67 vs 3.24）。即**低阶多项式归纳偏置确实在工作，
  但它用相位精度换取了轨迹平滑性，且在多频/强周期数据集上无法拟合**。
  （这是可测量的观察；"无法拟合"的具体机制——周期对齐、多项式阶数——是待验证假设。）

---

## 6. 最终模型：pure_full（计划"六"）

四模块全开 + `use_residual_head=False`（保证无 residual 分支）。

| setting | original | pure_full | Δ% |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4361 | −0.5 |
| ETTh1 h720 | 0.4179 | -- | -- |
| ETTh2 h336 | 0.3735 | 0.3828 | +2.5 |
| ETTh2 h720 | 0.4254 | **0.4074** | **−4.2** |
| ETTm1 h336 | 0.3585 | **0.6845** | **+90.9** |
| ETTm1 h720 | 0.4157 | **0.7085** | **+70.4** |
| Electricity h336 | 0.1661 | **0.2094** | **+26.1** |
| Electricity h720 | 0.2010 | **0.2518** | **+25.3** |
| Traffic h336 | 0.3912 | **0.6170** | **+57.7** |
| Traffic h720 | 0.4302 | -- | -- |

- 平均 **+33.5%**（2/8 wins）。pure_full 的误差曲线与 trajectory_decoder 几乎完全
  一致（两者在各 setting 的 MSE 差 <0.005），证实**最终模型的失败完全由 trajectory
  decoder 主导**，其余三个模块的贡献被其灾难性效应淹没。
- 唯一显著正结果：ETTh2 h720 **−4.2%**（MSE 0.4074 vs 0.4254）。在 ETTh1 h336
  （−0.5%）与 ETTh2 h336（+2.5%）之间符号不稳定，无法构成一致性结论。

---

## 7. 完整消融（计划"七"，Tables 1–4）

计划表格映射：Table 1=§2（representation）、Table 2=§3（evolution，含
velocity/deformation）、Table 3=§4（interaction）、Table 4=§5（decoder）。
逐 setting 全量数字见 §2–§6 各表（`research_runs/pure_phase_summary.csv`）。

**Table 4（Decoder，MSE）：**

| setting | original | predictor_mlp | trajectory_decoder |
|---|---|---|---|
| ETTh1 h336 | 0.4381 | 0.4232 | 0.4322 |
| ETTh2 h336 | 0.3735 | 0.3769 | 0.3715 |
| ETTh2 h720 | 0.4254 | 0.4177 | 0.4134 |
| ETTm1 h336 | 0.3585 | 0.3694 | 0.6830 |
| ETTm1 h720 | 0.4157 | 0.4238 | 0.7142 |
| Electricity h336 | 0.1661 | 0.1646 | 0.2094 |
| Electricity h720 | 0.2010 | 0.1996 | 0.2527 |
| Traffic h336 | 0.3912 | 0.3955 | 0.6237 |
| Traffic h720 | 0.4302 | 0.4308 | -- |

**消融观察汇总：**
- representation（multiscale）：+0.5% avg，负贡献或无贡献；
- evolution（deformation）：−0.1% avg，≈0；
- interaction（geo / graph）：−0.1% avg，≈0，相位对齐微改善但不转化为 MSE；
- decoder（trajectory）：+33.6% avg，强负贡献；
- **按计划 §7 的"每行去除/加入机制"口径，没有任何一个单机制能提供一致性正收益，
  trajectory decoder 是决定性负因子。**

---

## 8. 分析实验（计划"八"）

输出目录：`research_runs/pure_phase_analysis/`（4 个 CSV + `figures/`，对全部可用 run
重算 test 预测并采集模块诊断）。

### 8.1 Phase trajectory visualization
图：`figures/<setting>__phase_trajectory.png`（velocity / deformation / pure_full 的每槽
累计位移，对照 static 0 线）。
**可测量观察**：所有数据集的相位轨迹仍是**近似恒定速率**的单调漂移（如 ETTh2 h336
deformation 累计位移从槽 0 的 0.005 单调增至槽 23 的 0.112），幅度 ≤0.15 步，远小于
一个周期 24 步。deformation 并未学到明显的局部变速/非线性拐点，与 velocity 的
"恒定漂移"无实质区别。

### 8.2 Phase deformation visualization
图：`figures/<setting>__deformation_field.png`（每槽 rate / stretch−1 / 累计位移），
数据：`deformation_field.csv`。
**可测量观察**（phase_deformation，跨 setting 平均）：mean rate **−0.013**（≈0）、
mean (stretch−1) **−0.327**（即学到的拉伸因子 s≈0.67，整体**压缩**）、mean|累计位移|
**0.050**。即模型学到"时间压缩"但总位移 <0.1 步——变形场在数值上接近 inactive，
解释了与 baseline 的持平。

### 8.3 Frequency-phase consistency
数据：`frequency_phase_consistency.csv`（逐 24 步周期 argmax 圆环距离，沿用
analyze_peak_shift 口径）。跨 setting 平均：

| mode | peak shift err | within3 |
|---|---|---|
| original | 3.235 | 0.646 |
| multiscale_phase | 3.103 | 0.664 |
| phase_deformation | 3.109 | 0.663 |
| phase_geo | 3.113 | 0.662 |
| phase_graph | **3.076** | **0.667** |
| predictor_mlp | 3.140 | 0.659 |
| trajectory_decoder | 3.665 | 0.592 |
| pure_full | 3.875 | 0.565 |

**可测量观察**：四个 phase 模块（multiscale / deformation / geo / graph）均比 baseline
**改善**峰值对齐（peak shift 3.08–3.11 vs 3.24），phase_graph 最优；但 trajectory
decoder 与 pure_full **显著劣化**峰值对齐（3.67 / 3.88 vs 3.24，within3 0.565–0.592
vs 0.646）。即 geometry/representation 的相位精度收益被 decoder 完全逆转。

### 8.4 Trajectory smoothness
数据：`trajectory_smoothness.csv`（mean |y_{k+1}−y_k|，跨未来轴）。跨 setting 平均：
original 0.9446、predictor_mlp 0.9645、**trajectory_decoder 0.8940（−5.4%）**、
pure_full 0.9361、其余模块 0.95–0.96（≈baseline）。
**可测量观察**：trajectory decoder 的多项式约束确实产生全场最平滑的输出（满足计划的
"轨迹一致性内建"设计目标），但该平滑以相位峰值对齐（§8.3）和端到端 MSE 为代价——
**平滑性目标与预测精度目标在该数据集族上冲突**。

### 8.5 Multi-scale zeta observation
数据：`zeta_analysis.csv`；图：`figures/<setting>__multiscale_phase_zeta.png`。
**可测量观察**：multiscale_phase 的 zeta 门控**确实打开**（99.0% 维度 |ζ|>1e-4，
mean|ζ|=0.167，|Z_long|=0.968）；pure_full 中同样打开（99.9%，mean|ζ|=0.117）。
即长周期分支被模型实际使用，但 §2 显示它未转化为 MSE 收益——多尺度表达"被学到、
未被用出效果"。

---

## 9. 实验优先级核对（计划"九"）

| 优先级 | 实验 | 计划价值 | 实测 |
|---|---|---|---|
| 1 | Phase Deformation Field | ★★★★★ | 持平（avg −0.1%，5/9 wins），学到的变形幅度极小 |
| 2 | Multi-scale Phase Representation | ★★★★ | 微负（avg +0.5%，2/9 wins） |
| 3 | Phase Graph Interaction | ★★★★ | 持平（avg −0.1%，4/9 wins），相位对齐最优但无 MSE 收益 |
| 4 | Trajectory Decoder | ★★★ | **强负**（avg +33.6%，3/8 wins） |
| 5 | Phase Velocity | 不作为主要贡献 | 与 deformation 等价（confirm） |

**核对结论**：优先级最高的前两项（deformation、multiscale）实测均无正收益；trajectory
decoder 从"★★★ 实验"变成决定性负因子。计划的优先级排序与实测证据不匹配。

---

## 10. 论文故事 / 最终目标核对（计划"十"、"十一"）

- 论文故事主张 "Adaptive phase geometry modeling"，贡献为 dynamic deformation +
  geometry-aware interaction + trajectory decoder。
- 最终目标主张 "所有性能提升来自 phase representation / interaction / evolution /
  decoding，不依赖 residual branch"。
- **可测量核对**：本阶段 4 个机制的端到端收益（|avg ΔMSE|≤0.5% 或为负）远小于
  上阶段 residual 系列的收益（对比 `docs/PhaseFormer_next_stage_experiment_results_feedback.md`）；
  最终模型 pure_full 平均 +33.5%。**该论文叙事在当前协议与数据集族下不被数据支持。**

---

## 11. 不继续方向（建议）

基于可测量结果，以下方向不推荐作为下一步：
1. **Trajectory Decoder（低阶多项式跨整 horizon）**：平滑约束与相位精度冲突，在
   多频/强周期数据集上灾难性退化；若保留需限定在更短 horizon 或更高阶、分段多项式，
   但无证据表明可兑现论文级收益。
2. **继续堆叠 phase 表示/演化/交互模块**：multiscale/deformation/graph 的收益
   ≤±0.5% 且符号不稳定，模块被学到但无端到端效果——进一步增大表达力预计不会改变结论。
3. **纯 phase 无 residual 的"叙事优先"路线**：本阶段数据表明纯 phase 提升不足以
   覆盖 residual 的收益。

---

## 12. 数据落盘与复核

**生成文件：**
- 汇总表：`research_runs/pure_phase_summary.csv`（71 行，各 mode × setting 的
  MSE/MAE/Δ%/epochs，delta 以 original 为 base）
- 分析 CSV：`research_runs/pure_phase_analysis/frequency_phase_consistency.csv`、
  `trajectory_smoothness.csv`、`zeta_analysis.csv`、`deformation_field.csv`
- 分析图：`research_runs/pure_phase_analysis/figures/*.png`（每 setting × 5 类图：
  phase_trajectory / deformation_field / trajectory_smoothness / peak_shift_comparison /
  multiscale_phase_zeta）
- 运行日志：`research_runs/pure_phase_full_batches.log`、`research_runs/dyn_phase_full/logs/*.log`
- 每个 run：`research_runs/dyn_phase_full/dynphase_*_<mode>_*/`（metrics.csv /
  config.json / checkpoints/best.ckpt）

**复核动作：**
- 汇总表的 MSE 直接来自各 run 的 `metrics.csv`（test split），δ% 全量重算；
- 分析实验重算的 MSE/MAE 与 metrics.csv 一致（同一 best.ckpt 的 test 前向）；
- 表内缺失项（`--`）与 §0 披露的 9 个缺 run 一一对应；
- 单 seed（2021；ETTh1 h720 用 2026 以对齐既有 baseline），固定配置、无超参搜索、
  无 test-set 调参；结论为单 seed 配对比较。

**复现命令：**
```bash
bash scripts/run_pure_phase_full.sh                          # 70 run（缺失项可用 --resume 续跑补齐）
python scripts/summarize_pure_phase.py --output research_runs/pure_phase_summary.csv
python scripts/analyze_pure_phase.py --output research_runs/pure_phase_analysis
```

---

## 附录 A. 缺失 run 明细（实验中途停止）

| setting | 缺失 mode |
|---|---|
| Traffic h720 (seed2021) | trajectory_decoder、pure_full（停止时正在跑） |
| ETTh1 h720 (seed2026) | 全部 7 个 mode（batch 3 未启动） |

缺失项不参与任何平均与 wins 统计；如需补齐结论，用 `bash scripts/run_pure_phase_full.sh`
（`--resume` 会跳过已完成 run）即可续跑剩余 9 个。
