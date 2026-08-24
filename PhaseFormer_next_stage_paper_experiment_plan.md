# PhaseFormer 下一阶段论文级实验计划

## 研究目标

基于当前 weak-residual-phaseformer
分支实验结果，下一阶段研究重点从单纯增加 phase 模块转向：

**Adaptive Phase-Residual Trajectory Modeling**

核心问题：

> 如何让模型同时学习动态周期轨迹变化，以及根据数据特征自适应融合周期预测与趋势预测。

当前实验发现：

1.  Residual reconstruction 是主要性能来源；
2.  Dynamic Phase Correction 具有周期定位能力，但普遍 MSE 提升有限；
3.  Circular Geometry 有稳定但有限收益；
4.  Phase Rotation 当前实现存在负贡献；
5.  不同数据集对 phase 与 residual 的需求不同。

------------------------------------------------------------------------

# 一、研究假设

## Hypothesis 1：动态相位轨迹优于静态 phase representation

当前 PhaseFormer 使用固定 phase token：

    phase_1
    phase_2
    ...
    phase_P

但是实际周期存在：

-   周期提前
-   周期延迟
-   相位速度变化

因此提出：

    static phase
    → dynamic phase trajectory

------------------------------------------------------------------------

# 二、阶段1：Dynamic Phase Trajectory Modeling

## 目标

将当前 phase offset 建模升级为 phase velocity 建模。

新增：

    src/models/phase_velocity.py

结构：

    Phase token

    ↓

    Velocity Encoder

    ↓

    Δφ_t

    ↓

    Trajectory Integration

    ↓

    Phase Warping

    ↓

    Cross Phase Routing

由：

\[ φ'=φ+Δφ \]

升级为：

\[ φ_t=φ\_{t-1}+Δφ_t \]

------------------------------------------------------------------------

## 实验

比较：

  模型   机制
  ------ ----------------
  A0     Baseline
  A1     Phase Offset
  A2     Phase Velocity

重点测试：

-   ETTh2
-   Electricity

同时测试：

-   ETTh1
-   ETTm1
-   Traffic

指标：

-   MSE
-   MAE
-   Peak shift error

------------------------------------------------------------------------

# 三、阶段2：Geometry-aware Phase Interaction

## 目标

将周期结构从 position embedding 提升到 interaction 层。

新增：

Circular Attention Bias。

原：

\[ QK\^T \]

改：

\[ QK\^T-B\_{circle} \]

其中：

\[ B(i,j)=min(\|i-j\|,P-\|i-j\|) \]

实验：

  模型       Trajectory   Geometry Bias
  ---------- ------------ ---------------
  Baseline                
  B1         ✓            
  B2         ✓            ✓

------------------------------------------------------------------------

# 四、阶段3：Adaptive Residual Fusion

## 目标

解决 residual 固定开启导致的数据依赖问题。

当前：

    forecast = phase + residual

改为：

\[ y=y_p+lpha(x)y_r \]

新增：

    src/models/adaptive_residual_gate.py

结构：

    Phase feature

    ↓

    Gate Network

    ↓

    α

    ↓

    Phase + α Residual

------------------------------------------------------------------------

## 实验

比较：

  模型   Residual
  ------ ------------------------
  R0     无 residual
  R1     固定 residual
  R2     Adaptive residual gate

重点：

-   ETTh2
-   Electricity
-   Traffic

------------------------------------------------------------------------

# 五、最终模型

结构：

    Input

    ↓

    Phase Alignment

    ↓

    Phase Velocity Encoder

    ↓

    Circular Phase Interaction

    ↓

    Cross Phase Routing

    ↓

    Adaptive Residual Fusion

    ↓

    Forecast

------------------------------------------------------------------------

# 六、完整消融实验

## 模块消融

  Model      Velocity   Geometry   Residual Gate
  ---------- ---------- ---------- ---------------
  Baseline                         
  A          ✓                     
  B          ✓          ✓          
  C          ✓          ✓          ✓

------------------------------------------------------------------------

## Phase evolution 消融

比较：

  方法
  ----------------
  Static phase
  Offset phase
  Velocity phase

------------------------------------------------------------------------

## Residual 消融

比较：

  方式
  ----------
  None
  Fixed
  Adaptive

------------------------------------------------------------------------

# 七、分析实验

## 1. Phase trajectory visualization

展示不同周期中的 phase 演化。

比较：

Baseline：

固定轨迹

Ours：

动态轨迹。

------------------------------------------------------------------------

## 2. Residual gate visualization

展示不同数据集中的 α：

-   强趋势数据是否提高 residual 权重；
-   强周期数据是否降低 residual 权重。

------------------------------------------------------------------------

## 3. Error decomposition

分析：

phase error

-   

trend error

分别变化。

------------------------------------------------------------------------

# 八、不继续研究方向

## Phase Rotation

暂停。

原因：

已有实验显示多数 setting 负贡献。

## 单独 Harmonic Modulation

不作为主要创新。

原因：

收益主要集中于特定周期数据。

------------------------------------------------------------------------

# 九、论文贡献方向

## Contribution 1

动态 phase trajectory modeling。

解决固定 phase coordinate 问题。

## Contribution 2

Geometry-aware phase interaction。

利用周期拓扑增强 phase token 交互。

## Contribution 3

Adaptive residual fusion。

动态平衡周期预测与趋势预测。

------------------------------------------------------------------------

# 十、实验优先级

  优先级   实验                         价值
  -------- ---------------------------- -------
  1        Adaptive Residual Fusion     ★★★★★
  2        Phase Velocity Trajectory    ★★★★★
  3        Circular Phase Interaction   ★★★★
  4        Harmonic Modulation          ★★
  5        Phase Rotation               停止

------------------------------------------------------------------------

# 十一、最终研究目标

从：

    Static Phase Forecasting

发展为：

    Adaptive Phase-Residual Trajectory Forecasting

核心思想：

时间序列预测不仅需要学习周期模式，还需要学习周期轨迹如何变化，以及如何动态融合周期和趋势信息。
