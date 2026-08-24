# PhaseFormer 下一阶段论文级实验计划（Pure Phase Modeling路线）

## 研究目标

基于前两阶段实验结果，下一阶段明确回归核心研究主线：

> 在不依赖 residual reconstruction 的情况下，提升 PhaseFormer 的纯
> phase-space forecasting 能力。

当前实验发现：

1.  Phase Velocity 相比 Phase Offset 没有明显提升；
2.  Circular Attention Bias 有稳定但有限收益；
3.  Residual reconstruction
    虽然带来性能提升，但不符合本文研究叙事，因此不作为后续优化方向；
4.  当前 phase modeling 的主要问题不是增加更多模块，而是增强 phase
    representation、phase interaction 和 phase evolution。

因此下一阶段目标：

**构建纯 phase trajectory forecasting framework。**

------------------------------------------------------------------------

# 一、核心研究假设

## Hypothesis 1：当前 phase token 表达缺少动态周期结构

当前 PhaseFormer：

    Input
    ↓
    Phase Alignment
    ↓
    Phase tokens

默认每个 token 对应固定周期位置。

但是实际周期可能存在：

-   phase shift
-   phase deformation
-   local transition
-   cycle variation

因此需要从：

    static phase representation

转向：

    adaptive phase geometry representation

------------------------------------------------------------------------

# 二、阶段1：Phase Representation Enhancement

## 目标

验证增强 phase token 表达是否可以提升纯 phase forecasting。

## 方法：Multi-scale Phase Representation

新增：

    src/models/multiscale_phase.py

结构：

    Input

    ↓

    Short-period phase

    +

    Long-period phase

    ↓

    Phase Fusion

    ↓

    Cross Phase Routing

------------------------------------------------------------------------

## 实验

  模型       phase representation
  ---------- ----------------------
  Baseline   single phase
  M1         multi-scale phase

重点：

h336 / h720。

------------------------------------------------------------------------

# 三、阶段2：Dynamic Phase Deformation Modeling

## 目标

解决当前 Phase Velocity 只能学习近似恒定漂移的问题。

当前：

    phase + velocity

没有学习：

-   nonlinear shift
-   phase stretching
-   phase compression

------------------------------------------------------------------------

## 新模块

文件：

    src/models/phase_deformation.py

结构：

    Phase token

    ↓

    Deformation Encoder

    ↓

    Phase deformation field

    ↓

    Nonlinear phase warping

    ↓

    Cross Phase Routing

------------------------------------------------------------------------

## 实验

比较：

  模型                phase evolution
  ------------------- -----------------------
  Baseline            static
  Phase Offset        linear shift
  Phase Velocity      velocity shift
  Phase Deformation   nonlinear deformation

------------------------------------------------------------------------

## 指标

增加：

-   MSE
-   MAE
-   Peak shift error
-   Phase trajectory smoothness

------------------------------------------------------------------------

# 四、阶段3：Geometry-aware Phase Interaction

## 目标

让周期几何真正参与 phase interaction。

当前 Circular Bias：

只改变 attention score。

------------------------------------------------------------------------

## 改进

Phase Relation Graph。

新增：

    src/models/phase_graph.py

结构：

    Phase tokens

    ↓

    Circular phase graph

    ↓

    Graph message passing

    ↓

    Cross Phase Routing

------------------------------------------------------------------------

## 实验

  模型                      interaction
  ------------------------- -------------
  Original routing          
  Circular Bias             
  Phase Graph Interaction   

------------------------------------------------------------------------

# 五、阶段4：Pure Phase Forecasting Decoder

## 目标

增强 phase-only prediction。

当前 decoder：

linear projection。

------------------------------------------------------------------------

## 新模块

    src/models/phase_decoder.py

结构：

    Phase latent

    ↓

    Trajectory Decoder

    ↓

    Future phase sequence

加入：

-   trajectory consistency
-   phase smoothness

------------------------------------------------------------------------

## 实验

  Decoder
  --------------------
  Linear
  MLP
  Trajectory Decoder

------------------------------------------------------------------------

# 六、最终模型结构

    Input

    ↓

    Phase Alignment

    ↓

    Multi-scale Phase Representation

    ↓

    Dynamic Phase Deformation

    ↓

    Geometry-aware Phase Interaction

    ↓

    Cross Phase Routing

    ↓

    Trajectory Decoder

    ↓

    Forecast

------------------------------------------------------------------------

# 七、完整消融实验

## Table 1：Phase representation

  Model          Multi-scale
  -------------- -------------
  Baseline       
  +Multi-scale   

## Table 2：Phase evolution

  Model          Evolution
  -------------- -----------
  Static phase   
  Offset         
  Velocity       
  Deformation    

## Table 3：Phase interaction

  Model              Geometry
  ------------------ ----------
  Original routing   
  Circular Bias      
  Phase Graph        

## Table 4：Decoder

  Decoder
  --------------------
  Linear
  MLP
  Trajectory Decoder

------------------------------------------------------------------------

# 八、分析实验

## 1. Phase trajectory visualization

展示不同周期中的 phase evolution。

## 2. Phase deformation visualization

分析模型学习：

-   shift
-   stretch
-   compression

## 3. Frequency-phase consistency

比较预测周期位置与真实周期位置。

------------------------------------------------------------------------

# 九、实验优先级

  优先级   实验                               价值
  -------- ---------------------------------- ----------------
  1        Phase Deformation Field            ★★★★★
  2        Multi-scale Phase Representation   ★★★★
  3        Phase Graph Interaction            ★★★★
  4        Trajectory Decoder                 ★★★
  5        Phase Velocity                     不作为主要贡献

------------------------------------------------------------------------

# 十、论文故事

核心问题：

Existing phase forecasting methods assume static phase coordinates.

提出：

Adaptive phase geometry modeling。

贡献：

1.  Dynamic phase deformation modeling
2.  Geometry-aware phase interaction
3.  Pure phase trajectory forecasting decoder

------------------------------------------------------------------------

# 十一、最终目标

建立：

    Static Phase Forecasting

到：

    Adaptive Phase Geometry Forecasting

的转变。

所有性能提升来自：

-   phase representation
-   phase interaction
-   phase evolution
-   phase decoding

不依赖 residual branch。
