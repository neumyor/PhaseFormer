# PhaseFormer Weak-Residual 分支增强实验方案

## 目标

本实验计划基于 PhaseFormer `weak-residual-phaseformer`
分支代码，针对当前模型中的 phase representation、phase interaction 和
residual reconstruction 机制进行系统优化。

核心研究问题：

> 当前 PhaseFormer
> 是否能够进一步通过动态相位建模提升长期时间序列预测性能？

实验采用逐阶段验证方式：

1.  建立稳定 baseline
2.  分析当前 residual 分支贡献
3.  引入动态相位校正机制
4.  引入周期几何约束
5.  引入相位特征调制
6.  完成联合实验和消融分析

------------------------------------------------------------------------

# 一、当前 PhaseFormer 架构分析

当前代码主要位于：

    src/models/PhaseFormer.py
    src/models/phase_adapters.py

整体流程：

    Input sequence

    ↓

    Phase Alignment

    ↓

    Phase tokens

    ↓

    Cross Phase Routing

    ↓

    Phase Prediction

    ↓

    Residual Reconstruction

    ↓

    Forecast

主要模块：

-   PhaseEmbedding
-   CrossPhaseRoutingLayer
-   PhasePredictor
-   WeakPeriodResidualHead
-   PhaseLocalTrendHead
-   PhaseNoiseHighFreqDamping

------------------------------------------------------------------------

# 二、研究假设

## Hypothesis 1：动态相位变化建模能够提升长期预测

当前 phase token 默认对应固定周期位置：

    phase_1
    phase_2
    ...
    phase_P

但是实际时间序列中的周期可能存在：

-   周期提前
-   周期延迟
-   周期速度变化

因此增加：

    phase position + phase offset

使模型能够学习动态相位变化。

------------------------------------------------------------------------

## Hypothesis 2：周期几何结构能够改善 phase token 表达

当前 phase token 可以看作线性排列：

    1 2 3 ... P

但是周期具有循环性质：

    P → 1

因此加入周期结构编码，使模型感知周期边界连续性。

------------------------------------------------------------------------

## Hypothesis 3：多周期特征动态调制能够增强复杂周期数据预测

真实数据通常包含多个周期：

-   日周期
-   周周期
-   季节周期

因此需要让模型根据输入动态调整不同周期成分的重要性。

------------------------------------------------------------------------

# 三、阶段0：Baseline复现

## 目的

建立可靠实验基准。

## 不修改代码

运行：

``` bash
python run_etth1.py
```

测试：

-   ETTh1
-   ETTh2
-   ETTm1
-   Electricity
-   Traffic

预测长度：

    96
    192
    336
    720

记录：

  模型                        MSE   MAE
  --------------------------- ----- -----
  PhaseFormer weak residual   \-    \-

------------------------------------------------------------------------

# 四、阶段1：验证 Residual Branch 贡献

## 目的

分析当前 residual 模块是否已经有效解决趋势误差。

## 实验

比较：

1.  完整模型
2.  去除 residual head

关闭模块：

-   WeakPeriodResidualHead
-   PhaseLocalTrendHead

------------------------------------------------------------------------

## 修改位置

    src/models/phase_adapters.py

增加：

``` python
use_residual_head=False
```

forward：

``` python
if use_residual_head:
    residual=self.residual_head(x)
else:
    residual=0
```

------------------------------------------------------------------------

## 分析

观察：

-   长预测长度下降幅度
-   不同数据集表现

如果去除 residual 后明显下降：

说明后续改进需要保留 residual 分支。

------------------------------------------------------------------------

# 五、阶段2：Dynamic Phase Correction（核心实验）

## 研究目标

验证：

动态调整 phase position 是否能够提升预测。

------------------------------------------------------------------------

# 新增模块

文件：

    src/models/phase_correction.py

结构：

    Phase token

    ↓

    Offset predictor

    ↓

    Phase offset Δφ

    ↓

    Phase transformation

    ↓

    Cross Phase Routing

------------------------------------------------------------------------

## 模块代码设计

``` python
class PhaseCorrection(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(dim,dim),
            nn.GELU(),
            nn.Linear(dim,1)
        )


    def forward(self,x):

        delta=self.net(x)

        return delta
```

------------------------------------------------------------------------

## 接入位置

修改：

    src/models/PhaseFormer.py

原：

``` python
phase_tokens=self.phase_embedding(x)

phase_tokens=self.routing(phase_tokens)
```

改：

``` python
phase_tokens=self.phase_embedding(x)

delta=self.phase_corrector(
    phase_tokens
)

phase_tokens=phase_warp(
    phase_tokens,
    delta
)

phase_tokens=self.routing(
    phase_tokens
)
```

------------------------------------------------------------------------

## 测试方案

比较：

  模型
  -----------------------------
  Baseline
  \+ Dynamic Phase Correction

重点观察：

预测长度：

    336
    720

------------------------------------------------------------------------

## 分析指标

除了：

-   MSE
-   MAE

增加：

### Peak shift error

计算：

    预测峰值位置 - 真实峰值位置

观察周期定位误差。

------------------------------------------------------------------------

# 六、阶段3：Circular Phase Geometry

## 目标

增强周期结构表达。

------------------------------------------------------------------------

新增：

    src/models/phase_geometry.py

使用：

周期 Fourier embedding。

代码：

``` python
angle=2*pi*p/P

embedding=[
sin(angle),
cos(angle)
]
```

------------------------------------------------------------------------

替换：

原：

    learnable positional embedding

改：

    circular phase embedding

------------------------------------------------------------------------

实验：

  模型
  -----------------------------
  Baseline
  \+ Dynamic Phase Correction
  \+ Circular Geometry

------------------------------------------------------------------------

# 七、阶段4：Phase Rotation Mechanism

## 目标

让相位变化能够作用到 latent feature。

------------------------------------------------------------------------

新增：

    src/models/phase_rotation.py

二维旋转：

    (x,y)

    ↓

    rotation(theta)

    ↓

    (x',y')

实现：

``` python
x1,x2=x.chunk(2,-1)

out1=x1*cos(theta)-x2*sin(theta)

out2=x1*sin(theta)+x2*cos(theta)
```

------------------------------------------------------------------------

实验：

比较：

1.  Dynamic Phase Correction
2.  Dynamic Phase Correction + Rotation

------------------------------------------------------------------------

# 八、阶段5：Harmonic Feature Modulation

## 目标

增强多周期数据建模。

------------------------------------------------------------------------

新增：

    src/models/harmonic_modulation.py

采用：

Feature modulation:

    z'=gamma*z+beta

其中：

gamma、beta

由输入周期特征生成。

------------------------------------------------------------------------

插入位置：

    Cross Phase Routing

    ↓

    Harmonic Modulation

    ↓

    Prediction

------------------------------------------------------------------------

# 九、最终模型结构

    Input

    ↓

    Phase Alignment

    ↓

    Phase Embedding

    ↓

    Dynamic Phase Correction

    ↓

    Phase Geometry Encoding

    ↓

    Phase Rotation

    ↓

    Cross Phase Routing

    ↓

    Harmonic Modulation

    ↓

    Residual Reconstruction

    ↓

    Forecast

------------------------------------------------------------------------

# 十、完整消融实验

  实验       Dynamic Correction   Geometry   Rotation   Harmonic
  ---------- -------------------- ---------- ---------- ----------
  Baseline                                              
  A          ✓                                          
  B          ✓                    ✓                     
  C          ✓                    ✓          ✓          
  D          ✓                    ✓          ✓          ✓

------------------------------------------------------------------------

# 十一、结果分析逻辑

## 情况1

Dynamic Correction 提升最大：

说明：

模型主要瓶颈是固定 phase representation。

------------------------------------------------------------------------

## 情况2

Geometry 提升明显：

说明：

周期拓扑约束有效。

------------------------------------------------------------------------

## 情况3

Harmonic Modulation 提升：

说明：

数据包含明显多周期结构。

------------------------------------------------------------------------

# 十二、实验优先级

资源有限时：

优先：

1.  Dynamic Phase Correction
2.  Phase Rotation
3.  Circular Geometry
4.  Harmonic Modulation

------------------------------------------------------------------------

# 十三、最终研究方向

最终模型核心思想：

> 从静态 phase token 建模扩展到动态 phase trajectory 建模。

主要贡献：

1.  动态相位校正
2.  周期几何增强
3.  相位特征动态调制
4.  与 residual forecasting 联合优化
