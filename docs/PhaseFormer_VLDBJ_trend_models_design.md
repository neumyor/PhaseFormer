# 三种模型的趋势分支设计，以及移植到 PhaseFormer 的方式

本文记录 N-BEATS、ETSFormer、BasisFormer 是否专门设计了趋势分支、各自怎么设计，
以及我们在原版 PhaseFormer 旁边仿造出来的三个分支。实验结果见
`docs/PhaseFormer_VLDBJ_trend_branches.md`。

## 1. N-BEATS（Oreshkin et al., ICLR 2020，arXiv 1905.10437）

**有专门趋势分支。** 可解释版本的 N-BEATS 把网络拆成 trend stack 和 seasonality stack。

- **块结构**：每个 block 先用若干层全连接把输入窗口变成两组系数 θ^f、θ^b，再各自通过
  一个基函数层得到部分预测 ŷ 和 backcast x̂：ŷ = Σ_i θ^f_i v^f_i。
- **趋势基**：趋势块把 v^f 固定成多项式基，t = [0, 1, …, H−1] / H，
  ŷ = Σ_{i=0}^{p} θ^f_i t^i，论文的设置为 **p = 2**（二次多项式），趋势因此被约束成缓慢变化、
  近似单调的曲线。这是它唯一的"平滑/外推约束"。
- **季节基**：季节块用 Fourier 基，谐波数约 H/2，另加常数项。
- **双重残差堆叠**：x_ℓ = x_{ℓ−1} − x̂_{ℓ−1}，ŷ = Σ_ℓ ŷ_ℓ。每块只解释自己能解释的部分，
  剩下的交给后面的块。趋势堆叠在前，季节堆叠在后，因此两部分的输出可以分开画出来。

要点：趋势分支本身很小（一个线性映射到 p+1 个系数），表达力来自 MLP 和堆叠。

## 2. ETSFormer（Woo et al., ICLR 2022，arXiv 2202.01381）

**有专门的趋势分支，而且把它拆成 level 和 growth 两部分。**

- **分解**：每条序列被分解为 level E、growth B、seasonality S，直接对应 Holt-Winters。
- **Level**：E_t = α⊙(E_{t−1} − Linear(S_t)) + (1−α)⊙(E_{t−1} + Linear(B_{t−1}))，
  α 是每个 head 一个的可学习平滑系数。
- **Growth**：先用指数平滑注意力（ESA）对一阶差分做平滑，
  A_ES(V)_t = αV_t + (1−α)A_ES(V)_{t−1}，权重只与时间间隔有关、与内容无关，
  因此可以写成下三角矩阵并用 FFT 做到 O(L log L)。多头 ESA 的输出线性映射后即为 growth。
- **Seasonality**：频率注意力 FEA，对输入做 DFT，取振幅最大的 K 个频率（排除 k=1），
  用时域余弦形式重建，并可直接外推过预测窗口。
- **组合**：x̂ = Repeat(E_T) + Σ_i γ^i B_T + S，γ 是每个 head 的可学习阻尼系数。
  level 与 damped growth 合起来就是趋势。

要点：ETSFormer 的趋势是**显式的参数化指数平滑**（α、β、γ 都可学），不需要额外的趋势网络；
去掉 level 组件时精度下降最大。

## 3. BasisFormer（Ni et al., NeurIPS 2023，arXiv 2310.20496）

**没有专门的趋势分支。** 它做的是"可学习基 + 注意力系数"，趋势只是基向量可以吸收的成分之一。

- **基**：一组 N 条 (I+O) 长度的基向量 z，由窗口起始时间戳 τ = t/T 经四层 MLP（带跳连）生成，
  历史段和未来段对齐。基向量上加二阶差分平滑惩罚 L_smooth = ‖zS‖²，常数和线性斜坡不受罚，
  因此基可以吸收整体 level 与 trend。
- **系数**：Coef 模块用双向交叉注意力（BCAB）让序列与基互相 attend，
  再取内积得到 c[i,j,h]；这是序列与基之间的二部图相似度。
- **对比学习**：训练时把历史视角与未来视角的系数做 InfoNCE 对齐（L_align），推理时只用历史视角。
- **预测**：Forecast 模块把未来段基向量投影并分 head，
  ỹ[i,h] = Σ_j c[i,j,h]·z̃_y[j,h,:]，再用一个瓶颈 MLP 混合各 head。

要点：BasisFormer 不区分趋势/周期，它是"给预测找一组可解释的基"；趋势能力来自平滑基与系数学习，
不是独立的趋势分支。

## 4. 移植到 PhaseFormer 的三个分支

原版 PhaseFormer 已经是"相位分支 + 确定性锚点"的结构，本身没有显式趋势外推。
我们在其输出上加一个零初始化的 delta：`y_hat = PhaseFormer(x) + branch(x)`。三个分支各自仿造上面的设计：

| 分支 | 仿造对象 | 保留的设计 | 为了适配做的简化 |
|---|---|---|---|
| `PF+NB` | N-BEATS trend stack | 二次多项式基、MLP 出系数、backcast 从残差中减掉、两块堆叠 | 输入取历史减去末值后池化到 30 个周期均值（N-BEATS 原本直接吃整窗） |
| `PF+ETS` | ETSFormer level+growth | 可学习 α、β 的 Holt 递推、可学习阻尼 γ、level/growth 显式分离 | 递推在 30 个周期均值上做（原本每步做），季节部分交给 PhaseFormer 的相位分支 |
| `PF+Basis` | BasisFormer basis+Coef | 可学习平滑基、交叉注意力求系数、按系数加权组合基、二阶差分平滑惩罚 | 去掉对比学习与时间戳条件（需要未来视角，且对单分支实验是训练技巧）；基用低频余弦初始化 |

三个分支的参数量（以 Weather-96 为例）：

- `PF+NB`：16,578（两块，隐层 64）
- `PF+ETS`：84（α、β、γ 与门控各 21 个通道参数）
- `PF+Basis`：3,925（12 条基 + 64 维注意力）

三者在初始化时输出严格为 0（`nb` 靠零初始化输出层，`ets`/`basis` 靠零初始化逐通道门控），
因此每个 run 的第一步与 PhaseFormer 完全相同，之后分支按梯度自行打开。

## 5. 评估口径

- 总体精度：测试集 MSE / MAE，12 个 setting（6 数据集 × 96/192），seed 2021，E14 协议。
- 趋势成分：把预测误差按周期折叠，取每个周期一个水平值（cycle-level），
  得到趋势成分 MSE；同一套定义也用于周期成分。
- 对照：原版 PhaseFormer（同协议同 seed）、B0H+（调过参的相位 + level）。
  这样既能看分支相对 PhaseFormer 的增益，也能看它与我们已调好的 level 分支的差距。

## 参考

- Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio. N-BEATS: Neural basis expansion
  analysis for interpretable time series forecasting. ICLR 2020. arXiv:1905.10437.
- Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, Steven Hoi. ETSformer: Exponential Smoothing
  Transformers for Time-series Forecasting. arXiv:2202.01381.
- Zelin Ni, Hang Yu, Shizhan Liu, Jianguo Li, Weiyao Lin. BasisFormer: Attention-based Time Series
  Forecasting with Learnable and Interpretable Basis. NeurIPS 2023. arXiv:2310.20496.
