# PhaseFormer 方案一：Phase-conditioned Trend Token Profile Correction

> 状态：**架构设计与理论审查，未实现，未训练，未读取 test**
> 日期：2026-10-01
> 目标：在不破坏 `minipaper_0930` Theorem 2/3 论证主线的前提下，把跨周期趋势状态显式表示为 learned trend token，并让它与现有 phase tokens 发生受限 interaction。

## 0. 决策摘要

当前最稳妥的方案不是把 trend token 直接拼入 PhaseFormer 的 phase token 序列，而是把它作为**相位主干条件下的趋势系数估计器**：

```text
历史序列
  ├─ 原始 PhaseFormer phase path ────────────────> y_phi
  └─ learned trend encoder -> 一个 trend token
                         trend token 读取 phase tokens
                                      └──────────> profile coefficients λ

固定 horizon profiles A=[a1, a2, ...]
                         └──────────> r = A λ

y = (1-g) y_phi + g (last-value anchor + r)
```

方案的关键约束有三个：

1. trend encoder 可以从完整原始历史中学习，且每个变量独立处理；
2. trend token 可以读取 phase tokens，但不能反向修改原 PhaseFormer 的 phase latent；
3. 趋势分支的 horizon 输出必须落在固定的低频 profile 子空间中。

这样做的好处是：

- 原 PhaseFormer 的 phase forecast 和末端 LayerNorm 不变，因此 Theorem 2 的 level envelope 仍适用于 phase-only path；
- learned trend token 只是 Theorem 3 中 branch input `ζ(x)` 的一种更有归纳偏置的实现；
- 趋势分支最终只输出少数 profile 系数，因此 Theorem 3 的低秩补空间结论仍然成立；
- 当趋势 token 退化为一个常数位移系数时，模型退化为当前 `weak_residual + shared` 的 PhaseFormer-L。

本设计只提出机制和验证契约，不能把预期收益写成已有结果。

## 1. 为什么需要显式 trend token

`minipaper_0930` 对原 PhaseFormer 的描述是：历史窗口经过 RevIN 后折叠为

\[
\tilde X\in\mathbb R^{K\times P},
\qquad
\tilde x_p=\tilde X e_p\in\mathbb R^K,
\]

其中 `p` 是相位位置，`K` 是周期数，`P` 是周期长度。每个 phase token 读取一个固定相位位置跨周期的轨迹；相同的跨周期 level 轨迹会以共同平移的形式进入所有 phase tokens。

原理论已经给出三层结论：

1. phase path **能够**看到部分跨周期 level；
2. 末端 `LayerNorm` 和线性 predictor 对 phase forecast 的 level 施加一个 checkpoint-specific envelope；
3. envelope 之外的 conditional residual 主要落在恒定位移、斜率和曲率等少数 horizon profiles 上，因此最优线性补偿是低秩的。

当前 PhaseFormer-L 直接使用完整时间轴的 NLinear 风格头：

\[
z=\tilde x-\tilde x_L\mathbf 1_L,
\qquad
\hat y_r=\tilde x_L\mathbf 1_H+Wz+c.
\]

这个头足以验证补空间是否存在，但它没有把“趋势状态”作为内部可解释对象。方案一的目的不是再加一个任意 residual，而是把同一补空间写成：

\[
\text{历史序列}
\longrightarrow
\text{一个 learned trend state}
\longrightarrow
\text{少数固定 horizon profiles}.
\]

因此新架构的创新点是**受约束的状态抽取与 profile 解码**，而不是单纯增大 residual 分支容量。

## 2. 形式化定义

### 2.1 记号和张量形状

对每个变量 `c`，令：

- `L = seq_len`：历史长度；
- `P = period_len`：相位长度；
- `K = ceil(L/P)`：输入周期数；
- `H = pred_len`：预测长度；
- `x_c ∈ R^L`：原始历史；
- `\\tilde x_c ∈ R^L`：使用同一 RevIN 统计量后的历史；
- `h_{c,p} ∈ R^D`：现有 PhaseFormer 的第 `p` 个 phase latent；
- `T_c ∈ R^D`：单个 learned trend token。

批量输入的主要形状为：

```text
x_enc       : (B, L, C)
x_in        : (B, L, C)          # RevIN 后
phase_series: (B, C, P, K)
Z           : (B, C, P, D)       # routing 后的 phase latent
trend_token : (B, C, D)
lambda      : (B, C, m)
profile_out : (B, H, C)
```

### 2.2 输入归一化与锚点

趋势 encoder 使用与 phase path 相同的 RevIN 统计量，不重新计算 branch-specific 的均值和方差：

\[
\tilde x_c=(x_c-\mu_c)/\sigma_c.
\]

趋势 encoder 的主输入为末值锚定轨迹：

\[
z_c=\tilde x_c-\tilde x_{c,L}\mathbf 1_L.
\]

同时保留标量 \(\tilde x_{c,L}\) 作为 parameter-free anchor。这样做有两个作用：

- `z_c` 只承担 level 相对变化、局部 slope 和曲率，避免让 encoder 重复学习恒定位移；
- anchor 提供平移等变的 level 通道，与 `minipaper_0930` 的 anchored corrector 完全兼容。

`x_mark_enc` 暂不进入主方案。若以后加入时间标记，必须单独记录为外生变量扩展，不能混入第一轮趋势 token 结论。

### 2.3 Learned trend encoder

主方案使用一个**共享权重、逐通道独立**的时间 encoder。实现时将 `(B,C,L)` reshape 成 `(B*C,1,L)`，同一个 encoder 应用于所有变量，因此没有跨通道融合。

推荐的最小结构是：

```text
z_c (B*C, 1, L)
  -> causal depthwise temporal stem
  -> GELU + LayerNorm
  -> learned-query temporal pooling
  -> Linear projection
  -> T_c (B*C, D)
```

具体定义可写成：

\[
u_t=\operatorname{GELU}(\operatorname{LN}(\operatorname{Conv}_{\rm causal}(z))_t),
\]

\[
\alpha_t
=\operatorname{softmax}_t\left(
\frac{q^\top W_q u_t}{\sqrt d}+b_{\rm rec}(t)
\right),
\qquad
T= W_T\sum_{t=1}^{L}\alpha_t W_vu_t+b_T.
\]

其中 `b_rec(t)` 是可选的固定 recency bias，第一轮建议关闭或固定为零；如果启用，只允许使用历史位置，不能使用未来信息。推荐首轮固定：单通道共享 stem、一个 pooling query、`D_trend=D`，不加入多尺度分支。

这一结构比直接取周期均值更符合“从原始时间序列学习 trend encoder”，但仍然保留近期 level 的可解释性。后续可用 encoder 的 attention 权重、输入 Jacobian 和模板投影验证它是否真的学到了近期趋势。

### 2.4 Phase path 保持不变

现有 PhaseFormer path 不做结构修改：

```text
RevIN
  -> ring padding
  -> split into periods
  -> phase_series
  -> PhaseEmbedding
  -> CrossPhaseRouting layers
  -> phase predictor
  -> phase forecast y_phi
```

代码对应 `src/models/PhaseFormer.py` 的：

- RevIN 与周期重排：`forward` 中 1855–1909 行；
- phase embedding 与 routing：1950–2014 行；
- phase forecast 重组：2040–2043 行。

trend token 不被拼进 `phase_series`，也不被拼进原有 `CrossPhaseRoutingLayer` 的 token 序列。这样原 phase path 的最终 `norm2`、predictor 和 envelope 不变。

### 2.5 受限的 phase-conditioned interaction

趋势 token 读取 routing 后的 phase latent：

\[
T'_c
=T_c+
\operatorname{CrossAttn}(Q=T_c,K=H_c,V=H_c),
\qquad
H_c=(h_{c,1},\ldots,h_{c,P}).
\]

因为只有 trend token 作为 query，phase latent 不接收反向更新，所以该 interaction 只改变趋势系数，不改变 `y_phi`。这不是削弱 interaction，而是把 interaction 的功能明确限定为：

> 在已知 phase forecast 的条件下，判断当前窗口需要补多少 level/trend residual。

实现上可以把 `(B,C,P,D)` reshape 成 `(B*C,P,D)`，把 trend token reshape 成 `(B*C,1,D)`，使用一层单头或多头 cross-attention。第一轮建议单层、无 dropout，避免把 attention 容量与趋势机制混在一起。

### 2.6 Fixed horizon profile bank

定义固定 profile 矩阵：

\[
A=[a_1,\ldots,a_m]\in\mathbb R^{H\times m}.
\]

推荐第一轮只使用：

\[
a_1=\mathbf 1_H,
\]

并将 `m=1` 作为主架构。多 profile 作为预注册扩展：

\[
a_2=J_H t,
\qquad
a_3=J_H(t\odot t),
\qquad
J_H=I_H-\frac1H\mathbf 1_H\mathbf 1_H^\top.
\]

每个 profile 先做中心化、再做 Gram–Schmidt 正交化和 `||a_j||_2^2=H` 归一化。profile 不从 test 学习；第一轮可以直接由 horizon index 构造，后续若要学习 profile，必须另立理论版本。

### 2.7 Profile coefficient decoder

趋势 token 和 phase summary 只用于预测系数：

\[
q_{\phi,c}=\operatorname{Pool}_p(h_{c,p}),
\]

\[
\lambda_c
=W_\lambda[\,T'_c;q_{\phi,c};\tilde x_{c,L}\,]+b_\lambda
\in\mathbb R^m.
\]

趋势分支为：

\[
\hat{\tilde y}_{r,c}
=\tilde x_{c,L}\mathbf 1_H+A\lambda_c.
\]

在最严格的 `m=1` 版本中：

\[
\hat{\tilde y}_{r,c}
=\tilde x_{c,L}\mathbf 1_H+a_1\lambda_{c,1}.
\]

这里 `λ` 是相对于最近值 anchor 的校正系数。它不是独立预测未来全部轨迹的 unconstrained head。

### 2.8 门控融合与反归一化

主比较版本使用当前 PhaseFormer-L 的静态逐通道门：

\[
g_c=\operatorname{sigmoid}(\gamma_c),
\]

\[
\hat{\tilde y}_c
=(1-g_c)\hat{\tilde y}_{\phi,c}
 +g_c\hat{\tilde y}_{r,c}.
\]

最后使用原 RevIN 统计量：

\[
\hat y_c=\mu_c\mathbf 1_H+\sigma_c\hat{\tilde y}_c.
\]

动态 gate 不进入第一版主结论。它可以作为后续消融，研究 phase reliability 是否能预测 envelope binding；否则静态 gate 已足以对应 Theorem 3(d)，并能和现有 `weak_residual` 结果严格比较。

### 2.9 初始化与训练

推荐初始化：

- trend stem 使用标准 Xavier/Kaiming 初始化；
- `W_lambda` 零初始化，使趋势校正初始为零；
- `g` 使用固定的 `gate_init`，主实验沿用 `weak_residual` 的 0.2，同时增加 `g=0` 的 phase-only 对照；
- 不加入趋势 token 的独立 auxiliary loss；
- phase path、trend encoder、cross-attention 和 gate 联合训练。

`gate_init` 只是优化初始化，不是方法创新；所有模型比较必须固定该规则，不能靠极端 gate 初值制造收益。

## 3. Theorem 3 一致性论证

### 3.0 Profile 的必要性与直接融合对照

`profile` 不是趋势分支能够与 phase path 融合的必要条件。若令趋势
token 经过一个任意的 horizon decoder，得到

\[
R_c=\tilde x_{c,L}\mathbf 1_H+d_c(T'_c,q_{\phi,c}),
\qquad
\hat y_c=(1-g_c)F_c+g_cR_c,
\]

那么这条路径在工程上是合法的，也可以作为一个重要的 direct-fusion
control。它仍然只使用历史窗口，因此 Theorem 3 中“branch input 是
历史函数”的前提没有被破坏；但它不再保证修正输出属于固定的
`span(A)`。若 `d_c` 是从一个 `d` 维 token 通过线性层直接产生 `H` 个
时间点，则其线性输出秩一般只有

\[
\operatorname{rank}(D)\leq\min(H,d),
\]

而不是由 profile 数 `m` 控制。若 decoder 是非线性的，样本间的有效
输出子空间还可以随输入变化，Theorem 3(a) 的固定低秩补空间结论就更
不能直接使用。Theorem 2 对原始 phase-only path 的 envelope 仍可测量，
Theorem 3(d) 对固定 branch 的 gate 风险公式也仍成立；失去的是
Theorem 3(a)/(c) 的结构性 rank 上界，而不是融合操作本身的可行性。

因此 profile 的作用是一个可检验的归纳偏置：`trend encoder` 负责从历史
估计状态系数 `lambda`，固定 profile bank 负责规定这些状态在预测区间
上可以写成什么形状。直接 horizon decoder 同时学习这两件事，容量更大，
但会更接近 dense residual，也无法把收益解释为“低秩补空间被正确实现”。
当 `m=1` 且 profile 是常数时，直接输出一个标量再广播到整个 horizon
其实就是 profile 方案的等价写法；只有输出多个自由的 horizon 坐标时，
才真正放弃了固定 profile 约束。

### 3.0.1 单变量累积实验协议

后续实验按下面的阶梯推进，每一步只引入一个结构变量，并保留前一步的
训练协议、数据、seed、gate 形式和参数预算：

1. `B0 phase-only`：原始 PhaseFormer，作为唯一主参照。
2. `B1 token+m=1`：增加 history-only 的 per-channel trend token，使用
   一个固定 constant profile；不加入 phase-token interaction。第一版 gate
   固定，随后再单独比较 learned static gate。
3. `B2 token+m=1+interaction`：只打开 trend token 读取 phase latent 的
   单向 cross-attention，其他设置与 B1 完全相同。
4. `B3 token+m=2`：只把 profile bank 从 constant 扩展为
   constant+slope，用于验证额外 profile 维度的边际价值。
5. `B4 direct-horizon control`：复用 B2 的 encoder、interaction、gate 和
   参数规模，将固定 profile decoder 替换为直接 `H` 维 horizon decoder。
   该对照专门测量 profile 约束的代价和收益，不作为 Theorem 3 低秩实现的
   主模型。
6. 只有在上述结果可归因后，才分别研究动态 gate、多个 trend token 或
   更大的 profile 数；不得在同一轮同时引入这些机制。

每个新增分支都必须同时报告 phase forecast、trend forecast、gate、profile
coefficients（或 direct decoder 输出）的诊断，并做 branch-only 与 fused
forecast 的对照。若 B4 只带来收益而 B2/B3 没有，结论应是“需要更大的
输出空间”，不能把结果表述为 profile 化 trend token 已验证。

### 3.1 Learned trend token 是合法 branch input

Theorem 3 将 branch input 写成任意由历史窗口计算得到的 \(\zeta(\tilde x)\)。本方案令：

\[
\zeta_c(\tilde x_c)=
[\,T'_c,q_{\phi,c},\tilde x_{c,L}\,].
\]

`T'` 可以包含非线性 temporal encoder 和 cross-attention，但仍然只是历史窗口的可测函数，没有使用未来信息。因此 Theorem 3 的条件残差定义：

\[
D_{\rm cond}=\tilde y-\hat{\tilde y}_{\phi}
\]

仍然适用。

### 3.2 输出秩上界

趋势修正写成：

\[
r_c=A\lambda_c(\zeta_c).
\]

无论 \(\lambda_c\) 是线性还是非线性函数，`r_c` 都属于 `span(A)`。若对 `ζ` 做最小二乘线性预测，交叉矩阵满足：

\[
\operatorname{rank}\big(\mathbb E[D_{\rm cond}\zeta^\top]\big)
\le m+\operatorname{rank}(r_\varphi\text{ remainder}).
\]

当 phase shape remainder 很小时，主导 predictive spectrum 只能有 `m` 个 profile 方向。对于主方案 `m=1`：

\[
\operatorname{rank}(W_{\rm trend})\approx1.
\]

这比“训练后观察到权重矩阵接近低秩”更强：低秩来自输出 profile 的结构约束，而不是事后截断。

### 3.3 Anchored rank-one construction

当：

\[
A=\mathbf 1_H,
\qquad
\lambda_1=w^\top\tilde x,
\]

且 \(w^\top\mathbf 1_L=1\) 时，取：

\[
\hat{\tilde y}_r
=\tilde x_L\mathbf 1_H
+\mathbf 1_H u^\top(\tilde x-\tilde x_L\mathbf 1_L),
\]

其中 `u` 是对 `w` 的等价重参数化，即可恢复 `minipaper_0930` Theorem 3(c) 的 rank-one anchored branch。方案一因此不是另一个与 PhaseFormer-L 无关的模型，而是 PhaseFormer-L 的状态化、可解释化参数化。

### 3.4 Theorem 2 的继承条件

Theorem 2 的 level envelope 依赖：

1. phase latent 最后经过原有 `norm2`；
2. phase predictor 仍是原有线性输出；
3. trend branch 不改变 phase latent；
4. envelope 只测量 `y_phi`，不把最终融合输出误称为 phase-only 输出。

方案一满足这四个条件。因此可以继续声明：

> 原 PhaseFormer phase path 仍具有原 checkpoint-specific level envelope；trend token branch 负责以低秩 profile 方式补足 phase path 的 conditional residual。

不能声明 trend token 之后“原模型的 envelope 被消除了”。被消除的是最终融合预测的 level error，而不是 phase path 本身的 envelope。

### 3.5 Gate 与 envelope binding

对固定的 phase forecast `F` 和 trend forecast `R`，融合风险关于静态 gate `g` 是二次函数：

\[
g^*
=\frac{\mathbb E\langle D_{\rm cond},R-F\rangle}
{\mathbb E\|R-F\|^2}.
\]

因此趋势 token 的实证目标不是让 gate 永远变大，而是证明：

- 当窗口 level demand 落在 phase envelope 内，trend correction 与 residual 的 alignment 较弱；
- 当窗口落在 envelope 外，trend token 产生的 profile correction 与 residual alignment 增强；
- gate、trend coefficient magnitude 和 `dist(d, I_phi)` 具有可复现关联。

## 4. 与现有实现的对应关系

第一版实现可以局部加入现有 `PhaseFormer`，不需要新建完整模型类：

1. 新增 `src/models/trend_token.py`：trend stem、temporal pooling、phase-conditioned cross-attention、profile decoder；
2. 在 `PhaseFormer.__init__` 中增加 `use_trend_token`、`trend_token_dim`、`trend_profile_count` 等配置；
3. 在 RevIN 之后计算 `trend_token`；
4. 在 routing 完成后，使用最终 `Z` 对 trend token 做 cross-attention；
5. 在 phase forecast 重组后或其前的 normalized space 中生成 `residual_forecast_normalized`；
6. 复用现有 gate 和 RevIN denormalization 逻辑。

建议的 forward 顺序：

```python
x_in, stats = revin.normalize(x_enc)
trend_token = trend_encoder(x_in, last_value=x_in[:, -1:, :])

phase_series = build_phase_series(x_in)
Z = phase_embedding_and_routing(phase_series)

phase_forecast = phase_predictor(Z)
trend_token = trend_to_phase_cross_attention(trend_token, Z)
lambda_profile = profile_decoder(trend_token, pool(Z), last_value)
trend_forecast = anchored_profile_forecast(lambda_profile, last_value)

forecast = gated_fusion(phase_forecast, trend_forecast)
forecast = revin.denormalize(forecast, stats)
```

必须避免以下实现顺序：

```python
Z = concat([phase_tokens, trend_token])
Z = original_cross_phase_router(Z)
```

这会让 trend token 直接改变 phase path 的 final-LN envelope，无法继续使用原 Theorem 2 的精确结论。

## 5. 实践可行性二次审查

### 5.1 参数和计算量

对于 `L=720`、`P=24`、`D=8` 的典型设置，新增部分主要是：

- temporal stem：与 `D` 和卷积核相关，远小于 `L×H` 的 dense NLinear head；
- 一个 trend token 的 cross-attention：输入长度只有 `P`，复杂度为 `O(PD)`；
- profile decoder：`O(mD)`；
- profile bank：固定 buffer，不计可训练参数。

它不需要为每个 horizon 学习一个 `L×H` 稠密矩阵，因此应显著小于 dense PhaseFormer-L。精确参数量必须由实现后的 `parameter_count` 统计，不在设计阶段预先声称具体百分比。

### 5.2 优化稳定性

风险主要有四个：

1. **趋势 token 学到周期形状而非趋势。** 通过末值锚定、低频 profile 输出和输入 Jacobian 诊断控制；
2. **phase branch 与 trend branch 争抢 level。** 通过固定 profile、静态 gate 和 phase-only envelope 分解观察是否发生 level handoff；
3. **cross-attention 过拟合 phase slots。** 首轮只用一层、无 dropout、无多层 trend memory；
4. **profile basis 太强或太弱。** 先固定 `m=1`，只有在 predictive spectrum 支持时再进入 `m=2/3`。

零初始化 `W_lambda` 可以让校正项从零开始，避免第一步训练破坏原 phase forecast。若 branch 输出含 `last_value` anchor，则应同时报告 `g=0` 和固定 `g_init` 的控制，不能把初始化差异误当机制收益。

### 5.3 数据和信息泄漏

方案一只使用 `x_enc` 和已存在的 phase latent：

- 不读取 `y`；
- 不使用 decoder future marks；
- temporal pooling 只访问历史窗口；
- profile bank 只由 horizon index 构造；
- 不使用 test 统计量选择 encoder 或 profile 数量。

因此它可以沿用当前 PhaseFormer 的训练、验证和 best-checkpoint 协议。

### 5.4 与当前 PhaseFormer-L 的可比性

至少需要保留以下四个配置：

| 配置 | 含义 |
|---|---|
| `phase_only` | `use_residual_head=False`，原始相位主干 |
| `L_dense` | 当前 `weak_residual + shared`，完整时间轴线性头 |
| `L_trend_token_m1` | 本文方案一，单 trend token、一个 constant profile |
| `L_trend_token_m2` | 本文方案一的扩展，constant + centered slope |

`L_dense` 是容量较大的 probe，`L_trend_token_m1` 是受结构约束的 candidate。两者必须联合训练，不能冻结 phase path 后再训练趋势 token，否则无法检验 Theorem 3(c) 的 conditional complement。

## 6. 论证与实验计划

### 阶段 A：静态和退化关系验证

不训练，先验证：

1. `trend_token=0` 和 `g=0` 时输出逐元素等于 phase-only；
2. `m=1`、关闭 phase-conditioned attention 时，输出形态可映射到当前 PhaseFormer-L；
3. trend branch 输出始终位于 `span(A)`；
4. 不同 batch、不同通道和不完整 period padding 下形状正确；
5. phase path 的参数和输出在 trend flag off 时保持 bitwise 或数值等价。

### 阶段 B：合成数据理论检验

使用 `minipaper_0930` 的生成模型：

\[
X_{kp}=s_p+\ell_k+e_{kp},
\qquad
\ell_k=\ell_{k-1}+\eta_k.
\]

依次加入：

- persistent level step；
- random-walk level；
- local linear level trend；
- curvature；
- phase jitter；
- shape-only variation。

要验证的不是只有 MSE，而是：

- learned trend token 与真实 level state 的相关；
- `λ_1` 与未来 level demand 的相关；
- profile output 的常值/斜率对齐；
- predictive spectrum 是否集中在 `m` 个方向；
- `g` 是否随 envelope violation 增强；
- trend token 删除是否只损害 level error，而不显著损害 centered shape error。

### 阶段 C：小规模真实数据验证

先使用 validation-only 协议选择 encoder 的最小容量，不读取 test。建议设置：

- ETTh2-96、ETTh2-720：验证 envelope 外 level correction；
- ETTm2-192：验证高频采样下的趋势 token；
- ETTh1-192：作为较弱趋势信号的负对照。

每个 setting 至少包含三 seed，并与 `phase_only`、`L_dense` 配对。第一轮不搜索 profile 形状，不按数据集切换结构，不使用 dataset-aware gate。

### 阶段 D：机制解剖

对冻结 checkpoint 做：

1. trend token 输入 Jacobian 的时间位置质量；
2. temporal pooling 权重的 recency profile；
3. `λ` 与 level/slope/curvature 模板的对齐；
4. 删除 trend token、删除 phase-conditioned attention、删除 profile 分量；
5. 同维度随机 token 和 PCA token 对照；
6. branch-only loss 与 fused loss 同时报告。

如果删除趋势模式后 branch 自身误差下降但 fused error 上升，这将直接复现 `minipaper_0930` 对 conditional complement 的关键预测。

### 阶段 E：扩展到完整矩阵

只有在阶段 B–D 支持机制假设后，才扩展到完整数据集和 horizon 矩阵。正式结果应沿用仓库的：

- lowest-validation-loss checkpoint；
- 固定 seed；
- Golden 与 matched rerun 分开；
- test-set selection 单独披露；
- 对高误差和显著退化样本做可审计分析。

## 7. 预注册判定标准

以下标准用于判断“趋势 token 是否真的实现了理论补空间”，而不是只判断某个配置的 test MSE 是否下降。

### 理论通过条件

1. `g=0` 和 trend flag off 精确恢复 phase-only；
2. trend output 的数值 rank 不超过 `m`；
3. profile coefficient 的输入和输出语义在至少两个 seed 中稳定；
4. phase-conditioned token 的 fused contribution 高于同维度随机 token 对照；
5. 原 phase-only path 的 envelope 常数和边界性质不变。

### 实践支持条件

1. 相比 phase-only，至少在预注册的趋势不稳定 setting 上降低 level error；
2. 相比 `L_dense`，参数和计算量明显降低，且性能差距在预先设定容忍区间内；
3. 不在 ETTh1/ETTm1 等负对照上产生系统性过度修正；
4. 跨 seed 的趋势 token 语义和 gate 行为可重复；
5. 提升不能只来自 gate 初始化、训练轮数或数据集专用配置。

### 失败判定

出现以下任一情况时，不应继续声称“trend token 实现了 Theorem 3 补空间”：

- trend token 主要编码周期形状，且 profile 系数与 level demand 无关；
- 多 token 只有在取消 profile 约束后才有效；
- phase path 被 trend token 修改后 envelope 结论不再成立；
- 只有 test-set selection 后的单个 setting 改善；
- `L_trend_token_m1` 不能复现或接近当前 L 的 rank-one 行为。

## 8. 需要避免的理论和实践误读

1. **trend token 不是新的独立预测器。** 它是相位主干条件下的 residual state。
2. **单 token 不等于模型只能表达一个时间方向。** 它输出一个 profile 系数，但 phase-conditioned encoder 可以是非线性的；理论限制来自 horizon output geometry。
3. **低秩不是事后 SVD 压缩。** 本方案从 decoder 结构上保证低秩。
4. **原 phase path 仍然可能使用 level。** trend token 不是证明原模型完全看不到 level，而是显式承接其 envelope 外的部分。
5. **不要把 learned profile 与固定 profile 混在第一轮。** 一旦 profile 也自由学习，Theorem 3 的可解释输出空间需要重新定义。

## 9. 二次可行性结论

### 理论结论

方案一在以下意义下可行：

- learned trend encoder 是合法的历史函数输入；
- profile-constrained decoder 保留 Theorem 3 的 rank-`m` 补空间结构；
- `m=1` 时存在与现有 anchored PhaseFormer-L 等价的 rank-one construction；
- phase path 不被修改，因此原 Theorem 2 的 envelope 仍可对 phase-only forecast 逐 checkpoint 测量；
- gate 的风险表达式仍是 Theorem 3(d) 的静态 gate 形式。

它不能支持的更强表述是：trend token 让原 PhaseFormer 本身突破了 envelope。正确表述应是：

> 趋势 token 在保持原相位主干约束的同时，学习一个以相位预测为条件的低维状态，并通过固定 horizon profiles 补足 phase path 的 conditional residual。

### 实践结论

方案一也具有工程可行性：

- 只需新增一个局部 trend-token 模块和一个 profile decoder；
- 不需要改数据格式或训练入口；
- 可复用现有 RevIN、gate、best-checkpoint 和指标协议；
- 计算量应低于 dense NLinear 分支；
- `m=1` 可以作为极小改动的最小实现，`m=2/3` 可以作为可解释扩展；
- 通过 `g=0`、phase-only 和 dense L 三个控制可以清楚归因。

因此，下一步应先实现 `m=1`、单 token、phase-to-trend 单向 cross-attention、固定常值 profile 的最小版本。只有当它在合成数据和 validation-only 机制指标上复现 level readout、profile alignment 和 conditional contribution 后，才应进入多 profile 或双向 co-attention 扩展。
