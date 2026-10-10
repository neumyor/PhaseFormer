# Theorem 5 草稿：水平不变的形状调制（M1 的理论）

记录日期：2026-10-10。状态：**推导草稿，供期刊稿第5节使用**。5(a) 已有完整证明与数值核验；
5(b)、5(c) 证明完整但依赖明确列出的理想化假设；(c) 的圆族论证在成稿前需再走一遍逐项核对。
记号沿用 [minipaper_0930](minipaper_0930.md) 第2节。

## 0. 机制回顾

设 phase latent 为 $Z\in\mathbb R^{P\times d}$（$P=24$ 个相位槽，每槽 $d$ 维），
$\bar z=\frac1P\sum_p Z_p$，$S_p=Z_p-\bar z$。M1 在共享线性预测头之前写入

$$
Z'_p = Z_p + S_p\odot q,\qquad q = BAt\in\mathbb R^d,
$$

$A\in\mathbb R^{k\times r}$、$B\in\mathbb R^{d\times k}$，$k=\min(2,r,d)$，均无 bias，$B$ 零初始化；
$t\in\mathbb R^r$ 是 rank-3 水平分支的 cycle-mean 编码。b0（$q\equiv 0$）是 M1 的精确退化。

## 1. Theorem 5(a)：周期均值不变性

**定理 5(a)。** 设预测头为共享线性 $\hat y_p = WZ_p + c$（$W\in\mathbb R^{K'\times d}$），
horizon 按完整周期展开（$H=K'P$）。则对任意 $Z$、任意 $q$：

$$
\frac1P\sum_p \big[W(Z_p+S_p\odot q)\big] = \frac1P\sum_p WZ_p,
\qquad\text{即}\qquad
\frac1P\sum_p\big[\hat y'_p-\hat y_p\big]=0\in\mathbb R^{K'} .
$$

未来每个周期的均值（对相位槽取平均，逐 $k'$）不被调制改变。

**证明.** $\sum_p S_p = P\bar z - P\bar z = 0$，故
$\sum_p S_p\odot q = (\sum_p S_p)\odot q = 0$，线性 $W$ 与槽平均可交换。$\square$

**推论（融合不变）。** `components_static` 融合的水平通路只依赖两支路的周期均值
（水平门 $g_{\rm level}$ 加权的 level 差），其门权重是静态参数。由 5(a)，phase 支路的
周期均值不变，trend 支路不经调制，故 **b0 的整个水平通路与水平包络（0930 Theorem 2(b)
的 $\mathcal I_\varphi$）在 M1 中原样成立**；0930 Theorem 3 的水平互补分析可整段继承。

**数值核验（已有）**：float64 单元测试（预测头输出逐 cycle 均值差 < 1e-10）；
六个 H720 setting 最高精度前向核验最大差 5.96e-8–1.44e-7（见
[交互记录](PhaseFormer_interaction_results_record_20261010.md)）。
边界：训练时 dropout 可破坏逐次前向恒等；联合训练会间接改变主干水平预测。

## 2. Theorem 5(b)：乘性季节性的能力构造

**生成模型。** 原始坐标 $X_{kp}=\ell_k+(1+\alpha\ell_k)s_p+e_{kp}$，$\ell$ 随机游走，
$s$ 每周期均值为零，$\alpha$ 为水平–振幅耦合强度。RevIN 窗口标准化后，条件均值为

$$
\mathbb E[\tilde y\mid \tilde x] = d(\tilde x)\,\mathbf 1_H + \big(1+\alpha'\,d(\tilde x)\big)\,s(\tilde x),
\tag{2.1}
$$

其中 $d$ 是窗口标准差单位下的水平需求（0930 式(6)），$s$ 是归一化中心形状，
$\alpha'=\alpha(\bar u+\delta_u)$ 型常数（见第3节的圆族计算；(2.1) 中与 $d$ 无关的
常数振幅部分已并入 b0 的静态 $g_{\rm shape}$）。

**假设 H1（形状由中心 latent 承载，线性内容区）。** 在 0930 Lemma 1.3 的大偏移线性区，
嵌入与 LN 对内容透明，phase 路径满足 $W S_p = s_p\cdot(\text{窗口形状幅})$，
即中心 latent 线性承载归一化形状。

**假设 H2（水平线性可读）。** $t$ 为 cycle-mean 的线性编码，存在 $a$ 使 $a^\top t=d$。
（rank-3 编码器输入即 $K=30$ 个 cycle mean，随机游走水平的需求 $d$ 是它们的线性泛函。）

**定理 5(b)。** 在 H1、H2 下，取 M1 宽度 $k=1$：$A=a^\top$，$B=\alpha'\cdot\mathbf 1_d$，则

$$
W(S_p\odot q)=\alpha' d\,W S_p=\alpha' d\, s_p,
$$

即 M1 精确实现 (2.1) 的乘性项，同时由 5(a) 不写入任何周期均值。b0 中不存在参数组合能
产生该项（见 5(c)）。$\square$

要点：乘性修正 $\alpha' d\cdot s$ 是「水平读出 × 形状」的双线性项；trend 分支对窗口是
仿射的，双线性项不在其像内；M1 通过 $q$ 依赖 $t$ 把该项写成 $W(I+\operatorname{diag}(q))S$
族——恰好包含它。

## 3. Theorem 5(c)：b0 的振幅包络

**圆族。** 固定 $u\perp s\perp\mathbf 1$（$\|\cdot\|$ 归一），窗口族
$\tilde x_\theta=\cos\theta\,u+\sin\theta\,s$，$\theta\in[0,2\pi)$。该族窗口标准差为常数
$\sigma$；cycle mean 为 $\cos\theta\,u$ 的 cycle mean，per-phase 偏差为 $\sin\theta\,s_p$。
模型的贝叶斯归一化预测在该族上为（计算见附录）

$$
d(\theta)\mathbf 1_H+\Big(\sin\theta+\frac{A}{2}\sin 2\theta\Big)s,
\qquad d(\theta)=\frac{\delta_u}{\sigma}\cos\theta,\qquad
A=\frac{\alpha(\bar u+\delta_u)}{\sigma},
\tag{3.1}
$$

其中 $\delta_u$ 是 $u$ 的水平需求，随 $u$ 的窗内水平漂移增大而无界增长。

**引理 5.1（仿射支路只有一次谐波）。** trend 分支是 $\tilde x$ 的仿射映射，故其在圆族上的
输出形如 $\cos\theta\,Pu+\sin\theta\,Ps$——只含 $\cos\theta,\sin\theta$ 一次谐波，
**不含 $\sin 2\theta$ 分量**。

**引理 5.2（phase 支路形状有界）。** 由 0930 Theorem 2(a)，每个相位槽的预测落在固定
椭球 $\mathcal E_\theta^{\rm ckpt}$ 内，故 phase 支路对 horizon 中心化形状的贡献范数
不超过与输入无关的常数 $B_\varphi$（可从 checkpoint 读出）。

**定理 5(c)。** b0 的融合形状为 $(1-g_{\rm shape})\cdot(\text{phase 形状})+g_{\rm shape}\cdot
(\text{仿射形状})$。在圆族上，(3.1) 要求 $\sin2\theta$ 方向贡献
$\frac{A}{2}\|s\|$；由引理 5.1 仿射支路提供不了该方向，故它必须全部来自 phase 支路，
受 $B_\varphi$ 约束。于是存在与参数无关的下界：当 $u$ 的水平需求满足

$$
\frac{A}{2}\|s\| > (1-g_{\rm shape})\,B_\varphi\cdot c_H
\tag{3.2}
$$

时（$c_H$ 为把槽范数换算到 horizon 中心化范数的常数），b0 在该族上的形状误差至少为
左端与右端之差。**即 b0 存在振幅包络：乘性修正的需求随窗内水平漂移线性增长，超过
checkpoint 的形状预算后不可表示。** M1 在 H1/H2 下恰好表示整族（5(b)），且不触碰
水平包络（5(a)）。$\square$

**与 0930 Theorem 2 的对称性。** Theorem 2：水平需求超出 $\mathcal I_\varphi$ 不可表示，
低秩分支补全。Theorem 5(c)：振幅（乘性形状）需求超出 $B_\varphi$ 不可表示，
中心化调制补全。两个包络、两个互补，且 M1 的补全严格不破坏前者。

**成稿前需逐项核对**：(i) 引理 5.2 中 $B_\varphi$ 的精确形式（椭球常数到
horizon 中心化范数的换算）；(ii) (3.1) 的逐步推导（含 $\mu=\cos\theta\bar u$ 项并入
$\alpha'$ 的代数）；(iii) $c_H$ 的显式值。

## 4. 经验接口（预登记）

| 预测 | 检验 | 状态 |
|---|---|---|
| 数据中普遍存在水平–振幅耦合（$\alpha\neq0$） | V2：逐数据集 \|corr\| 远高于加性零假设地板 ~0.12 | **已测**：ETTm2 0.26、ETTh2 0.34、ETTm1 0.38、ETTh1 0.41、Weather 0.43、Electricity 0.60、Traffic 0.87（train，\|corr\| 均值；乘性锚 α=0.25 为 0.75，加性零假设 0.12） |
| 耦合越强的 setting，M1 相对 b0 收益越大 | V4：按 V2 耦合分位分层 M1−b0 逐窗口收益 | 待 28-setting 训练完成 |
| M1 收益集中在高水平需求窗口 | V4：按 \|d\| 分位分层 | 同上 |
| 直接周期均值不变 | V5：28 setting float64 核验 | 随 `--stage test` 自动执行 |
| 合成乘性数据上 M1>b0，加性数据上 M1≈b0 | V1：合成训练 | 待做（GPU 空闲时） |

V2 的负向解读已预登记：若某数据集耦合接近地板（如 ETTm2 的 0.26 仍在地板 3 倍以上），
M1 在该数据集的收益应小——这与交互实验中 ETTm2 上 M1 不稳定一致，写作时如实呈现。

## 5. 与 ICLR 定理的衔接

ICLR Theorem 1 把周期模式变化建模为 $X'=XS^\top+R$；跨周期水平漂移在折叠坐标下是
对所有 phase token 的共同加性扰动，落入残差 $R$，ICLR 稳定性界随其增大而退化。
期刊定理链从 ICLR 界停止保证处接手：水平（0930 Thm 2–3）→ 振幅（本文 Thm 5）。
（成稿前对照 ICLR 附录 A.6 记号核实此段。）
