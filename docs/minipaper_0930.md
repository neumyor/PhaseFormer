# PhaseFormer-L: Conditional Level Adaptation Beyond Phase Tokenization

*Draft of 2026-09-30. Compared with the 0926 draft, Section 2 has been replaced by
a formal analysis of the original PhaseFormer: a capability theorem, an
impossibility theorem, and a low-rank complement theorem (Sections 2–5). The
empirical sections are unchanged in substance and have been re-indexed against
the new theory.*

## Abstract

PhaseFormer represents recurring temporal structure by grouping observations at the
same phase across cycles. This representation is compact, but its success on
within-cycle shape changes does not settle how it handles cross-cycle level
changes. We give three results for the original architecture: RevIN, phase
folding, a LayerNorm-ed phase embedding, router-based cross-phase attention, and
a linear phase predictor.

First, a *capability theorem*. Every level shift shared by the phase tokens
passes exactly through router attention. The original design, with only two
content coordinates, can approximate the Bayes forecast for a seasonal shape
plus a random-walk cycle level to arbitrary uniform accuracy. That forecast
includes adaptive, recency-weighted level tracking pooled across phases.
Second, an *envelope theorem*. Because the last operation before the linear
predictor is a LayerNorm, the level of every PhaseFormer forecast, measured in
window standard deviations from the window mean, lies inside a
checkpoint-specific interval of half-width $C_\varphi$. This holds for any
input, any attention pattern, and any number of layers. A level departure $d$
beyond the interval costs at least $(|d|-C_\varphi)^2$ in normalized MSE.
Under instance normalization, such departures are scale-free: a persistent
level step that began $m$ samples before the forecast origin demands
$d=\sqrt{(L-m)/m}$, whatever its size. Third, a *complement theorem*. The
part of the level that the phase path cannot follow is a scalar field written
along a few fixed horizon profiles. Its least-squares linear predictor
therefore has rank at most the number of profiles, however nonlinear that
field is. Under a Gaussian design the readout is the Bayes recency kernel,
scaled by the probability of leaving the envelope. In joint training, an
anchored rank-one branch can take over the level exactly and leave the
backbone to carry the shape.

We then test these predictions with a deliberately unstructured anchored linear
branch. Across 28 train/validation settings, its predictive spectrum has a
leading-mode share of 0.642--0.862 and needs only 2--7 directions for 90% of
the attainable linear reduction. All 18 analyzed trained-head groups read a
recent weighted level. Twelve of them write a horizon-wide displacement; the
Weather groups write a smooth tilt or curvature. Three to five canonical modes
recover at least 95% of each of 72 frozen checkpoints' own improvement.
Deleting these semantic modes damages the fused forecast more than
same-dimensional random controls do, even when the deletion improves the
branch in isolation. The resulting PhaseFormer-L improves both MSE and MAE on
all 12 settings of ETTh2, ETTm2, and Weather in a three-seed matched
comparison, with MSE reductions of up to 8.73%. Its correction head compresses
to 3.5--6.1% of its dense parameters while retaining 92.4--101.9% of the
attainable branch value.

## 1. Introduction

Periodic signals vary along at least two related axes. Their shape within a
cycle may change, and the level around which that shape evolves may move from
one cycle to the next. PhaseFormer focuses on the first axis by collecting values
with the same phase across cycles. This makes recurring shape changes easy to
represent and explains why a small phase-domain backbone can forecast long
horizons efficiently.

For a journal extension, the useful question is sharper than whether
PhaseFormer can model level changes at all. We answer it with three theorems
about the original architecture, each proved from the actual computation graph
rather than from an idealized subspace model:

1. **Capability (Theorem 1).** The phase path is structurally able to model
   cross-cycle level variation. A common level shift reaches every phase token
   and passes exactly through router attention. The architecture can realize
   the Bayes forecast of a seasonal-shape plus random-walk-level model to any
   uniform accuracy, with content rank two.
2. **Envelope (Theorem 2).** The capability is bounded. The final LayerNorm
   places every forecast level inside a checkpoint-specific interval, measured
   in window standard deviations. Level departures beyond it cannot be
   modeled, and they incur an explicit error floor. Because RevIN makes the
   demand scale-free, recency rather than magnitude decides whether a level
   change falls outside the interval.
3. **Complement (Theorem 3).** What the phase path cannot follow is exactly
   what a low-rank linear branch represents. The unmodeled level is a scalar
   written along fixed horizon profiles, so the best linear corrector has rank
   at most the number of profiles. Its readout is a recency kernel, and an
   anchored rank-one branch can carry the level exactly when trained jointly.

The two positive statements (1 and 3) and the negative one (2) are compatible:
PhaseFormer models level inside its envelope, and the linear branch is the
minimal object that removes the envelope. The experiments then test
quantitative consequences of Theorem 3. A full-history linear branch is used as
a high-capacity probe; its reduced-rank spectrum measures how many directions
buy error reduction. Canonical modes of jointly trained heads identify what the
branch reads and writes. Mode-level interventions test whether those
directions matter to the fused forecast, not only to the branch.

The central claim is therefore:

> PhaseFormer can model cross-cycle level variation, but only within a bounded,
> scale-free envelope fixed by its final normalization. The complement beyond
> this envelope has a fixed output geometry, so it is low-rank. Its dominant
> mechanism is a recent weighted level readout followed by a persistent or
> smooth low-frequency correction over the horizon.

We keep three notions of dimension distinct: the *output rank* in Theorem 3
(the number of horizon profiles), the *predictive rank* (the number of
reduced-rank modes that buy most of the attainable linear reduction), and the
*functional rank* (the number of canonical modes that account for a trained
model's own improvement). The theorem bounds the first. The experiments measure
the second and third.

## 2. Setup: the original phase path as an operator

### 2.1 Normalization, folding, and the computation graph

Let $x\in\mathbb R^L$ be the lookback window of one channel, $L=KP$, with $P$ the
period length and $K$ the number of cycles. The main configuration has $L=720$,
$P=24$, and $K=30$. RevIN with `affine=False` computes

\[
\mu=\tfrac1L\mathbf 1_L^\top x,\qquad
\sigma=\big(\tfrac1L\|x-\mu\mathbf 1_L\|^2+\epsilon_r\big)^{1/2},\qquad
\tilde x=\frac{x-\mu\mathbf 1_L}{\sigma},
\qquad
\mathbf 1_L^\top\tilde x=0,\quad \|\tilde x\|^2\le L .
\tag{1}
\]

The window is folded into $\tilde X\in\mathbb R^{K\times P}$ with
$\tilde X_{kp}=\tilde x_{(k-1)P+p}$. The *phase token* $p$ is the column
$\tilde x_p=\tilde Xe_p\in\mathbb R^K$, and the *cycle-mean vector* is

\[
\bar r=\tfrac1P\tilde X\mathbf 1_P\in\mathbb R^K,
\qquad \mathbf 1_K^\top\bar r=0,\qquad \|\bar r\|^2\le K .
\tag{2}
\]

All LayerNorms act on $\mathbb R^D$ as

\[
\operatorname{LN}(u)=\gamma\odot\frac{Ju}{\sqrt{\|Ju\|^2/D+\epsilon}}+\beta,
\qquad J=I_D-\tfrac1D\mathbf 1_D\mathbf 1_D^\top .
\tag{3}
\]

One routing layer maps tokens $h_1,\dots,h_P\in\mathbb R^D$ as follows. With
learned routers $\rho_1,\dots,\rho_R$ and optional positional embeddings
$\pi_p$,

\[
\begin{aligned}
u_p&=h_p+\pi_p,&
B_j&=\operatorname{Attn}_s(\rho_j;\,u_{1:P}),\\
m_p&=\operatorname{Attn}_r(u_p;\,B_{1:R}),&
o_p&=\operatorname{LN}_1(u_p+m_p),\qquad
h_p^{+}=\operatorname{LN}_2\big(o_p+\operatorname{MLP}(o_p)\big).
\end{aligned}
\tag{4}
\]

Here $\operatorname{Attn}(q;s_{1:n})=W_o\operatorname{concat}_{\text{heads}}\big(\sum_i
\alpha_i(W_vs_i+c_v)\big)+c_o$. The weights $\alpha_i$ are a softmax over
$i$ of scaled scores $\langle W_qq+c_q,W_ks_i+c_k\rangle$, computed per head.
The input to the first layer is the embedding
$h_p^{(0)}=\operatorname{LN}_e(W_e\tilde x_p+c_e)$ with $W_e\in\mathbb R^{D\times K}$.
Middle units add an in-projection of the previous unit's out-projection before
routing. None of the results below depends on that detail, except through the
fact that *every unit ends with $\operatorname{LN}_2$*. The phase predictor and
the de-normalization are

\[
\hat{\tilde y}_p=W_dh_p^{(N)}+c_d\in\mathbb R^{K'},
\qquad
\hat y=\mu\mathbf 1_H+\sigma\,\hat{\tilde y},
\tag{5}
\]

where $\hat{\tilde y}\in\mathbb R^H$ unfolds the columns
$\hat{\tilde y}_1,\dots,\hat{\tilde y}_P$ cycle by cycle, and $H=K'P$. All
four horizons satisfy $P\mid H$. The code confirms this graph
(`CrossPhaseRoutingLayer`, `PhaseEmbedding`, `PhasePredictor`, RevIN with
population variance). In a multi-unit stack, the residual
$z_{\text{prev}}+\text{in\_proj}(\cdot)$ is added *before* routing, so the last
operation of every unit is $\operatorname{LN}_2$. For the three datasets studied
in Section 7 (ETTh2, ETTm2, Weather), the baseline presets use positional
embeddings, one head, $R=8$ routers, and $D=8$. The exception is ETTh2-720,
which uses $R=D=4$. Other datasets use wider presets; the theorems require only
$D\ge4$.

### 2.2 Level demand and the level part of the error

Let $y\in\mathbb R^H$ be the future and $\tilde y=(y-\mu\mathbf 1_H)/\sigma$ its
normalization in the *window's* coordinates. We define the *level demand* of a
window and the *level* of any horizon vector as

\[
d=\operatorname{lev}(\tilde y)=\frac{m_y-\mu}{\sigma},
\qquad
\operatorname{lev}(v)=\tfrac1H\mathbf 1_H^\top v ,
\tag{6}
\]

where $m_y$ is the future mean. The demand $d$ is the distance of the future
level from the window mean, measured in window standard deviations. For any
forecast, the per-step error splits exactly into a level part and a centered
part:

\[
\tfrac1H\|\hat{\tilde y}-\tilde y\|^2
=\big(\operatorname{lev}(\hat{\tilde y})-d\big)^2
+\tfrac1H\|J_H(\hat{\tilde y}-\tilde y)\|^2 .
\tag{7}
\]

Theorems 1 and 3 use a stylized generative model that separates recurring shape
from cycle level:

\[
X_{kp}=s_p+\ell_k+e_{kp},\qquad
\ell_k=\ell_{k-1}+\eta_k,\qquad
e_{kp}\sim\mathcal N(0,\sigma_e^2),\ \eta_k\sim\mathcal N(0,\sigma_\eta^2),
\tag{8}
\]

with flat priors on the seasonal profile $s\in\mathbb R^P$ and on $\ell_1$. In
folded coordinates a level trajectory adds the *same* vector $\ell\in\mathbb
R^K$ to every phase token, $x_p=s_p\mathbf 1_K+\ell+e_{\cdot p}$. The whole
analysis follows this common-shift structure.

## 3. Theorem 1: the phase path can model level variation

### 3.1 The target forecaster

For a weight vector $b\in\mathbb R^K$ with $\mathbf 1_K^\top b=1$ and
$\|b\|\le1$, define the *pooled level-adaptive seasonal forecaster*

\[
G_b(\tilde X)_p=\mathbf 1_{K'}\Big[\tfrac1K\mathbf 1_K^\top\tilde x_p
+\big(b-\tfrac1K\mathbf 1_K\big)^\top\bar r\Big],
\qquad p=1,\dots,P .
\tag{9}
\]

The first term is the phase's long-run value. The second is the recency-weighted
level of the cycle means relative to their long-run average. Its forecast level
is $\operatorname{lev}(G_b)=b^\top\bar r$, because $\mathbf 1^\top\bar r=0$.

Under (8), the Bayes weights are those of the Kalman forecast for a local level
model on the cycle means, with signal-to-noise ratio $q=P\sigma_\eta^2/\sigma_e^2$
and a diffuse initial level. They are nonnegative and sum to one. Away from the
first cycles they follow the steady-state exponential form

\[
b_k\approx\kappa(1-\kappa)^{K-k},\qquad
\kappa=\tfrac12\big(-q+\sqrt{q^2+4q}\big).
\tag{10}
\]

**Lemma 1.1 (Bayes optimality and RevIN consistency).** Under (8), $G_b$ with
the Kalman weights is the posterior-mean forecast of every future cycle. It is
linear and satisfies $G_b(\alpha X+c\mathbf 1\mathbf 1^\top)=\alpha G_b(X)+c\mathbf 1\mathbf 1^\top$.
Forecasting in RevIN coordinates and de-normalizing therefore reproduces $G_b$
on the raw window.

*Proof.* Write $r_k=\frac1P\sum_pX_{kp}=\bar s+\ell_k+\bar e_k$ and the phase
deviations $X_{kp}-r_k=(s_p-\bar s)+(e_{kp}-\bar e_k)$. For Gaussian noise, the
deviations are independent of $(\bar e_k)$ and do not involve $\ell$, so the
likelihood factorizes. The deviations identify $s_p-\bar s$, with posterior
mean $\frac1K\sum_k(X_{kp}-r_k)$. The cycle means form a local level model with
level $\bar s+\ell_k$ and observation variance $\sigma_e^2/P$. Its posterior
mean at every future cycle is the flat Kalman forecast $b^\top r$. Adding the
two posterior means gives (9) on raw data. Linearity is immediate, and
$(b-\frac1K\mathbf 1)^\top\mathbf 1=0$ gives the shift property. $\square$

### 3.2 Two exact structural facts

**Lemma 1.2 (common-shift pass-through).** Suppose every input token of a routing
layer is shifted by the same vector, $u_p\mapsto u_p+v$. Then
(i) every sender attention weight is unchanged. (ii) Every router buffer shifts
exactly by $V_sv$. (iii) Every receiver output shifts exactly by $V_rV_sv$ plus
the effect of the receiver's re-weighting of unshifted buffer contents. Here
$V_s=W_o^sW_v^s$ and $V_r=W_o^rW_v^r$:

\[
\alpha_{jp}(u+v\mathbf 1^\top)=\alpha_{jp}(u),\qquad
B_j(u+v\mathbf 1^\top)=B_j(u)+V_sv,\qquad
m_p(u+v\mathbf 1^\top)=\sum_j\beta'_{pj}\,(V_rB_j+\text{bias})+V_rV_sv .
\tag{11}
\]

*Proof.* The sender score of router $j$ and token $p$ changes by
$\langle W_q\rho_j+c_q,W_kv\rangle$ in every head. This term does not depend on
$p$ and cancels in the softmax over $p$, which gives (i). The buffer is a convex
combination of $W_v(u_p+v)+c_v$ followed by $W_o$, which gives (ii). The
receiver's weights $\beta'$ may change, but they are convex. Every value is
shifted by the same vector $W_v^rV_sv$, so the shift passes through $W_o^r$
unchanged, which gives (iii). $\square$

In folded coordinates, a level trajectory $\ell$ contributes the common
pre-activation shift $W_e\ell$ to every token (Section 2.2). Lemma 1.2 shows
that router attention neither hides nor distorts such a shift; only the
LayerNorms and the MLP act nonlinearly on it. The next lemma shows that the
LayerNorms can also be made transparent.

**Lemma 1.3 (large-offset linearization of LayerNorm).** Let $e_0\perp\mathbf
1_D$ with $\|e_0\|^2=D$, and let $w\perp\{\mathbf 1_D,e_0\}$. For $\gamma=\mathbf
1$, $\beta=0$, and $c>0$,

\[
c\,\operatorname{LN}(ce_0+w)=(1-\eta)(ce_0+w),
\qquad
0\le\eta\le\frac{\rho^2}{2c^2},\quad \rho^2=\frac{\|w\|^2}{D}+\epsilon .
\tag{12}
\]

The content $w$ is thus reproduced with relative error $O(\rho^2/c^2)$. The
directions $\mathbf 1_D$ and $e_0$ are consumed, so the linear content channel
has dimension at most $D-2$.

*Proof.* $J(ce_0+w)=ce_0+w$ and $\|ce_0+w\|^2/D=c^2+\|w\|^2/D$, so the
LayerNorm scale equals $c(1+\rho^2/c^2)^{1/2}$. The bound follows from
$1-(1+t)^{-1/2}\le t/2$. $\square$

### 3.3 Statement and proof

**Theorem 1 (structural level capability).** Consider the original PhaseFormer
with one routing layer, $D\ge4$, any $R\ge1$, and any number of heads. Then:

(a) *Global level and scale.* For every $\alpha>0$ and $c\in\mathbb R$,
$F(\alpha x+c\mathbf 1_L)=\alpha F(x)+c\mathbf 1_H$, up to the RevIN
$\epsilon_r$.

(b) *Cross-cycle level is visible to every token.* An additive cycle-level
trajectory enters every phase token as the same shift. Router attention
transmits this shift exactly, as in (11).

(c) *Adaptive level forecasting is realizable.* For every $b$ with
$\mathbf 1^\top b=1$ and $\|b\|\le1$ there is a family of parameters
$\theta_c$, $c^2\ge5L$, such that

\[
\sup_{\tilde x:\ \mathbf 1^\top\tilde x=0,\ \|\tilde x\|^2\le L}
\big\|F_{\theta_c}(\tilde x)-G_b(\tilde X)\big\|_\infty
\le\kappa\,\frac{L^{3/2}}{c^2},
\tag{13}
\]

with $\kappa$ an absolute constant ($\kappa\le30$ by the crude bound below). In
particular, by Lemma 1.1, PhaseFormer can approximate the Bayes forecast of
model (8) to any uniform accuracy.

*Proof.* (a) RevIN maps $\alpha x+c\mathbf 1$ to the same $\tilde x$, and (5)
restores $\alpha\sigma$ and $\alpha\mu+c$. (b) is Lemma 1.2 applied to
$W_e(\tilde x_p+\ell)=W_e\tilde x_p+W_e\ell$.

For (c), choose three mutually orthogonal directions $e_0,e_1,e_2\perp\mathbf
1_D$ with $\|e_i\|^2=D$. This is possible because $\dim\mathbf 1_D^\perp=D-1\ge3$.
Put $\alpha_1=\frac1K\mathbf 1_K$, $\alpha_2=b-\frac1K\mathbf 1_K$, and the
per-token scalars $f_p=(\alpha_1-\alpha_2)^\top\tilde x_p$ and
$g_p=\alpha_2^\top\tilde x_p$. Set

\[
\begin{aligned}
&W_e=e_1(\alpha_1-\alpha_2)^\top+e_2\alpha_2^\top,\quad c_e=ce_0,\quad
\gamma_e=\gamma_1=\gamma_2=\mathbf 1,\ \beta_e=\beta_1=\beta_2=0,\quad \pi\equiv0,\\
&W_q^s=0,\ c_q^s=0\ \ (\text{uniform sender weights}),\qquad
W_o^sW_v^s=\tfrac1De_2e_2^\top,\qquad W_o^rW_v^r=I,\ \text{all attention biases }0,\\
&\text{MLP output layer}=0,\qquad
W_d=\tfrac{c_*}{D}\mathbf 1_{K'}(e_1+e_2)^\top,\quad c_d=0 .
\end{aligned}
\tag{14}
\]

Here $c_*=c(1+O(\epsilon))$ is the constant that makes the map exact at zero
content.

Trace the graph. The embedding gives
$h_p=(ce_0+f_pe_1+g_pe_2)/n_p$ with $n_p=(c^2+f_p^2+g_p^2+\epsilon)^{1/2}$.
Uniform sender weights make every router buffer equal to
$B=\bar g_he_2$ with $\bar g_h=\frac1P\sum_qg_q/n_q$. Since all buffers are
identical, every receiver output equals $B$ for *any* receiver weights. The
residual stream becomes
$(ce_0+f_pe_1+(g_p+n_p\bar g_h)e_2)/n_p$. $\operatorname{LN}_1$ and
$\operatorname{LN}_2$ rescale it by factors of the form $(1+t)^{-1/2}$. The
predictor reads the $e_1+e_2$ coordinate and returns
$\mathbf 1_{K'}\,(c_*/M_p)(f_p+g_p+n_p\bar g_h)$, where $M_p$ is the total
scale. Exactly,
$f_p+g_p=\alpha_1^\top\tilde x_p$ and $\frac1P\sum_qg_q=\alpha_2^\top\bar r$,
so the zero-error limit is $G_b(\tilde X)_p$.

For the error, $\|\alpha_1-\alpha_2\|^2=\|b\|^2\le1$ and
$\|\alpha_2\|^2=\|b\|^2-\frac1K\le1$, and $\|\tilde x_p\|\le\sqrt L$ by (1).
Hence $|f_p|,|g_p|,|\bar g|\le\sqrt L$, and every scale argument satisfies
$t\le5L/c^2$. The three LayerNorm factors and the pooling ratios $n_p/n_q$ each
contribute relative error at most $t/2$ by Lemma 1.3. Multiplying by the target
magnitude $3\sqrt L$ gives (13) with $\kappa\le30$. $\square$

A direct numerical implementation of (14) with $K=30$, $P=24$, $D=8$, and
Kalman weights ($q=0.05$) on random-walk windows gives
$\max|F-G_b|=0.54,\,0.075,\,0.0078,\,7.6\times10^{-4},\,8.2\times10^{-5}$ for
$c=3,10,30,100,300$. This is the predicted $c^{-2}$ rate, with an empirical
constant $\approx4\times10^{-4}\ll\kappa$.

### 3.4 Remarks

**What Theorem 1 establishes.** The original architecture is not merely able to
*see* level. With two of its $D-2$ linear content coordinates, it can
*forecast* level optimally for a shape-plus-random-walk model: it estimates
level adaptively with recency weights, pools the estimate across all phases,
and adds it to each phase's recurring value. The baseline widths $D\in\{4,8\}$
satisfy $D\ge4$.

**The routing layer's linear class.** If the attention scores read only
positional coordinates, both attention stages become content-independent. The
sender weights are then a row-stochastic $A_s\in\mathbb R^{R\times P}$, the
receiver weights a row-stochastic $B_r\in\mathbb R^{P\times R}$, and in the
linear regime of Lemma 1.3 the realized forecasts form the class

\[
\operatorname{vec}\hat{\tilde Y}=\big(A_0\otimes I_P+A_1\otimes\Pi\big)\operatorname{vec}\tilde X,
\qquad \Pi=B_rA_s\ \text{row-stochastic},\ \operatorname{rank}\Pi\le R .
\tag{15}
\]

Here $A_0,A_1\in\mathbb R^{K'\times K}$ are maps of rank at most $D-2$ in
total. Construction (14) is the case $\Pi=\frac1P\mathbf 1\mathbf 1^\top$.

**Sample-resolution recency is exactly separable.** With positional embeddings
of vanishing amplitude and correspondingly large score weights ($D\ge5$ leaves
room for one positional coordinate), the sender can realize
$\Pi=\mathbf 1_Pv^\top$ for any $v$ in the simplex. The sample-level
exponential kernel then factorizes exactly into a cycle factor and a phase
factor,

\[
\varrho^{L-t}=\varrho^{P(K-k)}\,\varrho^{P-p},\qquad t=(k-1)P+p ,
\tag{16}
\]

so a sample-resolution EMA level readout lies in class (15), with
$b_k\propto\varrho^{P(K-k)}$ and $v_p\propto\varrho^{P-p}$.

**Soft clipping at finite $c$.** When a single content coordinate $t$ is
active, the embedding LayerNorm returns $t/\sqrt{c^2+t^2+\epsilon}$ on that
coordinate. Up to constant factors, the realized value is

\[
t\ \mapsto\ \frac{c\,t}{\sqrt{c^2+t^2}},
\tag{17}
\]

a soft clip that saturates at $\pm c$. Accuracy in Theorem 1 is bought by a
large offset $c$. The next theorem shows that $c$ is, necessarily, the size of
the model's level envelope.

**What Theorem 1 does not establish.** It is an existence result: a trained
checkpoint need not sit near $\theta_c$. It also leaves open how large a level
excursion a given checkpoint can follow. Theorem 2 answers that question for
every parameter value.

## 4. Theorem 2: some level departures cannot be modeled

### 4.1 The final-LayerNorm envelope

**Lemma 2.1.** For any $z\in\mathbb R^D$, the normalized part
$\varsigma=Jz/\sqrt{\|Jz\|^2/D+\epsilon}$ of (3) satisfies

\[
\varsigma\in\mathbf 1_D^\perp,\qquad
\|\varsigma\|^2=\frac{D\,\|Jz\|^2}{\|Jz\|^2+D\epsilon}<D .
\tag{18}
\]

*Proof.* Direct computation. $\square$

Let $A=W_d\operatorname{diag}(\gamma_2)\in\mathbb R^{K'\times D}$ and
$q'=W_d\beta_2+c_d$, both taken from the last unit. Define the envelope
constants

\[
c_0=\frac{\mathbf 1_{K'}^\top q'}{K'},\qquad
C_1=\frac{\sqrt D\,\big\|J\big(\gamma_2\odot W_d^\top\mathbf 1_{K'}\big)\big\|}{K'},
\qquad
\mathcal I_\varphi=(c_0-C_1,\,c_0+C_1),\qquad
C_\varphi=|c_0|+C_1 .
\tag{19}
\]

**Theorem 2 (level envelope of the original PhaseFormer).** For *every*
parameter value of the original architecture (any number of routing units, any
$R$, heads, positional embeddings, and MLP weights) and for *every* input
window:

(a) *Per-phase ellipsoid.* Each normalized phase forecast satisfies

\[
\hat{\tilde y}_p\in\mathcal E_\theta=\{\,q'+A\varsigma:\ \varsigma\in\mathbf
1_D^\perp,\ \|\varsigma\|<\sqrt D\,\},
\tag{20}
\]

a solid ellipsoid of dimension at most $\min(D-1,K')$. Consequently every
linear functional is bounded, with
$|\langle a,\hat{\tilde y}_p-q'\rangle|<\sqrt D\,\|JA^\top a\|$ for all
$a\in\mathbb R^{K'}$.

(b) *Level envelope.* $\operatorname{lev}(\hat{\tilde y})\in\mathcal I_\varphi$.
In original units, $|m_{\hat y}-\mu|<C_\varphi\,\sigma$: no forecast level can
lie farther than $C_\varphi$ window standard deviations from the window mean.

(c) *Error floor.* For every window with level demand $d$,

\[
\tfrac1H\|\hat{\tilde y}-\tilde y\|^2\ \ge\ \operatorname{dist}(d,\mathcal I_\varphi)^2
\ \ge\ (|d|-C_\varphi)_+^2,
\qquad
\tfrac1H\|\hat y-y\|^2\ \ge\ \sigma^2(|d|-C_\varphi)_+^2 .
\tag{21}
\]

(d) *Level–shape competition.* Let $v=JA^\top\mathbf 1_{K'}/K'$, so that
$C_1=\sqrt D\|v\|$, and let $\ell_p=\mathbf 1^\top\hat{\tilde y}_p/K'$ be the
level of phase $p$. Then

\[
\hat{\tilde y}_p=q'+\frac{\ell_p-c_0}{\|v\|^2}\,Av+A\varsigma_\perp,
\qquad
\|\varsigma_\perp\|^2<D\Big(1-\Big(\frac{\ell_p-c_0}{C_1}\Big)^2\Big),
\quad \varsigma_\perp\perp v .
\tag{22}
\]

As the level approaches the envelope boundary, the input-dependent part
$A\varsigma_\perp$ vanishes, and the phase forecast is forced onto the single
profile $Av$.

*Proof.* Every unit ends with $\operatorname{LN}_2$, so
$h^{(N)}_p=\gamma_2\odot\varsigma_p+\beta_2$ with $\varsigma_p$ as in Lemma 2.1.
Substituting into (5) gives $\hat{\tilde y}_p=q'+A\varsigma_p$, which is (20).
The functional bound is Cauchy–Schwarz together with $\varsigma=J\varsigma$:
$\langle a,A\varsigma\rangle=\langle JA^\top a,\varsigma\rangle$.

For (b), $\operatorname{lev}(\hat{\tilde y})=\frac1P\sum_p\ell_p$. Each
$\ell_p=c_0+\langle v,\varsigma_p\rangle$ lies in $\mathcal I_\varphi$ by the
bound with $a=\mathbf 1/K'$, and $A^\top\mathbf 1=\gamma_2\odot W_d^\top\mathbf 1$.
Averaging preserves the interval. De-normalization (5) gives the statement in
original units.

For (c), apply the split (7) and drop the centered term. The level term is at
least the squared distance from $d$ to $\mathcal I_\varphi$.

For (d), decompose $\varsigma_p$ along $v$ and its orthogonal complement. The
coefficient on $v$ is fixed by $\ell_p$, and the norm bound follows from
$\|\varsigma_p\|^2<D$. $\square$

The theorem uses nothing about routing, attention, or depth. Those components
decide *which* point of $\mathcal E_\theta$ is produced, never the set itself.
The constants in (19) are read directly from a checkpoint's final
$\operatorname{LN}_2$ and predictor.

### 4.2 Which level departures fall outside: scale-free demand

Instance normalization makes the demand $d$ dimensionless. For several basic
level patterns, $d$ is therefore independent of the amplitude of the level
change.

**Corollary 2.1 (steps and trends).** Let $L$ be the window length.

(i) *Persistent step.* Suppose the level jumps by $\Delta$ at $m$ samples before
the origin and persists. Suppose also that the rest of the window has zero-mean,
uncorrelated variation of variance $\sigma_0^2$. With $f=m/L$,

\[
d=\frac{\Delta(1-f)}{\sqrt{\Delta^2f(1-f)+\sigma_0^2}}
\ \xrightarrow{\ \Delta/\sigma_0\to\infty\ }\ \sqrt{\frac{L-m}{m}} .
\tag{23}
\]

For $L=720$ and $m=1,6,24,72,240$, the limit is
$d=26.81,\,10.91,\,5.39,\,3.00,\,1.41$. Any step more recent than
$m^*=L/(1+C_\varphi^2)$ samples demands a level outside every envelope of
half-width $C_\varphi$, however small the step is relative to its background.

(ii) *Continued linear trend.* A linear trend of any slope that continues
through the horizon has

\[
d=\frac{\sqrt3\,(L+H)}{\sqrt{L^2-1}},
\tag{24}
\]

which equals $1.96,\,2.19,\,2.54,\,3.46$ for $H=96,192,336,720$ at $L=720$.

*Proof.* (i) The window mean is $\Delta f$, and the variance is
$\Delta^2f(1-f)+\sigma_0^2$ (population variance, $\epsilon_r\to0$). The future
level is $\Delta$. (ii) The time index $1,\dots,L$ has mean $(L+1)/2$ and
population variance $(L^2-1)/12$. The future mean index is $L+(H+1)/2$. $\square$

Equation (23) also explains why ordinary within-cycle variation protects
PhaseFormer: a large $\sigma_0$ from recurring shape inflates the window's
standard deviation and shrinks $d$.

**Idealized illustration (simulation, not dataset evidence).** For a Brownian
level with unit terminal variance, $L=720$, and a persistent forecast of the
last level, $d=(\ell_L-\mu)/\sigma$ has the following distribution:

| window content | median $\lvert d\rvert$ | $P(\lvert d\rvert>1)$ | $P(\lvert d\rvert>2)$ | $P(\lvert d\rvert>3)$ | $\mathbb E(\lvert d\rvert-2)_+^2$ |
|---|---:|---:|---:|---:|---:|
| pure random-walk level | 1.15 | 0.569 | 0.154 | 0.014 | 0.052 |
| + seasonal amplitude 0.5 | 0.77 | 0.357 | 0.020 | 0.000 | 0.001 |
| + seasonal amplitude 1.0 | 0.48 | 0.134 | 0.000 | 0.000 | 0.000 |

The envelope binds on a *tail* of windows: recent steps, sustained trends, and
level-dominated windows with weak seasonal content. It does not bind on the
bulk of strongly periodic windows. This matches the empirical picture in
Section 7: the gain is systematic but moderate, and it is largest where level
dynamics dominate.

### 4.3 Capability costs envelope

**Corollary 2.2.** Every PhaseFormer that approximates $G_b$ of (9) uniformly
within $\delta$ in forecast level has

\[
C_\varphi(\theta)\ \ge\ \sup_{\tilde x}\big|\operatorname{lev}G_b(\tilde X)\big|-\delta
=\sqrt{K\|b\|^2-1}-\delta .
\tag{25}
\]

The construction of Theorem 1 attains accuracy $\delta\asymp L^{3/2}/c^2$
with $C_\varphi=\sqrt2\,c_*$.

*Proof.* $\operatorname{lev}G_b=b^\top\bar r$. Over $\mathbf 1^\top\bar r=0$ and
$\|\bar r\|^2\le K$, the maximum is $\sqrt K\|J_Kb\|=\sqrt{K\|b\|^2-1}$. It is
attained by a window that is constant within cycles, which is admissible. By
Theorem 2(b), the model's level cannot exceed $C_\varphi$. For (14),
$\gamma_2=\mathbf 1$ and $W_d^\top\mathbf 1=c_*(e_1+e_2)K'/D$, so
$C_1=\sqrt D\,c_*\sqrt{2D}/D=\sqrt2c_*$, while $c_0=0$. $\square$

For the most recent-weighted readout, $b\to e_K$, the bound (25) equals
$\sqrt{K-1}=5.39$ at $K=30$. This coincides with the demand of a one-cycle-old
step in (23), $\sqrt{(L-P)/P}$. Theorems 1 and 2 therefore describe one
trade-off. Accurate level tracking requires a large final-layer gain along
$W_d^\top\mathbf 1$. That gain is bounded in any trained checkpoint, and
by (22) it competes with the shape the same coordinates must carry.

### 4.4 Scope

Theorem 2 is exact, but whether it binds is a property of a checkpoint and a
data distribution. $C_\varphi$ is finite for every trained model and can be
read from its weights. The demand distribution of $d$ can be computed from
validation windows. We have not yet measured $C_\varphi$ on the trained
checkpoints; that measurement, together with the fraction of validation windows
with $|d|>C_\varphi$ and the level share of phase-only MSE in (7), is the
next verification step (Section 8). With `affine=True` RevIN, the envelope is
rescaled by the affine parameters and remains bounded.

## 5. Theorem 3: the unmodeled level is a low-rank linear complement

### 5.1 Setting

Let the conditional mean of the normalized future have the form

\[
\mathbb E[\tilde y\mid\tilde x]=S(\tilde x)+\sum_{j=1}^m a_j\,\lambda_j(\tilde x),
\tag{26}
\]

where $a_1=\mathbf 1_H$ is the *displacement* profile. The profiles
$a_2,\dots,a_m$ are fixed, horizon-centered, low-frequency profiles such as
tilt or curvature. The $\lambda_j$ are scalar state forecasts, and $S$ is a
horizon-centered shape term that the phase path is meant to carry. Model (8)
has $m=1$ and $\lambda_1=\operatorname{lev}G_b$. A local linear trend adds a
tilt, $m=2$.

Every phase forecast $F$ decomposes uniquely as

\[
F(\tilde x)=S(\tilde x)+\sum_{j=1}^m a_j\,\varphi_j(\tilde x)+r_\varphi(\tilde x),
\qquad r_\varphi(\tilde x)\perp\operatorname{span}\{a_j\}.
\tag{27}
\]

The scalars $\varphi_j$ are the projections of $F-S$ onto the profiles. The
remainder $r_\varphi$ is the phase path's shape error; it is small when the
phase path is *phase-consistent*. Because $a_1=\mathbf 1$ and the other terms
are centered, $\varphi_1=\operatorname{lev}(F)\in\mathcal I_\varphi$ by
Theorem 2. The conditional residual is then

\[
D_{\mathrm{cond}}=\tilde y-F(\tilde x)=\sum_{j=1}^m a_j\,\psi_j(\tilde x)+\nu-r_\varphi(\tilde x),
\qquad \psi_j=\lambda_j-\varphi_j,\quad \nu=\tilde y-\mathbb E[\tilde y\mid\tilde x],
\tag{28}
\]

and in particular $|\psi_1|\ge\operatorname{dist}(\lambda_1,\mathcal I_\varphi)$.
The unmodeled level of Theorem 2 is exactly the displacement coefficient
$\psi_1$. For the *envelope-optimal* phase path,
$\varphi_1=\Pi_{\mathcal I_\varphi}(\lambda_1)$ and $\psi_1$ is the
soft-threshold excess $\lambda_1-\Pi_{\mathcal I_\varphi}(\lambda_1)$.

### 5.2 Statement and proof

**Theorem 3 (low-rank linear complement).** Let $\zeta=\zeta(\tilde x)$ be any
branch input computed from the window, such as the anchored history, with
$\Sigma=\mathbb E[\zeta\zeta^\top]\succ0$.

(a) *Output rank, for any nonlinearity.* The least-squares linear predictor of
$D_{\mathrm{cond}}$ from $\zeta$ is

\[
W^*=\underbrace{\sum_{j=1}^m a_j\beta_j^\top}_{M,\ \operatorname{rank}\le m}
-\ \mathbb E[r_\varphi\zeta^\top]\Sigma^{-1},
\qquad \beta_j=\Sigma^{-1}\mathbb E[\psi_j\zeta] .
\tag{29}
\]

The innovation $\nu$ contributes nothing. With $\mathcal
R(W)=H^{-1}\mathbb E\|D_{\mathrm{cond}}-W\zeta\|^2$, the reduced-rank
predictive spectrum satisfies

\[
\mathcal R(M)-\mathcal R(W^*)=\tfrac1H\big\|\mathbb E[r_\varphi\zeta^\top]\Sigma^{-1/2}\big\|_F^2
\le\tfrac1H\mathbb E\|r_\varphi\|^2,
\qquad
\sum_{i>m}s_i^2\le\big\|\mathbb E[r_\varphi\zeta^\top]\Sigma^{-1/2}\big\|_F^2 ,
\tag{30}
\]

where $s_i$ are the singular values of $\mathbb E[D_{\mathrm{cond}}\zeta^\top]\Sigma^{-1/2}$.

(b) *Readout is the recency kernel.* Suppose $\zeta\sim\mathcal N(0,\Sigma)$,
$\lambda_1=w^\top\zeta+\text{const}$, and the phase path is envelope-optimal.
Then

\[
\beta_1=\mathbb E[\nabla_\zeta\psi_1]
=P\big(\lambda_1\notin\mathcal I_\varphi\big)\,w .
\tag{31}
\]

The best linear corrector reads the Bayes level kernel $w$, attenuated by the
probability that the envelope binds. It recovers the fraction

\[
\varrho=\frac{\mathbb E[(\beta_1^\top\zeta)^2]}{\mathbb E\psi_1^2}
=\frac{P(\lambda_1\notin\mathcal I_\varphi)^2\operatorname{Var}\lambda_1}{\mathbb E\psi_1^2}
\tag{32}
\]

of the unmodeled level energy.

(c) *In joint training the complement can be carried exactly by a rank-one
anchored branch.* Consider the PhaseFormer-L fusion
$\hat y=(1-g)F'+gR$ with $g\in(0,1)$, and the anchored branch
$R=\tilde x_L\mathbf 1_H+Wz+c$, where $z=\tilde x-\tilde x_L\mathbf 1_L$.
Suppose (26) holds with $m=1$, $\lambda_1=w^\top\tilde x$, and
$\mathbf 1^\top w=1$. Let the backbone produce $F'=S/(1-g)$, whose level is
zero, and choose

\[
W=\mathbf 1_H u^\top,\qquad u=\frac{w-(1-g)L^{-1}\mathbf 1_L}{g},\qquad c=0 .
\tag{33}
\]

Then $\hat y=\mathbb E[\tilde y\mid\tilde x]$ exactly, so the pair attains the
Bayes risk of (26). For every other admissible split of the level between the
two paths, the branch's required output still lies in
$\operatorname{span}\{a_j\}$. The split changes the readout, never the output
rank. The analogous construction with $m$ profiles gives $\operatorname{rank}W\le m$.

(d) *Gate.* For fixed $F$ and $R$ with $\Delta=R-F$, the risk
$\mathbb E\|\tilde y-F-g\Delta\|^2$ is minimized at

\[
g^*=\frac{\mathbb E\langle D_{\mathrm{cond}},\Delta\rangle}{\mathbb E\|\Delta\|^2},
\qquad
\text{reduction}=\frac{\big(\mathbb E\langle D_{\mathrm{cond}},\Delta\rangle\big)^2}{\mathbb E\|\Delta\|^2}.
\tag{34}
\]

The reduction is strictly positive if and only if the branch is aligned with
the residual, $\mathbb E\langle D_{\mathrm{cond}},\Delta\rangle\ne0$. For
$\Delta=\mathbf 1_H\beta_1^\top\zeta$ with $\beta_1$ as in (31), the alignment
equals $H\,P(\lambda_1\notin\mathcal I_\varphi)^2\operatorname{Var}\lambda_1$.
So the gate opens exactly when the envelope binds with positive probability.

*Proof.* (a) Because $\zeta$ is a function of $\tilde x$, the tower property
gives $\mathbb E[\nu\zeta^\top]=\mathbb E[\mathbb E[\nu\mid\tilde x]\zeta^\top]=0$.
Substituting (28) into $W^*=\mathbb E[D_{\mathrm{cond}}\zeta^\top]\Sigma^{-1}$
gives (29). For any $W$,
$\mathcal R(W)-\mathcal R(W^*)=H^{-1}\|(W-W^*)\Sigma^{1/2}\|_F^2$; setting $W=M$
gives the equality in (30). The inequality holds because
$\mathbb E[r_\varphi\zeta^\top]\Sigma^{-1}\zeta$ is the $L^2$ projection of
$r_\varphi$ onto linear functions of $\zeta$. The spectral bound follows from
Eckart–Young: $M\Sigma^{1/2}$ has rank at most $m$ and lies within
$\|\mathbb E[r_\varphi\zeta^\top]\Sigma^{-1/2}\|_F$ of the cross-covariance
matrix.

(b) Stein's lemma for centered Gaussian vectors,
$\mathbb E[\zeta\,\psi(\zeta)]=\Sigma\,\mathbb E[\nabla\psi(\zeta)]$, gives the
first equality. For the soft-threshold excess,
$\nabla\psi_1=\mathbb 1\{\lambda_1\notin\mathcal I_\varphi\}\,w$ almost
everywhere. Then (32) follows from
$\mathbb E[(\beta_1^\top\zeta)^2]=\beta_1^\top\Sigma\beta_1$.

(c) Since $u^\top\mathbf 1=1$, the anchor cancels:
$g(\tilde x_L+u^\top(\tilde x-\tilde x_L\mathbf 1))=g\,u^\top\tilde x$. RevIN's
$\mathbf 1^\top\tilde x=0$ then gives $g\,u^\top\tilde x=w^\top\tilde x$.
Hence $(1-g)F'+gR=S+\mathbf 1_H\lambda_1$. For a general split, the branch
must supply $[\mathbb E[\tilde y\mid\tilde x]-(1-g)F']/g-\tilde x_L\mathbf 1$.
If $F'$ carries the shape, this lies in $\operatorname{span}\{a_j\}$.

(d) The risk is a quadratic in $g$. Its minimizer and minimum follow directly,
and the alignment is computed from (28), (31), and $\mathbb E\langle
r_\varphi,\mathbf 1\rangle=0$. $\square$

A direct simulation of (31) with a five-dimensional correlated Gaussian input
and a symmetric envelope at one level standard deviation returns
$\beta_1/w=0.317\pm0.003$ in every coordinate, against
$P(\lambda_1\notin\mathcal I_\varphi)=0.316$.

### 5.3 Consequences

**Why the rank is small even though the gap is nonlinear.** The level that
PhaseFormer cannot follow is a thresholded, nonlinear function of the window.
Its *output geometry* is nevertheless fixed. It always acts through the
displacement and a few smooth profiles, and a linear predictor inherits the
output geometry, not the nonlinearity. This is sharper than the
0926 Proposition 2: the innovation noise no longer contributes to the rank
remainder, and only the phase path's own shape error $r_\varphi$ does.

**Why joint training matters.** For a *frozen* envelope-limited backbone, the
linear corrector is still rank one, but by (32) it recovers a shrinking fraction
of the excess as the envelope loosens. For Gaussian $\lambda_1$ with standard
deviation $s$ and a symmetric envelope of half-width $C$,
$\varrho=0.908,\,0.668,\,0.179,\,0.018$ at $C/s=0.5,1,2,3$. In *joint* training,
part (c) shows a better split. The backbone hands the level to the anchored
branch and keeps the shape. The level error then vanishes, and by (22) the
backbone recovers its full shape budget at $\ell_p=0$. The theory therefore
predicts that jointly trained branches are low-rank, level-reading, and poor as
standalone forecasters, yet indispensable to the fused model. Section 7.4
observes exactly this reversal.

**Anchoring identity.** For any $W=\mathbf 1_Hw^\top$ with $\mathbf 1^\top w=1$,
$\tilde x_L\mathbf 1_H+W(\tilde x-\tilde x_L\mathbf 1_L)=\mathbf 1_H\,w^\top\tilde
x$. The parameter-free anchor gives the branch an *unbounded*, exactly
shift-equivariant level channel. The phase path's level channel is confined to
$\mathcal I_\varphi$ by Theorem 2.

**Testable predictions.** Theorem 3 predicts:
(P1) the leading output direction of the correction is close to $\mathbf 1_H$
when the level persists, and uses smooth tilt or curvature when it evolves
within the horizon;
(P2) the leading readout is a recency-weighted level kernel;
(P3) the predictive spectrum is concentrated on at most a few profiles plus a
remainder;
(P4) the branch's value is defined relative to the backbone, so deleting its
level mode can help the branch but hurt the fusion;
(P5) the gain concentrates on windows with $|d|$ beyond the checkpoint's
envelope.
Section 7 tests P1–P4. P5 requires the checkpoint measurement described in
Section 4.4.

### 5.4 Measuring the right residual

Two residual targets serve different questions. The independent target
$D_{\mathrm{ind}}=\tilde y-\tilde x_L\mathbf 1_H$ measures what a linear branch
can add over last-value persistence. It is available uniformly across all 28
settings and is used for the broad closed-form spectrum. The conditional target
$D_{\mathrm{cond}}$ in (28) is the object of Theorem 3. In model (26) the level
parts of both targets are functions of the same scalar $\lambda_1$, so both
theories predict recency-kernel readouts along $\mathbf 1_H$. Section 7.2
checks directly that the two targets' leading input directions agree on the
settings used for the mechanism analysis.

## 6. PhaseFormer-L

### 6.1 Anchored temporal corrector

The branch is a full-history anchored linear map. For a normalized input
$x_n=(x-\mu)/\sigma$, let $x_{n,L}$ be the last input value and define

\[
z=x_n-x_{n,L}\mathbf 1_L,
\qquad
\hat y_r=\sigma\big(Wz+x_{n,L}\mathbf 1_H+c\big)+\mu\mathbf 1_H .
\tag{35}
\]

The phase path and the temporal branch are fused by a learned gate,

\[
\hat y=(1-g)\odot\hat y_\phi+g\odot\hat y_r,
\qquad g=\operatorname{sigmoid}(\gamma).
\tag{36}
\]

The backbone, branch, and gate are trained jointly, which is the setting of
Theorem 3(c). Because the normalization scale cancels in the linear term, the
branch is an anchored map of the original history, $W(x-x_L\mathbf 1_L)$, plus
its anchor. The dense branch is deliberately unstructured, so it is a probe:
Theorem 3 predicts its effective rank, and nothing in the parametrization
forces that rank.

### 6.2 Canonical modes and intervention

For a factorized head $W=VU$, we analyze the effective map rather than arbitrary
hidden coordinates:

\[
W=\sum_i s_i u_i v_i^\top .
\tag{37}
\]

Mode $i$ reads $v_i^\top z$ and writes the horizon profile $u_i$. We match input
vectors to a fixed dictionary of recent-level, local-trend, curvature,
periodic-shape, and change-point templates. Output vectors are matched to
displacement, tilt, curvature, and periodic-correction templates.

Theorem 3 defines the theoretical input and output spaces

\[
\mathcal T_{\mathrm{in}}=\operatorname{span}(w),
\qquad
\mathcal T_{\mathrm{out}}=\operatorname{span}\{a_1,\dots,a_m\}.
\tag{38}
\]

For every learned mode, we measure its input and output alignments,
$\alpha_i=\|P_{\mathcal T_{\mathrm{in}}}v_i\|^2$ and
$\beta_i=\|P_{\mathcal T_{\mathrm{out}}}u_i\|^2$. These are implemented through
the recent-level and displacement/tilt template dictionaries. A mode is
supported by the theory only when both alignments are high. Its importance is
then measured independently through its functional contribution $I_i$.

The intervention keeps the trained parameters and phase input fixed while
removing a selected input subspace or canonical mode from the branch. If
$\Delta\hat y_i=g\odot s_iu_i(v_i^\top z)$ is the forecast increment of mode
$i$, deleting that mode changes the fused squared error by

\[
\Delta\mathcal L_i=2\,\mathbb E[e^\top\Delta\hat y_i]+\mathbb E\|\Delta\hat y_i\|_2^2,
\qquad e=\hat y-y .
\tag{39}
\]

A theory-aligned mode with $I_i>0$ and $\Delta\mathcal L_i>0$ is therefore
required by the fused forecast under a controlled counterfactual.
Same-dimensional random subspaces use the same formula, but they have no
alignment to $\mathcal T_{\mathrm{in}}$ or $\mathcal T_{\mathrm{out}}$, so they
provide the specificity control.

### 6.3 Functional contribution

For a frozen checkpoint, each mode increment changes the fused prediction
additively. The exact MSE change from deleting a set of modes can therefore be
computed from the individual contributions $I_i$, with an algebraic forward
check. We define the functional rank $r_{95}$ as the smallest number of modes,
ordered by $I_i$, that recovers 95% of the checkpoint's own improvement over a
zeroed-map baseline. This is a realized model quantity, distinct from the
predictive spectrum in (30).

## 7. Selected evidence

We use a lookback of 720 and horizons $H\in\{96,192,336,720\}$. Main model
comparisons use three seeds and the matched phase-only baseline. Each witness
group below answers one question, and we avoid mixing incompatible
model-selection records.

### 7.1 The correction is useful in a coherent adaptation regime

The strongest matched comparison is the 12-setting group consisting of all four
horizons of ETTh2, ETTm2, and Weather. PhaseFormer-L improves both MSE and MAE
at all 12 settings in the three-seed comparison. MSE reductions range from
0.52% to 8.73%. The six settings added in the final training expansion also
improve both metrics. The effect is therefore neither single-horizon nor
single-dataset, and the comparison supplies the performance anchor for the
mechanism analysis.

| witness group | settings | dual-metric wins | largest MSE reduction |
|---|---:|---:|---:|
| ETTh2 + ETTm2 + Weather, $H\in\{96,192,336,720\}$ | 12 | 12/12 | 8.73% |

The size of the gain matches Section 4.2. Theorem 2 predicts a systematic
improvement concentrated on a tail of level-dominated windows, not a wholesale
failure of the phase path. That tail is the part of level dynamics the backbone
cannot follow, and it is the part the branch is meant to supply.

### 7.2 The attainable correction has a concentrated spectrum (P1, P3)

Closed-form reduced-rank regression was computed on 28 settings using the
last-value residual target. Across these settings, the leading mode accounts for
64.2%--86.2% of the total attainable linear reduction, and 2--7 directions
capture 90%. The leading output direction has cosine 0.890--0.989 with a
horizon-wide constant vector. On ETTh2 and ETTm2, the input mode consistently
matches a recency-weighted level kernel. Weather has a larger share of smooth
tilt and curvature.

| quantity | observed range | theoretical counterpart |
|---|---:|---|
| $s_1^2/\sum_i s_i^2$ | 0.642--0.862 | (30): few profiles plus a remainder |
| directions for 90% value | 2--7 | $m$ small; the tail is bounded by $r_\varphi$ |
| $\lvert\cos(u_1,\mathbf 1_H)\rvert$ | 0.890--0.989 | $a_1=\mathbf 1_H$ (displacement) |

As a direct check that this spectrum remains relevant after phase modeling, we
also projected the same training examples onto the phase-conditioned target
$D_{\mathrm{cond}}$. On the six primary settings carried into the head
analysis, the first input directions obtained from $D_{\mathrm{cond}}$ and from
the independent target have absolute cosine at least 0.9991. The conditional
analysis therefore points to the same recent-level coordinate in the settings
where the trained correction is later dissected. This supports the
conditional reading of Section 5.4; it does not assume that the two residual
targets are identical in every dataset.

### 7.3 Which modes are actually used? (P1, P2)

The trained-head dissection covers six principal settings, three seeds, and three
head capacities, giving 18 model-setting groups. In every group, the leading
input mode matches a recent weighted level, with explanation scores
0.57--0.73. In 12/18 groups, all from ETTh2 and ETTm2, the leading output is an
overall displacement, with explanation scores 0.91--0.97. The six Weather groups
use smooth curvature or tilt instead. This is the $m\ge2$ case of (26): the same
level readout is written onto a profile that evolves within the horizon.

| head-side pattern | groups | explanation score |
|---|---:|---:|
| recent weighted level input | 18/18 | 0.57--0.73 |
| level → horizon-wide displacement | 12/18 | 0.91--0.97 |
| level → Weather tilt/curvature | 6/18 | smooth low-frequency output |

The input and output explanation scores are the empirical dictionary estimates
of $\alpha_i$ and $\beta_i$ in (38). The leading ETT mode is aligned with both
theoretical factors, $w$ and $a_1=\mathbf 1_H$. The Weather modes keep the same
readout and use the smooth profiles $a_{j\ge2}$.

The canonical functional-rank analysis uses 72 frozen checkpoints (six settings,
four nominal compression levels, three seeds). Three to five modes recover at
least 95% of each checkpoint's own fitted improvement. A representative
ETTh2-720 rank-$H/8$ checkpoint has a first-mode contribution of 75.9%. Its
first five modes sum to 95.3%: a dominant level displacement plus smaller
periodic and higher-order corrections. The periodic modes are the empirical
footprint of the shape remainder $r_\varphi$ in (29).

| canonical mode group (ETTh2-720 witness) | cumulative contribution |
|---|---:|
| recent weighted level → displacement (mode 1) | 75.9% |
| periodic-shape corrections (modes 2–3) | 15.0% |
| higher-order level/curvature corrections (modes 4–5) | 4.3% |
| first five modes together | 95.3% |

### 7.4 Intervention establishes complementarity (P4)

We next remove the semantic input subspace spanned by the theory-aligned
recent-level directions, keeping the phase path and all trained weights fixed.
A same-dimensional random RRR subspace is the negative control. In all 18
analyzed groups, semantic deletion produces a fused MSE change outside the 95%
random-control interval. Across the underlying 72 checkpoint-level
interventions, deleting the semantic subspace improves the branch's own MSE in
52/72 cases but worsens the fused MSE in 72/72 cases. The median relative fused
increase is 30.43%.

This reversal is the signature predicted by Theorem 3(c). In the joint split,
the branch carries the level and the backbone carries the shape. Neither is a
good standalone forecaster, but the level mode is indispensable to their sum.
The theory predicts a readout in $\mathcal T_{\mathrm{in}}$ and a low-frequency
writeout in $\mathcal T_{\mathrm{out}}$. The trained heads contain those modes,
and the intervention in (39) changes the fused loss when exactly that subspace
is removed. The intervention is a counterfactual of the fitted model; it is not
a claim that the data-generating process has been causally identified.

### 7.5 From mode evidence to a compact implementation

The concentration results motivate a generic temporal factorization, $W=VU$,
with the input and output coordinates left trainable. The most aggressive
exploratory compression uses only 3.5--6.1% of the dense-head parameters and
retains 92.4--101.9% of the dense branch's attainable reference improvement
across the reported witness settings. In the broader 24-setting comparison, a
rank-$H/8$ head reduces the total model to 16.4--33.0% of the dense model, with
a mean absolute relative MSE difference of 0.72% from the dense PhaseFormer-L.

| compression view | retained capacity | evidence |
|---|---:|---|
| branch-only exploratory $q=1/32$ | 3.5--6.1% of dense head | 92.4--101.9% attainable value |
| full-model rank-$H/8$ | 16.4--33.0% of dense model | 0.72% mean absolute relative MSE difference |

The design implication is to choose rank from predictive accumulation and the
desired accuracy-capacity point, rather than to hard-code a single rank-one
filter. Theorem 3 explains why strong compression is possible: the output rank
is the number of profiles. The shape remainder and the secondary profiles
determine how much accuracy each additional rank retains.

## 8. Discussion: the theory-to-evidence chain

The paper forms an ordered chain.

1. **Capability (Theorem 1):** a common level shift reaches every phase token
   and passes exactly through router attention. The original design can
   realize the Bayes forecast of a seasonal-shape plus random-walk-level model
   with content rank two.
2. **Envelope (Theorem 2):** the final LayerNorm confines the forecast level to
   $\mathcal I_\varphi$ for every parameter value, input, and depth. The level
   demand is scale-free, so recent steps and sustained trends fall outside it
   regardless of their size. Capability and envelope are the same final-layer
   gain (Corollary 2.2).
3. **Complement (Theorem 3):** the unmodeled level is a scalar on fixed horizon
   profiles. Its best linear predictor has rank at most $m$, reads a recency
   kernel, and in joint training can be carried exactly by a rank-one anchored
   branch.
4. **Predictive rank (P3):** one mode buys 64.2--86.2% of the attainable linear
   value, and its output is horizon-wide.
5. **Conditional bridge:** independent and phase-conditioned leading input
   directions have absolute cosine at least 0.9991 on the six primary settings.
6. **Mode identity (P1, P2):** trained heads read a recent weighted level in all
   18 groups. They write a displacement on ETT data and a smooth tilt or
   curvature on Weather, the $m=1$ and $m\ge2$ cases of (26).
7. **Functional rank:** three to five canonical modes recover at least 95% of
   the fitted head's own gain.
8. **Complementarity (P4):** semantic deletion is more damaging than matched
   random deletion, and it can improve the branch while degrading the fused
   model, as predicted by the joint split of Theorem 3(c).
9. **Optimization:** generic low-rank factorization preserves most of the useful
   correction at a small parameter fraction.

Three percentages should not be conflated:

- **Model-level gain:** PhaseFormer-L versus phase-only, such as the 12/12
  dual-metric wins and the 8.73% maximum MSE reduction.
- **Branch-level capture:** the fraction of a linear branch's attainable or
  fitted improvement retained by a low-rank head.
- **Mode-level contribution:** the exact frozen-checkpoint contribution $I_i$
  and its cumulative functional rank.

**Assumptions and open verification.** Theorems 1 and 2 are exact statements
about the architecture. Theorem 1 is existential, and Theorem 2 binds only when
the level demand exceeds a checkpoint's $C_\varphi$. Theorem 3(a) needs only
the profile structure (26). Parts (b) and (c) add a Gaussian design and a
linear Bayes level, respectively; they are idealizations that explain the
observed readouts and reversal, not assumptions the data are known to satisfy.
The following measurements have not yet been made and would close the loop
between Theorem 2 and the observed gain:

- $C_\varphi$ computed from each trained phase-only checkpoint via (19);
- the fraction of validation windows with $|d|>C_\varphi$;
- the level share $(\operatorname{lev}(\hat{\tilde y})-d)^2$ of phase-only MSE
  in (7);
- whether PhaseFormer-L's per-window gain concentrates on those windows (P5).

The result is a conditional completion of PhaseFormer's representation. The phase
backbone carries recurring shape and the part of level variation it can reach.
The anchored linear branch removes the envelope with a compact
state-adaptation coordinate. Its dominant mechanism is a recent-level input
written as a persistent or smooth low-frequency output, while additional modes
absorb dataset-specific evolution.

## 9. Conclusion

PhaseFormer's phase representation is not incapable of modeling cross-cycle
level changes. It can, and optimally so for a seasonal-shape plus random-walk
level. Its level capability is bounded by an envelope set by the final
normalization and the predictor. Under instance normalization, the level
demands outside that envelope are determined by recency rather than
magnitude. What lies outside has a fixed output geometry, so the missing
complement is low-rank. Its readout is a recency kernel, and in joint training
a rank-one anchored branch can carry it exactly.

Reduced-rank analysis, trained-head mode semantics, functional contribution, and
controlled deletion all point to the mechanism the theory predicts: a recent
weighted level state converted into a horizon-wide displacement or a smooth
low-frequency correction. The correction is valuable because it complements
the phase path, not because it is a better standalone predictor. PhaseFormer-L
turns this into a practical extension: a gated anchored linear corrector whose
capacity can be reduced according to predictive value. In the chain presented
here, theory identifies what the backbone can do, where it must stop, and why
the remainder is low-rank. Spectra quantify the remainder, modes name its
content, interventions establish its dependence on the backbone, and low-rank
factorization provides the efficiency path.

## Evidence map and provenance

The numerical witnesses in this draft are selected from the audited experiment
records already present in the repository:

- principal performance: E14 main comparison and the consolidated PhaseFormer-L
  result tables;
- predictive spectrum and mode geometry: E15 closed-form dimension analysis;
- trained-head semantics and interventions: E16 dissection and the
  low-rank-checkpoint intervention records;
- functional rank and per-mode contribution: the 72-checkpoint functional-rank
  analysis;
- compression: the conditioned rank sweep and rank-capacity report.

The 12-setting performance group is a three-seed matched comparison. The
28-setting spectrum is a train/validation closed-form analysis. The 18-group
semantic dissection and 72-checkpoint functional-rank results are validation
analyses of frozen trained heads. These scopes correspond to different claims;
they are presented together because each closes a different link in the same
chain. The numerical checks in Sections 3.3 and 5.2 and the table in Section
4.2 are simulations of idealized models. They verify the derivations and
illustrate magnitudes, and they are not dataset evidence. The baseline
architecture facts used in Sections 2–4 (positional embeddings on, RevIN
`affine=False`, final `norm2` followed by a linear predictor, $D\in\{4,8\}$)
were checked against `src/models/PhaseFormer.py`,
`src/models/phaseformer_presets.py`, and the `confirm_*_no_residual_*` run
configurations.

## Reference

[1] Niu et al. *PhaseFormer*. ICLR 2026. arXiv:2510.04134.
