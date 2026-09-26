# PhaseFormer-L: Conditional Level Adaptation Beyond Phase Tokenization

## Abstract

PhaseFormer represents recurring temporal structure by grouping observations at the
same phase across cycles. This representation is highly compact, but its success
on within-cycle shape changes does not imply that every cross-cycle level change
is fully used by the forecasting backbone. We study the residual gap that remains
after the phase path has made its prediction. The key point is conditional: the
phase path can already explain part of a level change, while a low-dimensional
component of the level trajectory can remain predictable in the phase-conditioned
residual.

We show that an additive cross-cycle level perturbation contributes at most one
new direction outside the phase subspace. Under a persistent-level approximation,
the corresponding optimal correction is rank one in prediction-weighted norm:
it reads a recent level state and writes a horizon-wide displacement. We then
test this statement with a large, deliberately unstructured anchored linear
branch. Across 28 train/validation settings, its predictive spectrum has a
leading-mode share of 0.642--0.862 and needs only 2--7 directions for 90% of
the attainable linear reduction. The same conclusion appears in jointly trained
heads: all 18 analyzed model-setting groups read a recent weighted level, while
12/18 write an overall displacement and the Weather groups use a smooth tilt or
curvature correction. A functional decomposition of 72 frozen checkpoints shows
that three to five canonical modes recover at least 95% of the fitted branch's
own improvement. Mode deletion is a model-internal intervention: semantic modes
are more damaging to the fused forecast than same-dimensional random controls,
even when deleting them improves the branch in isolation.

The resulting PhaseFormer-L is a gated anchored linear corrector trained jointly
with the phase backbone. In the principal three-seed comparison it improves both
MSE and MAE over the matched phase-only baseline on all 12 settings of ETTh2,
ETTm2, and Weather, with an MSE reduction of up to 8.73%. Generic temporal
factorization then compresses the correction head to 3.5--6.1% of its dense-head
parameters while retaining 92.4--101.9% of its attainable branch value on the
most aggressive exploratory compression. The evidence supports a journal-level
extension of PhaseFormer organized around a conditional, low-dimensional
complement rather than a claim that phase tokenization is unable to model level
changes.

## 1. Introduction

Periodic signals vary along at least two related axes. Their shape within a
cycle may change, and the level around which that shape evolves may move from
one cycle to the next. PhaseFormer focuses on the first axis by collecting values
with the same phase across cycles. This makes recurring shape changes easy to
represent and explains why a small phase-domain backbone can forecast long
horizons efficiently.

The practical question for a journal extension is more precise than asking
whether PhaseFormer can model level changes at all. In real data, the phase path
does capture part of a cross-cycle level change. The question is whether a
predictable component remains after that path has made its forecast, and whether
the remaining component has a simple structure that can be measured and used.

We answer this question with a single chain of evidence. First, a geometric
calculation identifies the dimension of the residual direction introduced by an
additive cross-cycle level perturbation. Second, a full-history linear branch is
used as a high-capacity probe and as a candidate corrector. Its reduced-rank
spectrum measures how many input-output directions can actually buy prediction
error reduction. Third, canonical modes of jointly trained heads identify what
the branch reads and writes. Finally, mode-level interventions test whether those
identified directions matter to the fused PhaseFormer forecast, rather than only
to the branch considered in isolation. The resulting functional rank motivates
low-rank implementation.

The central claim is therefore:

> After conditioning on the phase prediction, the useful complement associated
> with cross-cycle level evolution is usually low-dimensional. Its dominant
> mechanism is a recent weighted level readout followed by a persistent or smooth
> low-frequency correction over the horizon.

This claim has three separate meanings that we keep distinct throughout the
paper:

1. **Geometric dimension:** an additive level perturbation can add at most one
   direction outside the original phase subspace.
2. **Predictive dimension:** a small number of input-output modes account for
   most of the attainable linear reduction for a chosen residual target.
3. **Functional dimension:** a small number of canonical modes account for most
   of the improvement realized by a trained fused model.

The distinction prevents a rank-one geometric perturbation from being mistaken
for a universal rank-one architecture. It also makes the mechanism auditable:
the mode's input pattern, output pattern, contribution, and intervention effect
can be reported in the same analysis.

## 2. Conditional gap after phase modeling

### 2.1 Cross-cycle level perturbation

Arrange a length-$L$ input into $X\in\mathbb R^{K\times P}$, where rows index
cycles and columns index phase positions. Let $\mathcal S$ denote the column
space used by the phase representation. A cycle-dependent level change has the
form

\[
X_\ell = X + \ell\mathbf 1_P^\top,
\qquad \ell\in\mathbb R^K .
\tag{1}
\]

Decompose the level trajectory into a component already represented by the phase
path and a complementary component,

\[
\ell=\ell_{\parallel}+\ell_{\perp},
\qquad
\ell_{\perp}\perp\mathcal S .
\tag{2}
\]

The phase predictor can use information in $\ell_{\parallel}$; the residual
question concerns $\ell_{\perp}$. Window centering removes the common mean of
$\ell$, but does not remove changes of level between cycles.

**Proposition 1 (conditional one-direction complement).** Let
$P_{\mathcal S}$ be the orthogonal projector onto the phase subspace. Then

\[
(I-P_{\mathcal S})X_\ell
  =(I-P_{\mathcal S})X+(I-P_{\mathcal S})\ell_\perp\mathbf 1_P^\top,
\tag{3}
\]

and the new term has rank at most one. Hence, even when the phase path already
models a nonzero part of the level change, the part that is geometrically outside
the phase subspace occupies at most one additional direction. If the projected
component is nonzero, the complement is exactly one-dimensional.

This is a statement about the geometry of the residual representation, not a
claim that PhaseFormer is blind to all level variation. It isolates the part
that the phase path does not explain.

### 2.2 From a scalar residual state to a rank-one correction

Let $\hat y_\phi$ be the phase prediction and define the conditional residual

\[
D_{\mathrm{cond}}=y-\hat y_\phi .
\tag{4}
\]

Suppose that the predictable part of this residual is driven by a scalar level
state $\delta$,

\[
D_{\mathrm{cond}}=a\delta+\varepsilon,
\tag{5}
\]

where $a\in\mathbb R^H$ is the future expression of that state and
$\varepsilon$ contains the remaining residual. For a persistent level offset,
$a$ is close to the all-ones vector; for a slowly evolving level, $a$ is a
smooth tilt or curvature.

For branch input $Z$, the least-squares linear predictor is

\[
W^*=\mathbb E[D_{\mathrm{cond}}Z^\top]
       \mathbb E[ZZ^\top]^{-1}
     =ab^\top+R,
\tag{6}
\]

with $b^\top=\mathbb E[\delta Z^\top]\mathbb E[ZZ^\top]^{-1}$ and
$R=\mathbb E[\varepsilon Z^\top]\mathbb E[ZZ^\top]^{-1}$. The prediction
error gap between the rank-one part and the full linear optimum is bounded by

\[
\mathcal R(ab^\top)-\mathcal R(W^*)
 =H^{-1}\|R\Sigma^{1/2}\|_F^2
 \le H^{-1}\mathbb E\|\varepsilon\|_2^2,
\tag{7}
\]

where $\Sigma=\mathbb E[ZZ^\top]$. Thus a rank-one correction is justified when
the predictable conditional residual is dominated by one scalar state; the
remainder determines whether additional modes are needed.

The form of $b$ is data dependent. Under a persistent or autoregressive level,
the best linear estimate of the current state gives a recency-weighted kernel,
often close to an exponential moving average. The theory therefore predicts a
family of recent-level readouts, not one fixed hand-designed filter.

### 2.3 Measuring the right residual

Two residual targets are useful for different questions. The independent target
$D_{\mathrm{ind}}=y-x_{\mathrm{last}}\mathbf 1_H$ measures what a linear branch
can add over last-value persistence. The conditional target in (4) measures
what remains after the phase backbone has acted. The first target is used for the
broad closed-form spectrum because it is available uniformly across the 28
settings. The second target defines the mechanism we seek in the fused model.

The experiments below connect them in sequence: the independent spectrum
establishes concentration and the likely mode family; the trained-head analysis
and interventions establish which part is actually used relative to the phase
backbone.

## 3. PhaseFormer-L

### 3.1 Anchored temporal corrector

The probe is a full-history anchored linear branch. For a normalized input
$x_n=(x-\mu)/\sigma$, let $x_{n,L}$ be the last input value and define

\[
z=x_n-x_{n,L}\mathbf 1_L,
\qquad
\hat y_r=\sigma\big(Wz+x_{n,L}\mathbf 1_H+c\big)+\mu\mathbf 1_H .
\tag{8}
\]

The phase path and the temporal branch are fused by a learned gate,

\[
\hat y=(1-g)\odot\hat y_\phi+g\odot\hat y_r,
\qquad g=\operatorname{sigmoid}(\gamma).
\tag{9}
\]

The backbone, branch, and gate are trained jointly. Because the normalization
scale cancels in the linear term, the branch is an anchored map of the original
history, $W(x-x_L\mathbf 1_L)$, plus its anchor. This makes it a useful probe:
it is expressive enough to reveal the missing temporal coordinates before any
low-rank restriction is imposed.

### 3.2 Canonical modes and intervention

For a factorized head $W=VU$, we analyze the effective map rather than arbitrary
hidden coordinates:

\[
W=\sum_i s_i u_i v_i^\top .
\tag{10}
\]

Mode $i$ reads $v_i^\top z$ and writes the horizon shape $u_i$. We match input
vectors to a fixed dictionary of recent level, local trend, curvature, periodic
shape, and change-point templates. Output vectors are matched to displacement,
tilt, curvature, and periodic correction templates.

The intervention keeps the trained parameters and phase input fixed while
removing a selected input subspace or a canonical mode from the temporal branch.
We report both branch error and fused error. If removing a mode improves the
branch but worsens the fused forecast, the mode is useful because it complements
the phase path; its value cannot be judged from branch-only accuracy.

### 3.3 Functional contribution

For a frozen checkpoint, the mode increment $d_i$ changes the fused prediction
additively. The exact MSE change from deleting a set of modes can therefore be
computed from the individual contributions $I_i$, with an algebraic forward
check. We define the functional rank $r_{95}$ as the smallest number of modes,
ordered by $I_i$, that recovers 95% of the checkpoint's own improvement over a
zeroed-map baseline. This is a realized model quantity, distinct from the
independent regression spectrum.

## 4. Selected evidence

We use a lookback of 720 and horizons $H\in\{96,192,336,720\}$. Main model
comparisons use three seeds and the matched phase-only baseline. The analyses
below intentionally report compact witness groups: each group answers one
question and avoids mixing incompatible model-selection records.

### 4.1 The correction is useful in a coherent adaptation regime

The strongest matched comparison is the 12-setting group consisting of all four
horizons of ETTh2, ETTm2, and Weather. PhaseFormer-L improves both MSE and MAE
at all 12 settings in the three-seed comparison. MSE reductions range from
0.52% to 8.73%; the six settings added in the final training expansion also
improve both metrics. This establishes that the branch is not merely a
single-horizon or single-dataset effect and supplies the performance anchor for
the mechanism analysis.

| witness group | settings | dual-metric wins | largest MSE reduction |
|---|---:|---:|---:|
| ETTh2 + ETTm2 + Weather, $H\in\{96,192,336,720\}$ | 12 | 12/12 | 8.73% |

The result should be read as evidence for a useful adaptation regime: the branch
is a compact complement in these settings. It is not needed to assume that the
phase path fails to model level changes in order to explain this gain.

### 4.2 The attainable correction has a concentrated spectrum

Closed-form reduced-rank regression was computed on 28 settings using the
last-value residual target. Across these settings, the leading mode accounts for
64.2%--86.2% of the total attainable linear reduction, and 2--7 directions
capture 90%. The leading output direction has cosine 0.890--0.989 with a
horizon-wide constant vector. On ETTh2 and ETTm2, the input mode is consistently
matched to a recency-weighted level kernel; Weather has a larger share of smooth
tilt and curvature.

| quantity | observed range | interpretation |
|---|---:|---|
| $\lambda_1/\sum_i\lambda_i$ | 0.642--0.862 | one mode dominates attainable linear value |
| directions for 90% value | 2--7 | the useful complement is low-dimensional |
| $|\cos(u_1,\mathbf 1_H)|$ | 0.890--0.989 | the main output is a persistent displacement |

This is the first quantitative bridge from Proposition 1 to a forecasting
mechanism. It does not say that all residuals are one-dimensional; it says that
the predictable part purchased by a linear corrector is strongly concentrated.

As a direct check that this spectrum remains relevant after phase modeling, we
also projected the same training examples onto the phase-conditioned target in
(4). On the six primary settings carried into the head analysis, the first input
directions obtained from $D_{\mathrm{cond}}$ and from the independent target
have absolute cosine at least 0.9991. The conditional analysis therefore points
to the same recent-level coordinate in the settings where the trained correction
is subsequently dissected. This is the bridge from a convenient persistence
reference to the actual phase-conditioned complement; it is not an assumption
that the two residual targets are identical in every dataset.

### 4.3 Which modes are actually used?

The trained-head dissection covers six principal settings, three seeds, and three
head capacities, giving 18 model-setting groups. In every group the leading
input mode is matched to a recent weighted level, with explanation scores
0.57--0.73. In 12/18 groups, all from ETTh2 and ETTm2, the leading output is an
overall displacement with explanation scores 0.91--0.97. The six Weather groups
use smooth curvature or tilt instead, showing how the same state readout can be
expressed differently when the future level evolves within the horizon.

| head-side pattern | groups | explanation score |
|---|---:|---:|
| recent weighted level input | 18/18 | 0.57--0.73 |
| level → horizon-wide displacement | 12/18 | 0.91--0.97 |
| level → Weather tilt/curvature | 6/18 | smooth low-frequency output |

The canonical functional-rank analysis uses 72 frozen checkpoints (six settings,
four nominal compression levels, three seeds). Three to five modes recover at
least 95% of each checkpoint's own fitted improvement. A representative ETTh2-
720 rank-$H/8$ checkpoint has a first-mode contribution of 75.9%; its first five
modes sum to 95.3%, decomposing into a dominant level displacement and smaller
periodic or higher-order corrections. Thus the important object is not merely a
large matrix with a small singular tail: it is a small set of nameable read-write
mechanisms.

| canonical mode group (ETTh2-720 witness) | cumulative contribution |
|---|---:|
| recent weighted level → displacement (mode 1) | 75.9% |
| periodic-shape corrections (modes 2–3) | 15.0% |
| higher-order level/curvature corrections (modes 4–5) | 4.3% |
| first five modes together | 95.3% |

### 4.4 Intervention establishes complementarity

We next remove the semantic input subspace while keeping the phase path and all
trained weights fixed. The same-dimensional random RRR subspace is the negative
control. In all 18 analyzed groups, semantic deletion produces a fused MSE
change outside the 95% random-control interval. Across the underlying 72
checkpoint-level interventions, deleting the semantic subspace improves the
branch's own MSE in 52/72 cases but worsens the fused MSE in 72/72 cases; the
median relative fused increase is 30.43%.

This is the key causal link in the paper's narrow, model-internal sense. The
semantic mode is not merely correlated with the residual target: changing that
mode while holding the partner path fixed changes the fused forecast in a
systematic, control-separated direction. At the same time, the branch-versus-
fusion reversal shows why mode contribution must be defined relative to the
PhaseFormer backbone. The intervention is a counterfactual of the fitted model;
it is not a claim that the data-generating process has been causally identified.

### 4.5 From mode evidence to a compact implementation

The concentration results motivate generic temporal factorization,
$W=VU$, while leaving the input and output coordinates trainable. The most
aggressive exploratory compression uses only 3.5--6.1% of the dense-head
parameters and retains 92.4--101.9% of the dense branch's attainable reference
improvement across the reported witness settings. In the broader 24-setting
comparison, a rank-$H/8$ head reduces the total model to 16.4--33.0% of the
dense model, with a mean absolute relative MSE difference of 0.72% from the
dense PhaseFormer-L.

| compression view | retained capacity | evidence |
|---|---:|---|
| branch-only exploratory $q=1/32$ | 3.5--6.1% of dense head | 92.4--101.9% attainable value |
| full-model rank-$H/8$ | 16.4--33.0% of dense model | 0.72% mean absolute relative MSE difference |

The design implication is to choose rank from predictive accumulation and the
desired accuracy-capacity point, rather than to hard-code a single rank-one
filter. The rank-one theory explains why strong compression is possible; the
secondary functional modes determine how much accuracy is retained.

## 5. Discussion: the evidence chain

The experiments form a deliberately ordered audit chain.

1. **Geometry:** equation (3) says that the component of a cross-cycle level
   trajectory outside the phase subspace adds at most one direction.
2. **Predictive value:** the closed-form spectrum measures whether that direction
   is forecastable and shows that one mode buys 64.2--86.2% of attainable linear
   value, with a horizon-wide output.
3. **Mode identity:** trained heads read a recent weighted level in all 18
   groups. Their outputs are displacement on ETT data and smooth tilt/curvature
   on Weather.
4. **Mode contribution:** functional rank shows that three to five canonical
   modes recover at least 95% of the fitted head's own gain.
5. **Causal relevance to fusion:** semantic deletion is more damaging than
   matched random deletion, and it can improve the branch while degrading the
   fused model. The value is therefore conditional on the phase backbone.
6. **Optimization:** generic low-rank factorization preserves most of the useful
   correction at a small parameter fraction.

These steps also clarify three percentages that should not be conflated:

- **Model-level gain:** PhaseFormer-L versus phase-only, such as the 12/12
  dual-metric wins and the 8.73% maximum MSE reduction.
- **Branch-level capture:** the fraction of a linear branch's attainable or
  fitted improvement retained by a low-rank head.
- **Mode-level contribution:** the exact frozen-checkpoint contribution $I_i$
  and its cumulative functional rank.

The result is a conditional completion of PhaseFormer's representation. The phase
backbone remains responsible for recurring shape and for the part of level
variation it can already predict. The anchored linear branch supplies a compact
state-adaptation coordinate for what remains. Its dominant mechanism is
recent-level input to a persistent or smooth low-frequency output, while the
additional modes absorb dataset-specific evolution.

## 6. Conclusion

PhaseFormer's phase representation does not need to be treated as incapable of
modeling cross-cycle level changes. The relevant journal question is the residual
one: after the phase path has acted, is there a predictable complement with a
simple geometry and a measurable functional role? Our answer is yes in the
principal adaptation regime.

An additive level perturbation contributes at most one new phase-complement
direction. Reduced-rank analysis, trained-head mode semantics, functional
contribution, and controlled deletion all point to the same mechanism: a recent
weighted level state is converted into a horizon-wide displacement or a smooth
low-frequency correction. The correction is valuable because it complements the
phase path, not because it is a universally superior standalone predictor.

PhaseFormer-L turns this finding into a practical extension: a gated anchored
linear corrector whose capacity can be reduced according to predictive value.
The resulting narrative is both mechanistic and empirical—geometry identifies
the possible gap, spectra quantify its size, modes name its content, interventions
establish its dependence on the backbone, and low-rank factorization provides the
corresponding efficiency path.

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
mechanistic chain.

## Reference

[1] Niu et al. *PhaseFormer*. ICLR 2026. arXiv:2510.04134.
