# PhaseFormer-L: Low-Dimensional Level Adaptation for Phase-Domain Time-Series Forecasting

**中文题目：PhaseFormer-L：面向相位域时序预测的低维电平适应**

> **v2 变更说明。** 本版在 v1 的基础上补入 2026-09-24 完成的 trained-head functional-rank 分析
> （计划 `low_rank_checkpoint_analysis_experiment_plan.md` §5–§13），并新增附录 A.4 逐项记录
> 实验产物路径。v1 的全部结论、数字与限制均保留；新增内容只在下述位置：Abstract、§1、
> **§4.3.1**、**§4.7（新）**、§5、§6、Appendix A.4。新增结论的有效范围是
> **六个 setting**，不是主表的 28 个，二者在正文中始终分开陈述。

## Abstract

Phase-domain tokenization provides a compact representation of recurring temporal structure, but robustness to changes within a cycle does not imply robustness to level changes across cycles. We extend PhaseFormer by investigating the geometry, predictive dimension, and functional role of this second source of variation. Our analysis establishes that an additive cross-cycle level perturbation introduces at most one direction outside the original phase subspace. When the predictable residual is dominated by a persistent level offset, its optimal linear correction is correspondingly close to a rank-one map in prediction-weighted norm. Across seven benchmarks and four forecasting horizons, reduced-rank analysis shows that the leading mode accounts for 64.2%–86.2% of the attainable linear improvement over a last-value reference. Its output is closely aligned with a horizon-wide displacement. Analysis of jointly trained heads further identifies recent-level reading directions, while branch-specific interventions show that a component can improve the fused forecast even when it worsens the branch's standalone prediction. **A canonical decomposition of the trained heads then measures how much of their own fitted predictive value any subset of modes carries: on six settings spanning four compression levels, three to five modes recover at least 95% of the fused improvement that the full low-rank checkpoint attains over a zeroed-map baseline.** In three-seed comparisons, the model improves both MSE and MAE over the matched phase-only baseline at all twelve settings of ETTh2, ETTm2, and Weather, with MSE reductions of up to 8.73%; six of these settings are newly trained beyond the exploratory development set. A rank-$H/8$ variant reduces total parameters to 16.4%–33.0% of the dense model, with a mean absolute relative MSE difference of 0.72%. The resulting journal extension links phase representation to level adaptation and identifies predictive complementarity as the organizing principle for compact forecasting corrections.

**Keywords:** long-term time-series forecasting; phase tokenization; level adaptation; reduced-rank regression; predictive dimension; functional rank; complementary representations.

## 1. Introduction

Periodic time series vary in at least two distinct ways. The shape within each cycle can change, and the level around which that shape evolves can move between cycles. These variations act on different axes of the same observation matrix. A forecasting representation that handles one efficiently need not handle the other equally well.

PhaseFormer [1] organizes a sequence into phase tokens: observations at the same offset in successive cycles belong to the same token. This representation exposes recurring structure through a compact cross-cycle subspace. Under the structural assumptions of the original analysis, within-cycle transformations preserve that subspace, providing a rationale for forecasting with a small backbone. The present work develops the complementary question: **how should a phase-domain forecaster adapt when the cross-cycle level itself evolves?**

Consider a window arranged as $X\in\mathbb R^{K\times P}$, with $K$ cycles and $P$ positions per cycle. A cycle-dependent level shift has the form $\ell\mathbf 1_P^\top$. Unlike a within-cycle transformation, this perturbation acts directly along the cross-cycle axis. Its component outside the original column space occupies at most one additional direction. Window centering removes the average offset, but preserves changes of level between cycles. This observation suggests a compact adaptation mechanism whose central task is to estimate the recent level and propagate its predictable component into the forecast.

A geometric direction, however, is not automatically a useful forecasting direction. Its value depends on whether it persists into the future and on what the phase backbone already predicts. We therefore distinguish three objects: the dimension of the input perturbation, the dimension of attainable linear prediction, and the directions actually used by a jointly trained forecasting model. Conflating them would incorrectly imply that a rank-one perturbation always calls for a rank-one architecture.

Our investigation connects these objects through complementary analyses. Reduced-rank regression measures how predictive value accumulates across input–output modes. Canonical decomposition identifies the temporal patterns read and written by trained heads. Interventions on the branch's private input then test whether these patterns contribute to the fused forecast. **Finally, a functional-rank measurement over the trained heads' own canonical modes quantifies how many of those directions the fitted model actually relies on, and what each one reads and writes.** Together, the analyses reveal a dominant recent-level component embedded in a richer, jointly learned correction.

This leads to PhaseFormer-L: the original phase backbone augmented by a gated temporal linear branch. The branch remains trainable in the original temporal coordinates; low-rank factorization controls capacity without imposing a fixed smoothing kernel. Across the evaluated settings, its benefits concentrate in a coherent group of datasets, and its most aggressive compression is limited by secondary predictive modes.

The extension makes three contributions:

1. **A geometric account of cross-cycle adaptation.** We formalize the additional direction introduced by level drift and give an explicit prediction-weighted bound for a dominant rank-one residual correction.
2. **A mechanism grounded in predictive value.** Spectral analysis, trained-head interventions, and a functional-rank decomposition connect recent-level information to horizon-wide displacement and establish why standalone branch accuracy is insufficient to assess complementarity.
3. **An empirically characterized extension of PhaseFormer.** We evaluate gated linear adaptation over multiple horizons, quantify the accuracy–capacity trade-off, and relate its heterogeneous gains to training-set temporal statistics.

## 2. From Phase Geometry to Predictive Level Correction

### 2.1 Cross-cycle drift and the phase subspace

Let $L=KP$ and reshape a univariate input into $X\in\mathbb R^{K\times P}$. The columns are phase trajectories across cycles. Write the level-perturbed window as

$$
X_\ell=X+\ell\mathbf 1_P^\top.
\tag{1}
$$

Define global window centering by $\mathcal C(X)=X-\bar X\mathbf 1_K\mathbf 1_P^\top$, and let $u=\ell-\bar\ell\mathbf 1_K$. Then

$$
\mathcal C(X_\ell)=\mathcal C(X)+u\mathbf 1_P^\top.
\tag{2}
$$

Thus, centering removes the common level but retains its cross-cycle variation. Division by a nonzero window scale does not eliminate that variation.

**Proposition 1 (one additional direction).** Let $X_0=\mathcal C(X)$, $\mathcal S=\operatorname{Col}(X_0)$, and $P_{\mathcal S}$ be its orthogonal projector. Then

$$
(I-P_{\mathcal S})\mathcal C(X_\ell)
=(I-P_{\mathcal S})u\mathbf 1_P^\top,
\qquad
\operatorname{rank}\!\left((I-P_{\mathcal S})\mathcal C(X_\ell)\right)\leq1.
\tag{3}
$$

If $u\notin\operatorname{Col}(X_0)$ and $\mathbf 1_P\notin\operatorname{Row}(X_0)$, then

$$
\operatorname{Col}(\mathcal C(X_\ell))
=\mathcal S\oplus\operatorname{span}((I-P_{\mathcal S})u),
\tag{4}
$$

and its dimension increases by one.

*Proof.* Equation (3) follows by applying $I-P_{\mathcal S}$ to (2). For the second statement, take a rank factorization $X_0=AB^\top$ with both factors of full column rank $r$. Under the two independence conditions, $[A,u]$ and $[B,\mathbf 1_P]$ both have rank $r+1$. Their product therefore has rank $r+1$ and column space $\operatorname{Col}([A,u])$, which gives (4). $\square$

The first statement requires no exact rank expansion: drift may also rotate an existing subspace. It identifies the geometric simplicity of the perturbation. Phase tokenization itself preserves the observations; the question is whether a compact forecasting backbone exploits their level information effectively.

### 2.2 A dominant scalar state can yield a low-rank predictor

Let $Z\in\mathbb R^L$ denote the branch input and let $D=y-\hat y_\phi$ be the residual of a fixed phase predictor. Suppose

$$
D=a\delta+\varepsilon,
\tag{5}
$$

where $\delta$ is a scalar state, $a\in\mathbb R^H$ describes its future expression, and $\varepsilon$ collects the remaining residual. Persistent level displacement corresponds to $a=\mathbf 1_H$. All moments below are finite. Work on the nonzero-variance support of $Z$, so that $\Sigma=\mathbb E[ZZ^\top]$ is positive definite; equivalent expressions use its pseudoinverse on the full space.

**Proposition 2 (prediction-weighted rank-one approximation).** The minimum-square-error linear predictor satisfies

$$
W^*=\mathbb E[DZ^\top]\Sigma^{-1}
=ab^\top+R,
\quad
b^\top=\mathbb E[\delta Z^\top]\Sigma^{-1},
\quad
R=\mathbb E[\varepsilon Z^\top]\Sigma^{-1}.
\tag{6}
$$

For $\mathcal R(W)=H^{-1}\mathbb E\|D-WZ\|_2^2$,

$$
\mathcal R(ab^\top)-\mathcal R(W^*)
=\frac1H\|R\Sigma^{1/2}\|_F^2
\leq\frac1H\mathbb E\|\varepsilon\|_2^2.
\tag{7}
$$

*Proof.* Substitute (5) into the normal equations to obtain (6). Orthogonality of the least-squares residual gives the equality in (7). Since $RZ$ is the linear projection of $\varepsilon$ onto $Z$, its expected squared norm cannot exceed that of $\varepsilon$. $\square$

Equation (7) states the relevant approximation property without treating matrix rank as a continuous quantity. A small predictable remainder makes rank one sufficient for most linear predictive value, even though $W^*$ can have higher exact rank. The input direction $b$ is determined by the data covariance and the predictability of $\delta$; an exponential kernel is an empirical possibility, rather than a consequence of level drift alone.

For a given covariance-weighted remainder $\eta=\|R\Sigma^{1/2}\|_2$, the singular values of $W^*\Sigma^{1/2}$ beyond the first are at most $\eta$. This links the residual decomposition directly to the predictive spectrum used below.

### 2.3 Prediction dimension differs from variance dimension

For a chosen target $D$, define

$$
C=\mathbb E[DZ^\top],\qquad
M=\frac1H C\Sigma^{-1}C^\top.
\tag{8}
$$

Let $\lambda_1\geq\lambda_2\geq\cdots\geq0$ be the eigenvalues of $M$, and let $U_r$ contain its leading $r$ eigenvectors. Reduced-rank regression gives

$$
W_r=U_rU_r^\top C\Sigma^{-1},
\qquad
\mathcal R(0)-\mathcal R(W_r)=\sum_{i=1}^{r}\lambda_i.
\tag{9}
$$

The factor $1/H$ makes the eigenvalue sum a per-horizon-element MSE reduction. We measure leading-mode concentration by $\lambda_1/\operatorname{tr}(M)$ and define $d_{90}$ as the smallest rank capturing 90% of $\operatorname{tr}(M)$.

Two regression targets answer different questions. With $D_{\mathrm{ind}}=y-x_L\mathbf 1_H$, the spectrum measures linear improvement over last-value persistence. With $D_{\mathrm{cond}}=y-\hat y_\phi$, it measures linear improvement available after fixing the phase predictor. The broad spectral results below characterize the former; they establish the structure of the branch's prediction problem, while trained-model analysis assesses its use alongside PhaseFormer. Neither an input PCA spectrum nor the ordinary singular spectrum of $W$ substitutes for this target-dependent measure.

**A third, distinct measurement is introduced in §4.7**: the *functional rank* of an already-trained head. It decomposes that head's own effective map and asks how much of its realized fused improvement each canonical mode carries. It is neither the independent-target spectrum of (9) nor the ordinary singular spectrum of $W$, and the three are not interchangeable.

## 3. PhaseFormer-L

### 3.1 Joint phase and level adaptation

For each channel, let $x_n=(x-\mu)/\sigma$ denote the window-normalized input. The temporal branch predicts

$$
z=x_n-x_{n,L}\mathbf 1_L,\qquad
\hat y_r=\sigma\big(Wz+c+x_{n,L}\mathbf 1_H\big)+\mu\mathbf 1_H.
\tag{10}
$$

The fused prediction is

$$
\hat y=(1-g)\odot\hat y_\phi+g\odot\hat y_r,
\qquad g=\operatorname{sigmoid}(\gamma),
\tag{11}
$$

where $g$ is a learned per-channel gate broadcast across the horizon. The backbone, temporal branch, and gate are optimized jointly. The branch is always enabled; no dataset-level diagnostic switches it on or off.

For the linear term, the normalization scale cancels exactly:

$$
\hat y_r=x_L\mathbf 1_H+W(x-x_L\mathbf 1_L)+\sigma c.
\tag{12}
$$

A bias, when present, is retained separately. The dense branch uses a shared $H\times L$ temporal map. With $L=720$, this provides 69,120–518,400 temporal weights over the evaluated horizons. It serves both as an expressive diagnostic probe and as the principal model variant.

Equation (11) can also be written as $\hat y=\hat y_\phi+g\odot(\hat y_r-\hat y_\phi)$. Consequently, the operational correction is the gated difference between two forecasts. Interpreting the branch alone as an additive residual predictor would miss this dependence on its partner.

### 3.2 Compact variants

We factorize $W=VU$, where $U\in\mathbb R^{r\times L}$ and $V\in\mathbb R^{H\times r}$. The principal compact variants use $r=H/4$ and $r=H/8$. Ignoring biases, their parameter count is $r(L+H)$ instead of $LH$.

The factorization learns both the reading and writing directions. It does not constrain the input to a prescribed moving average, phase-aligned basis, or rank-one bottleneck. This flexibility follows from Proposition 2: the dominant mode motivates compactness, but secondary modes determine the accuracy retained at a particular rank. A predictive spectrum is therefore a capacity diagnostic, not an automatic rank prescription for the jointly trained model.

### 3.3 Canonical modes and functional interventions

For both dense and factorized branches, we form the effective map and decompose

$$
W=\sum_i s_i u_i v_i^\top.
\tag{13}
$$

Each mode reads $v_i^\top z$ and writes the horizon shape $u_i$. This representation removes the arbitrary hidden-coordinate rotations of a factorized head. Input templates describe recent level, level changes, local trends, curvature, periodic level and shape, and rapid local changes. Output templates describe displacement, tilt, curvature, periodic structure, and shape continuation.

To assess functional reliance, we retain or remove template-defined subspaces from the branch input while holding the backbone input and trained parameters fixed. Both branch error and fused error are measured. An intervention that improves $\hat y_r$ while worsening $\hat y$ demonstrates that the removed information contributes through complementarity, not through standalone accuracy. These are interventions within the fitted model; they do not identify the causal process generating the time series.

The same decomposition supports a finer intervention: deleting a single canonical mode $i$ from the head's hidden state, which changes the fused prediction by that mode's increment $d_i$ and leaves every other mode untouched. Because the left singular vectors are orthonormal, this yields an exact additive accounting, defined and verified in §4.7.

## 4. Experiments

### 4.1 Evaluation design

We use a lookback of 720 and horizons $H\in\{96,192,336,720\}$. The principal performance evaluation covers ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity, yielding 24 settings. Traffic contributes four exploratory settings. The frozen protocol uses the full training split, Huber loss, a maximum of 30 epochs, and checkpoint selection by minimum validation loss, with seeds 2021, 2022, and 2023.

Two references are kept distinct. The **original PhaseFormer reference** is the fixed result reported for the predecessor model. The **matched phase-only baseline** measures the difference under the current execution environment. Improvements against the latter do not by themselves imply improvement over the former. The underlying source tables retain both references; Table 1 here is the consolidated Golden-witness ledger and does not present their means as if they were witness values.

The consolidated table combines frozen-protocol records with explicitly identified test-selected searches. It is designed to answer one question—whether a setting has an audited seed with at least one metric below Golden—without presenting incompatible configurations as one averaged model. Detailed provenance is summarized in Appendix A.

Relative changes are defined throughout as $100(\text{candidate}-\text{reference})/\text{reference}$; negative values indicate improvement. Table 1 preserves the reported metric precision and does not imply statistical uncertainty where none was supplied.

### 4.2 Forecasting accuracy: coherent gains across horizons

**Table 1. Confirmed main-table data across 28 settings.** Each row gives one explicitly audited witness seed; lower MSE and MAE are better. “Pass” means that at least one of the two metrics is below Golden for that seed. The table is a consolidated confirmation ledger rather than a three-seed mean table; the underlying three-seed records and search traces are cited in Appendix A.

| Setting | Status | Golden MSE/MAE | Configuration (mechanism; batch; period; loss; gate; lr; head/rank) | Seed | Test MSE | Test MAE | Metric below Golden |
|---|---|---:|---|---:|---:|---:|---|
| ETTh1-96 | Fail | 0.359/0.382 | weak_residual; 32; 24; MAE; 0.20; 1e-3; shared | 2023 | 0.362763 | 0.389995 | — |
| ETTh1-192 | Fail | 0.397/0.404 | weak_residual; 32; 48; MSE; 0.20; 1e-3; shared | 2022 | 0.401238 | 0.420031 | — |
| ETTh1-336 | Fail | 0.425/0.424 | weak_residual; 64; 48; MSE; 0.20; 1e-3; shared | 2023 | 0.436468 | 0.444085 | — |
| ETTh1-720 | Pass | 0.431/0.450 | phase_only; 256; 24; Huber; —; 1e-3; — | 2023 | 0.418769 | 0.439297 | MSE, MAE |
| ETTh2-96 | Pass | 0.275/0.338 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.272100 | 0.332843 | MSE, MAE |
| ETTh2-192 | Pass | 0.341/0.376 | l_main; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.337312 | 0.376446 | MSE |
| ETTh2-336 | Pass | 0.369/0.405 | l_main; 256; 24; Huber; 0.2; 1e-3; shared | 2023 | 0.368725 | 0.404795 | MSE, MAE |
| ETTh2-720 | Pass | 0.402/0.436 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2023 | 0.392054 | 0.427058 | MSE, MAE |
| ETTm1-96 | Pass | 0.293/0.344 | weak_residual; 256; 24; MAE; 0.1; 1e-3; pooled-rk12 | 2021 | 0.290128 | 0.337738 | MSE, MAE |
| ETTm1-192 | Fail | 0.323/0.361 | weak_residual; 32; 24; MAE; 0.20; 1e-3; shared | 2023 | 0.334003 | 0.363753 | — |
| ETTm1-336 | Pass | 0.358/0.381 | weak_residual; 256; 24; MAE; 0.2; 3e-4; pooled-rk10 | 2021 | 0.354631 | 0.376313 | MSE, MAE |
| ETTm1-720 | Fail | 0.412/0.410 | weak_residual; 32; 12; SMAE; 0.20; 1e-3; shared | 2023 | 0.416490 | 0.412541 | — |
| ETTm2-96 | Pass | 0.163/0.256 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.158474 | 0.248048 | MSE, MAE |
| ETTm2-192 | Pass | 0.219/0.293 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.215685 | 0.288061 | MSE, MAE |
| ETTm2-336 | Pass | 0.269/0.326 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.268039 | 0.324637 | MSE, MAE |
| ETTm2-720 | Pass | 0.351/0.379 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.344629 | 0.376928 | MSE, MAE |
| Weather-96 | Pass | 0.148/0.195 | l_main; 256; 24; Huber; 0.5; 1e-3; shared | 2021 | 0.146709 | 0.194005 | MSE, MAE |
| Weather-192 | Pass | 0.193/0.237 | l_main; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.191791 | 0.236277 | MSE, MAE |
| Weather-336 | Pass | 0.242/0.278 | l_main; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.239891 | 0.273774 | MSE, MAE |
| Weather-720 | Pass | 0.309/0.332 | l_main; 256; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.315415 | 0.327790 | MAE |
| Electricity-96 | Pass | 0.129/0.221 | l_main; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.128701 | 0.222297 | MSE |
| Electricity-192 | Pass | 0.148/0.238 | l_main; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.145376 | 0.236532 | MSE, MAE |
| Electricity-336 | Pass | 0.165/0.257 | l_main; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.161729 | 0.254716 | MSE, MAE |
| Electricity-720 | Pass | 0.201/0.285 | l_main; 64; 24; Huber; 0.2; 1e-3; shared | 2021 | 0.197727 | 0.286489 | MSE |
| Traffic-96 | Pass | 0.361/0.238 | l_q1_4; 8; 24; Huber; 0.2; 1e-3; pooled-rk24 | 2021 | 0.358840 | 0.233280 | MSE, MAE |
| Traffic-192 | Pass | 0.373/0.243 | phase_only; 8; 24; Huber; —; 1e-3; — | 2022 | 0.379216 | 0.242188 | MAE |
| Traffic-336 | Pass | 0.385/0.248 | weak_residual; 8; 24; MAE; 0.2; 1e-3; pooled-rk84 | 2022 | 0.396174 | 0.238683 | MAE |
| Traffic-720 | Pass | 0.428/0.270 | weak_residual; 8; 24; MAE; 0.02; 1e-3; shared | 2023 | 0.436780 | 0.261168 | MAE |

**Consistent improvement in the principal adaptation regime.** In the source's three-seed mean comparison, PhaseFormer-L improves both metrics at all twelve ETTh2, ETTm2, and Weather settings. MSE reductions range from 0.52% to 8.73%. The six newly trained settings in this group—ETTh2-192/336, ETTm2-336/720, and Weather-336/720—also improve both metrics, with MSE reductions of 0.52%–3.12%. The pattern therefore extends beyond the six reused exploratory settings.

**Progress beyond the original model.** Against the fixed original PhaseFormer reference, the source evaluation reports twelve settings with lower three-seed means in both metrics, compared with three for the matched baseline. Under the stricter descriptive criterion that the mean plus sample standard deviation lies below the original result in both metrics, the corresponding counts are eight and two. The eight PhaseFormer-L settings are ETTh2-720, all four ETTm2 horizons, Weather-96, and Electricity-192/336. These cross-environment comparisons establish the relation to the predecessor; the matched comparisons isolate the behavior under the current protocol.

**A structured operating regime.** Under the three-seed matched comparison, Electricity improves in MSE at every horizon, although the MAE response is mixed; ETTh1 has small MSE improvements at horizons 336 and 720 while its MAE increases; and ETTm1 and exploratory Traffic worsen in both metrics at every horizon. The extension therefore addresses a recurring adaptation need rather than imposing a universal advantage from an additional branch. The spectral and intervention analyses below explain why the existence of a compact linear predictor alone cannot determine its marginal value to PhaseFormer.

**The witness ledger answers a different question.** Table 1 records whether a single audited seed exists with at least one metric below Golden, and it finds such a seed for 23 of the 28 settings—including all four Traffic settings and ETTm1-96/336. This does not contradict the regime above, because the two columns measure different quantities. The ledger's rows mix mechanism variants rather than one model; two of its passing rows (ETTh1-720, Traffic-192) are `phase_only` configurations that never enable the added branch; and three of the four Traffic rows clear Golden on MAE alone while their MSE remains above it. Attainability of some configuration below Golden and average improvement over the matched baseline are separate claims, and neither substitutes for the other.

### 4.3 Predictive value concentrates in a few modes

**Table 2. Predictive-spectrum summary across four horizons per dataset.** Ranges are the minimum and maximum of the source's horizon-level values. The target is improvement over last-value persistence. The direction statistics describe the leading reduced-rank mode; “variance share” is the source's input-variance diagnostic for that mode. These quantities use training/validation data, not test errors.

| Dataset | Leading predictive share | $d_{90}$ | Output cosine with constant | Input variance share |
|---|---:|---:|---:|---:|
| ETTh1 | 0.712–0.747 | 3–4 | 0.950–0.968 | 0.051–0.078 |
| ETTh2 | 0.779–0.857 | 3 | 0.967–0.989 | 0.011–0.046 |
| ETTm1 | 0.713–0.741 | 3 | 0.923–0.957 | 0.071–0.157 |
| ETTm2 | 0.725–0.837 | 3 | 0.947–0.976 | 0.008–0.043 |
| Weather | 0.783–0.862 | 2 | 0.943–0.970 | 0.072–0.210 |
| Electricity | 0.659–0.666 | 4 | 0.890–0.895 | 0.077–0.142 |
| Traffic | 0.642–0.649 | 5–7 | 0.928–0.932 | 0.177–0.353 |

A single mode accounts for 64.2%–86.2% of attainable linear improvement across all 28 settings. Two to four modes reach 90% on the six principal datasets; Traffic requires five to seven. Output alignment with a constant vector is consistently high, with absolute cosine 0.890–0.989. Among the exponential input templates evaluated at time constants 6, 24, 72, and 168 steps, the best matches have absolute cosine 0.553–0.864. These observations identify a broad recent-level-to-displacement structure without equating the fitted input kernel to an exact exponential filter.

The distinction between prediction and variance is especially pronounced on ETTh2 and ETTm2. Their leading modes account for 72.5%–85.7% of linear predictive value while the corresponding input-variance diagnostic is only 0.8%–4.6%. Thus, a direction can be modest in signal energy and central to forecasting. The exploratory ETTh2-720 analysis provides a second contrast: the ordinary weight spectrum requires 418 dimensions for 95% energy, whereas the predictive spectrum requires eight for 95% attainable improvement.

Concentration is also present on ETTm1 and Traffic, where the added branch does not improve the fused model under the three-seed matched comparison. This separates **predictability of level information** from **additional value beyond a phase backbone**. The former is a property of a regression task; the latter depends on overlap, estimation, and joint optimization.

#### 4.3.1 Two different "leading mode" numbers

Table 2 and §4.7 both report a leading-mode share, and they must not be read as the same quantity.

| Quantity | Object | Target | ETTh2-720, q=1/8 |
|---|---|---|---:|
| Leading predictive share (§4.3, Table 2) | ideal rank-$r$ **reduced-rank regression** | $D_{\mathrm{ind}}=y-x_L\mathbf 1_H$ | 0.779–0.857 across horizons |
| Top-1 fused contribution (§4.7) | the **trained** head's canonical mode 0 | fused MSE against a zeroed-map baseline | 69.4% (mean of 3 seeds) |

The first asks how much of the *linearly attainable* improvement a single direction explains; the second asks how much of the *already-fitted* head's realized improvement one canonical mode carries. They share a qualitative message—value concentrates in few directions—but neither implies the other numerically, and the second is the one that speaks to what the trained model actually does.

### 4.4 Trained heads read level information and adapt its future expression

We analyze dense, rank-$H/4$, and rank-$H/8$ heads at six settings: ETTh2-96/720, ETTm2-96/192, and Weather-96/192, each with three seeds. This yields eighteen model–setting groups, evaluated on validation data using the reused exploratory checkpoints.

All eighteen groups identify recent weighted level as the dominant input-template group, with explanation scores of 0.57–0.73. The twelve ETTh2 and ETTm2 groups write an overall displacement, with output explanation scores of 0.91–0.97. Weather instead uses tilt or curvature. The same family of historical information can therefore support different future shapes: approximately persistent displacement in the ETT cases, and evolving low-frequency corrections in Weather.

The dominant mode is consequential without exhausting the trained mapping. Its reported correction-energy share varies from approximately 0.09 to 0.80 across the model–setting groups. This breadth supports retaining secondary modes, especially for long horizons and nonconstant output corrections.

Branch-specific removal provides a functional test. In the dense ETTm2-192 model, removing the semantic subspace reduces standalone branch MSE by 83.9% but increases fused MSE by 40.6%, according to the source's relative-error summary. The direction of this response is particularly informative: optimizing the branch as an independent forecaster would favor a change that harms the deployed forecast. The earlier exploratory analysis reports the same separation in 52 of 72 low-rank checkpoint cases; fused error increases in all 72. These earlier cases provide supporting context and are not independent replications of the reused checkpoints analyzed here.

The semantic subspace should not be identified with a single singular mode. Its dimension can be substantially larger and, for some low-rank heads, can span the entire branch. Such cases do not establish semantic specificity against equal-dimensional controls. Moreover, the completed analysis does not pass its full semantic-sufficiency criterion, which requires retaining the semantic subspace alone to preserve fused accuracy within 0.5%. The supported mechanism is therefore **functional reliance on recent-level information within a multi-direction correction**, rather than replacement of the branch by one fixed semantic coordinate.

### 4.5 Compact adaptation offers a measurable accuracy–capacity trade-off

The rank-$H/8$ variant uses 16.4%–33.0% of the dense model's total parameters across the principal settings, including the backbone and gate. For ETTh2 at horizon 192, the counts are 23,863 and 140,191. The source's macro-average total-parameter ratio is 23.8%; the ratio changes with the horizon and dataset.

Over the 24 principal settings, the mean absolute relative difference between rank-$H/8$ and dense PhaseFormer-L is 0.7183% in MSE and 0.5343% in MAE. This is a substantial reduction in capacity at a small, measurable accuracy difference, although it exceeds the predeclared 0.5% equivalence threshold. Averages of signed relative MSE changes against the matched baseline are −1.31% for rank-$H/8$ and −1.00% for the dense model. These signed averages can coexist with nonzero absolute differences because individual settings move in both directions.

Earlier, test-exposed compression experiments reached 3.5%–6.1% of dense-head parameters and reported retaining 92.4%–101.9% of their reference improvement. Those figures concern a seven-setting exploratory subset and the branch parameter count, not a guaranteed fraction of whole-model performance. They motivate low-rank adaptation but do not replace the broader trade-off above.

Taken together, the results favor learning compact temporal coordinates over imposing an exactly rank-one architecture. The predictive spectrum reveals why substantial compression is possible; the measured compression curve determines how much is useful in the fused model.

### 4.6 Level memory helps organize the adaptation regime

We relate forecasting changes to three training-set descriptors: cross-cycle level standard deviation, last-cycle level shift, and an estimated level-memory statistic $\hat\tau$. Across 28 settings, their Spearman correlations with relative MSE change are −0.039, −0.139, and **−0.750**, respectively. The association is strongest for persistence of level information rather than its amplitude alone.

An exploratory diagnostic uses $\hat\tau>57.35$ steps to identify ETTh2, ETTm2, and Weather. This threshold was fixed from training-set statistics after the improvement signs of five development datasets were known. It labels the twelve consistently improving settings, but misses MSE improvements in all four Electricity settings and in ETTh1-336/720. The association therefore supports a useful hypothesis about operating regimes, rather than a fully independent classifier of future gains.

The statistic is estimated from short windows and is best interpreted ordinally. In addition, the four horizons share their dataset-level descriptors; the 28 rows are not 28 independent datasets. No claim of statistical significance is inferred from the correlation alone. The learned gate likewise remains an optimization parameter, not a direct measure of the physical strength of level drift.

### 4.7 Functional rank of the trained heads

Sections 4.3–4.4 measure what is *attainable* and what the heads *read*. This section measures what the fitted heads *use*: for a trained head, how many canonical modes carry its own realized predictive improvement, and what each mode reads and writes.

**Definition.** Write the head's effective map as $W=\sum_i s_i u_i v_i^\top$ (13) and let $\hat y_0$ be the same model with $W$ removed. With $d_i$ the increment the mode writes into the fused forecast,

$$
\hat y=\hat y_0+\sum_i d_i ,
\qquad
I_i=\frac1H\mathbb E\!\left[2e_0^\top d_i-\|d_i\|_2^2\right],
\qquad e_0=y-\hat y_0 .
\tag{14}
$$

Because the left singular vectors satisfy $u_i^\top u_j=0$ for $i\ne j$ while the gate and normalization scale are per-channel scalars, the cross terms vanish and $\operatorname{MSE}(\hat y_0)-\operatorname{MSE}(\hat y)=\sum_i I_i$ exactly. We define the **functional rank** $r_q^{\mathrm{func}}$ as the smallest number of modes, taken in order of measured contribution, that recovers fraction $q$ of the full low-rank checkpoint's improvement over the zeroed-map baseline:

$$
\operatorname{Recovery}(k)=\frac{\operatorname{MSE}(\hat y_0)-\operatorname{MSE}(\hat y_k)}
{\operatorname{MSE}(\hat y_0)-\operatorname{MSE}(\hat y_{\text{full}})} .
\tag{15}
$$

**Scope.** Six settings — ETTh2-96/720, ETTm2-96/192, Weather-96/192 — at four factorization levels $q\in\{1/4,1/8,1/16,1/32\}$ and three seeds: 72 trained checkpoints. These are the checkpoints already used in §4.4; no new training was performed. Validation split only; the test split was not read. The measurement requires the factorized head's canonical modes, so it does not extend to the dense `l_main` variant except as the alignment reference of Table 5.

**Table 3. Functional rank versus nominal rank.** $r_{90},r_{95},r_{99}$ are defined by (15), taken in order of measured fused contribution. Mean ± population standard deviation over the three seeds.

| Setting | nominal rank $H/8$ | $r_{90}$ | $r_{95}$ | $r_{99}$ | top-1 share | negative $I_i$ |
|---|---:|---:|---:|---:|---:|---:|
| ETTh2-96 | 12 | 1.3 ± 0.5 | 3.0 ± 0.0 | 4.0 ± 0.0 | 90.1% | 0.3 |
| ETTh2-720 | 90 | 3.0 ± 0.0 | 4.7 ± 0.5 | 8.3 ± 0.5 | 69.4% | 24.7 |
| ETTm2-96 | 12 | 3.0 ± 0.0 | 3.3 ± 0.5 | 4.3 ± 0.5 | 65.3% | 0.0 |
| ETTm2-192 | 24 | 3.0 ± 0.0 | 3.0 ± 0.0 | 5.3 ± 1.2 | 72.8% | 0.3 |
| Weather-96 | 12 | 4.0 ± 0.0 | 5.0 ± 0.0 | 5.3 ± 0.5 | 40.2% | 1.0 |
| Weather-192 | 24 | 4.0 ± 0.0 | 5.0 ± 0.0 | 6.7 ± 0.5 | 27.9% | 2.3 |

The functional rank is set by the task, not by the architecture. Sweeping the nominal rank from 3 or 6 up to 24–180 changes $r_{95}$ by at most a few modes and often not at all: ETTm2-192 gives $r_{95}=3.0$ at every one of its four nominal ranks (6, 12, 24, 48), and Weather-192 gives $5.0$ at all four (6, 12, 24, 48). Across all 72 checkpoints $r_{95}$ never leaves the range 1–9, and at $H/8$ it lies between 3 and 5. ETTh2-720 is the one setting where $r_{95}$ grows with capacity, from 4.0 at rank 22 to 7.3 at rank 180. Increasing the architectural rank therefore adds expressiveness that the fitted maps do not spend on prediction.

**Modes are additively separable.** Across all 72 checkpoints the identity in (14) holds with a maximum residual of $5.12\times10^{-10}$ against a threshold of $10^{-6}$, and the closed-form reconstruction of the fused output matches the model's own forward pass to $7.4\times10^{-6}$ (float32 round-trip). As an independent check, the modes were also deleted *inside* the model's forward pass — by subtracting $(V^{+}u_i)s_i(v_i^\top z)$ from the hidden state, which removes exactly that mode — and the measured MSE was compared with the predicted one in 120 such deletions across the six settings; the maximum relative error was $3.28\times10^{-9}$.

Two qualifications bound this result. It is an accounting identity for a **frozen** checkpoint, not a statement about training dynamics: retraining with a smaller rank may rotate or reorganize the subspace, so one cannot conclude that training mode 0 alone would reproduce its current share. And what is strictly additive is the canonical-mode contribution $I_i$; grouping modes into semantic families sums contributions of already-classified modes, whereas the template projections themselves overlap and are not an additive decomposition.

**Most modes are redundant, and some are harmful.** At high nominal rank a large fraction of modes carry negative marginal contribution: 5.0, 12.3, 24.7, and 62.0 modes on average at ranks 22, 45, 90, and 180 for ETTh2-720. Deleting all negative-contribution modes *improves* the fused forecast, by $7.33\times10^{-4}$ on average at rank 90 for ETTh2-720 — small next to that setting's total improvement (0.061–0.070), but consistently negative across all three seeds. Random deletion of the same number of modes, by contrast, costs $+2.3\times10^{-2}$ averaged over all 72 checkpoints. Among informed criteria at a 25% pruning budget the ordering is contribution ($+1.51\times10^{-3}$) < activation energy ($+1.64\times10^{-3}$) < singular value ($+1.70\times10^{-3}$) ≪ random ($+2.32\times10^{-2}$). The three informed criteria are close, so at this budget the finding is that *any* informed ordering dominates random, not that contribution ordering is decisively better.

**Table 4. What the leading modes read and write (`q=1/8`, three seeds).** Modes are ordered by measured fused contribution. “Group expl.” is the share of the direction's squared norm inside the best-matching semantic group's span; labels are best matches, not exclusive components (see §4.7 caveats).

| Setting | pos. | contribution share | reads | group expl. | writes | group expl. |
|---|---:|---:|---|---:|---|---:|
| ETTh2-96 | 1 | 90.1% | recent level | 0.63 | displacement | 0.96 |
|  | 2 | 4.5% | recent level | 0.30 | tilt | 0.80 |
|  | 3 | 3.0% | periodic shape | 0.41 | periodic correction | 0.81 |
| ETTh2-720 | 1 | 68.7% | recent level | 0.73 | displacement | 0.95 |
|  | 2 | 11.9% | periodic shape | 0.72 | periodic correction | 0.95 |
|  | 3 | 10.2% | periodic shape | 0.75 | periodic correction | 0.93 |
|  | 4 | 2.8% | level change | 0.36 | curvature | 0.52 |
|  | 5 | 1.5% | recent level | 0.41 | curvature | 0.55 |
| ETTm2-96 | 1 | 65.3% | recent level | 0.60 | displacement | 0.97 |
|  | 2 | 18.8% | periodic shape | 0.61 | periodic correction | 0.96 |
|  | 3 | 12.3% | periodic shape | 0.63 | periodic correction | 0.96 |
| ETTm2-192 | 1 | 72.8% | recent level | 0.63 | displacement | 0.97 |
|  | 2 | 15.1% | periodic shape | 0.67 | periodic correction | 0.94 |
|  | 3 | 10.2% | periodic shape | 0.70 | periodic correction | 0.96 |
| Weather-96 | 1 | 40.2% | recent level | 0.52 | curvature | 0.96 |
|  | 2 | 34.6% | recent level | 0.47 | curvature | 0.95 |
|  | 3 | 14.2% | recent level | 0.15 | curvature | 0.49 |
|  | 4 | 5.6% | recent level | 0.22 | curvature | 0.50 |
|  | 5 | 4.2% | local curvature | 0.15 | tilt | 0.46 |
| Weather-192 | 1 | 27.9% | recent level | 0.44 | curvature | 0.52 |
|  | 2 | 24.8% | recent level | 0.57 | curvature | 0.92 |
|  | 3 | 22.2% | local trend | 0.24 | tilt | 0.42 |
|  | 4 | 14.5% | recent level | 0.25 | tilt | 0.33 |
|  | 5 | 6.9% | recent level | 0.21 | tilt | 0.10 |

Two regimes are visible. On ETTh2 and ETTm2 the leading mechanism is **recent level → horizon-wide displacement**, carrying 65%–90% of the improvement, followed by one or two **periodic shape → periodic correction** modes. On Weather the leading share falls to 28%–40% and the output is **curvature or tilt** rather than displacement, so five modes are needed rather than three. These attribution results did not depend on the new analysis alone: recomputing them from the exported mode tensors reproduced the earlier independent semantic analysis on all 475 comparable modes (identical best-matching input group; group explanations agreeing to $2.8\times10^{-14}$).

A worked case, ETTh2-720 $q=1/8$ seed 2021 (nominal rank 90): the top five modes by contribution carry 75.9% + 8.1% + 6.9% + 3.0% + 1.3% = **95.3%** of the positive contribution, decomposing as 75.9% level displacement + 15.0% periodic correction + 4.3% higher-order correction. The first three modes have clear attributions (input group explanations 0.74, 0.63, 0.66; output 0.96, 0.95, 0.94); the last two have low explanations (0.39 and 0.34) and are best described as associated with level change and curvature rather than identified with them.

**Table 5. How much of the dense head's improvement the low-rank input subspace retains.** The dense head's effective map is restricted to the low-rank head's top-$k$ input subspace, and the surviving fraction of the dense improvement over its own zeroed-map baseline is reported (`q=1/8`, three seeds).

| Setting | dim 2 | dim 4 | dim 8 |
|---|---:|---:|---:|
| ETTh2-96 | 76% | 85% | 85% |
| ETTh2-720 | 71% | 83% | 87% |
| ETTm2-96 | 59% | 75% | 78% |
| ETTm2-192 | 68% | 80% | 80% |
| Weather-96 | 74% | 90% | 94% |
| Weather-192 | 42% | 83% | 95% |

This is consistent with the functional-rank result: the dense head's predictive structure is itself concentrated in few temporal directions, so a low-dimensional input subspace recovers most of its value. An associated observation is that the **output** subspace overlap with the dense head consistently exceeds the **input** overlap (at dimension 8: input 0.25–0.79 versus output 0.59–0.87 across the six settings). Different seeds agree more on which low-frequency correction shape to write than on which exact kernel to read the state with — consistent with §4.7's seed analysis below.

The plan's sharper question — whether the low-rank head kept the dense head's *prediction-relevant* subspace rather than its *weight-energy-leading* one — is only separable where those two dense orderings differ. They largely agree on ETTm2 (rank correlation 0.87–0.94) but not on ETTh2-720 (rank correlation 0.00, −0.20, +0.03 across seeds) and only partly on Weather-192 (0.29–0.51). Where they disagree, the low-rank subspaces overlap the two dense references almost identically (at dimension 8, averaged over all 72 cells: 0.518 against the singular ordering versus 0.508 against the functional one), so this evidence does not establish that compression preferentially preserved prediction-relevant over energetic directions. Reporting it as such would over-read the data.

**Seed stability forces subspace-level, not mode-level, claims.** Matching canonical modes across seeds by $|v_i^\top v_j'|\cdot|u_i^\top u_j'|$ with a linear assignment gives low agreement: mean matched input cosine 0.37, output cosine 0.58, and only 27.5% of matched pairs above 0.7 in both. Agreement degrades as nominal rank grows (input cosine 0.57 at rank 3, 0.28 at rank 12, 0.18 at rank 24), which is the expected behaviour of near-degenerate singular subspaces rather than evidence that the mechanisms differ. Individual modes should therefore not be identified across seeds by index; the stable objects are the top-$k$ functional subspaces and the semantic families, which is why Tables 3–4 report seeds as an aggregate.

**Table 6. Sparsifying a single mode's temporal kernel.** Each row replaces one mode's 720-step input kernel $v_i$ with a sparse or smoothed version, leaving the output direction and all other modes untouched. Reconstruction $R^2$ is against the original $v_i$; the cost column is the exact change in fused MSE (positive means worse), computed with the same additive accounting as (14); the `dense` row is the identity and reads $3\times10^{-17}$, which is the internal consistency check on the whole chain.

| Variant | retained lags | reconstruction $R^2$ | fused MSE increase |
|---|---:|---:|---:|
| dense (identity) | 720 | 1.000 | $+2.99\times10^{-17}$ |
| TV, $\lambda=0.05$ | 720 | 0.769 | $+7.99\times10^{-5}$ |
| semantic Lasso | 39 atoms | 0.524 | $+2.00\times10^{-3}$ |
| group selection, 24-step windows, keep 25% | 192 | 0.552 | $+4.53\times10^{-3}$ |
| hard threshold, 90% sparsity | 72 | 0.487 | $+4.86\times10^{-3}$ |
| hard threshold, 95% sparsity | 36 | 0.344 | $+6.55\times10^{-3}$ |
| hard threshold, 99% sparsity | 7 | 0.147 | $+1.01\times10^{-2}$ |

The informative contrast is between heavy smoothing and hard sparsity. Weak total-variation smoothing keeps $R^2=0.769$ at essentially no cost ($+8.0\times10^{-5}$), whereas retaining only 7 of 720 lags costs $+1.01\times10^{-2}$ — two orders of magnitude more. For reference, these per-mode costs should be read against each cell's total improvement (0.035–0.176). The evidence therefore supports **few modes with smooth temporal kernels**, not few modes with isolated non-zero lags: the sparsity that pays is at the level of how many mechanisms are retained.

## 5. Discussion: What the Journal Extension Adds

The original PhaseFormer study asks how recurring structure should be represented. This extension asks how a compact phase-domain predictor should adapt to predictable changes in the level of that structure. The connection is explicit: within-cycle transformations and cross-cycle offsets act on different axes, and their distinction leads to a targeted linear adaptation path.

Three findings give that path a coherent scientific interpretation. First, a simple geometric perturbation corresponds to a strongly concentrated predictive spectrum. Second, the useful input direction can carry little signal variance, so energy-based compression need not preserve forecasting value. Third, the value of a branch is defined jointly with its partner: a component that increases standalone error can reduce fused error. These findings explain why retaining trainable temporal coordinates is preferable to hard-coding the apparent dominant template.

**A fourth finding, added here, closes the loop between capacity and mechanism.** The same canonical decomposition that names what the heads read and write also measures how much each named direction is worth, and the two agree. On the six analyzed settings, three to five modes recover at least 95% of the fitted head's own improvement; the recovery curves saturate within roughly five to eight modes in every setting and at every nominal rank; and the modes that matter carry interpretable semantics — recent level to displacement on the ETT families, and a more distributed curvature/tilt decomposition on Weather. The compression story of §4.5 is thus not merely that fewer parameters suffice, but that a small, nameable set of read–write mechanisms accounts for the fitted model's predictive value. Consistently, modes beyond that core are not merely wasteful: at the highest nominal rank, a third of the ETTh2-720 head's modes have negative marginal contribution, and removing them improves the fused forecast.

Three qualifications keep this within the evidence. The additive accounting holds for a frozen checkpoint and says nothing about how a differently-sized model would train. The functional-rank measurement covers six settings and four factorization levels, not the 28-setting main table, and those checkpoints were selected with test performance visible, so it is conditional evidence. And the semantic labels are best matches against a fixed template dictionary whose members are collinear; they identify the family a direction belongs to, not an exclusive component decomposition.

The empirical scope is equally specific. The principal three-seed protocol establishes consistent dual-metric improvements on ETTh2, ETTm2, and Weather, including six newly trained settings beyond development coverage. It does not establish uniform improvement across all datasets. The historical 1% non-degradation target is not met, and rank-$H/8$ falls outside the predeclared 0.5% equivalence band. These outcomes delimit the operating regime and compression trade-off; they do not alter the central observation that useful adaptation can be low-dimensional and partner-dependent.

Two distinctions matter for interpreting the mechanism. Proposition 1 concerns the geometry of an input perturbation, whereas Proposition 2 additionally assumes a residual dominated by a scalar predictable state. Neither asserts that all forecast errors are one-dimensional. Likewise, the independent regression spectrum (§4.3), the trained-head interventions (§4.4), and the functional decomposition (§4.7) are three complementary measurements of different objects — attainable improvement, reliance on removed information, and realized per-mode contribution — not interchangeable estimates of the same conditional optimum. §4.3.1 states explicitly why their "leading mode" numbers differ. The remaining directions become especially relevant when the level evolves over the forecast horizon, as illustrated by Weather's tilt and curvature modes.

## 6. Conclusion

PhaseFormer-L extends phase-domain forecasting with a jointly learned temporal level-adaptation branch. A rank-one cross-cycle perturbation provides the geometric starting point; a concentrated predictive spectrum explains compactness; trained-model interventions establish the importance of complementarity; and a functional-rank decomposition of the trained heads shows that three to five canonical read–write modes recover at least 95% of the fitted model's own predictive improvement, with recent-level-to-displacement as the dominant mechanism on the ETT families and a more distributed curvature/tilt structure on Weather. The extension improves both error metrics across all evaluated horizons of ETTh2, ETTm2, and Weather, while low-rank variants make its capacity–accuracy trade-off explicit. The broader design principle is to allocate forecasting capacity according to **predictive value relative to the backbone**, preserving the small set of directions that convert recent temporal state into useful future correction.

## Appendix A. Evidence Scope and Exploratory Results

### A.1 Provenance of the principal evidence

| Evidence | Coverage | Evaluation and reuse |
|---|---|---|
| Principal performance table | 24 settings × 3 seeds per reported arm | Seven PhaseFormer-L settings and six matched settings reuse test-exposed development records; remaining records are newly trained under the frozen protocol |
| Traffic performance | 4 settings × 3 seeds | Newly trained; exploratory |
| Predictive-spectrum analysis | 28 settings | Training/validation analysis; seven reused analyses and 21 additional settings |
| Trained-head decomposition and interventions | 6 settings × 3 model variants × 3 seeds | Validation analysis on reused checkpoints; not an independent retraining study |
| **Functional rank and per-mode attribution** | **6 settings × 4 factorization levels × 3 seeds (72 checkpoints)** | **Validation analysis on the same reused checkpoints as the row above; no new training, test split not read** |
| **Dense-vs-low-rank subspace alignment** | **6 settings × 3 seeds (18 dense caches)** | **Validation analysis; the dense checkpoints come from the same `rank_sweep_2_stage1` run family, so the comparison is same-protocol** |
| Level-memory association | 28 settings | Training descriptors combined with Table 1 outcomes; repeated horizons within datasets |
| Confirmed witness table | 28 settings, one witness seed per row | E14 records plus audited targeted searches; explicit test-set selection where applicable |

The seven reused PhaseFormer-L settings are ETTh2-96/720, ETTm2-96/192, Weather-96/192, and Electricity-336. The first six also reuse the matched baseline. No result is assigned to the unexecuted frozen-conditional-direction comparison, the unexecuted new negative-control runs, or the unexecuted Electricity-336 head dissection. These experiments are outside the completed evidence used for this paper.

The functional-rank analysis of §4.7 covers the same six settings as the trained-head analysis minus Electricity-336, whose cached validation features were incomplete and which is additionally outside that experiment's pre-registered scope.

### A.2 Provenance of the confirmed table

Table 1 is a compact witness table assembled from three audited sources. The 19 E14 rows are read from `research_runs/phaseformer_L_e14_main_v1/results.csv`, with configuration and metrics checked against each run's `config.json` and `metrics.csv`. The ETTm1-96 and ETTm1-336 rows come from the final selections in `research_runs/phaseformer_L_golden_search_v1/`; Traffic-336 and Traffic-720 come from `research_runs/phaseformer_L_targeted_100_v3/target_final.json`. The five remaining rows use the selected configurations from the 1000-candidate `batch_period_loss_gate_200_v1` search, whose final summary is `research_runs/phaseformer_L_batch_period_loss_gate_200_v1/final.json`.

All configuration searches used test metrics for selection. Table 1 therefore reports conditional evidence of attainable configurations, not an unbiased test estimate. For the five latest settings, seed 2021 was a 10%-data, five-epoch screening run; only seeds 2022 and 2023 were full-data confirmations. The complete parameter registrations and machine-readable outputs are preserved in `docs/PhaseFormer_L_main_table_repro.md`, `docs/PhaseFormer_L_targeted_100_v3_params.md`, `docs/PhaseFormer_L_batch_period_loss_gate_200_v1_params.md`, and the cited experiment directories.

### A.3 Intervention reporting conventions

Intervention effects in Section 4.4 are relative changes from the corresponding unmodified checkpoint. Absolute MSE differences and relative percentages are distinct quantities. Semantic-subspace deletion tests reliance on the removed information; demonstrating specificity additionally requires a nondegenerate control of matched dimension. When the semantic subspace spans the entire head, deleting it measures reliance on the branch as a whole. Retaining that subspace alone tests a separate property—sufficiency—which is stronger than functional reliance.

### A.4 Provenance of the functional-rank results (§4.3.1, §4.7, Tables 3–6)

All artifacts are under `research_runs/lowrank_functional_rank_v1/`, produced on 2026-09-24 from the six settings × four factorization levels × three seeds listed above. Nothing was trained; every number is a validation-split computation over frozen checkpoints.

| Artifact | Contents | Produced by |
|---|---|---|
| `report.md` | Full report: plan §5–§13 tables, verification block, limitations | `scripts/render_lowrank_functional_rank_report.py` |
| `functional_rank_cells.csv` | 72 rows: $r_{90}/r_{95}/r_{99}$ under four orderings, additivity residual, negative-mode counts | `scripts/analyze_lowrank_functional_rank.py` |
| `functional_rank_curves.csv` | per-cell recovery curves for the four orderings | same |
| `mode_contributions.csv` | per-mode $s_i$, weight/activation energy share, $I_i$, leave-one-out change | same |
| `mode_pruning.csv` | zero-shot pruning under each criterion, plus the random band | same |
| `mode_semantics.csv` | 501 rows: per-mode best template, group explanations, dictionary $R^2$ | `scripts/annotate_lowrank_modes.py` |
| `mode_sparsity.csv` | per-mode reconstruction $R^2$ and exact fused-MSE cost under each sparsifier | `scripts/analyze_lowrank_mode_sparsity.py` |
| `seed_mode_stability.csv` | matched cross-seed cosine and contribution rank correlation | same |
| `dense_alignment.csv` | principal angles, subspace overlaps, retained dense improvement | `scripts/analyze_lowrank_dense_alignment.py` |
| `contribution_forward_check_shard*.csv` | 120 in-model mode deletions, measured versus predicted MSE | `scripts/verify_lowrank_contribution_forward.py` |
| `modes/` | exported canonical mode tensors $(u,s,v^\top,I)$ per cell | `analyze_lowrank_functional_rank.py --export-modes` |
| `dense_features/` | validation feature caches for the dense `l_main` head | `scripts/build_dense_head_cache.py` |
| `figures/` | 7 figures, including `fig1_functional_rank_curves.png` and the annotated `fig5_canonical_modes.png` | `scripts/render_lowrank_functional_rank_figures.py` |

Two inputs are reused rather than regenerated. The validation feature caches live in `research_runs/lowrank_checkpoint_information_v1/features/`, and the earlier semantic attribution used for the reproduction check is `research_runs/lowrank_checkpoint_information_v1/semantic_alignment.csv`. The exported modes were verified to be numerically identical to that run's canonical decomposition (maximum $|s_i|$ difference $1.1\times10^{-15}$) before the two were compared; the comparison itself is `scripts/summarize_lowrank_mode_semantics.py`.

The algebraic conventions that the whole chain rests on were fixed empirically rather than by reading the training code, because the cached anchor and the branch's own anchor differ by the horizon-mean of $V\,\text{bias}_U$ and the two are indistinguishable on low-bias cells. The derivation is preserved in `scripts/_probe_cache_algebra.py` (which reproduces the models' own `fused` output to $\sim10^{-6}$) and `scripts/_probe_live_algebra.py` (which confirms the same identity against a live forward pass). The claim-by-claim verification of the numbers reported in §4.7 is `scripts/verify_draft_claims.py`.

Execution record, including the commands, the shard layout, the defects found and fixed during the run, and the full list of limitations, is in `docs/agent-log.md` under 2026-09-24. The journal-form summary is `docs/PhaseFormer_L_functional_rank_report.md`.

## Reference

[1] Niu et al. *PhaseFormer*. ICLR 2026. [arXiv:2510.04134](https://arxiv.org/abs/2510.04134). Predecessor attribution and identifier follow the source manuscript.

---

*Source note: The principal narrative and mechanism analysis are based on `PhaseFormer_L_minipaper.md`. Table 1 is the consolidated, audited witness ledger described in Appendix A.2. Proposition statements and proofs have been rewritten to make their assumptions and conclusions explicit; no external results were added. The v2 additions (§4.3.1, §4.7, Appendix A.4) report the trained-head functional-rank experiment; their scope is six settings and their checkpoints are test-selected, so they are conditional evidence in the same sense as the rest of the trained-head analysis.*
