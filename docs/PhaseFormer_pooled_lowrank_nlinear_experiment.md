# Pooled Low-Rank NLinear Screen

> Status: completed on September 10, 2026. This is a
> user-requested exploratory extension of Progressive IB Stage 2, not a
> replacement for its single-variable causal protocol.

## Scope

- Datasets: ETTh1 and ETTm1.
- Initial settings: lookback 720, horizon 96, seed 2021, Huber loss, 30 epochs,
  and the lowest-validation-loss checkpoint.
- Candidate prediction: an original PhaseFormer path and a pooled low-rank
  NLinear path are both initialized randomly and jointly optimized with the
  existing learned static fusion gate.
- Reference: the fixed PhaseFormer Golden table. Results also retain a matched
  jointly trained original-PhaseFormer baseline because the Golden environment
  is not identical to the A800 execution environment.

## Mechanism

The NLinear branch keeps the last-value anchor outside its dynamic input. It
first centers the history, optionally blends it with a fixed 24-step
moving-average view, adaptively pools the history in time, and then applies a
linear factorization:

```text
X - X_last -> optional smoothing -> pooled history -> rank-r -> horizon
```

The input pool factor and intermediate rank are explicit variables. The
smoothing ratio is a separate blend coefficient in `[0, 1]`.

## Search Order

1. With `smooth_ratio=0`, evaluate pool factors `{1, 2, 4, 8}` and ranks
   `{4, 8, 16, 32, 64, 96}` where the rank is valid for the pooled length.
2. Use validation MSE and MAE normalized by the matched jointly trained original
   baseline
   to select the best three non-full `(pool_factor, rank)` configurations per
   dataset.
3. For those three configurations, evaluate smoothing ratios
   `{0.10, 0.25, 0.50, 0.75}`.
4. Every planned configuration is independently and jointly trained. After its
   validation-selected checkpoint is restored, it receives one test evaluation.
   The test matrix is therefore test-set-exposed exploratory evidence;
   configuration selection remains based on validation only.

## Results

- Every dataset completed one original-PhaseFormer baseline, 23 no-smoothing
  pool/rank configurations, and 12 smoothing configurations.
- ETTh1 selected `pool=2, rank=8, smooth=0.75` by validation score. Its test
  MSE/MAE was `0.362479/0.390818`, or `+0.97%/+2.31%` relative to Golden.
- ETTm1 selected `pool=1, rank=16, smooth=0.25` by validation score. Its test
  MSE/MAE was `0.299431/0.350440`, or `+2.19%/+1.87%` relative to Golden.
- Neither selected candidate improves both Golden metrics. The experiment does
  not change a PhaseFormer preset or establish a replacement mechanism.
- The complete auditable result package is
  `research_runs/joint_pooled_lowrank_nlinear_h96_v1/`, including pool/rank
  and rank/smoothing Golden-delta heatmaps for MSE and MAE.

## Interpretation Limits

- Pooling and rank vary together in the first screen, so its result cannot
  isolate a pure low-rank effect.
- This H96, single-seed screen cannot establish cross-seed or cross-horizon
  stability.
- A negative Golden delta means a lower error than Golden. It is not by itself
  a paper-level improvement claim.
- The experiment does not change the current model preset.

## Controlled Follow-up Plan

> Status: planned only on September 10, 2026. No follow-up model has been
> trained and no further test prediction will be read until the selection is
> frozen.

### Question

The completed screen does not identify a monotonic low-rank or pooling effect:
pooling and rank changed jointly, ranks were not normalized to each pooled
input length, smoothing was only evaluated on three validation-selected cells,
and H96/one-seed results may reflect optimizer variance. The follow-up asks:

1. At fixed pooling, how does **relative factor rank** change validation
   error?
2. At fixed relative rank, how does pooling change validation error?
3. Does smoothing have a main effect, or only an interaction with pooling and
   rank?

The prior exploratory matrix is test-set-exposed and motivated this adaptive
follow-up scope. Its numeric values are excluded from all future configuration
ranking and selection, but every follow-up report must disclose this
test-informed lineage and must not be described as blind evaluation.

### Controls And Factors

- `phase_only`: original PhaseFormer.
- `direct_nlinear`: the existing unpooled, unfactorized
  `WeakPeriodResidualHead` jointly trained with PhaseFormer and the same static
  fusion gate. This distinguishes a bottleneck effect from merely replacing a
  direct linear map with two linear layers.
- `factorized_full`: pooled-low-rank head with `pool=1` and the maximum valid
  rank. This is the factorization-parameterization control.
- Pool factor: `p in {1, 2, 4}`. The `p=8` condition remains recorded as an
  exploratory boundary observation, but is outside the budgeted identification
  range; it can be added only if the smaller-pool response is non-monotonic.
- Relative rank:
  `q in {1/12, 1/3, 1}`. For each `(p, H)`, use
  `r = max(4, round_to_multiple_of_4(q * min(ceil(720 / p), H)))`, capped at
  the valid maximum. Results are analyzed by `q`, not by raw rank alone.
- Smoothing ratio: `s in {0, .25, .50}` with the fixed 24-step
  replicated-boundary moving-average definition already used in the initial
  screen.

All conditions keep L720, Huber, 30-epoch maximum, the lowest-validation-loss
checkpoint, identical data splits, and random joint optimization of the
PhaseFormer path, residual path, and fusion gate.

### Phase A: Capacity Identification, Validation Only

- Settings: `ETTh1` and `ETTm1`, H96, seeds `2021`, `2022`, and `2023`.
- Train `phase_only`, `direct_nlinear`, and the complete
  `p x q x s=0` factorial grid. Duplicate rank values created by capping are
  deduplicated but recorded.
- Do not call `trainer.test()` or otherwise read test predictions.
- Primary paired response for every seed is candidate MSE/MAE minus the
  same-seed `direct_nlinear` control. Golden is reported only after the final
  confirmation; it is not used for factor-effect estimation.

### Phase B: Smoothing Identification, Validation Only

- Preconditions: Phase A completes without data, checkpoint, or numerical
  failures; its analysis is recorded before this phase starts.
- Settings: `ETTh1` and `ETTm1`, H96, seeds `2021` and `2022`.
- Evaluate a predeclared balanced subset:
  `p in {1, 2, 4}` crossed with `q in {1/12, 1/3}`, and every smoothing
  ratio `s`. The `s=0` points are reused only when the exact setting and seed
  match Phase A; all nonzero-smoothing cells are independently trained.
- This grid measures smoothing at low and medium relative capacity across the
  entire pooling range, rather than only around prior validation winners.

### Budget And Stop Conditions

- Phase A has 11 jointly trained models per dataset-seed setting, or 66 runs:
  two controls plus nine `pool x relative-rank` cells.
- Phase B reuses exact `s=0` Phase A runs and adds at most 12 nonzero-smoothing
  runs per dataset-seed setting, or 48 additional runs: six pool/rank cells
  times two nonzero smoothing ratios, across two datasets and two seeds.
- Confirmation has one frozen shared candidate plus the two controls: at most
  36 full-train test runs across two datasets, two horizons, and three seeds.
- Stop after Phase A if neither MSE nor MAE yields a reproducible pool, rank,
  or pool-by-rank effect under the rules below. Stop after Phase B if smoothing
  has no reproducible conditional effect. In either case, retain the direct
  NLinear control and do not introduce pool/rank/smoothing as a preset.

### Analysis And Decision Rules

- Use seed-paired validation deltas and report means, standard deviations,
  paired bootstrap 95% intervals, and per-setting rank/pool response curves.
- Estimate fixed effects for `log2(p)`, `log2(q)`, their interaction, and, in
  Phase B, smoothing plus its interactions. Dataset and horizon remain
  explicitly reported strata; no dataset-specific mechanism is promoted from
  a single favorable cell.
- A Phase A effect qualifies only for final H192 confirmation when its paired
  direction agrees in both datasets for at least two of three seeds and its
  pooled bootstrap interval excludes zero for both MSE and MAE. Otherwise it
  is recorded as inconclusive rather than generalized.
- A smoothing claim additionally requires its direction to be consistent in
  both seeds at the affected `(p, q)` cells; H192 is evaluated only during
  final confirmation and does not participate in selection. A benefit limited
  to high rank is described as a
  capacity-by-smoothing interaction, not a global smoothing benefit.
- Select one shared configuration only after these validation rules
  are frozen. If neither passes, stop without a new test run or preset change.

### Confirmation And Test Boundary

- Only a validation-qualified shared configuration proceeds to full 30-epoch
  confirmation on ETTh1/ETTm1, H96/H192, seeds `2021`, `2022`, and `2023`,
  paired against `phase_only` and `direct_nlinear`.
- Test is read exactly once after the configuration is frozen. The final
  package reports Golden deltas, matched-control deltas, parameter count,
  training time, and sample-level error analysis.
- The required runner must add a validation-only mode that omits
  `--evaluate-test`; the existing test-reading matrix runner must not be
  reused for Phases A or B without that change.
