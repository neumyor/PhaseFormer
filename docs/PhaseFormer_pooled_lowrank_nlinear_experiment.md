# Pooled Low-Rank NLinear Screen

> Status: registered for execution on September 10, 2026. This is a
> user-requested exploratory extension of Progressive IB Stage 2, not a
> replacement for its single-variable causal protocol.

## Scope

- Datasets: ETTh1 and ETTm1.
- Initial settings: lookback 720, horizon 96, seed 2021, Huber loss, 30 epochs,
  and the lowest-validation-loss checkpoint.
- Candidate prediction: an original PhaseFormer path and a pooled low-rank
  NLinear path are both initialized randomly and jointly optimized with the
  existing learned static fusion gate.
- Reference: the fixed PhaseFormer Golden table. Results also retain the
  matched frozen-PhaseFormer metrics because the Golden environment is not
  identical to the A800 execution environment.

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
2. Use validation MSE and MAE normalized by the matched frozen-phase baseline
   to select the best three non-full `(pool_factor, rank)` configurations per
   dataset.
3. For those three configurations, evaluate smoothing ratios
   `{0.10, 0.25, 0.50, 0.75}`.
4. Every planned configuration is independently and jointly trained. After its
   validation-selected checkpoint is restored, it receives one test evaluation.
   The test matrix is therefore test-set-exposed exploratory evidence;
   configuration selection remains based on validation only.

## Interpretation Limits

- Pooling and rank vary together in the first screen, so its result cannot
  isolate a pure low-rank effect.
- This H96, single-seed screen cannot establish cross-seed or cross-horizon
  stability.
- A negative Golden delta means a lower error than Golden. It is not by itself
  a paper-level improvement claim.
- The experiment does not change the current model preset.
