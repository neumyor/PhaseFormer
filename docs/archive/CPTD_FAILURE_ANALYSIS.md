# CPTD Failure Analysis

## Decision

The output-space Circular Phase Transport Decoder (CPTD) was rejected and removed
from the active code. The user-provided result table showed regression on all
eight evaluated ETT tasks, including catastrophic ETTm1 failures. CPTD should
remain a negative research result, not an active tuning target.

## Reported results

All cells below are `MAE / MSE`.

| Dataset | Horizon | CPTD | Gold | Matched original | Existing latest |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3948 / 0.3878 | 0.382 / 0.359 | 0.3862 / 0.3608 | 0.3875 / 0.3567 |
| ETTh1 | 192 | 0.4254 / 0.4347 | 0.404 / 0.397 | 0.4093 / 0.4040 | 0.4118 / 0.3998 |
| ETTh2 | 96 | 0.3491 / 0.3025 | 0.338 / 0.275 | 0.3430 / 0.2808 | 0.3323 / 0.2730 |
| ETTh2 | 192 | 0.3859 / 0.3667 | 0.376 / 0.341 | 0.3835 / 0.3440 | 0.3740 / 0.3380 |
| ETTm1 | 96 | 0.4664 / 0.5445 | 0.344 / 0.293 | 0.3486 / 0.2987 | 0.3398 / 0.2926 |
| ETTm1 | 192 | 0.5275 / 0.7261 | 0.361 / 0.323 | 0.3651 / 0.3330 | 0.3581 / 0.3263 |
| ETTm2 | 96 | 0.2943 / 0.2064 | 0.256 / 0.163 | 0.2596 / 0.1700 | 0.2494 / 0.1607 |
| ETTm2 | 192 | 0.3362 / 0.2713 | 0.293 / 0.219 | 0.2985 / 0.2285 | 0.2876 / 0.2194 |

Average relative regression versus the fixed gold standard was approximately
15.7% MAE and 37.0% MSE. ETTm1-96 regressed 35.6% MAE / 85.8% MSE, while
ETTm1-192 regressed 46.1% MAE / 124.8% MSE.

## Root causes

1. Phase-mean pooling discarded local routed-token information before decoding.
2. The output was constrained to convex mixtures of three recently observed
   profiles, preventing generation of new phase shapes.
3. A single amplitude per future period could not represent phase-local changes.
4. Soft shift mixtures smoothed peaks, consistent with MSE degrading more than
   MAE.
5. Phase displacement was direct and limited to ±1 for every horizon, while
   level increments accumulated and could drift.
6. Replacing the original predictor removed the model's expressive fallback and
   made imperfect period choices much more damaging.

These findings motivated moving transport from observed output profiles into the
routed latent phase representation while retaining the original value-decoding
form in LPTD.
