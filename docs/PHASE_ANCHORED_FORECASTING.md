# Phase-Anchored Forecasting

## Status

Phase-Anchored Forecasting (PAF) is a default-off research mechanism. It
replaces the rejected LPTD experiment with a parameter-free change of phase
coordinates. It has not yet been evaluated, so no accuracy improvement is
claimed until results are compared with `docs/PHASEFORMER_GOLD_STANDARD.md`.

## Definition

Let `X[l,p]` be the normalized observation at phase slot `l` and input period
`p`. PAF selects the most recent real observation of every phase slot as its
origin:

```text
A[l]          = latest_real_observation(X[l,:])
X_center[l,p] = X[l,p] - A[l]
D_hat         = PhaseFormer(X_center)
Y_hat[l,j]    = A[l] + D_hat[l,j]
```

The model therefore forecasts how each phase coordinate evolves relative to
its own recent state. This is a coordinate reparameterization, not a second
forecast branch: embedding, cross-phase routing, and the original predictor all
remain unchanged.

## Design properties

1. **One phase path.** There is no parallel time-domain forecast, learned gate,
   transport head, or output mixture.
2. **No new parameters.** PAF only subtracts and restores a phase origin.
3. **No cumulative integration.** Every future period is decoded directly as a
   displacement from the same anchor, avoiding stepwise drift.
4. **Unrestricted value generation.** The original PhaseFormer predictor can
   generate arbitrary displacements; PAF imposes no convex transport or
   smoothing constraint.
5. **Phase-wise translation equivariance.** Within the normalized phase
   coordinate system, adding an arbitrary constant `c[l]` to the history of
   each phase slot leaves the centered input unchanged and adds exactly `c[l]`
   to the reconstructed forecast.

PAF cannot be combined with the weak-period time-domain residual. This keeps
the research mechanism identifiable as a single phase-native path.

## Incomplete periods

When the lookback is not divisible by `period_len`, each anchor is gathered
from the most recent real time index with the corresponding phase. Missing
slots in the final input period are filled with their anchors, so their centered
values are zero. Circular padding is never treated as a new observation in PAF.

PAF requires at least one complete input period. The original PhaseFormer path
retains its existing circular-padding behavior when PAF is disabled.

## Integration

- Transform: `src/models/phase_anchor.py`.
- Model integration: `src/models/PhaseFormer.py`.
- Preset and search name: `phase_anchor`.
- Configuration flag: `use_phase_anchor`.
- The model still returns reconstructed absolute phase predictions in
  `y_phase_steps`; intermediate `Z` is learned from centered phase trajectories.

## Initial validation

Use the same lookback, seed, optimizer, loss, and training budget as the matched
`original` run. Begin with the ten LPTD diagnostic tasks: ETTh1, ETTh2, ETTm1,
ETTm2, and Weather at horizons 96 and 192. Report deltas against both the matched
original and the fixed gold standard.

The key control is `original` versus `phase_anchor`; there are no mechanism
hyperparameters to tune. In addition to aggregate MSE and MAE, inspect whether
PAF reduces the phase-amplitude hallucination cases already documented for
ETTm1. Multi-seed confirmation is required for changes around one percent.
