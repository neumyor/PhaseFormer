# Trend-token interaction matrix

## Purpose

This round tests whether the learned trend token should only correct the final
forecast, read the phase tokens, write back to the phase tokens, or participate
in repeated/joint token interaction. The low-rank PhaseFormer-L results already
established the capacity settings used here; this round changes only the token
interaction graph.

## Controlled mechanisms

The arrow denotes information flow. `P` is the routed phase-token sequence and
`T` is one per-channel trend token with capacity `r=32`.

| Arm | Interaction | Definition |
|---|---|---|
| `trend_token_i0_late_fusion` | none | History token is decoded independently and fused at the output |
| `trend_token_i1_phase_to_trend` | `P -> T` | Trend token queries phase tokens before its horizon readout |
| `trend_token_i2_trend_to_phase` | `T -> P` | Phase tokens receive the trend state before the phase predictor |
| `trend_token_i3_parallel_bidir` | parallel `P <-> T` | Both directions read the original states and update in parallel |
| `trend_token_i6_alternating` | two-round alternating | Two sequential `P -> T -> P` rounds |
| `trend_token_i7_joint` | joint attention | Full self-attention over `[T, P_1, ..., P_M]` |

The phase-to-trend path preserves the original trend-token behavior. The
trend-to-phase single-token path is intentionally retained as a broadcast
control: with one key/value token its attention weights are necessarily one.
The joint and alternating arms test whether the phase path benefits from a
state update rather than only from a final residual forecast.

## Frozen protocol

- Data: ETTh1, ETTh2, ETTm1, ETTm2, Weather.
- Lookback: 720; phase period: 24.
- Horizons: 96, 192, 720.
- Seeds: 2021, 2022.
- Loss: Huber; epochs: 8; lowest validation-loss checkpoint.
- Trend token: `token_dim=32`, causal kernel 5, dilations `(1, 2)`.
- Readout: direct `token -> H` horizon readout.
- Fusion gate: static per-channel sigmoid, initialized at 0.2.
- Learning rate: the existing per-dataset/per-horizon base preset; no new grid.
- The primary matrix is validation-only. Test is not read for mechanism or
  hyperparameter selection.

The matrix contains the six requested arms plus a matched `phase_only` control:
`5 datasets × 3 horizons × 2 seeds × 7 arms = 210 jobs`.

## Comparability and reporting

Every arm uses the same history encoder, token dimension, direct horizon head,
gate initialization, loss, data split, and seed. Interaction modules are
constructed only for the arms that use them. The report must include parameter
counts and wall-clock time because joint/alternating attention has additional
capacity and compute.

The result is an architecture screen. Any later test evaluation must be a
separate, predeclared stage after validation results are frozen; it must not
replace this matrix or be described as an unbiased benchmark.

