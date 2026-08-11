# Circular Phase Transport Decoder

## Status

Circular Phase Transport Decoder (CPTD) is an implemented, default-off research
module. It has not yet been trained or evaluated. No accuracy improvement is
claimed until a frozen experiment is compared with
`docs/PHASEFORMER_GOLD_STANDARD.md` under a compatible protocol.

## Motivation

The original PhaseFormer groups observations by a fixed phase index and directly
regresses future phase values. Weak-period data can preserve a recognizable
cycle while its peaks move between neighboring phase slots, its amplitude
changes, or its period level drifts. A time-domain residual can absorb these
effects, but it bypasses the phase representation.

CPTD instead treats weak periodicity as evolution on a circular phase coordinate.
It replaces the direct value predictor and generates every future period only
from routed phase latents and observed phase profiles. It never maps the raw time
sequence directly to the forecast and never blends a separate sequence forecast
with the PhaseFormer output.

## Formulation

For recent period profile `x_r` with phase length `L`, CPTD separates the period
level and centered phase shape:

```text
mu_r = mean_phase(x_r)
q_r  = x_r - mu_r
```

For each future period `j`, a shared dynamics head consumes the phase-pooled
routing latent and normalized horizon coordinate. It predicts:

- convex recent-memory weights `omega[j, r]`;
- convex circular-shift weights `pi[j, r, s]`, for `s in [-S, S]`;
- bounded amplitude `a[j]`;
- bounded period-level increment `delta_mu[j]`.

The future centered shape and level are:

```text
q_hat[j]  = a[j] * sum_r omega[j,r] * sum_s pi[j,r,s] * roll(q_r, s)
mu_hat[j] = mu_last + cumulative_sum(delta_mu)[j]
x_hat[j]  = mu_hat[j] + q_hat[j]
```

The convex transport preserves the phase mean before the explicit level update.
Amplitude is bounded in log space, while level increments are scaled by recent
period-level changes plus a small normalized-series scale floor. These constraints
keep the decoder interpretable and prevent it from degenerating into an
unrestricted sequence regressor.

## Default configuration

| Setting | Default | Meaning |
|---|---:|---|
| `use_phase_transport_decoder` | `false` | Replace direct phase prediction |
| `phase_transport_hidden` | 8 | Shared dynamics-head width |
| `phase_transport_memory` | 3 | Number of recent period profiles |
| `phase_transport_max_shift` | 1 | Maximum circular phase displacement |
| `phase_transport_max_log_amplitude` | 0.5 | Bound on log-amplitude change |
| `phase_transport_max_level_step` | 1.0 | Multiplier for bounded level increments |
| `phase_transport_temperature` | 1.0 | Memory/shift softmax temperature |
| `phase_transport_prior_logit` | 3.0 | Persistence initialization strength |

The `phase_transport` ablation also disables absolute phase positional embeddings.
This lets the routing summary condition transport without assigning an absolute
preference to a phase label; phase order remains present in the observed profiles
and circular roll operator.

## Initialization and invariants

The final dynamics layer is zero-initialized with priors favoring the newest
period and zero phase shift. Amplitude starts at one and level drift at zero.
The initial model is therefore close to recent-period phase persistence rather
than a random future trajectory.

When `seq_len` is not divisible by `period_len`, transport memory is formed from
complete periods ending at the last real observation. The main PhaseFormer path
keeps its existing circular right-padding, but padded values are never treated as
observed transport state. CPTD requires `seq_len >= period_len`.

Conditional on the predicted transport parameters, circularly shifting every
input profile shifts the output by the same amount. Convex memory and shift
weights sum to one. The transport path preserves centered-profile mean; only the
explicit level path can change period mean.

## Integration and compatibility

- Implementation: `src/models/phase_transport.py`.
- Model integration: `src/models/PhaseFormer.py`.
- Preset mode: `phase_transport` in `src/models/phaseformer_presets.py`.
- Search mechanism: `phase_transport` in `scripts/search_phaseformer.py`.
- Diagnostics expose memory weights, shift weights, amplitude, level increments,
  and future levels for bad-case analysis.
- Enabling CPTD together with `use_weak_period_residual` raises `ValueError`.
  This prevents an experiment from being labeled phase-native while silently
  retaining a time-domain residual path.

Existing phase-only input or output calibrations remain technically composable,
but the first evaluation must use CPTD alone so its contribution is identifiable.

## Required validation

The first experiment round should compare original PhaseFormer, the strongest
residual reference, and CPTD on:

1. Weather-192, where channel-wise residual showed a strong validation signal;
2. ETTh2-720, where long-horizon drift favored residual prediction;
3. ETTm2-96, where existing phase adapters are strong;
4. Traffic-96, as a strong-period regression guardrail.

Required CPTD ablations are level only, shift only, level plus amplitude, the full
decoder, memory 1 versus 3, and removal of the circular transport constraint.
Search remains validation-isolated; final MSE and MAE claims use the repository
gold standard. Parameter count, training time, and inference time must accompany
accuracy results.

## Novelty boundary

The intended contribution is not a generic claim of first modeling phase
evolution. The narrower technical claim to investigate is a lightweight,
structured stochastic circulant transport decoder operating directly on explicit
PhaseFormer phase slots. Related work on phase alignment, cycle residuals,
general latent dynamics, and generative phase evolution must be reviewed before
making a novelty claim in a manuscript.
