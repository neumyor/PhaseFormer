# Latent Phase Transport Decoder

## Status

Latent Phase Transport Decoder (LPTD) is a default-off research module. It
replaces the failed output-space CPTD design, but it has not yet been trained or
evaluated. No improvement is claimed until results are compared with
`docs/PHASEFORMER_GOLD_STANDARD.md` under a compatible protocol.

## Design objective

Weak periodicity can move information between neighboring phase positions. The
original PhaseFormer uses static phase coordinates when decoding routed tokens,
while CPTD overcorrected this limitation by discarding the learned value decoder
and restricting forecasts to transported recent observations.

LPTD keeps PhaseFormer's learned latent representation and value-generation
capacity. It transports routed latent phase tokens separately for each future
period, then decodes values with the same horizon-specific linear form as the
original predictor. It never reads recent raw period profiles and never creates
a parallel time-domain forecast.

## Formulation

Let `Z[l]` be a routed latent token at phase position `l`. For future period `j`,
a shared local head receives `Z[l]` and the normalized horizon coordinate. It
predicts a distribution over neighboring circular shifts:

```text
pi[j,l,:] = softmax(TransportHead([Z[l], (j+1)/P_out]))
```

The future-period latent and value are:

```text
Z_tilde[j,l] = sum_s pi[j,l,s] * Z[(l-s) mod L]
y_hat[j,l]   = W[j] @ Z_tilde[j,l] + b[j]
```

The kernel is local and phase-specific: different phase slots may transport
different neighboring information. Cross-phase routing has already injected
global context into each token, so no phase pooling or separate global network
is required.

## Safety and expressivity properties

1. **Original-predictor containment.** When the transport kernel is the identity,
   `Z_tilde[j,l] = Z[l]` and LPTD is exactly the original linear
   `PhasePredictor`. Unlike CPTD, the hypothesis space does not collapse to a
   recent-observation dictionary.
2. **Local information preservation.** LPTD never averages `Z` across the phase
   axis before decoding.
3. **New-shape generation.** Values are generated from latent tokens with
   horizon-specific weights; they are not convex combinations of observed values.
4. **No accumulated level state.** LPTD directly predicts every future period,
   avoiding CPTD's cumulative level-step error.
5. **Circular equivariance of transport.** Relabeling all latent phase positions
   by a circular shift produces the same shift in transported outputs, conditional
   on the surrounding PhaseFormer representation.
6. **Linear memory.** The implementation loops over horizons and shifts instead
   of materializing `(B,C,P_out,L,S,D)`, keeping peak memory close to the original
   latent tensor.

## Defaults

| Setting | Default | Meaning |
|---|---:|---|
| `use_lptd` | `false` | Replace the original value predictor with LPTD |
| `lptd_hidden` | 8 | Width of the shared local transport head |
| `lptd_max_shift` | 1 | Circular neighborhood radius |
| `lptd_temperature` | 1.0 | Shift-distribution temperature |
| `lptd_prior_logit` | 5.0 | Identity-transport initialization strength |

The final transport layer is zero-initialized with a center-shift bias of 5.0.
For the default three-position kernel, the initial identity probability is about
98.7%. The horizon-specific value projection uses the same parameterization and
initialization family as the original `nn.Linear(D, P_out)` predictor.

## Integration

- Implementation: `src/models/latent_phase_transport.py`.
- Model integration: `src/models/PhaseFormer.py`.
- Preset/ablation mode: `lptd`.
- Search mechanism: `lptd` in `scripts/search_phaseformer.py`.
- Diagnostics: per-horizon and per-phase shift weights, entropy, and identity
  probability.
- LPTD and weak-period time-domain residual cannot be enabled together.
- LPTD currently targets PhaseFormer's default linear predictor; enabling the
  predictor MLP or predictor dropout with LPTD is rejected explicitly.

## First validation round

The first round should reproduce the eight CPTD tasks before expanding:

- ETTh1 and ETTh2, horizons 96 and 192;
- ETTm1 and ETTm2, horizons 96 and 192.

Compare original, latest, failed CPTD records, and LPTD. Report both the fixed
gold-standard delta and matched-run delta. The primary guardrail is that LPTD
must remove CPTD's ETTm collapse before it is considered for broader research.

Required low-cost ablations:

1. `max_shift=0`, which must behave as the original linear predictor;
2. `max_shift=1`, the proposed default;
3. shared phase kernel versus the proposed phase-local kernel;
4. identity-prior logits 3 and 5;
5. shift entropy and per-channel bad-case analysis.

If `max_shift=0` does not match an original rerun within expected seed variation,
the experiment pipeline or configuration is wrong and transport conclusions must
not be drawn.
