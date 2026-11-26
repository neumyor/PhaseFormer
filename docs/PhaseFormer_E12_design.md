# PhaseFormer-E1, -E2

Two cumulative structure changes on PhaseFormer-E (`docs/PhaseFormer_E_design.md`):
E1 = E + change 1, E2 = E1 + change 2. Each step is measured against the version before
it on the same seeds, so each change gets its own paired effect.

## Change 1 (E1): anchor the trend forecast on the last cycle mean

In E the trend encoder reads the slot mean of the un-normalized tokens. Because the
embedding is one linear map per slot, `mean_p U_p = W_e m + b_e`, where `m` holds the K
cycle means of the window: the trend state `g` is a linear function of the cycle means.
The forecast `F_b = x_L 1 + Psi_H (Gamma g + c)` is anchored on the last value
`x_L = m_K + s_last + e`, which carries the seasonal deviation of the last time step. The
windows end at every time of day, so `s_last` changes from window to window, and `m`
holds no information about it. The cycle means of `F_b`, which the level gate passes to
the forecast, therefore carry a noise term of the size of the seasonal component.

E1 anchors on the mean of the last cycle: `F_b = m_K 1 + Psi_H (Gamma g + c)` (and
`m_K 1 + W g + b` for the learned decoder heads). Anchor and encoder now read the same
information, and at initialization the trend forecast is the last cycle level.

## Change 2 (E2): Pre-LN routing layers

The routing layers of E are Post-LN, so the last operation before the predictor is a
LayerNorm over D. With D = 4 every slot token entering the predictor lies on a
two-dimensional sphere: the forecast of a slot cannot scale with its history, and the
model cannot express the per-phase linear map across cycles, the strongest simple
forecaster on periodic data.

E2 applies LayerNorm only to the inputs of the router attention and the MLP:

    h = LN1(x)
    x = x + Dropout(Receiver(h, Sender(R, h)))
    x = x + Dropout(MLP(LN2(x)))

The residual stream from `U` to the predictor is not normalized, so the phase forecast
contains `W_pred W_e X~`, a rank-D linear map from the history cycles to the future cycles
of every slot. For two or more layers the in-projection of later layers loses its
LayerNorm as well (`z = z_prev + W_in y`). The position embedding is still added in every
layer. LN1 and LN2 keep their parameters, so the parameter count changes only by the
removed in-projection LayerNorms.

## Module names (E1)

The two branches are drawn and named as mirrored pairs; `src/models/phaseformer_e1.py`
uses these names.

| Stage | Phase branch | Trend branch |
|---|---|---|
| Across phases | `CrossPhaseRouting`: P tokens -> M routers -> P tokens `h_p` | `CrossPhasePooling`: mean over P of `U`, then `W_g` -> state `g` |
| Decode | `PhaseDecoder` `Phi`: one linear map for all phases, `h_p` -> `F_p` | `TrendCorrection`: `F_b = m_K 1 + Psi_H g`, `dF = (g_l M_c + g_s J_c)(F_b - F_p)` |

Forecast: `y = F_p + dF`. Legacy names: `PhaseEmbedding` -> `PhaseTokenEmbedding`,
`routing_layers` -> `CrossPhaseRouting`, `predictor` (`PhasePredictor`) -> `PhaseDecoder`,
`lowrank_head.encoder` -> `CrossPhasePooling`, `lowrank_head.mix`/`decoder` together with
`lowrank_interaction_gate` and `lowrank_dynamic_fusion` -> `TrendCorrection`.

## Code

- Script: `scripts/phaseformer_e12.py` (jobs, check, run, report). It reuses
  `scripts/b0hp_structure_ablation.py`; arms `e1` and `e2` first build arm `e` and then
  apply changes 1 to n.
- Configuration and seeds: the recorded B0H+ configuration of every setting
  (`docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`) and seeds 2021-2023, as in
  `scripts/phaseformer_e.py`. The report reads B0H+ and E from
  `research_runs/phaseformer_e_v1` and E1, E2 from `research_runs/phaseformer_e12_v1`.
