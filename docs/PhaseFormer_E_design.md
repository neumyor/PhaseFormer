# PhaseFormer-E

PhaseFormer-E is the structure selected by the structure ablation of 2026-11-12
(`research_runs/b0hp_structure_v1_local/report.md`). It keeps the two-branch layout of
PhaseFormer+ and removes everything that reads the window directly.

## Structure

1. The normalized window is folded into cycles, `X~` of shape K x P.
2. Embedding: `U = W_e X~ + b_e`, shape d x P. There is no LayerNorm after the embedding.
3. Phase branch: `U` plus positional embeddings goes through N_r cross-phase routing layers
   (Post-LN inside the layers) and the shared predictor, which gives the phase forecast `F_p`.
4. Trend branch: the mean over the P phases of `U` goes through a linear encoder to a
   3-dimensional state `g`, then the polynomial decoder gives the absolute trend forecast
   `F_b = x_L 1 + Psi_H (Gamma g + c)`. The pooled history of the window is not an input.
5. Two static gates fuse the forecasts on the cycle levels (M_c) and the within-cycle
   shapes (J_c): `y = (1-g_l) M_c F_p + g_l M_c F_b + (1-g_s) J_c F_p + g_s J_c F_b`.

## Code

- Script: `scripts/phaseformer_e.py` (jobs, check, run, report). It reuses
  `scripts/b0hp_structure_ablation.py`, where arm `e` swaps the embedding LayerNorm for an
  identity and switches the level encoder to the token-only input (`level_input='phase_token'`).
- Configuration: every setting keeps the recorded B0H+ configuration of
  `docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`; only the model structure differs.
- Comparison: the B0H+ arm of the same script is the recorded model rerun with the same
  seeds and code, so each E run has a paired B0H+ run.

## Evidence so far

Six settings (ETTh1, ETTm1, Weather at 96 and 336), three seeds: E is 0.41% (MSE) behind
B0H+ on average, 3.0% behind on ETTh1-336 and within 0.75% on the other five.
