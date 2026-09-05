# Weak-residual best-on-TEST results per setting (with reproduction commands)

- date 2026-09-05; mechanism `weak_residual`, hypothesis `none`, variant `full`,
  capacity `base`, lookback 720, **seed 2021 (single seed only)**, `--percent 100`.
- Golden = fixed original-PhaseFormer paper test numbers (reference only).
- dMSE% = (Golden − model)/Golden × 100, positive = model better. MSE/MAE are
  all-channel averages on the TEST split (repo formal protocol).
- Interpreter: `/home/niuyiming/.conda/envs/py310/bin/python`. Results are
  deterministic for a fixed config/seed/hardware (verified by exact replica
  matches), so rerunning the commands below reproduces the numbers.

## ⚠ Selection / evaluation caveats (MANAGE_RULES)

- The 4 tuned settings (ETTh1-96/192, ETTm1-96/192) were selected by the
  **error on the TEST split** (user mandate 2026-09-05). Selection and
  measurement share the same seed-2021 test split → these are **not**
  blind/unbiased generalization, and any later "confirmation" on seed 2021 is
  itself affected by that exposure.
- **Single seed (2021) only; no 3-seed (2021/22/23) mean+std confirmation.** A
  formal "beat Golden" requires the 3-seed protocol with the winning config
  frozen. ETTh1-96 and ETTm1-96 show *nominal* dual-metric crossings here, but
  the MSE margins (+2.09% and +0.08%) are single-seed, test-selected numbers.
- Full per-candidate trajectories are preserved in the record docs
  `docs/Weak_residual_tune_4losing_settings_record.md` (round-1, 4 losing
  settings) and `docs/Weak_residual_etm1_round2_tuning_record.md` (round-2,
  ETTm1 only).

## Best-on-TEST per setting

Tuned configs shown with their short tag used in the tuning campaigns.

| Setting | Golden | source | best tag | parameter combo | best TEST | dMSE% | dMAE% |
|---|---|---:|---|---|---:|---:|---:|
| ETTh1_h96 | 0.359 / 0.382 | tuned r1 | `c13_mae_lr05_rlc` | mae, lr×0.5, gate0.2, `repeat_last_cycle`, p24, e40, pat10 | **0.3515 / 0.3782** | +2.09 | +0.98 |
| ETTh1_h192 | 0.397 / 0.404 | tuned r1 | `c13_mae_lr05_rlc` | mae, lr×0.5, gate0.2, `repeat_last_cycle`, p24, e40, pat10 | **0.3961 / 0.4054** | +0.23 | −0.34 |
| ETTh2_h96 | 0.275 / 0.338 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.2713 / 0.3338** | +1.35 | +1.24 |
| ETTh2_h192 | 0.341 / 0.376 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.3373 / 0.3764** | +1.08 | −0.12 |
| ETTm1_h96 | 0.293 / 0.344 | tuned r1+r2 | `n14_mae_rlc_lr060_e60` | mae, lr×0.6, gate0.2, `repeat_last_cycle`, p24, e60, pat15 | **0.2928 / 0.3378** | +0.08 | +1.81 |
| ETTm1_h192 | 0.323 / 0.361 | tuned r1+r2 | `n13_mae_rlc_lr040_e60` | mae, lr×0.4, gate0.2, `repeat_last_cycle`, p24, e60, pat15 | **0.3252 / 0.3569** | −0.68 | +1.13 |
| ETTm2_h96 | 0.163 / 0.256 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.1595 / 0.2506** | +2.15 | +2.11 |
| ETTm2_h192 | 0.219 / 0.293 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.2157 / 0.2881** | +1.51 | +1.69 |
| Weather_h96 | 0.148 / 0.195 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.1468 / 0.1920** | +0.78 | +1.53 |
| Weather_h192 | 0.193 / 0.237 | baseline | baseline-full weak residual | huber, lr×1.0, gate0.2, `shared`, p24, e30, pat8 | **0.1926 / 0.2360** | +0.23 | +0.40 |

Reading: **both metrics below Golden** on ETTh2/ETTm2/Weather (baseline) and on
ETTh1-96 / ETTm1-96 (tuned). **Mixed** on ETTh1-192 (MSE below, MAE +0.34% above)
and ETTm1-192 (MAE below, MSE −0.68% above). ETTh2-192 baseline is mixed on MAE
(−0.12%).

## Reproduction commands

All tuned bests use `--overrides` JSON to set the mechanism knobs (`patience`,
`weak_period_residual_head_type`); these land in `hyperparams` and win over the
defaults. Run from the repo root. The best checkpoint is selected by val_loss
(EarlyStopping patience as shown); the TEST numbers above come from one frozen
TEST read of that `best.ckpt`.

### Tuned best — ETTh1-96 and ETTh1-192: `c13_mae_lr05_rlc`

```
/home/niuyiming/.conda/envs/py310/bin/python scripts/search_phaseformer.py \
  --dataset ETTh1 --horizon 96 --stage input_components \
  --mechanism weak_residual --input-hypothesis none --input-variant full \
  --lookback 720 --seed 2021 --loss mae --lr-multiplier 0.5 --max-epochs 40 \
  --period 24 --capacity base --percent 100 --num-workers 4 --bad-case-limit 0 \
  --require-cuda \
  --output-dir research_runs/weak_residual_tune_4losing_scratch \
  --overrides '{"patience":10,"weak_period_residual_head_type":"repeat_last_cycle"}'
```
Same command with `--horizon 192` for ETTh1-192.

### Tuned best — ETTm1-96: `n14_mae_rlc_lr060_e60`

```
/home/niuyiming/.conda/envs/py310/bin/python scripts/search_phaseformer.py \
  --dataset ETTm1 --horizon 96 --stage input_components \
  --mechanism weak_residual --input-hypothesis none --input-variant full \
  --lookback 720 --seed 2021 --loss mae --lr-multiplier 0.6 --max-epochs 60 \
  --period 24 --capacity base --percent 100 --num-workers 4 --bad-case-limit 0 \
  --require-cuda \
  --output-dir research_runs/weak_residual_tune_etm1_scratch \
  --overrides '{"patience":15,"weak_period_residual_head_type":"repeat_last_cycle"}'
```

### Tuned best — ETTm1-192: `n13_mae_rlc_lr040_e60`

```
/home/niuyiming/.conda/envs/py310/bin/python scripts/search_phaseformer.py \
  --dataset ETTm1 --horizon 192 --stage input_components \
  --mechanism weak_residual --input-hypothesis none --input-variant full \
  --lookback 720 --seed 2021 --loss mae --lr-multiplier 0.4 --max-epochs 60 \
  --period 24 --capacity base --percent 100 --num-workers 4 --bad-case-limit 0 \
  --require-cuda \
  --output-dir research_runs/weak_residual_tune_etm1_scratch \
  --overrides '{"patience":15,"weak_period_residual_head_type":"repeat_last_cycle"}'
```

### Baseline bests (ETTh2-96/192, ETTm2-96/192, Weather-96/192) — not tuned

Baseline-full weak residual already beats/ties Golden on these; no tuning was run
(they were out of the tuning budget). Command template with the setting's
`--dataset/--horizon` (no overrides — pure defaults: patience 8, gate 0.2,
head `shared`):

```
/home/niuyiming/.conda/envs/py310/bin/python scripts/search_phaseformer.py \
  --dataset <ETTh2|ETTm2|Weather> --horizon <96|192> --stage input_components \
  --mechanism weak_residual --input-hypothesis none --input-variant full \
  --lookback 720 --seed 2021 --loss huber --lr-multiplier 1.0 --max-epochs 30 \
  --period 24 --capacity base --percent 100 --num-workers 4 --bad-case-limit 0 \
  --require-cuda \
  --output-dir research_runs/input_components_h134_scratch
```

## Search-space recap for the tuned rows

- **Round-1** (4 losing settings, 15 candidates/setting, tags `c01..c15`):
  loss {huber, mae} × lr× {0.3, 0.5, 1, 2} × gate {0.05, 0.2, 0.5} × head
  {shared, repeat_last_cycle, lowpass} × period {24; ETTh1 probes 12/48; ETTm1
  probes 48/96} × epochs/patience {30/8 … 100/25}. Winner everywhere: `c13`
  (mae × lr×0.5 × repeat_last_cycle × p24 × e40/pat10).
- **Round-2** (ETTm1 only, 30 candidates/setting, tags `n01..n30`): a fine grid
  around the round-1 winner (lr {.35,.40,.45,.55,.60,.70}, gate {.10,.15,.30,
  .45}, e {60,100}, shared/p48/p96/channel-head probes) plus an **mse-loss axis**
  (11 rows). The mse-loss axis was a clean negative result (best −1.2%/−1.8% vs
  the mae winners); the mae × `repeat_last_cycle` × p24 structure with longer
  training (e60) won on both ETTm1 horizons.

## Files of record (all committed with this doc)

- `docs/Weak_residual_tune_4losing_settings_record.md` — round-1 TEST-selection
  record, ETTh1/ETTm1 × H96/H192 per-candidate tables.
- `docs/Weak_residual_etm1_round2_tuning_record.md` — round-2 TEST-selection
  record, pooled (round-1 15 + round-2 30) ETTm1 per-candidate tables.
- Raw CSVs / run dirs stay under `research_runs/` (gitignored by design).
