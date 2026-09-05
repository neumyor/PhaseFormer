# Weak-residual round-2 tuning on ETTm1 H96/H192 — TEST-selection record

- date 2026-09-05; seed 2021 only; commit see git.

## ⚠ Selection mode (user-mandated, TEST-set selection)
- The best parameter combination per setting is chosen by the error on the **TEST split** (round-1 mandate carried into round-2). Results are **not** blind/unbiased generalization — selection and measurement share the same (seed-2021) test split. Full pooled trajectory in `etm1_all.csv` (round-1 15 + round-2 30 per setting). No 3-seed confirmation.
## Target & pool
- Golden: ETTm1-96 **0.293/0.344**, ETTm1-192 **0.323/0.361**. Round-1 winner both settings: c13_mae_lr05_rlc (ETTm1-96 0.2958/0.3400, ETTm1-192 0.3253/0.3565) — MAE already below Golden, MSE still ~0.7-1.0% above.
- Round-2 space (30 fresh retrains/setting): mae-core fine grid (lr {.35,.4,.45,.55,.6,.7} x gate {.1,.15,.3,.45} x e{60,100} x rlc/shared heads, p48/p96, channel head) + a never-tried **mse-loss** axis (lr {.3-.1}, gate {.1,.3}, e {40-100}). All capacity base, p24 default, seed 2021, L=720.
## Pooled winner (argmin TEST MSE across round1+round2)
| Setting | Golden | round1 winner | pooled winner | winner TEST | dMSE% | dMAE% |
|---|---:|---:|---|---:|---:|---|
| ETTm1-96 | 0.293/0.344 | c13_mae_lr05_rlc 0.2958/0.3399 | round2:n14_mae_rlc_lr060_e60 | **0.2928/0.3378** | +0.08 | +1.81 | BOTH |
| ETTm1-192 | 0.323/0.361 | c13_mae_lr05_rlc 0.3253/0.3565 | round2:n13_mae_rlc_lr040_e60 | **0.3252/0.3569** | -0.68 | +1.13 | MAE-only |

## ETTm1 H96 — pooled candidates by TEST MSE (source: r1/r2)

| rk | src | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | round2 | n14_mae_rlc_lr060_e60 | mae | 0.6 | 0.2 | repeat_last_cycle | 24 | 60 | 0.36228 | 0.29275 | 0.33776 |
| 2 | round2 | n19_mae_chan_lr050 | mae | 0.5 | 0.2 | channel | 24 | 40 | 0.36898 | 0.29343 | 0.33478 |
| 3 | round2 | n17_mae_rlc_p48 | mae | 0.5 | 0.2 | repeat_last_cycle | 48 | 40 | 0.36847 | 0.29360 | 0.33461 |
| 4 | round2 | n12_mae_rlc_lr050_e100 | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 100 | 0.36153 | 0.29399 | 0.33826 |
| 5 | round2 | n11_mae_rlc_lr050_e60 | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 60 | 0.36321 | 0.29416 | 0.33848 |
| 6 | round2 | n13_mae_rlc_lr040_e60 | mae | 0.4 | 0.2 | repeat_last_cycle | 24 | 60 | 0.36378 | 0.29495 | 0.33911 |
| 7 | round2 | n08_mae_rlc_g015 | mae | 0.5 | 0.15 | repeat_last_cycle | 24 | 40 | 0.36438 | 0.29509 | 0.33878 |
| 8 | round2 | n07_mae_rlc_g010 | mae | 0.5 | 0.1 | repeat_last_cycle | 24 | 40 | 0.36789 | 0.29511 | 0.33820 |
| 9 | round2 | n09_mae_rlc_g030 | mae | 0.5 | 0.3 | repeat_last_cycle | 24 | 40 | 0.36913 | 0.29577 | 0.33946 |
| 10 | round1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36462 | 0.29578 | 0.33995 |
| 11 | round2 | n03_mae_lr045_rlc | mae | 0.45 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36469 | 0.29583 | 0.34023 |
| 12 | round2 | n10_mae_rlc_g045 | mae | 0.5 | 0.45 | repeat_last_cycle | 24 | 40 | 0.36904 | 0.29584 | 0.33782 |
| 13 | round2 | n04_mae_lr055_rlc | mae | 0.55 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36434 | 0.29586 | 0.33975 |
| 14 | round2 | n05_mae_lr060_rlc | mae | 0.6 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36420 | 0.29590 | 0.33959 |
| 15 | round2 | n02_mae_lr040_rlc | mae | 0.4 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36531 | 0.29605 | 0.34057 |
| 16 | round2 | n22_mse_rlc_lr050_e100 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 100 | 0.38111 | 0.29651 | 0.34836 |
| 17 | round2 | n01_mae_lr035_rlc | mae | 0.35 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36667 | 0.29652 | 0.34089 |
| 18 | round1 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.36110 | 0.29717 | 0.34019 |
| 19 | round1 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.37590 | 0.29772 | 0.34727 |
| 20 | round2 | n06_mae_lr070_rlc | mae | 0.7 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36475 | 0.29805 | 0.34030 |
| 21 | round1 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.36178 | 0.29849 | 0.34133 |
| 22 | round2 | n15_mae_shr_lr045_e60 | mae | 0.45 | 0.2 | shared | 24 | 60 | 0.36072 | 0.29888 | 0.34148 |
| 23 | round1 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.36083 | 0.29897 | 0.34134 |
| 24 | round1 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.36936 | 0.29948 | 0.35013 |
| 25 | round2 | n21_mse_rlc_lr050_e60 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 60 | 0.38444 | 0.30008 | 0.35100 |
| 26 | round2 | n16_mae_shr_lr050_e80 | mae | 0.5 | 0.2 | shared | 24 | 80 | 0.36159 | 0.30061 | 0.34080 |
| 27 | round2 | n27_mse_rlc_lr060 | mse | 0.6 | 0.2 | repeat_last_cycle | 24 | 40 | 0.38523 | 0.30096 | 0.34948 |
| 28 | round2 | n24_mse_rlc_lr030_e60 | mse | 0.3 | 0.2 | repeat_last_cycle | 24 | 60 | 0.38835 | 0.30113 | 0.35055 |
| 29 | round1 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.36422 | 0.30140 | 0.34511 |
| 30 | round2 | n25_mse_rlc_lr070_e60 | mse | 0.7 | 0.2 | repeat_last_cycle | 24 | 60 | 0.38182 | 0.30176 | 0.35004 |
| 31 | round2 | n20_mse_rlc_lr050 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.38768 | 0.30268 | 0.35098 |
| 32 | round1 | c14_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.37477 | 0.30279 | 0.34587 |
| 33 | round2 | n26_mse_rlc_lr040 | mse | 0.4 | 0.2 | repeat_last_cycle | 24 | 40 | 0.38982 | 0.30392 | 0.35217 |
| 34 | round1 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.36867 | 0.30397 | 0.35034 |
| 35 | round2 | n29_mse_rlc_g030 | mse | 0.5 | 0.3 | repeat_last_cycle | 24 | 40 | 0.38927 | 0.30475 | 0.35408 |
| 36 | round2 | n23_mse_rlc_lr100 | mse | 1.0 | 0.2 | repeat_last_cycle | 24 | 30 | 0.38077 | 0.30525 | 0.35565 |
| 37 | round1 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.37285 | 0.30641 | 0.35294 |
| 38 | round2 | n30_mse_shr_lr050 | mse | 0.5 | 0.2 | shared | 24 | 40 | 0.37539 | 0.30647 | 0.35526 |
| 39 | round1 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.37187 | 0.30667 | 0.35071 |
| 40 | round1 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.37065 | 0.30672 | 0.35223 |
| 41 | round1 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.36958 | 0.30674 | 0.35182 |
| 42 | round2 | n28_mse_rlc_g010 | mse | 0.5 | 0.1 | repeat_last_cycle | 24 | 40 | 0.39188 | 0.30997 | 0.35727 |
| 43 | round1 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.37306 | 0.30997 | 0.35500 |
| 44 | round1 | c15_p96 | huber | 1.0 | 0.2 | shared | 96 | 30 | 0.38377 | 0.31026 | 0.35226 |
| 45 | round2 | n18_mae_rlc_p96 | mae | 0.5 | 0.2 | repeat_last_cycle | 96 | 40 | 0.39387 | 0.31971 | 0.35726 |

## ETTm1 H192 — pooled candidates by TEST MSE (source: r1/r2)

| rk | src | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | round2 | n13_mae_rlc_lr040_e60 | mae | 0.4 | 0.2 | repeat_last_cycle | 24 | 60 | 0.48069 | 0.32518 | 0.35692 |
| 2 | round1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48137 | 0.32527 | 0.35652 |
| 3 | round2 | n03_mae_lr045_rlc | mae | 0.45 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48208 | 0.32542 | 0.35666 |
| 4 | round2 | n04_mae_lr055_rlc | mae | 0.55 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48089 | 0.32544 | 0.35640 |
| 5 | round2 | n11_mae_rlc_lr050_e60 | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 60 | 0.48043 | 0.32565 | 0.35652 |
| 6 | round2 | n02_mae_lr040_rlc | mae | 0.4 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48329 | 0.32589 | 0.35702 |
| 7 | round2 | n09_mae_rlc_g030 | mae | 0.5 | 0.3 | repeat_last_cycle | 24 | 40 | 0.48635 | 0.32616 | 0.35719 |
| 8 | round2 | n14_mae_rlc_lr060_e60 | mae | 0.6 | 0.2 | repeat_last_cycle | 24 | 60 | 0.48052 | 0.32648 | 0.35645 |
| 9 | round2 | n01_mae_lr035_rlc | mae | 0.35 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48498 | 0.32680 | 0.35761 |
| 10 | round2 | n06_mae_lr070_rlc | mae | 0.7 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48121 | 0.32706 | 0.35717 |
| 11 | round2 | n05_mae_lr060_rlc | mae | 0.6 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48157 | 0.32721 | 0.35705 |
| 12 | round2 | n08_mae_rlc_g015 | mae | 0.5 | 0.15 | repeat_last_cycle | 24 | 40 | 0.47808 | 0.32737 | 0.35798 |
| 13 | round2 | n12_mae_rlc_lr050_e100 | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 100 | 0.47652 | 0.32765 | 0.35735 |
| 14 | round2 | n10_mae_rlc_g045 | mae | 0.5 | 0.45 | repeat_last_cycle | 24 | 40 | 0.48718 | 0.32790 | 0.35904 |
| 15 | round2 | n25_mse_rlc_lr070_e60 | mse | 0.7 | 0.2 | repeat_last_cycle | 24 | 60 | 0.49647 | 0.32881 | 0.36726 |
| 16 | round2 | n07_mae_rlc_g010 | mae | 0.5 | 0.1 | repeat_last_cycle | 24 | 40 | 0.48375 | 0.32912 | 0.35820 |
| 17 | round1 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48811 | 0.32917 | 0.36398 |
| 18 | round2 | n29_mse_rlc_g030 | mse | 0.5 | 0.3 | repeat_last_cycle | 24 | 40 | 0.50115 | 0.32991 | 0.36834 |
| 19 | round2 | n21_mse_rlc_lr050_e60 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 60 | 0.49678 | 0.32996 | 0.36754 |
| 20 | round2 | n17_mae_rlc_p48 | mae | 0.5 | 0.2 | repeat_last_cycle | 48 | 40 | 0.49139 | 0.33009 | 0.35964 |
| 21 | round2 | n23_mse_rlc_lr100 | mse | 1.0 | 0.2 | repeat_last_cycle | 24 | 30 | 0.50187 | 0.33042 | 0.36919 |
| 22 | round2 | n27_mse_rlc_lr060 | mse | 0.6 | 0.2 | repeat_last_cycle | 24 | 40 | 0.49833 | 0.33056 | 0.36809 |
| 23 | round2 | n28_mse_rlc_g010 | mse | 0.5 | 0.1 | repeat_last_cycle | 24 | 40 | 0.49775 | 0.33072 | 0.36849 |
| 24 | round2 | n20_mse_rlc_lr050 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.49958 | 0.33094 | 0.36846 |
| 25 | round2 | n26_mse_rlc_lr040 | mse | 0.4 | 0.2 | repeat_last_cycle | 24 | 40 | 0.50231 | 0.33115 | 0.36752 |
| 26 | round2 | n22_mse_rlc_lr050_e100 | mse | 0.5 | 0.2 | repeat_last_cycle | 24 | 100 | 0.49356 | 0.33135 | 0.36881 |
| 27 | round2 | n24_mse_rlc_lr030_e60 | mse | 0.3 | 0.2 | repeat_last_cycle | 24 | 60 | 0.50036 | 0.33156 | 0.36879 |
| 28 | round1 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.48441 | 0.33244 | 0.35877 |
| 29 | round1 | c14_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.48880 | 0.33375 | 0.36739 |
| 30 | round1 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.47918 | 0.33443 | 0.35988 |
| 31 | round2 | n15_mae_shr_lr045_e60 | mae | 0.45 | 0.2 | shared | 24 | 60 | 0.47685 | 0.33604 | 0.36152 |
| 32 | round1 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.48184 | 0.33640 | 0.36148 |
| 33 | round2 | n16_mae_shr_lr050_e80 | mae | 0.5 | 0.2 | shared | 24 | 80 | 0.47846 | 0.33644 | 0.36199 |
| 34 | round1 | c15_p96 | huber | 1.0 | 0.2 | shared | 96 | 30 | 0.49077 | 0.33665 | 0.37032 |
| 35 | round1 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.48839 | 0.33706 | 0.36662 |
| 36 | round2 | n19_mae_chan_lr050 | mae | 0.5 | 0.2 | channel | 24 | 40 | 0.48990 | 0.33728 | 0.35843 |
| 37 | round1 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.48076 | 0.33732 | 0.36150 |
| 38 | round1 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.48271 | 0.33733 | 0.36677 |
| 39 | round1 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.48720 | 0.33737 | 0.36685 |
| 40 | round1 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.49024 | 0.34043 | 0.36943 |
| 41 | round1 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.48820 | 0.34075 | 0.36819 |
| 42 | round2 | n30_mse_shr_lr050 | mse | 0.5 | 0.2 | shared | 24 | 40 | 0.49748 | 0.34093 | 0.37225 |
| 43 | round1 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.49107 | 0.34391 | 0.37257 |
| 44 | round1 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.48948 | 0.34409 | 0.37257 |
| 45 | round2 | n18_mae_rlc_p96 | mae | 0.5 | 0.2 | repeat_last_cycle | 96 | 40 | 0.51540 | 0.35991 | 0.37459 |

## Analysis & verdict (round 2)
- **ETTm1-96 — nominal dual-metric beat, but marginal.** Pooled winner `n14_mae_rlc_lr060_e60` TEST 0.29275/0.33776 vs Golden 0.293/0.344 → dMSE **+0.08%** (first time MSE < 0.293), dMAE +1.81%. This is single-seed (2021) **TEST-set-selected**: selection and measurement share the test split, and the +0.08% MSE margin is inside run-to-config noise, so it must be read as a marginal single-seed screening result, NOT a formal beat. A formal "beat Golden" requires the 3-seed (2021/22/23) mean+std protocol with a frozen (unexposed) config.
- **ETTm1-192 — still MAE-only.** Best pooled MSE is 0.3252 (`n13_mae_rlc_lr040_e60`, effectively tied with round-1 c13 0.3253); none of the 45 pooled candidates put MSE below Golden 0.323 (−0.68%). MAE is ~1.1-1.3% below Golden on the whole rlc/mae top cluster. The H192 MSE seems pinned ≈0.325 across lr {0.4-0.6} × e {40-100} × gate {0.1-0.45} × heads {rlc,shared,channel} — capacity-base weak residual saturates there.
- **mse-loss axis is a clean negative result.** Training directly on `loss_func='mse'` (11 candidates/setting, never tried in round 1) did NOT lower test MSE; best mse candidate was −1.2% (H96, n22) / −1.8% (H192, n25) vs the mae-core winners. mae-loss training generalizes better to the all-channel average MSE metric. Not worth pursuing further within this mechanism/capacity.
- **What transferred across horizons:** the winning structure is mae-loss × repeat_last_cycle × p24 × gate0.2 × lr∈{0.4-0.6} × e60 — n13 (lr0.4) is top on H192 and 6th on H96; n14 (lr0.6) is top on H96 and 8th on H192. Longer training (e60/e100) helped marginally on both; the channel head matched rlc on MSE (n19) with the best MAE on H96 (0.33478) but did not help H192.

## Artifacts
- pooled trajectory: `etm1_all.csv`; winner: `etm1_winner.csv`; record: this file
