# Weak-residual tuning on the 4 losing settings — TEST-selection record

> ⚠ **ETTm1 rows superseded by round-2 tuning (2026-09-05).** The best
> ETTm1-96 / ETTm1-192 configs are now round-2 winners
> `n14_mae_rlc_lr060_e60` (0.2928/0.3378) and `n13_mae_rlc_lr040_e60`
> (0.3252/0.3569); see `docs/Weak_residual_etm1_round2_tuning_record.md` and
> `research_runs/weak_residual_vs_golden_test_5ds/summary_best_test.md`. ETTh1
> rows below are unchanged (still current best).

- date 2026-09-05; commit see git.

## ⚠ Selection mode (user-mandated)
- 2026-09-05 the user REQUIRED the best parameter combination to be chosen by the error on the **TEST split**. This is **TEST-SET SELECTION**: the test split was used as the selection signal for all 60 candidates (one TEST read each).
- Consequence (MANAGE_RULES): results below are **not** blind/unbiased generalization — the reported winner was selected on the same split it is measured on. Full selection trajectory is preserved in `test_selection_all.csv`. Single seed 2021 only; no 3-seed confirmation. Any further 'confirmation' on fresh seeds is now itself affected by test exposure for seed 2021.
## TEST-selected winner vs Golden (positive d% = winner better)
| Setting | Golden | baseline TEST | winner | winner TEST | dMSE% | dMAE% |
|---|---:|---:|---|---:|---:|---|
| ETTh1-96 | 0.359/0.382 | 0.3656/0.3961 | c13_mae_lr05_rlc | **0.3515/0.3782** | +2.09 | +0.98 |
| ETTh1-192 | 0.397/0.404 | 0.4083/0.4206 | c13_mae_lr05_rlc | **0.3961/0.4054** | +0.23 | -0.34 |
| ETTm1-96 | 0.293/0.344 | 0.3067/0.3507 | c13_mae_lr05_rlc | **0.2958/0.3400** | -0.95 | +1.18 |
| ETTm1-192 | 0.323/0.361 | 0.3408/0.3682 | c13_mae_lr05_rlc | **0.3253/0.3565** | -0.70 | +1.24 |

## ETTh1 H96 — candidates by TEST MSE

| rk | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.72186 | 0.35150 | 0.37825 |
| 2 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.69158 | 0.35970 | 0.38806 |
| 3 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.69022 | 0.36005 | 0.38785 |
| 4 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.69181 | 0.36129 | 0.38852 |
| 5 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.70004 | 0.36335 | 0.39680 |
| 6 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.68654 | 0.36450 | 0.39536 |
| 7 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.67945 | 0.36462 | 0.39547 |
| 8 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.68660 | 0.36474 | 0.39495 |
| 9 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.68649 | 0.36556 | 0.39606 |
| 10 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.67839 | 0.36701 | 0.39615 |
| 11 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.68199 | 0.36807 | 0.39524 |
| 12 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.69343 | 0.36873 | 0.39030 |
| 13 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.69827 | 0.36882 | 0.38745 |
| 14 | c15_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.67515 | 0.37145 | 0.39977 |
| 15 | c14_p12 | huber | 1.0 | 0.2 | shared | 12 | 30 | 0.67565 | 0.37316 | 0.40007 |

## ETTh1 H192 — candidates by TEST MSE

| rk | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.96792 | 0.39607 | 0.40539 |
| 2 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.95989 | 0.40261 | 0.41380 |
| 3 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.96890 | 0.40344 | 0.41349 |
| 4 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.97081 | 0.40421 | 0.41407 |
| 5 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.96327 | 0.40563 | 0.41481 |
| 6 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.96850 | 0.40571 | 0.42069 |
| 7 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.96670 | 0.40592 | 0.41279 |
| 8 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.95471 | 0.40763 | 0.42200 |
| 9 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.95163 | 0.40831 | 0.42055 |
| 10 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.96078 | 0.40867 | 0.42175 |
| 11 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.95111 | 0.40895 | 0.41954 |
| 12 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.95911 | 0.40907 | 0.42249 |
| 13 | c14_p12 | huber | 1.0 | 0.2 | shared | 12 | 30 | 0.94743 | 0.40918 | 0.42200 |
| 14 | c15_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.95359 | 0.41259 | 0.42265 |
| 15 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.99162 | 0.41440 | 0.42369 |

## ETTm1 H96 — candidates by TEST MSE

| rk | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.36462 | 0.29578 | 0.33995 |
| 2 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.36110 | 0.29717 | 0.34019 |
| 3 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.37590 | 0.29772 | 0.34727 |
| 4 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.36178 | 0.29849 | 0.34133 |
| 5 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.36083 | 0.29897 | 0.34134 |
| 6 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.36936 | 0.29948 | 0.35013 |
| 7 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.36422 | 0.30140 | 0.34511 |
| 8 | c14_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.37477 | 0.30279 | 0.34587 |
| 9 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.36867 | 0.30397 | 0.35034 |
| 10 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.37285 | 0.30641 | 0.35294 |
| 11 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.37187 | 0.30667 | 0.35071 |
| 12 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.37065 | 0.30672 | 0.35223 |
| 13 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.36958 | 0.30674 | 0.35182 |
| 14 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.37306 | 0.30997 | 0.35500 |
| 15 | c15_p96 | huber | 1.0 | 0.2 | shared | 96 | 30 | 0.38377 | 0.31026 | 0.35226 |

## ETTm1 H192 — candidates by TEST MSE

| rk | tag | loss | lr | gate | head | p | e | val_mse | TEST mse | TEST mae |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | c13_mae_lr05_rlc | mae | 0.5 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48137 | 0.32527 | 0.35652 |
| 2 | c11_rlc | huber | 1.0 | 0.2 | repeat_last_cycle | 24 | 40 | 0.48811 | 0.32917 | 0.36398 |
| 3 | c07_mae_lr03 | mae | 0.3 | 0.2 | shared | 24 | 60 | 0.48441 | 0.33244 | 0.35877 |
| 4 | c14_p48 | huber | 1.0 | 0.2 | shared | 48 | 30 | 0.48880 | 0.33375 | 0.36739 |
| 5 | c06_mae_lr05 | mae | 0.5 | 0.2 | shared | 24 | 50 | 0.47918 | 0.33443 | 0.35988 |
| 6 | c10_mae_lr05_gate05 | mae | 0.5 | 0.5 | shared | 24 | 40 | 0.48184 | 0.33640 | 0.36148 |
| 7 | c15_p96 | huber | 1.0 | 0.2 | shared | 96 | 30 | 0.49077 | 0.33665 | 0.37032 |
| 8 | c03_lr05 | huber | 0.5 | 0.2 | shared | 24 | 50 | 0.48839 | 0.33706 | 0.36662 |
| 9 | c02_mae | mae | 1.0 | 0.2 | shared | 24 | 30 | 0.48076 | 0.33732 | 0.36150 |
| 10 | c12_lowpass | huber | 1.0 | 0.2 | lowpass | 24 | 40 | 0.48271 | 0.33733 | 0.36677 |
| 11 | c04_lr03 | huber | 0.3 | 0.2 | shared | 24 | 60 | 0.48720 | 0.33737 | 0.36685 |
| 12 | c08_gate005 | huber | 1.0 | 0.05 | shared | 24 | 30 | 0.49024 | 0.34043 | 0.36943 |
| 13 | c01_replica | huber | 1.0 | 0.2 | shared | 24 | 30 | 0.48820 | 0.34075 | 0.36819 |
| 14 | c05_lr20 | huber | 2.0 | 0.2 | shared | 24 | 30 | 0.49107 | 0.34391 | 0.37257 |
| 15 | c09_gate05 | huber | 1.0 | 0.5 | shared | 24 | 30 | 0.48948 | 0.34409 | 0.37257 |

## Artifacts
- trajectory (val+test, 60 rows): `test_selection_all.csv`
- winner list: `test_selection_winner.csv`
- record: this file
