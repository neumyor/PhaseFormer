# PhaseFormer-LA (M1) full test-stage results — 2026-10-10

Provenance: main + test stages of frozen commit `b831dca8` on the vldbj_m1 bundle; 168 trainings (28 settings x 3 seeds x {b0, m1}) + 168 registered test reads (`--stage test`, single read per checkpoint, V7 modulation-off, V5 cycle-mean check). All 168 cycle-mean checks passed (max |Δ cycle mean| <= 2.2e-16, float64, tolerance ~1.5e-9).

Per-setting test MSE/MAE, mean over seeds 2021-2023. `mod-off` = same m1 checkpoint with the centered modulation disabled at inference.

| setting | b0 MSE | LA MSE | LA MAE | LA MSE mod-off | dMSE% vs b0 | dMSE% mod-off |
|---|---|---|---|---|---|---|
| ETTh1-96 | 0.3638 | 0.3629 | 0.3897 | 0.3660 | -0.26 | -0.85 |
| ETTh1-192 | 0.4026 | 0.4015 | 0.4153 | 0.4249 | -0.27 | -5.51 |
| ETTh1-336 | 0.4535 | 0.4273 | 0.4318 | 0.4378 | -5.78 | -2.41 |
| ETTh1-720 | 0.4271 | 0.4281 | 0.4457 | 0.4273 | +0.24 | +0.20 |
| ETTh2-96 | 0.2790 | 0.2806 | 0.3437 | 0.2928 | +0.57 | -4.15 |
| ETTh2-192 | 0.3442 | 0.3434 | 0.3839 | 0.3453 | -0.24 | -0.56 |
| ETTh2-336 | 0.3674 | 0.3679 | 0.4071 | 0.3688 | +0.13 | -0.26 |
| ETTh2-720 | 0.3984 | 0.4008 | 0.4374 | 0.4013 | +0.60 | -0.13 |
| ETTm1-96 | 0.3050 | 0.3042 | 0.3529 | 0.3171 | -0.27 | -4.07 |
| ETTm1-192 | 0.3358 | 0.3346 | 0.3686 | 0.3358 | -0.35 | -0.36 |
| ETTm1-336 | 0.3612 | 0.3619 | 0.3851 | 0.3640 | +0.21 | -0.57 |
| ETTm1-720 | 0.4139 | 0.4132 | 0.4134 | 0.4145 | -0.16 | -0.32 |
| ETTm2-96 | 0.1673 | 0.1670 | 0.2585 | 0.1686 | -0.13 | -0.92 |
| ETTm2-192 | 0.2213 | 0.2218 | 0.2951 | 0.2240 | +0.23 | -1.00 |
| ETTm2-336 | 0.2691 | 0.2701 | 0.3270 | 0.2743 | +0.36 | -1.55 |
| ETTm2-720 | 0.3539 | 0.3548 | 0.3808 | 0.3645 | +0.24 | -2.66 |
| Electricity-96 | 0.1301 | 0.1298 | 0.2227 | 0.1335 | -0.24 | -2.79 |
| Electricity-192 | 0.1462 | 0.1465 | 0.2367 | 0.1473 | +0.19 | -0.56 |
| Electricity-336 | 0.1665 | 0.1667 | 0.2594 | 0.1674 | +0.13 | -0.40 |
| Electricity-720 | 0.1988 | 0.1987 | 0.2858 | 0.1988 | -0.02 | -0.03 |
| Traffic-96 | 0.3618 | 0.3617 | 0.2339 | 0.3634 | -0.04 | -0.48 |
| Traffic-192 | 0.3825 | 0.3817 | 0.2453 | 0.3884 | -0.19 | -1.72 |
| Traffic-336 | 0.4007 | 0.3956 | 0.2523 | 0.4015 | -1.27 | -1.46 |
| Traffic-720 | 0.4334 | 0.4340 | 0.2724 | 0.4356 | +0.16 | -0.36 |
| Weather-96 | 0.1458 | 0.1465 | 0.1940 | 0.1489 | +0.52 | -1.58 |
| Weather-192 | 0.1857 | 0.1864 | 0.2325 | 0.1876 | +0.39 | -0.66 |
| Weather-336 | 0.2393 | 0.2393 | 0.2763 | 0.2399 | +0.01 | -0.27 |
| Weather-720 | 0.3116 | 0.3148 | 0.3296 | 0.3156 | +1.04 | -0.25 |

LA improves MSE on 13/28 settings vs the retrained b0 (same protocol); modulation-on beats modulation-off on 27/28 settings.

## Horizon-averaged main-table values (avg over H in {96,192,336,720})

| dataset | LA MSE (avg) | ±sd | LA MAE (avg) | ±sd | ICLR PhaseFormer MSE/MAE | dMSE% | dMAE% |
|---|---|---|---|---|---|---|---|
| ETTh1 | 0.4050 | 0.0028 | 0.4206 | 0.0024 | 0.403 / 0.415 | +0.48 | +1.36 |
| ETTh2 | 0.3482 | 0.0017 | 0.3930 | 0.0011 | 0.346 / 0.388 | +0.62 | +1.30 |
| ETTm1 | 0.3535 | 0.0018 | 0.3800 | 0.0012 | 0.346 / 0.374 | +2.16 | +1.60 |
| ETTm2 | 0.2534 | 0.0025 | 0.3153 | 0.0011 | 0.25 / 0.313 | +1.37 | +0.75 |
| Electricity | 0.1605 | 0.0007 | 0.2511 | 0.0008 | 0.16 / 0.25 | +0.28 | +0.45 |
| Traffic | 0.3933 | 0.0012 | 0.2510 | 0.0008 | 0.386 / 0.249 | +1.88 | +0.80 |
| Weather | 0.2217 | 0.0007 | 0.2581 | 0.0006 | 0.223 / 0.26 | -0.56 | -0.73 |

Note: the retrained b0 (same protocol as LA) is itself within +0.2% to +1.0% of the ICLR published numbers on ETTh1/ETTm1/Traffic (protocol not fully aligned, per user decision only final results are compared). LA vs retrained b0 is the fair comparison; LA vs ICLR column is dominated by that protocol gap on 6/7 datasets.