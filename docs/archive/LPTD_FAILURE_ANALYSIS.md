# LPTD Failure Analysis

## Decision

Latent Phase Transport Decoder (LPTD) was rejected and removed from active code.
It eliminated CPTD's catastrophic expression collapse, but did not deliver a
stable improvement over original PhaseFormer and consistently trailed the
existing latest policies and fixed gold standard.

## Reported results

All result cells are `MSE / MAE`. Relative changes are positive when LPTD is
better.

| Task | LPTD | vs original | vs latest | vs gold |
|---|---:|---:|---:|---:|
| ETTh1-96 | 0.3580 / 0.3845 | +0.8% / +0.4% | -0.4% / +0.8% | +0.3% / -0.7% |
| ETTh1-192 | 0.4099 / 0.4126 | -1.4% / -0.8% | -2.5% / -0.2% | -3.2% / -2.1% |
| ETTh2-96 | 0.2763 / 0.3413 | +1.6% / +0.5% | -1.2% / -2.7% | -0.5% / -1.0% |
| ETTh2-192 | 0.3450 / 0.3831 | -0.3% / +0.1% | -2.1% / -2.4% | -1.2% / -1.9% |
| ETTm1-96 | 0.2980 / 0.3453 | +0.2% / +1.0% | -1.8% / -1.6% | -1.7% / -0.4% |
| ETTm1-192 | 0.3274 / 0.3627 | +1.7% / +0.7% | -0.3% / -1.3% | -1.4% / -0.5% |
| ETTm2-96 | 0.1736 / 0.2649 | -2.2% / -2.0% | -8.0% / -6.2% | -6.5% / -3.5% |
| ETTm2-192 | 0.2268 / 0.2959 | +0.7% / +0.9% | -3.4% / -2.9% | -3.6% / -1.0% |
| Weather-96 | 0.1512 / 0.1972 | -1.1% / -0.1% | -2.6% / -4.5% | -2.2% / -1.1% |
| Weather-192 | 0.1960 / 0.2413 | -1.3% / -1.6% | -1.3% / -1.6% | -1.6% / -1.8% |

Average relative change was approximately -0.13% MSE / -0.09% MAE against the
matched original, -2.36% / -2.26% against latest, and -2.16% / -1.40% against
gold.

## Minimal diagnostic result

The learned average identity weight fell from an initial value near 0.987 to
0.349--0.710, so LPTD did learn non-identity mixing. Shift entropy was
0.787--1.075, compared with the three-way maximum `log(3) = 1.099`; mixing was
especially diffuse on ETTm2 and Weather. All seven ETT variables and most
Weather variables had aggregate dominant shift `+1`.

On the complete ETTh1-96 validation control, `max_shift=0` achieved MSE 0.7408
versus 0.7534 for `max_shift=1`. Thus identity initialization was not the main
blocker: learned transport itself failed to improve this controlled task.

## Interpretation

LPTD learned a broadly shared, high-entropy neighbor mixture with a `+1` bias,
closer to directional smoothing than confident local phase alignment. Greater
identity weight was weakly associated with better results, while greater
entropy was weakly associated with worse results across the ten tasks. The
sample is too small for a statistical claim, but its direction agrees with the
controlled validation result.

Weak-period errors include amplitude, local-level, and irregular innovations
that conservative latent transport cannot generate. These findings motivated
Phase-Anchored Forecasting: retain the unrestricted original predictor, change
only the phase coordinate origin, and introduce no learned auxiliary mechanism.
