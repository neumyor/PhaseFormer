"""Unit tests for the E19 (minipaper §4.7) train-set level statistics.

These cover the parts that need no dataset and no GPU: the D7 window
descriptors must match the definitions registered in
``docs/PhaseFormer_structural_defect_research_narrative.md`` §A.6, and the new
``tau_hat`` estimator must behave correctly on synthetic series whose level
memory length is known in closed form.
"""

import math
import unittest

import numpy as np
import torch

from scripts.phaseformer_L.e19_predictive_stats import (
    DESCRIPTOR_KEYS,
    LOOKBACK,
    PERIOD,
    TAU_CYCLE_CAP,
    window_descriptors,
)

CYCLES = LOOKBACK // PERIOD  # 30


def make_batch(levels: np.ndarray, channels: int = 4, noise: float = 0.0,
               seed: int = 0) -> torch.Tensor:
    """Build (1, 720, C) where each cycle's mean equals the given level series.

    ``levels`` has shape (CYCLES,). The within-cycle shape is a fixed ramp so
    that ``cycle_amplitude_std`` stays non-degenerate.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(PERIOD, dtype=np.float64).reshape(PERIOD, 1)
    batch = np.zeros((1, LOOKBACK, channels), dtype=np.float64)
    for k in range(CYCLES):
        block = (base - base.mean()) + levels[k]
        batch[0, k * PERIOD:(k + 1) * PERIOD, :] = block
    if noise:
        batch += rng.normal(scale=noise, size=batch.shape)
    return torch.as_tensor(batch, dtype=torch.float32)


def _ar1_tau(phi: float, n_channels: int = 200, n_windows: int = 40,
             seed: int = 7) -> float:
    """Mean ``tau_hat`` (cycles) over windows/channels for an AR(1) level."""
    rng = np.random.default_rng(seed)
    batch = np.zeros((n_windows, LOOKBACK, n_channels), dtype=np.float64)
    base = np.arange(PERIOD, dtype=np.float64).reshape(PERIOD, 1)
    for w in range(n_windows):
        eps = rng.normal(scale=1.0, size=(CYCLES, n_channels))
        level = np.zeros((CYCLES, n_channels))
        for k in range(1, CYCLES):
            level[k] = phi * level[k - 1] + eps[k]
        for k in range(CYCLES):
            batch[w, k * PERIOD:(k + 1) * PERIOD, :] = (base - base.mean()) + level[k]
    out = window_descriptors(torch.as_tensor(batch, dtype=torch.float32))
    return float(np.nanmean(out["tau_hat_cycles"]))


class TestDescriptorContract(unittest.TestCase):
    def test_keys_and_shapes(self):
        out = window_descriptors(make_batch(np.zeros(CYCLES)))
        for key in DESCRIPTOR_KEYS:
            self.assertIn(key, out)
        for key in DESCRIPTOR_KEYS:
            self.assertEqual(np.asarray(out[key]).shape, (1,), msg=key)

    def test_rejects_wrong_lookback(self):
        bad = torch.zeros(1, LOOKBACK - 24, 3)
        with self.assertRaises(ValueError):
            window_descriptors(bad)

    def test_batch_dimension_is_preserved(self):
        batch = torch.cat([make_batch(np.zeros(CYCLES)), make_batch(np.ones(CYCLES))])
        out = window_descriptors(batch)
        self.assertEqual(np.asarray(out["cycle_level_std"]).shape, (2,))


class TestD7Descriptors(unittest.TestCase):
    """The census descriptors must reproduce the registered formulas."""

    def test_cycle_level_std_matches_registered_formula(self):
        levels = np.array([0.5, -0.3, 2.0, 1.1] * 7 + [0.2, 0.9])
        batch = make_batch(levels)
        got = window_descriptors(batch)["cycle_level_std"][0]

        a = batch.numpy()[0]
        cyc = a.reshape(CYCLES, PERIOD, a.shape[-1])
        expected = cyc.mean(1).std(0).mean()
        self.assertAlmostEqual(got, float(expected), places=6)

    def test_last_cycle_shift_matches_registered_formula(self):
        levels = np.arange(CYCLES, dtype=np.float64) * 0.25
        batch = make_batch(levels)
        got = window_descriptors(batch)["last_cycle_shift"][0]

        a = batch.numpy()[0]
        cyc = a.reshape(CYCLES, PERIOD, a.shape[-1])
        means = cyc.mean(1)
        expected = np.abs(means[-1] - means[:-1].mean(0)).mean()
        self.assertAlmostEqual(got, float(expected), places=6)

    def test_zero_level_variation_gives_zero_cycle_level_std(self):
        got = window_descriptors(make_batch(np.full(CYCLES, 3.0)))["cycle_level_std"][0]
        self.assertAlmostEqual(float(got), 0.0, places=6)


class TestTauHatEstimator(unittest.TestCase):
    """tau_hat = P * mean_c (-1 / ln rho_c) with rho the lag-1 level autocorr."""

    def test_linear_ramp_saturates_the_cap(self):
        # A monotone linear level has rho == 1 exactly, so the memory cannot be
        # resolved inside the window and must saturate, not blow up.
        levels = np.arange(CYCLES, dtype=np.float64)
        out = window_descriptors(make_batch(levels))
        self.assertAlmostEqual(float(out["tau_hat_cycles"][0]), TAU_CYCLE_CAP, places=4)
        self.assertAlmostEqual(
            float(out["tau_hat_steps"][0]), TAU_CYCLE_CAP * PERIOD, places=2
        )
        self.assertAlmostEqual(float(out["tau_capped_frac"][0]), 1.0, places=6)

    def test_near_unit_rho_is_also_capped(self):
        # Regression: -1/ln(rho) diverges as rho -> 1 from below, so a nearly
        # (but not exactly) linear level used to return memory far longer than
        # the window itself (observed: 2176 steps against a 720-step window).
        # The cap must therefore be applied to the RESULT, not only to rho >= 1.
        levels = np.arange(CYCLES, dtype=np.float64) + np.linspace(
            0.0, 1e-6, CYCLES
        )
        out = window_descriptors(make_batch(levels))
        cycles = float(out["tau_hat_cycles"][0])
        self.assertLessEqual(cycles, TAU_CYCLE_CAP + 1e-9)
        self.assertLessEqual(
            float(out["tau_hat_steps"][0]), TAU_CYCLE_CAP * PERIOD + 1e-6
        )
        self.assertGreater(float(out["tau_capped_frac"][0]), 0.0)

    def test_tau_hat_never_exceeds_the_window_in_steps(self):
        for phi in (0.5, 0.9, 0.99):
            self.assertLessEqual(_ar1_tau(phi), TAU_CYCLE_CAP + 1e-9)

    def test_alternating_level_has_no_memory(self):
        # rho == -1 means the level flips every cycle: no memory at all.
        levels = np.where(np.arange(CYCLES) % 2 == 0, 1.0, -1.0)
        out = window_descriptors(make_batch(np.asarray(levels, dtype=np.float64)))
        self.assertAlmostEqual(float(out["tau_hat_cycles"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(out["tau_hat_steps"][0]), 0.0, places=6)

    def test_ar1_level_recovers_the_known_timescale(self):
        # AR(1) level with phi = 0.5 has population tau = -1/ln(0.5) = 1.4427
        # cycles. The lag-1 estimator is downward-biased at K = 30 cycles, so
        # the assertion is a bias-aware band and the bias is documented in the
        # module docstring (it is monotone-preserving, so ranking is unaffected).
        phi = 0.5
        out = _ar1_tau(phi)
        expected_cycles = -1.0 / math.log(phi)          # 1.4427
        self.assertGreater(out, 0.8 * expected_cycles)
        self.assertLess(out, 1.1 * expected_cycles)

    def test_ar1_tau_is_monotone_in_the_true_memory(self):
        # The property §3.4.3's threshold and §4.7's Spearman rho actually rely
        # on: a longer true level memory must give a larger tau_hat.
        phis = (0.1, 0.3, 0.5, 0.7, 0.9)
        taus = [_ar1_tau(phi) for phi in phis]
        for earlier, later in zip(taus, taus[1:]):
            self.assertLess(earlier, later, msg=f"tau_hat not monotone: {taus}")
        self.assertGreater(taus[-1] / taus[0], 10.0)

    def test_flat_level_is_reported_as_non_finite_not_crashing(self):
        # A perfectly flat level has zero variance, so rho is undefined. The
        # estimator must not raise; it yields NaN, which the setting-level
        # aggregation surfaces instead of silently inventing a value.
        out = window_descriptors(make_batch(np.zeros(CYCLES)))
        self.assertTrue(np.isnan(float(out["tau_hat_cycles"][0])))


class TestConstants(unittest.TestCase):
    def test_lookback_is_divisible_by_period(self):
        self.assertEqual(LOOKBACK % PERIOD, 0)
        self.assertEqual(CYCLES, 30)


if __name__ == "__main__":
    unittest.main()
