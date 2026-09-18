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

    def test_alternating_level_has_no_memory(self):
        # rho == -1 means the level flips every cycle: no memory at all.
        levels = np.where(np.arange(CYCLES) % 2 == 0, 1.0, -1.0)
        out = window_descriptors(make_batch(np.asarray(levels, dtype=np.float64)))
        self.assertAlmostEqual(float(out["tau_hat_cycles"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(out["tau_hat_steps"][0]), 0.0, places=6)

    def test_ar1_level_recovers_the_known_timescale(self):
        # AR(1) level with phi = 0.5 has theoretical tau = -1/ln(0.5) cycles.
        phi = 0.5
        rng = np.random.default_rng(7)
        n_channels, n_windows = 200, 40
        batch = np.zeros((n_windows, LOOKBACK, n_channels), dtype=np.float64)
        base = np.arange(PERIOD, dtype=np.float64).reshape(PERIOD, 1)
        levels_all = np.zeros((n_windows, CYCLES, n_channels))
        for w in range(n_windows):
            eps = rng.normal(scale=1.0, size=(CYCLES, n_channels))
            level = np.zeros((CYCLES, n_channels))
            for k in range(1, CYCLES):
                level[k] = phi * level[k - 1] + eps[k]
            levels_all[w] = level
            for k in range(CYCLES):
                batch[w, k * PERIOD:(k + 1) * PERIOD, :] = (
                    (base - base.mean()) + level[k]
                )

        out = window_descriptors(torch.as_tensor(batch, dtype=torch.float32))
        expected_cycles = -1.0 / math.log(phi)          # 1.4427
        got = float(np.nanmean(out["tau_hat_cycles"]))
        self.assertAlmostEqual(got, expected_cycles, delta=0.12)
        self.assertAlmostEqual(
            float(np.nanmean(out["tau_hat_steps"])), expected_cycles * PERIOD, delta=3.0
        )
        self.assertLess(float(np.nanmean(out["tau_capped_frac"])), 0.01)

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
