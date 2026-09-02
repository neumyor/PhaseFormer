import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.input_component_ablation import (
    InputComponentConfig,
    InputComponentDataset,
    InputComponentTransformer,
    _fractional_circular_shift,
)
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


class FourTupleDataset(Dataset):
    def __init__(self, x):
        self.x = x
        self.timestamps = np.arange(100)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = np.full((8, self.x.shape[1]), index + 7.0, dtype=np.float32)
        x_mark = np.arange(len(self.x), dtype=np.float32)[:, None]
        y_mark = np.arange(len(target), dtype=np.float32)[:, None]
        return self.x.copy(), target, x_mark, y_mark


def synthetic_history(cycles=8, period=24, channels=2):
    phase = np.arange(period)
    template = np.stack(
        [np.sin(2 * np.pi * phase / period), np.cos(4 * np.pi * phase / period)],
        axis=1,
    )[:, :channels]
    values = []
    for cycle in range(cycles):
        values.append(template + 0.08 * cycle * np.linspace(-1, 1, period)[:, None])
    return np.concatenate(values).astype(np.float32)


class InputComponentTests(unittest.TestCase):
    def transform(self, hypothesis, variant, x, index=3):
        config = InputComponentConfig(
            hypothesis=hypothesis, variant=variant, period_len=24,
            intervention_seed=91,
        )
        return InputComponentTransformer(config).transform(
            x, sample_index=index, namespace="synthetic|test", return_metadata=True
        )

    def test_full_is_exact_identity_for_every_hypothesis(self):
        x = synthetic_history()
        for hypothesis in ("h1", "h3", "h4"):
            actual, _ = self.transform(hypothesis, "full", x)
            np.testing.assert_array_equal(actual, x)

    def test_h1_half_and_minus_form_an_exact_additive_ladder(self):
        x = synthetic_history()
        half, _ = self.transform("h1", "half_A", x)
        minus, metadata = self.transform("h1", "minus_A", x)
        np.testing.assert_allclose(half, 0.5 * (x + minus), atol=2e-7)
        np.testing.assert_allclose(x, metadata["base"] + metadata["component"], atol=1e-12)
        np.testing.assert_allclose(minus[-1], x[-1], atol=1e-7)

    def test_h1_sham_is_stable_but_sample_specific_and_endpoint_preserving(self):
        x = synthetic_history()
        first, first_meta = self.transform("h1", "sham", x, index=3)
        repeat, repeat_meta = self.transform("h1", "sham", x, index=3)
        other, other_meta = self.transform("h1", "sham", x, index=4)
        np.testing.assert_array_equal(first, repeat)
        np.testing.assert_array_equal(first_meta["permutation"], repeat_meta["permutation"])
        self.assertFalse(np.array_equal(first_meta["permutation"], other_meta["permutation"]))
        self.assertFalse(np.array_equal(first, other))
        np.testing.assert_allclose(first[-1], x[-1], atol=1e-6)

    def test_h3_variants_preserve_latest_observation(self):
        x = synthetic_history()
        half, _ = self.transform("h3", "half_A", x)
        minus, metadata = self.transform("h3", "minus_A", x)
        sham, _ = self.transform("h3", "sham", x)
        np.testing.assert_allclose(half, 0.5 * (x + minus), atol=2e-7)
        np.testing.assert_allclose(metadata["component"][-1], 0.0, atol=1e-12)
        np.testing.assert_allclose(half[-1], x[-1], atol=1e-7)
        np.testing.assert_allclose(minus[-1], x[-1], atol=1e-7)
        np.testing.assert_allclose(sham[-1], x[-1], atol=1e-6)

    def test_fractional_shift_preserves_real_energy(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(24, 3))
        shifted = _fractional_circular_shift(x, 2.35)
        self.assertTrue(np.isrealobj(shifted))
        np.testing.assert_allclose(np.sum(x * x, axis=0), np.sum(shifted * shifted, axis=0), rtol=1e-12, atol=1e-12)

    def test_h4_recovers_known_phase_drift_and_keeps_last_cycle(self):
        period = 24
        phase = np.arange(period)
        template = (
            np.sin(2 * np.pi * phase / period)
            + 0.35 * np.cos(4 * np.pi * phase / period + 0.3)
        )[:, None]
        physical_shifts = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        cycles = np.stack(
            [_fractional_circular_shift(template, shift) for shift in physical_shifts]
        )
        x = cycles.reshape(-1, 1).astype(np.float32)
        minus, metadata = self.transform("h4", "minus_A", x)
        self.assertTrue(metadata["identifiable"][0])
        estimated = metadata["phase_shift"][:, 0]
        np.testing.assert_allclose(estimated, physical_shifts, atol=0.35)
        np.testing.assert_allclose(
            minus[-period:], x[-period:], atol=2e-6,
        )
        aligned_cycles = minus.reshape(cycles.shape)
        self.assertLess(
            float(np.mean(np.var(aligned_cycles, axis=0))),
            0.05 * float(np.mean(np.var(cycles, axis=0))),
        )
        before_energy = np.sum(cycles * cycles, axis=1)
        after_energy = np.sum(minus.reshape(cycles.shape) ** 2, axis=1)
        np.testing.assert_allclose(before_energy, after_energy, rtol=2e-6, atol=2e-6)

    def test_h4_unidentifiable_constant_channel_is_unchanged(self):
        x = synthetic_history()
        x[:, 1] = 2.0
        minus, metadata = self.transform("h4", "minus_A", x)
        self.assertFalse(metadata["identifiable"][1])
        np.testing.assert_array_equal(minus[:, 1], x[:, 1])

    def test_dataset_wrapper_cannot_modify_targets_or_marks(self):
        x = synthetic_history()
        base = FourTupleDataset(x)
        wrapped = InputComponentDataset(
            base,
            InputComponentConfig(hypothesis="h1", variant="minus_A", period_len=24),
            "unit|train",
        )
        changed = wrapped[1]
        original = base[1]
        self.assertFalse(np.array_equal(changed[0], original[0]))
        for actual, expected in zip(changed[1:], original[1:]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(wrapped.timestamps, base.timestamps)

    def test_plain_rcrf_exports_reconstructable_branch_diagnostics(self):
        hp = build_hyperparams("ETTm2", 96, "rcrf_nlinear_plain")
        # Compact dimensions keep this diagnostic test inexpensive.
        hp.update(latent_dim=4, phase_encoder_hidden=4, predictor_hidden=4, layers=1, phase_attn_heads=1)
        args = make_exp_args("ETTm2", 720, 96, hp, batch_size=2)
        model = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp)).eval()
        x = torch.randn(2, 720, 7)
        with torch.inference_mode():
            output, _, _ = model(x)
        alpha = model.last_rcrf_alpha[:, None, :]
        reconstructed = (
            (1.0 - alpha) * model.last_phase_forecast
            + alpha * model.last_residual_forecast
        )
        torch.testing.assert_close(output, reconstructed, rtol=2e-5, atol=2e-5)
        self.assertEqual(tuple(model.last_rcrf_reliability.shape), (2, 7))


if __name__ == "__main__":
    unittest.main()
