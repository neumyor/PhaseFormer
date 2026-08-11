import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_warp import PhaseWarping
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


def _make_model(dataset, horizon, mode, seed=2021, **hyper_updates):
    hyperparams = build_hyperparams(dataset, horizon, mode)
    hyperparams.update(hyper_updates)
    pl.seed_everything(seed, workers=True)
    args = make_exp_args(dataset, 720, horizon, hyperparams)
    config = PhaseFormerPresetConfig(args, 720, horizon, hyperparams)
    return PhaseFormer(config), hyperparams


class PhaseWarpingTests(unittest.TestCase):
    def test_forward_shape_phase_warp_path(self):
        model, hp = _make_model("ETTm2", 96, "phase_warp", time_mark_dim=5)
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        y_hat, Z, y_phase_steps = model(x_enc, x_mark_enc, None, None)
        self.assertEqual(tuple(y_hat.shape), (2, 96, 7))
        self.assertEqual(tuple(Z.shape), (2, 7, 24, hp["latent_dim"]))
        self.assertEqual(tuple(y_phase_steps.shape), (2, 7, 24, 96 // 24))
        self.assertTrue(torch.isfinite(y_hat).all())

    def test_uniform_speed_is_identity_warp(self):
        warp = PhaseWarping(mark_dim=4)
        warp.eval()
        B, C, P, L = 2, 3, 5, 24
        x_periods = torch.randn(B, C, P, L)
        mark = torch.rand(B, P * L, 4)
        aligned = warp(x_periods, mark)
        expected = x_periods.permute(0, 1, 3, 2)
        # Uniform speed gives phi = L * (l * c) / (L * c) = l exactly in real
        # arithmetic; the normalized cumsum introduces ~1e-6 rounding, so the
        # identity is approximate to machine precision.
        torch.testing.assert_close(aligned, expected, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(warp.last_mean_phase_advance, 0.0, places=6)

    def test_warp_phase_is_monotonic_and_bounded(self):
        warp = PhaseWarping(mark_dim=4)
        B, C, P, L = 2, 3, 4, 24
        speed = torch.rand(B, C, P, L) + 0.1  # strictly positive
        phi = warp._warp_phase(speed)
        # non-decreasing along the phase (time) axis within each cycle
        self.assertTrue(torch.all(phi[..., 1:] >= phi[..., :-1]))
        self.assertTrue(torch.all(phi >= 0.0))
        self.assertTrue(torch.all(phi < L))

    def test_flag_on_init_matches_flag_off(self):
        off, _ = _make_model("ETTm2", 96, "original", time_mark_dim=5)
        on, _ = _make_model("ETTm2", 96, "phase_warp", time_mark_dim=5)
        off_keys = {k for k in off.state_dict() if not k.startswith("phase_warp.")}
        on_keys = {k for k in on.state_dict() if not k.startswith("phase_warp.")}
        self.assertEqual(off_keys, on_keys)
        for key in off_keys:
            torch.testing.assert_close(
                off.state_dict()[key], on.state_dict()[key], atol=0, rtol=0
            )
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        off.eval()
        on.eval()
        with torch.no_grad():
            y_off, _, _ = off(x_enc, x_mark_enc, None, None)
            y_on, _, _ = on(x_enc, x_mark_enc, None, None)
        torch.testing.assert_close(y_on, y_off, atol=1e-5, rtol=1e-5)

    def test_ablation_plumbing_and_mark_dim_fallback(self):
        hp = build_hyperparams("ETTm2", 96, "phase_warp")
        self.assertTrue(hp["use_phase_warp"])
        self.assertEqual(hp["phase_warp_hidden"], 8)
        model, _ = _make_model("ETTm2", 96, "phase_warp", time_mark_dim=5)
        self.assertEqual(model.phase_warp_mark_dim, 5)
        self.assertEqual(model.phase_warp.mark_dim, 5)

    def test_mutually_exclusive_with_phase_align(self):
        hp = build_hyperparams("ETTm2", 96, "original")
        hp["use_phase_align"] = True
        hp["use_phase_warp"] = True
        pl.seed_everything(2021, workers=True)
        args = make_exp_args("ETTm2", 720, 96, hp)
        config = PhaseFormerPresetConfig(args, 720, 96, hp)
        with self.assertRaises(ValueError):
            PhaseFormer(config)


if __name__ == "__main__":
    unittest.main()
