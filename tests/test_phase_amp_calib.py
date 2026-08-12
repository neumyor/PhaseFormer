import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_amp_calib import PhaseAmpCalibration
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


class PhaseAmpCalibrationTests(unittest.TestCase):
    def test_forward_shape_phase_amp_calib_path(self):
        model, hp = _make_model("ETTm2", 96, "phase_amp_calib", time_mark_dim=5)
        self.assertTrue(hp["use_phase_warp"])
        self.assertTrue(hp["use_phase_amp_calib"])
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        y_hat, Z, y_phase_steps = model(x_enc, x_mark_enc, None, None)
        self.assertEqual(tuple(y_hat.shape), (2, 96, 7))
        self.assertEqual(tuple(Z.shape), (2, 7, 24, hp["latent_dim"]))
        self.assertEqual(tuple(y_phase_steps.shape), (2, 7, 24, 96 // 24))
        self.assertTrue(torch.isfinite(y_hat).all())

    def test_identity_warm_start(self):
        calib = PhaseAmpCalibration(hidden=8)
        calib.eval()
        B, C, L, P = 2, 3, 24, 10
        phase_series = torch.randn(B, C, L, P)
        with torch.no_grad():
            out = calib(phase_series)
        # Zero-init final layer -> alpha=1, beta=0 -> identity.
        torch.testing.assert_close(out, phase_series, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(calib.last_mean_abs_log_alpha, 0.0, places=6)
        self.assertAlmostEqual(calib.last_mean_abs_beta, 0.0, places=6)

    def test_per_slot_scaling_broadcasts_over_periods(self):
        calib = PhaseAmpCalibration(hidden=8, max_scale=2.0)
        calib.eval()
        # Force every slot to the same scale_logit and shift by zeroing the
        # weight and fixing the bias: raw = b for all features.
        with torch.no_grad():
            calib.net[-1].weight.zero_()
            calib.net[-1].bias.copy_(torch.tensor([0.5, 0.2]))
        B, C, L, P = 2, 3, 24, 10
        phase_series = torch.randn(B, C, L, P)
        with torch.no_grad():
            out = calib(phase_series)
        alpha = 1.0 + 2.0 * torch.tanh(torch.tensor(0.5))
        expected = alpha * phase_series + 0.2
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_flag_on_init_matches_flag_off(self):
        # Same warp, amp_calib toggled. At construction amp_calib is identity,
        # so the two forward passes coincide and all shared state is identical.
        off, _ = _make_model(
            "ETTm2", 96, "original", time_mark_dim=5,
            use_phase_warp=True, phase_warp_hidden=8, use_phase_amp_calib=False,
        )
        on, _ = _make_model(
            "ETTm2", 96, "original", time_mark_dim=5,
            use_phase_warp=True, phase_warp_hidden=8, use_phase_amp_calib=True,
        )
        off_keys = {k for k in off.state_dict() if not k.startswith("phase_amp_calib.")}
        on_keys = {k for k in on.state_dict() if not k.startswith("phase_amp_calib.")}
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

    def test_ablation_plumbing(self):
        hp = build_hyperparams("ETTm2", 96, "phase_amp_calib")
        self.assertTrue(hp["use_phase_warp"])
        self.assertTrue(hp["use_phase_amp_calib"])
        self.assertEqual(hp["phase_amp_calib_hidden"], 8)
        self.assertEqual(hp["phase_amp_calib_max_scale"], 2.0)
        model, _ = _make_model("ETTm2", 96, "phase_amp_calib", time_mark_dim=5)
        self.assertIsNotNone(model.phase_amp_calib)
        self.assertIsNotNone(model.phase_warp)


if __name__ == "__main__":
    unittest.main()
