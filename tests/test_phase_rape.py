import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_rape import ReliabilityGate
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


class PhaseRapeTests(unittest.TestCase):
    def test_forward_shape_phase_rape_path(self):
        model, hp = _make_model("ETTm2", 96, "phase_rape", time_mark_dim=5)
        self.assertTrue(hp["use_phase_rape"])
        self.assertFalse(hp.get("use_phase_warp", False))
        self.assertFalse(hp.get("use_phase_amp_calib", False))
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        y_hat, Z, y_phase_steps = model(x_enc, x_mark_enc, None, None)
        self.assertEqual(tuple(y_hat.shape), (2, 96, 7))
        self.assertEqual(tuple(Z.shape), (2, 7, 24, hp["latent_dim"]))
        self.assertEqual(tuple(y_phase_steps.shape), (2, 7, 24, 96 // 24))
        self.assertTrue(torch.isfinite(y_hat).all())

    def test_at_init_matches_original(self):
        # At construction warp and amp calibration are identity, so the fused
        # RAPE phase equals the identity phase for any gate value -> the full
        # forward matches the original (warm start).
        off, _ = _make_model("ETTm2", 96, "original", time_mark_dim=5)
        on, _ = _make_model("ETTm2", 96, "phase_rape", time_mark_dim=5)
        off_keys = {k for k in off.state_dict() if not k.startswith(
            ("phase_warp.", "phase_amp_calib.", "reliability_gate."))}
        on_keys = {k for k in on.state_dict() if not k.startswith(
            ("phase_warp.", "phase_amp_calib.", "reliability_gate."))}
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
        torch.testing.assert_close(y_on, y_off, atol=1e-4, rtol=1e-4)

    def test_gate_is_bounded_and_shared(self):
        gate = ReliabilityGate(hidden=8)
        B, C, L, P = 2, 3, 24, 10
        x_in = torch.randn(B, L, C)
        phase_adapted = torch.randn(B, C, L, P)
        phase_identity = torch.randn(B, C, L, P)
        g = gate(x_in, phase_adapted, phase_identity)
        self.assertEqual(tuple(g.shape), (B, C))
        self.assertTrue(torch.all(g >= 0.0) and torch.all(g <= 1.0))
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue(0.0 <= gate.last_mean_gate <= 1.0)
        # zero-init final layer -> neutral start
        self.assertAlmostEqual(gate.last_mean_gate, 0.5, places=5)

    def test_gate_identity_input_gives_neutral(self):
        # When adapted == identity, the gate's adapt feature is 0, but the gate
        # still lies in (0, 1) and forward fusion is exactly the phase series.
        gate = ReliabilityGate(hidden=8)
        B, C, L, P = 2, 3, 24, 10
        x_in = torch.randn(B, L, C)
        phase = torch.randn(B, C, L, P)
        g = gate(x_in, phase, phase)
        fused = g.unsqueeze(-1).unsqueeze(-1) * phase + (
            1.0 - g
        ).unsqueeze(-1).unsqueeze(-1) * phase
        torch.testing.assert_close(fused, phase, atol=1e-6, rtol=1e-6)

    def test_ablation_plumbing(self):
        hp = build_hyperparams("ETTm2", 96, "phase_rape")
        self.assertTrue(hp["use_phase_rape"])
        self.assertEqual(hp["phase_rape_gate_hidden"], 8)
        self.assertEqual(hp["phase_amp_calib_max_scale"], 2.0)
        model, _ = _make_model("ETTm2", 96, "phase_rape", time_mark_dim=5)
        self.assertIsNotNone(model.phase_warp)
        self.assertIsNotNone(model.phase_amp_calib)
        self.assertIsNotNone(model.reliability_gate)

    def test_mutually_exclusive_with_individual_flags(self):
        for flag in ("use_phase_warp", "use_phase_amp_calib"):
            hp = build_hyperparams("ETTm2", 96, "original")
            hp[flag] = True
            hp["use_phase_rape"] = True
            pl.seed_everything(2021, workers=True)
            args = make_exp_args("ETTm2", 720, 96, hp)
            config = PhaseFormerPresetConfig(args, 720, 96, hp)
            with self.assertRaises(ValueError):
                PhaseFormer(config)


if __name__ == "__main__":
    unittest.main()
