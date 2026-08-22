import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_correction import PhaseCorrection
from src.models.phase_geometry import CircularPhaseEmbedding, build_circular_embedding
from src.models.phase_rotation import PhaseRotation
from src.models.harmonic_modulation import HarmonicModulation
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


def _forward_eval(model, x_enc=None, x_mark_enc=None):
    if x_enc is None:
        x_enc = torch.randn(2, 720, 7)
    if x_mark_enc is None:
        x_mark_enc = torch.rand(2, 720, 5)
    model.eval()
    with torch.no_grad():
        y_hat, Z, y_phase_steps = model(x_enc, x_mark_enc, None, None)
    return y_hat, Z, y_phase_steps


class PhaseCorrectionTests(unittest.TestCase):
    def test_forward_shape(self):
        corr = PhaseCorrection(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = corr(tokens)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_zero_init_is_identity(self):
        corr = PhaseCorrection(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = corr(tokens)
        torch.testing.assert_close(out, tokens, atol=0, rtol=0)
        self.assertEqual(corr.last_mean_delta, 0.0)

    def test_nonzero_delta_reorders_but_preserves_mass(self):
        corr = PhaseCorrection(dim=8, hidden=8)
        corr.net[-1].weight.data.normal_()
        corr.net[-1].bias.data.fill_(0.5)
        tokens = torch.randn(2, 3, 24, 8)
        out = corr(tokens)
        # Total token mass is preserved by the scatter (circular shift).
        torch.testing.assert_close(
            out.sum(dim=2), tokens.sum(dim=2), atol=1e-5, rtol=1e-5
        )
        self.assertGreater(corr.last_mean_delta, 0.0)


class CircularGeometryTests(unittest.TestCase):
    def test_embedding_shape_and_wrap(self):
        emb = build_circular_embedding(24, 8)
        self.assertEqual(tuple(emb.shape), (24, 8))
        # Slot P-1 is adjacent to slot 0 on the circle: angles differ by 2*pi/24,
        # so the first Fourier pair differs but is not orthogonal-degenerate.
        a0 = torch.atan2(emb[0, 0], emb[0, 1])
        a_last = torch.atan2(emb[-1, 0], emb[-1, 1])
        self.assertAlmostEqual(float(abs(a_last - a0 + 2 * torch.pi / 24) % (2 * torch.pi)), 0.0, places=5)

    def test_module_slices(self):
        m = CircularPhaseEmbedding(24, 8)
        self.assertEqual(tuple(m.forward(24).shape), (1, 24, 8))
        self.assertEqual(tuple(m.forward(10).shape), (1, 10, 8))
        # Non-persistent: not part of state_dict.
        self.assertEqual(len(list(m.state_dict())), 0)


class PhaseRotationTests(unittest.TestCase):
    def test_forward_shape_and_identity(self):
        rot = PhaseRotation(cond_dim=24, hidden=8)
        z = torch.randn(2, 7, 24, 8)
        cond = torch.randn(2, 7, 24, 24)
        out = rot(z, cond)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        torch.testing.assert_close(out, z, atol=0, rtol=0)
        self.assertEqual(rot.last_mean_theta, 0.0)

    def test_rotation_preserves_norm(self):
        rot = PhaseRotation(cond_dim=24, hidden=8)
        rot.net[-1].weight.data.fill_(3.0)  # saturate theta -> near pi/2
        z = torch.randn(2, 3, 24, 8)
        cond = torch.rand(2, 3, 24, 24)
        out = rot(z, cond)
        torch.testing.assert_close(
            out.norm(dim=-1), z.norm(dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_odd_dim_supported(self):
        rot = PhaseRotation(cond_dim=24, hidden=8)
        z = torch.randn(2, 3, 24, 9)
        cond = torch.rand(2, 3, 24, 24)
        out = rot(z, cond)
        self.assertEqual(tuple(out.shape), (2, 3, 24, 9))


class HarmonicModulationTests(unittest.TestCase):
    def test_forward_shape_and_identity(self):
        hm = HarmonicModulation(cond_dim=24, hidden=8)
        z = torch.randn(2, 7, 24, 8)
        cond = torch.randn(2, 7, 24, 24)
        out = hm(z, cond)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        torch.testing.assert_close(out, z, atol=0, rtol=0)
        self.assertEqual(hm.last_mean_abs_gamma, 0.0)
        self.assertEqual(hm.last_mean_abs_beta, 0.0)

    def test_scale_saturates(self):
        hm = HarmonicModulation(cond_dim=24, hidden=8, max_scale=2.0)
        hm.net[-1].weight.data.fill_(10.0)
        cond = torch.rand(2, 3, 24, 24)
        # Saturated scale: gamma stays within [1-max_scale, 1+max_scale].
        gamma = 1.0 + hm.max_scale * torch.tanh(hm.net(cond)[..., 0])
        self.assertTrue(((gamma - 1.0).abs() <= hm.max_scale + 1e-6).all())


class PhaseFormerDynamicMechanismTests(unittest.TestCase):
    MECHANISM_KWARGS = {
        "phase_correction": {"use_phase_correction": True},
        "circular_geometry": {"phase_use_circular_pos": True},
        "phase_rotation": {
            "use_phase_rotation": True,
            "phase_rotation_hidden": 8,
        },
        "harmonic_modulation": {
            "use_harmonic_modulation": True,
            "harmonic_modulation_hidden": 8,
        },
        "dyn_stack": {
            "use_phase_correction": True,
            "phase_use_circular_pos": True,
            "use_phase_rotation": True,
            "use_harmonic_modulation": True,
        },
    }
    # State-dict prefixes owned by each mechanism (the circular-geometry buffer
    # is non-persistent and never appears in state_dict).
    MECHANISM_PREFIXES = {
        "phase_correction": ("phase_correction.",),
        "circular_geometry": (),
        "phase_rotation": ("phase_rotation.",),
        "harmonic_modulation": ("harmonic_modulation.",),
        "dyn_stack": ("phase_correction.", "phase_rotation.", "harmonic_modulation."),
    }

    def test_flags_reach_model(self):
        model, _ = _make_model("ETTm2", 96, "original", **self.MECHANISM_KWARGS["dyn_stack"])
        self.assertTrue(model.use_phase_correction)
        self.assertTrue(model.phase_use_circular_pos)
        self.assertTrue(model.use_phase_rotation)
        self.assertTrue(model.use_harmonic_modulation)

    def test_flag_on_init_matches_flag_off_shared_params(self):
        off, _ = _make_model("ETTm2", 96, "original")
        for name, kwargs in self.MECHANISM_KWARGS.items():
            on, _ = _make_model("ETTm2", 96, "original", **kwargs)
            prefixes = self.MECHANISM_PREFIXES[name]
            off_keys = {k for k in off.state_dict() if not k.startswith(prefixes)}
            on_keys = {k for k in on.state_dict() if not k.startswith(prefixes)}
            self.assertEqual(off_keys, on_keys, msg=name)
            for key in off_keys:
                torch.testing.assert_close(
                    off.state_dict()[key], on.state_dict()[key], atol=0, rtol=0,
                    msg=f"{name}: {key}",
                )

    def test_identity_mechanisms_forward_equal_flag_off(self):
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        off_y, _, _ = _forward_eval(
            _make_model("ETTm2", 96, "original")[0], x_enc, x_mark_enc
        )
        for name in ["phase_correction", "phase_rotation", "harmonic_modulation"]:
            on, _ = _make_model("ETTm2", 96, "original", **self.MECHANISM_KWARGS[name])
            on_y, _, _ = _forward_eval(on, x_enc, x_mark_enc)
            torch.testing.assert_close(on_y, off_y, atol=1e-6, rtol=1e-6, msg=name)

    def test_residual_head_off_removes_residual(self):
        model, _ = _make_model("ETTm2", 96, "original", use_residual_head=False)
        self.assertFalse(model.use_weak_period_residual)
        self.assertFalse(model.use_phase_local_trend)
        self.assertFalse(hasattr(model, "weak_period_residual"))
        y_hat, Z, y_phase_steps = _forward_eval(model)
        self.assertEqual(tuple(y_hat.shape), (2, 96, 7))

    def test_residual_head_off_matches_original_params(self):
        # use_residual_head=False on the phase-only original is a no-op for the
        # phase path: identical state dict and identical forward.
        off, _ = _make_model("ETTm2", 96, "original", use_residual_head=False)
        orig, _ = _make_model("ETTm2", 96, "original")
        self.assertEqual(set(off.state_dict()), set(orig.state_dict()))
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        off_y, _, _ = _forward_eval(off, x_enc, x_mark_enc)
        orig_y, _, _ = _forward_eval(orig, x_enc, x_mark_enc)
        torch.testing.assert_close(off_y, orig_y, atol=0, rtol=0)

    def test_residual_full_enables_both_heads(self):
        full, _ = _make_model(
            "ETTm2", 96, "original",
            use_weak_period_residual=True,
            use_phase_local_trend=True,
        )
        self.assertTrue(full.use_weak_period_residual)
        self.assertTrue(full.use_phase_local_trend)
        self.assertTrue(hasattr(full, "weak_period_residual"))
        self.assertTrue(hasattr(full, "phase_local_trend"))
        # Both heads must be absent when use_residual_head is False.
        no_res, _ = _make_model("ETTm2", 96, "original", use_residual_head=False)
        residual_prefixes = ("weak_period_residual", "phase_local_trend")
        self.assertFalse(
            any(k.startswith(residual_prefixes) for k in no_res.state_dict())
        )


if __name__ == "__main__":
    unittest.main()
