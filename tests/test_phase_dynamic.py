import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_correction import PhaseCorrection
from src.models.phase_geometry import CircularPhaseEmbedding, build_circular_embedding
from src.models.phase_rotation import PhaseRotation
from src.models.harmonic_modulation import HarmonicModulation
from src.models.phase_velocity import PhaseVelocity
from src.models.adaptive_residual_gate import AdaptiveResidualGate
from src.models.multiscale_phase import MultiScalePhase
from src.models.phase_deformation import PhaseDeformation
from src.models.phase_graph import PhaseGraph
from src.models.phase_decoder import TrajectoryDecoder
from src.models.layers.SelfAttention_Family import FullAttention
from src.models.phaseformer_presets import (
    ABLATION_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    get_ablation_overrides,
    make_exp_args,
)
from src.models.phase_adapters import ReliabilityCoupledResidualFusion


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


class PhaseVelocityTests(unittest.TestCase):
    def test_forward_shape(self):
        vel = PhaseVelocity(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = vel(tokens)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_zero_init_is_identity(self):
        vel = PhaseVelocity(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = vel(tokens)
        torch.testing.assert_close(out, tokens, atol=0, rtol=0)
        self.assertEqual(vel.last_mean_velocity, 0.0)
        self.assertEqual(vel.last_mean_delta, 0.0)

    def test_nonzero_velocity_preserves_mass(self):
        # Constant nonzero velocity -> cumulative displacement grows along the
        # slot axis, but the circular scatter conserves total token mass.
        vel = PhaseVelocity(dim=8, hidden=8, velocity_scale=0.05)
        vel.net[-1].weight.data.normal_()
        vel.net[-1].bias.data.fill_(0.5)
        tokens = torch.randn(2, 3, 24, 8)
        out = vel(tokens)
        torch.testing.assert_close(
            out.sum(dim=2), tokens.sum(dim=2), atol=1e-5, rtol=1e-5
        )
        self.assertGreater(vel.last_mean_velocity, 0.0)
        self.assertGreater(vel.last_mean_delta, 0.0)
        # Velocity form couples slots: the last slot's displacement is a
        # cumsum, so it must be >= the first slot's displacement in magnitude
        # only when velocities share a sign; here we just check the hook.
        self.assertGreaterEqual(vel.last_mean_delta, vel.last_mean_velocity)


class AdaptiveResidualGateTests(unittest.TestCase):
    def test_forward_shape_and_warm_start(self):
        gate = AdaptiveResidualGate(phase_dim=8, enc_in=7, gate_init=0.5)
        Z = torch.randn(2, 7, 24, 8)
        x_in = torch.randn(2, 24, 7)
        alpha = gate(Z, x_in)
        self.assertEqual(tuple(alpha.shape), (2, 1, 7))
        # Warm start: alpha = gate_init at initialization (net zero-init).
        torch.testing.assert_close(
            alpha, torch.full_like(alpha, 0.5), atol=1e-4, rtol=1e-3
        )

    def test_gate_init_respects_gate_init_value(self):
        gate = AdaptiveResidualGate(phase_dim=8, enc_in=4, gate_init=0.2)
        Z = torch.randn(2, 4, 24, 8)
        x_in = torch.randn(2, 24, 4)
        alpha = gate(Z, x_in)
        torch.testing.assert_close(
            alpha, torch.full_like(alpha, 0.2), atol=1e-4, rtol=1e-3
        )


class PhaseDeformationTests(unittest.TestCase):
    def test_forward_shape(self):
        deform = PhaseDeformation(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = deform(tokens)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_zero_init_is_identity(self):
        deform = PhaseDeformation(dim=8, hidden=8)
        tokens = torch.randn(2, 7, 24, 8)
        out = deform(tokens)
        torch.testing.assert_close(out, tokens, atol=0, rtol=0)
        self.assertEqual(deform.last_mean_rate, 0.0)
        self.assertEqual(deform.last_mean_stretch, 0.0)
        self.assertEqual(deform.last_mean_delta, 0.0)

    def test_nonzero_rate_preserves_mass_and_bends(self):
        # Nonzero rate + nonzero stretch -> cumulative displacement that is not
        # a constant-rate drift; the circular scatter still conserves mass.
        deform = PhaseDeformation(dim=8, hidden=8, velocity_scale=0.2)
        deform.net_rate[-1].weight.data.normal_()
        deform.net_rate[-1].bias.data.fill_(0.5)
        deform.net_stretch[-1].weight.data.normal_()
        deform.net_stretch[-1].bias.data.fill_(0.3)
        tokens = torch.randn(2, 3, 24, 8)
        out = deform(tokens)
        torch.testing.assert_close(
            out.sum(dim=2), tokens.sum(dim=2), atol=1e-5, rtol=1e-5
        )
        self.assertGreater(deform.last_mean_rate, 0.0)
        self.assertGreater(deform.last_mean_stretch, 0.0)
        self.assertGreater(deform.last_mean_delta, 0.0)


class PhaseGraphTests(unittest.TestCase):
    def test_forward_shape_and_identity(self):
        g = PhaseGraph(dim=8, hidden=16, k=2)
        Z = torch.randn(2, 7, 24, 8)
        out = g(Z)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        # Zero-init message net -> message == 0 -> identity at init.
        torch.testing.assert_close(out, Z, atol=0, rtol=0)
        self.assertEqual(g.last_mean_message, 0.0)

    def test_activated_message_preserves_none_mass_but_changes_output(self):
        # With a nonzero message network the output differs from input.
        g = PhaseGraph(dim=8, hidden=16, k=2)
        g.msg_net[-1].weight.data.normal_()
        g.msg_net[-1].bias.data.fill_(0.1)
        Z = torch.randn(2, 3, 24, 8)
        out = g(Z)
        self.assertFalse(torch.equal(out, Z))
        self.assertGreater(g.last_mean_message, 0.0)

    def test_circular_shift_equivariance(self):
        # Message passing on the ring is translation-equivariant: shifting the
        # input by one slot shifts the output by one slot too.
        g = PhaseGraph(dim=8, hidden=16, k=2)
        g.msg_net[-1].weight.data.normal_()
        g.msg_net[-1].bias.data.fill_(0.1)
        Z = torch.randn(2, 3, 24, 8)
        out = g(Z)
        out_shifted = g(torch.roll(Z, 1, dims=2))
        torch.testing.assert_close(
            out_shifted, torch.roll(out, 1, dims=2), atol=1e-5, rtol=1e-5
        )


class TrajectoryDecoderTests(unittest.TestCase):
    def test_forward_shape(self):
        dec = TrajectoryDecoder(latent_dim=8, p_out=14, hidden=64, order=2)
        z = torch.randn(2, 7, 24, 8)
        y = dec(z)
        self.assertEqual(tuple(y.shape), (2, 7, 24, 14))
        self.assertTrue(torch.isfinite(y).all())

    def test_constant_coefs_give_constant_sequence(self):
        # If every coefficient except the constant is zero, the decoded phase
        # sequence is identical across the P_out period axis (trajectory
        # consistency: a stationary cycle shape).
        dec = TrajectoryDecoder(latent_dim=8, p_out=14, hidden=64, order=2)
        dec.coef_net[-1].weight.data.zero_()
        dec.coef_net[-1].bias.data.zero_()
        z = torch.randn(2, 3, 24, 8)
        y = dec(z)
        torch.testing.assert_close(
            y[..., 1:], y[..., :-1], atol=1e-6, rtol=1e-6
        )


class MultiScalePhaseTests(unittest.TestCase):
    def test_forward_shape_and_zero_gate_identity(self):
        ms = MultiScalePhase(latent_dim=8, period_len=24, num_periods_input=30, coarse=2)
        phase_series = torch.randn(2, 7, 24, 30)
        out = ms(phase_series)
        # zeta = 0 at init -> the long-branch contribution is exactly zero.
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_gate_open_contributes_long_branch(self):
        ms = MultiScalePhase(latent_dim=8, period_len=24, num_periods_input=30, coarse=2)
        ms.zeta.data.fill_(1.0)
        phase_series = torch.randn(2, 7, 24, 30)
        out = ms(phase_series)
        self.assertFalse(torch.equal(out, torch.zeros_like(out)))
        self.assertGreater(ms.last_mean_abs_long, 0.0)

    def test_odd_period_count_padding(self):
        # num_periods_input=29 (not divisible by coarse=2) pads with the last
        # column so the long branch still produces (B, C, L, 15).
        ms = MultiScalePhase(latent_dim=8, period_len=24, num_periods_input=29, coarse=2)
        phase_series = torch.randn(2, 7, 24, 29)
        out = ms(phase_series)
        self.assertEqual(tuple(out.shape), (2, 7, 24, 8))


class CircularAttentionBiasTests(unittest.TestCase):
    def test_fullattention_bias_changes_attention(self):
        attn = FullAttention(False, attention_dropout=0.0, output_attention=True)
        q = torch.randn(1, 8, 1, 8)
        k = torch.randn(1, 24, 1, 8)
        v = torch.randn(1, 24, 1, 8)
        _, A0 = attn(q, k, v, None)
        bias = torch.zeros(1, 1, 8, 24)
        bias[0, 0, :, 12] = 5.0  # penalize slot 12 hard
        _, A1 = attn(q, k, v, None, bias=bias)
        self.assertFalse(torch.equal(A0, A1))
        # Attention weights are (B, H, L, S); the (1, 1, 8, 24) bias broadcasts.
        self.assertEqual(tuple(A1.shape), (1, 1, 8, 24))

    def test_routing_layer_bias_flag_runs(self):
        model, _ = _make_model(
            "ETTm2", 96, "original",
            phase_use_circular_attn_bias=True,
            phase_circular_attn_bias_scale=1.0,
        )
        self.assertTrue(model.phase_use_circular_attn_bias)
        y_hat, Z, _ = _forward_eval(model)
        self.assertEqual(tuple(y_hat.shape), (2, 96, 7))
        self.assertTrue(torch.isfinite(y_hat).all())


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
        "phase_velocity": {
            "use_phase_velocity": True,
            "phase_velocity_hidden": 8,
            "phase_velocity_scale": 0.1,
        },
        "circular_attn_bias": {
            "phase_use_circular_attn_bias": True,
            "phase_circular_attn_bias_scale": 1.0,
        },
        "adaptive_residual_gate": {
            "use_adaptive_residual_gate": True,
            "adaptive_residual_gate_hidden": 8,
            "adaptive_residual_gate_init": 0.5,
        },
        "dyn_stack": {
            "use_phase_correction": True,
            "phase_use_circular_pos": True,
            "use_phase_rotation": True,
            "use_harmonic_modulation": True,
        },
        "multiscale_phase": {
            "use_multiscale_phase": True,
            "phase_multiscale_long_period": 48,
            "phase_multiscale_coarse": 2,
        },
        "phase_deformation": {
            "use_phase_deformation": True,
            "phase_deformation_hidden": 8,
            "phase_deformation_scale": 0.2,
        },
        "phase_graph": {
            "use_phase_graph": True,
            "phase_graph_hidden": 16,
            "phase_graph_k": 2,
        },
        "trajectory_decoder": {
            "use_trajectory_decoder": True,
            "phase_decoder_hidden": 64,
            "phase_decoder_order": 2,
        },
    }
    # State-dict prefixes owned by each mechanism (the circular-geometry and
    # circular-attn-bias buffers are non-persistent and never in state_dict).
    MECHANISM_PREFIXES = {
        "phase_correction": ("phase_correction.",),
        "circular_geometry": (),
        "phase_rotation": ("phase_rotation.",),
        "harmonic_modulation": ("harmonic_modulation.",),
        "phase_velocity": ("phase_velocity.",),
        "circular_attn_bias": (),
        "adaptive_residual_gate": ("adaptive_residual_gate.",),
        "dyn_stack": ("phase_correction.", "phase_rotation.", "harmonic_modulation."),
        "multiscale_phase": ("multiscale_phase.",),
        "phase_deformation": ("phase_deformation.",),
        "phase_graph": ("phase_graph.",),
        "trajectory_decoder": ("trajectory_decoder.",),
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
        for name in [
            "phase_correction",
            "phase_rotation",
            "harmonic_modulation",
            "phase_velocity",
            "circular_attn_bias",
            "adaptive_residual_gate",
            "multiscale_phase",
            "phase_deformation",
            "phase_graph",
        ]:
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

    def test_new_ablation_modes_build_and_forward(self):
        for mode in ["dyn_corr", "dyn_corr_geo", "dyn_corr_geo_rot", "dyn_stack",
                     "residual_full", "no_residual", "dyn_full",
                     "phase_velocity", "phase_vel_geo", "residual_adaptive",
                     "next_full"]:
            self.assertIn(mode, ABLATION_MODES, msg=mode)
            model, hp = _make_model("ETTh1", 336, mode)
            y_hat, Z, y_phase_steps = _forward_eval(model)
            self.assertEqual(tuple(y_hat.shape), (2, 336, 7), msg=mode)
            self.assertTrue(torch.isfinite(y_hat).all(), msg=mode)

    def test_new_pure_phase_modes_build_and_forward(self):
        for mode in ["multiscale_phase", "phase_deformation", "phase_geo",
                     "phase_graph", "predictor_mlp", "trajectory_decoder",
                     "pure_full"]:
            self.assertIn(mode, ABLATION_MODES, msg=mode)
            model, hp = _make_model("ETTh1", 336, mode)
            y_hat, Z, y_phase_steps = _forward_eval(model)
            self.assertEqual(tuple(y_hat.shape), (2, 336, 7), msg=mode)
            self.assertTrue(torch.isfinite(y_hat).all(), msg=mode)

    def test_pure_full_is_phase_only(self):
        model, hp = _make_model("ETTh1", 336, "pure_full")
        self.assertFalse(model.use_residual_head)
        self.assertFalse(model.use_weak_period_residual)
        self.assertTrue(model.use_multiscale_phase)
        self.assertTrue(model.use_phase_deformation)
        self.assertTrue(model.use_phase_graph)
        self.assertTrue(model.use_trajectory_decoder)
        self.assertFalse(
            any(k.startswith("weak_period_residual") for k in model.state_dict())
        )

    def test_trajectory_decoder_replaces_predictor_output(self):
        # The trajectory decoder and the linear predictor share the (B, C, L,
        # P_out) contract; switching the flag changes the decoding path.
        lin, _ = _make_model("ETTh1", 336, "original")
        traj, _ = _make_model("ETTh1", 336, "trajectory_decoder")
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        lin_y, _, lin_yps = _forward_eval(lin, x_enc, x_mark_enc)
        traj_y, _, traj_yps = _forward_eval(traj, x_enc, x_mark_enc)
        self.assertEqual(tuple(lin_yps.shape), tuple(traj_yps.shape))
        # Same seed -> shared params identical; only the predictor path differs.
        self.assertFalse(torch.equal(lin_y, traj_y))
        self.assertGreaterEqual(traj.trajectory_decoder.last_smoothness, 0.0)

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

    def test_next_full_flags_reach_model(self):
        model, _ = _make_model("ETTh1", 336, "next_full")
        self.assertTrue(model.use_phase_velocity)
        self.assertTrue(model.phase_use_circular_attn_bias)
        self.assertTrue(model.use_weak_period_residual)
        self.assertTrue(model.use_phase_local_trend)
        self.assertTrue(model.use_adaptive_residual_gate)
        self.assertTrue(hasattr(model, "phase_velocity"))
        self.assertTrue(hasattr(model, "adaptive_residual_gate"))

    def test_residual_adaptive_matches_residual_full_at_init(self):
        # Warm start: alpha = gate_init = 0.5 (fixed gate of residual_full), so
        # at initialization the adaptive gate reproduces the fixed fusion.
        fixed, _ = _make_model("ETTh1", 336, "residual_full")
        adaptive, _ = _make_model("ETTh1", 336, "residual_adaptive")
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        fy, _, _ = _forward_eval(fixed, x_enc, x_mark_enc)
        ay, _, _ = _forward_eval(adaptive, x_enc, x_mark_enc)
        torch.testing.assert_close(ay, fy, atol=1e-6, rtol=1e-6)

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


class ReliabilityCoupledResidualFusionTests(unittest.TestCase):
    """gold_combo_stability_v1: RCRF reliability, gate and integration."""

    def _strong_phase_series(self, B=2, C=3, L=24, P=30):
        # Perfectly periodic per-slot template: Var_l(mean_k) > 0, Var_k ~ 0.
        template = torch.linspace(-1.0, 1.0, L).view(1, 1, L, 1).expand(B, C, L, 1)
        series = template + 1e-3 * torch.randn(B, C, L, P)
        return series

    def _noisy_phase_series(self, B=2, C=3, L=24, P=30):
        return torch.randn(B, C, L, P)

    def _fused(self, module, phase_series):
        y_phase = torch.randn(2, 96, 3)
        y_residual = torch.randn(2, 96, 3)
        return module(y_phase, y_residual, phase_series)

    def test_reliability_limits(self):
        module = ReliabilityCoupledResidualFusion(sensitivity_init=0.0)
        r_strong = module._reliability(self._strong_phase_series())
        r_noisy = module._reliability(self._noisy_phase_series())
        self.assertTrue((r_strong > 0.9).all().item())
        self.assertTrue((r_noisy < 0.2).all().item())
        r_all = torch.cat([r_strong, r_noisy])
        self.assertTrue((r_all >= 0.0).all().item())
        self.assertTrue((r_all <= 1.0).all().item())

    def test_gate_shape_and_range(self):
        module = ReliabilityCoupledResidualFusion(sensitivity_init=2.0)
        y, alpha = self._fused(module, self._noisy_phase_series())
        self.assertEqual(tuple(y.shape), (2, 96, 3))
        self.assertEqual(tuple(alpha.shape), (2, 3))
        self.assertTrue(torch.isfinite(y).all().item())
        self.assertTrue((alpha > 0.0).all().item())
        self.assertTrue((alpha < 1.0).all().item())

    def test_initial_gate_matches_prior(self):
        # s0=0 -> alpha == prior (0.5) for every sample/channel.
        s0 = ReliabilityCoupledResidualFusion(sensitivity_init=0.0)
        _, alpha = self._fused(s0, self._noisy_phase_series())
        torch.testing.assert_close(alpha, torch.full_like(alpha, 0.5), atol=1e-6, rtol=0)
        self.assertEqual(s0.sensitivity, 0.0)
        # s0=2 -> initial sensitivity exactly 2; low reliability pushes alpha > 0.5.
        s2 = ReliabilityCoupledResidualFusion(sensitivity_init=2.0)
        self.assertAlmostEqual(s2.sensitivity, 2.0, places=6)
        _, alpha2 = self._fused(s2, self._noisy_phase_series())
        self.assertTrue((alpha2 > 0.5).all().item())

    def test_sensitivity_direction(self):
        # Same low-reliability input: higher sensitivity must raise alpha.
        low = self._noisy_phase_series()
        s0 = ReliabilityCoupledResidualFusion(sensitivity_init=0.0)
        s2 = ReliabilityCoupledResidualFusion(sensitivity_init=2.0)
        _, a0 = self._fused(s0, low)
        _, a2 = self._fused(s2, low)
        self.assertTrue((a2 > a0).all().item())
        # High-reliability input: coupling term (1-r) ~ 0, alpha stays at prior.
        strong = self._strong_phase_series()
        _, a_strong = self._fused(s2, strong)
        torch.testing.assert_close(
            a_strong, torch.full_like(a_strong, 0.5), atol=0.02, rtol=0
        )

    def test_forward_backward(self):
        module = ReliabilityCoupledResidualFusion(sensitivity_init=2.0)
        y, alpha = self._fused(module, self._noisy_phase_series())
        loss = y.sum() + alpha.sum()
        loss.backward()
        self.assertIsNotNone(module.gate_bias.grad)
        self.assertIsNotNone(module.s_raw.grad)
        self.assertTrue(torch.isfinite(module.gate_bias.grad).all().item())
        self.assertTrue(torch.isfinite(module.s_raw.grad).all().item())

    def test_mutual_exclusion_with_adaptive_gate(self):
        model, _ = _make_model(
            "ETTh1", 336, "original",
            use_weak_period_residual=True,
            use_rcrf_fusion=True,
            rcrf_alpha_init=0.5,
            rcrf_sensitivity_init=2.0,
            use_adaptive_weak_period_gate=True,
        )
        self.assertTrue(model.use_rcrf_fusion)
        self.assertTrue(hasattr(model, "rcrf_fusion"))
        # The adaptive gate must not be constructed under RCRF.
        self.assertFalse(hasattr(model, "adaptive_weak_period_gate"))
        y_hat, Z, y_phase_steps = _forward_eval(model)
        self.assertTrue(torch.isfinite(y_hat).all().item())

    def test_flag_off_regression(self):
        # Enabling RCRF replaces only the fixed-gate parameter: every other
        # parameter of the weak-residual model is bit-identical at construction.
        off, _ = _make_model(
            "ETTm2", 96, "original",
            use_weak_period_residual=True,
            use_rcrf_fusion=False,
            weak_period_residual_gate_init=0.5,
        )
        on, _ = _make_model(
            "ETTm2", 96, "original",
            use_weak_period_residual=True,
            use_rcrf_fusion=True,
        )
        off_keys = {k for k in off.state_dict() if k != "weak_period_residual_gate"}
        on_keys = {k for k in on.state_dict() if not k.startswith("rcrf_fusion")}
        self.assertEqual(off_keys, on_keys)
        for key in off_keys:
            torch.testing.assert_close(
                off.state_dict()[key], on.state_dict()[key], atol=0, rtol=0, msg=key
            )

    def test_fixed_gate_equals_s0_at_init(self):
        # RCRF with initial sensitivity 0 is exactly the fixed 0.5 gate: the
        # coupling term vanishes and alpha stays at the prior. Same shared params
        # under the same seed -> identical forward at construction time.
        fixed, _ = _make_model("ETTh1", 336, "gold_combo_fixed")
        s0, _ = _make_model("ETTh1", 336, "gold_combo_reliability_s0")
        x_enc = torch.randn(2, 720, 7)
        x_mark_enc = torch.rand(2, 720, 5)
        fy, _, _ = _forward_eval(fixed, x_enc, x_mark_enc)
        sy, _, _ = _forward_eval(s0, x_enc, x_mark_enc)
        torch.testing.assert_close(sy, fy, atol=1e-6, rtol=1e-6)

    def test_gold_combo_presets_reach_ablation_modes(self):
        for mode in ["gold_combo_fixed", "gold_combo_adaptive",
                     "gold_combo_reliability_s0", "gold_combo_reliability_s2"]:
            self.assertIn(mode, ABLATION_MODES, msg=mode)
            hp = build_hyperparams("ETTh2", 720, mode)
            self.assertEqual(hp["scheme_name"], mode)
            self.assertTrue(hp["use_weak_period_residual"])
            self.assertAlmostEqual(hp["weak_period_residual_gate_init"], 0.5)
            self.assertEqual(hp["phase_uncertainty_min"], 0.2)
            self.assertEqual(hp["phase_level_calib_gate_init"], 0.2)
            self.assertEqual(hp["phase_noise_hifreq_strength"], 0.8)
            model, _ = _make_model("ETTh2", 720, mode)
            y_hat, Z, y_phase_steps = _forward_eval(model)
            self.assertEqual(tuple(y_hat.shape), (2, 720, 7), msg=mode)
            self.assertTrue(torch.isfinite(y_hat).all(), msg=mode)

    def test_rcrf_preset_flags_and_sensitivity(self):
        s0 = get_ablation_overrides("gold_combo_reliability_s0")
        s2 = get_ablation_overrides("gold_combo_reliability_s2")
        self.assertTrue(s0["use_rcrf_fusion"])
        self.assertEqual(s0["rcrf_sensitivity_init"], 0.0)
        self.assertEqual(s2["rcrf_sensitivity_init"], 2.0)
        self.assertEqual(s0["rcrf_alpha_init"], 0.5)
        self.assertEqual(s0["rcrf_s_max"], 4.0)
        adaptive = get_ablation_overrides("gold_combo_adaptive")
        self.assertTrue(adaptive["use_adaptive_weak_period_gate"])
        fixed = get_ablation_overrides("gold_combo_fixed")
        self.assertNotIn("use_adaptive_weak_period_gate", fixed)
        self.assertNotIn("use_rcrf_fusion", fixed)

    def test_seed_passing_changes_init(self):
        m1, _ = _make_model("ETTh1", 336, "gold_combo_reliability_s2")
        m1b, _ = _make_model("ETTh1", 336, "gold_combo_reliability_s2")
        m2, _ = _make_model("ETTh1", 336, "gold_combo_reliability_s2", seed=2022)
        for key in m1.state_dict():
            torch.testing.assert_close(
                m1.state_dict()[key], m1b.state_dict()[key], atol=0, rtol=0, msg=key
            )
        differing = sum(
            1 for key in m1.state_dict()
            if not torch.equal(m1.state_dict()[key], m2.state_dict()[key])
        )
        self.assertGreater(differing, 0)

    def test_search_runner_reaches_gold_combo_modes(self):
        import sys
        import scripts.search_phaseformer as sp

        argv = [
            "search_phaseformer.py",
            "--dataset", "ETTh2", "--horizon", "720",
            "--stage", "mechanism_screen_1",
            "--mechanism", "gold_combo_reliability_s2",
            "--max-epochs", "8", "--percent", "30", "--seed", "2021",
            "--loss", "huber", "--num-workers", "0",
        ]
        old = sys.argv
        try:
            sys.argv = argv
            args = sp.parse_args()
            spec = sp.build_spec(args)
        finally:
            sys.argv = old
        self.assertTrue(spec["hyperparams"]["use_rcrf_fusion"])
        self.assertEqual(spec["hyperparams"]["rcrf_sensitivity_init"], 2.0)


if __name__ == "__main__":
    unittest.main()
