import unittest

import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_cycle_fusion import PhaseCycleFusionComposer
from src.models.phaseformer_presets import (
    PCTF_FUSION_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


def _composer(strategy):
    return PhaseCycleFusionComposer(
        120, 48, 24,
        strategy=strategy,
        d_model=8,
        num_heads=1,
        ffn_dim=16,
        masked_origins=2,
    )


class PhaseCycleFusionComposerTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        history = torch.randn(2, 120, 3)
        phase = torch.randn(2, 48, 3)
        phase_series = torch.randn(2, 3, 24, 5)
        return phase, history, phase_series

    def test_every_strategy_has_finite_shape_and_branch_diagnostics(self):
        phase, history, phase_series = self._inputs()
        for strategy in PhaseCycleFusionComposer.STRATEGIES:
            with self.subTest(strategy=strategy):
                model = _composer(strategy)
                output = model(phase, history, phase_series)
                self.assertEqual(tuple(output.shape), (2, 48, 3))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(tuple(model.last_trajectory.shape), (2, 48, 3))
                self.assertEqual(tuple(model.last_cycle.shape), (2, 48, 3))
                self.assertTrue(torch.isfinite(model.last_phase_reliability).all())
                self.assertTrue((model.last_phase_reliability >= 0).all())
                self.assertTrue((model.last_phase_reliability <= 1).all())

    def test_structured_composers_preserve_nlinear_absolute_mean(self):
        phase, history, phase_series = self._inputs()
        structured = (
            "component_scalar",
            "component_cycle",
            "monotonic_evidence",
            "mlp_evidence",
            "phase_modulation",
        )
        for strategy in structured:
            with self.subTest(strategy=strategy):
                model = _composer(strategy)
                output = model(phase, history, phase_series)
                torch.testing.assert_close(
                    output.mean(dim=1),
                    model.last_trajectory.mean(dim=1),
                    atol=2e-6,
                    rtol=1e-6,
                )
                self.assertLess(model.last_shape_cycle_mean_max, 2e-6)
                self.assertLess(model.last_horizon_mean_error_max, 2e-6)

    def test_negative_controls_mix_complete_forecasts(self):
        phase, history, phase_series = self._inputs()
        for strategy in ("uniform_control", "softmax_control"):
            with self.subTest(strategy=strategy):
                model = _composer(strategy)
                output = model(phase, history, phase_series)
                expected = (
                    model.last_phase + model.last_trajectory + model.last_cycle
                ) / 3.0
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(
                    model.last_branch_weights.sum(dim=-1),
                    torch.ones(2),
                )

    def test_monotonic_gate_directions_are_structural(self):
        model = _composer("monotonic_evidence")
        batch, channels = 2, 3
        zeros = torch.zeros(batch, channels)
        ones = torch.ones(batch, channels)
        level_base, shape_reliable = model._monotonic_gates(
            batch, channels, ones, zeros, ones, ones
        )
        level_drift, shape_unreliable = model._monotonic_gates(
            batch, channels, zeros, ones, ones, ones
        )
        level_low_conf, shape_low_conf = model._monotonic_gates(
            batch, channels, ones, zeros, zeros, zeros
        )
        self.assertTrue((shape_unreliable > shape_reliable).all())
        self.assertTrue((shape_reliable > shape_low_conf).all())
        self.assertTrue((level_base > level_drift).all())
        self.assertTrue((level_base > level_low_conf).all())

    def test_phase_modulation_has_bounded_differentiable_diagnostics(self):
        phase, history, phase_series = self._inputs()
        model = _composer("phase_modulation")
        model(phase, history, phase_series)
        self.assertEqual(tuple(model.last_expected_shift.shape), (2, 2, 3))
        self.assertEqual(tuple(model.last_amplitude.shape), (2, 2, 1, 3))
        self.assertTrue((model.last_amplitude >= 0.5).all())
        self.assertTrue((model.last_amplitude <= 2.0).all())

    def test_all_strategies_propagate_to_both_atomic_branches(self):
        phase, history, phase_series = self._inputs()
        for strategy in PhaseCycleFusionComposer.STRATEGIES:
            with self.subTest(strategy=strategy):
                model = _composer(strategy)
                phase_input = phase.clone().requires_grad_(True)
                loss = model(phase_input, history, phase_series).square().mean()
                loss.backward()
                self.assertGreater(float(phase_input.grad.abs().sum()), 0.0)
                self.assertGreater(
                    float(model.trajectory.linear.weight.grad.abs().sum()), 0.0
                )
                self.assertGreater(
                    float(model.cycle.out_proj.weight.grad.abs().sum()), 0.0
                )

    def test_masked_evidence_contexts_are_strictly_causal(self):
        model = _composer("monotonic_evidence")
        history = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 24, 1)
        contexts, targets = model._masked_contexts(history.reshape(1, 120, 1))
        values = contexts.reshape(2, 5, 24, 1)[:, :, 0, 0]
        torch.testing.assert_close(
            values, torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3.0]])
        )
        torch.testing.assert_close(targets[:, 0, 0], torch.tensor([3.0, 4.0]))


class PhaseCycleFusionPresetTests(unittest.TestCase):
    @staticmethod
    def _model(mode, horizon=96):
        hp = build_hyperparams("ETTm2", horizon, mode)
        args = make_exp_args("ETTm2", 720, horizon, hp, batch_size=2)
        return PhaseFormer(PhaseFormerPresetConfig(args, 720, horizon, hp))

    def test_presets_change_only_the_single_checkpoint_composer(self):
        for mode, strategy in PCTF_FUSION_MODES.items():
            with self.subTest(mode=mode):
                model = self._model(mode)
                self.assertTrue(model.use_phase_cycle_fusion)
                self.assertFalse(model.use_weak_period_residual)
                self.assertFalse(model.use_rcrf_fusion)
                self.assertFalse(model.use_triaxis_fusion)
                self.assertFalse(model.use_safe_triaxis)
                self.assertIsInstance(
                    model.phase_cycle_fusion, PhaseCycleFusionComposer
                )
                self.assertEqual(model.phase_cycle_fusion.strategy, strategy)

    def test_all_presets_have_complete_phaseformer_forward(self):
        for mode in PCTF_FUSION_MODES:
            with self.subTest(mode=mode):
                model = self._model(mode)
                output = model(torch.randn(1, 720, 7))[0]
                self.assertEqual(tuple(output.shape), (1, 96, 7))
                self.assertTrue(torch.isfinite(output).all())

    def test_h192_end_to_end_forward(self):
        model = self._model("pctf_fusion_monotonic", horizon=192)
        output = model(torch.randn(1, 720, 7))[0]
        self.assertEqual(tuple(output.shape), (1, 192, 7))
        self.assertTrue(torch.isfinite(output).all())

    def test_shared_phase_initialization_matches_a1(self):
        torch.manual_seed(2021)
        a1 = self._model("gold_combo_reliability_s2")
        for mode in PCTF_FUSION_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(2021)
                candidate = self._model(mode)
                for prefix in ("embedding.", "routing_layers.", "predictor."):
                    left = {
                        key: value for key, value in a1.state_dict().items()
                        if key.startswith(prefix)
                    }
                    right = {
                        key: value for key, value in candidate.state_dict().items()
                        if key.startswith(prefix)
                    }
                    self.assertEqual(set(left), set(right))
                    for key in left:
                        torch.testing.assert_close(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
