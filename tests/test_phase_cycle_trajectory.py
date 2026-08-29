import unittest

import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_cycle_trajectory import PhaseCycleTrajectoryResidualHead
from src.models.phaseformer_presets import (
    PCTF_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


class PhaseCycleTrajectoryHeadTests(unittest.TestCase):
    def test_difference_is_split_into_identifiable_orthogonal_components(self):
        head = PhaseCycleTrajectoryResidualHead(720, 96, 24)
        difference = torch.randn(2, 96, 3)
        shape, level = head.decompose_cycle_difference(difference)

        shape_cycles = shape.view(2, 4, 24, 3)
        level_cycles = level.view(2, 4, 24, 3)
        self.assertLess(float(shape_cycles.mean(dim=2).abs().max()), 1e-6)
        self.assertLess(float(level.mean(dim=1).abs().max()), 1e-6)
        self.assertLess(
            float((shape_cycles * level_cycles).sum(dim=2).abs().max()), 1e-5
        )
        torch.testing.assert_close(
            shape + level,
            difference - difference.mean(dim=1, keepdim=True),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_masked_contexts_are_strictly_causal(self):
        head = PhaseCycleTrajectoryResidualHead(
            120, 48, 24, masked_origins=2
        )
        history = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 24, 1)
        contexts, targets = head._masked_contexts(history.reshape(1, 120, 1))
        context_cycles = contexts.view(2, 5, 24, 1)[:, :, 0, 0]
        torch.testing.assert_close(
            context_cycles,
            torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3.0]]),
        )
        torch.testing.assert_close(targets[:, 0, 0], torch.tensor([3.0, 4.0]))

    def test_fixed_and_masked_confidences_are_finite_and_bounded(self):
        x = torch.randn(2, 120, 3)
        for confidence_mode in ("fixed", "masked_absolute", "masked_regret"):
            with self.subTest(confidence_mode=confidence_mode):
                head = PhaseCycleTrajectoryResidualHead(
                    120, 48, 24, confidence_mode=confidence_mode,
                    masked_origins=2,
                )
                output = head(x)
                self.assertEqual(tuple(output.shape), (2, 48, 3))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((head.last_shape_confidence >= 0.05).all())
                self.assertTrue((head.last_shape_confidence <= 1.0).all())
                self.assertTrue((head.last_level_confidence >= 0.05).all())
                self.assertTrue((head.last_level_confidence <= 1.0).all())
                self.assertEqual(tuple(head.last_shape_gate.shape), (2, 3, 2))
                self.assertEqual(tuple(head.last_level_gate.shape), (2, 3, 2))
                self.assertLess(head.last_shape_cycle_mean_max, 1e-6)
                self.assertLess(head.last_level_horizon_mean_max, 1e-6)
                torch.testing.assert_close(
                    (output - head.last_trajectory).mean(dim=1),
                    torch.zeros_like(output[:, 0]),
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_all_trainable_components_receive_gradients(self):
        head = PhaseCycleTrajectoryResidualHead(
            120, 48, 24, confidence_mode="masked_regret", masked_origins=2
        )
        # ICPT deliberately starts as RepeatLastCycle, so its relative-cycle
        # level correction is exactly zero on step zero.  A non-degenerate
        # learned state must make every component trainable thereafter.
        with torch.no_grad():
            head.cycle.out_proj.weight.normal_(std=0.01)
        loss = head(torch.randn(2, 120, 3)).square().mean()
        loss.backward()
        self.assertGreater(
            float(head.trajectory.linear.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(head.cycle.out_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.shape_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.level_logits.grad.abs().sum()), 0.0)

    def test_all_required_horizons_are_supported(self):
        for horizon in (96, 192, 336, 720):
            with self.subTest(horizon=horizon):
                head = PhaseCycleTrajectoryResidualHead(
                    720, horizon, 24, d_model=8, num_heads=1, ffn_dim=16
                )
                output = head(torch.randn(1, 720, 2))
                self.assertEqual(tuple(output.shape), (1, horizon, 2))
                self.assertTrue(torch.isfinite(output).all())


class PCTFPresetTests(unittest.TestCase):
    @staticmethod
    def _model(mode):
        hp = build_hyperparams("ETTm2", 96, mode)
        args = make_exp_args("ETTm2", 720, 96, hp, batch_size=2)
        return PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp))

    def test_presets_are_one_phaseformer_with_structured_residual(self):
        for mode, expected in PCTF_MODES.items():
            with self.subTest(mode=mode):
                model = self._model(mode)
                head = model.weak_period_residual
                self.assertIsInstance(head, PhaseCycleTrajectoryResidualHead)
                self.assertTrue(model.use_rcrf_fusion)
                self.assertFalse(model.use_triaxis_fusion)
                self.assertFalse(model.use_safe_triaxis)
                self.assertEqual(head.use_shape_correction, expected["shape"])
                self.assertEqual(head.use_level_correction, expected["level"])
                self.assertEqual(head.confidence_mode, expected["confidence"])

    def test_end_to_end_forward_uses_outer_rcrf(self):
        model = self._model("pctf_dual_masked")
        output = model(torch.randn(2, 720, 7))[0]
        self.assertEqual(tuple(output.shape), (2, 96, 7))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(model.rcrf_fusion.last_alpha)

    def test_shared_phase_initialization_matches_a1(self):
        torch.manual_seed(2021)
        a1 = self._model("gold_combo_reliability_s2")
        torch.manual_seed(2021)
        candidate = self._model("pctf_dual_masked")
        for prefix in ("embedding.", "routing_layers.", "predictor."):
            left = {k: v for k, v in a1.state_dict().items() if k.startswith(prefix)}
            right = {
                k: v for k, v in candidate.state_dict().items()
                if k.startswith(prefix)
            }
            self.assertEqual(set(left), set(right))
            for key in left:
                torch.testing.assert_close(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
