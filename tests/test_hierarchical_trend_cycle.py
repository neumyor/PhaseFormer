import unittest

import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.hierarchical_trend_cycle import HierarchicalTrendCycleResidualHead
from src.models.phaseformer_presets import (
    HPTC_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


class HierarchicalTrendCycleHeadTests(unittest.TestCase):
    def test_shape_and_zero_mean_correction(self):
        head = HierarchicalTrendCycleResidualHead(720, 96, 24, beta_init=0.25)
        output = head(torch.randn(2, 720, 7))
        self.assertEqual(tuple(output.shape), (2, 96, 7))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(head.last_correction_cycle_mean_max, 1e-6)
        self.assertEqual(tuple(head.last_beta.shape), (2, 7, 4))

    def test_fixed_and_rolling_confidence_are_distinct(self):
        x = torch.randn(2, 720, 3)
        fixed = HierarchicalTrendCycleResidualHead(
            720, 96, 24, beta_init=0.25, use_rolling_confidence=False
        )
        rolling = HierarchicalTrendCycleResidualHead(
            720, 96, 24, beta_init=0.25, use_rolling_confidence=True
        )
        fixed(x)
        rolling(x)
        torch.testing.assert_close(
            fixed.last_beta, torch.full_like(fixed.last_beta, 0.25)
        )
        self.assertGreater(float(rolling.last_beta.std()), 0.0)
        self.assertTrue((rolling.last_beta > 0).all())
        self.assertTrue((rolling.last_beta < 0.25).all())

    def test_both_components_receive_gradients(self):
        head = HierarchicalTrendCycleResidualHead(720, 96, 24, beta_init=0.25)
        loss = head(torch.randn(2, 720, 3)).square().mean()
        loss.backward()
        self.assertGreater(float(head.trajectory.linear.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.cycle_shape.out_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.beta_logit.grad.abs()), 0.0)

    def test_supported_horizons_include_long_unbacktestable_leads(self):
        for horizon in (96, 192, 336, 720):
            with self.subTest(horizon=horizon):
                head = HierarchicalTrendCycleResidualHead(720, horizon, 24)
                output = head(torch.randn(1, 720, 2))
                self.assertEqual(tuple(output.shape), (1, horizon, 2))
                self.assertTrue(torch.isfinite(output).all())


class HPTCPresetTests(unittest.TestCase):
    def _model(self, mode):
        hp = build_hyperparams("ETTm2", 96, mode)
        args = make_exp_args("ETTm2", 720, 96, hp, batch_size=2)
        return PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp))

    def test_all_presets_are_single_checkpoint_models(self):
        for mode, expected in HPTC_MODES.items():
            with self.subTest(mode=mode):
                model = self._model(mode)
                self.assertIsInstance(
                    model.weak_period_residual, HierarchicalTrendCycleResidualHead
                )
                self.assertFalse(model.use_triaxis_fusion)
                self.assertFalse(model.use_safe_triaxis)
                self.assertTrue(model.use_rcrf_fusion)
                self.assertEqual(
                    model.weak_period_residual.use_rolling_confidence,
                    expected["rolling"],
                )
                output = model(torch.randn(2, 720, 7))[0]
                self.assertEqual(tuple(output.shape), (2, 96, 7))

    def test_shared_phase_initialization_matches_a1(self):
        torch.manual_seed(2021)
        a1 = self._model("gold_combo_reliability_s2")
        torch.manual_seed(2021)
        candidate = self._model("hptc_rolling_b25")
        for prefix in ("embedding.", "routing_layers.", "predictor."):
            left = {k: v for k, v in a1.state_dict().items() if k.startswith(prefix)}
            right = {
                k: v for k, v in candidate.state_dict().items() if k.startswith(prefix)
            }
            self.assertEqual(set(left), set(right))
            for key in left:
                torch.testing.assert_close(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
