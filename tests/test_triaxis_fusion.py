import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)
from src.models.triaxis_fusion import (
    RollingTriAxisHistoryRouter,
    TriAxisHistoryRouter,
)


def _build_model(mode="triaxis_self_validating", horizon=96):
    hyperparams = build_hyperparams("ETTm2", horizon, mode)
    args = make_exp_args("ETTm2", 720, horizon, hyperparams, batch_size=2)
    config = PhaseFormerPresetConfig(args, 720, horizon, hyperparams)
    return PhaseFormer(config)


class TriAxisRouterTests(unittest.TestCase):
    def test_uniform_router_is_exact_arithmetic_mean(self):
        router = TriAxisHistoryRouter(96, 24, mode="uniform")
        history = torch.randn(2, 720, 3)
        experts = tuple(torch.randn(2, 96, 3) for _ in range(3))
        output, weights = router(*experts, history)
        expected = torch.stack(experts).mean(dim=0)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            weights, torch.full_like(weights, 1.0 / 3.0)
        )
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0])
        )

    def test_history_evidence_identifies_constructed_analogues(self):
        router = TriAxisHistoryRouter(48, 24)

        # A final period that continues the previous period's within-cycle line.
        cycles = torch.stack(
            (
                torch.zeros(24),
                torch.arange(24, dtype=torch.float32),
                torch.arange(24, 48, dtype=torch.float32),
            )
        ).view(1, 72, 1)
        risks, _ = router.history_evidence(cycles)
        self.assertLess(float(risks[..., 1].max()), 1e-5)

        # A final period that exactly continues an arbitrary inter-cycle delta.
        base = torch.sin(torch.arange(24) / 3.0)
        delta = torch.cos(torch.arange(24) / 5.0)
        cycles = torch.stack((base, base + delta, base + 2 * delta)).view(
            1, 72, 1
        )
        risks, _ = router.history_evidence(cycles)
        self.assertLess(float(risks[..., 2].max()), 1e-5)

    def test_weights_use_history_not_expert_predictions(self):
        torch.manual_seed(7)
        router = TriAxisHistoryRouter(96, 24, mode="self_validating")
        history = torch.randn(2, 720, 3)
        first = tuple(torch.randn(2, 96, 3) for _ in range(3))
        second = tuple(torch.randn(2, 96, 3) * 100 for _ in range(3))
        _, weights_a = router(*first, history)
        _, weights_b = router(*second, history)
        torch.testing.assert_close(weights_a, weights_b)

    def test_phase_and_future_cycle_biases_are_factorized(self):
        router = TriAxisHistoryRouter(48, 24, mode="structural")
        with torch.no_grad():
            router.phase_bias[3, 0] = 2.0
            router.future_cycle_bias[1, 2] = 3.0
        history = torch.randn(1, 72, 1)
        experts = tuple(torch.randn(1, 48, 1) for _ in range(3))
        _, weights = router(*experts, history)
        self.assertGreater(float(weights[0, 3, 0, 0]), 1.0 / 3.0)
        self.assertGreater(float(weights[0, 27, 0, 2]), 1.0 / 3.0)
        self.assertGreater(float(weights[0, 24, 0, 2]), 1.0 / 3.0)


class RollingTriAxisRouterTests(unittest.TestCase):
    def test_rolling_evidence_matches_horizon_and_uses_four_origins(self):
        for horizon in (96, 192):
            with self.subTest(horizon=horizon):
                router = RollingTriAxisHistoryRouter(
                    horizon, 24, mode="rolling_calibrated", origins=4
                )
                risks, risk_std, structural = router.rolling_history_evidence(
                    torch.randn(2, 720, 3)
                )
                expected = (2, 3, horizon // 24, 24, 3)
                self.assertEqual(tuple(risks.shape), expected)
                self.assertEqual(tuple(risk_std.shape), expected)
                self.assertEqual(tuple(structural.shape), expected)
                self.assertTrue(torch.isfinite(risks).all())
                self.assertTrue(torch.isfinite(risk_std).all())
                self.assertEqual(router.origins, 4)

    def test_horizon_matched_trajectory_backtest_is_exact_for_a_line(self):
        router = RollingTriAxisHistoryRouter(
            96, 24, mode="rolling_features", origins=4
        )
        history = torch.arange(720, dtype=torch.float32).view(1, 720, 1)
        risks, _, _ = router.rolling_history_evidence(history)
        self.assertLess(float(risks[..., 1].max()), 1e-4)

    def test_horizon_matched_cycle_backtest_is_exact_for_cycle_drift(self):
        router = RollingTriAxisHistoryRouter(
            96, 24, mode="rolling_features", origins=4
        )
        base = torch.sin(torch.arange(24, dtype=torch.float32) / 3.0)
        delta = torch.cos(torch.arange(24, dtype=torch.float32) / 5.0) * 0.1
        history = torch.stack([base + k * delta for k in range(30)]).view(
            1, 720, 1
        )
        risks, _, _ = router.rolling_history_evidence(history)
        self.assertLess(float(risks[..., 2].max()), 1e-4)

    def test_risk_prior_is_monotonic_and_uncertainty_shrinks_it(self):
        history = torch.randn(1, 720, 1)
        experts = tuple(torch.randn(1, 96, 1) for _ in range(3))
        risk = torch.tensor([0.0, 1.0, 2.0]).view(1, 1, 1, 1, 3).expand(
            1, 1, 4, 24, 3
        )
        structural = torch.zeros_like(risk)

        low = RollingTriAxisHistoryRouter(96, 24, mode="rolling_prior")
        low.rolling_history_evidence = lambda _: (
            risk, torch.zeros_like(risk), structural
        )
        _, low_weights = low(*experts, history)
        self.assertGreater(float(low_weights[..., 0].mean()), float(low_weights[..., 1].mean()))
        self.assertGreater(float(low_weights[..., 1].mean()), float(low_weights[..., 2].mean()))

        uncertain = RollingTriAxisHistoryRouter(96, 24, mode="rolling_prior")
        uncertain.load_state_dict(low.state_dict())
        uncertain.rolling_history_evidence = lambda _: (
            risk, torch.full_like(risk, 5.0), structural
        )
        _, uncertain_weights = uncertain(*experts, history)
        uniform = torch.full_like(uncertain_weights, 1.0 / 3.0)
        self.assertLess(
            float((uncertain_weights - uniform).abs().mean()),
            float((low_weights - uniform).abs().mean()),
        )

    def test_features_ablation_still_warm_starts_uniform(self):
        router = RollingTriAxisHistoryRouter(96, 24, mode="rolling_features")
        history = torch.randn(1, 720, 2)
        experts = tuple(torch.randn(1, 96, 2) for _ in range(3))
        output, weights = router(*experts, history)
        torch.testing.assert_close(
            weights, torch.full_like(weights, 1.0 / 3.0), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(output, torch.stack(experts).mean(dim=0))


class TriAxisPhaseFormerTests(unittest.TestCase):
    def test_all_presets_build_and_forward_at_two_horizons(self):
        for mode in (
            "triaxis_uniform",
            "triaxis_structural",
            "triaxis_self_validating",
            "triaxis_rolling_features",
            "triaxis_rolling_prior",
            "triaxis_rolling_calibrated",
        ):
            for horizon in (96, 192):
                with self.subTest(mode=mode, horizon=horizon):
                    pl.seed_everything(2021, workers=True)
                    model = _build_model(mode, horizon).eval()
                    x = torch.randn(1, 720, 7)
                    with torch.no_grad():
                        output, _, _ = model(x)
                    self.assertEqual(tuple(output.shape), (1, horizon, 7))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(
                        tuple(model.triaxis_weights.shape),
                        (1, horizon, 7, 3),
                    )
                    torch.testing.assert_close(
                        model.triaxis_weights.sum(dim=-1),
                        torch.ones_like(model.triaxis_weights[..., 0]),
                    )

    def test_auxiliary_losses_propagate_gradients(self):
        pl.seed_everything(2021, workers=True)
        model = _build_model().train()
        batch = (
            torch.randn(2, 720, 7),
            torch.randn(2, 96, 7),
            torch.randn(2, 720, 5),
            torch.randn(2, 96, 5),
        )
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        router_grad = model.triaxis_router.router[-1].weight.grad
        trajectory_grad = model.triaxis_trajectory_expert.linear.weight.grad
        cycle_grad = model.triaxis_cycle_expert.out_proj.weight.grad
        self.assertIsNotNone(router_grad)
        self.assertIsNotNone(trajectory_grad)
        self.assertIsNotNone(cycle_grad)
        self.assertGreater(float(router_grad.abs().sum()), 0.0)
        self.assertGreater(float(trajectory_grad.abs().sum()), 0.0)
        self.assertGreater(float(cycle_grad.abs().sum()), 0.0)

    def test_decoder_values_and_marks_cannot_change_output_or_weights(self):
        pl.seed_everything(2021, workers=True)
        model = _build_model("triaxis_rolling_calibrated").eval()
        x = torch.randn(1, 720, 7)
        first_marks = torch.randn(1, 96, 5)
        second_marks = torch.randn(1, 96, 5) * 100
        with torch.no_grad():
            first, _, _ = model(
                x, torch.randn(1, 720, 5), torch.randn(1, 96, 7), first_marks
            )
            weights = model.triaxis_weights.clone()
            second, _, _ = model(
                x, torch.randn(1, 720, 5), torch.randn(1, 96, 7) * 100,
                second_marks,
            )
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(weights, model.triaxis_weights)

    def test_cycle_calibrated_auxiliary_loss_propagates_gradients(self):
        pl.seed_everything(2021, workers=True)
        model = _build_model("triaxis_rolling_calibrated").train()
        batch = (
            torch.randn(2, 720, 7),
            torch.randn(2, 96, 7),
            torch.randn(2, 720, 5),
            torch.randn(2, 96, 5),
        )
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            float(model.triaxis_router.router[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.triaxis_router.risk_strength_raw.grad.abs()), 0.0
        )

    def test_existing_preset_has_no_triaxis_state(self):
        pl.seed_everything(2021, workers=True)
        model = _build_model("gold_combo_reliability_s2")
        self.assertFalse(model.use_triaxis_fusion)
        self.assertFalse(any("triaxis" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
