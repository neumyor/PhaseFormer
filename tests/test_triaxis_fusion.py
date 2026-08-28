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
    SafeRegretTriAxisRouter,
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


class SafeRegretRouterTests(unittest.TestCase):
    def test_exact_anchor_initialization_at_both_horizons(self):
        for horizon in (96, 192):
            with self.subTest(horizon=horizon):
                router = SafeRegretTriAxisRouter(horizon, 24)
                history = torch.randn(2, 720, 3)
                anchor = torch.randn(2, horizon, 3)
                experts = tuple(torch.randn_like(anchor) for _ in range(3))
                output, weights = router(anchor, *experts, history)
                torch.testing.assert_close(output, anchor, atol=0, rtol=0)
                torch.testing.assert_close(
                    weights[..., 0], torch.ones_like(weights[..., 0]),
                    atol=0, rtol=0,
                )
                torch.testing.assert_close(
                    weights[..., 1:], torch.zeros_like(weights[..., 1:]),
                    atol=0, rtol=0,
                )

    def test_closed_boundary_has_inward_prediction_gradient(self):
        router = SafeRegretTriAxisRouter(96, 24)
        history = torch.randn(2, 720, 1)
        anchor = torch.ones(2, 96, 1)
        better = tuple(torch.zeros_like(anchor) for _ in range(3))
        output, _ = router(anchor, *better, history)
        output.square().mean().backward()
        self.assertIsNotNone(router.raw_global_accept.grad)
        self.assertLess(float(router.raw_global_accept.grad), 0.0)

        # A temporarily negative raw gate still falls back exactly to A1, but
        # must not become a dead ReLU that can never recover.
        router.zero_grad(set_to_none=True)
        with torch.no_grad():
            router.raw_global_accept.fill_(-0.5)
        output, _ = router(anchor, *better, history)
        torch.testing.assert_close(output, anchor, atol=0, rtol=0)
        output.square().mean().backward()
        self.assertLess(float(router.raw_global_accept.grad), 0.0)

    def test_correction_is_clipped_and_weights_are_convex(self):
        router = SafeRegretTriAxisRouter(96, 24, correction_clip=2.0)
        with torch.no_grad():
            router.raw_global_accept.fill_(1.0)
        history = torch.randn(2, 720, 3)
        anchor = torch.zeros(2, 96, 3)
        experts = tuple(torch.full_like(anchor, 1e6) for _ in range(3))
        output, weights = router(anchor, *experts, history)
        limit = 2.0 * history.std(dim=1, unbiased=False).unsqueeze(1)
        self.assertTrue((output.abs() <= limit + 1e-6).all())
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0])
        )
        self.assertTrue((weights >= 0).all())

    def test_horizon_prior_is_monotone_in_action_logits(self):
        router = SafeRegretTriAxisRouter(
            192, 24, use_horizon_prior=True, horizon_prior_init=0.05
        )
        logits = router._action_logits(torch.randn(1, 720, 1))
        # Expert action order after no-op is phase, trajectory, cycle.
        self.assertGreater(float(logits[0, 0, -1, 1]), float(logits[0, 0, 0, 1]))
        self.assertLess(float(logits[0, 0, -1, 3]), float(logits[0, 0, 0, 3]))

    def test_actions_depend_only_on_encoder_history(self):
        router = SafeRegretTriAxisRouter(96, 24)
        history = torch.randn(1, 720, 2)
        anchor = torch.randn(1, 96, 2)
        first = tuple(torch.randn_like(anchor) for _ in range(3))
        second = tuple(torch.randn_like(anchor) * 100 for _ in range(3))
        router(anchor, *first, history)
        logits = router.last_cycle_action_logits.clone()
        router(anchor, *second, history)
        torch.testing.assert_close(logits, router.last_cycle_action_logits)


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


class SafeRegretPhaseFormerTests(unittest.TestCase):
    def test_a1_checkpoint_is_nested_and_initial_output_is_exact(self):
        pl.seed_everything(2021, workers=True)
        anchor = _build_model("gold_combo_reliability_s2").eval()
        candidate = _build_model("safe_triaxis_guarded").eval()
        incompat = candidate.load_state_dict(anchor.state_dict(), strict=False)
        self.assertFalse(incompat.unexpected_keys)
        self.assertTrue(incompat.missing_keys)
        self.assertTrue(
            all(key.startswith("safe_triaxis_") for key in incompat.missing_keys)
        )
        x = torch.randn(1, 720, 7)
        with torch.no_grad():
            expected, _, _ = anchor(x)
            actual, _, _ = candidate(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            actual, candidate.safe_triaxis_anchor_output, atol=0, rtol=0
        )

    def test_freeze_and_auxiliary_loss_update_only_safe_modules(self):
        pl.seed_everything(2021, workers=True)
        anchor = _build_model("gold_combo_reliability_s2")
        candidate = _build_model("safe_triaxis_guarded")
        candidate.load_state_dict(anchor.state_dict(), strict=False)
        candidate.freeze_safe_triaxis_anchor()
        candidate.train()
        candidate.on_train_epoch_start()
        trainable = [name for name, p in candidate.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("safe_triaxis_") for name in trainable))
        batch = (
            torch.randn(2, 720, 7),
            torch.randn(2, 96, 7),
            torch.randn(2, 720, 5),
            torch.randn(2, 96, 5),
        )
        loss = candidate.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            float(candidate.safe_triaxis_router.router[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(candidate.safe_triaxis_cycle_expert.out_proj.weight.grad.abs().sum()),
            0.0,
        )
        self.assertTrue(
            all(p.grad is None for name, p in candidate.named_parameters()
                if not name.startswith("safe_triaxis_"))
        )

    def test_all_safe_presets_forward(self):
        for mode in (
            "safe_triaxis_anchor",
            "safe_triaxis_regret",
            "safe_triaxis_guarded",
            "safe_triaxis_monotone",
        ):
            with self.subTest(mode=mode):
                model = _build_model(mode).eval()
                with torch.no_grad():
                    output, _, _ = model(torch.randn(1, 720, 7))
                self.assertEqual(tuple(output.shape), (1, 96, 7))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(
                    model.safe_triaxis_weights.sum(dim=-1),
                    torch.ones_like(model.safe_triaxis_weights[..., 0]),
                )


if __name__ == "__main__":
    unittest.main()
