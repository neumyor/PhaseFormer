import unittest

import torch
import torch.nn as nn

from src.models.multi_anchor import (
    ANCHOR_NAMES,
    MultiAnchorPhaseFormer,
    MultiAnchorRouter,
)
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


class ConstantOffsetAnchor(nn.Module):
    def __init__(self, horizon, offset):
        super().__init__()
        self.horizon = horizon
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, x_enc, *args, **kwargs):
        output = x_enc[:, -1:, :].expand(-1, self.horizon, -1) + self.offset
        return output, None, None


def _config(horizon=96):
    hp = build_hyperparams("ETTm2", horizon, "gold_combo_reliability_s2")
    hp["loss_func"] = "huber"
    args = make_exp_args("ETTm2", 720, horizon, hp, batch_size=2)
    return PhaseFormerPresetConfig(args, 720, horizon, hp)


def _bank(horizon, offsets):
    return {
        name: ConstantOffsetAnchor(horizon, offset)
        for name, offset in zip(ANCHOR_NAMES, offsets)
    }


class MultiAnchorRouterTests(unittest.TestCase):
    def test_hard_init_is_exact_a1_and_one_hot(self):
        for horizon in (96, 192):
            with self.subTest(horizon=horizon):
                router = MultiAnchorRouter(horizon, output_mode="hard").eval()
                history = torch.randn(2, 720, 4)
                anchors = tuple(torch.randn(2, horizon, 4) for _ in range(3))
                output, weights = router(history, *anchors)
                torch.testing.assert_close(output, anchors[0], atol=0, rtol=0)
                torch.testing.assert_close(
                    weights.sum(dim=-1), torch.ones_like(weights[..., 0])
                )
                self.assertTrue(torch.logical_or(weights == 0, weights == 1).all())
                self.assertEqual(tuple(router.last_logits.shape), (2, 4, horizon // 24, 3))
                self.assertTrue(torch.isfinite(router.last_features).all())

    def test_soft_output_is_a_convex_forecast(self):
        router = MultiAnchorRouter(96, output_mode="soft").eval()
        history = torch.randn(1, 720, 2)
        anchors = (
            torch.zeros(1, 96, 2),
            torch.ones(1, 96, 2),
            torch.full((1, 96, 2), 2.0),
        )
        output, weights = router(history, *anchors)
        self.assertTrue((output >= 0).all() and (output <= 2).all())
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0])
        )
        self.assertGreater(float(weights[..., 0].mean()), 0.97)

    def test_router_does_not_accept_future_truth_or_marks(self):
        router = MultiAnchorRouter(96).eval()
        history = torch.randn(1, 720, 2)
        anchors = tuple(torch.randn(1, 96, 2) for _ in range(3))
        first, weights = router(history, *anchors)
        second, second_weights = router(history.clone(), *[x.clone() for x in anchors])
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(weights, second_weights)
        with self.assertRaises(ValueError):
            router(history, *anchors, torch.randn(1, 96, 2))

    def test_structural_features_respond_to_history_and_forecasts(self):
        router = MultiAnchorRouter(96)
        history = torch.randn(1, 720, 2)
        anchors = tuple(torch.randn(1, 96, 2) for _ in range(3))
        router(history, *anchors)
        first = router.last_features.clone()
        changed = list(anchors)
        changed[1] = changed[1] + torch.linspace(0, 10, 96).view(1, 96, 1)
        router(history + torch.linspace(0, 5, 720).view(1, 720, 1), *changed)
        self.assertFalse(torch.equal(first, router.last_features))

    def test_straight_through_hard_route_has_gradients(self):
        router = MultiAnchorRouter(96, output_mode="hard").train()
        history = torch.randn(2, 720, 1)
        anchors = (
            torch.ones(2, 96, 1),
            torch.zeros(2, 96, 1),
            torch.full((2, 96, 1), 2.0),
        )
        output, _ = router(history, *anchors)
        output.square().mean().backward()
        self.assertIsNotNone(router.global_logits.grad)
        self.assertGreater(float(router.global_logits.grad.abs().sum()), 0.0)


class MultiAnchorPhaseFormerTests(unittest.TestCase):
    def test_training_uses_shadows_and_eval_uses_full_anchors(self):
        model = MultiAnchorPhaseFormer(
            _config(), _bank(96, (10, 11, 12)), _bank(96, (20, 21, 22))
        )
        x = torch.randn(1, 720, 7)
        model.train()
        train_output, _, _ = model(x)
        self.assertEqual(model.last_anchor_source, "shadow")
        torch.testing.assert_close(train_output, model.last_anchor_outputs[0])
        model.eval()
        eval_output, _, _ = model(x)
        self.assertEqual(model.last_anchor_source, "full")
        torch.testing.assert_close(eval_output, model.last_anchor_outputs[0])
        torch.testing.assert_close(eval_output - train_output, torch.full_like(eval_output, 10.0))

    def test_only_router_is_trainable_and_receives_gradient(self):
        model = MultiAnchorPhaseFormer(
            _config(),
            _bank(96, (0, 1, -1)),
            _bank(96, (0, 1, -1)),
            mean_regret_weight=0.05,
            cvar_weight=0.01,
        ).train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("router.") for name in trainable))
        x = torch.randn(2, 720, 7)
        target = x[:, -1:, :].expand(-1, 96, -1).clone()
        batch = (
            x,
            target,
            torch.randn(2, 720, 5),
            torch.randn(2, 96, 5),
        )
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.router.global_logits.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(
                value.grad is None
                for name, value in model.named_parameters()
                if not name.startswith("router.")
            )
        )
        self.assertTrue(torch.isfinite(model.last_relative_regret).all())

    def test_optimizer_contains_router_parameters_only(self):
        model = MultiAnchorPhaseFormer(
            _config(), _bank(96, (0, 1, 2)), _bank(96, (0, 1, 2))
        )
        configured = model.configure_optimizers()
        optimizer = configured[0][0] if isinstance(configured, tuple) else configured
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.router.parameters()})


if __name__ == "__main__":
    unittest.main()
