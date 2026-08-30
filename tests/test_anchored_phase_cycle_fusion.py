import unittest

import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.anchored_phase_cycle_fusion import (
    AnchoredPhaseCycleFusionComposer,
)
from src.models.phase_adapters import WeakPeriodResidualHead
from src.models.phaseformer_presets import (
    PCTF_ANCHORED_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


def _composer(strategy, period=24, pred_len=48):
    return AnchoredPhaseCycleFusionComposer(
        seq_len=120 if period == 24 else 720,
        pred_len=pred_len,
        cycle_period_len=period,
        strategy=strategy,
        d_model=8,
        num_heads=1,
        ffn_dim=16,
        masked_origins=2,
    )


class AnchoredPhaseCycleFusionComposerTests(unittest.TestCase):
    @staticmethod
    def _inputs(seq_len=120, pred_len=48, phase_period=24):
        history = torch.randn(2, seq_len, 3)
        anchor = torch.randn(2, pred_len, 3)
        phase = torch.randn(2, pred_len, 3)
        trajectory = torch.randn(2, pred_len, 3)
        phase_series = torch.randn(
            2, 3, phase_period, seq_len // phase_period
        )
        predictor = WeakPeriodResidualHead(seq_len, pred_len)
        return anchor, phase, trajectory, history, phase_series, predictor

    def test_every_strategy_is_exactly_the_anchor_at_initialization(self):
        values = self._inputs()
        for strategy in AnchoredPhaseCycleFusionComposer.STRATEGIES:
            with self.subTest(strategy=strategy):
                model = _composer(strategy)
                output = model(*values[:-1], trajectory_predictor=values[-1])
                self.assertTrue(torch.equal(output, values[0]))
                self.assertEqual(model.last_anchor_identity_max_abs, 0.0)
                self.assertEqual(model.last_update_horizon_mean_max, 0.0)

    def test_nonzero_component_updates_remain_orthogonal_and_mean_conserving(self):
        anchor, phase, trajectory, history, phase_series, predictor = self._inputs()
        model = _composer("component_cycle")
        with torch.no_grad():
            model.level_raw.fill_(0.4)
            model.shape_raw.fill_(-0.3)
        output = model(
            anchor, phase, trajectory, history, phase_series,
            trajectory_predictor=predictor,
        )
        self.assertGreater(float((output - anchor).abs().sum()), 0.0)
        torch.testing.assert_close(
            output.mean(dim=1), anchor.mean(dim=1), atol=2e-6, rtol=1e-6
        )
        self.assertLess(model.last_shape_cycle_mean_max, 2e-6)
        self.assertLess(model.last_projection_inner_product_max, 2e-5)
        self.assertLessEqual(
            float(model.last_level_coefficient.abs().max()), 0.25
        )
        self.assertLessEqual(
            float(model.last_shape_coefficient.abs().max()), 0.25
        )

    def test_explicit_legacy_defaults_preserve_nonzero_outputs(self):
        values = self._inputs()
        implicit = _composer("mlp_evidence")
        explicit = AnchoredPhaseCycleFusionComposer(
            seq_len=120,
            pred_len=48,
            cycle_period_len=24,
            strategy="mlp_evidence",
            d_model=8,
            num_heads=1,
            ffn_dim=16,
            masked_origins=2,
            detach_references=False,
            level_mode="horizon_centered",
            global_level_max=0.05,
        )
        explicit.load_state_dict(implicit.state_dict())
        with torch.no_grad():
            for model in (implicit, explicit):
                model.level_raw.fill_(0.2)
                model.shape_raw.fill_(-0.3)
                model.gate_mlp[-1].bias.copy_(torch.tensor([0.1, -0.1]))
        implicit_output = implicit(
            *values[:-1], trajectory_predictor=values[-1]
        )
        explicit_output = explicit(
            *values[:-1], trajectory_predictor=values[-1]
        )
        self.assertTrue(torch.equal(implicit_output, explicit_output))

    def test_horizon_matched_contexts_are_strictly_causal(self):
        model = _composer("monotonic_evidence")
        history = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 24, 1)
        cycle_contexts, full_contexts, targets, templates, origins = (
            model._rolling_contexts(history.reshape(1, 120, 1))
        )
        self.assertEqual(origins, 2)
        values = cycle_contexts.reshape(2, 5, 24, 1)[:, :, 0, 0]
        torch.testing.assert_close(
            values,
            torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 2.0]]),
        )
        torch.testing.assert_close(
            targets[:, :, 0, 0], torch.tensor([[2, 3], [3, 4.0]])
        )
        torch.testing.assert_close(
            templates[:, 0, 0], torch.tensor([0.5, 1.0])
        )
        torch.testing.assert_close(full_contexts, cycle_contexts)

    def test_period_is_decoupled_by_causal_history_trimming(self):
        values = self._inputs(seq_len=720, pred_len=96)
        model = _composer("monotonic_evidence", period=96, pred_len=96)
        output = model(*values[:-1], trajectory_predictor=values[-1])
        self.assertEqual(model.cycle_seq_len, 672)
        self.assertEqual(model.cycle_prefix_trim, 48)
        self.assertTrue(torch.equal(output, values[0]))
        self.assertEqual(tuple(model.last_shape_confidence.shape), (2, 3, 1))

    def test_zero_initialized_coefficients_receive_main_loss_gradients(self):
        anchor, phase, trajectory, history, phase_series, predictor = self._inputs()
        model = _composer("component_cycle")
        anchor = anchor.requires_grad_(True)
        output = model(
            anchor, phase, trajectory, history, phase_series,
            trajectory_predictor=predictor,
        )
        output.square().mean().backward()
        self.assertGreater(float(model.shape_raw.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.level_raw.grad.abs().sum()), 0.0)
        # Exact anchoring intentionally delays main-loss ICPT gradients until a
        # coefficient moves; the owning model supplies component auxiliaries.
        cycle_grad = model.cycle.out_proj.weight.grad
        self.assertTrue(cycle_grad is None or float(cycle_grad.abs().sum()) == 0.0)

    def test_history_referenced_level_repairs_the_one_cycle_null_space(self):
        values = self._inputs(seq_len=720, pred_len=96)
        legacy = AnchoredPhaseCycleFusionComposer(
            seq_len=720,
            pred_len=96,
            cycle_period_len=96,
            strategy="component_cycle",
            d_model=8,
            num_heads=1,
            ffn_dim=16,
            level_mode="horizon_centered",
        )
        repaired = AnchoredPhaseCycleFusionComposer(
            seq_len=720,
            pred_len=96,
            cycle_period_len=96,
            strategy="component_cycle",
            d_model=8,
            num_heads=1,
            ffn_dim=16,
            level_mode="history_referenced",
            global_level_max=0.05,
        )
        repaired.load_state_dict(legacy.state_dict())
        with torch.no_grad():
            legacy.level_raw.fill_(0.4)
            legacy.shape_raw.zero_()
            repaired.level_raw.fill_(0.4)
            repaired.shape_raw.zero_()
        legacy_output = legacy(
            *values[:-1], trajectory_predictor=values[-1]
        )
        repaired_output = repaired(
            *values[:-1], trajectory_predictor=values[-1]
        )
        self.assertTrue(torch.equal(legacy_output, values[0]))
        self.assertGreater(float((repaired_output - values[0]).abs().sum()), 0.0)
        self.assertGreater(
            float(repaired.last_global_level_update.abs().sum()), 0.0
        )
        self.assertLessEqual(
            float(repaired.last_level_coefficient.abs().max()), 0.25
        )

    def test_detached_references_do_not_receive_correction_gradients(self):
        anchor, phase, trajectory, history, phase_series, predictor = self._inputs()
        model = AnchoredPhaseCycleFusionComposer(
            seq_len=120,
            pred_len=48,
            cycle_period_len=24,
            strategy="component_cycle",
            d_model=8,
            num_heads=1,
            ffn_dim=16,
            detach_references=True,
            level_mode="history_referenced",
        )
        phase.requires_grad_(True)
        trajectory.requires_grad_(True)
        with torch.no_grad():
            model.level_raw.fill_(0.4)
            model.shape_raw.fill_(0.4)
        output = model(
            anchor, phase, trajectory, history, phase_series,
            trajectory_predictor=predictor,
        )
        output.square().mean().backward()
        self.assertIsNone(phase.grad)
        self.assertIsNone(trajectory.grad)
        self.assertGreater(
            float(model.cycle.out_proj.weight.grad.abs().sum()), 0.0
        )


class AnchoredPhaseCycleFusionPresetTests(unittest.TestCase):
    @staticmethod
    def _model(mode, horizon=96, cycle_period=None):
        hp = build_hyperparams("ETTm2", horizon, mode)
        if cycle_period is not None:
            hp["anchored_pctf_cycle_period_len"] = cycle_period
        args = make_exp_args("ETTm2", 720, horizon, hp, batch_size=2)
        return PhaseFormer(PhaseFormerPresetConfig(args, 720, horizon, hp))

    def test_presets_nest_the_complete_a2_anchor(self):
        for mode, strategy in PCTF_ANCHORED_MODES.items():
            with self.subTest(mode=mode):
                model = self._model(mode)
                self.assertTrue(model.use_anchored_phase_cycle_fusion)
                self.assertTrue(model.use_weak_period_residual)
                self.assertTrue(model.use_rcrf_fusion)
                self.assertTrue(model.use_periodic_residual_pe)
                self.assertEqual(
                    model.weak_period_residual.encoding_type, "lff"
                )
                self.assertEqual(
                    model.anchored_phase_cycle_fusion.strategy, strategy
                )

    def test_same_seed_candidate_contains_identical_a2_state_and_output(self):
        torch.manual_seed(2021)
        a2 = self._model("rcrf_pe_lff").eval()
        torch.manual_seed(2021)
        candidate = self._model(
            "pctf_anchor_monotonic", cycle_period=96
        ).eval()
        candidate_state = candidate.state_dict()
        for key, value in a2.state_dict().items():
            self.assertIn(key, candidate_state)
            torch.testing.assert_close(value, candidate_state[key])
        x = torch.randn(1, 720, 7)
        with torch.no_grad():
            expected = a2(x)[0]
            actual = candidate(x)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            torch.equal(actual, candidate.anchored_pctf_anchor_output)
        )

    def test_every_preset_has_complete_anchor_identical_forward(self):
        x = torch.randn(1, 720, 7)
        for mode in PCTF_ANCHORED_MODES:
            with self.subTest(mode=mode):
                model = self._model(mode).eval()
                with torch.no_grad():
                    output = model(x)[0]
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(
                    torch.equal(output, model.anchored_pctf_anchor_output)
                )

    def test_component_auxiliary_trains_icpt_at_zero_correction(self):
        model = self._model("pctf_anchor_monotonic", cycle_period=96).train()
        x = torch.randn(2, 720, 7)
        target = torch.randn(2, 96, 7)
        model(x)
        auxiliary, shape_loss, level_loss = (
            model._anchored_pctf_auxiliary_loss(target)
        )
        auxiliary.backward()
        gradient = model.anchored_phase_cycle_fusion.cycle.out_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(shape_loss))
        self.assertTrue(torch.isfinite(level_loss))

    def test_residual_and_marginal_auxiliaries_reach_icpt_and_gate(self):
        model = self._model(
            "pctf_anchor_repair_joint_marginal", cycle_period=96
        ).train()
        x = torch.randn(2, 720, 7)
        target = torch.randn(2, 96, 7)
        model(x)
        auxiliary, _, _ = model._anchored_pctf_auxiliary_loss(target)
        auxiliary.backward()
        cycle_gradient = (
            model.anchored_phase_cycle_fusion.cycle.out_proj.weight.grad
        )
        gate_gradient = (
            model.anchored_phase_cycle_fusion.gate_mlp[-1].weight.grad
        )
        self.assertIsNotNone(cycle_gradient)
        self.assertGreater(float(cycle_gradient.abs().sum()), 0.0)
        self.assertIsNotNone(gate_gradient)
        self.assertGreater(float(gate_gradient.abs().sum()), 0.0)
        self.assertGreater(
            float(model.anchored_pctf_last_gate_aux_loss.detach()), 0.0
        )

    def test_freeze_and_joint_lr_controls_have_the_intended_scope(self):
        frozen = self._model("pctf_anchor_diag_frozen_residual")
        frozen.freeze_anchored_pctf_anchor()
        trainable = [
            name for name, parameter in frozen.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith("anchored_phase_cycle_fusion.")
            for name in trainable
        ))

        joint = self._model("pctf_anchor_repair_joint_residual")
        optimizer = joint.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(len(optimizer.param_groups), 2)
        learning_rates = sorted(group["lr"] for group in optimizer.param_groups)
        base = float(joint.args.training_args.learning_rate)
        torch.testing.assert_close(
            torch.tensor(learning_rates),
            torch.tensor([base * 0.1, base]),
        )

    def test_h192_complete_forward(self):
        model = self._model(
            "pctf_anchor_phase_modulation", horizon=192, cycle_period=48
        ).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 720, 7))[0]
        self.assertEqual(tuple(output.shape), (1, 192, 7))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
