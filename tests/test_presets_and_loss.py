import unittest

import torch

from src.models.PhaseFormer import (
    PhaseFormer,
    PhaseUncertaintyShrinkage,
    WeakPeriodResidualHead,
)
from src.models.phase_adapters import (
    PhaseUncertaintyShrinkage as SplitPhaseUncertaintyShrinkage,
)
from src.models.phase_adapters import WeakPeriodResidualHead as SplitWeakPeriodResidualHead
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    get_latest_overrides,
    make_exp_args,
)


class PresetAndLossTests(unittest.TestCase):
    def test_legacy_huber_flag_is_normalized_to_real_loss_name(self):
        hyperparams = build_hyperparams("ETTm2", 96, "original")
        args = make_exp_args("ETTm2", 720, 96, hyperparams)
        self.assertEqual(args.training_args.loss_func, "huber")
        self.assertTrue(args.training_args.use_huber_loss)

    def test_explicit_mae_disables_huber_compatibility_flag(self):
        hyperparams = build_hyperparams("Exchange", 96, "latest")
        args = make_exp_args("Exchange", 720, 96, hyperparams)
        self.assertEqual(args.training_args.loss_func, "mae")
        self.assertFalse(args.training_args.use_huber_loss)

    def test_traffic_attention_settings_reach_model_config(self):
        hyperparams = build_hyperparams("Traffic", 96, "latest")
        args = make_exp_args("Traffic", 720, 96, hyperparams)
        config = PhaseFormerPresetConfig(args, 720, 96, hyperparams)
        self.assertEqual(config.phase_attn_window, 12)
        self.assertFalse(config.phase_use_pos_embed)

    def test_huber_matches_torch_reference(self):
        hyperparams = build_hyperparams("ETTh1", 720, "original")
        args = make_exp_args("ETTh1", 720, 720, hyperparams)
        config = PhaseFormerPresetConfig(args, 720, 720, hyperparams)
        model = PhaseFormer(config)
        prediction = torch.tensor([0.0, 0.5, 2.0])
        target = torch.zeros_like(prediction)
        actual = model._compute_loss(prediction, target)
        expected = torch.nn.functional.huber_loss(
            prediction, target, delta=config.huber_delta
        )
        torch.testing.assert_close(actual, expected)

    def test_split_adapters_keep_legacy_imports_and_state_keys(self):
        self.assertIs(WeakPeriodResidualHead, SplitWeakPeriodResidualHead)
        self.assertIs(PhaseUncertaintyShrinkage, SplitPhaseUncertaintyShrinkage)
        hyperparams = build_hyperparams("ETTm2", 96, "original")
        args = make_exp_args("ETTm2", 720, 96, hyperparams)
        config = PhaseFormerPresetConfig(args, 720, 96, hyperparams)
        source = PhaseFormer(config)
        restored = PhaseFormer(config)
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(tuple(source.state_dict()), tuple(restored.state_dict()))


class LatestPolicyTableTests(unittest.TestCase):
    def test_all_32_task_combos_resolve_and_gates_have_scheme(self):
        datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
        horizons = [96, 192, 336, 720]
        for ds in datasets:
            for hz in horizons:
                overrides = get_latest_overrides(ds, hz)
                self.assertIn("scheme_name", overrides, f"{ds} {hz} missing scheme_name")
                self.assertTrue(overrides["scheme_name"])

    def test_full_horizon_entry_covers_all_horizons(self):
        # Exchange/Whatever uses (ds, None) -> applies to every horizon.
        for hz in [96, 192, 336, 720]:
            self.assertEqual(
                get_latest_overrides("Exchange", hz)["scheme_name"],
                "latest_exchange_residual_mae",
            )

    def test_guardrail_untouched_datasets_use_original_phase_only(self):
        # Traffic has no weak-period mechanism in the latest policy.
        for hz in [96, 192, 336, 720]:
            self.assertEqual(
                get_latest_overrides("Traffic", hz)["scheme_name"],
                "latest_original_guardrail",
            )

    def test_policy_table_is_deep_copied(self):
        e = get_latest_overrides("Exchange", 96)
        e["scheme_name"] = "mutated"
        again = get_latest_overrides("Exchange", 96)
        self.assertEqual(again["scheme_name"], "latest_exchange_residual_mae")


if __name__ == "__main__":
    unittest.main()
