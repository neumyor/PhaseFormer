import unittest

import torch

from src.models.asymmetric_trend_components import (
    TREND_COMPONENTS,
    extract_trend_component,
)
from src.models.phase_adapters import RevIN
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)
from src.models.PhaseFormer import PhaseFormer


class AsymmetricTrendComponentTests(unittest.TestCase):
    def test_all_components_are_endpoint_anchored_and_shape_preserving(self):
        torch.manual_seed(7)
        x = torch.randn(2, 720, 3)
        for component in TREND_COMPONENTS:
            a = extract_trend_component(x, component)
            self.assertEqual(a.shape, x.shape)
            torch.testing.assert_close(a[:, -1, :], torch.zeros_like(a[:, -1, :]))

    def test_trend_filter_is_scale_equivariant(self):
        torch.manual_seed(8)
        x = torch.randn(2, 720, 3)
        a = extract_trend_component(x, "trend_filter", trend_filter_iterations=32)
        scaled = extract_trend_component(3.7 * x, "trend_filter", trend_filter_iterations=32)
        torch.testing.assert_close(scaled, 3.7 * a, rtol=2e-5, atol=2e-5)

    def test_causal_components_do_not_depend_on_future_history(self):
        torch.manual_seed(13)
        x = torch.randn(2, 720, 3)
        changed = x.clone()
        changed[:, 360:, :] += 5.0
        for component in ("causal_ema", "causal_local_linear", "holt_local_linear"):
            original = extract_trend_component(x, component)
            altered = extract_trend_component(changed, component)
            # Endpoint anchoring translates every A value when the final trend
            # changes; causal extraction requires the pre-cutoff *shape* to be
            # invariant after removing that common anchor translation.
            torch.testing.assert_close(
                original[:, :360, :] - original[:, :1, :],
                altered[:, :360, :] - altered[:, :1, :],
            )

    def test_shared_revin_stats_match_full_branch_when_component_is_zero(self):
        x = torch.randn(2, 720, 3)
        revin = RevIN(3, affine=True)
        normalized, stats = revin.normalize(x)
        torch.testing.assert_close(revin.normalize_with_stats(x, stats), normalized)

    def test_flag_off_matches_historical_weak_residual_forward(self):
        torch.manual_seed(11)
        hp = build_hyperparams("ETTm1", 96, "weak_residual")
        args = make_exp_args("ETTm1", 720, 96, hp)
        historical = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp)).eval()
        torch.manual_seed(11)
        hp_explicit = dict(hp, weak_residual_asymmetric_component="none")
        explicit = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp_explicit)).eval()
        x = torch.randn(1, 720, historical.enc_in)
        with torch.no_grad():
            historical_out = historical(x)[0]
            explicit_out = explicit(x)[0]
        torch.testing.assert_close(historical_out, explicit_out, rtol=0, atol=0)

    def test_asymmetric_preset_is_registered(self):
        hp = build_hyperparams("ETTm1", 96, "weak_residual_asymmetric_trend")
        self.assertEqual(hp["weak_residual_asymmetric_component"], "cycle_levels")

    def test_trend_filter_candidate_forward_runs(self):
        hp = build_hyperparams("ETTh1", 96, "weak_residual_asymmetric_trend")
        hp.update(
            weak_residual_asymmetric_component="trend_filter",
            weak_residual_trend_filter_iterations=4,
        )
        args = make_exp_args("ETTh1", 720, 96, hp)
        model = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hp)).eval()
        with torch.no_grad():
            output, _, _ = model(torch.randn(1, 720, model.enc_in))
        self.assertEqual(tuple(output.shape), (1, 96, model.enc_in))


if __name__ == "__main__":
    unittest.main()
