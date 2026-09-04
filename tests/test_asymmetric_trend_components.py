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


if __name__ == "__main__":
    unittest.main()
