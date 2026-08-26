import math
import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_adapters import PeriodPositionEncodedResidualHead
from src.models.phaseformer_presets import (
    PERIODIC_RESIDUAL_PE_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


def _calendar_marks(batch, length, start=0):
    step = torch.arange(start, start + length).repeat(batch, 1)
    minute = step.remainder(4)
    hour = torch.div(step, 4, rounding_mode="floor").remainder(24)
    day_index = torch.div(step, 96, rounding_mode="floor")
    weekday = day_index.remainder(7)
    day = day_index.remainder(28) + 1
    month = torch.div(day_index, 28, rounding_mode="floor").remainder(12) + 1
    return torch.stack((month, day, weekday, hour, minute), dim=-1).float()


class PeriodPositionEncodedResidualHeadTests(unittest.TestCase):
    def _head(self, encoding_type, **kwargs):
        return PeriodPositionEncodedResidualHead(
            seq_len=24,
            pred_len=12,
            period_len=6,
            encoding_type=encoding_type,
            pe_dim=16,
            temperature=0.1,
            cycle_decay=0.1,
            blend_init=0.1,
            **kwargs,
        )

    def test_all_encoding_types_have_finite_output_and_normalized_attention(self):
        x = torch.randn(2, 24, 3)
        x_mark = _calendar_marks(2, 24)
        y_mark = _calendar_marks(2, 18, start=18)
        for encoding_type in sorted(PeriodPositionEncodedResidualHead.ENCODING_TYPES):
            with self.subTest(encoding_type=encoding_type):
                head = self._head(encoding_type)
                out = head(x, x_mark, y_mark)
                self.assertEqual(tuple(out.shape), (2, 12, 3))
                self.assertTrue(torch.isfinite(out).all())
                attention = head.last_attention
                self.assertIn(attention.ndim, (2, 3))
                torch.testing.assert_close(
                    attention.sum(dim=-1),
                    torch.ones_like(attention.sum(dim=-1)),
                    atol=1e-6,
                    rtol=1e-6,
                )
                self.assertTrue(math.isfinite(head.last_attention_entropy))

    def test_horizon_blend_starts_at_registered_prior(self):
        head = self._head("harmonic")
        head(torch.randn(2, 24, 3))
        torch.testing.assert_close(
            head.last_beta,
            torch.full_like(head.last_beta, 0.1),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_cycle_encoding_prefers_same_phase(self):
        head = PeriodPositionEncodedResidualHead(
            24, 1, 6, "cycle", temperature=0.1, cycle_decay=0.0
        )
        attention = head._attention_weights(torch.randn(1, 24, 1))[0]
        # Forecast position 24 has the same phase as history positions 0/6/12/18.
        self.assertGreater(float(attention[18]), float(attention[21]))
        self.assertGreater(float(attention[12]), float(attention[15]))

    def test_cycle_decay_prefers_recent_equal_phase_observation(self):
        head = PeriodPositionEncodedResidualHead(
            24, 1, 6, "harmonic", temperature=0.1, cycle_decay=0.2
        )
        attention = head._attention_weights(torch.randn(1, 24, 1))[0]
        self.assertGreater(float(attention[18]), float(attention[12]))
        self.assertGreater(float(attention[12]), float(attention[6]))

    def test_calendar_attention_is_sample_specific(self):
        head = self._head("calendar")
        x = torch.randn(2, 24, 1)
        x_mark = _calendar_marks(2, 24)
        y_mark = _calendar_marks(2, 18, start=18)
        # Change only the second sample's future hour, so its matching kernel differs.
        y_mark[1, -12:, 3] = (y_mark[1, -12:, 3] + 7).remainder(24)
        attention = head._attention_weights(x, x_mark, y_mark)
        self.assertEqual(tuple(attention.shape), (2, 12, 24))
        self.assertFalse(torch.equal(attention[0], attention[1]))

    def test_learned_periodic_encodings_receive_gradients(self):
        for encoding_type, parameter_name in (
            ("time2vec", "time2vec_frequency"),
            ("lff", "lff_log_frequency_scale"),
        ):
            with self.subTest(encoding_type=encoding_type):
                head = self._head(encoding_type)
                output = head(torch.randn(2, 24, 3))
                output.square().mean().backward()
                grad = getattr(head, parameter_name).grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_invalid_encoding_and_calendar_without_marks_fail_clearly(self):
        with self.assertRaises(ValueError):
            PeriodPositionEncodedResidualHead(24, 12, 6, "unknown")
        with self.assertRaisesRegex(ValueError, "timestamp marks"):
            self._head("calendar")(torch.randn(2, 24, 3))


class PeriodicResidualPresetTests(unittest.TestCase):
    def test_all_presets_only_replace_the_rcrf_residual_head(self):
        parent = build_hyperparams("ETTm2", 96, "gold_combo_reliability_s2")
        for mode, encoding_type in PERIODIC_RESIDUAL_PE_MODES.items():
            with self.subTest(mode=mode):
                candidate = build_hyperparams("ETTm2", 96, mode)
                self.assertTrue(candidate["use_rcrf_fusion"])
                self.assertTrue(candidate["use_periodic_residual_pe"])
                self.assertEqual(candidate["weak_period_residual_head_type"], "periodic_pe")
                self.assertEqual(candidate["periodic_residual_pe_type"], encoding_type)
                for key in (
                    "use_phase_uncertainty_shrinkage",
                    "phase_uncertainty_min",
                    "use_phase_period_level_calibration",
                    "phase_level_calib_gate_init",
                    "use_phase_noise_hifreq_damping",
                    "rcrf_alpha_init",
                    "rcrf_sensitivity_init",
                    "rcrf_s_max",
                ):
                    self.assertEqual(candidate[key], parent[key])

    def test_calendar_preset_runs_full_phaseformer_forward_with_marks(self):
        hyperparams = build_hyperparams("ETTm2", 96, "rcrf_pe_calendar")
        pl.seed_everything(2021, workers=True)
        args = make_exp_args("ETTm2", 720, 96, hyperparams)
        config = PhaseFormerPresetConfig(args, 720, 96, hyperparams)
        model = PhaseFormer(config).eval()
        x = torch.randn(1, 720, 7)
        x_mark = _calendar_marks(1, 720)
        y_mark = _calendar_marks(1, 144, start=672)
        with torch.no_grad():
            output, _, _ = model(x, x_mark, None, y_mark)
        self.assertEqual(tuple(output.shape), (1, 96, 7))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(model.weak_period_residual.last_attention)


if __name__ == "__main__":
    unittest.main()
