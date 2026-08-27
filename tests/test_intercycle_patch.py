import io
import math
import unittest

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.intercycle_patch import (
    ICPTPE_TYPES,
    CycleNetStyleResidualHead,
    InterCyclePatchResidualHead,
    RepeatLastCycleResidualHead,
)
from src.models.phase_adapters import WeakPeriodResidualHead
from src.models.phaseformer_presets import (
    INTERCYCLE_PE_MODES,
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


class InterCyclePatchHeadTests(unittest.TestCase):
    def test_repeat_last_cycle_is_exact_persistence(self):
        head = RepeatLastCycleResidualHead(720, 96, 24)
        x = torch.randn(2, 720, 3)
        out = head(x)
        self.assertEqual(tuple(out.shape), (2, 96, 3))
        torch.testing.assert_close(
            out, x[:, -24:, :].repeat(1, 4, 1), atol=1e-6, rtol=1e-6
        )
        # Zero-parameter baseline.
        self.assertEqual(sum(p.numel() for p in head.parameters()), 0)

    def test_cycle_net_warm_starts_at_last_value_repeat(self):
        head = CycleNetStyleResidualHead(720, 96, 24)
        x = torch.randn(2, 720, 3)
        out = head(x)
        self.assertEqual(tuple(out.shape), (2, 96, 3))
        torch.testing.assert_close(
            out, x[:, -1:, :].expand(-1, 96, -1), atol=1e-6, rtol=1e-6
        )

    def test_all_pes_forward_finite_and_zero_init_is_repeat_last_cycle(self):
        x = torch.randn(2, 720, 3)
        x_mark = _calendar_marks(2, 720)
        y_mark = _calendar_marks(2, 144, start=672)
        for pe in ICPTPE_TYPES:
            with self.subTest(pe=pe):
                head = InterCyclePatchResidualHead(720, 96, 24, pe_type=pe)
                out = head(x, x_mark, y_mark) if pe == "calendar" else head(x)
                self.assertEqual(tuple(out.shape), (2, 96, 3))
                self.assertTrue(torch.isfinite(out).all())
                # W_out zero-init -> initial output is exactly RepeatLastCycle.
                torch.testing.assert_close(
                    out, x[:, -24:, :].repeat(1, 4, 1), atol=1e-6, rtol=1e-6
                )

    def test_all_pe_parameters_receive_gradients(self):
        x = torch.randn(2, 720, 3)
        x_mark = _calendar_marks(2, 720)
        y_mark = _calendar_marks(2, 144, start=672)
        for pe in ICPTPE_TYPES:
            with self.subTest(pe=pe):
                head = InterCyclePatchResidualHead(720, 96, 24, pe_type=pe)
                out = head(x, x_mark, y_mark) if pe == "calendar" else head(x)
                out.square().mean().backward()
                grads = [
                    p.grad for p in head.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                self.assertTrue(grads, f"{pe}: no gradient received")
                self.assertTrue(
                    any(float(g.abs().sum()) > 0.0 for g in grads),
                    f"{pe}: all gradients are zero",
                )

    def test_four_horizons_and_862_channels_forward(self):
        for pred_len in (96, 192, 336, 720):
            with self.subTest(pred_len=pred_len):
                head = InterCyclePatchResidualHead(720, pred_len, 24, pe_type="rope")
                out = head(torch.randn(1, 720, 862))
                self.assertEqual(tuple(out.shape), (1, pred_len, 862))
                self.assertTrue(torch.isfinite(out).all())

    def test_nondivisible_lengths_pad_and_trim(self):
        head = InterCyclePatchResidualHead(1000, 100, 24, pe_type="relative")
        x = torch.randn(2, 1000, 7)
        out = head(x)
        self.assertEqual(tuple(out.shape), (2, 100, 7))
        # Replicate-padding preserves the anchor: initial output repeats the
        # last 24 points.
        torch.testing.assert_close(
            out, x[:, -24:, :].repeat(1, 5, 1)[:, :100], atol=1e-6, rtol=1e-6
        )

    def test_b4_no_anchor_and_b5_no_attention(self):
        x = torch.randn(2, 720, 3)
        head = InterCyclePatchResidualHead(
            720, 96, 24, pe_type="lff", use_last_cycle_anchor=False
        )
        out = head(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(out.shape), (2, 96, 3))
        head = InterCyclePatchResidualHead(
            720, 96, 24, pe_type="none", use_attention=False
        )
        out = head(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(out.shape), (2, 96, 3))

    def test_calendar_without_marks_fails_clearly(self):
        head = InterCyclePatchResidualHead(720, 96, 24, pe_type="calendar")
        with self.assertRaises(ValueError):
            head(torch.randn(2, 720, 3))

    def test_checkpoint_round_trip_preserves_weights(self):
        head = InterCyclePatchResidualHead(720, 96, 24, pe_type="relative")
        head(torch.randn(2, 720, 3))
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)
        clone = InterCyclePatchResidualHead(720, 96, 24, pe_type="relative")
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(set(head.state_dict()), set(clone.state_dict()))
        for key in head.state_dict():
            torch.testing.assert_close(head.state_dict()[key], clone.state_dict()[key])

    def test_batch_one_noncontiguous_and_cpu(self):
        x = torch.randn(2, 7, 720)
        x_nc = x.transpose(1, 2)[:1]  # shape (1, 720, 7), non-contiguous view
        self.assertFalse(x_nc.is_contiguous())
        head = InterCyclePatchResidualHead(720, 96, 24, pe_type="relative")
        out = head(x_nc)
        self.assertEqual(tuple(out.shape), (1, 96, 7))
        self.assertTrue(torch.isfinite(out).all())

    def test_attention_diagnostics_are_finite(self):
        head = InterCyclePatchResidualHead(720, 96, 24, pe_type="relative")
        head(torch.randn(2, 720, 3))
        self.assertIsNotNone(head.last_attention)
        self.assertTrue(math.isfinite(head.last_attention_entropy))
        self.assertIsNotNone(head.last_top_lags)
        self.assertTrue(math.isfinite(head.last_delta_norm))
        self.assertTrue(math.isfinite(head.last_anchor_norm))


class InterCyclePatchPresetTests(unittest.TestCase):
    def test_all_icpt_presets_only_replace_the_rcrf_residual_head(self):
        parent = build_hyperparams("ETTm2", 96, "gold_combo_reliability_s2")
        for mode in sorted(INTERCYCLE_PE_MODES):
            with self.subTest(mode=mode):
                candidate = build_hyperparams("ETTm2", 96, mode)
                self.assertTrue(candidate["use_rcrf_fusion"])
                self.assertEqual(
                    candidate["weak_period_residual_head_type"], "intercycle"
                )
                self.assertEqual(
                    candidate["intercycle_pe_type"], INTERCYCLE_PE_MODES[mode]
                )
                # Phase stack, RCRF and training settings inherited unchanged.
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

    def test_gold_combo_head_is_still_nlinear_when_intercycle_off(self):
        # Flag-off: the intercycle head is only constructed on demand, so the
        # frozen RCRF candidate keeps the NLinear head type unchanged.
        hyperparams = build_hyperparams("ETTm2", 96, "gold_combo_reliability_s2")
        self.assertEqual(
            hyperparams.get("weak_period_residual_head_type", "shared"), "shared"
        )
        pl.seed_everything(2021, workers=True)
        args = make_exp_args("ETTm2", 720, 96, hyperparams)
        config = PhaseFormerPresetConfig(args, 720, 96, hyperparams)
        model = PhaseFormer(config).eval()
        self.assertIsInstance(model.weak_period_residual, WeakPeriodResidualHead)

    def test_calendar_preset_runs_full_phaseformer_forward_with_marks(self):
        hyperparams = build_hyperparams("ETTm2", 96, "rcrf_icpt_calendar")
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

    def test_ablation_modes_build_and_forward(self):
        for mode in ("icpt_only", "icpt_fixed_fusion", "icpt_patch16",
                     "icpt_no_anchor", "icpt_no_attention"):
            with self.subTest(mode=mode):
                hyperparams = build_hyperparams("ETTh2", 720, mode)
                pl.seed_everything(2021, workers=True)
                args = make_exp_args("ETTh2", 720, 720, hyperparams)
                config = PhaseFormerPresetConfig(args, 720, 720, hyperparams)
                model = PhaseFormer(config).eval()
                x = torch.randn(1, 720, 7)
                with torch.no_grad():
                    output, _, _ = model(x)
                self.assertEqual(tuple(output.shape), (1, 720, 7))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
