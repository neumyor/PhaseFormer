import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phase_adapters import (
    PeriodPositionEncodedResidualHead,
    WeakPeriodResidualHead,
)
from src.models.periodic_residual_experts import (
    AdaptiveMultiPeriodResidualHead,
    DualReliabilityPeriodicFusion,
    PhaseErrorPeriodicMemoryHead,
)
from src.models.phaseformer_presets import (
    PERIODIC_COMPLEMENT_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)
from scripts.run_periodic_residual_next_stage import (
    DATASETS,
    HORIZONS,
    MODES,
    SEEDS,
    build_commands,
    summarize,
)


class PhaseErrorPeriodicMemoryTests(unittest.TestCase):
    def test_warm_start_matches_nlinear_and_attention_is_normalized(self):
        x = torch.randn(2, 48, 3)
        head = PhaseErrorPeriodicMemoryHead(48, 12, 6)
        baseline = WeakPeriodResidualHead(48, 12)
        torch.testing.assert_close(head(x), baseline(x), atol=0.0, rtol=0.0)
        self.assertEqual(tuple(head.last_attention.shape), (2, 3, 7))
        torch.testing.assert_close(
            head.last_attention.sum(dim=-1),
            torch.ones(2, 3),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(math.isfinite(head.last_attention_entropy))
        torch.testing.assert_close(
            head.last_correction_gate,
            torch.zeros_like(head.last_correction_gate),
            atol=0.0,
            rtol=0.0,
        )

    def test_content_memory_is_sample_specific_and_trainable(self):
        head = PhaseErrorPeriodicMemoryHead(48, 12, 6)
        x = torch.randn(2, 48, 2)
        x[1, -6:] = 3.0 * x[1, -6:] + 1.0
        with torch.no_grad():
            head.correction_logits.fill_(0.2)
        output = head(x)
        self.assertFalse(torch.equal(head.last_attention[0], head.last_attention[1]))
        output.square().mean().backward()
        projection_grad = head.cycle_projection[0].weight.grad
        self.assertIsNotNone(projection_grad)
        self.assertGreater(float(projection_grad.abs().sum()), 0.0)

    def test_invalid_short_history_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "two full cycles"):
            PhaseErrorPeriodicMemoryHead(11, 6, 6)


class AdaptiveMultiPeriodResidualTests(unittest.TestCase):
    def test_warm_start_and_diagnostics(self):
        x = torch.randn(2, 48, 3)
        head = AdaptiveMultiPeriodResidualHead(48, 12, periods=(4, 6, 12))
        baseline = WeakPeriodResidualHead(48, 12)
        torch.testing.assert_close(head(x), baseline(x), atol=0.0, rtol=0.0)
        self.assertEqual(tuple(head.last_period_weights.shape), (2, 3, 3))
        torch.testing.assert_close(
            head.last_period_weights.sum(dim=-1),
            torch.ones(2, 3),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.isfinite(head.last_period_reliability).all())
        self.assertTrue((head.last_period_reliability >= 0).all())
        self.assertTrue((head.last_period_reliability <= 1).all())

    def test_router_prefers_matching_sinusoidal_period(self):
        steps = torch.arange(60, dtype=torch.float32)
        x = torch.sin(2.0 * math.pi * steps / 6.0).view(1, 60, 1)
        head = AdaptiveMultiPeriodResidualHead(
            60, 12, periods=(4, 6), routing_temperature=0.05
        )
        head(x)
        weights = head.last_period_weights[0, 0]
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_period_attention_is_phase_aligned(self):
        head = AdaptiveMultiPeriodResidualHead(48, 12, periods=(6,))
        attention = head.period_attention_0
        history = torch.arange(48)
        selected = attention[0] > 0
        self.assertTrue(((48 - history[selected]).remainder(6) == 0).all())
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(12))

    def test_period_router_and_correction_are_trainable(self):
        head = AdaptiveMultiPeriodResidualHead(48, 12, periods=(4, 6, 12))
        with torch.no_grad():
            head.correction_logits.fill_(0.2)
        output = head(torch.randn(2, 48, 3))
        output.square().mean().backward()
        self.assertGreater(float(head.period_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.linear.weight.grad.abs().sum()), 0.0)


class DualReliabilityFusionTests(unittest.TestCase):
    @staticmethod
    def _phase_series(error_cycles):
        cycles = torch.tensor(error_cycles, dtype=torch.float32)
        return cycles.transpose(0, 1).unsqueeze(0).unsqueeze(0)

    def test_repeating_error_cycles_receive_larger_periodic_gate(self):
        pattern = [1.0, -1.0, 1.0, -1.0]
        negative = [-value for value in pattern]
        high = self._phase_series((pattern, pattern, negative, negative))
        low = self._phase_series((pattern, negative, pattern, negative))
        phase_series = torch.cat((high, low), dim=0)
        fusion = DualReliabilityPeriodicFusion(6)
        reliability = fusion._periodic_reliability(phase_series)
        self.assertGreater(float(reliability[0, 0]), float(reliability[1, 0]))

        y_phase = torch.zeros(2, 6, 1)
        y_linear = torch.ones(2, 6, 1)
        y_periodic = 2.0 * torch.ones(2, 6, 1)
        output, _ = fusion(y_phase, y_linear, y_periodic, phase_series)
        self.assertEqual(tuple(output.shape), (2, 6, 1))
        self.assertGreater(
            float(fusion.last_periodic_gate[0].mean()),
            float(fusion.last_periodic_gate[1].mean()),
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_lff_component_refactor_preserves_original_blend(self):
        head = PeriodPositionEncodedResidualHead(
            48, 12, 6, encoding_type="lff", blend_init=0.1
        )
        x = torch.randn(2, 48, 3)
        y_linear, y_periodic = head.forward_components(x)
        output = head(x)
        beta = torch.sigmoid(head.blend_logits).view(1, 12, 1)
        torch.testing.assert_close(
            output, (1.0 - beta) * y_linear + beta * y_periodic
        )


class PeriodicComplementPresetTests(unittest.TestCase):
    def test_presets_inherit_the_frozen_rcrf_phase_stack(self):
        parent = build_hyperparams("ETTm2", 96, "gold_combo_reliability_s2")
        expected_heads = {
            "rcrf_phase_error_memory": "phase_error_memory",
            "rcrf_dual_reliability_lff": "periodic_pe",
            "rcrf_multiperiod": "adaptive_multiperiod",
        }
        for mode in PERIODIC_COMPLEMENT_MODES:
            with self.subTest(mode=mode):
                candidate = build_hyperparams("ETTm2", 96, mode)
                self.assertTrue(candidate["use_rcrf_fusion"])
                self.assertEqual(
                    candidate["weak_period_residual_head_type"], expected_heads[mode]
                )
                for key in (
                    "use_phase_uncertainty_shrinkage",
                    "phase_uncertainty_min",
                    "use_phase_period_level_calibration",
                    "use_phase_noise_hifreq_damping",
                    "rcrf_alpha_init",
                    "rcrf_sensitivity_init",
                    "rcrf_s_max",
                ):
                    self.assertEqual(candidate[key], parent[key])

    def test_all_six_datasets_build_both_horizon_presets(self):
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for mode in PERIODIC_COMPLEMENT_MODES:
                    with self.subTest(dataset=dataset, horizon=horizon, mode=mode):
                        candidate = build_hyperparams(dataset, horizon, mode)
                        self.assertEqual(candidate["scheme_name"], mode)
                        self.assertTrue(candidate["use_weak_period_residual"])

    def test_all_new_modes_run_full_phaseformer_at_both_horizons(self):
        x = torch.randn(1, 720, 7)
        for horizon in (96, 192):
            for mode in PERIODIC_COMPLEMENT_MODES:
                with self.subTest(horizon=horizon, mode=mode):
                    hyperparams = build_hyperparams("ETTm2", horizon, mode)
                    pl.seed_everything(2021, workers=True)
                    args = make_exp_args("ETTm2", 720, horizon, hyperparams)
                    config = PhaseFormerPresetConfig(args, 720, horizon, hyperparams)
                    model = PhaseFormer(config).eval()
                    with torch.no_grad():
                        output, _, _ = model(x, None, None, None)
                    self.assertEqual(tuple(output.shape), (1, horizon, 7))
                    self.assertTrue(torch.isfinite(output).all())

    def test_formal_runner_expands_to_288_model_runs(self):
        args = SimpleNamespace(
            datasets=",".join(DATASETS),
            horizons=",".join(map(str, HORIZONS)),
            seeds=",".join(map(str, SEEDS)),
            modes=",".join(MODES),
            num_workers=0,
            output_dir="research_runs/periodic_residual_next_stage_v1",
            resume=True,
            progress=False,
        )
        commands, run_count = build_commands(args)
        self.assertEqual(len(commands), 36)
        self.assertEqual(run_count, 288)
        self.assertTrue(all("--resume" in command for command in commands))

    def test_formal_summarizer_checks_and_aggregates_complete_matrix(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as output_dir:
            fieldnames = (
                "dataset", "horizon", "mode", "seed", "test_mse", "test_mae",
                "elapsed_sec",
            )
            for mode_index, mode in enumerate(("rcrf_pe_lff", "rcrf_multiperiod")):
                for seed_index, seed in enumerate(SEEDS):
                    run_dir = Path(output_dir) / f"run_{mode_index}_{seed}"
                    run_dir.mkdir()
                    with (run_dir / "metrics.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow(
                            {
                                "dataset": "ETTm2",
                                "horizon": 96,
                                "mode": mode,
                                "seed": seed,
                                "test_mse": 0.16 - 0.001 * mode_index + 1e-5 * seed_index,
                                "test_mae": 0.245 - 0.001 * mode_index + 1e-5 * seed_index,
                                "elapsed_sec": 1.0,
                            }
                        )
            args = SimpleNamespace(
                datasets="ETTm2",
                horizons="96",
                seeds=",".join(map(str, SEEDS)),
                modes="rcrf_pe_lff,rcrf_multiperiod",
                output_dir=output_dir,
            )
            self.assertEqual(summarize(args), 0)
            self.assertTrue((Path(output_dir) / "formal_summary.csv").exists())
            self.assertTrue((Path(output_dir) / "decision_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
