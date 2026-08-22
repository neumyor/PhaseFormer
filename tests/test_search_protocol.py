import unittest
from types import SimpleNamespace

from scripts.search_phaseformer import MECHANISMS, apply_compact, build_spec, run_id


class SearchProtocolTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(
            dataset="ETTh1", horizon=96, stage="period_screen", mechanism="original",
            period=12, lookback=720, percent=30, max_epochs=8, seed=2021,
            loss="huber", lr_multiplier=1.0, learning_rate=None, capacity="base",
            batch_size=None, evaluate_test=False, overrides="{}",
        )
        values.update(updates)
        return SimpleNamespace(**values)

    def test_run_id_is_deterministic(self):
        first = build_spec(self.args())
        second = build_spec(self.args())
        self.assertEqual(run_id(first), run_id(second))
        self.assertEqual(first["config_hash"], second["config_hash"])

    def test_plan_defines_expected_mechanisms(self):
        self.assertEqual(len(MECHANISMS), 23)
        self.assertIn("phase_align", MECHANISMS)
        self.assertIn("phase_warp", MECHANISMS)
        self.assertIn("phase_amp_calib", MECHANISMS)
        self.assertIn("phase_rape", MECHANISMS)
        # Dynamic-phase mechanisms (weak-residual-phaseformer plan stages 1-5).
        self.assertIn("residual_full", MECHANISMS)
        self.assertIn("no_residual", MECHANISMS)
        self.assertIn("phase_correction", MECHANISMS)
        self.assertIn("circular_geometry", MECHANISMS)
        self.assertIn("phase_rotation", MECHANISMS)
        self.assertIn("harmonic_modulation", MECHANISMS)
        self.assertIn("dyn_stack", MECHANISMS)

    def test_compact_latent_remains_head_divisible(self):
        hp = {"latent_dim": 10, "phase_attn_heads": 2, "phase_encoder_hidden": 3, "predictor_hidden": 5}
        apply_compact(hp)
        self.assertGreaterEqual(hp["latent_dim"], hp["phase_attn_heads"])
        self.assertEqual(hp["latent_dim"] % hp["phase_attn_heads"], 0)
        self.assertEqual(hp["phase_encoder_hidden"], 1)
        self.assertEqual(hp["predictor_hidden"], 2)

    def test_planned_fixed_batch_sizes(self):
        self.assertEqual(build_spec(self.args(dataset="Weather"))["batch_size"], 64)
        self.assertEqual(build_spec(self.args(dataset="Electricity"))["batch_size"], 64)
        self.assertEqual(build_spec(self.args(dataset="Traffic"))["batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
