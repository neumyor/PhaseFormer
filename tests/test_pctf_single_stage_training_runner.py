import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH = REPO_ROOT / "scripts" / "run_pctf_single_stage_training.py"
SPEC = importlib.util.spec_from_file_location("pctf_single_stage", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PctfSingleStageTrainingRunnerTests(unittest.TestCase):
    @staticmethod
    def _args(root=RUNNER.DEFAULT_OUTPUT_ROOT):
        return SimpleNamespace(
            output_root=str(root), screen_seeds="2021,2022",
            formal_seeds="2021,2022,2023", num_workers=0, progress=False,
            policies="",
        )

    def test_screen_is_one_stage_validation_only_and_complete(self):
        args = self._args()
        baselines = RUNNER.screen_baseline_commands(args)
        candidates = RUNNER.screen_candidate_commands(args)
        self.assertEqual(len(baselines), 8)
        self.assertEqual(len(candidates), 56)
        for command in baselines + candidates:
            self.assertNotIn("--init-checkpoint", command)
            self.assertNotIn("--evaluate-test", command)
            self.assertIn("--require-cuda", command)
            self.assertEqual(command[command.index("--percent") + 1], "100")
            self.assertEqual(command[command.index("--max-epochs") + 1], "30")
        override_values = {
            command[command.index("--overrides") + 1]
            for command in candidates
        }
        self.assertEqual(len(override_values), len(RUNNER.POLICIES))

        args.policies = "decoupled_protected"
        followup = RUNNER.screen_candidate_commands(args)
        self.assertEqual(len(followup), 8)
        self.assertTrue(all(
            json.loads(command[command.index("--overrides") + 1])[
                "anchored_pctf_decouple_anchor_gradient"
            ]
            for command in followup
        ))

    def test_unknown_policy_is_rejected(self):
        args = self._args()
        args.policies = "not_a_policy"
        with self.assertRaisesRegex(ValueError, "unknown policies"):
            RUNNER.screen_candidate_commands(args)

    @staticmethod
    def _write_result(path, row):
        path.mkdir(parents=True)
        with (path / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (path / "environment.json").write_text(json.dumps({
            "cuda_available": True, "gpu": "synthetic", "torch": "test",
            "cuda_runtime": "test", "lightning": "test",
            "git_commit": "synthetic",
        }))

    def test_screen_summary_freezes_only_an_eligible_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = 0
            for dataset, horizon in RUNNER.SETTINGS:
                for seed in RUNNER.SCREEN_SEEDS:
                    common = {
                        "dataset": dataset, "horizon": horizon, "seed": seed,
                        "val_mse": 0.2, "val_mae": 0.3,
                        "test_mse": "", "test_mae": "", "elapsed_sec": 1,
                        "epochs_completed": 10, "final_correction_scale": 1,
                    }
                    self._write_result(
                        root / "screen" / "baselines" / "runs" / f"a{serial}",
                        {**common, "mechanism": RUNNER.INCUMBENT},
                    )
                    for index, policy in enumerate(RUNNER.POLICIES):
                        ratio = 0.99 if policy == "warm5_mild" else 1.01 + index / 1000
                        self._write_result(
                            root / "screen" / "candidates" / policy
                            / "runs" / f"c{serial}",
                            {
                                **common, "mechanism": RUNNER.CANDIDATE,
                                "val_mse": 0.2 * ratio,
                                "val_mae": 0.3 * ratio,
                                "val_anchor_mse": 0.2,
                                "val_anchor_mae": 0.3,
                                "val_mse_ratio_vs_internal_anchor": ratio,
                                "val_mae_ratio_vs_internal_anchor": ratio,
                                "val_update_rms": 0.01,
                                "val_coefficient_regret_corr": 0.2,
                            },
                        )
                    serial += 1
            args = self._args(root)
            self.assertEqual(RUNNER.summarize_screen(args), 0)
            decision = json.loads((root / "screen_decision.json").read_text())
            self.assertFalse(decision["test_metrics_read"])
            self.assertEqual(decision["winner"], "warm5_mild")
            commands = RUNNER.formal_candidate_commands(args)
            self.assertEqual(len(commands), 12)
            self.assertTrue(all("--evaluate-test" in command for command in commands))
            self.assertTrue(all("--init-checkpoint" not in command for command in commands))


if __name__ == "__main__":
    unittest.main()
