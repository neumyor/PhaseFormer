import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH = REPO_ROOT / "scripts" / "run_pctf_single_stage_h192_tuning.py"
SPEC = importlib.util.spec_from_file_location("pctf_h192_tuning", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PctfSingleStageH192TuningRunnerTests(unittest.TestCase):
    def test_budget_is_ten_policies_and_validation_only(self):
        args = SimpleNamespace(
            output_root="research_runs/synthetic", max_epochs=30,
            num_workers=0, progress=False,
        )
        self.assertEqual(len(RUNNER.POLICIES), 10)
        commands = RUNNER.commands(args)
        self.assertEqual(len(commands), 40)
        for command in commands:
            self.assertIn("--require-cuda", command)
            self.assertNotIn("--evaluate-test", command)
            overrides = json.loads(command[command.index("--overrides") + 1])
            self.assertTrue(overrides["anchored_pctf_decouple_anchor_gradient"])
            self.assertTrue(overrides["anchored_pctf_detach_composer_inputs"])
            self.assertEqual(overrides["anchored_pctf_anchor_loss_weight"], 1.0)

    def test_long_budget_is_the_only_extended_epoch_policy(self):
        args = SimpleNamespace(
            output_root="research_runs/synthetic", max_epochs=30,
            num_workers=0, progress=False,
        )
        commands = RUNNER.commands(args)
        long_runs = [
            command for command in commands
            if "T9_long_budget" in command[command.index("--output-dir") + 1]
        ]
        self.assertEqual(len(long_runs), 4)
        self.assertTrue(all(
            command[command.index("--max-epochs") + 1] == "45"
            for command in long_runs
        ))


if __name__ == "__main__":
    unittest.main()
