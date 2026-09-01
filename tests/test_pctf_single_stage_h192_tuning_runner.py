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
    def test_budget_is_fifty_policies_and_validation_only(self):
        args = SimpleNamespace(
            output_root="research_runs/synthetic", max_epochs=30,
            num_workers=0, progress=False,
        )
        self.assertEqual(len(RUNNER.POLICIES), 50)
        commands = RUNNER.commands(args)
        self.assertEqual(len(commands), 200)
        for command in commands:
            self.assertIn("--require-cuda", command)
            self.assertNotIn("--evaluate-test", command)
            self.assertEqual(command[command.index("--stage") + 1], "finalist")
            overrides = json.loads(command[command.index("--overrides") + 1])
            self.assertTrue(overrides["anchored_pctf_decouple_anchor_gradient"])
            self.assertTrue(overrides["anchored_pctf_detach_composer_inputs"])
            self.assertEqual(overrides["anchored_pctf_anchor_loss_weight"], 1.0)

    def test_smoke_covers_both_h192_period_geometries(self):
        args = SimpleNamespace(
            output_root="research_runs/synthetic", max_epochs=30,
            num_workers=0, progress=False,
        )
        commands = RUNNER.smoke_commands(args)
        self.assertEqual(len(commands), 4)
        self.assertEqual({
            command[command.index("--dataset") + 1] for command in commands
        }, {"ETTh2", "ETTm2"})
        self.assertTrue(all(
            command[command.index("--percent") + 1] == "30"
            and command[command.index("--max-epochs") + 1] == "1"
            and "--evaluate-test" not in command
            for command in commands
        ))
        self.assertTrue(all(
            "/smoke/" in command[command.index("--output-dir") + 1]
            for command in commands
        ))


if __name__ == "__main__":
    unittest.main()
