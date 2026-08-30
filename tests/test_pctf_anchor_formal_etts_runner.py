import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH = REPO_ROOT / "scripts" / "run_pctf_anchor_formal_etts.py"
SPEC = importlib.util.spec_from_file_location("pctf_anchor_formal_etts", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PctfAnchorFormalEttsRunnerTests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            output_root="research_runs/pctf_anchor_formal_etts_v1",
            seeds="2021,2022,2023",
            num_workers=0,
            progress=False,
        )

    def test_formal_matrix_has_twelve_matched_runs_per_model(self):
        args = self._args()
        anchors = RUNNER.anchor_commands(args)
        candidates = RUNNER.candidate_commands(args, dry=True)
        self.assertEqual(len(anchors), 12)
        self.assertEqual(len(candidates), 12)
        for command in anchors + candidates:
            self.assertIn("--require-cuda", command)
            self.assertIn("--evaluate-test", command)
            self.assertEqual(command[command.index("--percent") + 1], "100")
            self.assertEqual(command[command.index("--max-epochs") + 1], "30")
            self.assertEqual(command[command.index("--lookback") + 1], "720")
        for command in candidates:
            self.assertIn("--init-checkpoint", command)

    def test_period_is_shared_within_each_dataset(self):
        commands = RUNNER.candidate_commands(self._args(), dry=True)
        observed = {}
        for command in commands:
            dataset = command[command.index("--dataset") + 1]
            period = int(command[command.index("--cycle-period") + 1])
            observed.setdefault(dataset, set()).add(period)
        self.assertEqual(observed, {"ETTh2": {48}, "ETTm2": {96}})

    @staticmethod
    def _write_result(path, row):
        path.mkdir(parents=True)
        with (path / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (path / "environment.json").write_text(json.dumps({
            "cuda_available": True,
            "gpu": "synthetic-gpu",
            "torch": "test",
            "cuda_runtime": "test",
            "lightning": "test",
            "git_commit": "synthetic-commit",
        }))

    def test_synthetic_formal_summary_contains_golden_and_a2_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = 0
            for dataset, horizon in RUNNER.SETTINGS:
                golden_mse, golden_mae = RUNNER.GOLDEN[(dataset, horizon)]
                for seed in RUNNER.SEEDS:
                    common = {
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "val_mse": 0.2,
                        "val_mae": 0.3,
                        "epochs_completed": 10,
                        "elapsed_sec": 1.0,
                        "peak_memory_bytes": 1024,
                        "parameter_count": 100,
                        "trainable_parameter_count": 100,
                    }
                    self._write_result(
                        root / "anchors" / "runs" / f"a{serial}",
                        {
                            **common,
                            "mechanism": RUNNER.INCUMBENT,
                            "test_mse": golden_mse * 0.99,
                            "test_mae": golden_mae * 0.99,
                        },
                    )
                    self._write_result(
                        root / "candidates" / "runs" / f"c{serial}",
                        {
                            **common,
                            "mechanism": RUNNER.CANDIDATE,
                            "test_mse": golden_mse * 0.98,
                            "test_mae": golden_mae * 0.98,
                            "anchor_identity_max_abs": 0.0,
                        },
                    )
                    serial += 1
            args = self._args()
            args.output_root = str(root)
            self.assertEqual(RUNNER.summarize(args), 0)
            decision = json.loads((root / "formal_decision.json").read_text())
            self.assertTrue(decision["candidate_replaces_a2_on_etts"])
            self.assertEqual(decision["candidate_both_metric_improve_settings"], 4)
            with (root / "formal_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["golden_mse"] for row in rows))


if __name__ == "__main__":
    unittest.main()
