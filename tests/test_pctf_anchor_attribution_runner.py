import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pctf_anchor_attribution.py"
SPEC = importlib.util.spec_from_file_location("pctf_anchor_attribution", SCRIPT_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PctfAnchorAttributionRunnerTests(unittest.TestCase):
    @staticmethod
    def _args(seeds="2021,2022"):
        return SimpleNamespace(
            output_root="research_runs/pctf_anchor_attribution_v3",
            seeds=seeds,
            num_workers=0,
            progress=False,
        )

    def test_dry_run_matrix_is_complete_and_validation_only(self):
        args = self._args()
        anchors = RUNNER.anchor_commands(args)
        candidates = RUNNER.candidate_commands(args, dry=True)
        self.assertEqual(len(anchors), 12)
        self.assertEqual(len(candidates), 72)
        for command in anchors + candidates:
            self.assertIn("--require-cuda", command)
            self.assertNotIn("--evaluate-test", command)
            self.assertEqual(command[command.index("--percent") + 1], "30")
            self.assertEqual(command[command.index("--max-epochs") + 1], "12")
        for command in candidates:
            self.assertIn("--init-checkpoint", command)

    def test_dry_candidates_cover_each_repair_for_each_setting_seed(self):
        args = self._args(seeds="2021")
        commands = RUNNER.candidate_commands(args, dry=True)
        observed = set()
        for command in commands:
            observed.add(command[command.index("--mechanism") + 1])
        self.assertEqual(observed, set(RUNNER.CANDIDATES))
        self.assertEqual(
            len(commands), len(RUNNER.SETTINGS) * len(RUNNER.CANDIDATES)
        )

    @staticmethod
    def _write_row(path, row):
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

    def _synthetic_matrix(self, root, leak_test=False):
        ratios = {
            RUNNER.CURRENT_CONTROL: 1.010,
            "pctf_anchor_diag_frozen_absolute": 0.999,
            "pctf_anchor_diag_frozen_residual": 0.998,
            "pctf_anchor_repair_joint_residual": 0.997,
            "pctf_anchor_repair_joint_marginal": 0.996,
            "pctf_anchor_repair_full": 0.995,
        }
        serial = 0
        for dataset, horizon in RUNNER.SETTINGS:
            anchor_row = {
                "dataset": dataset,
                "horizon": horizon,
                "seed": 2021,
                "mechanism": RUNNER.INCUMBENT,
                "val_mse": 0.1,
                "val_mae": 0.2,
                "test_mse": 0.1 if leak_test and serial == 0 else "",
                "test_mae": "",
                "checkpoint": "unused.ckpt",
            }
            self._write_row(
                root / "anchors" / "runs" / f"a{serial}", anchor_row
            )
            for mechanism, ratio in ratios.items():
                candidate_row = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": 2021,
                    "mechanism": mechanism,
                    "val_mse": 0.1 * ratio,
                    "val_mae": 0.2 * ratio,
                    "test_mse": "",
                    "test_mae": "",
                    "anchor_identity_max_abs": 0.0,
                    "anchor_frozen": mechanism in RUNNER.FROZEN_MODES,
                    "val_anchor_mse": 0.1,
                    "val_anchor_mae": 0.2,
                    "val_mse_ratio_vs_internal_anchor": ratio,
                    "val_mae_ratio_vs_internal_anchor": ratio,
                    "val_update_rms": 0.01,
                    "val_confidence_regret_corr": 0.1,
                    "val_coefficient_regret_corr": (
                        0.3
                        if mechanism in (
                            "pctf_anchor_repair_joint_marginal",
                            "pctf_anchor_repair_full",
                        )
                        else 0.2
                    ),
                }
                self._write_row(
                    root / "candidates" / "runs" / f"c{serial}_{mechanism}",
                    candidate_row,
                )
            serial += 1

    def test_summarizer_computes_registered_hypotheses_without_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._synthetic_matrix(root)
            args = self._args(seeds="2021")
            args.output_root = str(root)
            self.assertEqual(RUNNER.summarize(args), 0)
            decision = json.loads(
                (root / "attribution_decision.json").read_text()
            )
            self.assertFalse(decision["test_metrics_read"])
            self.assertTrue(all(decision["hypotheses"].values()))
            with (root / "attribution_aggregates.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(RUNNER.CANDIDATES))

    def test_summarizer_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._synthetic_matrix(root, leak_test=True)
            args = self._args(seeds="2021")
            args.output_root = str(root)
            with self.assertRaisesRegex(RuntimeError, "test leakage"):
                RUNNER.summarize(args)


if __name__ == "__main__":
    unittest.main()
