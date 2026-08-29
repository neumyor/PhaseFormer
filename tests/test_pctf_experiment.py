import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_pctf_experiment import (
    CANDIDATES,
    DATASETS,
    REFERENCES,
    SCREEN_MODES,
    formal_commands,
    screen_commands,
    summarize_formal,
    summarize_screen,
)


class PCTFExperimentProtocolTests(unittest.TestCase):
    @staticmethod
    def _args(root, **updates):
        values = dict(
            datasets=",".join(DATASETS),
            horizons="96,192",
            seeds="2021,2022,2023",
            output_root=str(root),
            num_workers=0,
            progress=False,
            champion=None,
        )
        values.update(updates)
        return SimpleNamespace(**values)

    @staticmethod
    def _write_screen_matrix(root, *, leak_test=False):
        serial = 0
        for dataset in DATASETS:
            for mode in SCREEN_MODES:
                if mode == "rcrf_pe_lff":
                    value = 1.0
                elif mode == "gold_combo_reliability_s2":
                    value = 1.01
                elif mode == "rcrf_icpt_none":
                    value = 1.02
                elif mode == "pctf_dual_fixed":
                    value = 0.99
                else:
                    value = 1.02
                row = {
                    "dataset": dataset,
                    "horizon": 96,
                    "mechanism": mode,
                    "seed": 2021,
                    "val_mse": value,
                    "val_mae": value,
                    "test_mse": "0.5" if leak_test and serial == 0 else "",
                    "test_mae": "0.5" if leak_test and serial == 0 else "",
                }
                run = root / "screen" / "runs" / f"run_{serial:03d}"
                run.mkdir(parents=True)
                with (run / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                serial += 1

    def test_dry_run_matrices_have_expected_size_and_test_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            screen = screen_commands(args)
            self.assertEqual(len(screen), 6 * 8)
            self.assertTrue(all("--evaluate-test" not in command for command in screen))

            args.champion = "pctf_dual_fixed"
            formal, champion = formal_commands(args, allow_override=True)
            self.assertEqual(champion, "pctf_dual_fixed")
            self.assertEqual(len(formal), 6 * 2 * 3 * 4)
            self.assertTrue(all("--evaluate-test" in command for command in formal))

    def test_screen_freezes_only_an_eligible_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen_matrix(root)
            summarize_screen(self._args(root))
            decision = json.loads((root / "screen_decision.json").read_text())
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["champion"], "pctf_dual_fixed")
            self.assertIn(decision["champion"], CANDIDATES)
            self.assertFalse(decision["test_metrics_read"])

    def test_validation_screen_rejects_any_test_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen_matrix(root, leak_test=True)
            with self.assertRaisesRegex(RuntimeError, "contains test metrics"):
                summarize_screen(self._args(root))

    def test_formal_summary_reads_only_the_frozen_champion_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen_matrix(root)
            summarize_screen(self._args(root))
            champion = "pctf_dual_fixed"
            serial = 0
            for mode in REFERENCES + (champion,):
                for seed in (2021, 2022, 2023):
                    value = 0.295 if mode == champion else 0.300
                    row = {
                        "dataset": "ETTh1",
                        "horizon": 96,
                        "mechanism": mode,
                        "seed": seed,
                        "test_mse": value + (seed - 2022) * 0.0001,
                        "test_mae": value + 0.05 + (seed - 2022) * 0.0001,
                    }
                    run = root / "formal" / "runs" / f"run_{serial:03d}"
                    run.mkdir(parents=True)
                    with (run / "metrics.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    serial += 1
            args = self._args(
                root, datasets="ETTh1", horizons="96", seeds="2021,2022,2023"
            )
            summarize_formal(args)
            decision = json.loads((root / "formal_decision.json").read_text())
            self.assertEqual(decision["champion"], champion)
            self.assertLess(decision["macro_ratio_vs_a2"], 1.0)
            with (root / "formal_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
