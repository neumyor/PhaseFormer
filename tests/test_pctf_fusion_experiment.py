import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_pctf_fusion_strategies import (
    DATASETS,
    NEGATIVE_CONTROLS,
    PAPER_CANDIDATES,
    REFERENCES,
    SCREEN_MODES,
    formal_commands,
    screen_commands,
    summarize_formal,
    summarize_screen,
)


class PCTFFusionExperimentTests(unittest.TestCase):
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
    def _write_screen(root, leak=False):
        serial = 0
        for dataset in DATASETS:
            for mode in SCREEN_MODES:
                if mode == "rcrf_pe_lff":
                    value = 1.0
                elif mode in REFERENCES:
                    value = 1.01
                elif mode in NEGATIVE_CONTROLS:
                    value = 1.005
                elif mode == "pctf_fusion_component_cycle":
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
                    "test_mse": "0.5" if leak and serial == 0 else "",
                    "test_mae": "0.5" if leak and serial == 0 else "",
                }
                run = root / "screen" / "runs" / f"run_{serial:03d}"
                run.mkdir(parents=True)
                with (run / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                serial += 1

    def test_dry_run_counts_and_test_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            screen = screen_commands(args)
            self.assertEqual(len(screen), 6 * 11)
            self.assertTrue(all("--evaluate-test" not in item for item in screen))
            args.champion = "pctf_fusion_component_cycle"
            formal, champion = formal_commands(args, dry_override=True)
            self.assertEqual(champion, args.champion)
            self.assertEqual(len(formal), 6 * 2 * 3 * 4)
            self.assertTrue(all("--evaluate-test" in item for item in formal))

    def test_screen_freezes_paper_candidate_but_never_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen(root)
            summarize_screen(self._args(root))
            decision = json.loads((root / "screen_decision.json").read_text())
            self.assertTrue(decision["passed"])
            self.assertEqual(
                decision["champion"], "pctf_fusion_component_cycle"
            )
            self.assertIn(decision["champion"], PAPER_CANDIDATES)
            self.assertNotIn(decision["champion"], NEGATIVE_CONTROLS)
            self.assertFalse(decision["negative_controls_eligible"])
            self.assertFalse(decision["test_metrics_read"])

    def test_screen_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen(root, leak=True)
            with self.assertRaisesRegex(RuntimeError, "contains test metrics"):
                summarize_screen(self._args(root))

    def test_formal_summary_uses_only_frozen_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_screen(root)
            summarize_screen(self._args(root))
            champion = "pctf_fusion_component_cycle"
            serial = 0
            for mode in REFERENCES + (champion,):
                for seed in (2021, 2022, 2023):
                    value = 0.35 if mode == champion else 0.36
                    row = {
                        "dataset": "ETTh1",
                        "horizon": 96,
                        "mechanism": mode,
                        "seed": seed,
                        "test_mse": value + (seed - 2022) * 0.0001,
                        "test_mae": value + 0.03 + (seed - 2022) * 0.0001,
                    }
                    run = root / "formal" / "runs" / f"run_{serial:03d}"
                    run.mkdir(parents=True)
                    with (run / "metrics.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    serial += 1
            summarize_formal(self._args(
                root, datasets="ETTh1", horizons="96", seeds="2021,2022,2023"
            ))
            decision = json.loads((root / "formal_decision.json").read_text())
            self.assertEqual(decision["champion"], champion)
            self.assertLess(decision["macro_ratio_vs_a2"], 1.0)
            with (root / "formal_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
