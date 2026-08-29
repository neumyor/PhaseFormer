import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_pctf_anchor_fusion_retest import (
    ANCHORED_MODES,
    DATASETS,
    HORIZONS,
    INCUMBENT,
    PAPER_CANDIDATES,
    PERIOD_CANDIDATES,
    PERIOD_PROBE,
    REFERENCES,
    SCREEN_MODES,
    formal_commands,
    period_commands,
    screen_commands,
    summarize_formal,
    summarize_period,
    summarize_screen,
)


class PCTFAnchorFusionRetestTests(unittest.TestCase):
    @staticmethod
    def _args(root, **updates):
        values = dict(
            datasets=",".join(DATASETS),
            horizons=",".join(map(str, HORIZONS)),
            seeds="2021,2022,2023",
            output_root=str(root),
            num_workers=0,
            progress=False,
            period_map=None,
            champion=None,
        )
        values.update(updates)
        return SimpleNamespace(**values)

    @staticmethod
    def _environment(cuda=True, gpu="NVIDIA GeForce RTX 4090"):
        return {
            "cuda_available": cuda,
            "gpu": gpu if cuda else None,
            "torch": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "lightning": "2.1.0",
        }

    @staticmethod
    def _write_run(root, serial, row, environment=None):
        run = root / "runs" / f"run_{serial:04d}"
        run.mkdir(parents=True)
        with (run / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (run / "environment.json").write_text(json.dumps(
            environment or PCTFAnchorFusionRetestTests._environment()
        ))

    def _write_period_matrix(self, root, *, cuda=True, mixed_gpu=False):
        serial = 0
        for dataset in DATASETS:
            preferred = PERIOD_CANDIDATES[dataset][1]
            for horizon in HORIZONS:
                common = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": 2021,
                    "val_mse": 1.0,
                    "val_mae": 1.0,
                    "test_mse": "",
                    "test_mae": "",
                    "anchor_identity_max_abs": "",
                }
                self._write_run(
                    root, serial,
                    dict(common, mechanism=INCUMBENT, cycle_period=""),
                    self._environment(
                        cuda=cuda,
                        gpu="different GPU" if mixed_gpu and serial == 0
                        else "NVIDIA GeForce RTX 4090",
                    ),
                )
                serial += 1
                for period in PERIOD_CANDIDATES[dataset]:
                    value = 0.99 if period == preferred else 1.01
                    self._write_run(
                        root, serial,
                        dict(
                            common,
                            mechanism=PERIOD_PROBE,
                            cycle_period=period,
                            val_mse=value,
                            val_mae=value,
                            anchor_identity_max_abs=0.0,
                        ),
                        self._environment(cuda=cuda),
                    )
                    serial += 1

    def _write_screen_matrix(
        self, root, periods, *, leak=False, anchor_error=False
    ):
        serial = 0
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for mode in SCREEN_MODES:
                    if mode == "pctf_anchor_component_cycle":
                        value = 0.99
                    elif mode == INCUMBENT:
                        value = 1.0
                    elif mode in REFERENCES:
                        value = 1.01
                    else:
                        value = 1.02
                    row = {
                        "dataset": dataset,
                        "horizon": horizon,
                        "mechanism": mode,
                        "seed": 2021,
                        "cycle_period": periods[dataset]
                        if mode in ANCHORED_MODES else "",
                        "val_mse": value,
                        "val_mae": value,
                        "test_mse": "0.5" if leak and serial == 0 else "",
                        "test_mae": "0.5" if leak and serial == 0 else "",
                        "anchor_identity_max_abs": (
                            1e-6 if anchor_error and mode in ANCHORED_MODES
                            and serial == len(REFERENCES) + len(("pctf_dual_fixed",))
                            else 0.0
                        ) if mode in ANCHORED_MODES else "",
                    }
                    self._write_run(root, serial, row)
                    serial += 1

    def test_dry_run_counts_boundaries_and_cuda_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            period = period_commands(args)
            self.assertEqual(len(period), 6 * 2 * 4)
            self.assertTrue(all("--require-cuda" in item for item in period))
            self.assertTrue(all("--evaluate-test" not in item for item in period))

            screen = screen_commands(args, dry_override=True)
            self.assertEqual(len(screen), 6 * 2 * len(SCREEN_MODES))
            self.assertTrue(all("--require-cuda" in item for item in screen))
            self.assertTrue(all("--evaluate-test" not in item for item in screen))

            args.champion = "pctf_anchor_component_cycle"
            formal, champion = formal_commands(args, dry_override=True)
            self.assertEqual(champion, args.champion)
            self.assertEqual(len(formal), 6 * 2 * 3 * 4)
            self.assertTrue(all("--require-cuda" in item for item in formal))
            self.assertTrue(all("--evaluate-test" in item for item in formal))

    def test_period_summary_freezes_one_shared_period_per_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_period_matrix(root / "period")
            summarize_period(self._args(root))
            decision = json.loads((root / "period_decision.json").read_text())
            expected = {
                dataset: PERIOD_CANDIDATES[dataset][1] for dataset in DATASETS
            }
            self.assertEqual(decision["selected_periods"], expected)
            self.assertFalse(decision["test_metrics_read"])

    def test_strategy_summary_selects_only_an_eligible_paper_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            periods = {
                dataset: PERIOD_CANDIDATES[dataset][1] for dataset in DATASETS
            }
            (root / "period_decision.json").write_text(json.dumps({
                "selected_periods": periods
            }))
            self._write_screen_matrix(root / "screen", periods)
            summarize_screen(self._args(root))
            decision = json.loads((root / "screen_decision.json").read_text())
            self.assertTrue(decision["passed"])
            self.assertEqual(
                decision["champion"], "pctf_anchor_component_cycle"
            )
            self.assertIn(decision["champion"], PAPER_CANDIDATES)
            self.assertFalse(decision["ablations_eligible"])
            self.assertFalse(decision["test_metrics_read"])

    def test_selection_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            periods = {dataset: 24 for dataset in DATASETS}
            (root / "period_decision.json").write_text(json.dumps({
                "selected_periods": periods
            }))
            self._write_screen_matrix(root / "screen", periods, leak=True)
            with self.assertRaisesRegex(RuntimeError, "contains test metrics"):
                summarize_screen(self._args(root))

    def test_selection_rejects_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_period_matrix(root / "period", cuda=False)
            with self.assertRaisesRegex(RuntimeError, "non-CUDA"):
                summarize_period(self._args(root))

    def test_selection_rejects_heterogeneous_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_period_matrix(root / "period", mixed_gpu=True)
            with self.assertRaisesRegex(RuntimeError, "heterogeneous"):
                summarize_period(self._args(root))

    def test_selection_rejects_nonzero_anchor_identity_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            periods = {dataset: 24 for dataset in DATASETS}
            (root / "period_decision.json").write_text(json.dumps({
                "selected_periods": periods
            }))
            self._write_screen_matrix(
                root / "screen", periods, anchor_error=True
            )
            with self.assertRaisesRegex(RuntimeError, "not exact A2"):
                summarize_screen(self._args(root))

    def test_formal_summary_uses_only_the_frozen_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            periods = {dataset: 24 for dataset in DATASETS}
            champion = "pctf_anchor_component_cycle"
            (root / "period_decision.json").write_text(json.dumps({
                "selected_periods": periods
            }))
            (root / "screen_decision.json").write_text(json.dumps({
                "passed": True, "champion": champion
            }))
            serial = 0
            for mode in REFERENCES + (champion,):
                for seed in (2021, 2022, 2023):
                    value = 0.35 if mode == champion else 0.36
                    row = {
                        "dataset": "ETTh1",
                        "horizon": 96,
                        "mechanism": mode,
                        "seed": seed,
                        "cycle_period": 24 if mode == champion else "",
                        "test_mse": value + (seed - 2022) * 0.0001,
                        "test_mae": value + 0.03 + (seed - 2022) * 0.0001,
                        "anchor_identity_max_abs": 0.0
                        if mode == champion else "",
                    }
                    self._write_run(root / "formal", serial, row)
                    serial += 1
            args = self._args(
                root, datasets="ETTh1", horizons="96",
                seeds="2021,2022,2023",
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
