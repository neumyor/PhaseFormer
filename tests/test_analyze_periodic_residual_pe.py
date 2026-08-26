import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.analyze_periodic_residual_pe import (
    build_zip,
    case_features,
    choose_cases,
    select_nonoverlap,
)


class PeriodicResidualAnalysisTest(unittest.TestCase):
    def test_lag_feature_detects_periodic_signal(self):
        x = np.sin(2 * np.pi * np.arange(720) / 24)
        lag_corr, energy, drift, volatility = case_features(x)
        self.assertGreater(lag_corr, 0.99)
        self.assertGreater(energy, 0.95)
        self.assertLess(drift, 1e-6)
        self.assertGreater(volatility, 0.0)

    def test_nonoverlap_selection_respects_channel_gap(self):
        score = np.arange(40, dtype=float).reshape(20, 2)
        selected = select_nonoverlap(score, score.shape, top_k=4, descending=True, min_gap=5)
        for i, (sid, channel) in enumerate(selected):
            for prior_sid, prior_channel in selected[:i]:
                if channel == prior_channel:
                    self.assertGreaterEqual(abs(sid - prior_sid), 5)

    def test_choose_cases_uses_mae_directions(self):
        base_mae = np.array([[3.0], [2.0], [1.0]])
        cand_mae = np.array([[2.0], [4.0], [0.9]])
        base = {"mae": base_mae}
        cand = {"mae": cand_mae}
        selected = choose_cases(base, cand, horizon=1, top_k=1)
        self.assertEqual(selected["baseline_high_error"], [(0, 0)])
        self.assertEqual(selected["candidate_regression"], [(1, 0)])
        self.assertEqual(selected["candidate_improvement"], [(0, 0)])

    def test_zip_contains_only_referenced_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figures = root / "figures"
            figures.mkdir()
            used = figures / "used.png"
            unused = figures / "unused.png"
            used.write_bytes(b"used")
            unused.write_bytes(b"unused")
            report = root / "objective_error_analysis.md"
            report.write_text("![图](figures/used.png)\n", encoding="utf-8")
            archive = root / "objective_error_analysis.zip"
            refs = build_zip(report, [used], archive)
            self.assertEqual(refs, ["figures/used.png"])
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(
                    set(handle.namelist()),
                    {"objective_error_analysis.md", "figures/used.png"},
                )


if __name__ == "__main__":
    unittest.main()
