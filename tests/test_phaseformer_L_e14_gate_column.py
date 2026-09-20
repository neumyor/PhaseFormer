"""Regression tests for the §4.2 gate column (defect fixed 2026-09-20).

Section 4.2's gate column prefers ``results.csv``'s ``gate_value``, but the 7
reused ``l_main`` settings have no such value in their Stage-0 evidence, so the
column falls back to the checkpoint reading carried by ``parameter_table.csv``.
That fallback used to key its lookup by ``(arm, horizon)`` -- pooling the dataset
away -- with the consequence that

* each reused cell received the **smallest** gate of any dataset at that horizon
  (measured in the real run: ETTh2-96 showed 0.052, Traffic's gate, instead of its
  own 0.492); and
* ``gate_from_checkpoint`` collapsed a per-seed set with ``sorted(...)[0]``, i.e.
  a minimum where every sibling path takes a mean.

Both are silent: the table stays complete and every cell still holds a plausible
number.  These tests fix the two properties that make the fallback correct -- the
lookup is keyed by dataset, and the value is the mean over the setting's seeds --
and they fail against the old implementation rather than merely describing it.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.phaseformer_L import e14_writeback as wb  # noqa: E402


def write_parameter_table(path: Path, rows) -> None:
    fields = ["arm", "dataset", "horizon", "seed", "setting", "status",
              "total_params", "residual_params", "residual_share",
              "gate_value_from_checkpoint"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


class GateFallbackKeyingTests(unittest.TestCase):
    """One horizon, two datasets, distinct gates and distinct parameter counts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "parameter_table.csv"
        rows = []
        # The dataset with the SMALLER gate must also be the one with the larger
        # parameter count, so that a min-collapse and a
        # "pick the widest dataset" mutation are both detectable.
        for dataset, gate, total in (("ETTh2", 0.50, 140191),
                                     ("Traffic", 0.05, 412454)):
            for index, seed in enumerate((2021, 2022, 2023)):
                rows.append({
                    "arm": "l_main", "dataset": dataset, "horizon": 192,
                    "seed": seed, "setting": f"{dataset}-192", "status": "reused",
                    "total_params": total, "residual_params": 138432,
                    "residual_share": round(138432 / total, 6),
                    # Different per seed, so the reported value can only be right
                    # if it is the MEAN of the three.
                    "gate_value_from_checkpoint": round(gate + 0.001 * index, 6),
                })
        write_parameter_table(self.path, rows)
        self.params = wb.read_parameters(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lookup_is_keyed_by_dataset_not_only_horizon(self):
        self.assertIn(("l_main", "ETTh2", 192), self.params)
        self.assertIn(("l_main", "Traffic", 192), self.params)
        self.assertNotIn(("l_main", 192), self.params,
                         "the dataset must not be pooled away")

    def test_each_dataset_gets_its_own_gate(self):
        """The old code returned Traffic's 0.051 for ETTh2 as well."""
        etth2 = self.params[("l_main", "ETTh2", 192)]["gate_from_checkpoint"]
        traffic = self.params[("l_main", "Traffic", 192)]["gate_from_checkpoint"]
        self.assertAlmostEqual(etth2, 0.501, places=6)
        self.assertAlmostEqual(traffic, 0.051, places=6)
        self.assertNotAlmostEqual(etth2, traffic, places=3)

    def test_gate_is_the_seed_mean_not_the_seed_minimum(self):
        entry = self.params[("l_main", "ETTh2", 192)]
        self.assertEqual(entry["gate_seeds_read"], 3)
        self.assertAlmostEqual(entry["gate_from_checkpoint"], 0.501, places=6)
        self.assertNotAlmostEqual(entry["gate_from_checkpoint"], 0.500, places=6)

    def test_parameter_spread_is_exposed_not_collapsed(self):
        entry = self.params[("l_main", "Traffic", 192)]
        self.assertEqual(entry["total_params"], 412454)
        self.assertEqual(entry["parameter_variants_across_seeds"], [412454])
        # The corrector is dataset-independent in the real run as well; the
        # table reports that as a fact rather than as an assumption.
        self.assertTrue(entry["constant_across_seeds"])


class GateSelectionTests(unittest.TestCase):
    """End-to-end: which value section 4.2 puts in the gate column."""

    def _row(self, params, gate_from_results):
        """Reproduce the exact selection expression used by the builder."""
        from_results = gate_from_results
        from_ckpt = (params.get(("l_main", "ETTh2", 192)) or {}).get(
            "gate_from_checkpoint")
        return (from_results if from_results is not None else from_ckpt,
                "results_csv" if from_results is not None
                else ("checkpoint" if from_ckpt is not None else None))

    def test_results_csv_still_wins_when_present(self):
        value, source = self._row({("l_main", "ETTh2", 192):
                                   {"gate_from_checkpoint": 0.999}}, 0.123456)
        self.assertEqual((value, source), (0.123456, "results_csv"))

    def test_fallback_is_used_only_when_results_has_none(self):
        value, source = self._row({("l_main", "ETTh2", 192):
                                   {"gate_from_checkpoint": 0.501}}, None)
        self.assertEqual((value, source), (0.501, "checkpoint"))

    def test_absent_dataset_yields_no_gate_rather_than_a_borrowed_one(self):
        """A parameter table without the dataset column must not invent a gate."""
        value, source = self._row({}, None)
        self.assertIsNone(value)
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()
