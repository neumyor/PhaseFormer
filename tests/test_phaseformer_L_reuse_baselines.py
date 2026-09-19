"""Unit tests for E18's E14 baseline provenance index (minipaper §4.6 rows 1/5).

E14's stage-A manifest records two kinds of cell, and ``load_baseline_index``
must admit both without trusting either on faith:

* ``new`` cells were launched by stage A and carry their argv, so the ``l_main``
  fingerprint is re-derived from ``--overrides``;
* ``reused`` cells were adopted from an earlier registered root and carry
  ``"command": null`` (all 81 of them on the live manifest), so the fingerprint
  is re-derived from the adopted run's own ``config.json``.

The second path exists because the 7 row-1 settings of §4.6 *are* exactly the
reused ones: without it every one of the 78 E18 rows carried empty baseline
provenance.  These tests need no dataset and no GPU -- they build manifests and
run directories in a temporary directory.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phaseformer_L.e14_main_matrix import (
    NEW_CELL_GATE_INIT,
    NEW_CELL_LR,
)
from scripts.phaseformer_L.e18_negative import (
    MECHANISM,
    load_baseline_index,
)

DATASET = "ETTh2"
HORIZON = 96
SEED = 2021


def l_main_hyperparams(**extra) -> dict:
    """Hyperparameters of an admitted ``l_main`` cell (arm table: shared head)."""
    hyper = {
        "weak_period_residual_head_type": "shared",
        "weak_period_residual_gate_init": NEW_CELL_GATE_INIT,
        "learning_rate": NEW_CELL_LR,
    }
    hyper.update(extra)
    return hyper


def write_run(root: Path, run_id: str, hyperparams: dict,
              horizon: int = HORIZON) -> Path:
    """Create ``<root>/runs/<run_id>/config.json`` and return the run dir."""
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mechanism": MECHANISM,
        "dataset": DATASET,
        "horizon": horizon,
        "seed": SEED,
        "hyperparams": hyperparams,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_dir


def reused_cell(run_dir, root, status="reused", arm="l_main") -> dict:
    return {
        "key": f"{arm}__{DATASET}-{HORIZON}-s{SEED}",
        "arm": arm,
        "dataset": DATASET,
        "horizon": HORIZON,
        "seed": SEED,
        "status": status,
        "command": None,
        "source": {"run_dir": str(run_dir), "root": str(root),
                   "config_hash": "deadbeef"},
    }


def new_cell(overrides: dict, argv_extra=None, seed=SEED) -> dict:
    argv = ["--output-dir", "/tmp/a-template-root",
            "--dataset", DATASET, "--horizon", str(HORIZON),
            "--overrides", json.dumps(overrides, sort_keys=True)]
    return {
        "key": f"l_main__{DATASET}-{HORIZON}-s{seed}",
        "arm": "l_main",
        "dataset": DATASET,
        "horizon": HORIZON,
        "seed": seed,
        "status": "new",
        "command": argv + list(argv_extra or []),
        "source": None,
    }


def write_manifest(directory: Path, cells: list) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage_a_manifest.json"
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


class BaselineIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_reused_cell_is_admitted_from_its_own_config(self):
        """A reused cell has no argv, so its run's config.json is the evidence."""
        run_dir = write_run(self.tmp / "adopted", "run_a",
                            l_main_hyperparams(
                                weak_period_residual_gate_init=0.5,
                                learning_rate=3e-4))
        manifest = write_manifest(
            self.tmp / "e14", [reused_cell(run_dir, self.tmp / "adopted")])
        index, report = load_baseline_index(manifest)

        self.assertEqual(report["rejected"], [])
        self.assertEqual(report["resolved"], 1)
        self.assertEqual(report["resolved_reused"], 1)
        self.assertEqual(report["resolved_new"], 0)
        entry = index[(DATASET, HORIZON, SEED)]
        # Gate and learning rate come from the run that was actually adopted --
        # the D-2 disclosure that a reused baseline keeps its own frozen values.
        self.assertEqual(entry["gate_init"], 0.5)
        self.assertEqual(entry["learning_rate"], 3e-4)
        self.assertEqual(entry["run_dir"], str(run_dir))
        self.assertEqual(entry["eval_root"], str(self.tmp / "adopted"))
        self.assertEqual(entry["status"], "reused")

    def test_reused_cell_whose_run_is_not_l_main_is_rejected(self):
        """Adoption is re-verified, not assumed: a wrong head is refused."""
        run_dir = write_run(
            self.tmp / "adopted", "run_b",
            l_main_hyperparams(weak_period_residual_head_type="pooled_lowrank"))
        manifest = write_manifest(
            self.tmp / "e14", [reused_cell(run_dir, self.tmp / "adopted")])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(report["resolved"], 0)
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("does not implement l_main",
                      report["rejected"][0]["reason"])

    def test_reused_cell_with_a_smoothed_run_is_rejected(self):
        """``l_main`` has smooth_ratio 0; a smoothed run is a different arm."""
        run_dir = write_run(
            self.tmp / "adopted", "run_c",
            l_main_hyperparams(weak_period_residual_smooth_ratio=0.5))
        manifest = write_manifest(
            self.tmp / "e14", [reused_cell(run_dir, self.tmp / "adopted")])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)

    def test_reused_cell_without_a_readable_config_is_rejected(self):
        """A missing run directory is rejected, and never silently blank."""
        missing = self.tmp / "adopted" / "runs" / "gone"
        manifest = write_manifest(
            self.tmp / "e14", [reused_cell(missing, self.tmp / "adopted")])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("no readable config.json",
                      report["rejected"][0]["reason"])

    def test_reused_cell_without_provenance_is_rejected(self):
        """No command and no source.run_dir is unauditable, so it is refused."""
        cell = reused_cell(self.tmp / "x", self.tmp / "x")
        cell["source"] = {"config_hash": "deadbeef"}
        manifest = write_manifest(self.tmp / "e14", [cell])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("neither a command nor source.run_dir",
                      report["rejected"][0]["reason"])

    def test_new_cell_is_admitted_from_its_command(self):
        """The argv path still works, and eval_root is the manifest's own root."""
        manifest = write_manifest(
            self.tmp / "e14", [new_cell(l_main_hyperparams())])
        index, report = load_baseline_index(manifest)

        self.assertEqual(report["rejected"], [])
        self.assertEqual(report["resolved_new"], 1)
        self.assertEqual(report["resolved_reused"], 0)
        entry = index[(DATASET, HORIZON, SEED)]
        self.assertEqual(entry["gate_init"], NEW_CELL_GATE_INIT)
        self.assertEqual(entry["learning_rate"], NEW_CELL_LR)
        # NOT the argv template: a new cell's run dir is not named yet, so the
        # root that holds the manifest is the only honest eval_root.
        self.assertEqual(entry["eval_root"], str(self.tmp / "e14"))
        self.assertNotEqual(entry["eval_root"], "/tmp/a-template-root")

    def test_new_cell_with_hand_edited_overrides_is_rejected(self):
        """The pre-existing argv check must survive the added reused path."""
        manifest = write_manifest(
            self.tmp / "e14",
            [new_cell(l_main_hyperparams(
                weak_period_residual_head_type="pooled_lowrank"))])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("overrides do not implement l_main",
                      report["rejected"][0]["reason"])

    def test_entry_reading_test_is_rejected(self):
        manifest = write_manifest(
            self.tmp / "e14",
            [new_cell(l_main_hyperparams(), argv_extra=["--evaluate-test"])])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("--evaluate-test", report["rejected"][0]["reason"])

    def test_other_arms_are_skipped_not_rejected(self):
        """Only l_main is this index's business; other arms are not failures."""
        run_dir = write_run(self.tmp / "adopted", "run_d", l_main_hyperparams())
        cell = reused_cell(run_dir, self.tmp / "adopted", arm="l_q1_4")
        manifest = write_manifest(self.tmp / "e14", [cell])
        index, report = load_baseline_index(manifest)

        self.assertEqual(index, {})
        self.assertEqual(report["rejected"], [])
        self.assertEqual(report["resolved"], 0)
        self.assertEqual(report["cells"], 1)

    def test_missing_manifest_is_reported_not_raised(self):
        index, report = load_baseline_index(self.tmp / "absent.json")
        self.assertEqual(index, {})
        self.assertFalse(report["exists"])
        self.assertEqual(len(report["rejected"]), 1)

    def test_manifest_without_cells_list_is_reported(self):
        path = self.tmp / "stage_a_manifest.json"
        path.write_text(json.dumps({"cells": None}), encoding="utf-8")
        index, report = load_baseline_index(path)
        self.assertEqual(index, {})
        self.assertEqual(len(report["rejected"]), 1)
        self.assertIn("no 'cells' list", report["rejected"][0]["reason"])

    def test_mixed_manifest_admits_both_kinds(self):
        """A manifest holding one reused and one new cell admits both."""
        run_dir = write_run(self.tmp / "adopted", "run_e", l_main_hyperparams())
        manifest = write_manifest(
            self.tmp / "e14",
            [reused_cell(run_dir, self.tmp / "adopted"),
             new_cell(l_main_hyperparams(), seed=SEED + 1)])
        index, report = load_baseline_index(manifest)

        self.assertEqual(report["rejected"], [])
        self.assertEqual(report["resolved"], 2)
        self.assertEqual((report["resolved_new"], report["resolved_reused"]),
                         (1, 1))
        self.assertEqual(sorted(key[2] for key in index), [SEED, SEED + 1])


if __name__ == "__main__":
    unittest.main()
