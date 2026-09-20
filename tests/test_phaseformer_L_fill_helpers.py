"""Tests for the two minipaper fill helpers.

The helpers themselves are deliberately small, but the fill they perform is the
step where a silent mistake is most expensive: an off-by-one row shift would put
one dataset's numbers under another dataset's name, and the paper would still
look complete.  These tests pin the three properties that make the fill safe:

* the artifact row lookup is keyed, so a paper row receives the numbers of ITS
  key rather than of whatever row happens to sit at the same position;
* the paper's own row order is preserved (the verifier in
  ``verify_minipaper_fill.py`` matches by key, so order is cosmetic -- but a
  reorder would silently break section 4.7, whose checker maps by INDEX);
* non-finite rho values become a literal ``NaN`` instead of a fabricated number.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLE_FILLER = REPO / "scripts" / "phaseformer_L" / "fill_minipaper_table.py"
RHO_FILLER = REPO / "scripts" / "phaseformer_L" / "fill_minipaper_47.py"


def run_self_test(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path), "--self-test"],
        capture_output=True, text=True,
    )


class TableFillerTests(unittest.TestCase):
    def test_self_test_passes(self):
        result = run_self_test(TABLE_FILLER)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_missing_artifact_is_refused_not_guessed(self):
        """A not-yet-produced artifact must exit 2, never write a partial fill."""
        result = subprocess.run(
            [sys.executable, str(TABLE_FILLER),
             "--minipaper", str(REPO / "docs" / "PhaseFormer_L_minipaper.md"),
             "--artifact", str(REPO / "does_not_exist.md")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not exist", result.stderr)


class RhoFillerTests(unittest.TestCase):
    def test_self_test_passes(self):
        result = run_self_test(RHO_FILLER)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_statistics_matches_the_verifier(self):
        """The drift guard: this tool's STATISTICS must equal the verifier's.

        Section 4.7 is filled by index, so a change to the verifier's tuple
        without the same change here would silently shift every row.
        """
        from scripts.phaseformer_L.fill_minipaper_47 import STATISTICS, spec_check
        from scripts.phaseformer_L.verify_minipaper_fill import STATISTICS as theirs

        self.assertEqual(tuple(STATISTICS), tuple(theirs))
        notes = spec_check(str(REPO))
        self.assertTrue(any("matches the verifier" in n for n in notes), notes)

    def test_non_finite_formats_as_nan(self):
        from scripts.phaseformer_L.fill_minipaper_47 import format_rho

        self.assertEqual(format_rho(float("nan")), "NaN")
        self.assertEqual(format_rho(None), "NaN")
        self.assertEqual(format_rho("not a number"), "NaN")
        self.assertEqual(format_rho(0.123456), "0.123")
        self.assertEqual(format_rho(-0.5), "-0.500")


if __name__ == "__main__":
    unittest.main()


class RendererMatchesVerifierTests(unittest.TestCase):
    """The composed section 4.4 cells must satisfy the verifier that reads them.

    This is the test that matters most for this table: the cells are *composed*
    rather than copied, so the filler and the verifier are two implementations of
    the same convention. If they disagree, every cell would be written in a shape
    the verifier rejects -- and the failure would only appear at fill time. So the
    filler's rendered cells are fed straight through the verifier's own checker
    here.

    It also pins the two easy-to-get-wrong details: the explanation rate must be the
    LAST number in the cell (group labels can contain slashes and digits), and the
    artifact's leading `model` column must not be written into the paper.
    """

    ROW = {
        "model": "PhaseFormer-L", "dataset": "ETTh2", "horizon": "96",
        "input_group_label": "周期形状/相位", "input_group_explanation": "0.6649",
        "output_group_label": "近端电平", "output_group_explanation": "0.5",
        "correction_energy_share": "0.6521771",
        "leading4_input_overlap": "0.777", "leading4_output_overlap": "0.666",
        "stable_semantics_verdict": "True",
    }

    def _paper(self) -> str:
        return "\n".join([
            "### 4.4 x",
            "",
            "| 模型 | Dataset | H | 主模式输入组 / 解释率 | 主模式输出组 / 解释率 |"
            " 修正能量份额 | 跨 seed leading4 重叠 | 稳定语义判定 |",
            "|---|---|---:|---|---|---:|---|---|",
            "| PhaseFormer-L | ETTh2 | 96 |  |  |  |  |  |",
            "",
            "### 4.5 y",
            "",
        ])

    def test_rendered_cells_pass_the_verifier(self):
        import csv
        import tempfile
        from pathlib import Path

        from scripts.phaseformer_L.fill_minipaper_44_dissection import render
        from scripts.phaseformer_L.verify_minipaper_fill import (
            E16, Report, check_4_4_dissection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / E16
            target.mkdir(parents=True)
            with (target / "dissection_table_44.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.ROW))
                writer.writeheader()
                writer.writerow(self.ROW)

            paper = root / "paper.md"
            paper.write_text(self._paper(), encoding="utf-8")

            # Fill the row the way the tool does, then verify it.
            lines = paper.read_text(encoding="utf-8").splitlines()
            cells = [c.strip() for c in lines[4].strip().strip("|").split("|")]
            lines[4] = "| " + " | ".join(cells[:3] + render(self.ROW)) + " |"
            paper.write_text("\n".join(lines) + "\n", encoding="utf-8")

            report = Report()
            check_4_4_dissection(report, root, paper)

        states = [row["state"] for row in report.rows]
        details = [row["detail"] for row in report.rows]
        self.assertNotIn("MISMATCH", states, details)
        self.assertNotIn("blank", states, details)
        # 5 composed columns -> five comparisons for the single row
        self.assertEqual(len(report.rows), 5, details)

    def test_rate_must_be_the_last_number(self):
        """A label carrying its own digits must not be mistaken for the rate."""
        from scripts.phaseformer_L.fill_minipaper_44_dissection import render

        row = dict(self.ROW)
        row["input_group_label"] = "组2/周期"
        row["input_group_explanation"] = "0.31"
        cell = render(row)[0]
        self.assertTrue(cell.startswith("组2/周期"))
        numbers = __import__("re").findall(r"[-+]?\d*\.?\d+", cell)
        self.assertEqual(float(numbers[-1]), 0.31)


class Section45FillerTests(unittest.TestCase):
    def test_self_test_passes(self):
        result = run_self_test(REPO / "scripts" / "phaseformer_L" / "fill_minipaper_45.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_h1_aggregation_is_imported_from_the_verifier(self):
        """The filler must not carry its own copy of the H1 grouping rule.

        Section 4.5's seventh column is a seed count derived from per-seed verdicts.
        If the filler and the verifier grouped differently, every H1 cell would be
        written in a shape the verifier rejects.  The filler imports the verifier's
        aggregate_h1 for exactly this reason, and this pins that it stays importable.
        """
        from scripts.phaseformer_L.fill_minipaper_45 import import_h1, render_h1

        self.assertIs(import_h1(str(REPO)), __import__(
            "scripts.phaseformer_L.verify_minipaper_fill",
            fromlist=["aggregate_h1"]).aggregate_h1)
        self.assertEqual(render_h1((3, 3)), "3/3 seed")
        self.assertEqual(render_h1((0, 3)), "0/3 seed")
        self.assertEqual(render_h1("evidence_missing"), "evidence_missing")


class Section46FillerTests(unittest.TestCase):
    def test_self_test_passes(self):
        result = run_self_test(REPO / "scripts" / "phaseformer_L" / "fill_minipaper_46.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_row_count_mismatch_refuses_to_write(self):
        """A positional fill with unequal row counts must write nothing."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "negative_table.md"
            artifact.write_text(
                "| 操作 | 对照 | 口径 | 既有结果 | 本文补做 |\n|---|---|---|---|---|\n"
                "| only-one-row | a | b | c | NEW |\n", encoding="utf-8")
            paper = root / "paper.md"
            original = (
                "### 4.6 x\n\n| 操作 | 对照 | 口径 | 既有结果 | 本文补做 |\n"
                "|---|---|---|---|---|\n| r1 | a | b | c |  |\n| r2 | a | b | c |  |\n\n### 4.7 y\n")
            paper.write_text(original, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "phaseformer_L" / "fill_minipaper_46.py"),
                 "--minipaper", str(paper), "--artifact", str(artifact), "--write"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("refusing a positional fill", result.stderr)
            self.assertEqual(paper.read_text(encoding="utf-8"), original)


class E16ShardMergeTests(unittest.TestCase):
    """E16 is assembled from 21 (setting, arm) shards; the merge must be auditable.

    Two properties carry the whole design: rows are concatenated (never
    recomputed), and the aggregates move in the safe direction -- parity is ANDed
    across shards and "did any shard read test" is ORed. A header mismatch must
    abort rather than merge shifted columns into nonsense.
    """

    MERGE = REPO / "scripts" / "phaseformer_L" / "merge_e16_shards.py"

    def test_self_test_passes(self):
        result = run_self_test(self.MERGE)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_shard_file_refuses_to_merge(self):
        import tempfile
        from pathlib import Path

        import csv as csv_mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shard = tmp / "phaseformer_L_e16_shard_00"
            shard.mkdir()
            # every CSV present, but the summary is missing
            for name in ("dissection_table.csv", "intervention_table.csv",
                         "canonical_modes.csv", "semantic_alignment.csv",
                         "cross_seed_alignment.csv"):
                with (shard / name).open("w", newline="") as handle:
                    writer = csv_mod.writer(handle)
                    writer.writerow(["arm", "setting"])
                    writer.writerow(["l_main", "ETTh2-96"])

            result = subprocess.run(
                [sys.executable, str(self.MERGE),
                 "--repo-root", str(tmp),
                 "--shard-glob", "phaseformer_L_e16_shard_*",
                 "--output-root", "out", "--write"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("lacks", result.stderr)
            self.assertFalse((tmp / "out").exists())
