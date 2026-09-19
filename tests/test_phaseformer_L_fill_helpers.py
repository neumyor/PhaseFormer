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
