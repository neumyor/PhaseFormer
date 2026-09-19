"""Pre-flight contract check: do the phase-2 consumers accept E14's real manifest?

Why this exists
---------------
Every phase-2 step derives its cells from E14's ``stage_a_manifest.json``.  A
consumer that mis-reads that artifact degrades *silently*: the step still exits
0 and still writes a table, but with empty columns or missing rows.  Two such
defects were found this way (``docs/PhaseFormer_L/audit/paper_code_consistency.md``
§16):

* E18's baseline index rejected every **reused** cell -- they carry
  ``"command": null``, because stage A did not launch them -- and so lost the
  baseline provenance of **all 78** of its rows;
* every **new** cell's recorded ``--output-dir`` was a template value
  (``/tmp/e14fix``), which a consumer would have copied into its artifact as the
  cell's ``eval_root``.

A first attempt at this check was a *static* extractor that scanned the
consumers' source for ``X.get("key")`` on **any** receiver.  It reported four
phantom gaps, because a locally built ``config``/``cell``/row dict is not a
manifest cell.  This script therefore calls the consumers' own loaders rather
than reading their source: **a gap reported here is a real gap.**

The manifest's two cell shapes (measured on the live artifact) are::

    new    : {arm, command, dataset, horizon, key, seed, source: null, status}
    reused : {arm, command: null, dataset, horizon, key, seed,
              source: {config_hash, gate_init, learning_rate, root, run_dir,
                       test_evidence}, status}

Checks, and their severity
--------------------------
============================  ========  ==========================================
check                         severity  what must hold
============================  ========  ==========================================
C1 E14 loader schema          FAIL      ``e14_read_test.load_manifest_cells`` accepts
                                        the manifest; cells carry the required
                                        fields and one uniform schema; the
                                        manifest's own ``counts.total`` agrees.
C2 E17 conditional index      FAIL      every declared ``l_main`` cell of E17's
                                        7 settings x 3 seeds resolves; no
                                        rejection.  They are all reused cells, so
                                        they exist before E14 finishes.
C3 E18 baseline provenance    FAIL      **every** (setting, seed) of E18's smooth
                                        stage resolves, and nothing is rejected.
                                        This is the check that would have caught
                                        the §16 defect before a 3-5 h run.
C4 E16 dissection plan        INFO      how many of E16's cells resolve now.  Its
                                        63 cells are reused ones, but the check
                                        stays informational because E16 owns its
                                        scope; 0 resolved still means FAIL.
C5 E18 SVD truncation plan    INFO      plan size and unresolved cells.  Unresolved
                                        cells are *expected* while stage A is
                                        unfinished.
============================  ========  ==========================================

C4/C5 import E16, the one consumer that needs numpy/torch at import time; if
those are unavailable the check is reported as SKIP instead of failing.

Usage::

    python scripts/phaseformer_L/check_phase2_consumers.py \
        --e14-root research_runs/phaseformer_L_e14_main_v1 [--json OUT.json]
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_E14_ROOT = "research_runs/phaseformer_L_e14_main_v1"
REQUIRED_FIELDS = ("arm", "dataset", "horizon", "seed", "status")

OK = "OK"
FAIL = "FAIL"
INFO = "INFO"
SKIP = "SKIP"


class Checker:
    def __init__(self) -> None:
        self.rows: list = []

    def record(self, name: str, state: str, detail: str) -> None:
        self.rows.append({"check": name, "state": state, "detail": detail})
        print(f"{state:>4} {name}: {detail}", flush=True)

    @property
    def failed(self) -> list:
        return [row for row in self.rows if row["state"] == FAIL]


def load_consumer(name: str):
    """Import a phase-2 consumer module by name.

    Returns ``None`` when a heavy third-party import (numpy/torch) is missing, so
    the check can degrade to SKIP instead of failing the pre-flight.
    """
    try:
        return importlib.import_module(f"scripts.phaseformer_L.{name}")
    except ImportError as exc:  # numpy/torch absent, not a contract violation
        print(f"  (cannot import {name}: {exc})", flush=True)
        return None


def check_e14_loader(checker: Checker, manifest_path: Path):
    """C1: the manifest is readable, complete and schema-uniform."""
    m14 = load_consumer("e14_read_test")
    try:
        manifest, cells = m14.load_manifest_cells(manifest_path)
    except SystemExit as exc:
        checker.record("C1 e14 loader schema", FAIL, f"loader raised: {exc}")
        return None
    problems = []
    declared = manifest.get("counts") or {}
    total = declared.get("total") if isinstance(declared, dict) else None
    if total is not None and int(total) != len(cells):
        problems.append(f"counts.total={total} but {len(cells)} cells")
    schemas = collections.Counter(tuple(sorted(cell.keys())) for cell in cells)
    if len(schemas) != 1:
        problems.append(f"{len(schemas)} distinct cell schemas: "
                        f"{[list(shape) for shape in schemas]}")
    for shape in schemas:
        missing = [field for field in REQUIRED_FIELDS if field not in shape]
        if missing:
            problems.append(f"cell shape {list(shape)} lacks {missing}")
    if problems:
        checker.record("C1 e14 loader schema", FAIL, "; ".join(problems))
        return None
    checker.record("C1 e14 loader schema", OK,
                   f"{len(cells)} cells, one schema, counts.total agrees")
    return cells


def check_e17_index(checker: Checker, e14_root: Path):
    """C2: E17 resolves every l_main cell it declares (all reused ones)."""
    m17 = load_consumer("e17_conditional")
    if m17 is None:
        checker.record("C2 e17 conditional index", SKIP, "module unavailable")
        return None
    seeds = m17.parse_list(m17.build_parser().parse_args([]).seeds, int)
    wanted = {(d, h, s) for (d, h) in m17.SETTINGS for s in seeds}
    index, rejected = m17.build_e14_index(e14_root, wanted)
    missing = sorted(wanted - set(index))
    if rejected or missing:
        checker.record(
            "C2 e17 conditional index", FAIL,
            f"declared={len(wanted)} resolved={len(index)} "
            f"missing={len(missing)} rejected={len(rejected)}"
            + (f" e.g. missing {missing[:3]}" if missing else "")
            + (f" e.g. rejected {rejected[0]}" if rejected else ""))
        return index
    pending = sum(1 for entry in index.values()
                  if entry.get("test", {}).get("test_evidence")
                  == "pending_e14_stage_b")
    checker.record("C2 e17 conditional index", OK,
                   f"{len(index)}/{len(wanted)} cells resolved, 0 rejected; "
                   f"{pending} still awaiting E14 stage B (expected)")
    return index


def check_e18_baselines(checker: Checker, manifest_path: Path):
    """C3: E18's smooth stage has a baseline for every (setting, seed).

    These are exactly the reused cells, so this must hold *before* E14 finishes;
    a rejection here means the E18 table would carry empty provenance.
    """
    m18 = load_consumer("e18_negative")
    index, report = m18.load_baseline_index(manifest_path)
    seeds = list(m18.SEEDS)
    wanted = {(d, h, s) for (d, h) in m18.SMOOTH_SETTINGS for s in seeds}
    missing = sorted(wanted - set(index))
    rejected = report.get("rejected") or []
    if rejected or missing:
        detail = (f"resolved={report.get('resolved')} "
                  f"(new={report.get('resolved_new')}, "
                  f"reused={report.get('resolved_reused')}) "
                  f"missing={len(missing)} rejected={len(rejected)}")
        if rejected:
            detail += f" e.g. {rejected[0]}"
        if missing:
            detail += f" e.g. missing {missing[:3]}"
        checker.record("C3 e18 baseline provenance", FAIL, detail)
        return index
    checker.record(
        "C3 e18 baseline provenance", OK,
        f"smooth coverage {len(wanted)}/{len(wanted)} setting-seed keys; "
        f"resolved={report.get('resolved')} "
        f"(new={report.get('resolved_new')}, "
        f"reused={report.get('resolved_reused')}), 0 rejected")
    return index


def check_e16_plan(checker: Checker, e14_root: Path):
    """C4 (INFO): what E16's own planner resolves right now."""
    m16 = load_consumer("e16_dissection")
    if m16 is None:
        checker.record("C4 e16 dissection plan", SKIP, "module unavailable")
        return
    args = m16.build_parser().parse_args([])
    args.e14_root = str(e14_root)
    args.allow_missing_cells = True
    try:
        cells = m16.build_cell_plan(args)
    except SystemExit as exc:
        checker.record("C4 e16 dissection plan", FAIL, f"planner raised: {exc}")
        return
    by_arm = collections.Counter(cell["arm"] for cell in cells)
    detail = f"{len(cells)} cell(s) resolvable now, by arm: {dict(by_arm)}"
    checker.record("C4 e16 dissection plan", INFO if cells else FAIL, detail)


def check_svd_plan(checker: Checker, manifest_path: Path):
    """C5 (INFO): E18's truncation plan builds; its unresolved cells are expected."""
    msvd = load_consumer("e18_svd_truncation")
    if msvd is None:
        checker.record("C5 e18 svd plan", SKIP, "module unavailable")
        return
    args = msvd.parse_args([])
    args.manifest = str(manifest_path)
    try:
        plan, problems, _path, _seeds, _ranks = msvd.build_plan(args)
    except SystemExit as exc:
        checker.record("C5 e18 svd plan", FAIL, f"planner raised: {exc}")
        return
    kinds = collections.Counter(
        str(item.get("problem") if isinstance(item, dict) else item)
        for item in problems)
    unresolved = sum(1 for record in plan if record.get("problems"))
    state = FAIL if not plan else INFO
    checker.record("C5 e18 svd plan", state,
                   f"{len(plan)} settings planned, {unresolved} carrying "
                   f"problems {dict(kinds)} (expected while stage A runs)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--e14-root", default=DEFAULT_E14_ROOT,
                        help="root holding stage_a_manifest.json and runs/")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="skip C4/C5, the checks that import E16 (numpy/torch)")
    parser.add_argument("--json", default="",
                        help="also write the summary to this JSON path")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    e14_root = Path(args.e14_root)
    if not e14_root.is_absolute():
        e14_root = ROOT / e14_root
    manifest_path = e14_root / "stage_a_manifest.json"

    print(f"E14 root: {e14_root}")
    print(f"manifest: {manifest_path}")
    print(f"exists  : {manifest_path.is_file()}")
    print()

    checker = Checker()
    if not manifest_path.is_file():
        checker.record("C0 manifest present", FAIL,
                       f"no stage_a_manifest.json under {e14_root}")
    else:
        check_e14_loader(checker, manifest_path)
        check_e17_index(checker, e14_root)
        check_e18_baselines(checker, manifest_path)
        if args.skip_heavy:
            checker.record("C4 e16 dissection plan", SKIP, "--skip-heavy")
            checker.record("C5 e18 svd plan", SKIP, "--skip-heavy")
        else:
            check_e16_plan(checker, e14_root)
            check_svd_plan(checker, manifest_path)

    failed = checker.failed
    states = collections.Counter(row["state"] for row in checker.rows)
    print()
    print("=" * 68)
    print("summary: " + ", ".join(f"{state}={n}" for state, n in
                                 sorted(states.items())))
    if failed:
        print(f"FAIL: {len(failed)} contract violation(s)")
        for row in failed:
            print(f"  - {row['check']}: {row['detail']}")
    else:
        print("OK: phase-2 consumers accept the live E14 manifest")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"e14_root": str(e14_root), "manifest": str(manifest_path),
             "checks": checker.rows,
             "failed": len(failed)}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"summary written: {path}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
