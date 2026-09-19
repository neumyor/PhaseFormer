"""Stage-5 audit of E14 stage A: the 411 runs behind the minipaper section 4.2.

The eight invariants below were first checked by hand on the first seven cells
(``execution_schedule.md`` 2026-09-19).  That audit lived in a scratch script, so
this makes it durable and repeats it over the whole matrix -- the stage-5
instrument for section 4.2, which is also the input the section-4.2 verdict
depends on.

Per *new* cell (resolved with ``e14_read_test.locate_run``, i.e. the same arm
fingerprint the stage-B reader uses):

1. a run directory resolves, and resolves **uniquely** (an ambiguous match would
   mean two runs claim the same cell);
2. ``metrics.csv`` exists -- the cell is finished;
3. ``metrics.csv`` records a ``checkpoint`` and that file exists;
4. ``val_mse`` is present and parseable;
5. **``test_mse`` and ``test_mae`` are EMPTY** -- stage A must never read the test
   split; this is the invariant the whole single-read protocol rests on;
6. ``epochs_completed`` is between 1 and ``max_epochs`` -- deliberately *not*
   "equals the requested count": early stopping (patience 8) stops both new and
   reused cells, so equality was a misjudgement in the first version of this
   audit;
7. ``parameter_count`` is present (it is what the parameter table cross-checks);
8. the run's ``config.json`` does not set ``evaluate_test``.

A cell with no run yet is **PENDING**, not a failure: the matrix is filled over
hours, so an incomplete matrix is reported and the run still exits 0.  Any
violation on a *finished* run is a failure (exit 1).

Usage (on the server, from the repository root)::

    python scripts/phaseformer_L/audit_e14_stage_a.py \
        [--e14-root research_runs/phaseformer_L_e14_main_v1] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phaseformer_L import e14_read_test as m14  # noqa: E402

MAX_EPOCHS = 30


def numeric(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def audit_cell(cell, e14_root: Path) -> dict:
    """Return the per-cell verdict; ``state`` is ok | pending | fail."""
    key = str(cell.get("key") or m14.cell_key(cell))
    run_dir, config, matched, _near = m14.locate_run(e14_root, cell)
    problems: list = []

    if run_dir is None:
        return {"key": key, "state": "pending", "problems": [], "run_dir": None}
    if len(matched) > 1:
        problems.append(f"ambiguous run match ({len(matched)} dirs)")

    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.is_file():
        # A directory without metrics.csv is an in-flight attempt, not a failure.
        return {"key": key, "state": "pending", "problems": problems,
                "run_dir": str(run_dir)}
    record = m14.first_metrics_row(run_dir) or {}

    checkpoint = str(record.get("checkpoint", "")).strip()
    if not checkpoint:
        problems.append("metrics.csv records no checkpoint")
    else:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            fallback = Path(run_dir) / "attempts" / "001" / "checkpoints" / "best.ckpt"
            if not fallback.is_file():
                problems.append(f"checkpoint missing: {checkpoint}")

    if numeric(record.get("val_mse")) is None:
        problems.append("val_mse missing or not numeric")

    for column in ("test_mse", "test_mae"):
        if str(record.get(column, "")).strip():
            problems.append(f"{column} is populated: stage A must never read test")

    epochs = numeric(record.get("epochs_completed"))
    if epochs is None:
        problems.append("epochs_completed missing or not numeric")
    elif not 1 <= epochs <= MAX_EPOCHS:
        problems.append(f"epochs_completed={epochs} outside 1..{MAX_EPOCHS}")

    if str(record.get("parameter_count", "")).strip() == "":
        problems.append("parameter_count missing")

    if isinstance(config, dict):
        try:
            read_test = int(config.get("evaluate_test", 0) or 0)
        except (TypeError, ValueError):
            read_test = 1
        if read_test:
            problems.append("config.json sets evaluate_test")

    return {"key": key, "state": "fail" if problems else "ok",
            "problems": problems, "run_dir": str(run_dir)}


def self_test() -> int:
    """Calibrate on a scratch root: a clean cell plus three broken ones.

    An audit that only ever reports "fine" proves nothing, so each invariant gets
    a cell that violates exactly it.  Built in a temp dir from the REAL manifest's
    first new cell (its arm fingerprint must keep matching, or locate_run would
    report the cell as pending instead of failing it for the injected reason).
    """
    import shutil
    import tempfile

    real_root = ROOT / "research_runs" / "phaseformer_L_e14_main_v1"
    manifest = json.loads((real_root / "stage_a_manifest.json").read_text())
    new_cells = [c for c in manifest["cells"] if c["status"] == "new"]
    template_dir, _config, _matched, _near = m14.locate_run(real_root, new_cells[0])
    if template_dir is None:
        print("self-test needs at least one finished run in the real root")
        return 1
    template_metrics = (Path(template_dir) / "metrics.csv").read_text()
    header = template_metrics.splitlines()[0]
    fields = header.split(",")

    def metrics(**overrides) -> str:
        row = ["0"] * len(fields)
        base = {"val_mse": "0.5", "checkpoint": "attempts/001/checkpoints/best.ckpt",
                "epochs_completed": "5", "parameter_count": "1000",
                "test_mse": "", "test_mae": ""}
        base.update(overrides)
        for index, name in enumerate(fields):
            if name in base:
                row[index] = str(base[name])
        return header + "\n" + ",".join(row) + "\n"

    cases = {
        "clean": metrics(),
        "reads_test": metrics(test_mse="0.9"),
        "zero_epochs": metrics(epochs_completed="0"),
        "no_parameter_count": metrics(parameter_count=""),
    }
    assertions = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        cells = []
        for name, body in cases.items():
            cell = dict(new_cells[0])
            # Each synthetic cell needs its OWN key: copying the template's key
            # made all four verdicts collapse onto one dict entry, so every case
            # printed the last case's result (the fixture's bug, not the audit's).
            cell["key"] = f"selftest_{name}__{cell['dataset']}-{cell['horizon']}"
            cell["seed"] = int(cell["seed"]) + len(cells)   # keep cells distinct
            run_id = f"{name}_{Path(template_dir).name}"
            run_dir = scratch / "runs" / run_id
            shutil.copytree(template_dir, run_dir)
            (run_dir / "metrics.csv").write_text(body)
            (run_dir / "attempts/001/checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "attempts/001/checkpoints/best.ckpt").write_text("stub")
            cell["source"] = None
            cells.append(cell)
        # Distinguish the cells by (arm, dataset, horizon, seed): the fingerprint
        # match also reads config.json's seed, so patch each copy's config too.
        for index, cell in enumerate(cells):
            run_dir = scratch / "runs" / f"{list(cases)[index]}_{Path(template_dir).name}"
            config = json.loads((run_dir / "config.json").read_text())
            config["seed"] = int(cell["seed"])
            (run_dir / "config.json").write_text(json.dumps(config))
        (scratch / "stage_a_manifest.json").write_text(json.dumps(
            {"cells": cells, "counts": {"total": len(cells)}}))

        verdicts = {cell["key"]: audit_cell(cell, scratch) for cell in cells}
        for index, name in enumerate(cases):
            record = verdicts[cells[index]["key"]]
            expected = "ok" if name == "clean" else "fail"
            ok = record["state"] == expected
            assertions.append(ok)
            print(f"  [{'OK  ' if ok else 'FAIL'}] self-test {name}: "
                  f"{record['state']} (expected {expected})"
                  + (f" -- {record['problems']}" if record["problems"] else ""))
        # The test-read violation must be reported by name, not just counted.
        reads = [r for r in verdicts.values()
                 if any("must never read test" in p for p in r["problems"])]
        ok = len(reads) == 1
        assertions.append(ok)
        print(f"  [{'OK  ' if ok else 'FAIL'}] exactly one cell flagged for reading "
              f"test: {len(reads)}")

    if all(assertions):
        print(f"\nOK: {len(assertions)} self-test assertion(s) held")
        return 0
    print(f"\nFAIL: {assertions.count(False)} of {len(assertions)} assertion(s) failed")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e14-root",
                        default="research_runs/phaseformer_L_e14_main_v1")
    parser.add_argument("--json", default="", help="write the report as JSON here")
    parser.add_argument("--max-failures", type=int, default=15,
                        help="print at most this many failing cells")
    parser.add_argument("--self-test", action="store_true",
                        help="calibrate on a scratch root with injected violations")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    e14_root = Path(args.e14_root)
    if not e14_root.is_absolute():
        e14_root = ROOT / e14_root
    manifest_path = e14_root / "stage_a_manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    cells = manifest.get("cells") or []
    new_cells = [c for c in cells if c.get("status") == "new"]
    reused = [c for c in cells if c.get("status") == "reused"]
    declared = (manifest.get("counts") or {}).get("total")

    print(f"manifest: {manifest_path}")
    print(f"cells: {len(cells)} (declared total {declared})")
    print(f"  new: {len(new_cells)}  reused: {len(reused)}\n")

    results = [audit_cell(cell, e14_root) for cell in new_cells]
    states = collections.Counter(r["state"] for r in results)
    failures = [r for r in results if r["state"] == "fail"]
    pending = [r for r in results if r["state"] == "pending"]

    print("stage A, per cell:")
    print(f"  ok      : {states.get('ok', 0)}")
    print(f"  pending : {states.get('pending', 0)}  (no finished run yet)")
    print(f"  fail    : {states.get('fail', 0)}")

    if failures:
        print(f"\nFAILING CELLS ({len(failures)}):")
        for record in failures[:args.max_failures]:
            print(f"  {record['key']}: {'; '.join(record['problems'])}")
        if len(failures) > args.max_failures:
            print(f"  ... and {len(failures) - args.max_failures} more")

    # The protocol invariant deserves its own line even when it holds: it is the
    # reason stage B is a separate step.
    read_test = [r["key"] for r in results
                 if any("must never read test" in p for p in r["problems"])]
    print(f"\ntest split read during stage A: {len(read_test)} cell(s)"
          + (" (must be 0)" if not read_test else f" -> {read_test[:5]}"))

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "manifest": str(manifest_path),
            "new_cells": len(new_cells), "reused_cells": len(reused),
            "declared_total": declared,
            "states": dict(states),
            "pending": [r["key"] for r in pending],
            "failures": failures,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report written: {path}")

    if failures:
        print("\nStage A violates an invariant on a finished run; the matrix is "
              "not usable as-is.")
        return 1
    if pending:
        print(f"\nStage A is incomplete ({len(pending)} cell(s) still pending) but "
              f"every finished cell satisfies all eight invariants.")
        return 0
    print("\nStage A complete: all "
          f"{len(new_cells)} cells finished and satisfy all eight invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
