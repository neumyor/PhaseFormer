#!/usr/bin/env python3
"""E14 stage B: read the test split exactly once for every *new* main-table cell.

`docs/PhaseFormer_L_execution_schedule.md` §2.2 (experiment E14, minipaper §4.2)
splits the §4.2 main table into two stages:

* **stage A** (`scripts/phaseformer_L/e14_main_matrix.py`) trains every *new* cell
  and writes `stage_a_manifest.json` into the output root.  It never passes
  `--evaluate-test`, so no test split is touched while the matrix is built.
* **stage B** (this script) runs only after every checkpoint is frozen, and reads
  the test split **exactly once per new cell**.  It is the E14 generalization of
  the proven E8 single-read protocol in
  `scripts/read_top2_direction_retention_test.py`.

For a `new` cell this script

1. locates the run directory under the output root by reading its `config.json`
   and matching the arm fingerprint of `e14_main_matrix.ARMS` / `_arm_match()`;
   a run carrying `weak_residual_projection` is a frozen-subspace arm and is
   never an E14 cell, so such near-miss runs are reported but never used;
2. rebuilds the trained model from `config["hyperparams"]` through
   `make_exp_args` + `PhaseFormerPresetConfig` and restores the run's
   best-validation checkpoint;
3. re-derives validation MSE from the restored checkpoint and requires it to
   reproduce the run's recorded `val_mse` within `VAL_REPRODUCE_TOL`.  A larger
   gap means the checkpoint or the protocol does not match, so the cell is
   reported as `rejected` and its test split is **not** read at all;
4. reads the test split once, recording the fused MSE/MAE, the residual branch's
   own MSE/MAE (`nlinear_mse` / `nlinear_mae`), the fusion gate (`gate_value`),
   the recomputed validation MSE and the reproduction gap.

A `reused` cell is never rebuilt and never re-read: its test metrics are copied
from the already-registered source that stage A audited -- either the run's own
inline `metrics.csv` (E3 lineage) or a registered evidence CSV such as
`research_runs/top2_direction_retention_v1/results.csv`, where the arm value for
E14's `l_main` row is `direct_nlinear` and for `phase_only` it is `phase_only`.

`torch` / `numpy` are imported lazily inside the GPU paths, so `--dry-run` runs
the whole pre-flight (arm fingerprint check, run location, reuse-evidence
verification) without them, on any machine, and writes nothing.

Outputs (in `--output-root`):

* `test_read_summary.json` -- `protocol` block plus one detail record per cell;
* `results.csv`             -- flat table, one row per cell;
* `test_read/<key>.json`    -- per-cell artifact, the idempotence guard that
  prevents a second read of an already-read checkpoint;
* `_logs/test_read_<key>.log` -- per-cell worker log.

Usage::

    # stage-2 gate: pre-flight the plan; imports no torch and writes nothing
    python scripts/phaseformer_L/e14_read_test.py \
        --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \
        --output-root research_runs/phaseformer_L_e14_main_v1 --dry-run

    # the single test read on 8 GPUs, one cell per GPU, one retry each
    python scripts/phaseformer_L/e14_read_test.py \
        --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \
        --output-root research_runs/phaseformer_L_e14_main_v1 \
        --gpus 0,1,2,3,4,5,6,7 --retries 1 --poll-seconds 15
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
STAGE_A_SCRIPT = HERE / "e14_main_matrix.py"

# --------------------------------------------------------------------------
# Protocol constants.  The first block is mirrored verbatim from
# ``scripts/phaseformer_L/e14_main_matrix.py`` (minipaper §4.0) and is compared
# against that file at startup -- see ``fingerprint_report()``.
# --------------------------------------------------------------------------
LOOKBACK = 720
PERIOD = 24
MAX_EPOCHS = 30
LOSS = "huber"
PERCENT = 100

# Arms of the §4.2 variant table.  ``rank_div`` (when set) means
# pooled_lowrank with rank = horizon // rank_div, pool_factor = 1.
ARMS = {
    "phase_only": {"mechanism": "no_residual", "rank_div": None},
    "l_main": {"mechanism": "weak_residual", "rank_div": None, "head": "shared"},
    "l_q1_4": {"mechanism": "weak_residual", "rank_div": 4, "head": "pooled_lowrank"},
    "l_q1_8": {"mechanism": "weak_residual", "rank_div": 8, "head": "pooled_lowrank"},
    "l_rcrf": {"mechanism": "rcrf_nlinear_plain", "rank_div": None},
    "a1": {"mechanism": "gold_combo_reliability_s2", "rank_div": None},
}

# Registered external single-test-read evidence for reused cells.  E8 launched
# its training runs without ``--evaluate-test``, so those runs' own
# ``metrics.csv`` has empty ``test_mse``/``test_mae`` and the audited numbers live
# in the experiment summary CSV instead.  Mirrors ``EXTERNAL_TEST_EVIDENCE`` in
# ``e14_main_matrix.py``.
EXTERNAL_TEST_EVIDENCE = (
    {
        "path": "research_runs/top2_direction_retention_v1/results.csv",
        "protocol": "top2-direction-retention-v1",
        "arm_column": "arm",
        "arm_values": {"phase_only": "phase_only", "l_main": "direct_nlinear"},
        "status_column": "test_read_status",
        "status_ok": ("read", "reused"),
        "val_column": "val_relative_difference",
        "val_tolerance": 1e-3,
    },
)

# Recomputing validation from the restored checkpoint cannot be bit-identical:
# the evaluation dataloader differs in worker count from the training run's, so
# float summation order differs.  Anything beyond float-level drift means the
# checkpoint or protocol does not match and the cell must not be reported.
# Kept identical to the E8 template.
VAL_REPRODUCE_WARN = 1e-4
VAL_REPRODUCE_TOL = 1e-3

# Stage A trains with ``--num-workers 4`` (schedule §4.2 principle 4); the same
# worker count is used here so the validation re-derivation stays comparable.
DEFAULT_NUM_WORKERS = 4

# Exact column order of ``results.csv`` (schedule §3.1 artifact naming).
RESULTS_FIELDS = (
    "arm", "dataset", "horizon", "seed", "setting", "status",
    "test_mse", "test_mae", "nlinear_mse", "nlinear_mae", "gate_value",
    "val_mse", "recorded_val_mse", "val_relative_difference", "test_size",
    "run_dir", "config_hash", "source",
)

# Statuses that mean "this cell is fine to report".
STATUS_PLANNED = "planned"
STATUS_READ = "read"
STATUS_VAL_DRIFT = "val_drift"
STATUS_ALREADY_READ = "already_read"
STATUS_REUSED = "reused"
OK_STATUSES = (STATUS_READ, STATUS_VAL_DRIFT, STATUS_ALREADY_READ, STATUS_REUSED,
               STATUS_PLANNED)

# Statuses that mean "this cell must not be reported" (non-zero exit).
PROBLEM_STATUSES = (
    "rejected",
    "missing_run",
    "missing_metrics",
    "missing_checkpoint",
    "missing_reused_source",
    "missing_cell",
    "unknown_manifest_status",
    "unknown_arm",
    "worker_failed",
)

# Statuses for which the test split has already been consumed.  A cell in one of
# these states is never read a second time, even across script invocations.
CONSUMED_STATUSES = (STATUS_READ, STATUS_VAL_DRIFT, STATUS_ALREADY_READ)

PROTOCOL_NAME = "phaseformer-L-e14-main-single-test-read-v1"

PROTOCOL_NOTE = (
    "One test read per new checkpoint: every manifest cell with status 'new' is "
    "evaluated on the test split exactly once, after stage A froze its "
    "best-validation checkpoint, and only after the validation MSE re-derived "
    "from the restored checkpoint reproduces the run's recorded val_mse within "
    "VAL_REPRODUCE_TOL.  Cells with status 'reused' are never re-read: their test "
    "metrics are copied verbatim from the already-registered source that stage A "
    "audited (an inline metrics.csv or a registered evidence CSV).  No "
    "configuration, checkpoint or model structure may change after this step."
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_list(raw, cast=str):
    return [cast(item) for item in str(raw).split(",") if item.strip()]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def num(value):
    """Parse a metrics value to float; "" / junk -> None."""
    raw = str(value if value is not None else "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def num_int(value):
    """Parse a count-like metrics value (test_size) to int; "" / junk -> None."""
    parsed = num(value)
    return None if parsed is None else int(parsed)


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def first_metrics_row(run_dir):
    """First row of a run's ``metrics.csv``, or None."""
    if run_dir is None:
        return None
    metrics = Path(run_dir) / "metrics.csv"
    if not metrics.exists():
        return None
    try:
        with metrics.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                return row
    except Exception:
        return None
    return None


def has_test_metrics(row) -> bool:
    """Mirrors ``_has_test_metrics`` in ``e14_main_matrix.py``."""
    if not row:
        return False
    return bool(str(row.get("test_mse", "")).strip()) and bool(
        str(row.get("test_mae", "")).strip()
    )


def rel_run_dir(run_dir):
    return repo_relative(run_dir) if run_dir else ""


# --------------------------------------------------------------------------
# Cell identity
# --------------------------------------------------------------------------

def cell_tuple(cell):
    return (
        str(cell["arm"]),
        str(cell["dataset"]),
        int(cell["horizon"]),
        int(cell["seed"]),
    )


def cell_key(cell) -> str:
    """Same key shape as ``e14_main_matrix.cell_key``."""
    arm, dataset, horizon, seed = cell_tuple(cell)
    return f"{arm}__{dataset}-{horizon}-s{seed}"


def cell_token(cell) -> str:
    arm, dataset, horizon, seed = cell_tuple(cell)
    return f"{arm}:{dataset}:{horizon}:{seed}"


def parse_cell_token(token):
    """Accept both ``arm:dataset:horizon:seed`` and E8's ``dataset:horizon:seed:arm``."""
    parts = [part.strip() for part in str(token).split(":")]
    if len(parts) != 4:
        raise SystemExit(f"cell token must have 4 ':'-separated fields: {token!r}")
    try:
        if parts[0] in ARMS:
            arm, dataset, horizon, seed = parts[0], parts[1], int(parts[2]), int(parts[3])
        elif parts[3] in ARMS:
            dataset, horizon, seed, arm = parts[0], int(parts[1]), int(parts[2]), parts[3]
        else:
            raise ValueError("no field names a known arm")
    except ValueError as exc:
        raise SystemExit(f"cannot parse cell token {token!r}: {exc}") from None
    return {"arm": arm, "dataset": dataset, "horizon": horizon, "seed": seed,
            "status": "", "source": None}


def empty_payload(cell) -> dict:
    """Every output field present, so results.csv can never drop a column."""
    arm, dataset, horizon, seed = cell_tuple(cell)
    return {
        "key": cell_key(cell),
        "arm": arm,
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "setting": f"{dataset}-{horizon}",
        "status": "",
        "reason": "",
        "plan": "",
        # outputs
        "test_mse": None,
        "test_mae": None,
        "nlinear_mse": None,
        "nlinear_mae": None,
        "gate_value": None,
        "val_mse": None,
        "recorded_val_mse": None,
        "val_relative_difference": None,
        "test_size": None,
        "run_dir": "",
        "config_hash": "",
        "source": "",
        # provenance / audit extras
        "gate_value_source": "",
        "nlinear_branch_present": None,
        "checkpoint": "",
        "recorded_val_mse_present": None,
        "checkpoint_missing_keys": [],
        "checkpoint_unexpected_keys": [],
        "eval_count": None,
        "val_size": None,
        "device": "",
        "gpu_name": "",
        "data_root": "",
        "matched_run_dirs": [],
        "near_miss_runs": [],
        "reuse_evidence": "",
        "reuse_evidence_status": "",
        "reuse_manifest_agreement": None,
        "protocol_failures": [],
        "protocol_drift_allowed": False,
        "self_read_during_training": False,
        "val_reproduce_tol": VAL_REPRODUCE_TOL,
        "val_reproduce_warn": VAL_REPRODUCE_WARN,
        "read_at": "",
        "warnings": [],
    }


# --------------------------------------------------------------------------
# Arm fingerprint: exact mirror of e14_main_matrix.ARMS / _arm_match()
# --------------------------------------------------------------------------

def arm_match(config, arm: str) -> bool:
    """Verbatim mirror of ``_arm_match`` in ``e14_main_matrix.py``.

    Kept as a literal copy (rather than routed through
    ``arm_exclusion_reasons``) so the two files can be diffed constant by
    constant; ``fingerprint_report()`` additionally proves behavioural parity
    against the stage-A module over ``PARITY_CONFIGS``.
    """
    spec = ARMS[arm]
    hyper = config.get("hyperparams", {}) or {}
    if config.get("mechanism") != spec["mechanism"]:
        return False
    # Exclude frozen-subspace / projection arms and any input intervention.
    if hyper.get("weak_residual_projection"):
        return False
    if str(hyper.get("input_hypothesis", "none") or "none") not in ("none", ""):
        return False
    if float(hyper.get("weak_period_residual_smooth_ratio", 0.0) or 0.0) != 0.0:
        return False
    if config.get("init_checkpoint"):
        return False
    head = spec.get("head")
    if head is None:
        return True
    if hyper.get("weak_period_residual_head_type") != head:
        return False
    div = spec.get("rank_div")
    if div is None:
        return True
    rank = hyper.get("weak_period_residual_rank")
    if rank is None or int(rank) != int(config["horizon"]) // div:
        return False
    return int(hyper.get("weak_period_residual_pool_factor", 1) or 1) == 1


def arm_exclusion_reasons(config, arm: str) -> list:
    """Diagnostic twin of ``arm_match``: why a run is *not* this arm.

    Used only to explain `missing_run` / near-miss runs in the audit output.
    """
    reasons = []
    spec = ARMS[arm]
    hyper = config.get("hyperparams", {}) or {}
    if config.get("mechanism") != spec["mechanism"]:
        reasons.append(f"mechanism={config.get('mechanism')} != {spec['mechanism']}")
    if hyper.get("weak_residual_projection"):
        reasons.append(
            "weak_residual_projection="
            f"{hyper.get('weak_residual_projection')} (frozen-subspace arm, not an "
            "E14 §4.2 cell)"
        )
    if str(hyper.get("input_hypothesis", "none") or "none") not in ("none", ""):
        reasons.append(f"input_hypothesis={hyper.get('input_hypothesis')}")
    if float(hyper.get("weak_period_residual_smooth_ratio", 0.0) or 0.0) != 0.0:
        reasons.append(
            "weak_period_residual_smooth_ratio="
            f"{hyper.get('weak_period_residual_smooth_ratio')}"
        )
    if config.get("init_checkpoint"):
        reasons.append(f"init_checkpoint={config.get('init_checkpoint')}")
    head = spec.get("head")
    if head is not None and hyper.get("weak_period_residual_head_type") != head:
        reasons.append(
            f"weak_period_residual_head_type={hyper.get('weak_period_residual_head_type')}"
            f" != {head}"
        )
    div = spec.get("rank_div")
    if div is not None:
        horizon = config.get("horizon")
        try:
            want_rank = int(horizon) // int(div)
        except Exception:
            want_rank = None
        rank = hyper.get("weak_period_residual_rank")
        if rank is None or int(rank) != want_rank:
            reasons.append(
                f"weak_period_residual_rank={rank} != horizon//{div}={want_rank}"
            )
        if int(hyper.get("weak_period_residual_pool_factor", 1) or 1) != 1:
            reasons.append(
                "weak_period_residual_pool_factor="
                f"{hyper.get('weak_period_residual_pool_factor')} != 1"
            )
    return reasons


def protocol_failures(config) -> list:
    """Mirrors ``_protocol_ok`` in ``e14_main_matrix.py`` (minipaper §4.0)."""
    checks = (
        ("lookback", LOOKBACK),
        ("loss", LOSS),
        ("max_epochs", MAX_EPOCHS),
        ("percent", PERCENT),
        ("period", PERIOD),
    )
    return [f"{name}={config.get(name)} != {want}" for name, want in checks
            if config.get(name) != want]


# Synthetic configs covering every arm and every exclusion rule of ``_arm_match``.
PARITY_CONFIGS = (
    ("phase_only", {"mechanism": "no_residual", "horizon": 96, "hyperparams": {}}),
    ("phase_only", {"mechanism": "no_residual", "horizon": 96,
                    "hyperparams": {"weak_residual_projection": "frozen_subspace"}}),
    ("phase_only", {"mechanism": "weak_residual", "horizon": 96,
                    "hyperparams": {"weak_period_residual_head_type": "shared"}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "hyperparams": {"weak_period_residual_head_type": "shared"}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192, "hyperparams": {}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 48,
                                "weak_period_residual_pool_factor": 1}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "hyperparams": {"weak_period_residual_head_type": "shared",
                                "weak_residual_projection": "frozen_subspace"}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "hyperparams": {"weak_period_residual_head_type": "shared",
                                "input_hypothesis": "d1"}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "hyperparams": {"weak_period_residual_head_type": "shared",
                                "weak_period_residual_smooth_ratio": 0.5}}),
    ("l_main", {"mechanism": "weak_residual", "horizon": 192,
                "init_checkpoint": "somewhere.ckpt",
                "hyperparams": {"weak_period_residual_head_type": "shared"}}),
    ("l_q1_4", {"mechanism": "weak_residual", "horizon": 720,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 180,
                                "weak_period_residual_pool_factor": 1}}),
    ("l_q1_4", {"mechanism": "weak_residual", "horizon": 720,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 90,
                                "weak_period_residual_pool_factor": 1}}),
    ("l_q1_4", {"mechanism": "weak_residual", "horizon": 720,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 180,
                                "weak_period_residual_pool_factor": 2}}),
    ("l_q1_4", {"mechanism": "weak_residual", "horizon": 720,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_pool_factor": 1}}),
    ("l_q1_8", {"mechanism": "weak_residual", "horizon": 720,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 90,
                                "weak_period_residual_pool_factor": 1}}),
    ("l_q1_8", {"mechanism": "weak_residual", "horizon": 96,
                "hyperparams": {"weak_period_residual_head_type": "pooled_lowrank",
                                "weak_period_residual_rank": 12,
                                "weak_period_residual_pool_factor": 1}}),
    ("l_rcrf", {"mechanism": "rcrf_nlinear_plain", "horizon": 336, "hyperparams": {}}),
    ("l_rcrf", {"mechanism": "rcrf_nlinear_plain", "horizon": 336,
                "hyperparams": {"weak_residual_projection": "frozen_subspace"}}),
    ("a1", {"mechanism": "gold_combo_reliability_s2", "horizon": 336,
            "hyperparams": {"weak_period_residual_gate_init": 0.2}}),
    ("a1", {"mechanism": "gold_combo_fixed", "horizon": 336, "hyperparams": {}}),
)


def fingerprint_report() -> dict:
    """Compare this script's arm/protocol tables with stage A's, then parity-test.

    Pure Python: it imports ``e14_main_matrix`` (stdlib only) and exercises both
    ``_arm_match`` implementations over ``PARITY_CONFIGS``.
    """
    report = {"sibling": str(STAGE_A_SCRIPT), "imported": False,
              "constants_equal": None, "differences": [], "parity_cases": 0,
              "parity_failures": []}
    module = None
    try:
        spec = importlib.util.spec_from_file_location("e14_main_matrix", STAGE_A_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report["imported"] = True
    except Exception as exc:  # pragma: no cover - reported, never fatal
        report["differences"].append(f"sibling import failed: {exc!r}")
        report["constants_equal"] = False
        return report

    if dict(module.ARMS) != dict(ARMS):
        report["differences"].append("ARMS differs from e14_main_matrix.ARMS")
    if tuple(module.EXTERNAL_TEST_EVIDENCE) != tuple(EXTERNAL_TEST_EVIDENCE):
        report["differences"].append(
            "EXTERNAL_TEST_EVIDENCE differs from e14_main_matrix.EXTERNAL_TEST_EVIDENCE"
        )
    for name in ("LOOKBACK", "PERIOD", "MAX_EPOCHS", "LOSS", "PERCENT"):
        if getattr(module, name, object()) != globals()[name]:
            report["differences"].append(
                f"{name}={globals()[name]} != e14_main_matrix.{name}="
                f"{getattr(module, name, None)}"
            )

    for case_index, (arm, config) in enumerate(PARITY_CONFIGS):
        report["parity_cases"] += 1
        mine = arm_match(config, arm)
        theirs = module._arm_match(config, arm)
        reasons_agree = (not arm_exclusion_reasons(config, arm)) == mine
        if mine != theirs or not reasons_agree:
            report["parity_failures"].append(
                {"case": case_index, "arm": arm, "ours": mine, "stage_a": theirs,
                 "exclusion_reasons_agree": reasons_agree}
            )
    report["constants_equal"] = not report["differences"]
    return report


# --------------------------------------------------------------------------
# Registered external test evidence
# --------------------------------------------------------------------------

def load_external_evidence() -> tuple[dict, list]:
    """Index externally registered single-test-read results.

    Mirrors ``load_external_test_evidence`` in ``e14_main_matrix.py`` and keys
    the rows by the *E14* arm name.  Returns ``(index, rejected_rows)``.
    """
    index: dict = {}
    rejected: list = []
    for spec in EXTERNAL_TEST_EVIDENCE:
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        try:
            handle = path.open(newline="", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for row in csv.DictReader(handle):
                arm_value = str(row.get(spec["arm_column"], "")).strip()
                arms = [a for a, value in spec["arm_values"].items()
                        if value == arm_value]
                if not arms:
                    continue
                where = {k: row.get(k) for k in ("dataset", "horizon", "seed")}
                if not str(row.get("test_mse", "")).strip():
                    rejected.append({"source": spec["path"], "reason": "empty test_mse",
                                     "row": where})
                    continue
                status = str(row.get(spec["status_column"], "")).strip()
                if status not in spec["status_ok"]:
                    rejected.append({"source": spec["path"],
                                     "reason": f"test_read_status={status!r}",
                                     "row": where})
                    continue
                raw_val = str(row.get(spec["val_column"], "")).strip()
                if raw_val:
                    try:
                        if float(raw_val) > spec["val_tolerance"]:
                            rejected.append({
                                "source": spec["path"],
                                "reason": f"val_relative_difference {raw_val} > "
                                          f"{spec['val_tolerance']}",
                                "row": where,
                            })
                            continue
                    except ValueError:
                        pass
                try:
                    key = (arms[0], row["dataset"], int(row["horizon"]),
                           int(row["seed"]))
                except (KeyError, TypeError, ValueError):
                    rejected.append({"source": spec["path"],
                                     "reason": "unparsable dataset/horizon/seed",
                                     "row": where})
                    continue
                index[key] = {
                    "evidence": spec["path"],
                    "protocol": spec["protocol"],
                    "arm_value": arm_value,
                    "status": status,
                    "test_mse": num(row.get("test_mse")),
                    "test_mae": num(row.get("test_mae")),
                    "nlinear_mse": num(row.get("nlinear_mse")),
                    "nlinear_mae": num(row.get("nlinear_mae")),
                    "gate_value": num(row.get("gate_value")),
                    "val_mse": num(row.get("val_mse")),
                    "val_relative_difference": num(raw_val),
                    "run_dir": str(row.get("run_dir", "") or ""),
                }
    return index, rejected


# --------------------------------------------------------------------------
# Run location
# --------------------------------------------------------------------------

def locate_run(output_root: Path, cell):
    """Find the stage-A run directory of one cell.

    Returns ``(run_dir, config, matched, near_misses)``.  Only configs that pass
    the full arm fingerprint are *matched*; configs for the same
    (dataset, horizon, seed) that fail it are returned as ``near_misses`` with
    the reasons, so a `missing_run` verdict is auditable (in particular, the
    frozen-subspace arms excluded by requirement).
    """
    arm, dataset, horizon, seed = cell_tuple(cell)
    runs_dir = Path(output_root) / "runs"
    matched, near = [], []
    if runs_dir.is_dir():
        for config_path in sorted(runs_dir.glob("*/config.json")):
            config = load_json(config_path)
            if not isinstance(config, dict):
                continue
            if config.get("dataset") != dataset:
                continue
            try:
                if int(config.get("horizon", -1)) != horizon:
                    continue
                if int(config.get("seed", -1)) != seed:
                    continue
            except (TypeError, ValueError):
                continue
            if arm_match(config, arm):
                matched.append((config_path.parent, config))
            else:
                near.append({
                    "run_dir": rel_run_dir(config_path.parent),
                    "reasons": arm_exclusion_reasons(config, arm),
                })
    if not matched:
        return None, None, [], near

    def sort_key(item):
        run_dir, _config = item
        row = first_metrics_row(run_dir)
        # A run whose metrics.csv already records a checkpoint is preferred; ties
        # are broken on the directory name for determinism.
        return (0 if (row or {}).get("checkpoint") else 1, str(run_dir))

    matched.sort(key=sort_key)
    run_dir, config = matched[0]
    return run_dir, config, [rel_run_dir(d) for d, _ in matched], near


def resolve_checkpoint(record, run_dir: Path):
    """Find the best-validation checkpoint of a run.

    ``metrics.csv`` stores a repo-relative path recorded when the run finished.
    The runs are relocatable, so fall back to resolving the same relative path
    against the run's own directory, then to the run's ``attempts/*/checkpoints``
    (the E8 template hardcoded ``attempts/001``; stage A may keep later attempts).
    """
    recorded = str((record or {}).get("checkpoint") or "").strip()
    if not recorded:
        return None
    raw = Path(recorded)
    candidates = [ROOT / raw, run_dir / raw]
    parts = raw.parts
    for index, part in enumerate(parts):
        if part == "runs" and index + 1 < len(parts):
            candidates.append(run_dir / Path(*parts[index + 1:]))
            break
    name = raw.name
    candidates.append(run_dir / "checkpoints" / name)
    candidates.extend(sorted((run_dir / "attempts").glob(f"*/checkpoints/{name}")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Reused cells: copy the registered test metrics, never re-read
# --------------------------------------------------------------------------

def read_reused_cell(cell, evidence_index: dict) -> dict:
    """Copy the already-registered test metrics of a reused cell.

    No checkpoint is loaded and no test split is touched.  Two registration
    styles exist (see ``resolve_reuse`` in ``e14_main_matrix.py``):

    * the source run's own ``metrics.csv`` already carries ``test_mse``/``test_mae``
      (E3 lineage: the ``pooled_lowrank`` q=1/4 and q=1/8 rows);
    * the numbers live in a registered evidence CSV and the source run's inline
      columns are empty (E8 lineage: ``phase_only`` 6 settings and ``l_main``'s
      ``direct_nlinear`` control).

    For the latter the row is re-read from the evidence file when it is available
    and cross-checked against the copy stage A froze into the manifest; when the
    evidence file is not on this machine the manifest's registered copy is used
    and a warning is recorded.
    """
    payload = empty_payload(cell)
    arm, dataset, horizon, seed = cell_tuple(cell)
    source = cell.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    source_run_dir = str(source.get("run_dir") or "")
    descriptor = source.get("test_evidence")

    # --- style 1: the source run's own metrics.csv carries the test read ------
    inline = isinstance(descriptor, str) and "inline" in descriptor
    if inline:
        src_dir = ROOT / source_run_dir if source_run_dir else None
        row = first_metrics_row(src_dir)
        if not has_test_metrics(row):
            payload["status"] = "missing_reused_source"
            payload["reason"] = (
                f"reuse source {source_run_dir or '<unset>'}/metrics.csv does not "
                "carry test_mse/test_mae; the registered inline test read is gone"
            )
            return payload
        payload.update({
            "status": STATUS_REUSED,
            "run_dir": source_run_dir,
            "source": f"{source_run_dir}/metrics.csv",
            "reuse_evidence": "inline metrics.csv",
            "test_mse": num(row.get("test_mse")),
            "test_mae": num(row.get("test_mae")),
            "test_size": num_int(row.get("test_size")),
            "checkpoint": str(row.get("checkpoint", "") or ""),
            "config_hash": str(source.get("config_hash")
                               or row.get("config_hash") or ""),
            "recorded_val_mse": num(row.get("val_mse")),
            "read_at": "",
        })
        return payload

    # --- style 2: a registered evidence CSV holds the numbers -----------------
    manifest_evidence = descriptor if isinstance(descriptor, dict) else {}
    recheck = evidence_index.get((arm, dataset, horizon, seed))
    if recheck is not None:
        payload.update({
            "status": STATUS_REUSED,
            "source": recheck["evidence"],
            "reuse_evidence": recheck["evidence"],
            "reuse_evidence_status": str(recheck["status"]),
            "test_mse": recheck["test_mse"],
            "test_mae": recheck["test_mae"],
            "nlinear_mse": recheck["nlinear_mse"],
            "nlinear_mae": recheck["nlinear_mae"],
            "gate_value": recheck["gate_value"],
            "gate_value_source": "registered",
            "recorded_val_mse": recheck["val_mse"],
            "val_relative_difference": recheck["val_relative_difference"],
            "run_dir": source_run_dir or recheck["run_dir"],
            "config_hash": str(source.get("config_hash") or ""),
        })
        manifest_mse = num(manifest_evidence.get("test_mse"))
        if manifest_mse is not None and recheck["test_mse"] is not None:
            agree = abs(manifest_mse - recheck["test_mse"]) <= 1e-12
            payload["reuse_manifest_agreement"] = agree
            if not agree:
                payload["warnings"].append(
                    "manifest test_mse "
                    f"{manifest_mse!r} != registered {recheck['test_mse']!r}"
                )
        if not payload["config_hash"] and payload["run_dir"]:
            config = load_json(ROOT / payload["run_dir"] / "config.json")
            if isinstance(config, dict):
                payload["config_hash"] = str(config.get("config_hash") or "")
        return payload

    manifest_mse = num(manifest_evidence.get("test_mse"))
    manifest_mae = num(manifest_evidence.get("test_mae"))
    if manifest_mse is None or manifest_mae is None:
        payload["status"] = "missing_reused_source"
        payload["reason"] = (
            f"reused cell has neither an inline test read in "
            f"{source_run_dir or '<unset>'}/metrics.csv nor a registered evidence "
            f"row for arm={arm} in {[s['path'] for s in EXTERNAL_TEST_EVIDENCE]}"
        )
        return payload
    payload.update({
        "status": STATUS_REUSED,
        "source": str(manifest_evidence.get("evidence", "") or ""),
        "reuse_evidence": "manifest registered copy",
        "reuse_evidence_status": str(manifest_evidence.get("status", "") or ""),
        "test_mse": manifest_mse,
        "test_mae": manifest_mae,
        "val_relative_difference": num(manifest_evidence.get("val_relative_difference")),
        "run_dir": source_run_dir,
        "config_hash": str(source.get("config_hash") or ""),
    })
    payload["warnings"].append(
        "reuse evidence not re-verified on this machine: used the registered copy "
        "frozen into stage_a_manifest.json"
    )
    return payload


# --------------------------------------------------------------------------
# New cells: torch-free pre-flight
# --------------------------------------------------------------------------

def preflight_new_cell(cell, output_root: Path, ignore_protocol_drift: bool):
    """Everything a new cell needs before any GPU work.

    Returns ``(payload, context)``.  ``context`` is None when the cell must not
    be read (the payload already carries the terminal status); otherwise it holds
    the run directory, config, metrics row and resolved checkpoint.  Importing no
    torch is what lets ``--dry-run`` validate the plan anywhere.
    """
    payload = empty_payload(cell)
    arm, dataset, horizon, seed = cell_tuple(cell)
    if arm not in ARMS:
        payload["status"] = "unknown_arm"
        payload["reason"] = f"arm {arm!r} is not an E14 §4.2 row"
        return payload, None

    run_dir, config, matched, near = locate_run(output_root, cell)
    payload["near_miss_runs"] = near
    if run_dir is None:
        payload["status"] = "missing_run"
        payload["reason"] = (
            f"no run under {rel_run_dir(output_root)}/runs matches arm={arm} "
            f"dataset={dataset} horizon={horizon} seed={seed}; "
            f"{len(near)} same-cell config(s) were excluded by the arm fingerprint"
        )
        return payload, None
    payload["matched_run_dirs"] = matched
    payload["run_dir"] = rel_run_dir(run_dir)
    payload["config_hash"] = str(config.get("config_hash") or "")

    failures = protocol_failures(config)
    payload["protocol_failures"] = failures
    if failures and not ignore_protocol_drift:
        payload["status"] = "rejected"
        payload["reason"] = (
            "protocol mismatch vs minipaper §4.0: " + "; ".join(failures)
            + " (pass --ignore-protocol-drift only with a documented reason)"
        )
        return payload, None
    if failures:
        payload["protocol_drift_allowed"] = True
        payload["warnings"].append("protocol drift allowed: " + "; ".join(failures))

    record = first_metrics_row(run_dir)
    if record is None:
        payload["status"] = "missing_metrics"
        payload["reason"] = f"{payload['run_dir']}/metrics.csv is missing or empty"
        return payload, None

    if has_test_metrics(record):
        # Stage A must never pass --evaluate-test.  If it did, the checkpoint was
        # not frozen before the read; the numbers are copied but flagged, and the
        # split is not read a second time.
        payload.update({
            "status": STATUS_ALREADY_READ,
            "self_read_during_training": True,
            "reason": (
                "run metrics.csv already carries test_mse/test_mae: stage A "
                "evaluated test before the matrix was frozen; copying that read "
                "instead of reading again"
            ),
            "source": f"{payload['run_dir']}/metrics.csv",
            "test_mse": num(record.get("test_mse")),
            "test_mae": num(record.get("test_mae")),
            "test_size": num_int(record.get("test_size")),
            "checkpoint": str(record.get("checkpoint", "") or ""),
            "recorded_val_mse": num(record.get("val_mse")),
        })
        return payload, None

    checkpoint = resolve_checkpoint(record, run_dir)
    if checkpoint is None:
        payload["status"] = "missing_checkpoint"
        payload["reason"] = (
            f"best-validation checkpoint not found for recorded path "
            f"{record.get('checkpoint')!r}"
        )
        return payload, None

    payload["recorded_val_mse"] = num(record.get("val_mse"))
    payload["recorded_val_mse_present"] = payload["recorded_val_mse"] is not None
    context = {
        "run_dir": run_dir,
        "config": config,
        "record": record,
        "checkpoint": checkpoint,
    }
    return payload, context


# --------------------------------------------------------------------------
# New cells: rebuild, gate on validation, then read test once
# --------------------------------------------------------------------------

def build_model(config, checkpoint_path, num_workers: int):
    """Rebuild the exact trained model, then restore its best checkpoint.

    Mirrors the training runner's construction order (``search_phaseformer.py``):
    ``set_float32_matmul_precision("medium")`` and the ETT data-root fallback, so
    the re-derived validation MSE is computed under the same numeric protocol
    that produced the run's recorded value.
    """
    import torch

    from src.dataset.data_factory import data_provider
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

    torch.set_float32_matmul_precision("medium")

    hyper = dict(config["hyperparams"])
    if hyper.get("weak_residual_projection"):
        raise RuntimeError(
            "refusing to rebuild a frozen-subspace arm: "
            f"weak_residual_projection={hyper['weak_residual_projection']!r}.  "
            "Those runs are not E14 §4.2 cells; its projection basis artifact is "
            "not part of the E14 stage-A output."
        )
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hyper,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = num_workers

    # The training runner falls back to the ETT-small directory when the
    # configured ETT root is absent; without it a rebuild can read different
    # files than training did (or fail outright).
    configured_root = Path(exp_args.dataset_args.root_path)
    if not configured_root.exists() and str(config["dataset"]).startswith("ETT"):
        fallback = ROOT / "resources" / "all_datasets" / "ETT-small"
        if fallback.exists():
            exp_args.dataset_args.root_path = str(fallback)

    train_set, _ = data_provider(exp_args.dataset_args, "train")
    if hasattr(train_set, "data_stamp"):
        hyper["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(
        PhaseFormerPresetConfig(exp_args, config["lookback"], config["horizon"], hyper)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompat.unexpected_keys}")
    return model, exp_args, list(incompat.missing_keys)


def evaluate_once(model, loader, split, target_var_index, static_gate):
    """MSE/MAE of the fused output and of the residual branch alone.

    ``nlinear_mse`` / ``nlinear_mae`` are the NLinear weak-period branch's own
    error; ``gate_value`` is the static fusion gate when the arm has one, and the
    mean input-dependent fusion weight observed during this pass otherwise
    (RCRF / A1 couples the gate to per-sample reliability, so their weight is a
    tensor, not a parameter).
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    fused_sq = fused_abs = branch_sq = branch_abs = 0.0
    count = 0
    branch_count = 0
    gate_sum = 0.0
    gate_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            pred = out[:, -model.pred_len:, :]
            true = batch_y.float()[:, -model.pred_len:, :]
            branch = model.last_residual_forecast
            if target_var_index != -1:
                true = true[:, :, target_var_index: target_var_index + 1]
                if branch is not None:
                    branch = branch[:, :, target_var_index: target_var_index + 1]
            err = pred - true
            fused_sq += float(err.pow(2).sum())
            fused_abs += float(err.abs().sum())
            count += err.numel()
            if branch is not None:
                branch_err = branch - true
                branch_sq += float(branch_err.pow(2).sum())
                branch_abs += float(branch_err.abs().sum())
                branch_count += branch_err.numel()
            alpha = getattr(model, "last_weak_residual_alpha", None)
            if alpha is None:
                alpha = getattr(model, "last_rcrf_alpha", None)
            if alpha is not None:
                gate_sum += float(alpha.detach().float().sum())
                gate_count += int(alpha.numel())
    dynamic_gate = (gate_sum / gate_count) if gate_count else None
    if static_gate is not None:
        gate_value, gate_source = static_gate, "static_gate"
    elif dynamic_gate is not None:
        gate_value, gate_source = dynamic_gate, "input_dependent_mean"
    else:
        gate_value, gate_source = None, "none"
    return {
        f"{split}_mse": fused_sq / max(count, 1),
        f"{split}_mae": fused_abs / max(count, 1),
        "nlinear_mse": (branch_sq / branch_count) if branch_count else None,
        "nlinear_mae": (branch_abs / branch_count) if branch_count else None,
        "gate_value": gate_value,
        "gate_value_source": gate_source,
        "nlinear_branch_present": branch_count > 0,
        "eval_count": count,
    }


def read_new_cell(cell, output_root: Path, artifact_dir: Path,
                  ignore_protocol_drift: bool, num_workers: int) -> dict:
    """Rebuild one new cell, gate on validation, then read its test split once."""
    # Pre-flight first and without torch: a cell with no run, no metrics or no
    # checkpoint must produce a clean payload (and exit 0) rather than a worker
    # crash, so the audit trail stays readable instead of becoming a traceback.
    payload, context = preflight_new_cell(cell, output_root, ignore_protocol_drift)
    if context is None:
        return payload

    # The validation reproduction needs the run's own recorded val_mse; without it
    # the checkpoint/protocol cannot be verified, so the test split is not read.
    if payload["recorded_val_mse"] is None:
        payload["status"] = "rejected"
        payload["reason"] = (
            "metrics.csv has no val_mse, so the restored checkpoint cannot be "
            "verified against the audited protocol; refusing to read test"
        )
        return payload
    if not payload["recorded_val_mse"]:
        payload["status"] = "rejected"
        payload["reason"] = (
            "metrics.csv records val_mse 0, so the relative reproduction gap is "
            "undefined; refusing to read test"
        )
        return payload

    import torch

    from src.dataset.data_factory import data_provider

    model, exp_args, missing_keys = build_model(
        context["config"], context["checkpoint"], num_workers
    )
    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES is set per worker by the parent, so device 0 is
        # exactly the GPU this cell was dispatched to.
        torch.cuda.set_device(0)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    payload["device"] = str(device)
    payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    payload["data_root"] = str(exp_args.dataset_args.root_path)
    payload["checkpoint"] = repo_relative(context["checkpoint"])
    payload["checkpoint_missing_keys"] = missing_keys
    static_gate = model.learned_residual_gate()

    # Determinism guard: recompute validation first.  The value must reproduce the
    # training run's own recorded val_mse, otherwise the test number this script
    # would produce does not belong to the audited protocol and the cell is
    # reported as rejected instead of being silently accepted.  The test split is
    # only read after this gate passes.
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    val_result = evaluate_once(
        model, val_loader, "val", model.target_var_index, static_gate
    )
    payload["val_size"] = len(val_set)
    recomputed_val = val_result["val_mse"]
    recorded_val = payload["recorded_val_mse"]
    relative = abs(recomputed_val - recorded_val) / recorded_val
    payload.update({
        "val_mse": recomputed_val,
        "val_relative_difference": relative,
        "nlinear_branch_present": val_result["nlinear_branch_present"],
    })
    if not math.isfinite(relative):
        payload["status"] = "rejected"
        payload["reason"] = (
            f"recomputed val_mse {recomputed_val!r} is not finite; the checkpoint "
            "or the data protocol is wrong -- test not read"
        )
        return payload
    if relative >= VAL_REPRODUCE_TOL:
        payload["status"] = "rejected"
        payload["reason"] = (
            f"recomputed val_mse {recomputed_val!r} differs from recorded "
            f"{recorded_val!r} by {relative!r} >= VAL_REPRODUCE_TOL "
            f"{VAL_REPRODUCE_TOL}; checkpoint or protocol mismatch -- test not read"
        )
        return payload

    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    test_result = evaluate_once(
        model, test_loader, "test", model.target_var_index, static_gate
    )
    payload.update({
        "status": STATUS_VAL_DRIFT if relative >= VAL_REPRODUCE_WARN else STATUS_READ,
        "test_mse": test_result["test_mse"],
        "test_mae": test_result["test_mae"],
        "nlinear_mse": test_result["nlinear_mse"],
        "nlinear_mae": test_result["nlinear_mae"],
        "gate_value": test_result["gate_value"],
        "gate_value_source": test_result["gate_value_source"],
        "nlinear_branch_present": test_result["nlinear_branch_present"],
        "test_size": len(test_set),
        "eval_count": test_result["eval_count"],
        "source": "stage_b_single_test_read",
        "read_at": now_iso(),
    })
    if payload["status"] == STATUS_VAL_DRIFT:
        payload["warnings"].append(
            f"val_relative_difference {relative!r} >= VAL_REPRODUCE_WARN "
            f"{VAL_REPRODUCE_WARN} but < VAL_REPRODUCE_TOL {VAL_REPRODUCE_TOL}"
        )
    return payload


# --------------------------------------------------------------------------
# Pre-flight plan (torch-free)
# --------------------------------------------------------------------------

def process_new_cell(cell, output_root: Path, artifact_dir: Path, args):
    """Idempotent single-cell read shared by the worker and in-process paths.

    Returns ``(payload, freshly_read)``.  A cell whose artifact already recorded
    a consumed test read (``read`` / ``val_drift`` / ``already_read``) is returned
    unchanged, which is what makes re-running stage B safe: the "exactly once"
    rule survives an operator re-invoking the script.  A cell whose previous
    attempt ended in ``rejected`` or ``missing_*`` never consumed the test split,
    so it is legitimately retried.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    key = cell_key(cell)
    if str(cell.get("status", "") or "") in PROBLEM_STATUSES and cell.get("reason"):
        # Already terminal before any GPU work (e.g. a --cell token that is not in
        # the manifest); keep the reason instead of overwriting it.
        return dict(cell), False
    previous = load_json(artifact_dir / f"{key}.json")
    if isinstance(previous, dict) and previous.get("status") in CONSUMED_STATUSES:
        return dict(previous, already_read_from_artifact=True), False
    payload = read_new_cell(cell, output_root, artifact_dir,
                            args.ignore_protocol_drift, args.num_workers)
    write_json(artifact_dir / f"{payload['key']}.json", payload)
    return payload, True


def plan_cell(cell, output_root: Path, evidence_index: dict, args):
    """Describe what stage B would do for one cell, without any GPU work."""
    status = str(cell.get("status", "") or "").strip()
    if status in PROBLEM_STATUSES and cell.get("reason"):
        # Already terminal (e.g. a --cell token that is not in the manifest).
        return dict(cell)
    if status == STATUS_REUSED:
        entry = read_reused_cell(cell, evidence_index)
        entry["plan"] = (
            "copy the already-registered test metrics from the reuse source "
            "(no rebuild, no test read)"
        )
        return entry
    if status == "new":
        entry, context = preflight_new_cell(cell, output_root,
                                            args.ignore_protocol_drift)
        entry["plan"] = (
            "rebuild from config.json + restore best checkpoint, re-derive "
            "val_mse, then read test exactly once"
        )
        if context is not None:
            entry["status"] = STATUS_PLANNED
            entry["checkpoint"] = repo_relative(context["checkpoint"])
        return entry
    entry = empty_payload(cell)
    entry["status"] = "unknown_manifest_status"
    entry["reason"] = (
        f"manifest cell status {status!r} is neither 'new' nor 'reused'"
    )
    return entry


def classify(entries):
    """Split plan/result entries into accepted cells, problems and warnings."""
    accepted, problems, warnings = [], [], []
    for entry in entries:
        status = entry.get("status")
        if status in PROBLEM_STATUSES:
            problems.append({"cell": entry.get("key"), "status": status,
                             "reason": entry.get("reason")})
        elif status in OK_STATUSES:
            accepted.append(entry)
            if entry.get("self_read_during_training"):
                warnings.append({
                    "cell": entry.get("key"),
                    "warning": "test_mse was already present in the run's own "
                               "metrics.csv: stage A evaluated test before the "
                               "checkpoint matrix was frozen",
                })
        else:
            problems.append({"cell": entry.get("key"), "status": status,
                             "reason": "unrecognised status"})
        for text in entry.get("warnings") or []:
            warnings.append({"cell": entry.get("key"), "warning": text})
    return accepted, problems, warnings


# --------------------------------------------------------------------------
# GPU dispatch: one cell per GPU, retries, poll loop
# --------------------------------------------------------------------------

def dispatch_new_cells(cells, gpus, args, manifest_path: Path, output_root: Path,
                       artifact_dir: Path):
    """One subprocess per cell, one cell per GPU, with retries.

    Same shape as ``e14_main_matrix.dispatch``: a free GPU takes the next pending
    cell, slow cells are handed out first because the manifest preserves stage A's
    expensive-first ordering, and a non-zero exit is retried up to ``--retries``.
    A worker that finishes with a payload (including a rejection) exits 0 and is
    never retried: a rejected cell is a research verdict, not a transient error.
    """
    pending = list(cells)
    active: dict = {}
    attempts: dict = {}
    results: list = []
    failed: list = []
    log_dir = Path(output_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())

    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell_key(cell)
            attempts[key] = attempts.get(key, 0) + 1
            argv = [
                sys.executable, script,
                "--manifest", str(manifest_path),
                "--output-root", str(output_root),
                "--cell", cell_token(cell),
                "--worker",
                "--num-workers", str(args.num_workers),
            ]
            if args.ignore_protocol_drift:
                argv.append("--ignore-protocol-drift")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(log_dir / f"test_read_{key}.log", "w")
            process = subprocess.Popen(
                argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = (cell, process, log)
            print(json.dumps({"event": "launch", "cell": key, "gpu": gpu,
                              "attempt": attempts[key], "pid": process.pid}),
                  flush=True)

        finished = []
        for gpu, (cell, process, log) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(gpu)
            key = cell_key(cell)
            log.close()
            payload = load_json(artifact_dir / f"{key}.json")
            if code == 0 and isinstance(payload, dict):
                results.append(payload)
                print(json.dumps({"event": "done", "cell": key, "gpu": gpu,
                                  "status": payload.get("status")}), flush=True)
            elif attempts[key] <= args.retries:
                pending.append(cell)
                print(json.dumps({"event": "retry", "cell": key, "gpu": gpu,
                                  "code": code}), flush=True)
            else:
                failed.append({"cell": key, "spec": cell,
                               "return_code": code,
                               "attempts": attempts[key],
                               "log": str(log_dir / f"test_read_{key}.log")})
                print(json.dumps({"event": "failed", "cell": key, "gpu": gpu,
                                  "code": code}), flush=True)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(args.poll_seconds)
    return results, failed


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def write_results_csv(path: Path, entries) -> None:
    """Flat one-row-per-cell table with the exact required column order."""
    arm_order = {arm: index for index, arm in enumerate(ARMS)}

    def sort_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    ordered = sorted(
        entries,
        key=lambda e: (arm_order.get(e.get("arm"), 99), str(e.get("dataset")),
                       sort_int(e.get("horizon")), sort_int(e.get("seed"))),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_FIELDS))
        writer.writeheader()
        for entry in ordered:
            writer.writerow({field: csv_value(entry.get(field))
                             for field in RESULTS_FIELDS})


def merge_cells(summary_path: Path, payloads):
    """Merge new per-cell records into any existing summary (incremental runs)."""
    existing = load_json(summary_path) or {}
    merged = {}
    for entry in existing.get("cells", []) or []:
        if isinstance(entry, dict) and entry.get("key"):
            merged[entry["key"]] = entry
    for entry in payloads:
        if entry.get("key"):
            merged[entry["key"]] = entry
    return [merged[key] for key in sorted(merged)]


def write_summary(path: Path, entries, args, manifest_path: Path, manifest,
                  output_root: Path, counts, problems, warnings, fingerprint,
                  evidence_rejected, failed) -> dict:
    summary = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "minipaper": "§4.0 (full-train, best-validation checkpoint, seeds "
                         "2021/2022/2023, one test read per checkpoint)",
            "schedule": "docs/PhaseFormer_L_execution_schedule.md §2.2 (E14 stage B)",
            "note": PROTOCOL_NOTE,
            "one_read_per_new_checkpoint": True,
            "reused_cells_never_reread": True,
            "new_cells_rebuilt_from_config": True,
            "already_read_guard": (
                "a run whose metrics.csv already carries test_mse, or a cell whose "
                "test_read/<key>.json was already written by this script, is never "
                "read again"
            ),
            "val_gate": {
                "checked_before_test_read": True,
                "warn": VAL_REPRODUCE_WARN,
                "tol": VAL_REPRODUCE_TOL,
                "on_failure": (
                    "status 'rejected'; the test split is not read and no test "
                    "number is reported for that cell"
                ),
            },
            "reused_cell_evidence": [spec["path"] for spec in EXTERNAL_TEST_EVIDENCE],
            "protocol_fixed_fields": {
                "lookback": LOOKBACK, "period": PERIOD, "loss": LOSS,
                "max_epochs": MAX_EPOCHS, "percent": PERCENT,
            },
            "num_workers": args.num_workers,
            "arm_fingerprint": {arm: ARMS[arm] for arm in sorted(ARMS)},
            "excluded_runs": (
                "any run with weak_residual_projection set is a frozen-subspace "
                "arm, not an E14 §4.2 cell, and is never matched"
            ),
        },
        "manifest": {
            "path": repo_relative(manifest_path),
            "experiment": (manifest or {}).get("experiment"),
            "reads_test": (manifest or {}).get("reads_test"),
            "cells": len((manifest or {}).get("cells", []) or []),
        },
        "run": {
            "output_root": repo_relative(output_root),
            "gpus": parse_list(args.gpus),
            "retries": args.retries,
            "poll_seconds": args.poll_seconds,
            "dry_run": bool(args.dry_run),
            "generated_at": now_iso(),
        },
        "fingerprint_check": fingerprint,
        "evidence_rejected_rows": evidence_rejected,
        "counts": counts,
        "problems": problems,
        "warnings": warnings,
        "failed_workers": failed,
        "cells": sorted(entries, key=lambda e: e.get("key", "")),
    }
    write_json(path, summary)
    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E14 stage B: read the test split exactly once per new §4.2 cell",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True,
                        help="<output-root>/stage_a_manifest.json (authoritative "
                             "cell list produced by e14_main_matrix.py --stage a)")
    parser.add_argument("--output-root",
                        default="research_runs/phaseformer_L_e14_main_v1",
                        help="stage-A output root: holds runs/, and receives "
                             "test_read_summary.json, results.csv and test_read/")
    parser.add_argument("--gpus", default="",
                        help="comma-separated GPU ids, one cell at a time per GPU "
                             "via subprocesses; empty = run in-process, sequentially")
    parser.add_argument("--retries", type=int, default=1,
                        help="extra attempts for a cell whose worker exits non-zero")
    parser.add_argument("--poll-seconds", type=int, default=15,
                        help="poll interval while workers are active")
    parser.add_argument("--dry-run", action="store_true",
                        help="pre-flight only: validate the manifest, the arm "
                             "fingerprint, run location and reuse evidence; "
                             "imports no torch and writes nothing")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="dataloader workers; 4 matches stage A")
    parser.add_argument("--cell", default="",
                        help="single cell as arm:dataset:horizon:seed (or E8's "
                             "dataset:horizon:seed:arm)")
    parser.add_argument("--cells-file", default="",
                        help="file of cell tokens, one per line (in-process mode)")
    parser.add_argument("--worker", action="store_true",
                        help="internal: process --cell and write only its artifact")
    parser.add_argument("--ignore-protocol-drift", action="store_true",
                        help="downgrade a §4.0 protocol-field mismatch from "
                             "'rejected' to a recorded warning (document why)")
    parser.add_argument("--skip-fingerprint-check", action="store_true",
                        help="do not compare the arm/protocol tables with "
                             "e14_main_matrix.py")
    return parser.parse_args(argv)


def load_manifest_cells(manifest_path: Path):
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"cannot read a JSON manifest at {manifest_path}")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SystemExit(f"manifest {manifest_path} has no non-empty 'cells' list")
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise SystemExit(f"manifest cell #{index} is not an object")
        for field in ("arm", "dataset", "horizon", "seed", "status"):
            if field not in cell:
                raise SystemExit(f"manifest cell #{index} lacks {field!r}")
    return manifest, cells


def select_cells(cells, args):
    if args.cell:
        wanted = parse_cell_token(args.cell)
        for cell in cells:
            if cell_tuple(cell) == cell_tuple(wanted):
                return [cell]
        payload = empty_payload(wanted)
        payload["status"] = "missing_cell"
        payload["reason"] = f"{cell_token(wanted)} is not in the manifest"
        return [payload]
    if args.cells_file:
        selected = []
        for line in Path(args.cells_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wanted = parse_cell_token(line)
            match = [c for c in cells if cell_tuple(c) == cell_tuple(wanted)]
            if match:
                selected.append(match[0])
            else:
                payload = empty_payload(wanted)
                payload["status"] = "missing_cell"
                payload["reason"] = f"{cell_token(wanted)} is not in the manifest"
                selected.append(payload)
        return selected
    return list(cells)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    if not args.skip_fingerprint_check:
        fingerprint = fingerprint_report()
        print(json.dumps({"event": "fingerprint_check", **fingerprint},
                         ensure_ascii=False), flush=True)
    else:
        fingerprint = {"skipped": True}

    manifest, manifest_cells = load_manifest_cells(manifest_path)
    cells = select_cells(manifest_cells, args)

    evidence_index, evidence_rejected = load_external_evidence()
    artifact_dir = output_root / "test_read"
    summary_path = output_root / "test_read_summary.json"
    results_path = output_root / "results.csv"

    manifest_counts = {"new": 0, "reused": 0, "other": 0}
    for cell in manifest_cells:
        bucket = str(cell.get("status", "") or "")
        manifest_counts[bucket if bucket in ("new", "reused") else "other"] += 1
    print(json.dumps({
        "event": "planned",
        "manifest": repo_relative(manifest_path),
        "cells_in_manifest": len(manifest_cells),
        "cells_selected": len(cells),
        "manifest_counts": manifest_counts,
        "reused_cells": sum(1 for c in cells if c.get("status") == "reused"),
        "new_cells": sum(1 for c in cells if c.get("status") == "new"),
        "external_evidence_rows": len(evidence_index),
        "external_evidence_rejected": len(evidence_rejected),
        "gpus": parse_list(args.gpus),
    }, ensure_ascii=False), flush=True)

    if args.worker:
        # Dispatched child: read exactly one cell, write only its artifact, and
        # let the parent own test_read_summary.json / results.csv.
        if not args.cell:
            raise SystemExit("--worker requires --cell")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cell = cells[0]
        if str(cell.get("status", "") or "").strip() == STATUS_REUSED:
            payload = read_reused_cell(cell, evidence_index)
            write_json(artifact_dir / f"{payload['key']}.json", payload)
        else:
            payload, _fresh = process_new_cell(cell, output_root, artifact_dir, args)
        print("E14READ " + json.dumps(payload, sort_keys=True, ensure_ascii=False),
              flush=True)
        return

    if args.dry_run:
        entries = [plan_cell(cell, output_root, evidence_index, args) for cell in cells]
        accepted, problems, warnings = classify(entries)
        for entry in entries:
            print("E14PLAN " + json.dumps(
                {k: entry.get(k) for k in
                 ("key", "status", "plan", "source", "run_dir", "checkpoint",
                  "reuse_evidence", "reason", "near_miss_runs")},
                sort_keys=True, ensure_ascii=False), flush=True)
        print(json.dumps({
            "event": "finished",
            "dry_run": True,
            "cells": len(entries),
            "accepted": len(accepted),
            "problems": len(problems),
            "warnings": len(warnings),
            "by_status": count_by(entries, "status"),
            "by_arm": count_by(entries, "arm"),
            "wrote_outputs": False,
        }, ensure_ascii=False), flush=True)
        if problems:
            print(json.dumps({"event": "dry_run_problems", "problems": problems,
                              "warnings": warnings}, ensure_ascii=False, indent=2),
                  flush=True)
            raise SystemExit("dry run found problems; refusing to proceed")
        return

    payloads = []
    failed = []
    reused_cells = [cell for cell in cells if cell.get("status") == "reused"]
    new_cells = [cell for cell in cells if cell.get("status") == "new"]
    other_cells = [cell for cell in cells
                   if cell.get("status") not in ("reused", "new")]

    # Reused cells need no GPU: copy the registered read in the parent process.
    for cell in reused_cells:
        payload = read_reused_cell(cell, evidence_index)
        payloads.append(payload)
        print(json.dumps({"event": "done", "cell": payload["key"], "gpu": None,
                          "status": payload["status"], "reused": True}),
              flush=True)
    for cell in other_cells:
        payload = plan_cell(cell, output_root, evidence_index, args)
        payloads.append(payload)
        print(json.dumps({"event": "failed", "cell": payload["key"],
                          "status": payload["status"]}), flush=True)

    if new_cells:
        gpus = parse_list(args.gpus)
        if gpus:
            results, failed = dispatch_new_cells(
                new_cells, gpus, args, manifest_path, output_root, artifact_dir
            )
            payloads.extend(results)
        else:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for cell in new_cells:
                key = cell_key(cell)
                payload, fresh = process_new_cell(cell, output_root,
                                                  artifact_dir, args)
                payloads.append(payload)
                if fresh:
                    print("E14READ " + json.dumps(payload, sort_keys=True,
                                                  ensure_ascii=False), flush=True)
                print(json.dumps({"event": "done", "cell": key, "gpu": None,
                                  "status": payload["status"],
                                  "freshly_read": fresh}), flush=True)

    # A worker that never produced a payload still gets a row, so results.csv
    # keeps one row per manifest cell and the gap is visible rather than missing.
    for failure in failed:
        spec = failure.get("spec")
        if spec is None:
            continue
        payload = empty_payload(spec)
        payload["status"] = "worker_failed"
        payload["reason"] = (
            f"worker exited {failure['return_code']} after {failure.get('attempts')} "
            f"attempt(s); see {failure.get('log')}"
        )
        payloads.append(payload)

    merged = merge_cells(summary_path, payloads)
    accepted, problems, warnings = classify(merged)
    counts = {
        "cells": len(merged),
        "accepted": len(accepted),
        "problems": len(problems),
        "warnings": len(warnings),
        "by_status": count_by(merged, "status"),
        "by_arm": count_by(merged, "arm"),
    }
    write_summary(summary_path, merged, args, manifest_path, manifest,
                  output_root, counts, problems, warnings, fingerprint,
                  evidence_rejected, failed)
    write_results_csv(results_path, merged)
    print(json.dumps({
        "event": "finished",
        "dry_run": False,
        "cells": len(merged),
        "accepted": len(accepted),
        "problems": len(problems),
        "warnings": len(warnings),
        "failed_workers": failed,
        "by_status": counts["by_status"],
        "summary": repo_relative(summary_path),
        "results_csv": repo_relative(results_path),
    }, ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit(f"E14 stage B had problems: {problems}")


def count_by(entries, field):
    counts: dict = {}
    for entry in entries:
        key = str(entry.get(field))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
