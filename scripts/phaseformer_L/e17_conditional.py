#!/usr/bin/env python3
"""E17 stage A: train the frozen-subspace arms of minipaper section 4.5.

Implements ``docs/PhaseFormer_L/e17_conditional/01_plan.md`` (which follows
``docs/PhaseFormer_L_execution_schedule.md`` section 2.3).  The four-arm table of
minipaper section 4.5 is assembled from three different lineages plus **24 new
training runs**:

* ``direct``                          -- reused: E14 ``l_main`` (7x3 = 21 cells),
* ``frozen_independent_direction_1``   -- reused: E8 ``keep_direction_1``
  (6x3 = 18 cells) **plus 3 new runs** (Electricity-336, which E8 never covered),
* ``frozen_conditional_direction_1``   -- **new**: 7 settings x 3 seeds = 21 runs,
* ``joint`` (PhaseFormer-L)            -- reused: E14 ``l_main`` (7x3 = 21 cells).

Three properties of this design are disclosed machine-readably in
``e17_summary.json`` (``disclosures``) and in the plan document, because they are
real properties of the implementation rather than caveats:

1. ``direct`` and ``joint`` are **the same configuration** here: PhaseFormer-L's
   corrector *is* the jointly trained unconstrained head, so the two columns
   coincide by construction (same run directory, same ``config_hash`` per cell);
2. the seven settings are test-set-selection-derived, and the reused cells carry
   the D-2 Stage-0-frozen ``(gate, lr)`` rather than the section 4.0 preset
   defaults;
3. both frozen projectors are fitted on the **train split only**.

Protocol (identical to E14): ``--lookback 720 --period 24 --max-epochs 30
--loss huber --percent 100 --stage confirm --require-cuda --resume``, one run per
GPU, retries, a JSON manifest, and ``--dry-run``.  **No cell ever passes
``--evaluate-test``** (``dispatch`` asserts it); the single test read is a
separate stage, and ``--test-read-plan`` only prints the list it would use.

The frozen projection is installed exactly as E8 does it: the
``--basis <path>`` flag plus ``--overrides {"weak_residual_projection":
"frozen_subspace", ...}``.  Both halves are required -- ``search_phaseformer.py``
has no ``--basis`` argument, so the flag is consumed by
``scripts/run_top2_direction_retention.py``, while the override is what lets
``PhaseFormer.__init__`` accept the installed basis (it validates
``configs.weak_residual_projection`` for the ``shared`` head).

Usage::

    # stage 2 gate: build the manifest, verify every reuse cell, train nothing
    python scripts/phaseformer_L/e17_conditional.py --stage plan --verify

    # stage 2 gate, the 24 commands printed one JSON object per line
    python scripts/phaseformer_L/e17_conditional.py --dry-run

    # the real matrix
    CUDA_VISIBLE_DEVICES= python scripts/phaseformer_L/e17_conditional.py \\
        --stage a --gpus 0,1,2,3,4,5,6,7 \\
        --output-root research_runs/phaseformer_L_e17_conditional_v1

    # what the later single test read would consume (no read is performed)
    python scripts/phaseformer_L/e17_conditional.py --stage plan --test-read-plan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phaseformer_L import e14_main_matrix as e14  # noqa: E402

# ---------------------------------------------------------------------------
# Lineage 1: the E14 main runner wraps search_phaseformer.py and is the only
# entry point that can install a frozen subspace (search_phaseformer.py itself
# has no --basis flag).
RUNNER = ROOT / "scripts" / "run_top2_direction_retention.py"
SEARCH_RUNNER = ROOT / "scripts" / "search_phaseformer.py"  # imported by RUNNER

# Protocol constants: imported from E14 rather than restated, so the two units
# cannot drift apart silently.
LOOKBACK = e14.LOOKBACK
PERIOD = e14.PERIOD
MAX_EPOCHS = e14.MAX_EPOCHS
LOSS = e14.LOSS
PERCENT = e14.PERCENT
SEEDS = e14.SEEDS
NEW_CELL_GATE_INIT = e14.NEW_CELL_GATE_INIT      # D-2 preset default 0.2
NEW_CELL_LR = e14.NEW_CELL_LR                    # D-2 preset default 1e-3
COST_HINT = e14.COST_HINT

E14_ARM = "l_main"
#: E8's own notation for the reused frozen-independent arm.
E8_ARM = "keep_direction_1"
#: ``weak_residual_projection_arm`` values written by *this* unit.  They must not
#: collide with E8's ``keep_direction_1``: that field feeds ``config_hash`` and the
#: run id, so a shared value would let two different projectors share one run id.
ARM_FROZEN_CONDITIONAL = "e17_frozen_conditional_direction_1"
ARM_FROZEN_INDEPENDENT = "e17_frozen_independent_direction_1"

# The seven test-selected settings (schedule section 2.1).  They are
# test-set-selection-derived; disclosed wherever they appear.
SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
)
# E8 shipped a frozen-independent direction-1 arm for six of them.
E8_SETTINGS = tuple(s for s in SETTINGS if s != ("Electricity", 336))

# ---------------------------------------------------------------------------
# Frozen per-setting (gate, lr) of the frozen arms (D-2 second half).
#
# Values are taken from ``scripts/run_top2_direction_retention_matrix.py::FROZEN``
# (which in turn cites ``scripts/run_rank_sweep_multiseed_v4.py::SETTING_TABLE``)
# and were re-checked against the six ``keep_direction_1`` rows of
# ``research_runs/top2_direction_retention_v1/results.csv``, whose recorded
# ``gate_init`` / ``learning_rate`` agree cell by cell.  Using them for the
# *conditional* arm keeps the two frozen arms paired on hyperparameters, which is
# what section 4.5 needs to read as "the target changed, nothing else did".
#
# Electricity-336 has no Stage-0 freeze (E8 never covered it), so the D-2 preset
# defaults are used and the cell is flagged ``hyperparams_source =
# "d2_preset_default_new_setting"``.
FROZEN_SETTING_TABLE = {
    ("ETTh2", 96): {"gate": 0.5, "lr": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "lr": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "lr": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "lr": 0.001},
    ("Weather", 96): {"gate": 0.2, "lr": 0.0003},
    ("Weather", 192): {"gate": 0.5, "lr": 0.001},
    ("Electricity", 336): {"gate": NEW_CELL_GATE_INIT, "lr": NEW_CELL_LR},
}
FROZEN_HYPERPARAMS_SOURCE = {
    ("ETTh2", 96): "e8_stage0_frozen",
    ("ETTh2", 720): "e8_stage0_frozen",
    ("ETTm2", 96): "e8_stage0_frozen",
    ("ETTm2", 192): "e8_stage0_frozen",
    ("Weather", 96): "e8_stage0_frozen",
    ("Weather", 192): "e8_stage0_frozen",
    ("Electricity", 336): "d2_preset_default_new_setting",
}

# Reuse scopes.  ``direct``/``joint`` share one scope because they are one object
# in this implementation (disclosure 1).
REUSE_SCOPE = {
    "direct": SETTINGS,
    "joint": SETTINGS,
    "frozen_independent_direction_1": E8_SETTINGS,
    "frozen_conditional_direction_1": (),
}
NEW_RUN_SCOPE = {
    "frozen_independent_direction_1": (("Electricity", 336),),
    "frozen_conditional_direction_1": SETTINGS,
}
ARMS = ("direct", "frozen_independent_direction_1",
        "frozen_conditional_direction_1", "joint")
TRAINED_ARMS = ("frozen_conditional_direction_1", "frozen_independent_direction_1")
FOUR_ARM_ORDER = (
    "direct",
    "frozen_independent_direction_1",
    "frozen_conditional_direction_1",
    "joint",
)

DEFAULT_OUTPUT_ROOT = "research_runs/phaseformer_L_e17_conditional_v1"
DEFAULT_E14_ROOT = "research_runs/phaseformer_L_e14_main_v1"
DEFAULT_E8_ROOT = "research_runs/top2_direction_retention_v1"
DEFAULT_E8_RESULTS = "research_runs/top2_direction_retention_v1/results.csv"
DEFAULT_PROJECTOR_DIR = f"{DEFAULT_OUTPUT_ROOT}/projectors"
DEFAULT_H1_EVIDENCE = (
    "docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md"
)
#: H1 majority rule (minipaper section 4.5 requires reporting a *seed count*).
H1_RANKS_PER_SEED = 4          # q in {1/16, 1/32, 1/4, 1/8} in table 5
H1_MAJORITY_FRACTION = 0.5     # strict majority of the rank rows within a seed

SOURCE_E14 = "reused_e14_stage_a_manifest"
SOURCE_E8 = "reused_e8_results_csv"
SOURCE_NEW = "new_trained_e17"

RESULTS_FIELDS = (
    "arm", "dataset", "horizon", "seed", "setting", "source", "status",
    "val_mse", "val_mae", "test_mse", "test_mae", "test_evidence",
    "gate_init", "learning_rate",
    "basis_file", "run_dir", "config_hash",
    "h1_cond_gt_indep_seed_majority", "h1_cond_minus_indep_mean",
    "h1_ranks", "h1_ranks_supporting", "note",
)


def parse_list(raw, cast=str) -> list:
    return [cast(item) for item in str(raw).split(",") if str(item).strip()]


def setting_name(dataset: str, horizon: int) -> str:
    return f"{dataset}-{int(horizon)}"


def repo_relative(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(raw) -> Path:
    """Absolute path of a possibly repo-relative argument."""
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e17_conditional.py",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["plan", "assemble", "a"], default="plan",
                        help="plan = build the manifest and stop; assemble = "
                        "re-read every cell (E14/E8/new runs) and rewrite "
                        "results.csv + e17_summary.json without training; "
                        "a = train the 24 new cells, then assemble")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--arms", default=",".join(TRAINED_ARMS),
                        help="arms whose *new* cells are trained here; the reused "
                        "arms (direct/joint) are only ever read")
    parser.add_argument("--datasets", default="",
                        help="empty = the datasets of the seven test-selected settings")
    parser.add_argument("--horizons", default="",
                        help="empty = the horizons of the seven test-selected settings")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--e14-root", default=DEFAULT_E14_ROOT)
    parser.add_argument("--e8-root", default=DEFAULT_E8_ROOT)
    parser.add_argument("--e8-results", default=DEFAULT_E8_RESULTS)
    parser.add_argument("--projector-dir", default=DEFAULT_PROJECTOR_DIR,
                        help="directory holding <ds>_<H>_Q1COND.npy / <ds>_<H>_Q1.npy")
    parser.add_argument("--projector-audit", default="",
                        help="projector_audit.json produced by "
                        "e17_conditional_projectors.py; empty = look inside "
                        "--projector-dir")
    parser.add_argument("--h1-evidence", default=DEFAULT_H1_EVIDENCE,
                        help="markdown/CSV table of the registered H1 evidence "
                        "(lowrank plan table 5, 72 rows); parsed read-only")
    parser.add_argument("--results-name", default="results.csv")
    parser.add_argument("--allow-missing-reuse", action="store_true",
                        help="keep going when a declared reuse cell cannot be "
                        "resolved (the missing cells are still reported)")
    parser.add_argument("--allow-missing-projector", action="store_true",
                        help="do not fail the plan gate when a new frozen cell has "
                        "no projector file yet")
    parser.add_argument("--verify", action="store_true",
                        help="fail if a declared reuse cell cannot be resolved or a "
                        "required projector is absent")
    parser.add_argument("--no-reuse", action="store_true",
                        help="treat every cell as new (protocol check / fallback); "
                        "never used for the section 4.5 table")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the 24 commands (one JSON object per line) and "
                        "train nothing; imports no torch")
    parser.add_argument("--test-read-plan", action="store_true",
                        help="also print the cells the later single test read would "
                        "consume; performs no read")
    parser.add_argument("--allow-test-read", action="store_true",
                        help="reserved flag; test reading is a separate stage and is "
                        "refused here (see the module docstring)")
    return parser


# ---------------------------------------------------------------------------
# Lineage 2: E14 stage-A manifest (direct / joint)
# ---------------------------------------------------------------------------
def load_e14_manifest(e14_root: Path) -> dict:
    path = e14_root / "stage_a_manifest.json"
    if not path.is_file():
        raise SystemExit(
            f"E14 stage-A manifest not found: {path}\n"
            "E17 depends on E14 (schedule section 4.1).  Run "
            "scripts/phaseformer_L/e14_main_matrix.py --stage a first, or point "
            "--e14-root at the root that holds stage_a_manifest.json."
        )
    manifest = json.loads(path.read_text())
    if not isinstance(manifest.get("cells"), list):
        raise SystemExit(f"{path} has no 'cells' list")
    return manifest


def _first_metrics_row(run_dir: Path) -> dict:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), None) or {}


def e14_test_evidence(cell: dict) -> dict:
    """Test evidence of one reused E14 cell, by E14's own two conventions.

    E14 records either the run's inline ``metrics.csv`` (E3 lineage) or a
    registered external evidence file (E8 lineage); both are mirrored here so the
    reused numbers carry their provenance instead of being re-read.
    """
    source = cell.get("source") or {}
    evidence = source.get("test_evidence")
    out = {
        "test_mse": None,
        "test_mae": None,
        "test_evidence": "missing",
        "val_mse": None,
        "val_mae": None,
        "nlinear_mse": None,
        "nlinear_mae": None,
        "gate_value": None,
    }
    run_dir = ROOT / source.get("run_dir", "") if source.get("run_dir") else None
    if isinstance(evidence, dict):
        out["test_mse"] = evidence.get("test_mse")
        out["test_mae"] = evidence.get("test_mae")
        out["test_evidence"] = (
            f"external:{evidence.get('evidence', '')}"
            f"|status={evidence.get('status', '')}"
        )
    elif evidence == "inline metrics.csv" and run_dir is not None:
        row = _first_metrics_row(run_dir)
        if str(row.get("test_mse", "")).strip():
            out["test_mse"] = row.get("test_mse")
            out["test_mae"] = row.get("test_mae")
            out["test_evidence"] = "inline:metrics.csv"
        else:
            out["test_evidence"] = "inline_recorded_but_empty"
        for key, name in (("val_mse", "val_mse"), ("val_mae", "val_mae"),
                          ("nlinear_mse", "nlinear_mse"),
                          ("nlinear_mae", "nlinear_mae"),
                          ("gate_value", "gate_value")):
            raw = str(row.get(name, "")).strip()
            if raw:
                out[key] = raw
    if run_dir is not None and out["val_mse"] is None:
        row = _first_metrics_row(run_dir)
        for key in ("val_mse", "val_mae"):
            raw = str(row.get(key, "")).strip()
            if raw:
                out[key] = raw
    return out


def build_e14_index(e14_root: Path, wanted: set) -> tuple[dict, list]:
    """Index the E14 ``l_main`` cells of ``wanted`` and their test evidence."""
    manifest = load_e14_manifest(e14_root)
    index: dict = {}
    rejected: list = []
    for cell in manifest["cells"]:
        if cell.get("arm") != E14_ARM:
            continue
        key = (cell.get("dataset"), int(cell.get("horizon", -1)),
               int(cell.get("seed", -1)))
        if key not in wanted:
            continue
        if key in index:
            rejected.append({"cell": setting_name(*key[:2]),
                             "reason": "duplicate l_main cell in the manifest"})
            continue
        status = cell.get("status")
        if status == "new":
            # An E14 cell trained by this very stage A has no test read yet; it is
            # still a valid `direct`/`joint` cell, but its test columns stay empty
            # until E14's stage B runs.
            index[key] = {
                "run_dir": None,
                "config_hash": None,
                "status": status,
                "source": cell.get("source"),
                "test": {"test_mse": None, "test_mae": None,
                         "test_evidence": "pending_e14_stage_b",
                         "val_mse": None, "val_mae": None,
                         "nlinear_mse": None, "nlinear_mae": None,
                         "gate_value": None},
            }
            continue
        evidence = e14_test_evidence(cell)
        index[key] = {
            "run_dir": (cell.get("source") or {}).get("run_dir"),
            "config_hash": (cell.get("source") or {}).get("config_hash"),
            "status": status,
            "source": cell.get("source"),
            "test": evidence,
        }
    return index, rejected


# ---------------------------------------------------------------------------
# Lineage 3: E8 results.csv (frozen independent direction 1, six settings)
# ---------------------------------------------------------------------------
def build_e8_index(results_path: Path, wanted: set) -> tuple[dict, list]:
    """Verify and index the reused ``keep_direction_1`` cells of E8.

    Every field is checked against the frozen table; a mismatch is a *rejection*
    with a reason (never a silent fallback), matching E14's reuse discipline.
    """
    index: dict = {}
    rejected: list = []
    if not results_path.is_file():
        return index, [{"reason": f"E8 results file not found: {results_path}"}]
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("arm", "")).strip() != E8_ARM:
            continue
        key = (row.get("dataset"), int(row["horizon"]), int(row["seed"]))
        if key not in wanted:
            continue
        dataset, horizon, seed = key
        frozen = FROZEN_SETTING_TABLE[(dataset, horizon)]
        problems = []
        try:
            if abs(float(row.get("gate_init") or "nan") - frozen["gate"]) > 1e-12:
                problems.append(
                    f"gate_init={row.get('gate_init')} != {frozen['gate']}"
                )
            if abs(float(row.get("learning_rate") or "nan") - frozen["lr"]) > 1e-12:
                problems.append(
                    f"learning_rate={row.get('learning_rate')} != {frozen['lr']}"
                )
        except ValueError:
            problems.append("gate_init/learning_rate is not numeric")
        status = str(row.get("test_read_status", "")).strip()
        if status not in ("read", "reused"):
            problems.append(f"test_read_status={status!r}")
        if not str(row.get("test_mse", "")).strip():
            problems.append("empty test_mse")
        if problems:
            rejected.append({
                "cell": f"{setting_name(dataset, horizon)}-s{seed}",
                "arm": E8_ARM,
                "failures": problems,
                "source": repo_relative(results_path),
            })
            continue
        if key in index:
            continue
        index[key] = {
            "run_dir": row.get("run_dir"),
            "config_hash": None,
            "test_mse": row.get("test_mse"),
            "test_mae": row.get("test_mae"),
            "val_mse": row.get("val_mse"),
            "val_mae": row.get("val_mae"),
            "nlinear_mse": row.get("nlinear_mse"),
            "nlinear_mae": row.get("nlinear_mae"),
            "gate_value": row.get("gate_value"),
            "test_read_status": status,
            "val_relative_difference": row.get("val_relative_difference"),
            "run_id": row.get("run_id"),
        }
    return index, rejected


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def projector_file(arm: str, dataset: str, horizon: int) -> str:
    """The frozen projector a new cell of ``arm`` must consume."""
    suffix = "Q1COND" if arm == "frozen_conditional_direction_1" else "Q1"
    return f"{dataset}_{horizon}_{suffix}.npy"


def arm_command(arm: str, dataset: str, horizon: int, seed: int,
                output_root: str, projector_dir: str, num_workers: int,
                max_epochs: int = MAX_EPOCHS) -> list:
    """argv (without the interpreter) of one new frozen cell.

    Mirrors E8's ``arm_command`` override structure and E14's protocol flags.
    ``--basis`` and ``weak_residual_projection=frozen_subspace`` are both
    required: the flag is consumed by the wrapper, the override is what
    ``PhaseFormer.__init__`` validates before it accepts the installed basis.
    """
    frozen = FROZEN_SETTING_TABLE[(dataset, horizon)]
    projection_arm = (
        ARM_FROZEN_CONDITIONAL if arm == "frozen_conditional_direction_1"
        else ARM_FROZEN_INDEPENDENT
    )
    overrides = {
        "learning_rate": frozen["lr"],
        "weak_period_residual_gate_init": frozen["gate"],
        "weak_period_residual_head_type": "shared",
        "weak_residual_projection": "frozen_subspace",
        "weak_residual_projection_arm": projection_arm,
    }
    return [
        str(RUNNER),
        "--output-dir", output_root,
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm",
        "--lookback", str(LOOKBACK),
        "--period", str(PERIOD),
        "--max-epochs", str(max_epochs),
        "--seed", str(seed),
        "--loss", LOSS,
        "--percent", str(PERCENT),
        "--learning-rate", str(frozen["lr"]),
        "--require-cuda",
        "--resume",
        "--num-workers", str(num_workers),
        "--bad-case-limit", "0",
        "--mechanism", "weak_residual",
        "--basis", f"{projector_dir}/{projector_file(arm, dataset, horizon)}",
        "--overrides", json.dumps(overrides, sort_keys=True),
    ]
    # ``--evaluate-test`` is never appended (section 5.3 of the plan).


def cell_key(arm: str, dataset: str, horizon: int, seed: int) -> str:
    return f"{arm}__{setting_name(dataset, horizon)}-s{seed}"


def build_cells(args, e14_index: dict, e8_index: dict) -> list:
    """One record per (arm, setting, seed) of the four-arm table."""
    requested_datasets = parse_list(args.datasets)
    requested_horizons = parse_list(args.horizons, int)
    seeds = parse_list(args.seeds, int)
    allowed = {
        (dataset, horizon) for dataset, horizon in SETTINGS
        if (not requested_datasets or dataset in requested_datasets)
        and (not requested_horizons or horizon in requested_horizons)
    }
    trained = parse_list(args.arms)
    for arm in trained:
        if arm not in TRAINED_ARMS:
            raise SystemExit(
                f"arm {arm!r} does not need training here; trainable arms: "
                f"{list(TRAINED_ARMS)} (direct/joint are reused from E14)"
            )

    cells: list = []
    for arm in FOUR_ARM_ORDER:
        for dataset, horizon in SETTINGS:
            if (dataset, horizon) not in allowed:
                continue
            for seed in seeds:
                key = (dataset, horizon, seed)
                cell = {
                    "arm": arm,
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "setting": setting_name(dataset, horizon),
                    "key": cell_key(arm, dataset, horizon, seed),
                    "status": None,
                    "source": None,
                    "command": None,
                    "basis_file": None,
                    "hyperparams_source": FROZEN_HYPERPARAMS_SOURCE[
                        (dataset, horizon)
                    ] if arm in TRAINED_ARMS else "reused_cell_own_protocol",
                    "gate_init": FROZEN_SETTING_TABLE[(dataset, horizon)]["gate"],
                    "learning_rate": FROZEN_SETTING_TABLE[(dataset, horizon)]["lr"],
                }
                reusable = arm in ("direct", "joint") or (
                    arm == "frozen_independent_direction_1"
                    and (dataset, horizon) in E8_SETTINGS
                )
                if arm in trained and (dataset, horizon) in NEW_RUN_SCOPE.get(arm, ()):
                    cell["status"] = "new"
                    cell["basis_file"] = (
                        f"{args.projector_dir}/"
                        f"{projector_file(arm, dataset, horizon)}"
                    )
                    cell["command"] = arm_command(
                        arm, dataset, horizon, seed, args.output_root,
                        args.projector_dir, args.num_workers,
                    )
                elif reusable and not args.no_reuse:
                    cell["status"] = "reused"
                    if arm in ("direct", "joint"):
                        cell["source"] = e14_index.get(key)
                    else:
                        cell["source"] = e8_index.get(key)
                elif arm in trained:
                    cell["status"] = "new"
                    cell["basis_file"] = (
                        f"{args.projector_dir}/"
                        f"{projector_file(arm, dataset, horizon)}"
                    )
                    cell["command"] = arm_command(
                        arm, dataset, horizon, seed, args.output_root,
                        args.projector_dir, args.num_workers,
                    )
                else:
                    # A reused arm outside its declared scope (e.g. the six
                    # settings of the independent arm with --no-reuse removed):
                    # recorded as a gap, never silently trained.
                    cell["status"] = "gap"
                cells.append(cell)
    cells.sort(key=lambda c: (
        -COST_HINT.get(c["dataset"], {}).get(c["horizon"], 500),
        c["arm"], c["dataset"], c["horizon"], c["seed"],
    ))
    return cells


# ---------------------------------------------------------------------------
# H1 evidence (registered table 5; never recomputed)
# ---------------------------------------------------------------------------
def parse_h1_markdown(path: Path) -> tuple[list, str]:
    """Parse the registered markdown copy of lowrank-plan table 5.

    Expected row shape::

        | Setting | q/rank | overlap with independent RRR |
        overlap with conditional RRR | 差值 | 支持 H1 |

    Only the setting, the two overlaps and the support flag are read; the rank
    label is kept verbatim for the audit trail.
    """
    rows: list = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.count("|") < 6:
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        setting, rank_label, indep, cond, _delta, support = parts[:6]
        if not setting or setting in ("Setting", "Setting ", "---") or "---" in setting:
            continue
        if setting.startswith(":"):
            continue
        try:
            indep_value = float(indep)
            cond_value = float(cond)
        except ValueError:
            continue
        if support not in ("是", "否"):
            continue
        rows.append({
            "setting": setting,
            "rank_label": rank_label,
            "overlap_independent_rrr": indep_value,
            "overlap_conditional_rrr": cond_value,
            "difference": cond_value - indep_value,
            "supports_h1": support == "是",
        })
    return rows, "markdown"


def parse_h1_csv(path: Path) -> tuple[list, str]:
    rows: list = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                indep = float(row.get("overlap_with_independent_rrr", ""))
                cond = float(row.get("overlap_with_conditional_rrr", ""))
            except ValueError:
                continue
            support_raw = str(row.get("supports_h1", "")).strip().lower()
            rows.append({
                "setting": str(row.get("setting", "")).strip(),
                "rank_label": str(row.get("rank", "")).strip(),
                "overlap_independent_rrr": indep,
                "overlap_conditional_rrr": cond,
                "difference": cond - indep,
                "supports_h1": support_raw in ("true", "1", "yes"),
            })
    return rows, "csv"


def load_h1_evidence(path: Path) -> tuple[dict, dict]:
    """``{(setting, seed): summary}`` plus a status block.

    Table 5 carries no seed column (its index is setting x rank x 3 seeds in
    file order), so the rows of one setting are grouped into consecutive blocks
    of ``H1_RANKS_PER_SEED``: rows 0-3 are seed 2021, 4-7 seed 2022, 8-11 seed
    2023.  The grouping is recorded in the status block so the convention is
    auditable rather than implicit.
    """
    status = {
        "path": repo_relative(path),
        "exists": path.is_file(),
        "format": None,
        "rows": 0,
        "seed_assignment": (
            f"consecutive blocks of {H1_RANKS_PER_SEED} rows per setting, in file "
            f"order -> seeds {list(SEEDS)}"
        ),
        "majority_rule": (
            f"seed-level H1 = strictly more than {H1_MAJORITY_FRACTION:.0%} of the "
            f"{H1_RANKS_PER_SEED} rank rows support H1 "
            f"(i.e. >= {int(H1_MAJORITY_FRACTION * H1_RANKS_PER_SEED) + 1}/"
            f"{H1_RANKS_PER_SEED})"
        ),
        "status": "missing",
        "settings_with_evidence": [],
        "settings_without_evidence": [
            setting_name(d, h) for d, h in SETTINGS
        ],
    }
    if not path.is_file():
        return {}, status
    try:
        rows, fmt = (
            parse_h1_csv(path) if path.suffix.lower() == ".csv"
            else parse_h1_markdown(path)
        )
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        status["status"] = "unparsable"
        status["error"] = repr(error)
        return {}, status
    status["format"] = fmt
    status["rows"] = len(rows)
    if not rows:
        status["status"] = "unparsable"
        return {}, status

    by_setting: dict = {}
    order: list = []
    for row in rows:
        name = row["setting"]
        if name not in by_setting:
            by_setting[name] = []
            order.append(name)
        by_setting[name].append(row)

    summary: dict = {}
    for name in order:
        block = by_setting[name]
        for seed_index, seed in enumerate(SEEDS):
            chunk = block[
                seed_index * H1_RANKS_PER_SEED:
                (seed_index + 1) * H1_RANKS_PER_SEED
            ]
            if not chunk:
                continue
            supporting = sum(1 for row in chunk if row["supports_h1"])
            total = len(chunk)
            summary[(name, seed)] = {
                "setting": name,
                "seed": seed,
                "ranks": total,
                "ranks_supporting": supporting,
                "seed_majority_supports_h1": bool(
                    supporting > H1_MAJORITY_FRACTION * total
                ),
                "mean_conditional_minus_independent": (
                    sum(row["difference"] for row in chunk) / total
                ),
                "rank_labels": [row["rank_label"] for row in chunk],
                "evidence_path": repo_relative(path),
                "evidence_format": fmt,
            }
    status["status"] = "loaded"
    status["settings_with_evidence"] = order
    status["settings_without_evidence"] = [
        setting_name(d, h) for d, h in SETTINGS
        if setting_name(d, h) not in order
    ]
    return summary, status


# ---------------------------------------------------------------------------
# New-run results (stage A writes no test metrics)
# ---------------------------------------------------------------------------
def resolve_run_dir(output_root: str, command: list):
    """The run directory a command will produce (search_phaseformer's own rule).

    ``search_phaseformer.build_spec`` + ``run_id`` are imported rather than
    copied so the directory name cannot drift from the runner's own convention.
    ``--basis`` / ``--basis-sha256`` are stripped first, exactly as the wrapper
    ``run_top2_direction_retention.py`` does before it hands over to
    ``search_phaseformer.parse_args`` (which knows neither flag).
    """
    try:
        import scripts.search_phaseformer as search
    except Exception as error:  # noqa: BLE001
        return None, f"cannot import search_phaseformer: {error!r}"
    stripped: list = []
    skip = 0
    for index, item in enumerate(command[1:]):
        if skip:
            skip -= 1
            continue
        if item in ("--basis", "--basis-sha256"):
            skip = 1
            continue
        stripped.append(item)
    saved = sys.argv
    try:
        sys.argv = [str(command[0]), *stripped]
        args = search.parse_args()
        spec = search.build_spec(args)
        rid = search.run_id(spec)
    except SystemExit as error:  # argparse rejects the argv
        return None, f"search_phaseformer refused the argv: {error!r}"
    except Exception as error:  # noqa: BLE001
        return None, f"cannot rebuild the run id: {error!r}"
    finally:
        sys.argv = saved
    return Path(output_root) / "runs" / rid, None


def read_new_run_record(run_dir: Path) -> dict:
    """Validation metrics of a finished stage-A run (test columns stay empty)."""
    record = {
        "val_mse": None, "val_mae": None, "test_mse": None, "test_mae": None,
        "nlinear_mse": None, "nlinear_mae": None, "gate_value": None,
        "config_hash": None, "checkpoint": None, "test_evidence":
        "pending_single_test_read",
    }
    if run_dir is None or not run_dir.is_dir():
        return record
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            record["config_hash"] = json.loads(
                config_path.read_text()
            ).get("config_hash")
        except json.JSONDecodeError:
            pass
    row = _first_metrics_row(run_dir)
    for key in ("val_mse", "val_mae", "nlinear_mse", "nlinear_mae",
                "gate_value", "checkpoint"):
        raw = str(row.get(key, "")).strip()
        if raw:
            record[key] = raw
    return record


# ---------------------------------------------------------------------------
# Dispatch (one run per GPU, retries, logs)
# ---------------------------------------------------------------------------
def assert_no_test_read(cells) -> None:
    """The single test read is a separate stage; refuse to smuggle it in."""
    offenders = [
        cell["key"] for cell in cells
        if cell["command"] and "--evaluate-test" in cell["command"]
    ]
    if offenders:
        raise SystemExit(
            f"refusing to run: {len(offenders)} cell command(s) carry "
            f"--evaluate-test (e.g. {offenders[0]}).  Test is read once, "
            "afterwards, by a separate stage."
        )


def dispatch(cells, output_root, gpus, retries, poll,
             allow_test_read: bool) -> tuple[list, list]:
    if allow_test_read:
        raise SystemExit(
            "--allow-test-read was given, but E17 stage A must not read the test "
            "split; the single read is a separate stage (plan section 5.3)."
        )
    assert_no_test_read(cells)
    pending = [cell for cell in cells if cell["status"] == "new"]
    active: dict = {}
    attempts: dict = {}
    completed: list = []
    failed: list = []
    log_dir = Path(output_root) if Path(output_root).is_absolute() else ROOT / output_root
    log_dir = log_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell["key"]
            attempts[key] = attempts.get(key, 0) + 1
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(log_dir / f"{key}.log", "w")
            process = subprocess.Popen(
                [sys.executable, *[str(item) for item in cell["command"]]],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            print(json.dumps({"event": "launch", "cell": key, "gpu": gpu,
                              "attempt": attempts[key]}), flush=True)
            active[gpu] = (cell, process, log)
        finished = []
        for gpu, (cell, process, log) in active.items():
            code = process.poll()
            if code is None:
                continue
            finished.append(gpu)
            key = cell["key"]
            log.close()
            if code == 0:
                completed.append(key)
                print(json.dumps({"event": "done", "cell": key, "gpu": gpu}),
                      flush=True)
            elif attempts[key] <= retries:
                pending.append(cell)
                print(json.dumps({"event": "retry", "cell": key, "gpu": gpu,
                                  "code": code}), flush=True)
            else:
                failed.append({"cell": key, "return_code": code})
                print(json.dumps({"event": "failed", "cell": key, "code": code}),
                      flush=True)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(poll)
    return completed, failed


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_results(cells, h1_summary: dict, e8_root: str = DEFAULT_E8_ROOT) -> list:
    rows: list = []
    for cell in cells:
        dataset, horizon, seed = cell["dataset"], cell["horizon"], cell["seed"]
        key = (dataset, horizon, seed)
        row = {name: "" for name in RESULTS_FIELDS}
        row.update({
            "arm": cell["arm"],
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "setting": cell["setting"],
            "status": cell["status"],
            "gate_init": cell["gate_init"],
            "learning_rate": cell["learning_rate"],
        })
        h1 = h1_summary.get((cell["setting"], seed))
        if h1:
            row["h1_cond_gt_indep_seed_majority"] = (
                "true" if h1["seed_majority_supports_h1"] else "false"
            )
            row["h1_cond_minus_indep_mean"] = round(
                h1["mean_conditional_minus_independent"], 6
            )
            row["h1_ranks"] = h1["ranks"]
            row["h1_ranks_supporting"] = h1["ranks_supporting"]
        elif h1_summary:
            row["h1_cond_gt_indep_seed_majority"] = "evidence_missing"
            row["note"] = "H1 evidence absent for this setting"
        else:
            row["h1_cond_gt_indep_seed_majority"] = "evidence_missing"

        if cell["status"] == "reused" and cell["source"] is not None:
            if cell["arm"] in ("direct", "joint"):
                source = cell["source"]
                test = source["test"]
                row.update({
                    "source": SOURCE_E14,
                    "val_mse": test.get("val_mse") or "",
                    "val_mae": test.get("val_mae") or "",
                    "test_mse": test.get("test_mse") or "",
                    "test_mae": test.get("test_mae") or "",
                    "test_evidence": test.get("test_evidence") or "",
                    "run_dir": source.get("run_dir") or "",
                    "config_hash": source.get("config_hash") or "",
                })
                if cell["arm"] == "joint":
                    row["note"] = (
                        "identical to the `direct` row: PhaseFormer-L's corrector "
                        "is the jointly trained shared head (disclosure 1)"
                    )
                if (test.get("test_mse") in (None, "")):
                    row["test_evidence"] = row["test_evidence"] or "unavailable"
            else:
                source = cell["source"]
                row.update({
                    "source": SOURCE_E8,
                    "val_mse": source.get("val_mse") or "",
                    "val_mae": source.get("val_mae") or "",
                    "test_mse": source.get("test_mse") or "",
                    "test_mae": source.get("test_mae") or "",
                    "test_evidence": (
                        f"e8_results_csv|status={source.get('test_read_status')}"
                    ),
                    "run_dir": source.get("run_dir") or "",
                    "config_hash": source.get("config_hash") or "",
                    "basis_file": (
                        f"{e8_root}/projectors/"
                        f"{cell['dataset']}_{cell['horizon']}_Q1.npy"
                    ),
                })
        elif cell["status"] == "new":
            run_dir, error = resolve_run_dir(
                cell["command"][cell["command"].index("--output-dir") + 1],
                cell["command"],
            )
            record = read_new_run_record(run_dir)
            row.update({
                "source": SOURCE_NEW,
                "val_mse": record["val_mse"] or "",
                "val_mae": record["val_mae"] or "",
                "test_mse": "",
                "test_mae": "",
                "test_evidence": record["test_evidence"],
                "basis_file": cell["basis_file"] or "",
                "run_dir": repo_relative(run_dir) if run_dir else "",
                "config_hash": record["config_hash"] or "",
                "note": (
                    f"hyperparams_source={cell['hyperparams_source']}"
                    + (f"; {error}" if error else "")
                ),
            })
        else:
            row["source"] = "gap"
            row["note"] = (
                "outside the declared reuse scope and not trained by E17; the cell "
                "is a documented gap"
            )
        rows.append(row)
    return rows


def h1_setting_summary(h1_summary: dict) -> dict:
    per_setting: dict = {}
    for (setting, seed), item in sorted(h1_summary.items()):
        bucket = per_setting.setdefault(setting, {
            "seeds": 0, "seeds_supporting": 0,
            "mean_conditional_minus_independent": 0.0,
        })
        bucket["seeds"] += 1
        bucket["seeds_supporting"] += 1 if item["seed_majority_supports_h1"] else 0
        bucket["mean_conditional_minus_independent"] += (
            item["mean_conditional_minus_independent"]
        )
    for bucket in per_setting.values():
        if bucket["seeds"]:
            bucket["mean_conditional_minus_independent"] = round(
                bucket["mean_conditional_minus_independent"] / bucket["seeds"], 6
            )
    return per_setting


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not RUNNER.is_file():
        raise SystemExit(f"the frozen-subspace wrapper is missing: {RUNNER}")

    e14_root = ROOT / args.e14_root
    e8_results = ROOT / args.e8_results
    projector_dir = (
        Path(args.projector_dir) if Path(args.projector_dir).is_absolute()
        else ROOT / args.projector_dir
    )

    seeds = parse_list(args.seeds, int)
    wanted_e14 = {(d, h, s) for (d, h) in SETTINGS for s in seeds}
    wanted_e8 = {(d, h, s) for (d, h) in E8_SETTINGS for s in seeds}
    e14_index, e14_rejected = build_e14_index(e14_root, wanted_e14)
    e8_index, e8_rejected = build_e8_index(e8_results, wanted_e8)

    reuse_summary = {
        "direct": {
            "declared": len(wanted_e14), "resolved": len(e14_index),
            "missing": sorted(
                setting_name(d, h) + f"-s{s}" for d, h, s in wanted_e14 - set(e14_index)
            ),
            "source": repo_relative(e14_root / "stage_a_manifest.json"),
            "arm": E14_ARM,
        },
        "joint": {
            "declared": len(wanted_e14), "resolved": len(e14_index),
            "missing": sorted(
                setting_name(d, h) + f"-s{s}" for d, h, s in wanted_e14 - set(e14_index)
            ),
            "source": repo_relative(e14_root / "stage_a_manifest.json"),
            "arm": E14_ARM,
            "note": "same index as `direct` (disclosure 1)",
        },
        "frozen_independent_direction_1": {
            "declared": len(wanted_e8), "resolved": len(e8_index),
            "missing": sorted(
                setting_name(d, h) + f"-s{s}" for d, h, s in wanted_e8 - set(e8_index)
            ),
            "source": repo_relative(e8_results),
            "arm": E8_ARM,
            "note": "Electricity-336 is not in E8's scope; it is the 3 new runs",
        },
        "frozen_conditional_direction_1": {
            "declared": 0, "resolved": 0, "missing": [],
            "source": "new: all 7 settings x 3 seeds trained here",
        },
    }

    cells = build_cells(args, e14_index, e8_index)
    counters: dict = {}
    for cell in cells:
        bucket = counters.setdefault(cell["arm"], {"new": 0, "reused": 0, "gap": 0})
        bucket[cell["status"]] += 1
    new_cells = [cell for cell in cells if cell["status"] == "new"]
    trained_new = [cell for cell in new_cells if cell["arm"] in parse_list(args.arms)]

    missing_projectors = sorted({
        cell["basis_file"] for cell in trained_new
        if cell["basis_file"] and not resolve_repo_path(cell["basis_file"]).is_file()
    })

    projector_audit_path = (
        Path(args.projector_audit) if args.projector_audit
        else projector_dir / "projector_audit.json"
    )
    projector_audit_present = projector_audit_path.is_file()
    projector_sha = {}
    if projector_audit_present:
        try:
            audit = json.loads(projector_audit_path.read_text())
            for record in audit:
                name = record.get("setting", "")
                projector_sha[name] = {
                    "q1": (record.get("standardized", {}) or {})
                    .get("projector", {}).get("sha256"),
                    "q1cond": (record.get("revin", {}) or {})
                    .get("conditional_projector", {}).get("sha256"),
                    "reproduction_abs_cos": (record.get("standardized", {}) or {})
                    .get("reproduction_gate", {}).get("abs_cos"),
                }
        except json.JSONDecodeError as error:
            projector_audit_present = False
            projector_sha = {"error": repr(error)}

    h1_summary, h1_status = load_h1_evidence(ROOT / args.h1_evidence)

    manifest = {
        "experiment": "E17 stage A (minipaper section 4.5 conditional-learning "
                      "four-arm table; training only)",
        "reads_test": False,
        "test_read_stage": (
            "separate; --evaluate-test is never passed and --allow-test-read is "
            "refused (plan section 5.3)"
        ),
        "protocol": {
            "runner": repo_relative(RUNNER),
            "inner_runner": repo_relative(SEARCH_RUNNER),
            "lookback": LOOKBACK, "period": PERIOD, "loss": LOSS,
            "max_epochs": MAX_EPOCHS, "percent": PERCENT,
            "stage": "confirm", "seeds": seeds,
            "checkpoint": "best validation loss",
            "require_cuda": True, "resume": True,
            "frozen_setting_table": {
                setting_name(d, h): FROZEN_SETTING_TABLE[(d, h)]
                for d, h in SETTINGS
            },
            "frozen_hyperparams_source": {
                setting_name(d, h): FROZEN_HYPERPARAMS_SOURCE[(d, h)]
                for d, h in SETTINGS
            },
            "projection": {
                "flag": "--basis <projector-dir>/<dataset>_<horizon>_Q1.npy|Q1COND.npy",
                "override_key": "weak_residual_projection",
                "override_value": "frozen_subspace",
                "install_path": "scripts/run_top2_direction_retention.py wraps "
                                 "search_phaseformer.py and installs the basis "
                                 "through PhaseFormer.__init__(projection_basis=...)",
            },
            "basis_naming": "<dataset>_<horizon>_Q1.npy (independent, E8 recipe) "
                            "/ <dataset>_<horizon>_Q1COND.npy (conditional, "
                            "train-split forward pass)",
        },
        "settings": [setting_name(d, h) for d, h in SETTINGS],
        "setting_scope_disclosure": (
            "the seven settings are test-set-selection-derived (schedule section "
            "2.1); they are not a blind selection"
        ),
        "gpus": parse_list(args.gpus, int),
        "arm_filters": parse_list(args.arms),
        "no_reuse": bool(args.no_reuse),
        "counts": {
            "total": len(cells),
            "by_arm": counters,
            # New runs of *this* unit: 21 frozen-conditional + 3 Electricity-336
            # frozen-independent = 24 (schedule section 2.3).
            "new_runs_total": len(
                [c for c in cells if c["arm"] in TRAINED_ARMS
                 and c["status"] == "new"]
            ),
            "new_runs_trained_here": len(trained_new),
        },
        "reuse_summary": reuse_summary,
        "reuse_rejected_candidates": e14_rejected + e8_rejected,
        "projector_dir": repo_relative(projector_dir),
        "projector_audit": repo_relative(projector_audit_path),
        "projector_audit_present": projector_audit_present,
        "projector_sha256": projector_sha,
        "missing_projectors": missing_projectors,
        "h1_evidence": h1_status,
        "cells": [
            {**cell, "command": cell["command"] if cell["status"] == "new" else None}
            for cell in cells
        ],
    }
    (output_root / "stage_a_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "event": "planned",
        "total_cells": len(cells),
        "by_arm": counters,
        "new_runs_trained_here": len(trained_new),
        "gpus": manifest["gpus"],
    }, ensure_ascii=False), flush=True)

    problems = []
    if not args.no_reuse:
        for arm in ("direct", "joint", "frozen_independent_direction_1"):
            if reuse_summary[arm]["missing"]:
                problems.append(
                    f"{arm}: unresolved reuse cells {reuse_summary[arm]['missing']}"
                )
    if missing_projectors and not args.allow_missing_projector:
        problems.append(
            f"{len(missing_projectors)} projector file(s) absent, e.g. "
            f"{missing_projectors[:3]}; build them with "
            "scripts/phaseformer_L/e17_conditional_projectors.py"
        )
    if args.dry_run and missing_projectors:
        # a dry-run is a wiring check; the projector build is a separate command
        print(json.dumps({"event": "warning", "problem": problems[-1]})
              if problems else "{}", flush=True)
    elif args.verify and problems:
        for problem in problems:
            print(json.dumps({"event": "verify_failed", "problem": problem}),
                  flush=True)
        raise SystemExit("E17 pre-flight verification failed; refusing to train")
    elif problems:
        for problem in problems:
            print(json.dumps({"event": "warning", "problem": problem}), flush=True)

    if args.dry_run or args.stage == "plan":
        for cell in trained_new:
            print(json.dumps({"cell": cell["key"],
                              "command": cell["command"]}, ensure_ascii=False))
        if args.test_read_plan:
            for cell in new_cells:
                run_dir, _error = resolve_run_dir(
                    cell["command"][cell["command"].index("--output-dir") + 1],
                    cell["command"],
                )
                print(json.dumps({
                    "event": "test_read_plan",
                    "cell": cell["key"],
                    "run_dir": repo_relative(run_dir) if run_dir else None,
                    "note": "no read performed; the single test read is a separate "
                            "stage",
                }, ensure_ascii=False))
        print(json.dumps({"event": "plan_only", "cells": len(cells),
                          "new_runs": len(trained_new)}))
        return

    if args.no_reuse:
        raise SystemExit(
            "--no-reuse is a protocol-check switch and cannot be combined with "
            "--stage a/assemble: the section 4.5 table is defined by its reuse "
            "lineage"
        )

    if args.stage == "assemble":
        summary = {
            "stage": "assemble",
            "cells_planned": len(cells),
            "new_runs_planned": len(trained_new),
            "trained": 0,
            "failed": [],
            "completed": [],
        }
        print(json.dumps({"event": "assemble_only",
                          "cells": len(cells)}, ensure_ascii=False), flush=True)
    else:
        completed, failed = dispatch(
            trained_new, args.output_root, manifest["gpus"], args.retries,
            args.poll_seconds, args.allow_test_read,
        )
        summary = {
            "stage": "a",
            "cells_planned": len(cells),
            "new_runs_planned": len(trained_new),
            "trained": len(completed),
            "failed": failed,
            "completed": completed,
        }
        (output_root / "stage_a_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failed:
            # Still assemble so the partial matrix and its gaps are auditable.
            print(json.dumps({"event": "failed_cells", "failed": failed}),
                  flush=True)

    # Read every cell back off disk (E14 manifest / E8 results / new run dirs).
    rows = assemble_results(cells, h1_summary, args.e8_root)
    results_path = output_root / args.results_name
    write_csv(results_path, rows)

    settings_summary = h1_setting_summary(h1_summary)
    e17_summary = {
        "experiment": "E17 / minipaper section 4.5 conditional-learning four arms",
        "output_root": args.output_root,
        "stage_a_summary": summary,
        "counts": manifest["counts"],
        "results_csv": args.results_name,
        "results_rows": len(rows),
        "results_rows_expected": len(FOUR_ARM_ORDER) * len(SETTINGS) * len(seeds),
        "conditional_target_route": "train_split_forward_pass",
        "conditional_target": "D_cond = y - y_phi",
        "independent_target": "D_ind = y - x_last",
        "h1_evidence_status": h1_status["status"],
        "h1_majority_rule": h1_status["majority_rule"],
        "h1_seed_assignment": h1_status["seed_assignment"],
        "h1_by_setting": settings_summary,
        "disclosures": {
            "direct_equals_joint": (
                "the `direct` and `joint` columns are the same configuration in "
                "this implementation: PhaseFormer-L's corrector is the jointly "
                "trained unconstrained (`shared`) head, so both columns read the "
                "same E14 l_main run per cell"
            ),
            "test_selected_settings": [setting_name(d, h) for d, h in SETTINGS],
            "mixed_hyperparameters": (
                "the reused cells keep the D-2 Stage-0-frozen (gate, lr) of their "
                "own lineage instead of the section 4.0 preset defaults; the new "
                "frozen cells reuse E8's FROZEN table for the same reason, and "
                "Electricity-336 (no Stage-0 freeze) uses the D-2 preset defaults"
            ),
            "frozen_projectors_train_only": (
                "both projectors are fitted on the train split only; the "
                "conditional one additionally consumes a trained phase backbone's "
                "train-split forward pass and no test statistic"
            ),
            "no_test_read_in_stage_a": (
                "no cell passes --evaluate-test; every test column of a new run is "
                "empty until the separate single-read stage"
            ),
        },
        "missing_projectors": missing_projectors,
    }
    (output_root / "e17_summary.json").write_text(
        json.dumps(e17_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "stage_finished",
        "stage": summary["stage"],
        "trained": summary["trained"],
        "failed": summary["failed"],
        "results_csv": repo_relative(results_path),
        "e17_summary": repo_relative(output_root / "e17_summary.json"),
    }, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(f"E17 stage A had failed cells: {summary['failed']}")


if __name__ == "__main__":
    main()
