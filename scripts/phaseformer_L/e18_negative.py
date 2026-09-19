#!/usr/bin/env python3
"""E18 stage A/B: train the two GPU rows of minipaper §4.6 (negative controls).

Implements `docs/PhaseFormer_L_execution_schedule.md` §2.3 (E18) and
`docs/Former_L_minipaper.md` §4.6.  §4.6 is a negative-control summary table with
three rows marked "本文补做"; this script covers the two that need training:

* **row 1 — ``--stage smooth``**: re-test *input smoothing* on **PhaseFormer-L**
  at the two frozen levels, on the 7 test-selected settings x 3 seeds x 2 levels
  = **42 runs**.

  | level | mechanism | head | override | operator |
  |---|---|---|---|---|
  | ``causal_ema_mid`` | ``weak_residual`` | ``shared`` | ``weak_period_residual_smooth_ratio=0.5`` + ``alpha=0.08`` | ``WeakPeriodResidualHead`` blends ``centered`` with a **causal EMA** |
  | ``causal_ema_max`` | ``weak_residual`` | ``shared`` | ``weak_period_residual_smooth_ratio=1.0`` + ``alpha=0.08`` | same blend at the strongest measured strength |

  Both levels use the operator the ``shared`` head actually implements and two
  *different* strengths from the probe grid ``s in {0, 0.25, 0.5, 0.75, 1}``.
  The original D-4 draft used one strength and varied only the EMA alpha, but
  ``alpha=0.08`` is already the operator's default, so those two "levels" were
  numerically identical; see the ``CAUSAL_EMA_ALPHA``/``SMOOTH_LEVELS`` comment
  for the correction and its evidence.  The mechanism is exactly PhaseFormer-L
  as defined by D-5: ``mechanism=weak_residual`` with the **dense** ``shared``
  head and the corrector always on.

* **row 5 — ``--stage rank12``**: boundary ablation, ``pooled_lowrank`` with
  ``pool_factor=1`` and an **absolute** ``weak_period_residual_rank`` in
  ``{1, 2}``, on the **6 settings of the E8 block** (ETTh2-96/720, ETTm2-96/192,
  Weather-96/192) x 3 seeds x 2 ranks = **36 runs**.

  This is **outside** the low-rank grid the minipaper calls "already measured"
  (deepest measured grid point ``q=1/32`` -> ranks 3/6/10/22).  §4.6
  pre-registers the expectation of degradation, and
  ``docs/PhaseFormer_L_experiment_plan.md`` §9.5 forbids reading the result as
  evidence for or against rank-2 necessity.

Protocol (minipaper §4.0, frozen decisions D-2/D-4; identical to E14):

* lookback 720, period 24, huber loss, 30 epochs, percent 100, best-validation
  checkpoint, seeds 2021/2022/2023;
* **every cell here is a new cell**, so it follows E14's **new-cell**
  hyperparameters ``gate_init=0.2`` / ``lr=1e-3`` (D-2) — *not* the Stage-0
  frozen ``(gate, lr)`` values that the 21 reused ``l_main`` cells of §4.2 carry.
  The ``_reused``/``new`` distinction is a property of a §4.2 cell, not of a
  setting: E14's cell table for these 21 cells is ``reused``, and this script's
  smoothing / rank cells are new runs with new config hashes;
* this stage **never passes ``--evaluate-test``**.  The single test read is a
  separate stage, exactly as in E14 (``scripts/phaseformer_L/e14_read_test.py``).

Reuse: the runner imports the E14 module and uses its ``_arm_match`` plus its
protocol constants, so "what is an ``l_main`` cell" is defined in exactly one
place.  ``--baseline-manifest`` points at E14's ``stage_a_manifest.json`` and
only *records* the matching ``l_main`` baseline cell (status, run dir, gate, lr)
next to every smoothed cell, so the §4.6 row-1 paired comparison is auditable;
it never changes what is trained.

Usage::

    # static check: build the manifest, train nothing
    python scripts/phaseformer_L/e18_negative.py --stage all --dry-run

    # row 1: 42 runs on 8 GPUs
    CUDA_VISIBLE_DEVICES= python scripts/phaseformer_L/e18_negative.py \
        --stage smooth --gpus 0,1,2,3,4,5,6,7 \
        --output-root research_runs/phaseformer_L_e18_negative_v1 \
        --baseline-manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json

    # row 5: 36 runs
    CUDA_VISIBLE_DEVICES= python scripts/phaseformer_L/e18_negative.py \
        --stage rank12 --gpus 0,1,2,3,4,5,6,7 \
        --output-root research_runs/phaseformer_L_e18_negative_v1
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

# The E14 module owns both the protocol constants and ``_arm_match``; importing
# it (stdlib only, no side effects at import time) keeps this runner's cell
# definitions provably identical to the §4.2 matrix's.
from scripts.phaseformer_L.e14_main_matrix import (  # noqa: E402
    LOOKBACK,
    LOSS,
    MAX_EPOCHS,
    NEW_CELL_GATE_INIT,
    NEW_CELL_LR,
    PERCENT,
    PERIOD,
    REUSE_SETTINGS_FULL,
    SEEDS,
    _arm_match,
    parse_list,
)

RUNNER = ROOT / "scripts" / "search_phaseformer.py"

STAGES = ("smooth", "rank12", "all")

# --- row 1 -----------------------------------------------------------------
# The 7 test-selected settings (schedule §2.1 / minipaper §4.1).  They are a
# test-set-selected set, not a blind sample, and must be disclosed as such.
SMOOTH_SETTINGS = tuple(REUSE_SETTINGS_FULL)
# The two frozen smoothing levels (D-4, corrected 2026-09-19).
#
# minipaper §4.6 row 1 reads "输入平滑（boxcar / causal EMA，各 5 档）… 在 PhaseFormer-L
# 上复测 2 档".  The pilot's "two operators" cannot both be reproduced on
# PhaseFormer-L, because the levels differ by *operator* only in the
# ``pooled_lowrank`` head: ``WeakPeriodResidualHead`` -- the ``shared`` head that
# PhaseFormer-L uses (D-5) -- implements ``smooth_ratio`` as a blend with
# ``_causal_ema`` (``src/models/phase_adapters.py:113-115``), whereas true boxcar
# (``F.avg_pool1d``) exists only in ``PooledLowRankWeakPeriodResidualHead``
# (``:168-179``).  So on PhaseFormer-L the operator is fixed and ``smooth_ratio``
# is a *strength*.
#
# The first draft of D-4 therefore collapsed: it used one strength (0.5) for both
# levels and varied only the EMA alpha, but ``alpha=0.08`` is already the default
# of both ``_causal_ema`` (``src/models/asymmetric_trend_components.py:116``) and
# the read key, so the two "levels" were numerically the same configuration.
#
# Corrected: the "2 档" are two genuinely different smoothing strengths of the
# causal-EMA blend, both inside the grid the probe sweep measured (s in
# {0, 0.25, 0.5, 0.75, 1}), namely the midpoint and the strongest setting.  The
# operator default alpha is written explicitly so each config self-describes.
CAUSAL_EMA_ALPHA = 0.08
SMOOTH_LEVELS = (
    {
        "level": "causal_ema_mid",
        "smooth_ratio": 0.5,
        "causal_ema_alpha": CAUSAL_EMA_ALPHA,
    },
    {
        "level": "causal_ema_max",
        "smooth_ratio": 1.0,
        "causal_ema_alpha": CAUSAL_EMA_ALPHA,
    },
)

# --- row 5 -----------------------------------------------------------------
# The 6-setting E8 block; Electricity-336 is deliberately excluded because the
# registered ablation is defined on these six (minipaper §4.6 row 5).
RANK12_SETTINGS = (
    ("ETTh2", 96), ("ETTh2", 720),
    ("ETTm2", 96), ("ETTm2", 192),
    ("Weather", 96), ("Weather", 192),
)
RANK12_RANKS = (1, 2)
POOL_FACTOR = 1

MECHANISM = "weak_residual"
SHARED_HEAD = "shared"
LOWRANK_HEAD = "pooled_lowrank"

# Rough per-setting training cost (seconds), mirrored from E14's COST_HINT, to
# launch expensive cells first so the matrix does not end on a long tail.
COST_HINT = {
    "ETTh2": {96: 47, 192: 37, 336: 60, 720: 57},
    "ETTm2": {96: 185, 192: 114, 336: 150, 720: 200},
    "Weather": {96: 1100, 192: 683, 336: 900, 720: 1200},
    "Electricity": {96: 1500, 192: 1600, 336: 1717, 720: 2000},
}


# --------------------------------------------------------------------------
# Cell construction
# --------------------------------------------------------------------------

def smooth_level_key(level: dict) -> str:
    """Cell key of a smoothing level, unique per configuration.

    The two levels differ in ``smooth_ratio`` and share the operator's default
    alpha, so the key carries the operator name and the strength.
    """
    return f"{level['level']}_s{level['smooth_ratio']:g}"


def smooth_overrides(level: dict) -> dict:
    """Override dict for one PhaseFormer-L smoothing cell.

    * ``weak_period_residual_head_type="shared"`` -> dense ``WeakPeriodResidualHead``
      (PhaseFormer-L, D-5);
    * ``weak_period_residual_smooth_ratio`` is the shared knob: for the dense
      head it is the blend weight between the raw centered input and the
      **causal EMA** of it (``src/models/phase_adapters.py:113-115``);
    * ``weak_period_residual_causal_ema_alpha`` pins the EMA rate to the
      operator's own default (0.08) explicitly.  It is written rather than left
      implicit so each config self-describes the operator, and so the two levels
      differ only in ``smooth_ratio``.  (The first D-4 draft varied the alpha
      instead of the strength and therefore produced two numerically identical
      levels, because 0.08 is already the default; corrected 2026-09-19.)
    * ``weak_period_residual_gate_init=0.2`` and ``learning_rate=1e-3`` are the
      D-2 new-cell values (see the cell's ``status`` == ``new``).
    """
    overrides = {
        "weak_period_residual_head_type": SHARED_HEAD,
        "weak_period_residual_smooth_ratio": level["smooth_ratio"],
        "weak_period_residual_gate_init": NEW_CELL_GATE_INIT,
        "learning_rate": NEW_CELL_LR,
    }
    if level["causal_ema_alpha"] is not None:
        overrides["weak_period_residual_causal_ema_alpha"] = level["causal_ema_alpha"]
    return overrides


def rank12_overrides(rank: int) -> dict:
    """Override dict for one absolute-rank boundary-ablation cell."""
    return {
        "weak_period_residual_head_type": LOWRANK_HEAD,
        "weak_period_residual_pool_factor": POOL_FACTOR,
        "weak_period_residual_rank": int(rank),
        "weak_period_residual_smooth_ratio": 0.0,
        "weak_period_residual_gate_init": NEW_CELL_GATE_INIT,
        "learning_rate": NEW_CELL_LR,
    }


def command(dataset: str, horizon: int, seed: int, mechanism: str,
            overrides: dict, output_root: str, num_workers: int,
            max_epochs: int = MAX_EPOCHS) -> list:
    """Build the runner argv (without the interpreter) for one E18 cell.

    No ``--evaluate-test`` anywhere: the single test read is a separate stage.
    """
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
        "--require-cuda",
        "--resume",
        "--num-workers", str(num_workers),
        "--bad-case-limit", "0",
        "--mechanism", mechanism,
        "--learning-rate", str(NEW_CELL_LR),
        "--overrides", json.dumps(overrides, sort_keys=True),
    ]


def build_cells(stages: tuple, datasets: list, seeds: list,
                levels: list, ranks: list) -> list:
    cells: list = []
    if "smooth" in stages:
        for dataset, horizon in SMOOTH_SETTINGS:
            if dataset not in datasets:
                continue
            for level in SMOOTH_LEVELS:
                if level["level"] not in levels:
                    continue
                key = smooth_level_key(level)
                for seed in seeds:
                    cells.append({
                        "stage": "smooth",
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "level": level["level"],
                        "level_key": key,
                        "rank": None,
                        "head_type": SHARED_HEAD,
                        "status": "new",
                        "overrides": smooth_overrides(level),
                    })
    if "rank12" in stages:
        for dataset, horizon in RANK12_SETTINGS:
            if dataset not in datasets:
                continue
            for rank in RANK12_RANKS:
                if rank not in ranks:
                    continue
                for seed in seeds:
                    cells.append({
                        "stage": "rank12",
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "level": None,
                        "level_key": f"absolute_rank{rank}",
                        "rank": int(rank),
                        "head_type": LOWRANK_HEAD,
                        "status": "new",
                        "overrides": rank12_overrides(rank),
                    })
    # Expensive settings first so the tail of the matrix is cheap.
    cells.sort(key=lambda c: (
        -COST_HINT.get(c["dataset"], {}).get(c["horizon"], 500),
        c["stage"], c["dataset"], c["horizon"], c["rank"] or 0, c["seed"],
    ))
    return cells


def cell_key(cell: dict) -> str:
    return (f"{cell['stage']}__{cell['dataset']}-{cell['horizon']}"
            f"-s{cell['seed']}__{cell['level_key']}")


# --------------------------------------------------------------------------
# E14 baseline resolution (row 1 annex only)
# --------------------------------------------------------------------------

def reused_arm_evidence(entry: dict) -> tuple:
    """Re-derive a *reused* cell's arm fingerprint from the run it adopted.

    A ``reused`` manifest entry carries no ``command``: stage A did not launch
    it, E14 adopted it from an earlier registered root after its own reuse audit
    passed (``e14_main_matrix._arm_match``).  So the argv path cannot be used
    for these cells -- and it is not a corner case, because the 7 row-1 settings
    of minipaper §4.6 *are* exactly the reused ones.

    The adopted run is on disk, so the fingerprint is re-derived from that run's
    own ``config.json`` with the same ``_arm_match`` test.  Returns
    ``(evidence, None)`` on success or ``(None, reason)`` on failure; a
    ``source`` dict is never trusted on its own.
    """
    source = entry.get("source") or {}
    if not isinstance(source, dict) or not source.get("run_dir"):
        return None, "reused cell has neither a command nor source.run_dir"
    run_dir = Path(str(source["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config = run_config(run_dir)
    if config is None:
        return None, f"reused run dir has no readable config.json: {run_dir}"
    if not _arm_match(config, "l_main"):
        return None, f"reused run config does not implement l_main: {run_dir}"
    hyper = config.get("hyperparams", {}) or {}
    # The root, not the run dir: E14's own cells are named
    # ``<root>/runs/<run-id>`` and the manifest records that root verbatim.
    root = source.get("root")
    return {
        "source": "e14_stage_a_manifest:reused_config",
        "arm": "l_main",
        "status": entry.get("status"),
        "run_dir": str(source["run_dir"]),
        "gate_init": hyper.get("weak_period_residual_gate_init"),
        "learning_rate": hyper.get("learning_rate"),
        "eval_root": str(root) if root else repo_relative(run_dir.parent),
    }, None


def load_baseline_index(path: Path) -> tuple:
    """Index E14's ``l_main`` cells from its stage-A manifest.

    Returns ``(index, report)`` with ``index[(dataset, horizon, seed)] = {...}``.
    A manifest entry is accepted only when it carries no ``--evaluate-test`` flag
    and implements the ``l_main`` arm, checked by one of two paths:

    * ``new`` cells were launched by stage A and carry their argv, so the
      fingerprint is re-derived from ``--overrides`` (an entry whose command was
      hand-edited is rejected);
    * ``reused`` cells carry no command, so ``reused_arm_evidence`` re-derives
      the fingerprint from the adopted run's ``config.json``.

    ``eval_root`` is recorded from evidence, never from a template: for a reused
    cell it is the adopted root the manifest recorded, and for a new cell it is
    the root that holds this manifest (``path.parent``), because a new cell's run
    directory is only named once the run exists.

    This function only *records* provenance; it never changes what E18 trains.
    """
    report = {"path": str(path), "exists": path.is_file(), "cells": 0,
              "resolved": 0, "resolved_new": 0, "resolved_reused": 0,
              "rejected": []}
    index: dict = {}
    if not report["exists"]:
        report["rejected"].append({"reason": "manifest file does not exist"})
        return index, report
    manifest = json.loads(path.read_text())
    entries = manifest.get("cells")
    if not isinstance(entries, list):
        report["rejected"].append({"reason": "manifest has no 'cells' list"})
        return index, report
    report["cells"] = len(entries)
    e14_root = repo_relative(path.parent)
    for entry in entries:
        argv = entry.get("command") or []
        if "--evaluate-test" in argv:
            report["rejected"].append({
                "cell": entry.get("key"),
                "reason": "stage-A command carries --evaluate-test",
            })
            continue
        if entry.get("arm") != "l_main":
            continue
        status = entry.get("status")
        if argv:
            overrides = {}
            if "--overrides" in argv:
                overrides = json.loads(argv[argv.index("--overrides") + 1])
            # The manifest records the arm's command; re-derive the fingerprint
            # from the arm table so an entry whose command was hand-edited is
            # rejected.
            candidate = {
                "mechanism": MECHANISM,
                "horizon": int(entry["horizon"]),
                "hyperparams": overrides,
            }
            if not _arm_match(candidate, "l_main"):
                report["rejected"].append({
                    "cell": entry.get("key"),
                    "reason": "overrides do not implement l_main",
                })
                continue
            evidence = {
                "source": "e14_stage_a_manifest",
                "arm": "l_main",
                "status": status,
                "run_dir": (entry.get("source") or {}).get("run_dir"),
                "gate_init": overrides.get("weak_period_residual_gate_init"),
                "learning_rate": overrides.get("learning_rate"),
                "eval_root": e14_root,
            }
            path_kind = "resolved_new"
        else:
            evidence, reason = reused_arm_evidence(entry)
            if evidence is None:
                report["rejected"].append({"cell": entry.get("key"),
                                           "reason": reason})
                continue
            path_kind = "resolved_reused"
        key = (entry["dataset"], int(entry["horizon"]), int(entry["seed"]))
        index[key] = evidence
        report["resolved"] += 1
        report[path_kind] += 1
    return index, report


def attach_baselines(cells: list, index: dict) -> None:
    for cell in cells:
        if cell["stage"] != "smooth":
            continue
        cell["baseline"] = index.get(
            (cell["dataset"], cell["horizon"], cell["seed"])
        )


# --------------------------------------------------------------------------
# Verification and summary
# --------------------------------------------------------------------------

def run_config(run_dir: Path):
    path = run_dir / "config.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _close(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) <= 1e-9
    except (TypeError, ValueError):
        return False


def config_matches(config, cell: dict) -> tuple:
    """Check one run directory's config against an E18 cell.

    Returns ``(ok, failures)``.  This is the fingerprint behind
    ``find_run_dir``, ``--verify`` and ``summarize``: a run is only counted as
    this cell's run when its config reproduces the cell's arm, protocol, head
    and D-2 new-cell hyperparameters exactly.
    """
    failures = []
    if config.get("mechanism") != MECHANISM:
        failures.append(f"mechanism={config.get('mechanism')!r}")
    hyper = config.get("hyperparams", {}) or {}
    if config.get("dataset") != cell["dataset"]:
        failures.append(f"dataset={config.get('dataset')!r}")
    if int(config.get("horizon", -1)) != int(cell["horizon"]):
        failures.append(f"horizon={config.get('horizon')!r}")
    if int(config.get("seed", -1)) != int(cell["seed"]):
        failures.append(f"seed={config.get('seed')!r}")
    for name, want in (("lookback", LOOKBACK), ("period", PERIOD),
                       ("max_epochs", MAX_EPOCHS), ("percent", PERCENT),
                       ("loss", LOSS)):
        if config.get(name) != want:
            failures.append(f"{name}={config.get(name)!r} != {want!r}")
    # E18 never reads test.  A config that says otherwise was not produced by
    # this runner's ``command()``, so it is rejected instead of being counted.
    try:
        read_test = int(config.get("evaluate_test", 0) or 0)
    except (TypeError, ValueError):
        read_test = 1
    if read_test:
        failures.append(f"evaluate_test={config.get('evaluate_test')!r}")
    if config.get("init_checkpoint"):
        failures.append(f"init_checkpoint={config.get('init_checkpoint')!r}")
    if hyper.get("weak_period_residual_head_type") != cell["head_type"]:
        failures.append(
            f"head_type={hyper.get('weak_period_residual_head_type')!r}")
    if not _close(hyper.get("weak_period_residual_gate_init"), NEW_CELL_GATE_INIT):
        failures.append(
            f"gate_init={hyper.get('weak_period_residual_gate_init')!r}")
    if not _close(hyper.get("learning_rate"), NEW_CELL_LR):
        failures.append(f"learning_rate={hyper.get('learning_rate')!r}")
    if hyper.get("weak_residual_projection"):
        failures.append(f"weak_residual_projection={hyper.get('weak_residual_projection')!r}")
    if str(hyper.get("input_hypothesis", "none") or "none") not in ("none", ""):
        failures.append(f"input_hypothesis={hyper.get('input_hypothesis')!r}")
    if cell["stage"] == "smooth":
        if not _close(hyper.get("weak_period_residual_smooth_ratio"),
                      cell["overrides"]["weak_period_residual_smooth_ratio"]):
            failures.append(
                f"smooth_ratio={hyper.get('weak_period_residual_smooth_ratio')!r}")
        want_alpha = cell["overrides"].get("weak_period_residual_causal_ema_alpha")
        got_alpha = hyper.get("weak_period_residual_causal_ema_alpha")
        if want_alpha is not None:
            # Both corrected levels write the operator's default alpha
            # explicitly, so the recorded config must match it exactly.
            if not _close(got_alpha, want_alpha):
                failures.append(f"causal_ema_alpha={got_alpha!r}")
        elif got_alpha is not None and not _close(got_alpha, CAUSAL_EMA_ALPHA):
            # An unoverridden alpha is only acceptable if it still equals the
            # operator default; anything else is not the frozen level.
            failures.append(f"causal_ema_alpha={got_alpha!r} (not the default)")
        if hyper.get("weak_period_residual_rank") is not None:
            failures.append(
                f"rank={hyper.get('weak_period_residual_rank')!r} (shared head)")
    else:
        if not int(hyper.get("weak_period_residual_rank", -1)) == int(cell["rank"]):
            failures.append(f"rank={hyper.get('weak_period_residual_rank')!r}")
        if not int(hyper.get("weak_period_residual_pool_factor", -1)) == POOL_FACTOR:
            failures.append(
                f"pool_factor={hyper.get('weak_period_residual_pool_factor')!r}")
        if not _close(hyper.get("weak_period_residual_smooth_ratio") or 0.0, 0.0):
            failures.append(
                f"smooth_ratio={hyper.get('weak_period_residual_smooth_ratio')!r}")
    return (not failures), failures


def find_run_dir(output_root: Path, cell: dict, max_epochs: int = MAX_EPOCHS):
    """Locate the run directory of one cell by matching its config.json.

    ``Mirrors E14's approach: the run id embeds a config hash, so the directory
    name cannot be predicted from the cell alone.  A run that carries
    ``metrics.csv`` (i.e. actually finished) outranks a config-only directory
    left behind by an interrupted attempt; ties are broken by path so the choice
    is deterministic.  Near-miss runs (same dataset/horizon/seed, wrong
    fingerprint) are returned so a missing cell is auditable.
    """
    matches = []
    near = []
    runs_dir = output_root / "runs"
    if not runs_dir.is_dir():
        return None, near
    for config_path in sorted(runs_dir.glob("*/config.json")):
        config = run_config(config_path.parent)
        if not config:
            continue
        if (config.get("dataset") != cell["dataset"]
                or int(config.get("horizon", -1)) != int(cell["horizon"])
                or int(config.get("seed", -1)) != int(cell["seed"])):
            continue
        ok, failures = config_matches(config, cell)
        if ok:
            matches.append((config_path.parent, config, failures))
        else:
            near.append({
                "run_dir": repo_relative(config_path.parent),
                "failures": failures,
            })
    matches.sort(key=lambda item: (
        0 if (item[0] / "metrics.csv").is_file() else 1, str(item[0])))
    if not matches:
        return None, near
    return (matches[0][0], matches[0][1], matches[0][2]), near


def verify_cells(cells: list, output_root: Path, max_epochs: int = MAX_EPOCHS) -> dict:
    """Every cell must have exactly one matching completed run."""
    report = {"checked": len(cells), "resolved": 0, "missing": [], "near_misses": []}
    for cell in cells:
        found, near = find_run_dir(output_root, cell, max_epochs)
        if found is None:
            report["missing"].append({
                "cell": cell_key(cell),
                "near_misses": near,
            })
            continue
        run_dir, _config, _failures = found
        if not (run_dir / "metrics.csv").is_file():
            report["missing"].append({
                "cell": cell_key(cell),
                "reason": f"no metrics.csv in {repo_relative(run_dir)}",
                "near_misses": near,
            })
            continue
        report["resolved"] += 1
    return report


RESULTS_FIELDS = [
    "stage", "dataset", "horizon", "seed", "cell", "level", "rank",
    "mechanism", "head_type", "gate_init", "learning_rate",
    "smooth_ratio", "causal_ema_alpha", "rank_override", "pool_factor",
    "val_mse", "val_mae", "best_val_loss", "epochs_completed",
    "parameter_count", "elapsed_sec", "run_id", "run_dir", "config_hash",
    "checkpoint", "test_mse_recorded", "test_mae_recorded",
    "baseline_status", "baseline_run_dir", "baseline_gate_init",
    "baseline_learning_rate", "baseline_eval_root",
]


def summarize(cells: list, output_root: Path, max_epochs: int = MAX_EPOCHS) -> list:
    """One row per cell, read back from the run directories that exist."""
    rows = []
    for cell in cells:
        found, _near = find_run_dir(output_root, cell, max_epochs)
        if found is None:
            continue
        run_dir, config, _failures = found
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(newline="") as handle:
            metrics = next(csv.DictReader(handle))
        hyper = config.get("hyperparams", {}) or {}
        baseline = cell.get("baseline") or {}
        rows.append({
            "stage": cell["stage"],
            "dataset": cell["dataset"],
            "horizon": cell["horizon"],
            "seed": cell["seed"],
            "cell": cell_key(cell),
            "level": cell["level"] or "",
            "rank": cell["rank"] if cell["rank"] is not None else "",
            "mechanism": config.get("mechanism", ""),
            "head_type": hyper.get("weak_period_residual_head_type", SHARED_HEAD),
            "gate_init": hyper.get("weak_period_residual_gate_init", ""),
            "learning_rate": hyper.get("learning_rate", ""),
            "smooth_ratio": hyper.get("weak_period_residual_smooth_ratio", 0.0),
            "causal_ema_alpha": hyper.get("weak_period_residual_causal_ema_alpha", ""),
            "rank_override": hyper.get("weak_period_residual_rank", ""),
            "pool_factor": hyper.get("weak_period_residual_pool_factor", ""),
            "val_mse": metrics.get("val_mse", ""),
            "val_mae": metrics.get("val_mae", ""),
            "best_val_loss": metrics.get("best_val_loss", ""),
            "epochs_completed": metrics.get("epochs_completed", ""),
            "parameter_count": metrics.get("parameter_count", ""),
            "elapsed_sec": metrics.get("elapsed_sec", ""),
            "run_id": metrics.get("run_id", ""),
            "run_dir": repo_relative(run_dir),
            "config_hash": config.get("config_hash", ""),
            "checkpoint": metrics.get("checkpoint", ""),
            "test_mse_recorded": bool(str(metrics.get("test_mse", "")).strip()),
            "test_mae_recorded": bool(str(metrics.get("test_mae", "")).strip()),
            "baseline_status": baseline.get("status", ""),
            "baseline_run_dir": baseline.get("run_dir", "") or "",
            "baseline_gate_init": baseline.get("gate_init", ""),
            "baseline_learning_rate": baseline.get("learning_rate", ""),
            "baseline_eval_root": baseline.get("eval_root", "") or "",
        })
    rows.sort(key=lambda row: (row["stage"], row["dataset"], row["horizon"],
                               row["rank"] if row["rank"] != "" else -1,
                               row["level"], row["seed"]))
    return rows


def write_results_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Dispatch (one run per GPU, retries, JSON progress; the E14 pattern)
# --------------------------------------------------------------------------

def repo_relative(path: Path) -> str:
    """Path relative to the repo when possible, else the path as given.

    ``--output-root`` may be absolute (or outside the repo), so a bare
    ``relative_to`` must never be allowed to fail the run.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def dispatch(cells: list, gpus: list, output_root: str, num_workers: int,
             retries: int, poll: int, max_epochs: int = MAX_EPOCHS):
    pending = list(cells)
    active: dict = {}
    attempts: dict = {}
    completed, failed = [], []
    log_dir = Path(output_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell_key(cell)
            attempts[key] = attempts.get(key, 0) + 1
            argv = command(cell["dataset"], cell["horizon"], cell["seed"],
                           MECHANISM, cell["overrides"], output_root,
                           num_workers, max_epochs)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(log_dir / f"{key}.log", "w")
            process = subprocess.Popen(
                [sys.executable, *argv], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
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
            key = cell_key(cell)
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--output-root",
                        default="research_runs/phaseformer_L_e18_negative_v1")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--datasets", default=",".join(
        sorted({d for d, _ in SMOOTH_SETTINGS} | {d for d, _ in RANK12_SETTINGS})))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--levels", default=",".join(
        level["level"] for level in SMOOTH_LEVELS))
    parser.add_argument("--ranks", default=",".join(str(r) for r in RANK12_RANKS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--baseline-manifest", default="",
                        help="E14 stage_a_manifest.json; records the l_main "
                             "baseline next to every smoothed cell")
    parser.add_argument("--verify", action="store_true",
                        help="fail if any planned cell has no matching completed run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    stages = STAGES[:2] if args.stage == "all" else (args.stage,)
    out_root = ROOT / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = parse_list(args.datasets)
    seeds = parse_list(args.seeds, int)
    levels = parse_list(args.levels)
    ranks = parse_list(args.ranks, int)
    gpus = parse_list(args.gpus, int)
    if not gpus:
        raise SystemExit("--gpus must contain at least one device")
    known_levels = {level["level"] for level in SMOOTH_LEVELS}
    if "smooth" in stages and not set(levels) <= known_levels:
        raise SystemExit(f"unknown --levels {sorted(set(levels) - known_levels)}; "
                         f"known: {sorted(known_levels)}")
    if "rank12" in stages and not set(ranks) <= set(RANK12_RANKS):
        raise SystemExit(f"unknown --ranks {sorted(set(ranks) - set(RANK12_RANKS))}; "
                         f"known: {sorted(RANK12_RANKS)}")

    cells = build_cells(stages, datasets, seeds, levels, ranks)

    baseline_index: dict = {}
    baseline_report = None
    if args.baseline_manifest:
        baseline_index, baseline_report = load_baseline_index(
            Path(args.baseline_manifest))
        attach_baselines(cells, baseline_index)

    counts: dict = {}
    for cell in cells:
        bucket = counts.setdefault(cell["stage"], {})
        bucket[cell["level_key"]] = bucket.get(cell["level_key"], 0) + 1

    manifest = {
        "experiment": "E18 (minipaper section 4.6 negative controls, training only)",
        "schedule_reference": "docs/PhaseFormer_L_execution_schedule.md section 2.3",
        "reads_test": False,
        "evaluate_test_passed": False,
        "test_read_note": (
            "this runner never builds the --evaluate-test flag; the single test "
            "read is a separate stage, as in E14 stage A/B"
        ),
        "stages": list(stages),
        "protocol": {
            "lookback": LOOKBACK, "period": PERIOD, "loss": LOSS,
            "max_epochs": MAX_EPOCHS, "percent": PERCENT,
            "seeds": seeds,
            "checkpoint": "best validation loss",
            "cell_status": "new (every E18 cell is a new cell)",
            "new_cell_overrides": {
                "weak_period_residual_gate_init": NEW_CELL_GATE_INIT,
                "learning_rate": NEW_CELL_LR,
                "rationale": "D-2: new cells use the preset defaults; D-4 fixes "
                             "the scope to the 7 test-selected settings",
            },
            "row1_levels": [
                {"level": level["level"],
                 "key": smooth_level_key(level),
                 "smooth_ratio": level["smooth_ratio"],
                 "causal_ema_alpha": (level["causal_ema_alpha"]
                                      if level["causal_ema_alpha"] is not None
                                      else CAUSAL_EMA_ALPHA),
                 "alpha_overridden": level["causal_ema_alpha"] is not None}
                for level in SMOOTH_LEVELS
            ],
            "row1_levels_frozen_note": (
                "Two different causal-EMA blend strengths (smooth_ratio 0.5 and "
                "1.0) at the operator's default alpha 0.08, both inside the "
                "probe grid. The earlier draft varied only the alpha, which is "
                "already the default and so produced numerically identical "
                "configurations; D-4 was corrected on 2026-09-19."
            ),
            "row5_ranks": list(RANK12_RANKS),
            "row5_grid_note": (
                "absolute rank 1/2 is outside the measured low-rank grid whose "
                "deepest point is q=1/32 (ranks 3/6/10/22); section 4.6 "
                "pre-registers degradation and experiment_plan section 9.5 "
                "forbids reading it as evidence about rank-2 necessity"
            ),
        },
        "settings": {
            "smooth": [f"{d}-{h}" for d, h in SMOOTH_SETTINGS],
            "rank12": [f"{d}-{h}" for d, h in RANK12_SETTINGS],
            "selection_disclosure": (
                "the 7 row-1 settings are the test-set-selected set of "
                "minipaper section 4.1; the 6 row-5 settings are the E8 block "
                "and exclude Electricity-336"
            ),
        },
        "gpus": gpus,
        "counts": {stage: {"total": sum(bucket.values()), "by_level": bucket}
                   for stage, bucket in counts.items()},
        "baseline_manifest": baseline_report,
        "cells": [
            {**{k: v for k, v in cell.items() if k != "baseline"},
             "key": cell_key(cell),
             "baseline": cell.get("baseline"),
             "command": command(cell["dataset"], cell["horizon"], cell["seed"],
                                MECHANISM, cell["overrides"], args.output_root,
                                args.num_workers)}
            for cell in cells
        ],
        "reuse_imported_from_e14": {
            "module": "scripts/phaseformer_L/e14_main_matrix.py",
            "arm_match_used_for_baselines": "_arm_match",
        },
    }
    manifest_path = out_root / "e18_negative_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "planned",
                      "manifest": repo_relative(manifest_path),
                      "counts": manifest["counts"]}, ensure_ascii=False))

    if args.verify:
        report = verify_cells(cells, out_root)
        (out_root / "e18_negative_verify.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "verify",
                          "resolved": report["resolved"],
                          "missing": len(report["missing"])}, ensure_ascii=False),
              flush=True)
        if report["missing"]:
            for item in report["missing"]:
                print(json.dumps({"event": "missing", **item}),
                      flush=True)
            raise SystemExit(
                f"verify failed: {len(report['missing'])} of {report['checked']} "
                "cells have no matching completed run"
            )

    if args.dry_run:
        for cell in cells[:40]:
            print(json.dumps({"cell": cell_key(cell),
                              "status": cell["status"],
                              "overrides": cell["overrides"]}, sort_keys=True))
        print(json.dumps({"event": "plan_only", "cells": len(cells)}))
        return

    completed, failed = dispatch(cells, gpus, args.output_root,
                                 args.num_workers, args.retries,
                                 args.poll_seconds)
    rows = summarize(cells, out_root)
    results_path = out_root / "results.csv"
    write_results_csv(rows, results_path)
    summary = {
        "stage": args.stage,
        "cells_planned": len(cells),
        "trained": len(completed),
        "failed": failed,
        "completed": completed,
        "results_rows": len(rows),
        "results_csv": repo_relative(results_path),
        "reads_test": False,
    }
    (out_root / f"e18_negative_summary_{args.stage}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "stage_finished", "stage": args.stage,
                      "trained": len(completed), "failed": len(failed),
                      "results_rows": len(rows)}, ensure_ascii=False))
    if failed:
        raise SystemExit(f"E18 stage {args.stage} had failed cells: {failed}")


if __name__ == "__main__":
    main()
