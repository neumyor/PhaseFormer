#!/usr/bin/env python3
"""E14 stage A: train the new §4.2 main-table cells (no test read).

Implements `docs/PhaseFormer_L_execution_schedule.md` §2.2.  The §4.2 main table
is 24 settings (ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity x H96/192/336/720)
plus a 4-setting Traffic appendix, with six variant rows.  Cells that already
have an audited three-seed record are **reused, not retrained**; every other
cell is trained here.

Protocol (minipaper §4.0, frozen decisions D-1..D-4):

* lookback 720, period 24, huber loss, 30 epochs, percent 100, best-validation
  checkpoint, seeds 2021/2022/2023;
* **new** cells use the preset defaults ``gate_init=0.2`` / ``lr=1e-3`` (D-2).
  Reused cells keep whatever Stage-0 frozen ``(gate, lr)`` their own run used,
  so the §4.2 table mixes two hyperparameter agreements and must say so;
* this stage **never passes ``--evaluate-test``**.  Test is read exactly once,
  afterwards, by ``scripts/phaseformer_L/e14_read_test.py``.

Reuse scope is the audited set only, discovered by scanning an explicit root
whitelist and validated field by field (see ``resolve_reuse``).  Roots produced
by E1 (``joint_lowrank_rank_sweep_v1``), by scratch runs, or by the selection /
intervention experiments are deliberately excluded -- E8's ``reuse_audit.json``
already rejected the E1 batches for ``gate_init``/``learning_rate`` mismatch.

Usage::

    # static check: build the manifest, validate the reuse set, train nothing
    python scripts/phaseformer_L/e14_main_matrix.py --stage plan --verify

    # real run on all 8 GPUs
    CUDA_VISIBLE_DEVICES= python scripts/phaseformer_L/e14_main_matrix.py \\
        --stage a --gpus 0,1,2,3,4,5,6,7 \\
        --output-root research_runs/phaseformer_L_e14_main_v1
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

RUNNER = ROOT / "scripts" / "search_phaseformer.py"

MAIN_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
TRAFFIC_DATASETS = ("Traffic",)
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)

# Protocol constants (minipaper §4.0).
LOOKBACK = 720
PERIOD = 24
MAX_EPOCHS = 30
LOSS = "huber"
PERCENT = 100
NEW_CELL_GATE_INIT = 0.2      # D-2: preset default for new cells
NEW_CELL_LR = 1e-3            # D-2

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

MAIN_ARMS = ("phase_only", "l_main", "l_q1_4", "l_q1_8", "l_rcrf")
OPTIONAL_ARMS = ("a1",)

# Audited reuse scope: the seven test-selected settings (minipaper §4.1/§4.2).
REUSE_SETTINGS_FULL = (
    ("ETTh2", 96), ("ETTh2", 720),
    ("ETTm2", 96), ("ETTm2", 192),
    ("Weather", 96), ("Weather", 192),
    ("Electricity", 336),
)
# phase_only additionally lacks Electricity-336 (E8 did not cover it).
REUSE_SETTINGS_PHASE_ONLY = tuple(
    s for s in REUSE_SETTINGS_FULL if s != ("Electricity", 336)
)

# Arm -> audited reuse settings.  An arm absent from this map has no reusable
# cell at all (rcrf/a1), which the 2026-09-18 audit confirmed on the server.
REUSE_SCOPE = {
    "phase_only": REUSE_SETTINGS_PHASE_ONLY,
    "l_main": REUSE_SETTINGS_FULL,
    "l_q1_4": REUSE_SETTINGS_FULL,
    "l_q1_8": REUSE_SETTINGS_FULL,
}

# Only these roots may supply reused cells.  Everything else -- E1, scratch,
# selection experiments, intervention sweeps -- is excluded on purpose.
REUSE_ROOT_WHITELIST = (
    "research_runs/rank_sweep_2_stage1",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_v3",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_v4",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_v5",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1",
    "research_runs/top2_direction_retention_v1",
)

# Rough per-setting training cost (seconds) from the server's 677 metrics.csv
# records, used only to launch expensive cells first so the 8 GPUs do not end
# with a long tail.  Unknown settings fall back to a mid-range estimate.
COST_HINT = {
    "ETTh1": {96: 85, 192: 74, 336: 110, 720: 160},
    "ETTh2": {96: 47, 192: 37, 336: 60, 720: 57},
    "ETTm1": {96: 253, 192: 244, 336: 300, 720: 380},
    "ETTm2": {96: 185, 192: 114, 336: 150, 720: 200},
    "Weather": {96: 1100, 192: 683, 336: 900, 720: 1200},
    "Electricity": {96: 1500, 192: 1600, 336: 1717, 720: 2000},
    "Traffic": {96: 4200, 192: 4200, 336: 4200, 720: 4200},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "a"], default="plan")
    parser.add_argument("--output-root", default="research_runs/phaseformer_L_e14_main_v1")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--arms", default=",".join(MAIN_ARMS),
                        help="comma list; default excludes the optional A1 row")
    parser.add_argument("--datasets", default=",".join(MAIN_DATASETS))
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--verify", action="store_true",
                        help="fail if a declared reuse cell cannot be resolved")
    parser.add_argument("--no-reuse", action="store_true",
                        help="train every cell (protocol check / fallback)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_list(raw, cast=str):
    return [cast(item) for item in str(raw).split(",") if item.strip()]


# --------------------------------------------------------------------------
# Reuse discovery and audit
# --------------------------------------------------------------------------

def _load_config(run_dir: Path):
    path = run_dir / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _has_test_metrics(run_dir: Path) -> bool:
    """A reusable cell must already carry its single test read."""
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return False
    try:
        with metrics.open() as handle:
            for row in csv.DictReader(handle):
                return bool(str(row.get("test_mse", "")).strip()) and bool(
                    str(row.get("test_mae", "")).strip()
                )
    except Exception:
        return False
    return False


def _protocol_ok(config) -> tuple[bool, list[str]]:
    checks = [
        ("lookback", config.get("lookback"), LOOKBACK),
        ("loss", config.get("loss"), LOSS),
        ("max_epochs", config.get("max_epochs"), MAX_EPOCHS),
        ("percent", config.get("percent"), PERCENT),
        ("period", config.get("period"), PERIOD),
    ]
    failures = [f"{n}={got} != {want}" for n, got, want in checks if got != want]
    return (not failures), failures


def _arm_match(config, arm: str) -> bool:
    """Does this run's config implement exactly the given arm?"""
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


def resolve_reuse(arm: str, wanted: set) -> tuple[dict, list]:
    """Find one audited run dir per wanted (dataset, horizon, seed) cell.

    Returns ``(index, rejected)`` where ``index`` maps the cell tuple to
    ``{"run_dir", "config_hash"}`` and ``rejected`` records near-misses so the
    audit trail shows why a candidate was not taken.
    """
    index: dict = {}
    rejected: list = []
    cells_by_setting: dict = {}
    for dataset, horizon, seed in wanted:
        cells_by_setting.setdefault((dataset, horizon), []).append(seed)

    for root_rel in REUSE_ROOT_WHITELIST:
        root = ROOT / root_rel
        if not root.exists():
            continue
        for config_path in sorted(root.glob("runs/*/config.json")):
            run_dir = config_path.parent
            config = _load_config(run_dir)
            if not config:
                continue
            key = (config.get("dataset"), int(config.get("horizon", -1)),
                   int(config.get("seed", -1)))
            if key[:2] not in cells_by_setting:
                continue
            if key[2] not in cells_by_setting[key[:2]]:
                continue
            if not _arm_match(config, arm):
                continue
            ok, failures = _protocol_ok(config)
            if not ok:
                rejected.append({"run_dir": str(run_dir.relative_to(ROOT)),
                                 "failures": failures})
                continue
            if not _has_test_metrics(run_dir):
                rejected.append({"run_dir": str(run_dir.relative_to(ROOT)),
                                 "failures": ["no test metrics recorded"]})
                continue
            if key in index:
                continue  # keep the first whitelisted root's match; deterministic
            index[key] = {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "config_hash": config.get("config_hash"),
                "root": root_rel,
            }
    return index, rejected


def build_reuse() -> tuple[dict, dict, list]:
    """Resolve reuse for every arm.  Returns (per-arm index, summary, rejected)."""
    per_arm: dict = {}
    summary: dict = {}
    rejected: list = []
    for arm, settings in REUSE_SCOPE.items():
        wanted = {(d, h, s) for (d, h) in settings for s in SEEDS}
        index, rej = resolve_reuse(arm, wanted)
        per_arm[arm] = index
        missing = sorted(wanted - set(index))
        summary[arm] = {
            "declared": len(wanted),
            "resolved": len(index),
            "missing": [f"{d}-{h}-s{s}" for d, h, s in missing],
        }
        rejected.extend({"arm": arm, **entry} for entry in rej)
    return per_arm, summary, rejected


# --------------------------------------------------------------------------
# Cell construction and commands
# --------------------------------------------------------------------------

def arm_command(arm: str, dataset: str, horizon: int, seed: int,
                output_root: str, num_workers: int) -> list:
    """Build the runner argv (without the interpreter) for one new cell."""
    spec = ARMS[arm]
    overrides: dict = {}
    argv = [
        str(RUNNER),
        "--output-dir", output_root,
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm",
        "--lookback", str(LOOKBACK),
        "--period", str(PERIOD),
        "--max-epochs", str(MAX_EPOCHS),
        "--seed", str(seed),
        "--loss", LOSS,
        "--percent", str(PERCENT),
        "--require-cuda",
        "--resume",
        "--num-workers", str(num_workers),
        "--bad-case-limit", "0",
        "--mechanism", spec["mechanism"],
    ]
    # D-2: new cells run at the preset defaults.  rcrf_nlinear_plain owns its
    # own gate prior inside the preset (0.5), so only the learning rate is set.
    if spec["mechanism"] == "weak_residual":
        overrides["weak_period_residual_gate_init"] = NEW_CELL_GATE_INIT
        if spec.get("head"):
            overrides["weak_period_residual_head_type"] = spec["head"]
        if spec.get("rank_div"):
            overrides["weak_period_residual_pool_factor"] = 1
            overrides["weak_period_residual_rank"] = int(horizon) // spec["rank_div"]
    argv += ["--learning-rate", str(NEW_CELL_LR)]
    overrides["learning_rate"] = NEW_CELL_LR
    argv += ["--overrides", json.dumps(overrides, sort_keys=True)]
    return argv


def build_cells(args, reuse) -> list:
    arms = parse_list(args.arms)
    datasets = parse_list(args.datasets)
    horizons = parse_list(args.horizons, int)
    seeds = parse_list(args.seeds, int)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {sorted(ARMS)}")

    cells = []
    for arm in arms:
        for dataset in datasets:
            for horizon in horizons:
                for seed in seeds:
                    key = (dataset, horizon, seed)
                    # Arms absent from REUSE_SCOPE (l_rcrf, a1) have no reusable
                    # cell at all, so their index is empty by construction.
                    source = None if args.no_reuse else reuse.get(arm, {}).get(key)
                    cells.append({
                        "arm": arm,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "status": "reused" if source else "new",
                        "source": source,
                    })
    # Launch expensive cells first so the tail of the matrix is cheap.
    cells.sort(key=lambda c: (
        -COST_HINT.get(c["dataset"], {}).get(c["horizon"], 500),
        c["arm"], c["dataset"], c["horizon"], c["seed"],
    ))
    return cells


def cell_key(cell) -> str:
    return f"{cell['arm']}__{cell['dataset']}-{cell['horizon']}-s{cell['seed']}"


def dispatch(cells, gpus, output_root, num_workers, retries, poll):
    pending = [c for c in cells if c["status"] == "new"]
    active: dict = {}
    attempts: dict = {}
    completed, failed = [], []
    log_dir = Path(output_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        free = [g for g in gpus if g not in active]
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            key = cell_key(cell)
            attempts[key] = attempts.get(key, 0) + 1
            argv = arm_command(cell["arm"], cell["dataset"], cell["horizon"],
                               cell["seed"], output_root, num_workers)
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


def main() -> None:
    args = parse_args()
    out_root = ROOT / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    reuse, reuse_summary, rejected = build_reuse()
    cells = build_cells(args, reuse)

    total = len(cells)
    counters: dict = {}
    for cell in cells:
        bucket = counters.setdefault(cell["arm"], {"new": 0, "reused": 0})
        bucket[cell["status"]] += 1

    manifest = {
        "experiment": "E14 stage A (minipaper §4.2 main matrix, training only)",
        "reads_test": False,
        "protocol": {
            "lookback": LOOKBACK, "period": PERIOD, "loss": LOSS,
            "max_epochs": MAX_EPOCHS, "percent": PERCENT,
            "seeds": parse_list(args.seeds, int),
            "checkpoint": "best validation loss",
            "new_cell_overrides": {"weak_period_residual_gate_init": NEW_CELL_GATE_INIT,
                                   "learning_rate": NEW_CELL_LR},
            "reused_cell_hyperparams": "kept from their own audited Stage-0 freeze (D-2)",
        },
        "reuse_roots_whitelist": list(REUSE_ROOT_WHITELIST),
        "reuse_scope": {arm: [f"{d}-{h}" for d, h in s]
                        for arm, s in REUSE_SCOPE.items()},
        "reuse_summary": reuse_summary,
        "reuse_rejected_candidates": rejected,
        "gpus": parse_list(args.gpus, int),
        "arm_filters": parse_list(args.arms),
        "no_reuse": bool(args.no_reuse),
        "counts": {"total": total, "by_arm": counters},
        "cells": [
            {**cell, "key": cell_key(cell),
             "command": None if cell["status"] == "reused" else arm_command(
                 cell["arm"], cell["dataset"], cell["horizon"], cell["seed"],
                 args.output_root, args.num_workers)}
            for cell in cells
        ],
    }
    (out_root / "stage_a_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_root / "stage_a_reuse_audit.json").write_text(
        json.dumps({"reuse_summary": reuse_summary,
                    "rejected_candidates": rejected,
                    "roots_whitelist": list(REUSE_ROOT_WHITELIST)},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"event": "planned", "total": total, "by_arm": counters,
                      "gpus": manifest["gpus"]}, ensure_ascii=False))

    # Static-check gate: every declared reuse cell must resolve.
    problems = []
    if not args.no_reuse:
        for arm, info in reuse_summary.items():
            if arm not in parse_list(args.arms):
                continue
            if info["missing"]:
                problems.append(f"{arm}: unresolved reuse cells {info['missing']}")
    if args.verify and problems:
        for problem in problems:
            print(json.dumps({"event": "verify_failed", "problem": problem}),
                  flush=True)
        raise SystemExit("reuse verification failed; refusing to train")
    if args.verify:
        print(json.dumps({"event": "verify_ok",
                          "reuse_cells_resolved": sum(
                              v["resolved"] for v in reuse_summary.values())}))
    if problems:
        for problem in problems:
            print(json.dumps({"event": "warning", "problem": problem}), flush=True)

    if args.stage == "plan" or args.dry_run:
        for cell in cells[:20]:
            print(json.dumps({"cell": cell_key(cell), "status": cell["status"]}))
        print(json.dumps({"event": "plan_only", "cells": total}))
        return

    completed, failed = dispatch(cells, manifest["gpus"], args.output_root,
                                args.num_workers, args.retries, args.poll_seconds)
    summary = {"stage": "a", "cells_planned": total,
               "trained": len(completed), "failed": failed,
               "completed": completed}
    (out_root / "stage_a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "stage_finished", "trained": len(completed),
                      "failed": len(failed)}, ensure_ascii=False))
    if failed:
        raise SystemExit(f"stage a had failed cells: {failed}")


if __name__ == "__main__":
    main()
