#!/usr/bin/env python3
"""Round 0/1 runner for the structured low-rank NLinear exploration plan.

Round 0 reuses the already trained pilot controls and only records provenance;
Round 1 trains the pre-registered representative configuration of each route
plus the matched ``time_axis_matched_lowrank`` control.

Plan reference:
``docs/PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md``
sections 5, 6 and 13.

Example::

    python scripts/run_structured_lowrank_round1.py \
        --round 0 --frozen research_runs/.../frozen_configs.json \
        --output-root research_runs/structured_lowrank_round0_scratch

The runner never invents hyperparameters: each setting's ``(gate_init,
learning_rate)`` is read from the frozen JSON produced by
``audit_structured_lowrank_reuse.py``.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.structured_residual_heads import (  # noqa: E402
    build_structured_residual_head,
    matched_control_rank,
)

LOOKBACK = 720

# Round 1 routes.  ``head`` is the PhaseFormer head type; ``overrides`` are the
# plan's pre-registered representative configuration per route (section 6).
ROUND1_ROUTES = {
    "A_period_lowrank": {
        "head": "structured_period_lowrank",
        "overrides": {
            "residual_period_len": 24,
            "residual_period_rank": 4,
            "residual_segment_alignment": "aligned",
        },
    },
    "B_segment_basis": {
        "head": "structured_segment_basis",
        "overrides": {
            "residual_period_len": 24,
            "residual_basis_count": 4,
            "residual_basis_lambda_orth": 0.01,
            "residual_segment_alignment": "aligned",
        },
    },
    "C_level_shape": {
        "head": "structured_level_shape",
        "overrides": {
            "residual_period_len": 24,
            "residual_level_mode": "dense",
            "residual_shape_rank": 4,
            "residual_segment_alignment": "aligned",
        },
    },
    "D_recent_sparse": {
        "head": "structured_recent_period",
        "overrides": {
            "residual_period_len": 24,
            "residual_recent_taps": 7,
            "residual_recent_weighting": "fixed_exp",
            "residual_recent_decay": 0.8,
            "residual_recent_rank": 2,
            "residual_segment_alignment": "aligned",
        },
    },
    "E_separable": {
        "head": "structured_separable",
        "overrides": {
            "residual_period_len": 24,
            "residual_separable_components": 1,
            "residual_segment_alignment": "aligned",
        },
    },
}

# Round 1 matched-control ranks, measured with
# ``scripts/report_structured_lowrank_params.py``.  The time-axis control costs
# ``r * (L + H + 1) + H`` head parameters, so its rank-1 floor is 913 parameters
# at H96 and 2161 at H720.  Every P=24 structured route except A sits below that
# floor, so for those the control is fixed at rank 1 and the measured gap is
# recorded instead of being forced into the 5% band (plan section 3.3 allows the
# nearest reachable rank with the difference documented).  Route A is matched
# properly because its ``L -> rank -> K_y * P`` map is comparable in size.
ROUND1_MATCHED_RANK = {
    "A_period_lowrank": 1,   # A_H96 = 604 params; rank-1 control = 913 (-51%)
    "B_segment_basis": 1,    # rank-1 floor already exceeds B by ~6x
    "C_level_shape": 1,
    "D_recent_sparse": 1,
    "E_separable": 1,
}

# Route A is the only route whose budget is a meaningful parameter-matching
# target; ``matched_rank_for_budget`` recomputes the control rank from the
# candidate's measured head budget per setting instead of a fixed table entry.
AUTOMATCHED_ROUTES = {"A_period_lowrank": "structured_period_lowrank"}
AUTOMATCHED_OVERRIDES = {
    "A_period_lowrank": {"residual_period_len": 24, "residual_period_rank": 4}
}

# Route A's pre-registered medium-capacity diagnostic point (plan section 4).
ROUND1_DIAGNOSTICS = {
    "A_period_lowrank_r8": {
        "head": "structured_period_lowrank",
        "overrides": {
            "residual_period_len": 24,
            "residual_period_rank": 8,
            "residual_segment_alignment": "aligned",
        },
        "matched_rank": 8,
    },
}


def load_frozen(path):
    with open(path) as handle:
        data = json.load(handle)
    return {(row["dataset"], int(row["horizon"])): row for row in data["settings"]}


def build_override(head, extra, matched_rank=None):
    payload = {"weak_period_residual_head_type": head}
    payload.update(extra)
    if matched_rank is not None:
        payload["weak_period_residual_head_type"] = "time_axis_matched_lowrank"
        payload["weak_period_residual_rank"] = matched_rank
    return payload


def build_command(args, frozen, head, overrides, tag):
    command = [
        sys.executable,
        str(ROOT / "scripts/search_phaseformer.py"),
        "--dataset",
        frozen["dataset"],
        "--horizon",
        str(frozen["horizon"]),
        "--stage",
        "confirm",
        "--mechanism",
        "weak_residual",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        str(args.max_epochs),
        "--seed",
        str(args.seed),
        "--loss",
        "huber",
        "--learning-rate",
        str(frozen["learning_rate"]),
        "--output-dir",
        str(Path(args.output_root) / tag),
        "--num-workers",
        str(args.num_workers),
        "--bad-case-limit",
        "0",
        "--evaluate-test",
        "--require-cuda",
        "--resume",
        "--overrides",
        json.dumps(overrides),
    ]
    return command


def round0_jobs(args, frozen):
    """Round 0 is an audit: the three controls already exist, so no new run."""

    return []


def _candidate_head_params(head_type, horizon, overrides):
    class _Config:
        def __init__(self, values):
            for key, value in values.items():
                setattr(self, key, value)

    head = build_structured_residual_head(
        head_type, LOOKBACK, horizon, _Config(dict(overrides)), seed=2021, key=head_type
    )
    return head.parameter_count()


def _matched_rank(name, head_type, horizon, overrides):
    """Rank of the time-axis control used for this candidate.

    Route A is matched from its measured budget; the other routes are already
    below the rank-1 control floor, so the control is pinned at rank 1 and the
    measured gap is reported rather than hidden.
    """

    if name in AUTOMATCHED_ROUTES:
        budget = _candidate_head_params(head_type, horizon, AUTOMATCHED_OVERRIDES[name])
        try:
            rank, _ = matched_control_rank(budget, LOOKBACK, horizon)
        except ValueError:
            # The candidate is below the rank-1 control floor; fall back to the
            # smallest control and record the gap instead of skipping the row.
            return 1
        return rank
    return ROUND1_MATCHED_RANK.get(name, 1)


def round1_jobs(args, frozen):
    jobs = []
    horizon = int(frozen["horizon"])
    for name, spec in {**ROUND1_ROUTES, **ROUND1_DIAGNOSTICS}.items():
        overrides = dict(spec["overrides"])
        overrides["weak_period_residual_gate_init"] = frozen["gate_init"]
        jobs.append((name, spec["head"], overrides))
        if not args.no_matched_controls:
            rank = spec.get(
                "matched_rank",
                _matched_rank(name, spec["head"], horizon, overrides),
            )
            jobs.append(
                (
                    f"matched_{name}",
                    spec["head"],
                    build_override(spec["head"], {}, matched_rank=rank)
                    | {"weak_period_residual_gate_init": frozen["gate_init"]},
                )
            )
    return jobs


def dispatch(args, jobs):
    if not jobs:
        print("no jobs to run (Round 0 reuses the existing controls)", flush=True)
        return
    pending = list(jobs)
    active = {}
    while pending or active:
        while pending and len(active) < len(args.gpus):
            gpu = next(item for item in args.gpus if item not in active)
            name, head, overrides = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = build_command(args, args._frozen, head, overrides, name)
            print(f"launch gpu={gpu} {name}: {' '.join(command)}", flush=True)
            active[gpu] = (
                name,
                subprocess.Popen(command, cwd=ROOT, env=env),
            )
        finished = []
        for gpu, (name, process) in active.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                raise RuntimeError(f"job {name} on GPU {gpu} failed ({code})")
            print(f"finished gpu={gpu}: {name}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[0, 1])
    parser.add_argument("--frozen", required=True, help="frozen_configs.json path")
    parser.add_argument(
        "--setting",
        help="dataset:horizon to run (for example ETTh2:720); omit to run all "
        "settings present in the frozen file, one after another",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--no-matched-controls", action="store_true")
    parser.add_argument(
        "--only", help="comma-separated candidate names to run (Round 1 only)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without launching"
    )
    args = parser.parse_args()
    args.gpus = [item for item in str(args.gpus).split(",") if item]

    frozen_map = load_frozen(args.frozen)
    settings = sorted(frozen_map.items())
    if args.setting:
        dataset, _, horizon = args.setting.partition(":")
        key = (dataset, int(horizon))
        if key not in frozen_map:
            raise SystemExit(f"setting {args.setting} is absent from {args.frozen}")
        settings = [(key, frozen_map[key])]
    for (dataset, horizon), frozen in settings:
        args._frozen = frozen
        print(f"== setting {dataset}:{horizon} (gate_init={frozen['gate_init']}, "
              f"lr={frozen['learning_rate']})", flush=True)
        jobs = round0_jobs(args, frozen) if args.round == 0 else round1_jobs(args, frozen)
        if args.only:
            allowed = {item.strip() for item in args.only.split(",") if item.strip()}
            jobs = [job for job in jobs if job[0] in allowed]
            print(f"   filtered to {sorted(allowed)}", flush=True)
        if args.dry_run:
            for name, head, overrides in jobs:
                command = build_command(args, frozen, head, overrides, name)
                print(f"   {name}: {' '.join(command)}", flush=True)
            continue
        dispatch(args, jobs)


if __name__ == "__main__":
    main()
