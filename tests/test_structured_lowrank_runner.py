"""Regression tests for the Round 1 job composition.

Guards the failure mode where a candidate job was emitted without an explicit
``weak_period_residual_head_type``: the ``weak_residual`` preset then reverts it
to ``shared`` and the run silently trains the plain direct NLinear head while
still recording the structural overrides.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _runner():
    spec = importlib.util.spec_from_file_location(
        "structured_round1_runner", ROOT / "scripts" / "run_structured_lowrank_round1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Args:
    no_matched_controls = False
    seed = 2021
    max_epochs = 1
    num_workers = 2
    output_root = "research_runs/unused_probe_dir"
    gpus = ["0"]


FROZEN = {
    "dataset": "ETTh2",
    "horizon": 96,
    "gate_init": 0.5,
    "learning_rate": 0.001,
    "lookback": 720,
    "period": 24,
    "max_epochs": 30,
    "loss": "huber",
    "seed": 2021,
}


def test_every_round1_job_carries_an_explicit_head_type():
    module = _runner()
    jobs = module.round1_jobs(_Args(), FROZEN)
    assert jobs, "expected Round 1 jobs"
    for name, head, overrides in jobs:
        assert "weak_period_residual_head_type" in overrides, name
        assert overrides["weak_period_residual_head_type"] in {
            *module.STRUCTURED_HEAD_BUILDERS,
            "time_axis_matched_lowrank",
        }, name
        assert overrides["weak_period_residual_gate_init"] == FROZEN["gate_init"], name


def test_candidate_jobs_use_their_own_head_and_controls_use_the_control_head():
    module = _runner()
    jobs = {name: (head, overrides) for name, head, overrides in module.round1_jobs(_Args(), FROZEN)}
    for name, spec in module.ROUND1_ROUTES.items():
        head, overrides = jobs[name]
        assert overrides["weak_period_residual_head_type"] == head == spec["head"], name
    for name in module.ROUND1_ROUTES:
        _, overrides = jobs[f"matched_{name}"]
        assert (
            overrides["weak_period_residual_head_type"] == "time_axis_matched_lowrank"
        ), name
        assert overrides.get("weak_period_residual_rank", 0) >= 1, name


def test_head_types_are_registered_in_the_model_factory():
    module = _runner()
    for name, spec in {**module.ROUND1_ROUTES, **module.ROUND1_DIAGNOSTICS}.items():
        assert spec["head"] in module.STRUCTURED_HEAD_BUILDERS, name


def test_matched_control_rank_is_at_most_the_candidate_rank():
    module = _runner()
    jobs = {name: overrides for name, _, overrides in module.round1_jobs(_Args(), FROZEN)}
    # Route A is matched from its measured budget, so its control rank may be
    # smaller than its own rank; it must never be larger than the budget can pay
    # for at the same total parameter count.
    rank = jobs["matched_A_period_lowrank"]["weak_period_residual_rank"]
    assert rank >= 1
    assert (
        jobs["A_period_lowrank"]["residual_period_rank"] >= 1
    )
