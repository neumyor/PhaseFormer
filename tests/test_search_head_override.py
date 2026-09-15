"""Regression test: an explicit residual head type must survive the preset.

``build_hyperparams(..., "weak_residual")`` hard-sets
``weak_period_residual_head_type`` to ``shared``.  ``build_spec`` applies CLI
overrides before that point, so without a re-application every run requested with
another registered head type silently trains the plain direct NLinear head.  This
test pins the behaviour for both the override and the untouched default.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _load_search_module():
    spec = importlib.util.spec_from_file_location(
        "search_phaseformer_head_override", ROOT / "scripts" / "search_phaseformer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Args:
    mechanism = "weak_residual"
    dataset = "ETTh2"
    horizon = 96
    lookback = 720
    percent = 100
    period = 24
    seed = 2021
    max_epochs = 30
    loss = "huber"
    learning_rate = 0.001
    lr_multiplier = 1.0
    capacity = "base"
    batch_size = None
    evaluate_test = True
    input_hypothesis = "none"
    input_variant = "full"
    intervention_seed = 9102
    input_d1_period = ""
    input_d1_sigma = ""
    input_d2_recent_length = ""
    input_d3_component = ""
    max_eval_samples = 0
    require_cuda = True
    init_checkpoint = ""
    stage = "confirm"
    cycle_period = None
    overrides = "{}"


def _spec_with(overrides):
    module = _load_search_module()
    args = type("Args", (_Args,), {"overrides": json.dumps(overrides)})()
    return module.build_spec(args)["hyperparams"]


def test_head_type_override_survives_the_weak_residual_preset():
    hyper = _spec_with(
        {
            "weak_period_residual_head_type": "structured_period_lowrank",
            "residual_period_len": 24,
            "residual_period_rank": 4,
        }
    )
    assert hyper["weak_period_residual_head_type"] == "structured_period_lowrank"
    assert hyper["residual_period_rank"] == 4
    assert hyper["residual_period_len"] == 24


def test_head_type_default_is_untouched_without_the_override():
    hyper = _spec_with({"residual_period_len": 24})
    assert hyper["weak_period_residual_head_type"] == "shared"


def test_matched_control_head_type_is_selectable():
    hyper = _spec_with(
        {
            "weak_period_residual_head_type": "time_axis_matched_lowrank",
            "weak_period_residual_rank": 2,
        }
    )
    assert hyper["weak_period_residual_head_type"] == "time_axis_matched_lowrank"
    assert hyper["weak_period_residual_rank"] == 2
