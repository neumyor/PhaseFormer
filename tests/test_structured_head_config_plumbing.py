"""The model must actually receive the structured-head configuration.

Round 1 was invalidated by exactly this failure: ``PhaseFormerPresetConfig``
never forwarded the ``residual_*`` override keys, so every structured candidate
was instantiated with the builder defaults (``residual_period_len=24``,
``residual_period_rank=4``, ``residual_basis_count=4``, ``recent_taps=7``,
``num_components=1``, ``level_mode="dense"``).  Routes then collapsed onto each
other -- the r8 diagnostic and the r4 route-A run built identical heads -- while
each run still recorded its own requested overrides in ``config.json``, so the
result tables looked like distinct candidates.

These tests instantiate the real model through the real preset path, so a missing
forwarding line fails loudly instead of silently training the wrong head.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.PhaseFormer import PhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    build_hyperparams,
    make_exp_args,
    PhaseFormerPresetConfig,
)


def _head(head_type, **overrides):
    hyper = build_hyperparams("ETTh2", 96, "original")
    hyper.update(overrides)
    hyper.update(
        {
            "weak_period_residual_head_type": head_type,
            "use_weak_period_residual": True,
            "weak_period_residual_gate_init": 0.5,
            "period_len": 24,
            "train_epochs": 30,
            "loss_func": "huber",
            "learning_rate": 0.001,
        }
    )
    args = make_exp_args("ETTh2", 720, 96, hyper)
    model = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hyper))
    return model.weak_period_residual


@pytest.mark.parametrize("rank", [1, 4, 8])
def test_period_lowrank_rank_reaches_the_head(rank):
    head = _head(
        "structured_period_lowrank",
        residual_period_len=24,
        residual_period_rank=rank,
    )
    assert head.rank == rank
    assert head.segment_encoder.weight.shape == (rank, 30)


@pytest.mark.parametrize("period", [24, 96])
def test_residual_period_len_reaches_the_head(period):
    head = _head(
        "structured_period_lowrank",
        residual_period_len=period,
        residual_period_rank=4,
    )
    assert head.period == period
    assert head.num_taps == -(-720 // period)


@pytest.mark.parametrize("basis", [2, 8])
def test_segment_basis_count_reaches_the_head(basis):
    head = _head(
        "structured_segment_basis",
        residual_period_len=24,
        residual_basis_count=basis,
        residual_basis_lambda_orth=0.01,
    )
    assert head.num_basis == basis
    assert head.lambda_orth == pytest.approx(0.01)


@pytest.mark.parametrize("mode,shape_rank", [("dense", 4), ("lowrank", None)])
def test_level_shape_configuration_reaches_the_head(mode, shape_rank):
    head = _head(
        "structured_level_shape",
        residual_period_len=24,
        residual_level_mode=mode,
        residual_level_rank=1,
        residual_shape_rank=shape_rank,
    )
    assert head.level_mode == mode
    assert head.shape_rank == shape_rank


@pytest.mark.parametrize("taps", [3, 15])
def test_recent_taps_reach_the_head(taps):
    head = _head(
        "structured_recent_period",
        residual_period_len=24,
        residual_recent_taps=taps,
        residual_recent_weighting="hard",
    )
    assert head.recent_taps == taps


@pytest.mark.parametrize("components", [1, 2])
def test_separable_component_count_reaches_the_head(components):
    head = _head(
        "structured_separable",
        residual_period_len=24,
        residual_separable_components=components,
    )
    assert head.num_components == components


@pytest.mark.parametrize("alignment", ["aligned", "shifted", "random"])
def test_segmentation_alignment_reaches_the_head(alignment):
    head = _head(
        "structured_period_lowrank",
        residual_period_len=24,
        residual_period_rank=4,
        residual_segment_alignment=alignment,
    )
    if alignment == "aligned":
        assert head.layout.offset == 0
    elif alignment == "shifted":
        assert head.layout.offset == 12
    else:
        assert 0 <= head.layout.offset < 24


def test_matched_control_rank_reaches_the_head():
    head = _head(
        "time_axis_matched_lowrank",
        weak_period_residual_rank=8,
    )
    assert head.rank == 8
    assert head.encoder.weight.shape == (8, 720)


def test_phase_path_period_len_is_never_changed_by_the_residual_head():
    """Plan section 10.1: the residual period is head-local."""

    head = _head(
        "structured_period_lowrank",
        residual_period_len=96,
        residual_period_rank=4,
    )
    assert head.period == 96
