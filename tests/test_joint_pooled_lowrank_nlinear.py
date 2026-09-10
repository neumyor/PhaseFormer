import torch

from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


def test_pooled_lowrank_head_warm_starts_at_last_value():
    head = PooledLowRankWeakPeriodResidualHead(
        12, 4, pool_factor=3, rank=2, smooth_ratio=0.5, smooth_window=5
    )
    x = torch.randn(2, 12, 3)
    expected = x[:, -1:, :].expand(-1, 4, -1)
    torch.testing.assert_close(head(x), expected)


def test_pooled_lowrank_head_validates_rank():
    try:
        PooledLowRankWeakPeriodResidualHead(12, 4, pool_factor=3, rank=5)
    except ValueError as exc:
        assert "factorized-map limit" in str(exc)
    else:
        raise AssertionError("expected invalid rank to fail")


def test_phaseformer_jointly_constructs_pooled_lowrank_branch():
    hyperparams = build_hyperparams("ETTm1", 96, "weak_residual")
    hyperparams.update(
        weak_period_residual_head_type="pooled_lowrank",
        weak_period_residual_pool_factor=4,
        weak_period_residual_rank=16,
        weak_period_residual_smooth_ratio=0.25,
        weak_period_residual_smooth_window=24,
    )
    args = make_exp_args("ETTm1", 720, 96, hyperparams, batch_size=2)
    model = PhaseFormer(PhaseFormerPresetConfig(args, 720, 96, hyperparams))
    assert isinstance(
        model.weak_period_residual, PooledLowRankWeakPeriodResidualHead
    )
    output, _, _ = model(torch.randn(2, 720, model.enc_in))
    assert output.shape == (2, 96, model.enc_in)
