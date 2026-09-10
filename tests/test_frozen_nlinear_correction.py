import torch

from src.models.frozen_nlinear_correction import PooledLowRankResidualHead


def test_pooled_low_rank_head_shape_and_endpoint_invariance():
    head = PooledLowRankResidualHead(
        12, 4, pool_factor=3, rank=2, smooth_ratio=0.0
    )
    x = torch.randn(2, 12, 3)
    output = head(x)
    shifted = head(x + 7.0)
    assert output.shape == (2, 4, 3)
    torch.testing.assert_close(output, shifted)


def test_pooled_low_rank_head_smoothing_is_finite():
    head = PooledLowRankResidualHead(
        12, 4, pool_factor=2, rank=2, smooth_ratio=0.5, smooth_window=5
    )
    output = head(torch.randn(2, 12, 1))
    assert torch.isfinite(output).all()


def test_pooled_low_rank_head_rejects_invalid_rank():
    try:
        PooledLowRankResidualHead(12, 4, pool_factor=3, rank=5)
    except ValueError as exc:
        assert "exceeds factorized map limit" in str(exc)
    else:
        raise AssertionError("expected rank validation failure")
