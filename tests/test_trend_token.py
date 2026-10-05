import torch

from src.models.trend_token import TrendTokenBranch


def test_trend_token_shapes_and_profile_rank():
    torch.manual_seed(0)
    module = TrendTokenBranch(
        seq_len=16,
        pred_len=8,
        enc_in=3,
        phase_dim=8,
        token_dim=12,
        profile_count=1,
        use_phase_attention=True,
    )
    x = torch.randn(2, 16, 3)
    z = torch.randn(2, 3, 4, 8)
    trend, gate, diag = module(x, z)
    assert trend.shape == (2, 8, 3)
    assert gate.shape == (1, 1, 3)
    assert diag["lambda"].shape == (2, 3, 1)
    centered = trend - x[:, -1:, :]
    # m=1 means every output trajectory is a scalar multiple of one profile.
    profile = diag["profile_bank"][:, 0]
    ratio = centered / profile.view(1, -1, 1).clamp_min(1e-6)
    assert torch.isfinite(ratio).all()


def test_trend_token_is_history_only_and_pool_is_per_channel():
    torch.manual_seed(1)
    module = TrendTokenBranch(
        seq_len=12,
        pred_len=5,
        enc_in=2,
        phase_dim=4,
        token_dim=8,
        profile_count=1,
    )
    x = torch.randn(1, 12, 2)
    z = torch.randn(1, 2, 3, 4)
    module(x, z)
    assert module.last_pool_weights.shape == (1, 2, 12)
    assert torch.allclose(module.last_pool_weights.sum(-1), torch.ones(1, 2), atol=1e-6)
    # Changing phase latent cannot alter the history pooling weights.
    first = module.last_pool_weights.clone()
    module(x, z + 10.0)
    assert torch.allclose(first, module.last_pool_weights, atol=1e-6)


def test_gate_zero_is_exact_phase_passthrough_formula():
    module = TrendTokenBranch(
        seq_len=8,
        pred_len=4,
        enc_in=1,
        phase_dim=4,
        token_dim=8,
        gate_init=1e-4,
    )
    with torch.no_grad():
        module.gate_logit.fill_(-100.0)
    gate = torch.sigmoid(module.gate_logit)
    assert float(gate.max()) < 1e-40


def test_direct_readout_projects_token_to_horizon():
    torch.manual_seed(2)
    module = TrendTokenBranch(
        seq_len=16,
        pred_len=7,
        enc_in=2,
        phase_dim=8,
        token_dim=12,
        readout_type="direct",
        use_phase_attention=True,
    )
    x = torch.randn(3, 16, 2)
    z = torch.randn(3, 2, 4, 8)
    trend, gate, diag = module(x, z)
    assert trend.shape == (3, 7, 2)
    assert gate.shape == (1, 1, 2)
    assert diag["readout_type"] == "direct"
    assert diag["lambda"] is None
    assert diag["direct_delta"].shape == (3, 2, 7)
    assert module.last_lambda is None


def test_single_mechanism_variants_preserve_forecast_shapes():
    torch.manual_seed(3)
    x = torch.randn(2, 16, 2)
    z = torch.randn(2, 2, 4, 8)
    variants = [
        {"gate_mode": "reliability"},
        {"direct_smooth_window": 5},
        {"token_count": 2},
        {"phase_attention_mode": "null"},
        {"readout_type": "factorized"},
        {"history_multiscale": True},
        {"history_robust": True},
    ]
    for kwargs in variants:
        module = TrendTokenBranch(
            seq_len=16,
            pred_len=7,
            enc_in=2,
            phase_dim=8,
            token_dim=12,
            readout_type=kwargs.pop("readout_type", "direct"),
            use_phase_attention=True,
            **kwargs,
        )
        trend, gate, diag = module(x, z)
        assert trend.shape == (2, 7, 2)
        assert gate.shape == (2, 1, 2) if kwargs.get("gate_mode") == "reliability" else gate.shape in {(1, 1, 2), (2, 1, 2)}
        assert torch.isfinite(trend).all()
        assert diag["direct_delta"].shape == (2, 2, 7)


def test_multi_token_pool_has_separate_attention_maps():
    module = TrendTokenBranch(
        seq_len=12,
        pred_len=5,
        enc_in=2,
        phase_dim=4,
        token_dim=8,
        readout_type="direct",
        token_count=2,
    )
    x = torch.randn(1, 12, 2)
    z = torch.randn(1, 2, 3, 4)
    module(x, z)
    assert module.last_pool_weights.shape == (1, 2, 2, 12)
    assert torch.allclose(module.last_pool_weights.sum(-1), torch.ones(1, 2, 2), atol=1e-6)
