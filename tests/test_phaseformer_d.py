"""PhaseFormer-D: shapes, the anchor at zero output, the roles of the two branches, per-branch compression."""
import pytest
import torch

from scripts.biaxial.model import DCfg, PhaseFormerD, count_parameters


def make(period=24, horizon=96, **over):
    return PhaseFormerD(DCfg(channels=3, pred_len=horizon, period=period, **over)).eval()


@pytest.mark.parametrize('period,horizon', [(24, 96), (48, 336), (24, 100), (72, 720)])
@pytest.mark.parametrize('fusion', ['sum', 'gate'])
def test_output_shape(period, horizon, fusion):
    out = make(period, horizon, fusion=fusion, p_t=period // 4)(torch.randn(2, 720, 3))
    assert out.shape == (2, horizon, 3) and torch.isfinite(out).all()


def zero_heads(m):
    for net in (m.cycle, m.phase):
        torch.nn.init.zeros_(net.predictor.weight)
        torch.nn.init.zeros_(net.predictor.bias)


def test_zero_heads_reproduce_the_last_cycle():
    torch.manual_seed(0)
    m = make(24, 96)
    zero_heads(m)
    x = torch.randn(2, 720, 3)
    assert torch.allclose(m(x), x[:, -24:].repeat(1, 4, 1), atol=1e-5)


def test_phase_branch_cannot_change_cycle_level():
    torch.manual_seed(0)
    m = make(24, 96, fusion='sum')
    zero_heads(m)
    torch.nn.init.normal_(m.phase.predictor.weight, std=0.5)
    x = torch.randn(2, 720, 3)
    out, last = m(x), x[:, -24:].repeat(1, 4, 1)
    diff = (out - last).view(2, 4, 24, 3)
    assert diff.abs().max() > 1e-3                         # it does change the shape
    assert diff.mean(2).abs().max() < 1e-5                 # but not the level of any forecast cycle


def test_branches_receive_their_own_normalization():
    seen = {}
    m = make(24, 96)
    m.cycle.register_forward_pre_hook(lambda mod, a: seen.__setitem__('t', a[0]))
    m.phase.register_forward_pre_hook(lambda mod, a: seen.__setitem__('s', a[0]))
    m(torch.randn(2, 720, 3) * 5 + 3)
    assert seen['t'].mean(1).abs().max() < 1e-5            # cycle branch: seasonal profile removed
    assert (seen['t'].square().mean((1, 2)) - 1).abs().max() < 0.02       # one unit scale per window (up to the floor)
    assert seen['s'].mean(2).abs().max() < 1e-5            # phase branch: every cycle has zero mean over phases
    assert (seen['s'].var(2, unbiased=False) - 1).abs().max() < 0.05


def test_compression_is_per_branch():
    m = make(24, 96, k_t=8, p_t=6, k_s=2, p_s=0)
    seen = {}
    m.cycle.inp.register_forward_hook(lambda mod, a, out: seen.__setitem__('t', out.shape))
    m.phase.inp.register_forward_hook(lambda mod, a, out: seen.__setitem__('s', out.shape))
    m(torch.randn(1, 720, 1))
    assert seen['t'][1:3] == (8, 6) and seen['s'][1:3] == (2, 24)


def test_gate_starts_as_sum():
    torch.manual_seed(0)
    a, b = make(fusion='gate'), make(fusion='sum')
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith('gate')})
    x = torch.randn(2, 720, 3)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_prefix_property_for_non_multiple_horizons():
    torch.manual_seed(0)
    short, long = make(24, 100), make(24, 120)
    long.load_state_dict(short.state_dict())
    x = torch.randn(2, 720, 3)
    assert torch.allclose(short(x), long(x)[:, :100], atol=1e-5)
