"""Pre-LN stream, linear+MLP embedding and Fourier phase identity: shapes, defaults unchanged, behaviour as designed."""
import pytest
import torch

from scripts.biaxial.model import Cfg, PhaseFormerPlus, count_parameters

FINAL = dict(arch='interleaved', layers=1, d=16, k=8, p=0, q=1, wp='identity', decoder='direct', anchor='last_cycle')
VARIANTS = {'preln': dict(norm='pre', embed='lin_mlp'), 'fourier': dict(phase_id='fourier'),
            'both': dict(norm='pre', embed='lin_mlp', phase_id='fourier')}


def make(period=24, horizon=96, seq_len=720, **over):
    return PhaseFormerPlus(Cfg(seq_len=seq_len, channels=3, pred_len=horizon, period=period, **{**FINAL, **over}))


@pytest.mark.parametrize('name', VARIANTS)
@pytest.mark.parametrize('period,horizon', [(24, 96), (48, 336), (24, 100)])
def test_variants_run(name, period, horizon):
    out = make(period, horizon, **VARIANTS[name]).eval()(torch.randn(2, 720, 3))
    assert out.shape == (2, horizon, 3) and torch.isfinite(out).all()


def test_default_has_no_new_parameters():
    base = make()
    assert not any(n.startswith(('phase_proj', 'slot', 'final_norm', 'lin')) for n, _ in base.named_parameters())
    assert count_parameters(base) == count_parameters(make(norm='post', embed='mlp_ln', phase_id='none'))


def test_fourier_parameters_do_not_grow_with_period():
    extra = lambda p: count_parameters(make(p, phase_id='fourier')) - count_parameters(make(p))
    # harmonics are capped at p // 2: same count once p // 2 >= 8, only the readout grows with p via nothing else
    assert extra(24) == extra(48) == extra(120)


def test_prenorm_residual_stream_is_linear_in_amplitude():
    """With the linear embedding and a pre-LN stream the cell value reaches the readout un-normalized:
    a cell value 100x larger gives a 100x larger embedding (the post-LN embedding saturates in direction)."""
    torch.manual_seed(0)
    m = make(norm='pre', embed='lin_mlp').eval().phase.inp
    z = torch.tensor([[[[0.5]]], [[[50.0]]]])
    lin = m.lin(z) - m.lin(torch.zeros_like(z))
    assert torch.allclose(lin[1], lin[0] * 100, rtol=1e-4)
    post = make().eval().phase.inp.embed
    a, b = post(torch.tensor([[1.0]])), post(torch.tensor([[1000.0]]))
    assert torch.nn.functional.cosine_similarity(a, b).item() < 0.999 and b.norm() < 8   # bounded norm: amplitude carried only by direction


def test_phase_identity_makes_phases_distinguishable():
    m = make(phase_id='fourier').eval().phase.inp
    X = torch.zeros(1, 30, 24)
    h = m(X)                                              # constant input: cells differ only through their identity
    assert (h[0, 0, 0] - h[0, 0, 12]).abs().max() > 1e-3
    assert not torch.allclose(h[0, 0, 5], h[0, 1, 5])
