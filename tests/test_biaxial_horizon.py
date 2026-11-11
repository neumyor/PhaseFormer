"""PhaseFormer-I accepts horizons that are not whole cycles; whole-cycle horizons behave exactly as before."""
import pytest
import torch

from scripts.biaxial.model import Cfg, PhaseFormerPlus

FINAL = dict(arch='interleaved', layers=1, d=16, k=8, p=0, q=1, wp='identity', decoder='direct', anchor='last_cycle')


def make(period, horizon, seq_len, **over):
    return PhaseFormerPlus(Cfg(seq_len=seq_len, channels=3, pred_len=horizon, period=period, **{**FINAL, **over}))


def test_output_length_for_non_multiple_horizons():
    for period, horizon, seq_len in [(96, 336, 672), (96, 720, 672), (144, 96, 720), (168, 192, 672), (24, 96, 720)]:
        x = torch.randn(2, seq_len, 3)
        assert make(period, horizon, seq_len)(x).shape == (2, horizon, 3), (period, horizon)


def test_prefix_equals_longer_whole_cycle_forecast():
    torch.manual_seed(0)
    x = torch.randn(2, 672, 3)
    short = make(96, 336, 672).eval()
    full = make(96, 384, 672).eval()
    full.load_state_dict(short.state_dict())
    assert torch.allclose(short(x), full(x)[:, :336], atol=0, rtol=0)


def test_all_anchors_and_trend_decoder_run():
    x = torch.randn(2, 672, 3)
    for anchor in ('last_cycle', 'last_cycle_mean', 'none', 'x_last'):
        assert make(96, 336, 672, anchor=anchor)(x).shape == (2, 336, 3)
    assert make(96, 336, 672, decoder='d0')(x).shape == (2, 336, 3)


def test_window_must_still_be_whole_cycles():
    with pytest.raises(ValueError):
        make(96, 336, 720)
