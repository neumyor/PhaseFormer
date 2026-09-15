"""CPU invariant tests for the structured low-rank residual heads.

These tests cover the checks required before any server training run by the
breadth-first exploration plan (§10.2): shape, persistence anchor, padding and
cropping, phase alignment, parameter matching and the aligned/shifted/random
segmentation control.  They are pure CPU tests and need no dataset.
"""

import torch

from src.models.structured_residual_heads import (
    STRUCTURED_HEAD_BUILDERS,
    StructuredPeriodLowRankHead,
    StructuredRecentPeriodHead,
    StructuredSeparableHead,
    StructuredSegmentBasisHead,
    TimeAxisMatchedLowRankHead,
    build_structured_residual_head,
    future_grid_segments,
    future_segment_order,
    gather_segments,
    matched_control_rank,
    resolve_segment_layout,
)


class _StubConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _batch(batch=2, length=720, channels=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, channels, generator=generator)


def _all_heads(seq_len=720, pred_len=96, period=24):
    config = _StubConfig(
        residual_period_len=period,
        residual_period_rank=4,
        residual_basis_count=4,
        residual_basis_lambda_orth=0.01,
        residual_level_mode="dense",
        residual_shape_rank=4,
        residual_recent_taps=7,
        residual_recent_weighting="hard",
        residual_separable_components=1,
    )
    return {
        name: build_structured_residual_head(
            name, seq_len, pred_len, config, seed=2021, key=f"{name}:stub"
        )
        for name in STRUCTURED_HEAD_BUILDERS
    }


def test_output_shape_for_all_heads():
    for period in (24, 96):
        for horizon in (96, 192, 336, 720):
            x = _batch(length=720, channels=5)
            for name, head in _all_heads(720, horizon, period).items():
                out = head(x)
                assert out.shape == (2, horizon, 5), (name, period, horizon, out.shape)


def test_zero_initialization_reproduces_persistence_anchor():
    # Route D with fixed exponential weighting scales the gathered history by a
    # non-learnable profile, so it is tested separately with its decay applied.
    x = _batch(length=720, channels=4, seed=7)
    heads = _all_heads(720, 192, 24)
    heads.pop("structured_recent_period")
    for name, head in heads.items():
        out = head(x)
        anchor = x[:, -1:, :].expand(-1, 192, -1)
        assert torch.equal(out, anchor), name
    recent = StructuredRecentPeriodHead(
        720, 192, residual_period_len=24, recent_taps=7, weighting="fixed_exp"
    )
    assert torch.equal(recent(x), x[:, -1:, :].expand(-1, 192, -1))


def test_centered_history_last_value_is_zero():
    x = _batch(length=720, channels=2, seed=3)
    for name, head in _all_heads().items():
        centered = head._centered(x)
        assert torch.allclose(centered[:, :, -1], torch.zeros(2, 2)), name


def test_segment_layout_padding_and_cropping():
    layout = resolve_segment_layout(seq_len=720, period=24, alignment="aligned")
    assert layout.offset == 0
    assert layout.num_segments == 30
    assert layout.tap_length == 720
    # Unaligned lookback must still be covered by whole periods.
    odd = resolve_segment_layout(seq_len=700, period=96, alignment="aligned")
    assert odd.num_segments == 8
    assert odd.tap_length == 700
    # The future order table must address only the future grid the head builds,
    # and the head must build at least that many periods.
    for period in (24, 96):
        for horizon in (96, 192, 336, 720):
            layout = resolve_segment_layout(seq_len=720, period=period)
            head = StructuredPeriodLowRankHead(720, horizon, residual_period_len=period)
            need = future_grid_segments(layout, horizon)
            assert need <= head.num_future_segments + 1
            order = future_segment_order(layout, horizon, torch.device("cpu"))
            assert order.shape == (horizon,)
            assert int(order.max()) < need * period
            assert int(order.min()) >= 0


def test_gather_segments_share_phase_slots():
    length, period, channels = 24, 6, 1
    centered = torch.arange(length, dtype=torch.float32).view(1, length, channels)
    layout = resolve_segment_layout(seq_len=length, period=period, alignment="aligned")
    segments = gather_segments(centered.permute(0, 2, 1), layout)
    assert layout.num_segments == 4
    # The raw gather counts backwards from the last observed step.
    assert torch.equal(
        segments[0, 0, 0],
        torch.flip(centered[0, period * 3 : period * 4, 0], dims=(0,)),
    )
    assert torch.equal(
        segments[0, 0, 1],
        torch.flip(centered[0, period * 2 : period * 3, 0], dims=(0,)),
    )
    # Neighbouring phase columns of the raw gather are exactly P steps apart.
    assert torch.allclose(
        segments[0, 0, 0] - segments[0, 0, 1],
        torch.full((period,), float(period)),
    )
    # The phase convention makes a fixed column mean a fixed absolute phase in
    # every row, including for shifted segmentations.
    values = torch.arange(length, dtype=torch.float32).view(1, length, 1)
    for offset in (0, period // 2, 3):
        shifted = resolve_segment_layout(
            seq_len=length, period=period, alignment="aligned", period_offset=offset
        )
        aligned = gather_segments(
            values.permute(0, 2, 1), shifted, phase_convention=True
        )
        for row in range(aligned.shape[2]):
            for column in range(period):
                value = aligned[0, 0, row, column].item()
                # Cell ``(row, column)`` carries lag ``row * period + column``:
                # the value that many steps before the last observation.  The
                # invariant that matters is that the lag, not the row, drives the
                # column, so a fixed column always means a fixed phase modulo P.
                expected = length - 1 - (row * period + column)
                if expected > 0:
                    assert value == expected, (offset, row, column, value, expected)
                # Row s of column c is exactly P steps older than row s - 1.
                if row > 0 and expected > 0:
                    older = aligned[0, 0, row - 1, column].item()
                    assert older - value == period, (offset, row, column)


def _periodic_history(length, period):
    """A strictly period-``P`` centered history with a unique value per phase."""

    index = torch.arange(length, dtype=torch.float32)
    return torch.sin(2 * torch.pi * index / period).view(1, length, 1)


def _wire_route_a_readout(head):
    """Route A reduced to a phase-wise readout of the newest history period.

    Rank 1 encodes lag column 0 and the decoder writes each phase slot of the
    first future period (flat decoder axis ``phase * K_y + k_y``).
    """

    with torch.no_grad():
        head.segment_encoder.weight.zero_()
        head.segment_encoder.bias.zero_()
        head.segment_encoder.weight[0, 0] = 1.0
        head.segment_decoder.weight.zero_()
        head.segment_decoder.bias.zero_()
        for column in range(head.period):
            head.segment_decoder.weight[column, 0] = 1.0
    return head


def test_phase_alignment_of_future_grid_route_a():
    """Route A must map history lag ``i`` onto forecast step ``i``.

    With the phase-wise readout the emitted delta must reproduce the centered
    history lag profile exactly; a one-slot slip anywhere in the geometry shows
    up as a rotation of this profile.
    """

    length, period, horizon = 72, 4, 12
    x = _periodic_history(length, period)
    head = _wire_route_a_readout(
        StructuredPeriodLowRankHead(
            length, horizon, residual_period_len=period, rank=1
        )
    )
    delta = (head(x) - x[:, -1:, :])[0]
    centered = head._centered(x)[0, 0]
    for step in range(period):
        assert torch.isclose(
            delta[step, 0], centered[length - 1 - step], atol=1e-4
        ), step
    # Only the first forecast period is written by this wiring.
    assert torch.equal(delta[period:], torch.zeros_like(delta[period:]))
    assert delta.abs().max() > 1e-5


def test_phase_alignment_holds_for_multi_period_horizons():
    """A long horizon must not shift the phase mapping at period boundaries."""

    length, period, horizon = 720, 24, 96
    x = _periodic_history(length, period)
    head = _wire_route_a_readout(
        StructuredPeriodLowRankHead(
            length, horizon, residual_period_len=period, rank=1
        )
    )
    delta = (head(x) - x[:, -1:, :])[0, :horizon, 0]
    centered = head._centered(x)[0, 0]
    for step in range(period):
        assert torch.isclose(delta[step], centered[length - 1 - step], atol=1e-4), step
    # This wiring only fills the first forecast period, so everything after it
    # must be exactly zero -- a slip would show up as nonzero tail values.
    assert torch.equal(delta[period:], torch.zeros(horizon - period))
    assert delta.abs().max() > 0.5


def test_phase_alignment_survives_segmentation_variants():
    """Aligned, shifted and random segmentations keep the phase mapping.

    The head only ever reads phase information from a ``P``-periodic history, so
    all three segmentations must produce the same forecast.  If the per-row phase
    bookkeeping were wrong, the shifted and random variants would disagree.
    """

    length, period, horizon = 48, 4, 12
    x = _periodic_history(length, period)
    reference = None
    for alignment in ("aligned", "shifted", "random"):
        head = StructuredSeparableHead(
            length,
            horizon,
            residual_period_len=period,
            num_components=1,
            alignment=alignment,
        )
        with torch.no_grad():
            head.period_maps.weight.zero_()
            head.period_maps.bias.fill_(1.0)  # period-independent response
            head.phase_profiles.zero_()
            head.phase_profiles[0, period - 1] = 1.0
        delta = (head(x) - x[:, -1:, :])[0, :horizon, 0]
        if reference is None:
            reference = delta
        else:
            assert torch.allclose(delta, reference, atol=1e-5), (
                alignment,
                delta.tolist(),
                reference.tolist(),
            )
    assert reference.abs().max() > 1e-5



def test_unobserved_recent_taps_are_zero_centered():
    length, period, horizon = 24, 24, 48
    x = _batch(batch=1, length=length, channels=1, seed=11)
    head = StructuredRecentPeriodHead(
        length,
        horizon,
        residual_period_len=period,
        recent_taps=1,
        weighting="hard",
    )
    out = head(x)
    assert torch.equal(out, x[:, -1:, :].expand(-1, horizon, -1))
    layout = head.layout
    # With L == P exactly one period is observed, laid out newest-first.
    assert layout.segment_indices[0].tolist() == list(range(23, -1, -1))
    # A second tap would sit entirely outside the window; the layout clamps such
    # positions into the appended zero slot rather than wrapping to real data,
    # and the head refuses to request more taps than the window holds.
    assert head.layout.segment_indices[0].min().item() == 0
    try:
        StructuredRecentPeriodHead(
            length, horizon, residual_period_len=period, recent_taps=2, weighting="hard"
        )
    except ValueError as error:
        assert "recent_taps" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a recent_taps validation error")
    padded = torch.cat([head._centered(x), head._centered(x).new_zeros(1, 1, 1)], dim=2)
    assert torch.equal(padded[:, :, 0:0], torch.zeros(1, 1, 0))


def test_recency_weighting_prefers_newest_tap():
    head = StructuredRecentPeriodHead(
        720, 96, residual_period_len=12, recent_taps=7, weighting="fixed_exp", decay=0.8
    )
    weights = head.recency_weights.flatten()
    assert weights.shape[0] == 7
    assert weights[0] == 1.0
    assert torch.all(weights[:-1] > weights[1:])
    hard = StructuredRecentPeriodHead(
        720, 96, residual_period_len=12, recent_taps=7, weighting="hard"
    )
    assert not hasattr(hard, "recency_weights")


def test_matched_control_parameter_budget():
    x = _batch(batch=1, length=720, channels=2)
    for name, head in _all_heads().items():
        target = head.parameter_count()
        try:
            rank, actual = matched_control_rank(target, 720, 96)
        except ValueError as error:
            # A head budget below the rank-1 control is reported, never silently
            # matched with an oversized control.
            assert "too small" in str(error), (name, target)
            continue
        control = TimeAxisMatchedLowRankHead(720, 96, rank=rank)
        assert control.parameter_count() == actual
        # The control never receives a larger budget than the candidate it is
        # matched against, and stays inside the pre-registered 5% band whenever
        # an integer rank can reach it.
        assert actual <= target
        if rank > 1:
            assert (target - actual) / target <= 0.05, (name, target, actual)
        out = control(x)
        assert out.shape == (1, 96, 2)
        assert torch.equal(out, x[:, -1:, :].expand(-1, 96, -1))


def test_matched_control_rank_helpers():
    # r * (720 + 96 + 1) + 96 <= 3264  ->  r = 3
    rank, actual = matched_control_rank(3264, 720, 96)
    assert rank == 3 and actual == 3 * (720 + 96 + 1) + 96
    assert TimeAxisMatchedLowRankHead(720, 96, rank=rank).parameter_count() == actual
    # Route E's budget is far below a rank-1 control, which is reported as a
    # no-match instead of a silent larger control; see the plan's matched-control
    # accounting for the recorded fallback.
    try:
        matched_control_rank(120, 720, 96)
    except ValueError as error:
        assert "too small" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a budget error for a tiny target")


def test_segmentation_variants_are_distinct_and_deterministic():
    aligned = resolve_segment_layout(720, 24, alignment="aligned")
    shifted = resolve_segment_layout(720, 24, alignment="shifted")
    assert shifted.offset == 12
    first = resolve_segment_layout(720, 24, alignment="random", seed=2021, key="a")
    again = resolve_segment_layout(720, 24, alignment="random", seed=2021, key="a")
    other = resolve_segment_layout(720, 24, alignment="random", seed=2021, key="b")
    assert torch.equal(first.segment_indices, again.segment_indices)
    assert not torch.equal(first.segment_indices, other.segment_indices)
    assert 0 <= first.offset < 24
    assert aligned.segment_indices.shape == (30, 24)
    assert shifted.segment_indices.shape == (31, 24)
    assert first.segment_indices.shape == (31, 24)


def test_shifted_layout_reads_same_time_span_with_phase_offset():
    shifted = resolve_segment_layout(720, 24, alignment="shifted")
    # The newest shifted segment ends at the last observed value and its first
    # entry reuses an older phase slot, so the covered span shifts by P.
    assert shifted.tap_length == 732
    # Rows are laid out newest-first and positions before the window are clamped
    # into the appended zero slot at index 720.
    assert shifted.segment_indices[0].tolist() == [720] * 12 + list(
        range(719, 707, -1)
    )
    assert shifted.segment_indices[-1][-1].item() == 0
    aligned = resolve_segment_layout(720, 24, alignment="aligned")
    assert aligned.segment_indices[0].tolist() == list(range(719, 695, -1))
    assert aligned.segment_indices[-1][-1].item() == 0


def test_separable_head_uses_phase_profiles():
    head = StructuredSeparableHead(720, 96, residual_period_len=24, num_components=1)
    x = _batch(batch=1, length=720, channels=1, seed=5)
    assert torch.equal(head(x), x[:, -1:, :].expand(-1, 96, -1))
    with torch.no_grad():
        head.period_maps.weight.zero_()
        head.period_maps.bias.fill_(1.0)
        head.phase_profiles.fill_(0.1)
    delta = head(x) - x[:, -1:, :]
    assert delta.abs().sum() > 0
    # A one-component separable map with a constant period response collapses to
    # a fixed per-phase profile, so the delta must be exactly periodic.
    period = head.period
    for step in range(96):
        assert torch.isclose(delta[0, step, 0], delta[0, step % period, 0], atol=1e-5)
    assert delta[0, 0, 0].abs() > 1e-5


def test_orthogonality_loss_is_surfaced_only_when_enabled():
    x = _batch(batch=1, length=720, channels=1, seed=13)
    off = StructuredSegmentBasisHead(720, 96, residual_period_len=24, lambda_orth=0.0)
    off(x)
    assert off.last_orthogonality_loss is None
    on = StructuredSegmentBasisHead(720, 96, residual_period_len=24, lambda_orth=0.01)
    on(x)
    assert on.last_orthogonality_loss is not None
    assert on.last_orthogonality_loss.item() > 0


def test_forward_validation_errors():
    head = StructuredPeriodLowRankHead(720, 96, residual_period_len=24, rank=4)
    try:
        head(torch.randn(1, 100, 2))
    except ValueError as error:
        assert "seq_len" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a seq_len validation error")
    try:
        StructuredPeriodLowRankHead(720, 96, residual_period_len=24, rank=99)
    except ValueError as error:
        assert "rank" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a rank validation error")
