"""Structured low-rank NLinear residual heads (breadth-first exploration plan).

These heads replace the time-point axis of the NLinear residual branch with a
period/segment coordinate system.  They are additive to the existing branch:
`src/models/phase_adapters.py` keeps ``shared``/``pooled_lowrank`` unchanged,
and PhaseFormer selects one of these heads through
``weak_period_residual_head_type`` with a ``structured_`` prefix.

Plan reference:
``docs/PhaseFormer_nlinear_structured_lowrank_breadth_first_exploration_plan.md``
(sections 4, 7 and 10.1).

Shared invariants (plan section 10.1), verified by ``tests/``:

- input ``(B, L, C)`` in the RevIN-normalized space, output ``(B, H, C)``;
- the centered history ``X - X_last`` may be split/gathered into period
  segments, but the last value never enters the dynamic input;
- every branch is zero-initialized so that at initialization the head returns
  ``delta + last`` with ``delta == 0``, i.e. the persistence anchor exactly;
- ``residual_period_len`` is a head-local field.  The PhaseFormer phase path
  keeps its own ``period_len``.

Two segmentation variants implement the required pseudo-structure control
(plan section 7): ``aligned`` splits from the window boundary, ``shifted`` uses
the fixed offset ``P // 2`` and ``random`` a deterministic per-run offset.
For an hourly dataset whose native cycle is ``P`` (for example ETT with
``period_len=24``), the aligned split is the physically meaningful one and the
shifted/random splits are near-orthogonal to it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


SEGMENT_ALIGNMENTS = ("aligned", "shifted", "random")

# Fixed exponential weighting for route D.  Route D compares the two weighting
# schemes pre-registered in the plan ("hard recent window" versus "fixed
# exponential weighting"); the decay is part of the representative
# configuration and is recorded with every run.
DEFAULT_RECENT_DECAY = 0.8


@dataclass(frozen=True)
class SegmentLayout:
    """Resolved index bookkeeping for one (head, setting) segmentation.

    ``tap_length = offset + L`` is the size of the padded centered history, and
    ``segment_indices[s]`` holds the ``P`` positions of segment ``s`` counted
    backwards from the end of that buffer (``x[-1]`` is the last observed
    value).  Counting from the end keeps the physical time span covered by a
    fixed number of taps invariant under ``offset``.
    """

    period: int
    offset: int
    tap_length: int
    segment_indices: torch.Tensor  # (K, P)

    @property
    def num_segments(self) -> int:
        return int(self.segment_indices.shape[0])


def resolve_segment_layout(
    seq_len: int,
    period: int,
    alignment: str = "aligned",
    seed: int = 2021,
    key: str = "",
    period_offset: Optional[int] = None,
) -> SegmentLayout:
    """Build the segment index map for one head instance.

    ``aligned`` uses offset 0, ``shifted`` uses ``period // 2`` and ``random``
    a deterministic offset derived from ``(seed, key)``.  Layouts are fixed at
    construction time: every sample of a run shares the same segmentation.
    """

    if period < 1:
        raise ValueError("residual_period_len must be >= 1")
    if alignment not in SEGMENT_ALIGNMENTS:
        raise ValueError(
            f"residual_segment_alignment must be one of {SEGMENT_ALIGNMENTS}"
        )
    if period_offset is not None:
        offset = int(period_offset)
    elif alignment == "shifted":
        offset = period // 2
    elif alignment == "random":
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
        offset = int(digest[:12], 16) % period
    else:
        offset = 0
    if offset < 0 or offset >= period:
        raise ValueError("residual_segment_offset must lie in [0, residual_period_len)")

    tap_length = offset + int(seq_len)
    num_segments = math.ceil(tap_length / period)
    # Newest-first positions: -1 is the last observed value, -(1 + P) the value
    # one period earlier, and so on.  Negative positions fall into the implicit
    # zero pad appended to the centered history, which is exactly "center to
    # zero" for values outside the observed window.
    positions = -(torch.arange(period, dtype=torch.long) + 1) - period * torch.arange(
        num_segments, dtype=torch.long
    ).unsqueeze(1)
    # Positions before the observed window are clamped into the implicit zero
    # pad appended to the centered history, which is exactly "center to zero" for
    # values outside the window.  The pad is a single extra slot at index
    # ``seq_len``; a whole old segment maps onto it, which is harmless because
    # those values are genuinely absent.
    segment_indices = (tap_length + positions).clamp_(min=0, max=int(seq_len))
    return SegmentLayout(
        period=int(period),
        offset=offset,
        tap_length=tap_length,
        segment_indices=segment_indices,
    )


def gather_segments(
    centered: torch.Tensor,
    layout: SegmentLayout,
    *,
    phase_convention: bool = False,
) -> torch.Tensor:
    """``(B, C, L)`` centered history -> ``(B, C, K, P)`` segments.

    In the raw layout row ``s`` is the period ending ``s`` periods before the last
    observation and column ``q`` counts backwards from that period's end, so the
    absolute index of cell ``(s, q)`` is ``tap_length - 1 - q - s * P``.

    With ``phase_convention`` every cell is instead addressed by its **lag**: the
    number of steps between the cell's timestamp and the end of the observed
    history.  Cell ``(s, c)`` holds lag ``s * P + c``, so column ``c`` always means
    lag ``c`` modulo ``P`` -- the same absolute phase in every row -- while the row
    index still advances by whole periods.  That is exactly the labelling a
    phase-wise weight (routes B, C and E) needs, and it makes the forecast
    equivalent to reading lag ``i`` for forecast step ``i``.  Lags reaching
    further back than the observed window land in a single appended zero slot,
    the "centre to zero" convention for unobserved values.
    """

    if centered.dim() != 3:
        raise ValueError("gather_segments expects (B, C, L)")
    length = centered.shape[2]
    device = centered.device
    period = layout.period
    taps = layout.num_segments
    padded = torch.cat(
        [centered, centered.new_zeros(centered.shape[0], centered.shape[1], 1)], dim=2
    )
    if not phase_convention:
        return padded[:, :, layout.segment_indices.to(device)]
    # Address every cell by its lag: the number of steps between the cell and the
    # end of the observed history.  Row ``s`` column ``c`` gets lag
    # ``lags = s * P + c``, so column ``c`` always means the same phase modulo
    # ``P``, while the row index still advances by whole periods.  A positive lag
    # lies inside the history and a negative lag (which only the shifted and
    # random segmentations reach) is served by the appended zero slot.
    rows = torch.arange(taps, dtype=torch.long, device=device).view(taps, 1)
    columns = torch.arange(period, dtype=torch.long, device=device).view(1, period)
    lags = rows * period + columns
    indices = torch.where(
        lags <= length - 1,
        length - 1 - lags,
        torch.full_like(lags, length),
    )
    indices = indices.clamp_(min=0, max=length)
    return padded[:, :, indices]



def future_segment_order(
    layout: SegmentLayout, pred_len: int, device: torch.device
) -> torch.Tensor:
    """Order in which the flat ``P * K_y`` grid serves the ``pred_len`` steps.

    Let ``T = offset + K * P`` be the exclusive end of the gathered history.  A
    gathered row ``s`` is a window of the history centred on the last observation
    of its own period, and the cells of that row hold the period's phases in
    decreasing order.  The phase slot that forecast phase ``p`` must read lives at
    column ``(offset + p + 1) mod P`` of row ``s``, and in flat terms that is

    ``index(i) = (i // P) * P + ((i + 1) mod P)``.

    One ``gather`` with this table produces the forecast in natural time order
    without any intermediate time buffer.
    """

    return torch.arange(int(pred_len), dtype=torch.long, device=device)


def future_grid_segments(layout: SegmentLayout, pred_len: int) -> int:
    """Number of future segments the order table actually addresses.

    The newest-first flat grid has ``K`` rows, which already covers the deepest
    forecast step, so ``ceil(pred_len / P)`` periods suffice for every alignment.
    """

    return max(1, math.ceil(int(pred_len) / layout.period))


class StructuredResidualHeadBase(nn.Module):
    """Shared centered/anchor bookkeeping for the structured residual heads."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__()
        if seq_len < 1 or pred_len < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.layout = resolve_segment_layout(
            seq_len=seq_len,
            period=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        self.num_taps = self.layout.num_segments
        self.period = self.layout.period
        self.num_future_segments = math.ceil(self.pred_len / self.period)
        # Auxiliary losses (currently only the route-B basis orthogonality term)
        # are surfaced here so PhaseFormer.training_step can add them without
        # changing the head call signature.
        self.last_orthogonality_loss: Optional[torch.Tensor] = None

    # -- helpers ---------------------------------------------------------
    def _centered(self, x: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(x)
        if tensor.dim() != 3:
            raise ValueError("structured residual heads expect (B, L, C) input")
        if tensor.shape[1] != self.seq_len:
            raise ValueError(
                f"expected seq_len={self.seq_len}, received {tensor.shape[1]}"
            )
        return (tensor - tensor[:, -1:, :]).permute(0, 2, 1).contiguous()

    def _order_future(self, segments: torch.Tensor) -> torch.Tensor:
        """``(B, C, K_y, P)`` future segments -> ``(B, H, C)`` ordered forecast.

        The head never writes the forecast into an intermediate time buffer.
        Instead the phase that each forecast step must read is known in closed
        form (see :func:`future_segment_order`), so one ``gather`` over the flat
        grid yields the forecast in natural time order.  For shifted
        segmentations the grid is padded with zero-centred future periods, which
        is the same "unobserved means zero" convention used on the history side.
        """

        batch, channels, num_segments, period = segments.shape
        required = future_grid_segments(self.layout, self.pred_len)
        if num_segments < required:
            pad = segments.new_zeros(
                batch, channels, required - num_segments, period
            )
            segments = torch.cat([segments, pad], dim=2)
        # Buffer column ``c`` of the grid must hold the future value of lag ``c``
        # (ascending phase), matching the lag labelling used on the history side;
        # the head emits its phase axis in that same order.
        grid = segments.reshape(batch, channels, segments.shape[2] * period)
        order = future_segment_order(self.layout, self.pred_len, segments.device)
        index = order.view(1, 1, -1).expand(batch, channels, -1)
        return torch.gather(grid, 2, index)

    def _finalize(self, delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Re-add the persistence anchor to an already ordered forecast."""

        if delta.shape[2] != self.pred_len:
            raise ValueError(
                f"ordered forecast has width {delta.shape[2]}, expected "
                f"{self.pred_len}"
            )
        return delta.permute(0, 2, 1).contiguous() + x[:, -1:, :]

    def _map_taps(self, layer: nn.Module, segments: torch.Tensor) -> torch.Tensor:
        """Apply a ``K -> R`` or ``K -> K_y`` map along the period/tap axis.

        ``segments`` is ``(B, C, K, P)``: the map runs along ``K`` for one phase
        slot while the phase slots are batched together, so the phase axis keeps
        its own weights per slot.
        """

        return layer(segments.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    def parameter_count(self) -> int:
        """Trainable parameter count of this head (plan section 3.3 main scope)."""

        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


class StructuredPeriodLowRankHead(StructuredResidualHeadBase):
    """Route A: rank bottleneck across the period axis, shared per phase slot.

    ``Y[k_y, p] = sum_k W[k_y, k] S[k, p]`` with ``W = decoder @ encoder``.  The
    encoder consumes the ``K`` period taps of every phase slot, producing a
    latent whose last axis is the rank; the decoder then writes *all* phase slots
    of the first future period, i.e. its output axis is laid out as
    ``(phase slot, k_y)`` so that one shared ``K -> K_y`` map serves every phase.
    The mapping is applied once; no predicted period is fed back as input.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        rank: int = 4,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__(
            seq_len,
            pred_len,
            residual_period_len=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank > self.num_taps:
            raise ValueError(
                f"rank={rank} exceeds the number of history segments {self.num_taps}"
            )
        self.rank = int(rank)
        self.segment_encoder = nn.Linear(self.num_taps, self.rank)
        # Flat decoder axis is ``phase_slot * K_y + k_y``.
        self.segment_decoder = nn.Linear(
            self.rank, self.period * self.num_future_segments
        )
        nn.init.zeros_(self.segment_decoder.weight)
        nn.init.zeros_(self.segment_decoder.bias)

    def forward(self, x):  # x: (B, L, C)
        self.last_orthogonality_loss = None
        centered = self._centered(x)
        # Column ``c`` carries lag ``c``, so the phase index is the shared axis
        # that lets one ``K -> K_y`` map serve every phase slot.
        segments = gather_segments(
            centered, self.layout, phase_convention=True
        )  # (B, C, K, P)
        batch, channels = segments.shape[0], segments.shape[1]
        latent = self.segment_encoder(segments.permute(0, 1, 3, 2))  # (B, C, P, r)
        decoded = self.segment_decoder(latent)  # (B, C, P, P * K_y)
        # Decoder output axis is ``phase * K_y + k_y``; regroup it so that period
        # ``k_y`` occupies one block of ``P`` consecutive phase slots.
        grouped = decoded.reshape(
            batch,
            channels,
            self.period,
            self.num_future_segments,
            self.period,
        )[:, :, :, :, 0]
        future = grouped.permute(0, 1, 3, 2)  # (B, C, K_y, P)
        return self._finalize(self._order_future(future), x)


class StructuredSegmentBasisHead(StructuredResidualHeadBase):
    """Route B: learn a small history basis and decode it into future segments.

    ``B = E(S)`` with ``E: K -> R`` and ``F = D(B)`` with ``D: R -> K_y``.  The
    optional ``lambda_orth`` term penalises redundancy between the learned
    basis directions; it is exposed as ``last_orthogonality_loss``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        num_basis: int = 4,
        lambda_orth: float = 0.0,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__(
            seq_len,
            pred_len,
            residual_period_len=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        if num_basis < 1:
            raise ValueError("num_basis must be >= 1")
        if num_basis > self.num_taps:
            raise ValueError(
                f"num_basis={num_basis} exceeds the number of history segments "
                f"{self.num_taps}"
            )
        if lambda_orth < 0.0:
            raise ValueError("lambda_orth must be non-negative")
        self.num_basis = int(num_basis)
        self.lambda_orth = float(lambda_orth)
        self.basis = nn.Linear(self.num_taps, self.num_basis)
        self.segment_decoder = nn.Linear(self.num_basis, self.num_future_segments)
        nn.init.zeros_(self.segment_decoder.weight)
        nn.init.zeros_(self.segment_decoder.bias)

    def forward(self, x):  # x: (B, L, C)
        self.last_orthogonality_loss = None
        centered = self._centered(x)
        segments = gather_segments(
            centered, self.layout, phase_convention=True
        )  # (B, C, K, P), one column per phase slot
        coefficients = self._map_taps(self.basis, segments)  # (B, C, R, P)
        future = self._map_taps(self.segment_decoder, coefficients)  # (B, C, K_y, P)
        if self.lambda_orth:
            basis = self.basis.weight  # (R, K)
            gram = basis @ basis.transpose(0, 1)
            identity = torch.eye(self.num_basis, device=gram.device, dtype=gram.dtype)
            self.last_orthogonality_loss = self.lambda_orth * (gram - identity).pow(2).mean()
        return self._finalize(self._order_future(future), x)


class StructuredLevelShapeHead(StructuredResidualHeadBase):
    """Route C: split each period into a level and a shape component.

    ``level_k = mean_p S[k, p]`` and ``shape_{k,p} = S[k,p] - level_k`` are
    modelled separately.  The level path may be dense or rank-limited, and the
    shape path always stays on the period axis (``K -> K_y``) shared across
    phase slots, so the three pre-registered Round-1 conditions differ only in
    the level parameterization and the presence of the shape path.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        level_mode: str = "dense",
        level_rank: int = 1,
        shape_rank: Optional[int] = 4,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__(
            seq_len,
            pred_len,
            residual_period_len=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        if level_mode not in {"dense", "lowrank"}:
            raise ValueError("level_mode must be 'dense' or 'lowrank'")
        self.level_mode = level_mode
        if level_mode == "lowrank":
            if level_rank < 1 or level_rank > self.num_taps:
                raise ValueError("level_rank must lie in [1, K]")
            self.level_rank = int(level_rank)
            self.level_encoder = nn.Linear(self.num_taps, self.level_rank)
            self.level_decoder = nn.Linear(self.level_rank, self.num_future_segments)
            nn.init.zeros_(self.level_decoder.weight)
            nn.init.zeros_(self.level_decoder.bias)
        else:
            self.level_map = nn.Linear(self.num_taps, self.num_future_segments)
            nn.init.zeros_(self.level_map.weight)
            nn.init.zeros_(self.level_map.bias)
        self.shape_rank = None if shape_rank is None else int(shape_rank)
        if self.shape_rank is not None:
            if self.shape_rank < 1 or self.shape_rank > self.num_taps:
                raise ValueError("shape_rank must lie in [1, K]")
            self.shape_encoder = nn.Linear(self.num_taps, self.shape_rank)
            self.shape_decoder = nn.Linear(self.shape_rank, self.num_future_segments)
            nn.init.zeros_(self.shape_decoder.weight)
            nn.init.zeros_(self.shape_decoder.bias)

    def _level_forecast(self, levels: torch.Tensor) -> torch.Tensor:
        """``(B, C, K)`` segment levels -> ``(B, C, K_y)`` future levels."""

        if self.level_mode == "lowrank":
            return self.level_decoder(self.level_encoder(levels))
        return self.level_map(levels)

    def forward(self, x):  # x: (B, L, C)
        self.last_orthogonality_loss = None
        centered = self._centered(x)
        segments = gather_segments(
            centered, self.layout, phase_convention=True
        )  # (B, C, K, P)
        levels = segments.mean(dim=3)  # (B, C, K)
        future_levels = self._level_forecast(levels)  # (B, C, K_y)
        delta_segments = future_levels.unsqueeze(-1).expand(
            -1, -1, -1, self.period
        ).clone()
        if self.shape_rank is not None:
            shapes = segments - levels.unsqueeze(-1)
            latent = self.shape_encoder(shapes.permute(0, 1, 3, 2))  # (B, C, P, r)
            future_shapes = self.shape_decoder(latent).permute(0, 1, 3, 2)
            delta_segments = delta_segments + future_shapes
        return self._finalize(self._order_future(delta_segments), x)


class StructuredRecentPeriodHead(StructuredResidualHeadBase):
    """Route D: keep only recent period taps, optionally with fixed decay.

    ``recent_n`` taps stay dense inside the ``K -> K_y`` map and the remaining
    history taps are dropped, so the sparse budget is explicit.  ``weighting``
    selects hard selection or a fixed exponential weighting; no learnable gate
    is introduced (plan section 4, route D).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        recent_taps: int = 7,
        weighting: str = "hard",
        decay: float = DEFAULT_RECENT_DECAY,
        rank: Optional[int] = None,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__(
            seq_len,
            pred_len,
            residual_period_len=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        if weighting not in {"hard", "fixed_exp"}:
            raise ValueError("weighting must be 'hard' or 'fixed_exp'")
        if recent_taps < 1 or recent_taps > self.num_taps:
            raise ValueError("recent_taps must lie in [1, K]")
        self.recent_taps = int(recent_taps)
        self.weighting = weighting
        self.decay = float(decay)
        if rank is None or rank <= 0:
            self.rank = None
            self.segment_map = nn.Linear(self.recent_taps, self.num_future_segments)
            nn.init.zeros_(self.segment_map.weight)
            nn.init.zeros_(self.segment_map.bias)
        else:
            if rank > self.recent_taps:
                raise ValueError("rank must not exceed recent_taps")
            self.rank = int(rank)
            self.segment_encoder = nn.Linear(self.recent_taps, self.rank)
            self.segment_decoder = nn.Linear(self.rank, self.num_future_segments)
            nn.init.zeros_(self.segment_decoder.weight)
            nn.init.zeros_(self.segment_decoder.bias)
        if weighting == "fixed_exp":
            # A fixed, non-learnable recency profile.  ``gather_segments`` returns
            # segments newest-first, so tap 0 is the most recent period and gets
            # weight 1; older taps decay geometrically.
            weights = torch.tensor(
                [self.decay**tap for tap in range(self.recent_taps)]
            )
            self.register_buffer("recency_weights", weights.view(1, 1, -1, 1))

    def forward(self, x):  # x: (B, L, C)
        self.last_orthogonality_loss = None
        centered = self._centered(x)
        segments = gather_segments(
            centered, self.layout, phase_convention=True
        )  # (B, C, K, P), newest tap first
        recent = segments[:, :, : self.recent_taps, :]
        if self.weighting == "fixed_exp":
            recent = recent * self.recency_weights
        if self.rank is None:
            future = self._map_taps(self.segment_map, recent)
        else:
            latent = self._map_taps(self.segment_encoder, recent)
            future = self._map_taps(self.segment_decoder, latent)
        return self._finalize(self._order_future(future), x)


class StructuredSeparableHead(StructuredResidualHeadBase):
    """Route E: separable period-by-phase components ``sum_j A_j (x) B_j``.

    Each component combines a ``K -> K_y`` period-axis map with a learnable
    phase-slot profile of length ``P``.  The phase profiles are zero-initialized,
    so the head starts at the persistence anchor while the period maps keep
    ordinary initialization and therefore receive gradient.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        residual_period_len: int = 24,
        num_components: int = 1,
        alignment: str = "aligned",
        seed: int = 2021,
        key: str = "",
        period_offset: Optional[int] = None,
    ):
        super().__init__(
            seq_len,
            pred_len,
            residual_period_len=residual_period_len,
            alignment=alignment,
            seed=seed,
            key=key,
            period_offset=period_offset,
        )
        if num_components < 1:
            raise ValueError("num_components must be >= 1")
        self.num_components = int(num_components)
        self.period_maps = nn.Linear(
            self.num_taps, self.num_components * self.num_future_segments
        )
        self.phase_profiles = nn.Parameter(
            torch.zeros(self.num_components, self.period)
        )

    def forward(self, x):  # x: (B, L, C)
        self.last_orthogonality_loss = None
        centered = self._centered(x)
        segments = gather_segments(
            centered, self.layout, phase_convention=True
        )  # (B, C, K, P)
        levels = segments.mean(dim=3)  # (B, C, K); the phase profile carries shape
        mapped = self.period_maps(levels)  # (B, C, J * K_y)
        mapped = mapped.view(
            centered.shape[0], centered.shape[1], self.num_components,
            self.num_future_segments,
        )
        phase = self.phase_profiles.view(1, 1, self.num_components, 1, self.period)
        delta_segments = (mapped.unsqueeze(-1) * phase).sum(dim=2)  # (B, C, K_y, P)
        return self._finalize(self._order_future(delta_segments), x)


class TimeAxisMatchedLowRankHead(nn.Module):
    """Time-point-axis low-rank control with a matched head parameter budget.

    ``centered -> Linear(720 -> r_match) -> Linear(r_match -> H) -> + last``.
    It performs no pooling, period segmentation, basis expansion, recent
    selection or smoothing, so comparing a structured route against this
    control isolates the coordinate system from the parameter reduction
    (plan section 3.3).
    """

    def __init__(self, seq_len: int, pred_len: int, *, rank: int):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank > min(seq_len, pred_len):
            raise ValueError(
                f"rank={rank} exceeds the factorized-map limit "
                f"{min(seq_len, pred_len)}"
            )
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.rank = int(rank)
        self.encoder = nn.Linear(self.seq_len, self.rank)
        self.decoder = nn.Linear(self.rank, self.pred_len)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):  # x: (B, L, C)
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        delta = self.decoder(self.encoder(centered)).permute(0, 2, 1).contiguous()
        return delta + last

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def matched_control_rank(
    target_head_params: int, seq_len: int, pred_len: int
) -> tuple[int, int]:
    """Nearest integer ``r_match`` for the matched time-axis control.

    Returns ``(rank, parameter_count)``.  The nearest rank is chosen without
    exceeding ``target_head_params``, so the control never receives a larger
    budget than the structured candidate it is matched against.
    """

    if target_head_params < seq_len + pred_len:
        raise ValueError("target budget is too small for a rank-1 control")
    best = 1
    for rank in range(1, min(seq_len, pred_len) + 1):
        if rank * (seq_len + pred_len) <= target_head_params:
            best = rank
        else:
            break
    return best, best * (seq_len + pred_len)


STRUCTURED_HEAD_BUILDERS = {
    "structured_period_lowrank": StructuredPeriodLowRankHead,
    "structured_segment_basis": StructuredSegmentBasisHead,
    "structured_level_shape": StructuredLevelShapeHead,
    "structured_recent_period": StructuredRecentPeriodHead,
    "structured_separable": StructuredSeparableHead,
}


def build_structured_residual_head(
    head_type: str,
    seq_len: int,
    pred_len: int,
    configs,
    *,
    seed: int = 2021,
    key: str = "",
) -> nn.Module:
    """Instantiate a structured head from a PhaseFormer config object.

    The factory reads ``residual_*`` fields from ``configs``; every field has a
    default so that a bare ``weak_period_residual_head_type`` override is
    runnable.  ``seed``/``key`` drive the deterministic ``random`` segmentation
    and are supplied by the caller so that the config object is not mutated.
    PhaseFormer keeps its own ``period_len``.
    """

    if head_type not in STRUCTURED_HEAD_BUILDERS:
        raise ValueError(f"unknown structured residual head type: {head_type}")
    get = lambda name, default: getattr(configs, name, default)  # noqa: E731
    common = dict(
        residual_period_len=int(get("residual_period_len", 24)),
        alignment=get("residual_segment_alignment", "aligned"),
        seed=int(get("residual_segment_seed", seed) or seed),
        key=get("residual_segment_key", "") or key,
        period_offset=get("residual_segment_offset", None),
    )
    if head_type == "structured_period_lowrank":
        return StructuredPeriodLowRankHead(
            seq_len,
            pred_len,
            rank=int(get("residual_period_rank", 4)),
            **common,
        )
    if head_type == "structured_segment_basis":
        return StructuredSegmentBasisHead(
            seq_len,
            pred_len,
            num_basis=int(get("residual_basis_count", 4)),
            lambda_orth=float(get("residual_basis_lambda_orth", 0.0)),
            **common,
        )
    if head_type == "structured_level_shape":
        shape_rank = get("residual_shape_rank", 4)
        return StructuredLevelShapeHead(
            seq_len,
            pred_len,
            level_mode=get("residual_level_mode", "dense"),
            level_rank=int(get("residual_level_rank", 1)),
            shape_rank=None if shape_rank is None else int(shape_rank),
            **common,
        )
    if head_type == "structured_recent_period":
        rank = get("residual_recent_rank", None)
        return StructuredRecentPeriodHead(
            seq_len,
            pred_len,
            recent_taps=int(get("residual_recent_taps", 7)),
            weighting=get("residual_recent_weighting", "hard"),
            decay=float(get("residual_recent_decay", DEFAULT_RECENT_DECAY)),
            rank=None if rank is None else int(rank),
            **common,
        )
    if head_type == "structured_separable":
        return StructuredSeparableHead(
            seq_len,
            pred_len,
            num_components=int(get("residual_separable_components", 1)),
            **common,
        )
    raise ValueError(f"unknown structured residual head type: {head_type}")
