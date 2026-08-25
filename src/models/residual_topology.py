"""Residual paths used by the residual-topology ablation.

The modules in this file are deliberately small and zero-initialized.  Turning
one on therefore preserves the PhaseFormer prediction at construction time and
changes only the location at which a residual path can be learned.
"""

import torch
import torch.nn as nn


class AdditiveOutputResidualHead(nn.Module):
    """Map centered normalized history to an additive forecast correction.

    Unlike :class:`WeakPeriodResidualHead`, this head does not add a persistence
    anchor and does not produce a competing full forecast.  Its output is a
    true correction ``delta_y`` for ``y = y_phase + gate * delta_y``.
    """

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.last_mean_abs_correction = 0.0

    def forward(self, x):  # x: (B, L, C), normalized scale
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        correction = self.linear(centered).permute(0, 2, 1).contiguous()
        self.last_mean_abs_correction = float(
            correction.detach().abs().mean().cpu()
        )
        return correction


class LatentResidualPath(nn.Module):
    """Project the initial phase latent into one or more routing depths.

    Each depth owns an independent ``D -> D`` bias-free projection.  All
    projections start at zero, so both a single long skip and repeated
    layer-wise injection are exact warm-start identities.
    """

    def __init__(self, latent_dim: int, num_injections: int):
        super().__init__()
        if num_injections < 1:
            raise ValueError("num_injections must be positive")
        self.projections = nn.ModuleList(
            nn.Linear(latent_dim, latent_dim, bias=False)
            for _ in range(num_injections)
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
        self.last_mean_abs_correction = 0.0

    def forward(self, anchor, injection_index: int = 0):
        correction = self.projections[injection_index](anchor)
        self.last_mean_abs_correction = float(
            correction.detach().abs().mean().cpu()
        )
        return correction


class PhaseSlotResidualHead(nn.Module):
    """Input-derived residual in the phase-slot domain ``(B, C, num_slots, P)``.

    This is the layer-wise (per routing depth) analogue of the single-point
    output residuals.  It lets an input-derived forecast be fused onto an
    intermediate routing layer's phase-series prediction ``y_phase_steps_p_in``
    before that prediction feeds the next layer, so the output residual gains a
    depth axis.

    Two forms, selected by ``anchor``:
      - convex (anchor=True): persistence anchor = the last observed period,
        plus a zero-init linear delta from the centered history.  This is the
        phase-domain analogue of :class:`WeakPeriodResidualHead`.
      - additive (anchor=False): zero-init linear correction only, an exact
        warm-start identity for any gate value.

    The linear maps the centered history ``(B, C, seq_len) -> (B, C, P)`` and
    the result is broadcast over the ``num_slots`` slot axis.
    """

    def __init__(
        self, seq_len: int, num_periods: int, num_slots: int, anchor: bool = False
    ):
        super().__init__()
        self.linear = nn.Linear(seq_len, num_periods)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.anchor = anchor
        self.num_slots = num_slots
        self.last_mean_abs_correction = 0.0

    def forward(self, x, last_period=None):
        # x: (B, seq_len, C) normalized scale; last_period: (B, C, P) or None
        centered = (x - x[:, -1:, :]).permute(0, 2, 1).contiguous()  # (B, C, seq_len)
        delta = self.linear(centered).unsqueeze(1)  # (B, 1, C, P)
        delta = delta.expand(-1, self.num_slots, -1, -1)  # (B, num_slots, C, P)
        out = delta.permute(0, 2, 1, 3).contiguous()  # (B, C, num_slots, P)
        if self.anchor:
            if last_period is None:
                raise ValueError("anchor form requires last_period")
            anchor = last_period.unsqueeze(1).expand(-1, self.num_slots, -1, -1)
            out = out + anchor.permute(0, 2, 1, 3).contiguous()
        self.last_mean_abs_correction = float(out.detach().abs().mean().cpu())
        return out
