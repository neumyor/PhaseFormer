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
