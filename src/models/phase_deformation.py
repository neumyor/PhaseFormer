import torch
import torch.nn as nn


class PhaseDeformation(nn.Module):
    """Nonlinear phase deformation field (pure-phase plan, stage 2).

    PhaseFormer's phase tokens live on a fixed cycle grid. PhaseCorrection
    shifts each slot by an independent delta in (-1, 1); PhaseVelocity lets a
    small cumulative drift evolve along the slot axis (a near-constant-rate
    trajectory). This module is strictly more expressive: a *deformation field*
    predicts both a per-slot advance rate `v` and a per-slot stretch factor `s`
    from the latent token, then builds the displacement field by cumulatively
    summing ``v * s`` along the slot axis:

        delta_l = sum_{k<l} v_k * s_k,   v in [-scale, scale], s in (0, 2)

    The stretch factor lets the phase advance non-uniformly across the cycle
    (compression where s < 1, stretching where s > 1) and reach larger net
    displacements than the fixed-scale velocity form, so the warp is a genuine
    nonlinear time re-parameterization instead of a constant drift.

    The warping reuses the k=2 scatter convention of `PhaseVelocity`/`PhaseCorrection`
    (identity scatter at delta == 0): with both heads zero-initialized the rates
    are zero, the displacement field is zero, and forward() is the identity, so
    the module is a warm-start deformation of the fixed phase grid.

    Diagnostics from the last forward pass (analysis only, no-grad):
      - last_mean_rate: mean |v|
      - last_mean_stretch: mean |s - 1|
      - last_mean_delta: mean |cumulative displacement|
      - last_rate: per-(sample, channel, slot) advance rate v
      - last_stretch: per-(sample, channel, slot) stretch factor s
      - last_delta: per-(sample, channel, slot) cumulative displacement
    """

    def __init__(self, dim: int, hidden: int = 8, velocity_scale: float = 0.2):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.velocity_scale = velocity_scale
        # Advance-rate head: per-slot velocity bounded by velocity_scale.
        self.net_rate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Stretch head: per-slot factor s = 1 + tanh(...) in (0, 2), init 1.
        self.net_stretch = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init final layers: rate = 0 (identity scatter), stretch = 1.
        nn.init.zeros_(self.net_rate[-1].weight)
        nn.init.zeros_(self.net_rate[-1].bias)
        nn.init.zeros_(self.net_stretch[-1].weight)
        nn.init.zeros_(self.net_stretch[-1].bias)

        # Diagnostic hooks (no parameters, no effect on the graph).
        self.last_mean_rate = 0.0
        self.last_mean_stretch = 0.0
        self.last_mean_delta = 0.0
        self.last_rate = None  # (B, C, L)
        self.last_stretch = None  # (B, C, L)
        self.last_delta = None  # (B, C, L)

    def forward(self, tokens):  # (B, C, L, D)
        B, C, L, D = tokens.shape
        rate = self.velocity_scale * torch.tanh(self.net_rate(tokens).squeeze(-1))  # (B, C, L)
        stretch = 1.0 + torch.tanh(self.net_stretch(tokens).squeeze(-1))  # (B, C, L) in (0, 2)
        # Nonlinear deformation field: cumulative displacement along the slot axis.
        delta = torch.cumsum(rate * stretch, dim=-1)  # (B, C, L)
        # Continuous position on the phase circle: pos = slot + delta.
        base = torch.arange(L, device=tokens.device, dtype=tokens.dtype).view(1, 1, L)
        pos = base + delta  # (B, C, L)
        i0 = pos.floor().long() % L  # floor(-0.5) = -1 -> wraps to L-1
        frac = pos - pos.floor()  # (B, C, L) in [0, 1)
        i1 = (i0 + 1) % L

        out = torch.zeros_like(tokens)
        for l in range(L):
            src = tokens[:, :, l, :].unsqueeze(2)  # (B, C, 1, D)
            t0 = i0[:, :, l].view(B, C, 1, 1).expand(-1, -1, 1, D)  # (B, C, 1, D)
            t1 = i1[:, :, l].view(B, C, 1, 1).expand(-1, -1, 1, D)
            w0 = (1.0 - frac[:, :, l]).view(B, C, 1, 1)
            w1 = frac[:, :, l].view(B, C, 1, 1)
            out.scatter_add_(2, t0, src * w0)
            out.scatter_add_(2, t1, src * w1)

        with torch.no_grad():
            self.last_mean_rate = float(rate.abs().mean())
            self.last_mean_stretch = float((stretch - 1.0).abs().mean())
            self.last_mean_delta = float(delta.abs().mean())
            self.last_rate = rate.detach()
            self.last_stretch = stretch.detach()
            self.last_delta = delta.detach()
        return out
