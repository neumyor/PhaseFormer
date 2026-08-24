import torch
import torch.nn as nn


class PhaseVelocity(nn.Module):
    """Phase velocity trajectory modeling (PhaseFormer next-stage plan, stage 1).

    Upgrades the static phase offset phi' = phi + delta (PhaseCorrection) to a
    dynamic phase *trajectory* phi_t = phi_{t-1} + delta_phi_t: a Velocity
    Encoder predicts a per-(sample, channel, slot) velocity delta_phi_t from
    the latent phase token, the Trajectory Integration cumulatively sums the
    velocities along the phase-slot axis (the integrated displacement of each
    slot on the phase circle), and Phase Warping re-aligns the token ordering
    by that displacement.

    The warping uses the same k=2 scatter convention as `PhaseCorrection`
    (identity scatter at delta == 0), so with a zero-init final layer the
    module is a warm-start deformation of the fixed phase grid: velocities are
    zero, the cumulative displacement is zero, and forward() is the identity.

    The velocity form lets the *rate of phase advance* vary across the cycle
    and across samples (period runs early, late, or at a changing speed),
    whereas the offset form only allows a single static per-slot shift.
    """

    def __init__(self, dim: int, hidden: int = 8, velocity_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.velocity_scale = velocity_scale
        # Velocity encoder: Linear(dim, hidden) -> GELU -> Linear(hidden, 1).
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so velocities are zero -> identity scatter.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hooks: mean |velocity| and mean |cumulative displacement|
        # from the last forward pass.
        self.last_mean_velocity = 0.0
        self.last_mean_delta = 0.0
        # Per-sample capture from the last forward pass (analysis/visualization
        # only; no parameters, no effect on the graph).
        self.last_vel = None  # (B, C, L)
        self.last_delta = None  # (B, C, L)

    def forward(self, tokens):  # (B, C, L, D)
        B, C, L, D = tokens.shape
        # Velocity per slot, bounded by velocity_scale.
        vel = self.velocity_scale * torch.tanh(self.net(tokens).squeeze(-1))  # (B, C, L)
        # Trajectory integration: cumulative displacement along the slot axis.
        delta = torch.cumsum(vel, dim=-1)  # (B, C, L)
        # Continuous position on the phase circle: pos = slot + cumulative delta.
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
            self.last_mean_velocity = float(vel.abs().mean())
            self.last_mean_delta = float(delta.abs().mean())
            self.last_vel = vel.detach()
            self.last_delta = delta.detach()
        return out
