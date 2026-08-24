import torch
import torch.nn as nn


class TrajectoryDecoder(nn.Module):
    """Trajectory decoder for pure phase forecasting (pure-phase plan, stage 4).

    The top-level PhasePredictor maps each phase latent (B, C, L, D) to a free
    per-period output (B, C, L, P_out): the value of each phase slot at each of
    the P_out future cycles, independently per cycle. This decoder instead
    models the *future phase sequence per slot* as a low-order polynomial
    trajectory in the period index:

        y[b, c, l, :] = sum_m coef[b, c, l, m] * t^m,   t in [-1, 1] over P_out

    The polynomial structure enforces trajectory consistency across the P_out
    axis (the future phase sequence evolves smoothly from cycle to cycle) and
    phase smoothness (low-order means no cycle-to-cycle discontinuity) — both
    are properties the plan asks the decoder to inject, rather than a loss term.
    With order=2 each slot has 3 free coefficients, so the decoder keeps per-slot
    freedom while strongly constraining the inter-cycle dynamics.

    Output signature matches PhasePredictor: (B, C, L, P_out), so it is a
    drop-in replacement for the top-level predictor in PhaseFormer.forward().

    Diagnostics from the last forward pass (analysis only, no-grad):
      - last_smoothness: mean |y_{k+1} - y_k| over the period axis (cycle
        smoothness of the decoded phase sequence)
    """

    def __init__(self, latent_dim: int, p_out: int, hidden: int = 64, order: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.p_out = p_out
        self.order = order
        self.coef_net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, order + 1),
        )
        # Fixed normalized period axis over the P_out future cycles.
        self.register_buffer("t", torch.linspace(-1.0, 1.0, p_out))
        # Diagnostic hook (analysis only, no parameters).
        self.last_smoothness = 0.0

    def forward(self, z):  # (B, C, L, D)
        B, C, L, D = z.shape
        coef = self.coef_net(z)  # (B, C, L, order + 1)
        t = self.t.view(1, 1, 1, self.p_out)  # (1, 1, 1, P_out)
        y = torch.zeros(B, C, L, self.p_out, device=z.device, dtype=z.dtype)
        t_pow = torch.ones_like(t)
        for m in range(self.order + 1):
            y = y + coef[..., m : m + 1] * t_pow
            t_pow = t_pow * t

        with torch.no_grad():
            self.last_smoothness = float((y[..., 1:] - y[..., :-1]).abs().mean())
        return y
