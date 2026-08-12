import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    """Reliability gate for adaptive phase evolution (RAPE).

    Adaptive phase warping + amplitude calibration always change the phase
    representation, even on stable strong-period windows where the original
    fixed-grid phase prior is already adequate. This module predicts a per-
    sample, per-channel reliability g in (0, 1) and fuses the fully adapted
    representation (warped + amplitude calibrated) with the original PhaseFormer
    phase prior:

        h~ = g * h_adapted + (1 - g) * h_identity

    g near 1 lets the model use the learned phase evolution; g near 0 falls
    back to the fixed phase grid. The gate is conditioned on history-window
    statistics (recent volatility, linear slope) and phase-domain statistics
    (same-slot instability across periods, magnitude of the adaptation). The
    final layer is zero-initialised so g = 0.5 at construction; because warp
    and amplitude calibration are both identity at construction, the fused
    output reduces to the identity phase regardless of g (warm start).
    """

    def __init__(self, hidden: int = 8, eps: float = 1e-6):
        super().__init__()
        self.hidden = hidden
        self.eps = max(float(eps), 1e-8)

        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init -> g = sigmoid(0) = 0.5 at construction (neutral start).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hook (plain float, never in the state_dict).
        self.last_mean_gate = 0.5

    def _features(self, x_in, phase_adapted, phase_identity):
        # x_in: (B, L, C) RevIN-normalized history; phase_*: (B, C, L, P)
        recent = x_in[:, 1:, :] - x_in[:, :-1, :]
        volatility = recent.abs().mean(dim=1)  # (B, C)
        L = x_in.shape[1]
        t = torch.arange(
            L, device=x_in.device, dtype=x_in.dtype
        ).view(1, L, 1) - (L - 1.0) / 2.0
        denom = torch.square(t).sum().clamp_min(self.eps)
        slope = (
            (x_in - x_in.mean(dim=1, keepdim=True)) * t
        ).sum(dim=1).abs() / denom  # (B, C)
        if phase_adapted.size(-1) >= 2:
            pdiff = phase_adapted[..., 1:] - phase_adapted[..., :-1]
            phase_instability = pdiff.abs().mean(dim=(2, 3))  # (B, C)
        else:
            phase_instability = torch.zeros_like(volatility)
        adapt = (phase_adapted - phase_identity).abs().mean(dim=(2, 3))  # (B, C)
        return torch.stack(
            [volatility, slope, phase_instability, adapt], dim=-1
        )  # (B, C, 4)

    def forward(self, x_in, phase_adapted, phase_identity):
        """Return per-channel gate g in (0, 1), shape (B, C)."""
        feat = self._features(x_in, phase_adapted, phase_identity)  # (B, C, 4)
        g = torch.sigmoid(self.net(feat).squeeze(-1))  # (B, C)
        with torch.no_grad():
            self.last_mean_gate = float(g.mean())
        return g
