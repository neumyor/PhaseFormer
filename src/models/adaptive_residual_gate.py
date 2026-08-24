import math

import torch
import torch.nn as nn


class AdaptiveResidualGate(nn.Module):
    """Adaptive residual fusion gate (PhaseFormer next-stage plan, stage 3).

    Replaces the fixed residual fusion (forecast = phase + residual) with an
    adaptive one:

        y = (1 - alpha) * y_p + alpha * y_r ,   alpha = sigmoid(Gate(Z, x))

    where y_p is the phase forecast and y_r is the residual (NLinear) forecast.
    Because `WeakPeriodResidualHead` already outputs a full forecast (anchored
    at the last value), the plan's additive form y = y_p + alpha * y_r is
    realized as a convex combination between the two full forecasts, so a
    dataset that does not want the residual head simply drives alpha -> 0 and
    recovers the phase-only forecast.

    The gate conditions on the latent phase feature Z (mean/std pooled over
    the phase-slot axis) plus the recent input volatility (an auxiliary trend
    signal), yielding one scalar alpha per (sample, channel).

    Warm start: the final layer is zero-init with a gate_init bias, so
    alpha = gate_init at initialization (default 0.5, matching the fixed
    residual_full gate).
    """

    def __init__(self, phase_dim: int, enc_in: int, hidden: int = 8, gate_init: float = 0.5):
        super().__init__()
        self.phase_dim = phase_dim
        self.hidden = hidden
        # Gate network: Linear(2*phase_dim + 1, hidden) -> GELU -> Linear(hidden, 1).
        self.net = nn.Sequential(
            nn.Linear(2 * phase_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.net[-1].bias, math.log(gate_init / (1.0 - gate_init)))
        # Per-channel bias so alpha can differ by series.
        self.channel_bias = nn.Parameter(torch.zeros(1, enc_in, 1))
        # Per-sample capture from the last forward pass (analysis/visualization
        # only; no parameters, no effect on the graph).
        self.last_alpha = None  # (B, 1, C)

    def forward(self, Z, x_in):
        # Z: (B, C, L, D) latent phase feature; x_in: (B, L, C) normalized input.
        z_mean = Z.mean(dim=2)  # (B, C, D)
        z_std = Z.std(dim=2)  # (B, C, D)
        diff = x_in[:, 1:, :] - x_in[:, :-1, :]
        vol = diff.abs().mean(dim=1).unsqueeze(-1)  # (B, C, 1)
        feat = torch.cat([z_mean, z_std, vol], dim=-1)  # (B, C, 2D + 1)
        logits = self.net(feat) + self.channel_bias  # (B, C, 1)
        alpha = torch.sigmoid(logits).permute(0, 2, 1)  # (B, 1, C)
        with torch.no_grad():
            self.last_alpha = alpha.detach()
        # (B, 1, C): broadcasts against y_hat (B, pred_len, C).
        return alpha
