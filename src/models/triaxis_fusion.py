"""History-only routing for phase, trajectory, and inter-cycle experts.

The router deliberately does not inspect future values, decoder marks, or the
experts' future predictions.  It backtests three inexpensive analogues on the
last observed cycle and combines their risks with structural history features.
The resulting weights are factorized by future-cycle index and phase slot.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TriAxisHistoryRouter(nn.Module):
    """Route three forecasting experts using evidence available at inference.

    Expert order is fixed to ``phase, trajectory, cycle``.  ``structural``
    uses only three descriptive history features, while ``self_validating``
    additionally uses pseudo-forecast errors computed entirely inside the
    encoder window.  The final projection and position biases are initialized
    to zero, so learned modes start from the controlled uniform mixture.
    """

    MODES = {"uniform", "structural", "self_validating"}
    EXPERT_NAMES = ("phase", "trajectory", "cycle")

    def __init__(
        self,
        pred_len: int,
        period_len: int,
        mode: str = "self_validating",
        hidden: int = 16,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported TriAxis router mode: {mode}")
        if pred_len <= 0 or period_len <= 1:
            raise ValueError("pred_len must be positive and period_len > 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.mode = mode
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.num_future_cycles = math.ceil(self.pred_len / self.period_len)

        if mode != "uniform":
            input_dim = 3 if mode == "structural" else 6
            self.router = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
            self.phase_bias = nn.Parameter(torch.zeros(self.period_len, 3))
            self.future_cycle_bias = nn.Parameter(
                torch.zeros(self.num_future_cycles, 3)
            )

        self.last_weights = None
        self.last_risks = None
        self.last_structural = None
        self.last_entropy = None

    def _cycle_view(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape (B, L, C)")
        B, L, C = history.shape
        num_cycles = L // self.period_len
        if num_cycles < 3:
            raise ValueError(
                "TriAxis history self-validation requires at least three cycles"
            )
        usable = history[:, -num_cycles * self.period_len :, :]
        return usable.view(B, num_cycles, self.period_len, C).permute(0, 3, 1, 2)

    def history_evidence(self, history: torch.Tensor):
        """Return pseudo risks and structural features as ``(B,C,P,3)``."""
        cycles = self._cycle_view(history)
        target = cycles[:, :, -1, :]
        previous = cycles[:, :, -2, :]
        previous2 = cycles[:, :, -3, :]
        reference = cycles[:, :, :-1, :]

        # Phase analogue: the mean value previously observed at each slot.
        phase_pred = reference.mean(dim=2)

        # Trajectory analogue: fit a line to the immediately preceding cycle
        # in chronological order and extrapolate the next period.
        p = torch.arange(
            self.period_len, device=history.device, dtype=history.dtype
        )
        p_centered = p - p.mean()
        denom = p_centered.square().sum().clamp_min(self.eps)
        prev_mean = previous.mean(dim=-1, keepdim=True)
        slope = (
            (previous - prev_mean) * p_centered.view(1, 1, -1)
        ).sum(dim=-1, keepdim=True) / denom
        future_p = p + self.period_len
        trajectory_pred = prev_mean + slope * (
            future_p - p.mean()
        ).view(1, 1, -1)

        # Cycle analogue: continue the most recent inter-cycle increment.
        cycle_pred = 2.0 * previous - previous2

        local_scale = cycles[:, :, -3:, :].std(
            dim=(2, 3), unbiased=False, keepdim=False
        ).unsqueeze(-1).clamp_min(self.eps)
        pseudo = torch.stack((phase_pred, trajectory_pred, cycle_pred), dim=-1)
        risks = ((pseudo - target.unsqueeze(-1)).abs() / local_scale.unsqueeze(-1))
        risks = risks.clamp(max=10.0)

        # Stable same-phase structure, recent level drift, and shape innovation.
        phase_signal = reference.mean(dim=2).var(dim=-1, unbiased=False)
        phase_noise = reference.var(dim=2, unbiased=False).mean(dim=-1)
        phase_reliability = phase_signal / (
            phase_signal + phase_noise + self.eps
        )
        level_drift = (
            (target.mean(dim=-1) - previous.mean(dim=-1)).abs()
            / local_scale.squeeze(-1)
        )
        shape_innovation = (
            (target - previous).std(dim=-1, unbiased=False)
            / local_scale.squeeze(-1)
        )
        structural_scalar = torch.stack(
            (phase_reliability, level_drift, shape_innovation), dim=-1
        )
        structural = structural_scalar.unsqueeze(2).expand(
            -1, -1, self.period_len, -1
        )
        return risks, structural

    def _history_weights(self, history: torch.Tensor) -> torch.Tensor:
        risks, structural = self.history_evidence(history)
        B, C, P, _ = risks.shape
        if self.mode == "uniform":
            weights = history.new_full((B, C, self.pred_len, 3), 1.0 / 3.0)
        else:
            features = structural if self.mode == "structural" else torch.cat(
                (risks, structural), dim=-1
            )
            phase_logits = self.router(features) + self.phase_bias.view(1, 1, P, 3)
            phase_index = torch.arange(self.pred_len, device=history.device) % P
            cycle_index = torch.div(
                torch.arange(self.pred_len, device=history.device),
                P,
                rounding_mode="floor",
            )
            logits = phase_logits[:, :, phase_index, :]
            logits = logits + self.future_cycle_bias[cycle_index].view(
                1, 1, self.pred_len, 3
            )
            weights = torch.softmax(logits / self.temperature, dim=-1)

        self.last_weights = weights.detach()
        self.last_risks = risks.detach()
        self.last_structural = structural.detach()
        self.last_entropy = (
            -(weights * weights.clamp_min(self.eps).log()).sum(dim=-1).mean().detach()
        )
        return weights

    def forward(
        self,
        phase_hat: torch.Tensor,
        trajectory_hat: torch.Tensor,
        cycle_hat: torch.Tensor,
        history: torch.Tensor,
    ):
        shapes = {tuple(x.shape) for x in (phase_hat, trajectory_hat, cycle_hat)}
        if len(shapes) != 1 or phase_hat.ndim != 3:
            raise ValueError("all experts must share shape (B, H, C)")
        if phase_hat.size(1) != self.pred_len:
            raise ValueError("expert horizon does not match router pred_len")
        weights = self._history_weights(history)  # (B,C,H,3)
        experts = torch.stack(
            (phase_hat, trajectory_hat, cycle_hat), dim=-1
        )  # (B,H,C,3)
        output = (experts * weights.permute(0, 2, 1, 3)).sum(dim=-1)
        return output, weights.permute(0, 2, 1, 3)
