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
import torch.nn.functional as F


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


class RollingTriAxisHistoryRouter(nn.Module):
    """Horizon-matched multi-origin router used by the TriAxis v2 experiment.

    For every requested future cycle ``q``, the router replays that exact lead
    time at several origins wholly contained in the encoder history.  It then
    estimates both mean pseudo-risk and origin-to-origin disagreement.  The
    ``prior`` and ``calibrated`` modes impose a monotonic low-risk preference;
    the bounded residual MLP can calibrate but cannot silently redefine the
    direction of the risk evidence at initialization.
    """

    MODES = {"rolling_features", "rolling_prior", "rolling_calibrated"}
    EXPERT_NAMES = TriAxisHistoryRouter.EXPERT_NAMES

    def __init__(
        self,
        pred_len: int,
        period_len: int,
        mode: str,
        hidden: int = 16,
        origins: int = 4,
        trajectory_window_cycles: int = 4,
        recency_decay: float = 0.5,
        risk_prior_strength: float = 1.0,
        correction_max: float = 0.5,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported rolling TriAxis mode: {mode}")
        if pred_len <= 0 or period_len <= 1:
            raise ValueError("pred_len must be positive and period_len > 1")
        if origins < 2 or trajectory_window_cycles < 1:
            raise ValueError("origins must be >=2 and trajectory window positive")
        if recency_decay < 0 or risk_prior_strength <= 0:
            raise ValueError("recency_decay must be non-negative and risk strength positive")
        if correction_max < 0 or temperature <= 0:
            raise ValueError("correction_max must be non-negative and temperature positive")
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.mode = mode
        self.origins = int(origins)
        self.trajectory_window_cycles = int(trajectory_window_cycles)
        self.recency_decay = float(recency_decay)
        self.correction_max = float(correction_max)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.num_future_cycles = math.ceil(self.pred_len / self.period_len)

        # mean risks (3), risk disagreement (3), structural evidence (3).
        self.router = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.phase_bias = nn.Parameter(torch.zeros(self.period_len, 3))
        self.future_cycle_bias = nn.Parameter(
            torch.zeros(self.num_future_cycles, 3)
        )
        if mode != "rolling_features":
            raw_strength = math.log(math.expm1(float(risk_prior_strength)))
            self.risk_strength_raw = nn.Parameter(torch.tensor(raw_strength))
            self.expert_prior_logits = nn.Parameter(torch.zeros(3))

        self.last_weights = None
        self.last_risks = None
        self.last_risk_std = None
        self.last_structural = None
        self.last_confidence = None
        self.last_entropy = None
        self.last_origin_count = None

    def _cycle_view(self, history):
        if history.ndim != 3:
            raise ValueError("history must have shape (B, L, C)")
        B, L, C = history.shape
        K = L // self.period_len
        minimum = self.num_future_cycles + self.origins + 2
        if K < minimum:
            raise ValueError(
                f"rolling TriAxis requires at least {minimum} complete cycles"
            )
        usable = history[:, -K * self.period_len :, :]
        return usable.view(B, K, self.period_len, C).permute(0, 3, 1, 2)

    def _trajectory_forecast(self, cycles, origin, lead):
        """Fit a recent chronological line and predict one target cycle."""
        start_cycle = max(0, origin + 1 - self.trajectory_window_cycles)
        history = cycles[:, :, start_cycle : origin + 1, :].flatten(2, 3)
        length = history.size(-1)
        t = torch.arange(length, device=history.device, dtype=history.dtype)
        centered_t = t - t.mean()
        centered_x = history - history.mean(dim=-1, keepdim=True)
        slope = (centered_x * centered_t).sum(dim=-1, keepdim=True) / (
            centered_t.square().sum().clamp_min(self.eps)
        )
        # target cycle begins (lead-1) full cycles after the next time point.
        target_t = length + (lead - 1) * self.period_len + torch.arange(
            self.period_len, device=history.device, dtype=history.dtype
        )
        return history.mean(dim=-1, keepdim=True) + slope * (
            target_t - t.mean()
        ).view(1, 1, -1)

    def rolling_history_evidence(self, history):
        """Return mean/std risks and structure as ``(B,C,Q,P,3)``."""
        cycles = self._cycle_view(history)
        B, C, K, P = cycles.shape
        risks_by_lead = []
        std_by_lead = []
        ages = torch.arange(
            self.origins - 1, -1, -1,
            device=history.device, dtype=history.dtype,
        )
        origin_weights = torch.softmax(-self.recency_decay * ages, dim=0)

        for lead in range(1, self.num_future_cycles + 1):
            origin_errors = []
            # Use the last R observable targets for every lead.  The associated
            # forecast origin moves back by `lead`, so no pseudo target leaks.
            for target_index in range(K - self.origins, K):
                origin = target_index - lead
                reference = cycles[:, :, : origin + 1, :]
                phase_pred = reference.mean(dim=2)
                trajectory_pred = self._trajectory_forecast(
                    cycles, origin, lead
                )
                last_cycle = cycles[:, :, origin, :]
                previous_cycle = cycles[:, :, origin - 1, :]
                cycle_pred = last_cycle + lead * (last_cycle - previous_cycle)
                prediction = torch.stack(
                    (phase_pred, trajectory_pred, cycle_pred), dim=-1
                )
                target = cycles[:, :, target_index, :]
                scale_start = max(0, origin + 1 - 4)
                scale = cycles[:, :, scale_start : origin + 1, :].std(
                    dim=(2, 3), unbiased=False
                ).clamp_min(self.eps)
                error = (prediction - target.unsqueeze(-1)).abs()
                error = (error / scale[:, :, None, None]).clamp(max=10.0)
                origin_errors.append(error)
            stacked = torch.stack(origin_errors, dim=2)  # (B,C,R,P,3)
            mean = (stacked * origin_weights.view(1, 1, -1, 1, 1)).sum(dim=2)
            variance = (
                (stacked - mean.unsqueeze(2)).square()
                * origin_weights.view(1, 1, -1, 1, 1)
            ).sum(dim=2)
            risks_by_lead.append(mean)
            std_by_lead.append(variance.clamp_min(0.0).sqrt())

        risks = torch.stack(risks_by_lead, dim=2)
        risk_std = torch.stack(std_by_lead, dim=2)

        reference = cycles[:, :, :-1, :]
        last = cycles[:, :, -1, :]
        previous = cycles[:, :, -2, :]
        local_scale = cycles[:, :, -4:, :].std(
            dim=(2, 3), unbiased=False
        ).clamp_min(self.eps)
        phase_signal = reference.mean(dim=2).var(dim=-1, unbiased=False)
        phase_noise = reference.var(dim=2, unbiased=False).mean(dim=-1)
        phase_reliability = phase_signal / (
            phase_signal + phase_noise + self.eps
        )
        level_drift = (
            (last.mean(dim=-1) - previous.mean(dim=-1)).abs() / local_scale
        )
        shape_innovation = (
            (last - previous).std(dim=-1, unbiased=False) / local_scale
        )
        structural_scalar = torch.stack(
            (phase_reliability, level_drift, shape_innovation), dim=-1
        )
        structural = structural_scalar[:, :, None, None, :].expand(
            B, C, self.num_future_cycles, P, 3
        )
        return risks, risk_std, structural

    def _history_weights(self, history):
        risks, risk_std, structural = self.rolling_history_evidence(history)
        features = torch.cat((risks, risk_std, structural), dim=-1)
        residual_logits = self.router(features)
        if self.mode == "rolling_features":
            logits = residual_logits
            confidence = torch.ones_like(risks[..., 0])
        else:
            centered = risks - risks.mean(dim=-1, keepdim=True)
            standardized = centered / risks.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(self.eps)
            # Disagreement across rolling origins shrinks the risk prior toward
            # the learned dataset-level expert prior.
            confidence = torch.exp(-risk_std.mean(dim=-1)).clamp(0.05, 1.0)
            risk_strength = F.softplus(self.risk_strength_raw)
            risk_logits = -risk_strength * confidence.unsqueeze(-1) * standardized
            bounded_correction = self.correction_max * torch.tanh(residual_logits)
            logits = (
                risk_logits
                + bounded_correction
                + self.expert_prior_logits.view(1, 1, 1, 1, 3)
            )
        logits = logits + self.phase_bias.view(1, 1, 1, self.period_len, 3)
        logits = logits + self.future_cycle_bias.view(
            1, 1, self.num_future_cycles, 1, 3
        )
        weights = torch.softmax(logits / self.temperature, dim=-1)
        weights = weights.reshape(
            history.size(0), history.size(2),
            self.num_future_cycles * self.period_len, 3,
        )[:, :, : self.pred_len, :]

        self.last_weights = weights.detach()
        self.last_risks = risks.detach()
        self.last_risk_std = risk_std.detach()
        self.last_structural = structural.detach()
        self.last_confidence = confidence.detach()
        self.last_entropy = (
            -(weights * weights.clamp_min(self.eps).log()).sum(dim=-1).mean().detach()
        )
        self.last_origin_count = self.origins
        return weights

    def forward(self, phase_hat, trajectory_hat, cycle_hat, history):
        shapes = {tuple(x.shape) for x in (phase_hat, trajectory_hat, cycle_hat)}
        if len(shapes) != 1 or phase_hat.ndim != 3:
            raise ValueError("all experts must share shape (B, H, C)")
        if phase_hat.size(1) != self.pred_len:
            raise ValueError("expert horizon does not match router pred_len")
        weights = self._history_weights(history)
        experts = torch.stack((phase_hat, trajectory_hat, cycle_hat), dim=-1)
        output = (experts * weights.permute(0, 2, 1, 3)).sum(dim=-1)
        return output, weights.permute(0, 2, 1, 3)


class SafeRegretTriAxisRouter(nn.Module):
    """A1-anchored router with an exact, learnable abstention path.

    The first action is always the complete A1 prediction.  The other actions
    are correction directions supplied by the phase, trajectory and cycle
    experts.  ``raw_global_accept`` starts at the closed boundary, so enabling
    this module and loading an A1 checkpoint is an exact functional identity.
    The boundary clamp has a usable inward gradient, allowing training to open
    the correction path only when the prediction objective supports it.
    """

    EXPERT_NAMES = RollingTriAxisHistoryRouter.EXPERT_NAMES

    # Reuse the audited rolling-origin implementation without constructing the
    # v2 router's unused projections or priors.
    _cycle_view = RollingTriAxisHistoryRouter._cycle_view
    _trajectory_forecast = RollingTriAxisHistoryRouter._trajectory_forecast
    rolling_history_evidence = RollingTriAxisHistoryRouter.rolling_history_evidence

    def __init__(
        self,
        pred_len: int,
        period_len: int,
        hidden: int = 16,
        origins: int = 4,
        trajectory_window_cycles: int = 4,
        recency_decay: float = 0.5,
        correction_clip: float = 2.0,
        max_accept: float = 1.0,
        use_horizon_prior: bool = False,
        horizon_prior_init: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        if pred_len <= 0 or period_len <= 1:
            raise ValueError("pred_len must be positive and period_len > 1")
        if origins < 2 or trajectory_window_cycles < 1:
            raise ValueError("origins must be >=2 and trajectory window positive")
        if correction_clip <= 0 or not 0 < max_accept <= 1:
            raise ValueError("correction_clip must be positive and max_accept in (0,1]")
        if horizon_prior_init <= 0:
            raise ValueError("horizon_prior_init must be positive")
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.num_future_cycles = math.ceil(self.pred_len / self.period_len)
        self.origins = int(origins)
        self.trajectory_window_cycles = int(trajectory_window_cycles)
        self.recency_decay = float(recency_decay)
        self.correction_clip = float(correction_clip)
        self.max_accept = float(max_accept)
        self.use_horizon_prior = bool(use_horizon_prior)
        self.eps = float(eps)

        # rolling mean risks (3), disagreement (3), structural evidence (3).
        self.router = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.raw_global_accept = nn.Parameter(torch.zeros(()))
        if self.use_horizon_prior:
            raw = math.log(math.expm1(float(horizon_prior_init)))
            self.horizon_prior_raw = nn.Parameter(torch.tensor(raw))

        self.last_weights = None
        self.last_cycle_action_logits = None
        self.last_cycle_accept = None
        self.last_selector = None
        self.last_global_accept = None
        self.last_risks = None
        self.last_risk_std = None
        self.last_structural = None

    def _action_logits(self, history):
        risks, risk_std, structural = self.rolling_history_evidence(history)
        # The decision unit is a future cycle, not an individual phase point.
        features = torch.cat((risks, risk_std, structural), dim=-1).mean(dim=3)
        raw = self.router(features)
        expert_logits = raw[..., :3]
        confidence_logit = raw[..., 3:4]
        if self.use_horizon_prior:
            q = torch.linspace(
                0.0, 1.0, self.num_future_cycles,
                device=history.device, dtype=history.dtype,
            ).view(1, 1, -1, 1)
            strength = F.softplus(self.horizon_prior_raw)
            signed = history.new_tensor((1.0, 0.0, -1.0)).view(1, 1, 1, 3)
            expert_logits = expert_logits + strength * q * signed
        # No-op has fixed score zero. Confidence shifts all expert actions
        # jointly, while their conditional softmax decides which correction.
        action_logits = torch.cat(
            (torch.zeros_like(confidence_logit), expert_logits + confidence_logit),
            dim=-1,
        )
        self.last_risks = risks.detach()
        self.last_risk_std = risk_std.detach()
        self.last_structural = structural.detach()
        return action_logits

    def forward(
        self, anchor_hat, phase_hat, trajectory_hat, cycle_hat, history
    ):
        shapes = {
            tuple(x.shape)
            for x in (anchor_hat, phase_hat, trajectory_hat, cycle_hat)
        }
        if len(shapes) != 1 or anchor_hat.ndim != 3:
            raise ValueError("anchor and experts must share shape (B,H,C)")
        if anchor_hat.size(1) != self.pred_len:
            raise ValueError("forecast horizon does not match router pred_len")

        cycle_logits = self._action_logits(history)  # (B,C,Q,4)
        soft_actions = torch.softmax(cycle_logits, dim=-1)
        selector = soft_actions[..., 1:] / soft_actions[..., 1:].sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        # Forward is a hard [0,max] abstention gate, while the straight-through
        # derivative remains usable even after an early batch moves the raw
        # scalar below zero.  This preserves exact A1 fallback without creating
        # a permanently dead boundary.
        hard_accept = self.raw_global_accept.detach().clamp(
            0.0, self.max_accept
        )
        global_accept = self.raw_global_accept + (
            hard_accept - self.raw_global_accept.detach()
        )
        cycle_accept = global_accept * (1.0 - soft_actions[..., :1])

        step = torch.arange(self.pred_len, device=history.device)
        cycle_index = torch.div(step, self.period_len, rounding_mode="floor")
        selector_step = selector[:, :, cycle_index, :].permute(0, 2, 1, 3)
        accept_step = cycle_accept[:, :, cycle_index, :].permute(0, 2, 1, 3)
        experts = torch.stack((phase_hat, trajectory_hat, cycle_hat), dim=-1)
        limit = self.correction_clip * history.std(
            dim=1, unbiased=False
        ).clamp_min(self.eps).unsqueeze(1).unsqueeze(-1)
        correction_directions = (experts - anchor_hat.unsqueeze(-1)).clamp(
            min=-limit, max=limit
        )
        correction = (selector_step * correction_directions).sum(dim=-1)
        output = anchor_hat + accept_step.squeeze(-1) * correction
        weights = torch.cat(
            (1.0 - accept_step, accept_step * selector_step), dim=-1
        )

        self.last_weights = weights
        self.last_cycle_action_logits = cycle_logits
        self.last_cycle_accept = cycle_accept
        self.last_selector = selector
        self.last_global_accept = global_accept
        return output, weights
