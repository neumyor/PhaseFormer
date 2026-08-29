"""One-checkpoint trend--cycle residual for a shared PhaseFormer.

The module does not ensemble complete forecasts. It factorizes one residual
forecast into a cycle-level trajectory and a zero-mean within-cycle shape:
NLinear predicts the trajectory, ICPT predicts mean-free cycle shape, and
rolling historical evidence only shrinks that orthogonal shape correction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.intercycle_patch import InterCyclePatchResidualHead
from src.models.phase_adapters import WeakPeriodResidualHead


class HierarchicalTrendCycleResidualHead(nn.Module):
    """Orthogonal trajectory plus reliability-shrunk cycle-shape correction."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int = 24,
        *,
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        beta_init: float = 0.25,
        use_rolling_confidence: bool = True,
        rolling_origins: int = 4,
        recency_decay: float = 0.5,
        risk_scale: float = 1.0,
        risk_std_weight: float = 1.0,
        confidence_floor: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        if seq_len <= 0 or pred_len <= 0 or period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period_len > 1")
        if rolling_origins < 2:
            raise ValueError("rolling_origins must be at least two")
        if recency_decay < 0 or risk_scale <= 0 or risk_std_weight < 0:
            raise ValueError("rolling-risk scales must be non-negative")
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")
        beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.num_future_cycles = math.ceil(pred_len / period_len)
        self.use_rolling_confidence = bool(use_rolling_confidence)
        self.rolling_origins = int(rolling_origins)
        self.recency_decay = float(recency_decay)
        self.risk_scale = float(risk_scale)
        self.risk_std_weight = float(risk_std_weight)
        self.confidence_floor = float(confidence_floor)
        self.eps = float(eps)

        # This first component has exactly the same constructor and RNG
        # consumption as A1. ICPT uses a forked RNG so enabling HPTC does not
        # shift the shared PhaseFormer trunk's paired-seed initialization.
        self.trajectory = WeakPeriodResidualHead(seq_len, pred_len)
        with torch.random.fork_rng(devices=[]):
            self.cycle_shape = InterCyclePatchResidualHead(
                seq_len=seq_len,
                pred_len=pred_len,
                period_len=period_len,
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                encoder_layers=1,
                decoder_layers=1,
                pe_type="none",
                use_last_cycle_anchor=True,
                use_attention=True,
                dropout=0.0,
                prediction_head="decoder",
                anchor_mode="last_cycle",
            )
        self.beta_logit = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(beta_init))))
        )

        self.last_beta = None
        self.last_risk = None
        self.last_risk_std = None
        self.last_confidence = None
        self.last_trajectory = None
        self.last_cycle_shape = None
        self.last_correction = None
        self.last_correction_cycle_mean_max = None

    def _cycle_view(self, series: torch.Tensor) -> torch.Tensor:
        """Return left-padded cycles as (B,C,K,P)."""
        if series.ndim != 3:
            raise ValueError("series must have shape (B,L,C)")
        B, L, C = series.shape
        cycles = math.ceil(L / self.period_len)
        pad = cycles * self.period_len - L
        if pad:
            series = F.pad(series.permute(0, 2, 1), (pad, 0), mode="replicate")
            series = series.permute(0, 2, 1)
        return series.view(B, cycles, self.period_len, C).permute(0, 3, 1, 2)

    @staticmethod
    def _mean_free(cycles: torch.Tensor) -> torch.Tensor:
        return cycles - cycles.mean(dim=-1, keepdim=True)

    def _shape_history(self, x: torch.Tensor) -> torch.Tensor:
        cycles = self._cycle_view(x)
        shape = self._mean_free(cycles)
        flat = shape.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(2))
        return flat[:, -self.seq_len :, :]

    def _future_cycle_shape(self, forecast: torch.Tensor) -> torch.Tensor:
        B, H, C = forecast.shape
        padded_horizon = self.num_future_cycles * self.period_len
        if H < padded_horizon:
            forecast = F.pad(
                forecast.permute(0, 2, 1),
                (0, padded_horizon - H),
                mode="replicate",
            ).permute(0, 2, 1)
        cycles = forecast.view(B, self.num_future_cycles, self.period_len, C)
        shape = cycles - cycles.mean(dim=2, keepdim=True)
        return shape.reshape(B, padded_horizon, C)[:, :H, :]

    def rolling_shape_evidence(self, history: torch.Tensor):
        """Horizon-matched shape extrapolation error from encoder history only."""
        cycles = self._cycle_view(history)
        shapes = self._mean_free(cycles)
        _, _, K, _ = cycles.shape
        origins = min(self.rolling_origins, max(2, K - 2))
        if K < origins + 3:
            raise ValueError("hierarchical trend-cycle head needs at least five cycles")
        ages = torch.arange(
            origins - 1, -1, -1, device=history.device, dtype=history.dtype
        )
        origin_weights = torch.softmax(-self.recency_decay * ages, dim=0)
        max_test_lead = max(1, K - origins - 1)
        risks, risk_stds = [], []
        for requested_lead in range(1, self.num_future_cycles + 1):
            # Leads beyond the encoder's backtestable range reuse the longest
            # observable lead; no target outside the history is accessed.
            lead = min(requested_lead, max_test_lead)
            origin_errors = []
            for target_index in range(K - origins, K):
                origin = target_index - lead
                current = shapes[:, :, origin, :]
                previous = shapes[:, :, origin - 1, :]
                prediction = current + lead * (current - previous)
                target = shapes[:, :, target_index, :]
                scale_start = max(0, origin + 1 - 4)
                scale = cycles[:, :, scale_start : origin + 1, :].std(
                    dim=(2, 3), unbiased=False
                ).clamp_min(self.eps)
                error = (prediction - target).abs().mean(dim=-1) / scale
                origin_errors.append(error.clamp(max=10.0))
            stacked = torch.stack(origin_errors, dim=-1)
            mean = (stacked * origin_weights).sum(dim=-1)
            variance = (
                (stacked - mean.unsqueeze(-1)).square() * origin_weights
            ).sum(dim=-1)
            risks.append(mean)
            risk_stds.append(variance.clamp_min(0.0).sqrt())
        return torch.stack(risks, dim=-1), torch.stack(risk_stds, dim=-1)

    def forward(self, x: torch.Tensor):
        trajectory = self.trajectory(x)
        shape_history = self._shape_history(x)
        cycle_shape = self._future_cycle_shape(self.cycle_shape(shape_history))
        trajectory_shape = self._future_cycle_shape(trajectory)
        correction = cycle_shape - trajectory_shape

        global_beta = torch.sigmoid(self.beta_logit)
        if self.use_rolling_confidence:
            risk, risk_std = self.rolling_shape_evidence(x)
            confidence = torch.exp(
                -self.risk_scale * (risk + self.risk_std_weight * risk_std)
            ).clamp(0.0, 1.0)
            confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * confidence
            beta_cycles = global_beta * confidence
        else:
            risk = risk_std = None
            confidence = x.new_ones((x.size(0), x.size(2), self.num_future_cycles))
            beta_cycles = global_beta * confidence
        beta = beta_cycles.repeat_interleave(self.period_len, dim=-1)
        beta = beta[..., : self.pred_len].permute(0, 2, 1)
        output = trajectory + beta * correction

        with torch.no_grad():
            correction_cycles = correction
            if self.pred_len < self.num_future_cycles * self.period_len:
                correction_cycles = F.pad(
                    correction.permute(0, 2, 1),
                    (0, self.num_future_cycles * self.period_len - self.pred_len),
                ).permute(0, 2, 1)
            correction_cycles = correction_cycles.view(
                x.size(0), self.num_future_cycles, self.period_len, x.size(2)
            )
            self.last_beta = beta_cycles.detach()
            self.last_risk = None if risk is None else risk.detach()
            self.last_risk_std = None if risk_std is None else risk_std.detach()
            self.last_confidence = confidence.detach()
            self.last_trajectory = trajectory.detach()
            self.last_cycle_shape = cycle_shape.detach()
            self.last_correction = correction.detach()
            self.last_correction_cycle_mean_max = float(
                correction_cycles.mean(dim=2).abs().max().detach()
            )
        return output
