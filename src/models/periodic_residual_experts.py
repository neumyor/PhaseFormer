"""Periodic residual experts designed to complement PhaseFormer's phase path.

The modules in this file preserve an NLinear path and add only evidence that is
not already represented by a same-phase template: content-matched template
errors, sample-wise residual-cycle reliability, or adaptively routed lag banks.
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phase_adapters import ReliabilityCoupledResidualFusion


def _bounded_raw(initial: float, maximum: float) -> torch.Tensor:
    initial = min(max(float(initial), 0.0), float(maximum) - 1e-4)
    return torch.atanh(torch.tensor(initial / float(maximum)))


class PhaseErrorPeriodicMemoryHead(nn.Module):
    """NLinear plus content-addressed memory of phase-template errors.

    Complete historical cycles are centered by their cross-cycle phase
    template. The latest error cycle queries earlier error cycles, so the
    memory retrieves recurring *deviations* rather than duplicating the main
    phase forecast. A zero-initialized signed correction gate makes the initial
    output exactly identical to the NLinear warm start.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        memory_dim: int = 16,
        temperature: float = 0.1,
        recency_decay: float = 0.1,
        max_correction: float = 0.5,
    ):
        super().__init__()
        if seq_len < 2 * period_len:
            raise ValueError("phase-error memory requires at least two full cycles")
        if pred_len <= 0 or period_len <= 1 or memory_dim <= 0:
            raise ValueError("pred_len/memory_dim must be positive and period_len > 1")
        if temperature <= 0 or recency_decay < 0 or max_correction <= 0:
            raise ValueError("invalid memory temperature/decay/correction bound")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.temperature = float(temperature)
        self.recency_decay = float(recency_decay)
        self.max_correction = float(max_correction)

        self.linear = nn.Linear(self.seq_len, self.pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.cycle_projection = nn.Sequential(
            nn.Linear(self.period_len, int(memory_dim), bias=False),
            nn.LayerNorm(int(memory_dim)),
        )
        # tanh(0)=0: flag-on starts as the exact NLinear head while retaining a
        # non-zero derivative for the correction gate.
        self.correction_logits = nn.Parameter(torch.zeros(self.pred_len))

        self.last_attention = None
        self.last_attention_entropy = None
        self.last_correction_gate = None

    def _error_cycles(self, x):
        usable = (self.seq_len // self.period_len) * self.period_len
        cycles = (
            x[:, -usable:, :]
            .permute(0, 2, 1)
            .contiguous()
            .view(x.size(0), x.size(2), usable // self.period_len, self.period_len)
        )
        template = cycles.mean(dim=2, keepdim=True)
        return cycles - template

    def _retrieve_error_cycle(self, error_cycles):
        past = error_cycles[:, :, :-1, :]
        query = self.cycle_projection(error_cycles[:, :, -1, :])
        keys = self.cycle_projection(past)
        query = F.normalize(query, dim=-1, eps=1e-6)
        keys = F.normalize(keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bcd,bckd->bck", query, keys)
        lag = torch.arange(
            past.size(2), 0, -1, dtype=similarity.dtype, device=similarity.device
        )
        logits = similarity / self.temperature - self.recency_decay * lag
        attention = torch.softmax(logits, dim=-1)
        retrieved = torch.einsum("bck,bckp->bcp", attention, past)
        return retrieved, attention

    def forward(self, x):
        if x.size(1) != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x.size(1)}")
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        linear_delta = self.linear(centered).permute(0, 2, 1).contiguous()

        error_cycles = self._error_cycles(x)
        retrieved_cycle, attention = self._retrieve_error_cycle(error_cycles)
        future_phase = torch.arange(
            self.pred_len, device=x.device
        ).remainder(self.period_len)
        periodic_error = retrieved_cycle.index_select(-1, future_phase)
        periodic_error = periodic_error.permute(0, 2, 1).contiguous()
        gate = self.max_correction * torch.tanh(self.correction_logits)
        gate = gate.view(1, self.pred_len, 1)
        output = last.expand(-1, self.pred_len, -1) + linear_delta + gate * periodic_error

        with torch.no_grad():
            self.last_attention = attention.detach()
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
            self.last_attention_entropy = float(entropy.mean().detach())
            self.last_correction_gate = gate.detach()
        return output


class AdaptiveMultiPeriodResidualHead(nn.Module):
    """NLinear plus a sample/channel-adaptive bank of periodic lag copies."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        periods: Iterable[int] = (12, 24, 48, 96),
        routing_temperature: float = 0.15,
        recency_decay: float = 0.1,
        max_correction: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        normalized = tuple(sorted({int(period) for period in periods}))
        if not normalized or any(period <= 1 or period >= seq_len for period in normalized):
            raise ValueError("periods must be unique integers in [2, seq_len)")
        if pred_len <= 0 or routing_temperature <= 0:
            raise ValueError("pred_len and routing_temperature must be positive")
        if recency_decay < 0 or max_correction <= 0:
            raise ValueError("recency_decay must be non-negative and max_correction positive")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.periods = normalized
        self.routing_temperature = float(routing_temperature)
        self.recency_decay = float(recency_decay)
        self.max_correction = float(max_correction)
        self.eps = max(float(eps), 1e-8)

        self.linear = nn.Linear(self.seq_len, self.pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.period_logits = nn.Parameter(torch.zeros(len(self.periods)))
        self.correction_logits = nn.Parameter(torch.zeros(self.pred_len))

        for index, period in enumerate(self.periods):
            self.register_buffer(
                f"period_attention_{index}", self._build_attention(period)
            )

        self.last_period_reliability = None
        self.last_period_weights = None
        self.last_correction_gate = None

    def _build_attention(self, period):
        history = torch.arange(self.seq_len, dtype=torch.long)
        future = torch.arange(
            self.seq_len, self.seq_len + self.pred_len, dtype=torch.long
        )
        lag = future[:, None] - history[None, :]
        same_phase = lag.remainder(period).eq(0)
        logits = -self.recency_decay * lag.float() / float(period)
        logits = logits.masked_fill(~same_phase, -torch.inf)
        return torch.softmax(logits, dim=-1)

    def _lag_reliability(self, x, period):
        centered = x - x.mean(dim=1, keepdim=True)
        recent = centered[:, period:, :]
        delayed = centered[:, :-period, :]
        numerator = (recent * delayed).sum(dim=1)
        denominator = recent.square().sum(dim=1).sqrt() * delayed.square().sum(dim=1).sqrt()
        valid = denominator > self.eps
        correlation = numerator / denominator.clamp_min(self.eps)
        reliability = 0.5 * (correlation.clamp(-1.0, 1.0) + 1.0)
        return torch.where(valid, reliability, torch.zeros_like(reliability))

    def forward(self, x):
        if x.size(1) != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x.size(1)}")
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        linear_delta = self.linear(centered).permute(0, 2, 1).contiguous()

        periodic = []
        reliability = []
        for index, period in enumerate(self.periods):
            attention = getattr(self, f"period_attention_{index}").to(dtype=x.dtype)
            periodic.append(torch.einsum("hl,bcl->bhc", attention, centered))
            reliability.append(self._lag_reliability(x, period))
        periodic = torch.stack(periodic, dim=-1)  # (B,H,C,Q)
        reliability = torch.stack(reliability, dim=-1)  # (B,C,Q)
        routing_logits = (
            reliability / self.routing_temperature
            + self.period_logits.view(1, 1, -1)
        )
        weights = torch.softmax(routing_logits, dim=-1)
        periodic_delta = (periodic * weights.unsqueeze(1)).sum(dim=-1)
        gate = self.max_correction * torch.tanh(self.correction_logits)
        gate = gate.view(1, self.pred_len, 1)
        output = last.expand(-1, self.pred_len, -1) + linear_delta + gate * periodic_delta

        with torch.no_grad():
            self.last_period_reliability = reliability.detach()
            self.last_period_weights = weights.detach()
            self.last_correction_gate = gate.detach()
        return output


class DualReliabilityPeriodicFusion(nn.Module):
    """RCRF outside, residual-cycle reliability inside.

    The outer RCRF gate uses raw phase reliability to mix PhaseFormer with a
    residual candidate. Inside that residual candidate, a second reliability
    score controls NLinear versus the periodic copy for each sample, channel,
    and forecast step.
    """

    def __init__(
        self,
        pred_len: int,
        alpha_init: float = 0.5,
        phase_sensitivity_init: float = 2.0,
        phase_s_max: float = 4.0,
        periodic_init: float = 0.1,
        periodic_sensitivity_init: float = 2.0,
        periodic_s_max: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if pred_len <= 0 or periodic_s_max <= 0:
            raise ValueError("pred_len and periodic_s_max must be positive")
        self.pred_len = int(pred_len)
        self.periodic_s_max = float(periodic_s_max)
        self.eps = max(float(eps), 1e-8)
        periodic_init = min(max(float(periodic_init), 1e-4), 1.0 - 1e-4)
        periodic_bias = float(torch.logit(torch.tensor(periodic_init)))
        self.periodic_bias = nn.Parameter(torch.full((self.pred_len,), periodic_bias))
        self.periodic_s_raw = nn.Parameter(
            _bounded_raw(periodic_sensitivity_init, self.periodic_s_max)
        )
        self.phase_fusion = ReliabilityCoupledResidualFusion(
            alpha_init=alpha_init,
            sensitivity_init=phase_sensitivity_init,
            s_max=phase_s_max,
            eps=self.eps,
        )

        self.last_periodic_reliability = None
        self.last_periodic_gate = None
        self.last_phase_reliability = None
        self.last_phase_gate = None

    @property
    def periodic_sensitivity(self):
        return float(self.periodic_s_max * torch.tanh(self.periodic_s_raw).detach())

    def _periodic_reliability(self, phase_series):
        template = phase_series.mean(dim=-1, keepdim=True)
        error_cycles = (phase_series - template).permute(0, 1, 3, 2)
        if error_cycles.size(2) < 2:
            return phase_series.new_zeros(phase_series.shape[:2])
        recent = error_cycles[:, :, 1:, :]
        previous = error_cycles[:, :, :-1, :]
        numerator = (recent * previous).sum(dim=-1)
        denominator = recent.square().sum(dim=-1).sqrt() * previous.square().sum(dim=-1).sqrt()
        valid = denominator > self.eps
        correlation = numerator / denominator.clamp_min(self.eps)
        pair_reliability = 0.5 * (correlation.clamp(-1.0, 1.0) + 1.0)
        pair_reliability = torch.where(
            valid, pair_reliability, torch.zeros_like(pair_reliability)
        )
        return pair_reliability.mean(dim=-1)

    def forward(self, y_phase, y_linear, y_periodic, phase_series):
        if y_phase.shape != y_linear.shape or y_phase.shape != y_periodic.shape:
            raise ValueError("phase, linear and periodic forecasts must have equal shapes")
        reliability = self._periodic_reliability(phase_series)
        sensitivity = self.periodic_s_max * torch.tanh(self.periodic_s_raw)
        logits = (
            self.periodic_bias.view(1, self.pred_len, 1)
            + sensitivity * (reliability.unsqueeze(1) - 0.5)
        )
        periodic_gate = torch.sigmoid(logits)
        residual = (1.0 - periodic_gate) * y_linear + periodic_gate * y_periodic
        output, phase_gate = self.phase_fusion(
            y_phase, residual, phase_series
        )

        with torch.no_grad():
            self.last_periodic_reliability = reliability.detach()
            self.last_periodic_gate = periodic_gate.detach()
            self.last_phase_reliability = self.phase_fusion.last_r
            self.last_phase_gate = phase_gate.detach()
        return output, phase_gate
