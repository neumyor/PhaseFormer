"""Single-checkpoint fusion strategies for phase, cycle, and trajectory.

Every strategy shares exactly one PhaseFormer phase forecast, one NLinear
trajectory head, and one no-PE ICPT cycle head.  The paper candidates compose
identifiable forecast components; the two complete-forecast mixtures are kept
only as negative controls.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.intercycle_patch import InterCyclePatchResidualHead
from src.models.phase_adapters import WeakPeriodResidualHead


class PhaseCycleFusionComposer(nn.Module):
    """Fuse PhaseFormer, NLinear, and ICPT inside one end-to-end model."""

    STRATEGIES = (
        "component_scalar",
        "component_cycle",
        "monotonic_evidence",
        "mlp_evidence",
        "phase_modulation",
        "uniform_control",
        "softmax_control",
    )

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int = 24,
        *,
        strategy: str = "component_cycle",
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        level_gate_init: float = 0.10,
        shape_gate_init: float = 0.10,
        deformation_gate_init: float = 0.05,
        masked_origins: int = 2,
        risk_scale: float = 1.0,
        risk_std_weight: float = 0.5,
        confidence_floor: float = 0.05,
        evidence_strength_init: float = 1.0,
        mlp_hidden: int = 16,
        mlp_correction_max: float = 2.0,
        modulation_temperature: float = 0.10,
        amplitude_min: float = 0.5,
        amplitude_max: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unsupported phase-cycle fusion strategy: {strategy}")
        if seq_len <= 0 or pred_len <= 0 or period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period_len > 1")
        if seq_len % period_len or pred_len % period_len:
            raise ValueError("fusion requires lengths divisible by period_len")
        if masked_origins < 1 or masked_origins >= seq_len // period_len:
            raise ValueError("masked_origins must leave at least one observed cycle")
        if risk_scale <= 0 or risk_std_weight < 0:
            raise ValueError("invalid evidence risk parameters")
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")
        if mlp_hidden < 1 or mlp_correction_max <= 0:
            raise ValueError("invalid MLP gate parameters")
        if modulation_temperature <= 0:
            raise ValueError("modulation_temperature must be positive")
        if not 0 < amplitude_min <= amplitude_max:
            raise ValueError("invalid amplitude range")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.num_history_cycles = seq_len // period_len
        self.num_future_cycles = pred_len // period_len
        self.strategy = str(strategy)
        self.masked_origins = int(masked_origins)
        self.risk_scale = float(risk_scale)
        self.risk_std_weight = float(risk_std_weight)
        self.confidence_floor = float(confidence_floor)
        self.mlp_correction_max = float(mlp_correction_max)
        self.modulation_temperature = float(modulation_temperature)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.eps = float(eps)

        # NLinear is constructed first and consumes exactly the same RNG as A1.
        # All additional random modules are isolated so the shared PhaseFormer
        # trunk remains paired-initialized with A1 for every strategy.
        self.trajectory = WeakPeriodResidualHead(seq_len, pred_len)
        with torch.random.fork_rng(devices=[]):
            self.cycle = InterCyclePatchResidualHead(
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
            if strategy == "mlp_evidence":
                self.gate_mlp = nn.Sequential(
                    nn.Linear(7, mlp_hidden),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, 2 * self.num_future_cycles),
                )
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)

        if strategy in ("component_scalar", "component_cycle"):
            size = 1 if strategy == "component_scalar" else self.num_future_cycles
            self.level_logits = nn.Parameter(
                torch.full((size,), self._logit(level_gate_init))
            )
            self.shape_logits = nn.Parameter(
                torch.full((size,), self._logit(shape_gate_init))
            )
        elif strategy in ("monotonic_evidence", "mlp_evidence"):
            self.level_logits = nn.Parameter(torch.full(
                (self.num_future_cycles,), self._logit(level_gate_init)
            ))
            self.shape_logits = nn.Parameter(torch.full(
                (self.num_future_cycles,), self._logit(shape_gate_init)
            ))
            if strategy == "monotonic_evidence":
                strength = self._inverse_softplus(evidence_strength_init)
                self.shape_unreliability_raw = nn.Parameter(torch.tensor(strength))
                self.shape_confidence_raw = nn.Parameter(torch.tensor(strength))
                self.level_confidence_raw = nn.Parameter(torch.tensor(strength))
                self.level_drift_raw = nn.Parameter(torch.tensor(strength))
        elif strategy == "phase_modulation":
            self.modulation_logits = nn.Parameter(torch.full(
                (self.num_future_cycles,), self._logit(shape_gate_init)
            ))
            self.deformation_logits = nn.Parameter(torch.full(
                (self.num_future_cycles,), self._logit(deformation_gate_init)
            ))
        elif strategy == "softmax_control":
            self.branch_logits = nn.Parameter(
                torch.zeros(self.num_future_cycles, 3)
            )

        self.last_phase = None
        self.last_trajectory = None
        self.last_cycle = None
        self.last_output = None
        self.last_level_gate = None
        self.last_shape_gate = None
        self.last_branch_weights = None
        self.last_phase_reliability = None
        self.last_drift = None
        self.last_shape_confidence = None
        self.last_level_confidence = None
        self.last_shape_risk = None
        self.last_level_risk = None
        self.last_expected_shift = None
        self.last_amplitude = None
        self.last_shape_cycle_mean_max = None
        self.last_horizon_mean_error_max = None

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return float(torch.logit(torch.tensor(value)))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        value = max(float(value), 1e-4)
        return float(torch.log(torch.expm1(torch.tensor(value))))

    def _cycles(self, series: torch.Tensor) -> torch.Tensor:
        if series.ndim != 3 or series.size(1) != self.pred_len:
            raise ValueError(
                f"forecast must have shape (B,{self.pred_len},C)"
            )
        return series.reshape(
            series.size(0), self.num_future_cycles,
            self.period_len, series.size(2),
        )

    def _history_cycles(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.size(1) != self.seq_len:
            raise ValueError(f"history must have shape (B,{self.seq_len},C)")
        return history.reshape(
            history.size(0), self.num_history_cycles,
            self.period_len, history.size(2),
        )

    @staticmethod
    def _decompose(cycles: torch.Tensor):
        cycle_mean = cycles.mean(dim=2, keepdim=True)
        global_mean = cycle_mean.mean(dim=1, keepdim=True)
        level = cycle_mean - global_mean
        shape = cycles - cycle_mean
        return global_mean, level, shape

    def _expand_gate(self, logits: torch.Tensor, batch: int, channels: int):
        gate = torch.sigmoid(logits).view(1, 1, -1)
        if gate.size(-1) == 1:
            gate = gate.expand(-1, -1, self.num_future_cycles)
        return gate.expand(batch, channels, -1)

    @staticmethod
    def _gate_to_cycles(gate: torch.Tensor):
        return gate.permute(0, 2, 1).unsqueeze(2)

    def _phase_reliability(self, phase_series: torch.Tensor):
        if phase_series.ndim != 4:
            raise ValueError("phase_series must have shape (B,C,L,K)")
        phase_signal = phase_series.mean(dim=-1).var(dim=2, unbiased=False)
        phase_noise = phase_series.var(dim=-1, unbiased=False).mean(dim=2)
        return phase_signal / (phase_signal + phase_noise + self.eps)

    def _drift(self, history: torch.Tensor):
        cycles = self._history_cycles(history)
        scale = history.std(dim=1, unbiased=False).clamp_min(self.eps)
        raw = (cycles[:, -1].mean(dim=1) - cycles[:, -2].mean(dim=1)).abs()
        normalized = raw / scale
        return normalized / (1.0 + normalized)

    def _masked_contexts(self, history: torch.Tensor):
        cycles = self._history_cycles(history)
        contexts, targets = [], []
        first = cycles[:, :1]
        start = self.num_history_cycles - self.masked_origins
        for target_index in range(start, self.num_history_cycles):
            observed = cycles[:, :target_index]
            padding = first.expand(
                -1, self.num_history_cycles - target_index, -1, -1
            )
            contexts.append(torch.cat((padding, observed), dim=1).reshape_as(history))
            targets.append(cycles[:, target_index])
        return torch.cat(contexts, dim=0), torch.cat(targets, dim=0)

    def _masked_confidence(self, history: torch.Tensor):
        batch, _, channels = history.shape
        contexts, targets = self._masked_contexts(history)
        with torch.no_grad():
            trajectory = self.trajectory(contexts)[:, : self.period_len]
            cycle = self.cycle(contexts)[:, : self.period_len]
            scale = contexts.std(dim=1, unbiased=False).clamp_min(self.eps)

            target_level = targets.mean(dim=1)
            trajectory_level = trajectory.mean(dim=1)
            cycle_level = cycle.mean(dim=1)
            level_cycle_error = (cycle_level - target_level).abs() / scale
            level_trajectory_error = (
                trajectory_level - target_level
            ).abs() / scale

            target_shape = targets - target_level.unsqueeze(1)
            trajectory_shape = trajectory - trajectory_level.unsqueeze(1)
            cycle_shape = cycle - cycle_level.unsqueeze(1)
            shape_cycle_error = (
                cycle_shape - target_shape
            ).abs().mean(dim=1) / scale
            shape_trajectory_error = (
                trajectory_shape - target_shape
            ).abs().mean(dim=1) / scale

            level_risk = torch.relu(torch.log(
                (level_cycle_error + self.eps)
                / (level_trajectory_error + self.eps)
            )).view(self.masked_origins, batch, channels)
            shape_risk = torch.relu(torch.log(
                (shape_cycle_error + self.eps)
                / (shape_trajectory_error + self.eps)
            )).view(self.masked_origins, batch, channels)

            level_mean = level_risk.mean(dim=0)
            shape_mean = shape_risk.mean(dim=0)
            level_std = level_risk.std(dim=0, unbiased=False)
            shape_std = shape_risk.std(dim=0, unbiased=False)
            level_score = level_mean + self.risk_std_weight * level_std
            shape_score = shape_mean + self.risk_std_weight * shape_std
            level_confidence = torch.exp(-self.risk_scale * level_score)
            shape_confidence = torch.exp(-self.risk_scale * shape_score)
            level_confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * level_confidence
            shape_confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * shape_confidence
        return shape_confidence, level_confidence, shape_mean, level_mean

    def _component_output(
        self,
        phase_cycles: torch.Tensor,
        trajectory_cycles: torch.Tensor,
        cycle_cycles: torch.Tensor,
        level_gate: torch.Tensor,
        shape_gate: torch.Tensor,
    ):
        trajectory_mean, trajectory_level, _ = self._decompose(trajectory_cycles)
        _, cycle_level, cycle_shape = self._decompose(cycle_cycles)
        _, _, phase_shape = self._decompose(phase_cycles)
        level_gate = self._gate_to_cycles(level_gate)
        shape_gate = self._gate_to_cycles(shape_gate)
        level = trajectory_level + level_gate * (cycle_level - trajectory_level)
        # Cycle-dependent gates can change the global mean; restore the
        # identifiable zero-mean level subspace after gating.
        level = level - level.mean(dim=1, keepdim=True)
        shape = phase_shape + shape_gate * (cycle_shape - phase_shape)
        output = trajectory_mean + level + shape
        return output, shape

    def _monotonic_gates(
        self, batch, channels, phase_reliability, drift,
        shape_confidence, level_confidence,
    ):
        shape_base = self.shape_logits.view(1, 1, -1)
        level_base = self.level_logits.view(1, 1, -1)
        shape_logits = (
            shape_base
            + F.softplus(self.shape_unreliability_raw)
            * (1.0 - phase_reliability).unsqueeze(-1)
            + F.softplus(self.shape_confidence_raw)
            * (shape_confidence - 0.5).unsqueeze(-1)
        )
        level_logits = (
            level_base
            + F.softplus(self.level_confidence_raw)
            * (level_confidence - 0.5).unsqueeze(-1)
            - F.softplus(self.level_drift_raw) * drift.unsqueeze(-1)
        )
        return (
            torch.sigmoid(level_logits).expand(batch, channels, -1),
            torch.sigmoid(shape_logits).expand(batch, channels, -1),
        )

    def _mlp_gates(
        self, phase, trajectory, cycle, history, phase_reliability, drift,
        shape_confidence, level_confidence,
    ):
        scale = history.std(dim=1, unbiased=False).clamp_min(self.eps)

        def disagreement(left, right):
            value = (left - right).abs().mean(dim=1) / scale
            return torch.log1p(value).clamp(max=3.0) / 3.0

        features = torch.stack((
            1.0 - phase_reliability,
            drift,
            shape_confidence,
            level_confidence,
            disagreement(phase, trajectory),
            disagreement(phase, cycle),
            disagreement(trajectory, cycle),
        ), dim=-1).detach()
        correction = self.mlp_correction_max * torch.tanh(self.gate_mlp(features))
        correction = correction.view(
            history.size(0), history.size(2), 2, self.num_future_cycles
        )
        level = torch.sigmoid(self.level_logits.view(1, 1, -1) + correction[:, :, 0])
        shape = torch.sigmoid(self.shape_logits.view(1, 1, -1) + correction[:, :, 1])
        return level, shape

    def _phase_modulation_output(
        self,
        phase_cycles: torch.Tensor,
        trajectory_cycles: torch.Tensor,
        cycle_cycles: torch.Tensor,
    ):
        trajectory_cycle_mean = trajectory_cycles.mean(dim=2, keepdim=True)
        phase_shape = phase_cycles - phase_cycles.mean(dim=2, keepdim=True)
        cycle_shape = cycle_cycles - cycle_cycles.mean(dim=2, keepdim=True)
        rolled = torch.stack(
            [
                torch.roll(phase_shape, shifts=shift, dims=2)
                for shift in range(self.period_len)
            ],
            dim=-1,
        )
        target = cycle_shape.unsqueeze(-1)
        numerator = (rolled * target).mean(dim=2)
        denominator = (
            rolled.square().mean(dim=2)
            * cycle_shape.square().mean(dim=2).unsqueeze(-1)
        ).clamp_min(self.eps).sqrt()
        alignment = numerator / denominator
        shift_weights = torch.softmax(
            alignment / self.modulation_temperature, dim=-1
        )
        aligned = (rolled * shift_weights.unsqueeze(2)).sum(dim=-1)
        amplitude = (
            (aligned * cycle_shape).sum(dim=2, keepdim=True)
            / aligned.square().sum(dim=2, keepdim=True).clamp_min(self.eps)
        ).clamp(self.amplitude_min, self.amplitude_max)
        modulated = amplitude * aligned
        deformation = cycle_shape - modulated

        batch, _, _, channels = phase_cycles.shape
        modulation_gate = self._gate_to_cycles(self._expand_gate(
            self.modulation_logits, batch, channels
        ))
        deformation_gate = self._gate_to_cycles(self._expand_gate(
            self.deformation_logits, batch, channels
        ))
        shape = (
            phase_shape
            + modulation_gate * (modulated - phase_shape)
            + deformation_gate * deformation
        )
        shifts = torch.arange(
            self.period_len, device=phase_cycles.device,
            dtype=phase_cycles.dtype,
        )
        signed_shifts = torch.where(
            shifts <= self.period_len // 2, shifts, shifts - self.period_len
        )
        expected_shift = (shift_weights * signed_shifts).sum(dim=-1)
        self.last_expected_shift = expected_shift.detach()
        self.last_amplitude = amplitude.detach()
        self.last_shape_gate = self._expand_gate(
            self.modulation_logits, batch, channels
        ).detach()
        self.last_level_gate = None
        return trajectory_cycle_mean + shape, shape

    def _control_output(self, phase_cycles, trajectory_cycles, cycle_cycles):
        forecasts = torch.stack(
            (phase_cycles, trajectory_cycles, cycle_cycles), dim=-1
        )
        if self.strategy == "uniform_control":
            weights = forecasts.new_full(
                (self.num_future_cycles, 3), 1.0 / 3.0
            )
        else:
            weights = torch.softmax(self.branch_logits, dim=-1)
        self.last_branch_weights = weights.detach()
        return (forecasts * weights.view(
            1, self.num_future_cycles, 1, 1, 3
        )).sum(dim=-1)

    def forward(
        self,
        phase: torch.Tensor,
        history: torch.Tensor,
        phase_series: torch.Tensor,
    ):
        if phase.shape != (history.size(0), self.pred_len, history.size(2)):
            raise ValueError("phase/history shapes are incompatible")
        trajectory = self.trajectory(history)
        cycle = self.cycle(history)
        phase_cycles = self._cycles(phase)
        trajectory_cycles = self._cycles(trajectory)
        cycle_cycles = self._cycles(cycle)
        phase_reliability = self._phase_reliability(phase_series)
        drift = self._drift(history)

        shape_confidence = level_confidence = history.new_ones(
            history.size(0), history.size(2)
        )
        shape_risk = level_risk = history.new_zeros(
            history.size(0), history.size(2)
        )
        shape = None
        if self.strategy in ("component_scalar", "component_cycle"):
            level_gate = self._expand_gate(
                self.level_logits, history.size(0), history.size(2)
            )
            shape_gate = self._expand_gate(
                self.shape_logits, history.size(0), history.size(2)
            )
            output_cycles, shape = self._component_output(
                phase_cycles, trajectory_cycles, cycle_cycles,
                level_gate, shape_gate,
            )
            self.last_level_gate = level_gate.detach()
            self.last_shape_gate = shape_gate.detach()
        elif self.strategy in ("monotonic_evidence", "mlp_evidence"):
            (
                shape_confidence, level_confidence,
                shape_risk, level_risk,
            ) = self._masked_confidence(history)
            if self.strategy == "monotonic_evidence":
                level_gate, shape_gate = self._monotonic_gates(
                    history.size(0), history.size(2), phase_reliability,
                    drift, shape_confidence, level_confidence,
                )
            else:
                level_gate, shape_gate = self._mlp_gates(
                    phase, trajectory, cycle, history, phase_reliability,
                    drift, shape_confidence, level_confidence,
                )
            output_cycles, shape = self._component_output(
                phase_cycles, trajectory_cycles, cycle_cycles,
                level_gate, shape_gate,
            )
            self.last_level_gate = level_gate.detach()
            self.last_shape_gate = shape_gate.detach()
        elif self.strategy == "phase_modulation":
            output_cycles, shape = self._phase_modulation_output(
                phase_cycles, trajectory_cycles, cycle_cycles
            )
        else:
            output_cycles = self._control_output(
                phase_cycles, trajectory_cycles, cycle_cycles
            )

        output = output_cycles.reshape_as(phase)
        with torch.no_grad():
            self.last_phase = phase.detach()
            self.last_trajectory = trajectory.detach()
            self.last_cycle = cycle.detach()
            self.last_output = output.detach()
            self.last_phase_reliability = phase_reliability.detach()
            self.last_drift = drift.detach()
            self.last_shape_confidence = shape_confidence.detach()
            self.last_level_confidence = level_confidence.detach()
            self.last_shape_risk = shape_risk.detach()
            self.last_level_risk = level_risk.detach()
            if shape is not None:
                self.last_shape_cycle_mean_max = float(
                    shape.mean(dim=2).abs().max().detach()
                )
                self.last_horizon_mean_error_max = float((
                    output.mean(dim=1) - trajectory.mean(dim=1)
                ).abs().max().detach())
            else:
                self.last_shape_cycle_mean_max = None
                self.last_horizon_mean_error_max = None
        return output
