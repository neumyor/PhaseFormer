"""A2-anchored phase--cycle fusion for one end-to-end PhaseFormer.

The first PCTF fusion study replaced useful subspaces of the incumbent
forecast, so none of its structured candidates could recover A2 for any gate
setting.  This module instead treats the complete A2 forecast as a trust-region
anchor and admits only two bounded ICPT innovations:

* relative cycle-level change with respect to the A2 trajectory branch; and
* within-cycle shape change with respect to the PhaseFormer branch.

All correction coefficients are tanh bounded and exactly zero initialized.
Consequently every candidate is functionally identical to A2 before training,
while the ICPT head is still trained by component-level auxiliary losses in the
owning PhaseFormer.  The model remains one jointly trained checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.intercycle_patch import InterCyclePatchResidualHead


class AnchoredPhaseCycleFusionComposer(nn.Module):
    """Add bounded, identifiable ICPT corrections to a complete A2 anchor."""

    STRATEGIES = (
        "component_scalar",
        "component_cycle",
        "shape_only",
        "level_only",
        "monotonic_evidence",
        "mlp_evidence",
        "phase_modulation",
    )

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        cycle_period_len: int = 24,
        *,
        strategy: str = "component_cycle",
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        correction_max: float = 0.25,
        deformation_max: float = 0.10,
        masked_origins: int = 3,
        risk_scale: float = 1.0,
        risk_std_weight: float = 0.5,
        confidence_floor: float = 0.05,
        risk_clip: float = 6.0,
        mlp_hidden: int = 16,
        modulation_temperature: float = 0.25,
        amplitude_min: float = 0.5,
        amplitude_max: float = 2.0,
        detach_references: bool = False,
        level_mode: str = "horizon_centered",
        global_level_max: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unsupported anchored fusion strategy: {strategy}")
        if seq_len <= 0 or pred_len <= 0 or cycle_period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period > 1")
        if pred_len % cycle_period_len:
            raise ValueError(
                "anchored fusion requires pred_len divisible by cycle_period_len"
            )
        cycle_seq_len = (seq_len // cycle_period_len) * cycle_period_len
        if cycle_seq_len < 2 * cycle_period_len:
            raise ValueError("cycle period leaves fewer than two history cycles")
        if masked_origins < 1:
            raise ValueError("masked_origins must be positive")
        if correction_max <= 0 or deformation_max <= 0:
            raise ValueError("correction bounds must be positive")
        if risk_scale <= 0 or risk_std_weight < 0 or risk_clip <= 0:
            raise ValueError("invalid evidence risk parameters")
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")
        if mlp_hidden < 1 or modulation_temperature <= 0:
            raise ValueError("invalid MLP or modulation parameters")
        if not 0 < amplitude_min <= amplitude_max:
            raise ValueError("invalid amplitude range")
        if level_mode not in ("horizon_centered", "history_referenced"):
            raise ValueError(f"unsupported anchored level mode: {level_mode}")
        if not 0 < global_level_max <= correction_max:
            raise ValueError(
                "global_level_max must be positive and no larger than correction_max"
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.cycle_period_len = int(cycle_period_len)
        self.cycle_seq_len = int(cycle_seq_len)
        self.cycle_prefix_trim = self.seq_len - self.cycle_seq_len
        self.num_history_cycles = self.cycle_seq_len // self.cycle_period_len
        self.num_future_cycles = self.pred_len // self.cycle_period_len
        self.strategy = str(strategy)
        self.correction_max = float(correction_max)
        self.deformation_max = float(deformation_max)
        self.masked_origins = min(
            int(masked_origins), max(1, self.num_history_cycles - 1)
        )
        # Use as many lead-aligned historical targets as causality permits.  If
        # H is longer than the available history, the final measured confidence
        # is conservatively carried forward to the remaining future cycles.
        self.evidence_cycles = min(
            self.num_future_cycles,
            max(1, self.num_history_cycles - self.masked_origins),
        )
        self.risk_scale = float(risk_scale)
        self.risk_std_weight = float(risk_std_weight)
        self.confidence_floor = float(confidence_floor)
        self.risk_clip = float(risk_clip)
        self.modulation_temperature = float(modulation_temperature)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.detach_references = bool(detach_references)
        self.level_mode = str(level_mode)
        self.global_level_max = float(global_level_max)
        self.eps = float(eps)

        # ICPT construction is RNG-isolated so adding this composer cannot
        # perturb the paired initialization of the complete A2 anchor.
        with torch.random.fork_rng(devices=[]):
            self.cycle = InterCyclePatchResidualHead(
                seq_len=self.cycle_seq_len,
                pred_len=self.pred_len,
                period_len=self.cycle_period_len,
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
                    nn.Linear(mlp_hidden, 2),
                )
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)

        scalar = strategy == "component_scalar"
        coefficient_count = 1 if scalar else self.num_future_cycles
        if strategy == "phase_modulation":
            self.modulation_raw = nn.Parameter(
                torch.zeros(self.num_future_cycles)
            )
            self.deformation_raw = nn.Parameter(
                torch.zeros(self.num_future_cycles)
            )
        else:
            self.level_raw = nn.Parameter(torch.zeros(coefficient_count))
            self.shape_raw = nn.Parameter(torch.zeros(coefficient_count))

        self.last_anchor = None
        self.last_phase = None
        self.last_trajectory = None
        self.last_cycle = None
        self.cycle_for_auxiliary = None
        self.level_correction_for_auxiliary = None
        self.shape_correction_for_auxiliary = None
        self.level_coefficient_for_auxiliary = None
        self.shape_coefficient_for_auxiliary = None
        self.last_output = None
        self.last_level_correction = None
        self.last_shape_correction = None
        self.last_level_update = None
        self.last_global_level_update = None
        self.last_shape_update = None
        self.last_level_coefficient = None
        self.last_shape_coefficient = None
        self.last_level_confidence = None
        self.last_shape_confidence = None
        self.last_level_risk = None
        self.last_shape_risk = None
        self.last_level_risk_std = None
        self.last_shape_risk_std = None
        self.last_phase_reliability = None
        self.last_drift = None
        self.last_expected_shift = None
        self.last_amplitude = None
        self.last_anchor_identity_max_abs = None
        self.last_update_horizon_mean_max = None
        self.last_shape_cycle_mean_max = None
        self.last_projection_inner_product_max = None

    def _history_tail(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.size(1) != self.seq_len:
            raise ValueError(f"history must have shape (B,{self.seq_len},C)")
        return history[:, -self.cycle_seq_len :, :]

    def _cycles(self, series: torch.Tensor) -> torch.Tensor:
        if series.ndim != 3 or series.size(1) != self.pred_len:
            raise ValueError(f"forecast must have shape (B,{self.pred_len},C)")
        return series.reshape(
            series.size(0), self.num_future_cycles,
            self.cycle_period_len, series.size(2),
        )

    def _history_cycles(self, history: torch.Tensor) -> torch.Tensor:
        tail = self._history_tail(history)
        return tail.reshape(
            tail.size(0), self.num_history_cycles,
            self.cycle_period_len, tail.size(2),
        )

    @staticmethod
    def _decompose(cycles: torch.Tensor):
        cycle_mean = cycles.mean(dim=2, keepdim=True)
        global_mean = cycle_mean.mean(dim=1, keepdim=True)
        level = cycle_mean - global_mean
        shape = cycles - cycle_mean
        return global_mean, level, shape

    def decompose_forecast(self, forecast: torch.Tensor):
        """Return horizon-shaped relative-level and within-cycle components."""
        cycles = self._cycles(forecast)
        _, level, shape = self._decompose(cycles)
        return level.expand_as(cycles).reshape_as(forecast), shape.reshape_as(forecast)

    def decompose_residual_target(self, residual: torch.Tensor):
        """Project an anchor residual into the admitted level/shape spaces.

        The legacy level space removes the horizon-wide mean.  The repaired
        space keeps cycle means so a one-cycle forecast still has a learnable,
        tightly bounded level correction.
        """
        cycles = self._cycles(residual)
        cycle_mean = cycles.mean(dim=2, keepdim=True)
        shape = cycles - cycle_mean
        if self.level_mode == "horizon_centered":
            level = cycle_mean - cycle_mean.mean(dim=1, keepdim=True)
        else:
            level = cycle_mean
        return (
            level.expand_as(cycles).reshape_as(residual),
            shape.reshape_as(residual),
        )

    def _phase_reliability(self, phase_series: torch.Tensor):
        if phase_series.ndim != 4:
            raise ValueError("phase_series must have shape (B,C,L,K)")
        signal = phase_series.mean(dim=-1).var(dim=2, unbiased=False)
        noise = phase_series.var(dim=-1, unbiased=False).mean(dim=2)
        return signal / (signal + noise + self.eps)

    def _drift(self, history: torch.Tensor):
        cycles = self._history_cycles(history)
        scale = self._history_tail(history).std(
            dim=1, unbiased=False
        ).clamp_min(self.eps)
        raw = (cycles[:, -1].mean(dim=1) - cycles[:, -2].mean(dim=1)).abs()
        normalized = raw / scale
        return normalized / (1.0 + normalized)

    def _expand_evidence(self, value: torch.Tensor) -> torch.Tensor:
        if value.size(-1) == self.num_future_cycles:
            return value
        suffix = value[..., -1:].expand(
            *value.shape[:-1], self.num_future_cycles - value.size(-1)
        )
        return torch.cat((value, suffix), dim=-1)

    def _rolling_contexts(self, history: torch.Tensor):
        """Create causal, horizon-matched pseudo-origins for ICPT evidence."""
        cycles = self._history_cycles(history)
        last_origin = self.num_history_cycles - self.evidence_cycles
        first_origin = max(1, last_origin - self.masked_origins + 1)
        origins = list(range(first_origin, last_origin + 1))
        cycle_contexts, full_contexts, targets, templates = [], [], [], []
        first_cycle = cycles[:, :1]
        for origin in origins:
            observed = cycles[:, :origin]
            padding = first_cycle.expand(
                -1, self.num_history_cycles - origin, -1, -1
            )
            cycle_context = torch.cat((padding, observed), dim=1)
            cycle_contexts.append(cycle_context.reshape(
                history.size(0), self.cycle_seq_len, history.size(2)
            ))
            if self.cycle_prefix_trim:
                prefix = cycle_context[:, :1, :1, :].expand(
                    -1, 1, self.cycle_prefix_trim, -1
                ).reshape(history.size(0), self.cycle_prefix_trim, history.size(2))
                full_context = torch.cat(
                    (prefix, cycle_contexts[-1]), dim=1
                )
            else:
                full_context = cycle_contexts[-1]
            full_contexts.append(full_context)
            targets.append(cycles[:, origin : origin + self.evidence_cycles])
            templates.append(observed.mean(dim=1))
        return (
            torch.cat(cycle_contexts, dim=0),
            torch.cat(full_contexts, dim=0),
            torch.cat(targets, dim=0),
            torch.cat(templates, dim=0),
            len(origins),
        )

    def _rolling_evidence(self, history, trajectory_predictor):
        if trajectory_predictor is None:
            raise ValueError("evidence strategies require the A2 trajectory predictor")
        batch, _, channels = history.shape
        (
            cycle_contexts, full_contexts, targets, phase_templates, origins,
        ) = self._rolling_contexts(history)
        evidence_len = self.evidence_cycles * self.cycle_period_len
        with torch.no_grad():
            cycle = self.cycle(cycle_contexts)[:, :evidence_len]
            trajectory = trajectory_predictor(full_contexts)[:, :evidence_len]
            cycle = cycle.reshape(
                origins * batch, self.evidence_cycles,
                self.cycle_period_len, channels,
            )
            trajectory = trajectory.reshape_as(cycle)
            scale = full_contexts.std(
                dim=1, unbiased=False
            ).clamp_min(self.eps).unsqueeze(1)

            target_level = targets.mean(dim=2)
            cycle_level = cycle.mean(dim=2)
            trajectory_level = trajectory.mean(dim=2)
            level_cycle_error = (cycle_level - target_level).abs() / scale
            level_reference_error = (
                trajectory_level - target_level
            ).abs() / scale

            target_shape = targets - target_level.unsqueeze(2)
            cycle_shape = cycle - cycle_level.unsqueeze(2)
            phase_shape = phase_templates - phase_templates.mean(
                dim=1, keepdim=True
            )
            phase_shape = phase_shape.unsqueeze(1).expand_as(target_shape)
            shape_cycle_error = (
                cycle_shape - target_shape
            ).abs().mean(dim=2) / scale
            shape_reference_error = (
                phase_shape - target_shape
            ).abs().mean(dim=2) / scale

            level_regret = torch.log(
                (level_cycle_error + self.eps)
                / (level_reference_error + self.eps)
            ).clamp(-self.risk_clip, self.risk_clip)
            shape_regret = torch.log(
                (shape_cycle_error + self.eps)
                / (shape_reference_error + self.eps)
            ).clamp(-self.risk_clip, self.risk_clip)

            def aggregate(value):
                value = value.reshape(
                    origins, batch, self.evidence_cycles, channels
                ).permute(1, 3, 2, 0)
                return value.mean(dim=-1), value.std(dim=-1, unbiased=False)

            level_risk, level_std = aggregate(level_regret)
            shape_risk, shape_std = aggregate(shape_regret)
            level_score = level_risk + self.risk_std_weight * level_std
            shape_score = shape_risk + self.risk_std_weight * shape_std
            level_confidence = torch.sigmoid(-self.risk_scale * level_score)
            shape_confidence = torch.sigmoid(-self.risk_scale * shape_score)
            level_confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * level_confidence
            shape_confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * shape_confidence
        return tuple(self._expand_evidence(value) for value in (
            shape_confidence, level_confidence,
            shape_risk, level_risk, shape_std, level_std,
        ))

    def _base_coefficient(self, raw: torch.Tensor, batch: int, channels: int):
        value = self.correction_max * torch.tanh(raw).view(1, 1, -1)
        if value.size(-1) == 1:
            value = value.expand(-1, -1, self.num_future_cycles)
        return value.expand(batch, channels, -1)

    def _component_coefficients(
        self, anchor, phase, trajectory, cycle, history, phase_reliability,
        shape_confidence, level_confidence,
    ):
        batch, _, channels = history.shape
        shape = self._base_coefficient(self.shape_raw, batch, channels)
        level = self._base_coefficient(self.level_raw, batch, channels)
        if self.strategy == "shape_only":
            level = torch.zeros_like(level)
        elif self.strategy == "level_only":
            shape = torch.zeros_like(shape)
        elif self.strategy == "monotonic_evidence":
            phase_allowance = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * (1.0 - phase_reliability).unsqueeze(-1)
            shape = shape * shape_confidence * phase_allowance
            level = level * level_confidence
        elif self.strategy == "mlp_evidence":
            scale = self._history_tail(history).std(
                dim=1, unbiased=False
            ).clamp_min(self.eps)
            phase_cycles = self._cycles(phase)
            trajectory_cycles = self._cycles(trajectory)
            cycle_cycles = self._cycles(cycle)
            _, trajectory_level, _ = self._decompose(trajectory_cycles)
            _, cycle_level, cycle_shape = self._decompose(cycle_cycles)
            if self.level_mode == "history_referenced":
                trajectory_level = trajectory_cycles.mean(dim=2, keepdim=True)
                cycle_level = cycle_cycles.mean(dim=2, keepdim=True)
            _, _, phase_shape = self._decompose(phase_cycles)

            def mean_abs(value):
                return (
                    value.abs().mean(dim=2) / scale.unsqueeze(1)
                ).permute(0, 2, 1)

            features = torch.stack((
                (1.0 - phase_reliability).unsqueeze(-1).expand(
                    -1, -1, self.num_future_cycles
                ),
                self._drift(history).unsqueeze(-1).expand(
                    -1, -1, self.num_future_cycles
                ),
                shape_confidence,
                level_confidence,
                mean_abs(cycle_shape - phase_shape),
                mean_abs(cycle_level - trajectory_level),
                mean_abs(cycle_cycles - self._cycles(anchor)),
            ), dim=-1).detach()
            correction = self.gate_mlp(features)
            shape_raw = self.shape_raw.view(1, 1, -1) + correction[..., 0]
            level_raw = self.level_raw.view(1, 1, -1) + correction[..., 1]
            shape = self.correction_max * torch.tanh(shape_raw)
            level = self.correction_max * torch.tanh(level_raw)
        return level, shape

    @staticmethod
    def _coefficient_to_cycles(coefficient: torch.Tensor):
        return coefficient.permute(0, 2, 1).unsqueeze(2)

    def _phase_modulation(
        self, phase_cycles: torch.Tensor, cycle_cycles: torch.Tensor,
        batch: int, channels: int,
    ):
        phase_shape = phase_cycles - phase_cycles.mean(dim=2, keepdim=True)
        cycle_shape = cycle_cycles - cycle_cycles.mean(dim=2, keepdim=True)
        rolled = torch.stack(
            [
                torch.roll(phase_shape, shifts=shift, dims=2)
                for shift in range(self.cycle_period_len)
            ],
            dim=-1,
        )
        numerator = (rolled * cycle_shape.unsqueeze(-1)).mean(dim=2)
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
        modulation = self.correction_max * torch.tanh(
            self.modulation_raw
        ).view(1, -1, 1, 1)
        deformation_scale = self.deformation_max * torch.tanh(
            self.deformation_raw
        ).view(1, -1, 1, 1)
        shape_update = (
            modulation * (modulated - phase_shape)
            + deformation_scale * deformation
        )
        shifts = torch.arange(
            self.cycle_period_len, device=phase_cycles.device,
            dtype=phase_cycles.dtype,
        )
        signed = torch.where(
            shifts <= self.cycle_period_len // 2,
            shifts, shifts - self.cycle_period_len,
        )
        self.last_expected_shift = (
            shift_weights * signed
        ).sum(dim=-1).detach()
        self.last_amplitude = amplitude.detach()
        self.last_shape_coefficient = modulation.expand(
            batch, -1, -1, channels
        ).squeeze(2).permute(0, 2, 1).detach()
        self.last_level_coefficient = torch.zeros_like(
            self.last_shape_coefficient
        )
        return shape_update

    def forward(
        self,
        anchor: torch.Tensor,
        phase: torch.Tensor,
        trajectory: torch.Tensor,
        history: torch.Tensor,
        phase_series: torch.Tensor,
        *,
        trajectory_predictor=None,
    ):
        expected = (history.size(0), self.pred_len, history.size(2))
        if anchor.shape != expected or phase.shape != expected or trajectory.shape != expected:
            raise ValueError("anchor, phase and trajectory shapes are incompatible")
        cycle = self.cycle(self._history_tail(history))
        # Keep one graph-carrying handle for the component auxiliary objective;
        # all public diagnostics below remain detached.
        self.cycle_for_auxiliary = cycle
        anchor_cycles = self._cycles(anchor)
        phase_cycles = self._cycles(phase)
        trajectory_cycles = self._cycles(trajectory)
        cycle_cycles = self._cycles(cycle)
        phase_reliability = self._phase_reliability(phase_series)
        drift = self._drift(history)

        batch, _, channels = history.shape
        ones = history.new_ones((batch, channels, self.num_future_cycles))
        zeros = history.new_zeros((batch, channels, self.num_future_cycles))
        shape_confidence = level_confidence = ones
        shape_risk = level_risk = shape_std = level_std = zeros
        if self.strategy in ("monotonic_evidence", "mlp_evidence"):
            (
                shape_confidence, level_confidence,
                shape_risk, level_risk, shape_std, level_std,
            ) = self._rolling_evidence(history, trajectory_predictor)

        level_update = torch.zeros_like(anchor_cycles)
        if self.strategy == "phase_modulation":
            shape_update = self._phase_modulation(
                phase_cycles, cycle_cycles, batch, channels
            )
            level_correction = torch.zeros_like(anchor_cycles)
            shape_correction = shape_update
            self.level_correction_for_auxiliary = level_correction.reshape_as(anchor)
            self.shape_correction_for_auxiliary = shape_correction.reshape_as(anchor)
            self.level_coefficient_for_auxiliary = zeros
            self.shape_coefficient_for_auxiliary = zeros
            global_level_update = torch.zeros_like(level_update)
        else:
            _, trajectory_level, _ = self._decompose(
                trajectory_cycles
            )
            _, cycle_level, cycle_shape = self._decompose(cycle_cycles)
            _, _, phase_shape = self._decompose(phase_cycles)
            if self.detach_references:
                trajectory_level = trajectory_level.detach()
                phase_shape = phase_shape.detach()
            if self.level_mode == "history_referenced":
                # Direct cycle-mean disagreement remains meaningful when the
                # forecast contains only one future cycle.  Its global part is
                # admitted through a much smaller trust region below.
                level_correction = (
                    cycle_cycles.mean(dim=2, keepdim=True)
                    - trajectory_cycles.mean(dim=2, keepdim=True).detach()
                    if self.detach_references
                    else cycle_cycles.mean(dim=2, keepdim=True)
                    - trajectory_cycles.mean(dim=2, keepdim=True)
                )
            else:
                level_correction = cycle_level - trajectory_level
            shape_correction = cycle_shape - phase_shape
            level_coefficient, shape_coefficient = self._component_coefficients(
                anchor, phase, trajectory, cycle, history, phase_reliability,
                shape_confidence, level_confidence,
            )
            level_scale = self._coefficient_to_cycles(level_coefficient)
            if self.level_mode == "history_referenced":
                global_level = level_correction.mean(dim=1, keepdim=True)
                relative_level = level_correction - global_level
                level_update = (level_scale * relative_level).expand_as(
                    anchor_cycles
                )
                level_update = level_update - level_update.mean(
                    dim=(1, 2), keepdim=True
                )
                global_scale = level_coefficient.mean(dim=-1, keepdim=True)
                global_scale = (
                    global_scale * self.global_level_max / self.correction_max
                ).permute(0, 2, 1).unsqueeze(2)
                global_level_update = (
                    global_scale * global_level
                ).expand_as(anchor_cycles)
                level_update = level_update + global_level_update
            else:
                level_update = (level_scale * level_correction).expand_as(
                    anchor_cycles
                )
                # Per-cycle coefficients can reintroduce a global offset.
                level_update = level_update - level_update.mean(
                    dim=(1, 2), keepdim=True
                )
                global_level_update = torch.zeros_like(level_update)
            shape_update = (
                self._coefficient_to_cycles(shape_coefficient)
                * shape_correction
            )
            self.level_correction_for_auxiliary = level_correction.expand_as(
                anchor_cycles
            ).reshape_as(anchor)
            self.shape_correction_for_auxiliary = shape_correction.reshape_as(anchor)
            self.level_coefficient_for_auxiliary = level_coefficient
            self.shape_coefficient_for_auxiliary = shape_coefficient
            self.last_level_coefficient = level_coefficient.detach()
            self.last_shape_coefficient = shape_coefficient.detach()

        output_cycles = anchor_cycles + level_update + shape_update
        output = output_cycles.reshape_as(anchor)
        with torch.no_grad():
            shape_cycle_mean = shape_update.mean(dim=2).abs().max()
            projection_inner = (level_update * shape_update).sum(dim=2).abs().max()
            self.last_anchor = anchor.detach()
            self.last_phase = phase.detach()
            self.last_trajectory = trajectory.detach()
            self.last_cycle = cycle.detach()
            self.last_output = output.detach()
            self.last_level_correction = level_correction.expand_as(
                anchor_cycles
            ).reshape_as(anchor).detach()
            self.last_shape_correction = shape_correction.reshape_as(anchor).detach()
            self.last_level_update = level_update.reshape_as(anchor).detach()
            self.last_global_level_update = global_level_update.reshape_as(
                anchor
            ).detach()
            self.last_shape_update = shape_update.reshape_as(anchor).detach()
            self.last_level_confidence = level_confidence.detach()
            self.last_shape_confidence = shape_confidence.detach()
            self.last_level_risk = level_risk.detach()
            self.last_shape_risk = shape_risk.detach()
            self.last_level_risk_std = level_std.detach()
            self.last_shape_risk_std = shape_std.detach()
            self.last_phase_reliability = phase_reliability.detach()
            self.last_drift = drift.detach()
            self.last_anchor_identity_max_abs = float(
                (output - anchor).abs().max().detach()
            )
            self.last_update_horizon_mean_max = float(
                (output - anchor).mean(dim=1).abs().max().detach()
            )
            self.last_shape_cycle_mean_max = float(shape_cycle_mean.detach())
            self.last_projection_inner_product_max = float(
                projection_inner.detach()
            )
        return output
