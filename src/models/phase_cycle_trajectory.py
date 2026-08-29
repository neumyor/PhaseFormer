"""Structured phase--cycle--trajectory residual for one PhaseFormer.

The PhaseFormer trunk remains the phase model.  Inside the residual path,
NLinear supplies the complete trajectory and a no-PE ICPT supplies a cycle
forecast.  ICPT is not averaged as a third complete prediction: only two
identifiable components of ``ICPT - NLinear`` are admitted:

* a within-cycle zero-mean shape correction; and
* a cycle-level correction whose mean over the whole forecast is zero.

The latter relaxes HPTC's overly strict *per-cycle* zero-mean constraint while
still reserving the absolute forecast level for NLinear.  Optional confidence
comes from causal, history-only masked-cycle reconstructions with the same
NLinear and ICPT modules.  It only shrinks corrections and never selects a
complete expert.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.intercycle_patch import InterCyclePatchResidualHead
from src.models.phase_adapters import WeakPeriodResidualHead


class PhaseCycleTrajectoryResidualHead(nn.Module):
    """NLinear trajectory plus structured ICPT cycle corrections."""

    CONFIDENCE_MODES = ("fixed", "masked_absolute", "masked_regret")

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int = 24,
        *,
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        shape_gate_init: float = 0.10,
        level_gate_init: float = 0.10,
        use_shape_correction: bool = True,
        use_level_correction: bool = True,
        confidence_mode: str = "fixed",
        masked_origins: int = 2,
        risk_scale: float = 1.0,
        risk_std_weight: float = 0.5,
        confidence_floor: float = 0.05,
        risk_clip: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if seq_len <= 0 or pred_len <= 0 or period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period_len > 1")
        if seq_len % period_len or pred_len % period_len:
            raise ValueError("PCTF requires seq_len and pred_len divisible by period_len")
        if confidence_mode not in self.CONFIDENCE_MODES:
            raise ValueError(f"unsupported PCTF confidence mode: {confidence_mode}")
        if not use_shape_correction and not use_level_correction:
            raise ValueError("PCTF must enable at least one cycle correction")
        if masked_origins < 1 or masked_origins >= seq_len // period_len:
            raise ValueError("masked_origins must leave at least one observed cycle")
        if risk_scale <= 0 or risk_std_weight < 0 or risk_clip <= 0:
            raise ValueError("invalid PCTF risk parameters")
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.num_history_cycles = seq_len // period_len
        self.num_future_cycles = pred_len // period_len
        self.use_shape_correction = bool(use_shape_correction)
        self.use_level_correction = bool(use_level_correction)
        self.confidence_mode = str(confidence_mode)
        self.masked_origins = int(masked_origins)
        self.risk_scale = float(risk_scale)
        self.risk_std_weight = float(risk_std_weight)
        self.confidence_floor = float(confidence_floor)
        self.risk_clip = float(risk_clip)
        self.eps = float(eps)

        # Constructing this first exactly matches A1's NLinear constructor and
        # RNG consumption.  Forking ICPT construction means the shared
        # PhaseFormer trunk retains paired-seed initialization with A1.
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
        self.shape_logits = nn.Parameter(
            torch.full(
                (self.num_future_cycles,),
                self._logit(shape_gate_init),
            )
        )
        self.level_logits = nn.Parameter(
            torch.full(
                (self.num_future_cycles,),
                self._logit(level_gate_init),
            )
        )

        self.last_trajectory = None
        self.last_cycle = None
        self.last_shape_correction = None
        self.last_level_correction = None
        self.last_shape_update = None
        self.last_level_update = None
        self.last_shape_risk = None
        self.last_level_risk = None
        self.last_shape_risk_std = None
        self.last_level_risk_std = None
        self.last_shape_confidence = None
        self.last_level_confidence = None
        self.last_shape_gate = None
        self.last_level_gate = None
        self.last_shape_cycle_mean_max = None
        self.last_level_horizon_mean_max = None
        self.last_projection_inner_product_max = None

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return float(torch.logit(torch.tensor(value)))

    def _cycles(self, series: torch.Tensor) -> torch.Tensor:
        if series.ndim != 3:
            raise ValueError("series must have shape (B,L,C)")
        if series.size(1) % self.period_len:
            raise ValueError("series length must be divisible by period_len")
        return series.view(
            series.size(0), series.size(1) // self.period_len,
            self.period_len, series.size(2),
        )

    def decompose_cycle_difference(self, difference: torch.Tensor):
        """Return orthogonal shape and conserved relative-level components."""
        if difference.size(1) != self.pred_len:
            raise ValueError(f"expected pred_len={self.pred_len}")
        cycles = self._cycles(difference)
        cycle_mean = cycles.mean(dim=2, keepdim=True)
        shape = cycles - cycle_mean
        # ICPT may redistribute level among future cycles, but it cannot change
        # the horizon-wide mean owned by NLinear.
        relative_level = cycle_mean - cycle_mean.mean(dim=1, keepdim=True)
        relative_level = relative_level.expand_as(cycles)
        return (
            shape.reshape_as(difference),
            relative_level.reshape_as(difference),
        )

    def _masked_contexts(self, history: torch.Tensor):
        """Build causal fixed-length contexts for the latest held-out cycles."""
        cycles = self._cycles(history)
        contexts, targets = [], []
        first = cycles[:, :1]
        start = self.num_history_cycles - self.masked_origins
        for target_index in range(start, self.num_history_cycles):
            observed = cycles[:, :target_index]
            left_pad = first.expand(
                -1, self.num_history_cycles - target_index, -1, -1
            )
            context = torch.cat((left_pad, observed), dim=1)
            contexts.append(context.reshape_as(history))
            targets.append(cycles[:, target_index])
        return torch.cat(contexts, dim=0), torch.cat(targets, dim=0)

    def _masked_risks(self, history: torch.Tensor):
        B, _, C = history.shape
        contexts, targets = self._masked_contexts(history)
        # Reliability is evidence, not a training target.  Detaching prevents
        # either expert from improving the main loss by gaming its confidence.
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

            if self.confidence_mode == "masked_regret":
                level = torch.relu(torch.log(
                    (level_cycle_error + self.eps)
                    / (level_trajectory_error + self.eps)
                ))
                shape = torch.relu(torch.log(
                    (shape_cycle_error + self.eps)
                    / (shape_trajectory_error + self.eps)
                ))
            else:
                level, shape = level_cycle_error, shape_cycle_error
            level = level.clamp(max=self.risk_clip).view(
                self.masked_origins, B, C
            )
            shape = shape.clamp(max=self.risk_clip).view(
                self.masked_origins, B, C
            )
            return (
                shape.mean(dim=0),
                level.mean(dim=0),
                shape.std(dim=0, unbiased=False),
                level.std(dim=0, unbiased=False),
            )

    def _confidence(self, risk: torch.Tensor, risk_std: torch.Tensor):
        score = risk + self.risk_std_weight * risk_std
        confidence = torch.exp(-self.risk_scale * score).clamp(0.0, 1.0)
        return self.confidence_floor + (1.0 - self.confidence_floor) * confidence

    def _cycle_gate(self, logits, confidence, enabled, dtype, device):
        if not enabled:
            return torch.zeros(
                confidence.size(0), confidence.size(1), self.num_future_cycles,
                dtype=dtype, device=device,
            )
        prior = torch.sigmoid(logits).view(1, 1, -1)
        return confidence.unsqueeze(-1) * prior

    def _expand_gate(self, gate: torch.Tensor):
        return gate.repeat_interleave(self.period_len, dim=-1).permute(0, 2, 1)

    def forward_components(self, x: torch.Tensor):
        trajectory = self.trajectory(x)
        cycle = self.cycle(x)
        return trajectory, cycle

    def forward(self, x: torch.Tensor):
        if x.size(1) != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x.size(1)}")
        if self.confidence_mode == "fixed":
            shape_risk = level_risk = x.new_zeros((x.size(0), x.size(2)))
            shape_std = level_std = x.new_zeros((x.size(0), x.size(2)))
            shape_confidence = level_confidence = x.new_ones(
                (x.size(0), x.size(2))
            )
        else:
            shape_risk, level_risk, shape_std, level_std = self._masked_risks(x)
            shape_confidence = self._confidence(shape_risk, shape_std)
            level_confidence = self._confidence(level_risk, level_std)

        trajectory, cycle = self.forward_components(x)
        shape, level = self.decompose_cycle_difference(cycle - trajectory)
        shape_gate = self._cycle_gate(
            self.shape_logits, shape_confidence, self.use_shape_correction,
            x.dtype, x.device,
        )
        level_gate = self._cycle_gate(
            self.level_logits, level_confidence, self.use_level_correction,
            x.dtype, x.device,
        )
        shape_update = self._expand_gate(shape_gate) * shape
        level_update = self._expand_gate(level_gate) * level
        # Per-cycle gates would otherwise reintroduce a non-zero global level.
        # Re-project after gating so the *actual* update, not only its ungated
        # basis, preserves NLinear's horizon-wide mean.
        level_update = level_update - level_update.mean(dim=1, keepdim=True)
        output = trajectory + shape_update + level_update

        with torch.no_grad():
            shape_cycles = self._cycles(shape_update)
            level_cycles = self._cycles(level_update)
            inner = (shape_cycles * level_cycles).sum(dim=2)
            self.last_trajectory = trajectory.detach()
            self.last_cycle = cycle.detach()
            self.last_shape_correction = shape.detach()
            self.last_level_correction = level.detach()
            self.last_shape_update = shape_update.detach()
            self.last_level_update = level_update.detach()
            self.last_shape_risk = shape_risk.detach()
            self.last_level_risk = level_risk.detach()
            self.last_shape_risk_std = shape_std.detach()
            self.last_level_risk_std = level_std.detach()
            self.last_shape_confidence = shape_confidence.detach()
            self.last_level_confidence = level_confidence.detach()
            self.last_shape_gate = shape_gate.detach()
            self.last_level_gate = level_gate.detach()
            self.last_shape_cycle_mean_max = float(
                shape_cycles.mean(dim=2).abs().max().detach()
            )
            self.last_level_horizon_mean_max = float(
                level_update.mean(dim=1).abs().max().detach()
            )
            self.last_projection_inner_product_max = float(
                inner.abs().max().detach()
            )
        return output
