"""Cross-attention variants built on the legacy PhaseFormer-L low-rank head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead


class PhaseContentCorrection(nn.Module):
    """Small content residual on the original pre-norm broadcast path.

    All modes have identical parameter shapes. The trend-only control replaces
    phase context with a deterministic pooled view of the same rank token.
    Diagnostics/interventions never change parameters or the low-rank decoder.
    """

    def __init__(self, rank, phase_dim, width, mode):
        super().__init__()
        if mode not in {"trend", "phase", "centered"}:
            raise ValueError(f"unsupported content mode: {mode}")
        self.mode = mode
        self.phase_dim = phase_dim
        self.context_bins = [(math.floor(i * rank / phase_dim),
                              math.ceil((i + 1) * rank / phase_dim))
                             for i in range(phase_dim)]
        self.trend_projection = nn.Linear(rank, width)
        # A context bias would cancel in the centered arm and confound matching.
        self.context_projection = nn.Linear(phase_dim, width, bias=False)
        self.output_projection = nn.Linear(width, phase_dim, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))
        self.intervention = "native"
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, trend, phase):
        if self.mode == "trend":
            # Equivalent adaptive-pool bins with deterministic CUDA backward.
            context = torch.stack([trend[..., start:end].mean(-1)
                                   for start, end in self.context_bins], dim=-1)
            context = context.expand(-1, phase.shape[1], -1)
        else:
            context = phase
        if self.intervention == "roll":
            context = context.roll(phase.shape[1] // 2, dims=1)
        elif self.intervention == "reverse":
            context = context.flip(1)
        raw = self.output_projection(
            self.trend_projection(trend) * self.context_projection(context)
        )
        if self.mode == "centered":
            raw = raw - raw.mean(1, keepdim=True)
        multiplier = {"off": 0., "half": .5, "double": 2.}.get(self.intervention, 1.)
        update = multiplier * self.scale * raw
        if self.capture_diagnostics:
            self.last_diagnostics = {"raw": raw.detach(), "update": update.detach()}
        return update


class LowRankPhaseInteractionBranch(nn.Module):
    """Legacy pooled-low-rank residual head with optional phase interaction.

    The no-interaction path is exactly the historical
    ``PooledLowRankWeakPeriodResidualHead``.  Cross-attention is applied to its
    rank-``r`` encoder output before the unchanged decoder; reverse-direction modes
    additionally return an updated phase latent for the PhaseFormer predictor.
    """

    MODES = {
        "none",
        "phase_to_trend",
        "trend_to_phase",
        "parallel_bidirectional",
        "alternating",
        "joint",
    }

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        phase_dim: int,
        *,
        pool_factor: int,
        rank: int,
        smooth_ratio: float = 0.0,
        smooth_window: int = 24,
        interaction_mode: str = "none",
        num_heads: int = 1,
        update_style: str = "post_norm",
        null_read: bool = False,
        selective_write: bool = False,
        content_mode: str = "none",
        content_width: int = 8,
    ) -> None:
        super().__init__()
        if interaction_mode not in self.MODES:
            raise ValueError(f"unsupported interaction_mode: {interaction_mode}")
        if phase_dim % num_heads:
            raise ValueError("phase_dim must be divisible by num_heads")
        if update_style not in {"post_norm", "pre_norm", "scaled_pre_norm"}:
            raise ValueError(f"unsupported update_style: {update_style}")
        if update_style != "post_norm" and (
            interaction_mode not in {"phase_to_trend", "trend_to_phase"}
            or num_heads != 1
        ):
            raise ValueError("pre-norm audit supports single-head, one-way interaction")
        if null_read and (
            update_style != "scaled_pre_norm" or interaction_mode != "phase_to_trend"
        ):
            raise ValueError("null read requires scaled pre-norm P -> T")
        if selective_write and (
            update_style != "scaled_pre_norm" or interaction_mode != "trend_to_phase"
        ):
            raise ValueError("selective write requires scaled pre-norm T -> P")
        self.interaction_mode = interaction_mode
        self.update_style = update_style
        self.null_read = bool(null_read)
        self.selective_write = bool(selective_write)
        self.receipt_intervention = "native"
        self.capture_diagnostics = False
        self.last_diagnostics = {}
        self.rank = int(rank)
        self.phase_dim = int(phase_dim)
        self.lowrank_head = PooledLowRankWeakPeriodResidualHead(
            seq_len,
            pred_len,
            pool_factor=pool_factor,
            rank=rank,
            smooth_ratio=smooth_ratio,
            smooth_window=smooth_window,
        )

        use_pt = interaction_mode in {
            "phase_to_trend",
            "parallel_bidirectional",
            "alternating",
        }
        use_tp = interaction_mode in {
            "trend_to_phase",
            "parallel_bidirectional",
            "alternating",
        }
        use_joint = interaction_mode == "joint"

        # P -> T: one rank latent queries the phase-token sequence.
        self.trend_query = nn.Linear(rank, phase_dim) if use_pt else None
        self.phase_key = nn.Linear(phase_dim, phase_dim) if use_pt else None
        self.phase_value = nn.Linear(phase_dim, phase_dim) if use_pt else None
        self.trend_out = nn.Linear(phase_dim, rank) if use_pt else None
        self.trend_norm = nn.LayerNorm(rank) if use_pt else None
        self.phase_to_trend_attention = (
            nn.MultiheadAttention(phase_dim, num_heads=num_heads, batch_first=True)
            if use_pt
            else None
        )

        # T -> P: phase slots query the single rank latent.  With one key/value
        # every softmax row is one, so this is deliberately a broadcast control.
        self.phase_query = nn.Linear(phase_dim, phase_dim) if use_tp else None
        self.trend_key = nn.Linear(rank, phase_dim) if use_tp else None
        self.trend_value = nn.Linear(rank, phase_dim) if use_tp else None
        self.phase_out = nn.Linear(phase_dim, phase_dim) if use_tp else None
        self.phase_norm = nn.LayerNorm(phase_dim) if use_tp else None
        self.trend_to_phase_attention = (
            nn.MultiheadAttention(phase_dim, num_heads=num_heads, batch_first=True)
            if use_tp
            else None
        )

        # Joint control: project the rank latent to the phase width, attend over
        # the concatenated sequence, then project the trend output back to rank.
        self.joint_trend_in = nn.Linear(rank, phase_dim) if use_joint else None
        self.joint_trend_out = nn.Linear(phase_dim, rank) if use_joint else None
        self.joint_phase_out = nn.Linear(phase_dim, phase_dim) if use_joint else None
        self.joint_attention = (
            nn.MultiheadAttention(phase_dim, num_heads=num_heads, batch_first=True)
            if use_joint
            else None
        )
        self.joint_trend_norm = nn.LayerNorm(rank) if use_joint else None
        self.joint_phase_norm = nn.LayerNorm(phase_dim) if use_joint else None

        if update_style != "post_norm":
            # Only the message sees normalized inputs; the rank latent bypasses LN.
            if self.trend_norm is None:
                self.trend_norm = nn.LayerNorm(rank)
            if self.phase_norm is None:
                self.phase_norm = nn.LayerNorm(phase_dim)
            if update_style == "scaled_pre_norm":
                self.interaction_scale = nn.Parameter(torch.zeros(()))
            if null_read or selective_write:
                self.null_score = nn.Parameter(torch.zeros(()))

        self.last_phase_attention = None
        if content_mode != "none" and (
            interaction_mode != "trend_to_phase" or update_style != "pre_norm"
            or null_read or selective_write or content_width < 1
        ):
            raise ValueError("content residual requires plain pre-norm T -> P and positive width")
        self.content_correction = (
            PhaseContentCorrection(rank, phase_dim, content_width, content_mode)
            if content_mode != "none" else None
        )

    def _phase_to_trend(self, latent, phase_latent):
        b, c, slots, _ = phase_latent.shape
        trend = latent.reshape(b * c, 1, self.rank)
        phase = phase_latent.reshape(b * c, slots, self.phase_dim)
        if self.update_style != "post_norm":
            return self._pre_norm_update(trend, phase, b, c, slots, to_trend=True)
        attended, weights = self.phase_to_trend_attention(
            self.trend_query(trend),
            self.phase_key(phase),
            self.phase_value(phase),
            need_weights=True,
        )
        trend = self.trend_norm(
            trend + self.trend_out(attended)
        )
        return trend.reshape(b, c, self.rank), weights.reshape(b, c, slots)

    def _trend_to_phase(self, latent, phase_latent):
        b, c, slots, _ = phase_latent.shape
        phase = phase_latent.reshape(b * c, slots, self.phase_dim)
        trend = latent.reshape(b * c, 1, self.rank)
        if self.update_style != "post_norm":
            return self._pre_norm_update(trend, phase, b, c, slots, to_trend=False)
        attended, weights = self.trend_to_phase_attention(
            self.phase_query(phase),
            self.trend_key(trend),
            self.trend_value(trend),
            need_weights=True,
        )
        phase = self.phase_norm(phase + self.phase_out(attended))
        return phase.reshape(b, c, slots, self.phase_dim), weights.reshape(b, c, slots)

    def _pre_norm_update(self, trend, phase, b, c, slots, *, to_trend):
        t, p = self.trend_norm(trend), self.phase_norm(phase)
        if to_trend:
            attention = self.phase_to_trend_attention
            q, k, v = self.trend_query(t), self.phase_key(p), self.phase_value(p)
        else:
            attention = self.trend_to_phase_attention
            q, k, v = self.phase_query(p), self.trend_key(t), self.trend_value(t)
        # Use the same projections as the legacy MHA; expose the null competitor
        # explicitly so an output bias cannot leak through a zero-valued null.
        weights_q, weights_k, weights_v = attention.in_proj_weight.chunk(3)
        bias_q, bias_k, bias_v = attention.in_proj_bias.chunk(3)
        q = F.linear(q, weights_q, bias_q)
        k = F.linear(k, weights_k, bias_k)
        v = F.linear(v, weights_v, bias_v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.phase_dim)
        if self.null_read or self.selective_write:
            scores = torch.cat([scores, self.null_score.expand_as(scores[..., :1])], -1)
            weights = scores.softmax(-1)[..., :-1]
        else:
            weights = scores.softmax(-1)
        if self.selective_write:
            if self.receipt_intervention == "mean":
                weights = weights.mean(1, keepdim=True).expand_as(weights)
            elif self.receipt_intervention == "roll":
                weights = weights.roll(slots // 2, dims=1)
        mass = weights.sum(-1, keepdim=True)
        attended = F.linear(weights @ v, attention.out_proj.weight)
        attended = attended + mass * attention.out_proj.bias
        output = self.trend_out if to_trend else self.phase_out
        message = F.linear(attended, output.weight) + mass * output.bias
        scale = self.interaction_scale if self.update_style == "scaled_pre_norm" else 1.0
        original = trend if to_trend else phase
        updated = original + scale * message
        if not to_trend and self.content_correction is not None:
            updated = updated + self.content_correction(t, p)
        if self.capture_diagnostics:
            self.last_diagnostics = {
                "message_norm": message.norm(dim=-1).mean(-1).reshape(b, c).detach(),
                "update_norm": (updated - original).norm(dim=-1).mean(-1).reshape(b, c).detach(),
                "original_norm": original.norm(dim=-1).mean(-1).reshape(b, c).detach(),
                "acceptance": mass.squeeze(-1).reshape(b, c, -1).detach(),
                "weights": weights.reshape(b, c, *weights.shape[1:]).detach(),
            }
        shape = (b, c, self.rank) if to_trend else (b, c, slots, self.phase_dim)
        return updated.reshape(shape), weights.reshape(b, c, slots)

    def _joint(self, latent, phase_latent):
        b, c, slots, _ = phase_latent.shape
        trend = self.joint_trend_in(latent).reshape(b * c, 1, self.phase_dim)
        phase = phase_latent.reshape(b * c, slots, self.phase_dim)
        joint = torch.cat([trend, phase], dim=1)
        attended, weights = self.joint_attention(joint, joint, joint, need_weights=True)
        trend = self.joint_trend_norm(
            latent.reshape(b * c, 1, self.rank)
            + self.joint_trend_out(attended[:, :1])
        )
        phase = self.joint_phase_norm(phase + self.joint_phase_out(attended[:, 1:]))
        return (
            trend.reshape(b, c, self.rank),
            phase.reshape(b, c, slots, self.phase_dim),
            weights.reshape(b, c, slots + 1, slots + 1),
        )

    def forward(self, x, phase_latent):
        latent, last = self.lowrank_head.encode_latent(x)
        phase_updated = phase_latent
        phase_attention = None
        joint_attention = None
        mode = self.interaction_mode
        if mode == "phase_to_trend":
            latent, phase_attention = self._phase_to_trend(latent, phase_latent)
        elif mode == "trend_to_phase":
            phase_updated, phase_attention = self._trend_to_phase(latent, phase_latent)
        elif mode == "parallel_bidirectional":
            original_latent = latent
            original_phase = phase_latent
            latent, phase_attention = self._phase_to_trend(original_latent, original_phase)
            phase_updated, _ = self._trend_to_phase(original_latent, original_phase)
        elif mode == "alternating":
            for _ in range(2):
                latent, phase_attention = self._phase_to_trend(latent, phase_updated)
                phase_updated, _ = self._trend_to_phase(latent, phase_updated)
        elif mode == "joint":
            latent, phase_updated, joint_attention = self._joint(latent, phase_latent)
        elif mode != "none":
            raise RuntimeError(f"unsupported interaction mode: {mode}")

        self.last_phase_attention = (
            phase_attention.detach() if phase_attention is not None else None
        )
        forecast = self.lowrank_head.decode_latent(latent, last)
        return forecast, {
            "updated_phase_latent": phase_updated,
            "rank_latent": latent,
            "phase_attention": self.last_phase_attention,
            "joint_attention": (
                joint_attention.detach() if joint_attention is not None else None
            ),
            "interaction_mode": mode,
        }
