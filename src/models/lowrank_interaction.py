"""Cross-attention variants built on the legacy PhaseFormer-L low-rank head."""

from __future__ import annotations

import torch
from torch import nn

from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead


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
    ) -> None:
        super().__init__()
        if interaction_mode not in self.MODES:
            raise ValueError(f"unsupported interaction_mode: {interaction_mode}")
        if phase_dim % num_heads:
            raise ValueError("phase_dim must be divisible by num_heads")
        self.interaction_mode = interaction_mode
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

        self.last_phase_attention = None

    def _phase_to_trend(self, latent, phase_latent):
        b, c, slots, _ = phase_latent.shape
        trend = latent.reshape(b * c, 1, self.rank)
        phase = phase_latent.reshape(b * c, slots, self.phase_dim)
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
        attended, weights = self.trend_to_phase_attention(
            self.phase_query(phase),
            self.trend_key(trend),
            self.trend_value(trend),
            need_weights=True,
        )
        phase = self.phase_norm(phase + self.phase_out(attended))
        return phase.reshape(b, c, slots, self.phase_dim), weights.reshape(b, c, slots)

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
