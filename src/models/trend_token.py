"""History-only trend token and low-rank profile correction.

The module deliberately keeps the phase path read-only.  A per-channel token is
encoded from the normalized history, reads the routed phase latent through a
single cross-attention, and emits coefficients for a fixed horizon profile
bank.  Consequently every correction is in ``span(profile_bank)`` by
construction, which is the implementation counterpart of the rank-m
complement argument in minipaper_0930 Theorem 3.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _causal_conv1d(x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
    """Apply a Conv1d with left-only padding (no future leakage)."""
    left = (int(layer.kernel_size[0]) - 1) * int(layer.dilation[0])
    return layer(F.pad(x, (left, 0)))


class TrendTokenBranch(nn.Module):
    """Single learned per-channel trend token with selectable horizon readout.

    ``profile`` keeps the fixed low-rank profile decoder used by the theorem
    path.  ``direct`` is the controlled ablation: the same token is mapped
    directly to all future steps by a shared horizon head.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        phase_dim: int,
        *,
        token_dim: int = 32,
        kernel_size: int = 5,
        dilations: Iterable[int] = (1, 2),
        profile_count: int = 1,
        profile_type: str = "constant",
        readout_type: str = "profile",
        use_phase_attention: bool = True,
        num_heads: int = 1,
        gate_init: float = 0.2,
        pool_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if profile_count < 1:
            raise ValueError("profile_count must be positive")
        if profile_type not in {"constant", "constant_slope"}:
            raise ValueError("profile_type must be 'constant' or 'constant_slope'")
        if readout_type not in {"profile", "direct"}:
            raise ValueError("readout_type must be 'profile' or 'direct'")
        if phase_dim % num_heads:
            raise ValueError("phase_dim must be divisible by trend attention heads")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if pool_temperature <= 0:
            raise ValueError("pool_temperature must be positive")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.phase_dim = int(phase_dim)
        self.token_dim = int(token_dim)
        self.profile_count = int(profile_count)
        self.readout_type = str(readout_type)
        self.use_phase_attention = bool(use_phase_attention)
        self.pool_temperature = float(pool_temperature)

        hidden = max(8, min(token_dim, 64))
        layers: list[nn.Module] = []
        in_channels = 1
        for dilation in tuple(int(d) for d in dilations):
            conv = nn.Conv1d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            layers.extend([conv, nn.GELU()])
            in_channels = hidden
        if not layers:
            layers = [nn.Conv1d(1, hidden, kernel_size=kernel_size), nn.GELU()]
        self.temporal_stem = nn.ModuleList(layers)
        self.to_token = nn.Linear(hidden, token_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.pool_query, std=0.02)

        self.phase_query = nn.Linear(token_dim, phase_dim)
        self.phase_key = nn.Linear(phase_dim, phase_dim)
        self.phase_value = nn.Linear(phase_dim, phase_dim)
        self.phase_out = nn.Linear(phase_dim, token_dim)
        self.phase_norm = nn.LayerNorm(token_dim)
        self.phase_attention = nn.MultiheadAttention(
            phase_dim, num_heads=num_heads, batch_first=True
        )

        readout_hidden = max(8, token_dim // 2)
        if self.readout_type == "profile":
            self.lambda_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, profile_count),
            )
            self.horizon_head = None
            # Zero initialization makes the added correction an exact no-op at
            # initialization, which is useful for controlled ablations.
            nn.init.zeros_(self.lambda_head[-1].weight)
            nn.init.zeros_(self.lambda_head[-1].bias)
        else:
            self.lambda_head = None
            self.horizon_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, pred_len),
            )
            # Keep direct-readout arms an exact no-op at initialization too.
            nn.init.zeros_(self.horizon_head[-1].weight)
            nn.init.zeros_(self.horizon_head[-1].bias)

        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(
            torch.full((1, 1, self.enc_in), float(torch.logit(torch.tensor(gate_init))))
        )

        profiles = self._make_profiles(self.pred_len, self.profile_count, profile_type)
        self.register_buffer("profile_bank", profiles, persistent=True)
        self.last_pool_weights = None
        self.last_phase_attention = None
        self.last_lambda = None
        self.last_trend_forecast = None

    @staticmethod
    def _make_profiles(pred_len: int, count: int, profile_type: str) -> torch.Tensor:
        t = torch.linspace(-1.0, 1.0, pred_len)
        profiles = [torch.ones_like(t)]
        if count > 1 and profile_type == "constant_slope":
            profiles.append(t)
        if count > 1 and profile_type == "constant":
            profiles.extend(torch.sin(math.pi * (i + 1) * (t + 1.0) / 2.0) for i in range(count - 1))
        if count > 2 and profile_type == "constant_slope":
            profiles.extend(t.pow(i) for i in range(2, count))
        bank = torch.stack(profiles[:count], dim=-1)
        bank = bank / bank.square().sum(dim=0, keepdim=True).clamp_min(1e-8).sqrt()
        return bank

    def _encode(self, x_in: torch.Tensor) -> torch.Tensor:
        # (B,L,C) -> (B*C,1,L); one shared encoder is used for all channels.
        b, l, c = x_in.shape
        h = x_in.permute(0, 2, 1).reshape(b * c, 1, l)
        for layer in self.temporal_stem:
            if isinstance(layer, nn.Conv1d):
                h = _causal_conv1d(h, layer)
            else:
                h = layer(h)
        h = h.transpose(1, 2)  # (B*C,L,H)
        scores = torch.einsum("blh,bqh->blq", h, self.pool_query.expand(b * c, -1, -1))
        scores = scores.squeeze(-1) / self.pool_temperature
        weights = scores.softmax(dim=-1)
        token = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        token = self.to_token(token).reshape(b, c, self.token_dim)
        self.last_pool_weights = weights.reshape(b, c, l).detach()
        return token

    def forward(self, x_in: torch.Tensor, phase_latent: torch.Tensor):
        if x_in.ndim != 3 or phase_latent.ndim != 4:
            raise ValueError("expected x_in=(B,L,C), phase_latent=(B,C,S,D)")
        b, _, c = x_in.shape
        if c != self.enc_in or phase_latent.shape[:2] != (b, c):
            raise ValueError("trend token input channel/batch dimensions do not match")
        token = self._encode(x_in)
        if self.use_phase_attention:
            s = phase_latent.shape[2]
            q = self.phase_query(token.reshape(b * c, 1, self.token_dim))
            phase = phase_latent.reshape(b * c, s, self.phase_dim)
            k = self.phase_key(phase)
            v = self.phase_value(phase)
            attended, weights = self.phase_attention(q, k, v, need_weights=True)
            token = self.phase_norm(token + self.phase_out(attended.squeeze(1)).reshape(b, c, self.token_dim))
            self.last_phase_attention = weights.reshape(b, c, s).detach()
        else:
            self.last_phase_attention = None
        profile = self.profile_bank.to(device=x_in.device, dtype=x_in.dtype)
        if self.readout_type == "profile":
            lam = self.lambda_head(token)
            delta = torch.einsum("bcm,hm->bhc", lam, profile)
            direct_delta = None
        else:
            lam = None
            # (B, C, D) -> (B, C, H) -> (B, H, C)
            direct_delta = self.horizon_head(token)
            delta = direct_delta.transpose(1, 2)
        anchor = x_in[:, -1:, :].expand(-1, self.pred_len, -1)
        trend = anchor + delta
        gate = torch.sigmoid(self.gate_logit).to(dtype=x_in.dtype)
        fused = trend * gate + (1.0 - gate) * 0.0  # caller supplies phase forecast
        self.last_lambda = lam.detach() if lam is not None else None
        self.last_trend_forecast = trend.detach()
        return trend, gate, {
            "lambda": lam,
            "direct_delta": direct_delta,
            "readout_type": self.readout_type,
            "pool_weights": self.last_pool_weights,
            "phase_attention": self.last_phase_attention,
            "profile_bank": profile,
            "fused_placeholder": fused,
        }
