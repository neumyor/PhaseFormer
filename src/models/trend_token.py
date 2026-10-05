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
    directly to all future steps by a shared horizon head.  The additional
    mechanisms are opt-in and are deliberately orthogonal so each hypothesis
    can be trained against the same B2 configuration.
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
        phase_attention_mode: str = "dense",
        num_heads: int = 1,
        gate_init: float = 0.2,
        gate_mode: str = "static",
        pool_temperature: float = 1.0,
        token_count: int = 1,
        direct_smooth_window: int = 0,
        history_multiscale: bool = False,
        history_robust: bool = False,
    ) -> None:
        super().__init__()
        if profile_count < 1:
            raise ValueError("profile_count must be positive")
        if profile_type not in {"constant", "constant_slope"}:
            raise ValueError("profile_type must be 'constant' or 'constant_slope'")
        if readout_type not in {"profile", "direct"}:
            if readout_type != "factorized":
                raise ValueError(
                    "readout_type must be 'profile', 'direct', or 'factorized'"
                )
        if phase_attention_mode not in {"dense", "null"}:
            raise ValueError("phase_attention_mode must be 'dense' or 'null'")
        if gate_mode not in {"static", "reliability"}:
            raise ValueError("gate_mode must be 'static' or 'reliability'")
        if token_count < 1 or token_dim % token_count:
            raise ValueError("token_count must divide token_dim")
        if direct_smooth_window < 0 or (
            direct_smooth_window > 0 and direct_smooth_window % 2 == 0
        ):
            raise ValueError("direct_smooth_window must be zero or a positive odd integer")
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
        self.phase_attention_mode = str(phase_attention_mode)
        self.gate_mode = str(gate_mode)
        self.token_count = int(token_count)
        self.token_part_dim = self.token_dim // self.token_count
        self.direct_smooth_window = int(direct_smooth_window)
        self.history_multiscale = bool(history_multiscale)
        self.history_robust = bool(history_robust)
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
        self.pool_hidden = hidden
        if self.history_multiscale:
            # A separate long-receptive-field path is the only change in the
            # multiscale candidate; the B2 default does not instantiate it.
            self.long_temporal_stem = nn.ModuleList([
                nn.Conv1d(1, hidden, kernel_size=kernel_size, dilation=8),
                nn.GELU(),
            ])
            self.pool_hidden *= 2
        else:
            self.long_temporal_stem = None
        self.to_token = nn.Linear(self.pool_hidden, self.token_part_dim)
        self.pool_query = nn.Parameter(
            torch.zeros(self.token_count, 1, self.pool_hidden)
        )
        nn.init.normal_(self.pool_query, std=0.02)

        self.phase_query = nn.Linear(token_dim, phase_dim)
        self.phase_key = nn.Linear(phase_dim, phase_dim)
        self.phase_value = nn.Linear(phase_dim, phase_dim)
        self.phase_out = nn.Linear(phase_dim, token_dim)
        self.phase_norm = nn.LayerNorm(token_dim)
        self.phase_attention = nn.MultiheadAttention(
            phase_dim, num_heads=num_heads, batch_first=True
        )
        if self.phase_attention_mode == "null":
            self.phase_null = nn.Parameter(torch.zeros(1, 1, phase_dim))
            nn.init.normal_(self.phase_null, std=0.02)
        else:
            self.phase_null = None

        readout_hidden = max(8, token_dim // 2)
        if self.readout_type == "profile":
            self.level_head = None
            self.shape_head = None
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
        elif self.readout_type == "direct":
            self.lambda_head = None
            self.level_head = None
            self.shape_head = None
            self.horizon_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, pred_len),
            )
            # Keep direct-readout arms an exact no-op at initialization too.
            nn.init.zeros_(self.horizon_head[-1].weight)
            nn.init.zeros_(self.horizon_head[-1].bias)
        else:
            self.lambda_head = None
            self.horizon_head = None
            readout_norm = nn.LayerNorm(token_dim)
            self.level_head = nn.Sequential(
                readout_norm,
                nn.Linear(token_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 1),
            )
            self.shape_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, pred_len),
            )
            nn.init.zeros_(self.level_head[-1].weight)
            nn.init.zeros_(self.level_head[-1].bias)
            nn.init.zeros_(self.shape_head[-1].weight)
            nn.init.zeros_(self.shape_head[-1].bias)

        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        if self.gate_mode == "static":
            self.gate_logit = nn.Parameter(
                torch.full((1, 1, self.enc_in), float(torch.logit(torch.tensor(gate_init))))
            )
            self.gate_head = None
        else:
            self.gate_logit = None
            self.gate_head = nn.Sequential(
                nn.Linear(self.token_dim + 2, max(8, self.token_dim // 2)),
                nn.GELU(),
                nn.Linear(max(8, self.token_dim // 2), 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, float(torch.logit(torch.tensor(gate_init))))

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
        if self.history_robust:
            center = x_in.median(dim=1, keepdim=True).values
            mad = (x_in - center).abs().median(dim=1, keepdim=True).values
            limit = (4.0 * mad).clamp_min(1e-3)
            x_in = center + (x_in - center).clamp(-limit, limit)
        h = x_in.permute(0, 2, 1).reshape(b * c, 1, l)
        for layer in self.temporal_stem:
            if isinstance(layer, nn.Conv1d):
                h = _causal_conv1d(h, layer)
            else:
                h = layer(h)
        if self.long_temporal_stem is not None:
            h_long = x_in.permute(0, 2, 1).reshape(b * c, 1, l)
            for layer in self.long_temporal_stem:
                if isinstance(layer, nn.Conv1d):
                    h_long = _causal_conv1d(h_long, layer)
                else:
                    h_long = layer(h_long)
            h = torch.cat([h, h_long], dim=1)
        h = h.transpose(1, 2)  # (B*C,L,H)
        query = self.pool_query.unsqueeze(0).expand(b * c, -1, -1, -1)
        query = query.reshape(b * c, self.token_count, self.pool_hidden)
        scores = torch.einsum("blh,bmh->bml", h, query) / self.pool_temperature
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bml,blh->bmh", weights, h)
        token = self.to_token(pooled).reshape(b, c, self.token_count, self.token_part_dim)
        token = token.reshape(b, c, self.token_dim)
        if self.token_count == 1:
            self.last_pool_weights = weights.reshape(b, c, l).detach()
        else:
            self.last_pool_weights = weights.reshape(b, c, self.token_count, l).detach()
        return token

    def forward(self, x_in: torch.Tensor, phase_latent: torch.Tensor):
        if x_in.ndim != 3 or phase_latent.ndim != 4:
            raise ValueError("expected x_in=(B,L,C), phase_latent=(B,C,S,D)")
        b, _, c = x_in.shape
        if c != self.enc_in or phase_latent.shape[:2] != (b, c):
            raise ValueError("trend token input channel/batch dimensions do not match")
        token = self._encode(x_in)
        phase_attention_for_gate = None
        if self.use_phase_attention:
            s = phase_latent.shape[2]
            q = self.phase_query(token.reshape(b * c, 1, self.token_dim))
            phase = phase_latent.reshape(b * c, s, self.phase_dim)
            if self.phase_null is not None:
                phase = torch.cat([
                    phase,
                    self.phase_null.expand(b * c, -1, -1).to(dtype=phase.dtype),
                ], dim=1)
            k = self.phase_key(phase)
            v = self.phase_value(phase)
            attended, weights = self.phase_attention(q, k, v, need_weights=True)
            token = self.phase_norm(token + self.phase_out(attended.squeeze(1)).reshape(b, c, self.token_dim))
            phase_attention_for_gate = weights.reshape(b, c, phase.shape[1])
            self.last_phase_attention = phase_attention_for_gate.detach()
        else:
            self.last_phase_attention = None
        profile = self.profile_bank.to(device=x_in.device, dtype=x_in.dtype)
        if self.readout_type == "profile":
            lam = self.lambda_head(token)
            delta = torch.einsum("bcm,hm->bhc", lam, profile)
            direct_delta = None
        elif self.readout_type == "direct":
            lam = None
            # (B, C, D) -> (B, C, H) -> (B, H, C)
            direct_delta = self.horizon_head(token)
            if self.direct_smooth_window > 1:
                direct_delta = F.avg_pool1d(
                    direct_delta,
                    kernel_size=self.direct_smooth_window,
                    stride=1,
                    padding=self.direct_smooth_window // 2,
                    count_include_pad=False,
                )
            delta = direct_delta.transpose(1, 2)
        else:
            lam = None
            level = self.level_head(token)
            shape = self.shape_head(token)
            shape = shape - shape.mean(dim=-1, keepdim=True)
            direct_delta = level + shape
            delta = direct_delta.transpose(1, 2)
        anchor = x_in[:, -1:, :].expand(-1, self.pred_len, -1)
        trend = anchor + delta
        if self.gate_mode == "static":
            gate = torch.sigmoid(self.gate_logit).to(dtype=x_in.dtype)
        else:
            if phase_attention_for_gate is None:
                entropy = torch.zeros((b, c, 1), device=x_in.device, dtype=x_in.dtype)
            else:
                weights = phase_attention_for_gate.to(dtype=x_in.dtype)
                entropy = -(weights.clamp_min(1e-8).log() * weights).sum(-1, keepdim=True)
            volatility = x_in.diff(dim=1).std(dim=1, keepdim=True).transpose(1, 2)
            features = torch.cat([token, entropy, volatility], dim=-1)
            gate = torch.sigmoid(self.gate_head(features)).transpose(1, 2)
        fused = trend * gate + (1.0 - gate) * 0.0  # caller supplies phase forecast
        self.last_lambda = lam.detach() if lam is not None else None
        self.last_trend_forecast = trend.detach()
        return trend, gate, {
            "lambda": lam,
            "direct_delta": direct_delta,
            "readout_type": self.readout_type,
            "gate_mode": self.gate_mode,
            "phase_attention_mode": self.phase_attention_mode,
            "token_count": self.token_count,
            "pool_weights": self.last_pool_weights,
            "phase_attention": self.last_phase_attention,
            "profile_bank": profile,
            "fused_placeholder": fused,
        }
