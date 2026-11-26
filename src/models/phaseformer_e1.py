"""PhaseFormer-E1: one set of phase tokens read by a phase branch and a trend branch.

    x --RevIN--> Fold by Cycle X~ (K x P) --Phase Token Embedding--> U (d x P, no LayerNorm)

    Phase branch   CrossPhaseRouting   P tokens -> M routers -> P tokens h_p
                   PhaseDecoder        each phase token -> its K' future cycles: F_p
    Trend branch   CrossPhasePooling   mean over P of U -> trend state g (r = 3)
                   TrendCorrection     F_b = m_K 1 + Psi_H(g), anchored on the last cycle mean m_K;
                                       dF = (g_l M_c + g_s J_c)(F_b - F_p)
    Forecast       y = F_p + dF, de-normalized

M_c keeps the per-cycle mean of a forecast and J_c the within-cycle remainder; g_l and g_s
are two static gates. The module follows the E1 arm of ``scripts/phaseformer_e12.py``,
which builds E1 by patching the legacy ``PhaseFormer`` (experiments still run that path);
``PhaseFormerE1.from_legacy`` copies the weights of such a model.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.layers.SelfAttention_Family import AttentionLayer, FullAttention
from src.models.phase_adapters import RevIN

PROFILE_COUNT = 3


def legendre_profiles(horizon, count=PROFILE_COUNT):
    """Displacement, tilt and curvature on the horizon, (H, count), unit RMS columns."""
    tau = torch.linspace(-1.0, 1.0, horizon, dtype=torch.float64)
    columns = []
    for power in range(count):
        v = tau ** power
        for u in columns:
            v = v - (v @ u) / (u @ u) * u
        columns.append(v / v.square().mean().sqrt())
    return torch.stack(columns, -1).float()


# ------------------------------------------------------------------ shared tokens
class PhaseTokenEmbedding(nn.Module):
    """U = W_e X~ + b_e: each phase slot's K cycle values become a d-dim token."""

    def __init__(self, cycles, width):
        super().__init__()
        self.proj = nn.Linear(cycles, width)

    def forward(self, series):  # (B, C, P, K) -> (B, C, P, d)
        return self.proj(series)


# ------------------------------------------------------------------ phase branch
class RoutingLayer(nn.Module):
    """M routers aggregate the P tokens and distribute back to them; Post-LN."""

    def __init__(self, width, routers, heads, period, dropout, position):
        super().__init__()
        self.period = period
        self.router = nn.Parameter(torch.empty(routers, width))
        nn.init.trunc_normal_(self.router, std=0.02)
        self.position = position
        if position:
            self.pos_embedding = nn.Parameter(torch.empty(period, width))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(dropout)
        self.aggregate = AttentionLayer(FullAttention(False, factor=5, attention_dropout=dropout,
                                                      output_attention=False), width, heads)
        self.distribute = AttentionLayer(FullAttention(False, factor=5, attention_dropout=dropout,
                                                       output_attention=False), width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):  # (B, C, P, d)
        b, c, slots, d = z.shape
        x = z.reshape(b * c, slots, d)
        if self.position:
            x = self.pos_dropout(x + self.pos_embedding[torch.arange(slots, device=z.device) % self.period])
        routers = self.router.unsqueeze(0).expand(b * c, -1, -1)
        buffer, _ = self.aggregate(routers, x, x, attn_mask=None)
        received, _ = self.distribute(x, buffer, buffer, attn_mask=None)
        x = self.norm1(x + self.dropout(received))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x.reshape(b, c, slots, d)


class CrossPhaseRouting(nn.Module):
    """N_r routing layers over the P phase tokens: U -> h_p.

    Between layers the tokens are read back to K values per slot and embedded again
    (Linear + LayerNorm) as an additive update of the token stream.
    """

    def __init__(self, cycles, width, layers, routers, heads, period, dropout, position):
        super().__init__()
        self.layers = nn.ModuleList(RoutingLayer(width, routers, heads, period, dropout, position)
                                    for _ in range(layers))
        self.readback = nn.ModuleList(nn.Linear(width, cycles) for _ in range(layers - 1))
        self.reembed = nn.ModuleList(nn.Sequential(nn.Linear(cycles, width), nn.LayerNorm(width))
                                     for _ in range(layers - 1))

    def forward(self, u):
        z = u
        for index, layer in enumerate(self.layers):
            if index:
                z = z + self.reembed[index - 1](series)
            z = layer(z)
            if index < len(self.readback):
                series = self.readback[index](z)
        return z


class PhaseDecoder(nn.Module):
    """F_p: one linear map, used by every phase slot, from h_p to the K' future cycles."""

    def __init__(self, width, out_cycles, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj = nn.Linear(width, out_cycles)

    def forward(self, h, pred_len):  # (B, C, P, d) -> (B, H, C)
        b, c = h.shape[:2]
        steps = self.proj(self.dropout(h))                     # (B, C, P, K')
        return steps.transpose(2, 3).reshape(b, c, -1)[..., :pred_len].transpose(1, 2)


# ------------------------------------------------------------------ trend branch
class CrossPhasePooling(nn.Module):
    """g = W_g mean_P(U): the P tokens pooled into one rank-r trend state."""

    def __init__(self, width, rank):
        super().__init__()
        self.proj = nn.Linear(width, rank)

    def forward(self, u):  # (B, C, P, d) -> (B, C, r)
        return self.proj(u.mean(2))


class TrendCorrection(nn.Module):
    """dF = (g_l M_c + g_s J_c)(F_b - F_p), with F_b = m_K 1 + Psi_H(g).

    ``decoder='legendre'`` writes g onto three fixed profiles (displacement, tilt,
    curvature); ``'learned'`` uses a free linear map to the horizon. Both start at zero,
    so F_b starts as the flat line at m_K.
    """

    def __init__(self, rank, pred_len, period, channels, decoder, gate_init, shared_gate):
        super().__init__()
        self.pred_len, self.period, self.decoder_kind = pred_len, period, decoder
        out = PROFILE_COUNT if decoder == 'legendre' else pred_len
        self.decoder = nn.Linear(rank, out)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        if decoder == 'legendre':
            self.register_buffer('profiles', legendre_profiles(pred_len), persistent=False)
        gates = 1 if shared_gate else channels
        logit = float(torch.logit(torch.tensor(gate_init)))
        self.level_logit = nn.Parameter(torch.full((1, gates, 1), logit))
        self.shape_offset = nn.Parameter(torch.zeros(1, gates, 1))

    def trend_forecast(self, g, anchor):  # (B, C, r), (B, 1, C) -> F_b (B, H, C)
        weights = self.decoder(g)
        delta = weights @ self.profiles.T if self.decoder_kind == 'legendre' else weights
        return delta.transpose(1, 2) + anchor

    def gates(self):
        return self.level_logit.sigmoid(), (self.level_logit + self.shape_offset).sigmoid()

    def forward(self, g, anchor, phase_forecast):
        trend_forecast = self.trend_forecast(g, anchor)
        b, h, c = phase_forecast.shape
        gap = (trend_forecast - phase_forecast).reshape(b, h // self.period, self.period, c)
        level = gap.mean(2, keepdim=True)                       # M_c
        g_level, g_shape = (gate.transpose(1, 2)[:, None] for gate in self.gates())   # (1, 1, 1, C)
        correction = g_level * level + g_shape * (gap - level)  # g_l M_c + g_s J_c
        return correction.reshape(b, h, c), trend_forecast


# ------------------------------------------------------------------ model
class PhaseFormerE1(nn.Module):

    def __init__(self, *, seq_len, pred_len, channels, period, width, layers, routers, heads,
                 dropout=0.1, position=True, predictor_dropout=0.0, rank=3, decoder='legendre',
                 gate_init=0.2, shared_gate=False):
        super().__init__()
        if pred_len % period:
            raise ValueError('the gates need whole forecast cycles')
        self.seq_len, self.pred_len, self.period = seq_len, pred_len, period
        self.cycles = -(-seq_len // period)
        self.revin = RevIN(channels)
        self.embedding = PhaseTokenEmbedding(self.cycles, width)
        self.routing = CrossPhaseRouting(self.cycles, width, layers, routers, heads, period, dropout, position)
        self.phase_decoder = PhaseDecoder(width, pred_len // period, predictor_dropout)
        self.pooling = CrossPhasePooling(width, rank)
        self.correction = TrendCorrection(rank, pred_len, period, channels, decoder, gate_init, shared_gate)

    def fold(self, x):  # (B, L, C) -> X~ as (B, C, P, K), ring-padded to whole cycles
        series = x.transpose(1, 2)
        pad = self.cycles * self.period - self.seq_len
        if pad:
            series = nn.functional.pad(series, (0, pad), mode='circular')
        b, c, _ = series.shape
        return series.reshape(b, c, self.cycles, self.period).transpose(2, 3)

    def forward(self, x):  # (B, L, C) -> (B, H, C)
        x, stats = self.revin.normalize(x)
        u = self.embedding(self.fold(x))
        phase_forecast = self.phase_decoder(self.routing(u), self.pred_len)
        anchor = x[:, -self.period:].mean(1, keepdim=True)       # m_K
        correction, _ = self.correction(self.pooling(u), anchor, phase_forecast)
        return self.revin.denormalize(phase_forecast + correction, stats)

    @classmethod
    def from_legacy(cls, model):
        """Copy a legacy PhaseFormer patched to E1 (``scripts/phaseformer_e12.py``)."""
        units = model.routing_layers
        first = units[0].interact
        branch = model.lowrank_interaction_branch.lowrank_head
        learned = hasattr(branch, 'decoder') and isinstance(branch.decoder, nn.Linear)
        gate = model.lowrank_interaction_gate
        if (first.use_circular_pos or first.use_circular_attn_bias or model.predictor.use_mlp
                or model.embedding.use_mlp or not model.use_revin or model.revin_affine
                or model.revin_eps != 1e-5):
            raise ValueError('options outside PhaseFormer-E1')
        new = cls(seq_len=model.seq_len, pred_len=model.pred_len, channels=model.enc_in,
                  period=model.period_len, width=model.latent_dim, layers=len(units),
                  routers=first.num_routers, heads=first.num_heads, dropout=first.dropout,
                  position=first.use_pos_embed, predictor_dropout=getattr(model.predictor.dropout, 'p', 0.0),
                  rank=branch.encoder.out_features, decoder='learned' if learned else 'legendre',
                  shared_gate=gate.numel() == 1)
        state = {'embedding.proj': model.embedding.projection,
                 'phase_decoder.proj': model.predictor.decoder,
                 'pooling.proj': branch.encoder,
                 'correction.decoder': branch.decoder if learned else branch.mix}
        for index, unit in enumerate(units):
            layer, old = new.routing.layers[index], unit.interact
            layer.router.data.copy_(old.router)
            if old.use_pos_embed:
                layer.pos_embedding.data.copy_(old.pos_embedding)
            for mine, theirs in (('aggregate', 'router_sender'), ('distribute', 'router_receiver'),
                                 ('norm1', 'norm1'), ('norm2', 'norm2'), ('mlp', 'mlp')):
                getattr(layer, mine).load_state_dict(getattr(old, theirs).state_dict())
            if unit.in_proj is not None:
                state[f'routing.reembed.{index - 1}'] = unit.in_proj
            if unit.out_proj is not None:
                state[f'routing.readback.{index}'] = unit.out_proj
        for name, module in state.items():
            new.get_submodule(name).load_state_dict(module.state_dict())
        new.correction.level_logit.data.copy_(gate.detach().reshape(1, -1, 1))
        new.correction.shape_offset.data.copy_(
            model.lowrank_dynamic_fusion.shape_bias_delta.detach().reshape(1, -1, 1))
        return new
