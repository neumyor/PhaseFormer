"""PhaseFormer+ (M0) and the bi-axial variants, one channel-independent model.

Shapes (N = batch x channels, K input cycles, P phases, K' output cycles):
  window x (N, L), folded X (N, K, P), forecast (N, H), H = K' * P.

Both families share the normalization, the trend pathway, and the two-gate
fusion. They differ in how the phase forecast F is produced:
  M0       embed each phase token (K -> d), stacked cross-phase routing, predictor.
  Bi-axial X -> W_t X W_p (k x p x q) -> embed -> (k, p, d) tokens -> routers along
           the cycle axis and/or the phase axis -> readout over (k, d) per phase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Cfg:
    seq_len: int = 720
    pred_len: int = 96
    channels: int = 7
    period: int = 24
    arch: str = 'm0'            # m0 | phase | temporal | dual | interleaved
    layers: int = 3
    d: int = 4
    routers: int = 8
    heads: int = 1
    dropout: float = 0.1
    # bi-axial input representation
    k: int = 8
    p: int = 24                 # 0 means p = period
    q: int = 1
    wp: str = 'learn'           # learn | identity
    gate_init: float = 0.2
    use_trend: bool = True
    w_init: str = 'random'      # random | poly: first rows of W_t are polynomials over the cycle index
    decoder: str = 'd0'         # d0 phase+trend+gates | direct | basis_fixed | basis_learn | hybrid_fixed | hybrid_learn
    anchor: str = 'x_last'      # x_last | last_cycle | last_cycle_mean | none (reference of the direct decoders)
    mean_token: bool = False    # inject the original mean phase token into every latent cell (E5 ablation)
    norm: str = 'post'          # post: residual then LayerNorm | pre: LayerNorm before each sublayer, un-normalized residual stream
    embed: str = 'mlp_ln'       # mlp_ln: Linear-GELU-Linear-LayerNorm | lin_mlp: Linear + Linear-GELU-Linear, no LayerNorm
    phase_id: str = 'none'      # none | fourier: sin/cos harmonics of the phase plus a cycle-slot embedding added at the input
    harmonics: int = 8          # Fourier harmonics of the phase (capped at p // 2)


def legendre_profiles(horizon: int, count: int = 3) -> torch.Tensor:
    tau = torch.linspace(-1.0, 1.0, horizon, dtype=torch.float64)
    columns = []
    for power in range(count):
        v = tau ** power
        for u in columns:
            v = v - (v @ u) / (u @ u) * u
        columns.append(v / v.square().mean().sqrt())
    return torch.stack(columns, -1).float()


class RouterAttn(nn.Module):
    """Gather-distribute routing over T tokens, with position embedding, residual and LayerNorm.

    Two multi-head attentions written with einsum: the routers query the tokens (gather), then
    the tokens query the routers' summaries (distribute). The router queries do not depend on the
    batch, so the gather stage never materializes per-sample queries.
    """

    def __init__(self, d, tokens, routers, heads, dropout, pre=False):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by the number of heads')
        self.h, self.dropout, self.pre = heads, dropout, pre
        self.router = nn.Parameter(torch.empty(routers, d))
        nn.init.trunc_normal_(self.router, std=0.02)
        self.pos = nn.Parameter(torch.empty(tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.q1, self.kv1, self.o1 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.keep_attn = False

    def _attend(self, q, kv):
        """q (N or 1, Tq, d), kv (N, Tk, 2d) -> (N, Tq, d)."""
        k, v = kv.chunk(2, -1)
        h = self.h
        split = lambda t: t.reshape(t.shape[0], t.shape[1], h, -1)
        q, k, v = split(q), split(k), split(v)
        scores = torch.einsum('nqhe,nkhe->nhqk', q.expand(k.shape[0], -1, -1, -1), k) / math.sqrt(q.shape[-1])
        weights = F.dropout(scores.softmax(-1), self.dropout, self.training)
        if self.keep_attn:
            self.last_weights.append(weights.detach())
        return torch.einsum('nhqk,nkhe->nqhe', weights, v).reshape(k.shape[0], q.shape[1], -1)

    def forward(self, x):  # (N, T, d)
        self.last_weights = []
        if self.pre:   # the position and the LayerNorm act on the sublayer input only; the residual stream keeps its scale
            y = self.norm(x) + self.pos
            buf = self.o1(self._attend(self.q1(self.router).unsqueeze(0), self.kv1(y)))
            return x + self.drop(self.o2(self._attend(self.q2(y), self.kv2(buf))))
        x = x + self.pos
        buf = self.o1(self._attend(self.q1(self.router).unsqueeze(0), self.kv1(x)))
        out = self.o2(self._attend(self.q2(x), self.kv2(buf)))
        return self.norm(x + self.drop(out))


class FFN(nn.Module):
    def __init__(self, d, dropout, pre=False):
        super().__init__()
        self.pre = pre
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.pre:
            return x + self.drop(self.mlp(self.norm(x)))
        return self.norm(x + self.drop(self.mlp(x)))


class RouterLayer(nn.Module):
    """One routing layer of the original PhaseFormer: routing, then the FFN."""

    def __init__(self, d, tokens, routers, heads, dropout):
        super().__init__()
        self.attn = RouterAttn(d, tokens, routers, heads, dropout)
        self.ffn = FFN(d, dropout)

    def forward(self, x):
        return self.ffn(self.attn(x))


class PhaseNetM0(nn.Module):
    """Original phase pathway: per-phase embedding, chained routing layers, shared predictor."""

    def __init__(self, c: Cfg, K: int, P: int, Kp: int):
        super().__init__()
        self.embed = nn.Linear(K, c.d)
        self.embed_norm = nn.LayerNorm(c.d)
        self.layers = nn.ModuleList(RouterLayer(c.d, P, c.routers, c.heads, c.dropout)
                                    for _ in range(c.layers))
        self.in_proj = nn.ModuleList(nn.Sequential(nn.Linear(K, c.d), nn.LayerNorm(c.d))
                                     for _ in range(c.layers - 1))
        self.out_proj = nn.ModuleList(nn.Linear(c.d, K) for _ in range(c.layers - 1))
        self.predictor = nn.Linear(c.d, Kp)

    def forward(self, X):  # X (N, K, P)
        phase = X.transpose(1, 2)                        # (N, P, K) phase tokens
        emb = self.embed_norm(self.embed(phase))
        z, cur = emb, phase
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = z + self.in_proj[i - 1](cur)
            z = layer(z)
            if i < len(self.layers) - 1:
                cur = self.out_proj[i](z)
        return self.predictor(z), emb.mean(1)            # (N, P, K'), (N, d)


class BiAxialInput(nn.Module):
    """Z^(h) = W_t^(h) X W_p^(h) + B^(h), embedded to (N, k, p, d)."""

    def __init__(self, c: Cfg, K: int, P: int):
        super().__init__()
        self.k, self.p, self.q = c.k, (c.p or P), c.q
        wt = torch.randn(self.q, self.k, K) / math.sqrt(K)
        if c.w_init == 'poly':
            basis = legendre_profiles(K, min(self.k, 4)).T / math.sqrt(K)   # (<=4, K), unit RMS rows
            wt[:, :basis.shape[0]] = basis
        self.Wt = nn.Parameter(wt)
        if c.wp == 'identity':
            if self.p != P:
                raise ValueError('identity W_p requires p = P')
            self.register_buffer('Wp', torch.eye(P).expand(self.q, P, P).clone(), persistent=False)
        else:
            self.Wp = nn.Parameter(torch.randn(self.q, P, self.p) / math.sqrt(P))
        self.bias = nn.Parameter(torch.zeros(self.q, self.k, self.p))
        if c.embed == 'mlp_ln':
            self.embed = nn.Sequential(nn.Linear(self.q, c.d), nn.GELU(), nn.Linear(c.d, c.d), nn.LayerNorm(c.d))
        elif c.embed == 'lin_mlp':   # the linear path keeps the amplitude of the cell value exactly
            self.lin = nn.Linear(self.q, c.d)
            self.embed = nn.Sequential(nn.Linear(self.q, c.d), nn.GELU(), nn.Linear(c.d, c.d))
        else:
            raise ValueError(c.embed)
        self.linear_embed = c.embed == 'lin_mlp'
        if c.phase_id == 'fourier':
            m = max(1, min(c.harmonics, self.p // 2))
            angle = 2 * math.pi * torch.arange(self.p)[:, None] * torch.arange(1, m + 1)[None] / self.p
            self.register_buffer('phase_feat', torch.cat([angle.sin(), angle.cos()], -1), persistent=False)
            self.phase_proj = nn.Linear(2 * m, c.d)
            self.slot = nn.Parameter(torch.empty(self.k, c.d))
            nn.init.trunc_normal_(self.slot, std=0.02)
        elif c.phase_id != 'none':
            raise ValueError(c.phase_id)
        self.phase_id = c.phase_id

    def forward(self, X):  # (N, K, P)
        Z = torch.einsum('qkc,ncP,qPp->nkpq', self.Wt, X, self.Wp) + self.bias.permute(1, 2, 0)
        h = self.embed(Z)
        if self.linear_embed:
            h = h + self.lin(Z)
        if self.phase_id == 'fourier':                    # the identity is part of the token, so both axes see it
            h = h + self.phase_proj(self.phase_feat)[None, None] + self.slot[None, :, None]
        return h                                          # (N, k, p, d)


class AxialBlock(nn.Module):
    """Routing along one axis of (N, k, p, d), then the FFN. axis: 'temporal' (over k) or 'phase' (over p)."""

    def __init__(self, c: Cfg, axis: str, k: int, p: int):
        super().__init__()
        self.axis = axis
        pre = c.norm == 'pre'
        self.attn = RouterAttn(c.d, k if axis == 'temporal' else p, c.routers, c.heads, c.dropout, pre)
        self.ffn = FFN(c.d, c.dropout, pre)

    def forward(self, h):
        return self.ffn(_along(self.attn, h, self.axis))


def _along(fn, h, axis):
    N, k, p, d = h.shape
    if axis == 'phase':
        return fn(h.reshape(N * k, p, d)).reshape(N, k, p, d)
    t = fn(h.transpose(1, 2).reshape(N * p, k, d))
    return t.reshape(N, p, k, d).transpose(1, 2)


class InterleavedBlock(nn.Module):
    """Temporal routing, phase routing, then one FFN."""

    def __init__(self, c: Cfg, k: int, p: int):
        super().__init__()
        pre = c.norm == 'pre'
        self.temporal = RouterAttn(c.d, k, c.routers, c.heads, c.dropout, pre)
        self.phase = RouterAttn(c.d, p, c.routers, c.heads, c.dropout, pre)
        self.ffn = FFN(c.d, c.dropout, pre)

    def forward(self, h):
        h = _along(self.temporal, h, 'temporal')
        h = _along(self.phase, h, 'phase')
        return self.ffn(h)


class PhaseNetBiAxial(nn.Module):
    def __init__(self, c: Cfg, K: int, P: int, Kp: int):
        super().__init__()
        self.inp = BiAxialInput(c, K, P)
        k, p = self.inp.k, self.inp.p
        self.arch = c.arch
        if c.arch in ('phase', 'temporal'):
            self.stack = nn.ModuleList(AxialBlock(c, c.arch, k, p) for _ in range(c.layers))
        elif c.arch == 'dual':
            self.stack = nn.ModuleList(AxialBlock(c, 'temporal', k, p) for _ in range(c.layers))
            self.stack2 = nn.ModuleList(AxialBlock(c, 'phase', k, p) for _ in range(c.layers))
            self.fuse = nn.Parameter(torch.zeros(c.d))
        elif c.arch == 'interleaved':
            self.stack = nn.ModuleList(InterleavedBlock(c, k, p) for _ in range(c.layers))
        elif c.arch == 'hybrid':
            # first half: independent branches; second half: interleaved blocks
            half = c.layers // 2
            self.stack = nn.ModuleList(AxialBlock(c, 'temporal', k, p) for _ in range(half))
            self.stack2 = nn.ModuleList(AxialBlock(c, 'phase', k, p) for _ in range(half))
            self.fuse = nn.Parameter(torch.zeros(c.d))
            self.tail = nn.ModuleList(InterleavedBlock(c, k, p) for _ in range(c.layers - half))
        else:
            raise ValueError(c.arch)
        if c.mean_token:
            self.token_embed = nn.Sequential(nn.Linear(K, c.d), nn.LayerNorm(c.d))
            self.token_write = nn.Linear(c.d, c.d)
        self.mean_token = c.mean_token
        self.expand = nn.Linear(p, P) if p != P else None
        self.final_norm = nn.LayerNorm(c.d) if c.norm == 'pre' else None
        self.predictor = nn.Linear(k * c.d, Kp)

    def forward(self, X):
        h0 = self.inp(X)
        if self.mean_token:
            token = self.token_embed(X.transpose(1, 2)).mean(1)       # (N, d): mean phase token
            h0 = h0 + self.token_write(token)[:, None, None, :]
        h = h0
        if self.arch in ('dual', 'hybrid'):
            t, s = h0, h0
            for bt, bp in zip(self.stack, self.stack2):
                t, s = bt(t), bp(s)
            g = torch.sigmoid(self.fuse)
            h = g * t + (1 - g) * s
            if self.arch == 'hybrid':
                for block in self.tail:
                    h = block(h)
        else:
            for block in self.stack:
                h = block(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.expand is not None:
            h = self.expand(h.transpose(2, 3)).transpose(2, 3)   # (N, k, P, d)
        N, k, P, d = h.shape
        z = h.permute(0, 2, 1, 3).reshape(N, P, k * d)
        self.pooled = z.mean(1)                                   # (N, k*d), read by the basis decoders
        return self.predictor(z), h0.mean((1, 2))


class TrendPathway(nn.Module):
    """Level token [pooled window ; mean phase token] -> 3-d state -> polynomial profiles."""

    def __init__(self, seq_len, pred_len, period, d):
        super().__init__()
        self.f = period
        self.encoder = nn.Linear(math.ceil(seq_len / period) + d, 3)
        self.mix = nn.Linear(3, 3)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.register_buffer('profiles', legendre_profiles(pred_len), persistent=False)

    def forward(self, x, zbar):  # x (N, L) normalized
        last = x[:, -1:]
        pooled = F.adaptive_avg_pool1d((x - last).unsqueeze(1), math.ceil(x.shape[1] / self.f)).squeeze(1)
        t = self.encoder(torch.cat([pooled, zbar], -1))
        return last + self.mix(t) @ self.profiles.T


def fold_window(x, K, P):
    """Windows (B, L, C) -> normalized windows (B*C, L), their fold (B*C, K, P) and the statistics (B, 1, C)."""
    B, L, C = x.shape
    mu = x.mean(1, keepdim=True)
    sd = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    xn = ((x - mu) / sd).permute(0, 2, 1).reshape(B * C, L)
    return xn, xn.view(B * C, K, P), mu, sd


class PhaseFormerPlus(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        P = c.period
        if c.seq_len % P:
            raise ValueError('window must be whole cycles')
        # The horizon need not be whole cycles: the model forecasts ceil(H / P) cycles and keeps the first H values.
        self.c, self.K, self.P, self.Kp = c, c.seq_len // P, P, -(-c.pred_len // P)
        self.Hc = self.Kp * P
        self.phase = (PhaseNetM0(c, self.K, P, self.Kp) if c.arch == 'm0'
                      else PhaseNetBiAxial(c, self.K, P, self.Kp))
        if c.decoder != 'd0':
            if c.arch == 'm0':
                raise ValueError('direct and basis decoders need a bi-axial phase network')
            if c.decoder != 'direct':
                self.coef = nn.Linear(c.k * c.d, 3)
                nn.init.zeros_(self.coef.weight)
                nn.init.zeros_(self.coef.bias)
                profiles = legendre_profiles(self.Hc)
                if c.decoder.endswith('learn'):
                    self.profiles = nn.Parameter(profiles)
                else:
                    self.register_buffer('profiles', profiles, persistent=False)
        elif c.use_trend:
            self.trend = TrendPathway(c.seq_len, self.Hc, P, c.d)
            logit = math.log(c.gate_init / (1 - c.gate_init))
            self.omega = nn.Parameter(torch.full((c.channels,), logit))
            self.omega_delta = nn.Parameter(torch.zeros(c.channels))

    def fold(self, x):
        """Normalized windows (N, L) and their fold (N, K, P) with the statistics (B, 1, C)."""
        return fold_window(x, self.K, self.P)

    def forward(self, x):  # x (B, L, C) -> (B, H, C)
        B, L, C = x.shape
        xn, X, mu, sd = self.fold(x)
        F_p, zbar = self.phase(X)                                   # (N, P, K')
        phase = F_p.transpose(1, 2).reshape(B * C, -1)              # (N, H), cycle-major
        if self.c.decoder != 'd0':
            out = self._decode(phase, X, xn)
        elif self.c.use_trend:
            trend = self.trend(xn, zbar)
            gl = torch.sigmoid(self.omega).repeat(B).unsqueeze(-1)
            gs = torch.sigmoid(self.omega + self.omega_delta).repeat(B).unsqueeze(-1)
            out = self._fuse(phase, trend, gl, gs)
        else:
            out = phase
        out = out.view(B, C, -1)[..., :self.c.pred_len].permute(0, 2, 1)
        return out * sd + mu

    def _anchor(self, X, xn):
        N = X.shape[0]
        mode = self.c.anchor
        if mode == 'x_last':
            return xn[:, -1:].expand(N, self.Hc)
        if mode == 'last_cycle':
            return X[:, -1, :].repeat(1, self.Kp)
        if mode == 'last_cycle_mean':
            return X[:, -1, :].mean(-1, keepdim=True).expand(N, self.Hc)
        if mode == 'none':
            return torch.zeros(N, self.Hc, device=X.device)
        raise ValueError(mode)

    def _decode(self, phase, X, xn):
        ref = self._anchor(X, xn)
        kind = self.c.decoder
        if kind == 'direct':
            return ref + phase
        trend = self.coef(self.phase.pooled) @ self.profiles.T
        if kind.startswith('basis'):                              # periodic part only
            cyc = phase.view(phase.shape[0], self.Kp, self.P)
            phase = (cyc - cyc.mean(-1, keepdim=True)).reshape_as(phase)
        return ref + trend + phase

    def _fuse(self, phase, trend, gl, gs):
        def split(v):
            cyc = v.view(v.shape[0], self.Kp, self.P)
            level = cyc.mean(-1, keepdim=True)
            return level.expand_as(cyc).reshape_as(v), v - level.expand_as(cyc).reshape_as(v)
        pm, pj = split(phase)
        tm, tj = split(trend)
        return (1 - gl) * pm + gl * tm + (1 - gs) * pj + gs * tj


@dataclass
class DCfg:
    """PhaseFormer-D: a cycle (temporal) branch and a phase branch, each with its own normalization and compression."""
    seq_len: int = 720
    pred_len: int = 96
    channels: int = 7
    period: int = 24
    d: int = 16
    routers: int = 8
    heads: int = 1
    dropout: float = 0.1
    # cycle branch: routing over the cycle axis of every phase column; compression K -> k_t cycles, P -> p_t phases
    k_t: int = 8
    p_t: int = 0                # 0 means P
    layers_t: int = 1
    norm_t: str = 'post'
    embed_t: str = 'mlp_ln'
    # phase branch: routing over the phase axis of every cycle row; compression K -> k_s cycles, P -> p_s phases
    k_s: int = 4
    p_s: int = 0                # 0 means P
    layers_s: int = 1
    norm_s: str = 'post'
    embed_s: str = 'mlp_ln'
    fusion: str = 'gate'        # sum | gate (window-adaptive branch gates, initialized to the plain sum)
    t_eps: float = 1e-2         # variance floor of the cycle-branch scale
    s_eps: float = 1e-2         # variance floor of the per-cycle scale of the phase branch
    phase_id: str = 'none'
    harmonics: int = 8


class PhaseFormerD(nn.Module):
    """Two forecasters of the window's last-cycle residual, fused in common units.

    The window is instance-normalized and folded to X (K cycles x P phases). Both branches predict a correction
    to the last observed cycle (the anchor), in the units of their own normalization, and de-normalize it with
    the statistics they normalized by, so zero outputs reproduce the anchor exactly.

    Cycle branch: how every phase evolves from cycle to cycle (level, drift, slow seasonality). Input: X with the
    seasonal profile (per-phase mean over cycles) removed and one scalar scale per window. The scale is shared by all
    phases, so the relative size of the phases' dynamics is kept; no per-phase scale, which would amplify noise in
    phases that hardly move.

    Phase branch: the shape of a cycle. Input: every cycle normalized by its own mean and standard deviation over
    the phases, so level and amplitude cannot reach it. Its output is made zero-mean over the phases of each
    forecast cycle and scaled by the last cycle's standard deviation: it can reshape a cycle but not move it, so
    level changes belong to the cycle branch alone.
    """

    def __init__(self, c: DCfg):
        super().__init__()
        P = c.period
        if c.seq_len % P:
            raise ValueError('window must be whole cycles')
        self.c, self.K, self.P, self.Kp = c, c.seq_len // P, P, -(-c.pred_len // P)
        self.Hc = self.Kp * P

        def branch(arch, k, p, layers, norm, embed):
            cfg = Cfg(seq_len=c.seq_len, pred_len=c.pred_len, channels=c.channels, period=P, arch=arch, layers=layers,
                      d=c.d, routers=c.routers, heads=c.heads, dropout=c.dropout, k=k, p=p, q=1,
                      wp='identity' if p in (0, P) else 'learn', decoder='direct', anchor='last_cycle',
                      norm=norm, embed=embed, phase_id=c.phase_id, harmonics=c.harmonics)
            return PhaseNetBiAxial(cfg, self.K, P, self.Kp)

        self.cycle = branch('temporal', c.k_t, c.p_t, c.layers_t, c.norm_t, c.embed_t)
        self.phase = branch('phase', c.k_s, c.p_s, c.layers_s, c.norm_s, c.embed_s)
        if c.fusion == 'gate':
            self.gate = nn.Linear(2 + (c.k_t + c.k_s) * c.d, 2)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif c.fusion != 'sum':
            raise ValueError(c.fusion)

    def forward(self, x):  # x (B, L, C) -> (B, H, C)
        B, L, C = x.shape
        c = self.c
        xn, X, mu, sd = fold_window(x, self.K, self.P)
        anchor = X[:, -1, :].repeat(1, self.Kp)                                   # (N, Hc)
        # cycle branch input: seasonal profile removed, one scale per window
        resid = X - X.mean(1, keepdim=True)
        scale_t = (resid.square().mean((1, 2), keepdim=True) + c.t_eps).sqrt()    # (N, 1, 1)
        f_t, _ = self.cycle(resid / scale_t)                                      # (N, P, Kp)
        delta_t = f_t.transpose(1, 2).reshape(X.shape[0], -1) * scale_t.flatten(1)
        # phase branch input: every cycle normalized over its phases
        row = X.mean(2, keepdim=True)
        scale_s = (X.var(2, keepdim=True, unbiased=False) + c.s_eps).sqrt()       # (N, K, 1)
        f_s, _ = self.phase((X - row) / scale_s)
        shape = f_s.transpose(1, 2)                                               # (N, Kp, P)
        shape = shape - shape.mean(-1, keepdim=True)
        delta_s = shape.reshape(X.shape[0], -1) * scale_s[:, -1]
        if c.fusion == 'gate':
            u = torch.cat([scale_t.flatten(1).log(), scale_s[:, -1].log(), self.cycle.pooled, self.phase.pooled], -1)
            g = 2 * torch.sigmoid(self.gate(u))                                   # (N, 2), equal to 1 at initialization
            delta_t, delta_s = g[:, :1] * delta_t, g[:, 1:] * delta_s
        out = (anchor + delta_t + delta_s).view(B, C, -1)[..., :c.pred_len].permute(0, 2, 1)
        return out * sd + mu


def make_model(cfg) -> nn.Module:
    return PhaseFormerD(cfg) if isinstance(cfg, DCfg) else PhaseFormerPlus(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
