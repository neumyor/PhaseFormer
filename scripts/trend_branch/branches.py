"""Trend branches modelled on N-BEATS, ETSFormer and BasisFormer.

Each branch sits next to an unchanged PhaseFormer and adds a delta to its forecast::

    y_hat = PhaseFormer(x) + branch(x)

and every branch is zero at initialization (or, for ``ets``/``basis``, scaled by a
zero-initialized per-channel gate), so training starts exactly at PhaseFormer.

* ``nb``     N-BEATS' interpretable trend stack (Oreshkin et al. 2020): a stack of
             blocks, each an MLP over the cycle means of the anchored history that
             emits polynomial-basis coefficients (degree 2) for the forecast and a
             backcast that is subtracted from the next block's input (doubly
             residual stacking).
* ``ets``    ETSFormer's level and growth (Woo et al. 2022): Holt's recursion with
             learned alpha/beta applied to the cycle means, extrapolated with a
             learned damping gamma.  ETSFormer applies it at every step; here it
             runs on the 30 cycle means so the branch stays cheap, and the seasonal
             part is deliberately left to PhaseFormer's phase branch.
* ``basis``  BasisFormer's learnable smooth basis (Ni et al. 2023): a learnable
             (N, H) basis, per-channel coefficients from a cross-attention between
             the pooled history and the basis embeddings, forecast as the weighted
             sum of basis rows.  The contrastive past/future alignment is omitted
             (it needs a second view and is training-only); a second-difference
             smoothness penalty on the basis is kept.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _NBBlock(nn.Module):
    def __init__(self, pooled, hidden, degree):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(pooled, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.to_forecast = nn.Linear(hidden, degree + 1)
        self.to_backcast = nn.Linear(hidden, pooled)
        nn.init.zeros_(self.to_forecast.weight)
        nn.init.zeros_(self.to_forecast.bias)

    def forward(self, residual):
        h = self.mlp(residual)
        return self.to_forecast(h), self.to_backcast(h)


class NBTrendBranch(nn.Module):
    """N-BEATS interpretable trend stack on top of PhaseFormer."""

    kind = 'nb'

    def __init__(self, seq_len, pred_len, channels, degree=2, blocks=2, hidden=64, pooled=30):
        super().__init__()
        self.pooled_len = min(int(pooled), int(seq_len))
        self.degree = int(degree)
        t = torch.linspace(0.0, 1.0, int(pred_len), dtype=torch.float32)
        self.register_buffer('poly', torch.stack([t ** k for k in range(degree + 1)], dim=1))
        self.blocks = nn.ModuleList(_NBBlock(self.pooled_len, hidden, degree) for _ in range(blocks))

    def forward(self, x):
        anchor = x[:, -1:]
        pooled = F.adaptive_avg_pool1d((x - anchor).transpose(1, 2), self.pooled_len)
        residual = pooled
        delta = torch.zeros(x.shape[0], self.poly.shape[0], x.shape[2],
                            dtype=x.dtype, device=x.device)
        for block in self.blocks:
            coefficients, backcast = block(residual)
            delta = delta + torch.einsum('bck,hk->bhc', coefficients, self.poly)
            residual = residual - backcast
        return delta


class ETSTrendBranch(nn.Module):
    """ETSFormer-style level and growth with damping, on the cycle means."""

    kind = 'ets'

    def __init__(self, seq_len, pred_len, channels, steps=30, gamma=0.9):
        super().__init__()
        self.pool_steps = min(int(steps), int(seq_len))
        self.alpha = nn.Parameter(torch.full((int(channels),), 0.5))
        self.beta = nn.Parameter(torch.full((int(channels),), 0.1))
        self.gamma = nn.Parameter(torch.full((int(channels),),
                                             math.log(gamma / (1.0 - gamma))))
        self.gate = nn.Parameter(torch.zeros(1, int(channels), 1))
        horizon_steps = torch.arange(1, int(pred_len) + 1, dtype=torch.float32)
        self.register_buffer('horizon_steps', horizon_steps)

    def forward(self, x):
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.pool_steps)      # (B, C, T)
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1)
        beta = torch.sigmoid(self.beta).view(1, -1, 1)
        gamma = torch.sigmoid(self.gamma)
        level = pooled[:, :, 0]
        growth = torch.zeros_like(level)
        for step in range(1, pooled.shape[2]):
            previous = level
            level = alpha[:, :, 0] * pooled[:, :, step] + (1 - alpha[:, :, 0]) * (level + growth)
            growth = beta[:, :, 0] * (level - previous) + (1 - beta[:, :, 0]) * growth
        damping = torch.cumsum(gamma.unsqueeze(0) ** self.horizon_steps.unsqueeze(-1), dim=0)  # (H, C)
        delta = ((level - x[:, -1]).unsqueeze(-1)
                 + growth.unsqueeze(-1) * damping.permute(1, 0).unsqueeze(0))                  # (B, C, H)
        return (delta * torch.tanh(self.gate)).transpose(1, 2)


class BasisTrendBranch(nn.Module):
    """BasisFormer-style learnable smooth basis with cross-attention coefficients."""

    kind = 'basis'

    def __init__(self, seq_len, pred_len, channels, n_basis=12, hidden=64, pooled=30,
                 smooth_weight=1e-3):
        super().__init__()
        self.pooled_len = min(int(pooled), int(seq_len))
        self.n_basis = int(n_basis)
        self.smooth_weight = float(smooth_weight)
        self.history = nn.Linear(self.pooled_len, hidden)
        self.basis_embedding = nn.Parameter(torch.randn(self.n_basis, hidden) * 0.02)
        self.basis = nn.Parameter(self._initial_basis(int(pred_len)))
        self.gate = nn.Parameter(torch.zeros(1, int(channels), 1))

    def _initial_basis(self, pred_len):
        """Low-frequency cosine basis: smooth at initialization, free afterwards."""
        t = torch.linspace(0.0, 1.0, pred_len, dtype=torch.float32)
        rows = []
        for index in range(self.n_basis):
            frequency = (index + 1) // 2
            if index % 2 == 0:
                rows.append(torch.cos(math.pi * frequency * t))
            else:
                rows.append(torch.sin(math.pi * frequency * t))
        return torch.stack(rows)

    def forward(self, x):
        anchor = x[:, -1:]
        pooled = F.adaptive_avg_pool1d((x - anchor).transpose(1, 2), self.pooled_len)
        token = self.history(pooled)                                     # (B, C, D)
        scores = torch.einsum('bcd,nd->bcn', token, self.basis_embedding) / math.sqrt(token.shape[-1])
        weights = torch.softmax(scores, dim=-1)                          # (B, C, N)
        delta = torch.einsum('bcn,nh->bch', weights, self.basis)                            # (B, C, H)
        return (delta * torch.tanh(self.gate)).transpose(1, 2)

    def regularization(self):
        second = self.basis[:, 2:] - 2 * self.basis[:, 1:-1] + self.basis[:, :-2]
        return self.smooth_weight * second.square().mean()


BRANCHES = {'nb': NBTrendBranch, 'ets': ETSTrendBranch, 'basis': BasisTrendBranch}
# Options each branch accepts from a job's ``branch_options``.
OPTIONS = {'nb': ('degree', 'blocks', 'hidden', 'pooled'),
           'ets': ('steps',),
           'basis': ('n_basis', 'hidden', 'pooled')}


def build(kind, seq_len, pred_len, channels, options=None):
    if kind not in BRANCHES:
        raise ValueError(kind)
    options = {k: v for k, v in (options or {}).items() if k in OPTIONS[kind]}
    return BRANCHES[kind](seq_len, pred_len, channels, **options)


def _optimizer_with_branch_rate(model, lr_mult):
    """Adam with the branch on its own learning-rate multiplier."""
    import types

    import torch

    base_lr = float(model.args.training_args.learning_rate)
    schedule = model.args.training_args.lr_schedule_config

    def configure_optimizers(self):
        branch, backbone = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (branch if name.startswith('trend_branch.') else backbone).append(parameter)
        groups = [{'params': backbone, 'lr': base_lr}]
        if branch:
            groups.append({'params': branch, 'lr': base_lr * lr_mult})
        optimizer = torch.optim.Adam(groups)
        if schedule.type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=schedule.tmax, eta_min=1e-8)
            return [optimizer], [scheduler]
        return optimizer

    model.configure_optimizers = types.MethodType(configure_optimizers, model)


def attach(model, kind, options=None, lr_mult=1.0):
    """Attach ``kind`` to a built PhaseFormer and route the forecast through it.

    ``options`` overrides the branch's architectural knobs (degree, basis count, ...);
    ``lr_mult`` scales the branch parameters' learning rate relative to the backbone.
    """
    import types

    branch = build(kind, model.seq_len, model.pred_len, model.enc_in, options)
    model.add_module('trend_branch', branch)
    original_forward = model.forward

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        out = original_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, *args, **kwargs)
        delta = self.trend_branch(x_enc.float())
        return (out[0] + delta,) + tuple(out[1:])

    model.forward = types.MethodType(forward, model)

    original_step = model.training_step

    def training_step(self, batch, batch_idx):
        loss = original_step(batch, batch_idx)
        regularizer = getattr(self.trend_branch, 'regularization', None)
        if regularizer is not None and self.training:
            loss = loss + regularizer()
        return loss

    model.training_step = types.MethodType(training_step, model)
    _optimizer_with_branch_rate(model, float(lr_mult))
    return model
