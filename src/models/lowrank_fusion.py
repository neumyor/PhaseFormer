"""Output gates for the original rank-r trend and phase forecasts."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LowRankDynamicFusion(nn.Module):
    MODES = {"reliability", "disagreement", "disagreement_without",
             "components", "components_static", "components_level_dynamic",
             "components_level_without"}

    def __init__(self, rank, phase_dim, channels, period, mode, width=16):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode, self.period = mode, period
        self.components = mode.startswith("components")
        self.level_only = mode in {"components_level_dynamic", "components_level_without"}
        if mode == "reliability":
            self.sensitivity = nn.Parameter(torch.zeros(()))
        elif mode == "components_static":
            self.shape_bias_delta = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            # rank state, its scale, phase distribution, forecast summaries.
            self.head = nn.Sequential(nn.Linear(rank + 2 + 2 * phase_dim + 7, width),
                                      nn.GELU(), nn.Linear(width, 2 if self.components and not self.level_only else 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            if self.level_only:
                self.shape_bias_delta = nn.Parameter(torch.zeros(1, channels, 1))
        self.override_gate = None  # Nonpersistent diagnostic intervention (B,C,1/2).
        self.last_gate = None
        self.last_reliability = None

    @staticmethod
    def reliability(history):
        signal = history.mean(-1).var(-1, unbiased=False)
        noise = history.var(-1, unbiased=False).mean(-1)
        return signal / (signal + noise + 1e-6)

    @staticmethod
    def features(trend, phase, yp, yt):
        # Gate gradients do not change the representations it reads. Both
        # forecast branches still receive gradients through the final mixture.
        trend, phase, yp, yt = (v.detach() for v in (trend, phase, yp, yt))
        tau = torch.linspace(-1, 1, yp.shape[1], device=yp.device, dtype=yp.dtype)
        slope = lambda y: (y * tau[None, :, None]).mean(1) / tau.square().mean()
        summaries = torch.stack([yp.mean(1), yt.mean(1), slope(yp), slope(yt),
                                 yp.std(1, unbiased=False), yt.std(1, unbiased=False),
                                 (yt-yp).square().mean(1).sqrt()], -1)
        return torch.cat([F.layer_norm(trend, (trend.shape[-1],)),
                          trend.mean(-1, keepdim=True), trend.std(-1, unbiased=False, keepdim=True),
                          phase.mean(2), phase.std(2, unbiased=False), summaries], -1)

    def forward(self, yp, yt, trend, phase, history, base_logit):
        b, _, c = yp.shape
        bias = base_logit.transpose(1, 2).expand(b, -1, -1)
        r = self.reliability(history.detach())
        if self.mode == "reliability":
            logits = bias + 4 * self.sensitivity.tanh() * (1-r).unsqueeze(-1)
        elif self.mode == "components_static":
            logits = torch.cat([bias, bias + self.shape_bias_delta], -1)
        else:
            features = self.features(trend, phase, yp, yt)
            if self.mode == "disagreement_without":
                features = torch.cat([features[..., :-7], torch.zeros_like(features[..., -7:])], -1)
            elif self.mode == "components_level_without":
                features = torch.zeros_like(features)
            logits = bias + self.head(features)
            if self.level_only:
                logits = torch.cat([logits, bias + self.shape_bias_delta], -1)
        gate = logits.sigmoid()
        if self.override_gate is not None:
            gate = self.override_gate.to(gate).expand_as(gate)
        self.last_gate, self.last_reliability = gate.detach(), r.detach()
        if not self.components:
            alpha = gate.transpose(1, 2)
            return (1-alpha)*yp + alpha*yt
        if yp.shape[1] % self.period:
            raise ValueError("component fusion requires whole forecast cycles")
        cycles_p = yp.reshape(b, -1, self.period, c)
        cycles_t = yt.reshape(b, -1, self.period, c)
        # Equivalent component formula expressed as scalar fusion plus a mean
        # adjustment. Equal level/shape gates recover the scalar path exactly.
        level_delta = cycles_t.mean(2, keepdim=True) - cycles_p.mean(2, keepdim=True)
        a_level, a_shape = gate[..., 0][:, None, None], gate[..., 1][:, None, None]
        mixed = (1-a_shape)*cycles_p + a_shape*cycles_t + (a_level-a_shape)*level_delta
        return mixed.reshape_as(yp)
