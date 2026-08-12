import torch
import torch.nn as nn


class PhaseAmpCalibration(nn.Module):
    """Phase-conditioned amplitude calibration of the phase representation.

    Adaptive phase warping learns *where* in the cycle events happen (the
    continuous phase phi), but it cannot change how strong a phase state is:
    two periods with the same shape but different amplitudes collapse to the
    same phase values, and a fixed phase grid has no per-state gain. This module
    adds the missing amplitude degree of freedom. For every phase slot l it
    predicts a scale alpha_l and a shift beta_l from the phase-slot position and
    per-slot statistics of the phase history (mean / std / abs-mean / last
    period / linear trend over periods), then applies

        h'[l, k] = alpha_l * h[l, k] + beta_l

    broadcast over the period axis k. alpha_l rescales a phase state by its
    observed amplitude envelope; beta_l compensates level drift between cycles.
    With zero-initialized outputs alpha=1 and beta=0, so the module is the
    identity at construction (warm start).

    The MLP weights are shared across channels and slots; the per-slot
    statistics make the calibration sample- and channel-specific. Output keeps
    the (B, C, L, P) layout, so every downstream consumer is untouched.
    """

    def __init__(self, hidden: int = 8, max_scale: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.hidden = hidden
        self.max_scale = max(max(float(max_scale), 0.0), 0.1)
        self.eps = max(float(eps), 1e-8)

        # Features per phase slot: [slot position, mean, std, abs-mean,
        # last-period value, linear trend over periods].
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Zero-init the final layer so scale_logit=0 (alpha=1) and beta=0 at
        # construction, i.e. the calibration is the identity (warm start).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hooks (plain floats, never in the state_dict). |alpha-1|
        # measures deviation from identity scaling; >1 indicates alpha<0 (a
        # phase sign-flip, reachable because max_scale allows negative alpha).
        self.last_mean_abs_log_alpha = 0.0
        self.last_mean_abs_beta = 0.0

    def _slot_stats(self, phase_series):
        # phase_series: (B, C, L, P) -> per-slot statistics (B, C, L)
        P = phase_series.size(-1)
        slot_mean = phase_series.mean(dim=-1)
        if P >= 2:
            slot_std = phase_series.var(dim=-1, unbiased=False).clamp_min(
                self.eps
            ).sqrt()
            centered = torch.arange(
                P, device=phase_series.device, dtype=phase_series.dtype
            ).view(1, 1, 1, P) - (P - 1.0) / 2.0
            denom = torch.square(centered).sum(dim=-1, keepdim=True).clamp_min(self.eps)
            slope = (
                (phase_series - slot_mean.unsqueeze(-1)) * centered
            ).sum(dim=-1, keepdim=True) / denom
            slope = slope.squeeze(-1)
        else:
            slot_std = torch.zeros_like(slot_mean)
            slope = torch.zeros_like(slot_mean)
        slot_absmean = phase_series.abs().mean(dim=-1)
        last = phase_series[..., -1]
        return slot_mean, slot_std, slot_absmean, last, slope

    def forward(self, phase_series):  # (B, C, L, P)
        B, C, L, P = phase_series.shape
        slot_mean, slot_std, slot_absmean, last, slope = self._slot_stats(phase_series)
        pos = torch.arange(
            L, device=phase_series.device, dtype=phase_series.dtype
        ).view(1, 1, L) / max(float(L), 1.0)
        pos = pos.expand(B, C, L)
        feat = torch.stack(
            [pos, slot_mean, slot_std, slot_absmean, last, slope], dim=-1
        )  # (B, C, L, 6)
        raw = self.net(feat)  # (B, C, L, 2)
        scale_logit, shift = raw[..., 0], raw[..., 1]
        # alpha in [1 - max_scale, 1 + max_scale] around the identity, beta
        # unbounded (zero at construction).
        alpha = 1.0 + self.max_scale * torch.tanh(scale_logit)
        beta = shift
        out = alpha.unsqueeze(-1) * phase_series + beta.unsqueeze(-1)

        with torch.no_grad():
            self.last_mean_abs_log_alpha = float((alpha - 1.0).abs().mean())
            self.last_mean_abs_beta = float(beta.abs().mean())
        return out
