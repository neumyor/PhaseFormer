import torch
import torch.nn as nn


class MultiScalePhase(nn.Module):
    """Multi-scale phase representation (pure-phase plan, stage 1).

    PhaseFormer represents the input as a single phase view ``phase_series`` of
    shape (B, C, L, P_in): L phase slots (positions within the cycle) by P_in
    features (one per consecutive cycle). This module adds a *long-period* view
    at the same phase-slot grid, obtained by averaging adjacent groups of
    ``coarse`` cycles along the period axis:

        phase_series_long[b, c, l, m] = mean over {k : k//coarse == m} of
                                         phase_series[b, c, l, k]

    Slot l stays aligned to the same phase position in both views, so the
    short (fine, single-cycle) and long (coarse, multi-cycle-averaged) views
    are directly fusible. The long view is embedded independently
    (Linear(P_in//coarse -> D) + LayerNorm) and gated into the short embedding:

        Z = Z_short + zeta * Z_long,   zeta a learnable (D,) vector init 0

    zeta = 0 at initialization makes the fusion the exact single-phase baseline
    (warm start); zeta becoming nonzero means the model actually uses the
    long-period structure.

    Diagnostics from the last forward pass (analysis only, no-grad):
      - last_mean_abs_long: mean |Z_long| (magnitude of the long branch output)
    """

    def __init__(self, latent_dim: int, period_len: int, num_periods_input: int,
                 coarse: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.period_len = period_len
        self.coarse = coarse
        self.num_periods_long = (num_periods_input + coarse - 1) // coarse
        self.embed_long = nn.Sequential(
            nn.Linear(self.num_periods_long, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # Gate on the long branch: 0 -> exact single-phase warm start.
        self.zeta = nn.Parameter(torch.zeros(latent_dim))

        # Diagnostic hook (analysis only, no parameters).
        self.last_mean_abs_long = 0.0

    def forward(self, phase_series):  # (B, C, L, P_in)
        B, C, L, P = phase_series.shape
        coarse = self.coarse
        Pc = self.num_periods_long
        if P % coarse == 0:
            grouped = phase_series.view(B, C, L, P // coarse, coarse).mean(dim=-1)
        else:
            pad = coarse - (P % coarse)
            if pad == coarse:
                pad = 0
            if pad:
                last = phase_series[..., -1:]
                phase_series_p = torch.cat(
                    [phase_series, last.expand(B, C, L, pad)], dim=-1
                )
            else:
                phase_series_p = phase_series
            grouped = phase_series_p.view(B, C, L, Pc, coarse).mean(dim=-1)
        # (B, C, L, Pc)
        Z_long = self.embed_long(grouped)  # (B, C, L, D)

        with torch.no_grad():
            self.last_mean_abs_long = float(Z_long.abs().mean())
        return self.zeta.view(1, 1, 1, self.latent_dim) * Z_long
