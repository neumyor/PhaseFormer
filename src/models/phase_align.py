import torch
import torch.nn as nn


class PhaseAlignment(nn.Module):
    """Data-driven phase alignment: re-map time points into phase slots by a
    learned continuous phase instead of the fixed `time % period_len` grid.

    For every (batch, channel, ring-position) the module predicts a phase
    correction `delta` in (-1, 1) from the normalized value and the time-mark
    features. The effective phase is `position_in_cycle + delta`, and the input
    evidence is soft-scattered onto the two neighbouring phase slots via linear
    interpolation on the phase circle (k=2). With delta == 0 the scatter reduces
    to the identity mapping, which makes the mechanism a smooth deformation of
    the fixed phase grid (residual warm start).

    The MLP weights are shared across channels; each channel's value makes its
    phase differ. The output has the same (B, C, L, P) layout that
    `_to_phase_series` produces, so every downstream consumer is untouched.
    """

    def __init__(
        self,
        mark_dim: int,
        hidden: int = 8,
        use_position_encoding: bool = False,
        chunk_t: int = 240,
    ):
        super().__init__()
        self.mark_dim = mark_dim
        self.hidden = hidden
        self.use_position_encoding = use_position_encoding
        self.chunk_t = chunk_t

        # Two-branch MLP. Splitting the input avoids materializing an expanded
        # (B, C, T, mark_dim) tensor; the mark branch is added via broadcasting.
        self.value_proj = nn.Linear(1, hidden)
        self.mark_proj = nn.Linear(mark_dim, hidden)
        if use_position_encoding:
            self.pos_proj = nn.Linear(1, hidden)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so delta == 0 at construction (identity).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hook: mean |delta| from the last forward pass. Plain float
        # so it never enters the state_dict.
        self.last_mean_delta = 0.0

    def _fit_mark_width(self, mark_ring):
        # Defensive: align the trailing mark width to self.mark_dim.
        if mark_ring.shape[-1] < self.mark_dim:
            pad = self.mark_dim - mark_ring.shape[-1]
            mark_ring = torch.nn.functional.pad(mark_ring, (0, pad))
        elif mark_ring.shape[-1] > self.mark_dim:
            mark_ring = mark_ring[..., : self.mark_dim]
        return mark_ring

    def _estimate_delta(self, x_ring, mark_ring):
        # x_ring: (B, C, T); mark_ring: (B, T, mark_dim) -> delta (B, C, T)
        B, C, T = x_ring.shape
        delta_parts = []
        for t0 in range(0, T, self.chunk_t):
            t1 = min(t0 + self.chunk_t, T)
            v = x_ring[:, :, t0:t1].unsqueeze(-1)  # (B, C, Tc, 1)
            m = mark_ring[:, t0:t1, :].unsqueeze(1)  # (B, 1, Tc, mark_dim)
            h = self.value_proj(v)  # (B, C, Tc, H)
            h = h + self.mark_proj(m)  # broadcast (B,1,Tc,H) -> (B,C,Tc,H)
            if self.use_position_encoding:
                pos = ((torch.arange(t0, t1, device=x_ring.device) % self.period_len)
                       / self.period_len - 0.5).view(1, 1, t1 - t0, 1)
                h = h + self.pos_proj(pos)
            d = self.net(h).squeeze(-1)  # (B, C, Tc)
            delta_parts.append(d)
        delta = torch.cat(delta_parts, dim=-1)  # (B, C, T)
        return torch.tanh(delta)

    def forward(self, x_periods, mark_ring):
        """Scatter input evidence into phase slots by learned phase.

        Args:
            x_periods: (B, C, P, L) values on the (ring-padded) fixed time grid.
            mark_ring: (B, T, mark_dim) time-mark features for ring positions,
                       T = P * L.
        Returns:
            (B, C, L, P) phase series, same layout as the identity permute.
        """
        B, C, P, L = x_periods.shape
        T = P * L
        self.period_len = L
        x_ring = x_periods.reshape(B, C, T)  # ring position g = p * L + l
        mark_ring = self._fit_mark_width(mark_ring)

        delta = self._estimate_delta(x_ring, mark_ring).view(B, C, P, L)  # (B,C,P,L)

        base = torch.arange(L, dtype=x_ring.dtype, device=x_ring.device).view(1, 1, 1, L)
        phi = base + delta  # (B, C, P, L)

        i0 = phi.floor().long() % L  # floor(-0.5) = -1 -> wraps to L-1
        frac = phi - phi.floor()  # (B, C, P, L) in [0, 1)
        i1 = (i0 + 1) % L

        out = torch.zeros_like(x_periods)
        for p in range(P):
            out[:, :, p, :].scatter_add_(
                2,
                i0[:, :, p, :],
                x_periods[:, :, p, :] * (1.0 - frac[:, :, p, :]),
            )
            out[:, :, p, :].scatter_add_(
                2,
                i1[:, :, p, :],
                x_periods[:, :, p, :] * frac[:, :, p, :],
            )

        with torch.no_grad():
            self.last_mean_delta = float(delta.abs().mean())
        return out.permute(0, 1, 3, 2).contiguous()  # (B, C, L, P)
