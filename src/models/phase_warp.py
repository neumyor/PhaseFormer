import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseWarping(nn.Module):
    """Data-driven monotonic phase warping for PhaseFormer.

    Instead of the fixed uniform phase advance (each time step = one phase
    slot), a small MLP predicts a per-point speed field from the normalized
    value and time-mark features. Within each cycle the cumulative normalized
    speed defines a monotonic map from time-in-cycle to a continuous phase:

        phi[l] = L * sum_{j < l} speed[j] / sum_j speed[j]

    so phi is non-decreasing in time, endpoints are fixed (phi[0] = 0,
    phi[L-1] = L-1), and a segment that evolves quickly gets more phase
    resolution while a slow segment is compressed. With a uniform speed field
    (softplus of a zero net output) the map reduces to the identity grid, so
    the mechanism is a warm-start deformation of the fixed phase grid.

    Input evidence is soft-scattered onto the two neighbouring phase slots via
    linear interpolation on the phase circle (k=2), identical to
    `PhaseAlignment`. The output has the (B, C, L, P) layout that
    `_to_phase_series` produces.
    """

    def __init__(self, mark_dim: int, hidden: int = 8, chunk_t: int = 240):
        super().__init__()
        self.mark_dim = mark_dim
        self.hidden = hidden
        self.chunk_t = chunk_t

        self.value_proj = nn.Linear(1, hidden)
        self.mark_proj = nn.Linear(mark_dim, hidden)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so speed = softplus(0) = ln 2 is uniform at
        # construction, which makes the warp the identity grid (warm start).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hook: mean |phi - identity| from the last forward pass.
        self.last_mean_phase_advance = 0.0

    def _fit_mark_width(self, mark_ring):
        if mark_ring.shape[-1] < self.mark_dim:
            mark_ring = F.pad(mark_ring, (0, self.mark_dim - mark_ring.shape[-1]))
        elif mark_ring.shape[-1] > self.mark_dim:
            mark_ring = mark_ring[..., : self.mark_dim]
        return mark_ring

    def _estimate_speed(self, x_ring, mark_ring):
        # x_ring: (B, C, T); mark_ring: (B, T, mark_dim) -> speed (B, C, T)
        B, C, T = x_ring.shape
        parts = []
        for t0 in range(0, T, self.chunk_t):
            t1 = min(t0 + self.chunk_t, T)
            v = x_ring[:, :, t0:t1].unsqueeze(-1)  # (B, C, Tc, 1)
            m = mark_ring[:, t0:t1, :].unsqueeze(1)  # (B, 1, Tc, mark_dim)
            h = self.value_proj(v)
            h = h + self.mark_proj(m)
            d = self.net(h).squeeze(-1)  # (B, C, Tc)
            parts.append(d)
        return F.softplus(torch.cat(parts, dim=-1))  # (B, C, T), > 0

    def _warp_phase(self, speed):
        # speed: (B, C, P, L) > 0 -> phi (B, C, P, L) monotonic in [0, L)
        cum = torch.cumsum(speed, dim=-1)  # (B, C, P, L)
        total = cum[..., -1:]  # (B, C, P, 1), > 0
        L = speed.shape[-1]
        # phi[l] = L * sum_{j<l} s_j / sum_j s_j ; phi[0] = 0, phi[L-1] = L-1
        return L * (cum - speed) / total

    def forward(self, x_periods, mark_ring):
        """Scatter input evidence into phase slots by a learned monotonic warp.

        Args:
            x_periods: (B, C, P, L) values on the (ring-padded) fixed time grid.
            mark_ring: (B, T, mark_dim) time-mark features for ring positions,
                       T = P * L.
        Returns:
            (B, C, L, P) phase series, same layout as the identity permute.
        """
        B, C, P, L = x_periods.shape
        T = P * L
        x_ring = x_periods.reshape(B, C, T)  # ring position g = p * L + l
        mark_ring = self._fit_mark_width(mark_ring)

        speed = self._estimate_speed(x_ring, mark_ring).view(B, C, P, L)
        phi = self._warp_phase(speed)  # (B, C, P, L) monotonic in [0, L)

        i0 = phi.floor().long() % L
        frac = phi - phi.floor()
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

        base = torch.arange(L, dtype=phi.dtype, device=phi.device).view(1, 1, 1, L)
        with torch.no_grad():
            self.last_mean_phase_advance = float((phi - base).abs().mean())
        return out.permute(0, 1, 3, 2).contiguous()  # (B, C, L, P)
