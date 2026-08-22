import torch
import torch.nn as nn


class PhaseCorrection(nn.Module):
    """Dynamic phase correction (PhaseFormer experiment plan, stage 2).

    PhaseFormer's phase tokens are aligned to the fixed grid position-in-cycle:
    token l occupies phase slot l. Real series can run early, late, or at a
    varying speed, so this module predicts a per-(sample, channel, slot) offset
    `delta` in (-1, 1) from the latent phase token itself, then re-aligns the
    token ordering along the phase-slot axis: token mass at slot l is linearly
    interpolated onto the two neighbouring slots `l + delta` on the phase circle
    (k=2, identical scatter convention to `PhaseAlignment`). With delta == 0 the
    scatter reduces to the identity, so the module is a warm-start deformation
    of the fixed phase grid.

    The module operates on the latent phase tokens (B, C, L, D) produced by the
    top-level `PhaseEmbedding`, exactly where the plan inserts it (embedding ->
    correction -> warp -> routing). Output keeps the (B, C, L, D) layout, so
    every downstream consumer is untouched.
    """

    def __init__(self, dim: int, hidden: int = 8):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        # Offset predictor matching the plan: Linear(dim, dim) -> GELU -> Linear(dim, 1),
        # with `hidden` defaulting to `dim` but kept configurable for capacity control.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer so delta = tanh(0) = 0 -> identity scatter.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hook: mean |delta| from the last forward pass.
        self.last_mean_delta = 0.0

    def forward(self, tokens):  # (B, C, L, D)
        B, C, L, D = tokens.shape
        delta = torch.tanh(self.net(tokens).squeeze(-1))  # (B, C, L)
        # Continuous position on the phase circle: pos = slot + delta.
        base = torch.arange(L, device=tokens.device, dtype=tokens.dtype).view(1, 1, L)
        pos = base + delta  # (B, C, L)
        i0 = pos.floor().long() % L  # floor(-0.5) = -1 -> wraps to L-1
        frac = pos - pos.floor()  # (B, C, L) in [0, 1)
        i1 = (i0 + 1) % L

        out = torch.zeros_like(tokens)
        for l in range(L):
            src = tokens[:, :, l, :].unsqueeze(2)  # (B, C, 1, D)
            t0 = i0[:, :, l].view(B, C, 1, 1).expand(-1, -1, 1, D)  # (B, C, 1, D)
            t1 = i1[:, :, l].view(B, C, 1, 1).expand(-1, -1, 1, D)
            w0 = (1.0 - frac[:, :, l]).view(B, C, 1, 1)
            w1 = frac[:, :, l].view(B, C, 1, 1)
            out.scatter_add_(2, t0, src * w0)
            out.scatter_add_(2, t1, src * w1)

        with torch.no_grad():
            self.last_mean_delta = float(delta.abs().mean())
        return out
