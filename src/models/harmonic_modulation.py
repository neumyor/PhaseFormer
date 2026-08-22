import torch
import torch.nn as nn


class HarmonicModulation(nn.Module):
    """Harmonic feature modulation (PhaseFormer experiment plan, stage 5).

    Rescales and shifts the latent representation with per-position
    ``gamma`` / ``beta`` generated from the input periodic features:

        z' = gamma * z + beta

    gamma is initialized to 1 and beta to 0 so the modulation starts as the
    identity (warm start). Inserted between Cross Phase Routing and the
    top-level Phase Predictor, exactly where the plan places it.

    Inputs:
        z: (B, C, L, D) latent phase tokens after routing
        cond: (B, C, L, cond_dim) input periodic features used to generate gamma/beta

    Output keeps the (B, C, L, D) layout.
    """

    def __init__(self, cond_dim: int, hidden: int = 8, max_scale: float = 2.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.max_scale = max_scale
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Zero-init final layer -> scale 0 -> gamma = 1, beta = 0: identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hooks from the last forward pass.
        self.last_mean_abs_gamma = 0.0
        self.last_mean_abs_beta = 0.0

    def forward(self, z, cond):
        raw = self.net(cond)  # (B, C, L, 2)
        gamma = 1.0 + self.max_scale * torch.tanh(raw[..., 0])  # (B, C, L)
        beta = raw[..., 1]  # (B, C, L)
        out = gamma.unsqueeze(-1) * z + beta.unsqueeze(-1)
        with torch.no_grad():
            self.last_mean_abs_gamma = float((gamma - 1.0).abs().mean())
            self.last_mean_abs_beta = float(beta.abs().mean())
        return out
