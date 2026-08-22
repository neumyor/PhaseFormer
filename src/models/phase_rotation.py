import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseRotation(nn.Module):
    """Phase rotation (PhaseFormer experiment plan, stage 4).

    Rotates pairs of latent phase features in the feature plane by a
    per-position angle ``theta`` predicted from the input periodic features:

        x1, x2 = chunk last dim into pairs
        x1' = x1*cos(theta) - x2*sin(theta)
        x2' = x1*sin(theta) + x2*cos(theta)

    A 2D rotation preserves feature magnitude and locally re-orients the phase
    representation, letting phase change act directly on the latent features.
    The final layer is zero-initialized so theta = tanh(0) = 0, a rotation of
    angle 0, i.e. the identity (warm start).

    Inputs:
        z: (B, C, L, D) latent phase tokens
        cond: (B, C, L, cond_dim) input periodic features used to predict theta

    Output keeps the (B, C, L, D) layout.
    """

    def __init__(self, cond_dim: int, hidden: int = 8):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Diagnostic hook: mean |theta| from the last forward pass.
        self.last_mean_theta = 0.0

    def forward(self, z, cond):
        B, C, L, D = z.shape
        theta = torch.tanh(self.net(cond).squeeze(-1))  # (B, C, L)
        cos = torch.cos(theta).unsqueeze(-1)  # (B, C, L, 1)
        sin = torch.sin(theta).unsqueeze(-1)

        # Pad the last dim to an even number of features for pairwise rotation.
        odd = D % 2
        z_padded = F.pad(z, (0, 1)) if odd else z
        x1 = z_padded[..., 0::2]
        x2 = z_padded[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.zeros_like(z_padded)
        out[..., 0::2] = o1
        out[..., 1::2] = o2
        if odd:
            out = out[..., :D]

        with torch.no_grad():
            self.last_mean_theta = float(theta.abs().mean())
        return out
