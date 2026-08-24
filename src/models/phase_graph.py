import torch
import torch.nn as nn


class PhaseGraph(nn.Module):
    """Circular phase graph message passing (pure-phase plan, stage 3).

    CrossPhaseRoutingLayer models token interaction with two-stage router
    attention; CircularAttentionBias only rescales the router attention scores.
    This module instead performs explicit *local message passing on the cycle
    graph* over the L phase slots: each slot gathers from its k nearest
    neighbors on each side of the phase circle (edges wrap around, so slot 0 is
    adjacent to slot L-1). Edge weights are shared per offset (translation
    invariant around the ring) and a message network transforms each neighbor
    latent before the weighted sum:

        message_l = sum_{o in [-k..k]\{0}} w_o * msg_net(roll_l(Z, o))
        Z_l'      = Z_l + message_l

    This is a graph convolution / diffusion step on the cycle, giving the
    interaction layer a geometry-aware inductive bias distinct from router
    attention: local, position-aware aggregation over cycle-adjacent phase
    tokens.

    Warm start: the message network's final layer is zero-initialized, so at
    initialization message == 0 and forward() is the identity (flag-off
    byte-equivalent behavior for the shared path).

    Diagnostics from the last forward pass (analysis only, no-grad):
      - last_mean_message: mean |message| over (sample, channel, slot)
      - last_edge_weight: (2k,) edge weights after the last forward
    """

    def __init__(self, dim: int, hidden: int = 16, k: int = 2):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.k = k
        # Edge weights shared per offset (exclude o=0 self-loop). Index i maps
        # offset o to: i = o - 1 for o > 0, i = k + o for o < 0 (i.e. offsets
        # -k..-1,1..k -> 0..2k-1).
        self.edge_w = nn.Parameter(torch.ones(2 * k))
        # Message network: transforms each neighbor latent before aggregation.
        self.msg_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init final layer -> message == 0 -> identity at init.
        nn.init.zeros_(self.msg_net[-1].weight)
        nn.init.zeros_(self.msg_net[-1].bias)

        # Diagnostic hooks (analysis only, no parameters).
        self.last_mean_message = 0.0

    @staticmethod
    def _offset_index(o, k):
        return o - 1 if o > 0 else k + o

    def forward(self, Z):  # (B, C, L, D)
        B, C, L, D = Z.shape
        message = torch.zeros_like(Z)
        for o in range(-self.k, self.k + 1):
            if o == 0:
                continue
            w = self.edge_w[self._offset_index(o, self.k)]
            shifted = torch.roll(Z, o, dims=2)  # circular shift along slots
            message = message + w * self.msg_net(shifted)
        out = Z + message

        with torch.no_grad():
            self.last_mean_message = float(message.abs().mean())
        return out
