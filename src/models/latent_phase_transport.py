import torch
import torch.nn as nn


class LatentPhaseTransportDecoder(nn.Module):
    """Forecast through horizon-conditioned transport of latent phase tokens.

    For each future period, a small local head predicts a circular transport
    kernel for every phase token. The transported latent is decoded with the
    same horizon-specific linear form as the original PhasePredictor. When the
    transport kernel is the identity, this module is exactly the original
    linear phase predictor; it never copies observed values or creates a
    separate time-domain forecast.

    Input:  z shaped (B, C, L, D)
    Output: future phase values shaped (B, C, L, P_out)
    """

    def __init__(
        self,
        *,
        p_out: int,
        latent_dim: int,
        hidden: int = 8,
        max_shift: int = 1,
        temperature: float = 1.0,
        prior_logit: float = 5.0,
    ):
        super().__init__()
        if p_out < 1:
            raise ValueError("p_out must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if hidden < 1:
            raise ValueError("hidden must be positive")
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.p_out = int(p_out)
        self.max_shift = int(max_shift)
        self.num_shifts = 2 * self.max_shift + 1
        self.temperature = float(temperature)
        self.shifts = tuple(range(-self.max_shift, self.max_shift + 1))

        # Cross-phase routing already injects global context into each local
        # token. Keeping the transport head local avoids the destructive phase
        # pooling used by CPTD while remaining channel-independent and small.
        self.transport_head = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_shifts),
        )
        self.value_projection = nn.Linear(latent_dim, self.p_out)
        self._initialize_identity_prior(prior_logit)

    def _initialize_identity_prior(self, prior_logit: float):
        final = self.transport_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[self.max_shift] = float(prior_logit)

    def _transport_weights(self, z, horizon_index: int):
        coordinate = float(horizon_index + 1) / float(self.p_out)
        horizon = torch.full_like(z[..., :1], coordinate)
        logits = self.transport_head(torch.cat([z, horizon], dim=-1))
        return torch.softmax(logits / self.temperature, dim=-1)

    def _transport(self, z, weights):
        transported = torch.zeros_like(z)
        for shift_index, shift in enumerate(self.shifts):
            shifted = torch.roll(z, shifts=shift, dims=2)
            weight = weights[..., shift_index : shift_index + 1]
            transported = transported + weight * shifted
        return transported

    def forward(self, z):
        if z.ndim != 4:
            raise ValueError("z must be a rank-4 tensor shaped (B, C, L, D)")

        future_phases = []
        for horizon_index in range(self.p_out):
            weights = self._transport_weights(z, horizon_index)
            transported = self._transport(z, weights)
            values = torch.einsum(
                "bcld,d->bcl",
                transported,
                self.value_projection.weight[horizon_index],
            )
            values = values + self.value_projection.bias[horizon_index]
            future_phases.append(values)
        return torch.stack(future_phases, dim=-1)

    def diagnostics(self, z):
        """Return transport distributions and entropy for bad-case analysis."""
        weights = torch.stack(
            [
                self._transport_weights(z, horizon_index)
                for horizon_index in range(self.p_out)
            ],
            dim=2,
        )  # (B,C,P_out,L,S)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "shift_weights": weights,
            "shift_entropy": entropy,
            "identity_weight": weights[..., self.max_shift],
        }
