import torch
import torch.nn as nn


class CircularPhaseTransportDecoder(nn.Module):
    """Generate future periods by transporting recent phase profiles.

    The decoder operates only on the phase tensor. It selects a convex mixture
    of recent period profiles, transports each profile by a small distribution
    of circular phase shifts, and predicts bounded amplitude and period-level
    evolution. There is no free-form time-domain forecast or residual fusion.

    Inputs:
        z: routed phase latents shaped (B, C, L, D).
        phase_series: observed normalized phase histories shaped (B, C, L, P).

    Output:
        Future phase values shaped (B, C, L, P_out).
    """

    def __init__(
        self,
        *,
        p_out: int,
        latent_dim: int,
        hidden: int = 8,
        memory_size: int = 3,
        max_shift: int = 1,
        max_log_amplitude: float = 0.5,
        max_level_step: float = 1.0,
        temperature: float = 1.0,
        prior_logit: float = 3.0,
    ):
        super().__init__()
        if p_out < 1:
            raise ValueError("p_out must be positive")
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if hidden < 1:
            raise ValueError("hidden must be positive")

        self.p_out = int(p_out)
        self.memory_size = int(memory_size)
        self.max_shift = int(max_shift)
        self.num_shifts = 2 * self.max_shift + 1
        self.max_log_amplitude = max(float(max_log_amplitude), 0.0)
        self.max_level_step = max(float(max_level_step), 0.0)
        self.temperature = max(float(temperature), 1e-4)
        self.shifts = tuple(range(-self.max_shift, self.max_shift + 1))

        # Horizon coordinates make one small head usable for every P_out.
        # Outputs: memory logits, per-memory shift logits, amplitude, level step.
        self.output_dim = (
            self.memory_size
            + self.memory_size * self.num_shifts
            + 2
        )
        self.dynamics_head = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.output_dim),
        )
        self._initialize_transport_prior(prior_logit)

    def _initialize_transport_prior(self, prior_logit: float):
        """Start close to last-period persistence with zero level drift."""
        final = self.dynamics_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            # Memory index zero is the most recent observed period.
            final.bias[0] = float(prior_logit)
            shift_start = self.memory_size
            center = self.max_shift
            for memory_index in range(self.memory_size):
                offset = shift_start + memory_index * self.num_shifts + center
                final.bias[offset] = float(prior_logit)

    def _recent_profiles(self, phase_series):
        """Return newest-first period profiles with a fixed memory dimension."""
        periods = phase_series.size(-1)
        available = min(self.memory_size, periods)
        recent = phase_series[..., -available:]
        if available < self.memory_size:
            oldest = recent[..., :1]
            padding = oldest.expand(
                *oldest.shape[:-1], self.memory_size - available
            )
            recent = torch.cat([padding, recent], dim=-1)
        # (B,C,L,K) -> (B,C,K,L), newest period at index zero.
        return recent.flip(dims=(-1,)).permute(0, 1, 3, 2).contiguous()

    def _predict_parameters(self, z):
        pooled = z.mean(dim=2)  # (B,C,D), invariant to phase-token ordering.
        horizon = torch.linspace(
            1.0 / self.p_out,
            1.0,
            self.p_out,
            device=z.device,
            dtype=z.dtype,
        )
        horizon = horizon.view(1, 1, self.p_out, 1).expand(
            z.size(0), z.size(1), -1, -1
        )
        context = torch.cat(
            [pooled.unsqueeze(2).expand(-1, -1, self.p_out, -1), horizon],
            dim=-1,
        )
        raw = self.dynamics_head(context)

        memory_end = self.memory_size
        shift_end = memory_end + self.memory_size * self.num_shifts
        memory_logits = raw[..., :memory_end]
        shift_logits = raw[..., memory_end:shift_end].view(
            *raw.shape[:-1], self.memory_size, self.num_shifts
        )
        amplitude_raw = raw[..., shift_end]
        level_raw = raw[..., shift_end + 1]
        return {
            "memory_weights": torch.softmax(
                memory_logits / self.temperature, dim=-1
            ),
            "shift_weights": torch.softmax(
                shift_logits / self.temperature, dim=-1
            ),
            "amplitude": torch.exp(
                self.max_log_amplitude * torch.tanh(amplitude_raw)
            ),
            "level_raw": level_raw,
        }

    def _level_scale(self, phase_series):
        period_levels = phase_series.mean(dim=2)  # (B,C,P)
        series_scale = phase_series.var(
            dim=(2, 3), unbiased=False
        ).clamp_min(1e-8).sqrt()
        if period_levels.size(-1) < 2:
            recent_change = torch.zeros_like(series_scale)
        else:
            level_diff = period_levels[..., 1:] - period_levels[..., :-1]
            window = min(self.memory_size, level_diff.size(-1))
            recent_change = level_diff[..., -window:].square().mean(dim=-1).sqrt()
        # The small series-scale floor keeps level learning alive for histories
        # whose observed period means happen to be constant.
        return (recent_change + 0.1 * series_scale).clamp_min(1e-4)

    def _decode(self, z, phase_series):
        if z.ndim != 4 or phase_series.ndim != 4:
            raise ValueError("z and phase_series must both be rank-4 tensors")
        if phase_series.size(-1) < 1:
            raise ValueError("phase_series must contain at least one observed period")
        if z.shape[:3] != phase_series.shape[:3]:
            raise ValueError("z and phase_series must share B, C, and L dimensions")

        parameters = self._predict_parameters(z)
        profiles = self._recent_profiles(phase_series)  # (B,C,K,L)
        profile_levels = profiles.mean(dim=-1, keepdim=True)
        centered_profiles = profiles - profile_levels
        rolled_profiles = torch.stack(
            [
                torch.roll(centered_profiles, shifts=shift, dims=-1)
                for shift in self.shifts
            ],
            dim=-2,
        )  # (B,C,K,S,L)

        joint_weights = (
            parameters["memory_weights"].unsqueeze(-1)
            * parameters["shift_weights"]
        )  # (B,C,P_out,K,S)
        transported_shape = torch.einsum(
            "bcjks,bcksl->bcjl", joint_weights, rolled_profiles
        )
        transported_shape = transported_shape * parameters["amplitude"].unsqueeze(-1)

        last_level = phase_series[..., -1].mean(dim=2)  # (B,C)
        level_step = (
            self.max_level_step
            * self._level_scale(phase_series).unsqueeze(-1)
            * torch.tanh(parameters["level_raw"])
        )
        future_level = last_level.unsqueeze(-1) + level_step.cumsum(dim=-1)
        future_periods = transported_shape + future_level.unsqueeze(-1)
        output = future_periods.permute(0, 1, 3, 2).contiguous()

        diagnostics = {
            **parameters,
            "level_step": level_step,
            "future_level": future_level,
        }
        return output, diagnostics

    def forward(self, z, phase_series):
        output, _ = self._decode(z, phase_series)
        return output

    def diagnostics(self, z, phase_series):
        """Expose interpretable transport parameters for bad-case analysis."""
        _, diagnostics = self._decode(z, phase_series)
        return diagnostics
