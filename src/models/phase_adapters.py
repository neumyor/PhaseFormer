import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Normalization over time (per-sample, per-variable).

    Normalizes inputs along the temporal axis for each sample and variable.
    Input is shaped (B, L, C). The stored statistics allow exact de-normalization
    at the output stage so predictions can be mapped back to the original scale.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):  # x: (B, L, C)
        mu = x.mean(dim=1, keepdim=True)  # (B,1,C)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mu, sigma)

    def normalize_with_stats(self, x, stats):
        """Normalize a second branch in the coordinate system of ``normalize``.

        This deliberately does not recompute per-branch moments.  It is used by
        asymmetric residual-input experiments, where only branch visibility is
        allowed to change.
        """
        mu, sigma = stats
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn

    def denormalize(self, y, stats):  # y: (B, L', C)
        mu, sigma = stats
        return y * sigma + mu


class WeakPeriodResidualHead(nn.Module):
    """NLinear-style temporal residual path for weakly periodic series.

    The phase path assumes that observations with the same phase index across
    periods are strongly related. Weak-periodic data can violate that assumption
    through drift or phase jitter, so this head directly extrapolates the
    centered recent trajectory and adds the last value back as a persistence
    anchor.
    """

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):  # x: (B, L, C), normalized scale
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        delta = self.linear(centered).permute(0, 2, 1).contiguous()
        return delta + last.expand(-1, delta.size(1), -1)


class PeriodPositionEncodedResidualHead(nn.Module):
    """NLinear residual augmented by a position-only periodic retrieval path.

    A positional kernel matches every forecast position to historical positions,
    applies the resulting weights to the centered history, and convexly blends
    that periodic copy with the ordinary NLinear delta.  Unlike adding a fixed
    position vector before a linear map (which can collapse into an ordinary
    bias/weight), this construction makes the historical-to-future matching
    induced by the position encoding directly observable.

    ``encoding_type`` selects only the position representation; the NLinear
    map, attention temperature, recency decay, and horizon-wise blend are shared
    across all variants for a controlled comparison.
    """

    ENCODING_TYPES = {
        "st_informer",
        "cycle",
        "harmonic",
        "traffic",
        "time2vec",
        "lff",
        "calendar",
    }

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        encoding_type: str = "harmonic",
        pe_dim: int = 16,
        temperature: float = 0.1,
        cycle_decay: float = 0.1,
        blend_init: float = 0.1,
        learn_blend: bool = True,
    ):
        super().__init__()
        if encoding_type not in self.ENCODING_TYPES:
            raise ValueError(
                f"Unsupported periodic residual position encoding: {encoding_type}"
            )
        if seq_len <= 0 or pred_len <= 0 or period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period_len > 1")
        if pe_dim < 2:
            raise ValueError("pe_dim must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if cycle_decay < 0:
            raise ValueError("cycle_decay must be non-negative")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.encoding_type = encoding_type
        self.pe_dim = int(pe_dim)
        self.temperature = float(temperature)
        self.cycle_decay = float(cycle_decay)

        # Keep exactly the same NLinear parameterization and zero initialization
        # as WeakPeriodResidualHead; PE adds only a structured periodic pathway.
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        blend_init = min(max(float(blend_init), 1e-4), 1.0 - 1e-4)
        blend_logit = float(torch.logit(torch.tensor(blend_init)))
        blend_logits = torch.full((self.pred_len,), blend_logit)
        if learn_blend:
            self.blend_logits = nn.Parameter(blend_logits)
        else:
            self.register_buffer("blend_logits", blend_logits)

        if self.encoding_type == "time2vec":
            num_periodic = max(1, self.pe_dim - 1)
            harmonic = torch.arange(1, num_periodic + 1, dtype=torch.float32)
            harmonic = (harmonic - 1).remainder(8) + 1
            init_frequency = 2.0 * math.pi * harmonic / self.period_len
            self.time2vec_frequency = nn.Parameter(init_frequency)
            self.time2vec_phase = nn.Parameter(torch.zeros(num_periodic))
            self.time2vec_linear_weight = nn.Parameter(torch.ones(1))
            self.time2vec_linear_bias = nn.Parameter(torch.zeros(1))
        elif self.encoding_type == "lff":
            num_frequency = max(1, self.pe_dim // 2)
            harmonic = torch.arange(1, num_frequency + 1, dtype=torch.float32)
            init_frequency = 2.0 * math.pi * harmonic / self.period_len
            self.register_buffer("lff_base_frequency", init_frequency)
            self.lff_log_frequency_scale = nn.Parameter(torch.zeros(num_frequency))

        self.last_beta = None
        self.last_attention = None
        self.last_attention_entropy = None
        self.last_top_lags = None

    @staticmethod
    def _paired_sincos(angle):
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(-2)

    def _st_informer_encoding(self, positions, dim=None):
        dim = int(dim or self.pe_dim)
        num_frequency = max(1, dim // 2)
        index = torch.arange(num_frequency, device=positions.device, dtype=positions.dtype)
        denominator = torch.pow(
            positions.new_tensor(1000.0), 2.0 * index / max(float(dim), 1.0)
        )
        angle = positions.unsqueeze(-1) / denominator
        return self._paired_sincos(angle)[..., :dim]

    def _cycle_encoding(self, positions, harmonics):
        harmonic = torch.arange(
            1, harmonics + 1, device=positions.device, dtype=positions.dtype
        )
        angle = 2.0 * math.pi * positions.unsqueeze(-1) * harmonic / self.period_len
        return self._paired_sincos(angle)

    def _index_encoding(self, positions):
        if self.encoding_type == "st_informer":
            return self._st_informer_encoding(positions)
        if self.encoding_type == "cycle":
            return self._cycle_encoding(positions, harmonics=1)
        if self.encoding_type == "harmonic":
            return self._cycle_encoding(positions, harmonics=4)
        if self.encoding_type == "traffic":
            absolute = self._st_informer_encoding(positions, dim=self.pe_dim)
            periodic = self._cycle_encoding(
                positions, harmonics=max(1, self.pe_dim // 2)
            )[..., : self.pe_dim]
            return absolute + periodic
        if self.encoding_type == "time2vec":
            scale = max(float(self.seq_len + self.pred_len - 1), 1.0)
            normalized = 2.0 * positions / scale - 1.0
            linear = (
                normalized.unsqueeze(-1) * self.time2vec_linear_weight
                + self.time2vec_linear_bias
            )
            periodic = torch.sin(
                positions.unsqueeze(-1) * self.time2vec_frequency
                + self.time2vec_phase
            )
            return torch.cat((linear, periodic), dim=-1)
        if self.encoding_type == "lff":
            frequency = self.lff_base_frequency * torch.exp(
                self.lff_log_frequency_scale
            )
            angle = positions.unsqueeze(-1) * frequency
            return self._paired_sincos(angle)[..., : self.pe_dim]
        raise RuntimeError("calendar encoding requires timestamp marks")

    @staticmethod
    def _calendar_encoding(marks):
        """Encode raw [month, day, weekday, hour, optional 15-min slot] marks."""
        if marks is None or marks.size(-1) < 4:
            raise ValueError("calendar PE requires raw timestamp marks with >=4 fields")
        month = marks[..., 0]
        day = marks[..., 1]
        weekday = marks[..., 2]
        hour = marks[..., 3]
        minute_slot = marks[..., 4] if marks.size(-1) >= 5 else torch.zeros_like(hour)
        time_of_day = (hour + minute_slot / 4.0) / 24.0
        cycles = torch.stack(
            (
                time_of_day,
                2.0 * time_of_day,
                weekday / 7.0,
                (day - 1.0) / 31.0,
                (month - 1.0) / 12.0,
            ),
            dim=-1,
        )
        return PeriodPositionEncodedResidualHead._paired_sincos(2.0 * math.pi * cycles)

    def _attention_weights(self, x, x_mark_enc=None, x_mark_dec=None):
        dtype, device = x.dtype, x.device
        history_pos = torch.arange(self.seq_len, device=device, dtype=dtype)
        future_pos = torch.arange(
            self.seq_len,
            self.seq_len + self.pred_len,
            device=device,
            dtype=dtype,
        )

        if self.encoding_type == "calendar":
            if x_mark_enc is None or x_mark_dec is None:
                raise ValueError("calendar PE requires encoder and decoder timestamp marks")
            history_embedding = self._calendar_encoding(x_mark_enc.to(dtype=dtype))
            future_embedding = self._calendar_encoding(
                x_mark_dec[:, -self.pred_len :, :].to(dtype=dtype)
            )
            history_embedding = F.normalize(history_embedding, dim=-1, eps=1e-6)
            future_embedding = F.normalize(future_embedding, dim=-1, eps=1e-6)
            similarity = torch.einsum(
                "bhd,bld->bhl", future_embedding, history_embedding
            )
        else:
            history_embedding = F.normalize(
                self._index_encoding(history_pos), dim=-1, eps=1e-6
            )
            future_embedding = F.normalize(
                self._index_encoding(future_pos), dim=-1, eps=1e-6
            )
            similarity = torch.matmul(future_embedding, history_embedding.transpose(0, 1))

        lag_cycles = (future_pos[:, None] - history_pos[None, :]) / self.period_len
        logits = similarity / self.temperature - self.cycle_decay * lag_cycles
        if similarity.ndim == 3:
            logits = logits.unsqueeze(0) if logits.ndim == 2 else logits
        return torch.softmax(logits, dim=-1)

    def forward_components(self, x, x_mark_enc=None, x_mark_dec=None):
        """Return the NLinear and periodic-copy forecasts before blending."""
        if x.size(1) != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x.size(1)}")
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()  # (B,C,L)
        linear_delta = self.linear(centered).permute(0, 2, 1).contiguous()
        attention = self._attention_weights(x, x_mark_enc, x_mark_dec)
        if attention.ndim == 2:
            periodic_delta = torch.einsum("hl,bcl->bhc", attention, centered)
        else:
            periodic_delta = torch.einsum("bhl,bcl->bhc", attention, centered)

        with torch.no_grad():
            self.last_attention = attention.detach()
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
            self.last_attention_entropy = float(entropy.mean().detach())
            mean_attention = attention.mean(dim=0) if attention.ndim == 3 else attention
            top_history = mean_attention.argmax(dim=-1)
            future_index = torch.arange(
                self.seq_len,
                self.seq_len + self.pred_len,
                device=x.device,
            )
            self.last_top_lags = (future_index - top_history).detach()

        anchor = last.expand(-1, self.pred_len, -1)
        return anchor + linear_delta, anchor + periodic_delta

    def forward(self, x, x_mark_enc=None, x_mark_dec=None):
        # x: (B, L, C), all computations stay on the normalized model scale.
        y_linear, y_periodic = self.forward_components(
            x, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec
        )
        beta = torch.sigmoid(self.blend_logits).view(1, self.pred_len, 1)
        with torch.no_grad():
            self.last_beta = beta.detach()
        return (1.0 - beta) * y_linear + beta * y_periodic


class ChannelWiseWeakPeriodResidualHead(nn.Module):
    """Channel-independent temporal residual path for heterogeneous variables."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(enc_in, pred_len, seq_len))
        self.bias = nn.Parameter(torch.zeros(enc_in, pred_len))

    def forward(self, x):  # x: (B, L, C), normalized scale
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        delta = torch.einsum("bcl,cpl->bcp", centered, self.weight)
        delta = delta + self.bias.unsqueeze(0)
        return delta.permute(0, 2, 1).contiguous() + last.expand(-1, delta.size(-1), -1)


class LowPassWeakPeriodResidualHead(nn.Module):
    """Low-pass temporal residual path for noisy weak-period drift.

    The branch estimates extrapolation coefficients from a moving-averaged
    trajectory, but anchors the forecast at the original last value so short-term
    level information is not smoothed away.
    """

    def __init__(self, seq_len: int, pred_len: int, window: int = 25):
        super().__init__()
        self.window = max(3, int(window))
        if self.window % 2 == 0:
            self.window += 1
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _smooth(self, x):
        pad = self.window // 2
        series = x.permute(0, 2, 1).contiguous()
        series = F.pad(series, (pad, pad), mode="replicate")
        smoothed = F.avg_pool1d(series, kernel_size=self.window, stride=1)
        return smoothed.permute(0, 2, 1).contiguous()

    def forward(self, x):  # x: (B, L, C), normalized scale
        last = x[:, -1:, :]
        smooth = self._smooth(x)
        smooth_last = smooth[:, -1:, :]
        centered = (smooth - smooth_last).permute(0, 2, 1).contiguous()
        delta = self.linear(centered).permute(0, 2, 1).contiguous()
        return delta + last.expand(-1, delta.size(1), -1)


class AdaptiveWeakPeriodGate(nn.Module):
    """Sample-wise residual gate driven by phase instability features."""

    def __init__(self, enc_in: int, hidden: int = 8, gate_init: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.net[-1].bias, float(torch.logit(torch.tensor(gate_init))))
        self.channel_bias = nn.Parameter(torch.zeros(1, enc_in, 1))

    def forward(self, x_in, phase_series):  # x_in: (B,L,C), phase_series: (B,C,L,P)
        phase_diff = phase_series[..., 1:] - phase_series[..., :-1]
        phase_instability = phase_diff.abs().mean(dim=(2, 3))
        recent_diff = x_in[:, 1:, :] - x_in[:, :-1, :]
        recent_volatility = recent_diff.abs().mean(dim=1)
        if phase_series.size(-1) >= 2:
            phase_trend = (phase_series[..., -1] - phase_series[..., -2]).abs().mean(dim=2)
        else:
            phase_trend = torch.zeros_like(phase_instability)
        features = torch.stack(
            [phase_instability, recent_volatility, phase_trend],
            dim=-1,
        )
        logits = self.net(features) + self.channel_bias
        return torch.sigmoid(logits).permute(0, 2, 1)


class TimeMarkAdjustmentHead(nn.Module):
    """Future time-feature correction for weakly periodic series."""

    def __init__(self, mark_dim: int, enc_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mark_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, enc_in),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_mark_dec, pred_len: int):
        future_mark = x_mark_dec[:, -pred_len:, :]
        return self.net(future_mark)


class PhaseLocalTrendHead(nn.Module):
    """Reliability-gated phase-shape drift correction.

    The correction only extrapolates within-period shape drift. Period-level
    drift is removed from the slope estimate, and a sample-wise reliability
    score suppresses the adapter when same-phase histories are noisy.
    """

    def __init__(
        self,
        num_periods_output: int,
        enc_in: int,
        window: int = 3,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.num_periods_output = num_periods_output
        self.window = max(2, int(window))
        self.max_gate = 0.005
        gate_init = min(max(float(gate_init), -self.max_gate + 1e-4), self.max_gate - 1e-4)
        gate_raw = torch.atanh(torch.tensor(gate_init / self.max_gate))
        self.gate = nn.Parameter(torch.full((1, enc_in, 1, 1), float(gate_raw)))

    def _diagnostics(self, phase_series):  # (B, C, L, P_in)
        template = phase_series.mean(dim=-1, keepdim=True)
        phase_signal = template.squeeze(-1).var(dim=2, unbiased=False, keepdim=True)
        phase_noise = phase_series.var(dim=-1, unbiased=False, keepdim=True)
        phase_noise = phase_noise.mean(dim=2, keepdim=True)
        reliability = phase_signal.unsqueeze(-1) / (
            phase_signal.unsqueeze(-1) + phase_noise + 1e-6
        )

        recent = phase_series[..., -min(self.window, phase_series.size(-1)) :]
        diffs = recent[..., 1:] - recent[..., :-1]
        mean_diff = diffs.mean(dim=-1)
        slope_consistency = mean_diff.abs() / (diffs.abs().mean(dim=-1) + 1e-6)
        raw_slope = mean_diff * slope_consistency
        shape_slope = raw_slope - raw_slope.mean(dim=2, keepdim=True)

        steps = torch.arange(
            1,
            self.num_periods_output + 1,
            device=phase_series.device,
            dtype=phase_series.dtype,
        )
        tau = max(float(self.num_periods_output) / 2.0, 1.0)
        saturated_steps = 1.0 - torch.exp(-steps / tau)
        correction = shape_slope.unsqueeze(-1) * saturated_steps.view(1, 1, 1, -1)
        horizon_scale = float(self.num_periods_output) ** -0.5
        base_gate = self.max_gate * horizon_scale * torch.tanh(self.gate)
        effective_gate = base_gate * reliability
        return {
            "template": template,
            "phase_signal": phase_signal,
            "phase_noise": phase_noise,
            "reliability": reliability,
            "slope_consistency": slope_consistency,
            "raw_slope": raw_slope,
            "shape_slope": shape_slope,
            "steps": saturated_steps,
            "base_gate": base_gate,
            "effective_gate": effective_gate,
            "correction": correction,
        }

    def diagnostics(self, phase_series):
        return self._diagnostics(phase_series)

    def forward(self, phase_series):  # (B, C, L, P_in)
        if phase_series.size(-1) < 2:
            return torch.zeros(
                *phase_series.shape[:-1],
                self.num_periods_output,
                device=phase_series.device,
                dtype=phase_series.dtype,
            )
        info = self._diagnostics(phase_series)
        return info["effective_gate"] * info["correction"]


class PhaseUncertaintyShrinkage(nn.Module):
    """Empirical-Bayes shrinkage of unreliable same-phase observations.

    Weak-period series can be written as x_{l,k}=p_l+d_k+eps_{l,k}, where l is
    the phase slot and k is the period index. High same-phase variance over k
    makes the phase history unreliable; useful periodic structure appears as
    variance of the phase template over l. This layer shrinks noisy period
    histories toward the phase template before routing, while a small learnable
    trend term preserves low-frequency drift across periods.
    """

    def __init__(
        self,
        enc_in: int,
        min_reliability: float = 0.35,
        trend_gate_init: float = 0.05,
        noise_floor: float = 1e-6,
    ):
        super().__init__()
        self.min_reliability = min(max(float(min_reliability), 0.0), 1.0)
        self.noise_floor = max(float(noise_floor), 1e-8)
        trend_gate_init = min(max(float(trend_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = torch.logit(torch.tensor(trend_gate_init))
        self.trend_gate = nn.Parameter(torch.full((1, enc_in, 1, 1), float(gate_logit)))

    def forward(self, phase_series):  # (B, C, L, P)
        template = phase_series.mean(dim=-1, keepdim=True)
        phase_signal = template.squeeze(-1).var(dim=2, unbiased=False, keepdim=True)
        phase_noise = phase_series.var(dim=-1, unbiased=False, keepdim=True)
        reliability = phase_signal.unsqueeze(-1) / (
            phase_signal.unsqueeze(-1) + phase_noise + self.noise_floor
        )
        reliability = self.min_reliability + (1.0 - self.min_reliability) * reliability
        shrunk = template + reliability * (phase_series - template)

        if phase_series.size(-1) < 2:
            return shrunk

        period_axis = torch.linspace(
            0.0,
            1.0,
            phase_series.size(-1),
            device=phase_series.device,
            dtype=phase_series.dtype,
        ).view(1, 1, 1, -1)
        centered_axis = period_axis - period_axis.mean(dim=-1, keepdim=True)
        denom = torch.square(centered_axis).sum(dim=-1, keepdim=True).clamp_min(
            self.noise_floor
        )
        slope = ((phase_series - template) * centered_axis).sum(dim=-1, keepdim=True) / denom
        trend = slope * centered_axis
        return shrunk + torch.sigmoid(self.trend_gate) * trend


class PhasePeriodLevelCalibration(nn.Module):
    """Calibrate forecast period means without removing phase-shape inputs."""

    def __init__(
        self,
        num_periods_output: int,
        enc_in: int,
        slope_window: int = 3,
        level_gate_init: float = 0.1,
        slope_gate_init: float = 0.05,
    ):
        super().__init__()
        self.num_periods_output = num_periods_output
        self.slope_window = max(2, int(slope_window))
        level_gate_init = min(max(float(level_gate_init), 1e-4), 1.0 - 1e-4)
        slope_gate_init = min(max(float(slope_gate_init), 1e-4), 1.0 - 1e-4)
        self.level_gate = nn.Parameter(
            torch.full((1, enc_in, 1, 1), float(torch.logit(torch.tensor(level_gate_init))))
        )
        self.slope_gate = nn.Parameter(
            torch.full((1, enc_in, 1, 1), float(torch.logit(torch.tensor(slope_gate_init))))
        )

    def _anchor_level(self, phase_series):
        period_level = phase_series.mean(dim=2, keepdim=True)
        last_level = period_level[..., -1:]
        if period_level.size(-1) < 2:
            return last_level.expand(-1, -1, -1, self.num_periods_output)
        window = min(self.slope_window, period_level.size(-1))
        recent = period_level[..., -window:]
        slope = (recent[..., 1:] - recent[..., :-1]).mean(dim=-1, keepdim=True)
        steps = torch.arange(
            1,
            self.num_periods_output + 1,
            device=phase_series.device,
            dtype=phase_series.dtype,
        ).view(1, 1, 1, -1)
        return last_level + torch.sigmoid(self.slope_gate) * slope * steps

    def forward(self, y_phase_steps, phase_series):  # (B,C,L,P_out), (B,C,L,P_in)
        anchor_level = self._anchor_level(phase_series)
        predicted_level = y_phase_steps.mean(dim=2, keepdim=True)
        correction = anchor_level - predicted_level
        return y_phase_steps + torch.sigmoid(self.level_gate) * correction


class PhaseSparseEventCalibration(nn.Module):
    """Restore sparse positive phase events without adding a sequence residual.

    Weather bad cases often have weakly periodic sparse peaks: the period mean is
    reasonable, but the forecast flattens rare positive excursions. This module
    estimates a recent phase-slot event envelope and only reallocates forecast
    mass toward historically active phase slots when the predicted positive
    excursion is too small.
    """

    def __init__(
        self,
        enc_in: int,
        window: int = 3,
        gate_init: float = 0.05,
        max_boost: float = 1.0,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.window = max(1, int(window))
        self.max_boost = max(float(max_boost), 0.0)
        self.temperature = max(float(temperature), 1e-4)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        gate_logit = torch.logit(torch.tensor(gate_init))
        self.gate = nn.Parameter(torch.full((1, enc_in, 1, 1), float(gate_logit)))

    def forward(self, y_phase_steps, phase_series):  # (B,C,L,P_out), (B,C,L,P_in)
        recent_periods = phase_series.permute(0, 1, 3, 2).contiguous()
        recent = recent_periods[:, :, -min(self.window, recent_periods.size(2)) :, :]
        recent_centered = recent - recent.mean(dim=-1, keepdim=True)
        event_template = recent_centered.clamp_min(0.0).mean(dim=2)
        event_peak = event_template.amax(dim=-1, keepdim=True).unsqueeze(-1)

        y_periods = y_phase_steps.permute(0, 1, 3, 2).contiguous()
        pred_mean = y_periods.mean(dim=-1, keepdim=True)
        pred_centered = y_periods - pred_mean
        pred_peak = pred_centered.clamp_min(0.0).amax(dim=-1, keepdim=True)

        gap = (event_peak - pred_peak).clamp(min=0.0, max=self.max_boost)
        weights = torch.softmax(event_template / self.temperature, dim=-1).unsqueeze(2)
        correction = gap * weights
        correction = correction - correction.mean(dim=-1, keepdim=True)
        calibrated = y_periods + torch.sigmoid(self.gate) * correction
        return calibrated.permute(0, 1, 3, 2).contiguous()


class PhaseNoiseHighFreqDamping(nn.Module):
    """Damp high-frequency forecast oscillations when phase history is noisy."""

    def __init__(
        self,
        strength: float = 0.5,
        noise_threshold: float = 1.0,
        noise_temperature: float = 0.2,
        window: int = 7,
    ):
        super().__init__()
        self.strength = min(max(float(strength), 0.0), 1.0)
        self.noise_threshold = float(noise_threshold)
        self.noise_temperature = max(float(noise_temperature), 1e-4)
        self.window = max(3, int(window))
        if self.window % 2 == 0:
            self.window += 1

    def _smooth(self, y_hat):
        pad = self.window // 2
        series = y_hat.permute(0, 2, 1).contiguous()
        series = F.pad(series, (pad, pad), mode="replicate")
        smoothed = F.avg_pool1d(series, kernel_size=self.window, stride=1)
        return smoothed.permute(0, 2, 1).contiguous()

    def forward(self, y_hat, phase_series):  # y_hat: (B,T,C), phase_series: (B,C,L,P)
        noise = phase_series.var(dim=-1, unbiased=False).mean(dim=2)
        trigger = torch.sigmoid((noise - self.noise_threshold) / self.noise_temperature)
        damping = 1.0 - self.strength * trigger.unsqueeze(1)
        smooth = self._smooth(y_hat)
        high_freq = y_hat - smooth
        return smooth + damping * high_freq


class ReliabilityCoupledResidualFusion(nn.Module):
    """Reliability-Coupled Residual Fusion (RCRF).

    Convexly fuses the phase forecast and the NLinear weak-period residual:

        r     = Var_l(mean_k x_lk) / (Var_l(mean_k x_lk) + mean_l Var_k(x_lk) + eps)
        s     = s_max * tanh(s_raw)
        alpha = sigmoid(logit(alpha_0) + s * (1 - r))
        y     = (1 - alpha) * y_phase + alpha * y_residual

    ``r`` is a per-sample/channel reliability computed from the RAW phase series
    (before uncertainty shrinkage), so the correction modules never change the
    evidence that gates their own contribution.  ``s`` is a learnable, tanh-bounded
    sensitivity: s=0 keeps the gate at the fixed prior alpha_0; larger s pushes the
    gate toward the residual path when same-phase histories are unreliable.
    Diagnostic hooks mirror the PhaseVelocity.last_delta convention: the last
    reliability / gate / sensitivity are captured outside the autograd graph.
    """

    def __init__(
        self,
        alpha_init: float = 0.5,
        sensitivity_init: float = 0.0,
        s_max: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.s_max = max(float(s_max), 1e-4)
        self.eps = max(float(eps), 1e-8)
        alpha_init = min(max(float(alpha_init), 1e-4), 1.0 - 1e-4)
        self.gate_bias = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(alpha_init))))
        )
        s0 = min(max(float(sensitivity_init), 0.0), self.s_max - 1e-4)
        self.s_raw = nn.Parameter(
            torch.tensor(float(torch.atanh(torch.tensor(s0 / self.s_max))))
        )
        self.last_r = None
        self.last_alpha = None
        self.last_sensitivity = None

    @property
    def sensitivity(self):
        """Current scalar sensitivity s = s_max * tanh(s_raw)."""
        return float(self.s_max * torch.tanh(self.s_raw).detach())

    def _reliability(self, phase_series):  # (B, C, L, P) pre-shrinkage
        template = phase_series.mean(dim=-1)  # (B, C, L)  mean over periods k
        phase_signal = template.var(dim=2, unbiased=False)  # (B, C)  Var_l(mean_k x_lk)
        phase_noise = phase_series.var(dim=-1, unbiased=False).mean(dim=2)  # (B, C)  mean_l Var_k
        return phase_signal / (phase_signal + phase_noise + self.eps)

    def forward(self, y_phase, y_residual, phase_series):
        # y_phase, y_residual: (B, T, C); phase_series: (B, C, L, P) pre-shrinkage.
        r = self._reliability(phase_series)  # (B, C)
        s = self.s_max * torch.tanh(self.s_raw)
        logit_alpha = self.gate_bias + s * (1.0 - r)
        alpha = torch.sigmoid(logit_alpha)  # (B, C)
        alpha_t = alpha.unsqueeze(1)  # (B, 1, C) broadcasts over the horizon
        y = (1.0 - alpha_t) * y_phase + alpha_t * y_residual
        with torch.no_grad():
            self.last_r = r.detach()
            self.last_alpha = alpha.detach()
            self.last_sensitivity = float(s.detach())
        return y, alpha
