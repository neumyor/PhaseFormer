"""Deterministic, endpoint-anchored trend components for branch ablations.

The PhaseFormer path keeps the complete history.  These functions only prepare
the history visible to a residual branch, so every returned component has a
zero final value and cannot remove the NLinear persistence anchor.
"""

import math

import torch
import torch.nn.functional as F


TREND_COMPONENTS = {
    "cycle_levels",
    "recent_linear",
    "global_linear",
    "smooth_local",
    "smooth_multiscale",
}


def _endpoint_anchor(component: torch.Tensor) -> torch.Tensor:
    return component - component[:, -1:, :]


def _linear_trend(x: torch.Tensor, start: int) -> torch.Tensor:
    """OLS linear trend fitted over ``x[:, start:]``, anchored at x's endpoint."""
    length = x.size(1)
    start = max(0, min(int(start), length - 2))
    time = torch.arange(start, length, device=x.device, dtype=x.dtype)
    centered_time = time - time.mean()
    values = x[:, start:, :]
    slope = (values * centered_time.view(1, -1, 1)).sum(dim=1, keepdim=True)
    slope = slope / centered_time.square().sum().clamp_min(torch.finfo(x.dtype).eps)
    full_time = torch.arange(length, device=x.device, dtype=x.dtype)
    return (full_time - (length - 1)).view(1, -1, 1) * slope


def _gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Replicate-padded Gaussian smoothing independently for each variable."""
    if sigma <= 0:
        raise ValueError("Gaussian smoothing sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)).square())
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    series = x.transpose(1, 2)
    padded = F.pad(series, (radius, radius), mode="replicate")
    smoothed = F.conv1d(
        padded.reshape(-1, 1, padded.size(-1)), kernel
    ).reshape_as(series)
    return smoothed.transpose(1, 2)


def extract_trend_component(
    x: torch.Tensor,
    component: str,
    *,
    period_len: int = 24,
    recent_window: int = 96,
    local_sigma: float = 24.0,
    long_sigma: float = 72.0,
) -> torch.Tensor:
    """Extract one of the five frozen trend components from ``(B,L,C)`` input."""
    if component not in TREND_COMPONENTS:
        raise ValueError(f"Unsupported asymmetric trend component: {component}")
    if x.ndim != 3:
        raise ValueError(f"Expected (B,L,C), got shape {tuple(x.shape)}")
    length = x.size(1)
    if component == "cycle_levels":
        if period_len <= 1 or length % period_len:
            raise ValueError("cycle_levels requires seq_len divisible by period_len > 1")
        cycles = x.reshape(x.size(0), length // period_len, period_len, x.size(2))
        levels = cycles.mean(dim=2, keepdim=True)
        return (
            (levels - levels[:, -1:, :, :])
            .expand(-1, -1, period_len, -1)
            .reshape_as(x)
        )
    if component == "recent_linear":
        return _linear_trend(x, length - recent_window)
    if component == "global_linear":
        return _linear_trend(x, 0)
    if component == "smooth_local":
        return _endpoint_anchor(_gaussian_smooth(x, local_sigma))
    # Difference of short- and long-scale smoothed trends.  Both smoothers use
    # the observed history only; endpoint anchoring retains NLinear's last value.
    return _endpoint_anchor(
        _gaussian_smooth(x, local_sigma) - _gaussian_smooth(x, long_sigma)
    )
