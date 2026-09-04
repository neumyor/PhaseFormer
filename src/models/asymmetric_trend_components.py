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
    "trend_filter",
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


def _second_difference(values: torch.Tensor) -> torch.Tensor:
    """Second finite difference along the history axis, ``(B,L,C)->(B,L-2,C)``."""
    return values[:, :-2, :] - 2.0 * values[:, 1:-1, :] + values[:, 2:, :]


def _second_difference_transpose(values: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`_second_difference` without a dense ``L×L`` matrix."""
    result = values.new_zeros(values.size(0), values.size(1) + 2, values.size(2))
    result[:, :-2, :] += values
    result[:, 1:-1, :] -= 2.0 * values
    result[:, 2:, :] += values
    return result


def _trend_filter(
    x: torch.Tensor,
    *,
    kappa: float,
    sample_interval_hours: float,
    iterations: int,
) -> torch.Tensor:
    """GPU-batched first-order trend-filter approximation.

    This solves ``min_f .5||x-f||² + lambda||D²f||₁`` by a fixed-iteration
    Chambolle--Pock primal-dual method.  It deliberately contains no CPU or
    per-series linear algebra in ``forward``.  Dividing each series by its
    own standard deviation makes the fixed normalized penalty exactly
    equivalent to ``lambda=kappa*std(x)*(1 hour/dt)²`` after rescaling.
    """
    if sample_interval_hours <= 0:
        raise ValueError("trend-filter sample interval must be positive")
    if iterations <= 0:
        raise ValueError("trend-filter iterations must be positive")
    # ``unbiased=False`` is the frozen population sample scale used by the
    # visual diagnostic.  Constant series have an identically zero component.
    scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(torch.finfo(x.dtype).eps)
    z = x / scale
    penalty = float(kappa) * (1.0 / float(sample_interval_hours)) ** 2
    # ||D²||² <= 16.  The strongly-convex primal acceleration is important
    # at the intentionally large frozen penalty (kappa=100).
    tau = sigma = 0.24
    fitted = z
    extrapolated = z
    dual = z.new_zeros(z.size(0), z.size(1) - 2, z.size(2))
    for _ in range(int(iterations)):
        dual = (dual + sigma * _second_difference(extrapolated)).clamp(-penalty, penalty)
        updated = (fitted + tau * z - tau * _second_difference_transpose(dual)) / (1.0 + tau)
        theta = (1.0 + 2.0 * tau) ** -0.5
        extrapolated = updated + theta * (updated - fitted)
        fitted = updated
        tau *= theta
        sigma /= theta
    return fitted * scale


def extract_trend_component(
    x: torch.Tensor,
    component: str,
    *,
    period_len: int = 24,
    recent_window: int = 96,
    local_sigma: float = 24.0,
    long_sigma: float = 72.0,
    trend_filter_kappa: float = 100.0,
    trend_filter_sample_interval_hours: float = 1.0,
    trend_filter_iterations: int = 128,
) -> torch.Tensor:
    """Extract one frozen trend component from ``(B,L,C)`` input."""
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
    if component == "trend_filter":
        return _endpoint_anchor(
            _trend_filter(
                x,
                kappa=trend_filter_kappa,
                sample_interval_hours=trend_filter_sample_interval_hours,
                iterations=trend_filter_iterations,
            )
        )
    # Difference of short- and long-scale smoothed trends.  Both smoothers use
    # the observed history only; endpoint anchoring retains NLinear's last value.
    return _endpoint_anchor(
        _gaussian_smooth(x, local_sigma) - _gaussian_smooth(x, long_sigma)
    )
