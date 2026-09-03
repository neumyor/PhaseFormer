"""Continuous, train-fitted input components for ETTm1 candidate discovery.

This module is intentionally separate from the historical H1/H3/H4 ablation
implementation.  It precomputes each candidate on the continuously scaled
ETTm1 series, using only the training prefix for fitted quantities, then exposes
window views.  Thus a value's base component never depends on which overlapping
window happens to contain it.  C7 is the documented exception: its *innovation*
is continuous, while the fixed last-24 support mask is relative to an origin.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


CANDIDATES = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
VARIANTS = ("full", "remove_025", "remove_050", "remove_full", "sham_025", "sham_050")


@dataclass(frozen=True)
class CandidateConfig:
    candidate: str
    variant: str = "full"
    period_len: int = 24
    daily_period: int = 96
    weekly_period: int = 672
    recent_length: int = 24
    intervention_seed: int = 9102

    def validate(self, seq_len: int) -> None:
        if self.candidate not in CANDIDATES:
            raise ValueError(f"unknown candidate: {self.candidate}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown candidate variant: {self.variant}")
        if seq_len < self.weekly_period:
            raise ValueError("candidate discovery requires seq_len >= 672")
        if self.period_len != 24 or self.daily_period != 96 or self.weekly_period != 672:
            raise ValueError("the preregistered ETTm1 candidate grid is fixed at 24/96/672")
        if self.recent_length not in (24, 48, 96, 192):
            raise ValueError("recent support must be one of 24/48/96/192 steps")


def _ema(values: np.ndarray, span: int, initial: np.ndarray) -> np.ndarray:
    """Causal EMA with a train-derived initial state."""
    alpha = 2.0 / (float(span) + 1.0)
    result = np.empty_like(values, dtype=np.float64)
    previous = np.asarray(initial, dtype=np.float64)
    for index, value in enumerate(values):
        previous = alpha * value + (1.0 - alpha) * previous
        result[index] = previous
    return result


def _fractional_shift(block: np.ndarray, shift: float) -> np.ndarray:
    spectrum = np.fft.rfft(block, axis=0)
    frequency = np.fft.rfftfreq(len(block))[:, None]
    phase = np.exp(-2j * np.pi * frequency * float(shift))
    phase[0] = 1.0
    if len(block) % 2 == 0:
        phase[-1] = 1.0
    return np.fft.irfft(spectrum * phase, n=len(block), axis=0)


class ContinuousCandidateBank:
    """Generate C1--C7 on a complete scaled ETTm1 series.

    The only train-fitted objects are templates, projections, the C7 causal
    predictor and phase/scale reference statistics.  Validation/test values
    are never used to fit them.
    """

    def __init__(self, dataset: Dataset, config: CandidateConfig):
        self.dataset = dataset
        self.config = config
        config.validate(int(dataset.seq_len))
        if Path(dataset.data_path).name != "ETTm1.csv":
            raise ValueError("continuous candidate discovery is intentionally scoped to ETTm1")
        self.seq_len = int(dataset.seq_len)
        self._values, self._start, self._train_end = self._load_scaled_series(dataset)
        self._components = self._build_components()

    @staticmethod
    def _load_scaled_series(dataset: Dataset):
        raw_path = Path(dataset.root_path) / dataset.data_path
        raw = pd.read_csv(raw_path)
        timestamps = pd.to_datetime(raw["date"]).to_numpy()
        if dataset.features in ("M", "MS"):
            values = raw.iloc[:, 1:].to_numpy(dtype=np.float64)
        else:
            values = raw[[dataset.target]].to_numpy(dtype=np.float64)
        values = dataset.scaler.transform(values)
        first = np.datetime64(pd.Timestamp(dataset.timestamps[0]))
        starts = np.flatnonzero(timestamps == first)
        if len(starts) != 1:
            raise RuntimeError("could not uniquely align split timestamps to ETTm1 source")
        start = int(starts[0])
        current = np.asarray(dataset.data_x, dtype=np.float64)
        reference = values[start : start + len(current), : current.shape[1]]
        if reference.shape != current.shape or not np.allclose(reference, current, atol=1e-7):
            raise RuntimeError("scaled continuous source does not match dataset split")
        train_end = 12 * 30 * 24 * 4
        if train_end >= len(values):
            raise RuntimeError("invalid ETTm1 train boundary")
        return values[:, : current.shape[1]], start, train_end

    def _templates(self):
        train = self._values[: self._train_end]
        phase96 = np.arange(self._train_end) % self.config.daily_period
        phase24 = np.arange(self._train_end) % self.config.period_len
        daily = np.stack([train[phase96 == phase].mean(axis=0) for phase in range(96)])
        short = np.stack([train[phase24 == phase].mean(axis=0) for phase in range(24)])
        return daily, short

    def _c1(self, daily, short):
        phase96 = np.arange(len(self._values)) % self.config.daily_period
        return daily[phase96] - short[phase96 % self.config.period_len]

    def _c2(self, c1):
        mean = self._values[: self._train_end].mean(axis=0)
        low = _ema(self._values, self.config.weekly_period, mean)
        return low - mean - c1

    def _c3(self, c1):
        mean = self._values[: self._train_end].mean(axis=0)
        band = _ema(self._values, 32, mean) - _ema(self._values, 80, mean)
        train_band, train_c1 = band[: self._train_end], c1[: self._train_end]
        denom = np.square(train_c1).sum(axis=0)
        beta = np.divide(
            (train_band * train_c1).sum(axis=0), denom,
            out=np.zeros_like(denom), where=denom > 1e-12,
        )
        return band - c1 * beta

    def _c4(self):
        values = self._values
        diff = np.zeros_like(values)
        diff[1:] = values[1:] - values[:-1]
        phase = np.arange(len(values)) % self.config.period_len
        train_phase = phase[1 : self._train_end]
        expected = np.stack(
            [np.median(diff[1 : self._train_end][train_phase == item], axis=0) for item in range(24)]
        )
        result = np.zeros_like(values)
        boundary = phase == 0
        result[boundary] = diff[boundary] - expected[0]
        return result

    def _c5(self, short):
        values = self._values
        template = short - short.mean(axis=0, keepdims=True)
        energy = np.square(template).sum(axis=0)
        baseline = np.ones(values.shape[1])
        result = np.zeros_like(values)
        starts = np.arange(0, len(values) - self.config.period_len + 1, self.config.period_len)
        amplitudes = []
        for start in starts:
            block = values[start : start + 24]
            centred = block - block.mean(axis=0, keepdims=True)
            amplitude = np.divide(
                (centred * template).sum(axis=0), energy,
                out=np.zeros_like(energy), where=energy > 1e-12,
            )
            if start < self._train_end:
                amplitudes.append(amplitude)
            result[start : start + 24] = (amplitude - baseline) * template
        if amplitudes:
            baseline = np.median(np.stack(amplitudes), axis=0)
            for start in starts:
                block = values[start : start + 24]
                centred = block - block.mean(axis=0, keepdims=True)
                amplitude = np.divide(
                    (centred * template).sum(axis=0), energy,
                    out=np.zeros_like(energy), where=energy > 1e-12,
                )
                result[start : start + 24] = (amplitude - baseline) * template
        return result

    def _c6(self, short):
        """First-order additive representation of smooth 24-step phase velocity."""
        template = short - short.mean(axis=0, keepdims=True)
        derivative = np.gradient(template, axis=0)
        cycles = len(self._values) // 24
        shifts = np.zeros((cycles, self._values.shape[1]), dtype=np.float64)
        candidates = np.arange(-6, 7)
        for cycle in range(cycles):
            block = self._values[cycle * 24 : (cycle + 1) * 24]
            centred = block - block.mean(axis=0, keepdims=True)
            scores = np.stack([(centred * np.roll(template, shift, axis=0)).sum(axis=0) for shift in candidates])
            shifts[cycle] = candidates[np.argmax(scores, axis=0)]
        smooth = _ema(shifts, 5, shifts[: min(8, cycles)].mean(axis=0))
        velocity = np.vstack([np.zeros_like(smooth[0]), np.diff(smooth, axis=0)])
        result = np.zeros_like(self._values)
        for cycle in range(cycles):
            result[cycle * 24 : (cycle + 1) * 24] = velocity[cycle] * derivative
        return result

    def _c7_innovation(self):
        values = self._values
        lag = 96
        train = values[: self._train_end]
        design = np.column_stack(
            [np.ones(self._train_end - lag), train[lag - 1 : -1], train[lag - 24 : -24], train[:-lag]]
        )
        # Fit each variable against its own three causal lags.  The design is
        # expanded channel-by-channel to avoid using other variables as an
        # accidental future proxy.
        residual = np.zeros_like(values)
        for channel in range(values.shape[1]):
            columns = [0, 1 + channel, 1 + values.shape[1] + channel, 1 + 2 * values.shape[1] + channel]
            beta, *_ = np.linalg.lstsq(design[:, columns], train[lag:, channel], rcond=None)
            matrix = np.column_stack(
                [np.ones(len(values) - lag), values[lag - 1 : -1, channel],
                 values[lag - 24 : -24, channel], values[:-lag, channel]]
            )
            residual[lag:, channel] = values[lag:, channel] - matrix @ beta
        return residual

    def _build_components(self):
        daily, short = self._templates()
        c1 = self._c1(daily, short)
        return {
            "c1": c1,
            "c2": self._c2(c1),
            "c3": self._c3(c1),
            "c4": self._c4(),
            "c5": self._c5(short),
            "c6": self._c6(short),
            "c7": self._c7_innovation(),
        }

    def _donor_start(self, index: int, candidate: str) -> int:
        token = f"{self.config.intervention_seed}|{candidate}|{index}".encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        max_start = self._train_end - self.seq_len - 1
        raw = int(np.random.default_rng(seed).integers(0, max_start))
        # Same 15-minute time-of-day phase, while preserving a full contiguous donor block.
        desired_phase = (self._start + index) % self.config.daily_period
        return raw - raw % self.config.daily_period + desired_phase

    def component(self, index: int) -> np.ndarray:
        start = self._start + int(index)
        raw = self._components[self.config.candidate][start : start + self.seq_len].copy()
        if self.config.candidate == "c7":
            raw[:-self.config.recent_length] = 0.0
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, self.config.recent_length))
            raw[-self.config.recent_length :] *= ramp[:, None]
        return raw

    def sham_component(self, index: int) -> np.ndarray:
        donor = self._donor_start(index, self.config.candidate)
        raw = self._components[self.config.candidate][donor : donor + self.seq_len].copy()
        if self.config.candidate == "c7":
            raw[:-self.config.recent_length] = 0.0
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, self.config.recent_length))
            raw[-self.config.recent_length :] *= ramp[:, None]
        return raw

    def transform(self, index: int, seq_x: np.ndarray) -> np.ndarray:
        x = np.asarray(seq_x, dtype=np.float64)
        if self.config.variant == "full":
            return x.copy()
        amount = 0.25 if self.config.variant.endswith("025") else (0.50 if self.config.variant.endswith("050") else 1.0)
        component = self.component(index)
        if self.config.variant.startswith("remove"):
            result = x - amount * component
        else:
            result = x - amount * self.sham_component(index)
        if not np.isfinite(result).all():
            raise FloatingPointError("candidate intervention produced non-finite values")
        return result.astype(seq_x.dtype, copy=False)


class CandidateDataset(Dataset):
    """A Dataset view that changes only the history input for C1--C7."""

    def __init__(self, dataset: Dataset, bank: ContinuousCandidateBank):
        self.dataset = dataset
        self.bank = bank

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset[index]
        return self.bank.transform(index, seq_x), seq_y, seq_x_mark, seq_y_mark

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class GaussianNotchBank:
    """Remove a target frequency from each history with a Gaussian FFT notch.

    The selected period is fixed from the training-only periodogram, while the
    intervention itself operates only on the available history window.  For a
    target frequency ``f0=1/period``, rFFT coefficient ``f`` is multiplied by
    ``1-exp(-(f-f0)^2/(2*sigma_f^2))``.  This gives a smooth band-stop filter
    rather than subtracting a globally fitted sinusoid.
    """

    def __init__(self, seq_len: int, period: float, sigma_frequency: float | None = None):
        if period <= 2:
            raise ValueError("spectral period must exceed two samples")
        self.seq_len = int(seq_len)
        if self.seq_len < 3:
            raise ValueError("Gaussian spectral notch requires seq_len >= 3")
        self.period = float(period)
        self.sigma_frequency = 1.0 / self.seq_len if sigma_frequency is None else float(sigma_frequency)
        if self.sigma_frequency <= 0:
            raise ValueError("Gaussian notch sigma_frequency must be positive")
        frequencies = np.fft.rfftfreq(self.seq_len)
        target = 1.0 / self.period
        gaussian = np.exp(-0.5 * np.square((frequencies - target) / self.sigma_frequency))
        self._keep = 1.0 - gaussian
        self._keep[0] = 1.0  # Do not alter the window mean / DC component.

    def transform(self, index: int, seq_x: np.ndarray) -> np.ndarray:
        x = np.asarray(seq_x, dtype=np.float64)
        if x.shape[0] != self.seq_len:
            raise ValueError("Gaussian notch received an unexpected history length")
        spectrum = np.fft.rfft(x, axis=0)
        result = np.fft.irfft(spectrum * self._keep[:, None], n=self.seq_len, axis=0)
        if not np.isfinite(result).all():
            raise FloatingPointError("Gaussian spectral notch produced non-finite values")
        return result.astype(seq_x.dtype, copy=False)


class TailZeroBank:
    """Set the final ``recent_length`` scaled input observations to zero."""

    def __init__(self, seq_len: int, recent_length: int):
        self.seq_len = int(seq_len)
        self.recent_length = int(recent_length)
        if not 1 <= self.recent_length <= self.seq_len:
            raise ValueError("recent_length must be within the history window")

    def transform(self, index: int, seq_x: np.ndarray) -> np.ndarray:
        result = np.asarray(seq_x).copy()
        if result.shape[0] != self.seq_len:
            raise ValueError("tail zeroing received an unexpected history length")
        result[-self.recent_length :] = 0.0
        return result


class TrajectoryComponentBank:
    """Remove endpoint-anchored trajectories that a full-time-axis head may use.

    Each component is computed only from an individual available history and is
    zero at its final observation (or final 24-step block).  The persistence
    anchor is therefore deliberately preserved while the requested trajectory
    information is removed.
    """

    COMPONENTS = {
        "global_linear", "recent_linear", "cycle_levels", "phase_drift", "cycle_amplitude"
    }

    def __init__(self, seq_len: int, component: str, period_len: int = 24, recent_length: int = 96):
        self.seq_len = int(seq_len)
        self.component = str(component)
        self.period_len = int(period_len)
        self.recent_length = int(recent_length)
        if self.component not in self.COMPONENTS:
            raise ValueError(f"unknown trajectory component {self.component}")
        if self.seq_len % self.period_len:
            raise ValueError("trajectory components require an integral number of periods")
        if not 2 <= self.recent_length <= self.seq_len:
            raise ValueError("recent linear component has invalid support")

    @staticmethod
    def _slope(values: np.ndarray, times: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=0, keepdims=True)
        time_centered = times - times.mean()
        return (time_centered[:, None] * centered).sum(axis=0) / np.square(time_centered).sum()

    def component_values(self, seq_x: np.ndarray) -> np.ndarray:
        x = np.asarray(seq_x, dtype=np.float64)
        if x.shape[0] != self.seq_len:
            raise ValueError("trajectory component received an unexpected history length")
        if self.component in {"global_linear", "recent_linear"}:
            support = x if self.component == "global_linear" else x[-self.recent_length :]
            support_time = np.arange(len(support), dtype=np.float64)
            slope = self._slope(support, support_time)
            time = np.arange(self.seq_len, dtype=np.float64) - (self.seq_len - 1)
            return time[:, None] * slope[None, :]

        cycles = x.reshape(-1, self.period_len, x.shape[1])
        if self.component == "cycle_levels":
            levels = cycles.mean(axis=1)
            return np.repeat((levels - levels[-1])[..., None, :], self.period_len, axis=1).reshape(x.shape)

        if self.component == "phase_drift":
            cycle_time = np.arange(len(cycles), dtype=np.float64)
            slopes = np.stack([self._slope(cycles[:, phase], cycle_time) for phase in range(self.period_len)])
            anchored_time = cycle_time - cycle_time[-1]
            return (anchored_time[:, None, None] * slopes[None, :, :]).reshape(x.shape)

        # A per-window phase template and its cycle-wise amplitude envelope.
        centered = cycles - cycles.mean(axis=1, keepdims=True)
        template = centered.mean(axis=0)
        energy = np.square(template).sum(axis=0)
        amplitude = np.divide(
            (centered * template[None]).sum(axis=1), energy[None],
            out=np.zeros_like(centered[:, 0]), where=energy[None] > 1e-12,
        )
        return ((amplitude - amplitude[-1])[:, None, :] * template[None]).reshape(x.shape)

    def transform(self, index: int, seq_x: np.ndarray) -> np.ndarray:
        result = np.asarray(seq_x, dtype=np.float64) - self.component_values(seq_x)
        if not np.isfinite(result).all():
            raise FloatingPointError("trajectory component removal produced non-finite values")
        return result.astype(seq_x.dtype, copy=False)
