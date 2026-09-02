"""History-only input interventions for the H1/H3/H4 attribution study.

The transformer operates on the already train-split-scaled ``seq_x`` array and
never receives a target or a future timestamp.  This boundary is deliberate:
it makes future leakage impossible inside the intervention implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset


HYPOTHESES = ("none", "h1", "h3", "h4")
VARIANTS = ("full", "half_A", "minus_A", "sham")


@dataclass(frozen=True)
class InputComponentConfig:
    hypothesis: str = "none"
    variant: str = "full"
    period_len: int = 24
    ema_window: int = 96
    intervention_seed: int = 9102
    max_phase_shift: int = 6
    mad_epsilon: float = 1e-6
    minimum_phase_correlation: float = 0.15

    def validate(self, seq_len: int | None = None) -> None:
        if self.hypothesis not in HYPOTHESES:
            raise ValueError(f"unknown input hypothesis: {self.hypothesis}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown input variant: {self.variant}")
        if self.period_len <= 1:
            raise ValueError("period_len must be greater than one")
        if self.ema_window <= 0:
            raise ValueError("ema_window must be positive")
        if self.max_phase_shift <= 0 or self.max_phase_shift >= self.period_len / 2:
            raise ValueError("max_phase_shift must be in (0, period_len / 2)")
        if seq_len is not None and seq_len % self.period_len:
            raise ValueError(
                f"seq_len={seq_len} is not divisible by period_len={self.period_len}"
            )


def _rms(x: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=axis))


def _match_rms(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match per-channel RMS without amplifying numerically empty controls."""
    candidate_rms = _rms(candidate, axis=0)
    reference_rms = _rms(reference, axis=0)
    scale = np.divide(
        reference_rms,
        candidate_rms,
        out=np.zeros_like(reference_rms),
        where=candidate_rms > 1e-12,
    )
    return candidate * scale


def _fractional_circular_shift(values: np.ndarray, shift: float) -> np.ndarray:
    """Apply a real, energy-preserving circular shift along the first axis.

    For even lengths the unpaired Nyquist coefficient cannot receive a complex
    phase while retaining a real signal, so it is held fixed.  Every retained
    Fourier coefficient therefore keeps exactly the same magnitude.
    """
    length = values.shape[0]
    spectrum = np.fft.rfft(values, axis=0)
    frequencies = np.fft.rfftfreq(length)
    phase = np.exp(-2j * np.pi * frequencies * float(shift))[:, None]
    phase[0] = 1.0
    if length % 2 == 0:
        phase[-1] = 1.0
    return np.fft.irfft(spectrum * phase, n=length, axis=0)


class InputComponentTransformer:
    """Extract and intervene on one hypothesized history component."""

    def __init__(self, config: InputComponentConfig):
        self.config = config
        config.validate()

    def _permutation(self, count: int, sample_index: int, namespace: str) -> np.ndarray:
        token = (
            f"{self.config.intervention_seed}|{namespace}|{sample_index}|"
            f"{self.config.hypothesis}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        return np.random.default_rng(seed).permutation(count)

    def _h1(self, x: np.ndarray, sample_index: int, namespace: str):
        cycles = x.reshape(-1, self.config.period_len, x.shape[1])
        template = np.median(cycles, axis=0)
        base0 = np.broadcast_to(template, cycles.shape).copy()
        offset = cycles[-1, -1] - base0[-1, -1]
        base = base0 + offset[None, None, :]
        component = cycles - base

        permutation = self._permutation(len(cycles), sample_index, namespace)
        sham = component[permutation].copy()
        sham -= sham[-1, -1][None, None, :]
        sham = _match_rms(sham.reshape(len(x), -1), component.reshape(len(x), -1))
        sham = sham.reshape(cycles.shape)
        metadata = {
            "component": component.reshape(x.shape),
            "base": base.reshape(x.shape),
            "permutation": permutation,
            "identifiable": np.ones(x.shape[1], dtype=bool),
        }
        return (
            base.reshape(x.shape),
            component.reshape(x.shape),
            sham.reshape(x.shape),
            metadata,
        )

    def _h3(self, x: np.ndarray, sample_index: int, namespace: str):
        cycles = x.reshape(-1, self.config.period_len, x.shape[1])
        template = np.median(cycles, axis=0)
        residual = x - np.tile(template, (len(cycles), 1))
        alpha = 2.0 / (self.config.ema_window + 1.0)
        ema = np.empty_like(residual, dtype=np.float64)
        init_count = min(self.config.period_len, len(residual))
        ema[0] = np.median(residual[:init_count], axis=0)
        for index in range(1, len(residual)):
            ema[index] = alpha * residual[index] + (1.0 - alpha) * ema[index - 1]
        component = ema - ema[-1]
        base = x - component
        sham = component[::-1].copy()
        sham -= sham[-1]
        sham = _match_rms(sham, component)
        metadata = {
            "component": component,
            "base": base,
            "permutation": np.arange(len(x) - 1, -1, -1),
            "identifiable": np.ones(x.shape[1], dtype=bool),
        }
        return base, component, sham, metadata

    def _estimate_phase(self, cycles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, channels = cycles.shape[1:]
        medians = np.median(cycles, axis=1, keepdims=True)
        mad = np.median(np.abs(cycles - medians), axis=1, keepdims=True)
        valid_scale = mad[:, 0, :] > self.config.mad_epsilon
        normalized = np.divide(
            cycles - medians,
            mad,
            out=np.zeros_like(cycles, dtype=np.float64),
            where=mad > self.config.mad_epsilon,
        )
        template = np.median(normalized, axis=0)
        shifts = np.arange(-self.config.max_phase_shift, self.config.max_phase_shift + 1)
        denominator = (
            np.linalg.norm(normalized, axis=1)
            * np.linalg.norm(template, axis=0)[None, :]
        )
        score_cube = np.stack(
            [
                np.sum(
                    normalized
                    * np.roll(template, int(shift), axis=0)[None],
                    axis=1,
                )
                for shift in shifts
            ],
            axis=0,
        )
        score_cube = np.divide(
            score_cube,
            denominator[None],
            out=np.full_like(score_cube, -1.0),
            where=denominator[None] > self.config.mad_epsilon,
        )
        peak = np.max(score_cube, axis=0)
        # Pre-registered tie break: smallest |shift|, then smaller signed shift.
        order = np.lexsort((shifts, np.abs(shifts)))
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        tied_rank = np.where(
            np.abs(score_cube - peak[None]) <= 1e-12,
            rank[:, None, None],
            len(shifts),
        )
        best = np.argmin(tied_rank, axis=0)
        valid = (
            valid_scale
            & (denominator > self.config.mad_epsilon)
            & (peak >= self.config.minimum_phase_correlation)
        )
        delta = shifts[best].astype(np.float64)
        interior = (best > 0) & (best < len(shifts) - 1) & valid
        rows, cols = np.indices(best.shape)
        left = score_cube[np.maximum(best - 1, 0), rows, cols]
        centre = score_cube[best, rows, cols]
        right = score_cube[np.minimum(best + 1, len(shifts) - 1), rows, cols]
        curvature = left - 2.0 * centre + right
        refinable = interior & (curvature < -1e-12)
        refinement = np.zeros_like(delta)
        refinement[refinable] = np.clip(
            0.5 * (left[refinable] - right[refinable]) / curvature[refinable],
            -0.5,
            0.5,
        )
        delta += refinement
        delta[~valid] = 0.0
        return delta, peak, valid

    def _h4(self, x: np.ndarray, sample_index: int, namespace: str):
        cycles = x.reshape(-1, self.config.period_len, x.shape[1])
        delta, peak, cycle_valid = self._estimate_phase(cycles)
        # A single flat/low-correlation day must not disable an otherwise
        # identifiable 30-cycle channel.  Such cycles stay byte-identical;
        # the channel is active only when the anchor (latest cycle) is valid
        # and at least half of its history supplies phase evidence.
        identifiable = cycle_valid[-1] & (cycle_valid.mean(axis=0) >= 0.5)
        latest = delta[-1]
        permutation = np.arange(len(cycles))
        if self.config.variant == "half_A":
            target = 0.5 * delta + 0.5 * latest[None, :]
        elif self.config.variant == "minus_A":
            target = np.broadcast_to(latest, delta.shape)
        elif self.config.variant == "sham":
            permutation = self._permutation(len(cycles), sample_index, namespace)
            target = delta[permutation] - delta[permutation[-1]] + latest
        else:
            target = delta

        transformable = cycle_valid & identifiable[None, :]
        if self.config.variant == "sham":
            # Do not inject an arbitrary zero estimate when the permuted donor
            # cycle was itself unidentifiable.
            transformable &= cycle_valid[permutation]
        correction = np.where(transformable, target - delta, 0.0)
        spectrum = np.fft.rfft(cycles, axis=1)
        frequencies = np.fft.rfftfreq(self.config.period_len)
        phase = np.exp(
            -2j * np.pi * frequencies[None, :, None] * correction[:, None, :]
        )
        phase[:, 0, :] = 1.0
        if self.config.period_len % 2 == 0:
            phase[:, -1, :] = 1.0
        transformed = np.fft.irfft(
            spectrum * phase, n=self.config.period_len, axis=1
        )
        metadata = {
            "phase_shift": delta,
            "phase_target": target,
            "peak_correlation": peak,
            "cycle_identifiable": cycle_valid,
            "cycle_transformable": transformable,
            "identifiable": identifiable,
            "permutation": permutation,
            "component": x - transformed.reshape(x.shape),
            "base": transformed.reshape(x.shape),
        }
        return transformed.reshape(x.shape), metadata

    def transform(
        self,
        seq_x: np.ndarray,
        *,
        sample_index: int = 0,
        namespace: str = "",
        return_metadata: bool = False,
    ):
        x = np.asarray(seq_x)
        if x.ndim != 2:
            raise ValueError(f"seq_x must have shape [time, channel], got {x.shape}")
        self.config.validate(len(x))
        original_dtype = x.dtype
        work = x.astype(np.float64, copy=False)
        hypothesis = self.config.hypothesis
        variant = self.config.variant
        if hypothesis == "none" or variant == "full":
            result = work.copy()
            metadata: Dict[str, Any] = {
                "component": np.zeros_like(work),
                "base": work.copy(),
                "identifiable": np.ones(work.shape[1], dtype=bool),
            }
        elif hypothesis == "h4":
            result, metadata = self._h4(work, sample_index, namespace)
        else:
            base, component, sham, metadata = (
                self._h1(work, sample_index, namespace)
                if hypothesis == "h1"
                else self._h3(work, sample_index, namespace)
            )
            if variant == "half_A":
                result = base + 0.5 * component
            elif variant == "minus_A":
                result = base
            else:
                result = base + sham
        result = result.astype(original_dtype, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("input component intervention produced non-finite values")
        return (result, metadata) if return_metadata else result


class InputComponentDataset(Dataset):
    """Dataset view that changes only ``seq_x`` and delegates all metadata."""

    def __init__(self, dataset: Dataset, config: InputComponentConfig, namespace: str):
        self.dataset = dataset
        self.config = config
        self.namespace = namespace
        self.transformer = InputComponentTransformer(config)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset[index]
        transformed = self.transformer.transform(
            seq_x, sample_index=int(index), namespace=self.namespace
        )
        return transformed, seq_y, seq_x_mark, seq_y_mark

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)
