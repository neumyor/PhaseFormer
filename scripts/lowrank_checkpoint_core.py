#!/usr/bin/env python3
"""Shared machinery for the low-rank checkpoint information analysis.

This module implements the pieces that
``docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md`` needs in
more than one entry point:

* the **effective linear map** of a trainable low-rank residual head
  (``decoder @ encoder`` plus the deterministic pooling operator), so that the
  rotation ambiguity of the hidden coordinates is removed before anything is
  named (plan section 3);
* **subspace utilities** (orthonormalization, principal angles, projection
  overlap) used by the cross-seed and cross-target comparisons;
* the **semantic dictionary** of input/output templates (plan section 4); and
* the **conditional reduced-rank regression** solver (plan section 5, Stage 3).

Nothing here trains a model or reads the test split.  The module is imported by
``analyze_lowrank_checkpoint_information.py``,
``compute_phase_conditional_rrr.py`` and
``evaluate_lowrank_semantic_interventions.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Effective map of the factorized low-rank head
# ---------------------------------------------------------------------------


def adaptive_avg_pool_operator(seq_len: int, pooled_len: int) -> np.ndarray:
    """Return the exact ``(pooled_len, seq_len)`` matrix of ``adaptive_avg_pool1d``.

    ``pooled_len == seq_len`` is the identity, which is the case for every
    formal checkpoint of this round (``pool_factor == 1``); the general branch
    exists so that a non-unit pool factor cannot silently be analysed as if the
    pooling step were absent.
    """
    if pooled_len == seq_len:
        return np.eye(seq_len, dtype=np.float64)
    matrix = np.zeros((pooled_len, seq_len), dtype=np.float64)
    for out_index in range(pooled_len):
        start = (out_index * seq_len) // pooled_len
        stop = -(-(out_index + 1) * seq_len // pooled_len)
        if stop <= start:
            stop = start + 1
        matrix[out_index, start:stop] = 1.0 / (stop - start)
    return matrix


def effective_map(
    encoder_weight: np.ndarray,
    decoder_weight: np.ndarray,
    pooled_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M, D)`` with ``delta_center = M z + D bias``.

    ``encoder_weight`` is ``(rank, pooled_len)``, ``decoder_weight`` is
    ``(pred_len, rank)`` and ``z`` is the *centered, un-pooled* branch input of
    length ``seq_len``.  ``M`` is therefore ``(pred_len, seq_len)`` and the
    encoder bias is folded into the returned ``c = decoder @ encoder.bias``
    term only through ``decode_bias``.
    """
    pool = adaptive_avg_pool_operator(pooled_len, encoder_weight.shape[1])
    return decoder_weight @ encoder_weight @ pool, decoder_weight


def decode_hidden(decoder_weight: np.ndarray, hidden: torch.Tensor) -> torch.Tensor:
    """``decoder(h)`` without the decoder bias, on an arbitrary hidden tensor."""
    weight = torch.as_tensor(decoder_weight, dtype=hidden.dtype, device=hidden.device)
    return hidden @ weight.T


# ---------------------------------------------------------------------------
# Subspace utilities
# ---------------------------------------------------------------------------


def orthonormalize(vectors: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for ``vectors`` given as rows (k, L)."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    u, s, _ = np.linalg.svd(matrix.T, full_matrices=False)
    tol = max(matrix.shape) * (s[0] if s.size else 0.0) * np.finfo(np.float64).eps
    keep = int(max(1, np.sum(s > tol)))
    return np.ascontiguousarray(u[:, :keep])


def orthonormality_error(basis: np.ndarray) -> float:
    gram = basis.T @ basis
    return float(np.abs(gram - np.eye(basis.shape[1])).max())


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between two orthonormal bases."""
    sigma = np.linalg.svd(
        np.asarray(basis_a, dtype=np.float64).T @ np.asarray(basis_b, dtype=np.float64),
        compute_uv=False,
    )
    sigma = np.clip(sigma, 0.0, 1.0)
    return np.degrees(np.arccos(sigma))


def projection_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Normalized subspace overlap ``||Qa^T Qb||_F^2 / k`` in ``[0, 1]``.

    ``1`` means the two subspaces coincide, ``0`` means they are orthogonal.
    The dimensions must match, otherwise the value is not comparable across
    ranks.
    """
    k = min(basis_a.shape[1], basis_b.shape[1])
    product = np.asarray(basis_a, dtype=np.float64).T @ np.asarray(basis_b, dtype=np.float64)
    return float(np.sum(product * product) / k)


# ---------------------------------------------------------------------------
# Semantic dictionary (plan section 4)
# ---------------------------------------------------------------------------


@dataclass
class Template:
    name: str
    vector: np.ndarray


@dataclass
class SemanticGroup:
    name: str
    templates: list[Template] = field(default_factory=list)
    basis: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        if self.basis is None:
            return 0
        return int(self.basis.shape[1])


def _tail(seq_len: int, length: int) -> np.ndarray:
    length = int(min(length, seq_len))
    vector = np.zeros(seq_len, dtype=np.float64)
    vector[-length:] = 1.0 / length
    return vector


def _tail_segment(seq_len: int, offset: int, length: int) -> np.ndarray:
    """Indicator segment ``offset`` steps back from the last step, mean-normalized."""
    vector = np.zeros(seq_len, dtype=np.float64)
    stop = seq_len - int(offset)
    start = max(0, stop - int(length))
    if stop <= start:
        return vector
    vector[start:stop] = 1.0 / (stop - start)
    return vector


def _ema(seq_len: int, tau: float) -> np.ndarray:
    """Causal exponential moving average weights, most recent step last."""
    index = np.arange(seq_len, dtype=np.float64)
    decay = math.log(2.0) / float(tau)
    weights = np.exp(-decay * (seq_len - 1 - index))
    return weights / weights.sum()


def _ramp(seq_len: int, length: int, anchored: bool) -> np.ndarray:
    vector = np.zeros(seq_len, dtype=np.float64)
    length = int(min(length, seq_len))
    x = np.arange(length, dtype=np.float64)
    if anchored:
        # A ramp that values the most recent step most, i.e. an endpoint anchor.
        weights = np.linspace(0.0, 1.0, length)
    else:
        weights = x - x.mean()
    vector[-length:] = weights
    return vector


def _quadratic(seq_len: int, length: int) -> np.ndarray:
    vector = np.zeros(seq_len, dtype=np.float64)
    length = int(min(length, seq_len))
    x = np.arange(length, dtype=np.float64)
    vector[-length:] = x * x
    return vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("degenerate semantic template")
    return vector / norm


def input_templates(
    seq_len: int,
    period: int,
) -> dict[str, list[Template]]:
    """Build the input-side semantic groups of plan section 4.1.

    ``period`` is the dataset's dominant physical cycle expressed in *steps*
    (24 for hourly data, 96 for ETTm2's 15-minute sampling), so period-shaped
    templates never treat 24 rows of an ETTm2 window as a full day.
    """
    tails = [6, 12, 24, 48, 72, 168]
    emas = [6, 12, 24, 48, 72, 168]
    trend_lengths = [24, 72, 168]

    templates: dict[str, list[Template]] = {}

    recent = []
    for length in tails:
        recent.append(Template(f"tail_mean_{length}", _normalize(_tail(seq_len, length))))
    for tau in emas:
        recent.append(Template(f"ema_tau{tau}", _normalize(_ema(seq_len, tau))))
    templates["recent_level"] = recent

    level_change = []
    for recent_length, prior_length in ((24, 24), (72, 72), (168, 168)):
        vector = _tail_segment(seq_len, 0, recent_length) - _tail_segment(
            seq_len, recent_length, prior_length
        )
        level_change.append(
            Template(f"level_delta_{recent_length}v{prior_length}", _normalize(vector))
        )
    for length in trend_lengths:
        vector = _tail_segment(seq_len, 0, length) - _tail_segment(
            seq_len, length, 4 * length
        )
        level_change.append(
            Template(f"level_contrast_tail{length}_vs4x", _normalize(vector))
        )
    templates["level_change"] = level_change

    local_trend = []
    for length in trend_lengths:
        local_trend.append(
            Template(f"ramp_linear_{length}", _normalize(_ramp(seq_len, length, False)))
        )
        local_trend.append(
            Template(f"ramp_anchored_{length}", _normalize(_ramp(seq_len, length, True)))
        )
    templates["local_trend"] = local_trend

    local_curvature = []
    for length in trend_lengths:
        constant = np.zeros(seq_len, dtype=np.float64)
        constant[-length:] = 1.0 / length
        linear = _ramp(seq_len, length, False)
        quadratic = _quadratic(seq_len, length)
        # Remove the linear and constant part so the template is the pure bend.
        design = np.stack(
            [
                constant / max(np.linalg.norm(constant), 1e-12),
                linear / max(np.linalg.norm(linear), 1e-12),
            ],
            axis=1,
        )
        residual = quadratic - design @ (design.T @ quadratic)
        local_curvature.append(
            Template(f"curvature_{length}", _normalize(residual))
        )
        local_curvature.append(
            Template(f"quadratic_{length}", _normalize(_quadratic(seq_len, length)))
        )
    templates["local_curvature"] = local_curvature

    period_level = []
    for multiple in (1, 2):
        block = period * multiple
        if block >= seq_len:
            continue
        vector = _tail_segment(seq_len, 0, block) - _tail_segment(
            seq_len, block, min(5 * block, seq_len - block)
        )
        period_level.append(
            Template(f"period_level_p{multiple}", _normalize(vector))
        )
        weekly_like = _tail_segment(seq_len, 0, block) - _tail_segment(
            seq_len, 0, 7 * block
        )
        period_level.append(
            Template(f"period_level_tail_vs_{7 * multiple}p", _normalize(weekly_like))
        )
    templates["period_level"] = period_level

    period_shape = []
    n_periods = seq_len // period
    for harmonic in (1, 2):
        phase_index = np.arange(seq_len, dtype=np.float64)
        period_shape.append(
            Template(
                f"sin_p{harmonic}",
                _normalize(np.sin(2.0 * math.pi * harmonic * phase_index / float(period))),
            )
        )
        period_shape.append(
            Template(
                f"cos_p{harmonic}",
                _normalize(np.cos(2.0 * math.pi * harmonic * phase_index / float(period))),
            )
        )
    if n_periods >= 2:
        # Recent cycle minus the mean of the earlier cycles, laid out over the
        # full window: a phase-aligned shape-contrast template.
        contrast = np.zeros(seq_len, dtype=np.float64)
        for step in range(period):
            positions = np.arange(seq_len - 1 - step, -1, -period)
            if positions.size < 2:
                continue
            contrast[positions[0]] = 1.0
            contrast[positions[1:]] = -1.0 / (positions.size - 1)
        if np.linalg.norm(contrast) > 1e-12:
            period_shape.append(
                Template("cycle_shape_contrast", _normalize(contrast))
            )
    templates["period_shape"] = period_shape

    fast = []
    first = np.zeros(seq_len, dtype=np.float64)
    first[-1] = 1.0
    first[-2] = -1.0
    fast.append(Template("first_difference_1", _normalize(first)))
    second = np.zeros(seq_len, dtype=np.float64)
    second[-1] = 1.0
    second[-2] = -2.0
    second[-3] = 1.0
    fast.append(Template("second_difference_1", _normalize(second)))
    for lag in (2, 4, 8):
        vector = np.zeros(seq_len, dtype=np.float64)
        vector[-1] = 1.0
        vector[-1 - lag] = -1.0
        fast.append(Template(f"contrast_lag{lag}", _normalize(vector)))
    for step in (2, 4, 8):
        vector = _tail_segment(seq_len, 0, step) - _tail_segment(seq_len, step, step)
        fast.append(Template(f"local_contrast_{step}", _normalize(vector)))
    for lag in (1, 4):
        weights = np.zeros(seq_len, dtype=np.float64)
        kernel = np.array([1.0, -2.0, 1.0])
        offset = seq_len - 3 - (lag - 1)
        weights[offset : offset + 3] = kernel
        fast.append(Template(f"local_second_diff_lag{lag}", _normalize(weights)))
    templates["fast_local_change"] = fast
    return templates


def output_templates(pred_len: int, period: int) -> dict[str, list[Template]]:
    """Build the output-side semantic groups of plan section 4.2."""
    templates: dict[str, list[Template]] = {}
    x = np.arange(pred_len, dtype=np.float64)

    displacement = [Template("constant", _normalize(np.ones(pred_len)))]
    templates["overall_displacement"] = displacement

    slow_tilt = [Template("ramp_linear", _normalize(2.0 * x - (pred_len - 1.0)))]
    slow_tilt.append(Template("ramp_capped", _normalize(np.minimum(x, pred_len / 2.0))))
    half = pred_len // 2
    if half > 0:
        slow_tilt.append(
            Template("step_half_positive", _normalize(np.where(x >= half, 1.0, 0.0)))
        )
        slow_tilt.append(
            Template("step_half_negative", _normalize(np.where(x >= half, -1.0, 0.0)))
        )
    templates["slow_tilt"] = [t for t in slow_tilt if np.linalg.norm(t.vector) > 1e-12]

    curvature = [
        Template("quadratic", _normalize((x - (pred_len - 1.0) / 2.0) ** 2))
    ]
    for knot in (pred_len / 3.0, 2.0 * pred_len / 3.0):
        piecewise = np.where(x <= knot, x - knot / 2.0, knot / 2.0)
        curvature.append(
            Template(f"piecewise_knot{int(knot)}", _normalize(piecewise))
        )
    templates["curvature"] = curvature

    periodic = []
    for harmonic in (1, 2):
        periodic.append(
            Template(
                f"sin_p{harmonic}",
                _normalize(np.sin(2.0 * math.pi * harmonic * x / float(period))),
            )
        )
        periodic.append(
            Template(
                f"cos_p{harmonic}",
                _normalize(np.cos(2.0 * math.pi * harmonic * x / float(period))),
            )
        )
    templates["periodic"] = periodic

    continuation = []
    for lag in (1, 2):
        # Copy the most recent cycle forward, decaying with horizon distance.
        positions = (pred_len - lag * period) + np.arange(pred_len)
        harmonic = np.cos(2.0 * math.pi * positions / float(period))
        curve = harmonic * (1.0 - (x / max(pred_len - 1.0, 1.0)))
        if np.linalg.norm(curve) > 1e-12:
            continuation.append(
                Template(f"decay_copy_lag{lag}", _normalize(curve))
            )
    ramp_decay = 1.0 - (x / max(pred_len - 1.0, 1.0))
    continuation.append(Template("linear_decay", _normalize(ramp_decay)))
    templates["recent_shape_continuation"] = continuation
    return templates


def build_groups(
    raw: dict[str, list[Template]],
    rank_tol: float = 1e-9,
) -> dict[str, SemanticGroup]:
    """Attach a stable orthonormal basis to every semantic group."""
    groups: dict[str, SemanticGroup] = {}
    for name, templates in raw.items():
        matrix = np.stack([template.vector for template in templates], axis=1)
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        threshold = rank_tol * (s[0] if s.size else 0.0) * max(matrix.shape)
        keep = int(max(1, np.sum(s > threshold)))
        groups[name] = SemanticGroup(
            name=name,
            templates=templates,
            basis=np.ascontiguousarray(u[:, :keep]),
        )
    return groups


def orthogonal_projection(direction: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares reconstruction of ``direction`` on ``basis`` and its R^2."""
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector), 0.0
    coefficients, *_ = np.linalg.lstsq(basis, vector, rcond=None)
    fitted = basis @ coefficients
    residual = float(np.linalg.norm(vector - fitted) ** 2)
    return fitted, float(max(0.0, 1.0 - residual / (norm ** 2)))


def group_explanation(direction: np.ndarray, group: SemanticGroup) -> float:
    """Share of the L2 norm of ``direction`` inside ``group``'s span."""
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return 0.0
    projected = group.basis @ (group.basis.T @ vector)
    return float(np.sum(projected * projected) / (norm ** 2))


def group_shapley(
    direction: np.ndarray,
    groups: dict[str, SemanticGroup],
    group_names: list[str],
) -> dict[str, float]:
    """Exact Shapley values of the dictionary R^2 over the group set.

    The value function is the R^2 of the least-squares reconstruction of
    ``direction`` on the span of the selected groups.  With at most seven
    groups the exact permutation average is cheap, so no sampling is used.
    """
    n = len(group_names)
    if n == 0:
        return {}
    if n == 1:
        # A single group has nothing to distribute: the exact Shapley value is
        # the group's own reconstruction R^2.
        _, value = orthogonal_projection(direction, groups[group_names[0]].basis)
        return {group_names[0]: float(value)}
    basis_cache: dict[int, np.ndarray] = {}

    def r2(mask: int) -> float:
        if mask == 0:
            return 0.0
        if mask not in basis_cache:
            columns = [
                groups[group_names[index]].basis
                for index in range(n)
                if mask >> index & 1
            ]
            basis_cache[mask] = orthonormalize(np.concatenate(columns, axis=1).T)
        _, value = orthogonal_projection(direction, basis_cache[mask])
        return value

    shapley = {name: 0.0 for name in group_names}
    # Exact Shapley weights over all n! orderings: each coalition S of size s is
    # the prefix of s!(n-s-1)! orderings, so the weight normalizes by (n-1)!.
    for mask in range(1 << n):
        size = bin(mask).count("1")
        weight = (
            math.factorial(size) * math.factorial(n - size - 1)
            / math.factorial(n - 1)
        )
        base = r2(mask)
        for index in range(n):
            if mask >> index & 1:
                continue
            shapley[group_names[index]] += weight * (r2(mask | 1 << index) - base)
    return shapley


# ---------------------------------------------------------------------------
# Input statistics used by the covariance-metric correlation
# ---------------------------------------------------------------------------


@dataclass
class CenteredMoments:
    """Second moments of the centered, RevIN-normalized branch input."""

    mean: np.ndarray
    cov: np.ndarray
    n: int

    def whitened(self) -> np.ndarray:
        """``C^+^{1/2}`` on the support of ``C`` (shape ``(L, L)``)."""
        values, vectors = np.linalg.eigh(0.5 * (self.cov + self.cov.T))
        keep = values > max(values.max(), 1e-300) * 1e-10
        return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T

    def whiten(self, vector: np.ndarray) -> np.ndarray:
        return self.whitened() @ np.asarray(vector, dtype=np.float64)


def covariance_correlation(
    direction: np.ndarray,
    template: np.ndarray,
    moments: CenteredMoments,
) -> float:
    """Correlation of ``direction^T z`` with ``template^T z`` under the data metric.

    ``z`` is centered by construction (``z = x_normalized - x_normalized_last``),
    so the linear functionals are already zero-mean on every channel.  Writing
    ``C`` for the training covariance of ``z`` and ``m`` for its mean, the
    Gram matrix of the two functionals is ``E[F F^T] + F m m^T F^T``; the
    whitening form of that expression makes the value invariant to the arbitrary
    L2 scaling of either vector.
    """
    whitening = moments.whitened()
    direction = np.asarray(direction, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    basis = np.stack([direction, template], axis=1)
    mean_image = whitening @ moments.mean
    demeaned = whitening @ basis
    gram_low = demeaned.T @ demeaned
    project = basis.T @ moments.mean
    gram = gram_low + np.outer(project, project)
    variance = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denom = float(variance[0] * variance[1])
    if denom <= 1e-14:
        return 0.0
    return float(np.clip(gram[0, 1] / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Reduced-rank regression
# ---------------------------------------------------------------------------


def independent_rrr(
    szz: np.ndarray,
    szy: np.ndarray,
    rank: int,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``rank`` input subspace of the reduced-rank regression on ``szy``.

    ``S = Szy^T (Szz + ridge I)^-1 Szy`` is eigendecomposed and the leading
    eigenvectors are mapped back to the input side with
    ``B = U^T Szy^T (Szz + ridge I)^-1``; the returned basis is the orthonormal
    span of ``B``'s leading rows.  For unweighted least squares in a negated
    loss this is the exact rank-``rank`` optimizer, which
    ``scripts/compute_top2_direction_projectors.py`` already relies on.
    """
    szz_r = szz + ridge * np.eye(szz.shape[0])
    szz_inv_szy = np.linalg.solve(szz_r, szy)
    s_mat = szy.T @ szz_inv_szy
    s_mat = 0.5 * (s_mat + s_mat.T)
    values, vectors = np.linalg.eigh(s_mat)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    directions = vectors[:, :rank].T @ szz_inv_szy.T
    return orthonormalize(directions), values[order]


def weighted_rrr_subspace(
    m_zz: np.ndarray,
    m_zy: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-``rank`` input subspace maximizing the weighted explained energy.

    For a weighted least-squares problem whose per-sample weights are shared
    across the whole horizon, the optimal rank-``rank`` subspace of a linear map
    is the span of the leading eigenvectors of ``M_zz`` where
    ``M_ab = sum_n w_n a_n b_n^T / N``.  This is the natural generalization of
    the unweighted RRR direction when the weights depend on the sample (through
    the learned gate and the RevIN scale difference) rather than on the horizon.

    Returns the orthonormal input basis and the eigenvalues in descending order.
    """
    m_zz = 0.5 * (m_zz + m_zz.T)
    values, vectors = np.linalg.eigh(m_zz)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    return np.ascontiguousarray(vectors[:, :rank]), values


def weighted_moments(
    z: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted second moments of ``(z, target)`` with per-sample weights."""
    zw = z * weights[:, None]
    return (
        z.T @ zw / z.shape[0],
        z.T @ (target * weights[:, None]) / z.shape[0],
        target.T @ (target * weights[:, None]) / z.shape[0],
    )
