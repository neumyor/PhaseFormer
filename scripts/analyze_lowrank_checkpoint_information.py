#!/usr/bin/env python3
"""Stage 0-2 of the low-rank checkpoint information analysis.

Implements ``docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md``:

* **Stage 0** -- checkpoint inventory, configuration and pool-factor audit, and
  the effective-map equivalence check ``decoder(encoder(pool(z))) == M z + c``
  with a hard ``1e-6`` maximum-absolute-error bound on a fixed validation batch;
* **Stage 1** -- canonical SVD modes of ``M``, singular shares, participation
  ratio, effective numerical rank, per-mode latent variance and correction
  energy, single-mode zeroing effects, bias-off effects, and the cross-seed
  input/output subspace alignment;
* **Stage 2** -- semantic dictionary attribution of the canonical input and
  output directions: single-template ``|cos|``, covariance-metric correlation,
  per-group projection explanation, dictionary R^2, leave-one-group-out and
  exact Shapley R^2, plus the paired input->output semantics required for
  ``表 4``.

The script reads the train split only (through the training-data covariance and
the semantic templates); all performance numbers come from the validation split
of the corresponding run configuration.  The test split is never read, no model
is trained, and no checkpoint is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lowrank_checkpoint_core import (  # noqa: E402
    CenteredMoments,
    build_groups,
    covariance_correlation,
    effective_map,
    group_explanation,
    group_shapley,
    input_templates,
    orthogonal_projection,
    output_templates,
    orthonormalize,
    orthonormality_error,
    principal_angles,
    projection_overlap,
)
from scripts.lowrank_checkpoint_inventory import (  # noqa: E402
    DIAGNOSTIC_RELATIVE_RANK,
    FORMAL_SEEDS,
    build_cells,
    inventory_rows,
    write_inventory,
)
from scripts.lowrank_checkpoint_model import DATASET_PERIOD_STEPS  # noqa: E402

# The five candidate mechanisms of the plan's 表 7, as input->output pairs.
INPUT_GROUP_LABELS = {
    "recent_level": "近期加权水平",
    "level_change": "水平变化",
    "local_trend": "局部趋势",
    "local_curvature": "局部曲率",
    "period_level": "周期 level",
    "period_shape": "周期形状/相位",
    "fast_local_change": "快速局部变化",
}
OUTPUT_GROUP_LABELS = {
    "overall_displacement": "整体位移",
    "slow_tilt": "倾斜修正",
    "curvature": "曲率修正",
    "periodic": "周期修正",
    "recent_shape_continuation": "近期形状延续",
}
MECHANISM_LABELS = {
    ("recent_level", "overall_displacement"): "近期加权水平 → 整体位移",
    ("level_change", "slow_tilt"): "水平变化/局部趋势 → 倾斜修正",
    ("local_trend", "slow_tilt"): "水平变化/局部趋势 → 倾斜修正",
    ("period_level", "periodic"): "周期 level/幅度/相位 → 周期修正",
    ("period_shape", "periodic"): "周期 level/幅度/相位 → 周期修正",
    ("fast_local_change", "curvature"): "快速局部变化 → 局部形状修正",
    ("fast_local_change", "recent_shape_continuation"): "快速局部变化 → 局部形状修正",
    ("local_curvature", "curvature"): "快速局部变化 → 局部形状修正",
    ("local_curvature", "recent_shape_continuation"): "快速局部变化 → 局部形状修正",
    ("recent_level", "recent_shape_continuation"): "近期加权形状 → 局部形状修正",
    ("period_shape", "recent_shape_continuation"): "周期 level/幅度/相位 → 周期修正",
}


@dataclass
class SettingDictionary:
    """Semantic dictionary plus the train-split metric for one setting."""

    setting: str
    dataset: str
    horizon: int
    period: int
    input_groups: dict
    output_groups: dict
    input_group_order: list[str]
    output_group_order: list[str]
    input_templates: list
    output_templates: list
    input_shapley: dict[str, float]
    output_shapley: dict[str, float]
    moments: CenteredMoments
    # The input modes live in the 720-wide lookback space and the output modes in
    # the horizon-wide forecast space, so their covariance metrics are different
    # objects of different sizes; sharing one would silently mix the two spaces.
    output_moments: CenteredMoments | None = None
    independent_rrr: dict[int, np.ndarray] = field(default_factory=dict)

    def group_shapley_input(self, direction: np.ndarray) -> dict[str, float]:
        return group_shapley(direction, self.input_groups, self.input_group_order)

    def group_shapley_output(self, direction: np.ndarray) -> dict[str, float]:
        return group_shapley(direction, self.output_groups, self.output_group_order)


def read_cached_features(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def load_train_centered_windows(
    dataset: str,
    horizon: int,
    limit_pairs: int,
    chunk: int = 4096,
) -> np.ndarray:
    """Centered, standardized training windows ``z = x - x_last`` as ``(N, L)``.

    Uses the repository's own split borders and train-fitted scaling (the same
    helper the frozen-projector scripts use) so the semantic templates and the
    covariance live in exactly the coordinate system the branch sees before
    RevIN.
    """
    from scripts.compute_top2_direction_projectors import DATASETS, load_split

    if dataset == "Electricity":
        DATASETS["Electricity"] = ("electricity", "electricity.csv", "custom")
    seq_len = 720
    segments, _ = load_split(dataset, seq_len, REPO_ROOT / "resources" / "all_datasets")
    train = segments["train"]
    total = len(train) - seq_len - horizon + 1
    if total <= 0:
        raise ValueError(f"train split too short for {dataset}-{horizon}")
    rows = []
    collected = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        windows = np.stack(
            [train[index : index + seq_len] for index in range(start, stop)], axis=0
        )
        centered = (windows - windows[:, -1:, :]).reshape(-1, seq_len)
        rows.append(centered.astype(np.float64))
        collected += centered.shape[0]
        if limit_pairs and collected >= limit_pairs:
            break
    return np.concatenate(rows, axis=0)


def centered_moments(z: np.ndarray) -> CenteredMoments:
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    return CenteredMoments(mean=mean, cov=cov, n=int(z.shape[0]))


def output_residual_moments(
    setting: str,
    feature_paths: list[Path],
    horizon: int,
    block: int = 400,
) -> CenteredMoments | None:
    """Second moments of the validation residuals in the forecast space.

    The output-side covariance metric needs ``horizon``-wide vectors, so it
    cannot reuse the input moments.  It is estimated from the residuals the
    checkpoints themselves leave in that space (``target - phase``), pooled over
    the available seeds and computed block-wise so the transient stays bounded.
    A small ridge keeps the whitening well defined when the pooled residuals are
    rank deficient.
    """
    residuals: list[np.ndarray] = []
    collected = 0
    for path in feature_paths:
        if not path.is_file():
            continue
        payload = np.load(path)
        target = payload["target"]
        phase = payload["phase"]
        samples = target.shape[0]
        for start in range(0, samples, block):
            stop = min(start + block, samples)
            piece = (
                target[start:stop].astype(np.float64)
                - phase[start:stop].astype(np.float64)
            ).reshape(-1, horizon)
            residuals.append(piece)
            collected += piece.shape[0]
        if collected >= 20000:
            break
    if not residuals:
        return None
    stacked = np.concatenate(residuals, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    cov = centered.T @ centered / max(stacked.shape[0] - 1, 1)
    scale = float(np.trace(cov)) / max(horizon, 1)
    cov = cov + np.eye(horizon) * max(scale, 1e-12) * 1e-6
    return CenteredMoments(mean=mean, cov=cov, n=int(stacked.shape[0]))


def build_dictionary(
    setting: str,
    dataset: str,
    horizon: int,
    train_z: np.ndarray,
) -> SettingDictionary:
    period = DATASET_PERIOD_STEPS[dataset]
    raw_inputs = input_templates(720, period)
    raw_outputs = output_templates(horizon, period)
    input_groups = build_groups(raw_inputs)
    output_groups = build_groups(raw_outputs)
    moments = centered_moments(train_z)
    return SettingDictionary(
        setting=setting,
        dataset=dataset,
        horizon=horizon,
        period=period,
        input_groups=input_groups,
        output_groups=output_groups,
        input_group_order=list(input_groups),
        output_group_order=list(output_groups),
        input_templates=[t for group in input_groups.values() for t in group.templates],
        output_templates=[t for group in output_groups.values() for t in group.templates],
        input_shapley={},
        output_shapley={},
        moments=moments,
    )


def align_direction(
    direction: np.ndarray,
    templates: list,
    groups: dict,
    group_order: list[str],
    moments: CenteredMoments,
    shapley_reference: dict[str, float] | None = None,
) -> dict:
    """Semantic attribution of one canonical direction."""
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return {"dictionary_r2": 0.0}
    matrix = np.stack([template.vector for template in templates], axis=1)
    cosines = np.abs(matrix.T @ vector) / norm
    order = np.argsort(cosines)[::-1]
    best = order[0]
    explanation = {
        name: group_explanation(vector, groups[name]) for name in group_order
    }
    ranked = sorted(explanation.items(), key=lambda item: (-item[1], item[0]))
    _, dictionary_r2 = orthogonal_projection(vector, matrix)
    leave_one_out = {}
    for name in group_order:
        remaining = [item for item in group_order if item != name]
        basis = orthonormalize(
            np.concatenate([groups[item].basis for item in remaining], axis=1).T
        )
        _, value = orthogonal_projection(vector, basis)
        leave_one_out[name] = float(dictionary_r2 - value)
    shapley = group_shapley(vector, groups, group_order)
    result = {
        "dictionary_r2": float(dictionary_r2),
        "best_template": templates[best].name,
        "best_template_abs_cos": float(cosines[best]),
        "best_group": ranked[0][0],
        "best_group_explanation": float(ranked[0][1]),
        "second_group": ranked[1][0] if len(ranked) > 1 else "",
        "second_group_explanation": float(ranked[1][1]) if len(ranked) > 1 else 0.0,
        "group_explanation": explanation,
        "leave_one_group_out_r2_drop": leave_one_out,
        "shapley_r2": shapley,
        "top_templates": [
            {
                "name": templates[index].name,
                "abs_cos": float(cosines[index]),
                "signed_cos": float(matrix[:, index] @ vector / norm),
            }
            for index in order[:5]
        ],
        "covariance_correlation_max": float(
            max(
                abs(covariance_correlation(vector, template.vector, moments))
                for template in templates
            )
        ),
    }
    if shapley_reference is not None:
        result["shapley_reference_available"] = True
    return result


def canonical_modes(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of the effective map: ``(u, s, vt)``."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u, s, vt


def singular_statistics(s: np.ndarray) -> dict:
    total = float(np.sum(s))
    energy = s * s
    energy_total = float(np.sum(energy))
    shares = energy / energy_total if energy_total > 0 else np.zeros_like(energy)
    participation = float(energy_total ** 2 / np.sum(energy ** 2)) if energy_total > 0 else 0.0
    threshold = max(matrix_rank_tolerance(s), 0.0)
    numerical_rank = int(np.sum(s > threshold))
    return {
        "singular_values": s,
        "singular_energy_shares": shares,
        "cumulative_energy_share": np.cumsum(shares),
        "participation_ratio": participation,
        "numerical_rank": numerical_rank,
        "singular_value_total": total,
        "rank_tolerance": threshold,
    }


def matrix_rank_tolerance(s: np.ndarray) -> float:
    if s.size == 0:
        return 0.0
    return float(s.max() * max(s.shape) * np.finfo(np.float64).eps)


def principal_angle_gap(s: np.ndarray) -> np.ndarray:
    """Relative singular-value gap ``(s_i - s_{i+1}) / s_i`` for gap checks."""
    if s.size < 2:
        return np.zeros(0)
    return (s[:-1] - s[1:]) / np.maximum(s[:-1], 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir",
        default="research_runs/lowrank_checkpoint_information_v1/features",
        help="cached per-checkpoint validation features from the evaluator",
    )
    parser.add_argument(
        "--output-dir",
        default="research_runs/lowrank_checkpoint_information_v1",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--train-pairs",
        type=int,
        default=0,
        help="cap on training window-channel pairs used for the covariance "
        "(0 means all of them)",
    )
    parser.add_argument("--settings", default="", help="comma separated subset")
    parser.add_argument("--seeds", default="", help="comma separated subset")
    parser.add_argument("--modes", type=int, default=8, help="canonical modes reported per checkpoint")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = repo_root / args.features_dir

    rows, offsets, shared_rank = inventory_rows(repo_root, hash_checkpoints=True)
    write_inventory(rows, output_dir / "checkpoint_inventory.csv")
    print(f"inventory rows: {len(rows)}")

    wanted_settings = {item for item in args.settings.split(",") if item}
    wanted_seeds = {int(item) for item in args.seeds.split(",") if item}
    formal = [row for row in rows if not row["is_diagnostic_only"]]
    if wanted_settings:
        formal = [row for row in formal if row["setting"] in wanted_settings]
    if wanted_seeds:
        formal = [row for row in formal if int(row["seed"]) in wanted_seeds]
    print(f"checkpoints selected for Stage 1/2: {len(formal)}")

    dictionaries: dict[str, SettingDictionary] = {}
    canonical_rows: list[dict] = []
    cross_seed_rows: list[dict] = []
    semantic_rows: list[dict] = []

    by_setting: dict[str, list[dict]] = {}
    for row in formal:
        by_setting.setdefault(row["setting"], []).append(row)

    for setting in sorted(by_setting):
        entries = by_setting[setting]
        dataset = entries[0]["dataset"]
        horizon = int(entries[0]["horizon"])
        train_z = load_train_centered_windows(dataset, horizon, args.train_pairs or 0)
        dictionary = build_dictionary(setting, dataset, horizon, train_z)
        dictionary.output_moments = output_residual_moments(
            setting,
            [
                features_dir
                / f"{setting}_seed{int(entry['seed'])}_{entry['cell'].replace('/', '-')}.npz"
                for entry in entries
            ],
            horizon,
        )
        if dictionary.output_moments is None:
            raise FileNotFoundError(
                f"no cached features available to estimate the output-space "
                f"moments for {setting}"
            )
        dictionaries[setting] = dictionary
        print(
            f"[dictionary] {setting}: input "
            f"{ {k: v.dimension for k, v in dictionary.input_groups.items()} } "
            f"output { {k: v.dimension for k, v in dictionary.output_groups.items()} }"
        )
        del train_z

        # Keyed by ``(seed, cell)``: a setting has one checkpoint per seed *and*
        # per compression level, so keying by seed alone silently kept only the
        # last cell written and then compared that single object with itself.
        modes_by_key: dict[tuple[int, str], dict] = {}
        for row in sorted(entries, key=lambda item: int(item["seed"])):
            seed = int(row["seed"])
            # The evaluator writes one cache per (setting, seed, cell); the cell
            # suffix is mandatory because a setting/seed pair can carry more than
            # one compression level.
            feature_path = (
                features_dir
                / f"{setting}_seed{seed}_{row['cell'].replace('/', '-')}.npz"
            )
            if not feature_path.is_file():
                raise FileNotFoundError(
                    f"missing cached validation features: {feature_path}"
                )
            features = read_cached_features(feature_path)
            encoder_weight = features["encoder_weight"].astype(np.float64)
            decoder_weight = features["decoder_weight"].astype(np.float64)
            matrix, _ = effective_map(
                encoder_weight, decoder_weight, encoder_weight.shape[1]
            )
            u, s, vt = canonical_modes(matrix)
            statistics = singular_statistics(s)
            gaps = principal_angle_gap(s)

            hidden = features["hidden"].astype(np.float64)
            n_samples = hidden.shape[0]
            # The cache stores ``sigma``/``mu`` as ``(N, 1, C)``; the horizon axis
            # is the window's RevIN statistics, constant over the horizon, so it is
            # dropped to recover the per-(sample, channel) scale the branch uses.
            sigma = features["sigma"].astype(np.float64).reshape(n_samples, -1)
            # ``correction_abs = decoder(h) * sigma`` is the low-rank branch
            # correction in the original value space, exactly as the evaluator
            # forms it.  The modes are taken in the canonical latent frame of the
            # decoder: its left singular vectors are orthonormal, so the modal
            # contributions ``(S_k u_k) <v_k, h>`` sum back to ``decoder(h)``
            # exactly.  Their squares therefore partition the decoded energy, and
            # each mode's energy is its mean square under the same RevIN scaling;
            # ``total_correction_energy`` is the evaluator's own total, so the
            # reported shares are computed against the quantity the intervention
            # arms use.
            # The reported spectrum comes from the effective map, while the
            # per-mode attribution below lives in the decoder's latent frame;
            # the two can differ in length, so the mode loop is clamped to the
            # shorter of the two rather than risking an out-of-range index.
            mode_count = min(int(args.modes), s.size)
            # ``u``/``s``/``vt`` above decompose the *effective map* and therefore
            # span the input space (720 wide), which the hidden state does not live
            # in.  The per-mode attribution below needs the decoder's own latent
            # frame, so it is taken separately; the spectrum statistics keep using
            # the effective map as planned.
            dec_u, dec_s, dec_vt = np.linalg.svd(decoder_weight, full_matrices=False)
            correction = (
                np.einsum("ncr,hr->nhc", hidden, decoder_weight)
                * sigma[:, None, :]
            )
            total_correction_energy = float(np.mean(correction ** 2))
            mode_energies = np.empty(dec_s.size)
            score_variance = np.empty(dec_s.size)
            for index in range(dec_s.size):
                score = np.einsum("ncr,r->nc", hidden, dec_vt[index])
                contribution = (
                    (dec_u[:, index] * dec_s[index])[None, :, None]
                    * score[:, None, :]
                    * sigma[:, None, :]
                )
                mode_energies[index] = float(np.mean(contribution ** 2))
                # The plan's ``Var(h_i)`` is the variance of the canonical latent
                # score, i.e. of the projection onto the mode's latent direction --
                # not of a raw hidden unit, which is not the same quantity.
                score_variance[index] = float(np.var(score))
            for index in range(min(mode_count, dec_s.size)):
                energy = float(mode_energies[index])
                coefficient_std = float(np.sqrt(max(score_variance[index], 0.0)))
                canonical_rows.append(
                    {
                        "setting": setting,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "cell": row["cell"],
                        "rank": row["rank"],
                        "mode_index": index,
                        "singular_value": float(s[index]),
                        "singular_value_share": float(statistics["singular_energy_shares"][index]),
                        "cumulative_singular_share": float(statistics["cumulative_energy_share"][index]),
                        "singular_gap_to_next": float(gaps[index]) if index < gaps.size else 0.0,
                        "latent_variance": float(score_variance[index]),
                        "latent_variance_share": float(
                            score_variance[index] / score_variance.sum()
                        ) if score_variance.sum() else 0.0,
                        "latent_variance_std": coefficient_std,
                        "correction_energy": energy,
                        "correction_energy_share": float(energy / total_correction_energy) if total_correction_energy else 0.0,
                        "bias_energy": 0.0,
                        "total_correction_energy": total_correction_energy,
                        "participation_ratio": statistics["participation_ratio"],
                        "numerical_rank": statistics["numerical_rank"],
                        "input_basis_orthonormality_error": orthonormality_error(vt[: s.size].T),
                        "output_basis_orthonormality_error": orthonormality_error(u[:, : s.size]),
                        "phase_gate_mean": float(features["gate"].mean()),
                        "sigma_mean": float(sigma.mean()),
                    }
                )
            modes_by_key[(seed, row["cell"])] = {
                "u": u,
                "s": s,
                "vt": vt,
                "statistics": statistics,
                "row": row,
                "total_correction_energy": total_correction_energy,
            }
            # A mode whose gap to the next singular value is small must not be
            # interpreted alone, so the flag travels with the checkpoint.
            print(
                f"[modes] {setting} seed={seed} {row['cell']} rank={row['rank']} "
                f"numerical_rank={statistics['numerical_rank']} "
                f"participation={statistics['participation_ratio']:.3f}"
            )
            del features, hidden, sigma, mode_energies

        # Cross-seed stability is only meaningful *within* a compression level:
        # different cells have different ranks, so pairing them would compare
        # unlike objects.  Each cell is therefore paired across the seeds that
        # have it, which is also what plan section 1 asks for (same setting and
        # same rank).
        cells = sorted({cell for (_, cell) in modes_by_key})
        for cell in cells:
            seeds_here = sorted(seed for (seed, c) in modes_by_key if c == cell)
            if len(seeds_here) < 2:
                continue
            reference = modes_by_key[(seeds_here[0], cell)]
            for other in seeds_here[1:]:
                candidate = modes_by_key[(other, cell)]
                for label, dimension in (("leading4", 4), ("leading8", 8), ("full", 0)):
                    dim = (
                        min(reference["s"].size, candidate["s"].size)
                        if dimension == 0
                        else min(dimension, reference["s"].size, candidate["s"].size)
                    )
                    if dim < 1:
                        continue
                    input_a, input_b = reference["vt"][:dim].T, candidate["vt"][:dim].T
                    output_a, output_b = reference["u"][:, :dim], candidate["u"][:, :dim]
                    angles_in = principal_angles(input_a, input_b)
                    angles_out = principal_angles(output_a, output_b)
                    matched = np.max(np.abs(input_a.T @ input_b), axis=1)
                    cross_seed_rows.append(
                        {
                            "setting": setting,
                            "dataset": dataset,
                            "horizon": horizon,
                            "cell": cell,
                        "seed_a": int(seeds_here[0]),
                            "seed_b": int(other),
                            "scope": label,
                            "dimension": int(dim),
                            "input_subspace_overlap": float(projection_overlap(input_a, input_b)),
                            "output_subspace_overlap": float(projection_overlap(output_a, output_b)),
                            "input_principal_angle_max_deg": float(angles_in.max()),
                            "input_principal_angle_mean_deg": float(angles_in.mean()),
                            "output_principal_angle_max_deg": float(angles_out.max()),
                            "output_principal_angle_mean_deg": float(angles_out.mean()),
                            "matched_mode_cosine_min": float(matched.min()),
                            "matched_mode_cosine_mean": float(matched.mean()),
                            "matched_modes_above_0p7": int(np.sum(matched >= 0.7)),
                            "cell_a": reference["row"]["cell"],
                            "cell_b": candidate["row"]["cell"],
                            "rank_a": reference["row"]["rank"],
                            "rank_b": candidate["row"]["rank"],
                        }
                    )

        for row in sorted(entries, key=lambda item: (int(item["seed"]), item["cell"])):
            seed = int(row["seed"])
            modes = modes_by_key[(seed, row["cell"])]
            for index in range(min(modes["s"].size, int(args.modes))):
                input_alignment = align_direction(
                    modes["vt"][index],
                    dictionary.input_templates,
                    dictionary.input_groups,
                    dictionary.input_group_order,
                    dictionary.moments,
                )
                output_alignment = align_direction(
                    modes["u"][:, index],
                    dictionary.output_templates,
                    dictionary.output_groups,
                    dictionary.output_group_order,
                    # Output modes are horizon-wide, so they use the residual
                    # moments of that space, not the lookback-space moments.
                    dictionary.output_moments,
                )
                semantic_rows.append(
                    {
                        "setting": setting,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "cell": row["cell"],
                        "rank": row["rank"],
                        "mode_index": index,
                        "singular_value": float(modes["s"][index]),
                        "singular_value_share": float(modes["statistics"]["singular_energy_shares"][index]),
                        "input_best_template": input_alignment["best_template"],
                        "input_best_template_abs_cos": input_alignment["best_template_abs_cos"],
                        "input_best_group": input_alignment["best_group"],
                        "input_group_explanation": input_alignment["best_group_explanation"],
                        "input_second_group": input_alignment["second_group"],
                        "input_second_group_explanation": input_alignment["second_group_explanation"],
                        "input_dictionary_r2": input_alignment["dictionary_r2"],
                        "input_covariance_correlation_max": input_alignment["covariance_correlation_max"],
                        "output_best_template": output_alignment["best_template"],
                        "output_best_template_abs_cos": output_alignment["best_template_abs_cos"],
                        "output_best_group": output_alignment["best_group"],
                        "output_group_explanation": output_alignment["best_group_explanation"],
                        "output_second_group": output_alignment["second_group"],
                        "output_second_group_explanation": output_alignment["second_group_explanation"],
                        "output_dictionary_r2": output_alignment["dictionary_r2"],
                        "output_covariance_correlation_max": output_alignment["covariance_correlation_max"],
                        "paired_mechanism": MECHANISM_LABELS.get(
                            (input_alignment["best_group"], output_alignment["best_group"]),
                            "未匹配预注册机制",
                        ),
                        "input_group_explanation_json": json.dumps(input_alignment["group_explanation"]),
                        "output_group_explanation_json": json.dumps(output_alignment["group_explanation"]),
                        "input_leave_one_out_json": json.dumps(input_alignment["leave_one_group_out_r2_drop"]),
                        "output_leave_one_out_json": json.dumps(output_alignment["leave_one_group_out_r2_drop"]),
                        "input_shapley_json": json.dumps(input_alignment["shapley_r2"]),
                        "output_shapley_json": json.dumps(output_alignment["shapley_r2"]),
                        "input_top_templates_json": json.dumps(input_alignment["top_templates"]),
                        "output_top_templates_json": json.dumps(output_alignment["top_templates"]),
                    }
                )

    write_csv(canonical_rows, output_dir / "canonical_modes.csv")
    write_csv(cross_seed_rows, output_dir / "cross_seed_alignment.csv")
    write_csv(semantic_rows, output_dir / "semantic_alignment.csv")
    summary = {
        "inventory_rows": len(rows),
        "formal_checkpoints": len(formal),
        "canonical_mode_rows": len(canonical_rows),
        "cross_seed_rows": len(cross_seed_rows),
        "semantic_rows": len(semantic_rows),
        "offsets": offsets,
        "shared_rank_cells": shared_rank,
        "dictionary_dimensions": {
            setting: {
                "input_groups": {k: v.dimension for k, v in dictionary.input_groups.items()},
                "output_groups": {k: v.dimension for k, v in dictionary.output_groups.items()},
                "input_templates": len(dictionary.input_templates),
                "output_templates": len(dictionary.output_templates),
                "train_pairs": dictionary.moments.n,
            }
            for setting, dictionary in dictionaries.items()
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("analysis complete")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
