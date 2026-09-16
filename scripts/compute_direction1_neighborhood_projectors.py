#!/usr/bin/env python3
"""Build train-only bootstrap neighborhoods around RRR direction 1.

The exact direction-1 projector is rank one.  To test whether useful
information lies in a stable local neighborhood around that direction, this
script:

1. partitions train-window origins into contiguous blocks;
2. resamples those blocks with replacement and re-estimates direction 1;
3. sign-aligns every bootstrap direction to the full-train direction 1;
4. applies PCA to the deviations orthogonal to direction 1; and
5. saves nested frozen bases Qcone1/Qcone2/Qcone4/Qcone8.

The word "cone" is descriptive.  A linear projector cannot preserve a literal
angular cone without also preserving its linear span, so the operative width
parameter is the retained tangent rank ``k - 1``.

Only the training split is read.  This script does not train PhaseFormer.

Example::

    python scripts/compute_direction1_neighborhood_projectors.py \
        --output-dir research_runs/direction1_neighborhood_v1/projectors
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_top2_direction_projectors import (  # noqa: E402
    DATASETS,
    load_split,
    orthonormalize,
    orthogonality_error,
    parse_settings,
    projection_idempotence_error,
    recover_error,
    rrr_directions,
)


DEFAULT_WIDTHS = (1, 2, 4, 8)
DEFAULT_SETTINGS = (
    "ETTh2:96",
    "ETTh2:720",
    "ETTm2:96",
    "ETTm2:192",
    "Weather:96",
    "Weather:192",
    "Electricity:336",
)
DATASETS["Electricity"] = ("electricity", "electricity.csv", "custom")


def parse_widths(raw: str) -> list[int]:
    widths = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not widths or widths[0] != 1:
        raise ValueError("widths must include 1 so the exact direction-1 control is saved")
    if any(width < 1 for width in widths):
        raise ValueError("all widths must be positive")
    return widths


def contiguous_origin_blocks(total_origins: int, n_blocks: int) -> list[tuple[int, int]]:
    if total_origins < 2:
        raise ValueError("at least two train-window origins are required")
    if n_blocks < 2:
        raise ValueError("bootstrap block count must be at least 2")
    n_blocks = min(n_blocks, total_origins)
    edges = np.linspace(0, total_origins, n_blocks + 1, dtype=int)
    return [
        (int(start), int(stop))
        for start, stop in zip(edges[:-1], edges[1:])
        if stop > start
    ]


def accumulate_block_moments(
    segment: np.ndarray,
    seq_len: int,
    pred_len: int,
    n_blocks: int,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return per-origin-block unnormalized Szz/Szy sums and pair counts."""
    total = len(segment) - seq_len - pred_len + 1
    if total <= 0:
        raise ValueError(
            f"segment length {len(segment)} is too short for "
            f"seq_len={seq_len}, pred_len={pred_len}"
        )
    blocks = contiguous_origin_blocks(total, n_blocks)
    x_all = sliding_window_view(segment, seq_len, axis=0)[:total]
    y_all = sliding_window_view(segment[seq_len:], pred_len, axis=0)[:total]

    szz_blocks = []
    szy_blocks = []
    pair_counts = []
    for block_start, block_stop in blocks:
        szz = np.zeros((seq_len, seq_len), dtype=np.float64)
        szy = np.zeros((seq_len, pred_len), dtype=np.float64)
        count = 0
        for start in range(block_start, block_stop, chunk):
            stop = min(start + chunk, block_stop)
            x = np.asarray(x_all[start:stop], dtype=np.float64)
            y = np.asarray(y_all[start:stop], dtype=np.float64)
            last = x[:, :, -1:]
            z = (x - last).reshape(-1, seq_len)
            d = (y - last).reshape(-1, pred_len)
            szz += z.T @ z
            szy += z.T @ d
            count += len(z)
        szz_blocks.append(szz)
        szy_blocks.append(szy)
        pair_counts.append(count)

    return (
        np.stack(szz_blocks),
        np.stack(szy_blocks),
        np.asarray(pair_counts, dtype=np.int64),
        blocks,
    )


def weighted_moments(
    szz_blocks: np.ndarray,
    szy_blocks: np.ndarray,
    pair_counts: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    count = float(weights @ pair_counts)
    if count <= 0:
        raise ValueError("bootstrap replicate has no samples")
    szz = np.tensordot(weights, szz_blocks, axes=(0, 0)) / count
    szy = np.tensordot(weights, szy_blocks, axes=(0, 0)) / count
    return szz, szy


def normalized_direction1(szz: np.ndarray, szy: np.ndarray, ridge: float) -> np.ndarray:
    _, directions, _ = rrr_directions(szz, szy, ridge)
    direction = np.asarray(directions[0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("direction 1 is numerically zero")
    return direction / norm


def sign_align(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return direction if float(direction @ reference) >= 0 else -direction


def bootstrap_direction_cloud(
    szz_blocks: np.ndarray,
    szy_blocks: np.ndarray,
    pair_counts: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    """Blocked multinomial bootstrap of the leading input direction."""
    if replicates < 2:
        raise ValueError("bootstrap replicate count must be at least 2")
    n_blocks = len(pair_counts)
    rng = np.random.default_rng(seed)
    cloud = []
    probabilities = np.full(n_blocks, 1.0 / n_blocks)
    for _ in range(replicates):
        weights = rng.multinomial(n_blocks, probabilities)
        szz, szy = weighted_moments(
            szz_blocks, szy_blocks, pair_counts, weights
        )
        cloud.append(sign_align(normalized_direction1(szz, szy, ridge), reference))
    return np.stack(cloud)


def tangent_pca(
    cloud: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of bootstrap deviations in the tangent space orthogonal to q1."""
    coefficients = cloud @ reference
    deviations = cloud - coefficients[:, None] * reference[None, :]
    _, singular_values, vh = np.linalg.svd(deviations, full_matrices=False)
    tolerance = max(deviations.shape) * np.finfo(np.float64).eps
    tolerance *= singular_values[0] if len(singular_values) else 0.0
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        raise ValueError("bootstrap directions have no nonzero tangent variation")
    tangent = vh[:rank].T
    tangent -= np.outer(reference, reference @ tangent)
    tangent, _ = np.linalg.qr(tangent)
    energy = singular_values[:rank] ** 2
    explained = energy / max(float(energy.sum()), 1e-12)
    return tangent, explained, deviations


def build_neighborhood_bases(
    reference: np.ndarray, tangent: np.ndarray, widths: list[int]
) -> dict[int, np.ndarray]:
    max_width = max(widths)
    if tangent.shape[1] < max_width - 1:
        raise ValueError(
            f"need {max_width - 1} tangent directions, found {tangent.shape[1]}"
        )
    bases = {}
    for width in widths:
        raw = np.column_stack([reference, tangent[:, : width - 1]])
        basis, _ = np.linalg.qr(raw)
        bases[width] = np.ascontiguousarray(basis[:, :width], dtype=np.float64)
    return bases


def subspace_statistics(
    basis: np.ndarray,
    szz: np.ndarray,
    szy: np.ndarray,
    total_predictive_gain: float,
    ridge: float,
) -> tuple[float, float]:
    """Visible input variance and optimal scalar/subspace MSE-gain capture."""
    trace = max(float(np.trace(szz)), 1e-12)
    variance_share = float(np.trace(basis.T @ szz @ basis) / trace)
    gram = basis.T @ (szz + ridge * np.eye(szz.shape[0])) @ basis
    cross = basis.T @ szy
    gain = float(np.trace(np.linalg.solve(gram, cross @ cross.T)))
    capture = gain / max(total_predictive_gain, 1e-12)
    return variance_share, capture


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()
    ).hexdigest()


def run_setting(
    dataset: str,
    horizon: int,
    *,
    seq_len: int,
    ridge: float,
    chunk: int,
    data_root: Path,
    output_dir: Path,
    widths: list[int],
    bootstrap_blocks: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict]:
    segments, borders = load_split(dataset, seq_len, data_root)
    train = segments["train"]
    szz_blocks, szy_blocks, pair_counts, blocks = accumulate_block_moments(
        train, seq_len, horizon, bootstrap_blocks, chunk
    )
    ones = np.ones(len(pair_counts), dtype=np.float64)
    szz, szy = weighted_moments(szz_blocks, szy_blocks, pair_counts, ones)
    eigenvalues, directions, _ = rrr_directions(szz, szy, ridge)
    reference = normalized_direction1(szz, szy, ridge)
    direction2 = sign_align(directions[1], reference)
    direction2 -= reference * float(reference @ direction2)
    direction2 /= max(float(np.linalg.norm(direction2)), 1e-12)

    cloud = bootstrap_direction_cloud(
        szz_blocks,
        szy_blocks,
        pair_counts,
        reference,
        replicates=bootstrap_replicates,
        ridge=ridge,
        seed=bootstrap_seed,
    )
    tangent, tangent_explained, deviations = tangent_pca(cloud, reference)
    bases = build_neighborhood_bases(reference, tangent, widths)
    qrrr2, _ = orthonormalize(directions[:2])
    qrrr2_path = output_dir / f"{dataset}_{horizon}_Qrrr2.npy"
    np.save(qrrr2_path, qrrr2)

    cosines = np.clip(cloud @ reference, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    total_gain = float(np.clip(eigenvalues, 0.0, None).sum())
    cumulative_tangent = np.cumsum(tangent_explained)
    rows = []
    previous_basis = None
    for width in widths:
        basis = bases[width]
        path = output_dir / f"{dataset}_{horizon}_Qcone{width}.npy"
        np.save(path, basis)
        variance_share, predictive_capture = subspace_statistics(
            basis, szz, szy, total_gain, ridge
        )
        tangent_rank = width - 1
        q2_overlap = (
            float(np.linalg.norm(basis[:, 1:].T @ direction2) ** 2)
            if tangent_rank
            else 0.0
        )
        nested_error = (
            0.0
            if previous_basis is None
            else float(
                np.max(
                    np.abs(
                        previous_basis
                        - basis @ (basis.T @ previous_basis)
                    )
                )
            )
        )
        row = {
            "setting": f"{dataset}-{horizon}",
            "dataset": dataset,
            "horizon": int(horizon),
            "seq_len": int(seq_len),
            "width": int(width),
            "tangent_rank": int(tangent_rank),
            "ridge": float(ridge),
            "bootstrap_blocks": int(len(blocks)),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "train_windows_per_channel": int(
                len(train) - seq_len - horizon + 1
            ),
            "train_window_channel_pairs": int(pair_counts.sum()),
            "block_origin_min": int(min(stop - start for start, stop in blocks)),
            "block_origin_max": int(max(stop - start for start, stop in blocks)),
            "angle_median_deg": float(np.median(angles)),
            "angle_p90_deg": float(np.quantile(angles, 0.90)),
            "angle_max_deg": float(np.max(angles)),
            "mean_tangent_norm": float(np.linalg.norm(deviations, axis=1).mean()),
            "bootstrap_tangent_effective_rank": int(tangent.shape[1]),
            "tangent_deviation_explained": (
                float(cumulative_tangent[tangent_rank - 1])
                if tangent_rank
                else 0.0
            ),
            "direction2_overlap": q2_overlap,
            "visible_variance_share": variance_share,
            "independent_predictive_capture": predictive_capture,
            "orthogonality_error": orthogonality_error(basis),
            "projector_idempotence_error": projection_idempotence_error(basis),
            "recover_direction1_error": recover_error(basis, reference),
            "nested_previous_error": nested_error,
            "basis_file": path.name,
            "basis_sha256": array_sha256(basis),
            "rrr2_basis_file": qrrr2_path.name,
            "rrr2_basis_sha256": array_sha256(qrrr2),
            "rrr2_orthogonality_error": orthogonality_error(qrrr2),
            "rrr2_projector_idempotence_error": projection_idempotence_error(qrrr2),
            "splits_read": ["train"],
            "raw_rows": int(borders["raw_rows"]),
            "n_channels": int(borders["n_channels"]),
        }
        row["checks"] = {
            "train_only": row["splits_read"] == ["train"],
            "orthonormal": row["orthogonality_error"] < 1e-8,
            "idempotent": row["projector_idempotence_error"] < 1e-8,
            "contains_direction1": row["recover_direction1_error"] < 1e-8,
            "nested": row["nested_previous_error"] < 1e-8,
            "rrr2_valid": (
                row["rrr2_orthogonality_error"] < 1e-8
                and row["rrr2_projector_idempotence_error"] < 1e-8
            ),
            "finite": all(
                np.isfinite(row[key])
                for key in (
                    "angle_median_deg",
                    "angle_p90_deg",
                    "visible_variance_share",
                    "independent_predictive_capture",
                )
            ),
        }
        row["checks"]["all_passed"] = all(row["checks"].values())
        rows.append(row)
        previous_basis = basis
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "checks"}
        flat["all_checks_passed"] = row["checks"]["all_passed"]
        flat_rows.append(flat)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Direction-1 bootstrap neighborhood Stage 0 audit",
        "",
        "Generated from the train split only. No PhaseFormer model is trained.",
        "",
        "| Setting | k | angle median/p90 | tangent variance explained | "
        "visible input variance | independent predictive capture | q2 overlap | QC |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {row['width']} | "
            f"{row['angle_median_deg']:.2f}° / {row['angle_p90_deg']:.2f}° | "
            f"{100 * row['tangent_deviation_explained']:.1f}% | "
            f"{100 * row['visible_variance_share']:.2f}% | "
            f"{100 * row['independent_predictive_capture']:.1f}% | "
            f"{100 * row['direction2_overlap']:.1f}% | "
            f"{'PASS' if row['checks']['all_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The neighborhood width is the retained tangent rank, not a literal",
            "angular threshold. All bases are nested and contain the exact full-train q1.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--bootstrap-blocks", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=64)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "resources" / "all_datasets")
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    widths = parse_widths(args.widths)
    if args.bootstrap_replicates < max(widths):
        raise ValueError(
            "bootstrap_replicates must be at least the largest requested width"
        )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, horizon in parse_settings(args.settings):
        print(
            f"[stage0] direction-1 neighborhood for {dataset}-{horizon}",
            flush=True,
        )
        rows.extend(
            run_setting(
                dataset,
                horizon,
                seq_len=args.seq_len,
                ridge=args.ridge,
                chunk=args.chunk,
                data_root=Path(args.data_root),
                output_dir=output_dir,
                widths=widths,
                bootstrap_blocks=args.bootstrap_blocks,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
        )

    index = {
        "protocol": "direction1-bootstrap-neighborhood-v1",
        "only_train_split_read": True,
        "seq_len": int(args.seq_len),
        "ridge": float(args.ridge),
        "bootstrap_blocks": int(args.bootstrap_blocks),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "widths": widths,
        "construction": (
            "blocked-bootstrap q1 estimates -> sign alignment -> tangent PCA -> "
            "nested [q1, tangent PCs] bases"
        ),
        "projectors": {
            f"{row['dataset']}-{row['horizon']}-k{row['width']}": {
                "file": row["basis_file"],
                "sha256": row["basis_sha256"],
                "all_checks_passed": row["checks"]["all_passed"],
            }
            for row in rows
        },
        "rrr2_projectors": {
            f"{row['dataset']}-{row['horizon']}": {
                "file": row["rrr2_basis_file"],
                "sha256": row["rrr2_basis_sha256"],
            }
            for row in rows
            if row["width"] == widths[0]
        },
        "all_checks_passed": all(row["checks"]["all_passed"] for row in rows),
    }
    (output_dir / "projectors.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "stage0_audit.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    write_csv(rows, output_dir / "stage0_audit.csv")
    (output_dir / "stage0_audit.md").write_text(render_markdown(rows))

    failed = [
        f"{row['setting']}-k{row['width']}"
        for row in rows
        if not row["checks"]["all_passed"]
    ]
    print(json.dumps({"all_checks_passed": not failed, "failed": failed}, indent=2))
    if failed:
        raise SystemExit(f"Stage 0 audit failed for: {failed}")


if __name__ == "__main__":
    main()
