#!/usr/bin/env python3
"""Section 10 of the low-rank checkpoint analysis plan: subspace alignment
between the dense temporal head and the low-rank head that compresses it.

Answers the plan's sharpest question -- does the low-rank head retain the
directions carrying the *dense* head's ordinary weight energy, or the directions
carrying its *prediction-relevant* energy?  Those are different objects, and the
plan (section 10.1) explicitly forbids reading prediction importance off the
dense singular values.

For every (setting, seed) with a dense cache this reports

* principal angles and projection overlap between the low-rank input/output
  subspaces and the dense input/output subspaces, taken both by ordinary
  singular order and by the dense *functional* order (modes ranked by their
  measured fused-MSE contribution);
* the share of the dense model's attainable improvement that survives when its
  effective map is restricted to the low-rank input subspace, which is the
  operational form of "what did the compression keep?".

Reads only cached validation tensors.  No training, no checkpoint modification,
and the test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_lowrank_functional_rank import analyse_cell, canonical_modes  # noqa: E402
from lowrank_checkpoint_core import principal_angles, projection_overlap  # noqa: E402


def orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the span of ``rows`` (k, L), returned as (L, d)."""
    if rows.shape[0] == 0:
        return np.zeros((rows.shape[1], 0))
    q, r = np.linalg.qr(rows.T)
    keep = int(np.sum(np.abs(np.diag(r)) > 1e-10 * max(rows.shape)))
    return np.ascontiguousarray(q[:, :max(keep, 1)])


def _rank_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties shared, so a rank correlation needs no scipy."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def top_k_rows(matrix: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    return matrix[order[:k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", default="research_runs/lowrank_functional_rank_v1/dense_features")
    parser.add_argument("--modes-dir", default="research_runs/lowrank_functional_rank_v1/modes")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--subspace-dims", default="2,4,8,16")
    parser.add_argument("--cells", default="q=1/4,q=1/8")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    dense_dir = repo_root / args.dense_dir
    modes_dir = repo_root / args.modes_dir
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = [int(value) for value in args.subspace_dims.split(",") if value]
    wanted_cells = [value for value in args.cells.split(",") if value]

    rows: list[dict] = []
    for dense_path in sorted(dense_dir.glob("*_dense.npz")):
        stem = dense_path.stem                       # e.g. ETTh2-96_seed2021_dense
        setting, seed_part, _ = stem.rsplit("_", 2)
        seed = int(seed_part.replace("seed", ""))
        dataset, horizon = setting.rsplit("-", 1)
        horizon = int(horizon)

        payload = {key: value.astype(np.float64) for key, value in np.load(dense_path).items()}
        cell_row, _, _, _, dense_modes = analyse_cell(
            payload, setting, dataset, horizon, seed, "dense", horizon,
            random_repeats=1, rng=np.random.default_rng(0), top_modes=1,
        )
        u_d, s_d, vt_d = dense_modes["u"], dense_modes["s"], dense_modes["vt"]
        I_d = dense_modes["contribution"]

        dw = payload["decoder_weight"]
        ew = payload["encoder_weight"]
        W_d = dw @ ew
        mapped_eb = dw @ payload["encoder_bias"]
        c_const = mapped_eb + payload["decoder_bias"]
        sigma = payload["sigma"]
        mu = payload["mu"]
        gate = payload["gate"]
        phase = payload["phase"]
        target = payload["target"]
        z = payload["z"]
        x_last = payload["x_last_norm"] + float(mapped_eb.mean())
        last_abs = sigma * x_last + mu

        # Dense baseline and the "no map" baseline, in the same algebra as the
        # low-rank analysis, so the ratios below are same-protocol.
        w_z_dense = np.einsum("hl,nlc->nhc", W_d, z, optimize=True)
        branch_dense = last_abs + sigma * (w_z_dense + c_const[None, :, None])
        fused_dense = (1.0 - gate) * phase + gate * branch_dense
        fused_zero = (1.0 - gate) * phase + gate * (
            last_abs + sigma * c_const[None, :, None]
        )
        mse_dense = float(np.mean((fused_dense - target) ** 2))
        mse_zero = float(np.mean((fused_zero - target) ** 2))
        improvement = mse_zero - mse_dense
        del w_z_dense, branch_dense, fused_dense, fused_zero

        singular_order = np.argsort(-s_d, kind="stable")
        functional_order = np.argsort(-I_d, kind="stable")

        # The plan's section 10.4 asks whether the dense head's ordinary
        # weight-energy order agrees with its prediction-relevant order at all.
        # If it does, the "low-rank kept the functional rather than the energetic
        # subspace" question is not separable on this model family, and that is
        # itself the finding; so the agreement is measured, not assumed.
        k_probe = int(min(16, s_d.size))
        basis_singular = orthonormal_basis(top_k_rows(vt_d, singular_order, k_probe))
        basis_functional = orthonormal_basis(top_k_rows(vt_d, functional_order, k_probe))
        order_rho = float(np.corrcoef(
            _rank_ranks(s_d), _rank_ranks(I_d)
        )[0, 1])
        rows.append({
            "setting": setting, "dataset": dataset, "horizon": horizon,
            "seed": seed, "cell": "dense", "rank": int(dense_modes["rank"]),
            "subspace_dim": k_probe, "dense_reference": "dense_singular_vs_functional",
            "input_overlap": projection_overlap(basis_singular, basis_functional),
            "input_cos_mean": "", "input_cos_min": "",
            "output_overlap": "", "output_cos_mean": "", "output_cos_min": "",
            "dense_improvement_retained": "", "dense_mse_with_restriction": "",
            "singular_vs_functional_rank_correlation": order_rho,
        })

        for cell in wanted_cells:
            mode_path = modes_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
            if not mode_path.is_file():
                continue
            low = np.load(mode_path)
            u_l, s_l, vt_l = low["u"], low["s"], low["vt"]
            I_l = low["contribution"]
            low_functional = np.argsort(-I_l, kind="stable")
            rank_l = int(low["rank"])

            for k in dims:
                k = min(k, rank_l)
                basis_in_low = orthonormal_basis(top_k_rows(vt_l, low_functional, k))
                basis_out_low = orthonormal_basis(top_k_rows(u_l.T, low_functional, k))
                for label, order in (("dense_singular", singular_order),
                                     ("dense_functional", functional_order)):
                    m = min(k, order.size)
                    basis_in_dense = orthonormal_basis(top_k_rows(vt_d, order, m))
                    basis_out_dense = orthonormal_basis(top_k_rows(u_d.T, order, m))
                    in_cos = np.cos(np.radians(principal_angles(basis_in_low, basis_in_dense)))
                    out_cos = np.cos(np.radians(principal_angles(basis_out_low, basis_out_dense)))
                    rows.append({
                        "setting": setting, "dataset": dataset, "horizon": horizon,
                        "seed": seed, "cell": cell, "rank": rank_l,
                        "subspace_dim": k, "dense_reference": label,
                        "input_overlap": projection_overlap(basis_in_low, basis_in_dense),
                        "input_cos_mean": float(in_cos.mean()),
                        "input_cos_min": float(in_cos.min()),
                        "output_overlap": projection_overlap(basis_out_low, basis_out_dense),
                        "output_cos_mean": float(out_cos.mean()),
                        "output_cos_min": float(out_cos.min()),
                        "dense_improvement_retained": "",
                        "dense_mse_with_restriction": "",
                        "singular_vs_functional_rank_correlation": "",
                    })

            # Operational retention: restrict the dense map to the low-rank
            # input subspace and measure the surviving improvement.
            for k in dims:
                k = min(k, rank_l)
                basis_in_low = orthonormal_basis(top_k_rows(vt_l, low_functional, k))
                projector = basis_in_low @ basis_in_low.T
                W_restricted = W_d @ projector
                w_z = np.einsum("hl,nlc->nhc", W_restricted, z, optimize=True)
                branch = last_abs + sigma * (w_z + c_const[None, :, None])
                fused = (1.0 - gate) * phase + gate * branch
                mse_restricted = float(np.mean((fused - target) ** 2))
                del w_z, branch, fused, projector, W_restricted
                rows.append({
                    "setting": setting, "dataset": dataset, "horizon": horizon,
                    "seed": seed, "cell": cell, "rank": rank_l,
                    "subspace_dim": k, "dense_reference": "input_subspace_restriction",
                    "input_overlap": "", "input_cos_mean": "", "input_cos_min": "",
                    "output_overlap": "", "output_cos_mean": "", "output_cos_min": "",
                    "dense_improvement_retained": (
                        (mse_zero - mse_restricted) / improvement
                        if abs(improvement) > 1e-15 else float("nan")
                    ),
                    "dense_mse_with_restriction": mse_restricted,
                    "singular_vs_functional_rank_correlation": "",
                })
        print(
            f"  {setting} seed={seed}: dense improvement={improvement:.6f} "
            f"dense rank={dense_modes['rank']}", flush=True,
        )

    path = output_dir / "dense_alignment.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


if __name__ == "__main__":
    main()
