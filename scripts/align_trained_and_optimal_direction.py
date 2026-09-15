#!/usr/bin/env python3
"""Compare the input subspace used by trained heads with the optimal one.

For each setting this script builds the row space (an input-side subspace of the
centered-window space) of

  * the analytically optimal rank-r map ``W_r = U_r U_r^T Szy^T Szz^-1`` computed
    from the saved second moments, and
  * the effective map of each trained checkpoint (``linear.weight`` for the
    direct head, ``decoder.weight @ encoder.weight`` for the pooled low-rank
    head), restricted to its top-r input directions,

and reports the subspace overlap ``trace(P P*) / r`` in [0, 1].  A high overlap
means stochastic training actually lands on the same few input directions that
the closed-form optimum uses, i.e. that the rank sweep's preserved performance
reflects a genuine data property rather than only an algebraic construction.

No training, no dataset access: only checkpoints plus the moments dump.

Usage:

    python scripts/align_trained_and_optimal_direction.py \
        --moments-dir research_runs/lowrank_data_property_v2 \
        --runs-root research_runs \
        --output-csv research_runs/lowrank_data_property_v2/trained_vs_optimal_alignment.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

RIDGE = 1e-8
RUN_RE = re.compile(
    r"confirm_(?P<dataset>[a-z0-9]+)_h(?P<horizon>\d+)_"
    r"(?P<mechanism>[a-z_]+)_p(?P<period>\d+)_.*_s(?P<seed>\d+)_"
)


def optimal_row_basis(szz, szy, rank):
    """Orthonormal basis of the row space of the optimal rank-r map."""
    szz_r = szz + RIDGE * np.eye(szz.shape[0])
    ols = np.linalg.solve(szz_r, szy).T  # H x L
    s_mat = szy.T @ np.linalg.solve(szz_r, szy)
    s_mat = 0.5 * (s_mat + s_mat.T)
    vals, vecs = np.linalg.eigh(s_mat)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    rr = min(rank, vecs.shape[1])
    return vecs[:, :rr].T @ ols  # rr x L


def orthonormalize(matrix, rank):
    """Orthonormal basis of the row space, taken from the SVD right factors.

    Dividing ``U^T @ matrix`` by the singular values would amplify numerically
    zero directions whenever the spectrum bottoms out (observed as row norms > 1
    on ETTh2-720 full-rank checkpoints), so use ``V^T`` rows directly.
    """
    _, s, vt = np.linalg.svd(matrix, full_matrices=False)
    tol = 1e-8 * max(float(s[0]), 1e-12)
    rr = min(rank, int(np.sum(s > tol)))
    if rr == 0:
        return None
    return vt[:rr]


def subspace_overlap(basis_a, basis_b, rank):
    rr = min(rank, basis_a.shape[0], basis_b.shape[0])
    cross = basis_a[:rr] @ basis_b[:rr].T
    return float((cross**2).sum() / rr)


def optimal_leading_direction(szz, szy):
    """Unit input functional of the optimal rank-1 map (the dominant direction)."""
    szz_r = szz + RIDGE * np.eye(szz.shape[0])
    ols = np.linalg.solve(szz_r, szy).T
    s_mat = szy.T @ np.linalg.solve(szz_r, szy)
    s_mat = 0.5 * (s_mat + s_mat.T)
    vals, vecs = np.linalg.eigh(s_mat)
    u1 = vecs[:, int(np.argmax(vals))]
    b1 = u1 @ ols
    return b1 / (np.linalg.norm(b1) + 1e-12)


def trained_map(state):
    if "weak_period_residual.linear.weight" in state:
        return state["weak_period_residual.linear.weight"].float().numpy(), None
    if "weak_period_residual.encoder.weight" in state:
        enc = state["weak_period_residual.encoder.weight"].float().numpy()
        dec = state["weak_period_residual.decoder.weight"].float().numpy()
        return dec @ enc, int(enc.shape[0])
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments-dir", default="research_runs/lowrank_data_property_v2")
    parser.add_argument("--runs-root", default="research_runs")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--max-seeds", type=int, default=3)
    args = parser.parse_args()

    moments = {}
    for path in sorted(glob.glob(os.path.join(args.moments_dir, "moments_*.npz"))):
        stem = os.path.basename(path).replace("moments_", "").replace(".npz", "")
        dataset, hs = stem.rsplit("_h", 1)
        moments[(dataset.lower(), int(hs))] = np.load(path)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_root, "**", "best.ckpt"), recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        match = RUN_RE.search(os.path.basename(run_dir))
        if not match or match.group("mechanism") != "weak_residual":
            continue
        key = (match.group("dataset"), int(match.group("horizon")))
        if key not in moments:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False).get("state_dict", {})
        weight, cfg_rank = trained_map(state)
        if weight is None:
            continue
        rank = int(cfg_rank) if cfg_rank else weight.shape[0]
        data = moments[key]

        star = orthonormalize(optimal_row_basis(data["szz"], data["szy"], rank), rank)
        got = orthonormalize(weight, rank)
        if star is None or got is None:
            continue
        b1 = optimal_leading_direction(data["szz"], data["szy"])
        # how much of the single dominant direction the trained row space reads
        dominant_cos = float(np.linalg.norm(got @ b1))
        rows.append(
            dict(
                dataset=match.group("dataset"),
                horizon=int(match.group("horizon")),
                seed=int(match.group("seed")),
                head="pooled_lowrank" if cfg_rank else "direct",
                rank=rank,
                subspace_overlap_r=round(subspace_overlap(got, star, rank), 4),
                dominant_direction_cos=round(dominant_cos, 4),
            )
        )

    if not rows:
        print("no matching checkpoints")
        return
    # Aggregate per (setting, head, rank)
    agg = {}
    for row in rows:
        k = (row["dataset"], row["horizon"], row["head"], row["rank"])
        agg.setdefault(k, []).append(row)
    print(f"{'setting':17}{'head':15}{'r':>5}{'n':>3}{'subspace_overlap':>17}{'|cos| with rank-1 dir':>23}")
    for k in sorted(agg, key=lambda k: (k[0], k[1], k[2], k[3])):
        vals = agg[k]
        print(
            f"{k[0] + '-' + str(k[1]):17}{k[2]:15}{k[3]:>5}{len(vals):>3}"
            f"{np.mean([v['subspace_overlap_r'] for v in vals]):>17.3f}"
            f"{np.mean([v['dominant_direction_cos'] for v in vals]):>23.3f}"
        )

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[written] {args.output_csv} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
