#!/usr/bin/env python3
"""Re-verify the reduced-rank-regression identity from saved second moments.

For ``Z = x - x_last`` and ``D = y - x_last`` in train-split-standardized units,

    E||D - W_r Z||^2  =  ( tr(Syy) - sum_{i<=r} lambda_i ) / H

where ``lambda`` are the eigenvalues of ``S = Szy^T Szz^{-1} Szy`` and ``W_r`` is
the optimal rank-r map (see ``scripts/analyze_optimal_lowrank_capture.py``).
This script recomputes the right-hand side from the ``moments_*.npz`` files
written by that script (``--save-moments``) and compares it against the
``train_mse`` column of its CSV output, so the reported capture/spectrum numbers
can be checked without re-accumulating moments or re-reading any dataset.

Usage (run inside the directory holding the moments and the CSV):

    python scripts/verify_optimal_rank_identity.py \
        --moments-dir research_runs/lowrank_data_property_v2 \
        --ranks 1,2,3,6
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

RIDGE = 1e-8  # same ridge as the analysis script; Szz is singular without it


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments-dir", default="research_runs/lowrank_data_property_v2")
    parser.add_argument("--ranks", default="1,2,3,6")
    parser.add_argument("--csv-name", default="optimal_rank_capture.csv")
    args = parser.parse_args()
    ranks = [int(x) for x in args.ranks.split(",") if x]

    csv_rows = {}
    with open(os.path.join(args.moments_dir, args.csv_name)) as handle:
        for row in csv.DictReader(handle):
            csv_rows[(row["dataset"], int(row["horizon"]), int(row["rank"]))] = float(
                row["train_mse"]
            )

    print("%-22s%4s%16s%16s%12s" % ("moments", "r", "identity MSE", "csv train_mse", "abs diff"))
    worst = 0.0
    checked = 0
    for path in sorted(glob.glob(os.path.join(args.moments_dir, "moments_*.npz"))):
        data = np.load(path)
        szz = data["szz"] + RIDGE * np.eye(data["szz"].shape[0])
        szy, syy = data["szy"], data["syy"]
        horizon = szy.shape[1]
        dataset = os.path.basename(path).replace("moments_", "").rsplit("_h", 1)[0]

        s_mat = szy.T @ np.linalg.solve(szz, szy)
        s_mat = 0.5 * (s_mat + s_mat.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(s_mat))[::-1]
        trace_syy = float(np.trace(syy))

        for rank in ranks:
            key = (dataset, horizon, rank)
            if rank > len(eigenvalues) or key not in csv_rows:
                continue
            identity = (trace_syy - float(eigenvalues[:rank].sum())) / horizon
            reference = csv_rows[key]
            diff = abs(identity - reference)
            worst = max(worst, diff)
            checked += 1
            print(
                "%-22s%4d%16.8f%16.8f%12.2e"
                % (f"{dataset}_h{horizon}", rank, identity, reference, diff)
            )
    print(f"checked {checked} cells; worst absolute difference {worst:.2e}")


if __name__ == "__main__":
    main()
