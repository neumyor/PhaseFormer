#!/usr/bin/env python3
"""Alignment of the trained level branch with the data's level subspace.

Reads the tensors written by ``level_pca_extract`` and, per dataset, measures
  spectral     cumulative PCA curves of the level token, the branch forecast and
               the level demand, all in the 30 pooled bins
  captured     share of the level demand's variance inside
                 the branch's three horizon directions,
                 the best three PCA directions,
                 the three fixed offset/tilt/curvature profiles
               for the full demand and for its predictable part (ridge from the
               level token, fitted on the first 70% of windows)
  angles       principal angles (degrees) between the branch subspace and the
               top three PCA directions of the predictable demand
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 12


def cum_curves(z):
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False) ** 2
    ratio = s / s.sum()
    cum = np.cumsum(ratio)
    r = np.pad(ratio[:K], (0, K - len(ratio[:K])))
    return r, np.pad(cum[:K], (0, K - K and K - len(cum[:K])))


def captured(demand, basis):
    """Variance share of ``demand`` inside the span of ``basis`` (columns)."""
    y = demand - demand.mean(0)
    q, _ = np.linalg.qr(basis)
    proj = y @ q @ q.T
    return (proj ** 2).sum() / (y ** 2).sum()


def top_dirs(y, k=3):
    y = y - y.mean(0)
    u, s, vt = np.linalg.svd(y, full_matrices=False)
    return vt[:k].T, s[:k] ** 2 / (s ** 2).sum()


def principal_angles(a, b):
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def legendre_basis(bins):
    tau = np.linspace(-1, 1, bins)
    cols = [tau ** p for p in range(3)]
    out = [np.ones_like(tau), tau, tau ** 2 - np.mean(tau ** 2)]
    return np.stack(out, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default=ROOT / 'research_runs/level_pca_v1')
    args = parser.parse_args()
    summary = {}
    for path in sorted(args.dir.glob('*_H720.npz')):
        d = np.load(path, allow_pickle=True)
        name = str(d['dataset'])
        zeta, pred, demand, basis = d['zeta'], d['pred'], d['demand'], d['basis'].T  # (30, r)
        bins = zeta.shape[-1]
        leg = legendre_basis(bins)
        rows = []
        for c in range(zeta.shape[1]):
            z, p, y = zeta[:, c], pred[:, c], demand[:, c]
            n = len(z)
            cut = int(0.7 * n)
            X, Y = z[:cut], y[:cut]
            mx, my = X.mean(0), Y.mean(0)
            Xc = X - mx
            reg = 1e-3 * np.eye(X.shape[1]) * (Xc.T @ Xc).trace() / X.shape[1]
            W = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ (Y - my))
            ypred = (z[cut:] - mx) @ W + my
            yfull = y[cut:]
            v_b = captured(yfull, basis)
            pc3, share = top_dirs(yfull, 3)
            v_p = share.sum()
            v_l = captured(yfull, leg)
            pb3, pshare = top_dirs(ypred, 3)
            vb3 = captured(ypred, basis)
            angles = principal_angles(basis, pb3)
            rows.append([v_b, v_p, v_l, vb3, pshare.sum(), captured(ypred, leg),
                         *angles, np.corrcoef(p, y)[0, 1]])
        rows = np.array(rows)
        rz, cz = cum_curves(zeta[:, 0])
        curves = {}
        for key, arr in (('zeta', zeta), ('pred', pred), ('demand', demand)):
            rr = [cum_curves(arr[:, c])[0] for c in range(arr.shape[1])]
            ratio = np.mean(rr, 0)
            curves[key] = {'ratio': ratio.tolist(), 'cum': np.cumsum(ratio).tolist()}
        summary[name] = {
            'run': str(d['run']), 'period': int(d['period']), 'rank': int(d['rank']),
            'bins': int(bins), 'channels': int(zeta.shape[1]), 'windows': int(zeta.shape[0]),
            'curves': curves,
            'demand_branch': float(np.median(rows[:, 0])),
            'demand_pca3': float(np.median(rows[:, 1])),
            'demand_leg3': float(np.median(rows[:, 2])),
            'pred_branch': float(np.median(rows[:, 3])),
            'pred_pca3': float(np.median(rows[:, 4])),
            'pred_leg3': float(np.median(rows[:, 5])),
            'angles_pred_pca3': np.median(rows[:, 6:9], 0).tolist(),
            'corr_pred_demand': float(np.median(rows[:, 9])),
            'demand_branch_iqr': np.percentile(rows[:, 0], [25, 75]).tolist(),
        }
        print(f"{name:12} branch3 {rows[:,0].mean():.3f} pca3 {rows[:,1].mean():.3f} "
              f"leg3 {rows[:,2].mean():.3f} | pred: branch {rows[:,3].mean():.3f} "
              f"pca3 {rows[:,4].mean():.3f} | corr {rows[:,9].mean():.3f}")
    (args.dir / 'model_summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
