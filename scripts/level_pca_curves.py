#!/usr/bin/env python3
"""PCA curves for the five objects of the level study, on the model's own test windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 12


def curve(z):
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False) ** 2
    ratio = s / s.sum()
    return np.pad(ratio[:K], (0, K - len(ratio[:K])))


def ridge(z, y):
    n = len(z)
    cut = int(0.7 * n)
    X, Y = z[:cut], y[:cut]
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    reg = 1e-3 * np.eye(X.shape[1]) * (Xc.T @ Xc).trace() / X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ (Y - my))
    pred = (z[cut:] - mx) @ W + my
    r2 = 1 - ((y[cut:] - pred) ** 2).sum() / ((y[cut:] - y[cut:].mean(0)) ** 2).sum()
    return pred, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default=ROOT / 'research_runs/level_pca_v1')
    args = parser.parse_args()
    out = {}
    for path in sorted(args.dir.glob('*_H720.npz')):
        d = np.load(path, allow_pickle=True)
        name = str(d['dataset'])
        if 'value' not in d.files:
            print(f'{name}: no value token yet, skipped')
            continue
        zeta, value, demand, pred = d['zeta'], d['value'], d['demand'], d['pred']
        curves, r2s = {k: [] for k in ('level', 'value', 'demand', 'ridge', 'branch')}, []
        for c in range(zeta.shape[1]):
            rp, r2 = ridge(zeta[:, c], demand[:, c])
            r2s.append(r2)
            for key, arr in (('level', zeta[:, c]), ('value', value[:, c]), ('demand', demand[:, c]),
                             ('ridge', rp), ('branch', pred[:, c])):
                curves[key].append(curve(arr))
        res = {}
        for key, rows in curves.items():
            ratio = np.mean(rows, 0)
            cum = np.cumsum(ratio)
            res[key] = {'ratio': ratio.tolist(), 'cum': cum.tolist(),
                        'n90': int(np.searchsorted(cum, .9) + 1) if cum.max() >= .9 else None,
                        'n95': int(np.searchsorted(cum, .95) + 1) if cum.max() >= .95 else None}
        res['r2_median'] = float(np.median(r2s))
        res['r2_mean'] = float(np.mean(r2s))
        res['run'] = str(d['run'])
        out[name] = res
        print(f"{name:12} " + ' | '.join(f"{k} @3 {res[k]['cum'][2]:.3f} n90 {res[k]['n90']}"
                                         for k in ('level', 'value', 'ridge', 'demand')) +
              f" | R2 med {res['r2_median']:.2f}")
    (args.dir / 'curves.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
