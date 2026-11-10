#!/usr/bin/env python3
"""PCA of the level token, the value token and the level demand, in the model's convention.

Per dataset (H=720, lookback 720, P=24, the level head's pooling):
  level token    pooled cycle means of the centered history, (30,)
  value token    the centered history sampled every P steps, (30,)
  level demand   the future cycle means minus the last observed value, (30,)
and the predictable part of the level demand: a ridge map from the level token,
fitted on the first 70% of the windows, applied to the rest.

Writes mean cumulative curves per dataset to research_runs/level_pca_v1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = {'ETTh1': 'ETT/ETTh1.csv', 'ETTh2': 'ETT/ETTh2.csv', 'ETTm1': 'ETT/ETTm1.csv',
        'ETTm2': 'ETT/ETTm2.csv', 'Weather': 'weather/weather.csv',
        'Electricity': 'electricity/electricity.csv', 'Traffic': 'traffic/traffic.csv'}
P, L, H, STRIDE, K = 24, 720, 720, 6, 12


def cumcurve(z, k=K):
    p = PCA(n_components=min(k, min(z.shape) - 1)).fit(z)
    ratio = p.explained_variance_ratio_
    return np.pad(ratio, (0, k - len(ratio))), np.pad(np.cumsum(ratio), (0, k - len(ratio)))


def ridge_predict(X, Y):
    n = len(X)
    cut = int(0.7 * n)
    Xtr, Ytr = X[:cut], Y[:cut]
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc = Xtr - mx
    reg = 1e-3 * np.eye(X.shape[1]) * (Xc.T @ Xc).trace() / X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ (Ytr - my))
    return (X[cut:] - mx) @ W + my


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'research_runs/level_pca_v1')
    parser.add_argument('--channels', type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, rel in DATA.items():
        v = pd.read_csv(args.data_root / rel).iloc[:, 1:].to_numpy('float32')
        chans = np.arange(v.shape[1]) if v.shape[1] <= args.channels else \
            np.sort(rng.choice(v.shape[1], args.channels, replace=False))
        acc = {k: [] for k in ('level_token', 'value_token', 'level_demand', 'level_pred')}
        for c in chans:
            s = v[:, c].astype('float64')
            starts = np.arange(0, len(s) - L - H, STRIDE)
            hist = np.stack([s[t:t+L] for t in starts])
            fut = np.stack([s[t+L:t+L+H] for t in starts])
            sd = hist.std(1)[:, None] + 1e-8
            last = hist[:, -1:]
            hc = hist.reshape(len(starts), -1, P).mean(-1)
            fc = fut.reshape(len(starts), -1, P).mean(-1)
            lvl_tok = (hc - last) / sd
            val_tok = (hist[:, ::P] - last) / sd
            lvl_tgt = (fc - last) / sd
            pred = ridge_predict(lvl_tok, lvl_tgt)
            for key, z in (('level_token', lvl_tok), ('value_token', val_tok),
                           ('level_demand', lvl_tgt), ('level_pred', pred)):
                acc[key].append(cumcurve(z))
        res = {}
        for key, chunks in acc.items():
            ratio = np.stack([a for a, _ in chunks]).mean(0)
            cum = np.cumsum(ratio)
            res[key] = {'ratio': ratio.tolist(), 'cum': cum.tolist(),
                        'n90': int(np.searchsorted(cum, .9) + 1) if cum.max() >= .9 else None,
                        'n95': int(np.searchsorted(cum, .95) + 1) if cum.max() >= .95 else None,
                        'n_channels': len(chunks)}
        (args.out / f'tokens_{name}.json').write_text(json.dumps(res, indent=1))
        print(name, {k: (round(v['cum'][2], 3), v['n90']) for k, v in res.items()})


if __name__ == '__main__':
    main()
