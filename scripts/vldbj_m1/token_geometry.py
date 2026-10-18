#!/usr/bin/env python3
"""Fig. 3 of Section 3.2: drift and dimensionality of patch and phase tokens.

For each dataset, a set of channels is z-normalized over the whole series and
cut into tokens of equal length P (one cycle), as a model sees them: input
windows start at any time step, so tokens start at any phase.
  patch token at start t: the P consecutive values x[t:t+P];
  phase token at start t: the P values one cycle apart,
                          x[t], x[t+P], ..., x[t+(P-1)P].
A cycle-aligned variant (patch tokens = rows of the fold only, x[kP:(k+1)P])
was also run on 2026-10-15: drift MMD^2 Traffic patch 0.390 vs phase 0.064,
Electricity 0.648 vs 0.184; two-PC share Traffic patch 0.781 vs phase 0.736,
Electricity 0.819 vs 0.791 (aligned patch tokens are the more compact there).
Tokens are grouped by the week of their start.

Drift: maximum mean discrepancy (RBF kernel, median-heuristic bandwidth)
between the tokens of each later week and those of the first week.
Dimensionality: cumulative explained-variance ratio of a PCA of all tokens.
Both are averaged over the channels.

Usage:
  python scripts/vldbj_m1/token_geometry.py <all_datasets> \
      --output research_runs/vldbj_token_geometry
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PERIOD = 24
DATASETS = {
    'Traffic': 'traffic/traffic.csv',
    'Electricity': 'electricity/electricity.csv',
}
CHANNELS = 16
WEEK_STEP = 2          # every second week enters the drift curve
SUB = 3                # every third token of a week enters the MMD


def mmd2(x, y, gamma):
    def k(a, b):
        return np.exp(-gamma * ((a[:, None] - b[None]) ** 2).sum(-1))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


def tokens(series):
    p = PERIOD
    n = len(series) - (p - 1) * p
    starts = np.arange(n)
    phase = series[starts[:, None] + p * np.arange(p)[None]]
    patch = series[starts[:, None] + np.arange(p)[None]]
    week = starts // (7 * p)
    return {'patch': patch, 'phase': phase}, {'patch': week, 'phase': week}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('--output', default='research_runs/vldbj_token_geometry')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    drift_rows, pca_rows = [], []
    for name, rel in DATASETS.items():
        values = pd.read_csv(Path(args.data_root) / rel).iloc[:, 1:].to_numpy('float64')
        channels = rng.choice(values.shape[1], CHANNELS, replace=False)
        for c in channels:
            s = values[:, c]
            s = (s - s.mean()) / (s.std() + 1e-8)
            toks, weeks_of = tokens(s)
            weeks = np.unique(weeks_of['phase'])[1:-1:WEEK_STEP]
            for kind, z in toks.items():
                week = weeks_of[kind]
                sub = SUB
                ratio = np.cumsum(PCA().fit(z).explained_variance_ratio_)
                for j, r in enumerate(ratio, 1):
                    pca_rows.append((name, int(c), kind, j, r))
                probe = z[::max(1, len(z) // 400)]
                gamma = 1.0 / np.median(((probe[:, None] - probe[None]) ** 2).sum(-1))
                base = z[week == 0][::sub]
                for w in weeks:
                    drift_rows.append((name, int(c), kind, int(w),
                                       mmd2(base, z[week == w][::sub], gamma)))
        print(name, 'done')
    drift = pd.DataFrame(drift_rows, columns=['dataset', 'channel', 'token', 'week', 'mmd2'])
    pca = pd.DataFrame(pca_rows, columns=['dataset', 'channel', 'token', 'component', 'cumulative'])
    drift.to_csv(out / 'drift.csv', index=False)
    pca.to_csv(out / 'pca.csv', index=False)
    summary = []
    for name in DATASETS:
        for kind in ('patch', 'phase'):
            d = drift[(drift.dataset == name) & (drift.token == kind)]
            p = pca[(pca.dataset == name) & (pca.token == kind)]
            curve = p.groupby('component').cumulative.mean()
            mmd = np.sqrt(np.clip(d.mmd2, 0, None))
            summary.append((name, kind, d.mmd2.mean(), mmd.mean(), curve.loc[2],
                            int((curve < 0.9).sum() + 1)))
    summary = pd.DataFrame(summary, columns=['dataset', 'token', 'mean_mmd2', 'mean_mmd',
                                             'top2', 'k90'])
    summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
