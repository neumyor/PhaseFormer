#!/usr/bin/env python3
"""PCA curves of Section 3.2: cumulative share of variance in the leading r directions.

Windows, normalization and folding are those of scripts/section3_dims.py (test split, length
720, 30 cycles x 24 phases, every channel). Cumulative shares for r = 1..R:

  phase   periodic part C = X - l 1^T of a window; its 24 phase tokens (columns of C, which sum to
          zero, so the SVD of C is the PCA of the tokens); averaged over windows (stride 24,
          at most WINDOWS per dataset, evenly spaced over all channels).
  patch   all length-24 patches of the periodic part of a window, at every start; same average.
  level   level token of a window (the 30 cycle means minus the last value x_L, window units);
          PCA over the windows of one channel (at most TREND_WINDOWS, horizon inside the test
          split), averaged over channels.

    python3 section3_pca_curves.py --data-root <all_datasets> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    'ETTh1': ('ETT', 'ETTh1.csv'), 'ETTh2': ('ETT', 'ETTh2.csv'), 'ETTm1': ('ETT', 'ETTm1.csv'),
    'ETTm2': ('ETT', 'ETTm2.csv'), 'Weather': ('weather', 'weather.csv'),
    'Electricity': ('electricity', 'electricity.csv'), 'Traffic': ('traffic', 'traffic.csv'),
}
P, K, L, H, R = 24, 30, 720, 720, 10
STRIDE, WINDOWS, TREND_WINDOWS, CHUNK = 24, 4000, 600, 200


def split_test(name, length):
    if name.startswith('ETT'):
        month = 30 * 24 * (4 if name.startswith('ETTm') else 1)
        return 16 * month, 20 * month
    return length - int(0.2 * length), length


def energy_curve(m):
    """Cumulative energy share (r = 1..R) of the leading singular directions of (n, a, b) matrices."""
    s2 = np.linalg.svd(m, compute_uv=False) ** 2
    return (np.cumsum(s2, 1) / s2.sum(1, keepdims=True))[:, :R]


def periodic_curves(values, lo, hi):
    part = values[lo:hi]
    count = max(0, (len(part) - L) // STRIDE + 1)
    idx = np.arange(count)[:, None] * STRIDE + np.arange(L)[None]
    w = part[idx].transpose(2, 0, 1).reshape(-1, L)
    w = w[w.std(1) > 1e-8]
    if len(w) > WINDOWS:
        w = w[np.linspace(0, len(w) - 1, WINDOWS).astype(int)]
    w = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    phase, patch = [], []
    starts = np.arange(L - P + 1)[:, None] + np.arange(P)[None]
    for a in range(0, len(w), CHUNK):
        folded = w[a:a + CHUNK].reshape(-1, K, P)
        c = folded - folded.mean(2, keepdims=True)
        phase.append(energy_curve(c))
        patch.append(energy_curve(c.reshape(-1, L)[:, starts]))
    phase, patch = np.concatenate(phase), np.concatenate(patch)
    return {'windows': int(len(w)), 'phase': phase.mean(0).tolist(), 'patch': patch.mean(0).tolist()}


def pca_curve(z):
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False) ** 2
    return np.cumsum(s)[:R] / s.sum()


def level_curve(values, lo, hi):
    starts = np.arange(max(0, lo - L), hi - L - H + 1)
    if len(starts) > TREND_WINDOWS:
        starts = starts[np.linspace(0, len(starts) - 1, TREND_WINDOWS).astype(int)]
    curves = []
    for c in range(values.shape[1]):
        s = values[:, c]
        w = np.stack([s[i:i + L] for i in starts])
        mu, sd = w.mean(1, keepdims=True), w.std(1, keepdims=True)
        ok = sd[:, 0] > 1e-8
        if ok.sum() < 50:
            continue
        wn = (w[ok] - mu[ok]) / sd[ok]
        level = wn.reshape(-1, K, P).mean(2) - wn[:, -1:]
        curves.append(pca_curve(level))
    return {'channels': len(curves), 'level': np.mean(curves, 0).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for name, (folder, file) in DATASETS.items():
        frame = pd.read_csv(args.data_root / folder / file)
        values = frame.drop(columns=[frame.columns[0]]).to_numpy('float64')
        lo, hi = split_test(name, len(values))
        out[name] = {**periodic_curves(values, lo, hi), **level_curve(values, lo, hi)}
        o = out[name]
        print(f"{name:12s} phase {o['phase'][1]:.3f}/{o['phase'][2]:.3f} patch {o['patch'][1]:.3f}/{o['patch'][2]:.3f}"
              f" level top3 {o['level'][2]:.3f} ({o['windows']} windows, {o['channels']} channels)", flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
