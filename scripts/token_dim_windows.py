#!/usr/bin/env python3
"""Dimensionality of the tokens a window yields, from window-sampled data.

Windows (length L=720, horizon H=720) start every STRIDE=7 steps, which is coprime to
the cycle length, so the start of a window falls on every phase of the cycle
as it does in training. Windows come from the test split (the inputs may begin
before it, as in the forecasting loaders). Each window is normalized to zero mean and
unit variance and folded into X (K=30 cycles x P=24 phases); C = X - l 1^T is its
periodic part, l the 30 cycle means.

The tokens are those the models read:
  phase token  a column of X (or C), a vector in R^30;
  patch token  a row of X (or C), a vector in R^24 (non-overlapping patches).
Tokens of all windows are pooled and PCA (centered) gives the cumulative explained
variance of the leading components. 'channel' averages the per-channel curves,
'shared' pools the tokens of all sampled channels, as a channel-independent model
shared by the channels reads them.

Trend: l (the 30 cycle means in window units) and the future cycle means m
(the 30 means of the next 720 steps in the same units); share of the energy of m
(uncentered) in the span of the offset, tilt and curvature profiles, PCA curves of
l and m.

    python3 token_dim_windows.py --data-root <all_datasets> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = {'ETTh1': 'ETT/ETTh1.csv', 'ETTh2': 'ETT/ETTh2.csv', 'ETTm1': 'ETT/ETTm1.csv',
        'ETTm2': 'ETT/ETTm2.csv', 'Weather': 'weather/weather.csv',
        'Electricity': 'electricity/electricity.csv', 'Traffic': 'traffic/traffic.csv'}
P, K, L, H = 24, 30, 720, 720
STRIDE, WINDOWS, CHANNELS, R = 7, 600, 16, 8


def test_bounds(name, n):
    if name.startswith('ETT'):
        month = 30 * 24 * (4 if name.startswith('ETTm') else 1)
        return 16 * month, 20 * month
    return n - int(0.2 * n), n


def window_starts(name, n):
    lo, hi = test_bounds(name, n)
    s = np.arange(max(0, lo - L), hi - L - H + 1, STRIDE)
    if len(s) > WINDOWS:
        s = s[np.linspace(0, len(s) - 1, WINDOWS).astype(int)]
    return s


class Gram:
    def __init__(self, d):
        self.s2, self.s1, self.n = np.zeros((d, d)), np.zeros(d), 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).astype('float64')
        self.s2 += x.T @ x
        self.s1 += x.sum(0)
        self.n += len(x)

    def merge(self, other):
        self.s2 += other.s2; self.s1 += other.s1; self.n += other.n

    def cum(self):
        c = self.s2 - np.outer(self.s1, self.s1) / self.n
        w = np.clip(np.linalg.eigvalsh(c)[::-1], 0, None)
        return (np.cumsum(w) / w.sum())[:R]

    def energy_in(self, basis):
        q, _ = np.linalg.qr(basis)
        return float(np.trace(q.T @ self.s2 @ q) / np.trace(self.s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    tau = np.linspace(-1, 1, K)
    profiles = np.stack([np.ones(K), tau, tau ** 2 - (tau ** 2).mean()], 1)
    kinds = (('phase', K), ('patch', P), ('phase_periodic', K), ('patch_periodic', P), ('level', K), ('future', K))
    out = {}
    for name, rel in DATA.items():
        v = pd.read_csv(args.data_root / rel).iloc[:, 1:].to_numpy('float64')
        chans = np.arange(v.shape[1]) if v.shape[1] <= CHANNELS else np.sort(rng.choice(v.shape[1], CHANNELS, replace=False))
        st = window_starts(name, len(v))
        shared = {k: Gram(d) for k, d in kinds}
        per_channel = {k: [] for k, _ in kinds}
        for c in chans:
            s = v[:, c]
            w = np.stack([s[i:i + L] for i in st]); f = np.stack([s[i + L:i + L + H] for i in st])
            mu, sd = w.mean(1, keepdims=True), w.std(1, keepdims=True)
            ok = sd[:, 0] > 1e-8
            w, f, mu, sd = w[ok], f[ok], mu[ok], sd[ok]
            X = ((w - mu) / sd).reshape(-1, K, P)
            lev = X.mean(2, keepdims=True)
            C = X - lev
            fut = ((f - mu) / sd).reshape(-1, K, P).mean(2)
            g = {k: Gram(d) for k, d in kinds}
            g['phase'].add(X.transpose(0, 2, 1)); g['patch'].add(X)
            g['phase_periodic'].add(C.transpose(0, 2, 1)); g['patch_periodic'].add(C)
            g['level'].add(lev[..., 0]); g['future'].add(fut)
            for k, _ in kinds:
                shared[k].merge(g[k]); per_channel[k].append(g[k].cum())
        res = {'windows_per_channel': int(len(st)), 'channels': int(len(chans)),
               'shared': {k: shared[k].cum().tolist() for k, _ in kinds},
               'channel': {k: np.mean(a, 0).tolist() for k, a in per_channel.items()},
               'future_in_profiles': shared['future'].energy_in(profiles),
               'future_in_profiles_channel': None}
        out[name] = res
        sh = res['shared']
        print(f"{name:12s} windows {len(st):3d}x{len(chans):2d} | shared top2/top3: periodic phase {sh['phase_periodic'][1]:.2f}/{sh['phase_periodic'][2]:.2f}"
              f" patch {sh['patch_periodic'][1]:.2f}/{sh['patch_periodic'][2]:.2f} | full phase {sh['phase'][1]:.2f} patch {sh['patch'][1]:.2f}"
              f" | level top3 {sh['level'][2]:.2f} future top3 {sh['future'][2]:.2f} profiles {res['future_in_profiles']:.2f}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
