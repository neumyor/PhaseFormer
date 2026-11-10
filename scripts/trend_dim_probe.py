#!/usr/bin/env python3
"""Dimensionality of the trend and of the periodic part of windows.

Windows lie inside the test split (as in token_energy.py: length 720, stride S),
every channel contributes its own windows and all sampled channels are pooled
(one model reads them all). A window is normalized to zero mean, unit variance
and folded into X (30 cycles x 24 phases).

periodic  per-window share of the energy of the periodic part C = X - l 1^T (and of X)
          in the leading r singular directions of the window, averaged over windows.
          The 24 phase tokens of a window are the columns of C.
trend     the level token l (30 cycle means) in four anchorings, PCA over windows
          (centered), per channel then averaged, and pooled over channels:
            raw        l
            lastcycle  l - l[-1]
            lastvalue  l - x[-1]          (the model's level token)
          and the future level m (30 cycle means of the next 720 steps, window
          units) anchored the same way, with the share of its energy (uncentered)
          in the offset/tilt/curvature profiles and in its best 3 directions.
Start offsets: 'fixed' windows start at the same hour (stride 24), 'any' at every
hour (stride 1, subsampled).

    python3 trend_dim_probe.py --data-root <all_datasets> --out <file.json>
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
P, K, L, H, R = 24, 30, 720, 720, 8


def test_bounds(name, n):
    if name.startswith('ETT'):
        month = 30 * 24 * (4 if name.startswith('ETTm') else 1)
        return 16 * month, 20 * month
    return n - int(0.2 * n), n


def starts(name, n, mode, cap=900):
    lo, hi = test_bounds(name, n)
    s = np.arange(max(0, lo - L), hi - L - H + 1, P if mode == 'fixed' else 1)
    if len(s) > cap:
        s = s[np.linspace(0, len(s) - 1, cap).astype(int)]
    return s


def cum_energy(m, center=False):
    if center:
        m = m - m.mean(-2, keepdims=True)
    s2 = np.linalg.svd(m, compute_uv=False) ** 2
    return (np.cumsum(s2, -1) / s2.sum(-1, keepdims=True))[..., :R]


def pca_cum(z):
    z = z - z.mean(0)
    w = np.linalg.svd(z, compute_uv=False) ** 2
    return (np.cumsum(w) / w.sum())[:R]


def share_in(y, basis):
    q, _ = np.linalg.qr(basis)
    return float(((y @ q) ** 2).sum() / (y ** 2).sum())


def best3(y):
    w = np.linalg.svd(y, compute_uv=False) ** 2
    return float(w[:3].sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--channels', type=int, default=16)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    tau = np.linspace(-1, 1, K)
    prof = np.stack([np.ones(K), tau, tau ** 2 - (tau ** 2).mean()], 1)
    out = {}
    for name, rel in DATA.items():
        v = pd.read_csv(args.data_root / rel).iloc[:, 1:].to_numpy('float64')
        chans = np.arange(v.shape[1]) if v.shape[1] <= args.channels else np.sort(rng.choice(v.shape[1], args.channels, replace=False))
        res = {}
        for mode in ('fixed', 'any'):
            st = starts(name, len(v), mode)
            per = {k: [] for k in ('X', 'C')}
            tr = {a: {'l': [], 'm': []} for a in ('raw', 'lastcycle', 'lastvalue')}
            tr_chan = {a: {'l': [], 'm': []} for a in tr}
            prof_share = {a: [] for a in tr}; b3 = {a: [] for a in tr}
            for c in chans:
                s = v[:, c]
                w = np.stack([s[i:i + L] for i in st]); f = np.stack([s[i + L:i + L + H] for i in st])
                mu, sd = w.mean(1, keepdims=True), w.std(1, keepdims=True)
                ok = sd[:, 0] > 1e-8
                w, f, mu, sd = w[ok], f[ok], mu[ok], sd[ok]
                wn = (w - mu) / sd
                X = wn.reshape(-1, K, P); lev = X.mean(2)
                C = X - lev[..., None]
                m = ((f - mu) / sd).reshape(-1, K, P).mean(2)
                per['X'].append(cum_energy(X)); per['C'].append(cum_energy(C))
                anch = {'raw': np.zeros((len(wn), 1)), 'lastcycle': lev[:, -1:], 'lastvalue': wn[:, -1:]}
                for a, off in anch.items():
                    l_a, m_a = lev - off, m - off
                    tr[a]['l'].append(l_a); tr[a]['m'].append(m_a)
                    tr_chan[a]['l'].append(pca_cum(l_a)); tr_chan[a]['m'].append(pca_cum(m_a))
                    prof_share[a].append(share_in(m_a, prof)); b3[a].append(best3(m_a))
            res[mode] = {
                'windows_per_channel': int(len(st)), 'channels': int(len(chans)),
                'X_energy': np.concatenate(per['X']).mean(0).tolist(), 'C_energy': np.concatenate(per['C']).mean(0).tolist(),
                'trend': {a: {'level_channel': np.mean(tr_chan[a]['l'], 0).tolist(),
                              'level_pooled': pca_cum(np.concatenate(tr[a]['l'])).tolist(),
                              'future_channel': np.mean(tr_chan[a]['m'], 0).tolist(),
                              'future_pooled': pca_cum(np.concatenate(tr[a]['m'])).tolist(),
                              'future_in_profiles_mean': float(np.mean(prof_share[a])),
                              'future_in_profiles_median': float(np.median(prof_share[a])),
                              'future_best3_mean': float(np.mean(b3[a]))} for a in tr}}
        out[name] = res
        f_, a_ = res['fixed'], res['any']
        print(f"{name:12s} X top2/3 {f_['X_energy'][1]:.2f}/{f_['X_energy'][2]:.2f} | C top1/2/3 {f_['C_energy'][0]:.2f}/{f_['C_energy'][1]:.2f}/{f_['C_energy'][2]:.2f}"
              f" || level top3 (ch avg) fixed: raw {f_['trend']['raw']['level_channel'][2]:.2f} cyc {f_['trend']['lastcycle']['level_channel'][2]:.2f} val {f_['trend']['lastvalue']['level_channel'][2]:.2f}"
              f" | any: val {a_['trend']['lastvalue']['level_channel'][2]:.2f}"
              f" || future in profiles fixed: raw {f_['trend']['raw']['future_in_profiles_mean']:.2f} cyc {f_['trend']['lastcycle']['future_in_profiles_mean']:.2f} val {f_['trend']['lastvalue']['future_in_profiles_mean']:.2f} | any val {a_['trend']['lastvalue']['future_in_profiles_mean']:.2f}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
