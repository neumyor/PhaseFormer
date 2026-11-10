#!/usr/bin/env python3
"""Low-dimensionality evidence of Section 3.2, from windows of every channel.

Periodicity. Windows are those of scripts/vldbj_m1/token_energy.py (test split,
length 720, stride 24, every channel), evenly subsampled to at most WINDOWS over all
channels of a dataset. A window is normalized to zero mean and unit variance and
folded into X (30 cycles x 24 phases); C = X - l 1^T is its periodic part. The 24
phase tokens of the window are the columns of C. For every window, the share of the
energy ||C||_F^2 in the leading r singular directions (r = 1, 2, 3) is computed and
averaged over the windows. Patch tokens are all length-24 patches of the window with
its cycle means removed, at every start; the same share is computed for them.
Patches aligned to the cycle are the rows of C and have the singular values of the
columns, so they are not a separate case.

Trend. Windows are those of the forecasting experiments (the horizon lies in the
test split, the input may begin before it; 720 steps followed by 720), starting at
every time step, evenly subsampled to at most TREND_WINDOWS per channel; every
channel. In window units, anchored at the last value x_L:
  level token   the 30 cycle means minus x_L;
  future level  the 30 cycle means of the next 720 steps minus x_L.
Per channel: share of the variance of the level token in its leading 3 principal
components over windows; share of the variance of the future level in the span of
the offset, tilt and curvature profiles and in its best 3 principal directions.
Channel results are averaged.

    python3 section3_dims.py --data-root <all_datasets> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vldbj_m1'))
from coupling import DATASETS, split_rows  # noqa: E402

P, K, L, H = 24, 30, 720, 720
STRIDE, WINDOWS, TREND_WINDOWS, CHUNK = 24, 4000, 600, 200


def energy_share(m):
    """Cumulative energy share (r = 1..3) of the leading singular directions of (n, a, b) matrices."""
    s2 = np.linalg.svd(m, compute_uv=False) ** 2
    return (np.cumsum(s2, 1) / s2.sum(1, keepdims=True))[:, :3]


def periodic_stats(values, lo, hi, rng):
    part = values[lo:hi]
    count = max(0, (len(part) - L) // STRIDE + 1)
    idx = np.arange(count)[:, None] * STRIDE + np.arange(L)[None]
    w = part[idx].transpose(2, 0, 1).reshape(-1, L)
    sd = w.std(1, keepdims=True)
    w = w[sd[:, 0] > 1e-8]
    if len(w) > WINDOWS:
        w = w[np.linspace(0, len(w) - 1, WINDOWS).astype(int)]
    w = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    phase, patch = [], []
    for a in range(0, len(w), CHUNK):
        x = w[a:a + CHUNK]
        folded = x.reshape(-1, K, P)
        c = folded - folded.mean(2, keepdims=True)
        phase.append(energy_share(c))
        wc = c.reshape(-1, L)
        starts = np.arange(L - P + 1)[:, None] + np.arange(P)[None]
        patch.append(energy_share(wc[:, starts]))
    phase, patch = np.concatenate(phase), np.concatenate(patch)
    return {'windows': int(len(w)), 'phase': phase.mean(0).tolist(), 'patch': patch.mean(0).tolist(),
            'phase_median': np.median(phase, 0).tolist(), 'patch_median': np.median(patch, 0).tolist()}


def pca_share3(z):
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False) ** 2
    return float(s[:3].sum() / s.sum())


def span_share(y, basis):
    y = y - y.mean(0)
    q, _ = np.linalg.qr(basis)
    return float(((y @ q) ** 2).sum() / (y ** 2).sum())


def trend_stats(values, lo, hi):
    starts = np.arange(max(0, lo - L), hi - L - H + 1)
    if len(starts) > TREND_WINDOWS:
        starts = starts[np.linspace(0, len(starts) - 1, TREND_WINDOWS).astype(int)]
    tau = np.linspace(-1, 1, K)
    profiles = np.stack([np.ones(K), tau, tau ** 2 - (tau ** 2).mean()], 1)
    level3, prof, best3 = [], [], []
    for c in range(values.shape[1]):
        s = values[:, c]
        w = np.stack([s[i:i + L] for i in starts]); f = np.stack([s[i + L:i + L + H] for i in starts])
        mu, sd = w.mean(1, keepdims=True), w.std(1, keepdims=True)
        ok = sd[:, 0] > 1e-8
        if ok.sum() < 50:
            continue
        w, f, mu, sd = w[ok], f[ok], mu[ok], sd[ok]
        wn = (w - mu) / sd
        last = wn[:, -1:]
        level = wn.reshape(-1, K, P).mean(2) - last
        future = ((f - mu) / sd).reshape(-1, K, P).mean(2) - last
        level3.append(pca_share3(level)); prof.append(span_share(future, profiles)); best3.append(pca_share3(future))
    return {'channels': len(level3), 'windows_per_channel': int(len(starts)),
            'level_token_top3': float(np.mean(level3)), 'future_in_profiles': float(np.mean(prof)),
            'future_best3': float(np.mean(best3)), 'level_token_top3_min': float(np.min(level3)),
            'future_in_profiles_median': float(np.median(prof))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for name, (folder, file, _) in DATASETS.items():
        frame = pd.read_csv(args.data_root / folder / file)
        values = frame.drop(columns=[frame.columns[0]]).to_numpy('float64')
        lo, hi = split_rows(name, len(values))['test']
        out[name] = {'periodic': periodic_stats(values, lo, hi, rng), 'trend': trend_stats(values, lo, hi)}
        p, t = out[name]['periodic'], out[name]['trend']
        print(f"{name:12s} phase r1/r2/r3 {p['phase'][0]:.2f}/{p['phase'][1]:.2f}/{p['phase'][2]:.2f}"
              f" patch {p['patch'][0]:.2f}/{p['patch'][1]:.2f}/{p['patch'][2]:.2f} ({p['windows']} win)"
              f" | level token top3 {t['level_token_top3']:.2f} | future in profiles {t['future_in_profiles']:.2f}"
              f" best3 {t['future_best3']:.2f} ({t['channels']} ch)", flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
