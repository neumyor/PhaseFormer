#!/usr/bin/env python3
"""V2: level-amplitude coupling of the benchmark datasets (read-only).

For every dataset split, channel and length-720 window (stride 24, 30 cycles
of 24 samples), compute the Pearson correlation between the 30 cycle means and
the 30 within-cycle deviation RMS values, plus the normalized regression
slope of amplitude on level.

Splits follow the forecasting experiments: the 12/4/4-month borders on ETT
and 70/10/20% on the other datasets.  (Version 1 split the whole ETT files
60/20/20%; its output is kept as coupling_fullfile_split_v1.csv.)

Besides the mean |corr|, each split reports the sign-consistent coupling:
the correlation is averaged over the windows of each channel, keeping its
sign, and the magnitudes of the channel averages are averaged over the
channels.  Chance correlations of either sign cancel in the channel average.
Its reference, the shifted pairing, computes the same statistic after pairing
the cycle means of every window with the amplitudes of the window half the
split later (circularly), which keeps the drift of both series but removes
their alignment.  This measures the multiplicative-seasonality
content that M1's modulation is designed to carry; a purely additive
seasonal signal gives both statistics zero.

Anchors: a synthetic multiplicative series with known coupling and an
additive-only null are appended for scale.

Usage:
  python scripts/vldbj_m1/coupling.py --output research_runs/vldbj_m1_coupling
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PERIOD = 24
LOOKBACK = 720
STRIDE = 24
CORR_THRESHOLD = 0.3

# (root relative to resources/all_datasets, file, value columns, train fraction)
DATASETS = {
    'ETTh1': ('ETT', 'ETTh1.csv', 0.6),
    'ETTh2': ('ETT', 'ETTh2.csv', 0.6),
    'ETTm1': ('ETT', 'ETTm1.csv', 0.6),
    'ETTm2': ('ETT', 'ETTm2.csv', 0.6),
    'Weather': ('weather', 'weather.csv', 0.7),
    'Electricity': ('electricity', 'electricity.csv', 0.7),
    'Traffic': ('traffic', 'traffic.csv', 0.7),
}


ETT_MONTH = 30 * 24  # rows per month of the hourly ETT files


def split_rows(name, length):
    """Row borders (train, val, test) of the forecasting experiments."""
    if name.startswith('ETT'):
        month = ETT_MONTH * (4 if name.startswith('ETTm') else 1)
        b = [0, 12 * month, 16 * month, 20 * month]
    else:
        train, test = int(0.7 * length), int(0.2 * length)
        b = [0, train, length - test, length]
    return {'train': (b[0], b[1]), 'val': (b[1], b[2]), 'test': (b[2], b[3])}


def split_fractions(train_fraction):
    if train_fraction == 0.6:
        return {'train': (0.0, 0.6), 'val': (0.6, 0.8), 'test': (0.8, 1.0)}
    return {'train': (0.0, 0.7), 'val': (0.7, 0.8), 'test': (0.8, 1.0)}


def window_statistics(values):
    """Correlation and normalized slope of amplitude on level.

    values: (rows, 720) float64, one length-720 window per row.  Rows whose
    window has no level or amplitude variation yield NaN and are dropped.
    """
    cycles = values.reshape(values.shape[0], -1, PERIOD)  # (C, K, P)
    means = cycles.mean(-1)  # (C, K)
    amps = np.sqrt(((cycles - means[..., None]) ** 2).mean(-1))  # (C, K)
    mean_c = means - means.mean(-1, keepdims=True)
    amp_c = amps - amps.mean(-1, keepdims=True)
    cov = (mean_c * amp_c).mean(-1)
    var_m = (mean_c ** 2).mean(-1)
    var_a = (amp_c ** 2).mean(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.sqrt(var_m * var_a)
        slope = cov / var_m  # amplitude units per level unit
    scale = values.std(-1)  # per-row window standard deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_slope = slope / scale
    return corr, norm_slope


def level_amplitude(values):
    cycles = values.reshape(values.shape[0], -1, PERIOD)
    means = cycles.mean(-1)
    return means, np.sqrt(((cycles - means[..., None]) ** 2).mean(-1))


def pearson(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (a * b).mean(-1) / np.sqrt((a ** 2).mean(-1) * (b ** 2).mean(-1))


def consistent(per_channel):
    """Mean magnitude of the per-channel average correlations."""
    means = [np.mean(c) for c in per_channel if len(c)]
    return float(np.mean(np.abs(means))) if means else float('nan')


def dataset_rows(name, values, splits):
    rows = []
    for split, (lo, hi) in splits.items():
        part = values[lo:hi]
        count = max(0, (len(part) - LOOKBACK) // STRIDE + 1)
        if count < 2:
            continue
        windows = np.stack([part[i * STRIDE:i * STRIDE + LOOKBACK]
                            for i in range(count)])  # (W, L, C)
        windows = np.ascontiguousarray(windows.transpose(2, 0, 1))  # (C, W, L)
        corrs, slopes, shifted = [], [], []
        for channel in windows:
            c, s = window_statistics(channel)
            ok = np.isfinite(c) & np.isfinite(s) & (c > -1) & (c < 1)
            corrs.append(c[ok])
            slopes.append(s[ok])
            means, amps = level_amplitude(channel)
            r = pearson(means, np.roll(amps, count // 2, axis=0))
            shifted.append(r[np.isfinite(r)])
        corr = np.concatenate(corrs) if corrs else np.array([])
        slope = np.concatenate(slopes) if slopes else np.array([])
        if not len(corr):
            continue
        rows.append({
            'dataset': name, 'split': split, 'windows': int(len(corr)),
            'mean_abs_corr': float(np.abs(corr).mean()),
            'median_abs_corr': float(np.median(np.abs(corr))),
            'fraction_abs_corr_ge_0.3': float((np.abs(corr) >= CORR_THRESHOLD).mean()),
            'share_positive': float((corr > 0).mean()),
            'consistent_coupling': consistent(corrs),
            'shifted_pairing': consistent(shifted),
            'mean_abs_norm_slope': float(np.abs(slope).mean()),
            'mean_signed_norm_slope': float(slope.mean()),
        })
    return rows


def synthetic(alpha, length=24 * 4000, seed=2021):
    rng = np.random.default_rng(seed)
    k = np.arange(length) // PERIOD
    phase = np.arange(length) % PERIOD
    shape = np.sin(2 * np.pi * phase / PERIOD)
    level = np.cumsum(rng.normal(0, 0.05, length))
    x = level + (1 + alpha * level) * shape + rng.normal(0, 0.1, length)
    return x[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=None,
                        help='resources/all_datasets location; defaults to the repo copy')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'research_runs/vldbj_m1_coupling')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    data_root = (args.data_root or ROOT / 'resources/all_datasets').resolve()
    for name, (folder, file, train_fraction) in DATASETS.items():
        path = data_root / folder / file
        frame = pd.read_csv(path)
        values = frame.drop(columns=[frame.columns[0]]).to_numpy(dtype='float64')
        rows += dataset_rows(name, values, split_rows(name, len(values)))
    for label, alpha in (('synthetic_mult-alpha0.25', 0.25), ('synthetic_additive-null', 0.0)):
        series = synthetic(alpha)
        rows += dataset_rows(label, series, split_rows(label, len(series)))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'coupling.csv', index=False)
    summary = {r['dataset'] + '/' + r['split']: {
        'mean_abs_corr': r['mean_abs_corr'],
        'fraction_ge_0.3': r['fraction_abs_corr_ge_0.3']} for r in rows}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
