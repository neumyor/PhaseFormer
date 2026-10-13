#!/usr/bin/env python3
"""Sign and weekday checks of the level-amplitude coupling (read-only).

Uses the test windows of coupling.py (length 720, stride 24, P=24, the
test split of the forecasting experiments) and reports, per dataset:

* the median signed correlation and, per channel, the average signed
  correlation and the share of positive windows;
* for the hourly sensor networks (Traffic, Electricity), the mean |corr| and
  the sign-consistent coupling over the weekday cycles of each window only.
  A cycle is assigned to the day of its middle sample; windows with fewer
  than 15 weekday cycles are dropped.

Usage:
  python scripts/vldbj_m1/coupling_extra.py --data-root <all_datasets> \
      --output research_runs/vldbj_m1_coupling
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coupling import DATASETS, LOOKBACK, PERIOD, ROOT, STRIDE, consistent, split_rows, window_statistics

WEEKDAY_SETS = ('Traffic', 'Electricity')


def test_windows(name, values, dates):
    a, b = split_rows(name, len(values))['test']
    part, part_dates = values[a:b], dates[a:b]
    count = (len(part) - LOOKBACK) // STRIDE + 1
    starts = np.arange(count) * STRIDE
    windows = np.stack([part[s:s + LOOKBACK] for s in starts])  # (W, L, C)
    return windows, part_dates, starts


def weekday_corr(windows, part_dates, starts):
    """Correlation over the weekday cycles of each window, per channel."""
    middle = np.arange(30) * PERIOD + PERIOD // 2
    out = []  # (W', C)
    for w, s in enumerate(starts):
        weekday = part_dates[s + middle].dayofweek < 5
        if weekday.sum() < 15:
            continue
        cycles = windows[w].T.reshape(windows.shape[2], 30, PERIOD)[:, weekday]  # (C, K', P)
        means = cycles.mean(-1)
        amps = np.sqrt(((cycles - means[..., None]) ** 2).mean(-1))
        mc = means - means.mean(-1, keepdims=True)
        ac = amps - amps.mean(-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = (mc * ac).mean(-1) / np.sqrt((mc ** 2).mean(-1) * (ac ** 2).mean(-1))
        out.append(corr)
    return np.stack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'research_runs/vldbj_m1_coupling')
    args = parser.parse_args()
    data_root = (args.data_root or ROOT / 'resources/all_datasets').resolve()
    rows = {}
    for name, (folder, file, _) in DATASETS.items():
        frame = pd.read_csv(data_root / folder / file)
        dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
        values = frame.drop(columns=[frame.columns[0]]).to_numpy(dtype='float64')
        windows, part_dates, starts = test_windows(name, values, dates)
        per_channel = []
        for c in range(windows.shape[2]):
            r = window_statistics(windows[:, :, c])[0]
            per_channel.append(r[np.isfinite(r) & (r > -1) & (r < 1)])
        corr = np.concatenate(per_channel)
        row = {'windows': int(len(corr)),
               'mean_abs_corr': float(np.abs(corr).mean()),
               'median_abs_corr': float(np.median(np.abs(corr))),
               'share_positive': float((corr > 0).mean()),
               'median_signed_corr': float(np.median(corr)),
               'consistent_coupling': consistent(per_channel)}
        if windows.shape[2] <= 21:
            row['channel_mean_corr'] = [round(float(np.mean(c)), 3) for c in per_channel]
            row['channel_share_positive'] = [round(float((c > 0).mean()), 3) for c in per_channel]
        if name in WEEKDAY_SETS:
            wd = weekday_corr(windows, part_dates, starts)
            cols = [wd[:, c][np.isfinite(wd[:, c])] for c in range(wd.shape[1])]
            flat = np.concatenate(cols)
            row.update({'weekday_windows': int(wd.shape[0]),
                        'weekday_mean_abs_corr': float(np.abs(flat).mean()),
                        'weekday_consistent_coupling': consistent(cols)})
        rows[name] = row
        print(name, json.dumps(row))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'coupling_extra.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
