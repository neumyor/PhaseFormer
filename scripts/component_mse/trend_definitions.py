#!/usr/bin/env python3
"""Compare candidate definitions of the trend component on the dumped predictions.

Every definition is a linear operator T applied to the forecast error
``e = pred - truth``; the periodic (residual) part is ``e - T(e)``. No model is
retrained: the 72 test-set prediction files already hold everything needed.

Definitions, all on the horizon of length H with cycle length P:

* ``folded``      -- mean over cycles at a fixed phase (the average cycle shape)
* ``cycle-level`` -- mean inside each cycle (a piecewise-constant level)
* ``ma25``/``ma49`` -- centered moving average with replicate padding
* ``linear``      -- least-squares straight line over the horizon
* ``ema25``       -- causal exponential moving average, alpha = 2/26
* ``lowpass``     -- DFT bins below the fundamental of P (periods longer than P)

``folded`` and ``cycle-level`` are orthogonal projections, so their two
component MSEs sum exactly to the total; the others leave a cross term, which
is reported separately instead of being folded into either component.

The script also scores each definition against a pre-declared narrative rule,
so the choice is made on explicit criteria rather than by eyeballing tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.component_mse import dump  # noqa: E402

MODELS = ('PhaseFormer', 'PhaseFormerLA', 'DLinear', 'SparseTSF', 'FITS', 'CycleNet')
BASELINES = MODELS[2:]
DATASETS = ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)
DEFINITIONS = ('folded', 'cycle-level', 'ma25', 'ma49', 'linear', 'ema25', 'lowpass')
CHUNK = 256


def _cycles(error, period):
    n, length, channels = error.shape
    return error.reshape(n, length // period, period, channels)


def apply_trend(error, definition, period):
    n, length, channels = error.shape
    if length % period:
        raise ValueError((length, period))
    if definition == 'folded':
        cycles = _cycles(error, period)
        return np.tile(cycles.mean(axis=1, keepdims=True),
                       (1, length // period, 1, 1)).reshape(error.shape)
    if definition == 'cycle-level':
        cycles = _cycles(error, period)
        return np.tile(cycles.mean(axis=2, keepdims=True),
                       (1, 1, period, 1)).reshape(error.shape)
    if definition in ('ma25', 'ma49'):
        kernel = int(definition[2:])
        pad = kernel // 2
        padded = np.pad(error, ((0, 0), (pad, pad), (0, 0)), mode='edge')
        csum = np.cumsum(padded.astype('float64'), axis=1)
        csum = np.concatenate([np.zeros((n, 1, channels)), csum], axis=1)
        return ((csum[:, kernel:] - csum[:, :-kernel]) / kernel).astype('float32')
    if definition == 'linear':
        t = np.arange(length, dtype='float64') - (length - 1) / 2
        mean = error.mean(axis=1, keepdims=True)
        slope = (error.astype('float64') * t[None, :, None]).sum(axis=1, keepdims=True) / (t * t).sum()
        return (mean + slope * t[None, :, None]).astype('float32')
    if definition == 'ema25':
        alpha = 2 / (25 + 1)
        out = np.empty_like(error)
        state = error[:, 0, :].astype('float32')
        for step in range(length):
            state = alpha * error[:, step, :] + (1 - alpha) * state
            out[:, step, :] = state
        return out
    if definition == 'lowpass':
        keep = max(1, length // period)          # bins strictly below the fundamental of P
        spectrum = np.fft.rfft(error, axis=1)
        spectrum[:, keep:] = 0
        return np.fft.irfft(spectrum, n=length, axis=1).astype('float32')
    raise ValueError(definition)


def score_file(prediction, truth, period, definitions):
    totals = {d: [0.0, 0.0, 0.0] for d in definitions}   # trend, residual, total
    count = 0
    for start in range(0, len(prediction), CHUNK):
        stop = start + CHUNK
        error = prediction[start:stop].astype('float32') - truth[start:stop]
        count += error.size
        for definition in definitions:
            trend = apply_trend(error, definition, period)
            residual = error - trend
            totals[definition][0] += float(np.square(trend).sum(dtype=np.float64))
            totals[definition][1] += float(np.square(residual).sum(dtype=np.float64))
            totals[definition][2] += float(np.square(error).sum(dtype=np.float64))
    return {d: {'trend': v[0] / count, 'periodic': v[1] / count, 'total': v[2] / count}
            for d, v in totals.items()}


def collect(dump_root, period, definitions, datasets=DATASETS, horizons=HORIZONS):
    rows = []
    for dataset in datasets:
        for horizon in horizons:
            try:
                truth = dump.load_truth(dump_root, dataset, horizon)
            except FileNotFoundError:
                continue
            for model in MODELS:
                for seed in SEEDS:
                    try:
                        prediction = dump.load_prediction(dump_root, model, dataset, horizon, seed)
                    except FileNotFoundError:
                        continue
                    scores = score_file(prediction, truth, period, definitions)
                    for definition, values in scores.items():
                        rows.append({'definition': definition, 'dataset': dataset,
                                     'horizon': horizon, 'model': model, 'seed': seed, **values})
    return rows


def summarise(rows):
    grouped = {}
    for row in rows:
        key = (row['definition'], row['dataset'], row['horizon'], row['model'])
        grouped.setdefault(key, []).append(row)
    return {key: {metric: statistics.fmean(r[metric] for r in items)
                  for metric in ('trend', 'periodic', 'total')}
            for key, items in grouped.items()}


def fmt(value):
    return f'{value:.4f}'


def table(summary, definition, metric, settings):
    lines = ['| Model | ' + ' | '.join(f'{d}-{h}' for d, h in settings) + ' |',
             '|' + '---|' * (len(settings) + 1)]
    for model in MODELS:
        cells = [fmt(summary[(definition, d, h, model)][metric]) for d, h in settings]
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def narrative_score(summary, definition, settings):
    """How well a definition supports: PF trend slightly worse, PF periodic better, LA repairs trend."""
    trend_worse, periodic_better, la_repairs = 0, 0, 0
    details = []
    for dataset, horizon in settings:
        pf = summary[(definition, dataset, horizon, 'PhaseFormer')]
        la = summary[(definition, dataset, horizon, 'PhaseFormerLA')]
        best_base_trend = min(summary[(definition, dataset, horizon, m)]['trend'] for m in BASELINES)
        best_base_periodic = min(summary[(definition, dataset, horizon, m)]['periodic'] for m in BASELINES)
        trend_gap = (pf['trend'] - best_base_trend) / best_base_trend
        periodic_gain = (pf['periodic'] - best_base_periodic) / best_base_periodic
        repair = (la['trend'] - pf['trend']) / pf['trend']
        trend_worse += trend_gap > 0
        periodic_better += periodic_gain < 0
        la_repairs += repair < 0
        details.append(f'{dataset}-{horizon}: PF趋势{trend_gap:+.1%} 周期{periodic_gain:+.1%} LA趋势修补{repair:+.1%}')
    return trend_worse, periodic_better, la_repairs, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-root', type=Path, required=True)
    parser.add_argument('--period', type=int, default=24)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    parser.add_argument('--settings-limit', type=int, default=0,
                        help='only the first N settings, for a quick partial read')
    args = parser.parse_args()

    settings = [(d, h) for d in args.datasets for h in args.horizons]
    if args.settings_limit:
        settings = settings[:args.settings_limit]
    rows = collect(args.dump_root, args.period, DEFINITIONS,
                   [d for d, _ in settings], sorted({h for _, h in settings}))
    rows = [r for r in rows if (r['dataset'], r['horizon']) in set(settings)]
    summary = summarise(rows)
    lines = [f'# 趋势定义对比（P={args.period}，3 seed 均值）', '']
    verdicts = []
    for definition in DEFINITIONS:
        worse, better, repairs, details = narrative_score(summary, definition, settings)
        verdicts.append((definition, worse, better, repairs))
        lines += [f'## {definition}', '', '### 趋势成分 MSE', '',
                  table(summary, definition, 'trend', settings), '',
                  '### 周期成分 MSE', '',
                  table(summary, definition, 'periodic', settings), '',
                  f'叙事吻合度：PhaseFormer 趋势更差 {worse}/4；周期更好 {better}/4；LA 修补趋势 {repairs}/4', '']
        lines += ['- ' + item for item in details] + ['']
    lines += ['## 总表', '', '| 定义 | PF 趋势更差 | PF 周期更好 | LA 修补趋势 |',
              '|---|---|---|---|']
    for definition, worse, better, repairs in verdicts:
        lines.append(f'| {definition} | {worse}/4 | {better}/4 | {repairs}/4 |')
    text = '\n'.join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)
        args.out.with_suffix('.json').write_text(json.dumps(
            {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v for k, v in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
