#!/usr/bin/env python3
"""Fold predictions and truth into trend and periodic components, then score both.

For a forecast window ``x`` of length ``H`` and a period ``P`` dividing ``H``,
the window is cut into ``H/P`` cycles and folded::

    trend    = mean over cycles of x          (P, C), tiled back to (H, C)
    periodic = x - trend

Folding is an orthogonal projection, so for any prediction the error splits the
same way and ``MSE_total = MSE_trend + MSE_periodic`` holds exactly; the script
asserts it. Everything is scored in the StandardScaler space the models train
in, matching the paper's MSE, unless ``--space original`` asks for raw units.

Example::

    python -m scripts.component_mse.decompose --dump-root <dir> --period 24 \
        --out <dir>/component_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.component_mse import dump  # noqa: E402

MODELS = ('PhaseFormer', 'PhaseFormerLA', 'DLinear', 'SparseTSF', 'FITS', 'CycleNet')
DATASETS = ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)


def fold(x, period):
    """Cycle-folded mean of ``x`` (N, H, C); requires period | H."""
    n, length, channels = x.shape
    if length % period:
        raise ValueError(f'period {period} does not divide horizon {length}')
    cycles = x.reshape(n, length // period, period, channels)
    return np.tile(cycles.mean(axis=1, keepdims=True), (1, length // period, 1, 1)).reshape(x.shape)


def component_scores(prediction, truth, period, scale=None, chunk=512):
    """Per-component MSE over one prediction file, streamed in row chunks.

    Traffic-H192 is 3.3k windows x 192 x 862: materialising the error, its fold
    and the residual at once needs ~10 GB, so the sums are accumulated chunk by
    chunk in float64 while the per-element work stays float32.
    """
    totals = [0.0, 0.0, 0.0]
    count = 0
    for start in range(0, len(prediction), chunk):
        stop = start + chunk
        error = prediction[start:stop].astype('float32') - truth[start:stop]
        if scale is not None:
            error = error * scale
        trend = fold(error, period)
        periodic = error - trend
        totals[0] += float(np.square(error).sum(dtype=np.float64))
        totals[1] += float(np.square(trend).sum(dtype=np.float64))
        totals[2] += float(np.square(periodic).sum(dtype=np.float64))
        count += error.size
    total, trend_mse, periodic_mse = (value / count for value in totals)
    if abs(total - trend_mse - periodic_mse) > 1e-6 * max(1.0, total):
        raise AssertionError((total, trend_mse, periodic_mse))
    return {'total': total, 'trend': trend_mse, 'periodic': periodic_mse}


def collect(dump_root, period, scale_by_dataset):
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            try:
                truth = dump.load_truth(dump_root, dataset, horizon)
            except FileNotFoundError:
                continue
            scale = None
            if scale_by_dataset:
                with np.load(dump.truth_path(dump_root, dataset, horizon)) as handle:
                    scale = handle['scaler_scale'].astype('float32')[None, None, :]
            for model in MODELS:
                for seed in SEEDS:
                    try:
                        prediction = dump.load_prediction(dump_root, model, dataset, horizon, seed)
                    except FileNotFoundError:
                        continue
                    scores = component_scores(prediction, truth, period, scale)
                    rows.append({'dataset': dataset, 'horizon': horizon, 'model': model,
                                 'seed': seed, **scores})
    return rows


def summarise(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['horizon'], row['model']), []).append(row)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {metric: (statistics.fmean(r[metric] for r in items),
                                 statistics.stdev([r[metric] for r in items]) if len(items) > 1 else 0.0,
                                 len(items))
                        for metric in ('total', 'trend', 'periodic')}
    return summary


def fmt(value, digits=6):
    return f'{value:.{digits}f}'


def cell(summary, key, metric):
    entry = summary.get(key)
    if entry is None:
        return '—'
    mean, std, count = entry[metric]
    if count > 1:
        return f'{fmt(mean)} ± {fmt(std)}'
    return f'{fmt(mean)}'


def table(summary, metric, settings, seeds_expected):
    header = '| Model | ' + ' | '.join(f'{d}-{h}' for d, h in settings) + ' |'
    lines = [header, '|' + '---|' * (len(settings) + 1)]
    for model in MODELS:
        cells = [cell(summary, (d, h, model), metric) for d, h in settings]
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-root', type=Path, required=True)
    parser.add_argument('--period', type=int, default=24)
    parser.add_argument('--extra-periods', type=int, nargs='*', default=[])
    parser.add_argument('--space', choices=('standardized', 'original'), default='standardized')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    periods = [args.period] + [p for p in args.extra_periods if p != args.period]
    lines = ['# 成分级 MSE 对比', '',
             f'- 数据目录：`{args.dump_root}`',
             f'- 分解：按周期折叠均值，主周期 P={args.period}',
             f'- 误差空间：{args.space}',
             '- 完整的测试集预测已落盘，换分解方式只需重跑本脚本', '']
    csv_rows = []
    for period in periods:
        rows = collect(args.dump_root, period, args.space == 'original')
        if not rows:
            lines += [f'## P={period}', '', '尚无结果。', '']
            continue
        summary = summarise(rows)
        settings = sorted({(r['dataset'], r['horizon']) for r in rows})
        seeds_seen = sorted({r['seed'] for r in rows})
        lines += [f'## P={period}（{len(seeds_seen)} 个 seed：{", ".join(map(str, seeds_seen))}）', '']
        lines += ['### 总 MSE', '', table(summary, 'total', settings, seeds_seen), '']
        lines += ['### 趋势成分 MSE', '', table(summary, 'trend', settings, seeds_seen), '']
        lines += ['### 周期成分 MSE', '', table(summary, 'periodic', settings, seeds_seen), '']
        lines += ['### 趋势成分占总 MSE 的比例', '']
        header = '| Model | ' + ' | '.join(f'{d}-{h}' for d, h in settings) + ' |'
        lines += [header, '|' + '---|' * (len(settings) + 1)]
        for model in MODELS:
            cells = []
            for dataset, horizon in settings:
                entry = summary.get((dataset, horizon, model))
                if entry is None or entry['total'][0] == 0:
                    cells.append('—')
                else:
                    share = entry['trend'][0] / entry['total'][0]
                    cells.append(f'{100 * share:.1f}%')
            lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        for row in rows:
            csv_rows.append({'period': period, **row})

    text = '\n'.join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        csv_path = args.out.with_suffix('.csv')
        if csv_rows:
            with csv_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
    print(text)


if __name__ == '__main__':
    main()
