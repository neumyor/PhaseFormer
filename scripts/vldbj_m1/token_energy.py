#!/usr/bin/env python3
"""Energy of a normalized window carried by its level and amplitude-scaled shape.

For every test channel and length-720 window (stride 24, 30 cycles of 24
samples), the window is normalized to zero mean and unit variance and folded
into a 30 x 24 matrix X with one row per cycle.  X splits orthogonally into the
level term l 1^T (the cycle means repeated over the phases) and the
within-cycle part R = X - l 1^T.  The best rank-one approximation a s^T of R
(its leading singular pair) is one amplitude per cycle times one shape over
the phases.  The script reports, per dataset, the mean shares of the energy
||X||_F^2 carried by l 1^T, by a s^T and by the remainder, and draws them as
stacked bars (paper Fig. token_energy).

Splits follow the forecasting experiments (see coupling.py).

Usage:
  python scripts/vldbj_m1/token_energy.py --data-root <all_datasets> \
      --output research_runs/vldbj_token_energy \
      --figure paper/vldbj/figures/token_energy.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coupling import DATASETS, LOOKBACK, PERIOD, STRIDE, split_rows

ROOT = Path(__file__).resolve().parents[2]
CYCLES = LOOKBACK // PERIOD


def windows_of(part):
    count = max(0, (len(part) - LOOKBACK) // STRIDE + 1)
    idx = np.arange(count)[:, None] * STRIDE + np.arange(LOOKBACK)[None]
    return part[idx].transpose(2, 0, 1).reshape(-1, LOOKBACK)  # (C*W, L)


def shares(values):
    """Mean energy shares (level, amplitude x shape, rest) of the windows."""
    scale = values.std(-1, keepdims=True)
    keep = scale[:, 0] > 1e-8
    x = (values[keep] - values[keep].mean(-1, keepdims=True)) / scale[keep]
    folded = x.reshape(-1, CYCLES, PERIOD)
    level = folded.mean(-1, keepdims=True)
    within = folded - level
    total = (folded ** 2).sum((1, 2))
    level_energy = PERIOD * (level[..., 0] ** 2).sum(-1)
    sv = np.linalg.svd(within, compute_uv=False)
    amp_shape = sv[:, 0] ** 2
    lev = level_energy / total
    amp = amp_shape / total
    return float(lev.mean()), float(amp.mean()), float((1 - lev - amp).mean()), int(len(x))


def draw(frame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                         'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6})
    names = list(frame['dataset'])[::-1]
    data = frame.set_index('dataset').loc[names]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    parts = [('level', 'Level  $\\ell\\,\\mathbf{1}^\\top$', '#2a78d6'),
             ('amplitude_shape', 'Amplitude $\\times$ shape  $a\\,s^\\top$', '#eb6834'),
             ('rest', 'Rest', '#c9c8c2')]
    left = np.zeros(len(names))
    for key, label, color in parts:
        width = data[key].to_numpy()
        ax.barh(names, width, left=left, color=color, edgecolor='white',
                linewidth=1.0, height=0.68, label=label)
        left += width
    for i, total in enumerate(data['level'] + data['amplitude_shape']):
        ax.text(total - 0.015, i, f'{total:.0%}', ha='right', va='center',
                color='white', fontsize=7, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Share of the energy of a normalized window', color='#52514e')
    ax.tick_params(length=0, colors='#0b0b0b')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0),
              frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'resources/all_datasets')
    parser.add_argument('--output', type=Path, default=ROOT / 'research_runs/vldbj_token_energy')
    parser.add_argument('--figure', type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (folder, file, _) in DATASETS.items():
        frame = pd.read_csv(args.data_root / folder / file)
        values = frame.drop(columns=[frame.columns[0]]).to_numpy(dtype='float64')
        lo, hi = split_rows(name, len(values))['test']
        lev, amp, rest, n = shares(windows_of(values[lo:hi]))
        rows.append({'dataset': name, 'windows': n, 'level': lev,
                     'amplitude_shape': amp, 'rest': rest,
                     'captured': lev + amp})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'token_energy.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=1))
    print(frame.to_string(index=False))
    if args.figure:
        draw(frame, args.figure)


if __name__ == '__main__':
    main()
