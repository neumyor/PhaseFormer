#!/usr/bin/env python3
"""Share of the level demand's variance inside three- and six-dimensional subspaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='level_pca_summary')
    args = parser.parse_args()
    m = json.loads(args.summary.read_text())
    names = [n for n in ORDER if n in m]
    leg = [m[n]['demand_leg3'] * 100 for n in names]
    pca3 = [m[n]['demand_pca3'] * 100 for n in names]
    pca6 = [100 * sum(m[n]['curves']['demand']['ratio'][:6]) for n in names]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.bar(x - w, leg, w, color='#75B9E6', label='three fixed profiles')
    ax.bar(x, pca3, w, color='#196F3D', label='best three PCA directions')
    ax.bar(x + w, pca6, w, color='#C9C9C9', label='best six PCA directions')
    for xi, v in zip(x - w, leg):
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', fontsize=10)
    for xi, v in zip(x, pca3):
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', fontsize=10)
    for xi, v in zip(x + w, pca6):
        ax.text(xi, v + 1, f'{v:.0f}', ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Share of the level demand's variance (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', ls=':', color='0.85')
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.13))
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f'{args.name}.png', dpi=300, bbox_inches='tight')
    print('wrote', args.out / f'{args.name}.png')


if __name__ == '__main__':
    main()
