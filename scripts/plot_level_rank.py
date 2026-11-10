#!/usr/bin/env python3
"""Test error against the rank of the level map, relative to rank three."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, ORANGE, GRAY = '#1F618D', '#196F3D', '#FFBF4C', '#7F8C8D'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='level_rank_sweep')
    args = parser.parse_args()
    d = json.loads(args.results.read_text())
    names = [k for k in d if d[k]]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6), squeeze=False)
    for ax, key in zip(axes.ravel(), names):
        rows = d[key]
        ranks = sorted(int(r) for r in rows)
        base = rows['3']['mse']
        delta = [(rows[str(r)]['mse'] - base) / base * 100 for r in ranks]
        ax.axhline(0, color=GRAY, lw=1, ls='--')
        ax.plot(ranks, delta, '-o', color=BLUE, lw=2, ms=7)
        for r, v in zip(ranks, delta):
            ax.annotate(f'{v:+.2f}', (r, v), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=9, color=BLUE)
        ax.set_title(key, fontsize=13)
        ax.set_xlabel('rank of the level map', fontsize=12)
        ax.set_xticks(ranks)
        ax.tick_params(labelsize=11)
        ax.grid(axis='y', ls=':', color='0.85')
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('test MSE relative to rank 3 (%)', fontsize=12)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / f'{args.name}.png', dpi=300, bbox_inches='tight')
    print('wrote', args.out / f'{args.name}.png')


if __name__ == '__main__':
    main()
