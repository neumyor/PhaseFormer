#!/usr/bin/env python3
"""Figures for the level-token PCA study, in the style of the ICLR phase-token PCA figure.

Bars: per-component explained ratio. Lines: cumulative ratio. Dashed horizontal
lines at 90% and 95%, dotted vertical lines where each series crosses 90%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE_BAR, GREEN_BAR = '#75B9E6', '#68B44D'
BLUE_LINE, GREEN_LINE = '#1F618D', '#196F3D'
ORANGE, RED = '#FFBF4C', '#FF4C4C'
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
K = 12


def panel(ax, a, b, name_a, name_b):
    x = np.arange(1, K + 1)
    w = 0.38
    ax.bar(x - w / 2, a['ratio'], w, color=BLUE_BAR, label=name_a, zorder=2)
    ax.bar(x + w / 2, b['ratio'], w, color=GREEN_BAR, label=name_b, zorder=2)
    ca, cb = np.array(a['cum']), np.array(b['cum'])
    ax.plot(x, ca, '-o', color=BLUE_LINE, lw=2.2, ms=7, zorder=3)
    ax.plot(x, cb, '-o', color=GREEN_LINE, lw=2.2, ms=7, zorder=3)
    ax.axhline(.90, color=ORANGE, ls='--', lw=2, zorder=1)
    ax.axhline(.95, color=RED, ls='--', lw=2, zorder=1)
    ax.text(0.5, .915, '90%', color=ORANGE, fontsize=14, fontweight='bold', va='bottom',
            bbox=dict(fc='white', ec='none', pad=1.5), zorder=6)
    ax.text(0.5, .96, '95%', color=RED, fontsize=14, fontweight='bold', va='bottom',
            bbox=dict(fc='white', ec='none', pad=1.5), zorder=6)
    for c, col, levels in ((ca, BLUE_LINE, (.9, .95)), (cb, GREEN_LINE, (.9,))):
        for lev in levels:
            hits = np.where(c >= lev)[0]
            if len(hits):
                ax.axvline(x[hits[0]], color=col, ls=':', lw=2, alpha=.85, zorder=1)
    ax.set_xlim(0.3, K + 0.7)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.tick_params(labelsize=15)
    ax.grid(axis='y', ls=':', color='0.8', zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Principal component', fontsize=17)
    ax.set_ylabel('Explained ratio / Cumulative', fontsize=17)
    ax.legend(loc='lower right', frameon=False, fontsize=16)


def figure(curves, key_a, key_b, name_a, name_b, out_dir, tag, grid=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        if name not in curves:
            continue
        d = curves[name]
        fig, ax = plt.subplots(figsize=(7.56, 5.24))
        panel(ax, d[key_a], d[key_b], name_a, name_b)
        fig.tight_layout()
        fig.savefig(out_dir / f'{tag}_{name.lower()}.png', dpi=300)
        plt.close(fig)
    names = [n for n in ORDER if n in curves]
    if grid and len(names) > 1:
        rows = (len(names) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(7.2, 2.9 * rows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            panel(ax, curves[name][key_a], curves[name][key_b], name_a, name_b)
            ax.set_title(name, fontsize=15)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend().remove()
        for ax in axes.ravel()[len(names):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f'{tag}_grid.png', dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curves', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    curves = json.loads(args.curves.read_text())
    figure(curves, 'ridge', 'demand', 'Predictable', 'Full demand', args.out, 'level_pca_predictable')
    figure(curves, 'level', 'value', 'Level', 'Value', args.out, 'level_pca_token')
    figure(curves, 'branch', 'demand', 'Level branch', 'Full demand', args.out, 'level_pca_branch')
    print('written to', args.out)


if __name__ == '__main__':
    main()
