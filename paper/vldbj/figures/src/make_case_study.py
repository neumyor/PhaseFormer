"""Figure: three test windows of ETTm2 (H=96) on which PhaseFormer+ corrects the cycle levels of PhaseFormer.

Thin lines are the series, thick translucent segments the mean of each future cycle (the trend term).
Data: case_trend.npz (window of the last 24 input steps, truth and both forecasts), mined with case_trend.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/Users/yimingniu/Code/PhaseFormer/paper/vldbj/figures/src')
sys.path.insert(0, str(HERE))
from figstyle import INK, MUTED, OURS, PHASE, TICK, apply, panel_title

apply()
OUT = Path('/Users/yimingniu/Code/PhaseFormer-overleaf/figures/case_study.pdf')
P, KEEP = 24, 24
d = np.load(HERE / 'case_trend.npz')
TITLES = ['Level offset', 'Level dip', 'Level rise']

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
fig.subplots_adjust(left=0.065, right=0.995, top=0.80, bottom=0.17, wspace=0.26)
for n, ax in enumerate(axes):
    hist, truth, pf, plus = d['hist'][n][-KEEP:], d['truth'][n], d['pf'][n], d['plus'][n]
    x = np.arange(len(truth))
    ax.axvline(0, color=MUTED, lw=0.5, ls=':')
    ax.plot(np.arange(-KEEP, 0), hist, color=MUTED, lw=0.8)
    for series, color, lw in ((truth, INK, 0.7), (pf, PHASE, 0.8), (plus, OURS, 0.8)):
        ax.plot(x, series, color=color, lw=lw)
    for c in range(len(truth) // P):
        sl = slice(c * P, (c + 1) * P)
        for series, color in ((truth, INK), (pf, PHASE), (plus, OURS)):
            ax.plot(x[sl], np.full(P, series[sl].mean()), color=color, lw=2.2, alpha=0.45, solid_capstyle='butt')
    ax.set_xlim(-KEEP, len(truth) - 1)
    ax.set_xlabel('Step')
    ax.grid(axis='y', ls=':', lw=0.45, color='0.85', zorder=0)
    panel_title(ax, 'abc'[n], TITLES[n])
    ax.set_title('trend error %.2f $\\rightarrow$ %.2f' % (d['tr_pf'][n], d['tr_plus'][n]), loc='right',
                 color=OURS, fontsize=TICK)
axes[0].set_ylabel('Value')
handles = [plt.Line2D([], [], color=c, lw=1.2) for c in (MUTED, INK, PHASE, OURS)]
leg = fig.legend(handles, ['Input', 'Truth', 'PhaseFormer', 'PhaseFormer+'], loc='upper center', ncol=4,
                 bbox_to_anchor=(0.53, 1.0), handlelength=1.8, columnspacing=2.4)
for text, color in zip(leg.get_texts(), (MUTED, INK, PHASE, OURS)):
    text.set_color(color)
fig.savefig(OUT)
print('wrote', OUT)
