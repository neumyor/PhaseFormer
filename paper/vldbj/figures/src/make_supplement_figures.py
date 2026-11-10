"""Hyperparameter sensitivity of the phase pathway (Section 5.6).

One row of four panels. In every panel one factor of the selected configuration is changed.
Bars: mean change of the test MSE over the settings; dots: the individual settings; gray band: the selected value
(change 0); data: exp_supplement.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/Users/yimingniu/Code/PhaseFormer/paper/vldbj/figures/src')
sys.path.insert(0, str(HERE))
from figstyle import INK, PHASE, REST, TICK, apply, panel_title

apply()
d = json.load(open(HERE / 'exp_supplement.json'))
base = {(r[0], r[1]): r[2] for r in d['settings']}
SETTINGS = [(ds, h) for ds in ('Traffic', 'Electricity', 'Weather') for h in (96, 336)]
OUT = Path('/Users/yimingniu/Code/PhaseFormer-overleaf/figures/exp_supplement.pdf')
rel = lambda v, s: 100 * (v / base[s] - 1)


def panel(ax, labels, groups, ymax):
    """groups[i]: per-setting changes (%) of category i; an empty list marks the selected category."""
    for i, vals in enumerate(groups):
        if not len(vals):
            ax.axvspan(i - 0.5, i + 0.5, color=REST, alpha=0.45, lw=0, zorder=0)
            ax.text(i, 0.04 * ymax, 'selected', ha='center', va='bottom', fontsize=TICK - 0.5, color=INK, zorder=5)
            continue
        mean = float(np.mean(vals))
        ax.bar(i, mean, 0.6, color=PHASE, lw=0, zorder=2)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(vals)), vals, s=6, color=INK, edgecolor='white',
                   linewidth=0.3, zorder=3)
        ax.text(i, max(max(vals), mean) + 0.035 * ymax, '%+.1f' % mean, ha='center', va='bottom', fontsize=TICK,
                color=INK, zorder=5)
    ax.axhline(0, color=INK, lw=0.6, zorder=4)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.04 * ymax, ymax)
    ax.grid(axis='y', ls=':', lw=0.45, color='0.85', zorder=0)


fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), gridspec_kw={'width_ratios': [3.6, 3.8, 6.2, 3.8]})
fig.subplots_adjust(left=0.065, right=0.995, top=0.87, bottom=0.17, wspace=0.30)

M = [2, 4, 8, 16]
groups = [[] if m == 8 else [rel(d['router']['%s-%d-M%d' % (ds, h, m)][0], (ds, h)) for ds, h in SETTINGS] for m in M]
panel(axes[0], [str(m) for m in M], groups, 5.2)
panel_title(axes[0], 'a', 'Number of routers $M$')

VAR = ['full_attn', 'linear_mix', 'none']
groups = [[]] + [[rel(d['routing']['%s-%d-%s' % (ds, h, v)][0], (ds, h)) for ds, h in SETTINGS] for v in VAR]
panel(axes[1], ['Router', 'Full\nattn.', 'Linear', 'None'], groups, 5.2)
panel_title(axes[1], 'b', 'Interaction across phases')

SEL = {96: 24, 336: 12}
P = [8, 12, 16, 24, 48]
ax = axes[2]
for k, (h, color) in enumerate(((96, PHASE), (336, '#A9DAF0'))):
    xs = np.arange(len(P)) + (k - 0.5) * 0.42
    for x, p in zip(xs, P):
        if p == SEL[h]:
            ax.scatter([x], [0.0], marker='^', s=14, color=INK, zorder=5, clip_on=False)
            continue
        v = rel(d['phase']['Traffic-%d-P%d' % (h, p)][0], ('Traffic', h))
        ax.bar(x, v, 0.40, color=color, lw=0, zorder=2)
        ax.text(x, max(v, 0) + 0.035 * 10.4, '%+.1f' % v, ha='center', va='bottom', fontsize=TICK - 1.0, color=INK)
ax.axhline(0, color=INK, lw=0.6, zorder=4)
ax.set_xticks(range(len(P)), [str(p) for p in P])
ax.set_xlim(-0.55, len(P) - 0.45)
ax.set_ylim(-0.04 * 10.4, 10.4)
ax.grid(axis='y', ls=':', lw=0.45, color='0.85', zorder=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE, lw=0), plt.Rectangle((0, 0), 1, 1, color='#A9DAF0', lw=0),
           plt.Line2D([], [], marker='^', ls='', color=INK, ms=3.5)]
ax.legend(handles, ['$H=96$', '$H=336$', 'selected'], loc='upper left', fontsize=TICK - 0.5, handlelength=1.0,
          borderaxespad=0.1, labelspacing=0.25)
panel_title(ax, 'c', 'Phase length $P$ (Traffic)')

L = [96, 192, 336, 720]
groups = [[] if l == 720 else [rel(d['input']['%s-%d-L%d' % (ds, h, l)][0], (ds, h)) for ds, h in SETTINGS] for l in L]
panel(axes[3], [str(l) for l in L], groups, 48)
panel_title(axes[3], 'd', 'Input length $L$')

for ax in axes:
    ax.set_ylabel('')
axes[0].set_ylabel('MSE change (%)')
fig.savefig(OUT)
print('wrote', OUT)
