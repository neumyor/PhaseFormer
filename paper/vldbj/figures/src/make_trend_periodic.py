"""Fig. 3 of Section 4.1: a folded window is a trend term plus a periodic term.

(a) One normalized Traffic window (sensor 580, thirty days from Monday
    2016-08-22, test period), folded with one row per day and one column per hour. X~ is the
    sum of the trend term r_bar 1^T (every row constant: the day means) and the
    periodic term C = X~ - r_bar 1^T (every row of zero mean). A column of X~ is a
    phase token; the mean of a row is the level of its cycle.
(b) Energy shares over the test windows of every dataset (token_energy.csv,
    scripts/vldbj_m1/token_energy.py): the trend term r_bar 1^T, the best rank-one
    periodic term a s^T of C, and the rest; the periodic term C is the second
    and the third part together.

    python3 make_trend_periodic.py [out.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from figstyle import INK, LEVEL, MUTED, PHASE, REST, SMALL, apply, panel_label

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'trend_periodic.pdf')
apply()

P, K = 24, 30
window = pd.read_csv(os.path.join(here, 'traffic_window_sensor580.csv'))
x = window['value'].to_numpy('float64')
x = (x - x.mean()) / x.std()
X = x.reshape(K, P)
level = X.mean(1, keepdims=True) * np.ones((1, P))
C = X - level

table = pd.read_csv(os.path.join(root, 'research_runs/vldbj_token_energy/token_energy.csv'))
data = table.set_index('dataset')
periodic_led = ['Traffic', 'Electricity', 'ETTh1']
trend_led = ['Weather', 'ETTm2', 'ETTm1', 'ETTh2']
names = (periodic_led + trend_led)[::-1]
data = data.loc[names]

COL, ROW = 14, 16                      # the phase token and the cycle highlighted in X

fig = plt.figure(figsize=(3.35, 2.75))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.62,
                         left=0.235, right=0.80, top=0.9, bottom=0.115)
maps = outer[0].subgridspec(1, 3, wspace=0.62)
vmax = np.abs(X).max()
panels = [(X, r'$\tilde X$', None, INK), (level, r'trend  $\bar r\,\mathbf{1}^\top$', '=', LEVEL),
          (C, r'periodic  $C$', '+', PHASE)]
axes = []
for i, (m, title, op, color) in enumerate(panels):
    ax = fig.add_subplot(maps[i])
    ax.imshow(m, cmap='Greys', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=7.2, pad=2, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.4)
    if i == 0:
        ax.set_ylabel('day', labelpad=1)
    if op:
        ax.text(-0.31, 0.5, op, transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=INK)
    axes.append(ax)

ax = axes[0]
ax.add_patch(Rectangle((COL - 0.5, -0.5), 1, K, fill=False, ec=PHASE, lw=1.3))
ax.add_patch(Rectangle((-0.5, ROW - 0.5), P, 1, fill=False, ec=LEVEL, lw=1.3))
axes[1].add_patch(Rectangle((-0.5, ROW - 0.5), P, 1, fill=False, ec=LEVEL, lw=1.3))
axes[2].add_patch(Rectangle((COL - 0.5, -0.5), 1, K, fill=False, ec=PHASE, lw=1.3))

fig.canvas.draw()
box0 = axes[0].get_position()
box2 = axes[2].get_position()
y_note = box0.y0 - 0.034
fig.text(box0.x0, y_note, 'column: phase token', ha='left', va='top', fontsize=SMALL, color=PHASE)
fig.text(box0.x0, y_note - 0.045, 'row mean: level', ha='left', va='top', fontsize=SMALL, color=LEVEL)
fig.text((box0.x0 + box2.x1) / 2 + 0.12, y_note, 'recurs across cycles', ha='left', va='top',
         fontsize=SMALL, color=PHASE)
fig.text((box0.x0 + box2.x1) / 2 + 0.12, y_note - 0.045, 'moves across cycles', ha='left', va='top',
         fontsize=SMALL, color=LEVEL)
panel_label(fig, '(a)', 0.012, 0.975)

ax = fig.add_subplot(outer[1])
rows = np.arange(len(names), dtype=float)
rows[len(trend_led):] += 0.45
trend = data['level'].to_numpy()
periodic = data['amplitude_shape'].to_numpy()
left = np.zeros(len(names))
for width, color, label in ((trend, LEVEL, r'trend $\bar r\,\mathbf{1}^\top$'),
                            (periodic, PHASE, r'periodic $a\,s^\top$'),
                            (data['rest'].to_numpy(), REST, r'rest $N$')):
    ax.barh(rows, width, left=left, color=color, edgecolor='white', linewidth=0.6, height=0.72,
            label=label)
    left += width
for row, t, p in zip(rows, trend, periodic):
    ax.text(1.02, row, f'{t:.0%}', ha='left', va='center', color=LEVEL, fontsize=SMALL,
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.2, row, f'{t + p:.0%}', ha='left', va='center', color=PHASE, fontsize=SMALL,
            transform=ax.get_yaxis_transform(), clip_on=False)
top = rows[-1] + 0.62
ax.text(1.02, top, 'trend', ha='left', va='bottom', color=LEVEL, fontsize=SMALL,
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.text(1.2, top, 'both', ha='left', va='bottom', color=PHASE, fontsize=SMALL,
        transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_yticks(rows)
ax.set_yticklabels(names)
ax.tick_params(axis='y', length=0, pad=2)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(['0', '50%', '100%'])
ax.set_xlabel('share of the energy of a normalized window')
ax.set_ylim(rows[0] - 0.55, rows[-1] + 0.55)
for color, lo, hi, text in ((LEVEL, rows[0], rows[len(trend_led) - 1], 'trend-led'),
                            (PHASE, rows[len(trend_led)], rows[-1], 'periodic-led')):
    ax.annotate('', xy=(-0.255, lo - 0.3), xytext=(-0.255, hi + 0.3),
                xycoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='-', color=color, lw=0.8))
    ax.text(-0.29, (lo + hi) / 2, text, transform=ax.get_yaxis_transform(), rotation=90,
            ha='center', va='center', fontsize=SMALL, color=color)
ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.02, 1.0), frameon=False, borderaxespad=0.2,
          fontsize=SMALL, handlelength=0.9, columnspacing=0.7, handletextpad=0.4)
panel_label(fig, '(b)', 0.012, 0.545)

fig.savefig(dst)
share = (level ** 2).sum() / (X ** 2).sum()
print(f'window trend share {share:.3f}; periodic {1 - share:.3f}; wrote {dst}')
