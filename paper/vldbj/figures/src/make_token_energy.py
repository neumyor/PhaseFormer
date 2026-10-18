"""Fig. 2 of Section 3.1: a window is a level term plus an amplitude-scaled shape.

(a) One normalized Traffic test window (sensor 345, thirty days from Monday
    2016-08-29), folded with one row per day, split into the level term
    l 1^T, the amplitude-scaled shape a s^T (leading singular pair of the
    within-cycle part) and the rest N.
(b) Mean energy shares of the three terms over all test windows of every
    dataset (token_energy.csv, scripts/vldbj_m1/token_energy.py), grouped by
    the term that dominates.

    python make_token_energy.py <all_datasets> [out.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import AMP, INK, LEVEL, MUTED, REST, SMALL, apply, panel_label

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
data_root = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, '..', 'token_energy.pdf')
apply()

START, SENSOR, P, K = 15048, 345, 24, 30

frame = pd.read_csv(os.path.join(data_root, 'traffic', 'traffic.csv'))
dates = pd.to_datetime(frame.iloc[:, 0])
assert dates.iloc[START].hour == 0 and dates.iloc[START].dayofweek == 0
x = frame.iloc[START:START + K * P, 1 + SENSOR].to_numpy('float64')
x = (x - x.mean()) / x.std()
X = x.reshape(K, P)
level = X.mean(1, keepdims=True) * np.ones((1, P))
u, sv, vt = np.linalg.svd(X - level, full_matrices=False)
amp_shape = sv[0] * np.outer(u[:, 0], vt[0])
rest = X - level - amp_shape

table = pd.read_csv(os.path.join(root, 'research_runs/vldbj_token_energy/token_energy.csv'))
data = table.set_index('dataset')
shape_led = ['Traffic', 'Electricity', 'ETTh1']
level_led = ['Weather', 'ETTm2', 'ETTm1', 'ETTh2']
names = (shape_led + level_led)[::-1]
data = data.loc[names]

fig = plt.figure(figsize=(3.35, 2.6))
outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.4], hspace=0.42,
                         left=0.235, right=0.89, top=0.93, bottom=0.125)
maps = outer[0].subgridspec(1, 4, wspace=0.5)
vmax = np.abs(X).max()
panels = [(X, '$X$', None, INK), (level, r'$\ell\,\mathbf{1}^\top$', '=', LEVEL),
          (amp_shape, r'$a\,s^\top$', '+', AMP), (rest, '$N$', '+', MUTED)]
for i, (m, title, op, color) in enumerate(panels):
    ax = fig.add_subplot(maps[i])
    ax.imshow(m, cmap='Greys', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=7.5, pad=2, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.4)
    if i == 0:
        ax.set_ylabel('day', labelpad=1)
    if op:
        ax.text(-0.3, 0.5, op, transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=INK)
panel_label(fig, '(a)', 0.012, 0.965)
box = ax.get_position()
fig.text((0.235 + 0.89) / 2 + 0.02, box.y0 - 0.012, 'hour of the day', ha='center', va='top', color=INK)

ax = fig.add_subplot(outer[1])
rows = np.arange(len(names), dtype=float)
rows[len(level_led):] += 0.45          # gap between the two groups
left = np.zeros(len(names))
for key, label, color in (('level', r'level $\ell\,\mathbf{1}^\top$', LEVEL),
                          ('amplitude_shape', r'amplitude-scaled shape $a\,s^\top$', AMP),
                          ('rest', 'rest $N$', REST)):
    width = data[key].to_numpy()
    ax.barh(rows, width, left=left, color=color, edgecolor='white', linewidth=0.6,
            height=0.72, label=label)
    left += width
for row, total in zip(rows, data['level'] + data['amplitude_shape']):
    ax.text(1.015, row, f'{total:.0%}', ha='left', va='center', color=INK,
            fontsize=SMALL, transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_yticks(rows)
ax.set_yticklabels(names)
ax.tick_params(axis='y', length=0, pad=2)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(['0', '50%', '100%'])
ax.set_xlabel('share of the energy of a normalized window')
ax.set_ylim(rows[0] - 0.55, rows[-1] + 0.55)
for color, lo, hi, text in ((LEVEL, rows[0], rows[len(level_led) - 1], 'level-led'),
                            (AMP, rows[len(level_led)], rows[-1], 'shape-led')):
    ax.annotate('', xy=(-0.255, lo - 0.3), xytext=(-0.255, hi + 0.3),
                xycoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='-', color=color, lw=0.8))
    ax.text(-0.29, (lo + hi) / 2, text, transform=ax.get_yaxis_transform(), rotation=90,
            ha='center', va='center', fontsize=SMALL, color=color)
ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.02, 1.0), frameon=False, borderaxespad=0.2,
          fontsize=SMALL, handlelength=0.9, columnspacing=0.7, handletextpad=0.4)
panel_label(fig, '(b)', 0.012, 0.6)

fig.savefig(dst)
share = 1 - (rest ** 2).sum() / (X ** 2).sum()
print(f'window share {share:.3f}; level {(level**2).sum()/(X**2).sum():.3f}')
