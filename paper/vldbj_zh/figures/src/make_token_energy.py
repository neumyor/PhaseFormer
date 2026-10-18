"""Fig. 2 of Section 3.1: a window is a level term plus an amplitude-scaled shape.

(a) The Traffic window of Fig. 4(a) (sensor 175, thirty days from Monday
    2016-10-03), normalized and folded, split into the level term l 1^T, the
    amplitude-scaled shape a s^T (leading singular pair of the within-cycle
    part) and the rest N.
(b) Mean energy shares of the three terms over all test windows of every
    dataset (token_energy.csv, scripts/vldbj_m1/token_energy.py).

    python make_token_energy.py <all_datasets> [out.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
data_root = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, '..', 'token_energy.pdf')

START, SENSOR, P, K = 15384, 175, 24, 30
LEVEL = '#1F5FA8'
SHAPE = '#E07B39'
REST = '#C9CDD2'
INK = '#222222'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelpad': 2,
})

frame = pd.read_csv(os.path.join(data_root, 'traffic', 'traffic.csv'))
x = frame.iloc[START:START + K * P, 1 + SENSOR].to_numpy('float64')
x = (x - x.mean()) / x.std()
X = x.reshape(K, P)
level = X.mean(1, keepdims=True) * np.ones((1, P))
within = X - level
u, sv, vt = np.linalg.svd(within, full_matrices=False)
amp_shape = sv[0] * np.outer(u[:, 0], vt[0])
rest = X - level - amp_shape
share = 1 - (rest ** 2).sum() / (X ** 2).sum()

table = pd.read_csv(os.path.join(root, 'research_runs/vldbj_token_energy/token_energy.csv'))
names = list(table['dataset'])[::-1]
data = table.set_index('dataset').loc[names]

fig = plt.figure(figsize=(3.35, 3.05))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.42,
                         left=0.17, right=0.97, top=0.95, bottom=0.11)
maps = outer[0].subgridspec(1, 4, wspace=0.55)
vmax = np.abs(X).max()
panels = [(X, '$X$', None), (level, '$\\ell\\,\\mathbf{1}^\\top$', '='),
          (amp_shape, '$a\\,s^\\top$', '+'), (rest, '$N$', '+')]
for i, (m, title, op) in enumerate(panels):
    ax = fig.add_subplot(maps[i])
    ax.imshow(m, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto',
              interpolation='nearest')
    ax.set_title(title, fontsize=7.5, pad=2, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.4)
    if i == 0:
        ax.set_ylabel('cycle', labelpad=1)
    ax.set_xlabel('phase', labelpad=1)
    if op:
        ax.text(-0.33, 0.5, op, transform=ax.transAxes, ha='center',
                va='center', fontsize=9, color=INK)
fig.text(0.02, 0.955, '(a)', fontsize=7.5, fontweight='bold', va='top')

ax = fig.add_subplot(outer[1])
parts = [('level', 'level $\\ell\\,\\mathbf{1}^\\top$', LEVEL),
         ('amplitude_shape', 'amplitude-scaled shape $a\\,s^\\top$', SHAPE),
         ('rest', 'rest $N$', REST)]
left = np.zeros(len(names))
for key, label, color in parts:
    width = data[key].to_numpy()
    ax.barh(names, width, left=left, color=color, edgecolor='white',
            linewidth=0.8, height=0.7, label=label)
    left += width
for i, total in enumerate(data['level'] + data['amplitude_shape']):
    ax.text(total - 0.012, i, f'{total:.0%}', ha='right', va='center',
            color='white', fontsize=6.5, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
ax.set_xlabel('share of the energy of a normalized test window')
ax.tick_params(axis='y', length=0)
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.42, 1.0), frameon=False,
          fontsize=6, handlelength=0.9, columnspacing=0.7, handletextpad=0.4)
fig.text(0.02, 0.535, '(b)', fontsize=7.5, fontweight='bold', va='top')

fig.savefig(dst)
print(f'window share {share:.3f}')
