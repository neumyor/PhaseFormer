"""Fig. 3 of Section 3.2: phase tokens stay in place and concentrate in two dimensions.

(a) Drift on Traffic: MMD between the tokens of each later week and those of the
    first week, averaged over the channels (drift.csv).
(b) Share of the variance explained by the first two principal components,
    Traffic and Electricity (pca.csv).
Both from scripts/vldbj_m1/token_geometry.py.

    python make_token_geometry.py [out.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import INK, PATCH, PHASE, SMALL, apply, panel_label

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'token_geometry.pdf')
runs = os.path.join(root, 'research_runs/vldbj_token_geometry')
apply()

drift = pd.read_csv(os.path.join(runs, 'drift.csv'))
pca = pd.read_csv(os.path.join(runs, 'pca.csv'))

fig = plt.figure(figsize=(3.35, 1.55))
grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.5,
                        left=0.13, right=0.985, top=0.86, bottom=0.235)
ax_d = fig.add_subplot(grid[0])
ax_p = fig.add_subplot(grid[1])

traffic = drift[drift.dataset == 'Traffic']
curves = {k: traffic[traffic.token == k].groupby('week').mmd2.mean() for k in ('patch', 'phase')}
cut = 1 - curves['phase'].mean() / curves['patch'].mean()
weeks = curves['patch'].index
ax_d.fill_between(weeks, curves['phase'].values, curves['patch'].values, color=PATCH,
                  alpha=0.12, lw=0)
ax_d.plot(weeks, curves['patch'].values, color=PATCH, lw=1.1, ls=(0, (3, 1.2)),
          label='patch tokens')
ax_d.plot(weeks, curves['phase'].values, color=PHASE, lw=1.3,
          label=f'phase tokens ($-${cut:.0%})')
ax_d.set_xlabel('week (Traffic)')
ax_d.set_ylabel(r'MMD$^2$ to week 1')
ax_d.set_ylim(0, 0.3)
ax_d.set_yticks([0, 0.1, 0.2])
ax_d.set_yticklabels(['0', '0.1', '0.2'])
ax_d.legend(loc='upper left', ncol=1, frameon=False, handlelength=1.6, borderaxespad=0.1,
            fontsize=SMALL, labelspacing=0.25)

names = ['Traffic', 'Electricity']
x = np.arange(len(names))
w = 0.36
for off, kind, color in ((-w / 2, 'patch', PATCH), (w / 2, 'phase', PHASE)):
    top2 = [pca[(pca.dataset == n) & (pca.token == kind) & (pca.component == 2)]
            .cumulative.mean() for n in names]
    ax_p.bar(x + off, top2, width=w, color=color)
    for xi, v in zip(x + off, top2):
        ax_p.text(xi, v + 0.02, f'{v:.0%}', ha='center', va='bottom', fontsize=SMALL,
                  color=INK, fontweight='bold' if kind == 'phase' else 'normal')
ax_p.set_xticks(x)
ax_p.set_xticklabels(names)
ax_p.tick_params(axis='x', length=0)
ax_p.set_ylim(0, 1)
ax_p.set_yticks([0, 0.5, 1])
ax_p.set_yticklabels(['0', '50%', '100%'])
ax_p.set_ylabel('variance in 2 PCs', labelpad=1)

panel_label(fig, '(a)', 0.012, 0.985)
panel_label(fig, '(b)', 0.6, 0.985)
fig.savefig(dst)
print(f'drift reduction {cut:.3f}; wrote {dst}')
