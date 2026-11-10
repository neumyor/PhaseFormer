"""Fig. 2 of Section 3.2: periodicity and trend are low-dimensional (single-column figure).

Data: research_runs/section3_pca_curves_v1/section3_pca_curves.json (scripts/section3_pca_curves.py),
windows of every channel of the seven datasets. Cumulative share of the variance captured by the
leading r principal directions of
  - the phase tokens of the periodic term of a window (averaged over windows),
  - the level token (PCA over windows, averaged over channels),
  - the patch tokens of the periodic term (reference; averaged over windows).
Lines: mean over the seven datasets. Bars at r = 2 and 3 (the values quoted in the text):
range over the seven datasets.

    python3 make_pca_curves.py [out.pdf]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figstyle import INK, LEVEL, MUTED, PATCH, PHASE, apply

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'pca_curves.pdf')
apply()

data = json.load(open(os.path.join(root, 'research_runs/section3_pca_curves_v1/section3_pca_curves.json')))
R = 10
r = np.arange(1, R + 1)
curves = {k: 100 * np.array([data[n][k][:R] for n in data]) for k in ('phase', 'level', 'patch')}

series = [  # key, color, marker, line style, dodge of the range bar, legend label
    ('phase', PHASE, 'o', '-', -0.2, 'Phase tokens (periodic term)'),
    ('level', LEVEL, 's', '-', 0.0, 'Level token (trend)'),
    ('patch', PATCH, 'D', (0, (3.2, 1.6)), 0.2, 'Patch tokens (reference)'),
]

fig, ax = plt.subplots(figsize=(3.4, 2.45))
fig.subplots_adjust(left=0.135, right=0.975, top=0.975, bottom=0.175)
for k in (2, 3):
    ax.axvspan(k - 0.5, k + 0.5, color='0.95', lw=0, zorder=0)
ax.axhline(90, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.text(1.0, 91.2, '90%', fontsize=6.5, color=MUTED, ha='left', va='bottom')
for key, color, marker, ls, dx, label in series:
    c = curves[key]
    ax.plot(r, c.mean(0), color=color, lw=1.2, ls=ls, zorder=3, label=label,
            marker=marker, ms=3.4 if marker != 'D' else 3.0, mec='white', mew=0.5)
    for k in (2, 3):
        lo, hi = c[:, k - 1].min(), c[:, k - 1].max()
        ax.vlines(k + dx, lo, hi, color=color, lw=1.6, alpha=0.55, zorder=2, capstyle='round')
ax.set_xlim(0.5, R + 0.5)
ax.set_ylim(20, 100)
ax.set_xticks(r)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_xlabel('Number of principal directions')
ax.set_ylabel('Cumulative share of variance (%)')
ax.grid(axis='y', ls=':', lw=0.4, color='0.88', zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.04), frameon=False, fontsize=6.8, handlelength=2.2,
          labelspacing=0.5, borderaxespad=0.2)
fig.savefig(dst)
print('wrote', dst)
