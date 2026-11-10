"""Fig. 2 of Section 3.1: the two token types of PhaseFormer+ on one real window.

The first four days (Monday 2016-08-22 to Thursday 2016-08-25) of the 30-day test window of
Traffic sensor 580 stored in traffic_window_sensor580.csv, z-normalized over all 720 steps as the
model sees it: black series, dashed gray cycle boundaries,
  phase token (PHASE): the values at one hour of every day, joined by a line;
  level token (LEVEL): the day means (horizontal bars).
Included at 0.48 of the text width.

    python3 make_two_tokens.py [out.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import INK, LEVEL, PHASE, PHASE_DARK, apply

here = os.path.dirname(os.path.abspath(__file__))
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'two_tokens.pdf')
apply()

P, K = 24, 4
HOUR = 16                                    # hour of the phase token shown (evening peak)
frame = pd.read_csv(os.path.join(here, 'traffic_window_sensor580.csv'))
assert len(frame) == 30 * P
dates = pd.to_datetime(frame['date'])[:K * P]
x = frame['value'].to_numpy('float64')
z = ((x - x.mean()) / x.std())[:K * P]
X = z.reshape(K, P)
means = X.mean(1)

fig, ax = plt.subplots(figsize=(3.44, 1.72))
fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.12)
t = np.arange(K * P)
for k in range(1, K):
    ax.axvline(k * P - 0.5, color='0.72', lw=0.6, ls=(0, (3, 2)), zorder=1)
for k in range(K):
    ax.plot([k * P, k * P + P - 1], [means[k], means[k]], color=LEVEL, lw=2.0, solid_capstyle='butt',
            zorder=2)
ax.plot(t, z, color=INK, lw=0.7, zorder=3)
ax.plot([k * P + HOUR for k in range(K)], X[:, HOUR], color=PHASE, lw=1.4, marker='o', ms=4.2,
        mec=INK, mew=0.5, zorder=4)
ax.set_xlim(-1, K * P)
ax.set_ylim(-1.2, 4.5)
ax.set_yticks([-1, 0, 1, 2, 3, 4])
ax.set_xticks([k * P + (P - 1) / 2 for k in range(K)], [d.strftime('%a %m-%d') for d in dates.iloc[::P]])
ax.tick_params(axis='x', length=0, pad=2)
handles = [plt.Line2D([], [], color=PHASE, lw=1.4, marker='o', ms=4.2, mec=INK, mew=0.5),
           plt.Line2D([], [], color=LEVEL, lw=2.0)]
leg = fig.legend(handles, ['phase token $\\tilde{x}_p$', 'level token $\\bar{r}_k$'], loc='upper center',
                 ncol=2, columnspacing=2.5, handlelength=2.0, bbox_to_anchor=(0.53, 1.0))
for txt, c in zip(leg.get_texts(), (PHASE_DARK, LEVEL)):
    txt.set_color(c)
    txt.set_fontweight('bold')
fig.savefig(dst)
print('wrote', dst)
