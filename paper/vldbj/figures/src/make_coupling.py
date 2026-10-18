"""Fig. 4 of Section 3.3: level-amplitude coupling (Insight 3).

(a) The standardized daily level and amplitude of thirty days of one Traffic
    sensor (sensor 175, from Monday 2016-10-03, test period), whose correlation
    (0.93) is at the Traffic median.
(b) Sign-consistent coupling of the test windows of every dataset, with the
    shifted-pairing reference drawn as the chance level (both from
    coupling.csv, scripts/vldbj_m1).

    python make_coupling.py <all_datasets> [out.pdf] [preview.png]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import AMP, INK, LEVEL, MUTED, SMALL, apply, panel_label

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
data_root = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, '..', 'coupling.pdf')

START, SENSOR, P, K = 15384, 175, 24, 30
NETWORKS = ('Electricity', 'Traffic')
apply()

frame = pd.read_csv(os.path.join(data_root, 'traffic', 'traffic.csv'))
dates = pd.to_datetime(frame.iloc[:, 0])
assert dates.iloc[START].hour == 0 and dates.iloc[START].dayofweek == 0
x = frame.iloc[START:START + K * P, 1 + SENSOR].to_numpy('float64')
cycles = x.reshape(K, P)
level = cycles.mean(1)
amp = np.sqrt(((cycles - level[:, None]) ** 2).mean(1))
r = np.corrcoef(level, amp)[0, 1]

table = pd.read_csv(os.path.join(root, 'research_runs/vldbj_m1_coupling/coupling.csv'))
test = table[(table.split == 'test') & ~table.dataset.str.startswith('synthetic')]
test = test.set_index('dataset').sort_values('consistent_coupling')

fig = plt.figure(figsize=(3.35, 2.5))
outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.4], hspace=0.6,
                         left=0.235, right=0.975, top=0.9, bottom=0.12)
ax_stats = fig.add_subplot(outer[0])
ax_bars = fig.add_subplot(outer[1])


def z(u):
    return (u - u.mean()) / u.std()


days = np.arange(K) + 1
ax_stats.plot(days, z(level), color=LEVEL, lw=1.4, marker='o', ms=2.4, mec='none',
              label='level of the day', zorder=3)
ax_stats.plot(days, z(amp), color=AMP, lw=1.1, marker='s', ms=2.0, mec='none',
              label='amplitude of the day', zorder=4)
ax_stats.set_yticks([])
ax_stats.spines['left'].set_visible(False)
ax_stats.set_xlim(0.5, K + 0.5)
ax_stats.set_xticks([1, 8, 15, 22, 29])
ax_stats.set_xlabel('day (one Traffic sensor)', labelpad=1)
ax_stats.text(0.995, 1.02, f'$r$ = {r:.2f}', transform=ax_stats.transAxes, ha='right',
              va='bottom', fontsize=7, color=INK)
ax_stats.legend(loc='lower left', bbox_to_anchor=(0.0, 0.98), ncol=2, frameon=False,
                handlelength=1.4, columnspacing=1.0, borderaxespad=0, handletextpad=0.4,
                fontsize=SMALL)

names = list(test.index)
rows = np.arange(len(names))
values = test['consistent_coupling'].to_numpy()
reference = test['shifted_pairing'].to_numpy()
ax_bars.barh(rows, values, height=0.72, zorder=2,
             color=[AMP if name in NETWORKS else '#EBA27A' for name in names])
ax_bars.plot(reference, rows, ls='none', marker='|', ms=7, mew=1.3, color=MUTED, zorder=4,
             label='chance level')
for row, value, name in zip(rows, values, names):
    ax_bars.text(value + 0.015, row, f'{value:.2f}', va='center', fontsize=SMALL, color=INK,
                 fontweight='bold' if name in NETWORKS else 'normal')
ax_bars.set_ylim(-0.55, len(names) - 0.45)
ax_bars.set_yticks(rows)
ax_bars.set_yticklabels(names)
ax_bars.tick_params(axis='y', length=0, pad=2)
ax_bars.set_xlim(0, 1)
ax_bars.set_xticks([0, 0.5, 1])
ax_bars.set_xticklabels(['0', '0.5', '1'])
ax_bars.set_xlabel('level–amplitude coupling', labelpad=1)
ax_bars.legend(loc='lower right', frameon=False, fontsize=SMALL, handlelength=1.0,
               handletextpad=0.4, borderaxespad=0.2)

panel_label(fig, '(a)', 0.012, 0.965)
panel_label(fig, '(b)', 0.012, 0.56)

fig.savefig(dst)
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=400)
print(f'corr = {r:.3f}; wrote {dst}')
