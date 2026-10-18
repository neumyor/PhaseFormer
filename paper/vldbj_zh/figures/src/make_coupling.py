"""Fig. 3 of Section 3.2: level-amplitude coupling (Insight 3).

(a) Thirty days of one Traffic sensor (sensor 175, from Monday 2016-10-03,
    test period) whose correlation (0.93) is at the Traffic median; below,
    the standardized daily level and amplitude.
(b) Sign-consistent coupling of the test windows of every dataset, with the
    shifted-pairing reference (both from coupling.csv, scripts/vldbj_m1).

    python make_coupling.py <all_datasets> [out.pdf] [preview.png]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, '../../../..'))
data_root = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, '..', 'coupling.pdf')

START, SENSOR, P, K = 15384, 175, 24, 30
NETWORKS = ('Electricity', 'Traffic')

BLUE = '#1F9FD8'       # phase-token blue of Figs. 1 and 2
DARK = '#1F5FA8'
RED = '#D62728'
OTHER = '#B4BEC9'
SHADE = '#EDEDED'
INK = '#222222'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelpad': 2,
})

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

fig = plt.figure(figsize=(3.35, 2.6))
outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.25], hspace=0.5,
                         left=0.175, right=0.965, top=0.93, bottom=0.115)
top = outer[0].subgridspec(2, 1, height_ratios=[1, 0.9], hspace=0.1)
ax_series = fig.add_subplot(top[0])
ax_stats = fig.add_subplot(top[1], sharex=ax_series)
ax_bars = fig.add_subplot(outer[1])

for k in range(K):
    if k % 7 in (5, 6):
        for ax in (ax_series, ax_stats):
            ax.axvspan(k, k + 1, color=SHADE, lw=0, zorder=0)

ax_series.plot(np.arange(K * P) / P, x, color='black', lw=0.45)
ax_series.set_ylabel('occupancy')
ax_series.set_yticks([0, 0.2])
ax_series.set_yticklabels(['0', '0.2'])
ax_series.tick_params(labelbottom=False)


def z(u):
    return (u - u.mean()) / u.std()


days = np.arange(K) + 0.5
ax_stats.plot(days, z(level), color=BLUE, lw=1.4, marker='o', ms=2.4, mec='none',
              label='cycle level', zorder=3)
ax_stats.plot(days, z(amp), color=RED, lw=0.9, marker='s', ms=1.9, mec='none',
              label='amplitude (RMS)', zorder=4)
low = min(z(level).min(), z(amp).min())
high = max(z(level).max(), z(amp).max())
ax_stats.set_ylim(low - 0.25, high + 0.25)
ax_stats.set_yticks([])
ax_stats.spines['left'].set_visible(False)
ax_stats.set_ylabel('z-score')
ax_stats.set_xlim(0, K)
ax_stats.set_xticks([0, 7, 14, 21, 28])
ax_stats.set_xlabel('day', labelpad=1)
ax_stats.text(0.4, low - 0.1, f'$r$ = {r:.2f}', ha='left', va='bottom', fontsize=7.5, color=INK)
handles, labels = ax_stats.get_legend_handles_labels()
handles.append(mpatches.Patch(color=SHADE, lw=0))
labels.append('weekend')
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.965, 0.935), ncol=3,
           frameon=False, handlelength=1.4, columnspacing=0.9, borderaxespad=0,
           handletextpad=0.4, fontsize=6.5)

names = list(test.index)
rows = np.arange(len(names))
values = test['consistent_coupling'].to_numpy()
reference = test['shifted_pairing'].to_numpy()
ax_bars.barh(rows, values, height=0.66, zorder=2,
             color=[DARK if name in NETWORKS else OTHER for name in names])
ax_bars.scatter(reference, rows, marker='|', s=46, linewidths=1.3, color=RED, zorder=4)
for row, value, name in zip(rows, values, names):
    ax_bars.text(value + 0.015, row, f'{value:.2f}', va='center', fontsize=6.5, color=INK,
                 fontweight='bold' if name in NETWORKS else 'normal')
ax_bars.set_ylim(-0.55, len(names) - 0.45)
ax_bars.set_yticks(rows)
ax_bars.set_yticklabels(names)
ax_bars.tick_params(axis='y', length=0, pad=2)
ax_bars.set_xlim(0, 1)
ax_bars.set_xticks([0, 0.5, 1])
ax_bars.set_xticklabels(['0', '0.5', '1'])
ax_bars.set_xlabel('sign-consistent coupling', labelpad=1)
ax_bars.legend([Line2D([], [], color=RED, marker='|', ls='none', ms=6, mew=1.3)],
               ['shifted pairing (reference)'], loc='lower right', frameon=False,
               fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)

ax_series.text(-0.215, 1.13, '(a)', transform=ax_series.transAxes, fontsize=8,
               fontweight='bold', va='bottom')
ax_bars.text(-0.215, 1.0, '(b)', transform=ax_bars.transAxes, fontsize=8,
             fontweight='bold', va='bottom')
fig.align_ylabels([ax_series, ax_stats])

fig.savefig(dst)
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=400)
print(f'corr = {r:.3f}; wrote {dst}')
