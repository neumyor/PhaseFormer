"""Learned gates under one common configuration against the trend energy share."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/Users/yimingniu/Code/PhaseFormer/paper/vldbj/figures/src')
from figstyle import INK, LEVEL, PHASE, TICK, apply, panel_title

apply()
gates = json.load(open('/Users/yimingniu/Code/PhaseFormer/paper/vldbj/figures/src/neutral_gates.json'))
share = {'Traffic': 8, 'Electricity': 10, 'ETTh1': 28, 'ETTh2': 52, 'ETTm1': 60, 'ETTm2': 63, 'Weather': 80}
OUT = Path('/Users/yimingniu/Code/PhaseFormer-overleaf/figures/gates.pdf')

fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True)
fig.subplots_adjust(left=0.15, right=0.99, top=0.84, bottom=0.19, wspace=0.42)
for ax, key, letter, title, color in ((axes[0], 'level', 'a', 'Level gate $g_\\ell$', LEVEL),
                                       (axes[1], 'shape', 'b', 'Shape gate $g_s$', PHASE)):
    for h, marker, label in ((96, 'o', '$H=96$'), (720, 's', '$H=720$')):
        pts = [(share[r['dataset']], r[key]) for r in gates if r['horizon'] == h]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, ms=3.4, lw=1.0,
                color=color, mec='white', mew=0.5, label=label, alpha=0.9 if h == 96 else 0.55)
    ax.set_xlabel('Trend energy share (%)')
    panel_title(ax, letter, title)
    ax.set_xlim(0, 88)
axes[0].set_ylabel('Gate value (sigmoid)')
axes[0].legend(frameon=False, loc='upper left', fontsize=TICK - 0.5, handlelength=1.2)
fig.savefig(OUT)
print('wrote', OUT)
