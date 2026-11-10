"""Data figures of Section 5 (experiments), in the shared style of figstyle.py.

  exp_horizons.pdf     Fig. 5, Section 5.2: MSE and MAE of the ten models of Table 1 at each
                       horizon, one pair of panels (MSE, MAE) per dataset; full text width.
  efficiency_bars.pdf  Fig. 6, Section 5.2: FLOPs per forecast and parameters on Traffic and
                       Electricity (input 720, horizon 96), logarithmic; full text width.
  exp_trend.pdf        Fig. 7, Sections 5.3 and 5.4: (a) the average MSE_trend and MSE_periodic of
                       every model in one plane, with lines of equal MSE_total; (b, c) the change of
                       MSE_trend and MSE_periodic of every model relative to PhaseFormer on each of the
                       twelve settings; full text width.
  exp_ablation.pdf     Fig. 8, Section 5.5: (a) MSE of PhaseFormer and of the two variants relative
                       to PhaseFormer+ per dataset; (b) on how many of the 28 settings PhaseFormer+ is
                       better than, equal to or worse than each of them; one column.

Data: exp_horizon.csv, exp_trend.csv, exp_ablation.csv (extract_exp_tables.py) and
efficiency.csv (parameters and FLOPs of the supplement table params_flops_details; PhaseFormer+
measured with scripts/vldbj_m1/efficiency.py, parameters of its phase pathway, with the trend
pathway and gates adding EXTRA).

    python3 make_exp_figures.py [output dir]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter

from figstyle import (ALT, AMP, GRID, INK, LEVEL, MODEL_STYLE, MUTED, OURS, PATCH, PHASE, REST,
                      TICK, apply, panel_title)

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..')
apply()

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']
HORIZONS = [96, 192, 336, 720]
TEXT_WIDTH = 7.16          # inches, IEEE two-column text width
COLUMN_WIDTH = 3.5
OTHER = '#A6ADB5'          # "other" models of the efficiency figure
OURS_LIGHT = '#F3C4C4'
EXTRA = {'Traffic': 2297, 'Electricity': 1323}   # trend pathway and per-channel gates


def read(name):
    with open(os.path.join(here, name)) as f:
        return list(csv.DictReader(f))


def minus(v, fmt):
    return (fmt % v).replace('-', '−')


def mark_ours(texts):
    for t in texts:
        if t.get_text() == 'PhaseFormer+':
            t.set_color(OURS)
            t.set_fontweight('bold')


# ------------------------------------------------------------------ Fig. 5: per-horizon results
def horizons():
    res = defaultdict(dict)                      # (dataset, metric) -> model -> horizon -> value
    for r in read('exp_horizon.csv'):
        for metric in ('mse', 'mae'):
            res[(r['dataset'], metric)].setdefault(r['model'], {})[int(r['horizon'])] = float(r[metric])
    order = ['PhaseFormer+', 'PhaseFormer', 'SparseTSF', 'TimeBase', 'PatchTST', 'Crossformer',
             'TimeMixer', 'FITS', 'iTransformer', 'FEDformer']
    x = np.arange(len(HORIZONS))

    fig, axes = plt.subplots(4, 4, figsize=(TEXT_WIDTH, 4.5))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.962, bottom=0.058, wspace=0.3, hspace=0.52)
    for k, ds in enumerate(DATASETS):
        for j, metric in enumerate(('mse', 'mae')):
            ax = axes[k // 2, 2 * (k % 2) + j]
            vals = res[(ds, metric)]
            # the y range covers every model except the two whose errors are far off the others;
            # their curves are clipped where they exceed it
            near = [v for m in order if m not in ('Crossformer', 'FEDformer') for v in vals[m].values()]
            lo, hi = min(near), max(near)
            pad = 0.06 * (hi - lo)
            for m in reversed(order):
                color, marker, ls = MODEL_STYLE[m]
                ours = m == 'PhaseFormer+'
                ax.plot(x, [vals[m][h] for h in HORIZONS], color=color, ls=ls, marker=marker,
                        lw=1.5 if ours else 0.8, ms=3.4 if ours else 2.4, mec='white', mew=0.4,
                        zorder=5 if ours else 3)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.25, len(HORIZONS) - 0.75)
            ax.set_xticks(x, [str(h) for h in HORIZONS])
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(**GRID)
            panel_title(ax, None, f'{ds} {metric.upper()}')
            if k // 2 == 3 or (k // 2 == 2 and k % 2 == 1):
                ax.set_xlabel('Horizon', fontsize=TICK, labelpad=1)
    for ax in axes[3, 2:]:
        ax.axis('off')
    handles = [Line2D([], [], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][1], ls=MODEL_STYLE[m][2],
                      lw=1.5 if m == 'PhaseFormer+' else 0.8, ms=3.6 if m == 'PhaseFormer+' else 2.8,
                      mec='white', mew=0.4) for m in order]
    leg = fig.legend(handles, order, loc='center', ncol=2, handlelength=2.4, columnspacing=1.6,
                     labelspacing=0.55, bbox_to_anchor=(0.775, 0.115))
    mark_ours(leg.get_texts())
    fig.savefig(os.path.join(out, 'exp_horizons.pdf'))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 6: efficiency
def efficiency():
    rows = read('efficiency.csv')
    fam = {r['model']: r['family'] for r in rows}
    color = {'phase': PHASE, 'both': PHASE, 'patch': PATCH, 'other': OTHER}
    panels = [('a', 'FLOPs on Traffic', 'mflops_traffic', 'GFLOPs', 'Traffic'),
              ('b', 'Parameters on Traffic', 'params_traffic', 'Parameters', 'Traffic'),
              ('c', 'FLOPs on Electricity', 'mflops_electricity', 'GFLOPs', 'Electricity'),
              ('d', 'Parameters on Electricity', 'params_electricity', 'Parameters', 'Electricity')]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.05))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.29, wspace=0.36)
    for ax, (letter, title, col, unit, ds) in zip(axes, panels):
        flops = unit == 'GFLOPs'
        val = {r['model']: float(r[col]) / (1000 if flops else 1) for r in rows}
        total = {n: v + (EXTRA[ds] if n == 'PhaseFormer+' and not flops else 0) for n, v in val.items()}
        names = sorted(val, key=lambda n: -total[n])
        for i, n in enumerate(names):
            f = fam[n]
            edge = OURS if n == 'PhaseFormer+' else (PATCH if f == 'both' else 'none')
            ax.bar(i, val[n], 0.72, color=color[f], edgecolor=edge, lw=0.9 if edge != 'none' else 0,
                   ls=(0, (2, 1)) if f == 'both' else '-', zorder=3)
            if n == 'PhaseFormer+' and not flops:
                ax.bar(i, EXTRA[ds], 0.72, bottom=val[n], color=OURS_LIGHT, edgecolor=OURS, lw=0.7,
                       ls=(0, (2, 1)), zorder=3)
        ax.set_yscale('log')
        ax.set_ylim(*((1e-4, 1e3) if flops else (1e2, 1e8)))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(names)), names, rotation=50, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, pad=2)
        mark_ours(ax.get_xticklabels())
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel(unit)
        ax.grid(**GRID)
        panel_title(ax, letter, title)
    handles = [Patch(color=PHASE, lw=0), Patch(color=PATCH, lw=0),
               Patch(facecolor=PHASE, edgecolor=PATCH, lw=0.9, ls=(0, (2, 1))), Patch(color=OTHER, lw=0),
               Patch(facecolor=OURS_LIGHT, edgecolor=OURS, lw=0.7, ls=(0, (2, 1)))]
    labels = ['Phase-based', 'Patch-based', 'Phase and patch', 'Others',
              'PhaseFormer+ trend pathway and gates (2.3K / 1.3K)']
    fig.legend(handles, labels, loc='upper center', ncol=5, handlelength=1.3, columnspacing=1.6,
               bbox_to_anchor=(0.52, 1.0))
    fig.savefig(os.path.join(out, 'efficiency_bars.pdf'))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 7: trend and periodic errors
TREND_ROWS = [  # model key, row label
    ('DLinear', 'DLinear'), ('SparseTSF', 'SparseTSF'), ('FITS', 'FITS'), ('CycleNet', 'CycleNet'),
    ('+N-BEATS', '+ N-BEATS'), ('+ETSformer', '+ ETSformer'), ('+BasisFormer', '+ BasisFormer'),
    ('PhaseFormer+', 'PhaseFormer+'),
]
BRANCH_COLOR = {'+N-BEATS': ALT[0], '+ETSformer': ALT[1], '+BasisFormer': ALT[2]}


def trend():
    d = defaultdict(dict)                        # setting -> model -> (trend, periodic)
    settings = []
    for r in read('exp_trend.csv'):
        t, tot = float(r['mse_trend']), float(r['mse_total'])
        d[r['setting']][r['model']] = np.array([t, tot - t])
        if r['setting'] not in settings:
            settings.append(r['setting'])
    models = [m for m, _ in TREND_ROWS] + ['PhaseFormer']
    avg = {m: np.mean([d[s][m] for s in settings], axis=0) for m in models}
    # relative change with respect to PhaseFormer, per setting and for the averages
    change = {m: np.array([100 * (d[s][m] / d[s]['PhaseFormer'] - 1) for s in settings]
                          + [100 * (avg[m] / avg['PhaseFormer'] - 1)]) for m, _ in TREND_ROWS}

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.6))
    ax = fig.add_axes([0.062, 0.15, 0.25, 0.74])

    # (a) the error plane: lines of equal MSE_total, the models, and the moves away from PhaseFormer
    X0, X1, Y0, Y1 = 0.1410, 0.1695, 0.1360, 0.1548
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    fig.canvas.draw()
    p0, p1 = ax.transData.transform([(X0, Y1), (X1, Y1 - (X1 - X0))])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))      # slope of x + y = const on screen
    for c in np.arange(0.28, 0.33, 0.01):
        ax.plot([X0, X1], [c - X0, c - X1], color='0.78', lw=0.5, ls=(0, (3, 2)), zorder=1)
        if X0 + 0.001 < c - Y0 < X1 - 0.001:     # label where the line meets the bottom edge
            ax.text(c - Y0 + 0.0004, Y0 + 0.0003, f'{c:.2f}', fontsize=TICK - 1, color=MUTED, ha='left',
                    va='bottom', rotation=angle, rotation_mode='anchor')
        elif Y0 < c - X1 < Y1:                    # or the right edge
            ax.text(X1 - 0.0003, c - X1 + 0.0003, f'{c:.2f}', fontsize=TICK - 1, color=MUTED, ha='right',
                    va='bottom', rotation=angle, rotation_mode='anchor')
    ax.text(X0 + 0.0004, Y1 - 0.0004, 'dashed lines: equal $\\mathrm{MSE}_{\\mathrm{total}}$', fontsize=TICK - 1,
            color=MUTED, ha='left', va='top')
    pf = avg['PhaseFormer']
    for m in BRANCH_COLOR:
        ax.annotate('', avg[m], pf, arrowprops=dict(arrowstyle='-|>', lw=0.6, color=ALT[2],
                                                     mutation_scale=4, shrinkA=1.5, shrinkB=0.5), zorder=4)
        ax.plot(*avg[m], 'o', ms=3.0, color=BRANCH_COLOR[m], mec=INK, mew=0.3, zorder=5)
    ax.annotate('', avg['PhaseFormer+'], pf, arrowprops=dict(arrowstyle='-|>', lw=1.0, color=OURS,
                                                              mutation_scale=6, shrinkA=2, shrinkB=2.5), zorder=4)
    for m, dx, dy, ha in [('DLinear', -0.0008, 0.0, 'right'), ('SparseTSF', 0.0008, 0.0, 'left'),
                          ('FITS', 0.0008, 0.0, 'left'), ('CycleNet', 0.0008, 0.0, 'left'),
                          ('PhaseFormer', 0.0009, 0.0003, 'left'), ('PhaseFormer+', 0.0008, -0.0003, 'left')]:
        color, marker, _ = MODEL_STYLE[m]
        ours = m == 'PhaseFormer+'
        ax.plot(*avg[m], marker, ms=5.0 if ours else 4.2, color=color, mec='white', mew=0.5, zorder=6)
        ax.text(avg[m][0] + dx, avg[m][1] + dy, m, fontsize=TICK, ha=ha, va='center',
                color=OURS if ours else INK, fontweight='bold' if ours else 'normal', zorder=7)
    cluster = np.mean([avg[m] for m in BRANCH_COLOR], axis=0)
    ax.annotate('PhaseFormer\n+ a trend branch', cluster, (cluster[0] + 0.0052, cluster[1] - 0.0030),
                fontsize=TICK - 0.5, color=ALT[2], ha='left', va='center', linespacing=1.0,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=ALT[2], shrinkA=1, shrinkB=3))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel('$\\mathrm{MSE}_{\\mathrm{trend}}$')
    ax.set_ylabel('$\\mathrm{MSE}_{\\mathrm{periodic}}$')
    panel_title(ax, 'a', 'Average over the twelve settings')

    # (b, c) per-setting change relative to PhaseFormer
    cmap = LinearSegmentedColormap.from_list('change', [LEVEL, PHASE, '#FFFFFF', '#F2B48A', AMP])
    norm = Normalize(-10, 10)
    cols = len(settings) + 1
    xpos = np.r_[np.arange(len(settings)), len(settings) + 0.35]
    for i, (letter, name) in enumerate([('b', 'trend'), ('c', 'periodic')]):
        hx = fig.add_axes([0.425, 0.565 - 0.4 * i, 0.51, 0.325])
        for r, (m, _) in enumerate(TREND_ROWS):
            for c in range(cols):
                v = change[m][c, i]
                hx.add_patch(plt.Rectangle((xpos[c] - 0.5, r - 0.5), 1, 1, color=cmap(norm(v)), lw=0))
                hx.text(xpos[c], r, minus(v, '%.1f'), ha='center', va='center', fontsize=5.2,
                        color='white' if abs(v) > 6.5 else INK,
                        fontweight='bold' if m == 'PhaseFormer+' else 'normal')
        for r in (3.5, 6.5):
            hx.axhline(r, color='white', lw=1.6)
        hx.axvline(len(settings) - 0.5 + 0.175, color='white', lw=3)
        hx.set_xlim(-0.5, xpos[-1] + 0.5)
        hx.set_ylim(len(TREND_ROWS) - 0.5, -0.5)
        hx.set_yticks(range(len(TREND_ROWS)), [lab for _, lab in TREND_ROWS], fontsize=TICK - 0.3)
        mark_ours(hx.get_yticklabels())
        hx.tick_params(length=0, pad=2)
        for s in hx.spines.values():
            s.set_visible(False)
        if i == 1:
            hx.set_xticks(xpos, [s.split('-')[1] for s in settings] + ['Avg.'], fontsize=TICK - 0.5)
            for k in range(0, len(settings), 2):             # dataset names under each pair of horizons
                hx.text((xpos[k] + xpos[k + 1]) / 2, len(TREND_ROWS) + 0.75, settings[k].split('-')[0],
                        ha='center', va='top', fontsize=TICK - 0.3, clip_on=False)
        else:
            hx.set_xticks([])
        panel_title(hx, letter, f'Change of $\\mathrm{{MSE}}_{{\\mathrm{{{name}}}}}$ relative to PhaseFormer (%)')
    cax = fig.add_axes([0.948, 0.165, 0.009, 0.725])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-10, -5, 0, 5, 10])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TICK - 0.5, length=2, width=0.5, pad=1.5)
    cb.ax.set_yticklabels(['≤−10', '−5', '0', '+5', '≥+10'])
    fig.savefig(os.path.join(out, 'exp_trend.pdf'))
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 8: ablation
def ablation():
    mse = defaultdict(dict)
    per = defaultdict(dict)
    for r in read('exp_ablation.csv'):
        mse[r['dataset']].setdefault(r['model'], []).append(float(r['mse']))
        per[(r['dataset'], r['horizon'])][r['model']] = float(r['mse'])
    models = ['PhaseFormer', 'Single gate', 'Linear']
    colors = [PHASE, ALT[1], ALT[2]]
    labels = ['PhaseFormer (no trend pathway)', 'Single gate', 'Linear layer']
    mean = {ds: {m: np.mean(v) for m, v in mse[ds].items()} for ds in DATASETS}
    allavg = {m: np.mean([mean[ds][m] for ds in DATASETS]) for m in models + ['PhaseFormer+']}
    gap = {m: [100 * (mean[ds][m] / mean[ds]['PhaseFormer+'] - 1) for ds in DATASETS]
           + [100 * (allavg[m] / allavg['PhaseFormer+'] - 1)] for m in models}
    ticks = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Elec.', 'Traffic', 'Avg.']
    x = np.arange(len(ticks)) + np.r_[np.zeros(len(DATASETS)), 0.35]
    w = 0.26

    fig = plt.figure(figsize=(COLUMN_WIDTH, 2.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.4], left=0.19, right=0.985, top=0.87,
                          bottom=0.13, hspace=0.6)
    ax = fig.add_subplot(gs[0])
    for k, (m, c) in enumerate(zip(models, colors)):
        ax.bar(x + (k - 1) * w, gap[m], w, color=c, lw=0, zorder=3)
        ax.text(x[-1] + (k - 1) * w, gap[m][-1] + 0.1, f'{gap[m][-1]:.1f}', fontsize=TICK - 1.0,
                color=INK, ha='center', va='bottom', rotation=90)
    ax.axhline(0, color=INK, lw=0.6, zorder=4)
    ax.axvline(len(DATASETS) - 0.5 + 0.175, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, ticks)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_ylabel('MSE above PhaseFormer+ (%)', fontsize=TICK)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(0, max(max(v) for v in gap.values()) + 0.6)
    ax.grid(**GRID)
    panel_title(ax, 'a', 'MSE averaged over the horizons of each dataset')
    fig.legend([Patch(color=c, lw=0) for c in colors], labels, loc='upper center', ncol=3,
               handlelength=1.2, columnspacing=1.0, fontsize=TICK, bbox_to_anchor=(0.56, 1.0))

    # (b) outcome on the 28 settings
    bx = fig.add_subplot(gs[1])
    seg_colors = [OURS, REST, ALT[1]]
    for k, m in enumerate(models):
        better = sum(v[m] > v['PhaseFormer+'] for v in per.values())
        tie = sum(v[m] == v['PhaseFormer+'] for v in per.values())
        left = 0
        for n, c in zip((better, tie, len(per) - better - tie), seg_colors):
            if n:
                bx.barh(k, n, 0.66, left=left, color=c, lw=0)
                bx.text(left + n / 2, k, str(n), ha='center', va='center', fontsize=TICK - 0.5,
                        color='white' if c != REST else INK)
            left += n
    bx.set_yticks(range(len(models)), ['PhaseFormer', 'Single gate', 'Linear layer'])
    bx.set_ylim(len(models) - 0.5, -0.5)
    bx.set_xlim(0, len(per))
    bx.set_xticks([0, 7, 14, 21, 28])
    bx.tick_params(axis='y', length=0)
    bx.spines['left'].set_visible(False)
    panel_title(bx, 'b', 'Settings, of 28, on which PhaseFormer+ is')
    bx.legend([Patch(color=c, lw=0) for c in seg_colors], ['better', 'equal', 'worse'],
              loc='upper center', bbox_to_anchor=(0.45, -0.32), ncol=3, fontsize=TICK, handlelength=1.0,
              columnspacing=1.2)
    fig.savefig(os.path.join(out, 'exp_ablation.pdf'))
    plt.close(fig)


horizons()
efficiency()
trend()
ablation()
