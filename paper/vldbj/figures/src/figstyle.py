"""Shared style of the data figures of the paper (Figs. 2 to 8).

One palette with fixed meanings across all figures, the framework diagram included:
  PHASE   phase tokens, the periodic term, the phase pathway, PhaseFormer   sky blue
  LEVEL   the level token, the trend term, the trend pathway                navy
  PATCH   patch tokens and patch-based models                               green
  OURS    PhaseFormer+                                                      red
  ALT     alternatives to the trend pathway (branches, variants)            grays
  REST    rest, reference, context                                          light gray
One type scale: tick labels and annotations TICK, axis labels and legends FONT, panel
titles TITLE; panel titles are left-aligned, "(a)" in bold followed by the title.
"""
import matplotlib.pyplot as plt

LEVEL = '#1B3A6B'
PHASE = '#1F9FD8'
PHASE_DARK = '#137DB0'   # phase-blue text on light backgrounds
PATCH = '#3E9B2A'
OURS = '#D62728'
AMP = '#E07B39'          # diverging counterpart of PHASE in heatmaps (an increase of the error)
REST = '#CED3D8'
INK = '#222222'
MUTED = '#6B7078'
ALT = ['#C9CDD2', '#8D939A', '#4E545C']

# models of Section 5, by family as in the efficiency figure: phase-based blue,
# patch-based green, the other models gray; PhaseFormer+ red
MODEL_STYLE = {  # name: (color, marker, line style)
    'PhaseFormer+': (OURS, 'o', '-'),
    'PhaseFormer': (PHASE, 's', '-'),
    'SparseTSF': ('#0B5C8C', 'v', '-'),
    'TimeBase': ('#5FB3C9', '^', (0, (3, 1.5))),
    'PatchTST': (PATCH, 'D', '-'),
    'Crossformer': ('#8CC63F', 'P', '-'),
    'TimeMixer': ('#1F5E14', 'X', '-'),
    'FITS': ('#3A3F45', '<', '-'),
    'iTransformer': ('#7F858C', '>', '-'),
    'FEDformer': ('#B4B9BF', 'h', '-'),
    'DLinear': ('#A6ADB5', 'p', '-'),
    'CycleNet': ('#5C6670', '*', '-'),
}

TICK = 6.5       # tick labels, value labels, annotations
FONT = 7         # axis labels, legends
TITLE = 7.5      # panel titles
SMALL = TICK
LABEL = TITLE
GRID = dict(axis='y', ls=':', lw=0.45, color='0.82', zorder=0)


def apply():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': FONT, 'pdf.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'axes.labelsize': FONT,
        'axes.titlesize': TITLE, 'axes.titlepad': 3, 'axes.labelpad': 2,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
        'legend.frameon': False, 'legend.fontsize': FONT,
    })


def panel_title(ax, letter, text):
    """Left-aligned panel title: bold "(a)" followed by the title."""
    head = f'$\\mathbf{{({letter})}}$ ' if letter else ''
    ax.set_title(head + text, loc='left', color=INK, fontsize=TITLE)


def panel_label(fig, text, x, y):
    """Bold panel label at figure coordinates (x, y), top-left anchored."""
    fig.text(x, y, text, fontsize=TITLE, fontweight='bold', va='top', ha='left', color=INK)
