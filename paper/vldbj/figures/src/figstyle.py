"""Shared style of the data figures of Section 3 (Figs. 2 to 4).

One palette with fixed meanings across the three figures:
  LEVEL   the level (term, curve)               navy
  AMP     the amplitude / amplitude-scaled shape orange
  PHASE   phase tokens                           blue of Fig. 1
  PATCH   patch tokens                           green of Fig. 1
  REST    rest, reference, context               light gray
"""
import matplotlib.pyplot as plt

LEVEL = '#1B3A6B'
AMP = '#E07B39'
PHASE = '#1F9FD8'
PATCH = '#3E9B2A'
REST = '#CED3D8'
INK = '#222222'
MUTED = '#6B7078'
SMALL = 6.5      # legends, value labels, annotations
LABEL = 7.5      # panel labels


def apply():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'pdf.fonttype': 42,
        'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelpad': 2,
    })


def panel_label(fig, text, x, y):
    """Bold panel label at figure coordinates (x, y), top-left anchored."""
    fig.text(x, y, text, fontsize=LABEL, fontweight='bold', va='top', ha='left', color=INK)
