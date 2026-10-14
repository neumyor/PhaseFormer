"""Efficiency figures as PowerPoint slides: Fig. 1(b) and the FLOPs/parameter bars.

Data: the parameter and FLOPs table of the electronic supplement
(Table params_flops_details; FLOPs per forecast in MFLOPs, input 720, horizon 96)
and the Traffic MAE averaged over the four horizons (Table main_results).
PhaseFormer+ is drawn with the 1.156K parameters of its phase pathway plus at
most about 2.1K for the level and amplitude pathways; the latter add less than
0.25% of the FLOPs, so its FLOPs are those of the phase pathway.

    python3 make_efficiency.py        # writes efficiency_plot.pptx, efficiency_bars.pptx
    sh export.sh efficiency_plot 243; sh export.sh efficiency_bars 495
"""
import math
from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

import pptfig
from pptfig import *  # noqa: F401,F403

HERE = Path(__file__).resolve().parent
PHASE = TOKEN_BLUE
PATCH = TOKEN_GREEN
OTHER = RGBColor(0x8C, 0x8C, 0x8C)
RED = RGBColor(0xD6, 0x27, 0x28)
LIGHT = {PHASE: RGBColor(0xBF, 0xE3, 0xF4), PATCH: RGBColor(0xC9, 0xE6, 0xC1),
         OTHER: RGBColor(0xDD, 0xDD, 0xDD)}
INK = RGBColor(0x22, 0x22, 0x22)
GRID = RGBColor(0xE2, 0xE2, 0xE2)
EXTRA = 2100            # at most about 2.1K parameters for the level and amplitude pathways

# name: (category, params Traffic, MFLOPs Traffic, params Electricity, MFLOPs Electricity, Traffic MAE)
MODELS = {
    'PhaseFormer+': ('phase', 1156, 13.9, 1156, 5.18, 0.252),
    'PatchTST': ('patch', 7.589e6, 498577.49, 1.373e6, 16122.93, 0.266),
    'iTransformer': ('other', 6.731e6, 11652.34, 5.153e6, 3347.97, 0.290),
    'Crossformer': ('patch', 22.954e6, 259209.90, 13.537e6, 96564.63, 0.282),
    'FEDformer': ('other', 21.206e6, 13679.70, 11.861e6, 6904.61, 0.400),
    'TimeBase': ('both', 214, 8.44, 214, 3.14, 0.278),
    'SparseTSF': ('phase', 17.949e3, 751.31, 4.509e3, 78.61, 0.280),
    'FITS': ('other', 1054, 1.76, 462, 0.28, 0.290),
    'TimeMixer': ('patch', 5.697e6, 2026.53, 5.584e6, 739.64, 0.298),
}
COLOR = {'phase': PHASE, 'both': PHASE, 'patch': PATCH, 'other': OTHER}


def flops_text(mflops):
    if mflops >= 1000:
        return f'{mflops / 1000:.3g} GFLOPs'
    return f'{mflops:.3g} MFLOPs'


def params_text(p):
    if p >= 1e6:
        return f'{p / 1e6:.3g}M'
    if p >= 1e3:
        return f'{p / 1e3:.3g}K'
    return f'{p:.0f}'


def oval(cx, cy, d, fill, line, lw, dash=None):
    return box(cx - d / 2, cy - d / 2, d, d, fill=fill, line=line, lw=lw, dash=dash,
               shape=MSO_SHAPE.OVAL)


def hline(x0, x1, y, color=GRID, lw=0.75, dash=None):
    path([(x0, y), (x1, y)], color=color, lw=lw, arrow=False, dash=dash)


def vline(x, y0, y1, color=GRID, lw=0.75, dash=None):
    path([(x, y0), (x, y1)], color=color, lw=lw, arrow=False, dash=dash)


# --------------------------------------------------------------------------
# Fig. 1(b): MAE against parameters on Traffic, marker size = FLOPs.
# --------------------------------------------------------------------------
def scatter():
    new_slide(6.0, 3.22)
    X0, X1, Y0, Y1 = 0.62, 5.88, 0.12, 2.56      # plot frame (inches)
    lx0, lx1 = 2.0, 9.0                          # log10 parameters
    my0, my1 = 0.244, 0.306                      # MAE

    def px(p):
        return X0 + (math.log10(p) - lx0) / (lx1 - lx0) * (X1 - X0)

    def py(m):
        return Y1 - (m - my0) / (my1 - my0) * (Y1 - Y0)

    for e in range(3, 9):
        vline(px(10 ** e), Y0, Y1)
    for m in (0.25, 0.26, 0.27, 0.28, 0.29, 0.30):
        hline(X0, X1, py(m))
    box(X0, Y0, X1 - X0, Y1 - Y0, fill=None, line=INK, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
    for e in range(2, 10):
        label(px(10 ** e) - 0.3, Y1 + 0.03, 0.6, 0.22, [f'$10^{e}$'], size=SMALL)
    for m in (0.25, 0.26, 0.27, 0.28, 0.29, 0.30):
        label(X0 - 0.5, py(m) - 0.1, 0.44, 0.2, [f'{m:.2f}'], size=SMALL, align=PP_ALIGN.RIGHT)
    label(X0, Y1 + 0.22, X1 - X0, 0.24, ['Parameters (Traffic)'], size=SMALL)
    tb = label(0.14 - 0.35, (Y0 + Y1) / 2 - 0.12, 0.7, 0.24, ['MAE'], size=SMALL)
    tb.rotation = -90

    def diameter(mflops):
        return 0.11 + 0.072 * (math.log10(mflops) + 1)

    # label offsets (dx, dy) in inches from the marker centre, and alignment
    place = {
        'PhaseFormer+': (0.22, 0.0, PP_ALIGN.LEFT),
        'TimeBase': (0.24, 0.0, PP_ALIGN.LEFT),
        'FITS': (0.0, -0.33, PP_ALIGN.CENTER),
        'SparseTSF': (0.33, 0.0, PP_ALIGN.LEFT),
        'iTransformer': (-0.33, 0.06, PP_ALIGN.RIGHT),
        'TimeMixer': (-0.3, -0.1, PP_ALIGN.RIGHT),
        'Crossformer': (0.36, 0.0, PP_ALIGN.LEFT),
        'PatchTST': (0.38, 0.0, PP_ALIGN.LEFT),
    }
    order = sorted((n for n in MODELS if n != 'FEDformer'), key=lambda n: -MODELS[n][2])
    for name in order:
        cat, p, fl, _, _, mae = MODELS[name]
        if name == 'PhaseFormer+':
            p = p + EXTRA
        cx, cy, d = px(p), py(mae), diameter(fl)
        if name == 'PhaseFormer+':
            oval(cx, cy, d, RED, PHASE, 1.75)
        elif cat == 'both':
            oval(cx, cy, d + 0.07, None, PATCH, 1.25, dash='dash')
            oval(cx, cy, d, LIGHT[PHASE], PHASE, 1.5)
        elif cat == 'patch':
            oval(cx, cy, d, LIGHT[PATCH], PATCH, 1.5, dash='dash')
        elif cat == 'phase':
            oval(cx, cy, d, LIGHT[PHASE], PHASE, 1.5)
        else:
            oval(cx, cy, d, LIGHT[OTHER], OTHER, 1.5, dash='sysDot')
    for name in order:
        cat, p, fl, _, _, mae = MODELS[name]
        if name == 'PhaseFormer+':
            p = p + EXTRA
        dx, dy, al = place[name]
        cx, cy = px(p), py(mae)
        color = RED if name == 'PhaseFormer+' else INK
        nm = f'**{name}**' if name == 'PhaseFormer+' else name
        w = 1.3
        x = cx + dx - (w / 2 if al == PP_ALIGN.CENTER else (w if al == PP_ALIGN.RIGHT else 0))
        label(x, cy + dy - 0.2, w, 0.4, [nm, flops_text(fl)], size=SMALL, align=al, color=color,
              spacing=0.9)
    # legend
    items = [('Phase-based', PHASE, None), ('Patch-based', PATCH, 'dash'),
             ('Others', OTHER, 'sysDot')]
    x, y = 1.55, 2.98
    for text, col, dash in items:
        oval(x + 0.08, y + 0.12, 0.15, None, col, 1.5, dash=dash)
        label(x + 0.22, y, 1.2, 0.24, [text], size=SMALL, align=PP_ALIGN.LEFT)
        x += 1.25
    save(HERE / 'efficiency_plot.pptx')


# --------------------------------------------------------------------------
# FLOPs and parameters per model on Traffic and Electricity (log bars).
# --------------------------------------------------------------------------
def bars():
    new_slide(12.2, 3.35)
    W, GAP, PX0 = 2.15, 0.85, 0.78
    panels = [('(a) FLOPs on Traffic', 2, 'GFLOPs', -3.0, 3.0),
              ('(b) Parameters on Traffic', 1, 'Parameters', 2.0, 8.0),
              ('(c) FLOPs on Electricity', 4, 'GFLOPs', -4.0, 2.0),
              ('(d) Parameters on Electricity', 3, 'Parameters', 2.0, 8.0)]
    Y0, Y1 = 0.36, 2.2
    for k, (title, col, unit, lo, hi) in enumerate(panels):
        x0 = PX0 + k * (W + GAP)
        x1 = x0 + W

        def py(v):
            return Y1 - (math.log10(v) - lo) / (hi - lo) * (Y1 - Y0)

        for e in range(int(lo) + 1, int(hi)):
            hline(x0, x1, py(10 ** e))
        for e in range(int(lo), int(hi) + 1, 1 if hi - lo <= 6 else 2):
            label(x0 - 0.52, py(10 ** e) - 0.1, 0.48, 0.2, [f'$10^{{{e}}}$'], size=SMALL,
                  align=PP_ALIGN.RIGHT)
        label(x0 - 0.25, 0.02, W + 0.5, 0.28, [f'**{title}**'], size=TEXT)
        names = sorted(MODELS, key=lambda n: -MODELS[n][col])
        n = len(names)
        slot = W / n
        bw = slot * 0.62
        for i, name in enumerate(names):
            cat = MODELS[name][0]
            v = MODELS[name][col] / (1000.0 if unit == 'GFLOPs' else 1.0)
            cx = x0 + slot * (i + 0.5)
            color = COLOR[cat]
            if name == 'PhaseFormer+' and unit == 'Parameters':
                top = py(v + EXTRA)
                box(cx - bw / 2, top, bw, py(v) - top, fill=LIGHT[PHASE], line=RED, lw=1.5,
                    dash='dash', shape=MSO_SHAPE.RECTANGLE)
                box(cx - bw / 2, py(v), bw, Y1 - py(v), fill=PHASE, line=RED, lw=1.75,
                    shape=MSO_SHAPE.RECTANGLE)
            elif name == 'PhaseFormer+':
                box(cx - bw / 2, py(v), bw, Y1 - py(v), fill=PHASE, line=RED, lw=1.75,
                    shape=MSO_SHAPE.RECTANGLE)
            else:
                box(cx - bw / 2, py(v), bw, Y1 - py(v), fill=color, line=None,
                    shape=MSO_SHAPE.RECTANGLE)
            lw_, lh_ = 1.2, 0.22
            off = lw_ / 2 * 0.7071
            tb = label(cx - off - lw_ / 2, Y1 + 0.06 + off - lh_ / 2, lw_, lh_,
                       [f'**{name}**' if name == 'PhaseFormer+' else name], size=SMALL,
                       align=PP_ALIGN.RIGHT, color=RED if name == 'PhaseFormer+' else INK)
            tb.rotation = -45
        box(x0, Y0, W, Y1 - Y0, fill=None, line=INK, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
        tb = label(x0 - 0.72 - 0.4, (Y0 + Y1) / 2 - 0.12, 0.8, 0.24, [unit], size=SMALL)
        tb.rotation = -90
    # legend
    LX, LY = 1.9, 3.06
    items = [('Phase-based', PHASE, None), ('Patch-based', PATCH, None), ('Others', OTHER, None)]
    x = LX
    for text, col, _ in items:
        box(x, LY + 0.06, 0.22, 0.14, fill=col, line=None, shape=MSO_SHAPE.RECTANGLE)
        label(x + 0.28, LY, 1.2, 0.26, [text], size=SMALL, align=PP_ALIGN.LEFT)
        x += 1.35
    box(x + 0.05, LY + 0.06, 0.22, 0.14, fill=LIGHT[PHASE], line=RED, lw=1.25, dash='dash',
        shape=MSO_SHAPE.RECTANGLE)
    label(x + 0.33, LY, 5.2, 0.26, ['level and amplitude pathways of PhaseFormer+ (at most about 2.1K)'],
          size=SMALL, align=PP_ALIGN.LEFT)
    save(HERE / 'efficiency_bars.pptx')


if __name__ == '__main__':
    scatter()
    bars()
