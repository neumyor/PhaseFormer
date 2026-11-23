"""Framework figure of PhaseFormer+, candidate V2: a two-lane pipeline with every module inline.

Drawn in the style of the conference figure (make_overview.py): Times, white module boxes with
black outlines, dashed sub-panels, small token rectangles, W boxes and Multi-Head Attention boxes.
  (a) phase-aligned tokens  x -> normalize, fold -> P x K matrix; one row is a phase token x_p,
                            the column means form the level token r; shared embedding W_e
  (b) phase pathway         Z -> aggregation MHA(R, Z, Z) -> R' -> distribution MHA(Z, R', R')
                            -> Add & LN, MLP, Add & LN (x N_r) -> shared predictor Phi -> F
                            (the positional embedding of Sec. 3.3 is left to the text)
  (c) trend pathway         z -> linear encoder E -> state t -> horizon basis Psi_H
                            (+) anchor r_K -> F_b
  (d) gated fusion          levels and shape of F and F_b mixed by the level gate g_l and the
                            shape gate g_s, summed, de-normalized -> y
Only key variables carry symbols, and each symbol sits next to words that name it.
Colors follow Fig. 3: phase token blue, level token green; routers indigo.
Symbols follow body/method.tex.

    python3 make_framework_cand2.py && ./export.sh framework_cand2 516
"""
import math
import sys
from pathlib import Path

from lxml import etree
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).parent))
import pptfig  # noqa: E402
from pptfig import (BLACK, BLUE_PANEL, GREEN_PANEL, PEACH, TOKEN_BLUE, TOKEN_GREEN,  # noqa: E402
                    TOKEN_LIGHT, WHITE, YELLOW, box, label, new_slide, path, save)

pptfig.SYMBOLS |= set('ΨΦΓ')

OUT = Path(__file__).with_name('framework_cand2.pptx')
W, H = 12.2, 4.0
TITLE, TEXT, SMALL = 16, 15, 12.5


def rgb(hexstr):
    return RGBColor.from_string(hexstr)


def lerp(c0, c1, t):
    return RGBColor(*(round(a + (b - a) * t) for a, b in zip(c0, c1)))


BAND_LINE = rgb('A6A6A6')
GRID_LINE = rgb('B4B4B4')
SERIES_GRAY = rgb('808080')
BLUE_LINE = rgb('1F3A5F')
BLUE_SOFT = rgb('8CC8EC')
GREEN_LINE = rgb('2E6B1F')
ROUTER = rgb('6A7DD0')
ROUTER_LINE = rgb('3B4C9A')
RED = rgb('C0392B')
PANEL_A = rgb('F7F7F7')
LANE_B = lerp(WHITE, BLUE_PANEL, 0.45)
LANE_C = lerp(WHITE, GREEN_PANEL, 0.45)
PANEL_D = lerp(WHITE, PEACH, 0.45)


# --------------------------------------------------------------------------- helpers
def band(x, y, w, h, title, fill):
    shape = box(x, y, w, h, fill=fill, line=BAND_LINE, lw=1.0, dash='dash', radius=0.1)
    label(x + 0.08, y + 0.02, w - 0.16, 0.28, [f'**{title}**'], size=TITLE, align=PP_ALIGN.LEFT)
    return shape


def translucent(shape, alpha):
    """Give the solid fill of a shape an opacity in [0, 1]."""
    clr = shape._element.spPr.find(qn('a:solidFill')).find(qn('a:srgbClr'))
    etree.SubElement(clr, qn('a:alpha')).set('val', str(round(alpha * 100000)))


def module(x, y, w, h, lines=(), size=TEXT, fill=WHITE, lw=1.25):
    return box(x, y, w, h, list(lines), fill=fill, line=BLACK, lw=lw, size=size, radius=0.08)


def rect(x, y, w, h, fill, line=BLACK, lw=0.75):
    box(x, y, w, h, fill=fill, line=line, lw=lw, shape=MSO_SHAPE.RECTANGLE)


def arrow(points, color=BLACK, lw=1.25, head=True, dash=None):
    path(points, color=color, lw=lw, arrow=head, dash=dash)


def operator(x, y, r=0.12):
    box(x - r, y - r, 2 * r, 2 * r, fill=WHITE, line=BLACK, lw=1.25, shape=MSO_SHAPE.OVAL)
    a = r * 0.55
    path([(x - a, y), (x + a, y)], color=BLACK, lw=1.25, arrow=False)
    path([(x, y - a), (x, y + a)], color=BLACK, lw=1.25, arrow=False)


def token_row(cx, cy, n, fill, line, w=0.11, h=0.12, gap=0.03):
    x = cx - (n * w + (n - 1) * gap) / 2
    for k in range(n):
        rect(x + k * (w + gap), cy - h / 2, w, h, fill, line, lw=1.0)
    return x, x + n * w + (n - 1) * gap


def token_col(cx, cy, n, fill, line, w=0.14, h=0.09, gap=0.025):
    y = cy - (n * h + (n - 1) * gap) / 2
    for k in range(n):
        rect(cx - w / 2, y + k * (h + gap), w, h, fill, line, lw=1.0)


def chip(cx, cy, w=0.5, h=0.2):
    """Router icon of the conference figure: a flat chip in perspective."""
    t = 0.07
    top = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy), (cx, cy - h / 2)]
    left = [(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx - w / 2, cy + t), (cx - w / 2, cy)]
    right = [(cx + w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx + w / 2, cy + t), (cx + w / 2, cy)]
    path(left, color=ROUTER_LINE, lw=0.75, arrow=False, fill=rgb('4E62B8'), closed=True)
    path(right, color=ROUTER_LINE, lw=0.75, arrow=False, fill=ROUTER_LINE, closed=True)
    path(top, color=ROUTER_LINE, lw=0.75, arrow=False, fill=rgb('8C9EE6'), closed=True)
    inner = [(cx, cy - h / 4), (cx + w / 4, cy), (cx, cy + h / 4), (cx - w / 4, cy), (cx, cy - h / 4)]
    path(inner, color=ROUTER, lw=0.75, arrow=False, fill=rgb('7A8DDE'), closed=True)


def daily(u):
    """Within-cycle shape with a morning and an evening peak, u in [0, 1)."""
    return 0.55 * math.exp(-((u - 0.32) / 0.08) ** 2) + 0.8 * math.exp(-((u - 0.7) / 0.1) ** 2)


def series(x, y, w, h, cycles, color, lw=1.0, drift=0.0, levels=None, centered=False, n=30):
    ys = []
    for k in range(cycles * n + 1):
        t = k / n
        c = min(int(t), cycles - 1)
        v = daily(t - int(t)) if t < cycles else daily(0.0)
        if centered:
            v -= 0.3
        v += drift * t / cycles
        if levels is not None:
            v += levels[c]
        ys.append(v)
    lo, hi = min(ys), max(ys)
    pts = [(x + w * k / (len(ys) - 1), y + h * (1 - (v - lo) / (hi - lo))) for k, v in enumerate(ys)]
    path(pts, color=color, lw=lw, arrow=False)


def steps(x, y, w, h, levels, color, lw=1.5):
    """Piecewise-constant cycle levels in [0, 1]."""
    n = len(levels)
    pts = []
    for c, v in enumerate(levels):
        yy = y + h * (1 - v)
        pts += [(x + w * c / n, yy), (x + w * (c + 1) / n, yy)]
    path(pts, color=color, lw=lw, arrow=False)


def profile(x, y, w, h, kind, color, lw=1.5):
    pts = []
    for k in range(21):
        u = k / 20
        v = {'offset': 0.5, 'tilt': 0.1 + 0.8 * u, 'curve': 0.9 - 3.2 * (u - 0.5) ** 2}[kind]
        pts.append((x + w * u, y + h * (1 - v)))
    path(pts, color=color, lw=lw, arrow=False)


def bezier(p0, p1, p2, p3, n=24):
    return [tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                  for a, b, c, d in zip(p0, p1, p2, p3)) for t in (k / n for k in range(n + 1))]


def wbox(cx, y, sub, sup, w=0.46, h=0.32):
    box(cx - w / 2, y, w, h, [f'$\\b{{W}}_{sub}$'], fill=WHITE, line=BLACK, lw=1.0,
        size=SMALL, radius=0.05, margin=0.0)


# --------------------------------------------------------------------------- figure
new_slide(W, H)
Y_P, Y_T = 1.09, 3.30                 # rows of the phase token and the level token
TOP, BOT = 0.04, 3.96

F0, F1 = 9.6, 12.16
translucent(band(F0, TOP, F1 - F0, BOT - TOP, '(d) Gated Fusion', rgb('D9D9D9')), 0.23)

# (a) phase-aligned tokens and the shared embedding -----------------------------------------
A0, A1 = 0.04, 2.9
band(A0, TOP, A1 - A0, BOT - TOP, '(a) Phase-Aligned Tokens', PANEL_A)
P, K = 6, 5
gx, cw, ch = 0.32, 0.1, 0.16
gy = Y_P - 1.5 * ch                    # row 1 is the highlighted phase token
gw = K * cw
series(gx, 0.39, gw, 0.2, K, BLACK, drift=0.35)
for k in range(1, K):
    path([(gx + k * cw, 0.37), (gx + k * cw, 0.61)], color=SERIES_GRAY, lw=0.75, arrow=False, dash='sysDash')
arrow([(gx + 0.08, 0.64), (gx + 0.08, gy - 0.03)])
label(gx + 0.16, 0.625, 0.4, 0.21, ['fold'], size=SMALL, align=PP_ALIGN.LEFT)
label(gx + gw + 0.06, 0.38, 0.9, 0.22, ['window $x$'], size=SMALL, align=PP_ALIGN.LEFT)

lo = rgb('F4F9FD')
levels = [0.0] * K
for p in range(P):
    shape = daily((p + 0.5) / P)
    for k in range(K):
        v = min(1.0, 0.1 + 0.65 * shape + 0.07 * k)
        levels[k] += v / P
        rect(gx + k * cw, gy + p * ch, cw, ch, lerp(lo, BLUE_SOFT, v), line=GRID_LINE, lw=0.6)
rect(gx - 0.025, gy + ch - 0.02, gw + 0.05, ch + 0.04, None, line=TOKEN_BLUE, lw=2.0)
label(0.19 - P * ch / 2, gy + P * ch / 2 - 0.12, P * ch, 0.24, ['$P$ phases'], size=SMALL).rotation = 270
label(gx + 0.15, gy + P * ch + 0.02, 0.8, 0.2, ['$K$ cycles'], size=SMALL, align=PP_ALIGN.LEFT)

# both tokens drawn as rows of cubes, as in Fig. 1: K cells, one per cycle
TX, S, D = 0.95, 0.12, 0.045             # left edge, cube size, depth offset
TOK_EDGE = rgb('5A6E7E')


def cuboid(x, y_bottom, w, h, fill, right=True):
    top = y_bottom - h
    path([(x, top), (x + D, top - D), (x + w + D, top - D), (x + w, top), (x, top)], color=TOK_EDGE,
         lw=0.5, arrow=False, fill=lerp(fill, WHITE, 0.4), closed=True)
    if right:
        path([(x + w, top), (x + w + D, top - D), (x + w + D, y_bottom - D), (x + w, y_bottom), (x + w, top)],
             color=TOK_EDGE, lw=0.5, arrow=False, fill=lerp(fill, BLACK, 0.25), closed=True)
    box(x, top, w, h, fill=fill, line=TOK_EDGE, lw=0.5, shape=MSO_SHAPE.RECTANGLE)


row = [min(1.0, 0.1 + 0.65 * daily(1.5 / P) + 0.07 * k) for k in range(K)]
for k, v in enumerate(row):            # the phase token x_p: row 1 of the grid
    cuboid(TX + k * S, Y_P + S / 2, S, S, lerp(rgb('D4EAF6'), rgb('2E9AD6'), k / (K - 1)), right=k == K - 1)
path([(gx + gw + 0.025, Y_P), (TX - 0.01, Y_P)], color=TOKEN_BLUE, lw=1.0, arrow=False, dash='sysDot')
label(TX - 0.12, Y_P + S / 2 + 0.04, K * S + 0.3, 0.42, ['phase token', '$x̃_p$'], size=SMALL, spacing=0.9,
      color=BLUE_LINE, anchor=MSO_ANCHOR.TOP)

YB = Y_T + 0.05                        # baseline of the level token, the cycle levels as bars
lmin, lmax = min(levels), max(levels)
for k, v in enumerate(levels):
    u = (v - lmin) / (lmax - lmin)
    cuboid(TX + k * S, YB, S, 0.1 + 0.26 * u, lerp(TOKEN_LIGHT, TOKEN_GREEN, u))
label(TX - 0.12, YB + 0.04, K * S + 0.3, 0.42, ['level token', '$r̅$'], size=SMALL, spacing=0.9,
      color=GREEN_LINE, anchor=MSO_ANCHOR.TOP)
MX = gx + 0.08
arrow([(MX, gy + P * ch + 0.03), (MX, Y_T), (TX - 0.03, Y_T)])
label(MX + 0.08, 2.29, 0.6, 0.57, ['mean', 'over', 'phases'], size=SMALL, align=PP_ALIGN.LEFT, spacing=0.9)

E0, E1 = 1.74, 2.84                    # shared embedding column
EX = (E0 + E1) / 2
translucent(module(E0, 0.42, E1 - E0, 3.48, fill=YELLOW, lw=0.5), 0.38)
label(E0 - 0.1, 0.45, E1 - E0 + 0.2, 0.42, ['**Shared**', '**Embedding**'], size=TEXT, spacing=0.85)
for yc, line in ((Y_P, BLUE_LINE), (Y_T, GREEN_LINE)):
    arrow([(TX + K * S + D + 0.02, yc), (EX - 0.25, yc)], color=line)
    box(EX - 0.25, yc - 0.16, 0.5, 0.32, ['$W_e$'], fill=WHITE, line=BLACK, lw=1.0, radius=0.05)
path([(EX, Y_P + 0.16), (EX, 1.85)], color=BLACK, lw=1.0, arrow=False, dash='dash')
path([(EX, 2.55), (EX, Y_T - 0.16)], color=BLACK, lw=1.0, arrow=False, dash='dash')
label(E0 - 0.1, 1.88, E1 - E0 + 0.2, 0.64, ['$z̄$ =', 'mean', 'of $z_p$'], size=SMALL, spacing=0.9)
label(EX + 0.25, Y_P + 0.03, E1 - EX - 0.25, 0.22, ['$\\b{Z}$'], size=TEXT)
label(EX + 0.25, Y_T + 0.03, E1 - EX - 0.25, 0.22, ['$z̄$'], size=TEXT)

# (b) phase pathway ---------------------------------------------------------------------------
B0, B1 = 2.96, 9.52
band(B0, TOP, B1 - B0, 2.5 - TOP, '(b) Phase Pathway', LANE_B)

R0, R1, RT, RB = 3.42, 8.22, 0.38, 2.42  # routing layer
module(R0 + 0.047, RT - 0.044, R1 - R0, RB - RT, lw=1.0)   # one layer behind: N_r stacked layers
module(R0, RT, R1 - R0, RB - RT, lw=1.0)
label(R0 + 0.08, RT + 0.02, 3.0, 0.24, ['**Cross-Phase Routing Layer**'], size=TEXT, align=PP_ALIGN.LEFT)
label(R1 + 0.103, RB - 0.26, 0.42, 0.26, ['$×N_r$'], align=PP_ALIGN.LEFT)

SUB_T, SUB_B = 0.66, 2.34
Y_W, Y_M, Y_O = 1.32, 1.74, 2.18      # W boxes, attention, outputs


def step_panel(x0, x1, title):
    box(x0, SUB_T, x1 - x0, SUB_B - SUB_T, fill=BLUE_PANEL, line=BLACK, lw=1.0, dash='dash', radius=0.05)
    label(x0 + 0.06, SUB_T + 0.03, x1 - x0 - 0.1, 0.22, [f'**{title}**'], size=SMALL, align=PP_ALIGN.LEFT)


def attention(x0, x1, kx, vx, qx, stage):
    for cx, sub in ((kx, 'K'), (vx, 'V'), (qx, 'Q')):
        wbox(cx, Y_W, sub, stage)
        arrow([(cx, Y_W + 0.32), (cx, Y_M)])
    module(x0 + 0.1, Y_M, x1 - x0 - 0.2, 0.28, ['Multi-Head Attention'], size=SMALL, lw=1.0)


# aggregation: routers R are the queries, Z~ the keys and values
G0, G1 = 3.5, 5.3
step_panel(G0, G1, 'Aggregation')
kx, vx, qx = 3.95, 4.43, 5.0
token_row((kx + vx) / 2, Y_P, 4, TOKEN_BLUE, BLUE_LINE)
arrow([(EX + 0.25, Y_P), ((kx + vx) / 2 - 0.29, Y_P)], color=BLUE_LINE)
for cx in (kx, vx):
    arrow([(cx, Y_P + 0.06), (cx, Y_W)])
label(kx + 0.03, Y_P + 0.05, vx - kx - 0.06, 0.2, ['$\\b{Z}$'], size=SMALL)
chip(qx, Y_P - 0.04)
label(G1 - 0.94, 0.7, 0.88, 0.22, ['routers $\\b{R}$'], size=SMALL, align=PP_ALIGN.RIGHT)
arrow([(qx, Y_P + 0.13), (qx, Y_W)])
attention(G0, G1, kx, vx, qx, 'agg')
RX = (G0 + G1) / 2 + 0.024
RTOK = dict(w=0.172, h=0.12, gap=0.047)  # router tokens R', wider than the phase tokens
arrow([(RX, Y_M + 0.28), (RX, Y_O - 0.06)])
ra, rb = token_row(RX, Y_O, 3, ROUTER, ROUTER_LINE, **RTOK)

# distribution: Z~ are the queries, R' the keys and values
D0, D1 = 5.42, 7.22
step_panel(D0, D1, 'Distribution')
kx, vx, qx = 5.75, 6.23, 6.85
da, db = token_row((kx + vx) / 2, Y_P, 3, ROUTER, ROUTER_LINE, **RTOK)
for cx in (kx, vx):
    arrow([(cx, Y_P + 0.06), (cx, Y_W)])
token_row(qx, Y_P, 4, TOKEN_BLUE, BLUE_LINE)
arrow([(qx, Y_P + 0.06), (qx, Y_W)])
label(qx + 0.07, Y_P + 0.05, 0.3, 0.2, ['$\\b{Z}$'], size=SMALL, align=PP_ALIGN.LEFT)
attention(D0, D1, kx, vx, qx, 'dist')
XG = (G1 + D0) / 2                      # hand-over of R' through the gap between the two steps
arrow([(rb, Y_O), (XG, Y_O), (XG, Y_P), (da, Y_P)], color=BLUE_LINE)
xc = (D0 + D1) / 2
arrow([(xc, Y_M + 0.28), (xc, Y_O - 0.06)])
za, zb = token_row(xc, Y_O, 4, TOKEN_BLUE, BLUE_LINE)

# residual block: Add & LN, MLP, Add & LN, bottom to top
C0, C1 = 7.28, 8.14
for yc, text in ((Y_O, 'Add & LN'), (1.72, 'MLP'), (1.26, 'Add & LN')):
    module(C0, yc - 0.14, C1 - C0, 0.28, [text], size=SMALL, lw=1.0)
arrow([(zb + 0.02, Y_O), (C0, Y_O)])
arrow([((C0 + C1) / 2, Y_O - 0.14), ((C0 + C1) / 2, 1.86)])
arrow([((C0 + C1) / 2, 1.58), ((C0 + C1) / 2, 1.40)])

# shared predictor, then unfold to the phase forecast F
Y_F = 1.26
module(8.36, 0.88, 1.0, 0.76, ['Shared', 'Predictor $Φ$'], size=14)
arrow([(C1, Y_F), (8.36, Y_F)])
arrow([(9.36, Y_F), (9.72, Y_F)])

# (c) trend pathway ---------------------------------------------------------------------------
band(B0, 2.58, B1 - B0, BOT - 2.58, '(c) Trend Pathway', LANE_C)
arrow([(EX + 0.25, Y_T), (3.2, Y_T)], color=GREEN_LINE)
module(3.2, 2.98, 1.3, 0.64, ['Linear', 'Encoder $E$'], size=TEXT)
arrow([(4.5, Y_T), (4.71, Y_T)])
token_col(4.79, Y_T, 3, TOKEN_GREEN, GREEN_LINE)
label(4.49, 3.49, 0.6, 0.42, ['state', '$t∈ℝ^r$'], size=SMALL)
HB0, HB1 = 5.3, 7.25
arrow([(4.87, Y_T), (HB0, Y_T)])
module(HB0, 2.92, HB1 - HB0, 0.98)
label(HB0, 2.94, HB1 - HB0, 0.24, ['Horizon Basis $Ψ_H$'], size=TEXT)
for x, kind in ((HB0 + 0.18, 'offset'), (HB0 + 0.75, 'tilt'), (HB0 + 1.32, 'curve')):
    profile(x, 3.24, 0.45, 0.2, kind, GREEN_LINE)
    label(x - 0.1, 3.46, 0.65, 0.2, [kind], size=SMALL)
label(HB0, 3.66, HB1 - HB0, 0.2, ['fixed or learned'], size=SMALL)
OX2 = 7.75
arrow([(HB1, Y_T), (OX2 - 0.12, Y_T)])
operator(OX2, Y_T)
arrow([(OX2, 3.86), (OX2, Y_T + 0.12)])
label(OX2 + 0.07, 3.62, 1.7, 0.24, ['anchor $r̅_K$ (last cycle)'], size=SMALL, align=PP_ALIGN.LEFT)
arrow([(OX2 + 0.12, Y_T), (9.72, Y_T)])
label(OX2 + 0.15, Y_T - 0.27, 1.6, 0.24, ['trend forecast $F_{\\r{b}}$'], size=SMALL)

# (d) gated fusion ----------------------------------------------------------------------------
X0, X1 = 9.72, 11.12
c1, c2 = X0 + (X1 - X0) / 4, X0 + 3 * (X1 - X0) / 4
label(c1 - 0.4, 0.64, 0.8, 0.24, ['levels'], size=SMALL)
label(c2 - 0.4, 0.64, 0.8, 0.24, ['shape'], size=SMALL)
for yc, color, lv, centered in ((Y_F, TOKEN_BLUE, [0.3, 0.55], True), (Y_T, TOKEN_GREEN, [0.35, 0.8], False)):
    module(X0, yc - 0.32, X1 - X0, 0.64, lw=1.0)
    path([((X0 + X1) / 2, yc - 0.28), ((X0 + X1) / 2, yc + 0.28)], color=GRID_LINE, lw=1.0, arrow=False,
         dash='dash')
    steps(c1 - 0.25, yc - 0.18, 0.5, 0.36, lv, color)
    if centered:
        series(c2 - 0.27, yc - 0.18, 0.54, 0.36, 2, color, lw=1.25)
    else:
        pts = []
        for c in range(2):
            pts += [(c2 - 0.27 + 0.27 * c, yc + 0.12), (c2 - 0.27 + 0.27 * (c + 1), yc - 0.12)]
        path(pts, color=color, lw=1.25, arrow=False)
Y_G = 2.28
for cx, gate in ((c1, '\\ell'), (c2, 's')):
    gate = 'ℓ' if gate == '\\ell' else gate
    arrow([(cx, Y_F + 0.32), (cx, Y_G - 0.12)])
    arrow([(cx, Y_T - 0.32), (cx, Y_G + 0.12)])
    operator(cx, Y_G)
    label(cx + 0.04, 1.8, 0.5, 0.24, [f'$1−g_{gate}$'], size=SMALL, align=PP_ALIGN.LEFT)
    label(cx + 0.04, 2.58, 0.5, 0.24, [f'$g_{gate}$'], size=SMALL, align=PP_ALIGN.LEFT)
arrow([(c1 + 0.12, Y_G), (c2 - 0.12, Y_G)])
label(X0 + 0.12, 3.675, 2.2, 0.21, ['$g_ℓ$: level gate       $g_s$: shape gate'], size=SMALL, align=PP_ALIGN.LEFT)
O0, O1 = 11.3, 12.08
arrow([(c2 + 0.12, Y_G), (O0, Y_G)])
module(O0, 1.742, O1 - O0, 1.02, lw=1.25)
label(O0, 1.782, O1 - O0, 0.26, ['forecast $ŷ$'], size=SMALL)
series(O0 + 0.1, 2.162, O1 - O0 - 0.2, 0.36, 2, RED, lw=1.25, levels=[0.0, 0.35])

save(OUT)
