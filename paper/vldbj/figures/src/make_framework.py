"""PhaseFormer+ framework (Fig. 1): a shared phase-aligned representation and two pathways.

Three panel columns and the output, mirror-symmetric about the middle line between the two rows:
  left   : shared panel (gray)  input window x -x~-> fold X~; one fork sends the phase tokens x~_p to the
           phase pathway and the level token [zeta; z_bar] to the trend pathway
  middle : the two pathways, column by column an encoder and a decoder
           phase (sky blue)  cross-phase routing R (xN_r) -h_p-> shared predictor Phi  -F->
           trend (navy)      linear encoder E            -t->   polynomial basis Psi_H -F_b->
  right  : gated fusion (gray)  F enters the gates 1-g_l | 1-g_s, F_b the gates g_l | g_s; each gate column
           feeds the projection it weights, M_c (cycle levels) | J_c (within-cycle shapes), and the two
           parts add up to the forecast y^
Conventions: boxes are modules or data, with their symbol in the title; variables label the arrows,
centered above them; arrows, labels and fills take the color of their pathway; all math is set in
Times New Roman. The embedding layer, the readout Gamma and the biases are not drawn.

    python3 make_framework.py            # writes framework.pptx next to this file
    sh export.sh framework 516
"""
import math
from pathlib import Path

import pptfig
from pptfig import *  # noqa: F401,F403

pptfig.SUBSCALE = 1.35          # sub/superscripts print at about 7pt

# All math in Times New Roman, accented letters included: letters italic, Greek capitals upright.
pptfig.SYM_FONT = 'Times New Roman'
pptfig.MATH_ITALIC = {}

OUT = Path(__file__).with_name('framework.pptx')

SERIES_GRAY = RGBColor(0x80, 0x80, 0x80)
# the hues of figstyle.py: phase sky blue, trend navy; the trend fills are darker so that the two
# pathways stay apart in grayscale
PHASE_PANEL = RGBColor(0xEA, 0xF5, 0xFC)
PHASE_TOKEN = RGBColor(0xBF, 0xE3, 0xF4)
PHASE_LINE = RGBColor(0x5D, 0xB8, 0xE3)
PHASE = RGBColor(0x13, 0x7D, 0xB0)
TREND_PANEL = RGBColor(0xD9, 0xE0, 0xEB)
TREND_BOX = RGBColor(0xB9, 0xC4, 0xD6)
TREND = RGBColor(0x1B, 0x3A, 0x6B)

LINE, FLOW, GLYPH, PANEL_LW, RAD = 1.25, 1.5, 1.0, 1.0, 0.08  # box, main arrow, glyph, panel, corner

TL, MACRON, HAT = '̃', '̅', '̂'   # combining tilde, overline, circumflex

# ------------------------------------------------------------------ rows (inches)
TS, BH, PAD, GAP = 0.42, 1.05, 0.14, 0.26         # panel title strip, box height, panel padding, row gap
HEAD = 0.34                                      # title line inside a box
PA0 = 0.05                                       # top-row panels
ya0 = PA0 + TS
YA = ya0 + BH / 2                                # arrow line of the top row
CA = ya0 + HEAD + (BH - HEAD) / 2                # glyph line of the top-row boxes
PA1 = ya0 + BH + PAD
PB0 = PA1 + GAP                                  # bottom-row panels
yb0 = PB0 + TS
YB = yb0 + BH / 2
CB = yb0 + HEAD + (BH - HEAD) / 2
PB1 = yb0 + BH + PAD
YF = (YA + YB) / 2                               # middle line of the figure
H = PB1 + 0.05

# ------------------------------------------------------------------ columns (inches)
SP0 = 0.05                                       # shared panel
LX0, BWL = SP0 + PAD, 2.4                        # input window and fold
XF = LX0 + BWL + 0.22                            # fork to the two pathways
SP1 = XF + PAD
GX0 = SP1 + 0.58                                 # pathway panels; the gap holds the token labels
C1X, BW1 = GX0 + PAD, 2.5                        # encoders: routing | linear encoder
C3X, C3W = C1X + BW1 + 0.55, 2.0                 # decoders: shared predictor | polynomial basis
GX1 = C3X + C3W + PAD
FUX0 = GX1 + 0.5                                 # fusion panel
SBX, SBW, SBH = FUX0 + PAD, 1.5, 0.46            # its three boxes
FUX1 = SBX + SBW + PAD
OX0, OW = FUX1 + 0.4, 1.25                       # forecast
W = OX0 + OW + 0.05
new_slide(W, H)


def daily(t, k=0):
    base = 0.15 + 0.55 * math.exp(-((t - 0.33) / 0.07) ** 2) + 0.45 * math.exp(-((t - 0.72) / 0.09) ** 2)
    return base + 0.03 * math.sin(7.3 * k)


def norm(vs):
    lo, hi = min(vs), max(vs)
    return [(v - lo) / (hi - lo) for v in vs]


def node(x, y, w, h, fill=WHITE):
    return box(x, y, w, h, fill=fill, line=BLACK, lw=LINE, radius=RAD)


def panel(x0, y0, x1, y1, fill, text):
    box(x0, y0, x1 - x0, y1 - y0, fill=fill, lw=PANEL_LW, dash=True, radius=RAD)
    label(x0 + 0.12, y0 + 0.06, x1 - x0 - 0.2, 0.3, [f'**{text}**'], align=PP_ALIGN.LEFT)


def head(x, y, w, text):
    label(x, y + 0.04, w, 0.3, [text])


def arrow(x0, y0, x1, y1, color=BLACK, lw=FLOW):
    path([(x0, y0), (x1, y1)], color=color, lw=lw)


def var(x0, x1, y, text, color=BLACK):
    """Variable centered above the horizontal arrow segment from x0 to x1 at height y."""
    label(x0, y - 0.31, x1 - x0, 0.28, [text], color=color)


def bar(x, y, w, h, fill, line):
    box(x, y, w, h, fill=fill, line=line, lw=GLYPH, shape=MSO_SHAPE.RECTANGLE)


def bars(x, yc, n, fill=PHASE_TOKEN, line=PHASE_LINE, w=0.1, h=0.4, gap=0.05):
    for k in range(n):
        bar(x + k * (w + gap), yc - h / 2, w, h, fill, line)
    return x + n * (w + gap) - gap


# ---------------------------------------------------------------- panels
panel(SP0, PA0, SP1, PB1, GRAY_PANEL, 'Phase-Aligned Representation')
panel(GX0, PA0, GX1, PA1, PHASE_PANEL, 'Phase Pathway')
panel(GX0, PB0, GX1, PB1, TREND_PANEL, 'Trend Pathway')
panel(FUX0, PA0, FUX1, PB1, GRAY_PANEL, 'Gated Fusion')

# ---------------------------------------------------------------- shared panel: input window over the fold
node(LX0, ya0, BWL, BH)
head(LX0, ya0, BWL, '**Input Window** $x$')
cw, ch = BWL - 0.4, 0.38
cx0 = LX0 + 0.2
curve(cx0, CA - ch / 2, cw, ch, norm([daily((k % 24) / 24, k) + 0.12 * (k // 24) / 3 for k in range(72)]),
      lw=GLYPH)
for c in (1, 2):
    xd = cx0 + cw * c / 3
    path([(xd, CA - ch / 2 - 0.03), (xd, CA + ch / 2 + 0.03)], color=SERIES_GRAY, lw=GLYPH, arrow=False,
         dash='sysDash')

xv = LX0 + BWL / 2
arrow(xv, ya0 + BH, xv, yb0)
label(xv + 0.08, (ya0 + BH + yb0) / 2 - 0.14, 0.5, 0.28, [f'$x{TL}$'], align=PP_ALIGN.LEFT)

# fold: three cycles, one per row, with their means (dashed); the boxed column is one phase token
node(LX0, yb0, BWL, BH)
head(LX0, yb0, BWL, f'**Fold by Cycle** $X{TL}$')
A, RS = 0.15, 0.185                              # curve amplitude and row spacing
fx0, fw = LX0 + 0.2, BWL - 0.4
for r in range(3):
    ry = CB - (2 * RS + A) / 2 + r * RS
    vs = norm([daily(k / 24, k + 24 * r) for k in range(24)])
    path([(fx0 + fw * k / 23, ry + A * (1 - v)) for k, v in enumerate(vs)], lw=GLYPH, arrow=False)
    my = ry + A * (1 - sum(vs) / len(vs))
    path([(fx0 - 0.06, my), (fx0 + fw + 0.06, my)], color=TREND, lw=GLYPH, arrow=False, dash='sysDash')
hx = fx0 + fw * 8 / 23
box(hx - 0.055, CB - (2 * RS + A) / 2 - 0.035, 0.11, 2 * RS + A + 0.07, fill=None, line=PHASE, lw=1.5,
    shape=MSO_SHAPE.RECTANGLE)

# one fork from the fold: the phase tokens to the phase pathway, the level token to the trend pathway
path([(LX0 + BWL, YB), (XF, YB)], lw=FLOW, arrow=False)
path([(XF, YB), (XF, YA), (C1X, YA)], color=PHASE, lw=FLOW)
arrow(XF, YB, C1X, YB, color=TREND)
box(XF - 0.05, YB - 0.05, 0.1, 0.1, fill=BLACK, line=None, shape=MSO_SHAPE.OVAL)
var(SP1, GX0, YA, f'$x{TL}_{{p}}$', PHASE)
var(SP1, GX0, YB, f'$[ζ; z{MACRON}]$', TREND)

# ---------------------------------------------------------------- encoders
# routing: phase tokens -> routers R (fewer, darker vectors of the same space) -> phase tokens
node(C1X, ya0, BW1, BH)
head(C1X, ya0, BW1, '**Cross-Phase Routing** $\\b{R}$')
ga = 0.36                                        # glyph arrow length, gaps of 0.06 on both sides
gx = C1X + (BW1 - (0.4 + 0.25 + 0.4 + 2 * (ga + 0.12))) / 2
gx = bars(gx, CA, 3)
arrow(gx + 0.06, CA, gx + 0.06 + ga, CA, lw=GLYPH)
gx = bars(gx + ga + 0.12, CA, 2, fill=PHASE, line=PHASE)
arrow(gx + 0.06, CA, gx + 0.06 + ga, CA, lw=GLYPH)
bars(gx + ga + 0.12, CA, 3)
label(C1X + BW1 + 0.05, ya0 + BH - 0.3, 0.5, 0.28, ['$×N_{r}$'], align=PP_ALIGN.LEFT)

# linear encoder: level token (cycle means zeta as bars, mean phase token z_bar as a token) -> state t
node(C1X, yb0, BW1, BH)
head(C1X, yb0, BW1, '**Linear Encoder** $E$')
nb, bw, bg = 6, 0.11, 0.045
zw = nb * bw + (nb - 1) * bg
gx = C1X + (BW1 - (zw + 0.1 + 0.1 + ga + 0.12 + 0.1)) / 2
base = CB + 0.2                                  # common baseline of zeta and z_bar
for k, v in enumerate((0.32, 0.45, 0.4, 0.58, 0.7, 0.8)):
    bar(gx + k * (bw + bg), base - 0.4 * v, bw, 0.4 * v, TREND_BOX, TREND)
gx += zw + 0.1
bar(gx, base - 0.4, 0.1, 0.4, PHASE_TOKEN, PHASE_LINE)
gx += 0.1
arrow(gx + 0.06, CB, gx + 0.06 + ga, CB, lw=GLYPH)
bar(gx + ga + 0.12, CB - 0.12, 0.1, 0.24, TREND_BOX, TREND)

# ---------------------------------------------------------------- decoders
arrow(C1X + BW1, YA, C3X, YA, color=PHASE)
var(C1X + BW1, C3X, YA, '$h_{p}$', PHASE)
arrow(C1X + BW1, YB, C3X, YB, color=TREND)
var(C1X + BW1, C3X, YB, '$t$', TREND)

node(C3X, ya0, C3W, BH, fill=YELLOW)
head(C3X, ya0, C3W, '**Shared Predictor** $Φ$')
curve(C3X + 0.3, CA - 0.16, C3W - 0.6, 0.32, norm([daily((k % 24) / 24, k) for k in range(48)]),
      color=PHASE, lw=LINE)

node(C3X, yb0, C3W, BH, fill=YELLOW)
head(C3X, yb0, C3W, '**Polynomial Basis** $Ψ_{H}$')
for k, prof in enumerate(([0.5] * 9, [i / 8 for i in range(9)], [1 - ((i - 4) / 4) ** 2 for i in range(9)])):
    icx = C3X + 0.3 + (C3W - 0.6) * (k + 0.5) / 3
    curve(icx - 0.2, CB - 0.16, 0.4, 0.32, prof, color=TREND, lw=LINE)

# ---------------------------------------------------------------- gated fusion and forecast
arrow(C3X + C3W, YA, SBX, YA, color=PHASE)
var(GX1, FUX0, YA, '$F$', PHASE)
arrow(C3X + C3W, YB, SBX, YB, color=TREND)
var(GX1, FUX0, YB, '$F_{\\r{b}}$', TREND)
for y, cells, fill in ((YA, ('$1 − g_{ℓ}$', '$1 − g_{s}$'), PHASE_PANEL),
                       (YF, ('$M_{c}$', '$J_{c}$'), WHITE),
                       (YB, ('$g_{ℓ}$', '$g_{s}$'), TREND_PANEL)):
    node(SBX, y - SBH / 2, SBW, SBH, fill=fill)
    path([(SBX + SBW / 2, y - SBH / 2), (SBX + SBW / 2, y + SBH / 2)], lw=GLYPH, arrow=False)
    for j, cell in enumerate(cells):
        label(SBX + j * SBW / 2, y - 0.15, SBW / 2, 0.3, [cell])
for j in range(2):                               # each gate column feeds the projection it weights
    xc = SBX + SBW * (2 * j + 1) / 4
    arrow(xc, YA + SBH / 2, xc, YF - SBH / 2, color=PHASE)
    arrow(xc, YB - SBH / 2, xc, YF + SBH / 2, color=TREND)
SR, XS = 0.12, SBX + SBW / 2                     # the two parts add up
box(XS - SR, YF - SR, 2 * SR, 2 * SR, fill=WHITE, lw=LINE, shape=MSO_SHAPE.OVAL)
path([(XS - 0.065, YF), (XS + 0.065, YF)], lw=LINE, arrow=False)
path([(XS, YF - 0.065), (XS, YF + 0.065)], lw=LINE, arrow=False)

arrow(SBX + SBW, YF, OX0, YF)
oy0 = YF - BH / 2
node(OX0, oy0, OW, BH)
head(OX0, oy0, OW, f'**Forecast** $y{HAT}$')
co = oy0 + HEAD + (BH - HEAD) / 2
curve(OX0 + 0.16, co - 0.19, OW - 0.32, 0.38,
      norm([daily((k % 24) / 24, k) + 0.12 * (3 + k / 24) / 3 for k in range(48)]), lw=GLYPH)

save(OUT)
