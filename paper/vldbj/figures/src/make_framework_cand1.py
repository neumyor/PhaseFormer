"""Fig. 2 candidate V1: PhaseFormer+ in the layout of the conference figure.

Three columns, as in figures/overview.pdf of the ICLR version (make_overview.py):
  left    Phase-Aligned Tokenization (fold, phase token, level token), a gray
          PhaseFormer+ bar, and the Gated Fusion of the two forecasts;
  middle  the module stack: shared embedding W_e, the trend pathway (mean over
          phases, linear encoder E, horizon basis Psi_H with the anchor) and the
          phase pathway (cross-phase routing layer x N_r, shared predictor Phi),
          both feeding the gated fusion;
  right   a zoom into one routing layer: phase-to-router aggregation and
          router-to-phase distribution.
Symbols follow body/method.tex.  Phase tokens are blue and the level token
green, as in Fig. 3; routers are indigo.

    python3 make_framework_cand1.py && ./export.sh framework_cand1 516
"""
import math
from pathlib import Path

from pptfig import *  # noqa: F401,F403

OUT = Path(__file__).with_name('framework_cand1.pptx')
FIG_H = 4.0
new_slide(12.2, FIG_H)

SERIES_GRAY = RGBColor(0x80, 0x80, 0x80)
BLUE_LINE = RGBColor(0x1F, 0x3A, 0x5F)
BLUE_TEXT = RGBColor(0x15, 0x6E, 0xA6)
LIGHT_BLUE = RGBColor(0xA6, 0xD8, 0xF2)
DARK_GREEN = RGBColor(0x2E, 0x6B, 0x1F)
ROUTER = RGBColor(0x6A, 0x7D, 0xD0)
ROUTER_LINE = RGBColor(0x3B, 0x4C, 0x9A)
LAVENDER = RGBColor(0xE6, 0xE8, 0xF7)
FUSE_RED = RGBColor(0xC0, 0x39, 0x2B)


def noise(k, amp):
    return amp * (math.sin(2.3 * k) + 0.6 * math.sin(5.1 * k + 1.3))


def plateau(t, k=0):
    """Daily traffic-like profile on t in [0, 1): low night, ramp, noisy plateau, decay."""
    if t < 0.12:
        v = 0.12
    elif t < 0.28:
        v = 0.12 + 0.68 * (t - 0.12) / 0.16
    elif t < 0.78:
        v = 0.8 + 0.06 * math.sin(14 * t) + noise(k, 0.025)
    elif t < 0.95:
        v = 0.8 - 0.68 * (t - 0.78) / 0.17
    else:
        v = 0.12
    return v


def series_box(x, y, w, h, cycles, prof, levels=None):
    box(x, y, w, h, fill=WHITE, line=SERIES_GRAY, lw=2, shape=MSO_SHAPE.RECTANGLE)
    n = 40 * cycles
    levels = levels or [0.0] * cycles
    ys = [0.75 * prof((k % 40) / 40, k) + levels[k // 40] for k in range(n)]
    curve(x + 0.04, y + 0.05, w - 0.08, h - 0.1, ys, lw=1.0)
    for c in range(1, cycles):
        xd = x + w * c / cycles
        path([(xd, y + 0.03), (xd, y + h - 0.03)], color=SERIES_GRAY, lw=1.5, arrow=False, dash='sysDash')


def chip(cx, cy, w=0.62, h=0.26):
    """Router icon: a flat chip in perspective (top face and two sides)."""
    t = 0.08
    top = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy), (cx, cy - h / 2)]
    left = [(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx - w / 2, cy + t), (cx - w / 2, cy)]
    right = [(cx + w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx + w / 2, cy + t), (cx + w / 2, cy)]
    path(left, color=ROUTER_LINE, lw=0.75, arrow=False, fill=RGBColor(0x4E, 0x62, 0xB8), closed=True)
    path(right, color=ROUTER_LINE, lw=0.75, arrow=False, fill=ROUTER_LINE, closed=True)
    path(top, color=ROUTER_LINE, lw=0.75, arrow=False, fill=RGBColor(0x8C, 0x9E, 0xE6), closed=True)
    inner = [(cx, cy - h / 4), (cx + w / 4, cy), (cx, cy + h / 4), (cx - w / 4, cy), (cx, cy - h / 4)]
    path(inner, color=RGBColor(0x6A, 0x7D, 0xD0), lw=0.75, arrow=False, fill=RGBColor(0x7A, 0x8D, 0xDE),
         closed=True)


def down(x, y0, y1, **kw):
    path([(x, y0), (x, y1)], **kw)


TOKEN_LINE = {id(TOKEN_BLUE): BLUE_LINE, id(ROUTER): ROUTER_LINE, id(TOKEN_GREEN): DARK_GREEN}


def token_bar(cx, y, n, color, w=0.12, h=0.15):
    """Contiguous small tokens (keys/values/queries) centered at cx."""
    x = cx - n * w / 2
    for k in range(n):
        box(x + k * w, y, w, h, fill=color, line=TOKEN_LINE[id(color)], lw=1.25, shape=MSO_SHAPE.RECTANGLE)
    return x


def token_row(cx, y, n, color, w=0.3, h=0.13, gap=0.08):
    x = cx - (n * w + (n - 1) * gap) / 2
    for k in range(n):
        box(x + k * (w + gap), y, w, h, fill=color, line=TOKEN_LINE[id(color)], lw=1.0,
            shape=MSO_SHAPE.RECTANGLE)


def operator(x, y, r=0.1, color=BLACK):
    box(x - r, y - r, 2 * r, 2 * r, fill=WHITE, line=color, lw=1.25, shape=MSO_SHAPE.OVAL)
    a = r * 0.6
    path([(x - a, y), (x + a, y)], color=color, lw=1.25, arrow=False)
    path([(x, y - a), (x, y + a)], color=color, lw=1.25, arrow=False)


# --------------------------------------------------------------------------
# Left column: tokenization, PhaseFormer+, gated fusion.
# --------------------------------------------------------------------------
LX, LW = 0.04, 2.96
T0, T1 = 0.04, 1.6             # tokenization box
B0, B1 = 1.72, 2.08            # PhaseFormer+ bar
G0, G1 = 2.2, FIG_H - 0.04     # gated fusion box

box(LX, T0, LW, T1 - T0, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
label(LX + 0.07, T0 + 0.03, 2.6, 0.3, ['**Phase-Aligned Tokens**'], size=TITLE, align=PP_ALIGN.LEFT)
label(LX + 0.03, 0.8, 0.24, 0.3, ['$x$'], size=TITLE)
series_box(0.3, 0.76, 1.0, 0.38, 3, plateau, levels=[0.0, 0.08, 0.16])
label(0.1, 1.2, 1.4, 0.26, ['normalize, fold'], size=SMALL)
block_arrow(1.36, 0.88, 0.18, 0.14)

# Folded window: one row per cycle (K rows), phases along the row.
GX, GW, GH, GG = 1.62, 0.78, 0.2, 0.05
rows_y = [0.64 + k * (GH + GG) for k in range(3)]
col_t = 0.62                   # phase of the highlighted column
col_x = GX + 0.03 + (GW - 0.06) * col_t
box(col_x - 0.05, rows_y[0] - 0.05, 0.1, rows_y[-1] + GH + 0.05 - (rows_y[0] - 0.05), fill=LIGHT_BLUE,
    line=BLUE_LINE, lw=1.25, shape=MSO_SHAPE.RECTANGLE)
for k, ry in enumerate(rows_y):
    box(GX, ry, GW, GH, fill=None, line=SERIES_GRAY, lw=1.5, shape=MSO_SHAPE.RECTANGLE)
    ys = [plateau(j / 40, 9 * k + j) for j in range(41)]
    curve(GX + 0.03, ry + 0.03, GW - 0.06, GH - 0.06, ys, lw=1.0)
    v = plateau(col_t, 9 * k + round(col_t * 40))
    cy = ry + 0.03 + (GH - 0.06) * (1 - v)
    r = 0.04
    box(col_x - r, cy - r, 2 * r, 2 * r, fill=TOKEN_BLUE, line=BLUE_LINE, lw=1.0, shape=MSO_SHAPE.OVAL)
    path([(GX + GW + 0.02, ry + GH / 2), (2.6, ry + GH / 2)], color=DARK_GREEN, lw=1.0)   # row mean
    box(2.6, ry + 0.02, 0.17, GH - 0.04, fill=TOKEN_GREEN, line=DARK_GREEN, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
label(col_x - 0.75, 0.36, 1.5, 0.26, ['phase token $x̃_p$'], size=SMALL, color=BLUE_TEXT)
label(1.85, 1.34, 1.1, 0.24, ['level token $r̄$'], size=SMALL, color=DARK_GREEN, align=PP_ALIGN.RIGHT)

path([(LX + LW / 2, T1), (LX + LW / 2, B0)])
box(LX, B0, LW, B1 - B0, ['**PhaseFormer+**'], fill=GRAY_BOX, line=PURPLE, lw=1.75, size=22,
    shape=MSO_SHAPE.RECTANGLE)
path([(LX + LW / 2, B1), (LX + LW / 2, G0)])

# Gated fusion: F and F_b split into levels (M_c) and shape (J_c), fused per part.
box(LX, G0, LW, G1 - G0, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
label(LX + 0.07, G0 + 0.02, 2.6, 0.3, ['**Gated Fusion**'], size=TITLE, align=PP_ALIGN.LEFT)
C1, C2, CW = 0.78, 1.52, 0.5
RF0, RF1 = 2.78, 3.02          # F cells
RB0, RB1 = 3.58, 3.82          # F_b cells
OY = (RF1 + RB0) / 2           # operators
label(C1 - 0.4, 2.52, 0.8, 0.24, ['levels $M_c$'], size=SMALL)
label(C2 - 0.4, 2.52, 0.8, 0.24, ['shape $J_c$'], size=SMALL)
label(LX + 0.04, RF0 - 0.03, 0.3, 0.3, ['$F$'])
label(LX + 0.04, RB0 - 0.03, 0.3, 0.3, ['$F_{b}$'])


def cell(cx, y0, y1, fill, line, ys, color):
    box(cx - CW / 2, y0, CW, y1 - y0, fill=fill, line=line, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
    curve(cx - CW / 2 + 0.05, y0 + 0.04, CW - 0.1, y1 - y0 - 0.08, ys, color=color, lw=1.25)


def steps(lv):
    return [lv[min(j // 10, len(lv) - 1)] for j in range(30)]


peaks = [0.1 + 0.85 * plateau((j % 15) / 15, j) for j in range(31)]
saw = [0.3 + 0.4 * ((j % 10) / 9) for j in range(30)]
cell(C1, RF0, RF1, BLUE_PANEL, BLUE_LINE, steps([0.45, 0.5, 0.48]), BLUE_LINE)
cell(C2, RF0, RF1, BLUE_PANEL, BLUE_LINE, peaks, BLUE_LINE)
cell(C1, RB0, RB1, GREEN_PANEL, DARK_GREEN, steps([0.15, 0.5, 0.85]), DARK_GREEN)
cell(C2, RB0, RB1, GREEN_PANEL, DARK_GREEN, saw, DARK_GREEN)
for cx, top, bottom in ((C1, '$1−g_{ℓ}$', '$g_{ℓ}$'), (C2, '$1−g_{s}$', '$g_{s}$')):
    down(cx, RF1, OY - 0.1, lw=1.25)
    down(cx, RB0, OY + 0.1, lw=1.25)
    operator(cx, OY)
    label(cx - 0.66, RF1 - 0.03, 0.52, 0.2, [top], size=SMALL, align=PP_ALIGN.RIGHT)
    label(cx - 0.66, OY + 0.1, 0.52, 0.2, [bottom], size=SMALL, align=PP_ALIGN.RIGHT)
path([(C1 + 0.1, OY), (C2 - 0.1, OY)], lw=1.25)
OX0, OX1 = 1.98, 2.96
path([(C2 + 0.1, OY), (OX0, OY)], color=FUSE_RED, lw=1.5)
box(OX0, 2.56, OX1 - OX0, 1.28, fill=WHITE, lw=1.25)
label(OX0, 2.6, OX1 - OX0, 0.28, ['**Forecast**'], size=SMALL)
label(OX0, 2.84, OX1 - OX0, 0.28, ['$ŷ$'])
fc = [0.1 + 0.75 * plateau((j % 20) / 20, j) + 0.1 * j / 40 for j in range(41)]
curve(OX0 + 0.1, 3.14, OX1 - OX0 - 0.2, 0.3, fc, color=FUSE_RED, lw=1.25)
label(OX0, 3.52, OX1 - OX0, 0.24, ['de-normalize'], size=SMALL)

# --------------------------------------------------------------------------
# Middle column: the module stack of PhaseFormer+.
# --------------------------------------------------------------------------
MX0, MX1 = 3.1, 6.2
CX0, CX1 = 3.18, 6.12          # content
box(MX0, 0.04, MX1 - MX0, FIG_H - 0.08, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
EY0, EY1 = 0.18, 0.52
down((CX0 + CX1) / 2, 0.04, EY0)
box(CX0, EY0, CX1 - CX0, EY1 - EY0, ['Shared Embedding $\\b{W}_{e}$'], fill=YELLOW)

# Trend pathway (left branch).
TC, TW = 3.72, 1.06
for y0, y1, lines in ((0.74, 1.14, ['Mean over', 'phases']),
                      (1.42, 1.82, ['Linear', 'encoder $E$']),
                      (2.1, 2.82, ['Horizon', 'basis $Ψ_{H}$', '+ anchor $r̄_{K}$'])):
    box(TC - TW / 2, y0, TW, y1 - y0, lines, fill=GREEN_PANEL, size=SMALL, spacing=0.9)
down(TC, EY1, 0.74)
down(TC, 1.14, 1.42)
label(TC + 0.05, 1.16, 0.4, 0.24, ['$z̄$'], align=PP_ALIGN.LEFT)
down(TC, 1.82, 2.1)
label(TC + 0.05, 1.84, 0.4, 0.24, ['$t$'], align=PP_ALIGN.LEFT)

# Phase pathway (right branch): routing layers and the shared predictor.
SX0, SX1, SY0, SY1 = 4.38, 5.86, 0.98, 2.62
PC = (SX0 + SX1) / 2
for off in (0.12, 0.06):
    box(SX0 + off, SY0 - off, SX1 - SX0, SY1 - SY0, fill=WHITE, line=GRAY_LINE, lw=0.75,
        shape=MSO_SHAPE.RECTANGLE)
down(PC, EY1, SY0)
operator(PC, 0.71, r=0.08)
label(PC - 0.5, 0.6, 0.4, 0.24, ['$\\b{Z}$'], align=PP_ALIGN.RIGHT)
label(PC + 0.12, 0.6, 0.6, 0.24, ['pos.'], size=SMALL, align=PP_ALIGN.LEFT)
box(SX0, SY0, SX1 - SX0, SY1 - SY0, fill=GRAY_PANEL, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
label(SX0, SY0 + 0.02, SX1 - SX0, 0.44, ['**Cross-Phase**', '**Routing Layer**'], size=TITLE, spacing=0.88)
box(SX0 + 0.1, 1.5, SX1 - SX0 - 0.2, 0.28, ['Aggregation'], fill=BLUE_PANEL, lw=1.25, dash=True)
box(SX0 + 0.1, 1.84, SX1 - SX0 - 0.2, 0.28, ['Distribution'], fill=LAVENDER, lw=1.25, dash=True)
box(SX0 + 0.06, 2.16, SX1 - SX0 - 0.12, 0.4, ['Add & LN, MLP,', 'Add & LN'], size=SMALL, spacing=0.9)
label(SX1 - 0.45, SY1 + 0.01, 0.45, 0.26, ['$×N_{r}$'], align=PP_ALIGN.RIGHT)
PY0, PY1 = 2.96, 3.38
down(PC, SY1, PY0)
label(PC - 0.5, SY1 + 0.0, 0.4, 0.24, ['$h_p$'], align=PP_ALIGN.RIGHT)
box(SX0, PY0, SX1 - SX0, PY1 - PY0, ['Shared', 'Predictor $Φ$'], fill=BLUE_PANEL, spacing=0.88)

# Gated fusion of the two forecasts.
FY0, FY1 = 3.62, 3.9
down(TC, 2.82, FY0)
label(TC + 0.05, 3.1, 0.4, 0.26, ['$F_{b}$'], align=PP_ALIGN.LEFT)
down(PC, PY1, FY0)
label(PC + 0.05, PY1 - 0.03, 0.4, 0.26, ['$F$'], align=PP_ALIGN.LEFT)
box(CX0, FY0, CX1 - CX0, FY1 - FY0, ['Gated Fusion $g_{ℓ}$, $g_{s}$'], fill=PEACH)

# Zoom lines.
RX0 = 6.32
for (a, b) in (((LX + LW, B0), (MX0, 0.04)), ((LX + LW, B1), (MX0, FIG_H - 0.04)),
               ((SX1 + 0.12, SY0 - 0.12), (RX0, 0.04)), ((SX1 + 0.12, SY1 - 0.12), (RX0, FIG_H - 0.04))):
    path([a, b], lw=2.0, arrow=False, dash='sysDot')

# --------------------------------------------------------------------------
# Right panel: the two steps of a cross-phase routing layer.
# --------------------------------------------------------------------------
box(RX0, 0.04, 12.16 - RX0, FIG_H - 0.08, fill=GRAY_PANEL, shape=MSO_SHAPE.RECTANGLE)
P0, P1 = 0.12, FIG_H - 0.12    # step panels
IN0, IN1 = 0.56, 1.14          # input boxes
W0, W1 = 1.38, 1.82            # projection boxes
YT = 2.1                       # token bars
YM0, YM1 = 2.5, 2.9            # multi-head attention
YO0, YO1 = 3.1, 3.66           # output box
STEP = 14                      # step titles


def step_panel(x0, x1, title, fill):
    box(x0, P0, x1 - x0, P1 - P0, fill=fill, lw=1.25, dash=True, shape=MSO_SHAPE.RECTANGLE)
    label(x0 + 0.1, P0 + 0.05, x1 - x0 - 0.15, 0.3, [f'**{title}**'], size=STEP, align=PP_ALIGN.LEFT)


def wbox(cx, sub, sup, w=0.54):
    box(cx - w / 2, W0, w, W1 - W0, [f'$\\b{{W}}_{sub}^{{({sup})}}$'], size=TITLE, margin=0.01)


def kqv(columns, sup):
    for cx, n, col, name in columns:
        wbox(cx, name[0], sup)
        down(cx, W1, YT)
        token_bar(cx, YT, n, col)
        label(cx + 0.08, W1 + 0.01, 0.34, 0.26, [f'${name}$'], align=PP_ALIGN.LEFT)
        down(cx, YT + 0.15, YM0)


def input_box(x0, x1, n, color, text, w=0.26, gap=0.07):
    box(x0, IN0, x1 - x0, IN1 - IN0, fill=WHITE)
    token_row((x0 + x1) / 2, IN0 + 0.07, n, color, w=w, gap=gap)
    label(x0, IN0 + 0.24, x1 - x0, 0.3, [text])


def output_box(x0, x1, n, color, text):
    c = (x0 + x1) / 2
    box(x0, YM0, x1 - x0, YM1 - YM0, ['Multi-Head Attention'])
    down(c, YM1, YO0)
    box(x0, YO0, x1 - x0, YO1 - YO0, fill=WHITE)
    token_row(c, YO0 + 0.07, n, color)
    label(x0, YO0 + 0.23, x1 - x0, 0.3, [text])


# Phase-to-router aggregation: R' = MHA(R, Z~, Z~).
A0, A1 = 6.4, 9.12
step_panel(A0, A1, 'Phase-to-Router Aggregation', BLUE_PANEL)
input_box(A0 + 0.1, 8.0, 4, TOKEN_BLUE, 'Phase Tokens $\\b{Z̃}$')
kx, vx, qx = 6.9, 7.6, 8.56
chip(qx, 0.68)
label(qx - 0.5, 0.84, 1.0, 0.28, ['__Routers__ $\\b{R}$'])
for cx in (kx, vx):
    down(cx, IN1, W0)
down(qx, 1.14, W0)
kqv(((kx, 4, TOKEN_BLUE, 'K_z'), (vx, 4, TOKEN_BLUE, 'V_z'), (qx, 3, ROUTER, 'Q_r')), 'a')
output_box(A0 + 0.1, A1 - 0.1, 3, ROUTER, 'Contextualized Routers $\\b{R}′$')

# Router-to-phase distribution: Z_attn = MHA(Z~, R', R').
D0, D1 = 9.2, 12.08
step_panel(D0, D1, 'Router-to-Phase Distribution', LAVENDER)
input_box(D0 + 0.1, 10.5, 3, ROUTER, 'Routers $\\b{R}′$')
input_box(10.58, D1 - 0.1, 4, TOKEN_BLUE, 'Phase Tokens $\\b{Z̃}$', w=0.22)
kx, vx, qx = 9.6, 10.2, 11.28
for cx in (kx, vx, qx):
    down(cx, IN1, W0)
kqv(((kx, 3, ROUTER, 'K_r'), (vx, 3, ROUTER, 'V_r'), (qx, 4, TOKEN_BLUE, 'Q_z')), 'd')
output_box(D0 + 0.1, D1 - 0.1, 4, TOKEN_BLUE, 'Enhanced Phase Tokens $\\b{Z}_{\\r{attn}}$')

# Curved hand-over of R' from aggregation to distribution.
p0, p1, p2, p3 = (A1 - 0.1, (YO0 + YO1) / 2), (9.2, (YO0 + YO1) / 2 - 0.06), (9.04, 1.0), (D0 + 0.1, 0.86)
bez = []
for k in range(25):
    t = k / 24
    bez.append(tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                     for a, b, c, d in zip(p0, p1, p2, p3)))
path(bez, lw=1.5)

save(OUT)
